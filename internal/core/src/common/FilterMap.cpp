// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include "common/FilterMap.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace milvus {
namespace {

struct SparseStorage {
    bool default_bit;
    size_t exception_cap;
    std::vector<int32_t> exceptions;
};

struct DenseStorage {
    std::shared_ptr<TargetBitmap> bitmap;
};

void
ValidateUniverse(size_t universe, size_t exception_cap) {
    if (universe > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument(
            "FilterMap universe exceeds the int32 row-ID range");
    }
    if (exception_cap > universe) {
        throw std::invalid_argument(
            "FilterMap exception cap exceeds its universe");
    }
}

bool
Contains(const std::vector<int32_t>& ids, int32_t id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

}  // namespace

struct FilterMap::Storage {
    Storage(size_t universe, SparseStorage sparse)
        : universe(universe), value(std::move(sparse)) {
    }

    Storage(size_t universe, DenseStorage dense)
        : universe(universe), value(std::move(dense)) {
    }

    size_t universe;
    std::variant<SparseStorage, DenseStorage> value;
};

FilterMap::FilterMap(std::shared_ptr<Storage> storage)
    : storage_(std::move(storage)) {
}

FilterMap
FilterMap::FromDense(std::shared_ptr<TargetBitmap> dense) {
    if (dense == nullptr) {
        throw std::invalid_argument("Dense FilterMap requires an owner");
    }
    return FilterMap(std::make_shared<Storage>(dense->size(),
                                               DenseStorage{std::move(dense)}));
}

FilterMap
FilterMap::Adaptive(size_t universe, bool default_bit, size_t exception_cap) {
    ValidateUniverse(universe, exception_cap);
    SparseStorage sparse{default_bit, exception_cap, {}};
    sparse.exceptions.reserve(exception_cap);
    return FilterMap(std::make_shared<Storage>(universe, std::move(sparse)));
}

bool
FilterMap::IsInitialized() const noexcept {
    return storage_ != nullptr;
}

bool
FilterMap::IsDense() const {
    return std::holds_alternative<DenseStorage>(GetStorage().value);
}

size_t
FilterMap::size() const {
    return GetStorage().universe;
}

size_t
FilterMap::count() const {
    const auto& storage = GetStorage();
    if (const auto* dense = std::get_if<DenseStorage>(&storage.value)) {
        return dense->bitmap->count();
    }
    const auto& sparse = std::get<SparseStorage>(storage.value);
    return sparse.default_bit ? storage.universe - sparse.exceptions.size()
                              : sparse.exceptions.size();
}

bool
FilterMap::test(size_t id) const {
    CheckId(id);
    const auto& storage = GetStorage();
    if (const auto* dense = std::get_if<DenseStorage>(&storage.value)) {
        return (*dense->bitmap)[id];
    }
    const auto& sparse = std::get<SparseStorage>(storage.value);
    return Contains(sparse.exceptions, static_cast<int32_t>(id))
               ? !sparse.default_bit
               : sparse.default_bit;
}

void
FilterMap::set(size_t id) {
    set(id, true);
}

void
FilterMap::reset(size_t id) {
    set(id, false);
}

void
FilterMap::set(size_t id, bool value) {
    CheckId(id);
    auto& storage = GetMutableStorage();
    if (std::holds_alternative<DenseStorage>(storage.value)) {
        GetMutableDense().set(id, value);
        return;
    }

    auto& sparse = std::get<SparseStorage>(storage.value);
    const auto row_id = static_cast<int32_t>(id);
    const auto it =
        std::find(sparse.exceptions.begin(), sparse.exceptions.end(), row_id);
    const bool is_exception = it != sparse.exceptions.end();
    if (value == sparse.default_bit) {
        if (is_exception) {
            sparse.exceptions.erase(it);
        }
        return;
    }
    if (is_exception) {
        return;
    }
    if (sparse.exceptions.size() < sparse.exception_cap) {
        sparse.exceptions.push_back(row_id);
        return;
    }

    PromoteToDense();
    GetMutableDense().set(id, value);
}

FilterMapCapability
FilterMap::capability() const {
    const auto& storage = GetStorage();
    if (std::holds_alternative<DenseStorage>(storage.value)) {
        return FilterMapCapability::RandomMembership;
    }
    if (std::get<SparseStorage>(storage.value).default_bit) {
        return FilterMapCapability::EnumerateOnly;
    }
    throw std::logic_error(
        "default-zero Sparse FilterMap must be Dense for unset-ID access");
}

void
FilterMap::AssignBatch(TargetBitmapView source,
                       const TargetBitmapView* validity,
                       size_t offset,
                       bool invert) {
    if (validity != nullptr && validity->size() != source.size()) {
        throw std::invalid_argument(
            "FilterMap source and validity batch sizes differ");
    }
    const auto universe = size();
    if (offset > universe || source.size() > universe - offset) {
        throw std::out_of_range("FilterMap batch exceeds universe");
    }
    if (IsDense()) {
        WriteDenseBatch(source, validity, offset, invert);
        return;
    }

    auto& storage = GetMutableStorage();
    auto& sparse = std::get<SparseStorage>(storage.value);
    const auto retained_prefix = sparse.exceptions.size();

    using Policy = TargetBitmapView::policy_type;
    using Word = TargetBitmapView::data_type;
    constexpr size_t kWordBits = sizeof(Word) * 8;
    bool exceeds_cap = false;
    for (size_t word_offset = 0; word_offset < source.size();
         word_offset += kWordBits) {
        const auto bits = std::min(kWordBits, source.size() - word_offset);
        Word logical =
            Policy::op_read(source.data(), source.offset() + word_offset, bits);
        if (validity != nullptr) {
            logical &= Policy::op_read(
                validity->data(), validity->offset() + word_offset, bits);
        }
        const Word used_mask = bits == kWordBits
                                   ? std::numeric_limits<Word>::max()
                                   : (static_cast<Word>(1) << bits) - 1;
        logical &= used_mask;
        if (invert) {
            logical = static_cast<Word>(~logical) & used_mask;
        }
        Word exceptions = sparse.default_bit
                              ? static_cast<Word>(~logical) & used_mask
                              : logical;
        while (exceptions != 0) {
            const auto bit = static_cast<size_t>(std::countr_zero(exceptions));
            if (sparse.exceptions.size() == sparse.exception_cap) {
                exceeds_cap = true;
                break;
            }
            sparse.exceptions.push_back(
                static_cast<int32_t>(offset + word_offset + bit));
            exceptions &= exceptions - 1;
        }
        if (exceeds_cap) {
            break;
        }
    }

    if (!exceeds_cap) {
        return;
    }

    // IDs tentatively appended from the triggering batch are discarded. The
    // retained prefix is backfilled once and this full batch is written once.
    sparse.exceptions.resize(retained_prefix);
    PromoteToDense();
    WriteDenseBatch(source, validity, offset, invert);
}

size_t
FilterMap::ReadUnsetBatch(FilterMapCursor& cursor,
                          std::span<int32_t> output) const {
    if (output.empty()) {
        return 0;
    }

    const auto& storage = GetStorage();
    if (const auto* sparse = std::get_if<SparseStorage>(&storage.value);
        sparse != nullptr && sparse->default_bit) {
        const auto remaining =
            sparse->exceptions.size() -
            std::min(cursor.position, sparse->exceptions.size());
        const auto count = std::min(remaining, output.size());
        std::copy_n(sparse->exceptions.begin() +
                        std::min(cursor.position, sparse->exceptions.size()),
                    count,
                    output.begin());
        cursor.position += count;
        return count;
    }

    size_t written = 0;
    while (cursor.position < storage.universe && written < output.size()) {
        const auto id = cursor.position++;
        if (!test(id)) {
            output[written++] = static_cast<int32_t>(id);
        }
    }
    return written;
}

TargetBitmap&
FilterMap::EnsureDense() {
    if (!IsDense()) {
        PromoteToDense();
    }
    return GetMutableDense();
}

const FilterMap::Storage&
FilterMap::GetStorage() const {
    if (storage_ == nullptr) {
        throw std::logic_error("FilterMap is not initialized");
    }
    return *storage_;
}

FilterMap::Storage&
FilterMap::GetMutableStorage() {
    if (storage_ == nullptr) {
        throw std::logic_error("FilterMap is not initialized");
    }
    if (!storage_.unique()) {
        storage_ = std::make_shared<Storage>(*storage_);
    }
    return *storage_;
}

TargetBitmap&
FilterMap::GetMutableDense() {
    auto& storage = GetMutableStorage();
    auto* dense = std::get_if<DenseStorage>(&storage.value);
    if (dense == nullptr) {
        throw std::logic_error("FilterMap is not Dense");
    }
    if (!dense->bitmap.unique()) {
        dense->bitmap = std::make_shared<TargetBitmap>(dense->bitmap->clone());
    }
    return *dense->bitmap;
}

void
FilterMap::PromoteToDense() {
    auto& storage = GetMutableStorage();
    auto* sparse = std::get_if<SparseStorage>(&storage.value);
    if (sparse == nullptr) {
        return;
    }

    auto bitmap =
        std::make_shared<TargetBitmap>(storage.universe, sparse->default_bit);
    for (const auto id : sparse->exceptions) {
        bitmap->set(static_cast<size_t>(id), !sparse->default_bit);
    }
    storage.value = DenseStorage{std::move(bitmap)};
}

void
FilterMap::WriteDenseBatch(TargetBitmapView source,
                           const TargetBitmapView* validity,
                           size_t offset,
                           bool invert) {
    auto& dense = GetMutableDense();
    if (validity != nullptr) {
        source.inplace_and(*validity, source.size());
    }
    if (invert) {
        source.flip();
    }
    TargetBitmapView::policy_type::op_copy(
        source.data(), source.offset(), dense.data(), offset, source.size());
}

void
FilterMap::CheckId(size_t id) const {
    const auto universe = size();
    if (id >= universe ||
        id > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::out_of_range("FilterMap row ID is outside universe");
    }
}

}  // namespace milvus
