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

struct SparseExceptionBitmapRep {
    bool default_bit;
    size_t exception_cap;
    std::vector<int32_t> exceptions;
};

struct DenseBitmapRep {
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

template <bool WordAligned, bool HasValidity, bool Complement>
bool
AppendBitmapExceptions(TargetBitmapView source,
                       const TargetBitmapView* validity,
                       size_t offset,
                       size_t exception_cap,
                       std::vector<int32_t>& exceptions) {
    using Policy = TargetBitmapView::policy_type;
    using Word = TargetBitmapView::data_type;
    constexpr size_t kWordBits = sizeof(Word) * 8;

    const auto append_word = [&](Word word, size_t word_offset) {
        while (word != 0) {
            if (exceptions.size() == exception_cap) {
                return false;
            }
            const auto bit = static_cast<size_t>(std::countr_zero(word));
            exceptions.push_back(
                static_cast<int32_t>(offset + word_offset + bit));
            word &= word - 1;
        }
        return true;
    };

    const auto full_words = source.size() / kWordBits;
    for (size_t word_index = 0; word_index < full_words; ++word_index) {
        const auto word_offset = word_index * kWordBits;
        Word word;
        if constexpr (WordAligned) {
            word = source.data()[source.offset() / kWordBits + word_index];
        } else {
            word = Policy::op_read(
                source.data(), source.offset() + word_offset, kWordBits);
        }
        if constexpr (HasValidity) {
            if constexpr (WordAligned) {
                word &=
                    validity
                        ->data()[validity->offset() / kWordBits + word_index];
            } else {
                word &= Policy::op_read(validity->data(),
                                        validity->offset() + word_offset,
                                        kWordBits);
            }
        }
        if constexpr (Complement) {
            word = static_cast<Word>(~word);
        }
        if (!append_word(word, word_offset)) {
            return false;
        }
    }

    const auto tail_bits = source.size() % kWordBits;
    if (tail_bits == 0) {
        return true;
    }
    const auto word_offset = full_words * kWordBits;
    Word word;
    if constexpr (WordAligned) {
        word = source.data()[source.offset() / kWordBits + full_words];
    } else {
        word = Policy::op_read(
            source.data(), source.offset() + word_offset, tail_bits);
    }
    if constexpr (HasValidity) {
        if constexpr (WordAligned) {
            word &=
                validity->data()[validity->offset() / kWordBits + full_words];
        } else {
            word &= Policy::op_read(
                validity->data(), validity->offset() + word_offset, tail_bits);
        }
    }
    const auto used_mask =
        (static_cast<Word>(1) << tail_bits) - static_cast<Word>(1);
    if constexpr (Complement) {
        word = static_cast<Word>(~word);
    }
    return append_word(word & used_mask, word_offset);
}

template <bool WordAligned, bool HasValidity>
bool
AppendBitmapExceptionsWithPolarity(TargetBitmapView source,
                                   const TargetBitmapView* validity,
                                   size_t offset,
                                   size_t exception_cap,
                                   std::vector<int32_t>& exceptions,
                                   bool complement) {
    return complement
               ? AppendBitmapExceptions<WordAligned, HasValidity, true>(
                     source, validity, offset, exception_cap, exceptions)
               : AppendBitmapExceptions<WordAligned, HasValidity, false>(
                     source, validity, offset, exception_cap, exceptions);
}

}  // namespace

struct FilterMap::Storage {
    Storage(size_t universe, SparseExceptionBitmapRep sparse)
        : universe(universe), value(std::move(sparse)) {
    }

    Storage(size_t universe, DenseBitmapRep dense)
        : universe(universe), value(std::move(dense)) {
    }

    size_t universe;
    std::variant<SparseExceptionBitmapRep, DenseBitmapRep> value;
};

FilterMap::FilterMap(std::shared_ptr<Storage> storage)
    : storage_(std::move(storage)) {
}

FilterMap
FilterMap::FromDense(std::shared_ptr<TargetBitmap> dense) {
    if (dense == nullptr) {
        throw std::invalid_argument("Dense FilterMap requires an owner");
    }
    return FilterMap(std::make_shared<Storage>(
        dense->size(), DenseBitmapRep{std::move(dense)}));
}

FilterMap
FilterMap::Adaptive(size_t universe, bool default_bit, size_t exception_cap) {
    ValidateUniverse(universe, exception_cap);
    SparseExceptionBitmapRep sparse{default_bit, exception_cap, {}};
    sparse.exceptions.reserve(exception_cap);
    return FilterMap(std::make_shared<Storage>(universe, std::move(sparse)));
}

bool
FilterMap::IsInitialized() const noexcept {
    return storage_ != nullptr;
}

bool
FilterMap::IsDense() const {
    return std::holds_alternative<DenseBitmapRep>(GetStorage().value);
}

size_t
FilterMap::size() const {
    return GetStorage().universe;
}

size_t
FilterMap::count() const {
    const auto& storage = GetStorage();
    if (const auto* dense = std::get_if<DenseBitmapRep>(&storage.value)) {
        return dense->bitmap->count();
    }
    const auto& sparse = std::get<SparseExceptionBitmapRep>(storage.value);
    return sparse.default_bit ? storage.universe - sparse.exceptions.size()
                              : sparse.exceptions.size();
}

bool
FilterMap::test(size_t id) const {
    CheckId(id);
    const auto& storage = GetStorage();
    if (const auto* dense = std::get_if<DenseBitmapRep>(&storage.value)) {
        return (*dense->bitmap)[id];
    }
    const auto& sparse = std::get<SparseExceptionBitmapRep>(storage.value);
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
    if (std::holds_alternative<DenseBitmapRep>(storage.value)) {
        GetMutableDense().set(id, value);
        return;
    }

    auto& sparse = std::get<SparseExceptionBitmapRep>(storage.value);
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
    if (std::holds_alternative<DenseBitmapRep>(storage.value)) {
        return FilterMapCapability::RandomMembership;
    }
    if (std::get<SparseExceptionBitmapRep>(storage.value).default_bit) {
        return FilterMapCapability::EnumerateOnly;
    }
    throw std::logic_error(
        "default-zero Sparse FilterMap must be Dense for unset-ID access");
}

void
FilterMap::AssignBitmapBatch(TargetBitmapView source,
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
    auto& sparse = std::get<SparseExceptionBitmapRep>(storage.value);
    const auto retained_prefix = sparse.exceptions.size();

    // exception = logical XOR default_bit, while logical is source XOR
    // invert after validity has been applied. Hoist both polarity and
    // validity dispatch out of the word loop so the common accepted-bitmap
    // path (invert == default_bit) compiles to source & validity directly.
    const bool complement = invert != sparse.default_bit;
    constexpr auto kWordBits = sizeof(TargetBitmapView::data_type) * 8;
    const bool word_aligned =
        source.offset() % kWordBits == 0 &&
        (validity == nullptr || validity->offset() % kWordBits == 0);
    bool within_cap;
    if (validity != nullptr) {
        within_cap = word_aligned
                         ? AppendBitmapExceptionsWithPolarity<true, true>(
                               source,
                               validity,
                               offset,
                               sparse.exception_cap,
                               sparse.exceptions,
                               complement)
                         : AppendBitmapExceptionsWithPolarity<false, true>(
                               source,
                               validity,
                               offset,
                               sparse.exception_cap,
                               sparse.exceptions,
                               complement);
    } else {
        within_cap = word_aligned
                         ? AppendBitmapExceptionsWithPolarity<true, false>(
                               source,
                               nullptr,
                               offset,
                               sparse.exception_cap,
                               sparse.exceptions,
                               complement)
                         : AppendBitmapExceptionsWithPolarity<false, false>(
                               source,
                               nullptr,
                               offset,
                               sparse.exception_cap,
                               sparse.exceptions,
                               complement);
    }

    if (within_cap) {
        return;
    }

    // IDs tentatively appended from the triggering batch are discarded. The
    // retained prefix is backfilled once and this full batch is written once.
    sparse.exceptions.resize(retained_prefix);
    PromoteToDense();
    WriteDenseBatch(source, validity, offset, invert);
}

void
FilterMap::AppendUniqueBits(std::span<const int32_t> ids, bool value) {
    const auto universe = size();
    for (const auto id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= universe) {
            throw std::out_of_range(
                "FilterMap producer ID is outside universe");
        }
    }
    if (ids.empty()) {
        return;
    }

    auto& storage = GetMutableStorage();
    if (std::holds_alternative<DenseBitmapRep>(storage.value)) {
        auto& dense = GetMutableDense();
        for (const auto id : ids) {
            dense.set(static_cast<size_t>(id), value);
        }
        return;
    }

    auto& sparse = std::get<SparseExceptionBitmapRep>(storage.value);
    if (value == sparse.default_bit) {
        return;
    }

    const auto available = sparse.exception_cap - sparse.exceptions.size();
    const auto sparse_count = std::min(available, ids.size());
    sparse.exceptions.insert(
        sparse.exceptions.end(), ids.begin(), ids.begin() + sparse_count);
    if (sparse_count == ids.size()) {
        return;
    }

    PromoteToDense();
    auto& dense = GetMutableDense();
    for (size_t index = sparse_count; index < ids.size(); ++index) {
        dense.set(static_cast<size_t>(ids[index]), value);
    }
}

size_t
FilterMap::ReadUnsetBatch(FilterMapCursor& cursor,
                          std::span<int32_t> output) const {
    if (output.empty()) {
        return 0;
    }

    const auto& storage = GetStorage();
    if (const auto* sparse =
            std::get_if<SparseExceptionBitmapRep>(&storage.value);
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
    auto* dense = std::get_if<DenseBitmapRep>(&storage.value);
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
    auto* sparse = std::get_if<SparseExceptionBitmapRep>(&storage.value);
    if (sparse == nullptr) {
        return;
    }

    auto bitmap =
        std::make_shared<TargetBitmap>(storage.universe, sparse->default_bit);
    for (const auto id : sparse->exceptions) {
        bitmap->set(static_cast<size_t>(id), !sparse->default_bit);
    }
    storage.value = DenseBitmapRep{std::move(bitmap)};
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
