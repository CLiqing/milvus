// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "common/CustomBitset.h"

namespace milvus {

// Canonical filtered-bit representation used by query execution. A set bit
// means that the corresponding row is excluded.
using TargetBitmap = CustomBitset;
using TargetBitmapView = CustomBitsetView;
using TargetBitmapPtr = std::unique_ptr<TargetBitmap>;

// Search consumers need to know what operations a filter can provide
// efficiently, but must not branch on its physical list/Roaring/Dense
// representation.
enum class FilterCapability : uint8_t {
    RandomMembership,
    EnumerateOnly,
};

// Opaque cursor for storage-independent enumeration of logical zero bits.
// A zero bit is a valid row at the vector-search filter boundary, matching the
// established TargetBitmap contract. Callers must not interpret the cursor.
struct FilterMapCursor {
    size_t position = 0;
};

struct FilterMapBatchResult {
    size_t ids_appended = 0;
    size_t ids_discarded_on_promotion = 0;
    size_t backfill_count = 0;
    size_t dense_words_written = 0;
    bool promoted = false;
    bool dense_batch_written = false;
};

// Canonical cross-operator filter result. Public operations retain ordinary
// bitmap semantics: set(id) stores logical 1, reset(id) stores logical 0 and
// test(id) returns that logical bit. Sparse storage is an implementation
// detail represented as exceptions to a default bit. Once the exception cap
// is exceeded the object promotes itself to the existing Dense bitmap without
// replaying any predicate rows.
class FilterMap {
 public:
    FilterMap() = default;
    // Copies share immutable storage. Mutating operations detach on first
    // write, so cache hits and cross-operator handoff remain O(1).
    FilterMap(const FilterMap&) = default;
    FilterMap&
    operator=(const FilterMap&) = default;

    FilterMap(FilterMap&&) noexcept = default;
    FilterMap&
    operator=(FilterMap&&) noexcept = default;

    static FilterMap
    FromDense(std::shared_ptr<TargetBitmap> dense) {
        if (dense == nullptr) {
            throw std::invalid_argument("Dense FilterMap requires an owner");
        }
        FilterMap result;
        result.initialized_ = true;
        result.universe_ = dense->size();
        result.dense_ = std::move(dense);
        result.exception_cap_ = 0;
        return result;
    }

    static FilterMap
    Adaptive(size_t universe, bool default_bit, size_t exception_cap) {
        FilterMap result;
        result.initialized_ = true;
        result.universe_ = universe;
        result.default_bit_ = default_bit;
        result.exception_cap_ = exception_cap;
        result.exceptions_ = std::make_shared<std::vector<int32_t>>();
        result.exceptions_->reserve(std::min(universe, exception_cap));
        return result;
    }

    static FilterMap
    FromExceptions(size_t universe,
                   bool default_bit,
                   std::shared_ptr<const std::vector<int32_t>> exceptions,
                   size_t exception_cap = std::numeric_limits<size_t>::max()) {
        if (exceptions == nullptr) {
            throw std::invalid_argument(
                "Sparse FilterMap requires an exception owner");
        }
        if (exceptions->size() > exception_cap) {
            throw std::invalid_argument(
                "Sparse FilterMap exception count exceeds cap");
        }
        auto owned = std::make_shared<std::vector<int32_t>>(*exceptions);
        ValidateExceptions(universe, *owned, /*require_unique=*/true);
        FilterMap result;
        result.initialized_ = true;
        result.universe_ = universe;
        result.default_bit_ = default_bit;
        result.exception_cap_ = exception_cap;
        result.exceptions_ = std::move(owned);
        return result;
    }

    // Trusted producer boundary: adopts a structurally unique ID buffer
    // without copying or rebuilding a validation set. Keep this entry point
    // inside execution/index producers whose one-row/one-ID contract is tested.
    static FilterMap
    AdoptExceptions(size_t universe,
                    bool default_bit,
                    std::shared_ptr<std::vector<int32_t>> exceptions,
                    size_t exception_cap = std::numeric_limits<size_t>::max()) {
        if (exceptions == nullptr) {
            throw std::invalid_argument(
                "Sparse FilterMap requires an exception owner");
        }
        if (exceptions->size() > exception_cap) {
            throw std::invalid_argument(
                "Sparse FilterMap exception count exceeds cap");
        }
        ValidateExceptions(universe, *exceptions, /*require_unique=*/false);
        FilterMap result;
        result.initialized_ = true;
        result.universe_ = universe;
        result.default_bit_ = default_bit;
        result.exception_cap_ = exception_cap;
        result.exceptions_ = std::move(exceptions);
        result.peak_exception_count_ = result.exceptions_->size();
        return result;
    }

    static FilterMap
    FromUnsetIds(size_t universe,
                 std::shared_ptr<const std::vector<int32_t>> unset_ids,
                 size_t exception_cap = std::numeric_limits<size_t>::max()) {
        return FromExceptions(
            universe, true, std::move(unset_ids), exception_cap);
    }

    static FilterMap
    AdoptUnsetIds(size_t universe,
                  std::shared_ptr<std::vector<int32_t>> unset_ids,
                  size_t exception_cap = std::numeric_limits<size_t>::max()) {
        return AdoptExceptions(
            universe, true, std::move(unset_ids), exception_cap);
    }

    size_t
    size() const {
        return universe_;
    }

    bool
    IsInitialized() const {
        return initialized_;
    }

    bool
    IsDense() const {
        return dense_ != nullptr;
    }

    FilterCapability
    capability() const {
        if (!initialized_) {
            throw std::logic_error("uninitialized FilterMap has no capability");
        }
        if (IsDense()) {
            return FilterCapability::RandomMembership;
        }
        if (!default_bit_) {
            throw std::logic_error(
                "default-zero sparse FilterMap has no efficient unset-ID "
                "capability; call EnsureDense");
        }
        return FilterCapability::EnumerateOnly;
    }

    bool
    test(size_t id) const {
        CheckId(id);
        if (dense_ != nullptr) {
            return (*dense_)[id];
        }
        return ContainsException(static_cast<int32_t>(id)) ? !default_bit_
                                                           : default_bit_;
    }

    void
    set(size_t id) {
        set(id, true);
    }

    void
    reset(size_t id) {
        set(id, false);
    }

    void
    set(size_t id, bool value) {
        CheckId(id);
        if (dense_ != nullptr) {
            EnsureUniqueDense();
            dense_->set(id, value);
            return;
        }

        EnsureUniqueExceptions();
        const auto row_id = static_cast<int32_t>(id);
        auto it = std::find(exceptions_->begin(), exceptions_->end(), row_id);
        const bool is_exception = it != exceptions_->end();
        if (value == default_bit_) {
            if (is_exception) {
                exceptions_->erase(it);
            }
            return;
        }
        if (is_exception) {
            return;
        }
        if (exceptions_->size() == exception_cap_) {
            EnsureDense();
            dense_->set(id, value);
            ++promotion_count_;
            return;
        }
        exceptions_->push_back(row_id);
        peak_exception_count_ =
            std::max(peak_exception_count_, exceptions_->size());
    }

    size_t
    count() const {
        if (dense_ != nullptr) {
            return dense_->count();
        }
        return default_bit_ ? universe_ - exceptions_->size()
                            : exceptions_->size();
    }

    bool
    all() const {
        return count() == universe_;
    }

    bool
    none() const {
        return count() == 0;
    }

    // Assigns one logical batch without exposing the current backend. When
    // validity is supplied, source bits are first ANDed with it; invert then
    // maps those source bits to their logical complement. This lets predicate
    // kernels retain their batch-local SIMD bitmap while FilterMap owns the
    // exception cap, T+1 promotion, prefix backfill and Dense continuation.
    FilterMapBatchResult
    AssignBatch(TargetBitmapView source,
                const TargetBitmapView* validity,
                size_t offset,
                bool invert) {
        if (validity != nullptr && validity->size() != source.size()) {
            throw std::invalid_argument(
                "FilterMap source and validity batch sizes differ");
        }
        if (offset > universe_ || source.size() > universe_ - offset) {
            throw std::out_of_range("FilterMap batch exceeds universe");
        }

        FilterMapBatchResult result;
        if (dense_ != nullptr) {
            EnsureUniqueDense();
            WriteDenseBatch(source, validity, offset, invert);
            result.dense_batch_written = true;
            result.dense_words_written =
                DestinationWordsTouched(offset, source.size());
            return result;
        }

        using Policy = TargetBitmapView::policy_type;
        using Word = TargetBitmapView::data_type;
        constexpr size_t kWordBits = sizeof(Word) * 8;
        const auto retained_prefix = exceptions_->size();
        EnsureUniqueExceptions();
        bool exceeds_cap = false;
        for (size_t word_offset = 0; word_offset < source.size();
             word_offset += kWordBits) {
            const auto bits = std::min(kWordBits, source.size() - word_offset);
            Word logical = Policy::op_read(
                source.data(), source.offset() + word_offset, bits);
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
            Word exceptions = default_bit_
                                  ? static_cast<Word>(~logical) & used_mask
                                  : logical;
            while (exceptions != 0) {
                const auto bit =
                    static_cast<size_t>(std::countr_zero(exceptions));
                if (exceptions_->size() == exception_cap_) {
                    exceeds_cap = true;
                    break;
                }
                exceptions_->push_back(
                    static_cast<int32_t>(offset + word_offset + bit));
                ++result.ids_appended;
                exceptions &= exceptions - 1;
            }
            if (exceeds_cap) {
                break;
            }
        }

        if (!exceeds_cap) {
            peak_exception_count_ =
                std::max(peak_exception_count_, exceptions_->size());
            return result;
        }

        result.ids_discarded_on_promotion =
            exceptions_->size() - retained_prefix;
        exceptions_->resize(retained_prefix);
        result.backfill_count = retained_prefix;
        EnsureDense();
        ++promotion_count_;
        result.promoted = true;
        WriteDenseBatch(source, validity, offset, invert);
        result.dense_batch_written = true;
        result.dense_words_written =
            DestinationWordsTouched(offset, source.size());
        return result;
    }

    // Copies logical zero IDs into output and advances cursor. Sparse
    // EnumerateOnly backends preserve producer order; callers must not assume
    // sorting. Dense enumeration is a correctness fallback, not the optimized
    // Dense consumer path.
    size_t
    ReadUnsetBatch(FilterMapCursor& cursor, std::span<int32_t> output) const {
        if (output.empty()) {
            return 0;
        }
        size_t written = 0;
        if (dense_ == nullptr && default_bit_) {
            while (cursor.position < exceptions_->size() &&
                   written < output.size()) {
                output[written++] = (*exceptions_)[cursor.position++];
            }
            return written;
        }
        while (cursor.position < universe_ && written < output.size()) {
            const auto id = cursor.position++;
            if (!test(id)) {
                output[written++] = static_cast<int32_t>(id);
            }
        }
        return written;
    }

    // Optional zero-copy enumeration fast path. EnumerateOnly storage used by
    // the current producer owns accepted/unset IDs contiguously; consumers
    // must fall back to ReadUnsetBatch when a future backend cannot expose a
    // stable span.
    std::optional<std::span<const int32_t>>
    TryGetUnsetIdsView() const {
        if (dense_ == nullptr && default_bit_ && exceptions_ != nullptr) {
            return std::span<const int32_t>(*exceptions_);
        }
        return std::nullopt;
    }

    // Applies a Dense logical-1 mask (for example MVCC invalid rows) using OR
    // semantics. A sparse default-1 map compacts only its zero exceptions;
    // other sparse polarity is materialized internally before the bulk op.
    void
    InplaceOr(TargetBitmapView mask) {
        if (mask.size() != universe_) {
            throw std::invalid_argument("FilterMap OR universe mismatch");
        }
        if (dense_ == nullptr && default_bit_) {
            EnsureUniqueExceptions();
            auto out = exceptions_->begin();
            for (auto it = exceptions_->begin(); it != exceptions_->end();
                 ++it) {
                const auto id = static_cast<size_t>(*it);
                if (!mask[id]) {
                    *out++ = *it;
                }
            }
            exceptions_->erase(out, exceptions_->end());
            return;
        }
        EnsureDense();
        dense_->inplace_or(mask, universe_);
    }

    // Read-only Dense owners are shared across FilterMap copies.  Accept the
    // owning bitmap directly so callers do not need to cast away const merely
    // to use it as an OR source.
    void
    InplaceOr(const TargetBitmap& mask) {
        if (mask.size() != universe_) {
            throw std::invalid_argument("FilterMap OR universe mismatch");
        }
        if (dense_ == nullptr && default_bit_) {
            EnsureUniqueExceptions();
            auto out = exceptions_->begin();
            for (auto it = exceptions_->begin(); it != exceptions_->end();
                 ++it) {
                const auto id = static_cast<size_t>(*it);
                if (!mask[id]) {
                    *out++ = *it;
                }
            }
            exceptions_->erase(out, exceptions_->end());
            return;
        }
        EnsureDense();
        dense_->inplace_or(mask, universe_);
    }

    TargetBitmap&
    EnsureDense() {
        if (dense_ == nullptr) {
            if (!initialized_ || exceptions_ == nullptr) {
                throw std::logic_error(
                    "cannot materialize an uninitialized FilterMap");
            }
            dense_ = std::make_shared<TargetBitmap>(universe_, default_bit_);
            for (const auto id : *exceptions_) {
                dense_->set(static_cast<size_t>(id), !default_bit_);
            }
            exceptions_.reset();
            ++dense_materialization_count_;
        }
        return *dense_;
    }

    const TargetBitmap*
    DenseData() const {
        return dense_.get();
    }

    std::shared_ptr<const TargetBitmap>
    DenseOwner() const {
        return dense_;
    }

    FilterMap
    Clone() const {
        return FilterMap(*this);
    }

    size_t
    StorageBytes() const {
        if (dense_ != nullptr) {
            return (universe_ + 7) / 8;
        }
        return exceptions_->size() * sizeof(int32_t);
    }

    // Storage-neutral immutable snapshot of logical zero IDs. This is used by
    // tests and compatibility adapters only; the production cross-repository
    // interface uses ReadUnsetBatch and does not expose a vector payload.
    std::shared_ptr<const std::vector<int32_t>>
    SnapshotUnsetIds() const {
        auto ids = std::make_shared<std::vector<int32_t>>();
        ids->reserve(universe_ - count());
        FilterMapCursor cursor;
        std::array<int32_t, 256> batch;
        while (true) {
            const auto n = ReadUnsetBatch(cursor, batch);
            if (n == 0) {
                break;
            }
            ids->insert(ids->end(), batch.begin(), batch.begin() + n);
        }
        return ids;
    }

    size_t
    promotion_count() const {
        return promotion_count_;
    }

    size_t
    peak_exception_count() const {
        return peak_exception_count_;
    }

    size_t
    dense_materialization_count() const {
        return dense_materialization_count_;
    }

    size_t
    detach_copy_count() const {
        return detach_copy_count_;
    }

 private:
    static size_t
    DestinationWordsTouched(size_t offset, size_t size) {
        constexpr size_t kWordBits = sizeof(TargetBitmapView::data_type) * 8;
        if (size == 0) {
            return 0;
        }
        return (offset % kWordBits + size + kWordBits - 1) / kWordBits;
    }

    void
    WriteDenseBatch(TargetBitmapView source,
                    const TargetBitmapView* validity,
                    size_t offset,
                    bool invert) {
        if (validity != nullptr) {
            source.inplace_and(*validity, source.size());
        }
        if (invert) {
            source.flip();
        }
        TargetBitmapView::policy_type::op_copy(source.data(),
                                               source.offset(),
                                               dense_->data(),
                                               offset,
                                               source.size());
    }

    void
    CheckId(size_t id) const {
        if (id >= universe_ ||
            id > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::out_of_range("FilterMap row ID is outside universe");
        }
    }

    bool
    ContainsException(int32_t id) const {
        return std::find(exceptions_->begin(), exceptions_->end(), id) !=
               exceptions_->end();
    }

    static void
    ValidateExceptions(size_t universe,
                       const std::vector<int32_t>& exceptions,
                       bool require_unique) {
        std::unordered_set<int32_t> unique;
        if (require_unique) {
            unique.reserve(exceptions.size());
        }
        for (const auto id : exceptions) {
            if (id < 0 || static_cast<size_t>(id) >= universe) {
                throw std::out_of_range(
                    "Sparse FilterMap exception is outside universe");
            }
            if (require_unique && !unique.insert(id).second) {
                throw std::invalid_argument(
                    "Sparse FilterMap exceptions must be unique");
            }
        }
    }

    void
    EnsureUniqueExceptions() {
        if (exceptions_ != nullptr && !exceptions_.unique()) {
            exceptions_ = std::make_shared<std::vector<int32_t>>(*exceptions_);
            ++detach_copy_count_;
        }
    }

    void
    EnsureUniqueDense() {
        if (dense_ != nullptr && !dense_.unique()) {
            dense_ = std::make_shared<TargetBitmap>(dense_->clone());
            ++detach_copy_count_;
        }
    }

    bool initialized_ = false;
    size_t universe_ = 0;
    bool default_bit_ = false;
    size_t exception_cap_ = 0;
    std::shared_ptr<std::vector<int32_t>> exceptions_;
    std::shared_ptr<TargetBitmap> dense_;
    size_t promotion_count_ = 0;
    size_t peak_exception_count_ = 0;
    size_t dense_materialization_count_ = 0;
    size_t detach_copy_count_ = 0;
};

}  // namespace milvus
