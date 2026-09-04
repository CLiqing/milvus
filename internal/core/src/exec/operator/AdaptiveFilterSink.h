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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "common/FilterBitmap.h"

namespace milvus::exec {

// Logical access counters for switch-point attribution.  These counters are
// deliberately independent of wall time: callers can combine them with unit
// costs measured from the same production implementation.
struct AdaptiveFilterSinkStats {
    uint64_t processed_rows = 0;
    // Actual vector append operations, including IDs from the triggering
    // batch that are discarded when that batch switches the sink to Dense.
    uint64_t ids_appended = 0;
    uint64_t ids_discarded_on_switch = 0;

    // A switch allocates and initializes the complete Dense universe once.
    uint64_t dense_allocations = 0;
    uint64_t dense_words_initialized = 0;

    // A Dense batch write is one bulk copy.  dense_words_written is the
    // logical number of 64-bit destination words touched by those copies;
    // overlapping boundary words in separate batches are counted separately.
    uint64_t dense_batch_writes = 0;
    uint64_t dense_words_written = 0;

    // Accepted IDs retained by complete earlier Sparse batches and reset in
    // the freshly initialized (all-filtered) Dense bitmap.
    uint64_t backfill_count = 0;
    uint64_t switch_count = 0;
};

struct AdaptiveFilterSinkNoStats {};

// Converts a stream of predicate batches into exactly one canonical filter
// representation.  Predicate data and validity use 1=true/valid semantics;
// the Dense output uses Milvus' established 1=filtered semantics.
//
// Batches must cover [0, universe) once, contiguously, in producer order.  In
// Sparse mode the sink keeps at most sparse_cap accepted row IDs.  If a batch
// contains the (sparse_cap + 1)-th accepted row, the sink initializes a final
// Dense bitmap, backfills only IDs retained from *earlier complete batches*,
// and writes the triggering batch as a whole.  Thus no predicate is rerun and
// no ID from the triggering batch is written twice.
//
// ConsumeBatch may mutate predicate_data after the sink switches to Dense:
// it turns the input view into 1=filtered form before its bulk copy.  Sparse
// batches remain untouched.  This makes the production path allocation-free
// per Dense batch while preserving an explicit ownership boundary for callers.
template <bool CollectStats = false>
class AdaptiveFilterSink {
 public:
    AdaptiveFilterSink(int64_t universe, int64_t sparse_cap)
        : universe_(universe), sparse_cap_(sparse_cap) {
        constexpr auto kMaxSparseId =
            static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        AssertInfo(universe_ >= 0 && universe_ <= kMaxSparseId,
                   "adaptive filter universe {} is outside the int32 row-ID "
                   "range",
                   universe_);
        AssertInfo(sparse_cap_ >= 0 && sparse_cap_ <= kMaxSparseId,
                   "adaptive filter Sparse cap {} is outside the int32 "
                   "row-ID range",
                   sparse_cap_);
        filter_map_ = FilterMap::Adaptive(static_cast<size_t>(universe_),
                                          /*default_bit=*/true,
                                          static_cast<size_t>(sparse_cap_));
    }

    AdaptiveFilterSink(const AdaptiveFilterSink&) = delete;
    AdaptiveFilterSink&
    operator=(const AdaptiveFilterSink&) = delete;
    AdaptiveFilterSink(AdaptiveFilterSink&&) = default;
    AdaptiveFilterSink&
    operator=(AdaptiveFilterSink&&) = default;

    // Nullable-column overload. A row is accepted only when both bits are 1.
    void
    ConsumeBatch(TargetBitmapView predicate_data,
                 TargetBitmapView valid_data,
                 int64_t batch_offset) {
        AssertInfo(predicate_data.size() == valid_data.size(),
                   "predicate batch size {} does not match validity size {}",
                   predicate_data.size(),
                   valid_data.size());
        ConsumeBatchImpl(predicate_data, &valid_data, batch_offset);
    }

    // Non-nullable overload: every predicate result bit is valid.
    void
    ConsumeBatch(TargetBitmapView predicate_data, int64_t batch_offset) {
        ConsumeBatchImpl(predicate_data, nullptr, batch_offset);
    }

    FilterMap
    Finish() {
        AssertInfo(!finished_, "adaptive filter sink was already finished");
        AssertInfo(next_offset_ == universe_,
                   "adaptive filter sink processed {} of {} rows",
                   next_offset_,
                   universe_);
        finished_ = true;

        return std::move(filter_map_);
    }

    const AdaptiveFilterSinkStats&
    stats() const
        requires(CollectStats)
    {
        return stats_;
    }

    bool
    IsDense() const {
        return filter_map_.IsDense();
    }

 private:
    void
    ValidateBatch(TargetBitmapView predicate_data, int64_t batch_offset) const {
        AssertInfo(!finished_,
                   "cannot add a batch to a finished adaptive filter sink");
        AssertInfo(batch_offset == next_offset_,
                   "adaptive filter batches must be contiguous: expected "
                   "offset {}, got {}",
                   next_offset_,
                   batch_offset);
        AssertInfo(batch_offset >= 0 && batch_offset <= universe_ &&
                       predicate_data.size() <=
                           static_cast<size_t>(universe_ - batch_offset),
                   "predicate batch [{}, {}) exceeds universe {}",
                   batch_offset,
                   batch_offset + static_cast<int64_t>(predicate_data.size()),
                   universe_);
    }

    void
    ConsumeBatchImpl(TargetBitmapView predicate_data,
                     const TargetBitmapView* valid_data,
                     int64_t batch_offset) {
        ValidateBatch(predicate_data, batch_offset);
        const auto batch_size = predicate_data.size();
        if constexpr (CollectStats) {
            stats_.processed_rows += batch_size;
        }

        const auto batch_result =
            filter_map_.AssignBatch(predicate_data,
                                    valid_data,
                                    static_cast<size_t>(batch_offset),
                                    /*invert=*/true);
        if constexpr (CollectStats) {
            stats_.ids_appended += batch_result.ids_appended;
            stats_.ids_discarded_on_switch +=
                batch_result.ids_discarded_on_promotion;
            stats_.backfill_count += batch_result.backfill_count;
            if (batch_result.promoted) {
                ++stats_.dense_allocations;
                stats_.dense_words_initialized +=
                    (static_cast<uint64_t>(universe_) + 63) / 64;
                ++stats_.switch_count;
            }
            if (batch_result.dense_batch_written) {
                ++stats_.dense_batch_writes;
                stats_.dense_words_written += batch_result.dense_words_written;
            }
        }
        next_offset_ += static_cast<int64_t>(batch_size);
    }

 private:
    int64_t universe_;
    int64_t sparse_cap_;
    int64_t next_offset_ = 0;
    bool finished_ = false;
    FilterMap filter_map_;
    [[no_unique_address]] std::conditional_t<CollectStats,
                                             AdaptiveFilterSinkStats,
                                             AdaptiveFilterSinkNoStats> stats_;
};

}  // namespace milvus::exec
