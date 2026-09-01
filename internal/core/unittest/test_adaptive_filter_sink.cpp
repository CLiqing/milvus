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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "common/Types.h"
#include "exec/operator/AdaptiveFilterSink.h"

namespace milvus::exec {
namespace {

template <typename T>
concept HasAdaptiveFilterSinkStats = requires(const T& sink) { sink.stats(); };

static_assert(!HasAdaptiveFilterSinkStats<AdaptiveFilterSink<false>>);
static_assert(HasAdaptiveFilterSinkStats<AdaptiveFilterSink<true>>);
static_assert(sizeof(AdaptiveFilterSink<false>) <
              sizeof(AdaptiveFilterSink<true>));

TargetBitmap
MakeBitmap(size_t size, std::initializer_list<size_t> set_bits) {
    TargetBitmap bitmap(size, false);
    for (const auto bit : set_bits) {
        EXPECT_LT(bit, size);
        bitmap.set(bit);
    }
    return bitmap;
}

void
ExpectSparseIds(const SparseFilterResult& result,
                std::initializer_list<int32_t> expected) {
    ASSERT_TRUE(result.IsSparse());
    ASSERT_FALSE(result.IsDense());
    ASSERT_NE(result.accepted_ids, nullptr);
    EXPECT_EQ(*result.accepted_ids, std::vector<int32_t>(expected));
}

void
ExpectDenseAccepted(const SparseFilterResult& result,
                    std::initializer_list<size_t> accepted) {
    ASSERT_TRUE(result.IsDense());
    ASSERT_FALSE(result.IsSparse());
    ASSERT_NE(result.filtered, nullptr);

    std::vector<bool> expected(result.universe, true);
    for (const auto id : accepted) {
        ASSERT_LT(id, expected.size());
        expected[id] = false;
    }
    ASSERT_EQ(result.filtered->size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ((*result.filtered)[i], expected[i]) << "row " << i;
    }
}

TEST(AdaptiveFilterSinkTest, ProductionDefaultCompilesOutStats) {
    AdaptiveFilterSink sink(/*universe=*/8, /*sparse_cap=*/2);
    auto predicate = MakeBitmap(8, {1, 6});
    sink.ConsumeBatch(TargetBitmapView(predicate), /*batch_offset=*/0);
    ExpectSparseIds(sink.Finish(), {1, 6});
}

TEST(AdaptiveFilterSinkTest, KeepsTMinusOneAndExactlyTSparse) {
    {
        AdaptiveFilterSink<true> sink(/*universe=*/10, /*sparse_cap=*/3);
        auto predicate = MakeBitmap(10, {1, 7});
        sink.ConsumeBatch(TargetBitmapView(predicate), /*batch_offset=*/0);
        auto result = sink.Finish();

        ExpectSparseIds(result, {1, 7});
        EXPECT_EQ(result.universe, 10);
        EXPECT_EQ(sink.stats().processed_rows, 10);
        EXPECT_EQ(sink.stats().ids_appended, 2);
        EXPECT_EQ(sink.stats().switch_count, 0);
        EXPECT_EQ(sink.stats().dense_allocations, 0);
    }

    {
        AdaptiveFilterSink<true> sink(/*universe=*/10, /*sparse_cap=*/3);
        auto predicate = MakeBitmap(10, {0, 4, 9});
        sink.ConsumeBatch(TargetBitmapView(predicate), /*batch_offset=*/0);
        auto result = sink.Finish();

        ExpectSparseIds(result, {0, 4, 9});
        EXPECT_EQ(sink.stats().ids_appended, 3);
        EXPECT_EQ(sink.stats().switch_count, 0);
    }
}

TEST(AdaptiveFilterSinkTest, TPlusOneSwitchWritesTriggeringBatchOnce) {
    AdaptiveFilterSink<true> sink(/*universe=*/12, /*sparse_cap=*/3);
    auto predicate = MakeBitmap(12, {1, 3, 5, 11});
    sink.ConsumeBatch(TargetBitmapView(predicate), /*batch_offset=*/0);
    auto result = sink.Finish();

    ExpectDenseAccepted(result, {1, 3, 5, 11});
    EXPECT_EQ(sink.stats().processed_rows, 12);
    // The first T candidates are appended once, then discarded in O(1).
    // None is point-backfilled because the triggering batch is bulk-written.
    EXPECT_EQ(sink.stats().ids_appended, 3);
    EXPECT_EQ(sink.stats().ids_discarded_on_switch, 3);
    EXPECT_EQ(sink.stats().backfill_count, 0);
    EXPECT_EQ(sink.stats().switch_count, 1);
    EXPECT_EQ(sink.stats().dense_allocations, 1);
    EXPECT_EQ(sink.stats().dense_words_initialized, 1);
    EXPECT_EQ(sink.stats().dense_batch_writes, 1);
    EXPECT_EQ(sink.stats().dense_words_written, 1);
}

TEST(AdaptiveFilterSinkTest, MultiBatchSwitchBackfillsOnlyCompletePrefix) {
    AdaptiveFilterSink<true> sink(/*universe=*/15, /*sparse_cap=*/3);

    auto first = MakeBitmap(5, {1, 4});
    sink.ConsumeBatch(TargetBitmapView(first), /*batch_offset=*/0);

    auto triggering = MakeBitmap(6, {0, 2});
    sink.ConsumeBatch(TargetBitmapView(triggering), /*batch_offset=*/5);

    auto dense_tail = MakeBitmap(4, {1, 3});
    sink.ConsumeBatch(TargetBitmapView(dense_tail), /*batch_offset=*/11);
    auto result = sink.Finish();

    ExpectDenseAccepted(result, {1, 4, 5, 7, 12, 14});
    EXPECT_EQ(sink.stats().processed_rows, 15);
    EXPECT_EQ(sink.stats().ids_appended, 3);
    EXPECT_EQ(sink.stats().ids_discarded_on_switch, 1);
    EXPECT_EQ(sink.stats().backfill_count, 2);
    EXPECT_EQ(sink.stats().switch_count, 1);
    EXPECT_EQ(sink.stats().dense_batch_writes, 2);
    EXPECT_EQ(sink.stats().dense_words_written, 2);
}

TEST(AdaptiveFilterSinkTest, NullableRowsAreRejectedDuringDenseSwitch) {
    AdaptiveFilterSink<true> sink(/*universe=*/8, /*sparse_cap=*/1);
    auto predicate = MakeBitmap(8, {0, 1, 2, 6});
    auto validity = MakeBitmap(8, {0, 2, 3, 4, 5, 6, 7});

    sink.ConsumeBatch(TargetBitmapView(predicate),
                      TargetBitmapView(validity),
                      /*batch_offset=*/0);
    auto result = sink.Finish();

    // Row 1 has predicate=true but validity=false and must be filtered.
    ExpectDenseAccepted(result, {0, 2, 6});
    EXPECT_EQ(sink.stats().backfill_count, 0);
    EXPECT_EQ(sink.stats().switch_count, 1);
}

TEST(AdaptiveFilterSinkTest, HandlesEmptyBatchAndCrossWordTail) {
    constexpr size_t kUniverse = 70;
    AdaptiveFilterSink<true> sink(kUniverse, /*sparse_cap=*/2);

    TargetBitmap empty(0, false);
    sink.ConsumeBatch(TargetBitmapView(empty), /*batch_offset=*/0);

    auto head = MakeBitmap(63, {0});
    sink.ConsumeBatch(TargetBitmapView(head), /*batch_offset=*/0);

    auto tail = MakeBitmap(7, {1, 6});
    sink.ConsumeBatch(TargetBitmapView(tail), /*batch_offset=*/63);
    auto result = sink.Finish();

    ExpectDenseAccepted(result, {0, 64, 69});
    EXPECT_EQ(sink.stats().processed_rows, kUniverse);
    EXPECT_EQ(sink.stats().ids_appended, 2);
    EXPECT_EQ(sink.stats().ids_discarded_on_switch, 1);
    EXPECT_EQ(sink.stats().backfill_count, 1);
    EXPECT_EQ(sink.stats().dense_words_initialized, 2);
    EXPECT_EQ(sink.stats().dense_batch_writes, 1);
    EXPECT_EQ(sink.stats().dense_words_written, 2);
}

TEST(AdaptiveFilterSinkTest, AllTrueBatchHonorsCapAndAllFalseStaysSparse) {
    {
        AdaptiveFilterSink<true> sink(/*universe=*/65, /*sparse_cap=*/64);
        TargetBitmap predicate(65, true);
        sink.ConsumeBatch(TargetBitmapView(predicate), /*batch_offset=*/0);
        auto result = sink.Finish();

        ASSERT_TRUE(result.IsDense());
        EXPECT_TRUE(result.filtered->none());
        EXPECT_EQ(sink.stats().switch_count, 1);
        EXPECT_EQ(sink.stats().dense_words_written, 2);
    }

    {
        AdaptiveFilterSink<true> sink(/*universe=*/65, /*sparse_cap=*/64);
        TargetBitmap predicate(65, false);
        sink.ConsumeBatch(TargetBitmapView(predicate), /*batch_offset=*/0);
        auto result = sink.Finish();

        ExpectSparseIds(result, {});
        EXPECT_EQ(sink.stats().ids_appended, 0);
        EXPECT_EQ(sink.stats().switch_count, 0);
    }
}

}  // namespace
}  // namespace milvus::exec
