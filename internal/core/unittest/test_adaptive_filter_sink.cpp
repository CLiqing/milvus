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

#include <array>
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
ExpectSparseIds(const FilterMap& result,
                std::initializer_list<int32_t> expected) {
    ASSERT_EQ(result.capability(), FilterCapability::EnumerateOnly);
    ASSERT_FALSE(result.IsDense());
    EXPECT_EQ(*result.SnapshotUnsetIds(), std::vector<int32_t>(expected));
}

void
ExpectDenseAccepted(const FilterMap& result,
                    std::initializer_list<size_t> accepted) {
    ASSERT_TRUE(result.IsDense());
    ASSERT_EQ(result.capability(), FilterCapability::RandomMembership);
    ASSERT_NE(result.DenseData(), nullptr);

    std::vector<bool> expected(result.size(), true);
    for (const auto id : accepted) {
        ASSERT_LT(id, expected.size());
        expected[id] = false;
    }
    ASSERT_EQ(result.DenseData()->size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ((*result.DenseData())[i], expected[i]) << "row " << i;
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
        EXPECT_EQ(result.size(), 10);
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
        EXPECT_TRUE(result.DenseData()->none());
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

TEST(FilterMapTest, PreservesBitmapSemanticsForBothDefaultBits) {
    auto default_zero = FilterMap::Adaptive(/*universe=*/8,
                                            /*default_bit=*/false,
                                            /*exception_cap=*/4);
    EXPECT_TRUE(default_zero.none());
    default_zero.set(6);
    default_zero.set(2);
    default_zero.set(6);  // Duplicate writes are idempotent.
    EXPECT_TRUE(default_zero.test(2));
    EXPECT_TRUE(default_zero.test(6));
    EXPECT_EQ(default_zero.count(), 2);
    default_zero.reset(2);
    EXPECT_FALSE(default_zero.test(2));
    EXPECT_EQ(default_zero.count(), 1);

    auto default_one = FilterMap::Adaptive(/*universe=*/8,
                                           /*default_bit=*/true,
                                           /*exception_cap=*/4);
    EXPECT_TRUE(default_one.all());
    default_one.reset(5);
    default_one.reset(1);
    default_one.reset(5);
    EXPECT_FALSE(default_one.test(1));
    EXPECT_FALSE(default_one.test(5));
    EXPECT_EQ(default_one.count(), 6);
    default_one.set(1);
    EXPECT_TRUE(default_one.test(1));
    EXPECT_EQ(default_one.count(), 7);
    EXPECT_EQ(*default_one.SnapshotUnsetIds(), (std::vector<int32_t>{5}));
}

TEST(FilterMapTest, PromotesAtTPlusOneAndBackfillsWithoutSemanticChange) {
    auto map = FilterMap::Adaptive(/*universe=*/10,
                                   /*default_bit=*/true,
                                   /*exception_cap=*/2);
    map.reset(8);
    map.reset(2);
    EXPECT_FALSE(map.IsDense());
    EXPECT_EQ(map.peak_exception_count(), 2);

    map.reset(6);
    ASSERT_TRUE(map.IsDense());
    EXPECT_EQ(map.promotion_count(), 1);
    EXPECT_EQ(map.count(), 7);
    for (size_t id = 0; id < map.size(); ++id) {
        EXPECT_EQ(map.test(id), id != 2 && id != 6 && id != 8) << id;
    }
    EXPECT_EQ(*map.SnapshotUnsetIds(), (std::vector<int32_t>{2, 6, 8}));
}

TEST(FilterMapTest, CursorAndMvccOrPreserveSparseResult) {
    auto ids = std::make_shared<const std::vector<int32_t>>(
        std::initializer_list<int32_t>{7, 1, 5, 3});
    auto map = FilterMap::FromUnsetIds(/*universe=*/9,
                                       ids,
                                       /*exception_cap=*/4);

    FilterMapCursor cursor;
    std::array<int32_t, 2> batch{};
    std::vector<int32_t> observed;
    while (const auto n = map.ReadUnsetBatch(cursor, batch)) {
        observed.insert(observed.end(), batch.begin(), batch.begin() + n);
    }
    EXPECT_EQ(observed, *ids);

    // MVCC/delete masks use logical 1=filtered. Invalidating rows 1 and 8
    // removes only row 1 from the sparse valid-row exceptions.
    auto invalid = MakeBitmap(9, {1, 8});
    map.InplaceOr(TargetBitmapView(invalid));
    EXPECT_EQ(*map.SnapshotUnsetIds(), (std::vector<int32_t>{7, 5, 3}));
    EXPECT_FALSE(map.IsDense());
}

TEST(FilterMapTest, CopyOnWritePreservesCachedSparseOwner) {
    auto ids = std::make_shared<std::vector<int32_t>>(
        std::initializer_list<int32_t>{7, 1, 5});
    auto cached = FilterMap::AdoptUnsetIds(/*universe=*/9, ids);
    auto query = cached;

    query.reset(3);
    EXPECT_EQ(query.detach_copy_count(), 1);
    EXPECT_EQ(*query.SnapshotUnsetIds(), (std::vector<int32_t>{7, 1, 5, 3}));
    EXPECT_EQ(*cached.SnapshotUnsetIds(), (std::vector<int32_t>{7, 1, 5}));
}

TEST(FilterMapTest, EnsureDenseIsOneWayIdempotentAndCopyOnWriteSafe) {
    auto ids = std::make_shared<std::vector<int32_t>>(
        std::initializer_list<int32_t>{4, 2});
    auto cached = FilterMap::AdoptUnsetIds(/*universe=*/8, ids);
    auto query = cached;

    auto* first = &query.EnsureDense();
    auto* second = &query.EnsureDense();
    EXPECT_EQ(first, second);
    EXPECT_EQ(query.dense_materialization_count(), 1);
    EXPECT_FALSE(query.test(2));
    EXPECT_FALSE(query.test(4));
    EXPECT_FALSE(cached.IsDense());

    auto dense_copy = query;
    dense_copy.set(2);
    EXPECT_EQ(dense_copy.detach_copy_count(), 1);
    EXPECT_TRUE(dense_copy.test(2));
    EXPECT_FALSE(query.test(2));
}

TEST(FilterMapTest, PublicFactoryRejectsDuplicateAndOutOfRangeIds) {
    auto duplicate = std::make_shared<const std::vector<int32_t>>(
        std::initializer_list<int32_t>{1, 1});
    EXPECT_THROW(FilterMap::FromUnsetIds(/*universe=*/4, duplicate),
                 std::invalid_argument);

    auto out_of_range = std::make_shared<const std::vector<int32_t>>(
        std::initializer_list<int32_t>{4});
    EXPECT_THROW(FilterMap::FromUnsetIds(/*universe=*/4, out_of_range),
                 std::out_of_range);
}

TEST(FilterMapTest, DefaultZeroSparseMustMaterializeBeforeHandoff) {
    auto map = FilterMap::Adaptive(/*universe=*/8,
                                   /*default_bit=*/false,
                                   /*exception_cap=*/4);
    map.set(3);
    EXPECT_THROW((void)map.capability(), std::logic_error);
    map.EnsureDense();
    EXPECT_EQ(map.capability(), FilterCapability::RandomMembership);
    EXPECT_TRUE(map.test(3));
}

TEST(FilterMapTest, DefaultConstructedMapIsRejectedByCapabilityBoundary) {
    FilterMap map;
    EXPECT_FALSE(map.IsInitialized());
    EXPECT_THROW((void)map.capability(), std::logic_error);
    EXPECT_THROW((void)map.EnsureDense(), std::logic_error);
}

}  // namespace
}  // namespace milvus::exec
