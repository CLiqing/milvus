// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include "common/FilterMap.h"

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <memory>
#include <vector>

namespace milvus {
namespace {

TargetBitmap
MakeBitmap(size_t size, std::initializer_list<size_t> set_bits) {
    TargetBitmap bitmap(size, false);
    for (const auto bit : set_bits) {
        bitmap.set(bit);
    }
    return bitmap;
}

std::vector<int32_t>
CollectUnset(const FilterMap& map) {
    std::vector<int32_t> result;
    std::array<int32_t, 3> batch{};
    FilterMapCursor cursor;
    while (const auto count = map.ReadUnsetBatch(cursor, batch)) {
        result.insert(result.end(), batch.begin(), batch.begin() + count);
    }
    return result;
}

TEST(FilterMapTest, PreservesBitmapSemanticsForBothDefaultBits) {
    auto zero = FilterMap::Adaptive(8, false, 4);
    zero.set(6);
    zero.set(2);
    zero.set(6);
    EXPECT_EQ(zero.count(), 2);
    EXPECT_TRUE(zero.test(2));
    EXPECT_TRUE(zero.test(6));
    zero.reset(2);
    EXPECT_EQ(zero.count(), 1);
    EXPECT_FALSE(zero.test(2));

    auto one = FilterMap::Adaptive(8, true, 4);
    one.reset(5);
    one.reset(1);
    one.reset(5);
    EXPECT_EQ(one.count(), 6);
    EXPECT_FALSE(one.test(1));
    EXPECT_FALSE(one.test(5));
    one.set(1);
    EXPECT_EQ(one.count(), 7);
    EXPECT_EQ(CollectUnset(one), (std::vector<int32_t>{5}));
}

TEST(FilterMapTest, KeepsTExceptionsAndPromotesAtTPlusOne) {
    auto map = FilterMap::Adaptive(10, true, 2);
    map.reset(8);
    map.reset(2);
    EXPECT_EQ(map.capability(), FilterMapCapability::EnumerateOnly);

    map.reset(6);
    EXPECT_EQ(map.capability(), FilterMapCapability::RandomMembership);
    EXPECT_EQ(map.count(), 7);
    for (size_t id = 0; id < map.size(); ++id) {
        EXPECT_EQ(map.test(id), id != 2 && id != 6 && id != 8) << id;
    }
}

TEST(FilterMapTest, BatchPromotionBackfillsOnlyCompletedPrefix) {
    auto map = FilterMap::Adaptive(15, true, 3);

    auto first = MakeBitmap(5, {1, 4});
    map.AssignBatch(TargetBitmapView(first), nullptr, 0, true);
    EXPECT_EQ(map.capability(), FilterMapCapability::EnumerateOnly);

    auto triggering = MakeBitmap(6, {0, 2});
    map.AssignBatch(TargetBitmapView(triggering), nullptr, 5, true);
    EXPECT_EQ(map.capability(), FilterMapCapability::RandomMembership);

    auto tail = MakeBitmap(4, {1, 3});
    map.AssignBatch(TargetBitmapView(tail), nullptr, 11, true);

    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{1, 4, 5, 7, 12, 14}));
}

TEST(FilterMapTest, BatchCombinesValidityBeforeInversion) {
    auto map = FilterMap::Adaptive(8, true, 1);
    auto predicate = MakeBitmap(8, {0, 1, 2, 6});
    auto validity = MakeBitmap(8, {0, 2, 3, 4, 5, 6, 7});
    TargetBitmapView validity_view(validity);

    map.AssignBatch(TargetBitmapView(predicate), &validity_view, 0, true);
    EXPECT_EQ(map.capability(), FilterMapCapability::RandomMembership);
    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{0, 2, 6}));
}

TEST(FilterMapTest, SparseCopyDetachesOnMutation) {
    auto original = FilterMap::Adaptive(9, true, 4);
    original.reset(7);
    original.reset(1);
    auto copy = original;

    copy.reset(5);
    EXPECT_EQ(CollectUnset(copy), (std::vector<int32_t>{7, 1, 5}));
    EXPECT_EQ(CollectUnset(original), (std::vector<int32_t>{7, 1}));
}

TEST(FilterMapTest, EnsureDenseIsIdempotentAndCopyOnWriteSafe) {
    auto original = FilterMap::Adaptive(8, true, 4);
    original.reset(4);
    original.reset(2);
    auto copy = original;

    auto* first = &copy.EnsureDense();
    auto* second = &copy.EnsureDense();
    EXPECT_EQ(first, second);
    EXPECT_FALSE(copy.test(2));
    EXPECT_FALSE(copy.test(4));
    EXPECT_EQ(original.capability(), FilterMapCapability::EnumerateOnly);

    auto dense_copy = copy;
    dense_copy.set(2);
    EXPECT_TRUE(dense_copy.test(2));
    EXPECT_FALSE(copy.test(2));
}

TEST(FilterMapTest, DenseOwnerDetachesBeforeMutation) {
    auto owner = std::make_shared<TargetBitmap>(6, false);
    owner->set(1);
    auto map = FilterMap::FromDense(owner);

    map.set(3);
    EXPECT_TRUE(map.test(3));
    EXPECT_FALSE((*owner)[3]);
}

TEST(FilterMapTest, RejectsInvalidStateAndBounds) {
    FilterMap empty;
    EXPECT_FALSE(empty.IsInitialized());
    EXPECT_THROW((void)empty.size(), std::logic_error);
    EXPECT_THROW((void)empty.EnsureDense(), std::logic_error);

    auto map = FilterMap::Adaptive(4, false, 2);
    EXPECT_THROW(map.set(4), std::out_of_range);
    EXPECT_THROW((void)map.capability(), std::logic_error);
    EXPECT_THROW((void)FilterMap::Adaptive(4, false, 5), std::invalid_argument);
}

}  // namespace
}  // namespace milvus
