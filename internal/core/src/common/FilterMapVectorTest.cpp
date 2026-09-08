// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include "common/Vector.h"
#include <array>
#include <gtest/gtest.h>

namespace milvus {

TEST(FilterMapVectorTest, UniqueDenseHandoffDoesNotCopyBytes) {
    TargetBitmap bitmap(129, false);
    bitmap.set(64);
    const auto* data = bitmap.data();
    ColumnVector column(std::move(bitmap), TargetBitmap(129, true));
    const auto& map = column.GetFilterMap();
    EXPECT_EQ(map.count(), 1);
    EXPECT_TRUE(map.test(64));
    EXPECT_EQ(column.GetRawData(), data);
    EXPECT_EQ(column.GetRawData(), data);
    EXPECT_TRUE(column.GetFilterMap().test(64));
}

TEST(FilterMapVectorTest, LegacyWritesDoNotModifyPublishedSnapshot) {
    TargetBitmap bitmap(129, false);
    bitmap.set(64);
    ColumnVector column(std::move(bitmap), TargetBitmap(129, true));
    auto snapshot = column.GetFilterMap();
    TargetBitmapView legacy(column.GetRawData(), column.size());
    legacy.set(65);
    EXPECT_FALSE(snapshot.test(65));
    EXPECT_TRUE(snapshot.test(64));
    EXPECT_TRUE(column.GetFilterMap().test(65));
}

TEST(FilterMapVectorTest, SparseCompatibilityPreservesNullAndFilteredBits) {
    auto map = FilterMap::Adaptive(129, true, 3);
    const std::array<int32_t, 3> accepted{128, 2, 64};
    map.AppendUniqueBits(accepted, false);
    auto validity = TargetBitmap(129, true);
    validity.reset(3);
    ColumnVector column(std::move(map), std::move(validity));
    auto snapshot = column.GetFilterMap();
    EXPECT_EQ(snapshot.capability(), FilterMapCapability::EnumerateOnly);
    TargetBitmapView legacy(column.GetRawData(), column.size());
    for (size_t id = 0; id < 129; ++id) {
        EXPECT_EQ(legacy[id], id != 128 && id != 2 && id != 64);
    }
    EXPECT_FALSE(column.ValidAt(3));
    // Simulate an existing visibility consumer excluding one previously valid
    // row. The actual MVCC/delete/TTL operators remain on their existing path.
    legacy.set(64);
    EXPECT_FALSE(snapshot.test(64));
    EXPECT_TRUE(column.GetFilterMap().test(64));
}

TEST(FilterMapVectorTest, TakeDenseConsumesMapAndInvalidatesCursor) {
    auto map = FilterMap::Adaptive(129, true, 2);
    map.reset(64);
    FilterMapCursor cursor;
    std::array<int32_t, 1> batch;
    ASSERT_EQ(map.ReadUnsetBatch(cursor, batch), 1);
    auto snapshot = map;
    auto dense = std::move(map).TakeDense();
    EXPECT_FALSE(map.IsInitialized());
    EXPECT_THROW(map.ReadUnsetBatch(cursor, batch), std::logic_error);
    dense.set(64);
    EXPECT_FALSE(snapshot.test(64));
}

TEST(FilterMapVectorTest, InvertedHandoffAndBitmapHelpers) {
    auto map = FilterMap::Adaptive(65, false, 2);
    map.flip();
    ColumnVector column(std::move(map), TargetBitmap(65, true));
    EXPECT_TRUE(column.AllTrue());
    EXPECT_FALSE(column.AllFalse());
    TargetBitmapView legacy(column.GetRawData(), column.size());
    legacy.reset();
    EXPECT_TRUE(column.AllFalse());
    EXPECT_EQ(column.GetFilterMap().count(), 0);
}

}  // namespace milvus
