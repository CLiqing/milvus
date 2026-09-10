// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include <gtest/gtest.h>
#include <array>
#include "query/FilterMapView.h"
#include "common/BitsetView.h"

namespace milvus {
TEST(FilterMapViewTest, OwnsSnapshotAndIndependentCursors) {
    auto map = FilterMap::Adaptive(129, true, 4);
    const std::array<int32_t, 4> ids{128, 3, 65, 0};
    map.AppendUniqueBits(ids, false);
    auto view = query::MakeFilterMapView(map);
    map.set(128);
    size_t a = 0, b = 0;
    std::array<int32_t, 2> out;
    ASSERT_EQ(view.read_unset(a, out.data(), out.size()), 2);
    EXPECT_EQ(out, (std::array<int32_t, 2>{128, 3}));
    ASSERT_EQ(view.read_unset(b, out.data(), out.size()), 2);
    EXPECT_EQ(out, (std::array<int32_t, 2>{128, 3}));
    // Legacy materialization must not invalidate either active cursor.
    EXPECT_NE(view.data(), nullptr);
    EXPECT_FALSE(view.test(128));
    EXPECT_TRUE(view.test(127));
    ASSERT_EQ(view.read_unset(a, out.data(), out.size()), 2);
    EXPECT_EQ(out, (std::array<int32_t, 2>{65, 0}));
    EXPECT_EQ(view.read_unset(a, out.data(), out.size()), 0);
    EXPECT_EQ(view.count(), 125);
}

TEST(FilterMapViewTest, FullCountKeepsEnumerationButSubrangeUsesDense) {
    auto map = FilterMap::Adaptive(129, true, 2);
    const std::array<int32_t, 2> ids{0, 128};
    map.AppendUniqueBits(ids, false);
    auto view = query::MakeFilterMapView(map);
    view.count_filtered_bits(0, 129);
    EXPECT_TRUE(view.enumerable());
    EXPECT_EQ(view.count(), 127);
    auto partial = view;
    partial.count_filtered_bits(64, 65);
    EXPECT_FALSE(partial.enumerable());
    EXPECT_EQ(partial.count(), 64);
    EXPECT_EQ(partial.size(), 65);
    EXPECT_TRUE(view.enumerable());
}

TEST(FilterMapViewTest, ExplicitDenseAndOffsetAreSafeCompatibilityBoundaries) {
    auto map = FilterMap::Adaptive(129, true, 2);
    map.reset(65);
    auto view = query::MakeFilterMapView(map);
    view.set_id_offset(64);
    view.set_vector_count(65);
    EXPECT_FALSE(view.enumerable());
    EXPECT_FALSE(view.test(1));
    EXPECT_TRUE(view.test(0));
    view.EnsureDense();
    const auto* data = view.data();
    view.EnsureDense();
    EXPECT_EQ(view.data(), data);
}
TEST(FilterMapViewTest, DenseSubviewRetainsOwnerAfterBaseClassCopy) {
    std::weak_ptr<std::vector<uint8_t>> lifetime;
    auto make_view = [&]() -> knowhere::BitsetView {
        auto owner = std::make_shared<std::vector<uint8_t>>(17, 0xff);
        (*owner)[65 >> 3] &= ~(uint8_t{1} << (65 & 7));
        (*owner)[128 >> 3] &= ~(uint8_t{1} << (128 & 7));
        lifetime = owner;
        auto view = knowhere::BitsetView::FromEnumerable(
            owner,
            owner.get(),
            129,
            127,
            [](const void*, size_t& cursor, int32_t* ids, size_t capacity) {
                const int32_t source[] = {65, 128};
                const auto count = std::min(capacity, size_t{2} - cursor);
                std::copy_n(source + cursor, count, ids);
                cursor += count;
                return count;
            },
            [](const void* p) {
                return static_cast<const std::vector<uint8_t>*>(p)->data();
            });
        BitsetView parent(std::move(view));
        return parent.subview(64, 65);
    };
    auto view = make_view();
    EXPECT_FALSE(lifetime.expired());
    EXPECT_FALSE(view.enumerable());
    EXPECT_FALSE(view.test(1));
    EXPECT_FALSE(view.test(64));
    EXPECT_TRUE(view.test(0));
    view = {};
    EXPECT_TRUE(lifetime.expired());
}
}  // namespace milvus
