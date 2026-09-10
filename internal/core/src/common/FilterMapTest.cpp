// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include "common/FilterMap.h"

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
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
    map.AssignBitmapBatch(TargetBitmapView(first), nullptr, 0, true);
    EXPECT_EQ(map.capability(), FilterMapCapability::EnumerateOnly);

    auto triggering = MakeBitmap(6, {0, 2});
    map.AssignBitmapBatch(TargetBitmapView(triggering), nullptr, 5, true);
    EXPECT_EQ(map.capability(), FilterMapCapability::RandomMembership);

    auto tail = MakeBitmap(4, {1, 3});
    map.AssignBitmapBatch(TargetBitmapView(tail), nullptr, 11, true);

    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{1, 4, 5, 7, 12, 14}));
}

TEST(FilterMapTest, BatchCombinesValidityBeforeInversion) {
    auto map = FilterMap::Adaptive(8, true, 1);
    auto predicate = MakeBitmap(8, {0, 1, 2, 6});
    auto validity = MakeBitmap(8, {0, 2, 3, 4, 5, 6, 7});
    TargetBitmapView validity_view(validity);

    map.AssignBitmapBatch(TargetBitmapView(predicate), &validity_view, 0, true);
    EXPECT_EQ(map.capability(), FilterMapCapability::RandomMembership);
    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{0, 2, 6}));
}

TEST(FilterMapTest, BitmapBatchHandlesUnalignedViewsAndTail) {
    auto map = FilterMap::Adaptive(10, true, 10);
    auto predicate = MakeBitmap(20, {4, 7, 11, 12});
    auto validity = MakeBitmap(24, {6, 9, 13, 17});
    TargetBitmapView predicate_view(predicate.data(), 3, 10);
    TargetBitmapView validity_view(validity.data(), 5, 10);

    map.AssignBitmapBatch(predicate_view, &validity_view, 0, true);
    EXPECT_EQ(map.capability(), FilterMapCapability::EnumerateOnly);
    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{1, 4, 8}));
}

TEST(FilterMapTest, BitmapBatchSupportsBothPolarities) {
    auto source = MakeBitmap(7, {1, 5});

    auto default_zero = FilterMap::Adaptive(7, false, 7);
    default_zero.AssignBitmapBatch(TargetBitmapView(source), nullptr, 0, false);
    EXPECT_EQ(default_zero.count(), 2);
    EXPECT_TRUE(default_zero.test(1));
    EXPECT_TRUE(default_zero.test(5));

    auto default_one = FilterMap::Adaptive(7, true, 7);
    default_one.AssignBitmapBatch(TargetBitmapView(source), nullptr, 0, false);
    EXPECT_EQ(default_one.count(), 2);
    for (size_t id = 0; id < default_one.size(); ++id) {
        EXPECT_EQ(default_one.test(id), id == 1 || id == 5) << id;
    }
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

TEST(FilterMapTest, AppendsUnorderedUniqueBitsWithoutLookup) {
    auto map = FilterMap::Adaptive(12, true, 4);
    const std::array<int32_t, 3> accepted{9, 1, 7};
    map.AppendUniqueBits(accepted, false);

    EXPECT_EQ(map.capability(), FilterMapCapability::EnumerateOnly);
    EXPECT_EQ(CollectUnset(map),
              (std::vector<int32_t>{accepted.begin(), accepted.end()}));
    EXPECT_EQ(map.count(), 9);
}

TEST(FilterMapTest, UniqueBitBatchPromotesAndWritesOnlyRemainingSuffix) {
    auto map = FilterMap::Adaptive(12, true, 3);
    const std::array<int32_t, 2> first{8, 2};
    const std::array<int32_t, 3> triggering{10, 1, 6};
    map.AppendUniqueBits(first, false);
    map.AppendUniqueBits(triggering, false);

    EXPECT_EQ(map.capability(), FilterMapCapability::RandomMembership);
    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{1, 2, 6, 8, 10}));
    EXPECT_EQ(map.count(), 7);
}

TEST(FilterMapTest, UniqueBitBatchValidatesBeforeMutation) {
    auto map = FilterMap::Adaptive(6, true, 4);
    const std::array<int32_t, 3> invalid{1, 6, 3};
    EXPECT_THROW(map.AppendUniqueBits(invalid, false), std::out_of_range);
    EXPECT_EQ(map.count(), 6);
    EXPECT_TRUE(CollectUnset(map).empty());
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

TEST(FilterMapTest, DenseBorrowIsReadOnlyAndCopiesRemainIsolated) {
    auto map = FilterMap::Adaptive(129, true, 2);
    const auto& dense = map.EnsureDense();
    static_assert(
        std::is_same_v<decltype(map.EnsureDense()), const TargetBitmap&>);
    static_assert(
        std::is_pointer_v<decltype(dense.data())> &&
        std::is_const_v<std::remove_pointer_t<decltype(dense.data())>>);
    auto copy = map;
    map.reset(64);
    EXPECT_TRUE(copy.test(64));
    EXPECT_FALSE(map.test(64));
    EXPECT_EQ(copy.size(), 129);
    EXPECT_EQ(copy.count(), 129);
    // The borrowed reference is no longer used after mutation.
    (void)dense;
    auto owner = std::make_shared<TargetBitmap>(8, false);
    auto wrapped = FilterMap::FromDense(owner);
    EXPECT_EQ(&wrapped.EnsureDense(), owner.get());
    EXPECT_THROW(FilterMap::FromDense(nullptr), std::invalid_argument);
}

TEST(FilterMapTest, UniqueIdsDifferentialAcrossPolaritiesAndBatchBoundaries) {
    std::mt19937 random(20260907);
    for (size_t n : {0, 1, 64, 129, 1024}) {
        for (bool default_bit : {false, true}) {
            for (bool value : {false, true}) {
                for (size_t cap :
                     {size_t(0), std::min(n, size_t(1)), n / 2, n}) {
                    auto map = FilterMap::Adaptive(n, default_bit, cap);
                    std::vector<bool> expected(n, default_bit);
                    std::vector<int32_t> ids(n);
                    std::iota(ids.begin(), ids.end(), 0);
                    std::shuffle(ids.begin(), ids.end(), random);
                    for (size_t start = 0; start < n; start += 17) {
                        const auto count = std::min(n - start, size_t(17));
                        map.AppendUniqueBits(
                            std::span<const int32_t>(ids.data() + start, count),
                            value);
                        for (size_t i = start; i < start + count; ++i)
                            expected[ids[i]] = value;
                        size_t ones = 0;
                        for (size_t i = 0; i < n; ++i) {
                            ASSERT_EQ(map.test(i), expected[i]);
                            ones += expected[i];
                        }
                        ASSERT_EQ(map.count(), ones);
                    }
                    if (!default_bit)
                        map.EnsureDense();
                    EXPECT_EQ(CollectUnset(map).size(), value ? 0 : n);
                }
            }
        }
    }
}

TEST(FilterMapTest, ConstructionStateDetachesAndInvalidBatchDoesNotAdvance) {
    auto map = FilterMap::Adaptive(128, true, 4);
    auto first = MakeBitmap(64, {1});
    map.AssignBitmapBatch(TargetBitmapView(first), nullptr, 0, true);
    auto copy = map;
    copy.reset(5);
    auto second = MakeBitmap(64, {2});
    auto wrong_validity = MakeBitmap(63, {});
    TargetBitmapView validity(wrong_validity);
    EXPECT_THROW(
        map.AssignBitmapBatch(TargetBitmapView(second), &validity, 64, true),
        std::invalid_argument);
    EXPECT_THROW(
        map.AssignBitmapBatch(TargetBitmapView(second), nullptr, 65, true),
        std::out_of_range);
    EXPECT_NO_THROW(
        map.AssignBitmapBatch(TargetBitmapView(second), nullptr, 64, true));
    EXPECT_FALSE(map.test(66));
    EXPECT_TRUE(copy.test(66));
    EXPECT_EQ(CollectUnset(copy), (std::vector<int32_t>{1, 5}));
}

TEST(FilterMapTest, RejectsOverlappingOrMixedConstruction) {
    auto source = MakeBitmap(4, {1});
    const std::array<int32_t, 1> ids{2};
    for (size_t cap : {size_t(0), size_t(8)}) {
        auto bitmap = FilterMap::Adaptive(8, false, cap);
        bitmap.AssignBitmapBatch(TargetBitmapView(source), nullptr, 0, false);
        EXPECT_THROW(bitmap.AssignBitmapBatch(
                         TargetBitmapView(source), nullptr, 0, false),
                     std::logic_error);
        EXPECT_THROW(bitmap.AppendUniqueBits(ids, true), std::logic_error);

        auto native = FilterMap::Adaptive(8, false, cap);
        native.AppendUniqueBits(ids, true);
        EXPECT_THROW(native.AssignBitmapBatch(
                         TargetBitmapView(source), nullptr, 4, false),
                     std::logic_error);

        auto point = FilterMap::Adaptive(8, false, cap);
        point.set(1);
        EXPECT_THROW(point.AssignBitmapBatch(
                         TargetBitmapView(source), nullptr, 0, false),
                     std::logic_error);
        EXPECT_THROW(point.AppendUniqueBits(ids, true), std::logic_error);
        EXPECT_TRUE(point.test(1));

        auto finished = FilterMap::Adaptive(8, true, cap);
        finished.EnsureDense();
        EXPECT_THROW(finished.AppendUniqueBits(ids, false), std::logic_error);
        EXPECT_THROW(finished.AssignBitmapBatch(
                         TargetBitmapView(source), nullptr, 0, false),
                     std::logic_error);
    }
}

TEST(FilterMapTest,
     BitmapWordsDifferentialAcrossAlignmentPolarityAndPromotion) {
    std::mt19937 random(20260907);
    for (size_t n : {0, 1, 63, 64, 65, 127, 128, 129, 257}) {
        for (size_t source_offset : {0, 1, 63, 64}) {
            for (size_t valid_offset : {0, 1, 63, 64}) {
                for (bool default_bit : {false, true}) {
                    for (bool invert : {false, true}) {
                        for (bool validity : {false, true}) {
                            for (size_t cap : {size_t(0),
                                               std::min(n, size_t(1)),
                                               n / 2,
                                               n}) {
                                auto map =
                                    FilterMap::Adaptive(n, default_bit, cap);
                                std::vector<bool> expected(n, default_bit);
                                for (size_t offset = 0; offset < n;
                                     offset += 129) {
                                    const auto count =
                                        std::min(size_t(129), n - offset);
                                    TargetBitmap source(source_offset + count,
                                                        false);
                                    TargetBitmap valid(valid_offset + count,
                                                       false);
                                    for (size_t i = 0; i < count; ++i) {
                                        const bool s = random() % 3 == 0;
                                        const bool v = random() % 4 != 0;
                                        source.set(source_offset + i, s);
                                        valid.set(valid_offset + i, v);
                                        expected[offset + i] =
                                            (s && (!validity || v)) != invert;
                                    }
                                    TargetBitmapView sv(
                                        source.data(), source_offset, count);
                                    TargetBitmapView vv(
                                        valid.data(), valid_offset, count);
                                    map.AssignBitmapBatch(
                                        sv,
                                        validity ? &vv : nullptr,
                                        offset,
                                        invert);
                                    size_t ones = 0;
                                    for (size_t i = 0; i < n; ++i) {
                                        ASSERT_EQ(map.test(i), expected[i])
                                            << n << ":" << i;
                                        ones += expected[i];
                                    }
                                    ASSERT_EQ(map.count(), ones);
                                }
                                std::vector<int32_t> expected_ids;
                                for (size_t i = 0; i < n; ++i) {
                                    if (!expected[i])
                                        expected_ids.push_back(i);
                                }
                                if (!default_bit)
                                    map.EnsureDense();
                                auto ids = CollectUnset(map);
                                std::sort(ids.begin(), ids.end());
                                ASSERT_EQ(ids, expected_ids);
                                map.EnsureDense();
                                ASSERT_EQ(CollectUnset(map), expected_ids);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(FilterMapTest, DenseCursorSkipsWordsAndResumesInsideWords) {
    for (size_t n : {0, 1, 63, 64, 65, 127, 128, 129, 1025}) {
        auto map = FilterMap::Adaptive(n, true, 0);
        std::vector<int32_t> expected;
        for (size_t i = 0; i < n; ++i) {
            if (i % 67 == 0 || i % 67 == 1 || i == n - 1) {
                map.reset(i);
                expected.push_back(i);
            }
        }
        map.EnsureDense();
        for (size_t batch_size : {1, 2, 3, 64, 65}) {
            std::vector<int32_t> buffer(batch_size), actual;
            FilterMapCursor cursor;
            EXPECT_EQ(map.ReadUnsetBatch(cursor, {}), 0);
            EXPECT_EQ(cursor.position, 0);
            while (auto count = map.ReadUnsetBatch(cursor, buffer)) {
                actual.insert(
                    actual.end(), buffer.begin(), buffer.begin() + count);
            }
            EXPECT_EQ(actual, expected);
            EXPECT_EQ(map.ReadUnsetBatch(cursor, buffer), 0);
        }
    }
    auto zero = FilterMap::Adaptive(8, false, 8);
    FilterMapCursor cursor;
    std::array<int32_t, 3> output;
    EXPECT_THROW(zero.ReadUnsetBatch(cursor, output), std::logic_error);
    zero.EnsureDense();
    EXPECT_EQ(CollectUnset(zero).size(), 8);
}

TEST(FilterMapTest, FlipPreservesLogicalWritesAndSharedCopies) {
    for (size_t cap : {size_t(0), size_t(8)}) {
        auto map = FilterMap::Adaptive(8, false, cap);
        map.set(6);
        map.set(1);
        auto original = map;
        map.flip();
        EXPECT_EQ(map.count(), 6);
        EXPECT_EQ(original.count(), 2);
        EXPECT_EQ(
            CollectUnset(map),
            cap ? (std::vector<int32_t>{6, 1}) : (std::vector<int32_t>{1, 6}));
        map.set(6);
        map.reset(3);
        EXPECT_TRUE(map.test(6));
        EXPECT_FALSE(map.test(3));
        EXPECT_TRUE(original.test(6));
        EXPECT_FALSE(original.test(3));
        auto copy = map;
        const auto& dense = map.EnsureDense();
        for (size_t i = 0; i < 8; ++i) EXPECT_EQ(dense[i], copy.test(i));
        copy.flip();
        for (size_t i = 0; i < 8; ++i) EXPECT_NE(map.test(i), copy.test(i));
        copy.flip();
        EXPECT_EQ(copy.count(), map.count());
    }
}

TEST(FilterMapTest, CursorRejectsConversionMutationAndDifferentOwner) {
    auto map = FilterMap::Adaptive(16, true, 4);
    const std::array<int32_t, 3> ids{12, 2, 8};
    map.AppendUniqueBits(ids, false);
    FilterMapCursor cursor;
    std::array<int32_t, 1> out;
    EXPECT_EQ(map.ReadUnsetBatch(cursor, out), 1);
    EXPECT_EQ(out[0], 12);
    auto copy = map;
    EXPECT_THROW(copy.ReadUnsetBatch(cursor, out), std::logic_error);
    map.EnsureDense();
    EXPECT_THROW(map.ReadUnsetBatch(cursor, out), std::logic_error);
    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{2, 8, 12}));
    cursor = {};
    map.ReadUnsetBatch(cursor, out);
    map.reset(1);
    EXPECT_THROW(map.ReadUnsetBatch(cursor, out), std::logic_error);
    cursor = {};
    map.ReadUnsetBatch(cursor, out);
    map.flip();
    EXPECT_THROW(map.ReadUnsetBatch(cursor, out), std::logic_error);
    cursor = {};
    map.ReadUnsetBatch(cursor, out);
    map = copy;
    EXPECT_THROW(map.ReadUnsetBatch(cursor, out), std::logic_error);
    cursor = {};
    map.ReadUnsetBatch(cursor, out);
    auto moved = std::move(map);
    EXPECT_THROW(map.ReadUnsetBatch(cursor, out), std::logic_error);
    EXPECT_EQ(CollectUnset(moved), (std::vector<int32_t>{12, 2, 8}));
}

TEST(FilterMapTest, FlipClosesConstructionAndRespectsNullBits) {
    auto map = FilterMap::Adaptive(8, false, 8);
    auto predicate = MakeBitmap(8, {0, 1, 2, 3, 6});
    auto validity = MakeBitmap(8, {0, 2, 3, 4, 5, 7});
    TargetBitmapView valid(validity);
    map.AssignBitmapBatch(TargetBitmapView(predicate), &valid, 0, false);
    auto matched = map;
    map.flip();
    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{0, 2, 3}));
    EXPECT_EQ(matched.count(), 3);
    const std::array<int32_t, 1> ids{7};
    EXPECT_THROW(map.AppendUniqueBits(ids, false), std::logic_error);
    EXPECT_THROW(
        map.AssignBitmapBatch(TargetBitmapView(predicate), &valid, 0, true),
        std::logic_error);
    map.EnsureDense();
    EXPECT_TRUE(map.test(1));
    EXPECT_TRUE(map.test(6));
}

TEST(FilterMapTest, VisibilityOrPreservesEnumerableSnapshot) {
    auto map = FilterMap::Adaptive(129, true, 4);
    const std::array<int32_t, 4> ids{128, 0, 64, 31};
    map.AppendUniqueBits(ids, false);
    auto snapshot = map;
    FilterMapCursor cursor;
    std::array<int32_t, 1> out;
    map.ReadUnsetBatch(cursor, out);
    auto mask = MakeBitmap(129, {0, 31, 63});
    map.InplaceOr(TargetBitmapView(mask));
    EXPECT_EQ(map.capability(), FilterMapCapability::EnumerateOnly);
    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{128, 64}));
    EXPECT_EQ(CollectUnset(snapshot), (std::vector<int32_t>{128, 0, 64, 31}));
    EXPECT_THROW(map.ReadUnsetBatch(cursor, out), std::logic_error);
    EXPECT_EQ(map.count(), 127);
}

TEST(FilterMapTest, VisibilityOrHandlesBothPolaritiesAndDense) {
    for (bool default_bit : {false, true}) {
        for (bool invert : {false, true}) {
            for (size_t cap : {size_t{1}, size_t{10}}) {
                auto map = FilterMap::Adaptive(129, default_bit, cap);
                map.set(0, !default_bit);
                map.set(64, !default_bit);
                if (invert) {
                    map.flip();
                }
                std::vector<bool> expected(129);
                for (size_t i = 0; i < 129; ++i) {
                    expected[i] = map.test(i);
                }
                auto mask = MakeBitmap(129, {1, 64, 128});
                map.InplaceOr(TargetBitmapView(mask));
                for (size_t i = 0; i < 129; ++i) {
                    EXPECT_EQ(map.test(i), expected[i] || mask[i]);
                }
            }
        }
    }
}

TEST(FilterMapTest, VisibilityOrRejectsWrongUniverseWithoutMutation) {
    auto map = FilterMap::Adaptive(65, true, 1);
    map.reset(64);
    auto wrong = MakeBitmap(64, {0});
    EXPECT_THROW(map.InplaceOr(TargetBitmapView(wrong)), std::invalid_argument);
    EXPECT_EQ(CollectUnset(map), (std::vector<int32_t>{64}));
}

}  // namespace
}  // namespace milvus
