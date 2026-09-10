// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include "common/FilterMapConfig.h"
#include "common/FilterMap.h"
#include "exec/QueryContext.h"

#include <gtest/gtest.h>
#include <limits>

namespace milvus {

TEST(FilterMapConfigTest, DefaultsPreserveDenseAndRatioScalesWithUniverse) {
    FilterMapConfig config;
    EXPECT_FALSE(config.ExceptionCap(1000000).has_value());
    config.enabled = true;
    EXPECT_FALSE(config.ExceptionCap(49999).has_value());
    EXPECT_EQ(config.ExceptionCap(50000), 200);
    EXPECT_EQ(config.ExceptionCap(1000000), 4000);
    EXPECT_EQ(config.ExceptionCap(3000000), 12000);
    EXPECT_EQ(config.ExceptionCap(50001), 200);
}

TEST(FilterMapConfigTest, RejectsInvalidSettingsIncludingWhenDisabled) {
    for (double ratio : {-0.1, 1.1,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
        FilterMapConfig config{false, 0, ratio};
        EXPECT_FALSE(config.IsValid());
        EXPECT_THROW(config.ExceptionCap(65), std::invalid_argument);
    }
    EXPECT_FALSE((FilterMapConfig{true, -1, 0.004}).IsValid());
    EXPECT_EQ((FilterMapConfig{true, 0, 0}).ExceptionCap(65), 0);
    EXPECT_EQ((FilterMapConfig{true, 0, 1}).ExceptionCap(65), 65);
}

TEST(FilterMapConfigTest, QueryConfigSnapshotsValidatedDefaults) {
    struct Restore {
        FilterMapConfig saved = GetDefaultFilterMapConfig();
        ~Restore() {
            SetDefaultFilterMapConfig(saved);
        }
    } restore;
    ASSERT_TRUE(SetDefaultFilterMapConfig({false, 50000, 0.004}));
    exec::QueryConfig before;
    ASSERT_TRUE(SetDefaultFilterMapConfig({true, 0, 0.5}));
    exec::QueryConfig after;
    EXPECT_FALSE(before.filter_map_config().ExceptionCap(65).has_value());
    EXPECT_EQ(after.filter_map_config().ExceptionCap(65), 32);
    EXPECT_FALSE(SetDefaultFilterMapConfig({false, -1, 0.1}));
    EXPECT_EQ(GetDefaultFilterMapConfig().ExceptionCap(65), 32);
}

TEST(FilterMapConfigTest, BatchIngestionFoldsNullAndPromotesAtRatioBoundary) {
    const FilterMapConfig config{true, 0, 0.004};
    constexpr size_t n = 1000;
    for (size_t accepted : {size_t(0), size_t(4), size_t(5), n}) {
        auto map = FilterMap::Adaptive(n, true, *config.ExceptionCap(n));
        for (size_t offset = 0; offset < n; offset += 65) {
            const auto size = std::min(size_t(65), n - offset);
            TargetBitmap batch(size, false), valid(size, true);
            for (size_t i = 0; i < size; ++i) {
                batch.set(i, offset + i < accepted);
            }
            // A TRUE+NULL is not accepted; it must not consume the sparse cap.
            if (offset == 65 && accepted <= 5) {
                batch.set(0);
                valid.reset(0);
            }
            TargetBitmapView view(batch), validity(valid);
            map.AssignBitmapBatch(view, &validity, offset, true);
        }
        EXPECT_EQ(map.count(), n - accepted);
        EXPECT_EQ(map.capability(), accepted <= 4
                                        ? FilterMapCapability::EnumerateOnly
                                        : FilterMapCapability::RandomMembership);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(map.test(i), i >= accepted);
        }
    }
}
}  // namespace milvus
