// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include "exec/operator/groupby/GroupMembership.h"
#include "exec/operator/groupby/SearchGroupByOperator.h"
#include "index/ScalarIndexSort.h"
#include "test_utils/DataGen.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec {

TEST(GroupByMapTest, StrictTracksLockedGroupsAndRemainingQuota) {
    GroupByMap<int64_t> groups(/*group_capacity=*/2,
                               /*group_size=*/3,
                               /*strict_group_size=*/true);

    EXPECT_TRUE(groups.Push(7));
    EXPECT_TRUE(groups.Push(7));
    EXPECT_TRUE(groups.Push(7));
    EXPECT_TRUE(groups.IsGroupFull(7));
    EXPECT_EQ(groups.GetRemainingGroupSize(7), 0);
    EXPECT_EQ(groups.GetEnoughGroupCount(), 1);
    EXPECT_FALSE(groups.IsGroupCapacityReached());
    EXPECT_FALSE(groups.IsGroupResEnough());

    EXPECT_TRUE(groups.Push(8));
    EXPECT_TRUE(groups.IsGroupCapacityReached());
    EXPECT_FALSE(groups.IsGroupResEnough());
    EXPECT_EQ(groups.GetGroupResultCount(8), 1);
    EXPECT_EQ(groups.GetRemainingGroupSize(8), 2);

    EXPECT_FALSE(groups.Push(9));
    EXPECT_FALSE(groups.Contains(9));
    EXPECT_EQ(groups.GetGroupCount(), 2);

    EXPECT_TRUE(groups.Push(8));
    EXPECT_TRUE(groups.Push(8));
    EXPECT_TRUE(groups.IsGroupResEnough());
    EXPECT_EQ(groups.GetEnoughGroupCount(), 2);
    EXPECT_FALSE(groups.Push(7));

    ASSERT_EQ(groups.GetGroupOrder().size(), 2);
    EXPECT_EQ(groups.GetGroupOrder()[0], std::optional<int64_t>(7));
    EXPECT_EQ(groups.GetGroupOrder()[1], std::optional<int64_t>(8));
}

TEST(GroupByMapTest, NonStrictStopsWhenCapacityIsReached) {
    GroupByMap<int64_t> groups(/*group_capacity=*/2,
                               /*group_size=*/3,
                               /*strict_group_size=*/false);

    EXPECT_TRUE(groups.Push(1));
    EXPECT_TRUE(groups.Push(1));
    EXPECT_FALSE(groups.IsGroupResEnough());
    EXPECT_TRUE(groups.Push(2));
    EXPECT_TRUE(groups.IsGroupResEnough());
    EXPECT_EQ(groups.GetRemainingGroupSize(1), 1);
    EXPECT_EQ(groups.GetRemainingGroupSize(2), 2);
}

TEST(GroupByMapTest, NullIsTrackedAsAStableGroupKey) {
    GroupByMap<int64_t> groups(/*group_capacity=*/2,
                               /*group_size=*/2,
                               /*strict_group_size=*/true);

    EXPECT_TRUE(groups.Push(std::nullopt));
    EXPECT_TRUE(groups.Contains(std::nullopt));
    EXPECT_EQ(groups.GetGroupResultCount(std::nullopt), 1);
    ASSERT_EQ(groups.GetGroupOrder().size(), 1);
    EXPECT_FALSE(groups.GetGroupOrder()[0].has_value());
}

TEST(GroupByResultCollectorTest, SortsAndAppendsByMetric) {
    {
        GroupByResultCollector<int64_t> collector;
        collector.Add(10, 0.8F, 1);
        collector.Add(20, 0.2F, 2);

        std::vector<GroupByValueType> groups;
        std::vector<int64_t> offsets;
        std::vector<float> distances;
        collector.SortAndAppend(
            knowhere::metric::L2, groups, offsets, distances);

        EXPECT_EQ(offsets, (std::vector<int64_t>{20, 10}));
        EXPECT_EQ(distances, (std::vector<float>{0.2F, 0.8F}));
        EXPECT_EQ(groups.size(), 2);
        EXPECT_EQ(collector.Size(), 0);
    }

    {
        GroupByResultCollector<int64_t> collector;
        collector.Add(10, 0.8F, 1);
        collector.Add(20, 0.2F, 2);

        std::vector<GroupByValueType> groups;
        std::vector<int64_t> offsets;
        std::vector<float> distances;
        collector.SortAndAppend(
            knowhere::metric::IP, groups, offsets, distances);

        EXPECT_EQ(offsets, (std::vector<int64_t>{10, 20}));
        EXPECT_EQ(distances, (std::vector<float>{0.8F, 0.2F}));
        EXPECT_EQ(groups.size(), 2);
        EXPECT_EQ(collector.Size(), 0);
    }
}

TEST(GroupMembershipTest, ScalarIndexAndRawFieldProduceIdenticalMembership) {
    constexpr int64_t kRowCount = 120;
    auto schema = std::make_shared<Schema>();
    auto pk_field = schema->AddDebugField("pk", DataType::INT64);
    auto group_field =
        schema->AddDebugField("nullable_group", DataType::INT64, true);
    schema->set_primary_field_id(pk_field);
    auto data = segcore::DataGen(schema,
                                 kRowCount,
                                 /*seed=*/42,
                                 /*ts_offset=*/0,
                                 /*repeat_count=*/4);

    auto raw_segment = CreateSealedWithFieldDataLoaded(schema, data);
    auto index_segment =
        segcore::CreateSealedSegment(schema, empty_index_meta, 7001);
    LoadGeneratedDataIntoSegment(
        data, index_segment.get(), false, {group_field.get()});

    auto values = data.get_col<int64_t>(group_field);
    auto valid = data.get_col_valid(group_field);
    auto scalar_index = index::CreateScalarIndexSort<int64_t>();
    scalar_index->Build(kRowCount, values.data(), valid.data());
    segcore::LoadIndexInfo load_info;
    load_info.field_id = group_field.get();
    load_info.field_type = DataType::INT64;
    load_info.index_params = GenIndexParams(scalar_index.get());
    load_info.cache_index =
        CreateTestCacheIndex("group-membership", std::move(scalar_index));
    index_segment->LoadIndex(load_info);

    TargetBitmap base_filter(kRowCount, false);
    base_filter[1] = true;  // filtered null
    base_filter[4] = true;  // filtered value group
    base_filter[117] = true;
    std::vector<std::optional<int64_t>> groups{
        std::nullopt, values[0], values[4], values[20]};

    auto raw = BuildGroupMembership<int64_t>(
        nullptr, *raw_segment, group_field, kRowCount, groups, &base_filter);
    auto indexed = BuildGroupMembership<int64_t>(
        nullptr, *index_segment, group_field, kRowCount, groups, &base_filter);
    ASSERT_TRUE(raw.has_value());
    ASSERT_TRUE(indexed.has_value());
    EXPECT_EQ(raw->Source(), GroupMembershipSource::RawField);
    EXPECT_EQ(indexed->Source(), GroupMembershipSource::ScalarIndex);
    EXPECT_EQ(raw->EligibleRowCount(), kRowCount - base_filter.count());
    EXPECT_EQ(raw->EligibleRowCount(), indexed->EligibleRowCount());
    EXPECT_EQ(raw->GroupRowCounts(), indexed->GroupRowCounts());

    std::vector<std::optional<int64_t>> batch{
        std::nullopt, values[4], values[20]};
    auto raw_bitmap = raw->BuildMembership(batch);
    auto index_bitmap = indexed->BuildMembership(batch);
    ASSERT_TRUE(raw_bitmap.has_value());
    ASSERT_TRUE(index_bitmap.has_value());
    ASSERT_EQ(raw_bitmap->size(), index_bitmap->size());
    for (size_t i = 0; i < raw_bitmap->size(); ++i) {
        EXPECT_EQ((*raw_bitmap)[i], (*index_bitmap)[i]) << "offset " << i;
        if (base_filter[i]) {
            EXPECT_FALSE((*raw_bitmap)[i]);
        }
    }
}

TEST(GroupMembershipTest, RawStringBoolAndNullGroupsRespectBaseFilter) {
    constexpr int64_t kRowCount = 40;
    auto schema = std::make_shared<Schema>();
    auto pk_field = schema->AddDebugField("pk", DataType::INT64);
    auto string_field =
        schema->AddDebugField("nullable_string", DataType::VARCHAR, true);
    auto bool_field = schema->AddDebugField("bool_group", DataType::BOOL);
    schema->set_primary_field_id(pk_field);
    auto data = segcore::DataGen(schema,
                                 kRowCount,
                                 /*seed=*/99,
                                 /*ts_offset=*/0,
                                 /*repeat_count=*/4);
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    TargetBitmap base_filter(kRowCount, false);
    base_filter[0] = true;
    base_filter[3] = true;
    base_filter[10] = true;

    auto strings = data.get_col<std::string>(string_field);
    auto valid = data.get_col_valid(string_field);
    std::vector<std::optional<std::string>> string_groups{
        std::nullopt, strings[0], strings[8]};
    auto string_membership = BuildGroupMembership<std::string>(nullptr,
                                                               *segment,
                                                               string_field,
                                                               kRowCount,
                                                               string_groups,
                                                               &base_filter);
    ASSERT_TRUE(string_membership.has_value());
    EXPECT_EQ(string_membership->Source(), GroupMembershipSource::RawField);
    auto string_bitmap = string_membership->BuildMembership(string_groups);
    ASSERT_TRUE(string_bitmap.has_value());

    std::vector<size_t> expected_counts(string_groups.size(), 0);
    for (size_t i = 0; i < kRowCount; ++i) {
        std::optional<std::string> value =
            valid[i] ? std::optional<std::string>(strings[i]) : std::nullopt;
        auto found =
            std::find(string_groups.begin(), string_groups.end(), value);
        auto expected = !base_filter[i] && found != string_groups.end();
        EXPECT_EQ((*string_bitmap)[i], expected) << "offset " << i;
        if (expected) {
            ++expected_counts[std::distance(string_groups.begin(), found)];
        }
    }
    EXPECT_EQ(string_membership->GroupRowCounts(), expected_counts);

    std::vector<std::optional<bool>> bool_groups{false, true};
    auto bool_membership = BuildGroupMembership<bool>(
        nullptr, *segment, bool_field, kRowCount, bool_groups, &base_filter);
    ASSERT_TRUE(bool_membership.has_value());
    EXPECT_EQ(bool_membership->GroupRowCounts()[0] +
                  bool_membership->GroupRowCounts()[1],
              kRowCount - base_filter.count());
    auto bool_bitmap = bool_membership->BuildMembership(bool_groups);
    ASSERT_TRUE(bool_bitmap.has_value());
    EXPECT_EQ(bool_bitmap->count(), kRowCount - base_filter.count());
}

TEST(GroupMembershipTest, RejectsMismatchedFilterSize) {
    auto schema = std::make_shared<Schema>();
    auto pk_field = schema->AddDebugField("pk", DataType::INT64);
    auto group_field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(pk_field);
    auto data = segcore::DataGen(schema, 10);
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    TargetBitmap wrong_size(9, false);

    auto membership = BuildGroupMembership<int64_t>(
        nullptr, *segment, group_field, 10, {0}, &wrong_size);
    EXPECT_FALSE(membership.has_value());
}

}  // namespace milvus::exec
