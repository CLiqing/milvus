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

#include <unordered_map>
#include <unordered_set>
#include <mutex>

#include "common/PrometheusClient.h"
#include "exec/operator/groupby/GroupMembership.h"
#include "exec/operator/groupby/SearchGroupByOperator.h"
#include "common/Consts.h"
#include "index/ScalarIndexSort.h"
#include "index/VectorMemIndex.h"
#include "exec/operator/Utils.h"
#include "monitor/Monitor.h"
#include "segcore/ChunkedSegmentSealedImpl.h"
#include "test_utils/DataGen.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec {

namespace {

class StrictGroupLogSink final : public google::LogSink {
 public:
    std::mutex mutex;
    std::vector<knowhere::Json> records;
    void
    send(google::LogSeverity,
         const char*,
         const char*,
         int,
         const std::tm*,
         const char* message,
         size_t length) override {
        const std::string text(message, length);
        const std::string marker = "strict_group_diagnostic ";
        auto pos = text.find(marker);
        if (pos != std::string::npos) {
            std::lock_guard<std::mutex> lock(mutex);
            records.push_back(
                knowhere::Json::parse(text.substr(pos + marker.size())));
        }
    }
};

class CountingScalarIndex : public index::ScalarIndexSort<int64_t> {
 public:
    size_t in_calls = 0;
    size_t in_values = 0;
    size_t null_calls = 0;

    const TargetBitmap
    In(size_t n, const int64_t* values) override {
        ++in_calls;
        in_values += n;
        return index::ScalarIndexSort<int64_t>::In(n, values);
    }

    const TargetBitmap
    IsNull() override {
        ++null_calls;
        return index::ScalarIndexSort<int64_t>::IsNull();
    }
};

class SequenceIterator final : public knowhere::IndexNode::iterator {
 public:
    explicit SequenceIterator(std::vector<std::pair<int64_t, float>> values)
        : values_(std::move(values)) {
    }

    std::pair<int64_t, float>
    Next() override {
        return values_.at(position_++);
    }

    bool
    HasNext() override {
        return position_ < values_.size();
    }

 private:
    std::vector<std::pair<int64_t, float>> values_;
    size_t position_{0};
};

std::shared_ptr<VectorIterator>
MakeSequenceVectorIterator(
    const std::vector<std::pair<int64_t, float>>& candidates,
    const BitsetView& invalid = {},
    bool empty_leading_chunk = false) {
    std::vector<std::pair<int64_t, float>> eligible;
    eligible.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (invalid.empty() || !invalid.test(candidate.first)) {
            eligible.emplace_back(candidate);
        }
    }
    auto iterator = std::make_shared<VectorIterator>(
        /*chunk_count=*/empty_leading_chunk ? 2 : 1,
        /*offset_mapping=*/nullptr);
    if (empty_leading_chunk) {
        iterator->AddIterator(std::make_shared<SequenceIterator>(
            std::vector<std::pair<int64_t, float>>{}));
    }
    iterator->AddIterator(std::make_shared<SequenceIterator>(eligible));
    iterator->seal();
    return iterator;
}

}  // namespace

TEST(VectorIteratorFilteredChunksTest, EmptyLeadingChunkKeepsSuccessors) {
    VectorIterator iterator(2, nullptr);
    iterator.AddIterator(std::make_shared<SequenceIterator>(
        std::vector<std::pair<int64_t, float>>{}));
    iterator.AddIterator(std::make_shared<SequenceIterator>(
        std::vector<std::pair<int64_t, float>>{{3, 1.0F}, {4, 2.0F}}));
    iterator.seal();
    ASSERT_TRUE(iterator.HasNext());
    EXPECT_EQ(iterator.Next()->first, 3);
    ASSERT_TRUE(iterator.HasNext());
    EXPECT_EQ(iterator.Next()->first, 4);
    EXPECT_FALSE(iterator.HasNext());
}

TEST(StrictGroupFilteredIteratorEligibilityTest,
     RequiresStrictMultiResultSingleQueryRowLevelSearch) {
    SearchInfo eligible;
    eligible.topk_ = 10;
    eligible.group_size_ = 3;
    eligible.strict_group_size_ = true;
    EXPECT_TRUE(query::CanUseStrictGroupFilteredIterator(eligible, 1));

    auto disabled = eligible;
    disabled.strict_group_acceptance_threshold_ = 0;
    EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(disabled, 1));

    auto non_strict = eligible;
    non_strict.strict_group_size_ = false;
    EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(non_strict, 1));

    auto single_result_group = eligible;
    single_result_group.group_size_ = 1;
    EXPECT_FALSE(
        query::CanUseStrictGroupFilteredIterator(single_result_group, 1));

    auto empty_topk = eligible;
    empty_topk.topk_ = 0;
    EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(empty_topk, 1));
    EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(eligible, 2));

    auto element_level = eligible;
    element_level.array_offsets_ = std::make_shared<ArrayOffsetsSealed>();
    EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(element_level, 1));
}

TEST(StrictGroupPhase2ExecutorTest,
     LocksGroupsAndRecreatesOneFilteredIterator) {
    constexpr int64_t kRowCount = 120;
    auto schema = std::make_shared<Schema>();
    auto pk_field = schema->AddDebugField("pk", DataType::INT64);
    auto group_field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(pk_field);
    auto data = segcore::DataGen(schema,
                                 kRowCount,
                                 /*seed=*/42,
                                 /*ts_offset=*/0,
                                 /*repeat_count=*/4);
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    auto group_values = data.get_col<int64_t>(group_field);

    std::unordered_map<int64_t, std::vector<int64_t>> rows_by_group;
    for (int64_t offset = 0; offset < kRowCount; ++offset) {
        rows_by_group[group_values[offset]].emplace_back(offset);
    }
    std::vector<int64_t> locked_groups;
    for (const auto& [group, rows] : rows_by_group) {
        if (rows.size() >= 3) {
            locked_groups.emplace_back(group);
            if (locked_groups.size() == 2) {
                break;
            }
        }
    }
    ASSERT_EQ(locked_groups.size(), 2);

    std::vector<std::pair<int64_t, float>> candidates;
    candidates.emplace_back(rows_by_group[locked_groups[0]][0], 0.0F);
    candidates.emplace_back(rows_by_group[locked_groups[1]][0], 1.0F);
    for (int64_t offset = 0; offset < kRowCount; ++offset) {
        if (group_values[offset] != locked_groups[0] &&
            group_values[offset] != locked_groups[1]) {
            candidates.emplace_back(offset,
                                    static_cast<float>(candidates.size()));
        }
    }
    for (auto group : locked_groups) {
        for (auto offset : rows_by_group[group]) {
            if (offset != candidates[0].first &&
                offset != candidates[1].first) {
                candidates.emplace_back(offset,
                                        static_cast<float>(candidates.size()));
            }
        }
    }

    SearchResult search_result;
    search_result.total_nq_ = 1;
    search_result.total_data_cnt_ = kRowCount;
    search_result.vector_iterators_ =
        std::vector<std::shared_ptr<VectorIterator>>{
            MakeSequenceVectorIterator(candidates)};

    int recreate_count = 0;
    TargetBitmap observed_filter;
    search_result.SetVectorIteratorRecreator(
        BitsetView{},
        [&](const BitsetView& invalid, SearchResult& recreated_result) {
            ++recreate_count;
            observed_filter = TargetBitmap(invalid.size(), false);
            for (size_t i = 0; i < invalid.size(); ++i) {
                observed_filter[i] = invalid.test(i);
            }
            recreated_result.vector_iterators_ =
                std::vector<std::shared_ptr<VectorIterator>>{
                    MakeSequenceVectorIterator(candidates, invalid, true)};
        });

    SearchInfo search_info;
    search_info.topk_ = 2;
    search_info.group_size_ = 3;
    search_info.strict_group_size_ = true;
    search_info.group_by_field_id_ = group_field;
    search_info.metric_type_ = knowhere::metric::L2;
    std::vector<GroupByValueType> output_groups;
    std::vector<int64_t> offsets;
    std::vector<float> distances;
    std::vector<size_t> prefix_sum;
    const auto before_candidates =
        milvus::monitor::internal_core_strict_group_phase2_phase1_candidates
            .Collect()
            .histogram;
    const auto before_latency =
        milvus::monitor::
            internal_core_strict_group_phase2_membership_build_latency.Collect()
                .histogram;
    const auto before_phase2 =
        milvus::monitor::internal_core_strict_group_phase2_phase2_candidates
            .Collect()
            .histogram;
    const auto before_bitmap =
        milvus::monitor::internal_core_strict_group_phase2_bitmap_build_latency
            .Collect()
            .histogram;
    const auto before_ratio =
        milvus::monitor::internal_core_strict_group_phase2_acceptance_ratio
            .Collect()
            .histogram;
    const auto started = std::chrono::steady_clock::now();
    SearchGroupBy(nullptr,
                  *search_result.vector_iterators_,
                  search_info,
                  output_groups,
                  *segment,
                  offsets,
                  distances,
                  prefix_sum,
                  &search_result);

    const auto elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    ASSERT_EQ(recreate_count, 1);
    ASSERT_EQ(observed_filter.size(), kRowCount);
    const auto after_candidates =
        milvus::monitor::internal_core_strict_group_phase2_phase1_candidates
            .Collect()
            .histogram;
    const auto after_latency =
        milvus::monitor::
            internal_core_strict_group_phase2_membership_build_latency.Collect()
                .histogram;
    EXPECT_EQ(after_candidates.sample_count - before_candidates.sample_count,
              1);
    EXPECT_EQ(after_candidates.sample_sum - before_candidates.sample_sum, 102);
    EXPECT_EQ(after_latency.sample_count - before_latency.sample_count, 1);
    EXPECT_GE(after_latency.sample_sum - before_latency.sample_sum, 0);
    EXPECT_LE(after_latency.sample_sum - before_latency.sample_sum, elapsed_ms);
    const auto after_phase2 =
        milvus::monitor::internal_core_strict_group_phase2_phase2_candidates
            .Collect()
            .histogram;
    const auto after_bitmap =
        milvus::monitor::internal_core_strict_group_phase2_bitmap_build_latency
            .Collect()
            .histogram;
    const auto after_ratio =
        milvus::monitor::internal_core_strict_group_phase2_acceptance_ratio
            .Collect()
            .histogram;
    EXPECT_EQ(after_phase2.sample_count - before_phase2.sample_count, 1);
    // Five candidates accept four rows; one belongs to a now-full group.
    EXPECT_EQ(after_phase2.sample_sum - before_phase2.sample_sum, 5);
    EXPECT_EQ(after_bitmap.sample_count - before_bitmap.sample_count, 1);
    EXPECT_LE(after_bitmap.sample_sum - before_bitmap.sample_sum, elapsed_ms);
    EXPECT_EQ(after_ratio.sample_count - before_ratio.sample_count, 1);
    EXPECT_EQ(after_ratio.sample_sum - before_ratio.sample_sum, 0);
    EXPECT_FALSE(search_result.CanRecreateVectorIterator());
    EXPECT_EQ(offsets.size(), 6);
    EXPECT_EQ(prefix_sum, (std::vector<size_t>{0, 6}));
    EXPECT_TRUE(observed_filter[candidates[0].first]);
    EXPECT_TRUE(observed_filter[candidates[1].first]);
    std::unordered_map<int64_t, size_t> output_counts;
    for (auto offset : offsets) {
        ++output_counts[group_values[offset]];
    }
    EXPECT_EQ(output_counts[locked_groups[0]], 3);
    EXPECT_EQ(output_counts[locked_groups[1]], 3);
    for (int64_t offset = 0; offset < kRowCount; ++offset) {
        auto belongs_to_locked_group =
            group_values[offset] == locked_groups[0] ||
            group_values[offset] == locked_groups[1];
        if (!belongs_to_locked_group) {
            EXPECT_TRUE(observed_filter[offset]);
        }
    }
}

TEST(StrictGroupPhase2ExecutorTest, EasyQuotaDoesNotRecreate) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto data = segcore::DataGen(schema, 1000, 42, 0, 1000);
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    auto values = data.get_col<int64_t>(field);
    std::vector<std::pair<int64_t, float>> candidates;
    for (int64_t i = 0; i < 1000; ++i) {
        if (values[i] == values[0]) {
            candidates.emplace_back(i, static_cast<float>(i));
        }
    }
    ASSERT_GE(candidates.size(), 2);
    SearchResult result;
    result.total_nq_ = 1;
    result.total_data_cnt_ = 1000;
    result.vector_iterators_ = std::vector<std::shared_ptr<VectorIterator>>{
        MakeSequenceVectorIterator(candidates)};
    auto filter_owner = std::make_shared<TargetBitmap>(1000, false);
    std::weak_ptr<TargetBitmap> weak_filter = filter_owner;
    result.vector_iterator_filter_owner_ = filter_owner;
    result.SetVectorIteratorRecreator(
        BitsetView(*filter_owner), [](auto&, auto&) {
            ADD_FAILURE() << "easy quota must use the original iterator";
        });
    EXPECT_EQ(result.vector_iterator_base_filter_, nullptr);
    filter_owner.reset();
    SearchInfo info;
    info.topk_ = 1;
    info.group_size_ = 2;
    info.strict_group_size_ = true;
    info.group_by_field_id_ = field;
    info.metric_type_ = knowhere::metric::L2;
    std::vector<GroupByValueType> groups;
    std::vector<int64_t> offsets;
    std::vector<float> distances;
    std::vector<size_t> prefix;
    SearchGroupBy(nullptr,
                  *result.vector_iterators_,
                  info,
                  groups,
                  *segment,
                  offsets,
                  distances,
                  prefix,
                  &result);
    EXPECT_EQ(offsets.size(), 2);
    EXPECT_TRUE(weak_filter.expired());
    EXPECT_EQ(result.vector_iterator_base_filter_, nullptr);
    EXPECT_FALSE(result.CanRecreateVectorIterator());
}

namespace {

void
CheckProbeScenario(
    const std::vector<int64_t>& labels,
    size_t candidate_count,
    int64_t topk,
    int64_t group_size,
    int expected_recreates,
    size_t expected_rows,
    std::optional<ErrorCode> recreate_error = std::nullopt,
    milvus::OpContext* op_ctx = nullptr,
    folly::CancellationSource* cancel_on_recreate = nullptr,
    std::optional<double> threshold = std::nullopt,
    std::optional<std::vector<int64_t>> recreated_offsets = std::nullopt,
    int64_t probe_budget = 100,
    StrictGroupStrategy strategy = StrictGroupStrategy::Sampling,
    bool debug = false) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto data = segcore::DataGen(schema, labels.size());
    for (auto& column : *data.raw_->mutable_fields_data()) {
        if (column.field_id() == field.get()) {
            auto* values = column.mutable_scalars()->mutable_long_data();
            for (size_t i = 0; i < labels.size(); ++i) {
                values->set_data(i, labels[i]);
            }
        }
    }
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    std::vector<std::pair<int64_t, float>> candidates;
    for (size_t i = 0; i < candidate_count; ++i) {
        candidates.emplace_back(i, static_cast<float>(i));
    }
    SearchResult result;
    result.total_nq_ = 1;
    result.total_data_cnt_ = labels.size();
    result.vector_iterators_ = std::vector<std::shared_ptr<VectorIterator>>{
        MakeSequenceVectorIterator(candidates)};
    int recreated = 0;
    result.SetVectorIteratorRecreator(
        BitsetView{}, [&](const BitsetView& invalid, SearchResult& batch) {
            ++recreated;
            if (cancel_on_recreate != nullptr) {
                cancel_on_recreate->requestCancellation();
            }
            if (recreate_error.has_value()) {
                throw SegcoreError(*recreate_error,
                                   "injected preparation failure");
            }
            // Only sampling consumes a probe after locking. The locked prefix
            // (and sampled rejected rows in these scenarios) must be excluded.
            const auto probe =
                strategy == StrictGroupStrategy::Sampling ? probe_budget : 0;
            for (int64_t i = 0; i < topk + probe; ++i) {
                EXPECT_TRUE(invalid.test(i)) << i;
            }
            EXPECT_FALSE(invalid.test(labels.size() - 1));
            auto fresh_candidates = candidates;
            if (recreated_offsets) {
                fresh_candidates.clear();
                for (auto offset : *recreated_offsets) {
                    fresh_candidates.emplace_back(offset,
                                                  static_cast<float>(offset));
                }
            }
            batch.vector_iterators_ =
                std::vector<std::shared_ptr<VectorIterator>>{
                    MakeSequenceVectorIterator(
                        fresh_candidates, invalid, true)};
        });
    SearchInfo info;
    info.topk_ = topk;
    info.group_size_ = group_size;
    info.strict_group_size_ = true;
    info.group_by_field_id_ = field;
    info.metric_type_ = knowhere::metric::L2;
    info.search_params_ = knowhere::Json::object();
    info.strict_group_acceptance_threshold_ = threshold.value_or(0.1);
    info.strict_group_probe_candidates_ = probe_budget;
    info.strict_group_strategy_ = strategy;
    info.strict_group_debug_ = debug;
    std::vector<GroupByValueType> groups;
    std::vector<int64_t> offsets;
    std::vector<float> distances;
    std::vector<size_t> prefix;
    SearchGroupBy(op_ctx,
                  *result.vector_iterators_,
                  info,
                  groups,
                  *segment,
                  offsets,
                  distances,
                  prefix,
                  &result);
    EXPECT_EQ(recreated, expected_recreates);
    EXPECT_EQ(offsets.size(), expected_rows);
    EXPECT_EQ(
        std::unordered_set<int64_t>(offsets.begin(), offsets.end()).size(),
        offsets.size());
    EXPECT_FALSE(result.CanRecreateVectorIterator());
}

}  // namespace

TEST(StrictGroupDiagnosticsTest, OptInAndFallbackStages) {
    StrictGroupLogSink sink;
    google::AddLogSink(&sink);
    Defer remove_sink([&] { google::RemoveLogSink(&sink); });
    std::vector<int64_t> labels(300, 2);
    labels[0] = labels[298] = labels[299] = 1;
    for (auto strategy : {StrictGroupStrategy::Sampling,
                          StrictGroupStrategy::FilteredIterator,
                          StrictGroupStrategy::PerGroup}) {
        for (bool debug : {false, true}) {
            sink.records.clear();
            // A short filtered iterator ends locally, without original fallback.
            // The test provider has no ordinary Search, so per_group falls back.
            const bool per_group = strategy == StrictGroupStrategy::PerGroup;
            CheckProbeScenario(labels,
                               labels.size(),
                               1,
                               3,
                               per_group ? 0 : 1,
                               per_group ? 3 : 2,
                               std::nullopt,
                               nullptr,
                               nullptr,
                               0.1,
                               std::vector<int64_t>{298},
                               100,
                               strategy,
                               debug);
            if (!debug) {
                EXPECT_TRUE(sink.records.empty());
                continue;
            }
            ASSERT_FALSE(sink.records.empty());
            EXPECT_EQ(sink.records.front()["stage"], "begin");
            EXPECT_EQ(sink.records.back()["stage"], "finish");
            EXPECT_EQ(sink.records.back()["remaining_locked_rows"],
                      per_group ? 0 : 1);
            EXPECT_EQ(sink.records.back()["original_iterator_skipped"],
                      !per_group);
            EXPECT_EQ(
                sink.records.back()["completion_reason"],
                per_group ? "quota_satisfied" : "filtered_iterator_exhausted");
            bool saw_fallback = false;
            for (const auto& record : sink.records) {
                EXPECT_EQ(record["acceptance_threshold"], 0.1);
                EXPECT_EQ(record["probe_budget"], 100);
                EXPECT_EQ(record["diagnostic_id"],
                          sink.records.front()["diagnostic_id"]);
                EXPECT_FALSE(record.contains("group_value"));
                if (record["stage"] == "original_begin") {
                    saw_fallback = true;
                    EXPECT_GT(record["remaining_locked_rows"].get<int64_t>(),
                              0);
                }
                if (record["stage"] == "union_filter_ready") {
                    EXPECT_EQ(record["available_rows"], 2);
                }
            }
            EXPECT_EQ(saw_fallback, per_group);
        }
        sink.records.clear();
        CheckProbeScenario(labels,
                           labels.size(),
                           1,
                           3,
                           0,
                           3,
                           std::nullopt,
                           nullptr,
                           nullptr,
                           0.0,
                           std::nullopt,
                           100,
                           strategy,
                           true);
        ASSERT_EQ(sink.records.size(), 1);
        EXPECT_EQ(sink.records.front()["stage"], "ineligible_original");
        EXPECT_EQ(sink.records.front()["acceptance_threshold"], 0.0);
    }
}

TEST(StrictGroupForcedIteratorTest, UnionFilterSkipsProbeAndStopsOnExhaustion) {
    // A fills before B/C lock; one B row is excluded by the original filter.
    const std::vector<int64_t> labels{
        10, 10, 10, 20, 30, 99, 20, 20, 20, 30, 30};
    for (auto threshold : {0.0, 0.000001, 1.0}) {
        for (auto probe_budget : {int64_t(1), int64_t(1000)}) {
            for (bool short_batch : {false, true}) {
                SCOPED_TRACE(testing::Message()
                             << threshold << "/" << probe_budget << "/"
                             << short_batch);
                auto schema = std::make_shared<Schema>();
                auto pk = schema->AddDebugField("pk", DataType::INT64);
                auto field = schema->AddDebugField("group", DataType::INT64);
                schema->set_primary_field_id(pk);
                auto data = segcore::DataGen(schema, labels.size());
                for (auto& column : *data.raw_->mutable_fields_data()) {
                    if (column.field_id() == field.get()) {
                        auto* values =
                            column.mutable_scalars()->mutable_long_data();
                        for (size_t i = 0; i < labels.size(); ++i) {
                            values->set_data(i, labels[i]);
                        }
                    }
                }
                auto segment = CreateSealedWithFieldDataLoaded(schema, data);
                TargetBitmap base(labels.size(), false);
                base[6] = true;
                std::vector<std::pair<int64_t, float>> candidates;
                for (size_t i = 0; i < labels.size(); ++i) {
                    candidates.emplace_back(i, static_cast<float>(i));
                }
                SearchResult result;
                result.total_nq_ = 1;
                result.total_data_cnt_ = labels.size();
                result.vector_iterators_ =
                    std::vector<std::shared_ptr<VectorIterator>>{
                        MakeSequenceVectorIterator(candidates,
                                                   BitsetView(base))};
                int recreated = 0;
                result.SetVectorSearchProvider(
                    BitsetView(base),
                    [&](const BitsetView& invalid,
                        std::optional<int64_t> topk,
                        SearchResult& batch) {
                        EXPECT_FALSE(
                            topk.has_value());  // Iterator, not Search.
                        ++recreated;
                        for (size_t i = 0; i < labels.size(); ++i) {
                            EXPECT_EQ(invalid.test(i), i < 7) << i;
                        }
                        const auto fresh =
                            short_batch
                                ? std::vector<std::pair<int64_t, float>>{{7,
                                                                          7.0F},
                                                                         {9,
                                                                          9.0F}}
                                : candidates;
                        batch.vector_iterators_ =
                            std::vector<std::shared_ptr<VectorIterator>>{
                                MakeSequenceVectorIterator(
                                    fresh, invalid, true)};
                    });
                SearchInfo info;
                info.topk_ = 3;
                info.group_size_ = 3;
                info.strict_group_size_ = true;
                info.group_by_field_id_ = field;
                info.metric_type_ = knowhere::metric::L2;
                info.strict_group_strategy_ =
                    StrictGroupStrategy::FilteredIterator;
                info.strict_group_acceptance_threshold_ = threshold;
                info.strict_group_probe_candidates_ = probe_budget;
                const auto probe_before =
                    milvus::monitor::
                        internal_core_strict_group_phase2_probe_candidates
                            .Collect()
                            .histogram.sample_sum;
                std::vector<GroupByValueType> groups;
                std::vector<int64_t> offsets;
                std::vector<float> distances;
                std::vector<size_t> prefix;
                SearchGroupBy(nullptr,
                              *result.vector_iterators_,
                              info,
                              groups,
                              *segment,
                              offsets,
                              distances,
                              prefix,
                              &result);
                EXPECT_EQ(recreated, threshold > 0 ? 1 : 0);
                const bool partial = threshold > 0 && short_batch;
                EXPECT_EQ(offsets.size(), partial ? 7 : 9);
                EXPECT_EQ(
                    std::unordered_set<int64_t>(offsets.begin(), offsets.end()),
                    (partial ? std::unordered_set<int64_t>{0, 1, 2, 3, 4, 7, 9}
                             : std::unordered_set<int64_t>{
                                   0, 1, 2, 3, 4, 7, 8, 9, 10}));
                EXPECT_EQ(milvus::monitor::
                              internal_core_strict_group_phase2_probe_candidates
                                  .Collect()
                                  .histogram.sample_sum,
                          probe_before);
                EXPECT_FALSE(result.CanRecreateVectorIterator());
            }
        }
    }
}

TEST(StrictGroupForcedIteratorTest, BackendErrorsPropagate) {
    std::vector<int64_t> labels(300, 2);
    labels[0] = labels[298] = labels[299] = 1;
    EXPECT_THROW(CheckProbeScenario(labels,
                                    labels.size(),
                                    1,
                                    3,
                                    1,
                                    3,
                                    ErrorCode::UnexpectedError,
                                    nullptr,
                                    nullptr,
                                    0.1,
                                    std::nullopt,
                                    100,
                                    StrictGroupStrategy::FilteredIterator),
                 SegcoreError);
}

TEST(StrictGroupLocalExhaustionTest,
     AvailabilityAndPartialResultsAcrossStrategies) {
    StrictGroupLogSink sink;
    google::AddLogSink(&sink);
    Defer remove_sink([&] { google::RemoveLogSink(&sink); });
    for (auto strategy : {StrictGroupStrategy::Sampling,
                          StrictGroupStrategy::FilteredIterator,
                          StrictGroupStrategy::PerGroup}) {
        for (int n = 120; n < 128; ++n) {
            for (int available : {0, 1, 2, 4}) {
                for (int topk : {1, 3}) {
                    for (bool debug : {false, true}) {
                        SCOPED_TRACE(testing::Message()
                                     << static_cast<int>(strategy) << "/" << n
                                     << "/" << available << "/" << topk << "/"
                                     << debug);
                        sink.records.clear();
                        auto schema = std::make_shared<Schema>();
                        auto pk = schema->AddDebugField("pk", DataType::INT64);
                        auto field =
                            schema->AddDebugField("group", DataType::INT64);
                        schema->set_primary_field_id(pk);
                        std::vector<int64_t> labels(n, 99);
                        labels[0] = 10;
                        if (topk == 3) {
                            labels[1] = 20;
                            labels[2] = 30;  // This group has no remaining row.
                            for (int i = n - 3; i < n; ++i) labels[i] = 20;
                        }
                        for (int i = 0; i < available; ++i)
                            labels[n - 10 + i] = 10;
                        labels[n - 6] = 10;
                        TargetBitmap base(n, false);
                        base[n - 6] =
                            true;  // Excluded by the original predicate.
                        auto data = segcore::DataGen(schema, n);
                        for (auto& column : *data.raw_->mutable_fields_data()) {
                            if (column.field_id() == field.get()) {
                                auto* values = column.mutable_scalars()
                                                   ->mutable_long_data();
                                for (int i = 0; i < n; ++i)
                                    values->set_data(i, labels[i]);
                            }
                        }
                        auto segment =
                            CreateSealedWithFieldDataLoaded(schema, data);
                        std::vector<std::pair<int64_t, float>> candidates;
                        for (int i = 0; i < n; ++i)
                            candidates.emplace_back(i, i);
                        SearchResult result;
                        result.total_nq_ = 1;
                        result.total_data_cnt_ = n;
                        auto original = MakeSequenceVectorIterator(
                            candidates, BitsetView(base));
                        result.vector_iterators_ =
                            std::vector<std::shared_ptr<VectorIterator>>{
                                original};
                        size_t provider_calls = 0;
                        result.SetVectorSearchProvider(
                            BitsetView(base),
                            [&](const BitsetView& invalid,
                                std::optional<int64_t> k,
                                SearchResult& batch) {
                                ++provider_calls;
                                EXPECT_EQ(
                                    k.has_value(),
                                    strategy == StrictGroupStrategy::PerGroup);
                                EXPECT_TRUE(invalid.test(n - 6));
                                for (int i = 0; i < topk; ++i)
                                    EXPECT_TRUE(invalid.test(i));
                                size_t valid_count = 0;
                                for (int i = 0; i < n; ++i)
                                    valid_count += !invalid.test(i);
                                EXPECT_GT(
                                    valid_count,
                                    0);  // Empty filters never reach the backend.
                                EXPECT_EQ(invalid.get_filtered_out_num_(),
                                          n - valid_count);
                                if (k) {
                                    for (int i = 0;
                                         i < n &&
                                         batch.seg_offsets_.size() < *k;
                                         ++i) {
                                        if (!invalid.test(i)) {
                                            batch.seg_offsets_.push_back(i);
                                            batch.distances_.push_back(i);
                                        }
                                    }
                                } else {
                                    batch.vector_iterators_ = std::vector<
                                        std::shared_ptr<VectorIterator>>{
                                        MakeSequenceVectorIterator(
                                            candidates, invalid, true)};
                                }
                            });
                        SearchInfo info;
                        info.topk_ = topk;
                        info.group_size_ = 3;
                        info.strict_group_size_ = true;
                        info.group_by_field_id_ = field;
                        info.metric_type_ = knowhere::metric::L2;
                        info.strict_group_strategy_ = strategy;
                        info.strict_group_probe_candidates_ = 1;
                        info.strict_group_debug_ = debug;
                        const auto before =
                            milvus::monitor::
                                internal_core_strict_group_phase2_original_remaining_candidates
                                    .Collect()
                                    .histogram.sample_sum;
                        std::vector<GroupByValueType> groups;
                        std::vector<int64_t> offsets;
                        std::vector<float> distances;
                        std::vector<size_t> prefix;
                        SearchGroupBy(nullptr,
                                      *result.vector_iterators_,
                                      info,
                                      groups,
                                      *segment,
                                      offsets,
                                      distances,
                                      prefix,
                                      &result);
                        const size_t expected =
                            1 + std::min(available, 2) + (topk == 3 ? 4 : 0);
                        EXPECT_EQ(offsets.size(), expected);
                        EXPECT_EQ(std::unordered_set<int64_t>(offsets.begin(),
                                                              offsets.end())
                                      .size(),
                                  expected);
                        EXPECT_EQ(
                            std::count(offsets.begin(), offsets.end(), n - 6),
                            0);
                        std::unordered_map<int64_t, size_t> counts;
                        for (auto offset : offsets) ++counts[labels[offset]];
                        EXPECT_EQ(counts[10], 1 + std::min(available, 2));
                        if (topk == 3) {
                            EXPECT_EQ(counts[20], 3);
                            EXPECT_EQ(counts[30], 1);
                        }
                        EXPECT_EQ(counts[99], 0);
                        EXPECT_EQ(provider_calls,
                                  strategy == StrictGroupStrategy::PerGroup
                                      ? (available > 0) + (topk == 3)
                                      : (available > 0 || topk == 3));
                        EXPECT_EQ(
                            milvus::monitor::
                                internal_core_strict_group_phase2_original_remaining_candidates
                                    .Collect()
                                    .histogram.sample_sum,
                            before);
                        // Directly prove the original consumer stopped at lock/probe.
                        ASSERT_TRUE(original->HasNext());
                        EXPECT_EQ(
                            original->Next()->first,
                            topk + (strategy == StrictGroupStrategy::Sampling
                                        ? 1
                                        : 0));
                        if (!debug) {
                            EXPECT_TRUE(sink.records.empty());
                        } else {
                            ASSERT_FALSE(sink.records.empty());
                            const auto& last = sink.records.back();
                            EXPECT_EQ(last["stage"], "finish");
                            EXPECT_EQ(last["original_iterator_skipped"], true);
                            EXPECT_EQ(last["original_remaining_candidates"], 0);
                            const auto union_available =
                                available + (topk == 3 ? 3 : 0);
                            const char* reason = "quota_satisfied";
                            if (topk == 1 && available == 0)
                                reason = "no_available_rows";
                            else if (topk == 3 || available < 2) {
                                reason =
                                    strategy == StrictGroupStrategy::PerGroup ||
                                            union_available < topk * 2
                                        ? "insufficient_available_rows"
                                        : "filtered_iterator_exhausted";
                            }
                            EXPECT_EQ(last["completion_reason"], reason);
                            for (const auto& record : sink.records) {
                                EXPECT_NE(record["stage"], "original_begin");
                                if (record["stage"] == "union_filter_ready") {
                                    EXPECT_EQ(record["available_rows"],
                                              union_available);
                                    EXPECT_EQ(record["remaining_locked_rows"],
                                              topk * 2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(StrictGroupForcedIteratorTest, AllStrategiesShareEligibilityGates) {
    for (auto strategy : {StrictGroupStrategy::Sampling,
                          StrictGroupStrategy::FilteredIterator,
                          StrictGroupStrategy::PerGroup}) {
        SearchInfo info;
        info.strict_group_strategy_ = strategy;
        info.topk_ = 3;
        info.group_size_ = 3;
        info.strict_group_size_ = true;
        EXPECT_TRUE(query::CanUseStrictGroupFilteredIterator(info, 1));
        EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(info, 2));
        info.strict_group_size_ = false;
        EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(info, 1));
        info.strict_group_size_ = true;
        info.group_size_ = 1;
        EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(info, 1));
        info.group_size_ = 3;
        info.strict_group_acceptance_threshold_ = 0;
        EXPECT_FALSE(query::CanUseStrictGroupFilteredIterator(info, 1));
    }
}

TEST(StrictGroupPerGroupTest, OrdinarySearchSettingsDoNotMutatePhaseOne) {
    SearchInfo original;
    original.topk_ = 50;
    original.group_by_field_id_ = FieldId(101);
    original.group_size_ = 3;
    original.strict_group_size_ = true;
    original.iterative_filter_execution = true;
    original.iterator_v2_info_ = SearchIteratorV2Info{};
    original.search_params_ = {{"ef", 123}, {knowhere::meta::TOPK, 50}};
    auto unchanged = query::StrictGroupSearchInfo(original, std::nullopt);
    EXPECT_TRUE(UseVectorIterator(unchanged));
    EXPECT_EQ(unchanged.topk_, 50);
    for (int64_t remaining : {1, 2}) {
        auto ordinary = query::StrictGroupSearchInfo(original, remaining);
        EXPECT_FALSE(UseVectorIterator(ordinary));
        EXPECT_FALSE(ordinary.iterator_v2_info_);
        EXPECT_FALSE(ordinary.strict_group_size_);
        EXPECT_EQ(ordinary.topk_, remaining);
        EXPECT_EQ(ordinary.search_params_[knowhere::meta::TOPK], remaining);
        EXPECT_EQ(ordinary.search_params_["ef"], 123);
    }
    EXPECT_TRUE(UseVectorIterator(original));
    EXPECT_EQ(original.topk_, 50);
    EXPECT_EQ(original.search_params_[knowhere::meta::TOPK], 50);
}

TEST(StrictGroupPerGroupTest, SearchProviderCapabilityAndErrors) {
    SearchResult result;
    auto filter = std::make_shared<TargetBitmap>(8, false);
    EXPECT_FALSE(result.SearchFilteredVectors(filter, 1));
    result.SetVectorSearchProvider(
        {}, [](const BitsetView&, std::optional<int64_t>, SearchResult&) {
            throw std::runtime_error("backend failure");
        });
    EXPECT_TRUE(result.CanSearchFilteredVectors());
    EXPECT_FALSE(result.SearchFilteredVectors(filter, 0));
    EXPECT_THROW(result.SearchFilteredVectors(filter, 1), std::runtime_error);
    result.ClearVectorIteratorRecreator();
    EXPECT_FALSE(result.CanSearchFilteredVectors());
    EXPECT_FALSE(result.SearchFilteredVectors(filter, 1));
}

TEST(StrictGroupPerGroupTest, IsolatedFiltersReuseStorageAndKeepShortResults) {
    StrictGroupLogSink sink;
    google::AddLogSink(&sink);
    Defer remove_sink([&] { google::RemoveLogSink(&sink); });
    // A fills before B locks; C is never locked. B/C have rows in later chunks.
    const std::vector<int64_t> labels{10, 10, 10, 20, 99, 20, 20, 30, 30};
    for (bool disabled : {false, true}) {
        // Full results, invalid padding, duplicate result, and empty result.
        for (int response : {0, 1, 2, 3}) {
            sink.records.clear();
            auto schema = std::make_shared<Schema>();
            auto pk = schema->AddDebugField("pk", DataType::INT64);
            auto field = schema->AddDebugField("group", DataType::INT64);
            schema->set_primary_field_id(pk);
            auto data = segcore::DataGen(schema, labels.size());
            for (auto& column : *data.raw_->mutable_fields_data()) {
                if (column.field_id() == field.get()) {
                    auto* values =
                        column.mutable_scalars()->mutable_long_data();
                    for (size_t i = 0; i < labels.size(); ++i) {
                        values->set_data(i, labels[i]);
                    }
                }
            }
            auto segment = CreateSealedWithFieldDataLoaded(schema, data);
            std::vector<std::pair<int64_t, float>> candidates;
            for (size_t i = 0; i < labels.size(); ++i) {
                candidates.emplace_back(i, static_cast<float>(i));
            }
            SearchResult result;
            result.total_nq_ = 1;
            result.total_data_cnt_ = labels.size();
            result.vector_iterators_ =
                std::vector<std::shared_ptr<VectorIterator>>{
                    MakeSequenceVectorIterator(candidates)};
            int batches = 0;
            result.SetVectorSearchProvider(
                {},
                [&](const BitsetView& invalid,
                    std::optional<int64_t> topk,
                    SearchResult& batch) {
                    ASSERT_EQ(topk, 2);
                    ++batches;
                    // A is already full, B's first result is retained, C excluded.
                    for (size_t i = 0; i < labels.size(); ++i) {
                        EXPECT_EQ(invalid.test(i), i != 5 && i != 6);
                    }
                    EXPECT_EQ(invalid.get_filtered_out_num_(),
                              labels.size() - 2);
                    if (response != 3) {
                        batch.seg_offsets_ = {5,
                                              response == 1
                                                  ? INVALID_SEG_OFFSET
                                                  : (response == 2 ? 5 : 6)};
                        batch.distances_ = {5.0F, 6.0F};
                    }
                });
            SearchInfo info;
            info.topk_ = 2;
            info.group_size_ = 3;
            info.strict_group_size_ = true;
            info.group_by_field_id_ = field;
            info.metric_type_ = knowhere::metric::L2;
            info.strict_group_strategy_ = StrictGroupStrategy::PerGroup;
            info.strict_group_acceptance_threshold_ = disabled ? 0 : 0.1;
            info.strict_group_debug_ = response % 2 == 1;
            auto before =
                milvus::monitor::
                    internal_core_strict_group_phase2_probe_candidates.Collect()
                        .histogram.sample_sum;
            std::vector<GroupByValueType> groups;
            std::vector<int64_t> offsets;
            std::vector<float> distances;
            std::vector<size_t> prefix;
            SearchGroupBy(nullptr,
                          *result.vector_iterators_,
                          info,
                          groups,
                          *segment,
                          offsets,
                          distances,
                          prefix,
                          &result);
            EXPECT_EQ(batches, disabled ? 0 : 1);
            const size_t expected_rows =
                disabled || response == 0 ? 6 : (response == 3 ? 4 : 5);
            EXPECT_EQ(offsets.size(), expected_rows);
            auto expected_offsets = std::unordered_set<int64_t>{0, 1, 2, 3};
            if (expected_rows >= 5)
                expected_offsets.insert(5);
            if (expected_rows == 6)
                expected_offsets.insert(6);
            EXPECT_EQ(
                std::unordered_set<int64_t>(offsets.begin(), offsets.end()),
                expected_offsets);
            if (info.strict_group_debug_ && !disabled) {
                ASSERT_FALSE(sink.records.empty());
                EXPECT_EQ(sink.records.back()["completion_reason"],
                          "search_short_result");
                EXPECT_EQ(sink.records.back()["original_iterator_skipped"],
                          true);
                EXPECT_EQ(sink.records.back()["original_remaining_candidates"],
                          0);
            }
            EXPECT_EQ(
                milvus::monitor::
                    internal_core_strict_group_phase2_probe_candidates.Collect()
                        .histogram.sample_sum,
                before);
        }
    }
}

class StrictGroupPerGroupPaddingTest
    : public ::testing::TestWithParam<int64_t> {};

TEST_P(StrictGroupPerGroupPaddingTest, MultipleGroupsAndSharedFilterLifetime) {
    const int64_t n = GetParam();
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto data = segcore::DataGen(schema, n, 42, 0, 4);
    // Keep group quotas satisfiable for every logical-size remainder.
    for (auto& column : *data.raw_->mutable_fields_data()) {
        if (column.field_id() == field.get()) {
            auto* values = column.mutable_scalars()->mutable_long_data();
            for (int64_t i = 0; i < n; ++i) {
                values->set_data(i, i % 4);
            }
        }
    }
    auto segment = CreateSealedWithFieldDataLoaded(schema, data);
    auto labels = data.get_col<int64_t>(field);
    std::unordered_map<int64_t, std::vector<int64_t>> by_group;
    for (int64_t i = 0; i < n; ++i) {
        by_group[labels[i]].push_back(i);
    }
    std::vector<std::optional<int64_t>> targets;
    std::vector<std::pair<int64_t, float>> candidates;
    for (auto& [group, rows] : by_group) {
        ASSERT_GE(rows.size(), 4);
        targets.emplace_back(group);
        candidates.emplace_back(rows[0], candidates.size());
        if (targets.size() == 3)
            break;
    }
    for (int64_t i = 0; i < n; ++i) candidates.emplace_back(i, n + i);
    // First group already has two hits when the third group locks.
    candidates.insert(candidates.begin() + 1, {by_group[*targets[0]][1], 0.5F});
    TargetBitmap base(n, false);
    auto excluded = by_group[*targets[1]][1];
    base[excluded] = true;
    SearchResult result;
    result.total_nq_ = 1;
    result.total_data_cnt_ = n;
    result.vector_iterators_ = std::vector<std::shared_ptr<VectorIterator>>{
        MakeSequenceVectorIterator(candidates, BitsetView(base))};
    size_t batches = 0;
    const uint8_t* bitmap_address = nullptr;
    result.SetVectorSearchProvider(
        BitsetView(base),
        [&](const BitsetView& invalid,
            std::optional<int64_t> topk,
            SearchResult& batch) {
            ASSERT_EQ(topk, batches == 0 ? 1 : 2);
            if (bitmap_address)
                EXPECT_EQ(bitmap_address, invalid.data());
            bitmap_address = invalid.data();
            auto target = *targets.at(batches++);
            for (int64_t i = 0; i < n; ++i) {
                bool accepted = std::any_of(
                    candidates.begin(),
                    candidates.begin() + 4,
                    [&](const auto& item) { return item.first == i; });
                EXPECT_EQ(invalid.test(i),
                          base[i] || labels[i] != target || accepted);
            }
            size_t expected_filtered = 0;
            for (int64_t i = 0; i < n; ++i) {
                expected_filtered += invalid.test(i);
            }
            EXPECT_EQ(invalid.get_filtered_out_num_(), expected_filtered);
            for (int64_t i = 0; i < n && batch.seg_offsets_.size() < *topk;
                 ++i) {
                if (!invalid.test(i)) {
                    batch.seg_offsets_.push_back(i);
                    batch.distances_.push_back(n + i);
                }
            }
        });
    SearchInfo info;
    info.topk_ = 3;
    info.group_size_ = 3;
    info.strict_group_size_ = true;
    info.group_by_field_id_ = field;
    info.metric_type_ = knowhere::metric::L2;
    info.strict_group_strategy_ = StrictGroupStrategy::PerGroup;
    std::vector<GroupByValueType> groups;
    std::vector<int64_t> offsets;
    std::vector<float> distances;
    std::vector<size_t> prefix;
    SearchGroupBy(nullptr,
                  *result.vector_iterators_,
                  info,
                  groups,
                  *segment,
                  offsets,
                  distances,
                  prefix,
                  &result);
    EXPECT_EQ(batches, 3);
    EXPECT_EQ(offsets.size(), 9);
    EXPECT_EQ(
        std::unordered_set<int64_t>(offsets.begin(), offsets.end()).size(), 9);
    EXPECT_EQ(std::count(offsets.begin(), offsets.end(), excluded), 0);
}

INSTANTIATE_TEST_SUITE_P(
    AllByteRemainders,
    StrictGroupPerGroupPaddingTest,
    ::testing::Values(120, 121, 122, 123, 124, 125, 126, 127));

TEST(StrictGroupPhase2ExecutorTest, ProbeBoundaryAndOnlyOneDecision) {
    for (int accepted : {9, 10, 11}) {
        SCOPED_TRACE(accepted);
        std::vector<int64_t> labels(300, 2);
        labels[0] = 1;
        for (int i = 1; i <= accepted; ++i) {
            labels[i] = 1;
        }
        // Do not re-evaluate even if the next window has no useful hits.
        for (size_t i = 201; i < labels.size(); ++i) {
            labels[i] = 1;
        }
        CheckProbeScenario(
            labels, labels.size(), 1, 20, accepted < 10 ? 1 : 0, 20);
    }
}

TEST(StrictGroupPhase2ExecutorTest, FullGroupHitsAreNotAcceptedProbeRows) {
    std::vector<int64_t> labels(120, 1);
    labels[0] = 0;
    for (size_t i = 2; i < 102; ++i) {
        labels[i] = 0;
    }
    // All 100 candidates hit locked groups, but only two fill a quota.
    // Acceptance is 2%, so recreate regardless of group-hit rate.
    CheckProbeScenario(labels, labels.size(), 2, 3, 1, 6);
}

TEST(StrictGroupPhase2ExecutorTest, ConfigurableProbeBudget) {
    for (int64_t budget : {1, 17, 100, 250}) {
        std::vector<int64_t> labels(budget + 4, 99);
        labels[0] = labels[budget + 1] = labels[budget + 2] =
            labels[budget + 3] = 1;
        auto before =
            milvus::monitor::internal_core_strict_group_phase2_probe_candidates
                .Collect()
                .histogram.sample_sum;
        CheckProbeScenario(labels,
                           labels.size(),
                           1,
                           3,
                           1,
                           3,
                           std::nullopt,
                           nullptr,
                           nullptr,
                           0.1,
                           std::nullopt,
                           budget);
        auto after =
            milvus::monitor::internal_core_strict_group_phase2_probe_candidates
                .Collect()
                .histogram.sample_sum;
        EXPECT_EQ(after - before, budget);
    }
    // With T=10, accepting one row is exactly 10%, not 1%.
    std::vector<int64_t> labels(14, 99);
    labels[0] = labels[1] = labels[11] = labels[12] = labels[13] = 1;
    for (double threshold : {0.1, 0.2}) {
        CheckProbeScenario(labels,
                           labels.size(),
                           1,
                           3,
                           threshold == 0.1 ? 0 : 1,
                           3,
                           std::nullopt,
                           nullptr,
                           nullptr,
                           threshold,
                           std::nullopt,
                           10);
    }
}

TEST(StrictGroupPhase2ExecutorTest, ConfigurableAcceptanceThreshold) {
    for (double threshold : {0.0, 0.01, 0.1, 0.5, 1.0}) {
        for (int accepted : {0, 1, 2, 9, 10, 11, 49, 50, 51, 100}) {
            SCOPED_TRACE(threshold);
            SCOPED_TRACE(accepted);
            std::vector<int64_t> labels(400, 2);
            labels[0] = 1;
            std::fill(labels.begin() + 1, labels.begin() + 1 + accepted, 1);
            std::fill(labels.begin() + 201, labels.end(), 1);
            CheckProbeScenario(labels,
                               labels.size(),
                               1,
                               150,
                               accepted / 100.0 < threshold ? 1 : 0,
                               150,
                               std::nullopt,
                               nullptr,
                               nullptr,
                               threshold);
        }
    }
}

TEST(StrictGroupPhase2ExecutorTest, RemainingQuotaDoesNotAffectDecision) {
    for (int quota : {3, 6, 7, 8, 100}) {
        std::vector<int64_t> labels(300, 2);
        labels[0] = labels[1] = 1;
        std::fill(labels.begin() + 101, labels.end(), 1);
        // A=1 triggers recreation even when only one row remains.
        CheckProbeScenario(labels, labels.size(), 1, quota, 1, quota);
    }
}

TEST(StrictGroupPhase2ExecutorTest, NonzeroLowAcceptance) {
    std::vector<int64_t> labels(250, 99);
    for (int i = 0; i < 50; ++i) {
        labels[i] = i;
        labels[150 + 2 * i] = labels[151 + 2 * i] = i;
    }
    labels[50] = 0;
    // A=1 out of 100 triggers recreation.
    CheckProbeScenario(labels, labels.size(), 50, 3, 1, 150);
}

TEST(StrictGroupPhase2ExecutorTest, ZeroAcceptanceWithFullGroupHits) {
    std::vector<int64_t> labels(106, 0);
    labels[3] = labels[104] = labels[105] = 1;
    // Group zero is full before locking group one. A=0 still recreates even
    // though all 100 probe candidates hit the full locked group (H=100).
    CheckProbeScenario(labels, labels.size(), 2, 3, 1, 6);
}

TEST(StrictGroupPhase2ExecutorTest, ProbeExhaustionDoesNotRecreate) {
    std::vector<int64_t> labels(120, 2);
    labels[0] = 1;
    CheckProbeScenario(labels, 50, 1, 3, 0, 1);
}

TEST(StrictGroupPhase2ExecutorTest, PreparationExceptionsPropagate) {
    std::vector<int64_t> labels(104, 99);
    labels[0] = labels[102] = labels[103] = 1;
    for (auto code : {ErrorCode::Unsupported,
                      ErrorCode::FileReadFailed,
                      ErrorCode::KnowhereError,
                      ErrorCode::MemAllocateFailed,
                      ErrorCode::FollyCancel,
                      ErrorCode::FollyOtherException,
                      ErrorCode::UnexpectedError,
                      ErrorCode::DataFormatBroken}) {
        EXPECT_THROW(
            CheckProbeScenario(labels, labels.size(), 1, 3, 1, 3, code),
            SegcoreError);
    }
    folly::CancellationSource source;
    milvus::OpContext op_ctx(source.getToken());
    EXPECT_THROW(CheckProbeScenario(labels,
                                    labels.size(),
                                    1,
                                    3,
                                    1,
                                    3,
                                    ErrorCode::Unsupported,
                                    &op_ctx,
                                    &source),
                 SegcoreError);
}

TEST(StrictGroupPhase2ExecutorTest,
     ExhaustedRecreatedIteratorKeepsPartialResults) {
    std::vector<int64_t> labels(106, 99);
    labels[0] = labels[102] = labels[103] = labels[104] = labels[105] = 1;
    // The fresh iterator may return nothing, or return 103 before the original
    // would reach 102,103,104. Never resume it; duplicate 103 adds no quota.
    for (auto fresh : {std::vector<int64_t>{},
                       std::vector<int64_t>{103},
                       std::vector<int64_t>{103, 103}}) {
        CheckProbeScenario(labels,
                           labels.size(),
                           1,
                           4,
                           1,
                           fresh.empty() ? 1 : 2,
                           std::nullopt,
                           nullptr,
                           nullptr,
                           0.1,
                           fresh);
    }
}

TEST(GroupMembershipTest, RawScansHonorCancellation) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(pk);
    constexpr size_t rows = 4096;
    auto data = segcore::DataGen(schema, rows);
    auto sealed = CreateSealedWithFieldDataLoaded(schema, data);
    auto growing = segcore::CreateGrowingSegment(schema, empty_index_meta);
    auto offset = growing->PreInsert(rows);
    growing->Insert(
        offset, rows, data.row_ids_.data(), data.timestamps_.data(), data.raw_);
    folly::CancellationSource source;
    milvus::OpContext ctx(source.getToken());
    source.requestCancellation();
    for (const auto* segment :
         {dynamic_cast<const segcore::SegmentInternalInterface*>(sealed.get()),
          dynamic_cast<const segcore::SegmentInternalInterface*>(
              growing.get())}) {
        ASSERT_NE(segment, nullptr);
        EXPECT_THROW(BuildGroupOffsets<int64_t>(
                         &ctx, *segment, field, rows, {int64_t(1)}, nullptr),
                     SegcoreError);
        try {
            BuildGroupMembership<int64_t>(
                &ctx, *segment, field, rows, {int64_t(1)}, nullptr);
            FAIL() << "cancelled membership scan completed";
        } catch (const SegcoreError& error) {
            EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
        }
    }
}

TEST(GroupMembershipTest, CancellationDuringRawScanStopsBeforeNextChunk) {
    // Cancel when the accessor reaches chunk 1, after chunk 0 was consumed.
    // No timing or thread scheduling dependency: the next periodic check must
    // throw before chunk 2 is pinned.
    class CancellingSegment : public segcore::ChunkedSegmentSealedImpl {
     public:
        CancellingSegment(SchemaPtr schema, folly::CancellationSource& source)
            : ChunkedSegmentSealedImpl(schema,
                                       empty_index_meta,
                                       segcore::SegcoreConfig::default_config(),
                                       991),
              source_(source),
              values_(2048, 1) {
        }
        bool
        HasFieldData(FieldId) const override {
            return true;
        }
        int64_t
        num_chunk_data(FieldId) const override {
            return 4;
        }
        int64_t
        size_per_chunk() const override {
            return 2048;
        }
        int64_t
        chunk_size(FieldId, int64_t) const override {
            return 2048;
        }
        int64_t
        num_rows_until_chunk(FieldId, int64_t id) const override {
            return id * 2048;
        }
        mutable int pins = 0;

     protected:
        PinWrapper<SpanBase>
        chunk_data_impl(milvus::OpContext*,
                        FieldId,
                        int64_t chunk) const override {
            ++pins;
            if (chunk == 1) {
                source_.requestCancellation();
            }
            return PinWrapper<SpanBase>(
                SpanBase(values_.data(), 2048, sizeof(int64_t)));
        }

     private:
        folly::CancellationSource& source_;
        std::vector<int64_t> values_;
    };
    auto schema = std::make_shared<Schema>();
    auto field = schema->AddDebugField("group", DataType::INT64);
    schema->set_primary_field_id(field);
    folly::CancellationSource source;
    milvus::OpContext ctx(source.getToken());
    CancellingSegment segment(schema, source);
    try {
        BuildGroupMembership<int64_t>(
            &ctx, segment, field, 8192, {int64_t(1)}, nullptr);
        FAIL() << "membership continued after cancellation";
    } catch (const SegcoreError& error) {
        EXPECT_EQ(error.get_error_code(), ErrorCode::FollyCancel);
    }
    EXPECT_EQ(segment.pins, 2);
}

TEST(StrictGroupPhase2ExecutorTest, SharedBaseFilterIsLazyAndReleased) {
    SearchResult result;
    auto owner = std::make_shared<TargetBitmap>(1000, false);
    (*owner)[17] = true;
    std::weak_ptr<TargetBitmap> weak_owner = owner;
    result.vector_iterator_filter_owner_ = owner;
    result.SetVectorIteratorRecreator(
        BitsetView(*owner), [](const BitsetView& filter, SearchResult&) {
            EXPECT_TRUE(filter.test(17));
            EXPECT_TRUE(filter.test(33));
        });
    EXPECT_EQ(result.vector_iterator_base_filter_, nullptr);
    owner.reset();
    EXPECT_FALSE(weak_owner.expired());
    TargetBitmap extra(1000, false);
    extra[33] = true;
    auto recreated = result.RecreateVectorIterators(std::move(extra));
    ASSERT_TRUE(recreated.has_value());
    EXPECT_NE(result.vector_iterator_base_filter_, nullptr);
    result.ClearVectorIteratorRecreator();
    EXPECT_TRUE(weak_owner.expired());
    EXPECT_FALSE(result.CanRecreateVectorIterator());
    EXPECT_EQ(result.GetVectorIteratorBaseFilter(), nullptr);
}

TEST(StrictGroupPhase2ExecutorTest, BackendPreparationPreservesTypedErrors) {
    class FailingIndex : public index::VectorMemIndex<float> {
     public:
        FailingIndex()
            : VectorMemIndex(
                  DataType::NONE,
                  "FLAT",
                  knowhere::metric::L2,
                  knowhere::Version::GetCurrentVersion().VersionNumber()) {
        }
        ErrorCode error = ErrorCode::FollyCancel;
        knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
        VectorIterators(const DatasetPtr,
                        const knowhere::Json&,
                        const BitsetView&) const override {
            throw SegcoreError(error, "injected backend preparation failure");
        }
    } index;
    SearchInfo info;
    info.group_by_field_id_ = FieldId(100);
    info.topk_ = 1;
    info.metric_type_ = knowhere::metric::L2;
    for (bool original : {false, true}) {
        SearchResult result;
        result.allow_vector_iterator_recreation_ = original;
        for (auto code : {ErrorCode::FollyCancel,
                          ErrorCode::FollyOtherException,
                          ErrorCode::DataFormatBroken,
                          ErrorCode::Unsupported}) {
            index.error = code;
            try {
                PrepareVectorIteratorsFromIndex(
                    info, 1, nullptr, result, BitsetView{}, index);
                FAIL() << "backend must throw";
            } catch (const SegcoreError& error) {
                EXPECT_EQ(error.get_error_code(),
                          original ? ErrorCode::Unsupported : code);
            }
        }
    }
}

TEST(StrictGroupPhase2ExecutorTest, FilledAtProbeBoundaryDoesNotRecreate) {
    std::vector<int64_t> labels(120, 2);
    labels[0] = labels[100] = 1;
    CheckProbeScenario(labels, labels.size(), 1, 2, 0, 2);
}

TEST(GroupMembershipTest, GrowingMmapStringUsesElementView) {
    auto& config = storage::MmapManager::GetInstance().GetMmapConfig();
    const bool previous = config.GetEnableGrowingMmap();
    config.SetEnableGrowingMmap(true);
    auto restore = std::shared_ptr<void>(
        nullptr, [&](void*) { config.SetEnableGrowingMmap(previous); });
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("pk", DataType::INT64);
    auto field = schema->AddDebugField("group", DataType::VARCHAR);
    schema->set_primary_field_id(pk);
    auto data = segcore::DataGen(schema, 100, 42, 0, 4);
    auto segment = segcore::CreateGrowingSegment(schema, empty_index_meta);
    auto offset = segment->PreInsert(100);
    segment->Insert(
        offset, 100, data.row_ids_.data(), data.timestamps_.data(), data.raw_);
    auto values = data.get_col<std::string>(field);
    auto* growing = dynamic_cast<segcore::SegmentGrowingImpl*>(segment.get());
    ASSERT_NE(growing, nullptr);
    ASSERT_TRUE(
        growing->get_insert_record().get_data<std::string>(field)->is_mmap());
    std::vector<std::optional<std::string>> groups{values[0]};
    auto membership = BuildGroupMembership<std::string>(
        nullptr, *growing, field, 100, groups, nullptr);
    auto classified = BuildGroupOffsets<std::string>(
        nullptr, *growing, field, 100, groups, nullptr);
    ASSERT_TRUE(classified);
    std::vector<int64_t> expected;
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] == values[0])
            expected.push_back(i);
    }
    EXPECT_EQ((*classified)[0], expected);
    ASSERT_TRUE(membership.has_value());
    auto bitmap = std::move(membership);
    ASSERT_TRUE(bitmap.has_value());
    for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_EQ((*bitmap)[i], values[i] == values[0]);
    }
}

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
    auto scalar_index = std::make_unique<CountingScalarIndex>();
    auto* counters = scalar_index.get();
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
    // Raw classification is one pass; index-only retains the original iterator.
    std::vector<std::optional<int64_t>> unique_groups;
    for (const auto& group : groups) {
        if (std::find(unique_groups.begin(), unique_groups.end(), group) ==
            unique_groups.end())
            unique_groups.push_back(group);
    }
    auto classified = BuildGroupOffsets<int64_t>(nullptr,
                                                 *raw_segment,
                                                 group_field,
                                                 kRowCount,
                                                 unique_groups,
                                                 &base_filter);
    ASSERT_TRUE(classified);
    for (size_t i = 0; i < unique_groups.size(); ++i) {
        std::vector<int64_t> expected;
        for (size_t row = 0; row < kRowCount; ++row) {
            std::optional<int64_t> key =
                valid[row] ? std::optional<int64_t>(values[row]) : std::nullopt;
            if (!base_filter[row] && key == unique_groups[i])
                expected.push_back(row);
        }
        EXPECT_EQ((*classified)[i], expected);
    }
    EXPECT_FALSE(BuildGroupOffsets<int64_t>(nullptr,
                                            *index_segment,
                                            group_field,
                                            kRowCount,
                                            unique_groups,
                                            &base_filter));
    ASSERT_TRUE(raw.has_value());
    ASSERT_TRUE(indexed.has_value());
    EXPECT_EQ(counters->in_calls, 1);
    EXPECT_EQ(counters->in_values, 3);
    EXPECT_EQ(counters->null_calls, 1);
    // Both sources present: phase two must use raw data, like phase one.
    LoadGeneratedDataIntoSegment(
        data,
        index_segment.get(),
        false,
        {pk_field.get(), RowFieldID.get(), TimestampFieldID.get()});
    ASSERT_TRUE(index_segment->HasFieldData(group_field));
    auto both = BuildGroupMembership<int64_t>(
        nullptr, *index_segment, group_field, kRowCount, groups, &base_filter);
    ASSERT_TRUE(both.has_value());
    EXPECT_EQ(counters->in_calls, 1);
    EXPECT_EQ(counters->null_calls, 1);
    // The union bitmap owns its bits and no longer needs the source column.
    raw_segment->DropFieldData(group_field);
    auto raw_bitmap = std::move(raw);
    auto index_bitmap = std::move(indexed);
    ASSERT_TRUE(raw_bitmap.has_value());
    ASSERT_TRUE(index_bitmap.has_value());
    ASSERT_EQ(raw_bitmap->size(), index_bitmap->size());
    for (size_t i = 0; i < raw_bitmap->size(); ++i) {
        EXPECT_EQ((*raw_bitmap)[i], (*index_bitmap)[i]) << "offset " << i;
        EXPECT_EQ((*raw_bitmap)[i], (*both)[i]) << "offset " << i;
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
    auto string_bitmap = std::move(string_membership);
    ASSERT_TRUE(string_bitmap.has_value());

    for (size_t i = 0; i < kRowCount; ++i) {
        std::optional<std::string> value =
            valid[i] ? std::optional<std::string>(strings[i]) : std::nullopt;
        auto found =
            std::find(string_groups.begin(), string_groups.end(), value);
        auto expected = !base_filter[i] && found != string_groups.end();
        EXPECT_EQ((*string_bitmap)[i], expected) << "offset " << i;
    }

    std::vector<std::optional<bool>> bool_groups{false, true};
    auto bool_membership = BuildGroupMembership<bool>(
        nullptr, *segment, bool_field, kRowCount, bool_groups, &base_filter);
    ASSERT_TRUE(bool_membership.has_value());
    auto bool_bitmap = std::move(bool_membership);
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
