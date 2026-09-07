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
#include "SearchGroupByOperator.h"

#include <chrono>

#include "common/Tracer.h"
#include "common/Consts.h"
#include "common/JsonUtils.h"
#include "exec/operator/groupby/GroupMembership.h"
#include "exec/operator/groupby/StrictGroupPhase2Planner.h"
#include "fmt/format.h"
#include "monitor/Monitor.h"
#include "query/Utils.h"

namespace milvus {
namespace exec {

namespace {

enum class StrictGroupPhase2FallbackReason {
    None,
    MissingRecreator,
    InvalidRowCount,
    Phase1Exhausted,
    Phase2NotNeeded,
    MembershipUnavailable,
    PlanUnavailable,
    MembershipBitmapUnavailable,
    RecreateUnavailable,
};

struct StrictGroupPhase2Stats {
    bool attempted = false;
    bool used = false;
    size_t phase1_candidates = 0;
    size_t phase2_candidates = 0;
    size_t eligible_rows = 0;
    size_t unfinished_group_rows = 0;
    size_t small_group_count = 0;
    size_t large_group_count = 0;
    size_t batch_count = 0;
    uint64_t membership_build_us = 0;
    uint64_t bitmap_build_us = 0;
    StrictGroupPhase2FallbackReason fallback_reason =
        StrictGroupPhase2FallbackReason::None;
};

struct StrictGroupPhase2Context {
    milvus::OpContext* op_ctx;
    const segcore::SegmentInternalInterface& segment;
    FieldId group_by_field_id;
    SearchResult* search_result;
    bool eligible;
};

const char*
FallbackReasonName(StrictGroupPhase2FallbackReason reason) {
    switch (reason) {
        case StrictGroupPhase2FallbackReason::None:
            return "none";
        case StrictGroupPhase2FallbackReason::MissingRecreator:
            return "missing_recreator";
        case StrictGroupPhase2FallbackReason::InvalidRowCount:
            return "invalid_row_count";
        case StrictGroupPhase2FallbackReason::Phase1Exhausted:
            return "phase1_exhausted";
        case StrictGroupPhase2FallbackReason::Phase2NotNeeded:
            return "phase2_not_needed";
        case StrictGroupPhase2FallbackReason::MembershipUnavailable:
            return "membership_unavailable";
        case StrictGroupPhase2FallbackReason::PlanUnavailable:
            return "plan_unavailable";
        case StrictGroupPhase2FallbackReason::MembershipBitmapUnavailable:
            return "membership_bitmap_unavailable";
        case StrictGroupPhase2FallbackReason::RecreateUnavailable:
            return "recreate_unavailable";
    }
    return "unknown";
}

void
RecordStrictGroupPhase2Stats(const StrictGroupPhase2Stats& stats) {
    if (!stats.attempted) {
        return;
    }
    milvus::monitor::internal_core_strict_group_phase2_phase1_candidates
        .Observe(stats.phase1_candidates);
    milvus::monitor::internal_core_strict_group_phase2_phase2_candidates
        .Observe(stats.phase2_candidates);
    milvus::monitor::internal_core_strict_group_phase2_batch_count.Observe(
        stats.batch_count);
    milvus::monitor::internal_core_strict_group_phase2_small_group_count
        .Observe(stats.small_group_count);
    milvus::monitor::internal_core_strict_group_phase2_large_group_count
        .Observe(stats.large_group_count);
    milvus::monitor::internal_core_strict_group_phase2_membership_build_latency
        .Observe(stats.membership_build_us);
    milvus::monitor::internal_core_strict_group_phase2_bitmap_build_latency
        .Observe(stats.bitmap_build_us);
    if (stats.eligible_rows > 0) {
        milvus::monitor::internal_core_strict_group_phase2_selectivity.Observe(
            static_cast<double>(stats.unfinished_group_rows) /
            stats.eligible_rows);
    }
    tracer::AddEvent(fmt::format(
        "strict_group_phase2: used={}, fallback={}, phase1_candidates={}, "
        "phase2_candidates={}, eligible_rows={}, unfinished_group_rows={}, "
        "small_groups={}, large_groups={}, batches={}, membership_ms={:.3f}, "
        "bitmap_ms={:.3f}",
        stats.used,
        FallbackReasonName(stats.fallback_reason),
        stats.phase1_candidates,
        stats.phase2_candidates,
        stats.eligible_rows,
        stats.unfinished_group_rows,
        stats.small_group_count,
        stats.large_group_count,
        stats.batch_count,
        stats.membership_build_us / 1000.0,
        stats.bitmap_build_us / 1000.0));
}

template <typename T, typename StopPredicate>
size_t
ConsumeGroupByIteratorUntil(const std::shared_ptr<VectorIterator>& iterator,
                            const std::shared_ptr<DataGetter<T>>& data_getter,
                            GroupByMap<T>& group_map,
                            GroupByResultCollector<T>& collector,
                            StopPredicate&& should_stop,
                            TargetBitmap* consumed_offsets = nullptr) {
    size_t candidates = 0;
    while (iterator->HasNext() && !should_stop()) {
        auto offset_dis_pair = iterator->Next();
        ++candidates;
        AssertInfo(
            offset_dis_pair.has_value(),
            "Wrong state! iterator cannot return valid result whereas it "
            "still tells hasNext, terminate groupBy operation");
        auto offset = offset_dis_pair->first;
        auto distance = offset_dis_pair->second;
        if (consumed_offsets != nullptr) {
            AssertInfo(offset >= 0 && static_cast<size_t>(offset) <
                                          consumed_offsets->size(),
                       "group-by iterator returned invalid logical offset {} "
                       "for row count {}",
                       offset,
                       consumed_offsets->size());
            (*consumed_offsets)[offset] = true;
        }
        auto group = data_getter->Get(offset);
        if (group_map.Push(group)) {
            collector.Add(offset, distance, std::move(group));
        }
    }
    return candidates;
}

template <typename T>
bool
AreGroupsFull(const GroupByMap<T>& group_map,
              const std::vector<std::optional<T>>& groups) {
    return std::all_of(groups.begin(), groups.end(), [&](const auto& group) {
        return group_map.IsGroupFull(group);
    });
}

template <typename T>
std::vector<std::optional<T>>
SelectBatchGroups(const std::vector<std::optional<T>>& groups,
                  const std::vector<size_t>& ordinals) {
    std::vector<std::optional<T>> batch;
    batch.reserve(ordinals.size());
    for (auto ordinal : ordinals) {
        AssertInfo(ordinal < groups.size(),
                   "strict group phase2 batch ordinal {} is out of range {}",
                   ordinal,
                   groups.size());
        batch.emplace_back(groups[ordinal]);
    }
    return batch;
}

template <typename T>
bool
TryStrictGroupFilteredPhase2(const std::shared_ptr<VectorIterator>& iterator,
                             const std::shared_ptr<DataGetter<T>>& data_getter,
                             GroupByMap<T>& group_map,
                             GroupByResultCollector<T>& collector,
                             const StrictGroupPhase2Context* context) {
    if (context == nullptr || !context->eligible ||
        context->search_result == nullptr) {
        return false;
    }

    StrictGroupPhase2Stats stats;
    stats.attempted = true;
    auto finish = [&] { RecordStrictGroupPhase2Stats(stats); };
    if (!context->search_result->CanRecreateVectorIterator()) {
        stats.fallback_reason =
            StrictGroupPhase2FallbackReason::MissingRecreator;
        finish();
        return false;
    }
    if (context->search_result->total_data_cnt_ < 0) {
        stats.fallback_reason =
            StrictGroupPhase2FallbackReason::InvalidRowCount;
        finish();
        return false;
    }

    TargetBitmap phase1_offsets(context->search_result->total_data_cnt_, false);
    stats.phase1_candidates = ConsumeGroupByIteratorUntil(
        iterator,
        data_getter,
        group_map,
        collector,
        [&] { return group_map.IsGroupCapacityReached(); },
        &phase1_offsets);

    if (!group_map.IsGroupCapacityReached()) {
        stats.fallback_reason =
            StrictGroupPhase2FallbackReason::Phase1Exhausted;
        finish();
        return true;
    }
    if (group_map.IsGroupResEnough()) {
        stats.fallback_reason =
            StrictGroupPhase2FallbackReason::Phase2NotNeeded;
        finish();
        return true;
    }

    std::vector<std::optional<T>> unfinished_groups;
    for (const auto& group : group_map.GetGroupOrder()) {
        if (!group_map.IsGroupFull(group)) {
            unfinished_groups.emplace_back(group);
        }
    }
    AssertInfo(!unfinished_groups.empty(),
               "strict group phase2 has no unfinished group");

    auto membership_start = std::chrono::steady_clock::now();
    auto membership = BuildGroupMembership<T>(
        context->op_ctx,
        context->segment,
        context->group_by_field_id,
        context->search_result->total_data_cnt_,
        unfinished_groups,
        context->search_result->GetVectorIteratorBaseFilter());
    stats.membership_build_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - membership_start)
            .count();
    if (!membership.has_value()) {
        stats.fallback_reason =
            StrictGroupPhase2FallbackReason::MembershipUnavailable;
        ConsumeGroupByIteratorUntil(
            iterator, data_getter, group_map, collector, [&] {
                return group_map.IsGroupResEnough();
            });
        finish();
        return true;
    }

    stats.eligible_rows = membership->EligibleRowCount();

    auto plan = BuildStrictGroupPhase2Plan(membership->EligibleRowCount(),
                                           membership->GroupRowCounts());
    if (!plan.has_value()) {
        stats.fallback_reason =
            StrictGroupPhase2FallbackReason::PlanUnavailable;
        ConsumeGroupByIteratorUntil(
            iterator, data_getter, group_map, collector, [&] {
                return group_map.IsGroupResEnough();
            });
        finish();
        return true;
    }
    stats.unfinished_group_rows = plan->combined_rows;
    stats.small_group_count = plan->small_group_count;
    stats.large_group_count = plan->large_group_count;
    stats.batch_count = plan->batches.size();

    bool phase2_started = false;
    for (const auto& batch_ordinals : plan->batches) {
        auto batch_groups =
            SelectBatchGroups(unfinished_groups, batch_ordinals);
        auto bitmap_start = std::chrono::steady_clock::now();
        auto batch_membership = membership->BuildMembership(batch_groups);
        if (!batch_membership.has_value()) {
            stats.bitmap_build_us +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - bitmap_start)
                    .count();
            stats.fallback_reason =
                StrictGroupPhase2FallbackReason::MembershipBitmapUnavailable;
            if (!phase2_started) {
                ConsumeGroupByIteratorUntil(
                    iterator, data_getter, group_map, collector, [&] {
                        return group_map.IsGroupResEnough();
                    });
                finish();
                return true;
            }
            finish();
            ThrowInfo(UnexpectedError,
                      "failed to build strict group phase2 membership after "
                      "phase2 started");
        }

        batch_membership->flip();
        *batch_membership |= phase1_offsets;
        stats.bitmap_build_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - bitmap_start)
                .count();

        auto recreated =
            context->search_result->RecreateVectorIterators(*batch_membership);
        if (!recreated.has_value()) {
            stats.fallback_reason =
                StrictGroupPhase2FallbackReason::RecreateUnavailable;
            if (!phase2_started) {
                ConsumeGroupByIteratorUntil(
                    iterator, data_getter, group_map, collector, [&] {
                        return group_map.IsGroupResEnough();
                    });
                finish();
                return true;
            }
            finish();
            ThrowInfo(UnexpectedError,
                      "failed to recreate strict group iterator after phase2 "
                      "started");
        }

        auto& batch_result = **recreated;
        AssertInfo(batch_result.vector_iterators_.has_value(),
                   "strict group phase2 recreated result has no iterator "
                   "container");
        AssertInfo(batch_result.vector_iterators_->size() <= 1,
                   "strict group phase2 expected at most one iterator, got {}",
                   batch_result.vector_iterators_->size());
        phase2_started = true;
        stats.used = true;
        if (batch_result.vector_iterators_->empty()) {
            context->search_result->search_storage_cost_ +=
                batch_result.search_storage_cost_;
            continue;
        }
        stats.phase2_candidates += ConsumeGroupByIteratorUntil(
            batch_result.vector_iterators_->front(),
            data_getter,
            group_map,
            collector,
            [&] { return AreGroupsFull(group_map, batch_groups); });
        context->search_result->search_storage_cost_ +=
            batch_result.search_storage_cost_;
        if (group_map.IsGroupResEnough()) {
            break;
        }
    }
    finish();
    return true;
}

}  // namespace

template <typename T>
void
GroupIteratorsByType(
    const std::vector<std::shared_ptr<VectorIterator>>& iterators,
    int64_t topK,
    int64_t group_size,
    bool strict_group_size,
    const std::shared_ptr<DataGetter<T>>& data_getter,
    std::vector<GroupByValueType>& group_by_values,
    std::vector<int64_t>& seg_offsets,
    std::vector<float>& distances,
    const knowhere::MetricType& metrics_type,
    std::vector<size_t>& topk_per_nq_prefix_sum,
    const StrictGroupPhase2Context* context = nullptr);

void
SearchGroupBy(milvus::OpContext* op_ctx,
              const std::vector<std::shared_ptr<VectorIterator>>& iterators,
              const SearchInfo& search_info,
              std::vector<GroupByValueType>& group_by_values,
              const segcore::SegmentInternalInterface& segment,
              std::vector<int64_t>& seg_offsets,
              std::vector<float>& distances,
              std::vector<size_t>& topk_per_nq_prefix_sum,
              SearchResult* search_result) {
    //1. get search meta
    FieldId group_by_field_id = search_info.group_by_field_id_.value();
    auto data_type = segment.GetFieldDataType(group_by_field_id);
    int max_total_size =
        search_info.topk_ * search_info.group_size_ * iterators.size();
    seg_offsets.reserve(max_total_size);
    distances.reserve(max_total_size);
    group_by_values.reserve(max_total_size);
    topk_per_nq_prefix_sum.reserve(iterators.size() + 1);
    StrictGroupPhase2Context phase2_context{
        op_ctx,
        segment,
        group_by_field_id,
        search_result,
        query::CanUseStrictGroupFilteredIterator(search_info,
                                                 iterators.size())};
    switch (data_type) {
        case DataType::INT8: {
            auto dataGetter =
                GetDataGetter<int8_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int8_t>(iterators,
                                         search_info.topk_,
                                         search_info.group_size_,
                                         search_info.strict_group_size_,
                                         dataGetter,
                                         group_by_values,
                                         seg_offsets,
                                         distances,
                                         search_info.metric_type_,
                                         topk_per_nq_prefix_sum,
                                         &phase2_context);
            break;
        }
        case DataType::INT16: {
            auto dataGetter =
                GetDataGetter<int16_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int16_t>(iterators,
                                          search_info.topk_,
                                          search_info.group_size_,
                                          search_info.strict_group_size_,
                                          dataGetter,
                                          group_by_values,
                                          seg_offsets,
                                          distances,
                                          search_info.metric_type_,
                                          topk_per_nq_prefix_sum,
                                          &phase2_context);
            break;
        }
        case DataType::INT32: {
            auto dataGetter =
                GetDataGetter<int32_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int32_t>(iterators,
                                          search_info.topk_,
                                          search_info.group_size_,
                                          search_info.strict_group_size_,
                                          dataGetter,
                                          group_by_values,
                                          seg_offsets,
                                          distances,
                                          search_info.metric_type_,
                                          topk_per_nq_prefix_sum,
                                          &phase2_context);
            break;
        }
        case DataType::INT64: {
            auto dataGetter =
                GetDataGetter<int64_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int64_t>(iterators,
                                          search_info.topk_,
                                          search_info.group_size_,
                                          search_info.strict_group_size_,
                                          dataGetter,
                                          group_by_values,
                                          seg_offsets,
                                          distances,
                                          search_info.metric_type_,
                                          topk_per_nq_prefix_sum,
                                          &phase2_context);
            break;
        }
        case DataType::TIMESTAMPTZ: {
            auto dataGetter =
                GetDataGetter<int64_t>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<int64_t>(iterators,
                                          search_info.topk_,
                                          search_info.group_size_,
                                          search_info.strict_group_size_,
                                          dataGetter,
                                          group_by_values,
                                          seg_offsets,
                                          distances,
                                          search_info.metric_type_,
                                          topk_per_nq_prefix_sum,
                                          &phase2_context);
            break;
        }
        case DataType::BOOL: {
            auto dataGetter =
                GetDataGetter<bool>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<bool>(iterators,
                                       search_info.topk_,
                                       search_info.group_size_,
                                       search_info.strict_group_size_,
                                       dataGetter,
                                       group_by_values,
                                       seg_offsets,
                                       distances,
                                       search_info.metric_type_,
                                       topk_per_nq_prefix_sum,
                                       &phase2_context);
            break;
        }
        case DataType::VARCHAR: {
            auto dataGetter =
                GetDataGetter<std::string>(op_ctx, segment, group_by_field_id);
            GroupIteratorsByType<std::string>(iterators,
                                              search_info.topk_,
                                              search_info.group_size_,
                                              search_info.strict_group_size_,
                                              dataGetter,
                                              group_by_values,
                                              seg_offsets,
                                              distances,
                                              search_info.metric_type_,
                                              topk_per_nq_prefix_sum,
                                              &phase2_context);
            break;
        }
        case DataType::JSON: {
            AssertInfo(search_info.json_path_.has_value(),
                       "json_path is required for json field when doing "
                       "search_group_by");
            if (search_info.json_type_.has_value()) {
                switch (search_info.json_type_.value()) {
                    case DataType::BOOL: {
                        auto data_getter = GetDataGetter<bool, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<bool>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::INT8: {
                        auto data_getter = GetDataGetter<int8_t, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<int8_t>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::INT16: {
                        auto data_getter = GetDataGetter<int16_t, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<int16_t>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::INT32: {
                        auto data_getter = GetDataGetter<int32_t, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<int32_t>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::INT64: {
                        auto data_getter = GetDataGetter<int64_t, milvus::Json>(
                            op_ctx,
                            segment,
                            group_by_field_id,
                            search_info.json_path_,
                            search_info.json_type_,
                            search_info.strict_cast_);
                        GroupIteratorsByType<int64_t>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    case DataType::VARCHAR: {
                        auto data_getter =
                            GetDataGetter<std::string, milvus::Json>(
                                op_ctx,
                                segment,
                                group_by_field_id,
                                search_info.json_path_,
                                search_info.json_type_,
                                search_info.strict_cast_);
                        GroupIteratorsByType<std::string>(
                            iterators,
                            search_info.topk_,
                            search_info.group_size_,
                            search_info.strict_group_size_,
                            data_getter,
                            group_by_values,
                            seg_offsets,
                            distances,
                            search_info.metric_type_,
                            topk_per_nq_prefix_sum);
                        break;
                    }
                    default: {
                        ThrowInfo(Unsupported,
                                  fmt::format("unsupported data type {} for "
                                              "group by operator",
                                              data_type));
                    }
                }
            } else {
                auto data_getter = GetDataGetter<std::string, milvus::Json>(
                    op_ctx,
                    segment,
                    group_by_field_id,
                    search_info.json_path_,
                    search_info.json_type_,
                    search_info.strict_cast_);
                GroupIteratorsByType<std::string>(
                    iterators,
                    search_info.topk_,
                    search_info.group_size_,
                    search_info.strict_group_size_,
                    data_getter,
                    group_by_values,
                    seg_offsets,
                    distances,
                    search_info.metric_type_,
                    topk_per_nq_prefix_sum);
            }
            break;
        }
        default: {
            ThrowInfo(
                Unsupported,
                fmt::format("unsupported data type {} for group by operator",
                            data_type));
        }
    }
}

template <typename T>
void
GroupIteratorResult(const std::shared_ptr<VectorIterator>& iterator,
                    int64_t topK,
                    int64_t group_size,
                    bool strict_group_size,
                    const std::shared_ptr<DataGetter<T>>& data_getter,
                    std::vector<GroupByValueType>& group_by_values,
                    std::vector<int64_t>& offsets,
                    std::vector<float>& distances,
                    const knowhere::MetricType& metrics_type,
                    const StrictGroupPhase2Context* context);

template <typename T>
void
GroupIteratorsByType(
    const std::vector<std::shared_ptr<VectorIterator>>& iterators,
    int64_t topK,
    int64_t group_size,
    bool strict_group_size,
    const std::shared_ptr<DataGetter<T>>& data_getter,
    std::vector<GroupByValueType>& group_by_values,
    std::vector<int64_t>& seg_offsets,
    std::vector<float>& distances,
    const knowhere::MetricType& metrics_type,
    std::vector<size_t>& topk_per_nq_prefix_sum,
    const StrictGroupPhase2Context* context) {
    topk_per_nq_prefix_sum.push_back(0);
    for (auto& iterator : iterators) {
        GroupIteratorResult<T>(iterator,
                               topK,
                               group_size,
                               strict_group_size,
                               data_getter,
                               group_by_values,
                               seg_offsets,
                               distances,
                               metrics_type,
                               context);
        topk_per_nq_prefix_sum.push_back(seg_offsets.size());
    }
}

template <typename T>
void
GroupIteratorResult(const std::shared_ptr<VectorIterator>& iterator,
                    int64_t topK,
                    int64_t group_size,
                    bool strict_group_size,
                    const std::shared_ptr<DataGetter<T>>& data_getter,
                    std::vector<GroupByValueType>& group_by_values,
                    std::vector<int64_t>& offsets,
                    std::vector<float>& distances,
                    const knowhere::MetricType& metrics_type,
                    const StrictGroupPhase2Context* context) {
    GroupByMap<T> group_map(topK, group_size, strict_group_size);
    GroupByResultCollector<T> collector;

    auto handled_by_filtered_phase2 =
        strict_group_size &&
        TryStrictGroupFilteredPhase2(
            iterator, data_getter, group_map, collector, context);
    if (!handled_by_filtered_phase2) {
        // Do iteration until fill the whole map or run out of all data. It may
        // enumerate every row in a segment and block following work.
        ConsumeGroupByIteratorUntil(
            iterator, data_getter, group_map, collector, [&] {
                return group_map.IsGroupResEnough();
            });
    }

    collector.SortAndAppend(metrics_type, group_by_values, offsets, distances);
}

}  // namespace exec
}  // namespace milvus
