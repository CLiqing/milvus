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
#include "common/Consts.h"
#include "exec/operator/groupby/GroupMembership.h"
#include "exec/operator/groupby/StrictGroupPhase2Planner.h"
#include "query/Utils.h"
#include "common/JsonUtils.h"

namespace milvus {
namespace exec {

namespace {

struct StrictGroupPhase2Context {
    milvus::OpContext* op_ctx;
    const segcore::SegmentInternalInterface& segment;
    FieldId group_by_field_id;
    SearchResult* search_result;
};

template <typename T, typename StopPredicate>
void
ConsumeGroupByIteratorUntil(const std::shared_ptr<VectorIterator>& iterator,
                            const std::shared_ptr<DataGetter<T>>& data_getter,
                            GroupByMap<T>& group_map,
                            GroupByResultCollector<T>& collector,
                            StopPredicate&& should_stop,
                            TargetBitmap* consumed_offsets = nullptr) {
    while (iterator->HasNext() && !should_stop()) {
        auto offset_dis_pair = iterator->Next();
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
                             int64_t topK,
                             int64_t group_size,
                             const std::shared_ptr<DataGetter<T>>& data_getter,
                             GroupByMap<T>& group_map,
                             GroupByResultCollector<T>& collector,
                             const StrictGroupPhase2Context* context) {
    if (context == nullptr || context->search_result == nullptr ||
        !context->search_result->CanRecreateVectorIterator() || topK <= 0 ||
        group_size <= 1 || context->search_result->total_nq_ != 1 ||
        context->search_result->total_data_cnt_ < 0) {
        return false;
    }

    TargetBitmap phase1_offsets(context->search_result->total_data_cnt_, false);
    ConsumeGroupByIteratorUntil(
        iterator,
        data_getter,
        group_map,
        collector,
        [&] { return group_map.IsGroupCapacityReached(); },
        &phase1_offsets);

    if (!group_map.IsGroupCapacityReached() || group_map.IsGroupResEnough()) {
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

    auto membership = BuildGroupMembership<T>(
        context->op_ctx,
        context->segment,
        context->group_by_field_id,
        context->search_result->total_data_cnt_,
        unfinished_groups,
        context->search_result->GetVectorIteratorBaseFilter());
    if (!membership.has_value()) {
        ConsumeGroupByIteratorUntil(
            iterator, data_getter, group_map, collector, [&] {
                return group_map.IsGroupResEnough();
            });
        return true;
    }

    auto plan = BuildStrictGroupPhase2Plan(membership->EligibleRowCount(),
                                           membership->GroupRowCounts());
    if (!plan.has_value()) {
        ConsumeGroupByIteratorUntil(
            iterator, data_getter, group_map, collector, [&] {
                return group_map.IsGroupResEnough();
            });
        return true;
    }

    bool phase2_started = false;
    for (const auto& batch_ordinals : plan->batches) {
        auto batch_groups =
            SelectBatchGroups(unfinished_groups, batch_ordinals);
        auto batch_membership = membership->BuildMembership(batch_groups);
        if (!batch_membership.has_value()) {
            if (!phase2_started) {
                ConsumeGroupByIteratorUntil(
                    iterator, data_getter, group_map, collector, [&] {
                        return group_map.IsGroupResEnough();
                    });
                return true;
            }
            ThrowInfo(UnexpectedError,
                      "failed to build strict group phase2 membership after "
                      "phase2 started");
        }

        batch_membership->flip();
        *batch_membership |= phase1_offsets;

        auto recreated =
            context->search_result->RecreateVectorIterators(*batch_membership);
        if (!recreated.has_value()) {
            if (!phase2_started) {
                ConsumeGroupByIteratorUntil(
                    iterator, data_getter, group_map, collector, [&] {
                        return group_map.IsGroupResEnough();
                    });
                return true;
            }
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
        if (batch_result.vector_iterators_->empty()) {
            context->search_result->search_storage_cost_ +=
                batch_result.search_storage_cost_;
            continue;
        }
        ConsumeGroupByIteratorUntil(
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
        op_ctx, segment, group_by_field_id, search_result};
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
        strict_group_size && TryStrictGroupFilteredPhase2(iterator,
                                                          topK,
                                                          group_size,
                                                          data_getter,
                                                          group_map,
                                                          collector,
                                                          context);
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
