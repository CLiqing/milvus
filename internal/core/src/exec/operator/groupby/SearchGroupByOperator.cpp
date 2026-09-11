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
#include <atomic>
#include <limits>

#include "common/Tracer.h"
#include "common/Consts.h"
#include "common/JsonUtils.h"
#include "exec/operator/groupby/GroupMembership.h"
#include "fmt/format.h"
#include "monitor/Monitor.h"
#include "query/Utils.h"
#include "segcore/Utils.h"

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
    ProbeAcceptanceHigh,
    RecreateUnavailable,
    RecreatedExhausted,
};

enum class StrictGroupDecision {
    NotEvaluated,
    AcceptanceHigh,
    AcceptanceLow,
    FilteredIterator,
    PerGroup
};

const char*
DecisionName(StrictGroupDecision decision) {
    switch (decision) {
        case StrictGroupDecision::NotEvaluated:
            return "not_evaluated";
        case StrictGroupDecision::AcceptanceHigh:
            return "acceptance_high";
        case StrictGroupDecision::AcceptanceLow:
            return "acceptance_low";
        case StrictGroupDecision::FilteredIterator:
            return "filtered_iterator";
        case StrictGroupDecision::PerGroup:
            return "per_group";
    }
    return "unknown";
}

struct StrictGroupPhase2Stats {
    bool attempted = false;
    bool used = false;
    size_t phase1_candidates = 0;
    size_t phase2_candidates = 0;
    size_t probe_candidates = 0;
    size_t probe_accepted = 0;
    size_t probe_group_hits = 0;
    size_t original_remaining_candidates = 0;
    StrictGroupDecision decision = StrictGroupDecision::NotEvaluated;
    size_t batch_count = 0;
    uint64_t membership_build_us = 0;
    uint64_t bitmap_build_us = 0;
    uint64_t recreate_us = 0;
    uint64_t search_us = 0;
    StrictGroupPhase2FallbackReason fallback_reason =
        StrictGroupPhase2FallbackReason::None;
};

struct StrictGroupPhase2Context {
    milvus::OpContext* op_ctx;
    const segcore::SegmentInternalInterface& segment;
    FieldId group_by_field_id;
    SearchResult* search_result;
    bool eligible;
    double acceptance_threshold;
    int64_t probe_candidates;
    StrictGroupStrategy strategy;
    const SearchInfo* search_info;
};

const char*
StrategyName(StrictGroupStrategy strategy) {
    switch (strategy) {
        case StrictGroupStrategy::Sampling:
            return "sampling";
        case StrictGroupStrategy::FilteredIterator:
            return "filtered_iterator";
        case StrictGroupStrategy::PerGroup:
            return "per_group";
    }
    return "unknown";
}

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
        case StrictGroupPhase2FallbackReason::ProbeAcceptanceHigh:
            return "probe_acceptance_high";
        case StrictGroupPhase2FallbackReason::RecreateUnavailable:
            return "recreate_unavailable";
        case StrictGroupPhase2FallbackReason::RecreatedExhausted:
            return "recreated_exhausted";
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
    milvus::monitor::internal_core_strict_group_phase2_probe_candidates.Observe(
        stats.probe_candidates);
    milvus::monitor::internal_core_strict_group_phase2_probe_accepted.Observe(
        stats.probe_accepted);
    milvus::monitor::internal_core_strict_group_phase2_probe_group_hits.Observe(
        stats.probe_group_hits);
    milvus::monitor::
        internal_core_strict_group_phase2_original_remaining_candidates.Observe(
            stats.original_remaining_candidates);
    milvus::monitor::internal_core_strict_group_phase2_membership_build_latency
        .Observe(stats.membership_build_us / 1000.0);
    milvus::monitor::internal_core_strict_group_phase2_bitmap_build_latency
        .Observe(stats.bitmap_build_us / 1000.0);
    milvus::monitor::internal_core_strict_group_phase2_recreate_latency.Observe(
        stats.recreate_us / 1000.0);
    milvus::monitor::internal_core_strict_group_phase2_search_latency.Observe(
        stats.search_us / 1000.0);
    if (stats.probe_candidates > 0) {
        milvus::monitor::internal_core_strict_group_phase2_acceptance_ratio
            .Observe(static_cast<double>(stats.probe_accepted) /
                     stats.probe_candidates);
    }
    tracer::AddEvent(fmt::format(
        "strict_group_phase2: used={}, fallback={}, phase1_candidates={}, "
        "phase2_candidates={}, probe_candidates={}, probe_accepted={}, "
        "probe_group_hits={}, "
        "original_remaining_candidates={}, decision={}, batches={}, "
        "membership_ms={:.3f}, "
        "bitmap_ms={:.3f}",
        stats.used,
        FallbackReasonName(stats.fallback_reason),
        stats.phase1_candidates,
        stats.phase2_candidates,
        stats.probe_candidates,
        stats.probe_accepted,
        stats.probe_group_hits,
        stats.original_remaining_candidates,
        DecisionName(stats.decision),
        stats.batch_count,
        stats.membership_build_us / 1000.0,
        stats.bitmap_build_us / 1000.0));
}

template <typename T, typename StopPredicate>
size_t
ConsumeGroupByIteratorUntil(
    const std::shared_ptr<VectorIterator>& iterator,
    const std::shared_ptr<DataGetter<T>>& data_getter,
    GroupByMap<T>& group_map,
    GroupByResultCollector<T>& collector,
    StopPredicate&& should_stop,
    size_t max_candidates = std::numeric_limits<size_t>::max(),
    size_t* accepted = nullptr,
    size_t* locked_group_hits = nullptr) {
    size_t candidates = 0;
    while (candidates < max_candidates && !should_stop() &&
           iterator->HasNext()) {
        auto offset_dis_pair = iterator->Next();
        ++candidates;
        AssertInfo(
            offset_dis_pair.has_value(),
            "Wrong state! iterator cannot return valid result whereas it "
            "still tells hasNext, terminate groupBy operation");
        auto offset = offset_dis_pair->first;
        auto distance = offset_dis_pair->second;
        if (collector.IsAcceptedOffset(offset)) {
            continue;
        }
        auto group = data_getter->Get(offset);
        if (locked_group_hits != nullptr && group_map.Contains(group)) {
            ++*locked_group_hits;
        }
        if (group_map.Push(group)) {
            collector.Add(offset, distance, std::move(group));
            if (accepted != nullptr) {
                ++*accepted;
            }
        }
    }
    return candidates;
}

template <typename T>
bool
TryStrictGroupFilteredPhase2(const std::shared_ptr<VectorIterator>& iterator,
                             const std::shared_ptr<DataGetter<T>>& data_getter,
                             GroupByMap<T>& group_map,
                             GroupByResultCollector<T>& collector,
                             const StrictGroupPhase2Context* context) {
    if (context == nullptr || context->search_result == nullptr) {
        return false;
    }

    StrictGroupPhase2Stats stats;
    const bool debug = context->search_info->strict_group_debug_;
    using Clock = std::chrono::steady_clock;
    Clock::time_point debug_start{}, last_log{};
    uint64_t diagnostic_id = 0;
    std::string trace_id;
    if (debug) {
        // Local ID correlates stages even when tracing was not propagated.
        // It is process-local, not a substitute for a cross-segment request ID.
        static std::atomic<uint64_t> next_id{0};
        diagnostic_id = ++next_id;
        debug_start = last_log = Clock::now();
        if (context->search_info->trace_ctx_.traceID != nullptr) {
            trace_id =
                tracer::GetTraceIDAsHexStr(&context->search_info->trace_ctx_);
        }
    }
    auto diagnostic = [&](const char* stage,
                          int64_t available_rows = -1,
                          int64_t group_ordinal = -1,
                          int64_t requested_k = -1) {
        if (!debug) {
            return;
        }
        const auto now = Clock::now();
        int64_t remaining_rows = 0;
        for (const auto& group : group_map.GetGroupOrder()) {
            remaining_rows += group_map.GetRemainingGroupSize(group);
        }
        // Unknown values are -1. No customer field values or vector payloads.
        knowhere::Json record = {
            {"stage", stage},
            {"diagnostic_id", diagnostic_id},
            {"trace_id", trace_id},
            {"segment_id", context->segment.get_segment_id()},
            {"strategy", StrategyName(context->strategy)},
            {"eligible", context->eligible},
            {"probe_budget", context->probe_candidates},
            {"acceptance_threshold", context->acceptance_threshold},
            {"topk", context->search_info->topk_},
            {"group_size", context->search_info->group_size_},
            {"segment_rows", context->search_result->total_data_cnt_},
            {"locked_groups", group_map.GetGroupCount()},
            {"unfinished_groups",
             group_map.GetGroupCount() - group_map.GetEnoughGroupCount()},
            {"remaining_locked_rows", remaining_rows},
            {"accepted_rows", collector.Size()},
            {"available_rows", available_rows},
            {"group_ordinal", group_ordinal},
            {"requested_k", requested_k},
            {"fallback", FallbackReasonName(stats.fallback_reason)},
            {"decision", DecisionName(stats.decision)},
            {"phase1_candidates_including_probe", stats.phase1_candidates},
            {"probe_candidates", stats.probe_candidates},
            {"probe_accepted", stats.probe_accepted},
            {"probe_group_hits", stats.probe_group_hits},
            {"phase2_candidates", stats.phase2_candidates},
            {"original_remaining_candidates",
             stats.original_remaining_candidates},
            {"batches", stats.batch_count},
            {"membership_us", stats.membership_build_us},
            {"bitmap_us", stats.bitmap_build_us},
            {"recreate_us", stats.recreate_us},
            {"search_us", stats.search_us},
            {"since_previous_log_us",
             std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                                   last_log)
                 .count()},
            {"elapsed_us",
             std::chrono::duration_cast<std::chrono::microseconds>(now -
                                                                   debug_start)
                 .count()}};
        LOG_INFO("strict_group_diagnostic {}", record.dump());
        last_log = now;
    };
    diagnostic(context->eligible ? "begin" : "ineligible_original");
    if (!context->eligible) {
        return false;
    }
    stats.attempted = true;
    auto finish = [&] {
        RecordStrictGroupPhase2Stats(stats);
        diagnostic("finish");
    };
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

    stats.phase1_candidates = ConsumeGroupByIteratorUntil(
        iterator, data_getter, group_map, collector, [&] {
            return group_map.IsGroupCapacityReached();
        });
    diagnostic("groups_locked");

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

    auto continue_original = [&] {
        diagnostic("original_begin");
        stats.original_remaining_candidates = ConsumeGroupByIteratorUntil(
            iterator, data_getter, group_map, collector, [&] {
                return group_map.IsGroupResEnough();
            });
        diagnostic("original_end");
        finish();
        return true;
    };

    if (context->strategy == StrictGroupStrategy::PerGroup) {
        stats.decision = StrictGroupDecision::PerGroup;
        if (!context->search_result->CanSearchFilteredVectors()) {
            stats.fallback_reason =
                StrictGroupPhase2FallbackReason::RecreateUnavailable;
            return continue_original();
        }
        std::vector<std::optional<T>> groups;
        for (const auto& group : group_map.GetGroupOrder()) {
            if (!group_map.IsGroupFull(group)) {
                groups.emplace_back(group);
            }
        }
        const auto start = std::chrono::steady_clock::now();
        auto offsets = BuildGroupOffsets<T>(
            context->op_ctx,
            context->segment,
            context->group_by_field_id,
            context->search_result->total_data_cnt_,
            groups,
            context->search_result->GetVectorIteratorBaseFilter());
        stats.membership_build_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        diagnostic("per_group_membership_ready");
        if (!offsets) {
            stats.fallback_reason =
                StrictGroupPhase2FallbackReason::MembershipUnavailable;
            return continue_original();
        }

        auto bitmap_start = std::chrono::steady_clock::now();
        auto filter = std::make_shared<TargetBitmap>(
            context->search_result->total_data_cnt_, true);
        stats.bitmap_build_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - bitmap_start)
                .count();
        collector.EnableOffsetDeduplication();
        for (size_t i = 0; i < groups.size(); ++i) {
            segcore::CheckCancellation(context->op_ctx,
                                       context->segment.get_segment_id(),
                                       context->group_by_field_id.get(),
                                       "strict per-group search");
            bitmap_start = std::chrono::steady_clock::now();
            size_t available = 0;
            for (auto offset : (*offsets)[i]) {
                if (!collector.IsAcceptedOffset(offset)) {
                    (*filter)[offset] = false;
                    ++available;
                }
            }
            stats.bitmap_build_us +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - bitmap_start)
                    .count();
            const auto remaining = group_map.GetRemainingGroupSize(groups[i]);
            diagnostic("group_search_begin", available, i, remaining);
            if (available == 0) {
                diagnostic("group_search_empty", available, i, remaining);
                continue;
            }
            auto search_start = std::chrono::steady_clock::now();
            auto batch = context->search_result->SearchFilteredVectors(
                filter, remaining);
            stats.search_us +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - search_start)
                    .count();
            if (!batch) {
                stats.fallback_reason =
                    StrictGroupPhase2FallbackReason::RecreateUnavailable;
                return continue_original();
            }
            auto& result = **batch;
            AssertInfo(result.seg_offsets_.size() == result.distances_.size() &&
                           result.seg_offsets_.size() <= remaining,
                       "invalid strict per-group Search result shape");
            stats.used = true;
            ++stats.batch_count;
            for (size_t j = 0; j < result.seg_offsets_.size(); ++j) {
                auto offset = result.seg_offsets_[j];
                if (offset == INVALID_SEG_OFFSET) {
                    continue;
                }
                // Consumer results, not internal ANN distance computations.
                ++stats.phase2_candidates;
                AssertInfo(offset >= 0 && offset < filter->size() &&
                               !(*filter)[offset],
                           "filtered group Search returned an excluded row");
                if (!collector.IsAcceptedOffset(offset) &&
                    group_map.Push(groups[i])) {
                    collector.Add(offset, result.distances_[j], groups[i]);
                }
            }
            context->search_result->search_storage_cost_ +=
                result.search_storage_cost_;
            diagnostic("group_search_end", available, i, remaining);
            // Search is synchronous; release temporary buffers before reuse.
            batch.reset();
            bitmap_start = std::chrono::steady_clock::now();
            for (auto offset : (*offsets)[i]) {
                (*filter)[offset] = true;
            }
            stats.bitmap_build_us +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - bitmap_start)
                    .count();
        }
        if (!group_map.IsGroupResEnough()) {
            stats.fallback_reason =
                StrictGroupPhase2FallbackReason::RecreatedExhausted;
            return continue_original();
        }
        finish();
        return true;
    }

    if (context->strategy == StrictGroupStrategy::Sampling) {
        // Probe once, using consumer candidates rather than backend graph visits.
        stats.probe_candidates = ConsumeGroupByIteratorUntil(
            iterator,
            data_getter,
            group_map,
            collector,
            [&] { return group_map.IsGroupResEnough(); },
            context->probe_candidates,
            &stats.probe_accepted,
            &stats.probe_group_hits);
        stats.phase1_candidates += stats.probe_candidates;
        diagnostic("probe_end");
        if (group_map.IsGroupResEnough() || !iterator->HasNext()) {
            stats.fallback_reason =
                StrictGroupPhase2FallbackReason::Phase2NotNeeded;
            finish();
            return true;
        }

        // Equality keeps the original iterator; never re-evaluate this decision later.
        if (static_cast<double>(stats.probe_accepted) /
                stats.probe_candidates >=
            context->acceptance_threshold) {
            stats.decision = StrictGroupDecision::AcceptanceHigh;
            stats.fallback_reason =
                StrictGroupPhase2FallbackReason::ProbeAcceptanceHigh;
            return continue_original();
        }
        stats.decision = StrictGroupDecision::AcceptanceLow;
    } else {
        // Explicit strategy: skip sampling but reuse the union-filter pipeline.
        stats.decision = StrictGroupDecision::FilteredIterator;
    }

    std::vector<std::optional<T>> unfinished_groups;
    for (const auto& group : group_map.GetGroupOrder()) {
        if (!group_map.IsGroupFull(group)) {
            unfinished_groups.emplace_back(group);
        }
    }
    AssertInfo(!unfinished_groups.empty(),
               "strict group phase2 has no unfinished group");

    auto prepare = [&]() -> std::optional<std::unique_ptr<SearchResult>> {
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
            return std::nullopt;
        }

        auto bitmap_start = std::chrono::steady_clock::now();
        membership->flip();
        collector.ExcludeAcceptedOffsets(*membership);
        stats.bitmap_build_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - bitmap_start)
                .count();
        // A full bitmap count is diagnostic-only; do not add it to timing runs.
        diagnostic("union_filter_ready",
                   debug ? static_cast<int64_t>(membership->size() -
                                                membership->count())
                         : -1);
        auto recreate_start = std::chrono::steady_clock::now();
        auto result = context->search_result->RecreateVectorIterators(
            std::move(*membership));
        stats.recreate_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - recreate_start)
                .count();
        diagnostic("iterator_created");
        return result;
    };
    std::optional<std::unique_ptr<SearchResult>> recreated;
    try {
        recreated = prepare();
    } catch (const std::exception& error) {
        LOG_WARN("strict group iterator preparation failed: {}", error.what());
        throw;
    }
    if (!recreated.has_value()) {
        if (stats.fallback_reason == StrictGroupPhase2FallbackReason::None) {
            stats.fallback_reason =
                StrictGroupPhase2FallbackReason::RecreateUnavailable;
        }
        return continue_original();
    }
    auto& batch_result = **recreated;
    AssertInfo(batch_result.vector_iterators_.has_value(),
               "strict group recreated result has no iterator container");
    AssertInfo(batch_result.vector_iterators_->size() <= 1,
               "strict group expected at most one iterator, got {}",
               batch_result.vector_iterators_->size());
    collector.EnableOffsetDeduplication();
    stats.used = true;
    stats.batch_count = 1;
    diagnostic("filtered_consume_begin");
    if (!batch_result.vector_iterators_->empty()) {
        stats.phase2_candidates = ConsumeGroupByIteratorUntil(
            batch_result.vector_iterators_->front(),
            data_getter,
            group_map,
            collector,
            [&] { return group_map.IsGroupResEnough(); });
    }
    diagnostic("filtered_consume_end");
    context->search_result->search_storage_cost_ +=
        batch_result.search_storage_cost_;
    if (!group_map.IsGroupResEnough()) {
        stats.fallback_reason =
            StrictGroupPhase2FallbackReason::RecreatedExhausted;
        return continue_original();
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
    Defer clear_recreator([&] {
        if (search_result != nullptr) {
            search_result->ClearVectorIteratorRecreator();
        }
    });
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
        query::CanUseStrictGroupFilteredIterator(search_info, iterators.size()),
        search_info.strict_group_acceptance_threshold_,
        search_info.strict_group_probe_candidates_,
        search_info.strict_group_strategy_,
        &search_info};
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
