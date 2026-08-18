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

#include "FilterBitsNode.h"

#include <algorithm>
#include <chrono>
#include <ratio>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "exec/QueryContext.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/ExprCache.h"
#include "expr/ITypeExpr.h"
#include "fmt/core.h"
#include "monitor/Monitor.h"
#include "plan/PlanNode.h"
#include "prometheus/histogram.h"

namespace milvus {
namespace exec {

namespace {

std::string
BuildExprCacheKey(const plan::FilterBitsNode& filter,
                  QueryContext* query_context) {
    auto key = filter.ToString();
    auto* segment =
        query_context != nullptr ? query_context->get_segment() : nullptr;
    if (segment != nullptr &&
        segment->get_schema_snapshot()->get_ttl_field_id().has_value()) {
        key += fmt::format("|entity_ttl_physical_time_us:{}",
                           query_context->get_entity_ttl_physical_time_us());
    }
    return key;
}

}  // namespace

bool
ConvertPredicateToFilteredBitset(TargetBitmapView data,
                                 TargetBitmapView valid,
                                 const size_t size) {
    // FilterBitsNode outputs a filtered-row bitset: 1 means excluded. A SQL-style
    // predicate passes only when it is definitely TRUE, so UNKNOWN/NULL must be
    // excluded together with FALSE.
    if (valid.all()) {
        data.flip();
        return true;
    }

    data.flip();
    TargetBitmap invalid(valid);
    invalid.flip();
    data.inplace_or(invalid, size);
    valid.set();
    return false;
}

PhyFilterBitsNode::PhyFilterBitsNode(
    int32_t operator_id,
    DriverContext* driverctx,
    const std::shared_ptr<const plan::FilterBitsNode>& filter)
    : Operator(driverctx,
               filter->output_type(),
               operator_id,
               filter->id(),
               "PhyFilterBitsNode") {
    ExecContext* exec_context = operator_context_->get_exec_context();
    query_context_ = exec_context->get_query_context();
    std::vector<expr::TypedExprPtr> filters;
    filters.emplace_back(filter->filter());
    // This operator folds UNKNOWN predicate rows into the excluded set
    // (ConvertPredicateToFilteredBitset), i.e. it is a null-rejecting
    // consumer: let conjunctions in the predicate tree drop UNKNOWN rows
    // from their active sets early.
    exprs_ = std::make_unique<ExprSet>(
        filters, exec_context, /*null_rejecting=*/true);
    need_process_rows_ = query_context_->get_active_count();
    num_processed_rows_ = 0;

    enable_expr_cache_ = query_context_->get_enable_expr_cache();
    if (enable_expr_cache_) {
        // Only cache the predicate result when EVERY expression in it is
        // cacheable. A bloom_match subtree is non-cacheable (its slim ToString
        // cache key cannot distinguish distinct filter blobs), and that
        // propagates up, so a predicate containing bloom_match is never cached
        // and can never reuse another query's bitmap.
        for (const auto& e : exprs_->exprs()) {
            if (e && !e->IsCacheable()) {
                enable_expr_cache_ = false;
                break;
            }
        }
    }
    if (enable_expr_cache_) {
        expr_cache_key_ = BuildExprCacheKey(*filter, query_context_);
    }
}

void
PhyFilterBitsNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

bool
PhyFilterBitsNode::AllInputProcessed() {
    if (num_processed_rows_ == need_process_rows_) {
        input_ = nullptr;
        return true;
    }
    return false;
}

bool
PhyFilterBitsNode::IsFinished() {
    return AllInputProcessed();
}

RowVectorPtr
PhyFilterBitsNode::GetOutput() {
    milvus::exec::checkCancellation(query_context_);

    if (AllInputProcessed()) {
        return nullptr;
    }

    const auto search_info = query_context_->get_search_info();
    // Retrieve plans have no vector-search params.  They still pass through
    // FilterBitsNode for ordinary scalar predicates, so treat a null params
    // value exactly as the default vector-search mode.
    const auto scan_mode = search_info.search_params_.is_object()
                               ? search_info.search_params_.value(
                                     "bf_filter_scan_mode",
                                     std::string{"auto"})
                               : std::string{"auto"};
    if (scan_mode == "roaring_valid_per_query") {
        ThrowInfo(ConfigInvalid,
                  "roaring_valid_per_query has been retired; use "
                  "valid_ids_per_query");
    }

    // The filter-result representation is decided once here and shared by the
    // cache gate and the native valid-ID path below.
    const bool use_sparse_representation =
        search_info.UseSparseFilterRepresentation();

    // Cache read: Stage 2 of two-stage search reuses the bitset cached by Stage 1.
    // Cache lives in the process-level ExprResCacheManager keyed by
    // (segment_id, FilterBitsNode signature + dynamic filter context), so
    // cross-query reuse is automatic only when the effective predicate matches.
    auto* cache_segment = query_context_->get_segment();
    const bool can_use_cache = !use_sparse_representation &&
                               enable_expr_cache_ && !expr_cache_key_.empty() &&
                               cache_segment != nullptr &&
                               cache_segment->type() == SegmentType::Sealed &&
                               ExprResCacheManager::IsEnabled();
    if (can_use_cache) {
        ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                     expr_cache_key_};
        ExprResCacheManager::Value cached;
        cached.active_count = need_process_rows_;
        if (ExprResCacheManager::Instance().Get(key, cached) &&
            cached.result != nullptr &&
            cached.result->size() == need_process_rows_) {
            num_processed_rows_ = need_process_rows_;
            std::vector<VectorPtr> col_res;
            col_res.push_back(std::make_shared<ColumnVector>(
                cached.result->clone(),
                cached.valid_result ? cached.valid_result->clone()
                                    : TargetBitmap(need_process_rows_, true)));
            return std::make_shared<RowVector>(col_res);
        }
    }

    // Native accepted-ID path for Cardinal.  BitmapIndex may retain a
    // Roaring posting internally, but the only payload crossing this boundary
    // is a valid-ID list.  A supported AND conjunction may additionally
    // retain that list across its second scalar predicate.  MvccNode
    // subsequently applies timestamp/delete visibility; every unsupported
    // expression keeps the dense path below.
    const bool native_list_mode = use_sparse_representation;
    const bool native_list_eligible = native_list_mode &&
                                      exprs_->size() == 1 &&
                                      cache_segment != nullptr &&
                                      cache_segment->type() == SegmentType::Sealed;
    if (native_list_eligible) {
        // Keep the scalar-index readiness contract identical to the regular
        // expression path below.  A native lookup must not race an
        // asynchronously prefetched BitmapIndex on the first request.
        exprs_->WaitPrefetch();
        const auto native_scalar_start =
            std::chrono::high_resolution_clock::now();
        EvalCtx native_eval_ctx(operator_context_->get_exec_context());
        auto native_ids = exprs_->expr(0)->TryGetNativeValidIds(native_eval_ctx);
        if (native_ids != nullptr) {
            const auto native_scalar_end =
                std::chrono::high_resolution_clock::now();
            const double native_scalar_cost =
                std::chrono::duration<double, std::micro>(native_scalar_end -
                                                           native_scalar_start)
                    .count();
            // Keep native-list producer timing in the same scalar histogram
            // as the Dense FilterBits path so E2E collection can compare
            // predicate-result construction without treating endpoint time
            // as a producer proxy.
            milvus::monitor::internal_core_search_latency_scalar.Observe(
                native_scalar_cost / 1000);
            query_context_->set_valid_id_payload(std::move(native_ids),
                                                  need_process_rows_);
            num_processed_rows_ = need_process_rows_;
            std::vector<VectorPtr> col_res;
            // MvccNode and VectorSearchNode recognize the native payload and
            // never inspect this placeholder. The driver nevertheless
            // requires every non-null RowVector to contain at least one row,
            // so use a one-bit sentinel rather than the former pair of
            // N-bit allocations in the no-delete fast path.
            col_res.push_back(std::make_shared<ColumnVector>(
                TargetBitmap(1, false), TargetBitmap(1, true)));
            return std::make_shared<RowVector>(col_res);
        }
    }

    tracer::AutoSpan span(
        "PhyFilterBitsNode::Execute", tracer::GetRootSpan(), true);
    tracer::AddEvent(fmt::format("input_rows: {}", need_process_rows_));

    exprs_->WaitPrefetch();

    std::chrono::high_resolution_clock::time_point scalar_start =
        std::chrono::high_resolution_clock::now();

    EvalCtx eval_ctx(operator_context_->get_exec_context());

    TargetBitmap bitset;
    TargetBitmap valid_bitset;

    // optimization: if all expressions can be executed at once,
    // execute in a single pass and flip in-place to avoid bitmap copies.
    if (exprs_->CanExecuteAllAtOnce()) {
        tracer::AddEvent("expr_execute_all_at_once");
        exprs_->SetExecuteAllAtOnce();

        exprs_->Eval(0, 1, true, eval_ctx, results_);
        AssertInfo(results_.size() == 1 && results_[0] != nullptr,
                   "PhyFilterBitsNode result size should be size one and not "
                   "be nullptr");
        auto col_vec = std::dynamic_pointer_cast<ColumnVector>(results_[0]);
        AssertInfo(col_vec && col_vec->IsBitmap(),
                   "PhyFilterBitsNode result should be bitmap ColumnVector");

        auto col_vec_size = col_vec->size();
        TargetBitmapView view(col_vec->GetRawData(), col_vec_size);
        TargetBitmapView valid_view(col_vec->GetValidRawData(), col_vec_size);
        ConvertPredicateToFilteredBitset(view, valid_view, col_vec_size);
        num_processed_rows_ = col_vec_size;

        AssertInfo(col_vec_size == need_process_rows_,
                   "bitset size: {}, need_process_rows_: {}",
                   col_vec_size,
                   need_process_rows_);

        if (can_use_cache) {
            ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                         expr_cache_key_};
            ExprResCacheManager::Value v;
            v.result = std::make_shared<TargetBitmap>(view);
            v.valid_result = std::make_shared<TargetBitmap>(valid_view);
            v.active_count = need_process_rows_;
            ExprResCacheManager::Instance().Put(key, v);
        }

        std::vector<VectorPtr> col_res;
        col_res.push_back(std::move(results_[0]));

        std::chrono::high_resolution_clock::time_point scalar_end =
            std::chrono::high_resolution_clock::now();
        double scalar_cost =
            std::chrono::duration<double, std::micro>(scalar_end - scalar_start)
                .count();
        milvus::monitor::internal_core_search_latency_scalar.Observe(
            scalar_cost / 1000);

        return std::make_shared<RowVector>(col_res);
    }

    while (num_processed_rows_ < need_process_rows_) {
        exprs_->Eval(0, 1, true, eval_ctx, results_);

        AssertInfo(results_.size() == 1 && results_[0] != nullptr,
                   "PhyFilterBitsNode result size should be size one and not "
                   "be nullptr");

        if (auto col_vec =
                std::dynamic_pointer_cast<ColumnVector>(results_[0])) {
            if (col_vec->IsBitmap()) {
                auto col_vec_size = col_vec->size();
                TargetBitmapView view(col_vec->GetRawData(), col_vec_size);
                bitset.append(view);
                TargetBitmapView valid_view(col_vec->GetValidRawData(),
                                            col_vec_size);
                valid_bitset.append(valid_view);
                num_processed_rows_ += col_vec_size;
            } else {
                ThrowInfo(UnexpectedError,
                          "PhyFilterBitsNode result should be bitmap");
            }
        } else {
            ThrowInfo(UnexpectedError,
                      "PhyFilterBitsNode result should be ColumnVector");
        }
    }
    TargetBitmapView bitset_view(bitset);
    TargetBitmapView valid_bitset_view(valid_bitset);
    ConvertPredicateToFilteredBitset(
        bitset_view, valid_bitset_view, bitset.size());

    AssertInfo(bitset.size() == need_process_rows_,
               "bitset size: {}, need_process_rows_: {}",
               bitset.size(),
               need_process_rows_);
    Assert(valid_bitset.size() == need_process_rows_);

    // Cache write: clone bitset into ExprResCacheManager — Stage 1 of two-stage
    // search. Must clone before move since Stage 1 still owns the bitset for
    // the ColumnVector return value below.
    if (can_use_cache) {
        ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                     expr_cache_key_};
        ExprResCacheManager::Value v;
        v.result = std::make_shared<TargetBitmap>(bitset.clone());
        v.valid_result = std::make_shared<TargetBitmap>(valid_bitset.clone());
        v.active_count = need_process_rows_;
        ExprResCacheManager::Instance().Put(key, v);
    }

    // num_processed_rows_ = need_process_rows_;
    std::vector<VectorPtr> col_res;
    col_res.push_back(std::make_shared<ColumnVector>(std::move(bitset),
                                                     std::move(valid_bitset)));
    std::chrono::high_resolution_clock::time_point scalar_end =
        std::chrono::high_resolution_clock::now();
    double scalar_cost =
        std::chrono::duration<double, std::micro>(scalar_end - scalar_start)
            .count();
    milvus::monitor::internal_core_search_latency_scalar.Observe(scalar_cost /
                                                                 1000);

    return std::make_shared<RowVector>(col_res);
}

}  // namespace exec
}  // namespace milvus
