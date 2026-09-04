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
#include "common/Common.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "exec/QueryContext.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/ExprCache.h"
#include "exec/expression/ConjunctExpr.h"
#include "exec/operator/AdaptiveFilterSink.h"
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

    // QueryContext is query-owned today, but FilterBits may be re-entered or
    // reused by future plan shapes.  Never let a Dense result inherit a Sparse
    // payload produced by an earlier execution of the same context.
    query_context_->clear_filter_map();

    const auto search_info = query_context_->get_search_info();
    // Retrieve plans have no vector-search params.  They still pass through
    // FilterBitsNode for ordinary scalar predicates, so treat a null params
    // value exactly as the default vector-search mode.
    const auto scan_mode = search_info.search_params_.is_object()
                               ? search_info.search_params_.value(
                                     "bf_filter_scan_mode", std::string{"auto"})
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
    if (use_sparse_representation &&
        !milvus::ENABLE_SPARSE_FILTER_RESULT.load()) {
        ThrowInfo(ConfigInvalid,
                  "adaptive sparse filter result is disabled; set "
                  "queryNode.segcore.enableSparseFilterResult=true");
    }
    const auto configured_sparse_max_cardinality =
        search_info.SparseResultMaxCardinality(
            milvus::SPARSE_FILTER_RESULT_MAX_CARDINALITY.load());
    const auto sparse_max_cardinality = ComputeSparseFilterResultCap(
        need_process_rows_,
        configured_sparse_max_cardinality,
        milvus::SPARSE_FILTER_RESULT_MIN_SEGMENT_ROWS.load(),
        milvus::SPARSE_FILTER_RESULT_MAX_RATIO.load());
    // Phase one deliberately leaves OR on the original Dense path.  This also
    // preserves its expression-cache behavior until Sparse union and SQL 3VL
    // semantics are designed and tested independently.
    const auto root_conjunct =
        exprs_->size() == 1
            ? std::dynamic_pointer_cast<PhyConjunctFilterExpr>(exprs_->expr(0))
            : nullptr;
    const bool phase_one_or =
        root_conjunct != nullptr && !root_conjunct->IsAnd();
    const bool selector_eligible = sparse_max_cardinality > 0;
    const bool adaptive_output_eligible = use_sparse_representation &&
                                          selector_eligible &&
                                          exprs_->size() == 1 && !phase_one_or;
    const std::string cache_signature =
        adaptive_output_eligible ? fmt::format("{}|adaptive_sparse:{}",
                                               expr_cache_key_,
                                               sparse_max_cardinality)
                                 : expr_cache_key_;
    if (use_sparse_representation && phase_one_or) {
        milvus::monitor::internal_core_adaptive_filter_output_or_dense
            .Increment();
    }

    // Cache read: Stage 2 of two-stage search reuses the bitset cached by Stage 1.
    // Cache lives in the process-level ExprResCacheManager keyed by
    // (segment_id, FilterBitsNode signature + dynamic filter context), so
    // cross-query reuse is automatic only when the effective predicate matches.
    auto* cache_segment = query_context_->get_segment();
    // Dense and Adaptive use representation-specific cache signatures.  An
    // Adaptive hit must return its cached Sparse list or threshold-Dense
    // result directly; scanning a cached N-bit Dense result on every request
    // recreates the universe cost and previously produced a 12% latency tail.
    const bool can_use_cache = enable_expr_cache_ && !expr_cache_key_.empty() &&
                               cache_segment != nullptr &&
                               cache_segment->type() == SegmentType::Sealed &&
                               ExprResCacheManager::IsEnabled();
    if (adaptive_output_eligible && !can_use_cache) {
        milvus::monitor::internal_core_adaptive_filter_cache_disabled
            .Increment();
    }
    if (can_use_cache) {
        ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                     cache_signature};
        if (adaptive_output_eligible) {
            ExprResCacheManager::SparseValue sparse_cached;
            sparse_cached.active_count = need_process_rows_;
            if (ExprResCacheManager::Instance().GetSparse(key, sparse_cached)) {
                milvus::monitor::internal_core_adaptive_filter_cache_sparse_hit
                    .Increment();
                num_processed_rows_ = need_process_rows_;
                auto filter_map = std::make_shared<FilterMap>(
                    sparse_cached.filter_map->Clone());
                query_context_->set_filter_map(std::move(filter_map));
                milvus::monitor::internal_core_adaptive_filter_output_sparse
                    .Increment();
                std::vector<VectorPtr> col_res;
                col_res.push_back(std::make_shared<ColumnVector>(
                    TargetBitmap(1, false), TargetBitmap(1, true)));
                return std::make_shared<RowVector>(col_res);
            }
        }
        ExprResCacheManager::Value cached;
        cached.active_count = need_process_rows_;
        if (ExprResCacheManager::Instance().Get(key, cached) &&
            cached.result != nullptr &&
            cached.result->size() == need_process_rows_) {
            if (adaptive_output_eligible) {
                milvus::monitor::internal_core_adaptive_filter_cache_dense_hit
                    .Increment();
            }
            num_processed_rows_ = need_process_rows_;
            std::vector<VectorPtr> col_res;
            if (adaptive_output_eligible) {
                milvus::monitor::internal_core_adaptive_filter_output_dense
                    .Increment();
                auto filter_map = std::make_shared<FilterMap>(
                    FilterMap::FromDense(std::make_shared<TargetBitmap>(
                        cached.result->clone())));
                query_context_->set_filter_map(std::move(filter_map));
                col_res.push_back(std::make_shared<ColumnVector>(
                    TargetBitmap(1, false), TargetBitmap(1, true)));
                return std::make_shared<RowVector>(col_res);
            }
            col_res.push_back(std::make_shared<ColumnVector>(
                cached.result->clone(),
                cached.valid_result ? cached.valid_result->clone()
                                    : TargetBitmap(need_process_rows_, true)));
            return std::make_shared<RowVector>(col_res);
        }
        if (adaptive_output_eligible) {
            milvus::monitor::internal_core_adaptive_filter_cache_miss
                .Increment();
        }
    }

    tracer::AutoSpan span(
        "PhyFilterBitsNode::Execute", tracer::GetRootSpan(), true);
    tracer::AddEvent(fmt::format("input_rows: {}", need_process_rows_));

    // Use one readiness/timing boundary for native Sparse, the adaptive
    // streaming sink, and the Dense baseline.
    exprs_->WaitPrefetch();
    const auto scalar_start = std::chrono::high_resolution_clock::now();
    const auto observe_scalar_cost = [&]() {
        const auto scalar_end = std::chrono::high_resolution_clock::now();
        const double scalar_cost =
            std::chrono::duration<double, std::micro>(scalar_end - scalar_start)
                .count();
        milvus::monitor::internal_core_search_latency_scalar.Observe(
            scalar_cost / 1000);
    };

    // Centralize Adaptive cache/counter/payload ownership. Dense cache state
    // is cloned because the returned ColumnVector takes the original bitmap.
    const auto finalize_adaptive = [&](FilterMap result) {
        AssertInfo(result.size() == static_cast<size_t>(need_process_rows_),
                   "adaptive predicate universe {} does not match filter "
                   "universe {}",
                   result.size(),
                   need_process_rows_);

        std::vector<VectorPtr> col_res;
        if (result.capability() == FilterCapability::EnumerateOnly) {
            AssertInfo(result.size() - result.count() <=
                           static_cast<size_t>(sparse_max_cardinality),
                       "adaptive Sparse result {} exceeds cap {}",
                       result.size() - result.count(),
                       sparse_max_cardinality);
            if (can_use_cache) {
                ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                             cache_signature};
                ExprResCacheManager::SparseValue cached;
                cached.filter_map =
                    std::make_shared<const FilterMap>(result.Clone());
                cached.active_count = result.size();
                ExprResCacheManager::Instance().PutSparse(key, cached);
                milvus::monitor::internal_core_adaptive_filter_cache_sparse_put
                    .Increment();
            }
            milvus::monitor::internal_core_adaptive_filter_output_sparse
                .Increment();
        } else {
            const auto dense = result.DenseData();
            AssertInfo(
                dense != nullptr &&
                    dense->size() == static_cast<size_t>(need_process_rows_),
                "adaptive Dense result size {} does not match {}",
                dense == nullptr ? 0 : dense->size(),
                need_process_rows_);
            if (can_use_cache) {
                ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                             cache_signature};
                ExprResCacheManager::Value cached;
                cached.result = std::make_shared<TargetBitmap>(dense->clone());
                cached.valid_result =
                    std::make_shared<TargetBitmap>(need_process_rows_, true);
                cached.active_count = need_process_rows_;
                ExprResCacheManager::Instance().Put(key, cached);
                milvus::monitor::internal_core_adaptive_filter_cache_dense_put
                    .Increment();
            }
            milvus::monitor::internal_core_adaptive_filter_output_dense
                .Increment();
        }
        query_context_->set_filter_map(
            std::make_shared<FilterMap>(std::move(result)));
        col_res.push_back(std::make_shared<ColumnVector>(
            TargetBitmap(1, false), TargetBitmap(1, true)));
        num_processed_rows_ = need_process_rows_;
        observe_scalar_cost();
        return std::make_shared<RowVector>(col_res);
    };

    // Native accepted-ID path.  BitmapIndex may retain a
    // Roaring posting internally, but the only payload crossing this boundary
    // is a valid-ID list.  A supported AND conjunction may additionally
    // retain that list across its second scalar predicate.  MvccNode
    // subsequently applies timestamp/delete visibility; every unsupported
    // expression keeps the dense path below.
    const bool native_list_eligible =
        adaptive_output_eligible && exprs_->size() == 1 &&
        cache_segment != nullptr &&
        cache_segment->type() == SegmentType::Sealed;
    bool native_dense_fallback = false;
    if (native_list_eligible) {
        EvalCtx native_eval_ctx(operator_context_->get_exec_context());
        auto native_expr = exprs_->expr(0);
        const auto preflight =
            native_expr->PreflightSparseFilter(native_eval_ctx,
                                               /*has_sparse_input=*/false,
                                               sparse_max_cardinality);
        if (preflight == SparseFilterPreflight::Sparse) {
            auto native_result = native_expr->TryApplySparseFilter(
                native_eval_ctx, std::nullopt, sparse_max_cardinality);
            AssertInfo(native_result.has_value(),
                       "Sparse capability preflight succeeded but root "
                       "predicate declined execution");
            AssertInfo(native_result->capability() ==
                               FilterCapability::RandomMembership ||
                           native_result->size() - native_result->count() <=
                               static_cast<size_t>(sparse_max_cardinality),
                       "native Sparse predicate violated result cap {}",
                       sparse_max_cardinality);
            return finalize_adaptive(std::move(*native_result));
        }
        native_dense_fallback = preflight == SparseFilterPreflight::Dense;
    }

    EvalCtx eval_ctx(operator_context_->get_exec_context());

    if (adaptive_output_eligible && !native_dense_fallback) {
        // Feed each ordinary predicate batch directly into the final
        // representation. Sparse success never allocates a complete
        // FilterBits bitmap; T+1 switches once with at most O(T) backfill.
        // Do not call SetExecuteAllAtOnce in this mode.
        AdaptiveFilterSink sink(need_process_rows_, sparse_max_cardinality);
        while (num_processed_rows_ < need_process_rows_) {
            exprs_->Eval(0, 1, true, eval_ctx, results_);
            AssertInfo(results_.size() == 1 && results_[0] != nullptr,
                       "PhyFilterBitsNode result size should be one and not "
                       "be nullptr");
            auto col_vec = std::dynamic_pointer_cast<ColumnVector>(results_[0]);
            AssertInfo(
                col_vec != nullptr && col_vec->IsBitmap(),
                "PhyFilterBitsNode result should be bitmap ColumnVector");
            const auto rows = col_vec->size();
            AssertInfo(
                rows > 0 && num_processed_rows_ + static_cast<int64_t>(rows) <=
                                need_process_rows_,
                "adaptive predicate batch {} exceeds remaining rows {}",
                rows,
                need_process_rows_ - num_processed_rows_);
            TargetBitmapView data(col_vec->GetRawData(), rows);
            TargetBitmapView valid(col_vec->GetValidRawData(), rows);
            sink.ConsumeBatch(data, valid, num_processed_rows_);
            num_processed_rows_ += rows;
        }
        return finalize_adaptive(sink.Finish());
    }

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
                                         cache_signature};
            ExprResCacheManager::Value v;
            v.result = std::make_shared<TargetBitmap>(view);
            v.valid_result = std::make_shared<TargetBitmap>(valid_view);
            v.active_count = need_process_rows_;
            ExprResCacheManager::Instance().Put(key, v);
            if (native_dense_fallback) {
                milvus::monitor::internal_core_adaptive_filter_cache_dense_put
                    .Increment();
            }
        }

        if (native_dense_fallback) {
            milvus::monitor::internal_core_adaptive_filter_output_dense
                .Increment();
        }
        observe_scalar_cost();
        std::vector<VectorPtr> col_res;
        col_res.push_back(std::move(results_[0]));
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
                                     cache_signature};
        ExprResCacheManager::Value v;
        v.result = std::make_shared<TargetBitmap>(bitset.clone());
        v.valid_result = std::make_shared<TargetBitmap>(valid_bitset.clone());
        v.active_count = need_process_rows_;
        ExprResCacheManager::Instance().Put(key, v);
        if (native_dense_fallback) {
            milvus::monitor::internal_core_adaptive_filter_cache_dense_put
                .Increment();
        }
    }

    if (native_dense_fallback) {
        milvus::monitor::internal_core_adaptive_filter_output_dense.Increment();
    }
    observe_scalar_cost();
    std::vector<VectorPtr> col_res;
    col_res.push_back(std::make_shared<ColumnVector>(std::move(bitset),
                                                     std::move(valid_bitset)));
    return std::make_shared<RowVector>(col_res);
}

}  // namespace exec
}  // namespace milvus
