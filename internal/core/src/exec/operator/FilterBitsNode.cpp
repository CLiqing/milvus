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
#include <optional>
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

constexpr int64_t kDownpushEstimatorSampleSize = 256;
constexpr double kDownpushFallbackFilterOutRatio = 0.90;

std::string
BuildExprCacheKey(const plan::FilterBitsNode& filter,
                  QueryContext* query_context) {
    auto key = filter.ToString();
    auto* segment =
        query_context != nullptr ? query_context->get_segment() : nullptr;
    if (segment != nullptr &&
        segment->get_schema().get_ttl_field_id().has_value()) {
        key += fmt::format("|entity_ttl_physical_time_us:{}",
                           query_context->get_entity_ttl_physical_time_us());
    }
    return key;
}

uint64_t
SplitMix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

OffsetVector
BuildSampleOffsets(int64_t active_count,
                   int64_t sample_size,
                   int64_t segment_id) {
    auto real_sample_size = std::min(active_count, sample_size);
    OffsetVector offsets;
    offsets.reserve(real_sample_size);
    for (int64_t i = 0; i < real_sample_size; ++i) {
        auto hash =
            SplitMix64(static_cast<uint64_t>(segment_id) ^
                       (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL));
        offsets.push_back(static_cast<int32_t>(hash % active_count));
    }
    return offsets;
}

std::optional<int64_t>
GetInt64Value(const proto::plan::GenericValue& value) {
    if (value.val_case() != proto::plan::GenericValue::kInt64Val) {
        return std::nullopt;
    }
    return value.int64_val();
}

std::optional<CardinalDownpushPredicateOp>
ToDownpushRangeOp(proto::plan::OpType op_type) {
    switch (op_type) {
        case proto::plan::OpType::GreaterEqual:
            return CardinalDownpushPredicateOp::Int64GreaterEqual;
        case proto::plan::OpType::GreaterThan:
            return CardinalDownpushPredicateOp::Int64GreaterThan;
        case proto::plan::OpType::LessEqual:
            return CardinalDownpushPredicateOp::Int64LessEqual;
        case proto::plan::OpType::LessThan:
            return CardinalDownpushPredicateOp::Int64LessThan;
        case proto::plan::OpType::Equal:
            return CardinalDownpushPredicateOp::Int64Equal;
        case proto::plan::OpType::NotEqual:
            return CardinalDownpushPredicateOp::Int64NotEqual;
        default:
            return std::nullopt;
    }
}

std::optional<CardinalDownpushPredicate>
TryCompileCardinalDownpushPredicate(const expr::TypedExprPtr& filter,
                                    QueryContext* query_context) {
    if (query_context == nullptr || filter == nullptr) {
        return std::nullopt;
    }
    auto* segment = query_context->get_segment();
    if (segment == nullptr || segment->type() != SegmentType::Sealed) {
        return std::nullopt;
    }

    auto try_field =
        [&](const expr::ColumnInfo& column) -> std::optional<FieldId> {
        if (column.data_type_ != DataType::INT64 || column.element_level_ ||
            column.nullable_) {
            return std::nullopt;
        }
        auto field_id = column.field_id_;
        if (!segment->HasFieldData(field_id) && !segment->HasIndex(field_id)) {
            return std::nullopt;
        }
        return field_id;
    };

    if (auto unary =
            std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(
                filter)) {
        auto field_id = try_field(unary->column_);
        auto value = GetInt64Value(unary->val_);
        auto op = ToDownpushRangeOp(unary->op_type_);
        if (!field_id.has_value() || !value.has_value() || !op.has_value()) {
            return std::nullopt;
        }
        CardinalDownpushPredicate predicate;
        predicate.field_id_ = field_id.value();
        predicate.op_ = op.value();
        predicate.arg0_ = value.value();
        return predicate;
    }

    if (auto arith =
            std::dynamic_pointer_cast<const expr::BinaryArithOpEvalRangeExpr>(
                filter)) {
        auto field_id = try_field(arith->column_);
        auto modulus = GetInt64Value(arith->right_operand_);
        auto threshold = GetInt64Value(arith->value_);
        if (!field_id.has_value() || !modulus.has_value() ||
            !threshold.has_value() || modulus.value() <= 0 ||
            threshold.value() < 0 || threshold.value() > modulus.value() ||
            arith->arith_op_type_ != proto::plan::ArithOpType::Mod ||
            arith->op_type_ != proto::plan::OpType::LessThan) {
            return std::nullopt;
        }
        CardinalDownpushPredicate predicate;
        predicate.field_id_ = field_id.value();
        predicate.op_ = CardinalDownpushPredicateOp::Int64ModLessThan;
        predicate.arg0_ = modulus.value();
        predicate.arg1_ = threshold.value();
        return predicate;
    }

    return std::nullopt;
}

std::optional<int64_t>
EstimateFilteredOutCountBySample(QueryContext* query_context,
                                 ExprSet* exprs,
                                 ExecContext* exec_context) {
    if (query_context == nullptr || exprs == nullptr ||
        exec_context == nullptr) {
        return std::nullopt;
    }
    auto active_count = query_context->get_active_count();
    if (active_count <= 0) {
        return 0;
    }
    auto* segment = query_context->get_segment();
    auto segment_id = segment != nullptr ? segment->get_segment_id() : 0;
    auto offsets = BuildSampleOffsets(
        active_count, kDownpushEstimatorSampleSize, segment_id);
    if (offsets.empty()) {
        return 0;
    }

    EvalCtx eval_ctx(exec_context, &offsets);
    std::vector<VectorPtr> results;
    exprs->Eval(0, 1, true, eval_ctx, results);
    if (results.size() != 1 || results[0] == nullptr) {
        return std::nullopt;
    }
    auto col_vec = std::dynamic_pointer_cast<ColumnVector>(results[0]);
    if (!col_vec || !col_vec->IsBitmap()) {
        return std::nullopt;
    }
    TargetBitmapView passed(col_vec->GetRawData(), col_vec->size());
    auto passed_count = static_cast<int64_t>(passed.count());
    auto filtered_sample_count =
        static_cast<int64_t>(offsets.size()) - passed_count;
    auto estimate =
        static_cast<int64_t>((static_cast<double>(filtered_sample_count) /
                              static_cast<double>(offsets.size())) *
                             static_cast<double>(active_count));
    return std::clamp<int64_t>(estimate, 0, active_count);
}

}  // namespace

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
    exprs_ = std::make_unique<ExprSet>(filters, exec_context);
    need_process_rows_ = query_context_->get_active_count();
    num_processed_rows_ = 0;

    if (query_context_->get_search_info().cardinal_downpush_execution) {
        auto predicate = TryCompileCardinalDownpushPredicate(filter->filter(),
                                                             query_context_);
        std::optional<int64_t> estimated_filtered_out_count;
        if (predicate.has_value()) {
            ExprSet sample_exprs(filters, exec_context);
            estimated_filtered_out_count = EstimateFilteredOutCountBySample(
                query_context_, &sample_exprs, exec_context);
        }
        if (estimated_filtered_out_count.has_value()) {
            auto ratio = need_process_rows_ > 0
                             ? static_cast<double>(
                                   estimated_filtered_out_count.value()) /
                                   static_cast<double>(need_process_rows_)
                             : 0.0;
            if (ratio < kDownpushFallbackFilterOutRatio) {
                predicate->estimated_filtered_out_count_ =
                    need_process_rows_ > 0
                        ? std::max<int64_t>(
                              1, estimated_filtered_out_count.value())
                        : 0;
                cardinal_downpush_predicate_ = predicate;
                cardinal_downpush_enabled_ = true;
            }
        }
    }

    enable_expr_cache_ = query_context_->get_enable_expr_cache();
    if (enable_expr_cache_ && !cardinal_downpush_enabled_) {
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

    // Cache read: Stage 2 of two-stage search reuses the bitset cached by Stage 1.
    // Cache lives in the process-level ExprResCacheManager keyed by
    // (segment_id, FilterBitsNode signature + dynamic filter context), so
    // cross-query reuse is automatic only when the effective predicate matches.
    auto* cache_segment = query_context_->get_segment();
    const bool can_use_cache = enable_expr_cache_ && !expr_cache_key_.empty() &&
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

    tracer::AutoSpan span(
        "PhyFilterBitsNode::Execute", tracer::GetRootSpan(), true);
    tracer::AddEvent(fmt::format("input_rows: {}", need_process_rows_));

    if (cardinal_downpush_enabled_) {
        query_context_->set_cardinal_downpush_predicate(
            cardinal_downpush_predicate_.value());
        num_processed_rows_ = need_process_rows_;
        std::vector<VectorPtr> col_res;
        col_res.push_back(std::make_shared<ColumnVector>(
            TargetBitmap(need_process_rows_, false),
            TargetBitmap(need_process_rows_, true)));
        return std::make_shared<RowVector>(col_res);
    }

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
        // flip in-place on the result bitmap, no extra copy
        TargetBitmapView view(col_vec->GetRawData(), col_vec_size);
        view.flip();
        num_processed_rows_ = col_vec_size;

        AssertInfo(col_vec_size == need_process_rows_,
                   "bitset size: {}, need_process_rows_: {}",
                   col_vec_size,
                   need_process_rows_);

        if (can_use_cache) {
            TargetBitmapView valid_view(col_vec->GetValidRawData(),
                                        col_vec_size);
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
                ThrowInfo(ExprInvalid,
                          "PhyFilterBitsNode result should be bitmap");
            }
        } else {
            ThrowInfo(ExprInvalid,
                      "PhyFilterBitsNode result should be ColumnVector");
        }
    }
    bitset.flip();

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
