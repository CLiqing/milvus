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
#include <cmath>
#include <cstdint>
#include <ratio>
#include <string>
#include <type_traits>
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

constexpr const char* kCardinalExprDownpushParam = "cardinal_expr_downpush";
constexpr const char* kLegacyHnswExprDownpushParam = "hnsw_expr_downpush";
constexpr const char* kCardinalExprModPParam = "cardinal_expr_mod_p";
constexpr const char* kCardinalExprModTParam = "cardinal_expr_mod_t";
constexpr const char* kCardinalExprThresholdParam = "cardinal_expr_threshold";
constexpr const char* kCardinalExprFilteredOutCountParam =
    "cardinal_expr_filtered_out_count";
constexpr const char* kCardinalExprFilterRatioParam =
    "cardinal_expr_filter_ratio";

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

template <typename T>
std::optional<T>
GetJsonNumber(const knowhere::Json& params, const char* key) {
    if (!params.contains(key)) {
        return std::nullopt;
    }
    const auto& value = params[key];
    try {
        if (value.is_number()) {
            return value.get<T>();
        }
        if (value.is_string()) {
            if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(std::stoll(value.get<std::string>()));
            } else {
                return static_cast<T>(std::stod(value.get<std::string>()));
            }
        }
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

bool
GetJsonBool(const knowhere::Json& params, const char* key) {
    if (!params.contains(key)) {
        return false;
    }
    const auto& value = params[key];
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_string()) {
        return value.get<std::string>() == "true" ||
               value.get<std::string>() == "1";
    }
    return false;
}

std::optional<int64_t>
GetInt64Value(const proto::plan::GenericValue& value) {
    if (value.val_case() != proto::plan::GenericValue::kInt64Val) {
        return std::nullopt;
    }
    return value.int64_val();
}

std::optional<CardinalExprDownpushInfo>
TryBuildCardinalExprDownpushInfo(const expr::TypedExprPtr& filter,
                                 QueryContext* query_context,
                                 int64_t active_count) {
    if (query_context == nullptr) {
        return std::nullopt;
    }
    const auto& search_params = query_context->get_search_info().search_params_;
    const auto enabled =
        GetJsonBool(search_params, kCardinalExprDownpushParam) ||
        GetJsonBool(search_params, kLegacyHnswExprDownpushParam);
    if (!enabled) {
        return std::nullopt;
    }

    auto* segment = query_context->get_segment();
    if (segment == nullptr || segment->type() != SegmentType::Sealed) {
        ThrowInfo(UnexpectedError,
                  "cardinal expr downpush requires sealed segment");
    }

    auto filtered_out_count = GetJsonNumber<int64_t>(
        search_params, kCardinalExprFilteredOutCountParam);
    auto filter_ratio =
        GetJsonNumber<double>(search_params, kCardinalExprFilterRatioParam);
    if (!filtered_out_count.has_value() && !filter_ratio.has_value()) {
        ThrowInfo(UnexpectedError,
                  "cardinal expr downpush requires {} or {}",
                  kCardinalExprFilteredOutCountParam,
                  kCardinalExprFilterRatioParam);
    }

    auto set_filtered_count = [&](CardinalExprDownpushInfo& info) {
        if (filtered_out_count.has_value()) {
            info.filtered_out_count_ = std::clamp<int64_t>(
                filtered_out_count.value(), 0, active_count);
        } else {
            auto ratio = std::clamp<double>(filter_ratio.value(), 0.0, 1.0);
            info.filtered_out_count_ = std::clamp<int64_t>(
                std::llround(active_count * ratio), 0, active_count);
        }
    };

    auto unary_expr =
        std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(filter);
    if (unary_expr != nullptr) {
        if (unary_expr->column_.data_type_ != DataType::INT64 ||
            unary_expr->column_.element_level_) {
            ThrowInfo(UnexpectedError,
                      "cardinal expr downpush only supports non-element INT64 "
                      "range filter");
        }
        auto& field_meta = segment->get_schema()[unary_expr->column_.field_id_];
        if (field_meta.get_name().get() != "id") {
            ThrowInfo(UnexpectedError,
                      "cardinal expr downpush only supports id >= threshold, "
                      "got field {}",
                      field_meta.get_name().get());
        }
        if (unary_expr->op_type_ != proto::plan::OpType::GreaterEqual) {
            ThrowInfo(UnexpectedError,
                      "cardinal expr downpush only supports id >= threshold");
        }
        auto expr_threshold = GetInt64Value(unary_expr->val_);
        if (!expr_threshold.has_value()) {
            ThrowInfo(UnexpectedError,
                      "cardinal expr downpush id >= threshold requires INT64 "
                      "threshold");
        }
        auto threshold =
            GetJsonNumber<int64_t>(search_params, kCardinalExprThresholdParam)
                .value_or(expr_threshold.value());
        if (threshold != expr_threshold.value()) {
            ThrowInfo(UnexpectedError,
                      "cardinal expr downpush threshold param {} mismatches "
                      "expression value {}",
                      threshold,
                      expr_threshold.value());
        }

        CardinalExprDownpushInfo info;
        info.field_id_ = unary_expr->column_.field_id_;
        info.predicate_ = CardinalExprDownpushPredicate::Int64GreaterEqual;
        info.threshold_ = threshold;
        set_filtered_count(info);
        return info;
    }

    auto arith_expr =
        std::dynamic_pointer_cast<const expr::BinaryArithOpEvalRangeExpr>(
            filter);
    if (arith_expr == nullptr) {
        ThrowInfo(UnexpectedError,
                  "cardinal expr downpush only supports id >= threshold or "
                  "pk % P < T demo expression");
    }
    auto primary_field_id = segment->get_schema().get_primary_field_id();
    if (!primary_field_id.has_value() ||
        arith_expr->column_.field_id_ != primary_field_id.value() ||
        arith_expr->column_.data_type_ != DataType::INT64 ||
        arith_expr->column_.element_level_) {
        ThrowInfo(UnexpectedError,
                  "cardinal expr downpush mod demo requires INT64 primary "
                  "field");
    }
    if (arith_expr->arith_op_type_ != proto::plan::ArithOpType::Mod ||
        arith_expr->op_type_ != proto::plan::OpType::LessThan) {
        ThrowInfo(UnexpectedError,
                  "cardinal expr downpush mod demo only supports pk % P < T");
    }

    auto expr_modulus = GetInt64Value(arith_expr->right_operand_);
    auto expr_threshold = GetInt64Value(arith_expr->value_);
    auto modulus = GetJsonNumber<int64_t>(search_params, kCardinalExprModPParam)
                       .value_or(expr_modulus.value_or(0));
    auto threshold =
        GetJsonNumber<int64_t>(search_params, kCardinalExprModTParam)
            .value_or(expr_threshold.value_or(0));
    if (expr_modulus.has_value() && expr_modulus.value() != modulus) {
        ThrowInfo(UnexpectedError,
                  "cardinal expr downpush modulus param {} mismatches "
                  "expression value {}",
                  modulus,
                  expr_modulus.value());
    }
    if (expr_threshold.has_value() && expr_threshold.value() != threshold) {
        ThrowInfo(UnexpectedError,
                  "cardinal expr downpush mod threshold param {} mismatches "
                  "expression value {}",
                  threshold,
                  expr_threshold.value());
    }
    if (modulus <= 0 || threshold < 0 || threshold > modulus) {
        ThrowInfo(UnexpectedError,
                  "invalid cardinal expr downpush mod params: modulus={}, "
                  "threshold={}",
                  modulus,
                  threshold);
    }

    CardinalExprDownpushInfo info;
    info.field_id_ = primary_field_id.value();
    info.predicate_ = CardinalExprDownpushPredicate::ModLessThan;
    info.modulus_ = modulus;
    info.threshold_ = threshold;
    set_filtered_count(info);
    return info;
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

    auto downpush_info = TryBuildCardinalExprDownpushInfo(
        filter->filter(), query_context_, need_process_rows_);
    if (downpush_info.has_value()) {
        cardinal_expr_downpush_enabled_ = true;
        cardinal_expr_downpush_info_ = downpush_info.value();
    }

    enable_expr_cache_ = query_context_->get_enable_expr_cache();
    if (enable_expr_cache_ && !cardinal_expr_downpush_enabled_) {
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

    if (cardinal_expr_downpush_enabled_) {
        query_context_->set_cardinal_expr_downpush_info(
            cardinal_expr_downpush_info_);
        num_processed_rows_ = need_process_rows_;
        std::vector<VectorPtr> col_res;
        col_res.push_back(std::make_shared<ColumnVector>(
            TargetBitmap(need_process_rows_, false),
            TargetBitmap(need_process_rows_, true)));
        return std::make_shared<RowVector>(col_res);
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
