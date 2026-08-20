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
#include <limits>
#include <optional>
#include <ratio>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "common/RegexQuery.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "exec/QueryContext.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/ExprCache.h"
#include "expr/ITypeExpr.h"
#include "fmt/core.h"
#include "log/Log.h"
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

std::optional<double>
GetDoubleValue(const proto::plan::GenericValue& value) {
    switch (value.val_case()) {
        case proto::plan::GenericValue::kInt64Val:
            return static_cast<double>(value.int64_val());
        case proto::plan::GenericValue::kFloatVal:
            return value.float_val();
        default:
            return std::nullopt;
    }
}

std::optional<std::string>
GetStringValue(const proto::plan::GenericValue& value) {
    if (value.val_case() != proto::plan::GenericValue::kStringVal) {
        return std::nullopt;
    }
    return value.string_val();
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
        case proto::plan::OpType::PrefixMatch:
            return CardinalDownpushPredicateOp::StringPrefixMatch;
        case proto::plan::OpType::PostfixMatch:
            return CardinalDownpushPredicateOp::StringPostfixMatch;
        case proto::plan::OpType::InnerMatch:
            return CardinalDownpushPredicateOp::StringInnerMatch;
        case proto::plan::OpType::Match:
            return CardinalDownpushPredicateOp::StringLikeMatch;
        default:
            return std::nullopt;
    }
}

bool
IsStringMatchOp(CardinalDownpushPredicateOp op) {
    return op == CardinalDownpushPredicateOp::StringPrefixMatch ||
           op == CardinalDownpushPredicateOp::StringPostfixMatch ||
           op == CardinalDownpushPredicateOp::StringInnerMatch ||
           op == CardinalDownpushPredicateOp::StringLikeMatch;
}

bool
IsDownpushIntField(DataType data_type) {
    return data_type == DataType::INT8 || data_type == DataType::INT16 ||
           data_type == DataType::INT32 || data_type == DataType::INT64 ||
           data_type == DataType::TIMESTAMPTZ;
}

bool
IsDownpushFloatField(DataType data_type) {
    return data_type == DataType::FLOAT;
}

bool
IsDownpushStringField(DataType data_type) {
    return data_type == DataType::VARCHAR || data_type == DataType::STRING;
}

std::optional<CardinalDownpushPredicateValueType>
GetDownpushValueType(DataType data_type) {
    if (IsDownpushIntField(data_type)) {
        return CardinalDownpushPredicateValueType::Int64;
    }
    if (IsDownpushFloatField(data_type)) {
        return CardinalDownpushPredicateValueType::Float;
    }
    if (IsDownpushStringField(data_type)) {
        return CardinalDownpushPredicateValueType::String;
    }
    return std::nullopt;
}

bool
FillPredicateArg(CardinalDownpushPredicate& predicate,
                 const proto::plan::GenericValue& value,
                 bool second_arg = false) {
    switch (predicate.value_type_) {
        case CardinalDownpushPredicateValueType::Int64: {
            auto arg = GetInt64Value(value);
            if (!arg.has_value()) {
                return false;
            }
            if (second_arg) {
                predicate.arg1_ = arg.value();
            } else {
                predicate.arg0_ = arg.value();
            }
            return true;
        }
        case CardinalDownpushPredicateValueType::Float: {
            auto arg = GetDoubleValue(value);
            if (!arg.has_value()) {
                return false;
            }
            if (second_arg) {
                predicate.double_arg1_ = arg.value();
            } else {
                predicate.double_arg0_ = arg.value();
            }
            return true;
        }
        case CardinalDownpushPredicateValueType::String: {
            auto arg = GetStringValue(value);
            if (!arg.has_value()) {
                return false;
            }
            if (second_arg) {
                predicate.string_arg1_ = arg.value();
            } else {
                predicate.string_arg0_ = arg.value();
            }
            return true;
        }
    }
    return false;
}

bool
FillPredicateTermArgs(CardinalDownpushPredicate& predicate,
                      const std::vector<proto::plan::GenericValue>& values) {
    if (values.empty()) {
        return false;
    }

    switch (predicate.value_type_) {
        case CardinalDownpushPredicateValueType::Int64:
            predicate.int64_terms_.reserve(values.size());
            for (const auto& value : values) {
                auto arg = GetInt64Value(value);
                if (!arg.has_value()) {
                    return false;
                }
                predicate.int64_terms_.push_back(arg.value());
            }
            std::sort(predicate.int64_terms_.begin(),
                      predicate.int64_terms_.end());
            predicate.int64_terms_.erase(
                std::unique(predicate.int64_terms_.begin(),
                            predicate.int64_terms_.end()),
                predicate.int64_terms_.end());
            return true;
        case CardinalDownpushPredicateValueType::Float:
            predicate.double_terms_.reserve(values.size());
            for (const auto& value : values) {
                auto arg = GetDoubleValue(value);
                if (!arg.has_value()) {
                    return false;
                }
                predicate.double_terms_.push_back(arg.value());
            }
            std::sort(predicate.double_terms_.begin(),
                      predicate.double_terms_.end());
            predicate.double_terms_.erase(
                std::unique(predicate.double_terms_.begin(),
                            predicate.double_terms_.end()),
                predicate.double_terms_.end());
            return true;
        case CardinalDownpushPredicateValueType::String:
            predicate.string_terms_.reserve(values.size());
            for (const auto& value : values) {
                auto arg = GetStringValue(value);
                if (!arg.has_value()) {
                    return false;
                }
                predicate.string_terms_.push_back(arg.value());
            }
            std::sort(predicate.string_terms_.begin(),
                      predicate.string_terms_.end());
            predicate.string_terms_.erase(
                std::unique(predicate.string_terms_.begin(),
                            predicate.string_terms_.end()),
                predicate.string_terms_.end());
            return true;
    }
    return false;
}

enum class DownpushLikeTokenType : uint8_t {
    Literal = 0,
    AnyOne = 1,
    AnyMany = 2,
};

bool
CompileLikePattern(CardinalDownpushPredicate& predicate) {
    const auto& pattern = predicate.string_arg0_;
    if (pattern.size() > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    auto add_token = [&](DownpushLikeTokenType type,
                         size_t offset,
                         size_t size) {
        predicate.like_token_offsets_.push_back(static_cast<uint32_t>(offset));
        predicate.like_token_sizes_.push_back(static_cast<uint32_t>(size));
        predicate.like_token_types_.push_back(static_cast<uint8_t>(type));
    };

    for (size_t i = 0; i < pattern.size();) {
        const auto c = pattern[i];
        if (c == '\\') {
            ++i;
            if (i == pattern.size()) {
                return false;
            }
            const auto char_len = Utf8ValidatedCharByteLen(pattern.data() + i,
                                                           pattern.size() - i);
            add_token(DownpushLikeTokenType::Literal, i, char_len);
            i += char_len;
            continue;
        }
        if (c == '%') {
            if (predicate.like_token_types_.empty() ||
                predicate.like_token_types_.back() !=
                    static_cast<uint8_t>(DownpushLikeTokenType::AnyMany)) {
                add_token(DownpushLikeTokenType::AnyMany, i, 1);
            }
            ++i;
            continue;
        }
        if (c == '_') {
            add_token(DownpushLikeTokenType::AnyOne, i, 1);
            ++i;
            continue;
        }
        const auto char_len =
            Utf8ValidatedCharByteLen(pattern.data() + i, pattern.size() - i);
        add_token(DownpushLikeTokenType::Literal, i, char_len);
        i += char_len;
    }
    return true;
}

bool
TryFoldInt64TermsToRange(CardinalDownpushPredicate& predicate) {
    if (predicate.value_type_ != CardinalDownpushPredicateValueType::Int64 ||
        predicate.int64_terms_.empty()) {
        return false;
    }

    const auto first = predicate.int64_terms_.front();
    const auto last = predicate.int64_terms_.back();
    if (last < first) {
        return false;
    }

    const auto expected_size =
        static_cast<__int128>(last) - static_cast<__int128>(first) + 1;
    if (expected_size <= 0 ||
        expected_size != static_cast<__int128>(predicate.int64_terms_.size())) {
        return false;
    }

    predicate.arg0_ = first;
    predicate.arg1_ = last;
    predicate.lower_inclusive_ = true;
    predicate.upper_inclusive_ = true;
    predicate.int64_terms_.clear();
    predicate.op_ = CardinalDownpushPredicateOp::ScalarRange;
    return true;
}

std::optional<CardinalDownpushPredicateOp>
ToDownpushArithLessThanOp(proto::plan::ArithOpType arith_op) {
    switch (arith_op) {
        case proto::plan::ArithOpType::Add:
            return CardinalDownpushPredicateOp::ScalarAddLessThan;
        case proto::plan::ArithOpType::Sub:
            return CardinalDownpushPredicateOp::ScalarSubLessThan;
        case proto::plan::ArithOpType::Mul:
            return CardinalDownpushPredicateOp::ScalarMulLessThan;
        case proto::plan::ArithOpType::Div:
            return CardinalDownpushPredicateOp::ScalarDivLessThan;
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

    auto try_field = [&](const expr::ColumnInfo& column)
        -> std::optional<CardinalDownpushPredicate> {
        if (column.element_level_ || !column.nested_path_.empty()) {
            return std::nullopt;
        }
        auto field_id = column.field_id_;
        auto value_type = GetDownpushValueType(column.data_type_);
        if (!value_type.has_value()) {
            return std::nullopt;
        }
        if (column.nullable_ &&
            value_type.value() != CardinalDownpushPredicateValueType::String) {
            return std::nullopt;
        }
        if (!segment->HasFieldData(field_id) && !segment->HasIndex(field_id)) {
            return std::nullopt;
        }
        CardinalDownpushPredicate predicate;
        predicate.field_id_ = field_id;
        predicate.field_data_type_ = column.data_type_;
        predicate.value_type_ = value_type.value();
        return predicate;
    };

    if (auto unary =
            std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(
                filter)) {
        auto predicate = try_field(unary->column_);
        auto op = ToDownpushRangeOp(unary->op_type_);
        if (!predicate.has_value() || !op.has_value() ||
            !FillPredicateArg(predicate.value(), unary->val_)) {
            return std::nullopt;
        }
        if (IsStringMatchOp(op.value()) &&
            predicate->value_type_ !=
                CardinalDownpushPredicateValueType::String) {
            return std::nullopt;
        }
        predicate->op_ = op.value();
        if (predicate->op_ == CardinalDownpushPredicateOp::StringLikeMatch &&
            !CompileLikePattern(predicate.value())) {
            return std::nullopt;
        }
        return predicate;
    }

    if (auto binary =
            std::dynamic_pointer_cast<const expr::BinaryRangeFilterExpr>(
                filter)) {
        auto predicate = try_field(binary->column_);
        if (!predicate.has_value() ||
            !FillPredicateArg(predicate.value(), binary->lower_val_) ||
            !FillPredicateArg(predicate.value(), binary->upper_val_, true)) {
            return std::nullopt;
        }
        predicate->op_ = CardinalDownpushPredicateOp::ScalarRange;
        predicate->lower_inclusive_ = binary->lower_inclusive_;
        predicate->upper_inclusive_ = binary->upper_inclusive_;
        return predicate;
    }

    if (auto term =
            std::dynamic_pointer_cast<const expr::TermFilterExpr>(filter)) {
        auto predicate = try_field(term->column_);
        if (!predicate.has_value() ||
            !FillPredicateTermArgs(predicate.value(), term->vals_)) {
            return std::nullopt;
        }
        if (TryFoldInt64TermsToRange(predicate.value())) {
            return predicate;
        }
        predicate->op_ = CardinalDownpushPredicateOp::ScalarTerm;
        return predicate;
    }

    if (auto arith =
            std::dynamic_pointer_cast<const expr::BinaryArithOpEvalRangeExpr>(
                filter)) {
        auto predicate = try_field(arith->column_);
        if (!predicate.has_value() ||
            arith->op_type_ != proto::plan::OpType::LessThan) {
            return std::nullopt;
        }

        if (arith->arith_op_type_ == proto::plan::ArithOpType::Mod) {
            auto modulus = GetInt64Value(arith->right_operand_);
            auto threshold = GetInt64Value(arith->value_);
            if (predicate->value_type_ !=
                    CardinalDownpushPredicateValueType::Int64 ||
                !modulus.has_value() || !threshold.has_value() ||
                modulus.value() <= 0 || threshold.value() < 0 ||
                threshold.value() > modulus.value()) {
                return std::nullopt;
            }
            predicate->op_ = CardinalDownpushPredicateOp::Int64ModLessThan;
            predicate->arg0_ = modulus.value();
            predicate->arg1_ = threshold.value();
            return predicate;
        }

        auto arith_less_than_op =
            ToDownpushArithLessThanOp(arith->arith_op_type_);
        if (arith_less_than_op.has_value()) {
            if (predicate->value_type_ ==
                CardinalDownpushPredicateValueType::Int64) {
                auto right_operand = GetInt64Value(arith->right_operand_);
                auto threshold = GetInt64Value(arith->value_);
                if (!right_operand.has_value() || !threshold.has_value() ||
                    (arith->arith_op_type_ == proto::plan::ArithOpType::Div &&
                     right_operand.value() == 0)) {
                    return std::nullopt;
                }
                predicate->op_ = arith_less_than_op.value();
                predicate->arg0_ = right_operand.value();
                predicate->arg1_ = threshold.value();
                return predicate;
            }
            if (predicate->value_type_ ==
                CardinalDownpushPredicateValueType::Float) {
                auto right_operand = GetDoubleValue(arith->right_operand_);
                auto threshold = GetDoubleValue(arith->value_);
                if (!right_operand.has_value() || !threshold.has_value() ||
                    (arith->arith_op_type_ == proto::plan::ArithOpType::Div &&
                     right_operand.value() == 0.0)) {
                    return std::nullopt;
                }
                predicate->op_ = arith_less_than_op.value();
                predicate->double_arg0_ = right_operand.value();
                predicate->double_arg1_ = threshold.value();
                return predicate;
            }
        }
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
        // downpush hint (ann filter fusing): attempt to fuse the scalar
        // predicate into the vector index. The hint is advisory — if any
        // precondition is unmet we silently fall back to the normal ExprSet
        // path below and never break correctness.
        TryEnableCardinalDownpush(*filter, exec_context);
    }

    enable_expr_cache_ = query_context_->get_enable_expr_cache();
    if (enable_expr_cache_ && !cardinal_downpush_enabled_) {
        expr_cache_key_ = BuildExprCacheKey(*filter, query_context_);
    }
}

void
PhyFilterBitsNode::TryEnableCardinalDownpush(
    const plan::FilterBitsNode& filter,
    ExecContext* exec_context) {
    const auto& search_info = query_context_->get_search_info();

    auto fallback = [](const char* reason) {
        milvus::monitor::internal_core_downpush_fallback_count_family.Add(
            {{"reason", reason}})
            .Increment();
    };

    // Fusion is not implemented for element-level (array-of-vectors) search.
    if (search_info.element_level()) {
        LOG_DEBUG("downpush fallback: element-level vector search unsupported");
        fallback("element_level");
        return;
    }

    // The vector index must support predicate fusion. Backend-agnostic
    // capability query so this node never reads the raw index type.
    if (!query_context_->get_segment()->SupportsDownpush(
            search_info.field_id_)) {
        LOG_DEBUG("downpush fallback: vector index does not support fusion");
        fallback("unsupported_index");
        return;
    }

    // entity TTL: ExprSet compilation injects the TTL predicate into exprs_.
    // Because only the *user* predicate is deferred into the vector index, the
    // TTL predicate is compiled separately and kept as a normal logical-space
    // bitset (evaluated in GetOutput).
    auto ttl_expr = CreateTTLFieldFilterExpression(query_context_);
    if (ttl_expr != nullptr) {
        // ExprSet normally injects entity TTL into its first source. This
        // source is already the TTL expression itself, so disable injection
        // here to avoid compiling TTL AND TTL.
        ttl_exprs_ = std::make_unique<ExprSet>(
            std::vector<expr::TypedExprPtr>{ttl_expr}, exec_context, false);
    }

    auto predicate =
        TryCompileCardinalDownpushPredicate(filter.filter(), query_context_);
    if (!predicate.has_value()) {
        LOG_DEBUG(
            "downpush fallback: unsupported predicate shape (only sealed "
            "scalar int/float/varchar range/term/match/arith/mod are fused)");
        fallback("unsupported_predicate");
        return;
    }

    // A LIKE predicate requires raw varchar values.  A sealed segment backed
    // by STL_SORT can expose dictionary IDs for equality, inequality and term
    // predicates while not exposing the raw string chunk view needed by LIKE.
    // Do not defer that availability check until VectorSearchNode: FilterBits
    // has already skipped materializing the normal bitmap by then, so failure
    // there would turn an optional hint into a request error.  Phase 1 must
    // always preserve correctness by falling back to the ordinary filter.
    if (IsStringMatchOp(predicate->op_)) {
        LOG_DEBUG(
            "downpush fallback: varchar match needs a raw value source that "
            "is unavailable for some sealed scalar-index layouts");
        fallback("unsupported_predicate");
        return;
    }

    std::vector<expr::TypedExprPtr> filters{filter.filter()};
    ExprSet sample_exprs(filters, exec_context);
    auto estimated_filtered_out_count = EstimateFilteredOutCountBySample(
        query_context_, &sample_exprs, exec_context);
    if (!estimated_filtered_out_count.has_value()) {
        LOG_DEBUG("downpush fallback: failed to estimate filter ratio");
        fallback("estimate_failed");
        return;
    }

    auto ratio = need_process_rows_ > 0
                     ? static_cast<double>(estimated_filtered_out_count.value()) /
                           static_cast<double>(need_process_rows_)
                     : 0.0;
    if (ratio >= kDownpushFallbackFilterOutRatio) {
        LOG_DEBUG("downpush fallback: filter-out ratio {} >= threshold {}",
                  ratio,
                  kDownpushFallbackFilterOutRatio);
        fallback("ratio_threshold");
        return;
    }

    predicate->estimated_filtered_out_count_ =
        need_process_rows_ > 0
            ? std::max<int64_t>(1, estimated_filtered_out_count.value())
            : 0;
    cardinal_downpush_predicate_ = predicate;
    cardinal_downpush_enabled_ = true;
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
        if (ttl_exprs_ != nullptr) {
            // entity TTL stays a normal logical-space bitset (`1` = exclude),
            // while the user predicate is deferred into the vector index.
            EvalCtx ttl_eval_ctx(operator_context_->get_exec_context());
            std::vector<VectorPtr> ttl_results;
            ttl_exprs_->Eval(0, 1, true, ttl_eval_ctx, ttl_results);
            AssertInfo(ttl_results.size() == 1 && ttl_results[0] != nullptr,
                       "TTL filter should produce a single bitmap result");
            auto ttl_col = std::dynamic_pointer_cast<ColumnVector>(ttl_results[0]);
            AssertInfo(ttl_col && ttl_col->IsBitmap(),
                       "TTL filter result should be a bitmap ColumnVector");
            AssertInfo(static_cast<int64_t>(ttl_col->size()) ==
                           need_process_rows_,
                       "TTL filter result size {} != need_process_rows_ {}",
                       ttl_col->size(),
                       need_process_rows_);
            // Eval yields `1` = keep (TTL not expired); flip to `1` = exclude
            // to match the exclude-bitset convention consumed downstream.
            TargetBitmapView ttl_view(ttl_col->GetRawData(), ttl_col->size());
            ttl_view.flip();
            col_res.push_back(std::move(ttl_results[0]));
        } else {
            col_res.push_back(std::make_shared<ColumnVector>(
                TargetBitmap(need_process_rows_, false),
                TargetBitmap(need_process_rows_, true)));
        }
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
