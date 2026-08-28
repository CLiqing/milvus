// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/DownpushPredicateProvider.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "exec/expression/CandidateEvaluator.h"

namespace milvus::exec {
namespace {

enum class Int64EvaluatorOp : uint8_t {
    GreaterEqual,
    ModLessThan,
    GreaterThan,
    LessEqual,
    LessThan,
    Equal,
    NotEqual,
    Range,
    Term,
    AddLessThan,
    SubLessThan,
    MulLessThan,
    DivLessThan,
};

struct Int64EvaluatorState {
    Int64CandidateSourceView source;
    Int64EvaluatorOp op = Int64EvaluatorOp::Equal;
    int64_t arg0 = 0;
    int64_t arg1 = 0;
    bool lower_inclusive = true;
    bool upper_inclusive = true;
    std::vector<int64_t> terms;
};

struct FloatEvaluatorState {
    FloatCandidateSourceView source;
    Int64EvaluatorOp op = Int64EvaluatorOp::Equal;
    float arg0 = 0.0F;
    float arg1 = 0.0F;
    bool lower_inclusive = true;
    bool upper_inclusive = true;
    std::vector<float> terms;
};

const int64_t*
ResolveInt64CandidateValue(const Int64CandidateSourceView& source,
                           int64_t row_id) noexcept {
    if (row_id < 0 || static_cast<size_t>(row_id) >= source.row_count) {
        return nullptr;
    }
    if (source.row_values != nullptr) {
        return source.row_values + row_id;
    }
    if (source.chunk_values == nullptr || source.chunk_offsets == nullptr ||
        source.num_chunks == 0) {
        return nullptr;
    }
    if (source.num_chunks == 1) {
        return source.chunk_values[0] == nullptr
                   ? nullptr
                   : source.chunk_values[0] + row_id;
    }
    const auto* upper =
        std::upper_bound(source.chunk_offsets,
                         source.chunk_offsets + source.num_chunks + 1,
                         row_id);
    if (upper == source.chunk_offsets) {
        return nullptr;
    }
    const auto chunk_id =
        static_cast<size_t>((upper - source.chunk_offsets) - 1);
    if (chunk_id >= source.num_chunks ||
        source.chunk_values[chunk_id] == nullptr) {
        return nullptr;
    }
    return source.chunk_values[chunk_id] +
           (row_id - source.chunk_offsets[chunk_id]);
}

const float*
ResolveFloatCandidateValue(const FloatCandidateSourceView& source,
                           int64_t row_id) noexcept {
    if (row_id < 0 || static_cast<size_t>(row_id) >= source.row_count) {
        return nullptr;
    }
    if (source.row_values != nullptr) {
        return source.row_values + row_id;
    }
    if (source.chunk_values == nullptr || source.chunk_offsets == nullptr ||
        source.num_chunks == 0) {
        return nullptr;
    }
    if (source.num_chunks == 1) {
        return source.chunk_values[0] == nullptr
                   ? nullptr
                   : source.chunk_values[0] + row_id;
    }
    const auto* upper =
        std::upper_bound(source.chunk_offsets,
                         source.chunk_offsets + source.num_chunks + 1,
                         row_id);
    if (upper == source.chunk_offsets) {
        return nullptr;
    }
    const auto chunk_id =
        static_cast<size_t>((upper - source.chunk_offsets) - 1);
    if (chunk_id >= source.num_chunks ||
        source.chunk_values[chunk_id] == nullptr) {
        return nullptr;
    }
    return source.chunk_values[chunk_id] +
           (row_id - source.chunk_offsets[chunk_id]);
}

bool
EvaluateInt64Value(const Int64EvaluatorState& state, int64_t value) noexcept {
    switch (state.op) {
        case Int64EvaluatorOp::GreaterEqual:
            return value >= state.arg0;
        case Int64EvaluatorOp::ModLessThan:
            return value % state.arg0 < state.arg1;
        case Int64EvaluatorOp::GreaterThan:
            return value > state.arg0;
        case Int64EvaluatorOp::LessEqual:
            return value <= state.arg0;
        case Int64EvaluatorOp::LessThan:
            return value < state.arg0;
        case Int64EvaluatorOp::Equal:
            return value == state.arg0;
        case Int64EvaluatorOp::NotEqual:
            return value != state.arg0;
        case Int64EvaluatorOp::Range: {
            const bool lower_ok = state.lower_inclusive
                                      ? value >= state.arg0
                                      : value > state.arg0;
            const bool upper_ok = state.upper_inclusive
                                      ? value <= state.arg1
                                      : value < state.arg1;
            return lower_ok && upper_ok;
        }
        case Int64EvaluatorOp::Term:
            return std::binary_search(
                state.terms.begin(), state.terms.end(), value);
        case Int64EvaluatorOp::AddLessThan:
            return static_cast<__int128>(value) +
                       static_cast<__int128>(state.arg0) <
                   static_cast<__int128>(state.arg1);
        case Int64EvaluatorOp::SubLessThan:
            return static_cast<__int128>(value) -
                       static_cast<__int128>(state.arg0) <
                   static_cast<__int128>(state.arg1);
        case Int64EvaluatorOp::MulLessThan:
            return static_cast<__int128>(value) *
                       static_cast<__int128>(state.arg0) <
                   static_cast<__int128>(state.arg1);
        case Int64EvaluatorOp::DivLessThan:
            return static_cast<__int128>(value) /
                       static_cast<__int128>(state.arg0) <
                   static_cast<__int128>(state.arg1);
    }
    return false;
}

bool
EvaluateFloatValue(const FloatEvaluatorState& state, float raw_value) noexcept {
    const float value = raw_value;
    switch (state.op) {
        case Int64EvaluatorOp::GreaterEqual:
            return value >= state.arg0;
        case Int64EvaluatorOp::GreaterThan:
            return value > state.arg0;
        case Int64EvaluatorOp::LessEqual:
            return value <= state.arg0;
        case Int64EvaluatorOp::LessThan:
            return value < state.arg0;
        case Int64EvaluatorOp::Equal:
            return value == state.arg0;
        case Int64EvaluatorOp::NotEqual:
            return value != state.arg0;
        case Int64EvaluatorOp::Range: {
            const bool lower_ok = state.lower_inclusive
                                      ? value >= state.arg0
                                      : value > state.arg0;
            const bool upper_ok = state.upper_inclusive
                                      ? value <= state.arg1
                                      : value < state.arg1;
            return lower_ok && upper_ok;
        }
        case Int64EvaluatorOp::Term:
            return !std::isnan(value) &&
                   std::binary_search(
                       state.terms.begin(), state.terms.end(), value);
        case Int64EvaluatorOp::AddLessThan:
            return value + state.arg0 < state.arg1;
        case Int64EvaluatorOp::SubLessThan:
            return value - state.arg0 < state.arg1;
        case Int64EvaluatorOp::MulLessThan:
            return value * state.arg0 < state.arg1;
        case Int64EvaluatorOp::DivLessThan:
            return value / state.arg0 < state.arg1;
        case Int64EvaluatorOp::ModLessThan:
            return false;
    }
    return false;
}

int32_t
EvaluateInt64Candidates(const void* opaque,
                        const int64_t* row_ids,
                        uint32_t count,
                        uint64_t active_mask,
                        uint64_t* valid_mask) noexcept {
    if (opaque == nullptr || row_ids == nullptr || valid_mask == nullptr ||
        count > 64) {
        return -1;
    }
    const auto& state = *static_cast<const Int64EvaluatorState*>(opaque);

    const uint64_t lane_mask =
        count == 64 ? UINT64_MAX
                    : (count == 0 ? uint64_t{0}
                                  : ((uint64_t{1} << count) - 1));
    const uint64_t active = active_mask & lane_mask;

    std::array<const int64_t*, 64> values{};
    for (uint32_t lane = 0; lane < count; ++lane) {
        if ((active & (uint64_t{1} << lane)) == 0) {
            continue;
        }
        values[lane] = ResolveInt64CandidateValue(state.source, row_ids[lane]);
        if (values[lane] != nullptr) {
            __builtin_prefetch(values[lane], 0, 1);
        }
    }

    uint64_t accepted = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        if (values[lane] != nullptr &&
            EvaluateInt64Value(state, *values[lane])) {
            accepted |= uint64_t{1} << lane;
        }
    }
    *valid_mask = accepted;
    return 0;
}

int32_t
EvaluateInt64ContiguousCandidates(const void* opaque,
                                  int64_t first_row_id,
                                  uint32_t count,
                                  uint64_t active_mask,
                                  uint64_t* valid_mask) noexcept {
    if (opaque == nullptr || valid_mask == nullptr || count > 64 ||
        first_row_id < 0) {
        return -1;
    }
    const auto& state = *static_cast<const Int64EvaluatorState*>(opaque);
    const uint64_t lane_mask =
        count == 64 ? UINT64_MAX
                    : (count == 0 ? uint64_t{0}
                                  : ((uint64_t{1} << count) - 1));
    const uint64_t active = active_mask & lane_mask;
    if (static_cast<size_t>(first_row_id) > state.source.row_count ||
        count > state.source.row_count - static_cast<size_t>(first_row_id)) {
        return -1;
    }

    uint64_t accepted = 0;
    if (state.source.row_values != nullptr) {
        const auto* values = state.source.row_values + first_row_id;
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint64_t lane_bit = uint64_t{1} << lane;
            if ((active & lane_bit) != 0 &&
                EvaluateInt64Value(state, values[lane])) {
                accepted |= lane_bit;
            }
        }
    } else {
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint64_t lane_bit = uint64_t{1} << lane;
            if ((active & lane_bit) == 0) {
                continue;
            }
            const auto* value = ResolveInt64CandidateValue(
                state.source, first_row_id + static_cast<int64_t>(lane));
            if (value != nullptr && EvaluateInt64Value(state, *value)) {
                accepted |= lane_bit;
            }
        }
    }
    *valid_mask = accepted;
    return 0;
}

int32_t
EvaluateFloatCandidates(const void* opaque,
                        const int64_t* row_ids,
                        uint32_t count,
                        uint64_t active_mask,
                        uint64_t* valid_mask) noexcept {
    if (opaque == nullptr || row_ids == nullptr || valid_mask == nullptr ||
        count > 64) {
        return -1;
    }
    const auto& state = *static_cast<const FloatEvaluatorState*>(opaque);
    const uint64_t lane_mask =
        count == 64 ? UINT64_MAX
                    : (count == 0 ? uint64_t{0}
                                  : ((uint64_t{1} << count) - 1));
    const uint64_t active = active_mask & lane_mask;

    std::array<const float*, 64> values{};
    for (uint32_t lane = 0; lane < count; ++lane) {
        if ((active & (uint64_t{1} << lane)) == 0) {
            continue;
        }
        values[lane] = ResolveFloatCandidateValue(state.source, row_ids[lane]);
        if (values[lane] != nullptr) {
            __builtin_prefetch(values[lane], 0, 1);
        }
    }
    uint64_t accepted = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        if (values[lane] != nullptr &&
            EvaluateFloatValue(state, *values[lane])) {
            accepted |= uint64_t{1} << lane;
        }
    }
    *valid_mask = accepted;
    return 0;
}

int32_t
EvaluateFloatContiguousCandidates(const void* opaque,
                                  int64_t first_row_id,
                                  uint32_t count,
                                  uint64_t active_mask,
                                  uint64_t* valid_mask) noexcept {
    if (opaque == nullptr || valid_mask == nullptr || count > 64 ||
        first_row_id < 0) {
        return -1;
    }
    const auto& state = *static_cast<const FloatEvaluatorState*>(opaque);
    const uint64_t lane_mask =
        count == 64 ? UINT64_MAX
                    : (count == 0 ? uint64_t{0}
                                  : ((uint64_t{1} << count) - 1));
    const uint64_t active = active_mask & lane_mask;
    if (static_cast<size_t>(first_row_id) > state.source.row_count ||
        count > state.source.row_count - static_cast<size_t>(first_row_id)) {
        return -1;
    }

    uint64_t accepted = 0;
    if (state.source.row_values != nullptr) {
        const auto* values = state.source.row_values + first_row_id;
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint64_t lane_bit = uint64_t{1} << lane;
            if ((active & lane_bit) != 0 &&
                EvaluateFloatValue(state, values[lane])) {
                accepted |= lane_bit;
            }
        }
    } else {
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint64_t lane_bit = uint64_t{1} << lane;
            if ((active & lane_bit) == 0) {
                continue;
            }
            const auto* value = ResolveFloatCandidateValue(
                state.source, first_row_id + static_cast<int64_t>(lane));
            if (value != nullptr && EvaluateFloatValue(state, *value)) {
                accepted |= lane_bit;
            }
        }
    }
    *valid_mask = accepted;
    return 0;
}

std::optional<Int64EvaluatorOp>
ToInt64EvaluatorOp(CardinalDownpushPredicateOp op) {
    switch (op) {
        case CardinalDownpushPredicateOp::Int64GreaterEqual:
            return Int64EvaluatorOp::GreaterEqual;
        case CardinalDownpushPredicateOp::Int64ModLessThan:
            return Int64EvaluatorOp::ModLessThan;
        case CardinalDownpushPredicateOp::Int64GreaterThan:
            return Int64EvaluatorOp::GreaterThan;
        case CardinalDownpushPredicateOp::Int64LessEqual:
            return Int64EvaluatorOp::LessEqual;
        case CardinalDownpushPredicateOp::Int64LessThan:
            return Int64EvaluatorOp::LessThan;
        case CardinalDownpushPredicateOp::Int64Equal:
            return Int64EvaluatorOp::Equal;
        case CardinalDownpushPredicateOp::Int64NotEqual:
            return Int64EvaluatorOp::NotEqual;
        case CardinalDownpushPredicateOp::ScalarRange:
            return Int64EvaluatorOp::Range;
        case CardinalDownpushPredicateOp::ScalarTerm:
            return Int64EvaluatorOp::Term;
        case CardinalDownpushPredicateOp::ScalarAddLessThan:
            return Int64EvaluatorOp::AddLessThan;
        case CardinalDownpushPredicateOp::ScalarSubLessThan:
            return Int64EvaluatorOp::SubLessThan;
        case CardinalDownpushPredicateOp::ScalarMulLessThan:
            return Int64EvaluatorOp::MulLessThan;
        case CardinalDownpushPredicateOp::ScalarDivLessThan:
            return Int64EvaluatorOp::DivLessThan;
        default:
            return std::nullopt;
    }
}

std::optional<int64_t>
AsInt64(const proto::plan::GenericValue& value) {
    return value.val_case() == proto::plan::GenericValue::kInt64Val
               ? std::optional<int64_t>(value.int64_val())
               : std::nullopt;
}

std::optional<double>
AsDouble(const proto::plan::GenericValue& value) {
    if (value.val_case() == proto::plan::GenericValue::kInt64Val) {
        return static_cast<double>(value.int64_val());
    }
    if (value.val_case() == proto::plan::GenericValue::kFloatVal) {
        return value.float_val();
    }
    return std::nullopt;
}

class NumericProvider final : public DownpushPredicateProvider {
 public:
    bool
    Supports(const expr::ColumnInfo& column) const override {
        const auto type = column.data_type_;
        const bool supported =
            type == DataType::INT8 || type == DataType::INT16 ||
            type == DataType::INT32 || type == DataType::INT64 ||
            type == DataType::TIMESTAMPTZ || type == DataType::FLOAT;
        return supported && !column.nullable_ && !column.element_level_ &&
               column.nested_path_.empty();
    }

    CardinalDownpushPredicate
    NewPredicate(const expr::ColumnInfo& column) const override {
        CardinalDownpushPredicate predicate;
        predicate.field_id_ = column.field_id_;
        predicate.field_data_type_ = column.data_type_;
        predicate.value_type_ = column.data_type_ == DataType::FLOAT
                                    ? CardinalDownpushPredicateValueType::Float
                                    : CardinalDownpushPredicateValueType::Int64;
        return predicate;
    }

    bool
    SupportsRangeOp(CardinalDownpushPredicateOp op) const override {
        return op != CardinalDownpushPredicateOp::StringPrefixMatch &&
               op != CardinalDownpushPredicateOp::StringPostfixMatch &&
               op != CardinalDownpushPredicateOp::StringInnerMatch &&
               op != CardinalDownpushPredicateOp::StringLikeMatch;
    }

    bool
    FillArg(CardinalDownpushPredicate& predicate,
            const proto::plan::GenericValue& value,
            bool second_arg) const override {
        if (predicate.value_type_ ==
            CardinalDownpushPredicateValueType::Int64) {
            const auto converted = AsInt64(value);
            if (!converted.has_value()) {
                return false;
            }
            (second_arg ? predicate.arg1_ : predicate.arg0_) = *converted;
            return true;
        }
        const auto converted = AsDouble(value);
        if (!converted.has_value()) {
            return false;
        }
        (second_arg ? predicate.double_arg1_ : predicate.double_arg0_) =
            *converted;
        return true;
    }

    bool
    FillTerms(
        CardinalDownpushPredicate& predicate,
        const std::vector<proto::plan::GenericValue>& values) const override {
        if (values.empty()) {
            return false;
        }
        if (predicate.value_type_ ==
            CardinalDownpushPredicateValueType::Int64) {
            for (const auto& value : values) {
                const auto converted = AsInt64(value);
                if (!converted.has_value()) {
                    return false;
                }
                predicate.int64_terms_.push_back(*converted);
            }
            std::sort(predicate.int64_terms_.begin(),
                      predicate.int64_terms_.end());
            predicate.int64_terms_.erase(
                std::unique(predicate.int64_terms_.begin(),
                            predicate.int64_terms_.end()),
                predicate.int64_terms_.end());
            return true;
        }
        for (const auto& value : values) {
            const auto converted = AsDouble(value);
            if (!converted.has_value()) {
                return false;
            }
            predicate.double_terms_.push_back(*converted);
        }
        std::sort(predicate.double_terms_.begin(),
                  predicate.double_terms_.end());
        predicate.double_terms_.erase(
            std::unique(predicate.double_terms_.begin(),
                        predicate.double_terms_.end()),
            predicate.double_terms_.end());
        return true;
    }

    bool
    FillArithmetic(
        CardinalDownpushPredicate& predicate,
        const expr::BinaryArithOpEvalRangeExpr& expression) const override {
        if (expression.op_type_ != proto::plan::OpType::LessThan) {
            return false;
        }
        if (expression.arith_op_type_ == proto::plan::ArithOpType::Mod) {
            const auto modulus = AsInt64(expression.right_operand_);
            const auto threshold = AsInt64(expression.value_);
            if (predicate.value_type_ !=
                    CardinalDownpushPredicateValueType::Int64 ||
                !modulus.has_value() || !threshold.has_value() ||
                *modulus <= 0 || *threshold < 0 || *threshold > *modulus) {
                return false;
            }
            predicate.op_ = CardinalDownpushPredicateOp::Int64ModLessThan;
            predicate.arg0_ = *modulus;
            predicate.arg1_ = *threshold;
            return true;
        }
        CardinalDownpushPredicateOp op;
        switch (expression.arith_op_type_) {
            case proto::plan::ArithOpType::Add:
                op = CardinalDownpushPredicateOp::ScalarAddLessThan;
                break;
            case proto::plan::ArithOpType::Sub:
                op = CardinalDownpushPredicateOp::ScalarSubLessThan;
                break;
            case proto::plan::ArithOpType::Mul:
                op = CardinalDownpushPredicateOp::ScalarMulLessThan;
                break;
            case proto::plan::ArithOpType::Div:
                op = CardinalDownpushPredicateOp::ScalarDivLessThan;
                break;
            default:
                return false;
        }
        predicate.op_ = op;
        if (predicate.value_type_ ==
            CardinalDownpushPredicateValueType::Int64) {
            const auto operand = AsInt64(expression.right_operand_);
            const auto threshold = AsInt64(expression.value_);
            if (!operand.has_value() || !threshold.has_value() ||
                (expression.arith_op_type_ == proto::plan::ArithOpType::Div &&
                 *operand == 0)) {
                return false;
            }
            predicate.arg0_ = *operand;
            predicate.arg1_ = *threshold;
            return true;
        }
        const auto operand = AsDouble(expression.right_operand_);
        const auto threshold = AsDouble(expression.value_);
        if (!operand.has_value() || !threshold.has_value() ||
            (expression.arith_op_type_ == proto::plan::ArithOpType::Div &&
             *operand == 0.0)) {
            return false;
        }
        predicate.double_arg0_ = *operand;
        predicate.double_arg1_ = *threshold;
        return true;
    }
};

}  // namespace

const DownpushPredicateProvider&
NumericDownpushPredicateProvider() {
    static const NumericProvider provider;
    return provider;
}

std::optional<PreparedCandidateEvaluator>
PrepareInt64ModCandidateEvaluator(const Int64CandidateSourceView& source,
                                  int64_t divisor,
                                  int64_t upper_bound) {
    CardinalDownpushPredicate predicate;
    predicate.value_type_ = CardinalDownpushPredicateValueType::Int64;
    predicate.op_ = CardinalDownpushPredicateOp::Int64ModLessThan;
    predicate.arg0_ = divisor;
    predicate.arg1_ = upper_bound;
    return PrepareInt64CandidateEvaluator(source, predicate);
}

std::optional<PreparedCandidateEvaluator>
PrepareInt64CandidateEvaluator(const Int64CandidateSourceView& source,
                               const CardinalDownpushPredicate& predicate) {
    const bool has_contiguous_source = source.row_values != nullptr;
    const bool has_chunked_source = source.chunk_values != nullptr &&
                                    source.chunk_offsets != nullptr &&
                                    source.num_chunks > 0;
    const auto op = ToInt64EvaluatorOp(predicate.op_);
    if (predicate.value_type_ !=
            CardinalDownpushPredicateValueType::Int64 ||
        !op.has_value() || source.row_count == 0 ||
        (!has_contiguous_source && !has_chunked_source)) {
        return std::nullopt;
    }
    if (*op == Int64EvaluatorOp::ModLessThan &&
        (predicate.arg0_ <= 0 || predicate.arg1_ < 0 ||
         predicate.arg1_ > predicate.arg0_)) {
        return std::nullopt;
    }
    if (*op == Int64EvaluatorOp::Term && predicate.int64_terms_.empty()) {
        return std::nullopt;
    }
    if (*op == Int64EvaluatorOp::DivLessThan && predicate.arg0_ == 0) {
        return std::nullopt;
    }
    auto owner = std::make_shared<Int64EvaluatorState>(Int64EvaluatorState{
        source,
        *op,
        predicate.arg0_,
        predicate.arg1_,
        predicate.lower_inclusive_,
        predicate.upper_inclusive_,
        predicate.int64_terms_});
    if (*op == Int64EvaluatorOp::Term) {
        std::sort(owner->terms.begin(), owner->terms.end());
        owner->terms.erase(
            std::unique(owner->terms.begin(), owner->terms.end()),
            owner->terms.end());
    }
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    prepared.view.eval_batch = &EvaluateInt64Candidates;
    prepared.view.eval_contiguous = &EvaluateInt64ContiguousCandidates;
    return prepared;
}

std::optional<PreparedCandidateEvaluator>
PrepareFloatCandidateEvaluator(const FloatCandidateSourceView& source,
                               const CardinalDownpushPredicate& predicate) {
    const bool has_contiguous_source = source.row_values != nullptr;
    const bool has_chunked_source = source.chunk_values != nullptr &&
                                    source.chunk_offsets != nullptr &&
                                    source.num_chunks > 0;
    const auto op = ToInt64EvaluatorOp(predicate.op_);
    if (predicate.value_type_ != CardinalDownpushPredicateValueType::Float ||
        !op.has_value() || *op == Int64EvaluatorOp::ModLessThan ||
        source.row_count == 0 ||
        (!has_contiguous_source && !has_chunked_source)) {
        return std::nullopt;
    }
    if (*op == Int64EvaluatorOp::Term && predicate.double_terms_.empty()) {
        return std::nullopt;
    }
    if (*op == Int64EvaluatorOp::DivLessThan &&
        predicate.double_arg0_ == 0.0) {
        return std::nullopt;
    }
    auto owner = std::make_shared<FloatEvaluatorState>(FloatEvaluatorState{
        source,
        *op,
        static_cast<float>(predicate.double_arg0_),
        static_cast<float>(predicate.double_arg1_),
        predicate.lower_inclusive_,
        predicate.upper_inclusive_,
        {}});
    if (*op == Int64EvaluatorOp::Term) {
        owner->terms.reserve(predicate.double_terms_.size());
        for (const auto value : predicate.double_terms_) {
            owner->terms.push_back(static_cast<float>(value));
        }
        owner->terms.erase(
            std::remove_if(owner->terms.begin(),
                           owner->terms.end(),
                           [](float value) { return std::isnan(value); }),
            owner->terms.end());
        std::sort(owner->terms.begin(), owner->terms.end());
        owner->terms.erase(
            std::unique(owner->terms.begin(), owner->terms.end()),
            owner->terms.end());
    }
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    prepared.view.eval_batch = &EvaluateFloatCandidates;
    prepared.view.eval_contiguous = &EvaluateFloatContiguousCandidates;
    return prepared;
}

}  // namespace milvus::exec
