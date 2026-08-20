// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/DownpushPredicateProvider.h"

#include <algorithm>

namespace milvus::exec {
namespace {

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

}  // namespace milvus::exec
