// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#include "exec/AnnFusingPlan.h"

namespace milvus::exec {

std::optional<AnnFusingLeafFacts>
DescribeAnnFusingLeaf(const expr::TypedExprPtr& expression,
                      const segcore::SegmentInternalInterface& segment,
                      OpContext* op_context) {
    const expr::ColumnInfo* column = nullptr;
    std::string operation;
    if (auto range =
            dynamic_cast<const expr::UnaryRangeFilterExpr*>(expression.get())) {
        column = &range->column_;
        switch (range->op_type_) {
            case proto::plan::GreaterThan:
            case proto::plan::GreaterEqual:
            case proto::plan::LessThan:
            case proto::plan::LessEqual:
                operation = "range";
                break;
            default:
                operation = proto::plan::OpType_Name(range->op_type_);
        }
    } else if (auto range = dynamic_cast<const expr::BinaryRangeFilterExpr*>(
                   expression.get())) {
        column = &range->column_;
        operation = "range";
    } else if (auto arithmetic =
                   dynamic_cast<const expr::BinaryArithOpEvalRangeExpr*>(
                       expression.get())) {
        column = &arithmetic->column_;
        operation = proto::plan::ArithOpType_Name(arithmetic->arith_op_type_);
    }
    // V1 describes scalar leaves only. Do not mislabel JSON/array/nested
    // indexes as ordinary scalar indexes when collecting planner facts.
    if (!column || column->element_level_ || !column->nested_path_.empty()) {
        return std::nullopt;
    }
    std::string index_type = "NONE";
    if (segment.HasIndex(column->field_id_)) {
        auto pinned = segment.PinIndex(op_context, column->field_id_);
        if (!pinned.empty()) {
            index_type = pinned.front().get()->Type();
        }
    }
    return AnnFusingLeafFacts{
        column->field_id_,
        proto::schema::DataType_Name(
            static_cast<proto::schema::DataType>(column->data_type_)),
        std::move(operation),
        std::move(index_type)};
}

}  // namespace milvus::exec
