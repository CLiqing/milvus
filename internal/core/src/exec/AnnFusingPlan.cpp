// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#include "exec/AnnFusingPlan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "exec/expression/OffsetExpressionEvaluator.h"
#include "log/Log.h"

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

std::optional<double>
SampleAnnFusingRejection(const expr::TypedExprPtr& expression,
                         FieldId field_id,
                         ExecContext* exec_context) {
    auto* query = exec_context->get_query_context();
    const auto params = query->get_search_info().search_params_;
    // Development-only input for isolating CAP scalar costs from estimator
    // error. It is a ratio, never the bitmap's exact count.
    if (params.contains("ann_fusing_rejection_ratio")) {
        const double ratio =
            params.at("ann_fusing_rejection_ratio").get<double>();
        AssertInfo(std::isfinite(ratio) && ratio >= 0 && ratio <= 1,
                   "ann_fusing_rejection_ratio must be in [0,1]");
        LOG_DEBUG("ann_fusing sample source=injected rejection_ratio={}",
                  ratio);
        return ratio;
    }
    const int requested = params.value("ann_fusing_sample_rows", 20);
    AssertInfo(requested == 10 || requested == 20,
               "ann_fusing_sample_rows must be 10 or 20");
    const auto* segment = query->get_segment();
    const auto active = query->get_active_count();
    if (active <= 0 || active > std::numeric_limits<int32_t>::max() ||
        !segment->HasFieldData(field_id)) {
        return std::nullopt;
    }
    const auto chunks = segment->num_chunk_data(field_id);
    if (chunks <= 0) {
        return std::nullopt;
    }
    // Select offsets, never generate data values. Sampling uses the server's
    // scalar chunk boundaries, not client parquet row groups.
    std::mt19937_64 random(static_cast<uint64_t>(segment->get_segment_id()) ^
                           query->get_query_timestamp());
    const auto chunk =
        std::uniform_int_distribution<int64_t>(0, chunks - 1)(random);
    const auto first = segment->num_rows_until_chunk(field_id, chunk);
    const auto rows =
        std::min<int64_t>(segment->chunk_size(field_id, chunk), active - first);
    if (first < 0 || rows <= 0) {
        return std::nullopt;
    }
    const auto count = std::min<int64_t>(requested, rows);
    OffsetVector offsets;
    offsets.reserve(count);
    std::uniform_int_distribution<int64_t> choose(0, rows - 1);
    while (offsets.size() < count) {
        const auto offset = first + choose(random);
        if (std::find(offsets.begin(), offsets.end(), offset) ==
            offsets.end()) {
            offsets.push_back(offset);
        }
    }
    PreparedOffsetExpressionEvaluator prepared(expression, exec_context, true);
    auto workspace = prepared.CreateWorkspace();
    if (!workspace->SupportsOffsetInput()) {
        return std::nullopt;
    }
    auto result = workspace->EvalOffsets(offsets);
    TargetBitmapView truth(result->GetRawData(), result->size());
    TargetBitmapView valid(result->GetValidRawData(), result->size());
    size_t accepted = 0;
    for (size_t i = 0; i < offsets.size(); ++i) {
        accepted += truth[i] && valid[i];
    }
    const double ratio = 1.0 - static_cast<double>(accepted) / count;
    LOG_DEBUG(
        "ann_fusing sample source=chunk field={} chunk={} rows={} "
        "chunk_first={} chunk_rows={} rejection_ratio={}",
        field_id.get(),
        chunk,
        count,
        first,
        rows,
        ratio);
    return ratio;
}

}  // namespace milvus::exec
