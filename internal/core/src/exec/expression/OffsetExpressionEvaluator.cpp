// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/OffsetExpressionEvaluator.h"

#include <algorithm>
#include <utility>

#include "common/EasyAssert.h"
#include "common/Types.h"
#include "exec/expression/EvalCtx.h"

namespace milvus::exec {
namespace {

uint64_t
LaneMask(uint32_t count) {
    return count == 64
               ? ~uint64_t{0}
               : (count == 0 ? uint64_t{0} : (uint64_t{1} << count) - 1);
}

}  // namespace

OffsetExpressionWorkspace::OffsetExpressionWorkspace(
    const std::vector<expr::TypedExprPtr>& logical_expressions,
    ExecContext* exec_context,
    bool null_rejecting)
    : exec_context_(exec_context),
      expr_set_(logical_expressions, exec_context, null_rejecting) {
    AssertInfo(exec_context_ != nullptr,
               "offset expression evaluator requires an ExecContext");
    compact_offsets_.reserve(64);
    compact_to_lane_.reserve(64);
}

bool
OffsetExpressionWorkspace::SupportsOffsetInput() const {
    return std::all_of(expr_set_.exprs().begin(),
                       expr_set_.exprs().end(),
                       [](const auto& expression) {
                           return expression != nullptr &&
                                  expression->SupportOffsetInput();
                       });
}

std::shared_ptr<ColumnVector>
OffsetExpressionWorkspace::EvalOffsets(OffsetVector& row_ids) {
    EvalCtx eval_context(exec_context_, &row_ids);
    results_.clear();
    expr_set_.Eval(0, 1, true, eval_context, results_);
    AssertInfo(results_.size() == 1 && results_[0] != nullptr,
               "offset expression must produce exactly one result");
    auto output = std::dynamic_pointer_cast<ColumnVector>(results_[0]);
    AssertInfo(output != nullptr && output->IsBitmap(),
               "offset expression result must be a bitmap ColumnVector");
    AssertInfo(output->size() == row_ids.size(),
               "offset expression result size {} does not match input {}",
               output->size(),
               row_ids.size());
    return output;
}

OffsetExpressionTruth
OffsetExpressionWorkspace::EvalBatch(const int32_t* row_ids,
                                     uint32_t count,
                                     uint64_t active_mask) {
    AssertInfo(count <= 64,
               "offset expression batch size {} exceeds 64",
               count);
    const auto lane_mask = LaneMask(count);
    AssertInfo((active_mask & ~lane_mask) == 0,
               "offset expression active mask contains lanes outside batch");
    AssertInfo(count == 0 || row_ids != nullptr,
               "offset expression row IDs cannot be null");

    OffsetExpressionTruth truth;
    if (active_mask == 0) {
        return truth;
    }

    compact_offsets_.clear();
    compact_to_lane_.clear();
    for (uint32_t lane = 0; lane < count; ++lane) {
        if ((active_mask & (uint64_t{1} << lane)) != 0) {
            compact_offsets_.push_back(row_ids[lane]);
            compact_to_lane_.push_back(lane);
        }
    }

    auto output = EvalOffsets(compact_offsets_);

    TargetBitmapView data(output->GetRawData(), output->size());
    TargetBitmapView valid(output->GetValidRawData(), output->size());
    for (size_t compact_lane = 0; compact_lane < compact_to_lane_.size();
         ++compact_lane) {
        const auto original_lane = compact_to_lane_[compact_lane];
        if (valid[compact_lane]) {
            truth.known_mask |= uint64_t{1} << original_lane;
            if (data[compact_lane]) {
                truth.true_mask |= uint64_t{1} << original_lane;
            }
        }
    }
    return truth;
}

PreparedOffsetExpressionEvaluator::PreparedOffsetExpressionEvaluator(
    std::vector<expr::TypedExprPtr> logical_expressions,
    ExecContext* exec_context,
    bool null_rejecting)
    : logical_expressions_(std::move(logical_expressions)),
      exec_context_(exec_context),
      null_rejecting_(null_rejecting) {
    AssertInfo(!logical_expressions_.empty(),
               "offset expression evaluator requires an expression");
    AssertInfo(exec_context_ != nullptr,
               "offset expression evaluator requires an ExecContext");
}

std::unique_ptr<OffsetExpressionWorkspace>
PreparedOffsetExpressionEvaluator::CreateWorkspace() const {
    return std::make_unique<OffsetExpressionWorkspace>(
        logical_expressions_, exec_context_, null_rejecting_);
}

}  // namespace milvus::exec
