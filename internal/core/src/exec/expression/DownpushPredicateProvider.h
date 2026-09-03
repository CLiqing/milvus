// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <optional>

#include "exec/expression/OffsetExpressionEvaluator.h"
#include "exec/operator/DownpushSearchContext.h"
#include "expr/ITypeExpr.h"

namespace milvus::exec {

// Numeric arithmetic owns its typed lowering and preparation. The returned
// leaf is opaque to the logical fusing pipeline.
std::optional<CandidateLeafPlan>
TryCompileNumericArithmeticCandidateLeaf(
    const expr::BinaryArithOpEvalRangeExpr& expression);

// Numeric comparison/range/TERM lowering is entirely type-owned.  The
// generic FilterBits composer neither maps Numeric operators nor stores their
// literals.
std::optional<CandidateLeafPlan>
TryCompileNumericCandidateLeaf(const expr::TypedExprPtr& expression);

// Stage-8E vertical slice: prepare one sealed, non-nullable INT64 comparison
// or constrained MOD leaf as the shared random-offset evaluator. Unsupported
// layouts and expressions return nullptr so iterative keeps its existing path.
std::shared_ptr<const PreparedOffsetExpressionEvaluator>
PrepareNumericOffsetExpressionEvaluator(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    const expr::TypedExprPtr& expression);

std::optional<CandidateLeafPlan>
TryCompileStringCandidateLeaf(const expr::TypedExprPtr& expression);

}  // namespace milvus::exec
