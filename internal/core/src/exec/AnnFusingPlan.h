// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#pragma once

#include <optional>
#include <string>

#include "common/AnnFusingPlugin.h"
#include "expr/ITypeExpr.h"
#include "segcore/SegmentInterface.h"

namespace milvus::exec {

class ExecContext;

// Facts only, not a second predicate implementation. Policy ownership stays
// in the plugin; expression truth stays in the original physical Exprs.
struct AnnFusingLeafFacts {
    FieldId field_id;
    std::string data_type;
    std::string operation;
    std::string index_type;

    MilvusAnnFusingRuleV1
    view() const {
        return {sizeof(MilvusAnnFusingRuleV1),
                data_type.c_str(),
                operation.c_str(),
                index_type.c_str()};
    }
};

std::optional<AnnFusingLeafFacts>
DescribeAnnFusingLeaf(const expr::TypedExprPtr& expression,
                      const segcore::SegmentInternalInterface& segment,
                      OpContext* op_context);

// Evaluate 10/20 original offsets from one actual scalar chunk. This owns a
// separate workspace, so sampling never consumes the baseline ExprSet cursor.
// Only the estimate escapes; sampled offsets remain local to this call.
std::optional<double>
SampleAnnFusingFilterRatio(const expr::TypedExprPtr& expression,
                         FieldId field_id,
                         ExecContext* exec_context);

struct AnnFusingExecutionPlan {
    expr::TypedExprPtr baseline;  // null means no user predicate before search
    expr::TypedExprPtr residual;  // null means entirely baseline
    // Advisory residual filter ratio, never a bitmap count/result capacity.
    std::optional<double> residual_filter_ratio;
};

AnnFusingExecutionPlan
PlanAnnFusingExpression(const expr::TypedExprPtr& expression,
                        ExecContext* exec_context,
                        AnnFilterFusingRequest request);

}  // namespace milvus::exec
