// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#pragma once

#include <optional>
#include <string>

#include "common/AnnFusingPlugin.h"
#include "expr/ITypeExpr.h"
#include "segcore/SegmentInterface.h"

namespace milvus::exec {

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

}  // namespace milvus::exec
