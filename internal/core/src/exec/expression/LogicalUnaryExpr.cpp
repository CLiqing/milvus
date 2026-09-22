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

#include "LogicalUnaryExpr.h"

#include "common/EasyAssert.h"
#include "common/Tracer.h"
#include "common/ValueOp.h"
#include "exec/expression/Utils.h"

namespace milvus {
namespace exec {

void
PhyLogicalUnaryExpr::Eval(EvalCtx& context, VectorPtr& result) {
    tracer::AutoSpan span("PhyLogicalUnaryExpr::Eval", tracer::GetRootSpan());

    AssertInfo(inputs_.size() == 1,
               "logical unary expr must has one input, but now {}",
               inputs_.size());

    inputs_[0]->Eval(context, result);
    if (expr_->op_type_ == milvus::expr::LogicalUnaryExpr::OpType::LogicalNot) {
        common::ThreeValuedLogicOp::Not(GetColumnVector(result));
    }
}

bool
PhyLogicalUnaryExpr::MayDeferFiltering(const FilterScheduleContext& context) {
    return inputs_[0]->MayDeferFiltering(context);
}

FilterSchedule
PhyLogicalUnaryExpr::ScheduleFiltering(const FilterScheduleContext& context,
                                       bool) {
    // NOT is a hard boundary for lifting mandatory conditions. The child must
    // retain three-valued truth until the original NOT evaluator consumes it.
    auto child = inputs_[0]->ScheduleFiltering(context, false);
    AssertInfo(!(child.baseline && child.residual),
               "NOT child was incorrectly split");
    if (!child.residual) {
        return {logical_source_, nullptr};
    }
    return {nullptr,
            std::make_shared<expr::LogicalUnaryExpr>(expr_->op_type_,
                                                     child.residual)};
}

}  //namespace exec
}  // namespace milvus
