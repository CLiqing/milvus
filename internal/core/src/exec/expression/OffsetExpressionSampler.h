// Licensed under the Apache License, Version 2.0.
#pragma once
#include "exec/expression/OffsetExpressionEvaluator.h"
namespace milvus::exec {
// Caller supplies the already-checked shared physical chunk layout. Sampling
// performs actual shared Expr evaluation, never a disposable capability probe.
std::optional<double>
SampleOffsetFilterRatio(const expr::TypedExprPtr& expression,
                        ExecContext* context,
                        std::shared_ptr<const ChunkedColumnInterface> column);
}  // namespace milvus::exec
