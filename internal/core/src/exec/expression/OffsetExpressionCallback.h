// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#pragma once

#include <exception>
#include <mutex>

#include "exec/expression/OffsetExpressionEvaluator.h"
#include "knowhere/candidate_evaluator.h"

namespace milvus::exec {

// Thin production transport over the foundation Expr API, not another kernel.
// The query owns this factory and the leased source snapshot until all search
// tasks have joined. Each bind creates its own context, Expr and read budget.
class OffsetExpressionCallback final {
 public:
    OffsetExpressionCallback(expr::TypedExprPtr expression,
                             ExecContext* context,
                             int64_t row_count,
                             OffsetExpressionOptions options = {});

    knowhere::CandidateEvaluatorView
    view(bool memoize = true) const;

    // After search tasks join, restore the original storage/Expr/cancellation
    // exception rather than classifying it from a generic backend string.
    void
    RethrowIfFailed() const;

    TargetBitmap
    EvalBaseline() const;

 private:
    struct Worker;
    void
    CaptureException() const noexcept;
    static knowhere::CandidateEvalStatus
    CreateWorker(const void* context, void** output) noexcept;
    static void
    DestroyWorker(void* worker) noexcept;
    static knowhere::CandidateEvalStatus
    EvalBatch(void* worker,
              const int32_t* ids,
              size_t count,
              const uint64_t* active_mask,
              uint64_t* accepted_mask) noexcept;

    expr::TypedExprPtr expression_;
    PreparedOffsetExpressionEvaluator prepared_;
    QueryContext* query_context_;
    int64_t row_count_;
    mutable std::mutex failure_mutex_;
    mutable std::exception_ptr first_failure_;
};
}  // namespace milvus::exec
