// Licensed under the Apache License, Version 2.0.
#pragma once

#include <exception>
#include <mutex>
#include "exec/QueryContext.h"
#include "expr/ITypeExpr.h"
#include "knowhere/candidate_evaluator.h"

namespace milvus::exec {

// Lives in the query context until the synchronous vector search has joined its tasks.
class OffsetPredicate {
 public:
    OffsetPredicate(expr::TypedExprPtr predicate, QueryContext* query);
    const knowhere::CandidateEvaluatorView&
    view() const {
        return view_;
    }
    void
    RethrowIfFailed() const;

 private:
    struct Worker;
    void
    CaptureFailure() const noexcept;
    expr::TypedExprPtr predicate_;
    QueryContext* query_;
    knowhere::CandidateEvaluatorView view_;
    mutable std::mutex mutex_;
    mutable std::exception_ptr failure_;
};

}  // namespace milvus::exec
