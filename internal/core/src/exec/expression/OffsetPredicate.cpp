// Licensed under the Apache License, Version 2.0.
#include "exec/expression/OffsetPredicate.h"
#include <algorithm>
#include <limits>
#include "exec/expression/Expr.h"

namespace milvus::exec {
using EvalStatus = knowhere::CandidateEvalStatus;

struct OffsetPredicate::Worker {
    explicit Worker(const OffsetPredicate& owner)
        : owner(owner),
          query(owner.query_->MakeOffsetContext()),
          ctx(query.get()),
          root(CompileExpression(owner.predicate_, query.get(), {}, false)) {
        AssertInfo(root && root->SupportOffsetInput(),
                   "offset predicate is not supported");
        root->MarkNullRejecting();
    }

    // Compile a task-private physical root directly: sequential filter-cache
    // and batch-LIKE rewrites for a full scan do not apply to arbitrary offsets.
    const OffsetPredicate& owner;
    std::unique_ptr<QueryContext> query;
    ExecContext ctx;
    ExprPtr root;
    OffsetVector offsets;
    std::vector<size_t> lanes;
};

OffsetPredicate::OffsetPredicate(expr::TypedExprPtr predicate,
                                 QueryContext* query)
    : predicate_(std::move(predicate)), query_(query) {
    view_.row_count = query_->get_active_count();
    view_.context = this;
    view_.create_worker = [](const void* ctx, void** out) noexcept {
        if (!out || !ctx)
            return EvalStatus::InvalidArgument;
        *out = nullptr;
        const auto& owner = *static_cast<const OffsetPredicate*>(ctx);
        try {
            *out = new Worker(owner);
            return EvalStatus::Success;
        } catch (...) {
            owner.CaptureFailure();
            return EvalStatus::Failed;
        }
    };
    view_.destroy_worker = [](void* worker) noexcept {
        delete static_cast<Worker*>(worker);
    };
    view_.eval_batch = [](void* state,
                          const int32_t* ids,
                          size_t count,
                          const uint64_t* active,
                          uint64_t* accepted) noexcept {
        if (!state || (!ids && count) || (!accepted && count) ||
            count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            return EvalStatus::InvalidArgument;
        auto& worker = *static_cast<Worker*>(state);
        try {
            checkCancellation(worker.query.get());
            worker.offsets.clear();
            worker.lanes.clear();
            std::fill_n(accepted, (count + 63) / 64, uint64_t{0});
            for (size_t i = 0; i < count; ++i) {
                if (active && !(active[i / 64] & (uint64_t{1} << (i % 64))))
                    continue;
                if (ids[i] < 0 ||
                    size_t(ids[i]) >= worker.owner.view_.row_count)
                    return EvalStatus::InvalidArgument;
                worker.offsets.push_back(ids[i]);
                if (active)
                    worker.lanes.push_back(i);
            }
            if (worker.offsets.empty())
                return EvalStatus::Success;
            EvalCtx eval(&worker.ctx, &worker.offsets);
            VectorPtr result;
            worker.root->Eval(eval, result);
            auto column = std::dynamic_pointer_cast<ColumnVector>(result);
            AssertInfo(
                column && column->IsBitmap() &&
                    column->size() == worker.offsets.size(),
                "offset predicate must produce one boolean per input row");
            TargetBitmapView truth(column->GetRawData(), column->size());
            TargetBitmapView valid(column->GetValidRawData(), column->size());
            for (size_t i = 0; i < worker.offsets.size(); ++i) {
                if (valid[i] && truth[i]) {
                    const auto lane = active ? worker.lanes[i] : i;
                    accepted[lane / 64] |= uint64_t{1} << (lane % 64);
                }
            }
            return EvalStatus::Success;
        } catch (...) {
            worker.owner.CaptureFailure();
            return EvalStatus::Failed;
        }
    };
}

void
OffsetPredicate::CaptureFailure() const noexcept {
    std::lock_guard lock(mutex_);
    if (!failure_)
        failure_ = std::current_exception();
}

void
OffsetPredicate::RethrowIfFailed() const {
    std::lock_guard lock(mutex_);
    if (failure_)
        std::rethrow_exception(failure_);
}
}  // namespace milvus::exec
