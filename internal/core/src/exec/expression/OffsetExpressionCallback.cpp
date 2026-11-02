// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#include "exec/expression/OffsetExpressionCallback.h"

#include <array>

namespace milvus::exec {
using Status = knowhere::CandidateEvalStatus;
namespace {
// Result transport only: preserve the Expr's packed bitmaps instead of
// unpacking/repacking every accepted row. Read allocated bytes only, without
// alignment or host-endianness assumptions. The caller masks the final byte.
uint64_t
ReadPackedLanes(const void* data, uint32_t count) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t mask = 0;
    for (uint32_t i = 0; i < (count + 7) / 8; ++i) {
        mask |= uint64_t{bytes[i]} << (8 * i);
    }
    return mask;
}
}  // namespace

struct OffsetExpressionCallback::Worker {
    explicit Worker(const OffsetExpressionCallback* factory)
        : factory(factory),
          query(factory->query_context_->MakeOffsetContext()),
          context(query.get()),
          workspace(factory->prepared_.CreateWorkspace(&context)) {
        AssertInfo(workspace->SupportsOffsetInput(),
                   "callback expression does not support offsets");
        offsets.reserve(64);
    }
    const OffsetExpressionCallback* factory;
    std::unique_ptr<QueryContext> query;
    ExecContext context;
    std::unique_ptr<OffsetExpressionWorkspace> workspace;
    OffsetVector offsets;
    std::array<uint32_t, 64> lanes;
};

OffsetExpressionCallback::OffsetExpressionCallback(
    expr::TypedExprPtr expression, ExecContext* context, int64_t row_count,
    OffsetExpressionOptions options)
    : expression_(std::move(expression)),
      prepared_(expression_, options),
      query_context_(context ? context->get_query_context() : nullptr),
      row_count_(row_count) {
    AssertInfo(query_context_ != nullptr && row_count >= 0 &&
                   row_count <= query_context_->get_active_count(),
               "callback requires a query context and valid active domain");
}

knowhere::CandidateEvaluatorView
OffsetExpressionCallback::view() const {
    return {1, sizeof(knowhere::CandidateEvaluatorView), this,
            CreateWorker, DestroyWorker, EvalBatch};
}

TargetBitmap
OffsetExpressionCallback::EvalBaseline() const {
    auto query = query_context_->MakeOffsetContext();
    ExecContext context(query.get());
    ExprSet expressions({expression_}, &context, true);
    EvalCtx eval(&context);
    return EvalExprSetOverAllBatches(expressions, eval, row_count_,
                                    "deferred-entry baseline materialization");
}

void
OffsetExpressionCallback::CaptureException() const noexcept {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    if (!first_failure_) {
        first_failure_ = std::current_exception();
    }
}

void
OffsetExpressionCallback::RethrowIfFailed() const {
    std::exception_ptr failure;
    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        failure = first_failure_;
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

Status
OffsetExpressionCallback::CreateWorker(const void* context,
                                       void** output) noexcept {
    if (output == nullptr) {
        return Status::InvalidArgument;
    }
    *output = nullptr;
    if (context == nullptr) {
        return Status::InvalidArgument;
    }
    const auto* factory = static_cast<const OffsetExpressionCallback*>(context);
    try {
        *output = new Worker(factory);
        return Status::Success;
    } catch (...) {
        factory->CaptureException();
        return Status::Failed;
    }
}

void
OffsetExpressionCallback::DestroyWorker(void* worker) noexcept {
    delete static_cast<Worker*>(worker);
}

Status
OffsetExpressionCallback::EvalBatch(void* opaque, const int32_t* ids,
                                   uint32_t count, uint64_t active,
                                   uint64_t* accepted) noexcept {
    if (opaque == nullptr || accepted == nullptr || count > 64 ||
        (count != 0 && ids == nullptr) ||
        (count < 64 && (active >> count) != 0)) {
        return Status::InvalidArgument;
    }
    *accepted = 0;
    auto& worker = *static_cast<Worker*>(opaque);
    try {
        checkCancellation(worker.query.get());
        worker.offsets.clear();
        const auto lanes = count == 64 ? ~uint64_t{0}
                                      : (uint64_t{1} << count) - 1;
        const bool all_active = active == lanes;
        if (all_active) {
            // No positional restoration is needed when every input lane is
            // active. Keep the input copy contiguous and omit the lane map.
            if (count != 0) {
                worker.offsets.assign(ids, ids + count);
            }
            for (const auto id : worker.offsets) {
                AssertInfo(id >= 0 && id < worker.factory->row_count_,
                           "callback offset outside active domain");
            }
        } else {
            for (uint32_t lane = 0; lane < count; ++lane) {
                if ((active & (uint64_t{1} << lane)) == 0) {
                    continue;
                }
                AssertInfo(ids[lane] >= 0 && ids[lane] < worker.factory->row_count_,
                           "callback offset outside active domain");
                worker.lanes[worker.offsets.size()] = lane;
                worker.offsets.push_back(ids[lane]);
            }
        }
        if (worker.offsets.empty()) {
            return Status::Success;
        }
        const auto result = worker.workspace->EvalOffsets(worker.offsets);
        if (all_active) {
            *accepted = ReadPackedLanes(result->GetRawData(), count) &
                        ReadPackedLanes(result->GetValidRawData(), count) & lanes;
            return Status::Success;
        }
        TargetBitmapView truth(result->GetRawData(), result->size());
        TargetBitmapView known(result->GetValidRawData(), result->size());
        uint64_t mask = 0;
        for (size_t i = 0; i < worker.offsets.size(); ++i) {
            if (truth[i] && known[i]) {
                mask |= uint64_t{1} << worker.lanes[i];
            }
        }
        *accepted = mask;
        return Status::Success;
    } catch (...) {
        worker.factory->CaptureException();
        return Status::Failed;
    }
}
}  // namespace milvus::exec
