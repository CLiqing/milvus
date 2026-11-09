// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#include "exec/expression/OffsetExpressionCallback.h"

#include <algorithm>
#include <limits>

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
        lanes.reserve(64);
    }
    const OffsetExpressionCallback* factory;
    std::unique_ptr<QueryContext> query;
    ExecContext context;
    std::unique_ptr<OffsetExpressionWorkspace> workspace;
    OffsetVector offsets;
    std::vector<size_t> lanes;
};

OffsetExpressionCallback::OffsetExpressionCallback(
    expr::TypedExprPtr expression,
    ExecContext* context,
    int64_t row_count,
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
OffsetExpressionCallback::view(bool memoize) const {
    return {2,
            sizeof(knowhere::CandidateEvaluatorView),
            this,
            CreateWorker,
            DestroyWorker,
            EvalBatch,
            memoize};
}

TargetBitmap
OffsetExpressionCallback::EvalBaseline() const {
    auto query = query_context_->MakeOffsetContext();
    ExecContext context(query.get());
    ExprSet expressions({expression_}, &context, true);
    EvalCtx eval(&context);
    return EvalExprSetOverAllBatches(expressions,
                                     eval,
                                     row_count_,
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
OffsetExpressionCallback::EvalBatch(void* opaque,
                                    const int32_t* ids,
                                    size_t count,
                                    const uint64_t* active,
                                    uint64_t* accepted) noexcept {
    if (opaque == nullptr || accepted == nullptr ||
        (count != 0 && ids == nullptr)) {
        return Status::InvalidArgument;
    }
    const size_t words = count / 64 + (count % 64 != 0);
    const uint64_t tail = count == 0        ? 0
                          : count % 64 == 0 ? ~uint64_t{0}
                                            : (uint64_t{1} << (count % 64)) - 1;
    if (active != nullptr && words != 0 && (active[words - 1] & ~tail) != 0)
        return Status::InvalidArgument;
    std::fill_n(accepted, words, uint64_t{0});
    auto& worker = *static_cast<Worker*>(opaque);
    try {
        checkCancellation(worker.query.get());
        bool all_active = true;
        if (active != nullptr) {
            for (size_t word = 0; word < words; ++word) {
                if (active[word] != (word + 1 == words ? tail : ~uint64_t{0})) {
                    all_active = false;
                    break;
                }
            }
        }
        // Existing Expr kernels use signed int lengths. This is an internal
        // safety window, not a transport cap or 64-row Expr subdivision.
        constexpr size_t window =
            (size_t(std::numeric_limits<int>::max()) / 64) * 64;
        for (size_t begin = 0; begin < count;) {
            const size_t length = std::min(window, count - begin);
            worker.offsets.clear();
            worker.lanes.clear();
            if (all_active) {
                worker.offsets.assign(ids + begin, ids + begin + length);
                for (const auto id : worker.offsets) {
                    AssertInfo(id >= 0 && id < worker.factory->row_count_,
                               "callback offset outside active domain");
                }
            } else {
                for (size_t lane = begin; lane < begin + length; ++lane) {
                    if (((active[lane / 64] >> (lane % 64)) & 1) == 0)
                        continue;
                    AssertInfo(ids[lane] >= 0 &&
                                   ids[lane] < worker.factory->row_count_,
                               "callback offset outside active domain");
                    worker.lanes.push_back(lane);
                    worker.offsets.push_back(ids[lane]);
                }
            }
            if (!worker.offsets.empty()) {
                checkCancellation(worker.query.get());
                const auto result =
                    worker.workspace->EvalOffsets(worker.offsets);
                if (all_active) {
                    const auto* truth =
                        static_cast<const uint8_t*>(result->GetRawData());
                    const auto* known =
                        static_cast<const uint8_t*>(result->GetValidRawData());
                    for (size_t local = 0; local < length; local += 64) {
                        const auto n = static_cast<uint32_t>(
                            std::min<size_t>(64, length - local));
                        accepted[(begin + local) / 64] =
                            ReadPackedLanes(truth + local / 8, n) &
                            ReadPackedLanes(known + local / 8, n) &
                            (n == 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1);
                    }
                } else {
                    TargetBitmapView truth(result->GetRawData(),
                                           result->size());
                    TargetBitmapView known(result->GetValidRawData(),
                                           result->size());
                    for (size_t i = 0; i < worker.offsets.size(); ++i) {
                        if (truth[i] && known[i]) {
                            const auto lane = worker.lanes[i];
                            accepted[lane / 64] |= uint64_t{1} << (lane % 64);
                        }
                    }
                }
            }
            begin += length;
        }
        return Status::Success;
    } catch (...) {
        worker.factory->CaptureException();
        return Status::Failed;
    }
}
}  // namespace milvus::exec
