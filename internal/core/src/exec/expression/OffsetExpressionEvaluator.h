// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace milvus::exec {

enum class OffsetEvalStatus : int32_t {
    Success = 0,
    InvalidArgument = 1,
    EvaluationError = 2,
    InvalidResult = 3,
};

// SQL three-valued truth for at most 64 candidate lanes. A TRUE lane is also
// known; a lane absent from known_mask is SQL NULL. Consumers accept only
// true_mask & known_mask at the root of the expression.
struct OffsetTruthMask {
    uint64_t true_mask{0};
    uint64_t known_mask{0};
};

// Mutable scratch belongs to one search worker (or one serial iterator). It is
// never shared between concurrent EvalBatch calls. Concrete expression
// modules derive from this type to retain reusable gather/result buffers.
class OffsetEvalWorkspace {
 public:
    virtual ~OffsetEvalWorkspace() = default;
};

using CreateOffsetEvalWorkspaceFn =
    std::unique_ptr<OffsetEvalWorkspace> (*)(const void* prepared_program);

// Evaluate logical segment row IDs selected by active_mask. Implementations
// may mutate only workspace; prepared_program is query-scoped immutable state.
// Throwing implementations are contained by PreparedOffsetExpressionEvaluator
// and reported as EvaluationError.
using EvalOffsetBatchFn = int32_t (*)(const void* prepared_program,
                                      OffsetEvalWorkspace& workspace,
                                      const int64_t* row_ids,
                                      uint32_t count,
                                      uint64_t active_mask,
                                      OffsetTruthMask* result);

// Query-scoped immutable offset evaluator shared by iterative filter and ANN
// filter fusing. It owns the prepared expression program and source pins; each
// concurrent consumer must create and retain its own workspace.
class PreparedOffsetExpressionEvaluator final {
 public:
    static std::shared_ptr<const PreparedOffsetExpressionEvaluator>
    Create(std::shared_ptr<const void> prepared_program,
           CreateOffsetEvalWorkspaceFn create_workspace,
           EvalOffsetBatchFn eval_batch);

    std::unique_ptr<OffsetEvalWorkspace>
    CreateWorkspace() const;

    OffsetEvalStatus
    EvalBatch(OffsetEvalWorkspace& workspace,
              const int64_t* row_ids,
              uint32_t count,
              uint64_t active_mask,
              OffsetTruthMask* result) const noexcept;

 private:
    PreparedOffsetExpressionEvaluator(
        std::shared_ptr<const void> prepared_program,
        CreateOffsetEvalWorkspaceFn create_workspace,
        EvalOffsetBatchFn eval_batch);

    std::shared_ptr<const void> prepared_program_;
    CreateOffsetEvalWorkspaceFn create_workspace_{nullptr};
    EvalOffsetBatchFn eval_batch_{nullptr};
};

enum class OffsetExpressionNodeType : uint8_t {
    Leaf,
    Not,
    And,
    Or,
};

// Post-order expression node. Leaf.left is an index into the evaluator list;
// logical nodes refer only to earlier node indices.
struct OffsetExpressionNode {
    OffsetExpressionNodeType type{OffsetExpressionNodeType::Leaf};
    size_t left{0};
    size_t right{0};
};

// Builds an immutable SQL three-valued logical composer around prepared leaf
// evaluators. Each composer workspace owns one workspace per leaf.
std::shared_ptr<const PreparedOffsetExpressionEvaluator>
ComposeOffsetExpressionEvaluators(
    std::vector<std::shared_ptr<const PreparedOffsetExpressionEvaluator>>
        leaves,
    std::vector<OffsetExpressionNode> nodes,
    size_t root);

}  // namespace milvus::exec
