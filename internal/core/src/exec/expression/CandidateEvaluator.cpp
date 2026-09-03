// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/CandidateEvaluator.h"

#include <array>
#include <utility>

#include "exec/expression/OffsetExpressionEvaluator.h"
#include "exec/operator/DownpushSearchContext.h"

namespace milvus::exec {
namespace {

struct TruthMask {
    uint64_t true_mask{0};
    uint64_t known_mask{0};
};

struct CompositeEvaluatorState {
    std::vector<PreparedCandidateEvaluator> leaves;
    std::vector<CandidatePredicateNode> nodes;
    size_t root{0};
};

struct OffsetEvaluatorAdapterState {
    std::shared_ptr<const PreparedOffsetExpressionEvaluator> evaluator;
};

uint64_t
LaneMask(uint32_t count) noexcept {
    return count == 64
               ? ~uint64_t{0}
               : (count == 0 ? uint64_t{0} : ((uint64_t{1} << count) - 1));
}

int32_t
EvalOffsetWithWorkspace(const void* opaque,
                        void* opaque_workspace,
                        const int64_t* ids,
                        uint32_t count,
                        uint64_t active_mask,
                        uint64_t* accepted_mask) noexcept {
    if (opaque == nullptr || opaque_workspace == nullptr ||
        accepted_mask == nullptr) {
        return -1;
    }
    const auto& state =
        *static_cast<const OffsetEvaluatorAdapterState*>(opaque);
    auto& workspace = *static_cast<OffsetEvalWorkspace*>(opaque_workspace);
    OffsetTruthMask truth;
    const auto status =
        state.evaluator->EvalBatch(workspace, ids, count, active_mask, &truth);
    if (status != OffsetEvalStatus::Success) {
        *accepted_mask = 0;
        return -1;
    }
    *accepted_mask = truth.true_mask & truth.known_mask;
    return 0;
}

int32_t
EvalOffsetContiguousWithWorkspace(const void* opaque,
                                  void* opaque_workspace,
                                  int64_t first_id,
                                  uint32_t count,
                                  uint64_t active_mask,
                                  uint64_t* accepted_mask) noexcept {
    if (first_id < 0 || count > 64) {
        return -1;
    }
    std::array<int64_t, 64> ids{};
    for (uint32_t lane = 0; lane < count; ++lane) {
        ids[lane] = first_id + static_cast<int64_t>(lane);
    }
    return EvalOffsetWithWorkspace(opaque,
                                   opaque_workspace,
                                   ids.data(),
                                   count,
                                   active_mask,
                                   accepted_mask);
}

void*
CreateOffsetWorkspace(const void* opaque) noexcept {
    if (opaque == nullptr) {
        return nullptr;
    }
    try {
        const auto& state =
            *static_cast<const OffsetEvaluatorAdapterState*>(opaque);
        return state.evaluator->CreateWorkspace().release();
    } catch (...) {
        return nullptr;
    }
}

void
ReleaseOffsetWorkspace(void* workspace) noexcept {
    delete static_cast<OffsetEvalWorkspace*>(workspace);
}

// Compatibility callbacks are not used by workspace-aware Cardinal. They
// keep generic Knowhere consumers safe and correct by creating one temporary
// workspace for an isolated call.
int32_t
EvalOffsetCompatibility(const void* opaque,
                        const int64_t* ids,
                        uint32_t count,
                        uint64_t active_mask,
                        uint64_t* accepted_mask) noexcept {
    std::unique_ptr<OffsetEvalWorkspace> workspace(
        static_cast<OffsetEvalWorkspace*>(CreateOffsetWorkspace(opaque)));
    if (workspace == nullptr) {
        return -1;
    }
    return EvalOffsetWithWorkspace(
        opaque, workspace.get(), ids, count, active_mask, accepted_mask);
}

int32_t
EvalOffsetContiguousCompatibility(const void* opaque,
                                  int64_t first_id,
                                  uint32_t count,
                                  uint64_t active_mask,
                                  uint64_t* accepted_mask) noexcept {
    std::unique_ptr<OffsetEvalWorkspace> workspace(
        static_cast<OffsetEvalWorkspace*>(CreateOffsetWorkspace(opaque)));
    if (workspace == nullptr) {
        return -1;
    }
    return EvalOffsetContiguousWithWorkspace(
        opaque, workspace.get(), first_id, count, active_mask, accepted_mask);
}

int32_t
EvalLeaf(const PreparedCandidateEvaluator& leaf,
         const int64_t* ids,
         uint32_t count,
         uint64_t active,
         TruthMask* result) noexcept {
    if (leaf.eval_truth_batch != nullptr) {
        return leaf.eval_truth_batch(leaf.view.context,
                                     ids,
                                     count,
                                     active,
                                     &result->true_mask,
                                     &result->known_mask);
    }
    uint64_t accepted = 0;
    const auto status =
        leaf.view.eval_batch(leaf.view.context, ids, count, active, &accepted);
    if (status == 0) {
        result->true_mask = accepted & active;
        result->known_mask = active;
    }
    return status;
}

int32_t
EvalNode(const CompositeEvaluatorState& state,
         size_t node_index,
         const int64_t* ids,
         uint32_t count,
         uint64_t active,
         TruthMask* result) noexcept {
    if (node_index >= state.nodes.size() || result == nullptr) {
        return -1;
    }
    const auto& node = state.nodes[node_index];
    if (node.type == CandidatePredicateNodeType::Leaf) {
        if (node.left >= state.leaves.size()) {
            return -1;
        }
        return EvalLeaf(state.leaves[node.left], ids, count, active, result);
    }

    TruthMask left;
    if (EvalNode(state, node.left, ids, count, active, &left) != 0) {
        return -1;
    }
    if (node.type == CandidatePredicateNodeType::Not) {
        result->known_mask = left.known_mask & active;
        result->true_mask = (~left.true_mask) & result->known_mask;
        return 0;
    }

    uint64_t right_active = active;
    if (node.type == CandidatePredicateNodeType::And) {
        const auto definitely_false = left.known_mask & ~left.true_mask;
        right_active &= ~definitely_false;
    } else if (node.type == CandidatePredicateNodeType::Or) {
        const auto definitely_true = left.known_mask & left.true_mask;
        right_active &= ~definitely_true;
    } else {
        return -1;
    }

    TruthMask right;
    if (EvalNode(state, node.right, ids, count, right_active, &right) != 0) {
        return -1;
    }
    if (node.type == CandidatePredicateNodeType::And) {
        result->true_mask = left.true_mask & right.true_mask;
        result->known_mask = (left.known_mask & ~left.true_mask) |
                             (right.known_mask & ~right.true_mask) |
                             (left.known_mask & right.known_mask);
    } else {
        result->true_mask = left.true_mask | right.true_mask;
        result->known_mask = (left.known_mask & left.true_mask) |
                             (right.known_mask & right.true_mask) |
                             (left.known_mask & right.known_mask);
    }
    result->true_mask &= active;
    result->known_mask &= active;
    return 0;
}

int32_t
EvaluateCompositeTruth(const void* opaque,
                       const int64_t* ids,
                       uint32_t count,
                       uint64_t active_mask,
                       uint64_t* true_mask,
                       uint64_t* known_mask) noexcept {
    if (opaque == nullptr || ids == nullptr || true_mask == nullptr ||
        known_mask == nullptr || count > 64) {
        return -1;
    }
    const auto& state = *static_cast<const CompositeEvaluatorState*>(opaque);
    TruthMask result;
    const auto status = EvalNode(
        state, state.root, ids, count, active_mask & LaneMask(count), &result);
    if (status == 0) {
        *true_mask = result.true_mask;
        *known_mask = result.known_mask;
    }
    return status;
}

int32_t
EvaluateComposite(const void* opaque,
                  const int64_t* ids,
                  uint32_t count,
                  uint64_t active_mask,
                  uint64_t* accepted_mask) noexcept {
    uint64_t known = 0;
    return EvaluateCompositeTruth(
        opaque, ids, count, active_mask, accepted_mask, &known);
}

int32_t
EvaluateCompositeContiguous(const void* opaque,
                            int64_t first_id,
                            uint32_t count,
                            uint64_t active_mask,
                            uint64_t* accepted_mask) noexcept {
    if (first_id < 0 || count > 64) {
        return -1;
    }
    std::array<int64_t, 64> ids{};
    for (uint32_t lane = 0; lane < count; ++lane) {
        ids[lane] = first_id + static_cast<int64_t>(lane);
    }
    return EvaluateComposite(
        opaque, ids.data(), count, active_mask, accepted_mask);
}

}  // namespace

std::optional<PreparedCandidateEvaluator>
ComposeCandidateEvaluators(std::vector<PreparedCandidateEvaluator> leaves,
                           const std::vector<CandidatePredicateNode>& nodes,
                           size_t root) {
    if (leaves.empty() || nodes.empty() || root >= nodes.size()) {
        return std::nullopt;
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        if (node.type == CandidatePredicateNodeType::Leaf) {
            if (node.left >= leaves.size()) {
                return std::nullopt;
            }
        } else if (node.left >= i ||
                   (node.type != CandidatePredicateNodeType::Not &&
                    node.right >= i)) {
            return std::nullopt;
        }
    }
    for (const auto& leaf : leaves) {
        if (!leaf) {
            return std::nullopt;
        }
    }
    auto owner = std::make_shared<CompositeEvaluatorState>(
        CompositeEvaluatorState{std::move(leaves), nodes, root});
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    prepared.view.eval_batch = &EvaluateComposite;
    prepared.view.eval_contiguous = &EvaluateCompositeContiguous;
    prepared.eval_truth_batch = &EvaluateCompositeTruth;
    return prepared;
}

std::optional<PreparedCandidateEvaluator>
AdaptOffsetExpressionEvaluator(
    std::shared_ptr<const PreparedOffsetExpressionEvaluator> evaluator) {
    if (evaluator == nullptr) {
        return std::nullopt;
    }
    auto owner = std::make_shared<OffsetEvaluatorAdapterState>(
        OffsetEvaluatorAdapterState{std::move(evaluator)});
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    prepared.view.eval_batch = &EvalOffsetCompatibility;
    prepared.view.eval_contiguous = &EvalOffsetContiguousCompatibility;
    prepared.view.abi_capabilities |=
        knowhere::kCandidateEvaluatorCapabilityWorkerWorkspace;
    prepared.view.create_workspace = &CreateOffsetWorkspace;
    prepared.view.release_workspace = &ReleaseOffsetWorkspace;
    prepared.view.eval_batch_with_workspace = &EvalOffsetWithWorkspace;
    prepared.view.eval_contiguous_with_workspace =
        &EvalOffsetContiguousWithWorkspace;
    return prepared;
}

}  // namespace milvus::exec
