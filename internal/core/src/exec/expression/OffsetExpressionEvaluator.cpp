// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/OffsetExpressionEvaluator.h"

#include <utility>
#include <vector>

namespace milvus::exec {
namespace {

uint64_t
LaneMask(uint32_t count) noexcept {
    if (count == 0) {
        return 0;
    }
    return count == 64 ? ~uint64_t{0} : (uint64_t{1} << count) - 1;
}

struct CompositeProgram {
    std::vector<std::shared_ptr<const PreparedOffsetExpressionEvaluator>>
        leaves;
    std::vector<OffsetExpressionNode> nodes;
    size_t root{0};
};

struct CompositeWorkspace : OffsetEvalWorkspace {
    std::vector<std::unique_ptr<OffsetEvalWorkspace>> leaf_workspaces;
};

std::unique_ptr<OffsetEvalWorkspace>
CreateCompositeWorkspace(const void* opaque) {
    const auto& program = *static_cast<const CompositeProgram*>(opaque);
    auto workspace = std::make_unique<CompositeWorkspace>();
    workspace->leaf_workspaces.reserve(program.leaves.size());
    for (const auto& leaf : program.leaves) {
        auto leaf_workspace = leaf->CreateWorkspace();
        if (leaf_workspace == nullptr) {
            return nullptr;
        }
        workspace->leaf_workspaces.emplace_back(std::move(leaf_workspace));
    }
    return workspace;
}

int32_t
EvalCompositeNode(const CompositeProgram& program,
                  CompositeWorkspace& workspace,
                  size_t node_index,
                  const int64_t* ids,
                  uint32_t count,
                  uint64_t active,
                  OffsetTruthMask* result) {
    if (node_index >= program.nodes.size()) {
        return -1;
    }
    const auto& node = program.nodes[node_index];
    if (node.type == OffsetExpressionNodeType::Leaf) {
        if (node.left >= program.leaves.size()) {
            return -1;
        }
        return program.leaves[node.left]->EvalBatch(
                   *workspace.leaf_workspaces[node.left],
                   ids,
                   count,
                   active,
                   result) == OffsetEvalStatus::Success
                   ? 0
                   : -1;
    }

    OffsetTruthMask left;
    if (EvalCompositeNode(
            program, workspace, node.left, ids, count, active, &left) != 0) {
        return -1;
    }
    if (node.type == OffsetExpressionNodeType::Not) {
        result->known_mask = left.known_mask & active;
        result->true_mask = (~left.true_mask) & result->known_mask;
        return 0;
    }

    uint64_t right_active = active;
    if (node.type == OffsetExpressionNodeType::And) {
        right_active &= ~(left.known_mask & ~left.true_mask);
    } else if (node.type == OffsetExpressionNodeType::Or) {
        right_active &= ~(left.known_mask & left.true_mask);
    } else {
        return -1;
    }

    OffsetTruthMask right;
    if (EvalCompositeNode(
            program, workspace, node.right, ids, count, right_active, &right) !=
        0) {
        return -1;
    }
    if (node.type == OffsetExpressionNodeType::And) {
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
EvalComposite(const void* opaque,
              OffsetEvalWorkspace& opaque_workspace,
              const int64_t* ids,
              uint32_t count,
              uint64_t active,
              OffsetTruthMask* result) {
    const auto& program = *static_cast<const CompositeProgram*>(opaque);
    auto& workspace = static_cast<CompositeWorkspace&>(opaque_workspace);
    return EvalCompositeNode(
        program, workspace, program.root, ids, count, active, result);
}

}  // namespace

PreparedOffsetExpressionEvaluator::PreparedOffsetExpressionEvaluator(
    std::shared_ptr<const void> prepared_program,
    CreateOffsetEvalWorkspaceFn create_workspace,
    EvalOffsetBatchFn eval_batch)
    : prepared_program_(std::move(prepared_program)),
      create_workspace_(create_workspace),
      eval_batch_(eval_batch) {
}

std::shared_ptr<const PreparedOffsetExpressionEvaluator>
PreparedOffsetExpressionEvaluator::Create(
    std::shared_ptr<const void> prepared_program,
    CreateOffsetEvalWorkspaceFn create_workspace,
    EvalOffsetBatchFn eval_batch) {
    if (prepared_program == nullptr || create_workspace == nullptr ||
        eval_batch == nullptr) {
        return nullptr;
    }
    return std::shared_ptr<const PreparedOffsetExpressionEvaluator>(
        new PreparedOffsetExpressionEvaluator(
            std::move(prepared_program), create_workspace, eval_batch));
}

std::unique_ptr<OffsetEvalWorkspace>
PreparedOffsetExpressionEvaluator::CreateWorkspace() const {
    return create_workspace_(prepared_program_.get());
}

OffsetEvalStatus
PreparedOffsetExpressionEvaluator::EvalBatch(
    OffsetEvalWorkspace& workspace,
    const int64_t* row_ids,
    uint32_t count,
    uint64_t active_mask,
    OffsetTruthMask* result) const noexcept {
    if (result == nullptr) {
        return OffsetEvalStatus::InvalidArgument;
    }
    *result = {};
    if (count > 64 || (count != 0 && row_ids == nullptr)) {
        return OffsetEvalStatus::InvalidArgument;
    }
    const auto lane_mask = LaneMask(count);
    if ((active_mask & ~lane_mask) != 0) {
        return OffsetEvalStatus::InvalidArgument;
    }
    if (count == 0 || active_mask == 0) {
        return OffsetEvalStatus::Success;
    }

    int32_t status = 0;
    try {
        status = eval_batch_(prepared_program_.get(),
                             workspace,
                             row_ids,
                             count,
                             active_mask,
                             result);
    } catch (...) {
        *result = {};
        return OffsetEvalStatus::EvaluationError;
    }
    if (status != 0) {
        *result = {};
        return OffsetEvalStatus::EvaluationError;
    }
    if ((result->true_mask & ~result->known_mask) != 0 ||
        ((result->true_mask | result->known_mask) & ~active_mask) != 0) {
        *result = {};
        return OffsetEvalStatus::InvalidResult;
    }
    return OffsetEvalStatus::Success;
}

std::shared_ptr<const PreparedOffsetExpressionEvaluator>
ComposeOffsetExpressionEvaluators(
    std::vector<std::shared_ptr<const PreparedOffsetExpressionEvaluator>>
        leaves,
    std::vector<OffsetExpressionNode> nodes,
    size_t root) {
    if (leaves.empty() || nodes.empty() || root >= nodes.size()) {
        return nullptr;
    }
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        if (node.type == OffsetExpressionNodeType::Leaf) {
            if (node.left >= leaves.size()) {
                return nullptr;
            }
        } else if (node.left >= i ||
                   (node.type != OffsetExpressionNodeType::Not &&
                    node.right >= i)) {
            return nullptr;
        }
    }
    for (const auto& leaf : leaves) {
        if (leaf == nullptr) {
            return nullptr;
        }
    }
    auto program = std::make_shared<const CompositeProgram>(
        CompositeProgram{std::move(leaves), std::move(nodes), root});
    return PreparedOffsetExpressionEvaluator::Create(
        std::move(program), &CreateCompositeWorkspace, &EvalComposite);
}

}  // namespace milvus::exec
