// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "knowhere/candidate_evaluator.h"

namespace milvus::exec {

struct CandidatePredicateNode;

using CandidateEvaluatorV1 = knowhere::CandidateEvaluatorV1;
using CandidateEvalBatchFn = knowhere::CandidateEvalBatchFn;
using CandidateEvalContiguousFn = knowhere::CandidateEvalContiguousFn;

// Milvus-only leaf/composer callback preserving SQL three-valued truth.
//
// context: immutable Milvus-owned prepared leaf or composer state.
// row_ids: `count` logical segment row IDs.
// count: number of lanes in [0, 64].
// active_mask: input lanes to evaluate; bits >= count must be zero.
// true_mask: output lanes whose expression result is SQL TRUE.
// known_mask: output lanes whose result is not SQL NULL; true_mask must be a
//             subset of known_mask, and both outputs must be subsets of
//             active_mask with no bits set above count.
// return: zero on success, non-zero on failure; exceptions may not cross the
//         noexcept boundary.
using CandidateEvalTruthBatchFn = int32_t (*)(const void* context,
                                              const int64_t* row_ids,
                                              uint32_t count,
                                              uint64_t active_mask,
                                              uint64_t* true_mask,
                                              uint64_t* known_mask) noexcept;

// Milvus-owned executable view. The owner keeps the callback context and all
// referenced scalar source memory alive for the complete query execution.
struct PreparedCandidateEvaluator {
    CandidateEvaluatorV1 view;
    std::shared_ptr<const void> owner;
    // Milvus-only three-valued result used by the logical composer. `true`
    // is always a subset of `known`; unknown lanes model SQL NULL. The public
    // candidate ABI intentionally exposes only root-level accepted lanes.
    CandidateEvalTruthBatchFn eval_truth_batch = nullptr;

    explicit operator bool() const noexcept {
        return owner != nullptr && view.context != nullptr &&
               view.eval_batch != nullptr;
    }
};

std::optional<PreparedCandidateEvaluator>
ComposeCandidateEvaluators(std::vector<PreparedCandidateEvaluator> leaves,
                           const std::vector<CandidatePredicateNode>& nodes,
                           size_t root);

}  // namespace milvus::exec
