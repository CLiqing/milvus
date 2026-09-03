// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "exec/expression/CandidateEvaluator.h"

namespace milvus {
class OpContext;
namespace segcore {
class SegmentInternalInterface;
}
namespace exec {

enum class CandidatePredicateNodeType : uint8_t {
    Leaf,
    And,
    Or,
    Not,
};

struct CandidatePredicateNode {
    CandidatePredicateNodeType type{CandidatePredicateNodeType::Leaf};
    size_t left{0};
    size_t right{0};
};

// Opaque, query-scoped owner for every pin and converted value buffer used by
// a fused scalar predicate. Preparing this context is part of the FilterBits
// eligibility decision, so an advisory hint can always fall back before the
// normal bitmap is skipped.
struct PreparedFusingBundle;

// Type-erased Milvus-internal leaf plan. The owning expression module keeps
// its typed immutable state and supplies the preparation callback; the logical
// composer never needs to know the value type, opcode, or literal layout.
struct PreparedCandidateLeaf {
    PreparedCandidateEvaluator evaluator;
    // Populated only after this leaf has migrated to the Milvus shared offset
    // evaluator. Unmigrated expression families keep using evaluator above.
    std::shared_ptr<const PreparedOffsetExpressionEvaluator> offset_evaluator;
    std::vector<std::shared_ptr<const void>> resource_owners;
};

using PrepareCandidateLeafFn = std::optional<PreparedCandidateLeaf> (*)(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    const void* typed_state);

struct CandidateLeafPlan {
    std::shared_ptr<const void> typed_state;
    PrepareCandidateLeafFn prepare = nullptr;

    explicit operator bool() const noexcept {
        return typed_state != nullptr && prepare != nullptr;
    }
};

// Milvus-internal logical program. Logical nodes compose opaque prepared
// leaves; neither the program nor lower layers inspect typed leaf state.
struct AnnFilterFusingProgram {
    std::vector<CandidateLeafPlan> leaves;
    std::vector<CandidatePredicateNode> nodes;
    size_t root{0};
    int64_t estimated_filtered_out_count{0};
};

std::shared_ptr<PreparedFusingBundle>
PrepareAnnFilterFusingBundle(const segcore::SegmentInternalInterface* segment,
                             OpContext* op_context,
                             const AnnFilterFusingProgram& program);

const char*
AnnFilterFusingSourceName(const PreparedFusingBundle& context);

}  // namespace exec
}  // namespace milvus
