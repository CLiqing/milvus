// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "common/Downpush.h"

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

// Milvus-internal logical program. Leaves retain the existing type-owned
// predicate descriptions; logical nodes only compose their prepared batch
// results. This program never crosses into Knowhere or Cardinal.
struct CardinalDownpushPredicateProgram {
    std::vector<CardinalDownpushPredicate> leaves;
    std::vector<bool> leaf_nullable;
    std::vector<CandidatePredicateNode> nodes;
    size_t root{0};
    int64_t estimated_filtered_out_count{0};
};

// Opaque, query-scoped owner for every pin and converted value buffer used by
// a fused scalar predicate. Preparing this context is part of the FilterBits
// eligibility decision, so an advisory hint can always fall back before the
// normal bitmap is skipped.
struct CardinalDownpushSearchContext;

std::shared_ptr<CardinalDownpushSearchContext>
PrepareCardinalDownpushSearchContext(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    const CardinalDownpushPredicate& predicate);

std::shared_ptr<CardinalDownpushSearchContext>
PrepareCardinalDownpushSearchContext(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    const CardinalDownpushPredicateProgram& program);

const char*
CardinalDownpushSourceName(const CardinalDownpushSearchContext& context);

}  // namespace exec
}  // namespace milvus
