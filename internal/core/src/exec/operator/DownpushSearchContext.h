// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information.
// Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>

#include "common/Downpush.h"

namespace milvus {
class OpContext;
namespace segcore {
class SegmentInternalInterface;
}
namespace exec {

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

const char*
CardinalDownpushSourceName(const CardinalDownpushSearchContext& context);

}  // namespace exec
}  // namespace milvus
