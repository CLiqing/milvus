// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cachinglayer/CacheSlot.h"
#include "common/Span.h"
#include "exec/expression/NumericCandidateEvaluator.h"

namespace milvus::exec {

// Query-scoped Numeric source owners. Materialized and pinned-chunk layouts
// are deliberately distinct so a leaf retains only the representation it
// actually uses.
struct Int64MaterializedCandidateSourceOwner {
    std::shared_ptr<std::vector<int64_t>> materialized_values;

    Int64CandidateSourceView
    view(size_t row_count) const noexcept {
        return {
            .row_values = materialized_values == nullptr
                              ? nullptr
                              : materialized_values->data(),
            .row_count = row_count,
        };
    }
};

struct Int64ChunkedCandidateSourceOwner {
    std::vector<milvus::cachinglayer::PinWrapper<Span<int64_t>>> pins;
    std::vector<const int64_t*> chunk_values;
    std::vector<int64_t> chunk_offsets;

    Int64CandidateSourceView
    view(size_t row_count) const noexcept {
        return {
            .row_count = row_count,
            .chunk_values =
                chunk_values.empty() ? nullptr : chunk_values.data(),
            .chunk_offsets =
                chunk_offsets.empty() ? nullptr : chunk_offsets.data(),
            .num_chunks = chunk_values.size(),
        };
    }
};

struct FloatMaterializedCandidateSourceOwner {
    std::shared_ptr<std::vector<float>> materialized_values;

    FloatCandidateSourceView
    view(size_t row_count) const noexcept {
        return {
            .row_values = materialized_values == nullptr
                              ? nullptr
                              : materialized_values->data(),
            .row_count = row_count,
        };
    }
};

struct FloatChunkedCandidateSourceOwner {
    std::vector<milvus::cachinglayer::PinWrapper<Span<float>>> pins;
    std::vector<const float*> chunk_values;
    std::vector<int64_t> chunk_offsets;

    FloatCandidateSourceView
    view(size_t row_count) const noexcept {
        return {
            .row_count = row_count,
            .chunk_values =
                chunk_values.empty() ? nullptr : chunk_values.data(),
            .chunk_offsets =
                chunk_offsets.empty() ? nullptr : chunk_offsets.data(),
            .num_chunks = chunk_values.size(),
        };
    }
};

}  // namespace milvus::exec
