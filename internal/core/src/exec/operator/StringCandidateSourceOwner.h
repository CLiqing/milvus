// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cachinglayer/CacheSlot.h"
#include "common/Chunk.h"
#include "exec/expression/StringCandidateEvaluator.h"
#include "index/Index.h"

namespace milvus::exec {

struct RawStringCandidateSourceOwner {
    std::vector<milvus::cachinglayer::PinWrapper<RawStringChunkView>> pins;
    std::vector<const char*> chunk_bases;
    std::vector<const uint32_t*> chunk_value_offsets;
    std::vector<const bool*> chunk_valid_data;
    std::vector<size_t> chunk_row_counts;
    std::vector<int64_t> chunk_row_offsets;
    size_t uniform_chunk_rows = 0;

    StringCandidateSourceView
    view(size_t row_count) const noexcept {
        return {
            .chunk_bases = chunk_bases.empty() ? nullptr : chunk_bases.data(),
            .chunk_value_offsets = chunk_value_offsets.empty()
                                       ? nullptr
                                       : chunk_value_offsets.data(),
            .chunk_valid_data = chunk_valid_data.empty()
                                    ? nullptr
                                    : chunk_valid_data.data(),
            .chunk_row_counts = chunk_row_counts.empty()
                                    ? nullptr
                                    : chunk_row_counts.data(),
            .chunk_row_offsets = chunk_row_offsets.empty()
                                     ? nullptr
                                     : chunk_row_offsets.data(),
            .num_chunks = pins.size(),
            .row_count = row_count,
            .uniform_chunk_rows = uniform_chunk_rows,
        };
    }
};

struct StringDictionaryCandidateSourceOwner {
    std::vector<milvus::cachinglayer::PinWrapper<const index::IndexBase*>>
        index_pins;
    const int32_t* row_dictionary_ids = nullptr;
    size_t row_count = 0;
    int32_t target_dictionary_id = -1;
    bool target_dictionary_id_found = false;

    StringCandidateSourceView
    view() const noexcept {
        return {
            .row_count = row_count,
            .row_dictionary_ids = row_dictionary_ids,
            .target_dictionary_id = target_dictionary_id,
            .target_dictionary_id_found = target_dictionary_id_found,
        };
    }
};

}  // namespace milvus::exec
