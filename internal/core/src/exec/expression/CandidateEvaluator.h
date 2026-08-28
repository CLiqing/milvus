// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace milvus {
struct CardinalDownpushPredicate;
}

namespace milvus::exec {

inline constexpr uint32_t kCandidateEvaluatorAbiMajor = 1;

using CandidateEvalBatchFn = int32_t (*)(
    const void*, const int64_t*, uint32_t, uint64_t, uint64_t*
) noexcept;
using CandidateEvalContiguousFn = int32_t (*)(
    const void*, int64_t, uint32_t, uint64_t, uint64_t*
) noexcept;

// Milvus-owned executable view. The owner keeps the callback context and all
// referenced scalar source memory alive for the complete query execution.
struct CandidateEvaluatorV1 {
    uint32_t abi_major = kCandidateEvaluatorAbiMajor;
    uint32_t struct_size = sizeof(CandidateEvaluatorV1);
    uint64_t abi_capabilities = 0;
    const void* context = nullptr;
    CandidateEvalBatchFn eval_batch = nullptr;
    CandidateEvalContiguousFn eval_contiguous = nullptr;
};

struct PreparedCandidateEvaluator {
    CandidateEvaluatorV1 view;
    std::shared_ptr<const void> owner;

    explicit operator bool() const noexcept {
        return owner != nullptr && view.context != nullptr &&
               view.eval_batch != nullptr;
    }
};

// Internal Numeric-module source contract. It never crosses into Knowhere or
// Cardinal; it lets the Numeric evaluator retain the existing zero-copy
// contiguous/chunked source layout without placing source details in the
// public candidate-evaluator ABI.
struct Int64CandidateSourceView {
    const int64_t* row_values = nullptr;
    size_t row_count = 0;
    const int64_t* const* chunk_values = nullptr;
    const int64_t* chunk_offsets = nullptr;
    size_t num_chunks = 0;
};

struct FloatCandidateSourceView {
    const float* row_values = nullptr;
    size_t row_count = 0;
    const float* const* chunk_values = nullptr;
    const int64_t* chunk_offsets = nullptr;
    size_t num_chunks = 0;
};

// String-module source contract. Raw chunks preserve zero-copy access to the
// sealed string column. Dictionary IDs are an optional EQ/NE specialization;
// neither representation crosses the generic Knowhere/Cardinal ABI.
struct StringCandidateSourceView {
    const char* const* chunk_bases = nullptr;
    const uint32_t* const* chunk_value_offsets = nullptr;
    const bool* const* chunk_valid_data = nullptr;
    const size_t* chunk_row_counts = nullptr;
    const int64_t* chunk_row_offsets = nullptr;
    size_t num_chunks = 0;
    size_t row_count = 0;
    size_t uniform_chunk_rows = 0;
    const int32_t* row_dictionary_ids = nullptr;
    int32_t target_dictionary_id = -1;
    bool target_dictionary_id_found = false;
};

std::optional<PreparedCandidateEvaluator>
PrepareInt64ModCandidateEvaluator(const Int64CandidateSourceView& source,
                                  int64_t divisor,
                                  int64_t upper_bound);

// Numeric-owned lowering entry point.  It deliberately accepts Milvus'
// internal predicate description and returns only the generic evaluator ABI;
// no numeric opcode or literal crosses into Knowhere/Cardinal.
std::optional<PreparedCandidateEvaluator>
PrepareInt64CandidateEvaluator(const Int64CandidateSourceView& source,
                               const CardinalDownpushPredicate& predicate);

std::optional<PreparedCandidateEvaluator>
PrepareFloatCandidateEvaluator(const FloatCandidateSourceView& source,
                               const CardinalDownpushPredicate& predicate);

std::optional<PreparedCandidateEvaluator>
PrepareStringCandidateEvaluator(const StringCandidateSourceView& source,
                                const CardinalDownpushPredicate& predicate);

}  // namespace milvus::exec
