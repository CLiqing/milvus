// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/Types.h"
#include "exec/expression/CandidateEvaluator.h"

namespace milvus::exec {

// String-owned source contract. Raw chunks preserve zero-copy access to the
// sealed VARCHAR column. Dictionary IDs are an EQ/NE specialization and are
// intentionally absent from the generic evaluator ABI.
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

enum class StringCandidateComparisonOp : uint8_t {
    GreaterEqual,
    GreaterThan,
    LessEqual,
    LessThan,
    Equal,
    NotEqual,
};

struct StringComparisonCandidatePredicate {
    FieldId field_id;
    DataType field_data_type{DataType::NONE};
    bool nullable{false};
    StringCandidateComparisonOp op{StringCandidateComparisonOp::Equal};
    std::string value;
};

struct StringRangeCandidatePredicate {
    FieldId field_id;
    DataType field_data_type{DataType::NONE};
    bool nullable{false};
    std::string lower;
    std::string upper;
    bool lower_inclusive{true};
    bool upper_inclusive{true};
};

struct StringTermCandidatePredicate {
    FieldId field_id;
    DataType field_data_type{DataType::NONE};
    bool nullable{false};
    std::vector<std::string> terms;
};

struct StringLikeCandidatePredicate {
    FieldId field_id;
    DataType field_data_type{DataType::NONE};
    bool nullable{false};
    std::string pattern;
};

using StringCandidatePredicate =
    std::variant<StringComparisonCandidatePredicate,
                 StringRangeCandidatePredicate,
                 StringTermCandidatePredicate,
                 StringLikeCandidatePredicate>;

std::optional<PreparedCandidateEvaluator>
PrepareStringCandidateEvaluator(const StringCandidateSourceView& source,
                                const StringCandidatePredicate& predicate);

}  // namespace milvus::exec
