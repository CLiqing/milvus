// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "common/Types.h"
#include "exec/expression/CandidateEvaluator.h"
#include "pb/plan.pb.h"

namespace milvus::exec {

// Numeric-owned source layouts. These views never cross the public
// Knowhere/Cardinal ABI. A leaf uses either one contiguous materialized column
// or zero-copy sealed chunks; chunk_offsets has num_chunks + 1 entries.
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

enum class NumericCandidatePredicateOp : uint8_t {
    GreaterEqual,
    GreaterThan,
    LessEqual,
    LessThan,
    Equal,
    NotEqual,
    Range,
    Term,
};

// Comparison/range/TERM descriptors are owned by the Numeric expression
// module.  Each value family contains only its own literals; the generic
// logical composer receives these objects as opaque CandidateLeafPlan state.
struct Int64CandidatePredicate {
    FieldId field_id;
    DataType field_data_type{DataType::NONE};
    NumericCandidatePredicateOp op{NumericCandidatePredicateOp::Equal};
    int64_t arg0{0};
    int64_t arg1{0};
    bool lower_inclusive{true};
    bool upper_inclusive{true};
    std::vector<int64_t> terms;
};

struct FloatCandidatePredicate {
    FieldId field_id;
    DataType field_data_type{DataType::NONE};
    NumericCandidatePredicateOp op{NumericCandidatePredicateOp::Equal};
    float arg0{0.0F};
    float arg1{0.0F};
    bool lower_inclusive{true};
    bool upper_inclusive{true};
    std::vector<float> terms;
};

// Numeric arithmetic descriptors contain only the fields required by their
// own value family. They are Milvus-internal and never cross the public
// Knowhere/Cardinal evaluator ABI.
struct Int64ArithmeticCandidatePredicate {
    FieldId field_id;
    DataType field_data_type{DataType::NONE};
    proto::plan::ArithOpType arithmetic_op;
    int64_t operand{0};
    int64_t target{0};
};

struct FloatArithmeticCandidatePredicate {
    FieldId field_id;
    DataType field_data_type{DataType::NONE};
    proto::plan::ArithOpType arithmetic_op;
    float operand{0.0F};
    float target{0.0F};
};

std::optional<PreparedCandidateEvaluator>
PrepareInt64ModCandidateEvaluator(const Int64CandidateSourceView& source,
                                  int64_t divisor,
                                  int64_t upper_bound);

std::optional<PreparedCandidateEvaluator>
PrepareInt64ArithmeticCandidateEvaluator(
    const Int64CandidateSourceView& source,
    const Int64ArithmeticCandidatePredicate& predicate);

std::optional<PreparedCandidateEvaluator>
PrepareFloatArithmeticCandidateEvaluator(
    const FloatCandidateSourceView& source,
    const FloatArithmeticCandidatePredicate& predicate);

std::optional<PreparedCandidateEvaluator>
PrepareInt64CandidateEvaluator(const Int64CandidateSourceView& source,
                               const Int64CandidatePredicate& predicate);

std::optional<PreparedCandidateEvaluator>
PrepareFloatCandidateEvaluator(const FloatCandidateSourceView& source,
                               const FloatCandidatePredicate& predicate);

}  // namespace milvus::exec
