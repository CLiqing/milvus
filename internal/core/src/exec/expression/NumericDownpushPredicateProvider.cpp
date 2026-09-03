// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/DownpushPredicateProvider.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

#include "exec/expression/NumericCandidateEvaluator.h"
#include "exec/expression/BinaryArithOpEvalRangeExprUtils.h"
#include "exec/expression/UnaryPredicateUtils.h"
#include "exec/operator/NumericCandidateSourceOwner.h"
#include "segcore/SegmentInterface.h"

namespace milvus::exec {
namespace {

enum class Int64EvaluatorOp : uint8_t {
    GreaterEqual,
    ModLessThan,
    GreaterThan,
    LessEqual,
    LessThan,
    Equal,
    NotEqual,
    Range,
    Term,
    AddLessThan,
    SubLessThan,
    MulLessThan,
    DivLessThan,
};

struct Int64EvaluatorState {
    Int64CandidateSourceView source;
    Int64EvaluatorOp op = Int64EvaluatorOp::Equal;
    int64_t arg0 = 0;
    int64_t arg1 = 0;
    bool lower_inclusive = true;
    bool upper_inclusive = true;
    std::vector<int64_t> terms;
};

struct FloatEvaluatorState {
    FloatCandidateSourceView source;
    Int64EvaluatorOp op = Int64EvaluatorOp::Equal;
    float arg0 = 0.0F;
    float arg1 = 0.0F;
    bool lower_inclusive = true;
    bool upper_inclusive = true;
    std::vector<float> terms;
};

struct NumericOffsetWorkspace : OffsetEvalWorkspace {};

struct NumericOffsetProgram {
    PreparedCandidateLeaf leaf;
};

std::unique_ptr<OffsetEvalWorkspace>
CreateNumericOffsetWorkspace(const void*) {
    return std::make_unique<NumericOffsetWorkspace>();
}

int32_t
EvaluateNumericOffsetBatch(const void* opaque,
                           OffsetEvalWorkspace&,
                           const int64_t* row_ids,
                           uint32_t count,
                           uint64_t active_mask,
                           OffsetTruthMask* result) {
    const auto& program = *static_cast<const NumericOffsetProgram*>(opaque);
    const auto& evaluator = program.leaf.evaluator;
    if (evaluator.eval_truth_batch != nullptr) {
        return evaluator.eval_truth_batch(evaluator.view.context,
                                          row_ids,
                                          count,
                                          active_mask,
                                          &result->true_mask,
                                          &result->known_mask);
    }
    const auto status = evaluator.view.eval_batch(evaluator.view.context,
                                                  row_ids,
                                                  count,
                                                  active_mask,
                                                  &result->true_mask);
    if (status == 0) {
        result->known_mask = active_mask;
    }
    return status;
}

const int64_t*
ResolveInt64CandidateValue(const Int64CandidateSourceView& source,
                           int64_t row_id) noexcept {
    if (row_id < 0 || static_cast<size_t>(row_id) >= source.row_count) {
        return nullptr;
    }
    if (source.row_values != nullptr) {
        return source.row_values + row_id;
    }
    if (source.chunk_values == nullptr || source.chunk_offsets == nullptr ||
        source.num_chunks == 0) {
        return nullptr;
    }
    if (source.num_chunks == 1) {
        return source.chunk_values[0] == nullptr
                   ? nullptr
                   : source.chunk_values[0] + row_id;
    }
    const auto* upper =
        std::upper_bound(source.chunk_offsets,
                         source.chunk_offsets + source.num_chunks + 1,
                         row_id);
    if (upper == source.chunk_offsets) {
        return nullptr;
    }
    const auto chunk_id =
        static_cast<size_t>((upper - source.chunk_offsets) - 1);
    if (chunk_id >= source.num_chunks ||
        source.chunk_values[chunk_id] == nullptr) {
        return nullptr;
    }
    return source.chunk_values[chunk_id] +
           (row_id - source.chunk_offsets[chunk_id]);
}

const float*
ResolveFloatCandidateValue(const FloatCandidateSourceView& source,
                           int64_t row_id) noexcept {
    if (row_id < 0 || static_cast<size_t>(row_id) >= source.row_count) {
        return nullptr;
    }
    if (source.row_values != nullptr) {
        return source.row_values + row_id;
    }
    if (source.chunk_values == nullptr || source.chunk_offsets == nullptr ||
        source.num_chunks == 0) {
        return nullptr;
    }
    if (source.num_chunks == 1) {
        return source.chunk_values[0] == nullptr
                   ? nullptr
                   : source.chunk_values[0] + row_id;
    }
    const auto* upper =
        std::upper_bound(source.chunk_offsets,
                         source.chunk_offsets + source.num_chunks + 1,
                         row_id);
    if (upper == source.chunk_offsets) {
        return nullptr;
    }
    const auto chunk_id =
        static_cast<size_t>((upper - source.chunk_offsets) - 1);
    if (chunk_id >= source.num_chunks ||
        source.chunk_values[chunk_id] == nullptr) {
        return nullptr;
    }
    return source.chunk_values[chunk_id] +
           (row_id - source.chunk_offsets[chunk_id]);
}

template <Int64EvaluatorOp Op>
bool
EvaluateInt64ValueSpecialized(const Int64EvaluatorState& state,
                              int64_t value) noexcept {
    if constexpr (Op == Int64EvaluatorOp::GreaterEqual) {
        return EvaluateUnaryPredicate<proto::plan::OpType::GreaterEqual>(
            value, state.arg0);
    } else if constexpr (Op == Int64EvaluatorOp::ModLessThan) {
        return EvaluateNumericArithmeticLessThan<
            proto::plan::ArithOpType::Mod>(value, state.arg0, state.arg1);
    } else if constexpr (Op == Int64EvaluatorOp::GreaterThan) {
        return EvaluateUnaryPredicate<proto::plan::OpType::GreaterThan>(
            value, state.arg0);
    } else if constexpr (Op == Int64EvaluatorOp::LessEqual) {
        return EvaluateUnaryPredicate<proto::plan::OpType::LessEqual>(
            value, state.arg0);
    } else if constexpr (Op == Int64EvaluatorOp::LessThan) {
        return EvaluateUnaryPredicate<proto::plan::OpType::LessThan>(
            value, state.arg0);
    } else if constexpr (Op == Int64EvaluatorOp::Equal) {
        return EvaluateUnaryPredicate<proto::plan::OpType::Equal>(value,
                                                                  state.arg0);
    } else if constexpr (Op == Int64EvaluatorOp::NotEqual) {
        return EvaluateUnaryPredicate<proto::plan::OpType::NotEqual>(
            value, state.arg0);
    } else if constexpr (Op == Int64EvaluatorOp::Range) {
        const bool lower_ok =
            state.lower_inclusive ? value >= state.arg0 : value > state.arg0;
        const bool upper_ok =
            state.upper_inclusive ? value <= state.arg1 : value < state.arg1;
        return lower_ok && upper_ok;
    } else if constexpr (Op == Int64EvaluatorOp::Term) {
        return std::binary_search(
            state.terms.begin(), state.terms.end(), value);
    } else if constexpr (Op == Int64EvaluatorOp::AddLessThan) {
        return EvaluateNumericArithmeticLessThan<
            proto::plan::ArithOpType::Add>(value, state.arg0, state.arg1);
    } else if constexpr (Op == Int64EvaluatorOp::SubLessThan) {
        return EvaluateNumericArithmeticLessThan<
            proto::plan::ArithOpType::Sub>(value, state.arg0, state.arg1);
    } else if constexpr (Op == Int64EvaluatorOp::MulLessThan) {
        return EvaluateNumericArithmeticLessThan<
            proto::plan::ArithOpType::Mul>(value, state.arg0, state.arg1);
    } else if constexpr (Op == Int64EvaluatorOp::DivLessThan) {
        return EvaluateNumericArithmeticLessThan<
            proto::plan::ArithOpType::Div>(value, state.arg0, state.arg1);
    }
    return false;
}

bool
EvaluateFloatValue(const FloatEvaluatorState& state, float raw_value) noexcept {
    const float value = raw_value;
    switch (state.op) {
        case Int64EvaluatorOp::GreaterEqual:
            return value >= state.arg0;
        case Int64EvaluatorOp::GreaterThan:
            return value > state.arg0;
        case Int64EvaluatorOp::LessEqual:
            return value <= state.arg0;
        case Int64EvaluatorOp::LessThan:
            return value < state.arg0;
        case Int64EvaluatorOp::Equal:
            return value == state.arg0;
        case Int64EvaluatorOp::NotEqual:
            return value != state.arg0;
        case Int64EvaluatorOp::Range: {
            const bool lower_ok = state.lower_inclusive ? value >= state.arg0
                                                        : value > state.arg0;
            const bool upper_ok = state.upper_inclusive ? value <= state.arg1
                                                        : value < state.arg1;
            return lower_ok && upper_ok;
        }
        case Int64EvaluatorOp::Term:
            return !std::isnan(value) &&
                   std::binary_search(
                       state.terms.begin(), state.terms.end(), value);
        case Int64EvaluatorOp::AddLessThan:
            return EvaluateNumericArithmeticLessThan<
                proto::plan::ArithOpType::Add>(value, state.arg0, state.arg1);
        case Int64EvaluatorOp::SubLessThan:
            return EvaluateNumericArithmeticLessThan<
                proto::plan::ArithOpType::Sub>(value, state.arg0, state.arg1);
        case Int64EvaluatorOp::MulLessThan:
            return EvaluateNumericArithmeticLessThan<
                proto::plan::ArithOpType::Mul>(value, state.arg0, state.arg1);
        case Int64EvaluatorOp::DivLessThan:
            return EvaluateNumericArithmeticLessThan<
                proto::plan::ArithOpType::Div>(value, state.arg0, state.arg1);
        case Int64EvaluatorOp::ModLessThan:
            return false;
    }
    return false;
}

template <Int64EvaluatorOp Op>
int32_t
EvaluateInt64Candidates(const void* opaque,
                        const int64_t* row_ids,
                        uint32_t count,
                        uint64_t active_mask,
                        uint64_t* valid_mask) noexcept {
    if (opaque == nullptr || row_ids == nullptr || valid_mask == nullptr ||
        count > 64) {
        return -1;
    }
    const auto& state = *static_cast<const Int64EvaluatorState*>(opaque);

    const uint64_t lane_mask =
        count == 64 ? UINT64_MAX
                    : (count == 0 ? uint64_t{0} : ((uint64_t{1} << count) - 1));
    const uint64_t active = active_mask & lane_mask;

    std::array<const int64_t*, 64> values{};
    const auto& source = state.source;
    if (source.row_values != nullptr) {
        for (uint32_t lane = 0; lane < count; ++lane) {
            const auto row_id = row_ids[lane];
            if ((active & (uint64_t{1} << lane)) != 0 && row_id >= 0 &&
                static_cast<size_t>(row_id) < source.row_count) {
                values[lane] = source.row_values + row_id;
                __builtin_prefetch(values[lane], 0, 1);
            }
        }
    } else if (source.num_chunks == 1 && source.chunk_values != nullptr &&
               source.chunk_values[0] != nullptr) {
        const auto* chunk = source.chunk_values[0];
        for (uint32_t lane = 0; lane < count; ++lane) {
            const auto row_id = row_ids[lane];
            if ((active & (uint64_t{1} << lane)) != 0 && row_id >= 0 &&
                static_cast<size_t>(row_id) < source.row_count) {
                values[lane] = chunk + row_id;
                __builtin_prefetch(values[lane], 0, 1);
            }
        }
    } else {
        for (uint32_t lane = 0; lane < count; ++lane) {
            if ((active & (uint64_t{1} << lane)) == 0) {
                continue;
            }
            values[lane] = ResolveInt64CandidateValue(source, row_ids[lane]);
            if (values[lane] != nullptr) {
                __builtin_prefetch(values[lane], 0, 1);
            }
        }
    }

    uint64_t accepted = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        if (values[lane] != nullptr &&
            EvaluateInt64ValueSpecialized<Op>(state, *values[lane])) {
            accepted |= uint64_t{1} << lane;
        }
    }
    *valid_mask = accepted;
    return 0;
}

template <Int64EvaluatorOp Op>
int32_t
EvaluateInt64ContiguousCandidates(const void* opaque,
                                  int64_t first_row_id,
                                  uint32_t count,
                                  uint64_t active_mask,
                                  uint64_t* valid_mask) noexcept {
    if (opaque == nullptr || valid_mask == nullptr || count > 64 ||
        first_row_id < 0) {
        return -1;
    }
    const auto& state = *static_cast<const Int64EvaluatorState*>(opaque);
    const uint64_t lane_mask =
        count == 64 ? UINT64_MAX
                    : (count == 0 ? uint64_t{0} : ((uint64_t{1} << count) - 1));
    const uint64_t active = active_mask & lane_mask;
    if (static_cast<size_t>(first_row_id) > state.source.row_count ||
        count > state.source.row_count - static_cast<size_t>(first_row_id)) {
        return -1;
    }

    uint64_t accepted = 0;
    if (state.source.row_values != nullptr) {
        const auto* values = state.source.row_values + first_row_id;
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint64_t lane_bit = uint64_t{1} << lane;
            if ((active & lane_bit) != 0 &&
                EvaluateInt64ValueSpecialized<Op>(state, values[lane])) {
                accepted |= lane_bit;
            }
        }
    } else {
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint64_t lane_bit = uint64_t{1} << lane;
            if ((active & lane_bit) == 0) {
                continue;
            }
            const auto* value = ResolveInt64CandidateValue(
                state.source, first_row_id + static_cast<int64_t>(lane));
            if (value != nullptr &&
                EvaluateInt64ValueSpecialized<Op>(state, *value)) {
                accepted |= lane_bit;
            }
        }
    }
    *valid_mask = accepted;
    return 0;
}

template <Int64EvaluatorOp Op>
int32_t
EvaluateInt64Truth(const void* opaque,
                   const int64_t* row_ids,
                   uint32_t count,
                   uint64_t active_mask,
                   uint64_t* true_mask,
                   uint64_t* known_mask) noexcept {
    if (known_mask == nullptr) {
        return -1;
    }
    const auto status = EvaluateInt64Candidates<Op>(
        opaque, row_ids, count, active_mask, true_mask);
    if (status == 0) {
        const auto lanes =
            count == 64
                ? ~uint64_t{0}
                : (count == 0 ? uint64_t{0} : ((uint64_t{1} << count) - 1));
        *known_mask = active_mask & lanes;
    }
    return status;
}

void
ConfigureInt64EvaluatorCallbacks(Int64EvaluatorOp op,
                                 PreparedCandidateEvaluator& prepared) {
#define SET_INT64_EVALUATOR_CALLBACKS(op_name)                             \
    case Int64EvaluatorOp::op_name:                                        \
        prepared.view.eval_batch =                                         \
            &EvaluateInt64Candidates<Int64EvaluatorOp::op_name>;           \
        prepared.view.eval_contiguous =                                    \
            &EvaluateInt64ContiguousCandidates<Int64EvaluatorOp::op_name>; \
        prepared.eval_truth_batch =                                        \
            &EvaluateInt64Truth<Int64EvaluatorOp::op_name>;                \
        return

    switch (op) {
        SET_INT64_EVALUATOR_CALLBACKS(GreaterEqual);
        SET_INT64_EVALUATOR_CALLBACKS(ModLessThan);
        SET_INT64_EVALUATOR_CALLBACKS(GreaterThan);
        SET_INT64_EVALUATOR_CALLBACKS(LessEqual);
        SET_INT64_EVALUATOR_CALLBACKS(LessThan);
        SET_INT64_EVALUATOR_CALLBACKS(Equal);
        SET_INT64_EVALUATOR_CALLBACKS(NotEqual);
        SET_INT64_EVALUATOR_CALLBACKS(Range);
        SET_INT64_EVALUATOR_CALLBACKS(Term);
        SET_INT64_EVALUATOR_CALLBACKS(AddLessThan);
        SET_INT64_EVALUATOR_CALLBACKS(SubLessThan);
        SET_INT64_EVALUATOR_CALLBACKS(MulLessThan);
        SET_INT64_EVALUATOR_CALLBACKS(DivLessThan);
    }
#undef SET_INT64_EVALUATOR_CALLBACKS
}

int32_t
EvaluateFloatCandidates(const void* opaque,
                        const int64_t* row_ids,
                        uint32_t count,
                        uint64_t active_mask,
                        uint64_t* valid_mask) noexcept {
    if (opaque == nullptr || row_ids == nullptr || valid_mask == nullptr ||
        count > 64) {
        return -1;
    }
    const auto& state = *static_cast<const FloatEvaluatorState*>(opaque);
    const uint64_t lane_mask =
        count == 64 ? UINT64_MAX
                    : (count == 0 ? uint64_t{0} : ((uint64_t{1} << count) - 1));
    const uint64_t active = active_mask & lane_mask;

    std::array<const float*, 64> values{};
    const auto& source = state.source;
    if (source.row_values != nullptr) {
        for (uint32_t lane = 0; lane < count; ++lane) {
            const auto row_id = row_ids[lane];
            if ((active & (uint64_t{1} << lane)) != 0 && row_id >= 0 &&
                static_cast<size_t>(row_id) < source.row_count) {
                values[lane] = source.row_values + row_id;
                __builtin_prefetch(values[lane], 0, 1);
            }
        }
    } else if (source.num_chunks == 1 && source.chunk_values != nullptr &&
               source.chunk_values[0] != nullptr) {
        const auto* chunk = source.chunk_values[0];
        for (uint32_t lane = 0; lane < count; ++lane) {
            const auto row_id = row_ids[lane];
            if ((active & (uint64_t{1} << lane)) != 0 && row_id >= 0 &&
                static_cast<size_t>(row_id) < source.row_count) {
                values[lane] = chunk + row_id;
                __builtin_prefetch(values[lane], 0, 1);
            }
        }
    } else {
        for (uint32_t lane = 0; lane < count; ++lane) {
            if ((active & (uint64_t{1} << lane)) == 0) {
                continue;
            }
            values[lane] = ResolveFloatCandidateValue(source, row_ids[lane]);
            if (values[lane] != nullptr) {
                __builtin_prefetch(values[lane], 0, 1);
            }
        }
    }
    uint64_t accepted = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        if (values[lane] != nullptr &&
            EvaluateFloatValue(state, *values[lane])) {
            accepted |= uint64_t{1} << lane;
        }
    }
    *valid_mask = accepted;
    return 0;
}

int32_t
EvaluateFloatTruth(const void* opaque,
                   const int64_t* row_ids,
                   uint32_t count,
                   uint64_t active_mask,
                   uint64_t* true_mask,
                   uint64_t* known_mask) noexcept {
    if (known_mask == nullptr) {
        return -1;
    }
    const auto status =
        EvaluateFloatCandidates(opaque, row_ids, count, active_mask, true_mask);
    if (status == 0) {
        const auto lanes =
            count == 64
                ? ~uint64_t{0}
                : (count == 0 ? uint64_t{0} : ((uint64_t{1} << count) - 1));
        *known_mask = active_mask & lanes;
    }
    return status;
}

int32_t
EvaluateFloatContiguousCandidates(const void* opaque,
                                  int64_t first_row_id,
                                  uint32_t count,
                                  uint64_t active_mask,
                                  uint64_t* valid_mask) noexcept {
    if (opaque == nullptr || valid_mask == nullptr || count > 64 ||
        first_row_id < 0) {
        return -1;
    }
    const auto& state = *static_cast<const FloatEvaluatorState*>(opaque);
    const uint64_t lane_mask =
        count == 64 ? UINT64_MAX
                    : (count == 0 ? uint64_t{0} : ((uint64_t{1} << count) - 1));
    const uint64_t active = active_mask & lane_mask;
    if (static_cast<size_t>(first_row_id) > state.source.row_count ||
        count > state.source.row_count - static_cast<size_t>(first_row_id)) {
        return -1;
    }

    uint64_t accepted = 0;
    if (state.source.row_values != nullptr) {
        const auto* values = state.source.row_values + first_row_id;
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint64_t lane_bit = uint64_t{1} << lane;
            if ((active & lane_bit) != 0 &&
                EvaluateFloatValue(state, values[lane])) {
                accepted |= lane_bit;
            }
        }
    } else {
        for (uint32_t lane = 0; lane < count; ++lane) {
            const uint64_t lane_bit = uint64_t{1} << lane;
            if ((active & lane_bit) == 0) {
                continue;
            }
            const auto* value = ResolveFloatCandidateValue(
                state.source, first_row_id + static_cast<int64_t>(lane));
            if (value != nullptr && EvaluateFloatValue(state, *value)) {
                accepted |= lane_bit;
            }
        }
    }
    *valid_mask = accepted;
    return 0;
}

std::optional<Int64EvaluatorOp>
ToNumericEvaluatorOp(NumericCandidatePredicateOp op) {
    switch (op) {
        case NumericCandidatePredicateOp::GreaterEqual:
            return Int64EvaluatorOp::GreaterEqual;
        case NumericCandidatePredicateOp::GreaterThan:
            return Int64EvaluatorOp::GreaterThan;
        case NumericCandidatePredicateOp::LessEqual:
            return Int64EvaluatorOp::LessEqual;
        case NumericCandidatePredicateOp::LessThan:
            return Int64EvaluatorOp::LessThan;
        case NumericCandidatePredicateOp::Equal:
            return Int64EvaluatorOp::Equal;
        case NumericCandidatePredicateOp::NotEqual:
            return Int64EvaluatorOp::NotEqual;
        case NumericCandidatePredicateOp::Range:
            return Int64EvaluatorOp::Range;
        case NumericCandidatePredicateOp::Term:
            return Int64EvaluatorOp::Term;
    }
    return std::nullopt;
}

std::optional<NumericCandidatePredicateOp>
ToNumericPredicateOp(proto::plan::OpType op) {
    switch (op) {
        case proto::plan::OpType::GreaterEqual:
            return NumericCandidatePredicateOp::GreaterEqual;
        case proto::plan::OpType::GreaterThan:
            return NumericCandidatePredicateOp::GreaterThan;
        case proto::plan::OpType::LessEqual:
            return NumericCandidatePredicateOp::LessEqual;
        case proto::plan::OpType::LessThan:
            return NumericCandidatePredicateOp::LessThan;
        case proto::plan::OpType::Equal:
            return NumericCandidatePredicateOp::Equal;
        case proto::plan::OpType::NotEqual:
            return NumericCandidatePredicateOp::NotEqual;
        default:
            return std::nullopt;
    }
}

std::optional<Int64EvaluatorOp>
ToArithmeticEvaluatorOp(proto::plan::ArithOpType op) {
    switch (op) {
        case proto::plan::ArithOpType::Mod:
            return Int64EvaluatorOp::ModLessThan;
        case proto::plan::ArithOpType::Add:
            return Int64EvaluatorOp::AddLessThan;
        case proto::plan::ArithOpType::Sub:
            return Int64EvaluatorOp::SubLessThan;
        case proto::plan::ArithOpType::Mul:
            return Int64EvaluatorOp::MulLessThan;
        case proto::plan::ArithOpType::Div:
            return Int64EvaluatorOp::DivLessThan;
        default:
            return std::nullopt;
    }
}

std::optional<int64_t>
AsInt64(const proto::plan::GenericValue& value) {
    return value.val_case() == proto::plan::GenericValue::kInt64Val
               ? std::optional<int64_t>(value.int64_val())
               : std::nullopt;
}

std::optional<double>
AsDouble(const proto::plan::GenericValue& value) {
    if (value.val_case() == proto::plan::GenericValue::kInt64Val) {
        return static_cast<double>(value.int64_val());
    }
    if (value.val_case() == proto::plan::GenericValue::kFloatVal) {
        return value.float_val();
    }
    return std::nullopt;
}

std::shared_ptr<std::vector<int64_t>>
MaterializeInt64CandidateValues(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    FieldId field_id,
    int64_t row_count) {
    std::vector<int64_t> offsets(row_count);
    std::iota(offsets.begin(), offsets.end(), 0);
    auto field_data = segment->bulk_subscript(
        op_context, field_id, offsets.data(), row_count);
    if (field_data == nullptr || !field_data->has_scalars()) {
        return nullptr;
    }
    auto values = std::make_shared<std::vector<int64_t>>();
    values->reserve(row_count);
    const auto& scalars = field_data->scalars();
    if (scalars.has_long_data() &&
        scalars.long_data().data_size() == row_count) {
        const auto& data = scalars.long_data().data();
        values->assign(data.begin(), data.end());
        return values;
    }
    if (scalars.has_int_data() &&
        scalars.int_data().data_size() == row_count) {
        for (const auto value : scalars.int_data().data()) {
            values->push_back(static_cast<int64_t>(value));
        }
        return values;
    }
    if (scalars.has_timestamptz_data() &&
        scalars.timestamptz_data().data_size() == row_count) {
        const auto& data = scalars.timestamptz_data().data();
        values->assign(data.begin(), data.end());
        return values;
    }
    return nullptr;
}

std::shared_ptr<std::vector<float>>
MaterializeFloatCandidateValues(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    FieldId field_id,
    int64_t row_count) {
    std::vector<int64_t> offsets(row_count);
    std::iota(offsets.begin(), offsets.end(), 0);
    auto field_data = segment->bulk_subscript(
        op_context, field_id, offsets.data(), row_count);
    if (field_data == nullptr || !field_data->has_scalars() ||
        !field_data->scalars().has_float_data() ||
        field_data->scalars().float_data().data_size() != row_count) {
        return nullptr;
    }
    const auto& data = field_data->scalars().float_data().data();
    return std::make_shared<std::vector<float>>(data.begin(), data.end());
}

template <typename PrepareEvaluator>
std::optional<PreparedCandidateLeaf>
PrepareInt64SourceLeaf(const segcore::SegmentInternalInterface* segment,
                       OpContext* op_context,
                       FieldId field_id,
                       DataType field_data_type,
                       PrepareEvaluator&& prepare_evaluator) {
    if (segment == nullptr || segment->type() != SegmentType::Sealed ||
        segment->get_schema()[field_id].get_data_type() != field_data_type) {
        return std::nullopt;
    }
    const auto row_count = segment->get_row_count();
    PreparedCandidateLeaf leaf;
    if ((field_data_type == DataType::INT64 ||
         field_data_type == DataType::TIMESTAMPTZ) &&
        segment->HasFieldData(field_id)) {
        auto owner = std::make_shared<Int64ChunkedCandidateSourceOwner>();
        const auto num_chunks = segment->num_chunk_data(field_id);
        if (num_chunks <= 0) {
            return std::nullopt;
        }
        owner->pins.reserve(num_chunks);
        owner->chunk_values.reserve(num_chunks);
        owner->chunk_offsets.reserve(num_chunks + 1);
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            owner->chunk_offsets.push_back(
                segment->num_rows_until_chunk(field_id, chunk_id));
            auto pin =
                segment->chunk_data<int64_t>(op_context, field_id, chunk_id);
            owner->chunk_values.push_back(pin.get().data());
            owner->pins.push_back(std::move(pin));
        }
        owner->chunk_offsets.push_back(row_count);
        auto evaluator =
            prepare_evaluator(owner->view(static_cast<size_t>(row_count)));
        if (!evaluator.has_value()) {
            return std::nullopt;
        }
        leaf.evaluator = std::move(*evaluator);
        leaf.resource_owners.push_back(std::move(owner));
        return leaf;
    }

    if (field_data_type != DataType::INT8 &&
        field_data_type != DataType::INT16 &&
        field_data_type != DataType::INT32) {
        return std::nullopt;
    }
    auto owner = std::make_shared<Int64MaterializedCandidateSourceOwner>();
    owner->materialized_values = MaterializeInt64CandidateValues(
        segment, op_context, field_id, row_count);
    if (owner->materialized_values == nullptr) {
        return std::nullopt;
    }
    auto evaluator =
        prepare_evaluator(owner->view(static_cast<size_t>(row_count)));
    if (!evaluator.has_value()) {
        return std::nullopt;
    }
    leaf.evaluator = std::move(*evaluator);
    leaf.resource_owners.push_back(std::move(owner));
    return leaf;
}

template <typename PrepareEvaluator>
std::optional<PreparedCandidateLeaf>
PrepareFloatSourceLeaf(const segcore::SegmentInternalInterface* segment,
                       OpContext* op_context,
                       FieldId field_id,
                       DataType field_data_type,
                       PrepareEvaluator&& prepare_evaluator) {
    if (segment == nullptr || segment->type() != SegmentType::Sealed ||
        field_data_type != DataType::FLOAT ||
        segment->get_schema()[field_id].get_data_type() != DataType::FLOAT) {
        return std::nullopt;
    }
    const auto row_count = segment->get_row_count();
    PreparedCandidateLeaf leaf;
    if (segment->HasFieldData(field_id)) {
        auto owner = std::make_shared<FloatChunkedCandidateSourceOwner>();
        const auto num_chunks = segment->num_chunk_data(field_id);
        if (num_chunks <= 0) {
            return std::nullopt;
        }
        owner->pins.reserve(num_chunks);
        owner->chunk_values.reserve(num_chunks);
        owner->chunk_offsets.reserve(num_chunks + 1);
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            owner->chunk_offsets.push_back(
                segment->num_rows_until_chunk(field_id, chunk_id));
            auto pin =
                segment->chunk_data<float>(op_context, field_id, chunk_id);
            owner->chunk_values.push_back(pin.get().data());
            owner->pins.push_back(std::move(pin));
        }
        owner->chunk_offsets.push_back(row_count);
        auto evaluator =
            prepare_evaluator(owner->view(static_cast<size_t>(row_count)));
        if (!evaluator.has_value()) {
            return std::nullopt;
        }
        leaf.evaluator = std::move(*evaluator);
        leaf.resource_owners.push_back(std::move(owner));
        return leaf;
    }
    auto owner = std::make_shared<FloatMaterializedCandidateSourceOwner>();
    owner->materialized_values = MaterializeFloatCandidateValues(
        segment, op_context, field_id, row_count);
    if (owner->materialized_values == nullptr) {
        return std::nullopt;
    }
    auto evaluator =
        prepare_evaluator(owner->view(static_cast<size_t>(row_count)));
    if (!evaluator.has_value()) {
        return std::nullopt;
    }
    leaf.evaluator = std::move(*evaluator);
    leaf.resource_owners.push_back(std::move(owner));
    return leaf;
}

std::optional<PreparedCandidateLeaf>
PrepareInt64ArithmeticLeaf(const segcore::SegmentInternalInterface* segment,
                           OpContext* op_context,
                           const void* typed_state) {
    if (typed_state == nullptr) {
        return std::nullopt;
    }
    const auto& predicate =
        *static_cast<const Int64ArithmeticCandidatePredicate*>(typed_state);
    return PrepareInt64SourceLeaf(
        segment,
        op_context,
        predicate.field_id,
        predicate.field_data_type,
        [&](const Int64CandidateSourceView& source) {
            return PrepareInt64ArithmeticCandidateEvaluator(source,
                                                             predicate);
        });
}

std::optional<PreparedCandidateLeaf>
PrepareFloatArithmeticLeaf(const segcore::SegmentInternalInterface* segment,
                           OpContext* op_context,
                           const void* typed_state) {
    if (typed_state == nullptr) {
        return std::nullopt;
    }
    const auto& predicate =
        *static_cast<const FloatArithmeticCandidatePredicate*>(typed_state);
    return PrepareFloatSourceLeaf(
        segment,
        op_context,
        predicate.field_id,
        predicate.field_data_type,
        [&](const FloatCandidateSourceView& source) {
            return PrepareFloatArithmeticCandidateEvaluator(source,
                                                             predicate);
        });
}

std::optional<PreparedCandidateLeaf>
PrepareInt64PredicateLeaf(const segcore::SegmentInternalInterface* segment,
                          OpContext* op_context,
                          const void* typed_state) {
    if (typed_state == nullptr) {
        return std::nullopt;
    }
    const auto& predicate =
        *static_cast<const Int64CandidatePredicate*>(typed_state);
    return PrepareInt64SourceLeaf(
        segment,
        op_context,
        predicate.field_id,
        predicate.field_data_type,
        [&](const Int64CandidateSourceView& source) {
            return PrepareInt64CandidateEvaluator(source, predicate);
        });
}

std::optional<PreparedCandidateLeaf>
PrepareFloatPredicateLeaf(const segcore::SegmentInternalInterface* segment,
                          OpContext* op_context,
                          const void* typed_state) {
    if (typed_state == nullptr) {
        return std::nullopt;
    }
    const auto& predicate =
        *static_cast<const FloatCandidatePredicate*>(typed_state);
    return PrepareFloatSourceLeaf(
        segment,
        op_context,
        predicate.field_id,
        predicate.field_data_type,
        [&](const FloatCandidateSourceView& source) {
            return PrepareFloatCandidateEvaluator(source, predicate);
        });
}

}  // namespace

std::optional<CandidateLeafPlan>
TryCompileNumericCandidateLeaf(const expr::TypedExprPtr& expression) {
    if (expression == nullptr) {
        return std::nullopt;
    }

    auto is_supported_column = [](const expr::ColumnInfo& column) {
        const auto type = column.data_type_;
        const bool numeric =
            type == DataType::INT8 || type == DataType::INT16 ||
            type == DataType::INT32 || type == DataType::INT64 ||
            type == DataType::TIMESTAMPTZ || type == DataType::FLOAT;
        return numeric && !column.nullable_ && !column.element_level_ &&
               column.nested_path_.empty();
    };

    auto make_int_plan = [](Int64CandidatePredicate predicate) {
        auto state =
            std::make_shared<Int64CandidatePredicate>(std::move(predicate));
        return CandidateLeafPlan{std::move(state), &PrepareInt64PredicateLeaf};
    };
    auto make_float_plan = [](FloatCandidatePredicate predicate) {
        auto state =
            std::make_shared<FloatCandidatePredicate>(std::move(predicate));
        return CandidateLeafPlan{std::move(state), &PrepareFloatPredicateLeaf};
    };

    if (auto unary =
            std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(
                expression)) {
        const auto& column = unary->column_;
        const auto op = ToNumericPredicateOp(unary->op_type_);
        if (!is_supported_column(column) || !op.has_value()) {
            return std::nullopt;
        }
        if (column.data_type_ == DataType::FLOAT) {
            const auto arg = AsDouble(unary->val_);
            if (!arg.has_value()) {
                return std::nullopt;
            }
            FloatCandidatePredicate predicate;
            predicate.field_id = column.field_id_;
            predicate.field_data_type = column.data_type_;
            predicate.op = *op;
            predicate.arg0 = static_cast<float>(*arg);
            return make_float_plan(std::move(predicate));
        }
        const auto arg = AsInt64(unary->val_);
        if (!arg.has_value()) {
            return std::nullopt;
        }
        Int64CandidatePredicate predicate;
        predicate.field_id = column.field_id_;
        predicate.field_data_type = column.data_type_;
        predicate.op = *op;
        predicate.arg0 = *arg;
        return make_int_plan(std::move(predicate));
    }

    if (auto range =
            std::dynamic_pointer_cast<const expr::BinaryRangeFilterExpr>(
                expression)) {
        const auto& column = range->column_;
        if (!is_supported_column(column)) {
            return std::nullopt;
        }
        if (column.data_type_ == DataType::FLOAT) {
            const auto lower = AsDouble(range->lower_val_);
            const auto upper = AsDouble(range->upper_val_);
            if (!lower.has_value() || !upper.has_value()) {
                return std::nullopt;
            }
            FloatCandidatePredicate predicate;
            predicate.field_id = column.field_id_;
            predicate.field_data_type = column.data_type_;
            predicate.op = NumericCandidatePredicateOp::Range;
            predicate.arg0 = static_cast<float>(*lower);
            predicate.arg1 = static_cast<float>(*upper);
            predicate.lower_inclusive = range->lower_inclusive_;
            predicate.upper_inclusive = range->upper_inclusive_;
            return make_float_plan(std::move(predicate));
        }
        const auto lower = AsInt64(range->lower_val_);
        const auto upper = AsInt64(range->upper_val_);
        if (!lower.has_value() || !upper.has_value()) {
            return std::nullopt;
        }
        Int64CandidatePredicate predicate;
        predicate.field_id = column.field_id_;
        predicate.field_data_type = column.data_type_;
        predicate.op = NumericCandidatePredicateOp::Range;
        predicate.arg0 = *lower;
        predicate.arg1 = *upper;
        predicate.lower_inclusive = range->lower_inclusive_;
        predicate.upper_inclusive = range->upper_inclusive_;
        return make_int_plan(std::move(predicate));
    }

    auto term =
        std::dynamic_pointer_cast<const expr::TermFilterExpr>(expression);
    if (term == nullptr || !is_supported_column(term->column_) ||
        term->vals_.empty()) {
        return std::nullopt;
    }
    const auto& column = term->column_;
    if (column.data_type_ == DataType::FLOAT) {
        FloatCandidatePredicate predicate;
        predicate.field_id = column.field_id_;
        predicate.field_data_type = column.data_type_;
        predicate.op = NumericCandidatePredicateOp::Term;
        predicate.terms.reserve(term->vals_.size());
        for (const auto& value : term->vals_) {
            const auto converted = AsDouble(value);
            if (!converted.has_value()) {
                return std::nullopt;
            }
            predicate.terms.push_back(static_cast<float>(*converted));
        }
        std::sort(predicate.terms.begin(), predicate.terms.end());
        predicate.terms.erase(
            std::unique(predicate.terms.begin(), predicate.terms.end()),
            predicate.terms.end());
        return make_float_plan(std::move(predicate));
    }

    Int64CandidatePredicate predicate;
    predicate.field_id = column.field_id_;
    predicate.field_data_type = column.data_type_;
    predicate.op = NumericCandidatePredicateOp::Term;
    predicate.terms.reserve(term->vals_.size());
    for (const auto& value : term->vals_) {
        const auto converted = AsInt64(value);
        if (!converted.has_value()) {
            return std::nullopt;
        }
        predicate.terms.push_back(*converted);
    }
    std::sort(predicate.terms.begin(), predicate.terms.end());
    predicate.terms.erase(
        std::unique(predicate.terms.begin(), predicate.terms.end()),
        predicate.terms.end());

    // A dense integral TERM is equivalent to an inclusive range and avoids a
    // binary search in the graph hot path.
    const auto first = predicate.terms.front();
    const auto last = predicate.terms.back();
    const auto expected_size =
        static_cast<__int128>(last) - static_cast<__int128>(first) + 1;
    if (expected_size > 0 &&
        expected_size == static_cast<__int128>(predicate.terms.size())) {
        predicate.op = NumericCandidatePredicateOp::Range;
        predicate.arg0 = first;
        predicate.arg1 = last;
        predicate.terms.clear();
    }
    return make_int_plan(std::move(predicate));
}

std::shared_ptr<const PreparedOffsetExpressionEvaluator>
PrepareNumericOffsetExpressionEvaluator(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    const expr::TypedExprPtr& expression) {
    if (segment == nullptr || segment->type() != SegmentType::Sealed ||
        segment->get_schema().get_ttl_field_id().has_value()) {
        return nullptr;
    }

    bool is_int64 = false;
    if (const auto unary =
            std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(
                expression)) {
        is_int64 = unary->column_.data_type_ == DataType::INT64;
    } else if (const auto arithmetic = std::dynamic_pointer_cast<
                   const expr::BinaryArithOpEvalRangeExpr>(expression)) {
        is_int64 = arithmetic->column_.data_type_ == DataType::INT64 &&
                   arithmetic->arith_op_type_ == proto::plan::ArithOpType::Mod;
    }
    if (!is_int64) {
        return nullptr;
    }

    std::optional<CandidateLeafPlan> plan;
    if (const auto arithmetic =
            std::dynamic_pointer_cast<const expr::BinaryArithOpEvalRangeExpr>(
                expression)) {
        plan = TryCompileNumericArithmeticCandidateLeaf(*arithmetic);
    } else {
        plan = TryCompileNumericCandidateLeaf(expression);
    }
    if (!plan.has_value()) {
        return nullptr;
    }
    std::optional<PreparedCandidateLeaf> leaf;
    try {
        leaf = plan->prepare(segment, op_context, plan->typed_state.get());
    } catch (...) {
        return nullptr;
    }
    if (!leaf.has_value() || !static_cast<bool>(leaf->evaluator)) {
        return nullptr;
    }
    auto program = std::make_shared<const NumericOffsetProgram>(
        NumericOffsetProgram{std::move(*leaf)});
    return PreparedOffsetExpressionEvaluator::Create(
        std::move(program),
        &CreateNumericOffsetWorkspace,
        &EvaluateNumericOffsetBatch);
}

std::optional<CandidateLeafPlan>
TryCompileNumericArithmeticCandidateLeaf(
    const expr::BinaryArithOpEvalRangeExpr& expression) {
    const auto& column = expression.column_;
    if (expression.op_type_ != proto::plan::OpType::LessThan ||
        column.nullable_ || column.element_level_ ||
        !column.nested_path_.empty() ||
        !ToArithmeticEvaluatorOp(expression.arith_op_type_).has_value()) {
        return std::nullopt;
    }

    if (column.data_type_ == DataType::FLOAT) {
        if (expression.arith_op_type_ == proto::plan::ArithOpType::Mod) {
            return std::nullopt;
        }
        const auto operand = AsDouble(expression.right_operand_);
        const auto target = AsDouble(expression.value_);
        if (!operand.has_value() || !target.has_value() ||
            (expression.arith_op_type_ == proto::plan::ArithOpType::Div &&
             *operand == 0.0)) {
            return std::nullopt;
        }
        auto state = std::make_shared<FloatArithmeticCandidatePredicate>(
            FloatArithmeticCandidatePredicate{column.field_id_,
                                              column.data_type_,
                                              expression.arith_op_type_,
                                              static_cast<float>(*operand),
                                              static_cast<float>(*target)});
        return CandidateLeafPlan{std::move(state),
                                 &PrepareFloatArithmeticLeaf};
    }

    if (column.data_type_ != DataType::INT8 &&
        column.data_type_ != DataType::INT16 &&
        column.data_type_ != DataType::INT32 &&
        column.data_type_ != DataType::INT64 &&
        column.data_type_ != DataType::TIMESTAMPTZ) {
        return std::nullopt;
    }
    // Until sequential SIMD, random offsets, scalar-index lookup and
    // SkipIndex share one explicit overflow contract, integral fusing admits
    // only the constrained MOD form below.  TIMESTAMPTZ arithmetic is also
    // excluded; it must not inherit numeric MOD semantics accidentally.
    if (column.data_type_ == DataType::TIMESTAMPTZ ||
        expression.arith_op_type_ != proto::plan::ArithOpType::Mod) {
        return std::nullopt;
    }
    const auto operand = AsInt64(expression.right_operand_);
    const auto target = AsInt64(expression.value_);
    if (!operand.has_value() || !target.has_value() ||
        ((expression.arith_op_type_ == proto::plan::ArithOpType::Div) &&
         *operand == 0) ||
        (expression.arith_op_type_ == proto::plan::ArithOpType::Mod &&
         (*operand <= 0 || *target < 0 || *target > *operand))) {
        return std::nullopt;
    }
    auto state = std::make_shared<Int64ArithmeticCandidatePredicate>(
        Int64ArithmeticCandidatePredicate{column.field_id_,
                                          column.data_type_,
                                          expression.arith_op_type_,
                                          *operand,
                                          *target});
    return CandidateLeafPlan{std::move(state), &PrepareInt64ArithmeticLeaf};
}

std::optional<PreparedCandidateEvaluator>
PrepareInt64ModCandidateEvaluator(const Int64CandidateSourceView& source,
                                  int64_t divisor,
                                  int64_t upper_bound) {
    return PrepareInt64ArithmeticCandidateEvaluator(
        source,
        Int64ArithmeticCandidatePredicate{FieldId{},
                                          DataType::INT64,
                                          proto::plan::ArithOpType::Mod,
                                          divisor,
                                          upper_bound});
}

std::optional<PreparedCandidateEvaluator>
PrepareInt64ArithmeticCandidateEvaluator(
    const Int64CandidateSourceView& source,
    const Int64ArithmeticCandidatePredicate& predicate) {
    const bool has_contiguous_source = source.row_values != nullptr;
    const bool has_chunked_source = source.chunk_values != nullptr &&
                                    source.chunk_offsets != nullptr &&
                                    source.num_chunks > 0;
    const auto op = ToArithmeticEvaluatorOp(predicate.arithmetic_op);
    if (!op.has_value() || source.row_count == 0 ||
        (!has_contiguous_source && !has_chunked_source) ||
        (*op == Int64EvaluatorOp::ModLessThan &&
         (predicate.operand <= 0 || predicate.target < 0 ||
          predicate.target > predicate.operand)) ||
        (*op == Int64EvaluatorOp::DivLessThan && predicate.operand == 0)) {
        return std::nullopt;
    }
    auto owner = std::make_shared<Int64EvaluatorState>(Int64EvaluatorState{
        source, *op, predicate.operand, predicate.target, true, true, {}});
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    ConfigureInt64EvaluatorCallbacks(*op, prepared);
    return prepared;
}

std::optional<PreparedCandidateEvaluator>
PrepareFloatArithmeticCandidateEvaluator(
    const FloatCandidateSourceView& source,
    const FloatArithmeticCandidatePredicate& predicate) {
    const bool has_contiguous_source = source.row_values != nullptr;
    const bool has_chunked_source = source.chunk_values != nullptr &&
                                    source.chunk_offsets != nullptr &&
                                    source.num_chunks > 0;
    const auto op = ToArithmeticEvaluatorOp(predicate.arithmetic_op);
    if (!op.has_value() || *op == Int64EvaluatorOp::ModLessThan ||
        source.row_count == 0 ||
        (!has_contiguous_source && !has_chunked_source) ||
        (*op == Int64EvaluatorOp::DivLessThan && predicate.operand == 0.0F)) {
        return std::nullopt;
    }
    auto owner = std::make_shared<FloatEvaluatorState>(FloatEvaluatorState{
        source,
        *op,
        predicate.operand,
        predicate.target,
        true,
        true,
        {}});
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    prepared.view.eval_batch = &EvaluateFloatCandidates;
    prepared.view.eval_contiguous = &EvaluateFloatContiguousCandidates;
    prepared.eval_truth_batch = &EvaluateFloatTruth;
    return prepared;
}

std::optional<PreparedCandidateEvaluator>
PrepareInt64CandidateEvaluator(const Int64CandidateSourceView& source,
                               const Int64CandidatePredicate& predicate) {
    const bool has_contiguous_source = source.row_values != nullptr;
    const bool has_chunked_source = source.chunk_values != nullptr &&
                                    source.chunk_offsets != nullptr &&
                                    source.num_chunks > 0;
    const auto op = ToNumericEvaluatorOp(predicate.op);
    if (!op.has_value() || source.row_count == 0 ||
        (!has_contiguous_source && !has_chunked_source)) {
        return std::nullopt;
    }
    if (*op == Int64EvaluatorOp::Term && predicate.terms.empty()) {
        return std::nullopt;
    }
    auto owner = std::make_shared<Int64EvaluatorState>(
        Int64EvaluatorState{source,
                            *op,
                            predicate.arg0,
                            predicate.arg1,
                            predicate.lower_inclusive,
                            predicate.upper_inclusive,
                            predicate.terms});
    if (*op == Int64EvaluatorOp::Term) {
        std::sort(owner->terms.begin(), owner->terms.end());
        owner->terms.erase(
            std::unique(owner->terms.begin(), owner->terms.end()),
            owner->terms.end());
    }
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    ConfigureInt64EvaluatorCallbacks(*op, prepared);
    return prepared;
}

std::optional<PreparedCandidateEvaluator>
PrepareFloatCandidateEvaluator(const FloatCandidateSourceView& source,
                               const FloatCandidatePredicate& predicate) {
    const bool has_contiguous_source = source.row_values != nullptr;
    const bool has_chunked_source = source.chunk_values != nullptr &&
                                    source.chunk_offsets != nullptr &&
                                    source.num_chunks > 0;
    const auto op = ToNumericEvaluatorOp(predicate.op);
    if (!op.has_value() || source.row_count == 0 ||
        (!has_contiguous_source && !has_chunked_source)) {
        return std::nullopt;
    }
    if (*op == Int64EvaluatorOp::Term && predicate.terms.empty()) {
        return std::nullopt;
    }
    auto owner = std::make_shared<FloatEvaluatorState>(
        FloatEvaluatorState{source,
                            *op,
                            predicate.arg0,
                            predicate.arg1,
                            predicate.lower_inclusive,
                            predicate.upper_inclusive,
                            predicate.terms});
    if (*op == Int64EvaluatorOp::Term) {
        owner->terms.erase(
            std::remove_if(owner->terms.begin(),
                           owner->terms.end(),
                           [](float value) { return std::isnan(value); }),
            owner->terms.end());
        std::sort(owner->terms.begin(), owner->terms.end());
        owner->terms.erase(
            std::unique(owner->terms.begin(), owner->terms.end()),
            owner->terms.end());
    }
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    prepared.view.eval_batch = &EvaluateFloatCandidates;
    prepared.view.eval_contiguous = &EvaluateFloatContiguousCandidates;
    prepared.eval_truth_batch = &EvaluateFloatTruth;
    return prepared;
}

}  // namespace milvus::exec
