// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/DownpushPredicateProvider.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>

#include "common/RegexQuery.h"
#include "exec/expression/CandidateEvaluator.h"

namespace milvus::exec {
namespace {

enum class LikeTokenType : uint8_t {
    Literal = 0,
    AnyOne = 1,
    AnyMany = 2,
};

enum class StringEvaluatorOp : uint8_t {
    GreaterEqual,
    GreaterThan,
    LessEqual,
    LessThan,
    Equal,
    NotEqual,
    Range,
    Term,
    Like,
};

struct StringEvaluatorState {
    StringCandidateSourceView source;
    StringEvaluatorOp op = StringEvaluatorOp::Equal;
    std::string arg0;
    std::string arg1;
    bool lower_inclusive = true;
    bool upper_inclusive = true;
    std::vector<std::string> terms;
    std::unique_ptr<LikePatternMatcher> like_matcher;
};

std::optional<StringEvaluatorOp>
ToStringEvaluatorOp(CardinalDownpushPredicateOp op) {
    switch (op) {
        case CardinalDownpushPredicateOp::Int64GreaterEqual:
            return StringEvaluatorOp::GreaterEqual;
        case CardinalDownpushPredicateOp::Int64GreaterThan:
            return StringEvaluatorOp::GreaterThan;
        case CardinalDownpushPredicateOp::Int64LessEqual:
            return StringEvaluatorOp::LessEqual;
        case CardinalDownpushPredicateOp::Int64LessThan:
            return StringEvaluatorOp::LessThan;
        case CardinalDownpushPredicateOp::Int64Equal:
            return StringEvaluatorOp::Equal;
        case CardinalDownpushPredicateOp::Int64NotEqual:
            return StringEvaluatorOp::NotEqual;
        case CardinalDownpushPredicateOp::ScalarRange:
            return StringEvaluatorOp::Range;
        case CardinalDownpushPredicateOp::ScalarTerm:
            return StringEvaluatorOp::Term;
        case CardinalDownpushPredicateOp::StringLikeMatch:
            return StringEvaluatorOp::Like;
        default:
            return std::nullopt;
    }
}

bool
ResolveRawStringCandidate(const StringCandidateSourceView& source,
                          int64_t row_id,
                          std::string_view* value) noexcept {
    if (value == nullptr || row_id < 0 ||
        static_cast<size_t>(row_id) >= source.row_count ||
        source.chunk_bases == nullptr ||
        source.chunk_value_offsets == nullptr ||
        source.chunk_row_counts == nullptr ||
        source.chunk_row_offsets == nullptr || source.num_chunks == 0) {
        return false;
    }
    size_t chunk_idx = 0;
    size_t local_offset = static_cast<size_t>(row_id);
    if (source.num_chunks > 1) {
        if (source.uniform_chunk_rows > 0) {
            chunk_idx = local_offset / source.uniform_chunk_rows;
            if (chunk_idx >= source.num_chunks) {
                return false;
            }
            local_offset -= chunk_idx * source.uniform_chunk_rows;
        } else {
            const auto* upper =
                std::upper_bound(source.chunk_row_offsets,
                                 source.chunk_row_offsets +
                                     source.num_chunks + 1,
                                 row_id);
            if (upper == source.chunk_row_offsets) {
                return false;
            }
            chunk_idx = static_cast<size_t>(
                (upper - source.chunk_row_offsets) - 1);
            if (chunk_idx >= source.num_chunks) {
                return false;
            }
            local_offset -=
                static_cast<size_t>(source.chunk_row_offsets[chunk_idx]);
        }
    }
    const auto* base = source.chunk_bases[chunk_idx];
    const auto* offsets = source.chunk_value_offsets[chunk_idx];
    const auto* valid_data = source.chunk_valid_data == nullptr
                                 ? nullptr
                                 : source.chunk_valid_data[chunk_idx];
    if (base == nullptr || offsets == nullptr ||
        local_offset >= source.chunk_row_counts[chunk_idx] ||
        (valid_data != nullptr && !valid_data[local_offset])) {
        return false;
    }
    const auto begin = offsets[local_offset];
    const auto end = offsets[local_offset + 1];
    if (end < begin) {
        return false;
    }
    *value = std::string_view(base + begin, end - begin);
    return true;
}

bool
EvaluateStringValue(const StringEvaluatorState& state,
                    std::string_view value) noexcept {
    switch (state.op) {
        case StringEvaluatorOp::GreaterEqual:
            return value >= state.arg0;
        case StringEvaluatorOp::GreaterThan:
            return value > state.arg0;
        case StringEvaluatorOp::LessEqual:
            return value <= state.arg0;
        case StringEvaluatorOp::LessThan:
            return value < state.arg0;
        case StringEvaluatorOp::Equal:
            return value == state.arg0;
        case StringEvaluatorOp::NotEqual:
            return value != state.arg0;
        case StringEvaluatorOp::Range: {
            const bool lower_ok = state.lower_inclusive
                                      ? value >= state.arg0
                                      : value > state.arg0;
            const bool upper_ok = state.upper_inclusive
                                      ? value <= state.arg1
                                      : value < state.arg1;
            return lower_ok && upper_ok;
        }
        case StringEvaluatorOp::Term:
            return std::binary_search(
                state.terms.begin(), state.terms.end(), value);
        case StringEvaluatorOp::Like:
            return state.like_matcher != nullptr && (*state.like_matcher)(value);
    }
    return false;
}

bool
EvaluateStringCandidate(const StringEvaluatorState& state,
                        int64_t row_id) noexcept {
    if (row_id < 0 || static_cast<size_t>(row_id) >= state.source.row_count) {
        return false;
    }
    if (state.source.row_dictionary_ids != nullptr) {
        const auto value_id = state.source.row_dictionary_ids[row_id];
        if (value_id < 0) {
            return false;
        }
        if (state.op == StringEvaluatorOp::Equal) {
            return state.source.target_dictionary_id_found &&
                   value_id == state.source.target_dictionary_id;
        }
        if (state.op == StringEvaluatorOp::NotEqual) {
            return !state.source.target_dictionary_id_found ||
                   value_id != state.source.target_dictionary_id;
        }
        return false;
    }
    std::string_view value;
    return ResolveRawStringCandidate(state.source, row_id, &value) &&
           EvaluateStringValue(state, value);
}

int32_t
EvaluateStringCandidates(const void* context,
                         const int64_t* candidate_ids,
                         uint32_t count,
                         uint64_t active_mask,
                         uint64_t* valid_mask) noexcept {
    if (context == nullptr || candidate_ids == nullptr ||
        valid_mask == nullptr || count > 64 ||
        (active_mask & ~((count == 64) ? ~uint64_t{0}
                                      : ((uint64_t{1} << count) - 1))) != 0) {
        return -1;
    }
    const auto& state = *static_cast<const StringEvaluatorState*>(context);
    uint64_t accepted = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        const auto lane_bit = uint64_t{1} << lane;
        if ((active_mask & lane_bit) != 0 &&
            EvaluateStringCandidate(state, candidate_ids[lane])) {
            accepted |= lane_bit;
        }
    }
    *valid_mask = accepted;
    return 0;
}

int32_t
EvaluateStringContiguousCandidates(const void* context,
                                   int64_t first_candidate_id,
                                   uint32_t count,
                                   uint64_t active_mask,
                                   uint64_t* valid_mask) noexcept {
    if (count > 64) {
        return -1;
    }
    std::array<int64_t, 64> candidate_ids{};
    for (uint32_t lane = 0; lane < count; ++lane) {
        candidate_ids[lane] = first_candidate_id + lane;
    }
    return EvaluateStringCandidates(
        context, candidate_ids.data(), count, active_mask, valid_mask);
}

class StringProvider final : public DownpushPredicateProvider {
 public:
    bool
    Supports(const expr::ColumnInfo& column) const override {
        return (column.data_type_ == DataType::VARCHAR ||
                column.data_type_ == DataType::STRING) &&
               !column.element_level_ && column.nested_path_.empty();
    }

    CardinalDownpushPredicate
    NewPredicate(const expr::ColumnInfo& column) const override {
        CardinalDownpushPredicate predicate;
        predicate.field_id_ = column.field_id_;
        predicate.field_data_type_ = column.data_type_;
        predicate.value_type_ = CardinalDownpushPredicateValueType::String;
        return predicate;
    }

    bool
    SupportsRangeOp(CardinalDownpushPredicateOp op) const override {
        return op != CardinalDownpushPredicateOp::Int64ModLessThan;
    }

    bool
    FillArg(CardinalDownpushPredicate& predicate,
            const proto::plan::GenericValue& value,
            bool second_arg) const override {
        if (value.val_case() != proto::plan::GenericValue::kStringVal) {
            return false;
        }
        (second_arg ? predicate.string_arg1_ : predicate.string_arg0_) =
            value.string_val();
        return true;
    }

    bool
    FillTerms(
        CardinalDownpushPredicate& predicate,
        const std::vector<proto::plan::GenericValue>& values) const override {
        if (values.empty()) {
            return false;
        }
        for (const auto& value : values) {
            if (value.val_case() != proto::plan::GenericValue::kStringVal) {
                return false;
            }
            predicate.string_terms_.push_back(value.string_val());
        }
        std::sort(predicate.string_terms_.begin(),
                  predicate.string_terms_.end());
        predicate.string_terms_.erase(
            std::unique(predicate.string_terms_.begin(),
                        predicate.string_terms_.end()),
            predicate.string_terms_.end());
        return true;
    }

    bool
    FinalizeUnary(CardinalDownpushPredicate& predicate) const override {
        if (predicate.op_ != CardinalDownpushPredicateOp::StringLikeMatch) {
            return true;
        }
        const auto& pattern = predicate.string_arg0_;
        if (pattern.size() > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        auto add_token = [&](LikeTokenType type, size_t offset, size_t size) {
            predicate.like_token_offsets_.push_back(
                static_cast<uint32_t>(offset));
            predicate.like_token_sizes_.push_back(static_cast<uint32_t>(size));
            predicate.like_token_types_.push_back(static_cast<uint8_t>(type));
        };
        for (size_t i = 0; i < pattern.size();) {
            const auto c = pattern[i];
            if (c == '\\') {
                ++i;
                if (i == pattern.size()) {
                    return false;
                }
                const auto size = Utf8ValidatedCharByteLen(pattern.data() + i,
                                                           pattern.size() - i);
                add_token(LikeTokenType::Literal, i, size);
                i += size;
            } else if (c == '%') {
                if (predicate.like_token_types_.empty() ||
                    predicate.like_token_types_.back() !=
                        static_cast<uint8_t>(LikeTokenType::AnyMany)) {
                    add_token(LikeTokenType::AnyMany, i, 1);
                }
                ++i;
            } else if (c == '_') {
                add_token(LikeTokenType::AnyOne, i, 1);
                ++i;
            } else {
                const auto size = Utf8ValidatedCharByteLen(pattern.data() + i,
                                                           pattern.size() - i);
                add_token(LikeTokenType::Literal, i, size);
                i += size;
            }
        }
        return true;
    }

    bool
    FillArithmetic(CardinalDownpushPredicate&,
                   const expr::BinaryArithOpEvalRangeExpr&) const override {
        return false;
    }
};

}  // namespace

const DownpushPredicateProvider&
StringDownpushPredicateProvider() {
    static const StringProvider provider;
    return provider;
}

std::optional<PreparedCandidateEvaluator>
PrepareStringCandidateEvaluator(const StringCandidateSourceView& source,
                                const CardinalDownpushPredicate& predicate) {
    const auto op = ToStringEvaluatorOp(predicate.op_);
    const bool has_raw_source =
        source.chunk_bases != nullptr &&
        source.chunk_value_offsets != nullptr &&
        source.chunk_row_counts != nullptr &&
        source.chunk_row_offsets != nullptr && source.num_chunks > 0;
    const bool dictionary_op =
        op.has_value() && (*op == StringEvaluatorOp::Equal ||
                           *op == StringEvaluatorOp::NotEqual);
    const bool has_dictionary_source =
        source.row_dictionary_ids != nullptr && dictionary_op;
    if (predicate.value_type_ != CardinalDownpushPredicateValueType::String ||
        !op.has_value() || source.row_count == 0 ||
        (!has_raw_source && !has_dictionary_source) ||
        (*op == StringEvaluatorOp::Term && predicate.string_terms_.empty())) {
        return std::nullopt;
    }
    auto owner = std::make_shared<StringEvaluatorState>();
    owner->source = source;
    owner->op = *op;
    owner->arg0 = predicate.string_arg0_;
    owner->arg1 = predicate.string_arg1_;
    owner->lower_inclusive = predicate.lower_inclusive_;
    owner->upper_inclusive = predicate.upper_inclusive_;
    owner->terms = predicate.string_terms_;
    if (*op == StringEvaluatorOp::Term) {
        std::sort(owner->terms.begin(), owner->terms.end());
        owner->terms.erase(
            std::unique(owner->terms.begin(), owner->terms.end()),
            owner->terms.end());
    }
    if (*op == StringEvaluatorOp::Like) {
        try {
            // Reuse the same prepared matcher as the baseline String
            // expression path.  Cardinal receives only the generic batch
            // callback and remains unaware of LIKE syntax or tokenization.
            owner->like_matcher =
                std::make_unique<LikePatternMatcher>(owner->arg0);
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }
    PreparedCandidateEvaluator prepared;
    prepared.owner = owner;
    prepared.view.context = owner.get();
    prepared.view.eval_batch = &EvaluateStringCandidates;
    prepared.view.eval_contiguous = &EvaluateStringContiguousCandidates;
    return prepared;
}

}  // namespace milvus::exec
