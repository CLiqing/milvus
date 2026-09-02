// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/DownpushPredicateProvider.h"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "common/RegexQuery.h"
#include "exec/expression/StringCandidateEvaluator.h"
#include "exec/operator/StringCandidateSourceOwner.h"
#include "index/StringIndex.h"
#include "log/Log.h"
#include "segcore/SegmentInterface.h"

namespace milvus::exec {
namespace {

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
ToStringEvaluatorOp(StringCandidateComparisonOp op) {
    switch (op) {
        case StringCandidateComparisonOp::GreaterEqual:
            return StringEvaluatorOp::GreaterEqual;
        case StringCandidateComparisonOp::GreaterThan:
            return StringEvaluatorOp::GreaterThan;
        case StringCandidateComparisonOp::LessEqual:
            return StringEvaluatorOp::LessEqual;
        case StringCandidateComparisonOp::LessThan:
            return StringEvaluatorOp::LessThan;
        case StringCandidateComparisonOp::Equal:
            return StringEvaluatorOp::Equal;
        case StringCandidateComparisonOp::NotEqual:
            return StringEvaluatorOp::NotEqual;
    }
    return std::nullopt;
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
            const auto* upper = std::upper_bound(
                source.chunk_row_offsets,
                source.chunk_row_offsets + source.num_chunks + 1,
                row_id);
            if (upper == source.chunk_row_offsets) {
                return false;
            }
            chunk_idx =
                static_cast<size_t>((upper - source.chunk_row_offsets) - 1);
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
            const bool lower_ok = state.lower_inclusive ? value >= state.arg0
                                                        : value > state.arg0;
            const bool upper_ok = state.upper_inclusive ? value <= state.arg1
                                                        : value < state.arg1;
            return lower_ok && upper_ok;
        }
        case StringEvaluatorOp::Term:
            return std::binary_search(
                state.terms.begin(), state.terms.end(), value);
        case StringEvaluatorOp::Like:
            return state.like_matcher != nullptr &&
                   (*state.like_matcher)(value);
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
        (active_mask &
         ~((count == 64) ? ~uint64_t{0} : ((uint64_t{1} << count) - 1))) != 0) {
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
EvaluateStringTruth(const void* context,
                    const int64_t* candidate_ids,
                    uint32_t count,
                    uint64_t active_mask,
                    uint64_t* true_mask,
                    uint64_t* known_mask) noexcept {
    if (context == nullptr || candidate_ids == nullptr ||
        true_mask == nullptr || known_mask == nullptr || count > 64 ||
        (active_mask &
         ~((count == 64) ? ~uint64_t{0} : ((uint64_t{1} << count) - 1))) != 0) {
        return -1;
    }
    const auto& state = *static_cast<const StringEvaluatorState*>(context);
    uint64_t accepted = 0;
    uint64_t known = 0;
    for (uint32_t lane = 0; lane < count; ++lane) {
        const auto lane_bit = uint64_t{1} << lane;
        if ((active_mask & lane_bit) == 0) {
            continue;
        }
        const auto row_id = candidate_ids[lane];
        if (row_id < 0 ||
            static_cast<size_t>(row_id) >= state.source.row_count) {
            continue;
        }
        bool passed = false;
        if (state.source.row_dictionary_ids != nullptr) {
            const auto value_id = state.source.row_dictionary_ids[row_id];
            if (value_id < 0) {
                continue;
            }
            known |= lane_bit;
            if (state.op == StringEvaluatorOp::Equal) {
                passed = state.source.target_dictionary_id_found &&
                         value_id == state.source.target_dictionary_id;
            } else if (state.op == StringEvaluatorOp::NotEqual) {
                passed = !state.source.target_dictionary_id_found ||
                         value_id != state.source.target_dictionary_id;
            }
        } else {
            std::string_view value;
            if (!ResolveRawStringCandidate(state.source, row_id, &value)) {
                continue;
            }
            known |= lane_bit;
            passed = EvaluateStringValue(state, value);
        }
        if (passed) {
            accepted |= lane_bit;
        }
    }
    *true_mask = accepted;
    *known_mask = known;
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

FieldId
StringPredicateFieldId(const StringCandidatePredicate& predicate) {
    return std::visit([](const auto& value) { return value.field_id; },
                      predicate);
}

DataType
StringPredicateFieldType(const StringCandidatePredicate& predicate) {
    return std::visit([](const auto& value) { return value.field_data_type; },
                      predicate);
}

std::optional<std::string_view>
DictionaryLookupValue(const StringCandidatePredicate& predicate) {
    const auto* comparison =
        std::get_if<StringComparisonCandidatePredicate>(&predicate);
    if (comparison == nullptr ||
        (comparison->op != StringCandidateComparisonOp::Equal &&
         comparison->op != StringCandidateComparisonOp::NotEqual)) {
        return std::nullopt;
    }
    return comparison->value;
}

std::optional<PreparedCandidateLeaf>
PrepareStringPredicateLeaf(const segcore::SegmentInternalInterface* segment,
                           OpContext* op_context,
                           const void* typed_state) {
    if (segment == nullptr || typed_state == nullptr ||
        segment->type() != SegmentType::Sealed) {
        return std::nullopt;
    }
    const auto& predicate =
        *static_cast<const StringCandidatePredicate*>(typed_state);
    const auto field_id = StringPredicateFieldId(predicate);
    const auto field_type = StringPredicateFieldType(predicate);
    if ((field_type != DataType::VARCHAR && field_type != DataType::STRING) ||
        segment->get_schema()[field_id].get_data_type() != field_type) {
        return std::nullopt;
    }
    const auto row_count = segment->get_row_count();
    PreparedCandidateLeaf leaf;
    if (segment->HasFieldData(field_id)) {
        auto owner = std::make_shared<RawStringCandidateSourceOwner>();
        const auto num_chunks = segment->num_chunk_data(field_id);
        if (num_chunks <= 0) {
            return std::nullopt;
        }
        owner->pins.reserve(num_chunks);
        owner->chunk_bases.reserve(num_chunks);
        owner->chunk_value_offsets.reserve(num_chunks);
        owner->chunk_valid_data.reserve(num_chunks);
        owner->chunk_row_counts.reserve(num_chunks);
        owner->chunk_row_offsets.reserve(num_chunks + 1);
        int64_t expected_row_offset = 0;
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            const auto chunk_row_offset =
                segment->num_rows_until_chunk(field_id, chunk_id);
            if (chunk_row_offset != expected_row_offset) {
                return std::nullopt;
            }
            auto pin =
                segment->raw_string_chunk_view(op_context, field_id, chunk_id);
            const auto view = pin.get();
            if (view.base == nullptr || view.offsets == nullptr ||
                view.row_count == 0 || expected_row_offset > row_count ||
                static_cast<int64_t>(view.row_count) >
                    row_count - expected_row_offset) {
                return std::nullopt;
            }
            owner->chunk_row_offsets.push_back(chunk_row_offset);
            owner->chunk_bases.push_back(view.base);
            owner->chunk_value_offsets.push_back(view.offsets);
            owner->chunk_valid_data.push_back(view.valid_data);
            owner->chunk_row_counts.push_back(view.row_count);
            owner->pins.push_back(std::move(pin));
            expected_row_offset += view.row_count;
        }
        if (expected_row_offset != row_count) {
            return std::nullopt;
        }
        owner->chunk_row_offsets.push_back(row_count);
        const auto uniform_rows = owner->chunk_row_counts.front();
        bool uniform = uniform_rows > 0;
        for (size_t i = 1; i + 1 < owner->chunk_row_counts.size(); ++i) {
            uniform = uniform && owner->chunk_row_counts[i] == uniform_rows;
        }
        if (owner->chunk_row_counts.back() > uniform_rows) {
            uniform = false;
        }
        owner->uniform_chunk_rows = uniform ? uniform_rows : 0;
        auto evaluator = PrepareStringCandidateEvaluator(
            owner->view(static_cast<size_t>(row_count)), predicate);
        if (!evaluator.has_value()) {
            return std::nullopt;
        }
        leaf.evaluator = std::move(*evaluator);
        leaf.resource_owners.push_back(std::move(owner));
        return leaf;
    }

    const auto dictionary_value = DictionaryLookupValue(predicate);
    if (!dictionary_value.has_value() || !segment->HasIndex(field_id)) {
        return std::nullopt;
    }
    auto pins = segment->PinIndex(op_context, field_id);
    if (pins.size() != 1) {
        return std::nullopt;
    }
    const auto* string_index =
        dynamic_cast<const index::StringIndex*>(pins.front().get());
    if (string_index == nullptr) {
        return std::nullopt;
    }
    auto dictionary = string_index->GetDictionaryIdColumnView(
        std::string(*dictionary_value));
    if (!dictionary.has_value() || dictionary->row_value_ids == nullptr ||
        dictionary->row_count != static_cast<size_t>(row_count)) {
        return std::nullopt;
    }
    auto owner = std::make_shared<StringDictionaryCandidateSourceOwner>();
    owner->row_dictionary_ids = dictionary->row_value_ids;
    owner->row_count = dictionary->row_count;
    owner->target_dictionary_id = dictionary->target_dictionary_id;
    owner->target_dictionary_id_found =
        dictionary->target_dictionary_id_found;
    owner->index_pins = std::move(pins);
    auto evaluator = PrepareStringCandidateEvaluator(owner->view(), predicate);
    if (!evaluator.has_value()) {
        return std::nullopt;
    }
    leaf.evaluator = std::move(*evaluator);
    leaf.resource_owners.push_back(std::move(owner));
    return leaf;
}

}  // namespace

std::optional<CandidateLeafPlan>
TryCompileStringCandidateLeaf(const expr::TypedExprPtr& expression) {
    if (expression == nullptr) {
        return std::nullopt;
    }
    auto supported = [](const expr::ColumnInfo& column) {
        return (column.data_type_ == DataType::VARCHAR ||
                column.data_type_ == DataType::STRING) &&
               !column.element_level_ && column.nested_path_.empty();
    };
    auto as_string = [](const proto::plan::GenericValue& value)
        -> std::optional<std::string> {
        if (value.val_case() != proto::plan::GenericValue::kStringVal) {
            return std::nullopt;
        }
        return value.string_val();
    };
    auto make_plan = [](StringCandidatePredicate predicate) {
        auto state =
            std::make_shared<StringCandidatePredicate>(std::move(predicate));
        return CandidateLeafPlan{std::move(state), &PrepareStringPredicateLeaf};
    };

    if (auto unary =
            std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(
                expression)) {
        const auto& column = unary->column_;
        auto value = as_string(unary->val_);
        if (!supported(column) || !value.has_value()) {
            return std::nullopt;
        }
        auto comparison_op = [&]()
            -> std::optional<StringCandidateComparisonOp> {
            switch (unary->op_type_) {
                case proto::plan::OpType::GreaterEqual:
                    return StringCandidateComparisonOp::GreaterEqual;
                case proto::plan::OpType::GreaterThan:
                    return StringCandidateComparisonOp::GreaterThan;
                case proto::plan::OpType::LessEqual:
                    return StringCandidateComparisonOp::LessEqual;
                case proto::plan::OpType::LessThan:
                    return StringCandidateComparisonOp::LessThan;
                case proto::plan::OpType::Equal:
                    return StringCandidateComparisonOp::Equal;
                case proto::plan::OpType::NotEqual:
                    return StringCandidateComparisonOp::NotEqual;
                default:
                    return std::nullopt;
            }
        }();
        if (comparison_op.has_value()) {
            return make_plan(StringComparisonCandidatePredicate{
                column.field_id_, column.data_type_, column.nullable_,
                *comparison_op, std::move(*value)});
        }

        std::string pattern;
        if (unary->op_type_ == proto::plan::OpType::Match) {
            pattern = std::move(*value);
        } else if (unary->op_type_ == proto::plan::OpType::PrefixMatch ||
                   unary->op_type_ == proto::plan::OpType::PostfixMatch ||
                   unary->op_type_ == proto::plan::OpType::InnerMatch) {
            for (const char c : *value) {
                if (c == '%' || c == '_' || c == '\\') {
                    pattern.push_back('\\');
                }
                pattern.push_back(c);
            }
            if (unary->op_type_ != proto::plan::OpType::PrefixMatch) {
                pattern.insert(pattern.begin(), '%');
            }
            if (unary->op_type_ != proto::plan::OpType::PostfixMatch) {
                pattern.push_back('%');
            }
        } else {
            return std::nullopt;
        }
        try {
            LikePatternMatcher validation(pattern);
        } catch (const std::exception&) {
            return std::nullopt;
        }
        return make_plan(StringLikeCandidatePredicate{column.field_id_,
                                                      column.data_type_,
                                                      column.nullable_,
                                                      std::move(pattern)});
    }

    if (auto range =
            std::dynamic_pointer_cast<const expr::BinaryRangeFilterExpr>(
                expression)) {
        const auto& column = range->column_;
        auto lower = as_string(range->lower_val_);
        auto upper = as_string(range->upper_val_);
        if (!supported(column) || !lower.has_value() || !upper.has_value()) {
            return std::nullopt;
        }
        return make_plan(StringRangeCandidatePredicate{column.field_id_,
                                                       column.data_type_,
                                                       column.nullable_,
                                                       std::move(*lower),
                                                       std::move(*upper),
                                                       range->lower_inclusive_,
                                                       range->upper_inclusive_});
    }

    auto term =
        std::dynamic_pointer_cast<const expr::TermFilterExpr>(expression);
    if (term == nullptr || !supported(term->column_) || term->vals_.empty()) {
        return std::nullopt;
    }
    std::vector<std::string> terms;
    terms.reserve(term->vals_.size());
    for (const auto& value : term->vals_) {
        auto converted = as_string(value);
        if (!converted.has_value()) {
            return std::nullopt;
        }
        terms.push_back(std::move(*converted));
    }
    std::sort(terms.begin(), terms.end());
    terms.erase(std::unique(terms.begin(), terms.end()), terms.end());
    return make_plan(StringTermCandidatePredicate{term->column_.field_id_,
                                                  term->column_.data_type_,
                                                  term->column_.nullable_,
                                                  std::move(terms)});
}

std::optional<PreparedCandidateEvaluator>
PrepareStringCandidateEvaluator(const StringCandidateSourceView& source,
                                const StringCandidatePredicate& predicate) {
    auto owner = std::make_shared<StringEvaluatorState>();
    owner->source = source;
    std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T,
                                         StringComparisonCandidatePredicate>) {
                owner->op = *ToStringEvaluatorOp(value.op);
                owner->arg0 = value.value;
            } else if constexpr (std::is_same_v<
                                     T, StringRangeCandidatePredicate>) {
                owner->op = StringEvaluatorOp::Range;
                owner->arg0 = value.lower;
                owner->arg1 = value.upper;
                owner->lower_inclusive = value.lower_inclusive;
                owner->upper_inclusive = value.upper_inclusive;
            } else if constexpr (std::is_same_v<T,
                                                StringTermCandidatePredicate>) {
                owner->op = StringEvaluatorOp::Term;
                owner->terms = value.terms;
            } else {
                owner->op = StringEvaluatorOp::Like;
                owner->arg0 = value.pattern;
            }
        },
        predicate);
    const bool has_raw_source = source.chunk_bases != nullptr &&
                                source.chunk_value_offsets != nullptr &&
                                source.chunk_row_counts != nullptr &&
                                source.chunk_row_offsets != nullptr &&
                                source.num_chunks > 0;
    const bool dictionary_op = owner->op == StringEvaluatorOp::Equal ||
                               owner->op == StringEvaluatorOp::NotEqual;
    const bool has_dictionary_source =
        source.row_dictionary_ids != nullptr && dictionary_op;
    if (source.row_count == 0 ||
        (!has_raw_source && !has_dictionary_source) ||
        (owner->op == StringEvaluatorOp::Term && owner->terms.empty())) {
        return std::nullopt;
    }
    if (owner->op == StringEvaluatorOp::Term) {
        std::sort(owner->terms.begin(), owner->terms.end());
        owner->terms.erase(
            std::unique(owner->terms.begin(), owner->terms.end()),
            owner->terms.end());
    }
    if (owner->op == StringEvaluatorOp::Like) {
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
    prepared.eval_truth_batch = &EvaluateStringTruth;
    return prepared;
}

}  // namespace milvus::exec
