// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/DownpushPredicateProvider.h"

#include <algorithm>
#include <limits>

#include "common/RegexQuery.h"

namespace milvus::exec {
namespace {

enum class LikeTokenType : uint8_t {
    Literal = 0,
    AnyOne = 1,
    AnyMany = 2,
};

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

}  // namespace milvus::exec
