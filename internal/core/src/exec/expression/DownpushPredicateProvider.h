// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include <memory>
#include <optional>

#include "common/Downpush.h"
#include "expr/ITypeExpr.h"

namespace milvus::exec {

// Type-owned conversion boundary for ANN filter fusing. The pipeline owns
// expression composition; numeric, varchar, JSON and future value families
// own literal conversion and operator capability here.
class DownpushPredicateProvider {
 public:
    virtual ~DownpushPredicateProvider() = default;

    virtual bool
    Supports(const expr::ColumnInfo& column) const = 0;

    virtual CardinalDownpushPredicate
    NewPredicate(const expr::ColumnInfo& column) const = 0;

    virtual bool
    SupportsRangeOp(CardinalDownpushPredicateOp op) const = 0;

    virtual bool
    FillArg(CardinalDownpushPredicate& predicate,
            const proto::plan::GenericValue& value,
            bool second_arg) const = 0;

    virtual bool
    FillTerms(CardinalDownpushPredicate& predicate,
              const std::vector<proto::plan::GenericValue>& values) const = 0;

    virtual bool
    FinalizeUnary(CardinalDownpushPredicate& predicate) const {
        return true;
    }

    virtual bool
    FillArithmetic(
        CardinalDownpushPredicate& predicate,
        const expr::BinaryArithOpEvalRangeExpr& expression) const = 0;
};

const DownpushPredicateProvider*
FindDownpushPredicateProvider(const expr::ColumnInfo& column);

const DownpushPredicateProvider&
NumericDownpushPredicateProvider();

const DownpushPredicateProvider&
StringDownpushPredicateProvider();

}  // namespace milvus::exec
