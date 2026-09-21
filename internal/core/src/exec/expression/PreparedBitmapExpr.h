// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#pragma once

#include "exec/expression/Expr.h"

namespace milvus::exec {

// Query/segment-owned expression result, not an operator IR and not a
// cross-query cache. Truth and validity remain immutable for every worker.
class PreparedBitmapExpr final : public expr::ITypeFilterExpr {
 public:
    PreparedBitmapExpr(const expr::TypedExprPtr& source,
                       const std::shared_ptr<ColumnVector>& result)
        : source_(source),
          truth_(TargetBitmapView(result->GetRawData(), result->size())),
          valid_(TargetBitmapView(result->GetValidRawData(), result->size())) {
    }
    std::string
    ToString() const override {
        return "PreparedBitmap(" + source_->ToString() + ")";
    }
    const TargetBitmap&
    truth() const {
        return truth_;
    }
    const TargetBitmap&
    valid() const {
        return valid_;
    }

 private:
    expr::TypedExprPtr source_;
    const TargetBitmap truth_;
    const TargetBitmap valid_;
};

class PhyPreparedBitmapExpr final : public Expr {
 public:
    PhyPreparedBitmapExpr(std::shared_ptr<const PreparedBitmapExpr> source,
                          OpContext* op_context,
                          int64_t batch_size)
        : Expr(DataType::BOOL, {}, "PhyPreparedBitmapExpr", op_context),
          source_(std::move(source)),
          batch_size_(batch_size) {
    }
    void
    Eval(EvalCtx& context, VectorPtr& result) override;
    void
    MoveCursor() override;
    bool
    CanExecuteAllAtOnce() const override {
        return true;
    }
    void
    SetExecuteAllAtOnce() override {
        batch_size_ = source_->truth().size();
    }
    bool
    IsCacheable() const override {
        return false;
    }
    bool
    IsSource() const override {
        return true;
    }
    std::optional<expr::ColumnInfo>
    GetColumnInfo() const override {
        return std::nullopt;
    }
    std::string
    ToString() const override {
        return source_->ToString();
    }

 private:
    std::shared_ptr<const PreparedBitmapExpr> source_;
    int64_t batch_size_;
    int64_t cursor_{0};
};

}  // namespace milvus::exec
