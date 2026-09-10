// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include "exec/expression/ConjunctExpr.h"

namespace milvus::exec {
namespace {

class TracePredicate : public Expr {
 public:
    TracePredicate(size_t size, size_t limit, bool nullable = false)
        : Expr(DataType::BOOL, {}, "TracePredicate", nullptr),
          size_(size),
          limit_(limit),
          nullable_(nullable) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override {
        const auto* offsets = context.get_offset_input();
        const auto count =
            offsets ? offsets->size() : std::min(size_ - position_, size_t{64});
        TargetBitmap truth(count), valid(count, true);
        for (size_t i = 0; i < count; ++i) {
            const auto id = offsets ? (*offsets)[i] : position_ + i;
            visited.push_back(id);
            truth[i] = id < limit_;
            valid[i] = !nullable_ || id % 3 != 0;
        }
        if (!offsets) {
            position_ += count;
        }
        result =
            std::make_shared<ColumnVector>(std::move(truth), std::move(valid));
    }

    void
    MoveCursor() override {
        position_ += std::min(size_ - position_, size_t{64});
    }
    std::string
    ToString() const override {
        return "TracePredicate";
    }
    std::vector<size_t> visited;

 private:
    size_t size_, limit_, position_ = 0;
    bool nullable_;
};

TEST(FilterMapExprTest, AndOnlyEvaluatesAcceptedIdsInChosenOrder) {
    constexpr size_t n = 129;
    QueryContext query("map_predicate", nullptr, n, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    auto a = std::make_shared<TracePredicate>(n, 5);
    auto b = std::make_shared<TracePredicate>(n, n, true);
    auto c = std::make_shared<TracePredicate>(n, 4);
    std::vector<ExprPtr> children{a, b, c};
    PhyConjunctFilterExpr conjunction(std::move(children), true, nullptr);
    conjunction.MarkNullRejecting();
    auto map = conjunction.EvalFilterMap(context, n, 8);
    EXPECT_EQ(a->visited.size(), n);
    EXPECT_EQ(b->visited, (std::vector<size_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(c->visited, (std::vector<size_t>{1, 2, 4}));
    EXPECT_EQ(map.count(), n - 2);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(map.test(i), i != 1 && i != 2);
    }
    EXPECT_EQ(context.get_offset_input(), nullptr);
}

TEST(FilterMapExprTest, DensePredecessorCanShrinkWithoutRepeatingPredicates) {
    constexpr size_t n = 129;
    QueryContext query("map_predicate", nullptr, n, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    auto narrow = std::make_shared<TracePredicate>(n, 5);
    auto wide = std::make_shared<TracePredicate>(n, n);
    std::vector<ExprPtr> children{narrow, wide};
    PhyConjunctFilterExpr conjunction(std::move(children), true, nullptr);
    conjunction.MarkNullRejecting();
    conjunction.Reorder({1, 0});
    auto map = conjunction.EvalFilterMap(context, n, 8);
    EXPECT_EQ(wide->visited.size(), n);
    EXPECT_EQ(narrow->visited.size(), n);
    EXPECT_EQ(map.capability(), FilterMapCapability::EnumerateOnly);
    EXPECT_EQ(map.count(), n - 5);
}

TEST(FilterMapExprTest, OrDoesNotRestrictSecondPredicateToFirstMatches) {
    constexpr size_t n = 129;
    QueryContext query("map_predicate", nullptr, n, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    auto a = std::make_shared<TracePredicate>(n, 2, true);
    auto b = std::make_shared<TracePredicate>(n, 5);
    std::vector<ExprPtr> children{a, b};
    PhyConjunctFilterExpr conjunction(std::move(children), false, nullptr);
    conjunction.MarkNullRejecting();
    auto map = conjunction.EvalFilterMap(context, n, 8);
    EXPECT_EQ(map.count(), n - 5);
    EXPECT_EQ(a->visited.size(), n);
    EXPECT_EQ(b->visited.size(), n);
}

}  // namespace
}  // namespace milvus::exec
