// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include "exec/expression/ConjunctExpr.h"

namespace milvus::exec {
namespace {

class FilterMapExprTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        old_ = GetDefaultFilterMapConfig();
        ASSERT_TRUE(SetDefaultFilterMapConfig({true, 0, 0.125}));
    }
    void
    TearDown() override {
        SetDefaultFilterMapConfig(old_);
    }
    FilterMapConfig old_;
};

FilterMap
Evaluate(Expr& expr,
         EvalCtx& context,
         std::optional<FilterMap> input = std::nullopt) {
    EvalCtx selected(context.get_exec_context());
    selected.EnableFilterOutput();
    if (input) {
        input->flip();  // Test inputs use final excluded-bit semantics.
        selected.set_filter_input(std::move(*input));
    }
    VectorPtr result;
    expr.Eval(selected, result);
    auto values = GetColumnVector(result);
    values->ConvertToFiltered();
    return values->GetFilterMap();
}

class TracePredicate : public Expr {
 public:
    TracePredicate(size_t size, size_t limit, bool nullable = false)
        : Expr(DataType::BOOL, {}, "TracePredicate", nullptr),
          size_(size),
          limit_(limit),
          nullable_(nullable) {
    }

    void
    EvalImpl(EvalCtx& context, VectorPtr& result) override {
        const auto* offsets = context.get_offset_input();
        const auto count = offsets ? offsets->size()
                                   : std::min(size_ - position_,
                                              all_at_once ? size_ : size_t{64});
        calls.push_back(count);
        const auto& active = context.get_bitmap_input();
        TargetBitmap truth(count), valid(count, true);
        for (size_t i = 0; i < count; ++i) {
            if (!active.empty() && honor_mask && !active[i]) {
                continue;
            }
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
        if (fail_after_eval) {
            throw std::runtime_error("injected predicate failure");
        }
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
    std::vector<size_t> calls;
    bool honor_mask = true;
    bool all_at_once = false;
    bool fail_after_eval = false;
    bool support_offsets = true;
    bool
    SupportOffsetInput() override {
        return support_offsets;
    }
    bool
    CanExecuteAllAtOnce() const override {
        return all_at_once;
    }

 private:
    size_t size_, limit_, position_ = 0;
    bool nullable_;
};

TEST_F(FilterMapExprTest, AndOnlyEvaluatesAcceptedIdsInChosenOrder) {
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
    auto map = Evaluate(conjunction, context);
    EXPECT_EQ(a->visited.size(), n);
    EXPECT_EQ(b->visited, (std::vector<size_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(c->visited, (std::vector<size_t>{1, 2, 4}));
    EXPECT_EQ(map.count(), n - 2);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(map.test(i), i != 1 && i != 2);
    }
    EXPECT_EQ(context.get_offset_input(), nullptr);
}

TEST_F(FilterMapExprTest, DensePredecessorCanShrinkWithoutRepeatingPredicates) {
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
    auto map = Evaluate(conjunction, context);
    EXPECT_EQ(wide->visited.size(), n);
    EXPECT_EQ(narrow->visited.size(), n);
    EXPECT_EQ(map.capability(), FilterMapCapability::EnumerateOnly);
    EXPECT_EQ(map.count(), n - 5);
}

TEST_F(FilterMapExprTest, OrDoesNotRestrictSecondPredicateToFirstMatches) {
    constexpr size_t n = 129;
    QueryContext query("map_predicate", nullptr, n, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    auto a = std::make_shared<TracePredicate>(n, 2, true);
    auto b = std::make_shared<TracePredicate>(n, 5);
    std::vector<ExprPtr> children{a, b};
    PhyConjunctFilterExpr conjunction(std::move(children), false, nullptr);
    conjunction.MarkNullRejecting();
    auto map = Evaluate(conjunction, context);
    EXPECT_EQ(map.count(), n - 5);
    EXPECT_EQ(a->visited.size(), n);
    // Existing OR active-mask pruning skips only the definitely TRUE row 1.
    EXPECT_EQ(b->visited.size(), n - 1);
    EXPECT_EQ(std::count(b->visited.begin(), b->visited.end(), 1), 0);
}

TEST_F(FilterMapExprTest, WholeSparseInputUsesOneEvalBeyondConfiguredBatch) {
    constexpr size_t n = 1000000;
    QueryContext query("map_predicate", nullptr, n, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    const size_t v = context.get_query_config()->get_expr_batch_size() + 17;
    ASSERT_LT(v, n);
    auto input = FilterMap::Adaptive(n, true, v);
    std::vector<int32_t> ids(v);
    for (size_t i = 0; i < v; ++i) {
        ids[i] = n - 1 - i;
    }
    input.AppendUniqueBits(ids, false);
    TracePredicate predicate(n, n, true);
    auto map = Evaluate(predicate, context, std::move(input));
    EXPECT_EQ(predicate.calls, (std::vector<size_t>{v}));
    EXPECT_EQ(predicate.visited, std::vector<size_t>(ids.begin(), ids.end()));
    size_t accepted = 0;
    for (auto id : ids) {
        EXPECT_EQ(map.test(id), id % 3 == 0);
        accepted += id % 3 != 0;
    }
    EXPECT_EQ(map.count(), n - accepted);
    EXPECT_EQ(context.get_offset_input(), nullptr);
    EXPECT_TRUE(context.get_bitmap_input().empty());
}

TEST_F(FilterMapExprTest, EmptySparseInputDoesNotCallPredicate) {
    constexpr size_t n = 129;
    QueryContext query("map_predicate", nullptr, n, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    TracePredicate predicate(n, n);
    auto map = Evaluate(predicate, context, FilterMap::Adaptive(n, true, 8));
    EXPECT_TRUE(predicate.calls.empty());
    EXPECT_EQ(map.count(), n);
}

TEST_F(FilterMapExprTest, EmptyUniverseDoesNotCallPredicate) {
    QueryContext query("empty_filter_range", nullptr, 0, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    TracePredicate predicate(0, 0);
    auto map = Evaluate(predicate, context);
    EXPECT_TRUE(predicate.calls.empty());
    EXPECT_EQ(map.size(), 0);
    EXPECT_EQ(map.count(), 0);
}

TEST_F(FilterMapExprTest, DenseMaskIsAppliedBeforeEvalAndAlignedToEveryBatch) {
    constexpr size_t n = 129;
    for (bool honor_mask : {true, false}) {
        QueryContext query("map_predicate", nullptr, n, 0);
        ExecContext exec(&query);
        EvalCtx context(&exec);
        auto dense = std::make_shared<TargetBitmap>(n, true);
        for (auto id : {5, 64, 128}) {
            (*dense)[id] = false;
        }
        TracePredicate predicate(n, 100);
        predicate.honor_mask = honor_mask;
        auto map = Evaluate(
            predicate, context, FilterMap::FromDense(std::move(dense)));
        if (honor_mask) {
            EXPECT_EQ(predicate.visited, (std::vector<size_t>{5, 64, 128}));
        } else {
            EXPECT_EQ(predicate.visited.size(), n);
        }
        EXPECT_EQ(predicate.calls, (std::vector<size_t>{64, 64, 1}));
        for (size_t id = 0; id < n; ++id) {
            EXPECT_EQ(map.test(id), id != 5 && id != 64);
        }
        EXPECT_TRUE(context.get_bitmap_input().empty());
        EXPECT_EQ(context.get_offset_input(), nullptr);
    }
}

TEST_F(FilterMapExprTest, AllAtOncePredicateReceivesDenseActiveMask) {
    constexpr size_t n = 129;
    QueryContext query("map_predicate", nullptr, n, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    auto dense = std::make_shared<TargetBitmap>(n, true);
    (*dense)[128] = false;
    TracePredicate predicate(n, n);
    predicate.all_at_once = true;
    auto map =
        Evaluate(predicate, context, FilterMap::FromDense(std::move(dense)));
    EXPECT_EQ(predicate.calls, (std::vector<size_t>{n}));
    EXPECT_EQ(predicate.visited, (std::vector<size_t>{128}));
    EXPECT_EQ(map.count(), n - 1);
    EXPECT_FALSE(map.test(128));
    EXPECT_TRUE(context.get_bitmap_input().empty());
}

TEST_F(FilterMapExprTest, PredicateFailureDoesNotLeakMaskOrOffsetsToParent) {
    constexpr size_t n = 129;
    for (bool sparse : {true, false}) {
        QueryContext query("map_predicate", nullptr, n, 0);
        ExecContext exec(&query);
        EvalCtx context(&exec);
        auto input = FilterMap::Adaptive(n, true, 8);
        input.set(5, false);
        if (!sparse) {
            input.EnsureDense();
        }
        TracePredicate predicate(n, n);
        predicate.fail_after_eval = true;
        EXPECT_THROW(Evaluate(predicate, context, std::move(input)),
                     std::runtime_error);
        EXPECT_EQ(context.get_offset_input(), nullptr);
        EXPECT_TRUE(context.get_bitmap_input().empty());
    }
}

TEST_F(FilterMapExprTest,
       EmptyCandidatesShortCircuitEveryRepresentationAndKernel) {
    constexpr size_t n = 129;
    for (bool sparse : {false, true}) {
        for (bool offsets : {false, true}) {
            QueryContext query("empty_candidates", nullptr, n, 0);
            ExecContext exec(&query);
            EvalCtx context(&exec);
            auto input = FilterMap::Adaptive(n, true, 8);
            if (!sparse)
                input.EnsureDense();
            TracePredicate predicate(n, n);
            predicate.support_offsets = offsets;
            auto map = Evaluate(predicate, context, std::move(input));
            EXPECT_EQ(map.count(), n);
            EXPECT_TRUE(predicate.calls.empty());
            EXPECT_TRUE(predicate.visited.empty());
        }
    }
}

TEST_F(FilterMapExprTest, AndStopsBeforeUnsupportedSuccessorsAndNestedAnd) {
    constexpr size_t n = 129;
    QueryContext query("empty_and", nullptr, n, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    auto empty = std::make_shared<TracePredicate>(n, 0);
    auto b = std::make_shared<TracePredicate>(n, n);
    b->support_offsets = false;
    auto c = std::make_shared<TracePredicate>(n, n);
    auto nested = std::make_shared<PhyConjunctFilterExpr>(
        std::vector<ExprPtr>{b, c}, true, nullptr);
    PhyConjunctFilterExpr conjunction(
        std::vector<ExprPtr>{empty, nested}, true, nullptr);
    conjunction.MarkNullRejecting();
    auto map = Evaluate(conjunction, context);
    EXPECT_EQ(map.count(), n);
    EXPECT_TRUE(b->calls.empty());
    EXPECT_TRUE(c->calls.empty());
}

class BatchPattern : public Expr {
 public:
    explicit BatchPattern(bool first_empty)
        : Expr(DataType::BOOL, {}, "batch_pattern", nullptr),
          empty_(first_empty) {
    }
    void
    EvalImpl(EvalCtx&, VectorPtr& result) override {
        ++calls;
        const bool value = !empty_ || position != 0;
        result = std::make_shared<ColumnVector>(TargetBitmap(64, value),
                                                TargetBitmap(64, true));
        ++position;
    }
    void
    MoveCursor() override {
        ++position;
        ++skips;
    }
    size_t position = 0, calls = 0, skips = 0;

 private:
    bool empty_;
};

TEST_F(FilterMapExprTest, EmptyLegacyBatchAdvancesSiblingBeforeNonemptyBatch) {
    QueryContext query("batch_alignment", nullptr, 128, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);  // No full-range consumer; retain batch contract.
    auto a = std::make_shared<BatchPattern>(true);
    auto b = std::make_shared<BatchPattern>(false);
    PhyConjunctFilterExpr conjunction(
        std::vector<ExprPtr>{a, b}, true, nullptr);
    conjunction.MarkNullRejecting();
    VectorPtr result;
    conjunction.Eval(context, result);
    EXPECT_EQ(b->calls, 0);
    EXPECT_EQ(b->skips, 1);
    EXPECT_EQ(b->position, 1);
    EXPECT_EQ(GetColumnVector(result)->GetFilterMap().count(), 0);
    conjunction.Eval(context, result);
    EXPECT_EQ(b->calls, 1);
    EXPECT_EQ(b->position, 2);
    EXPECT_EQ(GetColumnVector(result)->GetFilterMap().count(), 64);
    EXPECT_TRUE(context.get_bitmap_input().empty());
}

TEST_F(FilterMapExprTest, DefaultDisabledKeepsOriginalBatchEntry) {
    ASSERT_TRUE(SetDefaultFilterMapConfig({false, 0, 0.125}));
    QueryContext query("disabled", nullptr, 129, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    context.set_bitmap_input(TargetBitmap{});  // Legacy empty means no mask.
    context.EnableFilterOutput();
    EXPECT_FALSE(context.filter_rows());
    TracePredicate predicate(129, 129);
    VectorPtr result;
    predicate.Eval(context, result);
    EXPECT_EQ(result->size(), 64);
    EXPECT_EQ(predicate.calls, (std::vector<size_t>{64}));
}

TEST_F(FilterMapExprTest, EmptyLegacyInputSkipsUnsupportedKernelAndAdvances) {
    QueryContext query("empty_legacy", nullptr, 128, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    context.set_bitmap_input(TargetBitmap(64, false));
    TracePredicate predicate(128, 128);
    predicate.support_offsets = false;
    VectorPtr result;
    predicate.Eval(context, result);
    EXPECT_TRUE(predicate.calls.empty());
    EXPECT_EQ(result->size(), 64);
    context.clear_bitmap_input();
    predicate.Eval(context, result);
    ASSERT_EQ(predicate.visited.size(), 64);
    EXPECT_EQ(predicate.visited.front(), 64);
}

TEST_F(FilterMapExprTest, ConjunctionRestoresIncomingMaskOnException) {
    QueryContext query("exception_scope", nullptr, 129, 0);
    ExecContext exec(&query);
    EvalCtx context(&exec);
    auto mask = FilterMap::Adaptive(64, false, 8);
    mask.set(1);
    context.set_filter_input(mask);
    auto a = std::make_shared<TracePredicate>(129, 5);
    auto b = std::make_shared<TracePredicate>(129, 129);
    b->fail_after_eval = true;
    PhyConjunctFilterExpr conjunction(
        std::vector<ExprPtr>{a, b}, true, nullptr);
    conjunction.MarkNullRejecting();
    VectorPtr result;
    EXPECT_THROW(conjunction.Eval(context, result), std::runtime_error);
    ASSERT_TRUE(context.filter_input());
    EXPECT_EQ(context.filter_input()->count(), 1);
    EXPECT_TRUE(context.filter_input()->test(1));
    EXPECT_EQ(context.get_offset_input(), nullptr);
}

TEST_F(FilterMapExprTest, SameEvalOutputPromotesExactlyBeyondRatioCap) {
    constexpr size_t n = 129; // floor(129 * 0.125) == 16
    for (const size_t accepted : {size_t{16}, size_t{17}}) {
        QueryContext query("cap_boundary", nullptr, n, 0);
        ExecContext exec(&query);
        EvalCtx context(&exec);
        TracePredicate predicate(n, accepted);
        auto map = Evaluate(predicate, context);
        EXPECT_EQ(map.count(), n - accepted);
        EXPECT_EQ(map.capability(), accepted == 16
                                       ? FilterMapCapability::EnumerateOnly
                                       : FilterMapCapability::RandomMembership);
        EXPECT_EQ(predicate.visited.size(), n);
    }
}

}  // namespace
}  // namespace milvus::exec
