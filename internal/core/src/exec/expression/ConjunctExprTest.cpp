// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/Common.h"
#include "common/Types.h"
#include "common/QueryInfo.h"
#include "common/Vector.h"
#include "exec/QueryContext.h"
#include "exec/expression/ConjunctExpr.h"
#include "exec/expression/EvalCtx.h"

namespace milvus::exec {
namespace {

class FixedBitmapExpr : public Expr {
 public:
    FixedBitmapExpr(TargetBitmap data, TargetBitmap valid)
        : Expr(DataType::BOOL, {}, "FixedBitmapExpr", nullptr),
          data_(std::move(data)),
          valid_(std::move(valid)) {
    }

    void
    Eval(EvalCtx&, VectorPtr& result) override {
        ++eval_count_;
        result = std::make_shared<ColumnVector>(data_.clone(), valid_.clone());
    }

    void
    MoveCursor() override {
        ++move_count_;
    }

    std::string
    ToString() const override {
        return "FixedBitmapExpr";
    }

    std::optional<milvus::expr::ColumnInfo>
    GetColumnInfo() const override {
        return std::nullopt;
    }

    int eval_count_ = 0;
    int move_count_ = 0;

 private:
    TargetBitmap data_;
    TargetBitmap valid_;
};

class FixedSparseProducerExpr : public Expr {
 public:
    explicit FixedSparseProducerExpr(std::vector<int32_t> ids)
        : Expr(DataType::BOOL, {}, "FixedSparseProducerExpr", nullptr),
          ids_(std::make_shared<std::vector<int32_t>>(std::move(ids))) {
    }

    std::shared_ptr<std::vector<int32_t>>
    TryGetNativeValidIds() override {
        return ids_;
    }

    bool
    CanApplySparseFilter(EvalCtx&, bool has_sparse_input, int64_t) override {
        return !has_sparse_input;
    }

    std::string
    ToString() const override {
        return "FixedSparseProducerExpr";
    }

    std::optional<milvus::expr::ColumnInfo>
    GetColumnInfo() const override {
        return std::nullopt;
    }

 private:
    std::shared_ptr<std::vector<int32_t>> ids_;
};

class CountingAllTrueNativeExpr : public Expr {
 public:
    explicit CountingAllTrueNativeExpr(int64_t universe)
        : Expr(DataType::BOOL, {}, "CountingAllTrueNativeExpr", nullptr),
          universe_(universe) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override {
        ++eval_count_;
        const auto* offsets = context.get_offset_input();
        const auto rows = offsets == nullptr ? static_cast<size_t>(universe_)
                                             : offsets->size();
        result = std::make_shared<ColumnVector>(TargetBitmap(rows, true),
                                                TargetBitmap(rows, true));
    }

    bool
    CanApplySparseFilter(EvalCtx&, bool has_sparse_input, int64_t) override {
        return !has_sparse_input;
    }

    std::optional<FilterMap>
    TryApplySparseFilter(EvalCtx&,
                         std::optional<FilterMap> input,
                         int64_t max_cardinality) override {
        ++apply_count_;
        last_cap_ = max_cardinality;
        if (input.has_value()) {
            return std::nullopt;
        }
        auto ids = std::make_shared<std::vector<int32_t>>();
        ids->reserve(universe_);
        for (int32_t id = 0; id < universe_; ++id) {
            ids->push_back(id);
        }
        return FilterMap::FromUnsetIds(
            universe_, std::move(ids), static_cast<size_t>(max_cardinality));
    }

    std::string
    ToString() const override {
        return "CountingAllTrueNativeExpr";
    }

    std::optional<milvus::expr::ColumnInfo>
    GetColumnInfo() const override {
        return std::nullopt;
    }

    int apply_count_ = 0;
    int eval_count_ = 0;
    int64_t last_cap_ = -1;

 private:
    int64_t universe_;
};

class OffsetMembershipExpr : public Expr {
 public:
    explicit OffsetMembershipExpr(std::unordered_set<int32_t> accepted)
        : Expr(DataType::BOOL, {}, "OffsetMembershipExpr", nullptr),
          accepted_(std::move(accepted)) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override {
        ++eval_count_;
        auto* offsets = context.get_offset_input();
        ASSERT_NE(offsets, nullptr);
        seen_offsets_ = offsets->size();
        TargetBitmap data(offsets->size(), false);
        TargetBitmap valid(offsets->size(), true);
        for (size_t i = 0; i < offsets->size(); ++i) {
            data[i] = accepted_.contains((*offsets)[i]);
        }
        result =
            std::make_shared<ColumnVector>(std::move(data), std::move(valid));
    }

    std::shared_ptr<std::vector<int32_t>>
    TryFilterNativeValidIds(EvalCtx&, std::span<const int32_t> input) override {
        seen_offsets_ = input.size();
        auto output = std::make_shared<std::vector<int32_t>>();
        output->reserve(input.size());
        for (const auto id : input) {
            if (accepted_.contains(id)) {
                output->push_back(id);
            }
        }
        return output;
    }

    bool
    CanApplySparseFilter(EvalCtx&, bool has_sparse_input, int64_t) override {
        return has_sparse_input;
    }

    std::string
    ToString() const override {
        return "OffsetMembershipExpr";
    }

    std::optional<milvus::expr::ColumnInfo>
    GetColumnInfo() const override {
        return std::nullopt;
    }

    size_t seen_offsets_ = 0;
    int eval_count_ = 0;

 private:
    std::unordered_set<int32_t> accepted_;
};

// Test double for the representation-neutral predicate contract.  It can
// seed either representation and can consume Sparse candidates.
class AdaptiveMembershipExpr : public Expr {
 public:
    AdaptiveMembershipExpr(std::vector<int32_t> accepted,
                           int64_t universe,
                           bool seed_dense)
        : Expr(DataType::BOOL, {}, "AdaptiveMembershipExpr", nullptr),
          accepted_(accepted.begin(), accepted.end()),
          universe_(universe),
          seed_dense_(seed_dense) {
    }

    void
    Eval(EvalCtx&, VectorPtr& result) override {
        ++eval_count_;
        TargetBitmap data(universe_, false);
        TargetBitmap valid(universe_, true);
        for (const auto id : accepted_) {
            data[id] = true;
        }
        result =
            std::make_shared<ColumnVector>(std::move(data), std::move(valid));
    }

    std::optional<FilterMap>
    TryApplySparseFilter(EvalCtx& context,
                         std::optional<FilterMap> input,
                         int64_t max_cardinality) override {
        ++apply_count_;
        last_cap_ = max_cardinality;
        if (input.has_value()) {
            return Expr::TryApplySparseFilter(
                context, std::move(input), max_cardinality);
        }
        if (seed_dense_) {
            auto filtered = std::make_shared<TargetBitmap>(universe_, true);
            for (const auto id : accepted_) {
                (*filtered)[id] = false;
            }
            return FilterMap::FromDense(std::move(filtered));
        }
        auto ids = std::make_shared<std::vector<int32_t>>();
        ids->reserve(accepted_.size());
        for (int64_t id = 0; id < universe_; ++id) {
            if (accepted_.contains(static_cast<int32_t>(id))) {
                ids->push_back(static_cast<int32_t>(id));
            }
        }
        return FilterMap::FromUnsetIds(
            universe_, std::move(ids), static_cast<size_t>(max_cardinality));
    }

    bool
    CanApplySparseFilter(EvalCtx&, bool, int64_t) override {
        return true;
    }

    std::shared_ptr<std::vector<int32_t>>
    TryFilterNativeValidIds(EvalCtx&, std::span<const int32_t> input) override {
        ++filter_count_;
        auto output = std::make_shared<std::vector<int32_t>>();
        for (const auto id : input) {
            if (accepted_.contains(id)) {
                output->push_back(id);
            }
        }
        return output;
    }

    std::string
    ToString() const override {
        return "AdaptiveMembershipExpr";
    }

    std::optional<milvus::expr::ColumnInfo>
    GetColumnInfo() const override {
        return std::nullopt;
    }

    int apply_count_ = 0;
    int eval_count_ = 0;
    int filter_count_ = 0;
    int64_t last_cap_ = -1;

 private:
    std::unordered_set<int32_t> accepted_;
    int64_t universe_;
    bool seed_dense_;
};

// Each row given as {data, valid}.
std::shared_ptr<FixedBitmapExpr>
FixedRows(std::initializer_list<std::pair<bool, bool>> rows) {
    TargetBitmap data_bitmap(rows.size(), false);
    TargetBitmap valid_bitmap(rows.size(), false);
    size_t i = 0;
    for (const auto& [data, valid] : rows) {
        data_bitmap[i] = data;
        valid_bitmap[i] = valid;
        ++i;
    }
    return std::make_shared<FixedBitmapExpr>(std::move(data_bitmap),
                                             std::move(valid_bitmap));
}

std::shared_ptr<FixedBitmapExpr>
FixedBool(const bool data, const bool valid) {
    return FixedRows({{data, valid}});
}

// A boolean node that is not a conjunction (stand-in for NOT in the
// null-rejection propagation tests).
class PassThroughExpr : public Expr {
 public:
    explicit PassThroughExpr(ExprPtr input)
        : Expr(DataType::BOOL, {std::move(input)}, "PassThroughExpr", nullptr) {
    }

    void
    Eval(EvalCtx& context, VectorPtr& result) override {
        inputs_[0]->Eval(context, result);
    }

    std::string
    ToString() const override {
        return "PassThroughExpr";
    }

    std::optional<milvus::expr::ColumnInfo>
    GetColumnInfo() const override {
        return std::nullopt;
    }
};

}  // namespace

TEST(ConjunctExprTest, AndKeepsUnknownRowsActiveForFollowingFalse) {
    auto unknown = FixedBool(false, false);
    auto true_expr = FixedBool(true, true);
    auto false_expr = FixedBool(false, true);

    std::vector<ExprPtr> inputs{unknown, true_expr, false_expr};
    PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);

    QueryContext query_context("conjunct_test", nullptr, 1, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    VectorPtr result;
    conjunct.Eval(eval_context, result);

    auto output = std::dynamic_pointer_cast<ColumnVector>(result);
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(output->IsBitmap());

    TargetBitmapView data(output->GetRawData(), output->size());
    TargetBitmapView valid(output->GetValidRawData(), output->size());
    ASSERT_EQ(output->size(), 1);
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(valid[0]);

    EXPECT_EQ(unknown->eval_count_, 1);
    EXPECT_EQ(true_expr->eval_count_, 1);
    EXPECT_EQ(false_expr->eval_count_, 1);
    EXPECT_EQ(false_expr->move_count_, 0);
}

TEST(ConjunctExprTest, NullRejectingAndSkipsFollowingForAllUnknown) {
    auto unknown = FixedBool(false, false);
    auto heavy = FixedBool(true, true);

    std::vector<ExprPtr> inputs{unknown, heavy};
    auto conjunct = std::make_shared<PhyConjunctFilterExpr>(
        std::move(inputs), true, nullptr);
    conjunct->MarkNullRejecting();

    QueryContext query_context("conjunct_test", nullptr, 1, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    VectorPtr result;
    conjunct->Eval(eval_context, result);

    auto output = std::dynamic_pointer_cast<ColumnVector>(result);
    ASSERT_NE(output, nullptr);
    TargetBitmapView data(output->GetRawData(), output->size());
    TargetBitmapView valid(output->GetValidRawData(), output->size());
    ASSERT_EQ(output->size(), 1);
    // The row stays UNKNOWN, which the null-rejecting consumer treats as
    // FALSE.
    EXPECT_FALSE(data[0]);
    EXPECT_FALSE(valid[0]);

    // The UNKNOWN row can never become TRUE under AND, so the following
    // expression must be skipped, not evaluated.
    EXPECT_EQ(unknown->eval_count_, 1);
    EXPECT_EQ(heavy->eval_count_, 0);
    EXPECT_EQ(heavy->move_count_, 1);
}

TEST(ConjunctExprTest, NullRejectingAndStillEvaluatesForTrueRows) {
    auto first = FixedRows({{false, false}, {true, true}});
    auto second = FixedRows({{true, true}, {false, true}});

    std::vector<ExprPtr> inputs{first, second};
    auto conjunct = std::make_shared<PhyConjunctFilterExpr>(
        std::move(inputs), true, nullptr);
    conjunct->MarkNullRejecting();

    QueryContext query_context("conjunct_test", nullptr, 2, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    VectorPtr result;
    conjunct->Eval(eval_context, result);

    // Row 1 is TRUE after the first expression, so the second must run.
    EXPECT_EQ(second->eval_count_, 1);
    EXPECT_EQ(second->move_count_, 0);

    auto output = std::dynamic_pointer_cast<ColumnVector>(result);
    ASSERT_NE(output, nullptr);
    TargetBitmapView data(output->GetRawData(), output->size());
    TargetBitmapView valid(output->GetValidRawData(), output->size());
    ASSERT_EQ(output->size(), 2);
    // Row 1: TRUE AND FALSE = FALSE.
    EXPECT_FALSE(data[1]);
    EXPECT_TRUE(valid[1]);
    // Row 0 is excluded either way (UNKNOWN or FALSE).
    EXPECT_FALSE(data[0] && valid[0]);
}

TEST(ConjunctExprTest, NullRejectingMatchesDefaultIncludedSet) {
    // The included set (definite TRUE rows) must be identical with and
    // without null-rejection; only excluded rows may differ between
    // UNKNOWN and FALSE.
    auto build_and_eval = [](bool null_rejecting) {
        auto first = FixedRows({{false, false}, {true, true}, {false, true}});
        auto second = FixedRows({{false, true}, {true, true}, {true, true}});
        std::vector<ExprPtr> inputs{first, second};
        auto conjunct = std::make_shared<PhyConjunctFilterExpr>(
            std::move(inputs), true, nullptr);
        if (null_rejecting) {
            conjunct->MarkNullRejecting();
        }
        QueryContext query_context("conjunct_test", nullptr, 3, 0);
        ExecContext exec_context(&query_context);
        EvalCtx eval_context(&exec_context);
        VectorPtr result;
        conjunct->Eval(eval_context, result);
        auto output = std::dynamic_pointer_cast<ColumnVector>(result);
        TargetBitmapView data(output->GetRawData(), output->size());
        TargetBitmapView valid(output->GetValidRawData(), output->size());
        std::vector<bool> included;
        for (size_t i = 0; i < output->size(); ++i) {
            included.push_back(data[i] && valid[i]);
        }
        return included;
    };

    EXPECT_EQ(build_and_eval(true), build_and_eval(false));
}

TEST(ConjunctExprTest, NullRejectingOrKeepsUnknownRowsActive) {
    // Under OR an UNKNOWN row can still become TRUE (UNKNOWN OR TRUE =
    // TRUE), so null-rejection must not skip the following expression.
    auto unknown = FixedBool(false, false);
    auto true_expr = FixedBool(true, true);

    std::vector<ExprPtr> inputs{unknown, true_expr};
    auto disjunct = std::make_shared<PhyConjunctFilterExpr>(
        std::move(inputs), false, nullptr);
    disjunct->MarkNullRejecting();

    QueryContext query_context("conjunct_test", nullptr, 1, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    VectorPtr result;
    disjunct->Eval(eval_context, result);

    EXPECT_EQ(true_expr->eval_count_, 1);

    auto output = std::dynamic_pointer_cast<ColumnVector>(result);
    ASSERT_NE(output, nullptr);
    TargetBitmapView data(output->GetRawData(), output->size());
    TargetBitmapView valid(output->GetValidRawData(), output->size());
    // UNKNOWN OR TRUE = TRUE: the row must be included.
    EXPECT_TRUE(data[0]);
    EXPECT_TRUE(valid[0]);
}

TEST(ConjunctExprTest, AdaptiveSparseOutputIsAndOnlyInPhaseOne) {
    std::vector<ExprPtr> and_inputs{FixedBool(true, true),
                                    FixedBool(true, true)};
    PhyConjunctFilterExpr conjunction(std::move(and_inputs), true, nullptr);
    EXPECT_TRUE(conjunction.IsAnd());

    std::vector<ExprPtr> or_inputs{FixedBool(false, true),
                                   FixedBool(true, true)};
    PhyConjunctFilterExpr disjunction(std::move(or_inputs), false, nullptr);
    EXPECT_FALSE(disjunction.IsAnd());
}

TEST(ConjunctExprTest, AdaptiveRequestAndThresholdParsing) {
    SearchInfo default_info;
    EXPECT_FALSE(default_info.RequestsAdaptiveFilterRepresentation());
    EXPECT_EQ(default_info.SparseResultMaxCardinality(1000), 1000);

    SearchInfo adaptive_info;
    adaptive_info.search_params_ = knowhere::Json{
        {"filter_result_representation", "adaptive"},
        {"sparse_result_max_cardinality", 4096},
    };
    EXPECT_TRUE(adaptive_info.RequestsAdaptiveFilterRepresentation());
    EXPECT_TRUE(adaptive_info.UseSparseFilterRepresentation());
    EXPECT_EQ(adaptive_info.SparseResultMaxCardinality(1000), 1000);
    EXPECT_EQ(adaptive_info.SparseResultMaxCardinality(9000), 4096);

    SearchInfo compatibility_info;
    compatibility_info.search_params_ =
        knowhere::Json{{"filter_result_representation", "sparse"}};
    EXPECT_TRUE(compatibility_info.RequestsAdaptiveFilterRepresentation());

    const auto invalid_cap =
        static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    SearchInfo invalid_request;
    invalid_request.search_params_ = knowhere::Json{
        {"filter_result_representation", "adaptive"},
        {"sparse_result_max_cardinality", invalid_cap},
    };
    EXPECT_ANY_THROW(invalid_request.SparseResultMaxCardinality(1000));
    EXPECT_ANY_THROW(default_info.SparseResultMaxCardinality(invalid_cap));
    EXPECT_ANY_THROW(
        SetSparseFilterResultConfig(true, invalid_cap, 50000, 0.009));
}

TEST(ConjunctExprTest, SparseSelectorUsesPerSegmentRatioAndAbsoluteCap) {
    constexpr int64_t kAbsoluteCap = 6000;
    constexpr int64_t kMinRows = 50000;
    constexpr double kRatio = 0.006;

    EXPECT_EQ(ComputeSparseFilterResultCap(
                  kMinRows - 1, kAbsoluteCap, kMinRows, kRatio),
              0);
    EXPECT_EQ(
        ComputeSparseFilterResultCap(kMinRows, kAbsoluteCap, kMinRows, kRatio),
        300);
    EXPECT_EQ(ComputeSparseFilterResultCap(
                  kMinRows + 1, kAbsoluteCap, kMinRows, kRatio),
              300);
    EXPECT_EQ(
        ComputeSparseFilterResultCap(100000, kAbsoluteCap, kMinRows, kRatio),
        600);
    EXPECT_EQ(
        ComputeSparseFilterResultCap(250000, kAbsoluteCap, kMinRows, kRatio),
        1500);
    EXPECT_EQ(
        ComputeSparseFilterResultCap(1000000, kAbsoluteCap, kMinRows, kRatio),
        6000);
    EXPECT_EQ(
        ComputeSparseFilterResultCap(10000000, kAbsoluteCap, kMinRows, kRatio),
        kAbsoluteCap);

    // A request-level absolute cap may make the selector more conservative,
    // but it cannot bypass the global ratio or minimum-segment guards.
    EXPECT_EQ(ComputeSparseFilterResultCap(1000000, 4096, kMinRows, kRatio),
              4096);
    EXPECT_EQ(ComputeSparseFilterResultCap(49999, 4096, kMinRows, kRatio), 0);
}

TEST(ConjunctExprTest, OffsetInputErasesUnmaterializedReservedLikeSlot) {
    // ReorderConjunctExpr reserves index inputs_.size() for a runtime
    // PhyLikeConjunctExpr and records the LIKE positions via SetLikeIndices.
    // The batch-ngram path cannot be used with an offset input, so the init
    // block must erase the reserved slot on that path too — otherwise both
    // the Eval loop and the early-exit path (SkipFollowingExprs) would
    // index inputs_ past its end.
    auto first = FixedBool(false, true);  // definite FALSE -> early exit
    auto second = FixedBool(true, true);

    std::vector<ExprPtr> inputs{first, second};
    auto conjunct = std::make_shared<PhyConjunctFilterExpr>(
        std::move(inputs), true, nullptr);
    conjunct->SetLikeIndices({0, 1});
    conjunct->Reorder({0, 1, 2});

    QueryContext query_context("conjunct_test", nullptr, 1, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);
    OffsetVector offsets;
    offsets.push_back(0);
    eval_context.set_offset_input(&offsets);

    VectorPtr result;
    conjunct->Eval(eval_context, result);

    auto output = std::dynamic_pointer_cast<ColumnVector>(result);
    ASSERT_NE(output, nullptr);
    TargetBitmapView data(output->GetRawData(), output->size());
    TargetBitmapView valid(output->GetValidRawData(), output->size());
    ASSERT_EQ(output->size(), 1);
    EXPECT_FALSE(data[0]);
    EXPECT_TRUE(valid[0]);
    // The definite-FALSE first input early-exits: the second expression is
    // skipped via MoveCursor and the erased reserved slot is never touched.
    EXPECT_EQ(first->eval_count_, 1);
    EXPECT_EQ(second->eval_count_, 0);
    EXPECT_EQ(second->move_count_, 1);
}

TEST(ConjunctExprTest, MarkNullRejectingStopsAtNonConjunctNodes) {
    std::vector<ExprPtr> inner_inputs{FixedBool(true, true),
                                      FixedBool(true, true)};
    auto inner_and = std::make_shared<PhyConjunctFilterExpr>(
        std::move(inner_inputs), true, nullptr);

    std::vector<ExprPtr> hidden_inputs{FixedBool(true, true),
                                       FixedBool(true, true)};
    auto hidden_and = std::make_shared<PhyConjunctFilterExpr>(
        std::move(hidden_inputs), true, nullptr);
    auto wrapper = std::make_shared<PassThroughExpr>(hidden_and);

    std::vector<ExprPtr> outer_inputs{inner_and, wrapper};
    auto outer_or = std::make_shared<PhyConjunctFilterExpr>(
        std::move(outer_inputs), false, nullptr);

    outer_or->MarkNullRejecting();

    // Propagates through nested AND/OR ...
    EXPECT_TRUE(outer_or->IsNullRejecting());
    EXPECT_TRUE(inner_and->IsNullRejecting());
    // ... but stops at any non-conjunct node (e.g. NOT), where FALSE and
    // UNKNOWN produce different results.
    EXPECT_FALSE(hidden_and->IsNullRejecting());
}

TEST(ConjunctExprTest, SparseAndPassesOnlyAcceptedOffsetsToSecondPredicate) {
    auto producer = std::make_shared<FixedSparseProducerExpr>(
        std::vector<int32_t>{3, 10, 18, 25});
    auto consumer = std::make_shared<OffsetMembershipExpr>(
        std::unordered_set<int32_t>{10, 25});
    std::vector<ExprPtr> inputs{producer, consumer};
    PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);

    QueryContext query_context("sparse_conjunct_test", nullptr, 32, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    const auto ids = conjunct.TryGetNativeValidIds(eval_context);
    ASSERT_NE(ids, nullptr);
    EXPECT_EQ(*ids, (std::vector<int32_t>{10, 25}));
    EXPECT_EQ(consumer->seen_offsets_, 4);
    EXPECT_EQ(consumer->eval_count_, 0);
}

TEST(ConjunctExprTest, SparseAndDoesNotSearchLaterProducer) {
    // Phase one must not evaluate children opportunistically until one happens
    // to produce Sparse.  If the first child cannot produce the representation,
    // the native chain declines and FilterBits uses the established Dense AND
    // path before making the final adaptive representation decision.
    auto first_consumer = std::make_shared<OffsetMembershipExpr>(
        std::unordered_set<int32_t>{10, 25});
    auto second_producer = std::make_shared<FixedSparseProducerExpr>(
        std::vector<int32_t>{3, 10, 18, 25});
    std::vector<ExprPtr> inputs{first_consumer, second_producer};
    PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);

    QueryContext query_context(
        "sparse_conjunct_late_producer_test", nullptr, 32, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    const auto ids = conjunct.TryGetNativeValidIds(eval_context);
    EXPECT_EQ(ids, nullptr);
    EXPECT_EQ(first_consumer->seen_offsets_, 0);
    EXPECT_EQ(first_consumer->eval_count_, 0);
}

TEST(ConjunctExprTest, SparseAndChainsAcceptedOffsetsAcrossPredicates) {
    auto producer = std::make_shared<FixedSparseProducerExpr>(
        std::vector<int32_t>{3, 10, 18, 25});
    auto second = std::make_shared<OffsetMembershipExpr>(
        std::unordered_set<int32_t>{3, 10, 25});
    auto third = std::make_shared<OffsetMembershipExpr>(
        std::unordered_set<int32_t>{10, 25});
    std::vector<ExprPtr> inputs{producer, second, third};
    PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);

    QueryContext query_context("sparse_conjunct_chain_test", nullptr, 32, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    const auto ids = conjunct.TryGetNativeValidIds(eval_context);
    ASSERT_NE(ids, nullptr);
    EXPECT_EQ(*ids, (std::vector<int32_t>{10, 25}));
    EXPECT_EQ(second->seen_offsets_, 4);
    EXPECT_EQ(third->seen_offsets_, 3);
    EXPECT_EQ(second->eval_count_, 0);
    EXPECT_EQ(third->eval_count_, 0);
}

TEST(ConjunctExprTest, SparseApplyPreservesUniverseAcrossNestedAnd) {
    auto producer = std::make_shared<FixedSparseProducerExpr>(
        std::vector<int32_t>{3, 10, 18, 25});
    auto inner_consumer = std::make_shared<OffsetMembershipExpr>(
        std::unordered_set<int32_t>{3, 10, 25});
    std::vector<ExprPtr> inner_inputs{producer, inner_consumer};
    auto inner = std::make_shared<PhyConjunctFilterExpr>(
        std::move(inner_inputs), true, nullptr);
    auto outer_consumer = std::make_shared<OffsetMembershipExpr>(
        std::unordered_set<int32_t>{10, 25});
    std::vector<ExprPtr> outer_inputs{inner, outer_consumer};
    PhyConjunctFilterExpr outer(std::move(outer_inputs), true, nullptr);

    QueryContext query_context("sparse_apply_nested_test", nullptr, 32, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    const auto result =
        outer.TryApplySparseFilter(eval_context, std::nullopt, 1000);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 32);
    EXPECT_EQ(*result->SnapshotUnsetIds(), (std::vector<int32_t>{10, 25}));
    EXPECT_EQ(inner_consumer->seen_offsets_, 4);
    EXPECT_EQ(outer_consumer->seen_offsets_, 3);
}

TEST(ConjunctExprTest,
     SparseAndFallsBackWhenSecondPredicateHasNoSparseConsumer) {
    auto producer = std::make_shared<FixedSparseProducerExpr>(
        std::vector<int32_t>{3, 10, 18, 25});
    auto consumer = FixedBool(true, true);
    std::vector<ExprPtr> inputs{producer, consumer};
    PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);

    QueryContext query_context("sparse_conjunct_fallback_test", nullptr, 32, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    EXPECT_EQ(conjunct.TryGetNativeValidIds(eval_context), nullptr);
    EXPECT_EQ(consumer->eval_count_, 0);
}

TEST(ConjunctExprTest,
     SparsePreflightRunsUnsupportedAndChildrenOnceInEitherOrder) {
    for (const bool native_first : {true, false}) {
        auto native = std::make_shared<CountingAllTrueNativeExpr>(4);
        auto unsupported = FixedRows(
            {{true, true}, {false, true}, {true, true}, {false, true}});
        std::vector<ExprPtr> inputs =
            native_first ? std::vector<ExprPtr>{native, unsupported}
                         : std::vector<ExprPtr>{unsupported, native};
        PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);

        QueryContext query_context("sparse_preflight_fallback", nullptr, 4, 0);
        ExecContext exec_context(&query_context);
        EvalCtx eval_context(&exec_context);

        EXPECT_FALSE(conjunct.CanApplySparseFilter(
            eval_context, /*has_sparse_input=*/false, /*cap=*/1000));
        EXPECT_FALSE(
            conjunct.TryApplySparseFilter(eval_context, std::nullopt, 1000)
                .has_value());
        EXPECT_EQ(native->apply_count_, 0);
        EXPECT_EQ(native->eval_count_, 0);
        EXPECT_EQ(unsupported->eval_count_, 0);

        VectorPtr result;
        conjunct.Eval(eval_context, result);
        auto output = std::dynamic_pointer_cast<ColumnVector>(result);
        ASSERT_NE(output, nullptr);
        ASSERT_EQ(output->size(), 4);
        TargetBitmapView data(output->GetRawData(), output->size());
        TargetBitmapView valid(output->GetValidRawData(), output->size());
        EXPECT_TRUE(data[0]);
        EXPECT_FALSE(data[1]);
        EXPECT_TRUE(data[2]);
        EXPECT_FALSE(data[3]);
        EXPECT_TRUE(valid.all());
        EXPECT_EQ(native->apply_count_, 0);
        EXPECT_EQ(native->eval_count_, 1);
        EXPECT_EQ(unsupported->eval_count_, 1);
    }
}

TEST(ConjunctExprTest, AdaptiveDenseIntermediateMergesNextPredicateOnce) {
    auto first = std::make_shared<AdaptiveMembershipExpr>(
        std::vector<int32_t>{0, 1, 2, 4}, 6, true);
    auto second = std::make_shared<AdaptiveMembershipExpr>(
        std::vector<int32_t>{1, 2, 3}, 6, false);
    std::vector<ExprPtr> inputs{first, second};
    PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);

    QueryContext query_context("adaptive_dense_chain", nullptr, 6, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);
    auto result =
        conjunct.TryApplySparseFilter(eval_context, std::nullopt, 1000);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->capability(), FilterCapability::EnumerateOnly);
    EXPECT_EQ(result->size(), 6);
    EXPECT_EQ(*result->SnapshotUnsetIds(), (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(first->apply_count_, 1);
    EXPECT_EQ(first->eval_count_, 0);
    EXPECT_EQ(second->apply_count_, 1);
    EXPECT_EQ(second->eval_count_, 0);
}

TEST(ConjunctExprTest, AdaptiveAndIsCorrectInEitherPredicateOrder) {
    const auto evaluate = [](bool reverse, bool first_is_dense) {
        auto a = std::make_shared<AdaptiveMembershipExpr>(
            std::vector<int32_t>{0, 1, 2, 4}, 6, first_is_dense && !reverse);
        auto b = std::make_shared<AdaptiveMembershipExpr>(
            std::vector<int32_t>{1, 2, 3}, 6, first_is_dense && reverse);
        std::vector<ExprPtr> inputs =
            reverse ? std::vector<ExprPtr>{b, a} : std::vector<ExprPtr>{a, b};
        PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);
        QueryContext query_context("adaptive_order", nullptr, 6, 0);
        ExecContext exec_context(&query_context);
        EvalCtx eval_context(&exec_context);
        auto result =
            conjunct.TryApplySparseFilter(eval_context, std::nullopt, 1000);
        EXPECT_EQ(a->apply_count_, 1);
        EXPECT_EQ(b->apply_count_, 1);
        EXPECT_EQ(a->eval_count_, 0);
        EXPECT_EQ(b->eval_count_, 0);
        return result;
    };

    for (const bool first_is_dense : {false, true}) {
        const auto ab = evaluate(false, first_is_dense);
        const auto ba = evaluate(true, first_is_dense);
        ASSERT_TRUE(ab.has_value());
        ASSERT_TRUE(ba.has_value());
        ASSERT_EQ(ab->capability(), FilterCapability::EnumerateOnly);
        ASSERT_EQ(ba->capability(), FilterCapability::EnumerateOnly);
        EXPECT_EQ(*ab->SnapshotUnsetIds(), (std::vector<int32_t>{1, 2}));
        EXPECT_EQ(*ab->SnapshotUnsetIds(), *ba->SnapshotUnsetIds());
    }
}

TEST(ConjunctExprTest, AdaptiveAndEvaluatesEachChildOnceAtEveryTestedDepth) {
    for (const size_t depth : {size_t{2}, size_t{4}, size_t{8}}) {
        std::vector<std::shared_ptr<AdaptiveMembershipExpr>> predicates;
        std::vector<ExprPtr> inputs;
        predicates.reserve(depth);
        inputs.reserve(depth);
        for (size_t i = 0; i < depth; ++i) {
            auto predicate = std::make_shared<AdaptiveMembershipExpr>(
                std::vector<int32_t>{1, 2, 3}, 6, false);
            predicates.push_back(predicate);
            inputs.push_back(predicate);
        }
        PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);
        QueryContext query_context("adaptive_depth", nullptr, 6, 0);
        ExecContext exec_context(&query_context);
        EvalCtx eval_context(&exec_context);

        const auto result =
            conjunct.TryApplySparseFilter(eval_context, std::nullopt, 1000);
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(result->capability(), FilterCapability::EnumerateOnly);
        EXPECT_EQ(*result->SnapshotUnsetIds(), (std::vector<int32_t>{1, 2, 3}));
        for (const auto& predicate : predicates) {
            EXPECT_EQ(predicate->apply_count_, 1) << "depth=" << depth;
            EXPECT_EQ(predicate->eval_count_, 0) << "depth=" << depth;
        }
    }
}

TEST(ConjunctExprTest, AdaptiveExecutionKeepsPreflightCapAcrossEntireChain) {
    auto first = std::make_shared<AdaptiveMembershipExpr>(
        std::vector<int32_t>{1, 2, 3}, 6, false);
    auto second = std::make_shared<AdaptiveMembershipExpr>(
        std::vector<int32_t>{2, 3}, 6, false);
    std::vector<ExprPtr> inputs{first, second};
    PhyConjunctFilterExpr conjunct(std::move(inputs), true, nullptr);
    QueryContext query_context("adaptive_cap_propagation", nullptr, 6, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    constexpr int64_t kRequestCap = 37;
    ASSERT_TRUE(conjunct.CanApplySparseFilter(
        eval_context, /*has_sparse_input=*/false, kRequestCap));
    const auto result =
        conjunct.TryApplySparseFilter(eval_context, std::nullopt, kRequestCap);

    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->capability(), FilterCapability::EnumerateOnly);
    EXPECT_EQ(*result->SnapshotUnsetIds(), (std::vector<int32_t>{2, 3}));
    EXPECT_EQ(first->last_cap_, kRequestCap);
    EXPECT_EQ(second->last_cap_, kRequestCap);
}

TEST(ConjunctExprTest, AdaptiveOrAlwaysDeclinesSparseExecution) {
    auto first = std::make_shared<AdaptiveMembershipExpr>(
        std::vector<int32_t>{0}, 4, false);
    auto second = std::make_shared<AdaptiveMembershipExpr>(
        std::vector<int32_t>{1}, 4, false);
    std::vector<ExprPtr> inputs{first, second};
    PhyConjunctFilterExpr disjunction(std::move(inputs), false, nullptr);
    QueryContext query_context("adaptive_or", nullptr, 4, 0);
    ExecContext exec_context(&query_context);
    EvalCtx eval_context(&exec_context);

    EXPECT_FALSE(
        disjunction.TryApplySparseFilter(eval_context, std::nullopt, 1000)
            .has_value());
    EXPECT_EQ(first->apply_count_, 0);
    EXPECT_EQ(second->apply_count_, 0);
}

}  // namespace milvus::exec
