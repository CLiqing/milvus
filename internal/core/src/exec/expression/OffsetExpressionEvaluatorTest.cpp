// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "common/Schema.h"
#include "exec/QueryContext.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/OffsetExpressionEvaluator.h"
#include "expr/ITypeExpr.h"
#include "segcore/SegmentSealed.h"
#include "test_utils/DataGen.h"
#include "test_utils/storage_test_utils.h"

namespace milvus::exec {
namespace {

struct EvaluatorFixture {
    static constexpr int32_t kRows = 256;

    explicit EvaluatorFixture(std::vector<int64_t> pattern = {},
                              bool nullable = true)
        : nullable(nullable) {
        schema = std::make_shared<Schema>();
        id_field = schema->AddDebugField("id", DataType::INT64);
        value_field = schema->AddDebugField("value", DataType::INT64, nullable);
        schema->set_primary_field_id(id_field);

        segment = segcore::CreateSealedSegment(schema);
        auto insert_data = std::make_unique<InsertRecordProto>();
        std::vector<int64_t> ids(kRows);
        values.resize(kRows);
        for (int32_t row = 0; row < kRows; ++row) {
            ids[row] = row;
            values[row] =
                pattern.empty() ? row - 100 : pattern[row % pattern.size()];
        }
        segcore::InsertCol(insert_data.get(), ids, (*schema)[id_field], false);
        // For a nullable field random_valid=false produces the deterministic
        // validity pattern row % 2 == 0.
        segcore::InsertCol(
            insert_data.get(), values, (*schema)[value_field], false);

        segcore::GeneratedData data;
        data.schema_ = schema;
        data.raw_ = insert_data.release();
        data.raw_->set_num_rows(kRows);
        for (int32_t row = 0; row < kRows; ++row) {
            data.row_ids_.push_back(row);
            data.timestamps_.push_back(row);
        }
        LoadGeneratedDataIntoSegment(data, segment.get(), true);

        query_context = std::make_unique<QueryContext>(
            "offset-expression-test", segment.get(), kRows, MAX_TIMESTAMP);
        exec_context = std::make_unique<ExecContext>(query_context.get());
    }

    expr::TypedExprPtr
    ModLessThan(int64_t divisor, int64_t threshold) const {
        return ArithmeticLessThan(
            proto::plan::ArithOpType::Mod, divisor, threshold);
    }

    expr::TypedExprPtr
    ArithmeticLessThan(proto::plan::ArithOpType operation,
                       int64_t operand,
                       int64_t threshold) const {
        proto::plan::GenericValue rhs;
        rhs.set_int64_val(operand);
        proto::plan::GenericValue target;
        target.set_int64_val(threshold);
        return std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            expr::ColumnInfo(value_field, DataType::INT64, {}, nullable),
            proto::plan::OpType::LessThan,
            operation,
            target,
            rhs);
    }

    expr::TypedExprPtr
    GreaterThan(int64_t threshold) const {
        proto::plan::GenericValue target;
        target.set_int64_val(threshold);
        return std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(value_field, DataType::INT64, {}, nullable),
            proto::plan::OpType::GreaterThan,
            target);
    }

    bool nullable;
    std::vector<int64_t> values;
    std::shared_ptr<Schema> schema;
    FieldId id_field;
    FieldId value_field;
    segcore::SegmentSealedUPtr segment;
    std::unique_ptr<QueryContext> query_context;
    std::unique_ptr<ExecContext> exec_context;
};

OffsetExpressionTruth
EvaluateReference(ExprSet& expressions,
                  ExecContext* exec_context,
                  const int32_t* row_ids,
                  uint32_t count,
                  uint64_t active_mask) {
    OffsetVector compact;
    std::vector<uint32_t> lanes;
    for (uint32_t lane = 0; lane < count; ++lane) {
        if ((active_mask & (uint64_t{1} << lane)) != 0) {
            compact.push_back(row_ids[lane]);
            lanes.push_back(lane);
        }
    }
    if (compact.empty()) {
        return {};
    }

    EvalCtx eval_context(exec_context, &compact);
    std::vector<VectorPtr> results;
    expressions.Eval(0, 1, true, eval_context, results);
    auto output = std::dynamic_pointer_cast<ColumnVector>(results.at(0));
    EXPECT_NE(output, nullptr);
    TargetBitmapView data(output->GetRawData(), output->size());
    TargetBitmapView valid(output->GetValidRawData(), output->size());

    OffsetExpressionTruth truth;
    for (size_t compact_lane = 0; compact_lane < lanes.size(); ++compact_lane) {
        const auto lane = lanes[compact_lane];
        if (valid[compact_lane]) {
            truth.known_mask |= uint64_t{1} << lane;
            if (data[compact_lane]) {
                truth.true_mask |= uint64_t{1} << lane;
            }
        }
    }
    return truth;
}

void
ExpectEvaluatorMatchesExprSet(const expr::TypedExprPtr& expression,
                              EvaluatorFixture& fixture) {
    std::vector<expr::TypedExprPtr> expressions{expression};
    PreparedOffsetExpressionEvaluator prepared(
        expression, fixture.exec_context.get(), /*null_rejecting=*/false);
    auto workspace = prepared.CreateWorkspace();
    ASSERT_TRUE(workspace->SupportsOffsetInput());
    ExprSet reference(
        expressions, fixture.exec_context.get(), /*null_rejecting=*/false);

    std::array<int32_t, 64> row_ids{};
    for (uint32_t lane = 0; lane < row_ids.size(); ++lane) {
        row_ids[lane] = static_cast<int32_t>((lane * 7 + 3) % fixture.kRows);
    }
    for (const uint32_t count : {1U, 8U, 16U, 32U, 64U}) {
        const auto active =
            count == 64 ? ~uint64_t{0} : (uint64_t{1} << count) - 1;
        const auto expected = EvaluateReference(reference,
                                                fixture.exec_context.get(),
                                                row_ids.data(),
                                                count,
                                                active);
        const auto actual =
            workspace->EvalTruthBatch(row_ids.data(), count, active);
        EXPECT_EQ(actual.true_mask, expected.true_mask) << "count=" << count;
        EXPECT_EQ(actual.known_mask, expected.known_mask) << "count=" << count;
    }

    // Inactive lanes are compacted before ExprSet sees the offsets.  Invalid
    // IDs in those lanes therefore cannot accidentally access segment data.
    row_ids.fill(-1);
    row_ids[1] = 12;
    row_ids[7] = 18;
    const uint64_t sparse_active = (uint64_t{1} << 1) | (uint64_t{1} << 7);
    const auto expected = EvaluateReference(reference,
                                            fixture.exec_context.get(),
                                            row_ids.data(),
                                            8,
                                            sparse_active);
    const auto actual =
        workspace->EvalTruthBatch(row_ids.data(), 8, sparse_active);
    EXPECT_EQ(actual.true_mask, expected.true_mask);
    EXPECT_EQ(actual.known_mask, expected.known_mask);
}

}  // namespace

TEST(OffsetExpressionEvaluatorTest, ReusesExprSetForModUnaryAndLogicalTree) {
    EvaluatorFixture fixture;
    const auto mod = fixture.ModLessThan(5, 3);
    const auto unary = fixture.GreaterThan(-20);
    const auto logical = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or,
        std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, mod, unary),
        fixture.GreaterThan(120));

    ExpectEvaluatorMatchesExprSet(mod, fixture);
    ExpectEvaluatorMatchesExprSet(unary, fixture);
    ExpectEvaluatorMatchesExprSet(logical, fixture);
    ExpectEvaluatorMatchesExprSet(
        std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, logical),
        fixture);
}

TEST(OffsetExpressionEvaluatorTest, GivesEachConcurrentWorkerPrivateExprState) {
    EvaluatorFixture fixture;
    auto prepared = std::make_shared<PreparedOffsetExpressionEvaluator>(
        fixture.ModLessThan(7, 4), fixture.exec_context.get(), false);

    using Batches = std::vector<OffsetExpressionTruth>;
    std::vector<std::future<Batches>> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.emplace_back(
            std::async(std::launch::async, [prepared, worker]() {
                auto workspace = prepared->CreateWorkspace();
                Batches actual;
                for (int iteration = 0; iteration < 100; ++iteration) {
                    std::array<int32_t, 64> ids{};
                    const auto count = (iteration * 7 + worker) % 65;
                    uint64_t active = 0;
                    for (int lane = 0; lane < count; ++lane) {
                        // Unsorted, repeated IDs, variable tails and holes.
                        ids[lane] =
                            ((lane % 17) * 97 + iteration * 13 + worker) % 256;
                        if ((lane + iteration) % 3 != 0) {
                            active |= uint64_t{1} << lane;
                        }
                    }
                    actual.push_back(
                        workspace->EvalTruthBatch(ids.data(), count, active));
                }
                return actual;
            }));
    }
    for (int worker = 0; worker < workers.size(); ++worker) {
        const auto actual = workers[worker].get();
        ASSERT_EQ(actual.size(), 100);
        for (int iteration = 0; iteration < actual.size(); ++iteration) {
            OffsetExpressionTruth expected;
            const auto count = (iteration * 7 + worker) % 65;
            for (int lane = 0; lane < count; ++lane) {
                const auto id =
                    ((lane % 17) * 97 + iteration * 13 + worker) % 256;
                if ((lane + iteration) % 3 != 0 && id % 2 == 0) {
                    expected.known_mask |= uint64_t{1} << lane;
                    if ((id - 100) % 7 < 4) {
                        expected.true_mask |= uint64_t{1} << lane;
                    }
                }
            }
            EXPECT_EQ(actual[iteration].known_mask, expected.known_mask)
                << worker << ":" << iteration;
            EXPECT_EQ(actual[iteration].true_mask, expected.true_mask)
                << worker << ":" << iteration;
        }
    }
}

TEST(OffsetExpressionEvaluatorTest, ExactTruthAndNullRejectingAcceptance) {
    EvaluatorFixture fixture;
    proto::plan::GenericValue zero;
    zero.set_int64_val(0);
    const auto always_false = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(fixture.id_field, DataType::INT64),
        proto::plan::OpType::LessThan,
        zero);
    const auto root = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And,
        fixture.GreaterThan(0),
        always_false);
    const auto negated = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, root);
    std::array<int32_t, 4> ids{1, 150, 3, 100};
    for (const auto& expression :
         {expr::TypedExprPtr(root), expr::TypedExprPtr(negated)}) {
        PreparedOffsetExpressionEvaluator precise(
            expression, fixture.exec_context.get(), false);
        PreparedOffsetExpressionEvaluator rejecting(
            expression, fixture.exec_context.get(), true);
        auto exact = precise.CreateWorkspace();
        auto accepted = rejecting.CreateWorkspace();
        const auto truth = exact->EvalTruthBatch(ids.data(), 4, 15);
        EXPECT_EQ(truth.known_mask, 15);
        EXPECT_EQ(truth.true_mask, expression == root ? 0 : 15);
        EXPECT_EQ(accepted->EvalAcceptedBatch(ids.data(), 4, 15),
                  truth.accepted_mask());
        EXPECT_ANY_THROW(accepted->EvalTruthBatch(ids.data(), 4, 15));
    }
}

TEST(OffsetExpressionEvaluatorTest, RejectsInvalidContractAndSkipsEmptyBatch) {
    EvaluatorFixture fixture;
    EXPECT_ANY_THROW(
        PreparedOffsetExpressionEvaluator(nullptr, fixture.exec_context.get()));
    EXPECT_ANY_THROW(
        PreparedOffsetExpressionEvaluator(fixture.ModLessThan(5, 3), nullptr));
    auto workspace = PreparedOffsetExpressionEvaluator(
                         fixture.ModLessThan(5, 3), fixture.exec_context.get())
                         .CreateWorkspace();
    EXPECT_EQ(workspace->EvalTruthBatch(nullptr, 0, 0).known_mask, 0);
    const std::array<int32_t, 1> ids{-1};
    EXPECT_EQ(workspace->EvalAcceptedBatch(ids.data(), 1, 0), 0);
    EXPECT_ANY_THROW(workspace->EvalTruthBatch(nullptr, 1, 1));
    EXPECT_ANY_THROW(workspace->EvalTruthBatch(ids.data(), 65, 0));
    EXPECT_ANY_THROW(workspace->EvalTruthBatch(ids.data(), 1, 2));
}

TEST(OffsetExpressionEvaluatorTest, NativeIterativeBatchHasNo64LaneLimit) {
    EvaluatorFixture fixture;
    auto workspace = PreparedOffsetExpressionEvaluator(
                         fixture.ModLessThan(7, 3), fixture.exec_context.get())
                         .CreateWorkspace();
    OffsetVector ids;
    for (int lane = 0; lane < 193; ++lane) {
        ids.push_back((lane * 97 + 3) % fixture.kRows);
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto result = workspace->EvalOffsets(ids);
        ASSERT_EQ(result->size(), ids.size());
        TargetBitmapView data(result->GetRawData(), result->size());
        TargetBitmapView valid(result->GetValidRawData(), result->size());
        for (size_t lane = 0; lane < ids.size(); ++lane) {
            EXPECT_EQ(valid[lane], ids[lane] % 2 == 0);
            if (valid[lane]) {
                EXPECT_EQ(data[lane], (ids[lane] - 100) % 7 < 3);
            }
        }
    }
}

TEST(OffsetExpressionEvaluatorTest,
     ArithmeticOffsetMatchesSequentialDefinedDomain) {
    using Op = proto::plan::ArithOpType;
    const auto low = std::numeric_limits<int64_t>::min();
    const auto high = std::numeric_limits<int64_t>::max();
    // Test extrema without executing signed overflow (undefined upstream).
    // Negative MOD and division follow C++ truncation toward zero.
    const std::vector<std::pair<Op, int64_t>> cases{{Op::Add, 0},
                                                    {Op::Sub, 0},
                                                    {Op::Mul, 1},
                                                    {Op::Div, 7},
                                                    {Op::Mod, 7},
                                                    {Op::Add, 7},
                                                    {Op::Sub, 7},
                                                    {Op::Mul, 7},
                                                    {Op::Div, -7},
                                                    {Op::Mod, -7}};
    for (const auto& [operation, operand] : cases) {
        const bool needs_headroom =
            operand == 7 && (operation == Op::Add || operation == Op::Sub ||
                             operation == Op::Mul);
        for (bool nullable : {false, true}) {
            EvaluatorFixture fixture({needs_headroom ? low / 16 : low,
                                      needs_headroom ? high / 16 : high,
                                      -8,
                                      8,
                                      -7,
                                      7,
                                      -1,
                                      0,
                                      1},
                                     nullable);
            const auto expression =
                fixture.ArithmeticLessThan(operation, operand, 0);
            ExprSet sequential({expression}, fixture.exec_context.get(), false);
            EvalCtx eval(fixture.exec_context.get());
            std::vector<VectorPtr> results;
            sequential.Eval(0, 1, true, eval, results);
            ASSERT_EQ(results.size(), 1);
            auto output = std::dynamic_pointer_cast<ColumnVector>(results[0]);
            ASSERT_NE(output, nullptr);
            ASSERT_EQ(output->size(), fixture.kRows);
            TargetBitmapView data(output->GetRawData(), output->size());
            TargetBitmapView valid(output->GetValidRawData(), output->size());
            auto workspace = PreparedOffsetExpressionEvaluator(
                                 expression, fixture.exec_context.get())
                                 .CreateWorkspace();
            for (int start = 0; start < fixture.kRows; start += 32) {
                std::array<int32_t, 32> ids{};
                for (int lane = 0; lane < ids.size(); ++lane) {
                    ids[lane] = start + lane;
                }
                const auto actual = workspace->EvalTruthBatch(
                    ids.data(), ids.size(), 0xffffffff);
                for (int lane = 0; lane < ids.size(); ++lane) {
                    const auto id = ids[lane];
                    const bool known = !nullable || id % 2 == 0;
                    __int128 value = fixture.values[id];
                    switch (operation) {
                        case Op::Add:
                            value += operand;
                            break;
                        case Op::Sub:
                            value -= operand;
                            break;
                        case Op::Mul:
                            value *= operand;
                            break;
                        case Op::Div:
                            value /= operand;
                            break;
                        case Op::Mod:
                            value %= operand;
                            break;
                        default:
                            FAIL() << "unexpected test operation";
                    }
                    const bool accepted = known && value < 0;
                    EXPECT_EQ(valid[id], known);
                    EXPECT_EQ(bool(valid[id] && data[id]), accepted);
                    EXPECT_EQ(bool(actual.known_mask & (uint64_t{1} << lane)),
                              known);
                    EXPECT_EQ(bool(actual.true_mask & (uint64_t{1} << lane)),
                              accepted);
                }
            }
        }
    }
}

TEST(OffsetExpressionEvaluatorTest,
     ArithmeticZeroDivisorStillRaisesQueryError) {
    using Op = proto::plan::ArithOpType;
    EvaluatorFixture fixture({}, false);
    for (const auto operation : {Op::Div, Op::Mod}) {
        const auto expression = fixture.ArithmeticLessThan(operation, 0, 0);
        ExprSet sequential({expression}, fixture.exec_context.get(), false);
        EvalCtx eval(fixture.exec_context.get());
        std::vector<VectorPtr> results;
        EXPECT_ANY_THROW(sequential.Eval(0, 1, true, eval, results));
        auto workspace = PreparedOffsetExpressionEvaluator(
                             expression, fixture.exec_context.get())
                             .CreateWorkspace();
        const int32_t id = 0;
        EXPECT_ANY_THROW(workspace->EvalAcceptedBatch(&id, 1, 1));
    }
}

}  // namespace milvus::exec
