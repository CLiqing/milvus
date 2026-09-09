// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <future>
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

    EvaluatorFixture() {
        schema = std::make_shared<Schema>();
        id_field = schema->AddDebugField("id", DataType::INT64);
        value_field =
            schema->AddDebugField("value", DataType::INT64, true);
        schema->set_primary_field_id(id_field);

        segment = segcore::CreateSealedSegment(schema);
        auto insert_data = std::make_unique<InsertRecordProto>();
        std::vector<int64_t> ids(kRows);
        std::vector<int64_t> values(kRows);
        for (int32_t row = 0; row < kRows; ++row) {
            ids[row] = row;
            values[row] = row - 100;
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
        proto::plan::GenericValue rhs;
        rhs.set_int64_val(divisor);
        proto::plan::GenericValue target;
        target.set_int64_val(threshold);
        return std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            expr::ColumnInfo(value_field, DataType::INT64, {}, true),
            proto::plan::OpType::LessThan,
            proto::plan::ArithOpType::Mod,
            target,
            rhs);
    }

    expr::TypedExprPtr
    GreaterThan(int64_t threshold) const {
        proto::plan::GenericValue target;
        target.set_int64_val(threshold);
        return std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(value_field, DataType::INT64, {}, true),
            proto::plan::OpType::GreaterThan,
            target);
    }

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
        expressions, fixture.exec_context.get(), /*null_rejecting=*/false);
    auto workspace = prepared.CreateWorkspace();
    ASSERT_TRUE(workspace->SupportsOffsetInput());
    ExprSet reference(
        expressions, fixture.exec_context.get(), /*null_rejecting=*/false);

    std::array<int32_t, 64> row_ids{};
    for (uint32_t lane = 0; lane < row_ids.size(); ++lane) {
        row_ids[lane] = static_cast<int32_t>((lane * 7 + 3) % fixture.kRows);
    }
    for (const uint32_t count : {1U, 8U, 16U, 32U, 64U}) {
        const auto active = count == 64 ? ~uint64_t{0}
                                        : (uint64_t{1} << count) - 1;
        const auto expected = EvaluateReference(reference,
                                                fixture.exec_context.get(),
                                                row_ids.data(),
                                                count,
                                                active);
        const auto actual =
            workspace->EvalBatch(row_ids.data(), count, active);
        EXPECT_EQ(actual.true_mask, expected.true_mask) << "count=" << count;
        EXPECT_EQ(actual.known_mask, expected.known_mask)
            << "count=" << count;
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
        workspace->EvalBatch(row_ids.data(), 8, sparse_active);
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
}

TEST(OffsetExpressionEvaluatorTest, GivesEachConcurrentWorkerPrivateExprState) {
    EvaluatorFixture fixture;
    std::vector<expr::TypedExprPtr> expressions{fixture.ModLessThan(7, 4)};
    auto prepared = std::make_shared<PreparedOffsetExpressionEvaluator>(
        expressions, fixture.exec_context.get(), /*null_rejecting=*/true);

    std::array<int32_t, 32> row_ids{};
    for (uint32_t lane = 0; lane < row_ids.size(); ++lane) {
        row_ids[lane] = static_cast<int32_t>(lane * 3);
    }
    const auto all_active = (uint64_t{1} << row_ids.size()) - 1;

    std::vector<std::future<uint64_t>> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.emplace_back(std::async(
            std::launch::async, [prepared, row_ids, all_active]() mutable {
                auto workspace = prepared->CreateWorkspace();
                uint64_t checksum = 0;
                for (int iteration = 0; iteration < 100; ++iteration) {
                    checksum ^= workspace
                                    ->EvalBatch(row_ids.data(),
                                                row_ids.size(),
                                                all_active)
                                    .accepted_mask();
                }
                return checksum;
            }));
    }
    for (auto& worker : workers) {
        EXPECT_EQ(worker.get(), 0U);
    }
}

}  // namespace milvus::exec
