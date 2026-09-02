// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/OffsetExpressionEvaluator.h"

#include <array>
#include <atomic>
#include <future>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace milvus::exec {
namespace {

struct FakeProgram {
    int64_t divisor;
};

struct FakeWorkspace : OffsetEvalWorkspace {
    uint64_t calls{0};
};

std::unique_ptr<OffsetEvalWorkspace>
CreateFakeWorkspace(const void*) {
    return std::make_unique<FakeWorkspace>();
}

int32_t
EvalFake(const void* opaque,
         OffsetEvalWorkspace& workspace,
         const int64_t* ids,
         uint32_t count,
         uint64_t active,
         OffsetTruthMask* result) {
    const auto& program = *static_cast<const FakeProgram*>(opaque);
    auto& scratch = static_cast<FakeWorkspace&>(workspace);
    ++scratch.calls;
    for (uint32_t lane = 0; lane < count; ++lane) {
        const auto bit = uint64_t{1} << lane;
        if ((active & bit) == 0) {
            continue;
        }
        if (ids[lane] < 0) {
            continue;  // negative IDs model SQL NULL in this fake leaf.
        }
        result->known_mask |= bit;
        if (ids[lane] % program.divisor == 0) {
            result->true_mask |= bit;
        }
    }
    return 0;
}

std::shared_ptr<const PreparedOffsetExpressionEvaluator>
MakeFakeEvaluator() {
    return PreparedOffsetExpressionEvaluator::Create(
        std::make_shared<const FakeProgram>(FakeProgram{3}),
        &CreateFakeWorkspace,
        &EvalFake);
}

TEST(OffsetExpressionEvaluatorTest, ValidatesBatchShapesAndThreeValuedMasks) {
    auto evaluator = MakeFakeEvaluator();
    ASSERT_NE(evaluator, nullptr);
    auto workspace = evaluator->CreateWorkspace();
    ASSERT_NE(workspace, nullptr);

    std::array<int64_t, 64> ids{};
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<int64_t>(i);
    }
    ids[1] = -1;

    for (const uint32_t count : {0U, 1U, 31U, 32U, 64U}) {
        const auto active = count == 64
                                ? ~uint64_t{0}
                                : (count == 0 ? 0 : (uint64_t{1} << count) - 1);
        OffsetTruthMask result;
        EXPECT_EQ(evaluator->EvalBatch(
                      *workspace, ids.data(), count, active, &result),
                  OffsetEvalStatus::Success);
        EXPECT_EQ(result.true_mask & ~result.known_mask, 0);
        EXPECT_EQ((result.true_mask | result.known_mask) & ~active, 0);
        if (count > 1) {
            EXPECT_EQ(result.known_mask & uint64_t{2}, 0);
        }
    }

    OffsetTruthMask result;
    EXPECT_EQ(evaluator->EvalBatch(*workspace, ids.data(), 65, 0, &result),
              OffsetEvalStatus::InvalidArgument);
    EXPECT_EQ(evaluator->EvalBatch(*workspace, ids.data(), 1, 2, &result),
              OffsetEvalStatus::InvalidArgument);
    EXPECT_EQ(evaluator->EvalBatch(*workspace, nullptr, 1, 1, &result),
              OffsetEvalStatus::InvalidArgument);
}

TEST(OffsetExpressionEvaluatorTest, HonorsSparseActiveMask) {
    auto evaluator = MakeFakeEvaluator();
    auto workspace = evaluator->CreateWorkspace();
    const std::array<int64_t, 8> ids{0, 3, 4, -1, 6, 7, 9, 12};
    constexpr uint64_t active = 0b11011001;
    OffsetTruthMask result;
    ASSERT_EQ(evaluator->EvalBatch(
                  *workspace, ids.data(), ids.size(), active, &result),
              OffsetEvalStatus::Success);
    EXPECT_EQ(result.known_mask, active & ~(uint64_t{1} << 3));
    EXPECT_EQ(result.true_mask, uint64_t{0b11010001});
}

struct TruthTableProgram {
    std::array<int8_t, 8> values;
};

int32_t
EvalTruthTable(const void* opaque,
               OffsetEvalWorkspace&,
               const int64_t* ids,
               uint32_t count,
               uint64_t active,
               OffsetTruthMask* result) {
    const auto& program = *static_cast<const TruthTableProgram*>(opaque);
    for (uint32_t lane = 0; lane < count; ++lane) {
        const auto bit = uint64_t{1} << lane;
        if ((active & bit) == 0) {
            continue;
        }
        const auto value = program.values.at(ids[lane]);
        if (value < 0) {
            continue;
        }
        result->known_mask |= bit;
        if (value != 0) {
            result->true_mask |= bit;
        }
    }
    return 0;
}

std::shared_ptr<const PreparedOffsetExpressionEvaluator>
MakeTruthTableEvaluator(std::array<int8_t, 8> values) {
    return PreparedOffsetExpressionEvaluator::Create(
        std::make_shared<const TruthTableProgram>(
            TruthTableProgram{std::move(values)}),
        &CreateFakeWorkspace,
        &EvalTruthTable);
}

TEST(OffsetExpressionEvaluatorTest, ComposesSqlAndOrNotTruth) {
    auto a = MakeTruthTableEvaluator({0, 0, 0, 1, 1, 1, -1, -1});
    auto b = MakeTruthTableEvaluator({0, 1, -1, 0, 1, -1, 0, 1});
    // NOT(A) OR B. Nodes are post-order and logical nodes reference earlier
    // node indices.
    auto evaluator = ComposeOffsetExpressionEvaluators(
        {std::move(a), std::move(b)},
        {{OffsetExpressionNodeType::Leaf, 0, 0},
         {OffsetExpressionNodeType::Not, 0, 0},
         {OffsetExpressionNodeType::Leaf, 1, 0},
         {OffsetExpressionNodeType::Or, 1, 2}},
        3);
    ASSERT_NE(evaluator, nullptr);
    auto workspace = evaluator->CreateWorkspace();
    ASSERT_NE(workspace, nullptr);
    const std::array<int64_t, 8> ids{0, 1, 2, 3, 4, 5, 6, 7};
    OffsetTruthMask result;
    ASSERT_EQ(
        evaluator->EvalBatch(*workspace, ids.data(), ids.size(), 0xff, &result),
        OffsetEvalStatus::Success);
    // TRUE, TRUE, TRUE, FALSE, TRUE, NULL, NULL, TRUE.
    EXPECT_EQ(result.true_mask, uint64_t{0x97});
    EXPECT_EQ(result.known_mask, uint64_t{0x9f});
}

TEST(OffsetExpressionEvaluatorTest, ContainsErrorsAndRejectsInvalidResults) {
    struct EmptyWorkspace : OffsetEvalWorkspace {};
    auto factory = +[](const void*) -> std::unique_ptr<OffsetEvalWorkspace> {
        return std::make_unique<EmptyWorkspace>();
    };
    auto throwing =
        +[](const void*,
            OffsetEvalWorkspace&,
            const int64_t*,
            uint32_t,
            uint64_t,
            OffsetTruthMask*) -> int32_t { throw std::runtime_error("test"); };
    auto invalid = +[](const void*,
                       OffsetEvalWorkspace&,
                       const int64_t*,
                       uint32_t,
                       uint64_t,
                       OffsetTruthMask* result) -> int32_t {
        result->true_mask = 1;
        result->known_mask = 0;
        return 0;
    };
    auto owner = std::make_shared<const int>(1);
    const int64_t id = 0;
    OffsetTruthMask result;

    auto throwing_evaluator =
        PreparedOffsetExpressionEvaluator::Create(owner, factory, throwing);
    auto workspace = throwing_evaluator->CreateWorkspace();
    EXPECT_EQ(throwing_evaluator->EvalBatch(*workspace, &id, 1, 1, &result),
              OffsetEvalStatus::EvaluationError);

    auto invalid_evaluator =
        PreparedOffsetExpressionEvaluator::Create(owner, factory, invalid);
    workspace = invalid_evaluator->CreateWorkspace();
    EXPECT_EQ(invalid_evaluator->EvalBatch(*workspace, &id, 1, 1, &result),
              OffsetEvalStatus::InvalidResult);
}

TEST(OffsetExpressionEvaluatorTest, UsesOneWorkspacePerConcurrentWorker) {
    const auto evaluator = MakeFakeEvaluator();
    std::vector<std::future<uint64_t>> workers;
    for (size_t worker = 0; worker < 8; ++worker) {
        workers.emplace_back(std::async(std::launch::async, [evaluator] {
            auto workspace = evaluator->CreateWorkspace();
            std::array<int64_t, 32> ids{};
            for (size_t i = 0; i < ids.size(); ++i) {
                ids[i] = static_cast<int64_t>(i);
            }
            for (size_t round = 0; round < 1000; ++round) {
                OffsetTruthMask result;
                if (evaluator->EvalBatch(*workspace,
                                         ids.data(),
                                         ids.size(),
                                         0xffffffffU,
                                         &result) !=
                    OffsetEvalStatus::Success) {
                    return uint64_t{0};
                }
            }
            return static_cast<FakeWorkspace&>(*workspace).calls;
        }));
    }
    for (auto& worker : workers) {
        EXPECT_EQ(worker.get(), 1000);
    }
}

}  // namespace
}  // namespace milvus::exec
