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

#include <boost/format.hpp>
#include <boost/optional/optional.hpp>
#include <folly/FBVector.h>
#include <stddef.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ExprTestBase.h"
#include "NamedType/named_type_impl.hpp"
#include "bitset/bitset.h"
#include "common/Common.h"
#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "common/IndexMeta.h"
#include "common/Schema.h"
#include "common/Types.h"
#include "common/Vector.h"
#include "common/protobuf_utils.h"
#include "exec/QueryContext.h"
#include "exec/Task.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/CandidateEvaluator.h"
#include "exec/expression/DownpushPredicateProvider.h"
#include "exec/expression/NumericCandidateEvaluator.h"
#include "exec/expression/StringCandidateEvaluator.h"
#include "expr/ITypeExpr.h"
#include "gtest/gtest.h"
#include "index/BitmapIndex.h"
#include "index/Index.h"
#include "index/Meta.h"
#include "index/StringIndex.h"
#include "index/StringIndexMarisa.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "pb/plan.pb.h"
#include "plan/PlanNode.h"
#include "query/ExecPlanNodeVisitor.h"
#include "query/Plan.h"
#include "query/PlanImpl.h"
#include "query/PlanNode.h"
#include "segcore/ChunkedSegmentSealedImpl.h"
#include "segcore/SegmentGrowing.h"
#include "segcore/SegmentGrowingImpl.h"
#include "segcore/SegmentSealed.h"
#include "segcore/Types.h"
#include "test_utils/DataGen.h"
#include "test_utils/GenExprProto.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

EXPR_TEST_INSTANTIATE();

TEST(CandidateEvaluatorTest,
     Int64ModSupportsContiguousChunkedAndPartialBatches) {
    static constexpr std::array<int64_t, 10> values = {
        -9, -1, 0, 1, 2, 3, 4, 5, 8, 13};
    static constexpr std::array<const int64_t*, 3> chunks = {
        values.data(), values.data() + 3, values.data() + 7};
    static constexpr std::array<int64_t, 4> chunk_offsets = {0, 3, 7, 10};
    static constexpr std::array<int64_t, 8> row_ids = {0, 1, 2, 3, 6, 7, 8, 9};

    auto check = [&](const exec::Int64CandidateSourceView& source) {
        auto prepared = exec::PrepareInt64ModCandidateEvaluator(source, 5, 2);
        ASSERT_TRUE(prepared.has_value());
        ASSERT_TRUE(static_cast<bool>(prepared.value()));
        uint64_t valid_mask = 0;
        const auto status = prepared->view.eval_batch(prepared->view.context,
                                                      row_ids.data(),
                                                      row_ids.size(),
                                                      0b11101111,
                                                      &valid_mask);
        EXPECT_EQ(status, 0);

        uint64_t expected = 0;
        for (size_t lane = 0; lane < row_ids.size(); ++lane) {
            if ((0b11101111 & (uint64_t{1} << lane)) != 0 &&
                values[row_ids[lane]] % 5 < 2) {
                expected |= uint64_t{1} << lane;
            }
        }
        EXPECT_EQ(valid_mask, expected);

        ASSERT_NE(prepared->view.eval_contiguous, nullptr);
        valid_mask = 0;
        constexpr int64_t first_row_id = 1;
        constexpr uint32_t contiguous_count = 7;
        constexpr uint64_t contiguous_active = 0b1101111;
        EXPECT_EQ(prepared->view.eval_contiguous(prepared->view.context,
                                                 first_row_id,
                                                 contiguous_count,
                                                 contiguous_active,
                                                 &valid_mask),
                  0);
        expected = 0;
        for (uint32_t lane = 0; lane < contiguous_count; ++lane) {
            if ((contiguous_active & (uint64_t{1} << lane)) != 0 &&
                values[first_row_id + lane] % 5 < 2) {
                expected |= uint64_t{1} << lane;
            }
        }
        EXPECT_EQ(valid_mask, expected);
    };

    exec::Int64CandidateSourceView contiguous;
    contiguous.row_values = values.data();
    contiguous.row_count = values.size();
    check(contiguous);

    exec::Int64CandidateSourceView chunked;
    chunked.row_count = values.size();
    chunked.chunk_values = chunks.data();
    chunked.chunk_offsets = chunk_offsets.data();
    chunked.num_chunks = chunks.size();
    check(chunked);

    EXPECT_FALSE(
        exec::PrepareInt64ModCandidateEvaluator(contiguous, 0, 1).has_value());
    EXPECT_FALSE(
        exec::PrepareInt64ModCandidateEvaluator(contiguous, 5, 6).has_value());
}

TEST(OffsetExpressionE0, IterativeOffsetAndCandidateModAgree) {
    auto schema = std::make_shared<Schema>();
    const auto id_fid = schema->AddDebugField("id", DataType::INT64);
    const auto bucket_fid = schema->AddDebugField("bucket", DataType::INT64);
    schema->set_primary_field_id(id_fid);

    static constexpr std::array<int64_t, 16> values = {
        -13, -9, -5, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17};
    std::vector<int64_t> ids(values.size());
    std::iota(ids.begin(), ids.end(), int64_t{0});

    auto insert_data = std::make_unique<InsertRecordProto>();
    InsertCol(insert_data.get(), ids, (*schema)[id_fid], false);
    InsertCol(insert_data.get(),
              std::vector<int64_t>(values.begin(), values.end()),
              (*schema)[bucket_fid],
              false);
    GeneratedData raw_data;
    raw_data.schema_ = schema;
    raw_data.raw_ = insert_data.release();
    raw_data.raw_->set_num_rows(values.size());
    raw_data.row_ids_ = ids;
    raw_data.timestamps_.assign(values.size(), 0);

    auto segment = CreateSealedSegment(schema);
    LoadGeneratedDataIntoSegment(raw_data, segment.get(), true);

    proto::plan::GenericValue threshold;
    threshold.set_int64_val(2);
    proto::plan::GenericValue divisor;
    divisor.set_int64_val(5);
    auto expression = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        expr::ColumnInfo(bucket_fid, DataType::INT64),
        proto::plan::OpType::LessThan,
        proto::plan::ArithOpType::Mod,
        threshold,
        divisor);
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expression);

    exec::OffsetVector offsets = {15, 0, 7, 3, 9, 4, 12, 1, 14, 6, 10};
    const auto iterative_result = milvus::test::gen_filter_res(
        plan.get(), segment.get(), values.size(), MAX_TIMESTAMP, &offsets);
    const BitsetTypeView iterative_truth(iterative_result->GetRawData(),
                                         iterative_result->size());

    exec::Int64CandidateSourceView source;
    source.row_values = values.data();
    source.row_count = values.size();
    const auto candidate =
        exec::PrepareInt64ModCandidateEvaluator(source, 5, 2);
    ASSERT_TRUE(candidate.has_value());
    std::array<int64_t, 11> candidate_ids{};
    std::copy(offsets.begin(), offsets.end(), candidate_ids.begin());
    uint64_t candidate_truth = 0;
    const uint64_t active = (uint64_t{1} << offsets.size()) - 1;
    ASSERT_EQ(candidate->view.eval_batch(candidate->view.context,
                                         candidate_ids.data(),
                                         candidate_ids.size(),
                                         active,
                                         &candidate_truth),
              0);

    ASSERT_EQ(iterative_truth.size(), offsets.size());
    for (size_t lane = 0; lane < offsets.size(); ++lane) {
        EXPECT_EQ(iterative_truth[lane],
                  (candidate_truth & (uint64_t{1} << lane)) != 0)
            << "lane=" << lane << " row=" << offsets[lane];
    }

    OpContext op_context;
    auto shared = exec::PrepareNumericOffsetExpressionEvaluator(
        segment.get(), &op_context, expression);
    ASSERT_NE(shared, nullptr);
    auto workspace = shared->CreateWorkspace();
    ASSERT_NE(workspace, nullptr);
    exec::OffsetTruthMask shared_truth;
    ASSERT_EQ(shared->EvalBatch(*workspace,
                                candidate_ids.data(),
                                candidate_ids.size(),
                                active,
                                &shared_truth),
              exec::OffsetEvalStatus::Success);
    EXPECT_EQ(shared_truth.true_mask & shared_truth.known_mask,
              candidate_truth);
}

TEST(OffsetExpressionE0, ComparisonVarcharAndLogicalTreesAgree) {
    auto schema = std::make_shared<Schema>();
    const auto id_fid = schema->AddDebugField("id", DataType::INT64);
    const auto bucket_fid = schema->AddDebugField("bucket", DataType::INT64);
    const auto tag_fid = schema->AddDebugField("tag", DataType::VARCHAR);
    schema->set_primary_field_id(id_fid);

    const std::vector<int64_t> ids = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<int64_t> buckets = {-2, -1, 0, 2, 3, 5, 7, 9};
    const std::vector<std::string> tags = {
        "ant", "bee", "cat", "dog", "eel", "fox", "gnu", "猫"};
    auto insert_data = std::make_unique<InsertRecordProto>();
    InsertCol(insert_data.get(), ids, (*schema)[id_fid], false);
    InsertCol(insert_data.get(), buckets, (*schema)[bucket_fid], false);
    InsertCol(insert_data.get(), tags, (*schema)[tag_fid], false);
    GeneratedData raw_data;
    raw_data.schema_ = schema;
    raw_data.raw_ = insert_data.release();
    raw_data.raw_->set_num_rows(ids.size());
    raw_data.row_ids_ = ids;
    raw_data.timestamps_.assign(ids.size(), 0);
    auto segment = CreateSealedSegment(schema);
    LoadGeneratedDataIntoSegment(raw_data, segment.get(), true);

    exec::OffsetVector offsets = {7, 0, 5, 2, 6, 1, 4, 3};
    std::array<int64_t, 8> candidate_ids{};
    std::copy(offsets.begin(), offsets.end(), candidate_ids.begin());
    constexpr uint64_t active = 0xff;
    auto assert_same = [&](const expr::TypedExprPtr& expression,
                           const exec::PreparedCandidateEvaluator& candidate) {
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           expression);
        const auto iterative_result = milvus::test::gen_filter_res(
            plan.get(), segment.get(), ids.size(), MAX_TIMESTAMP, &offsets);
        const BitsetTypeView iterative_truth(iterative_result->GetRawData(),
                                             iterative_result->size());
        uint64_t candidate_truth = 0;
        ASSERT_EQ(candidate.view.eval_batch(candidate.view.context,
                                            candidate_ids.data(),
                                            candidate_ids.size(),
                                            active,
                                            &candidate_truth),
                  0);
        ASSERT_EQ(iterative_truth.size(), offsets.size());
        for (size_t lane = 0; lane < offsets.size(); ++lane) {
            EXPECT_EQ(iterative_truth[lane],
                      (candidate_truth & (uint64_t{1} << lane)) != 0)
                << "lane=" << lane << " row=" << offsets[lane];
        }
    };

    proto::plan::GenericValue three;
    three.set_int64_val(3);
    auto greater_equal = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(bucket_fid, DataType::INT64),
        proto::plan::OpType::GreaterEqual,
        three);
    exec::Int64CandidateSourceView numeric_source;
    numeric_source.row_values = buckets.data();
    numeric_source.row_count = buckets.size();
    OpContext op_context;
    auto assert_numeric_shared =
        [&](const expr::TypedExprPtr& expression,
            const exec::PreparedCandidateEvaluator& candidate) {
            assert_same(expression, candidate);
            auto shared = exec::PrepareNumericOffsetExpressionEvaluator(
                segment.get(), &op_context, expression);
            ASSERT_NE(shared, nullptr);
            auto workspace = shared->CreateWorkspace();
            ASSERT_NE(workspace, nullptr);
            exec::OffsetTruthMask shared_truth;
            ASSERT_EQ(shared->EvalBatch(*workspace,
                                        candidate_ids.data(),
                                        candidate_ids.size(),
                                        active,
                                        &shared_truth),
                      exec::OffsetEvalStatus::Success);
            uint64_t candidate_truth = 0;
            ASSERT_EQ(candidate.view.eval_batch(candidate.view.context,
                                                candidate_ids.data(),
                                                candidate_ids.size(),
                                                active,
                                                &candidate_truth),
                      0);
            EXPECT_EQ(shared_truth.true_mask & shared_truth.known_mask,
                      candidate_truth);
        };

    exec::Int64CandidatePredicate ge_predicate;
    ge_predicate.op = exec::NumericCandidatePredicateOp::GreaterEqual;
    ge_predicate.arg0 = 3;
    auto ge_candidate =
        exec::PrepareInt64CandidateEvaluator(numeric_source, ge_predicate);
    ASSERT_TRUE(ge_candidate.has_value());
    assert_numeric_shared(greater_equal, *ge_candidate);

    proto::plan::GenericValue lower;
    lower.set_int64_val(-1);
    proto::plan::GenericValue upper;
    upper.set_int64_val(7);
    auto range = std::make_shared<expr::BinaryRangeFilterExpr>(
        expr::ColumnInfo(bucket_fid, DataType::INT64),
        lower,
        upper,
        false,
        true);
    exec::Int64CandidatePredicate range_predicate;
    range_predicate.op = exec::NumericCandidatePredicateOp::Range;
    range_predicate.arg0 = -1;
    range_predicate.arg1 = 7;
    range_predicate.lower_inclusive = false;
    range_predicate.upper_inclusive = true;
    auto range_candidate = exec::PrepareInt64CandidateEvaluator(
        numeric_source, range_predicate);
    ASSERT_TRUE(range_candidate.has_value());
    assert_numeric_shared(range, *range_candidate);

    std::vector<proto::plan::GenericValue> term_values;
    for (const auto value : {-2, 2, 7}) {
        proto::plan::GenericValue term_value;
        term_value.set_int64_val(value);
        term_values.push_back(std::move(term_value));
    }
    auto term = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(bucket_fid, DataType::INT64), term_values);
    exec::Int64CandidatePredicate term_predicate;
    term_predicate.op = exec::NumericCandidatePredicateOp::Term;
    term_predicate.terms = {-2, 2, 7};
    auto term_candidate =
        exec::PrepareInt64CandidateEvaluator(numeric_source, term_predicate);
    ASSERT_TRUE(term_candidate.has_value());
    assert_numeric_shared(term, *term_candidate);

    proto::plan::GenericValue fox;
    fox.set_string_val("fox");
    auto not_fox = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(tag_fid, DataType::VARCHAR),
        proto::plan::OpType::NotEqual,
        fox);
    std::string string_storage;
    std::array<uint32_t, 9> string_offsets{};
    for (size_t i = 0; i < tags.size(); ++i) {
        string_storage.append(tags[i]);
        string_offsets[i + 1] = string_storage.size();
    }
    const std::array<const char*, 1> string_bases = {string_storage.data()};
    const std::array<const uint32_t*, 1> value_offsets = {
        string_offsets.data()};
    const std::array<size_t, 1> row_counts = {tags.size()};
    const std::array<int64_t, 2> row_offsets = {
        0, static_cast<int64_t>(tags.size())};
    exec::StringCandidateSourceView string_source;
    string_source.chunk_bases = string_bases.data();
    string_source.chunk_value_offsets = value_offsets.data();
    string_source.chunk_row_counts = row_counts.data();
    string_source.chunk_row_offsets = row_offsets.data();
    string_source.num_chunks = 1;
    string_source.row_count = tags.size();
    string_source.uniform_chunk_rows = tags.size();
    exec::StringCandidatePredicate ne_predicate =
        exec::StringComparisonCandidatePredicate{
            tag_fid,
            DataType::VARCHAR,
            false,
            exec::StringCandidateComparisonOp::NotEqual,
            "fox"};
    auto ne_candidate =
        exec::PrepareStringCandidateEvaluator(string_source, ne_predicate);
    ASSERT_TRUE(ne_candidate.has_value());
    assert_same(not_fox, *ne_candidate);

    proto::plan::GenericValue five;
    five.set_int64_val(5);
    auto equal_five = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(bucket_fid, DataType::INT64),
        proto::plan::OpType::Equal,
        five);
    proto::plan::GenericValue zero;
    zero.set_int64_val(0);
    auto less_zero = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(bucket_fid, DataType::INT64),
        proto::plan::OpType::LessThan,
        zero);
    auto not_equal_five = std::make_shared<expr::LogicalUnaryExpr>(
        expr::LogicalUnaryExpr::OpType::LogicalNot, equal_five);
    auto ge_and_not_five = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::And, greater_equal, not_equal_five);
    auto logical = std::make_shared<expr::LogicalBinaryExpr>(
        expr::LogicalBinaryExpr::OpType::Or, ge_and_not_five, less_zero);

    exec::Int64CandidatePredicate eq_predicate;
    eq_predicate.op = exec::NumericCandidatePredicateOp::Equal;
    eq_predicate.arg0 = 5;
    exec::Int64CandidatePredicate lt_predicate;
    lt_predicate.op = exec::NumericCandidatePredicateOp::LessThan;
    lt_predicate.arg0 = 0;
    auto eq_candidate =
        exec::PrepareInt64CandidateEvaluator(numeric_source, eq_predicate);
    auto lt_candidate =
        exec::PrepareInt64CandidateEvaluator(numeric_source, lt_predicate);
    ASSERT_TRUE(eq_candidate.has_value());
    ASSERT_TRUE(lt_candidate.has_value());
    auto composed = exec::ComposeCandidateEvaluators(
        {std::move(*ge_candidate),
         std::move(*eq_candidate),
         std::move(*lt_candidate)},
        {{exec::CandidatePredicateNodeType::Leaf, 0, 0},
         {exec::CandidatePredicateNodeType::Leaf, 1, 0},
         {exec::CandidatePredicateNodeType::Not, 1, 0},
         {exec::CandidatePredicateNodeType::And, 0, 2},
         {exec::CandidatePredicateNodeType::Leaf, 2, 0},
         {exec::CandidatePredicateNodeType::Or, 3, 4}},
        5);
    ASSERT_TRUE(composed.has_value());
    assert_same(logical, *composed);
}

TEST(OffsetExpressionE2Benchmark, DISABLED_ModBatch32) {
    constexpr size_t row_count = 8192;
    constexpr size_t batch_count = 2048;
    constexpr size_t repeats = 20;
    constexpr size_t width = 32;

    auto schema = std::make_shared<Schema>();
    const auto id_fid = schema->AddDebugField("id", DataType::INT64);
    const auto bucket_fid = schema->AddDebugField("bucket", DataType::INT64);
    schema->set_primary_field_id(id_fid);
    std::vector<int64_t> ids(row_count);
    std::vector<int64_t> buckets(row_count);
    std::iota(ids.begin(), ids.end(), int64_t{0});
    for (size_t row = 0; row < row_count; ++row) {
        buckets[row] = static_cast<int64_t>(row % 10000);
    }
    auto insert_data = std::make_unique<InsertRecordProto>();
    InsertCol(insert_data.get(), ids, (*schema)[id_fid], false);
    InsertCol(insert_data.get(), buckets, (*schema)[bucket_fid], false);
    GeneratedData raw_data;
    raw_data.schema_ = schema;
    raw_data.raw_ = insert_data.release();
    raw_data.raw_->set_num_rows(row_count);
    raw_data.row_ids_ = ids;
    raw_data.timestamps_.assign(row_count, 0);
    auto segment = CreateSealedSegment(schema);
    LoadGeneratedDataIntoSegment(raw_data, segment.get(), true);

    proto::plan::GenericValue threshold;
    threshold.set_int64_val(3);
    proto::plan::GenericValue divisor;
    divisor.set_int64_val(5);
    auto expression = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        expr::ColumnInfo(bucket_fid, DataType::INT64),
        proto::plan::OpType::LessThan,
        proto::plan::ArithOpType::Mod,
        threshold,
        divisor);

    auto query_context = std::make_shared<exec::QueryContext>(
        DEAFULT_QUERY_ID, segment.get(), row_count, MAX_TIMESTAMP);
    auto exec_context =
        std::make_unique<exec::ExecContext>(query_context.get());
    exec::ExprSet legacy({expression}, exec_context.get());
    exec::OffsetVector legacy_offsets;
    legacy_offsets.reserve(width);
    exec::EvalCtx eval_ctx(exec_context.get(), &legacy_offsets);
    std::vector<VectorPtr> legacy_results;

    OpContext op_context;
    auto shared = exec::PrepareNumericOffsetExpressionEvaluator(
        segment.get(), &op_context, expression);
    ASSERT_NE(shared, nullptr);
    auto workspace = shared->CreateWorkspace();
    ASSERT_NE(workspace, nullptr);

    std::vector<std::array<int32_t, width>> batches(batch_count);
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (auto& batch : batches) {
        for (auto& row : batch) {
            state ^= state << 7;
            state ^= state >> 9;
            row = static_cast<int32_t>(state % row_count);
        }
    }

    auto run_legacy = [&] {
        uint64_t checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        for (size_t repeat = 0; repeat < repeats; ++repeat) {
            for (const auto& batch : batches) {
                legacy_offsets.assign(batch.begin(), batch.end());
                legacy.Eval(0, 1, true, eval_ctx, legacy_results);
                auto result =
                    std::dynamic_pointer_cast<ColumnVector>(legacy_results[0]);
                const BitsetTypeView truth(result->GetRawData(), width);
                for (size_t lane = 0; lane < width; ++lane) {
                    checksum += truth[lane];
                }
            }
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        return std::pair{
            checksum,
            std::chrono::duration<double, std::nano>(elapsed).count() /
                (repeats * batch_count * width)};
    };
    auto run_shared = [&] {
        uint64_t checksum = 0;
        std::array<int64_t, width> widened{};
        const auto start = std::chrono::steady_clock::now();
        for (size_t repeat = 0; repeat < repeats; ++repeat) {
            for (const auto& batch : batches) {
                std::copy(batch.begin(), batch.end(), widened.begin());
                exec::OffsetTruthMask truth;
                const auto status = shared->EvalBatch(
                    *workspace, widened.data(), width, 0xffffffffU, &truth);
                if (status != exec::OffsetEvalStatus::Success) {
                    throw std::runtime_error("shared evaluator failed");
                }
                checksum += std::popcount(truth.true_mask & truth.known_mask);
            }
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        return std::pair{
            checksum,
            std::chrono::duration<double, std::nano>(elapsed).count() /
                (repeats * batch_count * width)};
    };

    run_legacy();
    run_shared();
    const auto [legacy_checksum_a, legacy_ns_a] = run_legacy();
    const auto [shared_checksum_b1, shared_ns_b1] = run_shared();
    const auto [shared_checksum_b2, shared_ns_b2] = run_shared();
    const auto [legacy_checksum_a2, legacy_ns_a2] = run_legacy();
    EXPECT_EQ(legacy_checksum_a2, legacy_checksum_a);
    EXPECT_EQ(shared_checksum_b1, legacy_checksum_a);
    EXPECT_EQ(shared_checksum_b2, legacy_checksum_a);
    const auto legacy_ns = (legacy_ns_a + legacy_ns_a2) / 2.0;
    const auto shared_ns = (shared_ns_b1 + shared_ns_b2) / 2.0;
    std::cout << "offset_evaluator_mod_batch32 legacy_ns_per_candidate="
              << legacy_ns << " shared_ns_per_candidate=" << shared_ns
              << " ratio=" << shared_ns / legacy_ns << std::endl;
}

TEST(CandidateEvaluatorTest, Int64ComparisonRangeAndTermStayMilvusOwned) {
    static constexpr std::array<int64_t, 9> values = {
        -8, -1, 0, 1, 2, 3, 4, 8, 13};
    static constexpr std::array<const int64_t*, 3> chunks = {
        values.data(), values.data() + 2, values.data() + 6};
    static constexpr std::array<int64_t, 4> chunk_offsets = {0, 2, 6, 9};
    static constexpr std::array<int64_t, 9> row_ids = {
        8, 0, 7, 1, 6, 2, 5, 3, 4};

    exec::Int64CandidateSourceView contiguous;
    contiguous.row_values = values.data();
    contiguous.row_count = values.size();
    exec::Int64CandidateSourceView chunked;
    chunked.row_count = values.size();
    chunked.chunk_values = chunks.data();
    chunked.chunk_offsets = chunk_offsets.data();
    chunked.num_chunks = chunks.size();

    struct TestCase {
        exec::NumericCandidatePredicateOp op;
        int64_t arg0;
        int64_t arg1;
        bool lower_inclusive;
        bool upper_inclusive;
        std::function<bool(int64_t)> expected;
    };
    const std::vector<TestCase> cases = {
        {exec::NumericCandidatePredicateOp::GreaterEqual,
         3,
         0,
         true,
         true,
         [](int64_t v) { return v >= 3; }},
        {exec::NumericCandidatePredicateOp::GreaterThan,
         3,
         0,
         true,
         true,
         [](int64_t v) { return v > 3; }},
        {exec::NumericCandidatePredicateOp::LessEqual,
         3,
         0,
         true,
         true,
         [](int64_t v) { return v <= 3; }},
        {exec::NumericCandidatePredicateOp::LessThan,
         3,
         0,
         true,
         true,
         [](int64_t v) { return v < 3; }},
        {exec::NumericCandidatePredicateOp::Equal,
         3,
         0,
         true,
         true,
         [](int64_t v) { return v == 3; }},
        {exec::NumericCandidatePredicateOp::NotEqual,
         3,
         0,
         true,
         true,
         [](int64_t v) { return v != 3; }},
        {exec::NumericCandidatePredicateOp::Range,
         -1,
         4,
         false,
         true,
         [](int64_t v) { return v > -1 && v <= 4; }},
    };

    for (const auto& test_case : cases) {
        exec::Int64CandidatePredicate predicate;
        predicate.op = test_case.op;
        predicate.arg0 = test_case.arg0;
        predicate.arg1 = test_case.arg1;
        predicate.lower_inclusive = test_case.lower_inclusive;
        predicate.upper_inclusive = test_case.upper_inclusive;

        uint64_t reference_mask = 0;
        for (size_t lane = 0; lane < row_ids.size(); ++lane) {
            // Leave lane 3 inactive to exercise active-mask propagation.
            if (lane != 3 && test_case.expected(values[row_ids[lane]])) {
                reference_mask |= uint64_t{1} << lane;
            }
        }
        for (const auto& source : {contiguous, chunked}) {
            auto prepared =
                exec::PrepareInt64CandidateEvaluator(source, predicate);
            ASSERT_TRUE(prepared.has_value());
            uint64_t valid_mask = 0;
            EXPECT_EQ(prepared->view.eval_batch(prepared->view.context,
                                                row_ids.data(),
                                                row_ids.size(),
                                                ~(uint64_t{1} << 3),
                                                &valid_mask),
                      0);
            EXPECT_EQ(valid_mask, reference_mask);
        }
    }

    exec::Int64CandidatePredicate term;
    term.op = exec::NumericCandidatePredicateOp::Term;
    term.terms = {-8, 1, 4, 13};
    auto prepared = exec::PrepareInt64CandidateEvaluator(contiguous, term);
    ASSERT_TRUE(prepared.has_value());
    uint64_t valid_mask = 0;
    EXPECT_EQ(prepared->view.eval_batch(prepared->view.context,
                                        row_ids.data(),
                                        row_ids.size(),
                                        UINT64_MAX,
                                        &valid_mask),
              0);
    uint64_t expected_mask = 0;
    for (size_t lane = 0; lane < row_ids.size(); ++lane) {
        if (std::binary_search(term.terms.begin(),
                               term.terms.end(),
                               values[row_ids[lane]])) {
            expected_mask |= uint64_t{1} << lane;
        }
    }
    EXPECT_EQ(valid_mask, expected_mask);

    term.terms.clear();
    EXPECT_FALSE(
        exec::PrepareInt64CandidateEvaluator(contiguous, term).has_value());
}

TEST(CandidateEvaluatorTest, StringComparisonRangeAndTermStayMilvusOwned) {
    static constexpr char chunk0[] = "antbeecat";
    static constexpr char chunk1[] = "dogeelfox";
    static constexpr std::array<const char*, 2> bases = {chunk0, chunk1};
    static constexpr std::array<uint32_t, 4> offsets0 = {0, 3, 6, 9};
    static constexpr std::array<uint32_t, 4> offsets1 = {0, 3, 6, 9};
    static constexpr std::array<const uint32_t*, 2> offsets = {offsets0.data(),
                                                               offsets1.data()};
    static constexpr std::array<bool, 3> valid0 = {true, false, true};
    static constexpr std::array<bool, 3> valid1 = {true, true, true};
    static constexpr std::array<const bool*, 2> validity = {valid0.data(),
                                                            valid1.data()};
    static constexpr std::array<size_t, 2> row_counts = {3, 3};
    static constexpr std::array<int64_t, 3> row_offsets = {0, 3, 6};
    static constexpr std::array<int64_t, 6> row_ids = {5, 0, 3, 1, 4, 2};

    exec::StringCandidateSourceView source;
    source.chunk_bases = bases.data();
    source.chunk_value_offsets = offsets.data();
    source.chunk_valid_data = validity.data();
    source.chunk_row_counts = row_counts.data();
    source.chunk_row_offsets = row_offsets.data();
    source.num_chunks = bases.size();
    source.row_count = 6;
    source.uniform_chunk_rows = 3;

    auto evaluate = [&](exec::StringCandidatePredicate predicate) {
        auto prepared =
            exec::PrepareStringCandidateEvaluator(source, predicate);
        EXPECT_TRUE(prepared.has_value());
        uint64_t valid_mask = 0;
        EXPECT_EQ(prepared->view.eval_batch(prepared->view.context,
                                            row_ids.data(),
                                            row_ids.size(),
                                            (uint64_t{1} << row_ids.size()) - 1,
                                            &valid_mask),
                  0);
        return valid_mask;
    };

    exec::StringCandidatePredicate equal =
        exec::StringComparisonCandidatePredicate{
            FieldId{},
            DataType::VARCHAR,
            false,
            exec::StringCandidateComparisonOp::Equal,
            "dog"};
    EXPECT_EQ(evaluate(equal), uint64_t{1} << 2);

    exec::StringCandidatePredicate range =
        exec::StringRangeCandidatePredicate{FieldId{},
                                            DataType::VARCHAR,
                                            false,
                                            "cat",
                                            "fox",
                                            true,
                                            false};
    EXPECT_EQ(evaluate(range),
              (uint64_t{1} << 2) | (uint64_t{1} << 4) | (uint64_t{1} << 5));

    exec::StringCandidatePredicate term =
        exec::StringTermCandidatePredicate{
            FieldId{}, DataType::VARCHAR, false, {"fox", "ant"}};
    EXPECT_EQ(evaluate(term), (uint64_t{1} << 0) | (uint64_t{1} << 1));

    static constexpr std::array<int32_t, 4> dictionary_ids = {1, -1, 2, 1};
    exec::StringCandidateSourceView dictionary;
    dictionary.row_count = dictionary_ids.size();
    dictionary.row_dictionary_ids = dictionary_ids.data();
    dictionary.target_dictionary_id = 1;
    dictionary.target_dictionary_id_found = true;
    static constexpr std::array<int64_t, 4> dictionary_rows = {0, 1, 2, 3};
    auto prepared = exec::PrepareStringCandidateEvaluator(dictionary, equal);
    ASSERT_TRUE(prepared.has_value());
    uint64_t valid_mask = 0;
    EXPECT_EQ(
        prepared->view.eval_batch(prepared->view.context,
                                  dictionary_rows.data(),
                                  dictionary_rows.size(),
                                  (uint64_t{1} << dictionary_rows.size()) - 1,
                                  &valid_mask),
        0);
    EXPECT_EQ(valid_mask, (uint64_t{1} << 0) | (uint64_t{1} << 3));

    exec::StringCandidatePredicate like =
        exec::StringLikeCandidatePredicate{
            FieldId{}, DataType::VARCHAR, false, "_o%"};
    EXPECT_EQ(evaluate(like), (uint64_t{1} << 0) | (uint64_t{1} << 2));

    std::get<exec::StringLikeCandidatePredicate>(like).pattern = "d\\%g";
    EXPECT_EQ(evaluate(like), uint64_t{0});

    std::get<exec::StringLikeCandidatePredicate>(like).pattern = "broken\\";
    EXPECT_FALSE(
        exec::PrepareStringCandidateEvaluator(source, like).has_value());
}

TEST(CandidateEvaluatorTest, StringLikePreservesUtf8AndEscapeSemantics) {
    const std::vector<std::string> values = {
        "猫", "猫a", "é", "🙂", "100%", "a_b", R"(path\root)", "axxbYcZZd"};
    std::string storage;
    std::vector<uint32_t> offsets = {0};
    for (const auto& value : values) {
        storage.append(value);
        offsets.push_back(static_cast<uint32_t>(storage.size()));
    }
    const std::array<const char*, 1> bases = {storage.data()};
    const std::array<const uint32_t*, 1> value_offsets = {offsets.data()};
    const std::array<size_t, 1> row_counts = {values.size()};
    const std::array<int64_t, 2> row_offsets = {
        0, static_cast<int64_t>(values.size())};
    const std::array<int64_t, 8> row_ids = {0, 1, 2, 3, 4, 5, 6, 7};

    exec::StringCandidateSourceView source;
    source.chunk_bases = bases.data();
    source.chunk_value_offsets = value_offsets.data();
    source.chunk_row_counts = row_counts.data();
    source.chunk_row_offsets = row_offsets.data();
    source.num_chunks = 1;
    source.row_count = values.size();
    source.uniform_chunk_rows = values.size();

    auto evaluate = [&](std::string pattern) {
        exec::StringCandidatePredicate like =
            exec::StringLikeCandidatePredicate{FieldId{},
                                               DataType::VARCHAR,
                                               false,
                                               std::move(pattern)};
        auto prepared = exec::PrepareStringCandidateEvaluator(source, like);
        EXPECT_TRUE(prepared.has_value());
        uint64_t valid_mask = 0;
        EXPECT_EQ(prepared->view.eval_batch(prepared->view.context,
                                            row_ids.data(),
                                            row_ids.size(),
                                            (uint64_t{1} << row_ids.size()) - 1,
                                            &valid_mask),
                  0);
        return valid_mask;
    };

    EXPECT_EQ(evaluate("_"),
              (uint64_t{1} << 0) | (uint64_t{1} << 2) | (uint64_t{1} << 3));
    EXPECT_EQ(evaluate("猫_"), uint64_t{1} << 1);
    EXPECT_EQ(evaluate(R"(100\%)"), uint64_t{1} << 4);
    EXPECT_EQ(evaluate(R"(a\_b)"), uint64_t{1} << 5);
    EXPECT_EQ(evaluate(R"(path\\%)"), uint64_t{1} << 6);
    EXPECT_EQ(evaluate("a%b_c%d"), uint64_t{1} << 7);
}

TEST(CandidateEvaluatorTest, LogicalComposerPreservesThreeValuedSemantics) {
    static constexpr std::array<int64_t, 8> numeric_values = {
        0, 1, 2, 3, 4, 5, 6, 7};
    static constexpr std::array<int32_t, 8> dictionary_ids = {
        0, 1, 1, 1, 0, -1, 1, 0};
    static constexpr std::array<int64_t, 8> row_ids = {0, 1, 2, 3, 4, 5, 6, 7};

    exec::Int64CandidateSourceView numeric_source;
    numeric_source.row_values = numeric_values.data();
    numeric_source.row_count = numeric_values.size();

    exec::Int64CandidatePredicate greater_equal;
    greater_equal.op = exec::NumericCandidatePredicateOp::GreaterEqual;
    greater_equal.arg0 = 3;
    auto ge =
        exec::PrepareInt64CandidateEvaluator(numeric_source, greater_equal);
    ASSERT_TRUE(ge.has_value());

    exec::StringCandidateSourceView string_source;
    string_source.row_count = dictionary_ids.size();
    string_source.row_dictionary_ids = dictionary_ids.data();
    string_source.target_dictionary_id = 0;
    string_source.target_dictionary_id_found = true;
    exec::StringCandidatePredicate string_equal =
        exec::StringComparisonCandidatePredicate{
            FieldId{},
            DataType::VARCHAR,
            true,
            exec::StringCandidateComparisonOp::Equal,
            "x"};
    auto eq =
        exec::PrepareStringCandidateEvaluator(string_source, string_equal);
    ASSERT_TRUE(eq.has_value());

    exec::Int64CandidatePredicate equal_zero;
    equal_zero.op = exec::NumericCandidatePredicateOp::Equal;
    equal_zero.arg0 = 0;
    auto zero =
        exec::PrepareInt64CandidateEvaluator(numeric_source, equal_zero);
    ASSERT_TRUE(zero.has_value());

    // (numeric >= 3 AND NOT(string == "x")) OR numeric == 0
    const std::vector<exec::CandidatePredicateNode> nodes = {
        {exec::CandidatePredicateNodeType::Leaf, 0, 0},
        {exec::CandidatePredicateNodeType::Leaf, 1, 0},
        {exec::CandidatePredicateNodeType::Not, 1, 0},
        {exec::CandidatePredicateNodeType::And, 0, 2},
        {exec::CandidatePredicateNodeType::Leaf, 2, 0},
        {exec::CandidatePredicateNodeType::Or, 3, 4},
    };
    std::vector<exec::PreparedCandidateEvaluator> leaves;
    leaves.push_back(std::move(*ge));
    leaves.push_back(std::move(*eq));
    leaves.push_back(std::move(*zero));
    auto composite =
        exec::ComposeCandidateEvaluators(std::move(leaves), nodes, 5);
    ASSERT_TRUE(composite.has_value());

    uint64_t accepted = 0;
    EXPECT_EQ(composite->view.eval_batch(composite->view.context,
                                         row_ids.data(),
                                         row_ids.size(),
                                         UINT64_MAX,
                                         &accepted),
              0);
    // Row 5 is NULL in the string source. NOT(NULL) stays NULL and therefore
    // is not accepted at the root; it must not be mistaken for NOT(false).
    EXPECT_EQ(accepted,
              (uint64_t{1} << 0) | (uint64_t{1} << 3) | (uint64_t{1} << 6));
}

TEST(CandidateEvaluatorTest, Int64ArithmeticUsesWideIntermediate) {
    static constexpr std::array<int64_t, 8> values = {
        INT64_MIN, -100, -7, 0, 7, 100, INT64_MAX - 1, INT64_MAX};
    static constexpr std::array<int64_t, 8> row_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    exec::Int64CandidateSourceView source;
    source.row_values = values.data();
    source.row_count = values.size();

    struct TestCase {
        proto::plan::ArithOpType op;
        int64_t operand;
        int64_t threshold;
        std::function<bool(int64_t)> expected;
    };
    const std::vector<TestCase> cases = {
        {proto::plan::ArithOpType::Add,
         9,
         50,
         [](int64_t v) { return static_cast<__int128>(v) + 9 < 50; }},
        {proto::plan::ArithOpType::Sub,
         -9,
         50,
         [](int64_t v) { return static_cast<__int128>(v) - (-9) < 50; }},
        {proto::plan::ArithOpType::Mul,
         3,
         50,
         [](int64_t v) { return static_cast<__int128>(v) * 3 < 50; }},
        {proto::plan::ArithOpType::Div,
         -1,
         50,
         [](int64_t v) { return static_cast<__int128>(v) / (-1) < 50; }},
    };

    for (const auto& test_case : cases) {
        exec::Int64ArithmeticCandidatePredicate predicate{
            FieldId{},
            DataType::INT64,
            test_case.op,
            test_case.operand,
            test_case.threshold};
        auto prepared = exec::PrepareInt64ArithmeticCandidateEvaluator(
            source, predicate);
        ASSERT_TRUE(prepared.has_value());
        uint64_t valid_mask = 0;
        EXPECT_EQ(prepared->view.eval_batch(prepared->view.context,
                                            row_ids.data(),
                                            row_ids.size(),
                                            UINT64_MAX,
                                            &valid_mask),
                  0);
        uint64_t expected_mask = 0;
        for (size_t lane = 0; lane < values.size(); ++lane) {
            if (test_case.expected(values[lane])) {
                expected_mask |= uint64_t{1} << lane;
            }
        }
        EXPECT_EQ(valid_mask, expected_mask);
    }

    exec::Int64ArithmeticCandidatePredicate divide_by_zero{
        FieldId{},
        DataType::INT64,
        proto::plan::ArithOpType::Div,
        0,
        1};
    EXPECT_FALSE(exec::PrepareInt64ArithmeticCandidateEvaluator(
                     source, divide_by_zero)
                     .has_value());
}

TEST(CandidateEvaluatorTest,
     IntegralFusingCompilationAdmitsOnlyConstrainedMod) {
    auto compile = [](DataType data_type,
                      proto::plan::ArithOpType arithmetic_op,
                      int64_t operand,
                      int64_t threshold) {
        proto::plan::GenericValue value;
        value.set_int64_val(threshold);
        proto::plan::GenericValue right_operand;
        right_operand.set_int64_val(operand);
        expr::BinaryArithOpEvalRangeExpr expression(
            expr::ColumnInfo(FieldId{100}, data_type),
            proto::plan::OpType::LessThan,
            arithmetic_op,
            value,
            right_operand);
        return exec::TryCompileNumericArithmeticCandidateLeaf(expression);
    };

    EXPECT_TRUE(compile(DataType::INT64, proto::plan::ArithOpType::Mod, 5, 2)
                    .has_value());
    EXPECT_FALSE(compile(DataType::INT64, proto::plan::ArithOpType::Mod, 0, 0)
                     .has_value());
    EXPECT_FALSE(compile(DataType::INT64, proto::plan::ArithOpType::Mod, 5, 6)
                     .has_value());

    for (auto arithmetic_op : {proto::plan::ArithOpType::Add,
                               proto::plan::ArithOpType::Sub,
                               proto::plan::ArithOpType::Mul,
                               proto::plan::ArithOpType::Div}) {
        EXPECT_FALSE(compile(DataType::INT64, arithmetic_op, 2, 1).has_value());
    }
    EXPECT_FALSE(
        compile(DataType::TIMESTAMPTZ, proto::plan::ArithOpType::Mod, 5, 2)
            .has_value());
}

TEST(CandidateEvaluatorTest, FloatLeavesPreserveIeeeSemantics) {
    const std::array<float, 9> values = {
        -std::numeric_limits<float>::infinity(),
        -3.5F,
        -0.0F,
        0.0F,
        1.25F,
        3.5F,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        8.0F};
    const std::array<const float*, 2> chunks = {values.data(),
                                                values.data() + 4};
    const std::array<int64_t, 3> chunk_offsets = {0, 4, 9};
    const std::array<int64_t, 9> row_ids = {8, 7, 6, 5, 4, 3, 2, 1, 0};
    exec::FloatCandidateSourceView contiguous;
    contiguous.row_values = values.data();
    contiguous.row_count = values.size();
    exec::FloatCandidateSourceView chunked;
    chunked.row_count = values.size();
    chunked.chunk_values = chunks.data();
    chunked.chunk_offsets = chunk_offsets.data();
    chunked.num_chunks = chunks.size();

    auto check = [&](exec::FloatCandidatePredicate predicate,
                     const std::function<bool(float)>& expected) {
        for (const auto& source : {contiguous, chunked}) {
            auto prepared =
                exec::PrepareFloatCandidateEvaluator(source, predicate);
            ASSERT_TRUE(prepared.has_value());
            uint64_t valid_mask = 0;
            EXPECT_EQ(prepared->view.eval_batch(prepared->view.context,
                                                row_ids.data(),
                                                row_ids.size(),
                                                UINT64_MAX,
                                                &valid_mask),
                      0);
            uint64_t expected_mask = 0;
            for (size_t lane = 0; lane < row_ids.size(); ++lane) {
                if (expected(values[row_ids[lane]])) {
                    expected_mask |= uint64_t{1} << lane;
                }
            }
            EXPECT_EQ(valid_mask, expected_mask);
        }
    };

    exec::FloatCandidatePredicate predicate;
    predicate.op = exec::NumericCandidatePredicateOp::NotEqual;
    predicate.arg0 = 3.5F;
    check(predicate, [](float value) { return value != 3.5; });

    predicate.op = exec::NumericCandidatePredicateOp::Range;
    predicate.arg0 = -3.5F;
    predicate.arg1 = 3.5F;
    predicate.lower_inclusive = false;
    predicate.upper_inclusive = true;
    check(predicate, [](float value) { return value > -3.5 && value <= 3.5; });

    predicate.op = exec::NumericCandidatePredicateOp::Term;
    predicate.terms = {
        std::numeric_limits<float>::quiet_NaN(), -3.5F, 0.0F, 8.0F};
    check(predicate, [](float value) {
        return value == -3.5F || value == 0.0F || value == 8.0F;
    });

    auto check_arithmetic =
        [&](exec::FloatArithmeticCandidatePredicate arithmetic,
            const std::function<bool(float)>& expected) {
            for (const auto& source : {contiguous, chunked}) {
                auto prepared =
                    exec::PrepareFloatArithmeticCandidateEvaluator(
                        source, arithmetic);
                ASSERT_TRUE(prepared.has_value());
                uint64_t valid_mask = 0;
                EXPECT_EQ(prepared->view.eval_batch(prepared->view.context,
                                                    row_ids.data(),
                                                    row_ids.size(),
                                                    UINT64_MAX,
                                                    &valid_mask),
                          0);
                uint64_t expected_mask = 0;
                for (size_t lane = 0; lane < row_ids.size(); ++lane) {
                    if (expected(values[row_ids[lane]])) {
                        expected_mask |= uint64_t{1} << lane;
                    }
                }
                EXPECT_EQ(valid_mask, expected_mask);
            }
        };
    check_arithmetic({FieldId{},
                      DataType::FLOAT,
                      proto::plan::ArithOpType::Add,
                      2.0F,
                      5.0F},
                     [](float value) { return value + 2.0F < 5.0F; });
    check_arithmetic({FieldId{},
                      DataType::FLOAT,
                      proto::plan::ArithOpType::Div,
                      -2.0F,
                      1.0F},
                     [](float value) { return value / -2.0F < 1.0F; });

    exec::FloatArithmeticCandidatePredicate float_divide_by_zero{
        FieldId{},
        DataType::FLOAT,
        proto::plan::ArithOpType::Div,
        -0.0F,
        1.0F};
    EXPECT_FALSE(exec::PrepareFloatArithmeticCandidateEvaluator(
                     contiguous, float_divide_by_zero)
                     .has_value());
}

TEST_P(ExprTest, TestBinaryArithOpEvalRangeExpr_forbigint_mod) {
    // test (bigint mod 10 == 0)
    auto schema = std::make_shared<Schema>();
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    auto json_fid = schema->AddDebugField("json", DataType::JSON);
    schema->set_primary_field_id(int64_fid);

    auto seg = CreateSealedSegment(schema);
    size_t N = 1000;
    auto insert_data = std::make_unique<InsertRecordProto>();
    {
        // insert pk fid
        auto field_meta = schema->operator[](int64_fid);
        std::vector<int64_t> data(N);
        for (int i = 0; i < N; i++) {
            data[i] = i;
        }
        InsertCol(insert_data.get(), data, field_meta, false);
    }

    BitsetType expect(N, false);
    {
        auto field_meta = schema->operator[](json_fid);
        std::vector<std::string> data(N);

        auto start = 1ULL << 54;
        for (int i = 0; i < N; i++) {
            data[i] = R"({"meta":)" + std::to_string(start + i) + "}";
            if ((start + i) % 10 == 0) {
                expect.set(i);
            }
        }
        InsertCol(insert_data.get(), data, field_meta, false);
    }

    GeneratedData raw_data;
    raw_data.schema_ = schema;
    raw_data.raw_ = insert_data.release();
    raw_data.raw_->set_num_rows(N);
    for (int i = 0; i < N; ++i) {
        raw_data.row_ids_.push_back(i);
        raw_data.timestamps_.push_back(i);
    }

    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    proto::plan::GenericValue val1;
    val1.set_int64_val(10);
    proto::plan::GenericValue val2;
    val2.set_int64_val(0);
    auto expr = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
        expr::ColumnInfo(json_fid, DataType::JSON, {"meta"}),
        proto::plan::OpType::Equal,
        proto::plan::ArithOpType::Mod,
        val2,
        val1);

    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    auto final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    EXPECT_EQ(final.size(), expect.size())
        << "final size: " << final.size() << " expect size: " << expect.size();
    for (auto i = 0; i < final.size(); i++) {
        EXPECT_EQ(final[i], expect[i])
            << "i: " << i << " final: " << final[i] << " expect: " << expect[i];
    }
}

TEST_P(ExprTest, TestMutiInConvert) {
    auto schema = std::make_shared<Schema>();
    auto pk = schema->AddDebugField("id", DataType::INT64);
    schema->AddDebugField("bool", DataType::BOOL);
    schema->AddDebugField("bool1", DataType::BOOL);
    schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    schema->AddDebugField("float", DataType::FLOAT);
    schema->AddDebugField("float1", DataType::FLOAT);
    schema->AddDebugField("double", DataType::DOUBLE);
    schema->AddDebugField("double1", DataType::DOUBLE);
    schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    schema->AddDebugField("json", DataType::JSON, false);
    schema->AddDebugField("str_array", DataType::ARRAY, DataType::VARCHAR);
    schema->set_primary_field_id(pk);

    auto seg = CreateSealedSegment(schema);
    size_t N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    auto build_expr = [&](int index) -> expr::TypedExprPtr {
        switch (index) {
            case 0: {
                proto::plan::GenericValue val1;
                val1.set_int64_val(100);
                auto expr1 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    proto::plan::OpType::Equal,
                    val1);
                proto::plan::GenericValue val2;
                val2.set_int64_val(200);
                auto expr2 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    proto::plan::OpType::Equal,
                    val2);
                auto expr3 = std::make_shared<expr::LogicalBinaryExpr>(
                    expr::LogicalBinaryExpr::OpType::Or, expr1, expr2);
                proto::plan::GenericValue val3;
                val3.set_int64_val(300);
                auto expr4 = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    proto::plan::OpType::Equal,
                    val3);
                return std::make_shared<expr::LogicalBinaryExpr>(
                    expr::LogicalBinaryExpr::OpType::Or, expr3, expr4);
            };
            default:
                ThrowInfo(ErrorCode::UnexpectedError, "not implement");
        }
    };

    auto expr = build_expr(0);
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    auto final1 = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    auto prev_optimize_expr_enabled = OPTIMIZE_EXPR_ENABLED.load();
    OPTIMIZE_EXPR_ENABLED.store(false);
    auto final2 = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    EXPECT_EQ(final1.size(), final2.size());
    for (auto i = 0; i < final1.size(); i++) {
        EXPECT_EQ(final1[i], final2[i]);
    }
    OPTIMIZE_EXPR_ENABLED.store(prev_optimize_expr_enabled,
                                std::memory_order_release);
}

TEST(Expr, TestExprPerformance) {
    GTEST_SKIP() << "Skip performance test, open it when test performance";
    auto schema = std::make_shared<Schema>();
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto str2_fid = schema->AddDebugField("string2", DataType::VARCHAR);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    std::map<DataType, FieldId> fids = {{DataType::INT8, int8_fid},
                                        {DataType::INT16, int16_fid},
                                        {DataType::INT32, int32_fid},
                                        {DataType::INT64, int64_fid},
                                        {DataType::VARCHAR, str2_fid},
                                        {DataType::FLOAT, float_fid},
                                        {DataType::DOUBLE, double_fid}};

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    // load field data
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    enum ExprType {
        UnaryRangeExpr = 0,
        TermExprImpl = 1,
        CompareExpr = 2,
        LogicalUnaryExpr = 3,
        BinaryRangeExpr = 4,
        LogicalBinaryExpr = 5,
        BinaryArithOpEvalRangeExpr = 6,
    };

    auto build_unary_range_expr = [&](DataType data_type,
                                      int64_t value) -> expr::TypedExprPtr {
        if (IsIntegerDataType(data_type)) {
            proto::plan::GenericValue val;
            val.set_int64_val(value);
            return std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::LessThan,
                val,
                std::vector<proto::plan::GenericValue>{});
        } else if (IsFloatDataType(data_type)) {
            proto::plan::GenericValue val;
            val.set_float_val(float(value));
            return std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::LessThan,
                val,
                std::vector<proto::plan::GenericValue>{});
        } else if (IsStringDataType(data_type)) {
            proto::plan::GenericValue val;
            val.set_string_val(std::to_string(value));
            return std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::LessThan,
                val,
                std::vector<proto::plan::GenericValue>{});
        } else {
            throw std::runtime_error("not supported type");
        }
    };

    auto build_binary_range_expr = [&](DataType data_type,
                                       int64_t low,
                                       int64_t high) -> expr::TypedExprPtr {
        if (IsIntegerDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_int64_val(low);
            proto::plan::GenericValue val2;
            val2.set_int64_val(high);
            return std::make_shared<expr::BinaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                val1,
                val2,
                true,
                true);
        } else if (IsFloatDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_float_val(float(low));
            proto::plan::GenericValue val2;
            val2.set_float_val(float(high));
            return std::make_shared<expr::BinaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                val1,
                val2,
                true,
                true);
        } else if (IsStringDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_string_val(std::to_string(low));
            proto::plan::GenericValue val2;
            val2.set_string_val(std::to_string(low));
            return std::make_shared<expr::BinaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                val1,
                val2,
                true,
                true);
        } else {
            throw std::runtime_error("not supported type");
        }
    };

    auto build_compare_expr = [&](DataType data_type) -> expr::TypedExprPtr {
        if (IsIntegerDataType(data_type) || IsFloatDataType(data_type) ||
            IsStringDataType(data_type)) {
            return std::make_shared<expr::CompareExpr>(
                fids[data_type],
                fids[data_type],
                data_type,
                data_type,
                proto::plan::OpType::LessThan);
        } else {
            throw std::runtime_error("not supported type");
        }
    };

    auto build_logical_unary_expr =
        [&](DataType data_type) -> expr::TypedExprPtr {
        auto child_expr = build_unary_range_expr(data_type, 10);
        return std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, child_expr);
    };

    auto build_logical_binary_expr =
        [&](DataType data_type) -> expr::TypedExprPtr {
        auto child1_expr = build_unary_range_expr(data_type, 10);
        auto child2_expr = build_unary_range_expr(data_type, 10);
        return std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child1_expr, child2_expr);
    };

    auto build_multi_logical_binary_expr =
        [&](DataType data_type) -> expr::TypedExprPtr {
        auto child1_expr = build_unary_range_expr(data_type, 100);
        auto child2_expr = build_unary_range_expr(data_type, 100);
        auto child3_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child1_expr, child2_expr);
        auto child4_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child1_expr, child2_expr);
        auto child5_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child3_expr, child4_expr);
        auto child6_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child3_expr, child4_expr);
        return std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child5_expr, child6_expr);
    };

    auto build_arith_op_expr = [&](DataType data_type,
                                   int64_t right_val,
                                   int64_t val) -> expr::TypedExprPtr {
        if (IsIntegerDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_int64_val(right_val);
            proto::plan::GenericValue val2;
            val2.set_int64_val(val);
            return std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::Equal,
                proto::plan::ArithOpType::Add,
                val1,
                val2);
        } else if (IsFloatDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_float_val(float(right_val));
            proto::plan::GenericValue val2;
            val2.set_float_val(float(val));
            return std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::Equal,
                proto::plan::ArithOpType::Add,
                val1,
                val2);
        } else {
            throw std::runtime_error("not supported type");
        }
    };

    auto test_case_base = [=, &seg](expr::TypedExprPtr expr) {
        query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        for (int i = 0; i < 100; i++) {
            final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
            EXPECT_EQ(final.size(), N);
        }
    };

    auto expr = build_unary_range_expr(DataType::INT8, 10);
    test_case_base(expr);
    expr = build_unary_range_expr(DataType::INT16, 10);
    test_case_base(expr);
    expr = build_unary_range_expr(DataType::INT32, 10);
    test_case_base(expr);
    expr = build_unary_range_expr(DataType::INT64, 10);
    test_case_base(expr);
    expr = build_unary_range_expr(DataType::FLOAT, 10);
    test_case_base(expr);
    expr = build_unary_range_expr(DataType::DOUBLE, 10);
    test_case_base(expr);
    expr = build_unary_range_expr(DataType::VARCHAR, 10);
    test_case_base(expr);

    expr = build_binary_range_expr(DataType::INT8, 10, 100);
    test_case_base(expr);
    expr = build_binary_range_expr(DataType::INT16, 10, 100);
    test_case_base(expr);
    expr = build_binary_range_expr(DataType::INT32, 10, 100);
    test_case_base(expr);
    expr = build_binary_range_expr(DataType::INT64, 10, 100);
    test_case_base(expr);
    expr = build_binary_range_expr(DataType::FLOAT, 10, 100);
    test_case_base(expr);
    expr = build_binary_range_expr(DataType::DOUBLE, 10, 100);
    test_case_base(expr);
    expr = build_binary_range_expr(DataType::VARCHAR, 10, 100);
    test_case_base(expr);

    expr = build_compare_expr(DataType::INT8);
    test_case_base(expr);
    expr = build_compare_expr(DataType::INT16);
    test_case_base(expr);
    expr = build_compare_expr(DataType::INT32);
    test_case_base(expr);
    expr = build_compare_expr(DataType::INT64);
    test_case_base(expr);
    expr = build_compare_expr(DataType::FLOAT);
    test_case_base(expr);
    expr = build_compare_expr(DataType::DOUBLE);
    test_case_base(expr);
    expr = build_compare_expr(DataType::VARCHAR);
    test_case_base(expr);

    expr = build_arith_op_expr(DataType::INT8, 10, 100);
    test_case_base(expr);
    expr = build_arith_op_expr(DataType::INT16, 10, 100);
    test_case_base(expr);
    expr = build_arith_op_expr(DataType::INT32, 10, 100);
    test_case_base(expr);
    expr = build_arith_op_expr(DataType::INT64, 10, 100);
    test_case_base(expr);
    expr = build_arith_op_expr(DataType::FLOAT, 10, 100);
    test_case_base(expr);
    expr = build_arith_op_expr(DataType::DOUBLE, 10, 100);
    test_case_base(expr);

    expr = build_logical_unary_expr(DataType::INT8);
    test_case_base(expr);
    expr = build_logical_unary_expr(DataType::INT16);
    test_case_base(expr);
    expr = build_logical_unary_expr(DataType::INT32);
    test_case_base(expr);
    expr = build_logical_unary_expr(DataType::INT64);
    test_case_base(expr);
    expr = build_logical_unary_expr(DataType::FLOAT);
    test_case_base(expr);
    expr = build_logical_unary_expr(DataType::DOUBLE);
    test_case_base(expr);
    expr = build_logical_unary_expr(DataType::VARCHAR);
    test_case_base(expr);

    expr = build_logical_binary_expr(DataType::INT8);
    test_case_base(expr);
    expr = build_logical_binary_expr(DataType::INT16);
    test_case_base(expr);
    expr = build_logical_binary_expr(DataType::INT32);
    test_case_base(expr);
    expr = build_logical_binary_expr(DataType::INT64);
    test_case_base(expr);
    expr = build_logical_binary_expr(DataType::FLOAT);
    test_case_base(expr);
    expr = build_logical_binary_expr(DataType::DOUBLE);
    test_case_base(expr);
    expr = build_logical_binary_expr(DataType::VARCHAR);
    test_case_base(expr);

    expr = build_multi_logical_binary_expr(DataType::INT8);
    test_case_base(expr);
    expr = build_multi_logical_binary_expr(DataType::INT16);
    test_case_base(expr);
    expr = build_multi_logical_binary_expr(DataType::INT32);
    test_case_base(expr);
    expr = build_multi_logical_binary_expr(DataType::INT64);
    test_case_base(expr);
    expr = build_multi_logical_binary_expr(DataType::FLOAT);
    test_case_base(expr);
    expr = build_multi_logical_binary_expr(DataType::DOUBLE);
    test_case_base(expr);
    expr = build_multi_logical_binary_expr(DataType::VARCHAR);
    test_case_base(expr);
}

TEST(Expr, TestExprNOT) {
    auto schema = std::make_shared<Schema>();
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8, true);
    schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16, true);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32, true);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64, true);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto str2_fid = schema->AddDebugField("string2", DataType::VARCHAR, true);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT, true);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE, true);
    schema->set_primary_field_id(str1_fid);

    std::map<DataType, FieldId> fids = {{DataType::INT8, int8_fid},
                                        {DataType::INT16, int16_fid},
                                        {DataType::INT32, int32_fid},
                                        {DataType::INT64, int64_fid},
                                        {DataType::VARCHAR, str2_fid},
                                        {DataType::FLOAT, float_fid},
                                        {DataType::DOUBLE, double_fid}};

    auto seg = CreateSealedSegment(schema);
    FixedVector<bool> valid_data_i8;
    FixedVector<bool> valid_data_i16;
    FixedVector<bool> valid_data_i32;
    FixedVector<bool> valid_data_i64;
    FixedVector<bool> valid_data_str;
    FixedVector<bool> valid_data_float;
    FixedVector<bool> valid_data_double;
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    valid_data_i8 = raw_data.get_col_valid(int8_fid);
    valid_data_i16 = raw_data.get_col_valid(int16_fid);
    valid_data_i32 = raw_data.get_col_valid(int32_fid);
    valid_data_i64 = raw_data.get_col_valid(int64_fid);
    valid_data_str = raw_data.get_col_valid(str2_fid);
    valid_data_float = raw_data.get_col_valid(float_fid);
    valid_data_double = raw_data.get_col_valid(double_fid);

    // load field data
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    enum ExprType {
        UnaryRangeExpr = 0,
        TermExprImpl = 1,
        CompareExpr = 2,
        LogicalUnaryExpr = 3,
        BinaryRangeExpr = 4,
        LogicalBinaryExpr = 5,
        BinaryArithOpEvalRangeExpr = 6,
    };

    auto build_unary_range_expr = [&](DataType data_type,
                                      int64_t value) -> expr::TypedExprPtr {
        if (IsIntegerDataType(data_type)) {
            proto::plan::GenericValue val;
            val.set_int64_val(value);
            return std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::LessThan,
                val,
                std::vector<proto::plan::GenericValue>{});
        } else if (IsFloatDataType(data_type)) {
            proto::plan::GenericValue val;
            val.set_float_val(float(value));
            return std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::LessThan,
                val,
                std::vector<proto::plan::GenericValue>{});
        } else if (IsStringDataType(data_type)) {
            proto::plan::GenericValue val;
            val.set_string_val(std::to_string(value));
            return std::make_shared<expr::UnaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::LessThan,
                val,
                std::vector<proto::plan::GenericValue>{});
        } else {
            throw std::runtime_error("not supported type");
        }
    };

    auto build_binary_range_expr = [&](DataType data_type,
                                       int64_t low,
                                       int64_t high) -> expr::TypedExprPtr {
        if (IsIntegerDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_int64_val(low);
            proto::plan::GenericValue val2;
            val2.set_int64_val(high);
            return std::make_shared<expr::BinaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                val1,
                val2,
                true,
                true);
        } else if (IsFloatDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_float_val(float(low));
            proto::plan::GenericValue val2;
            val2.set_float_val(float(high));
            return std::make_shared<expr::BinaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                val1,
                val2,
                true,
                true);
        } else if (IsStringDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_string_val(std::to_string(low));
            proto::plan::GenericValue val2;
            val2.set_string_val(std::to_string(low));
            return std::make_shared<expr::BinaryRangeFilterExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                val1,
                val2,
                true,
                true);
        } else {
            throw std::runtime_error("not supported type");
        }
    };

    auto build_compare_expr = [&](DataType data_type) -> expr::TypedExprPtr {
        if (IsIntegerDataType(data_type) || IsFloatDataType(data_type) ||
            IsStringDataType(data_type)) {
            return std::make_shared<expr::CompareExpr>(
                fids[data_type],
                fids[data_type],
                data_type,
                data_type,
                proto::plan::OpType::LessThan);
        } else {
            throw std::runtime_error("not supported type");
        }
    };

    auto build_logical_binary_expr =
        [&](DataType data_type) -> expr::TypedExprPtr {
        auto child1_expr = build_unary_range_expr(data_type, 10);
        auto child2_expr = build_unary_range_expr(data_type, 10);
        return std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child1_expr, child2_expr);
    };

    auto build_multi_logical_binary_expr =
        [&](DataType data_type) -> expr::TypedExprPtr {
        auto child1_expr = build_unary_range_expr(data_type, 100);
        auto child2_expr = build_unary_range_expr(data_type, 100);
        auto child3_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child1_expr, child2_expr);
        auto child4_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child1_expr, child2_expr);
        auto child5_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child3_expr, child4_expr);
        auto child6_expr = std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child3_expr, child4_expr);
        return std::make_shared<expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child5_expr, child6_expr);
    };

    auto build_arith_op_expr = [&](DataType data_type,
                                   int64_t right_val,
                                   int64_t val) -> expr::TypedExprPtr {
        if (IsIntegerDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_int64_val(right_val);
            proto::plan::GenericValue val2;
            val2.set_int64_val(val);
            return std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::Equal,
                proto::plan::ArithOpType::Add,
                val1,
                val2);
        } else if (IsFloatDataType(data_type)) {
            proto::plan::GenericValue val1;
            val1.set_float_val(float(right_val));
            proto::plan::GenericValue val2;
            val2.set_float_val(float(val));
            return std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                expr::ColumnInfo(fids[data_type], data_type),
                proto::plan::OpType::Equal,
                proto::plan::ArithOpType::Add,
                val1,
                val2);
        } else {
            throw std::runtime_error("not supported type");
        }
    };

    auto build_logical_unary_expr =
        [&](DataType data_type) -> expr::TypedExprPtr {
        auto child_expr = build_unary_range_expr(data_type, 10);
        return std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, child_expr);
    };

    auto test_ans = [=, &seg](expr::TypedExprPtr expr,
                              FixedVector<bool> valid_data) {
        query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan.get(), seg.get(), N, MAX_TIMESTAMP, &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);

        for (int i = 0; i < N; i++) {
            if (!valid_data[i]) {
                EXPECT_EQ(final[i], false);
                if (i % 2 == 0) {
                    EXPECT_EQ(view[int(i / 2)], false);
                }
            }
        }
    };

    auto expr = build_unary_range_expr(DataType::INT8, 10);
    test_ans(expr, valid_data_i8);
    expr = build_unary_range_expr(DataType::INT16, 10);
    test_ans(expr, valid_data_i16);
    expr = build_unary_range_expr(DataType::INT32, 10);
    test_ans(expr, valid_data_i32);
    expr = build_unary_range_expr(DataType::INT64, 10);
    test_ans(expr, valid_data_i64);
    expr = build_unary_range_expr(DataType::FLOAT, 10);
    test_ans(expr, valid_data_float);
    expr = build_unary_range_expr(DataType::DOUBLE, 10);
    test_ans(expr, valid_data_double);
    expr = build_unary_range_expr(DataType::VARCHAR, 10);
    test_ans(expr, valid_data_str);

    expr = build_binary_range_expr(DataType::INT8, 10, 100);
    test_ans(expr, valid_data_i8);
    expr = build_binary_range_expr(DataType::INT16, 10, 100);
    test_ans(expr, valid_data_i16);
    expr = build_binary_range_expr(DataType::INT32, 10, 100);
    test_ans(expr, valid_data_i32);
    expr = build_binary_range_expr(DataType::INT64, 10, 100);
    test_ans(expr, valid_data_i64);
    expr = build_binary_range_expr(DataType::FLOAT, 10, 100);
    test_ans(expr, valid_data_float);
    expr = build_binary_range_expr(DataType::DOUBLE, 10, 100);
    test_ans(expr, valid_data_double);
    expr = build_binary_range_expr(DataType::VARCHAR, 10, 100);
    test_ans(expr, valid_data_str);

    expr = build_compare_expr(DataType::INT8);
    test_ans(expr, valid_data_i8);
    expr = build_compare_expr(DataType::INT16);
    test_ans(expr, valid_data_i16);
    expr = build_compare_expr(DataType::INT32);
    test_ans(expr, valid_data_i32);
    expr = build_compare_expr(DataType::INT64);
    test_ans(expr, valid_data_i64);
    expr = build_compare_expr(DataType::FLOAT);
    test_ans(expr, valid_data_float);
    expr = build_compare_expr(DataType::DOUBLE);
    test_ans(expr, valid_data_double);
    expr = build_compare_expr(DataType::VARCHAR);
    test_ans(expr, valid_data_str);

    expr = build_arith_op_expr(DataType::INT8, 10, 100);
    test_ans(expr, valid_data_i8);
    expr = build_arith_op_expr(DataType::INT16, 10, 100);
    test_ans(expr, valid_data_i16);
    expr = build_arith_op_expr(DataType::INT32, 10, 100);
    test_ans(expr, valid_data_i32);
    expr = build_arith_op_expr(DataType::INT64, 10, 100);
    test_ans(expr, valid_data_i64);
    expr = build_arith_op_expr(DataType::FLOAT, 10, 100);
    test_ans(expr, valid_data_float);
    expr = build_arith_op_expr(DataType::DOUBLE, 10, 100);
    test_ans(expr, valid_data_double);

    expr = build_logical_unary_expr(DataType::INT8);
    test_ans(expr, valid_data_i8);
    expr = build_logical_unary_expr(DataType::INT16);
    test_ans(expr, valid_data_i16);
    expr = build_logical_unary_expr(DataType::INT32);
    test_ans(expr, valid_data_i32);
    expr = build_logical_unary_expr(DataType::INT64);
    test_ans(expr, valid_data_i64);
    expr = build_logical_unary_expr(DataType::FLOAT);
    test_ans(expr, valid_data_float);
    expr = build_logical_unary_expr(DataType::DOUBLE);
    test_ans(expr, valid_data_double);
    expr = build_logical_unary_expr(DataType::VARCHAR);
    test_ans(expr, valid_data_str);

    expr = build_logical_binary_expr(DataType::INT8);
    test_ans(expr, valid_data_i8);
    expr = build_logical_binary_expr(DataType::INT16);
    test_ans(expr, valid_data_i16);
    expr = build_logical_binary_expr(DataType::INT32);
    test_ans(expr, valid_data_i32);
    expr = build_logical_binary_expr(DataType::INT64);
    test_ans(expr, valid_data_i64);
    expr = build_logical_binary_expr(DataType::FLOAT);
    test_ans(expr, valid_data_float);
    expr = build_logical_binary_expr(DataType::DOUBLE);
    test_ans(expr, valid_data_double);
    expr = build_logical_binary_expr(DataType::VARCHAR);
    test_ans(expr, valid_data_str);

    expr = build_multi_logical_binary_expr(DataType::INT8);
    test_ans(expr, valid_data_i8);
    expr = build_multi_logical_binary_expr(DataType::INT16);
    test_ans(expr, valid_data_i16);
    expr = build_multi_logical_binary_expr(DataType::INT32);
    test_ans(expr, valid_data_i32);
    expr = build_multi_logical_binary_expr(DataType::INT64);
    test_ans(expr, valid_data_i64);
    expr = build_multi_logical_binary_expr(DataType::FLOAT);
    test_ans(expr, valid_data_float);
    expr = build_multi_logical_binary_expr(DataType::DOUBLE);
    test_ans(expr, valid_data_double);
    expr = build_multi_logical_binary_expr(DataType::VARCHAR);
    test_ans(expr, valid_data_str);
}

TEST_P(ExprTest, test_term_pk) {
    auto schema = std::make_shared<Schema>();
    schema->AddField(FieldName("Timestamp"),
                     FieldId(1),
                     DataType::INT64,
                     false,
                     std::nullopt);
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    schema->AddDebugField("string1", DataType::VARCHAR);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->set_primary_field_id(int64_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    std::vector<proto::plan::GenericValue> retrieve_ints;
    for (int i = 0; i < 10; ++i) {
        proto::plan::GenericValue val;
        val.set_int64_val(i);
        retrieve_ints.push_back(val);
    }
    auto expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(int64_fid, DataType::INT64), retrieve_ints);
    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    BitsetType final;
    auto plan =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    EXPECT_EQ(final.size(), N);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(final[i], true);
    }
    for (int i = 10; i < N; ++i) {
        EXPECT_EQ(final[i], false);
    }
    retrieve_ints.clear();
    for (int i = 0; i < 10; ++i) {
        proto::plan::GenericValue val;
        val.set_int64_val(i + N);
        retrieve_ints.push_back(val);
    }
    expr = std::make_shared<expr::TermFilterExpr>(
        expr::ColumnInfo(int64_fid, DataType::INT64), retrieve_ints);
    plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
    final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    EXPECT_EQ(final.size(), N);

    // specify some offsets and do scalar filtering on these offsets
    milvus::exec::OffsetVector offsets;
    offsets.reserve(N / 2);
    for (auto i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            offsets.emplace_back(i);
        }
    }
    auto col_vec = milvus::test::gen_filter_res(
        plan.get(), seg.get(), N, MAX_TIMESTAMP, &offsets);
    BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
    EXPECT_EQ(view.size(), N / 2);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(final[i], false);
        if (i % 2 == 0) {
            EXPECT_EQ(view[int(i / 2)], false);
        }
    }
}

TEST_P(ExprTest, TestGrowingSegmentGetBatchSize) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateGrowingSegment(schema, empty_index_meta);
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    seg->PreInsert(N);
    seg->Insert(0,
                N,
                raw_data.row_ids_.data(),
                raw_data.timestamps_.data(),
                raw_data.raw_);

    proto::plan::GenericValue val;
    val.set_int64_val(10);
    auto expr = std::make_shared<expr::UnaryRangeFilterExpr>(
        expr::ColumnInfo(int8_fid, DataType::INT8),
        proto::plan::OpType::GreaterThan,
        val,
        std::vector<proto::plan::GenericValue>{});
    auto plan_node =
        std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);

    std::vector<int64_t> test_batch_size = {
        8192, 10240, 20480, 30720, 40960, 102400, 204800, 307200};

    for (const auto& batch_size : test_batch_size) {
        EXEC_EVAL_EXPR_BATCH_SIZE.store(batch_size);
        auto plan = plan::PlanFragment(plan_node);
        auto query_context = std::make_shared<milvus::exec::QueryContext>(
            "query id", seg.get(), N, MAX_TIMESTAMP);

        auto task =
            milvus::exec::Task::Create("task_expr", plan, 0, query_context);
        auto last_num = N % batch_size;
        auto iter_num = last_num == 0 ? N / batch_size : N / batch_size + 1;
        int iter = 0;
        for (;;) {
            auto result = task->Next();
            if (!result) {
                break;
            }
            auto childrens = result->childrens();
            if (++iter != iter_num) {
                EXPECT_EQ(childrens[0]->size(), batch_size);
            } else {
                EXPECT_EQ(childrens[0]->size(), last_num);
            }
        }
    }
}

TEST_P(ExprTest, TestConjuctExpr) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    schema->AddDebugField("float", DataType::FLOAT);
    schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    // load field data
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);
    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    auto build_expr = [&](int l, int r) -> expr::TypedExprPtr {
        ::milvus::proto::plan::GenericValue value;
        value.set_int64_val(l);
        auto left = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(int64_fid, DataType::INT64),
            proto::plan::OpType::GreaterThan,
            value,
            std::vector<proto::plan::GenericValue>{});
        value.set_int64_val(r);
        auto right = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(int64_fid, DataType::INT64),
            proto::plan::OpType::LessThan,
            value,
            std::vector<proto::plan::GenericValue>{});

        return std::make_shared<milvus::expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, left, right);
    };

    std::vector<std::pair<int, int>> test_case = {
        {100, 0}, {0, 100}, {8192, 8194}};
    for (auto& pair : test_case) {
        auto expr = build_expr(pair.first, pair.second);
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        BitsetType final;
        final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan.get(), seg.get(), N, MAX_TIMESTAMP, &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(final[i], pair.first < i && i < pair.second) << i;
            if (i % 2 == 0) {
                EXPECT_EQ(view[int(i / 2)], pair.first < i && i < pair.second)
                    << i;
            }
        }
    }
}

TEST_P(ExprTest, TestConjuctExprNullable) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int8_nullable", DataType::INT8);
    schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int16_nullable", DataType::INT16);
    schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int32_nullable", DataType::INT32);
    schema->AddDebugField("int64", DataType::INT64);
    auto int64_nullable_fid =
        schema->AddDebugField("int64_nullable", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    schema->AddDebugField("float", DataType::FLOAT);
    schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    auto build_expr = [&](int l, int r) -> expr::TypedExprPtr {
        ::milvus::proto::plan::GenericValue value;
        value.set_int64_val(l);
        auto left = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(int64_nullable_fid, DataType::INT64),
            proto::plan::OpType::GreaterThan,
            value,
            std::vector<proto::plan::GenericValue>{});
        value.set_int64_val(r);
        auto right = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(int64_nullable_fid, DataType::INT64),
            proto::plan::OpType::LessThan,
            value,
            std::vector<proto::plan::GenericValue>{});

        return std::make_shared<milvus::expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, left, right);
    };

    std::vector<std::pair<int, int>> test_case = {
        {100, 0}, {0, 100}, {8192, 8194}};
    for (auto& pair : test_case) {
        auto expr = build_expr(pair.first, pair.second);
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        BitsetType final;
        final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan.get(), seg.get(), N, MAX_TIMESTAMP, &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(final[i], pair.first < i && i < pair.second) << i;
            if (i % 2 == 0) {
                EXPECT_EQ(view[int(i / 2)], pair.first < i && i < pair.second)
                    << i;
            }
        }
    }
}

TEST_P(ExprTest, TestUnaryBenchTest) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    // load field data
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    std::vector<std::pair<FieldId, DataType>> test_cases = {
        {int8_fid, DataType::INT8},
        {int16_fid, DataType::INT16},
        {int32_fid, DataType::INT32},
        {int64_fid, DataType::INT64},
        {float_fid, DataType::FLOAT},
        {double_fid, DataType::DOUBLE}};
    for (const auto& pair : test_cases) {
        proto::plan::GenericValue val;
        if (pair.second == DataType::FLOAT || pair.second == DataType::DOUBLE) {
            val.set_float_val(10);
        } else {
            val.set_int64_val(10);
        }
        auto expr = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pair.first, pair.second),
            proto::plan::OpType::GreaterThan,
            val,
            std::vector<proto::plan::GenericValue>{});
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        for (int i = 0; i < 10; i++) {
            final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        }
    }
}

TEST_P(ExprTest, TestBinaryRangeBenchTest) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    // load field data
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    std::vector<std::pair<FieldId, DataType>> test_cases = {
        {int8_fid, DataType::INT8},
        {int16_fid, DataType::INT16},
        {int32_fid, DataType::INT32},
        {int64_fid, DataType::INT64},
        {float_fid, DataType::FLOAT},
        {double_fid, DataType::DOUBLE}};

    for (const auto& pair : test_cases) {
        proto::plan::GenericValue lower;
        if (pair.second == DataType::FLOAT || pair.second == DataType::DOUBLE) {
            lower.set_float_val(10);
        } else {
            lower.set_int64_val(10);
        }
        proto::plan::GenericValue upper;
        if (pair.second == DataType::FLOAT || pair.second == DataType::DOUBLE) {
            upper.set_float_val(45);
        } else {
            upper.set_int64_val(45);
        }
        auto expr = std::make_shared<expr::BinaryRangeFilterExpr>(
            expr::ColumnInfo(pair.first, pair.second),
            lower,
            upper,
            true,
            true);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        for (int i = 0; i < 10; i++) {
            final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        }
    }
}

TEST_P(ExprTest, TestLogicalUnaryBenchTest) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    // load field data
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    std::vector<std::pair<FieldId, DataType>> test_cases = {
        {int8_fid, DataType::INT8},
        {int16_fid, DataType::INT16},
        {int32_fid, DataType::INT32},
        {int64_fid, DataType::INT64},
        {float_fid, DataType::FLOAT},
        {double_fid, DataType::DOUBLE}};

    for (const auto& pair : test_cases) {
        proto::plan::GenericValue val;
        if (pair.second == DataType::FLOAT || pair.second == DataType::DOUBLE) {
            val.set_float_val(10);
        } else {
            val.set_int64_val(10);
        }
        auto child_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pair.first, pair.second),
            proto::plan::OpType::GreaterThan,
            val,
            std::vector<proto::plan::GenericValue>{});
        auto expr = std::make_shared<expr::LogicalUnaryExpr>(
            expr::LogicalUnaryExpr::OpType::LogicalNot, child_expr);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        for (int i = 0; i < 50; i++) {
            final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        }
    }
}

TEST_P(ExprTest, TestBinaryLogicalBenchTest) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    // load field data
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    std::vector<std::pair<FieldId, DataType>> test_cases = {
        {int8_fid, DataType::INT8},
        {int16_fid, DataType::INT16},
        {int32_fid, DataType::INT32},
        {int64_fid, DataType::INT64},
        {float_fid, DataType::FLOAT},
        {double_fid, DataType::DOUBLE}};

    for (const auto& pair : test_cases) {
        proto::plan::GenericValue val;
        if (pair.second == DataType::FLOAT || pair.second == DataType::DOUBLE) {
            val.set_float_val(-1000000);
        } else {
            val.set_int64_val(-1000000);
        }
        proto::plan::GenericValue val1;
        if (pair.second == DataType::FLOAT || pair.second == DataType::DOUBLE) {
            val1.set_float_val(-100);
        } else {
            val1.set_int64_val(-100);
        }
        auto child1_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pair.first, pair.second),
            proto::plan::OpType::LessThan,
            val,
            std::vector<proto::plan::GenericValue>{});
        auto child2_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(pair.first, pair.second),
            proto::plan::OpType::NotEqual,
            val1,
            std::vector<proto::plan::GenericValue>{});
        auto expr = std::make_shared<const expr::LogicalBinaryExpr>(
            expr::LogicalBinaryExpr::OpType::And, child1_expr, child2_expr);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        for (int i = 0; i < 50; i++) {
            final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        }
    }
}

TEST_P(ExprTest, TestBinaryArithOpEvalRangeBenchExpr) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    std::vector<std::pair<FieldId, DataType>> test_cases = {
        {int8_fid, DataType::INT8},
        {int16_fid, DataType::INT16},
        {int32_fid, DataType::INT32},
        {int64_fid, DataType::INT64},
        {float_fid, DataType::FLOAT},
        {double_fid, DataType::DOUBLE}};

    for (const auto& pair : test_cases) {
        proto::plan::GenericValue val;
        if (pair.second == DataType::FLOAT || pair.second == DataType::DOUBLE) {
            val.set_float_val(100);
        } else {
            val.set_int64_val(100);
        }
        proto::plan::GenericValue right;
        if (pair.second == DataType::FLOAT || pair.second == DataType::DOUBLE) {
            right.set_float_val(10);
        } else {
            right.set_int64_val(10);
        }
        auto expr = std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
            expr::ColumnInfo(pair.first, pair.second),
            proto::plan::OpType::Equal,
            proto::plan::ArithOpType::Add,
            val,
            right);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        for (int i = 0; i < 50; i++) {
            final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        }
    }
}

TEST(BitmapIndexTest, PatternMatchTest) {
    // Initialize bitmap index
    using namespace milvus::index;
    BitmapIndex<std::string> index;

    // Add test data
    std::vector<std::string> data = {"apple", "banana", "orange", "pear"};

    // Build index
    index.Build(data.size(), data.data(), nullptr);

    // Create test datasets with different operators
    auto prefix_dataset = std::make_shared<Dataset>();
    prefix_dataset->Set(OPERATOR_TYPE, OpType::PrefixMatch);
    prefix_dataset->Set(MATCH_VALUE, std::string("a"));

    auto contains_dataset = std::make_shared<Dataset>();
    contains_dataset->Set(OPERATOR_TYPE, OpType::InnerMatch);
    contains_dataset->Set(MATCH_VALUE, std::string("an"));

    auto posix_dataset = std::make_shared<Dataset>();
    posix_dataset->Set(OPERATOR_TYPE, OpType::PostfixMatch);
    posix_dataset->Set(MATCH_VALUE, std::string("a"));

    // Execute queries
    auto prefix_result = index.Query(prefix_dataset);
    auto contains_result = index.Query(contains_dataset);
    auto posix_result = index.Query(posix_dataset);

    // Verify results
    EXPECT_TRUE(prefix_result[0]);
    EXPECT_FALSE(prefix_result[2]);

    EXPECT_FALSE(contains_result[0]);
    EXPECT_TRUE(contains_result[1]);
    EXPECT_TRUE(contains_result[2]);

    EXPECT_FALSE(posix_result[0]);
    EXPECT_TRUE(posix_result[1]);
    EXPECT_FALSE(posix_result[2]);

    auto prefix_result2 =
        index.PatternMatch(std::string("a"), OpType::PrefixMatch);
    auto contains_result2 =
        index.PatternMatch(std::string("an"), OpType::InnerMatch);
    auto posix_result2 =
        index.PatternMatch(std::string("a"), OpType::PostfixMatch);

    EXPECT_TRUE(prefix_result == prefix_result2);
    EXPECT_TRUE(contains_result == contains_result2);
    EXPECT_TRUE(posix_result == posix_result2);
}

TEST(Expr, TestExprNull) {
    auto schema = std::make_shared<Schema>();
    auto bool_fid = schema->AddDebugField("bool", DataType::BOOL, true);
    auto bool_1_fid = schema->AddDebugField("bool1", DataType::BOOL);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8, true);
    auto int8_1_fid = schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16, true);
    auto int16_1_fid = schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32, true);
    auto int32_1_fid = schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64, true);
    auto int64_1_fid = schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto str2_fid = schema->AddDebugField("string2", DataType::VARCHAR, true);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT, true);
    auto float_1_fid = schema->AddDebugField("float1", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE, true);
    auto double_1_fid = schema->AddDebugField("double1", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    std::map<DataType, FieldId> fids = {{DataType::BOOL, bool_fid},
                                        {DataType::INT8, int8_fid},
                                        {DataType::INT16, int16_fid},
                                        {DataType::INT32, int32_fid},
                                        {DataType::INT64, int64_fid},
                                        {DataType::VARCHAR, str2_fid},
                                        {DataType::FLOAT, float_fid},
                                        {DataType::DOUBLE, double_fid}};

    std::map<DataType, FieldId> fids_not_nullable = {
        {DataType::BOOL, bool_1_fid},
        {DataType::INT8, int8_1_fid},
        {DataType::INT16, int16_1_fid},
        {DataType::INT32, int32_1_fid},
        {DataType::INT64, int64_1_fid},
        {DataType::VARCHAR, str1_fid},
        {DataType::FLOAT, float_1_fid},
        {DataType::DOUBLE, double_1_fid}};

    auto seg = CreateSealedSegment(schema);
    FixedVector<bool> valid_data_bool;
    FixedVector<bool> valid_data_i8;
    FixedVector<bool> valid_data_i16;
    FixedVector<bool> valid_data_i32;
    FixedVector<bool> valid_data_i64;
    FixedVector<bool> valid_data_str;
    FixedVector<bool> valid_data_float;
    FixedVector<bool> valid_data_double;

    int N = 1000;
    auto raw_data = DataGen(schema, N);
    valid_data_bool = raw_data.get_col_valid(bool_fid);
    valid_data_i8 = raw_data.get_col_valid(int8_fid);
    valid_data_i16 = raw_data.get_col_valid(int16_fid);
    valid_data_i32 = raw_data.get_col_valid(int32_fid);
    valid_data_i64 = raw_data.get_col_valid(int64_fid);
    valid_data_str = raw_data.get_col_valid(str2_fid);
    valid_data_float = raw_data.get_col_valid(float_fid);
    valid_data_double = raw_data.get_col_valid(double_fid);

    FixedVector<bool> valid_data_all_true(N, true);

    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    auto build_nullable_expr = [&](DataType data_type,
                                   NullExprType op) -> expr::TypedExprPtr {
        return std::make_shared<expr::NullExpr>(
            expr::ColumnInfo(fids[data_type], data_type, {}, true), op);
    };

    auto build_not_nullable_expr = [&](DataType data_type,
                                       NullExprType op) -> expr::TypedExprPtr {
        return std::make_shared<expr::NullExpr>(
            expr::ColumnInfo(
                fids_not_nullable[data_type], data_type, {}, false),
            op);
    };

    auto test_is_null_ans = [=, &seg](expr::TypedExprPtr expr,
                                      FixedVector<bool> valid_data) {
        query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);
        for (int i = 0; i < N; i++) {
            EXPECT_NE(final[i], valid_data[i]);
        }
    };

    auto test_is_not_null_ans = [=, &seg](expr::TypedExprPtr expr,
                                          FixedVector<bool> valid_data) {
        query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);
        for (int i = 0; i < N; i++) {
            EXPECT_EQ(final[i], valid_data[i]);
        }
    };

    auto expr = build_nullable_expr(DataType::BOOL,
                                    proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_bool);
    expr = build_nullable_expr(DataType::INT8,
                               proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_i8);
    expr = build_nullable_expr(DataType::INT16,
                               proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_i16);
    expr = build_nullable_expr(DataType::INT32,
                               proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_i32);
    expr = build_nullable_expr(DataType::INT64,
                               proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_i64);
    expr = build_nullable_expr(DataType::FLOAT,
                               proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_float);
    expr = build_nullable_expr(DataType::DOUBLE,
                               proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_double);
    expr = build_nullable_expr(DataType::FLOAT,
                               proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_float);
    expr = build_nullable_expr(DataType::DOUBLE,
                               proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_double);
    expr = build_nullable_expr(DataType::BOOL,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_bool);
    expr = build_nullable_expr(DataType::INT8,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_i8);
    expr = build_nullable_expr(DataType::INT16,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_i16);
    expr = build_nullable_expr(DataType::INT32,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_i32);
    expr = build_nullable_expr(DataType::INT64,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_i64);
    expr = build_nullable_expr(DataType::FLOAT,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_float);
    expr = build_nullable_expr(DataType::DOUBLE,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_double);
    expr = build_nullable_expr(DataType::FLOAT,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_float);
    expr = build_nullable_expr(DataType::DOUBLE,
                               proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_double);
    //not nullable expr
    expr = build_not_nullable_expr(DataType::BOOL,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::INT8,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::INT16,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::INT32,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::INT64,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::FLOAT,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::DOUBLE,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::FLOAT,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::DOUBLE,
                                   proto::plan::NullExpr_NullOp_IsNull);
    test_is_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::BOOL,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::INT8,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::INT16,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::INT32,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::INT64,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::FLOAT,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::DOUBLE,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::FLOAT,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
    expr = build_not_nullable_expr(DataType::DOUBLE,
                                   proto::plan::NullExpr_NullOp_IsNotNull);
    test_is_not_null_ans(expr, valid_data_all_true);
}

TEST_P(ExprTest, TestCompareExprBenchTest) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    auto int8_1_fid = schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    auto int32_1_fid = schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    auto int64_1_fid = schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto float_1_fid = schema->AddDebugField("float1", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    auto double_1_fid = schema->AddDebugField("double1", DataType::DOUBLE);

    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);

    std::vector<
        std::pair<std::pair<FieldId, DataType>, std::pair<FieldId, DataType>>>
        test_cases = {
            {{int8_fid, DataType::INT8}, {int8_1_fid, DataType::INT8}},
            {{int16_fid, DataType::INT16}, {int16_fid, DataType::INT16}},
            {{int32_fid, DataType::INT32}, {int32_1_fid, DataType::INT32}},
            {{int64_fid, DataType::INT64}, {int64_1_fid, DataType::INT64}},
            {{float_fid, DataType::FLOAT}, {float_1_fid, DataType::FLOAT}},
            {{double_fid, DataType::DOUBLE}, {double_1_fid, DataType::DOUBLE}}};

    for (const auto& pair : test_cases) {
        auto expr = std::make_shared<expr::CompareExpr>(pair.first.first,
                                                        pair.second.first,
                                                        pair.first.second,
                                                        pair.second.second,
                                                        OpType::LessThan);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        for (int i = 0; i < 10; i++) {
            final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
        }
    }
}

TEST_P(ExprTest, TestRefactorExprs) {
    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto int8_fid = schema->AddDebugField("int8", DataType::INT8);
    auto int8_1_fid = schema->AddDebugField("int81", DataType::INT8);
    auto int16_fid = schema->AddDebugField("int16", DataType::INT16);
    schema->AddDebugField("int161", DataType::INT16);
    auto int32_fid = schema->AddDebugField("int32", DataType::INT32);
    schema->AddDebugField("int321", DataType::INT32);
    auto int64_fid = schema->AddDebugField("int64", DataType::INT64);
    schema->AddDebugField("int641", DataType::INT64);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    schema->AddDebugField("string2", DataType::VARCHAR);
    auto float_fid = schema->AddDebugField("float", DataType::FLOAT);
    auto double_fid = schema->AddDebugField("double", DataType::DOUBLE);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);

    LoadGeneratedDataIntoSegment(raw_data, seg.get(), true);

    enum ExprType {
        UnaryRangeExpr = 0,
        TermExprImpl = 1,
        CompareExpr = 2,
        LogicalUnaryExpr = 3,
        BinaryRangeExpr = 4,
        LogicalBinaryExpr = 5,
        BinaryArithOpEvalRangeExpr = 6,
    };

    auto build_expr = [&](enum ExprType test_type,
                          int n) -> expr::TypedExprPtr {
        switch (test_type) {
            case UnaryRangeExpr: {
                proto::plan::GenericValue val;
                val.set_int64_val(10);
                return std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    proto::plan::OpType::GreaterThan,
                    val,
                    std::vector<proto::plan::GenericValue>{});
            }
            case TermExprImpl: {
                std::vector<proto::plan::GenericValue> retrieve_ints;
                for (int i = 0; i < n; ++i) {
                    proto::plan::GenericValue val;
                    val.set_float_val(i);
                    retrieve_ints.push_back(val);
                }
                return std::make_shared<expr::TermFilterExpr>(
                    expr::ColumnInfo(double_fid, DataType::DOUBLE),
                    retrieve_ints);
            }
            case CompareExpr: {
                auto compare_expr =
                    std::make_shared<expr::CompareExpr>(int8_fid,
                                                        int8_1_fid,
                                                        DataType::INT8,
                                                        DataType::INT8,
                                                        OpType::LessThan);
                return compare_expr;
            }
            case BinaryRangeExpr: {
                proto::plan::GenericValue lower;
                lower.set_int64_val(10);
                proto::plan::GenericValue upper;
                upper.set_int64_val(45);
                return std::make_shared<expr::BinaryRangeFilterExpr>(
                    expr::ColumnInfo(int64_fid, DataType::INT64),
                    lower,
                    upper,
                    true,
                    true);
            }
            case LogicalUnaryExpr: {
                proto::plan::GenericValue val;
                val.set_int64_val(10);
                auto child_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int8_fid, DataType::INT8),
                    proto::plan::OpType::GreaterThan,
                    val,
                    std::vector<proto::plan::GenericValue>{});
                return std::make_shared<expr::LogicalUnaryExpr>(
                    expr::LogicalUnaryExpr::OpType::LogicalNot, child_expr);
            }
            case LogicalBinaryExpr: {
                proto::plan::GenericValue val;
                val.set_int64_val(10);
                auto child1_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int8_fid, DataType::INT8),
                    proto::plan::OpType::GreaterThan,
                    val,
                    std::vector<proto::plan::GenericValue>{});
                auto child2_expr = std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int8_fid, DataType::INT8),
                    proto::plan::OpType::NotEqual,
                    val,
                    std::vector<proto::plan::GenericValue>{});
                ;
                return std::make_shared<const expr::LogicalBinaryExpr>(
                    expr::LogicalBinaryExpr::OpType::And,
                    child1_expr,
                    child2_expr);
            }
            case BinaryArithOpEvalRangeExpr: {
                proto::plan::GenericValue val;
                val.set_int64_val(100);
                proto::plan::GenericValue right;
                right.set_int64_val(10);
                return std::make_shared<expr::BinaryArithOpEvalRangeExpr>(
                    expr::ColumnInfo(int8_fid, DataType::INT8),
                    proto::plan::OpType::Equal,
                    proto::plan::ArithOpType::Add,
                    val,
                    right);
            }
            default: {
                proto::plan::GenericValue val;
                val.set_int64_val(10);
                return std::make_shared<expr::UnaryRangeFilterExpr>(
                    expr::ColumnInfo(int8_fid, DataType::INT8),
                    proto::plan::OpType::GreaterThan,
                    val,
                    std::vector<proto::plan::GenericValue>{});
            }
        }
    };
    auto test_case = [&](int n) {
        auto expr = build_expr(UnaryRangeExpr, n);
        query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
        BitsetType final;
        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        final = ExecuteQueryExpr(plan, seg.get(), N, MAX_TIMESTAMP);
    };
    test_case(3);
    test_case(10);
    test_case(20);
    test_case(30);
    test_case(50);
    test_case(100);
    test_case(200);
    // test_case(500);
}

TEST_P(ExprTest, TestCompareWithScalarIndexMaris) {
    std::vector<
        std::tuple<std::string, std::function<bool(std::string, std::string)>>>
        testcases = {
            {"string1 < string2",
             [](std::string a, std::string b) { return a.compare(b) < 0; }},
            {"string1 <= string2",
             [](std::string a, std::string b) { return a.compare(b) <= 0; }},
            {"string1 > string2",
             [](std::string a, std::string b) { return a.compare(b) > 0; }},
            {"string1 >= string2",
             [](std::string a, std::string b) { return a.compare(b) >= 0; }},
            {"string1 == string2",
             [](std::string a, std::string b) { return a.compare(b) == 0; }},
            {"string1 != string2",
             [](std::string a, std::string b) { return a.compare(b) != 0; }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto str2_fid = schema->AddDebugField("string2", DataType::VARCHAR);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    segcore::LoadIndexInfo load_index_info;

    // load index for string1 field
    auto str1_col = raw_data.get_col<std::string>(str1_fid);
    auto str1_index = milvus::index::CreateStringIndexMarisa();
    str1_index->Build(N, str1_col.data());
    load_index_info.field_id = str1_fid.get();
    load_index_info.field_type = DataType::VARCHAR;
    load_index_info.index_params = GenIndexParams(str1_index.get());
    load_index_info.cache_index =
        CreateTestCacheIndex("test", std::move(str1_index));
    seg->LoadIndex(load_index_info);

    // load index for string2 field
    auto str2_col = raw_data.get_col<std::string>(str2_fid);
    auto str2_index = milvus::index::CreateStringIndexMarisa();
    str2_index->Build(N, str2_col.data());
    load_index_info.field_id = str2_fid.get();
    load_index_info.field_type = DataType::VARCHAR;
    load_index_info.index_params = GenIndexParams(str2_index.get());
    load_index_info.cache_index =
        CreateTestCacheIndex("test", std::move(str2_index));
    seg->LoadIndex(load_index_info);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto binary_plan = create_search_plan_from_expr(expr_str);
        auto plan = CreateSearchPlanByExpr(
            schema, binary_plan.data(), binary_plan.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg.get(),
            N,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg.get(),
            N,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);

        for (int i = 0; i < N; ++i) {
            auto ans = final[i];
            auto val1 = str1_col[i];
            auto val2 = str2_col[i];
            auto ref = ref_func(val1, val2);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}

TEST_P(ExprTest, TestCompareWithScalarIndexMarisNullable) {
    std::vector<std::tuple<std::string,
                           std::function<bool(std::string, std::string, bool)>>>
        testcases = {
            {"string1 < nullable_fid",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) < 0;
             }},
            {"string1 <= nullable_fid",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) <= 0;
             }},
            {"string1 > nullable_fid",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) > 0;
             }},
            {"string1 >= nullable_fid",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) >= 0;
             }},
            {"string1 == nullable_fid",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) == 0;
             }},
            {"string1 != nullable_fid",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) != 0;
             }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto nullable_fid =
        schema->AddDebugField("nullable_fid", DataType::VARCHAR, true);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    segcore::LoadIndexInfo load_index_info;

    // load index for string1 field
    auto str1_col = raw_data.get_col<std::string>(str1_fid);
    auto str1_index = milvus::index::CreateStringIndexMarisa();
    str1_index->Build(N, str1_col.data());
    load_index_info.field_id = str1_fid.get();
    load_index_info.field_type = DataType::VARCHAR;
    load_index_info.index_params = GenIndexParams(str1_index.get());
    load_index_info.cache_index =
        CreateTestCacheIndex("test", std::move(str1_index));
    seg->LoadIndex(load_index_info);

    // load index for nullable_fid field
    auto nullable_col = raw_data.get_col<std::string>(nullable_fid);
    auto valid_data_col = raw_data.get_col_valid(nullable_fid);
    auto str2_index = milvus::index::CreateStringIndexMarisa();
    str2_index->Build(N, nullable_col.data(), valid_data_col.data());
    load_index_info.field_id = nullable_fid.get();
    load_index_info.field_type = DataType::VARCHAR;
    load_index_info.index_params = GenIndexParams(str2_index.get());
    load_index_info.cache_index =
        CreateTestCacheIndex("test", std::move(str2_index));
    seg->LoadIndex(load_index_info);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto binary_plan = create_search_plan_from_expr(expr_str);
        auto plan = CreateSearchPlanByExpr(
            schema, binary_plan.data(), binary_plan.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg.get(),
            N,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg.get(),
            N,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);

        for (int i = 0; i < N; ++i) {
            auto ans = final[i];
            auto val1 = str1_col[i];
            auto val2 = nullable_col[i];
            auto ref = ref_func(val1, val2, valid_data_col[i]);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}

TEST_P(ExprTest, TestCompareWithScalarIndexMarisNullable2) {
    std::vector<std::tuple<std::string,
                           std::function<bool(std::string, std::string, bool)>>>
        testcases = {
            {"nullable_fid < string1",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) < 0;
             }},
            {"nullable_fid <= string1",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) <= 0;
             }},
            {"nullable_fid > string1",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) > 0;
             }},
            {"nullable_fid >= string1",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) >= 0;
             }},
            {"nullable_fid == string1",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) == 0;
             }},
            {"nullable_fid != string1",
             [](std::string a, std::string b, bool valid) {
                 if (!valid) {
                     return false;
                 }
                 return a.compare(b) != 0;
             }},
        };

    auto schema = std::make_shared<Schema>();
    schema->AddDebugField("fakevec", data_type, 16, metric_type);
    auto str1_fid = schema->AddDebugField("string1", DataType::VARCHAR);
    auto nullable_fid =
        schema->AddDebugField("nullable_fid", DataType::VARCHAR, true);
    schema->set_primary_field_id(str1_fid);

    auto seg = CreateSealedSegment(schema);
    int N = 1000;
    auto raw_data = DataGen(schema, N);
    segcore::LoadIndexInfo load_index_info;

    // load index for string1 field
    auto str1_col = raw_data.get_col<std::string>(str1_fid);
    auto str1_index = milvus::index::CreateStringIndexMarisa();
    str1_index->Build(N, str1_col.data());
    load_index_info.field_id = str1_fid.get();
    load_index_info.field_type = DataType::VARCHAR;
    load_index_info.index_params = GenIndexParams(str1_index.get());
    load_index_info.cache_index =
        CreateTestCacheIndex("test", std::move(str1_index));
    seg->LoadIndex(load_index_info);

    // load index for nullable_fid field
    auto nullable_col = raw_data.get_col<std::string>(nullable_fid);
    auto valid_data_col = raw_data.get_col_valid(nullable_fid);
    auto str2_index = milvus::index::CreateStringIndexMarisa();
    str2_index->Build(N, nullable_col.data(), valid_data_col.data());
    load_index_info.field_id = nullable_fid.get();
    load_index_info.field_type = DataType::VARCHAR;
    load_index_info.index_params = GenIndexParams(str2_index.get());
    load_index_info.cache_index =
        CreateTestCacheIndex("test", std::move(str2_index));
    seg->LoadIndex(load_index_info);

    query::ExecPlanNodeVisitor visitor(*seg, MAX_TIMESTAMP);
    SetSchema(schema);
    for (auto [expr_str, ref_func] : testcases) {
        auto binary_plan = create_search_plan_from_expr(expr_str);
        auto plan = CreateSearchPlanByExpr(
            schema, binary_plan.data(), binary_plan.size());
        BitsetType final;
        final = ExecuteQueryExpr(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0],
            seg.get(),
            N,
            MAX_TIMESTAMP);
        EXPECT_EQ(final.size(), N);

        // specify some offsets and do scalar filtering on these offsets
        milvus::exec::OffsetVector offsets;
        offsets.reserve(N / 2);
        for (auto i = 0; i < N; ++i) {
            if (i % 2 == 0) {
                offsets.emplace_back(i);
            }
        }
        auto col_vec = milvus::test::gen_filter_res(
            plan->plan_node_->plannodes_->sources()[0]->sources()[0].get(),
            seg.get(),
            N,
            MAX_TIMESTAMP,
            &offsets);
        BitsetTypeView view(col_vec->GetRawData(), col_vec->size());
        EXPECT_EQ(view.size(), N / 2);

        for (int i = 0; i < N; ++i) {
            auto ans = final[i];
            auto val1 = nullable_col[i];
            auto val2 = str1_col[i];
            auto ref = ref_func(val1, val2, valid_data_col[i]);
            ASSERT_EQ(ans, ref) << expr_str << "@" << i << "!!"
                                << boost::format("[%1%, %2%]") % val1 % val2;
            if (i % 2 == 0) {
                ASSERT_EQ(view[int(i / 2)], ref)
                    << expr_str << "@" << i << "!!"
                    << boost::format("[%1%, %2%]") % val1 % val2;
            }
        }
    }
}
