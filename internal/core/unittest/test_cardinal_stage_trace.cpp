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

#include <stdexcept>
#include <string>
#include <vector>

#include "index/CardinalStageTraceMetadata.h"
#include "segcore/CardinalStageTrace.h"

namespace {

using Json = nlohmann::json;

Json
TraceEnvelope(size_t coarse_count = 257) {
    Json coarse = Json::array();
    for (size_t i = 0; i < coarse_count; ++i) {
        coarse.push_back({{"offset", i * 3 + 2}, {"score", 0.9 - i * 0.001}});
    }
    Json query{{"query_index", 0},
               {"query_hash", "0123456789abcdef"},
               {"local_topk", 2},
               {"candidate_limit", coarse_count},
               {"level", 7},
               {"metric", "COSINE"},
               {"capture_complete", true},
               {"coarse_compute_count", 12345},
               {"phase_us", {{"fine", 24.5}}},
               {"coarse", coarse},
               {"local", Json::array({coarse[4], coarse[1]})}};
    return {{"cardinal_stage_trace",
             {{"schema_version", 1},
              {"capture_complete", true},
              {"expected_queries", 1},
              {"request_tag", "hotpotqa:saq13:q0007"},
              {"tag_valid", true},
              {"queries", Json::array({query})}}}};
}

std::vector<int64_t>
NonIdentityPks(const std::vector<int64_t>& offsets) {
    std::vector<int64_t> result;
    for (auto offset : offsets) {
        result.push_back(1000003 + offset * 17);
    }
    return result;
}

TEST(CardinalStageTrace, CorrelationTagDoesNotReachSearchParameters) {
    Json params{{"trace_visit", true},
                {"debug_stage_trace_tag", "laion:saq22:q0016"},
                {"level", 7},
                {"search_list", 319}};
    const auto request = milvus::index::TakeCardinalStageTraceRequest(params);
    EXPECT_TRUE(request.enabled);
    EXPECT_TRUE(request.tag_valid);
    EXPECT_EQ(request.tag, "laion:saq22:q0016");
    EXPECT_FALSE(params.contains("debug_stage_trace_tag"));
    EXPECT_EQ(params.at("level"), 7);
    EXPECT_EQ(params.at("search_list"), 319);
    EXPECT_TRUE(params.at("trace_visit").get<bool>());
}

TEST(CardinalStageTrace, DisabledCaptureDoesNotCopyMetadata) {
    auto dataset = std::make_shared<knowhere::DataSet>();
    dataset->SetJsonInfo(TraceEnvelope().dump());
    std::string result;
    result = "stale previous request";
    milvus::index::PreserveCardinalStageTrace(
        dataset, milvus::index::CardinalStageTraceRequest{}, result);
    EXPECT_TRUE(result.empty());
}

TEST(CardinalStageTrace, MissingMetadataCannotReusePreviousCapture) {
    auto dataset = std::make_shared<knowhere::DataSet>();
    dataset->SetJsonInfo(TraceEnvelope().dump());
    std::string result;
    const milvus::index::CardinalStageTraceRequest tagged{
        true, true, "hotpotqa:saq13:q0007"};
    milvus::index::PreserveCardinalStageTrace(dataset, tagged, result);
    ASSERT_FALSE(result.empty());
    dataset->SetJsonInfo("");
    milvus::index::PreserveCardinalStageTrace(dataset, tagged, result);
    const auto incomplete = Json::parse(result);
    EXPECT_FALSE(
        incomplete["cardinal_stage_trace"]["capture_complete"].get<bool>());
    EXPECT_EQ(incomplete["cardinal_stage_trace"]["request_tag"], tagged.tag);
    dataset->SetJsonInfo("{\"unrelated_backend_visit\":{}}");
    milvus::index::PreserveCardinalStageTrace(
        dataset,
        milvus::index::CardinalStageTraceRequest{true, false, ""},
        result);
    EXPECT_TRUE(result.empty());
}

TEST(CardinalStageTrace, MapsActualOffsetsAndProvidesCompleteChunks) {
    const auto records = milvus::segcore::BuildCardinalStageTraceRecords(
        TraceEnvelope().dump(), 77, 101, NonIdentityPks);
    ASSERT_EQ(records.size(), 5);  // C:3chunks, L:1chunk, manifest.
    EXPECT_EQ(records[0].at("stage"), "C");
    EXPECT_EQ(records[0].at("candidate_count"), 257);
    EXPECT_EQ(records[0].at("chunk_count"), 3);
    EXPECT_EQ(records[0].at("candidates").size(), 128);
    const auto first = records[0].at("candidates")[0];
    EXPECT_EQ(first.at("offset"), 2);
    EXPECT_EQ(first.at("pk"), 1000037);
    EXPECT_NE(first.at("offset"), first.at("pk"));
    EXPECT_EQ(records[0].at("chunk_hash"),
              milvus::segcore::StageTraceCandidateChecksum(
                  records[0].at("candidates")));
    const auto local = records[3].at("candidates");
    EXPECT_EQ(records[3].at("stage"), "L");
    EXPECT_EQ(local[0].at("offset"), 14);
    EXPECT_EQ(local[1].at("offset"), 5);
    EXPECT_EQ(records.back().at("record_type"), "manifest");
    EXPECT_TRUE(records.back().at("capture_complete").get<bool>());
    EXPECT_EQ(records.back().at("expected_chunk_records"), 4);
    EXPECT_EQ(records.back().at("coarse_compute_count"), 12345);
    EXPECT_EQ(records.back().at("phase_us").at("fine"), 24.5);
}

TEST(CardinalStageTrace, ChecksumUsesPortableBinary32Scores) {
    const Json rows = Json::array(
        {{{"position", 0}, {"offset", 2}, {"pk", 1000037}, {"score", 1.0}}});
    EXPECT_EQ(milvus::segcore::StageTraceCandidateChecksum(rows),
              milvus::segcore::StageTraceFnv1a64("0:2:1000037:1065353216;"));
}

TEST(CardinalStageTrace, IncompleteInputOrWrongMappingCannotProduceSuccess) {
    auto envelope = TraceEnvelope();
    envelope["cardinal_stage_trace"]["queries"][0]["capture_complete"] = false;
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
    EXPECT_THROW(
        milvus::segcore::BuildCardinalStageTraceRecords(
            TraceEnvelope().dump(),
            77,
            101,
            [](const std::vector<int64_t>&) { return std::vector<int64_t>{}; }),
        std::invalid_argument);
}

TEST(CardinalStageTrace, LocalIdsMustBelongToTheRealCoarsePool) {
    auto envelope = TraceEnvelope();
    envelope["cardinal_stage_trace"]["queries"][0]["local"][0]["offset"] =
        99999;
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
}

TEST(CardinalStageTrace, GlobalBatchCompletenessCannotBeIgnored) {
    auto envelope = TraceEnvelope();
    envelope["cardinal_stage_trace"]["capture_complete"] = false;
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
    envelope = TraceEnvelope();
    envelope["cardinal_stage_trace"]["expected_queries"] = 2;
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
    envelope = TraceEnvelope();
    envelope["cardinal_stage_trace"]["expected_queries"] = 2;
    auto duplicate = envelope["cardinal_stage_trace"]["queries"][0];
    envelope["cardinal_stage_trace"]["queries"].push_back(duplicate);
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
}

TEST(CardinalStageTrace, MissingOrUnsafeTagsInvalidateCapture) {
    Json config{{"trace_visit", true},
                {"debug_stage_trace_tag", "bad\ncorrelation"}};
    const auto request = milvus::index::TakeCardinalStageTraceRequest(config);
    EXPECT_FALSE(request.tag_valid);
    EXPECT_TRUE(request.tag.empty());
    auto envelope = TraceEnvelope();
    envelope["cardinal_stage_trace"]["tag_valid"] = false;
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
}

TEST(CardinalStageTrace, DiagnosticLimitsCannotProducePartialSuccess) {
    auto envelope = TraceEnvelope(8193);
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
    envelope = TraceEnvelope();
    auto& trace = envelope["cardinal_stage_trace"];
    const auto query = trace["queries"][0];
    trace["queries"] = Json::array();
    trace["expected_queries"] = 17;
    for (size_t i = 0; i < 17; ++i) {
        auto next = query;
        next["query_index"] = i;
        trace["queries"].push_back(next);
    }
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
    envelope = TraceEnvelope();
    envelope["cardinal_stage_trace"]["queries"][0]["coarse"][0]["score"] =
        1e100;
    EXPECT_THROW(milvus::segcore::BuildCardinalStageTraceRecords(
                     envelope.dump(), 77, 101, NonIdentityPks),
                 std::invalid_argument);
}

TEST(CardinalStageTrace, MetadataFailuresAreDiagnosticOnly) {
    std::string result = "stale";
    const milvus::index::CardinalStageTraceRequest request{
        true, true, "hotpotqa:saq13:q0007"};
    EXPECT_NO_THROW(
        milvus::index::PreserveCardinalStageTrace(nullptr, request, result));
    ASSERT_FALSE(result.empty());
    EXPECT_FALSE(Json::parse(result)["cardinal_stage_trace"]["capture_complete"]
                     .get<bool>());
    auto dataset = std::make_shared<knowhere::DataSet>();
    dataset->SetJsonInfo("not JSON");
    EXPECT_NO_THROW(
        milvus::index::PreserveCardinalStageTrace(dataset, request, result));
    EXPECT_FALSE(Json::parse(result)["cardinal_stage_trace"]["capture_complete"]
                     .get<bool>());
}

}  // namespace
