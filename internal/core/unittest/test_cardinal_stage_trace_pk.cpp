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

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "common/OpContext.h"
#include "common/QueryResult.h"
#include "common/Schema.h"
#include "common/Types.h"
#include "index/IndexFactory.h"
#include "index/IndexInfo.h"
#include "index/IndexStats.h"
#include "index/VectorDiskIndex.h"
#include "index/VectorIndex.h"
#include "knowhere/dataset.h"
#include "knowhere/operands.h"
#include "pb/common.pb.h"
#include "pb/plan.pb.h"
#include "query/Plan.h"
#include "query/PlanImpl.h"
#include "query/Utils.h"
#include "segcore/CardinalStageTrace.h"
#include "segcore/ChunkedSegmentSealedImpl.h"
#include "segcore/SegmentInterface.h"
#include "segcore/SegmentSealed.h"
#include "segcore/Utils.h"
#include "storage/RemoteChunkManagerSingleton.h"
#include "storage/Util.h"
#include "test_utils/DataGen.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "test_utils/storage_test_utils.h"

using namespace milvus;

TEST(CardinalStageTracePk, UsesActualPinnedPrimaryKeys) {
    constexpr int64_t rows = 16;
    auto schema = std::make_shared<Schema>();
    const auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, 4, knowhere::metric::L2);
    schema->set_primary_field_id(pk);
    auto dataset = segcore::DataGen(schema, rows);
    for (auto& field : *dataset.raw_->mutable_fields_data()) {
        if (field.field_id() == pk.get()) {
            for (int64_t i = 0; i < rows; ++i) {
                field.mutable_scalars()->mutable_long_data()->set_data(
                    i, 1000003 + i * 17);
            }
        }
    }
    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();
    const auto load = PrepareInsertBinlog(
        kCollectionID, kPartitionID, kSegmentID, dataset, cm);
    auto segment =
        segcore::CreateSealedSegment(schema, empty_index_meta, kSegmentID);
    segment->LoadFieldData(load);
    const auto snapshot = segment->CaptureReadSnapshot();
    ASSERT_NE(snapshot, nullptr);

    // Real APIs preserve a row's PK. Deliberately publish different real PK
    // data through the test hook so old-vs-latest snapshot reads are observable.
    auto* chunked =
        dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(segment.get());
    ASSERT_NE(chunked, nullptr);
    auto newer_dataset = segcore::DataGen(schema, rows, 43);
    for (auto& field : *newer_dataset.raw_->mutable_fields_data()) {
        if (field.field_id() == pk.get()) {
            for (int64_t i = 0; i < rows; ++i) {
                field.mutable_scalars()->mutable_long_data()->set_data(
                    i, 9000007 + i * 31);
            }
        }
    }
    auto newer_segment = segcore::CreateSealedSegment(
        schema, empty_index_meta, kSegmentID + 1);
    newer_segment->LoadFieldData(PrepareInsertBinlog(
        kCollectionID, kPartitionID, kSegmentID + 1, newer_dataset, cm));
    auto* newer_chunked = dynamic_cast<segcore::ChunkedSegmentSealedImpl*>(
        newer_segment.get());
    ASSERT_NE(newer_chunked, nullptr);
    auto newer_runtime = newer_chunked->TestCloneMutableRuntimeResourceState();
    chunked->TestPublishRuntimeResourceState(std::move(newer_runtime));
    const auto newer_snapshot = segment->CaptureReadSnapshot();
    ASSERT_NE(
        segcore::ChunkedSegmentSealedImpl::ToPublishedState(snapshot),
        segcore::ChunkedSegmentSealedImpl::ToPublishedState(newer_snapshot));
    const int64_t first_offset = 3;
    milvus::OpContext newer_context;
    const auto newer_values = segcore::BulkSubscriptWithSnapshot(
        segment.get(), newer_snapshot, &newer_context, pk,
        schema->operator[](pk), &first_offset, 1);
    ASSERT_EQ(newer_values->scalars().long_data().data_size(), 1);
    EXPECT_EQ(newer_values->scalars().long_data().data(0), 9000100);
    using Json = nlohmann::json;
    Json coarse = Json::array({{{"offset", 3}, {"score", .1}},
                               {{"offset", 7}, {"score", .2}},
                               {{"offset", 11}, {"score", .3}}});
    const Json metadata{
        {"cardinal_stage_trace",
         {{"schema_version", 1},
          {"capture_complete", true},
          {"expected_queries", 1},
          {"request_tag", "sealed:pk-map:q0"},
          {"tag_valid", true},
          {"queries",
           Json::array({{{"query_index", 0},
                         {"query_hash", "0123456789abcdef"},
                         {"local_topk", 1},
                         {"candidate_limit", 3},
                         {"level", 1},
                         {"metric", "L2"},
                         {"capture_complete", true},
                         {"coarse", coarse},
                         {"local", Json::array({coarse[1]})}}})}}}};
    const auto records = segcore::BuildCardinalStageTraceRecords(
        metadata.dump(),
        kSegmentID,
        pk.get(),
        [&](const std::vector<int64_t>& offsets) {
            milvus::OpContext context;
            auto data =
                segcore::BulkSubscriptWithSnapshot(segment.get(),
                                                   snapshot,
                                                   &context,
                                                   pk,
                                                   schema->operator[](pk),
                                                   offsets.data(),
                                                   offsets.size());
            const auto& values = data->scalars().long_data().data();
            return std::vector<int64_t>(values.begin(), values.end());
        });
    ASSERT_EQ(records.size(), 3);
    EXPECT_EQ(records[0]["candidates"][0]["pk"], 1000054);
    EXPECT_EQ(records[0]["candidates"][2]["pk"], 1000190);
    query::Plan plan(schema);
    SearchResult ordinary;
    ordinary.seg_offsets_ = {7};
    ordinary.distances_ = {0.0F};
    ordinary.read_snapshot_ = snapshot;
    segment->FillPrimaryKeys(&plan, ordinary);
    ASSERT_EQ(ordinary.primary_keys_.size(), 1);
    // Current FillPrimaryKeys uses latest state. Its behavior is not changed
    // here; the diagnostic resolver must independently retain the pinned state.
    EXPECT_EQ(std::get<int64_t>(ordinary.primary_keys_[0]), 9000224);
    EXPECT_EQ(records[1]["candidates"][0]["pk"], 1000122);
    EXPECT_NE(records[1]["candidates"][0]["pk"],
              std::get<int64_t>(ordinary.primary_keys_[0]));
    EXPECT_NE(records[1]["candidates"][0]["pk"], 7);
}

TEST(CardinalStageTracePk, NormalSegmentSearchCarriesActualCardinalCandidates) {
    constexpr int64_t rows = 1024;
    constexpr int64_t dim = 128;
    constexpr int64_t topk = 10;
    constexpr int64_t query_row = 17;
    auto schema = std::make_shared<Schema>();
    const auto vec = schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, dim, knowhere::metric::L2);
    const auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto dataset = segcore::DataGen(schema, rows, 42);
    for (auto& field : *dataset.raw_->mutable_fields_data()) {
        if (field.field_id() == pk.get()) {
            for (int64_t i = 0; i < rows; ++i) {
                field.mutable_scalars()->mutable_long_data()->set_data(
                    i, 1000003 + i * 17);
            }
        }
    }
    const auto vectors = dataset.get_col<float>(vec);
    index::CreateIndexInfo create_info;
    create_info.field_type = DataType::VECTOR_FLOAT;
    create_info.metric_type = knowhere::metric::L2;
    create_info.index_type = knowhere::IndexEnum::INDEX_HNSW;
    create_info.index_engine_version = 11;
    auto indexing = index::IndexFactory::GetInstance().CreateIndex(
        create_info, storage::FileManagerContext());
    const knowhere::Json build_params{{"metric_type", "L2"},
                                      {"M", 16},
                                      {"efConstruction", 64},
                                      {"max_degree", 16},
                                      {"search_list_size", 64},
                                      {"graph_build_algo", "Vamana"},
                                      {"graph_search_dim_threshold", 0},
                                      {"search_quant_type", "RBQ2"},
                                      {"refine_quant_type", "RBQ4"},
                                      {"skip_preheat", true}};
    indexing->BuildWithDataset(knowhere::GenDataSet(rows, dim, vectors.data()),
                               build_params);
    segcore::LoadIndexInfo load_index;
    load_index.field_id = vec.get();
    load_index.index_params = GenIndexParams(indexing.get());
    load_index.index_params["metric_type"] = "L2";
    load_index.cache_index =
        CreateTestCacheIndex("cardinal-stage-trace", std::move(indexing));

    const auto cm = storage::RemoteChunkManagerSingleton::GetInstance()
                        .GetRemoteChunkManager();
    auto segment =
        segcore::CreateSealedSegment(schema, empty_index_meta, kSegmentID);
    segment->LoadFieldData(PrepareInsertBinlog(
        kCollectionID, kPartitionID, kSegmentID, dataset, cm));
    segment->DropFieldData(vec);
    segment->LoadIndex(load_index);
    auto make_plan = [&](bool trace) {
        knowhere::Json params{{"ef", 64},
                              {"level", 3},
                              {"refine_ratio", 1.5},
                              {"skip_refine", false}};
        if (trace) {
            params["trace_visit"] = true;
            params["debug_stage_trace_tag"] = "local:segment-search:q0";
        }
        proto::plan::PlanNode node;
        auto* anns = node.mutable_vector_anns();
        anns->set_vector_type(proto::plan::VectorType::FloatVector);
        anns->set_field_id(vec.get());
        anns->set_placeholder_tag("$0");
        auto* info = anns->mutable_query_info();
        info->set_topk(topk);
        info->set_metric_type("L2");
        info->set_search_params(params.dump());
        info->set_round_decimal(-1);
        return query::CreateSearchPlanFromPlanNode(schema, node);
    };
    auto plain_plan = make_plan(false);
    auto trace_plan = make_plan(true);
    const auto placeholder = segcore::CreatePlaceholderGroupFromBlob(
        1, dim, vectors.data() + query_row * dim);
    const auto plain_group = query::ParsePlaceholderGroup(
        plain_plan.get(), placeholder.SerializeAsString());
    const auto trace_group = query::ParsePlaceholderGroup(
        trace_plan.get(), placeholder.SerializeAsString());
    auto plain = segment->Search(plain_plan.get(), plain_group.get(), 1000000);
    auto traced = segment->Search(trace_plan.get(), trace_group.get(), 1000000);
    ASSERT_EQ(traced->seg_offsets_, plain->seg_offsets_);
    ASSERT_EQ(traced->distances_, plain->distances_);
    ASSERT_EQ(traced->distances_.size(), topk);
    ASSERT_EQ(std::memcmp(traced->distances_.data(), plain->distances_.data(),
                          traced->distances_.size() * sizeof(float)), 0);
    EXPECT_TRUE(plain->cardinal_stage_trace_json_.empty());
    ASSERT_FALSE(traced->cardinal_stage_trace_json_.empty());
    const auto envelope =
        nlohmann::json::parse(traced->cardinal_stage_trace_json_);
    const auto& trace = envelope.at("cardinal_stage_trace");
    ASSERT_TRUE(trace.at("capture_complete").get<bool>());
    ASSERT_EQ(trace.at("request_tag"), "local:segment-search:q0");
    const auto& query_trace = trace.at("queries").at(0);
    EXPECT_GT(query_trace.at("refine_compute_count").get<int64_t>(), 0);
    EXPECT_GT(query_trace.at("coarse").size(), topk);
    ASSERT_EQ(query_trace.at("local").size(), topk);
    ASSERT_EQ(traced->seg_offsets_.size(), topk);
    segment->FillPrimaryKeys(trace_plan.get(), *traced);
    for (size_t i = 0; i < topk; ++i) {
        EXPECT_EQ(query_trace.at("local").at(i).at("offset"),
                  traced->seg_offsets_[i]);
        EXPECT_FLOAT_EQ(query_trace.at("local").at(i).at("score").get<float>(),
                        traced->distances_[i]);
        EXPECT_EQ(std::get<int64_t>(traced->primary_keys_[i]),
                  1000003 + traced->seg_offsets_[i] * 17);
    }
    if (const auto* artifact_path =
            std::getenv("MILVUS_STAGE_TRACE_RESULT_PATH")) {
        std::vector<int64_t> pks;
        for (const auto& value : traced->primary_keys_) {
            pks.push_back(std::get<int64_t>(value));
        }
        const nlohmann::json artifact{
            {"schema_version", 1},
            {"segment_id", kSegmentID},
            {"field_id", vec.get()},
            {"request_tag", trace.at("request_tag")},
            {"query_hash", query_trace.at("query_hash")},
            {"query_vector_fp32", std::vector<float>(
                 vectors.data() + query_row * dim,
                 vectors.data() + (query_row + 1) * dim)},
            {"normal_offsets", traced->seg_offsets_},
            {"normal_pks", pks},
            {"normal_scores", traced->distances_},
            {"coarse_count", query_trace.at("coarse").size()},
            {"local_count", query_trace.at("local").size()}};
        std::ofstream output(artifact_path);
        ASSERT_TRUE(output.is_open());
        output << artifact.dump(2) << std::endl;
        ASSERT_TRUE(output.good());
    }
}

#ifdef BUILD_DISK_ANN
namespace {

void
CheckTieredSaqSegmentTrace(int coarse_bits) {
    SCOPED_TRACE("CARDINAL_TIERED SAQ" + std::to_string(coarse_bits));
    constexpr int64_t rows = 4096;
    constexpr int64_t dim = 640;
    constexpr int64_t topk = 10;
    constexpr int64_t query_row = 17;
    auto schema = std::make_shared<Schema>();
    const auto vec = schema->AddDebugField(
        "vec", DataType::VECTOR_FLOAT, dim, knowhere::metric::COSINE);
    const auto pk = schema->AddDebugField("pk", DataType::INT64);
    schema->set_primary_field_id(pk);
    auto dataset = segcore::DataGen(schema, rows, 42);
    std::mt19937 generator(20261027);
    std::normal_distribution<float> normal;
    for (auto& field : *dataset.raw_->mutable_fields_data()) {
        if (field.field_id() == pk.get()) {
            for (int64_t i = 0; i < rows; ++i) {
                field.mutable_scalars()->mutable_long_data()->set_data(
                    i, 1000003 + i * 17);
            }
        } else if (field.field_id() == vec.get()) {
            auto* values = field.mutable_vectors()
                               ->mutable_float_vector()
                               ->mutable_data();
            std::vector<float> row(dim);
            for (int64_t i = 0; i < rows; ++i) {
                double norm = 0;
                for (int64_t j = 0; j < dim; ++j) {
                    row[j] = normal(generator) *
                             std::exp(-1.5f * static_cast<float>(j) / dim);
                    norm += static_cast<double>(row[j]) * row[j];
                }
                norm = std::sqrt(norm);
                for (int64_t j = 0; j < dim; ++j) {
                    values->Set(i * dim + j,
                                static_cast<float>(row[j] / norm));
                }
            }
        }
    }
    const auto vectors = dataset.get_col<float>(vec);
    auto storage_config = get_default_local_storage_config();
    auto chunk_manager = storage::CreateChunkManager(storage_config);
    auto fs = storage::InitArrowFileSystem(storage_config);
    const storage::FieldDataMeta field_meta{
        kCollectionID, kPartitionID, kSegmentID, vec.get()};
    const storage::IndexMeta index_meta{
        kSegmentID, vec.get(), 910000 + coarse_bits, 1};
    const storage::FileManagerContext context(
        field_meta, index_meta, chunk_manager, fs);
    index::CreateIndexInfo create;
    create.field_type = DataType::VECTOR_FLOAT;
    create.metric_type = knowhere::metric::COSINE;
    create.index_type = "CARDINAL_TIERED";
    create.index_engine_version = 11;
    auto indexing =
        index::IndexFactory::GetInstance().CreateIndex(create, context);
    ASSERT_NE(dynamic_cast<index::VectorDiskAnnIndex<knowhere::fp32>*>(
                  indexing.get()),
              nullptr);
    // Exact HotpotQA build-budget inputs, on a small full-rank local fixture.
    // This checks the actual disk/SQ8/SAQ/SAQM registration and storage path;
    // it is neither a benchmark nor a claim that the learned plan is identical.
    const knowhere::Json build{
        {"metric_type", "COSINE"},
        {"dim", dim},
        {"index_algo", "IVF"},
        {"nlist", 32},
        {"build_quant_type", "SQ8"},
        {"search_quant_type", "SAQ" + std::to_string(coarse_bits)},
        {"refine_quant_type", "SAQM"},
        {"retrieve_friendly", true},
        {"skip_refine", false},
        {"saq_budget_in_bytes", coarse_bits == 1 ? 105 : 193},
        {"saq_nested_total_budget_in_bytes", 386},
        {"saq_min_budget_in_bytes", coarse_bits == 1 ? 87 : 159},
        {"saq_nested_min_refine_budget_in_bytes",
         coarse_bits == 1 ? 231 : 159},
        {"skip_preheat", true}};
    indexing->BuildWithDataset(
        knowhere::GenDataSet(rows, dim, vectors.data()), build);
    const auto uploaded = indexing->Upload();
    const auto index_files = uploaded->GetIndexFiles();
    ASSERT_FALSE(index_files.empty());
    indexing.reset();
    auto loaded =
        index::IndexFactory::GetInstance().CreateIndex(create, context);
    auto* disk = dynamic_cast<index::VectorDiskAnnIndex<knowhere::fp32>*>(
        loaded.get());
    ASSERT_NE(disk, nullptr);
    knowhere::Json load{{"metric_type", "COSINE"},
                         {"dim", dim},
                         {"index_files", index_files},
                         {"skip_preheat", true}};
    load[milvus::LOAD_PRIORITY] =
        milvus::proto::common::LoadPriority::HIGH;
    disk->Load(milvus::tracer::TraceContext{}, load);
    ASSERT_EQ(disk->Count(), rows);
    ASSERT_EQ(disk->GetDim(), dim);

    segcore::LoadIndexInfo load_index;
    load_index.field_id = vec.get();
    load_index.index_params = GenIndexParams(loaded.get());
    load_index.cache_index = CreateTestCacheIndex(
        "cardinal-stage-trace-tiered-" + std::to_string(coarse_bits),
        std::move(loaded));
    auto segment =
        segcore::CreateSealedSegment(schema, empty_index_meta, kSegmentID);
    segment->LoadFieldData(PrepareInsertBinlog(
        kCollectionID, kPartitionID, kSegmentID, dataset, chunk_manager));
    segment->DropFieldData(vec);
    segment->LoadIndex(load_index);
    const auto tag = "local:tiered-saq" + std::to_string(coarse_bits) + ":q17";
    auto make_plan = [&](bool trace) {
        knowhere::Json params{{"ef", 64},
                              {"level", 3},
                              {"refine_ratio", 1.5},
                              {"skip_refine", false}};
        if (trace) {
            params["trace_visit"] = true;
            params["debug_stage_trace_tag"] = tag;
        }
        proto::plan::PlanNode node;
        auto* anns = node.mutable_vector_anns();
        anns->set_vector_type(proto::plan::VectorType::FloatVector);
        anns->set_field_id(vec.get());
        anns->set_placeholder_tag("$0");
        auto* info = anns->mutable_query_info();
        info->set_topk(topk);
        info->set_metric_type("COSINE");
        info->set_search_params(params.dump());
        info->set_round_decimal(-1);
        return query::CreateSearchPlanFromPlanNode(schema, node);
    };
    auto plain_plan = make_plan(false);
    auto trace_plan = make_plan(true);
    const auto placeholder = segcore::CreatePlaceholderGroupFromBlob(
        1, dim, vectors.data() + query_row * dim);
    const auto blob = placeholder.SerializeAsString();
    const auto plain_group = query::ParsePlaceholderGroup(plain_plan.get(), blob);
    const auto trace_group = query::ParsePlaceholderGroup(trace_plan.get(), blob);
    auto plain = segment->Search(plain_plan.get(), plain_group.get(), 1000000);
    auto traced = segment->Search(trace_plan.get(), trace_group.get(), 1000000);
    ASSERT_EQ(traced->seg_offsets_, plain->seg_offsets_);
    ASSERT_EQ(traced->distances_, plain->distances_);
    ASSERT_EQ(traced->distances_.size(), topk);
    ASSERT_EQ(std::memcmp(traced->distances_.data(), plain->distances_.data(),
                          traced->distances_.size() * sizeof(float)), 0);
    ASSERT_EQ(traced->seg_offsets_.size(), topk);
    EXPECT_EQ(traced->seg_offsets_.front(), query_row);
    EXPECT_TRUE(plain->cardinal_stage_trace_json_.empty());
    ASSERT_FALSE(traced->cardinal_stage_trace_json_.empty());
    const auto envelope = nlohmann::json::parse(
        traced->cardinal_stage_trace_json_);
    const auto& trace = envelope.at("cardinal_stage_trace");
    ASSERT_TRUE(trace.at("capture_complete").get<bool>());
    ASSERT_EQ(trace.at("request_tag"), tag);
    const auto& capture = trace.at("queries").at(0);
    ASSERT_GT(capture.at("refine_compute_count").get<int64_t>(), 0);
    ASSERT_GT(capture.at("coarse").size(), topk);
    ASSERT_EQ(capture.at("local").size(), topk);
    segment->FillPrimaryKeys(trace_plan.get(), *traced);
    for (size_t i = 0; i < topk; ++i) {
        EXPECT_EQ(capture.at("local").at(i).at("offset"),
                  traced->seg_offsets_[i]);
        EXPECT_FLOAT_EQ(capture.at("local").at(i).at("score").get<float>(),
                        traced->distances_[i]);
        EXPECT_EQ(std::get<int64_t>(traced->primary_keys_[i]),
                  1000003 + 17 * traced->seg_offsets_[i]);
    }
    if (const auto* dir = std::getenv("MILVUS_STAGE_TRACE_TIERED_RESULT_DIR")) {
        std::filesystem::create_directories(dir);
        std::vector<int64_t> pks;
        for (const auto& value : traced->primary_keys_) {
            pks.push_back(std::get<int64_t>(value));
        }
        const auto path = std::filesystem::path(dir) /
                          ("tiered-saq" + std::to_string(coarse_bits) + ".json");
        const nlohmann::json artifact{
            {"index_type", "CARDINAL_TIERED"},
            {"index_engine_version", 11},
            {"build_params", build},
            {"segment_id", kSegmentID},
            {"field_id", vec.get()},
            {"query_vector_fp32", std::vector<float>(
                 vectors.data() + query_row * dim,
                 vectors.data() + (query_row + 1) * dim)},
            {"normal_offsets", traced->seg_offsets_},
            {"normal_pks", pks},
            {"normal_scores", traced->distances_},
            {"capture", trace}};
        std::ofstream output(path);
        ASSERT_TRUE(output.is_open());
        output << artifact.dump(2) << std::endl;
        ASSERT_TRUE(output.good());
    }
}

}  // namespace

TEST(CardinalStageTracePk, TieredSaq1NestedNormalSegmentSearch) {
    CheckTieredSaqSegmentTrace(1);
}

TEST(CardinalStageTracePk, TieredSaq2NestedNormalSegmentSearch) {
    CheckTieredSaqSegmentTrace(2);
}
#endif
