// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include <gtest/gtest.h>
#include <memory>
#include <regex>
#include <vector>
#include <chrono>

#include "query/PlanProto.h"

namespace {

std::shared_ptr<milvus::Schema>
CreateTestSchema() {
    auto schema = std::make_shared<milvus::Schema>();
    schema->AddDebugField(
        "fakevec", milvus::DataType::VECTOR_FLOAT, 16, knowhere::metric::L2);
    auto i64_fid = schema->AddDebugField("age", milvus::DataType::INT64);
    schema->set_primary_field_id(i64_fid);
    return schema;
}

milvus::proto::plan::PlanNode
CreateVectorPlanNode(const std::string& search_params) {
    milvus::proto::plan::PlanNode plan_node;
    auto vector_anns = plan_node.mutable_vector_anns();
    vector_anns->set_vector_type(milvus::proto::schema::DataType::FloatVector);
    vector_anns->set_placeholder_tag("$0");
    vector_anns->set_field_id(100);
    auto query_info = vector_anns->mutable_query_info();
    query_info->set_topk(10);
    query_info->set_round_decimal(3);
    query_info->set_metric_type("L2");
    query_info->set_search_params(search_params);
    return plan_node;
}

}  // namespace

TEST(PlanProto, NotSetUnsupported) {
    using namespace milvus;
    using namespace milvus::query;
    auto schema = CreateTestSchema();

    proto::plan::Expr expr_pb;
    ProtoParser parser(schema);
    ASSERT_ANY_THROW(parser.ParseExprs(expr_pb));
}

TEST(PlanProto, ConsumeS3ReadPathSearchParams) {
    using namespace milvus::query;
    auto schema = CreateTestSchema();
    ProtoParser parser(schema);
    auto plan_node_proto = CreateVectorPlanNode(R"({
        "nprobe": 10,
        "s3_read_path": "curl_multi",
        "s3_read_max_inflight": "100",
        "s3_read_eventloops": 8,
        "s3_read_crt_max_connections": 64,
        "s3_read_crt_throughput_gbps": "30"
    })");

    auto plan_node = parser.PlanNodeFromProto(plan_node_proto);
    const auto& search_info = plan_node->search_info_;

    EXPECT_EQ(search_info.search_params_["nprobe"], 10);
    EXPECT_FALSE(search_info.search_params_.contains("s3_read_path"));
    EXPECT_FALSE(search_info.search_params_.contains("s3_read_max_inflight"));
    EXPECT_FALSE(search_info.search_params_.contains("s3_read_eventloops"));
    EXPECT_TRUE(search_info.s3_read_path_config_.override_enabled);
    EXPECT_EQ(search_info.s3_read_path_config_.mode, "curl_multi");
    ASSERT_TRUE(search_info.s3_read_path_config_.max_inflight.has_value());
    EXPECT_EQ(*search_info.s3_read_path_config_.max_inflight, 100);
    ASSERT_TRUE(search_info.s3_read_path_config_.event_loops.has_value());
    EXPECT_EQ(*search_info.s3_read_path_config_.event_loops, 8);
    ASSERT_TRUE(
        search_info.s3_read_path_config_.crt_max_connections.has_value());
    EXPECT_EQ(*search_info.s3_read_path_config_.crt_max_connections, 64);
    ASSERT_TRUE(
        search_info.s3_read_path_config_.crt_throughput_gbps.has_value());
    EXPECT_DOUBLE_EQ(*search_info.s3_read_path_config_.crt_throughput_gbps, 30);
}

TEST(PlanProto, RejectInvalidS3ReadPathSearchParams) {
    using namespace milvus::query;
    auto schema = CreateTestSchema();
    ProtoParser parser(schema);
    auto plan_node_proto = CreateVectorPlanNode(R"({
        "nprobe": 10,
        "s3_read_path": "unknown"
    })");

    EXPECT_ANY_THROW(parser.PlanNodeFromProto(plan_node_proto));
}
