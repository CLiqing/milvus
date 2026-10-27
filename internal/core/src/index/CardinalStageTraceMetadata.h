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

#pragma once

#include <stdexcept>
#include <string>

#include "knowhere/dataset.h"
#include "nlohmann/json.hpp"

namespace milvus::index {

struct CardinalStageTraceRequest {
    bool enabled = false;
    bool tag_valid = false;
    std::string tag;
};

// Remove the diagnostic-only correlation tag before Knowhere validates params.
// It is never used to choose a search path or a candidate budget.
inline CardinalStageTraceRequest
TakeCardinalStageTraceRequest(nlohmann::json& search_config) noexcept {
    CardinalStageTraceRequest request;
    try {
        const auto visit = search_config.find("trace_visit");
        request.enabled = visit != search_config.end() && visit->is_boolean() &&
                          visit->get<bool>();
        auto tag = search_config.find("debug_stage_trace_tag");
        if (tag != search_config.end()) {
            if (tag->is_string()) {
                const auto& tag_value = tag->get_ref<const std::string&>();
                request.tag_valid =
                    !tag_value.empty() && tag_value.size() <= 128;
                if (request.tag_valid) {
                    for (const unsigned char c : tag_value) {
                        request.tag_valid =
                            request.tag_valid &&
                            ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                             (c >= '0' && c <= '9') || c == ':' || c == '_' ||
                             c == '-' || c == '.');
                    }
                }
                if (request.tag_valid) {
                    request.tag = tag_value;
                }
            }
            search_config.erase(tag);
        }
        if (!request.tag_valid) {
            request.tag.clear();
        }
    } catch (...) {
        request.tag_valid = false;
        request.tag.clear();
        // Strip diagnostic-only fields even when copying their value failed.
        try {
            search_config.erase("debug_stage_trace_tag");
        } catch (...) {
        }
    }
    return request;
}

inline void
PreserveCardinalStageTrace(const knowhere::DataSetPtr& dataset,
                           const CardinalStageTraceRequest& request,
                           std::string& output) noexcept {
    output.clear();
    if (!request.enabled) {
        return;
    }
    try {
        if (!dataset) {
            throw std::invalid_argument("missing trace dataset");
        }
        const auto metadata = dataset->GetJsonInfo();
        if (metadata.empty() || metadata.size() > 32 * 1024 * 1024) {
            throw std::length_error("missing or oversized trace metadata");
        }
        auto root = nlohmann::json::parse(metadata);
        auto trace = root.find("cardinal_stage_trace");
        if (trace == root.end() || !trace->is_object()) {
            // Other backends preserve their usual output. A tagged Cardinal
            // diagnostic must make missing metadata visible to its collector.
            if (!request.tag.empty()) {
                throw std::invalid_argument("missing Cardinal trace");
            }
            return;
        }
        (*trace)["request_tag"] = request.tag;
        (*trace)["tag_valid"] = request.tag_valid;
        output = root.dump();
    } catch (...) {
        // Diagnostics never replace a successful ordinary search with a new
        // error category. The segment emits an explicit incomplete record.
        try {
            output =
                nlohmann::json{
                    {"cardinal_stage_trace",
                     {{"request_tag", request.tag},
                      {"tag_valid", request.tag_valid},
                      {"capture_complete", false},
                      {"reason", "invalid_cardinal_trace_metadata"}}}}
                    .dump();
        } catch (...) {
            // An allocation failure cannot turn a successful search into an
            // error. Missing records invalidate capture in the collector.
            output.clear();
        }
    }
}

}  // namespace milvus::index
