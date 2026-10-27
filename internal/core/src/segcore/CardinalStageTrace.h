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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace milvus::segcore {

using StageTracePkResolver =
    std::function<std::vector<int64_t>(const std::vector<int64_t>&)>;

inline std::string
StageTraceFnv1a64(const std::string& bytes) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char c : bytes) {
        hash ^= c;
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

// Hash a language-independent representation rather than a JSON float format.
inline std::string
StageTraceCandidateChecksum(const nlohmann::json& candidates) {
    std::string bytes;
    for (const auto& row : candidates) {
        const float score = row.at("score").get<float>();
        uint32_t bits;
        std::memcpy(&bits, &score, sizeof(bits));
        bytes += std::to_string(row.at("position").get<size_t>()) + ":" +
                 std::to_string(row.at("offset").get<int64_t>()) + ":" +
                 std::to_string(row.at("pk").get<int64_t>()) + ":" +
                 std::to_string(bits) + ";";
    }
    return StageTraceFnv1a64(bytes);
}

inline nlohmann::json
IncompleteCardinalStageTrace(int64_t segment_id,
                             const std::string& request_tag,
                             const std::string& reason) {
    return {{"event", "CARDINAL_EXACT_STAGE_TRACE"},
            {"schema_version", 1},
            {"record_type", "manifest"},
            {"segment_id", segment_id},
            {"request_tag", request_tag},
            {"capture_complete", false},
            {"reason", reason}};
}

// Build all records before emitting any of them. A failed PK lookup therefore
// cannot masquerade as a successfully captured empty/partial candidate set.
inline std::vector<nlohmann::json>
BuildCardinalStageTraceRecords(const std::string& metadata,
                               int64_t segment_id,
                               int64_t field_id,
                               const StageTracePkResolver& resolve_pks) {
    using Json = nlohmann::json;
    constexpr size_t kChunkSize = 128;
    constexpr size_t kMaxQueries = 16;
    constexpr size_t kMaxCandidates = 8192;
    const auto root = Json::parse(metadata);
    const auto& trace = root.at("cardinal_stage_trace");
    const std::string tag = trace.value("request_tag", std::string{});
    if (!trace.value("tag_valid", false) || tag.empty()) {
        throw std::invalid_argument("invalid diagnostic correlation tag");
    }
    if (trace.value("schema_version", 0) != 1 ||
        !trace.value("capture_complete", false) || !trace.contains("queries") ||
        !trace.at("queries").is_array() || trace.at("queries").empty() ||
        trace.at("queries").size() > kMaxQueries) {
        throw std::invalid_argument("unsupported diagnostic envelope");
    }

    if (trace.at("expected_queries").get<size_t>() !=
        trace.at("queries").size()) {
        throw std::invalid_argument("incomplete diagnostic query batch");
    }
    std::vector<Json> output;
    std::set<size_t> query_indices;
    for (const auto& query : trace.at("queries")) {
        const auto query_index = query.at("query_index").get<size_t>();
        if (query_index >= trace.at("queries").size() ||
            !query_indices.insert(query_index).second) {
            throw std::invalid_argument("invalid diagnostic query index");
        }
        if (!query.value("capture_complete", false)) {
            throw std::invalid_argument("incomplete native trace");
        }
        const auto& coarse = query.at("coarse");
        const auto& local = query.at("local");
        if (!coarse.is_array() || !local.is_array() ||
            coarse.size() > kMaxCandidates || local.size() > kMaxCandidates) {
            throw std::invalid_argument("invalid candidate arrays");
        }
        const auto local_topk = query.at("local_topk").get<size_t>();
        if (local.size() > local_topk || local.size() > coarse.size()) {
            throw std::invalid_argument("invalid local result size");
        }
        std::vector<int64_t> offsets;
        std::map<int64_t, size_t> positions;
        for (const auto* rows : {&coarse, &local}) {
            for (const auto& row : *rows) {
                const auto offset = row.at("offset").get<int64_t>();
                const auto score = row.at("score").get<float>();
                if (offset < 0 || !std::isfinite(score)) {
                    throw std::invalid_argument("invalid captured candidate");
                }
                if (positions.emplace(offset, offsets.size()).second) {
                    offsets.push_back(offset);
                }
            }
        }
        std::map<int64_t, bool> coarse_members;
        for (const auto& row : coarse) {
            if (!coarse_members.emplace(row.at("offset").get<int64_t>(), true)
                     .second) {
                throw std::invalid_argument("duplicate coarse candidate");
            }
        }
        std::map<int64_t, bool> local_members;
        for (const auto& row : local) {
            const auto id = row.at("offset").get<int64_t>();
            if (coarse_members.find(id) == coarse_members.end() ||
                !local_members.emplace(id, true).second) {
                throw std::invalid_argument(
                    "local output is not a coarse subset");
            }
        }
        const auto pks = resolve_pks(offsets);
        if (pks.size() != offsets.size()) {
            throw std::invalid_argument("PK mapping count mismatch");
        }
        Json base{{"event", "CARDINAL_EXACT_STAGE_TRACE"},
                  {"schema_version", 1},
                  {"segment_id", segment_id},
                  {"field_id", field_id},
                  {"request_tag", tag},
                  {"query_hash", query.at("query_hash")},
                  {"query_index", query.at("query_index")},
                  {"local_topk", local_topk},
                  {"candidate_limit", query.at("candidate_limit")},
                  {"level", query.at("level")},
                  {"metric", query.at("metric")}};
        for (const auto* key : {"score_domain",
                                "search_candidates_limit",
                                "skip_refine",
                                "refine_ratio",
                                "ivf_search_ratio",
                                "should_refine",
                                "coarse_compute_count",
                                "refine_compute_count",
                                "ivf_search_count",
                                "graph_search_count",
                                "bf_search_count",
                                "fallback_search_count",
                                "phase_us"}) {
            if (query.contains(key)) {
                base[key] = query.at(key);
            }
        }
        base["expected_queries"] = trace.at("expected_queries");
        size_t chunks_emitted = 0;
        std::string chunk_hashes;
        for (const auto& stage :
             {std::string("coarse"), std::string("local")}) {
            const auto& rows = query.at(stage);
            const size_t count = rows.size();
            const size_t chunks = (count + kChunkSize - 1) / kChunkSize;
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                Json candidates = Json::array();
                const auto begin = chunk * kChunkSize;
                const auto end = std::min(count, begin + kChunkSize);
                for (size_t i = begin; i < end; ++i) {
                    const auto offset = rows[i].at("offset").get<int64_t>();
                    candidates.push_back({{"position", i},
                                          {"offset", offset},
                                          {"pk", pks.at(positions.at(offset))},
                                          {"score", rows[i].at("score")}});
                }
                auto record = base;
                const auto hash = StageTraceCandidateChecksum(candidates);
                record["record_type"] = "chunk";
                record["stage"] = stage == "coarse" ? "C" : "L";
                record["candidate_count"] = count;
                record["chunk_index"] = chunk;
                record["chunk_count"] = chunks;
                record["hash_algorithm"] = "fnv1a64-v1";
                record["chunk_hash_format"] =
                    "position:offset:pk:score_binary32_uint;";
                record["chunk_hash"] = hash;
                record["candidates"] = std::move(candidates);
                output.push_back(std::move(record));
                chunk_hashes += hash;
                ++chunks_emitted;
            }
        }
        auto manifest = base;
        manifest["record_type"] = "manifest";
        manifest["capture_complete"] = true;
        manifest["coarse_count"] = coarse.size();
        manifest["local_count"] = local.size();
        manifest["expected_chunk_records"] = chunks_emitted;
        manifest["hash_algorithm"] = "fnv1a64-v1";
        manifest["ordered_chunk_hashes_hash"] = StageTraceFnv1a64(chunk_hashes);
        output.push_back(std::move(manifest));
    }
    return output;
}

}  // namespace milvus::segcore
