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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

#include "ArrayOffsets.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "knowhere/config.h"

namespace milvus {

struct SearchIteratorV2Info {
    std::string token = "";
    uint32_t batch_size = 0;
    std::optional<float> last_bound = std::nullopt;
};

// Brute-force index params sourced from the collection-level index metadata at
// plan creation. Used only by brute force when a segment predates a field added
// by add_function_field, so its per-segment metadata does not carry the field.
struct BruteForceIndexParams {
    std::optional<float> bm25_k1_;
    std::optional<float> bm25_b_;
    std::optional<int64_t> minhash_lsh_band_;
    std::optional<int64_t> minhash_element_bit_width_;
};

struct SearchInfo {
    int64_t topk_{0};
    int64_t group_size_{1};
    bool strict_group_size_{false};
    int64_t round_decimal_{0};
    FieldId field_id_;
    MetricType metric_type_;
    knowhere::Json search_params_;
    BruteForceIndexParams brute_force_index_params_;
    std::vector<FieldId>
        group_by_field_ids_;  // Group by field IDs (single or multi-field)
    tracer::TraceContext trace_ctx_;
    bool materialized_view_involved = false;
    bool iterative_filter_execution = false;
    std::optional<SearchIteratorV2Info> iterator_v2_info_ = std::nullopt;
    std::optional<std::string> json_path_;
    std::optional<milvus::DataType> json_type_;
    bool strict_cast_{false};
    std::shared_ptr<const IArrayOffsets> array_offsets_{
        nullptr};  // For element-level search
    bool global_refine_enable_{false};
    float search_topk_ratio_{0.0f};
    float refine_topk_ratio_{0.0f};
    // Number of rows visible to a growing-segment search, in logical row space.
    // Growing plans decide it ONCE from get_active_count(timestamp) and carry
    // it down so no kernel re-derives it. Sealed plans leave it unset: their
    // indexes are immutable and retain the empty-bitset fast path.
    //
    // Re-deriving it is not safe on a growing segment: a concurrent insert
    // publishes rows into the column storage (and into a nullable field's
    // offset mapping) before ack_responder_ advances, so a later read sees
    // rows this search must not touch. Reduce then validates offsets against
    // the acknowledged count and rejects them ("invalid offset ... rows num
    // ..."), or, if the ack catches up first, silently returns rows newer
    // than the query's MVCC timestamp.
    //
    // -1 means "not supplied" (sealed searches and direct callers). Growing
    // kernels fall back to computing it themselves for direct calls.
    int64_t active_count_{-1};

    bool
    element_level() const {
        return array_offsets_ != nullptr;
    }

    bool
    has_group_by() const {
        return !group_by_field_ids_.empty();
    }

    // The filter-result representation the search wants the scalar filter to
    // produce.  `sparse` means the filter emits a query-owned sparse result
    // instead of the Dense filtered bitmap.  Search dispatch is independent:
    // the downstream consumer decides whether to enumerate it (BF) or use a
    // membership adapter (Graph/IVF).  The legacy
    // `bf_filter_scan_mode == "valid_ids_per_query"` experiment knob implies
    // the same representation, kept for compatibility.
    bool
    RequestsAdaptiveFilterRepresentation() const {
        if (!search_params_.is_object()) {
            return false;
        }
        const auto representation = search_params_.value(
            "filter_result_representation", std::string{"dense"});
        if (representation == "sparse" || representation == "adaptive") {
            return true;
        }
        return false;
    }

    bool
    UseSparseFilterRepresentation() const {
        if (RequestsAdaptiveFilterRepresentation()) {
            return true;
        }
        if (!search_params_.is_object()) {
            return false;
        }
        return search_params_.value("bf_filter_scan_mode", std::string{"auto"}) ==
               "valid_ids_per_query";
    }

    int64_t
    SparseResultMaxCardinality(int64_t configured_default) const {
        AssertInfo(configured_default > 0,
                   "configured sparse_result_max_cardinality must be positive, got {}",
                   configured_default);
        AssertInfo(
            configured_default <= std::numeric_limits<int32_t>::max(),
            "configured sparse_result_max_cardinality {} exceeds the int32 "
            "row-ID limit {}",
            configured_default,
            std::numeric_limits<int32_t>::max());
        auto value = configured_default;
        if (search_params_.is_object() &&
            search_params_.contains("sparse_result_max_cardinality")) {
            const auto requested =
                search_params_.at("sparse_result_max_cardinality")
                    .get<int64_t>();
            AssertInfo(requested > 0,
                       "sparse_result_max_cardinality must be positive, got {}",
                       requested);
            AssertInfo(
                requested <= std::numeric_limits<int32_t>::max(),
                "sparse_result_max_cardinality {} exceeds the int32 row-ID "
                "limit {}",
                requested,
                std::numeric_limits<int32_t>::max());
            // A request may make the experiment more conservative, but must
            // never enlarge the instance-wide absolute safety bound.
            value = std::min(configured_default, requested);
        }
        return value;
    }
};

using SearchInfoPtr = std::shared_ptr<SearchInfo>;

}  // namespace milvus
