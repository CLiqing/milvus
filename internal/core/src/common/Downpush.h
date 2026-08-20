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

#include <cstdint>
#include <string>
#include <vector>

#include "common/Types.h"
#include "knowhere/comp/index_param.h"

namespace milvus {

enum class CardinalDownpushPredicateOp {
    Int64GreaterEqual = 0,
    Int64ModLessThan = 1,
    Int64GreaterThan = 2,
    Int64LessEqual = 3,
    Int64LessThan = 4,
    Int64Equal = 5,
    Int64NotEqual = 6,
    ScalarRange = 7,
    ScalarAddLessThan = 8,
    ScalarTerm = 9,
    ScalarSubLessThan = 10,
    ScalarMulLessThan = 11,
    ScalarDivLessThan = 12,
    StringPrefixMatch = 13,
    StringPostfixMatch = 14,
    StringInnerMatch = 15,
    StringLikeMatch = 16,
};

enum class CardinalDownpushPredicateValueType {
    Int64 = 0,
    Float = 1,
    String = 2,
};

struct CardinalDownpushPredicate {
    FieldId field_id_;
    DataType field_data_type_{DataType::NONE};
    CardinalDownpushPredicateValueType value_type_{
        CardinalDownpushPredicateValueType::Int64};
    CardinalDownpushPredicateOp op_{
        CardinalDownpushPredicateOp::Int64GreaterEqual};
    int64_t arg0_{0};
    int64_t arg1_{0};
    double double_arg0_{0.0};
    double double_arg1_{0.0};
    std::string string_arg0_;
    std::string string_arg1_;
    std::vector<int64_t> int64_terms_;
    std::vector<double> double_terms_;
    std::vector<std::string> string_terms_;
    std::vector<uint32_t> like_token_offsets_;
    std::vector<uint32_t> like_token_sizes_;
    std::vector<uint8_t> like_token_types_;
    bool lower_inclusive_{true};
    bool upper_inclusive_{true};
    int64_t estimated_filtered_out_count_{0};
};

// Whether a vector index type supports scalar-predicate fusion (downpush, a.k.a.
// "ann filter fusing"). Backend-agnostic: today the graph family (Cardinal +
// HNSW) supports it; DISKANN and flat/IVF remain unsupported in v1.
inline bool
IsDownpushSupportedIndexType(const knowhere::IndexType& index_type) {
    return index_type == knowhere::IndexEnum::INDEX_CARDINAL_TIERED ||
           index_type == knowhere::IndexEnum::INDEX_HNSW ||
           index_type == knowhere::IndexEnum::INDEX_HNSW_SQ ||
           index_type == knowhere::IndexEnum::INDEX_HNSW_PQ ||
           index_type == knowhere::IndexEnum::INDEX_HNSW_PRQ;
}

}  // namespace milvus
