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

#include <string>

#include "knowhere/comp/index_param.h"

namespace milvus {

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
