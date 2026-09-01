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

#include <atomic>
#include <iostream>
#include <utility>
#include <variant>
#include "common/Consts.h"
#include "storage/ThreadPool.h"

namespace milvus {

extern std::atomic<int64_t> FILE_SLICE_SIZE;
extern std::atomic<int64_t> EXEC_EVAL_EXPR_BATCH_SIZE;
extern std::atomic<int64_t> DELETE_DUMP_BATCH_SIZE;
extern std::atomic<bool> ENABLE_LATEST_DELETE_SNAPSHOT_OPTIMIZATION;
extern std::atomic<bool> OPTIMIZE_EXPR_ENABLED;
extern std::atomic<bool> ENABLE_DRIVER_PREFETCH;
extern std::atomic<bool> JSON_KEY_STATS_ENABLED;
extern std::atomic<bool> GROWING_JSON_KEY_STATS_ENABLED;
extern std::atomic<bool> CONFIG_PARAM_TYPE_CHECK_ENABLED;
extern std::atomic<bool> ENABLE_PARQUET_STATS_SKIP_INDEX;
extern std::atomic<bool> ENABLE_SPARSE_FILTER_RESULT;
extern std::atomic<int64_t> SPARSE_FILTER_RESULT_MAX_CARDINALITY;
extern std::atomic<int64_t> SPARSE_FILTER_RESULT_MIN_SEGMENT_ROWS;
extern std::atomic<double> SPARSE_FILTER_RESULT_MAX_RATIO;

void
SetIndexSliceSize(const int64_t size);

void
SetLoadTransientBudgetBytes(int64_t bytes);

void
SetDefaultExecEvalExprBatchSize(int64_t val);

void
SetDefaultDeleteDumpBatchSize(int64_t val);

void
SetDefaultOptimizeExprEnable(bool val);

void
SetDefaultDriverPrefetchEnable(bool val);

void
SetDefaultJSONKeyStatsEnable(bool val);

void
SetDefaultGrowingJSONKeyStatsEnable(bool val);

void
SetDefaultConfigParamTypeCheck(bool val);

void
SetDefaultEnableParquetStatsSkipIndex(bool val);

void
SetEnableLatestDeleteSnapshotOptimization(bool val);

void
SetSparseFilterResultConfig(bool enabled,
                            int64_t max_cardinality,
                            int64_t min_segment_rows,
                            double max_ratio);

// Returns the per-segment Sparse cap.  Zero means that the segment must keep
// the Dense representation.  The absolute cap may already contain a
// request-level override; the segment-size and ratio guards remain global
// safety bounds.
int64_t
ComputeSparseFilterResultCap(int64_t segment_rows,
                             int64_t max_cardinality,
                             int64_t min_segment_rows,
                             double max_ratio);

void
SetLogLevel(const char* level);

struct BufferView {
    struct Element {
        const char* data_;
        uint32_t* offsets_;
        int start_;
        int end_;
    };

    std::variant<std::vector<Element>, std::pair<char*, size_t>> data_;
};

}  // namespace milvus
