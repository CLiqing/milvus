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

#include "common/Common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string.h>

#include "common/Consts.h"
#include "common/EasyAssert.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "log/Log.h"
#include "storage/EntryStreamUtils.h"
#include "tantivy-binding.h"

namespace milvus {

std::atomic<int64_t> FILE_SLICE_SIZE(DEFAULT_INDEX_FILE_SLICE_SIZE);
std::atomic<int64_t> EXEC_EVAL_EXPR_BATCH_SIZE(
    DEFAULT_EXEC_EVAL_EXPR_BATCH_SIZE);
std::atomic<int64_t> DELETE_DUMP_BATCH_SIZE(DEFAULT_DELETE_DUMP_BATCH_SIZE);
std::atomic<bool> ENABLE_LATEST_DELETE_SNAPSHOT_OPTIMIZATION(
    DEFAULT_ENABLE_LATEST_DELETE_SNAPSHOT_OPTIMIZATION);
std::atomic<bool> OPTIMIZE_EXPR_ENABLED(DEFAULT_OPTIMIZE_EXPR_ENABLED);
std::atomic<bool> ENABLE_DRIVER_PREFETCH(DEFAULT_ENABLE_DRIVER_PREFETCH);

std::atomic<bool> JSON_KEY_STATS_ENABLED(DEFAULT_JSON_KEY_STATS_ENABLED);

std::atomic<bool> GROWING_JSON_KEY_STATS_ENABLED(
    DEFAULT_GROWING_JSON_KEY_STATS_ENABLED);
std::atomic<bool> CONFIG_PARAM_TYPE_CHECK_ENABLED(
    DEFAULT_CONFIG_PARAM_TYPE_CHECK_ENABLED);
std::atomic<bool> ENABLE_PARQUET_STATS_SKIP_INDEX(
    DEFAULT_ENABLE_PARQUET_STATS_SKIP_INDEX);
std::atomic<bool> ENABLE_SPARSE_FILTER_RESULT(false);
std::atomic<int64_t> SPARSE_FILTER_RESULT_MAX_CARDINALITY(6000);
std::atomic<int64_t> SPARSE_FILTER_RESULT_MIN_SEGMENT_ROWS(50000);
std::atomic<double> SPARSE_FILTER_RESULT_MAX_RATIO(0.006);

void
SetIndexSliceSize(const int64_t size) {
    FILE_SLICE_SIZE.store(size << 20);
    LOG_INFO("set config index slice size (byte): {}", FILE_SLICE_SIZE.load());
}

void
SetLoadTransientBudgetBytes(int64_t bytes) {
    if (bytes < 0) {
        LOG_WARN("ignore invalid load transient budget bytes: {}", bytes);
        return;
    }
    storage::TransientMemoryBudget::SetLoadTransientBudgetBytes(
        static_cast<size_t>(bytes));
    LOG_INFO("set load transient budget bytes: {}", bytes);
}

void
SetDefaultExecEvalExprBatchSize(int64_t val) {
    EXEC_EVAL_EXPR_BATCH_SIZE.store(val);
    LOG_INFO("set default expr eval batch size: {}",
             EXEC_EVAL_EXPR_BATCH_SIZE.load());
}

void
SetDefaultDeleteDumpBatchSize(int64_t val) {
    DELETE_DUMP_BATCH_SIZE.store(val);
    LOG_INFO("set default delete dump batch size: {}",
             DELETE_DUMP_BATCH_SIZE.load());
}

void
SetDefaultOptimizeExprEnable(bool val) {
    OPTIMIZE_EXPR_ENABLED.store(val);
    LOG_INFO("set default optimize expr enabled: {}",
             OPTIMIZE_EXPR_ENABLED.load());
}

void
SetDefaultDriverPrefetchEnable(bool val) {
    ENABLE_DRIVER_PREFETCH.store(val);
    LOG_INFO("set default driver prefetch enabled: {}",
             ENABLE_DRIVER_PREFETCH.load());
}

void
SetDefaultJSONKeyStatsEnable(bool val) {
    JSON_KEY_STATS_ENABLED.store(val);
    LOG_INFO("set default json key stats enabled: {}",
             JSON_KEY_STATS_ENABLED.load());
}

void
SetDefaultGrowingJSONKeyStatsEnable(bool val) {
    GROWING_JSON_KEY_STATS_ENABLED.store(val);
    LOG_INFO("set default growing json key index enable: {}",
             GROWING_JSON_KEY_STATS_ENABLED.load());
}

void
SetDefaultConfigParamTypeCheck(bool val) {
    CONFIG_PARAM_TYPE_CHECK_ENABLED.store(val);
    LOG_INFO("set default config param type check enabled: {}",
             CONFIG_PARAM_TYPE_CHECK_ENABLED.load());
}

void
SetDefaultEnableParquetStatsSkipIndex(bool val) {
    ENABLE_PARQUET_STATS_SKIP_INDEX.store(val);
    LOG_INFO("set default enable parquet stats: {}",
             ENABLE_PARQUET_STATS_SKIP_INDEX.load());
}

void
SetEnableLatestDeleteSnapshotOptimization(bool val) {
    ENABLE_LATEST_DELETE_SNAPSHOT_OPTIMIZATION.store(val);
    LOG_INFO("set default enable latest delete snapshot optimization: {}",
             ENABLE_LATEST_DELETE_SNAPSHOT_OPTIMIZATION.load());
}

void
SetSparseFilterResultConfig(bool enabled,
                            int64_t max_cardinality,
                            int64_t min_segment_rows,
                            double max_ratio) {
    AssertInfo(max_cardinality > 0,
               "sparse filter result max cardinality must be positive, got {}",
               max_cardinality);
    AssertInfo(
        max_cardinality <= std::numeric_limits<int32_t>::max(),
        "sparse filter result max cardinality {} exceeds the int32 row-ID "
        "limit {}",
        max_cardinality,
        std::numeric_limits<int32_t>::max());
    AssertInfo(min_segment_rows > 0,
               "sparse filter result minimum segment rows must be positive, "
               "got {}",
               min_segment_rows);
    AssertInfo(std::isfinite(max_ratio) && max_ratio > 0.0 &&
                   max_ratio <= 1.0,
               "sparse filter result maximum ratio must be in (0, 1], got {}",
               max_ratio);
    ENABLE_SPARSE_FILTER_RESULT.store(enabled);
    SPARSE_FILTER_RESULT_MAX_CARDINALITY.store(max_cardinality);
    SPARSE_FILTER_RESULT_MIN_SEGMENT_ROWS.store(min_segment_rows);
    SPARSE_FILTER_RESULT_MAX_RATIO.store(max_ratio);
    LOG_INFO("set sparse filter result config: enabled={}, "
             "max_cardinality={}, min_segment_rows={}, max_ratio={}",
             enabled,
             max_cardinality,
             min_segment_rows,
             max_ratio);
}

int64_t
ComputeSparseFilterResultCap(int64_t segment_rows,
                             int64_t max_cardinality,
                             int64_t min_segment_rows,
                             double max_ratio) {
    AssertInfo(segment_rows >= 0,
               "sparse filter segment rows must be non-negative, got {}",
               segment_rows);
    AssertInfo(max_cardinality > 0,
               "sparse filter result max cardinality must be positive, got {}",
               max_cardinality);
    AssertInfo(min_segment_rows > 0,
               "sparse filter result minimum segment rows must be positive, "
               "got {}",
               min_segment_rows);
    AssertInfo(std::isfinite(max_ratio) && max_ratio > 0.0 &&
                   max_ratio <= 1.0,
               "sparse filter result maximum ratio must be in (0, 1], got {}",
               max_ratio);
    if (segment_rows < min_segment_rows) {
        return 0;
    }

    const long double scaled = static_cast<long double>(segment_rows) *
                               static_cast<long double>(max_ratio);
    // Decimal configuration such as 0.006 can land infinitesimally below an
    // integer after binary conversion.  The relative epsilon only repairs
    // that representation error; it is far below one row.
    const long double epsilon =
        std::max(1.0L, std::abs(scaled)) * 1e-12L;
    const auto ratio_cap =
        static_cast<int64_t>(std::floor(scaled + epsilon));
    return std::min(max_cardinality, ratio_cap);
}

void
SetLogLevel(const char* level) {
    LOG_INFO("set log level: {}", level);
    if (strcmp(level, "debug") == 0) {
        gflags::SetCommandLineOption("minloglevel", "0");
        gflags::SetCommandLineOption("v", "5");
    } else if (strcmp(level, "trace") == 0) {
        gflags::SetCommandLineOption("minloglevel", "0");
        gflags::SetCommandLineOption("v", "6");
    } else {
        gflags::SetCommandLineOption("v", "4");
        if (strcmp(level, "info") == 0) {
            gflags::SetCommandLineOption("minloglevel", "0");
        } else if (strcmp(level, "warn") == 0) {
            gflags::SetCommandLineOption("minloglevel", "1");
        } else if (strcmp(level, "error") == 0) {
            gflags::SetCommandLineOption("minloglevel", "2");
        }
    }
    tantivy_set_log_level(level);
}

}  // namespace milvus
