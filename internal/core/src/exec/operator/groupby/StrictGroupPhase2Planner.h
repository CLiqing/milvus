// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace milvus::exec {

constexpr size_t kStrictGroupPhase2MaxSelectivityPercent = 10;
constexpr size_t kStrictGroupSmallGroupSelectivityPercent = 1;

// Each entry is a list of indexes into the unfinished locked-group array.
// Batches are disjoint and retain the phase-1 lock order.
struct StrictGroupPhase2Plan {
    std::vector<std::vector<size_t>> batches;
};

inline bool
IsBelowSelectivity(size_t rows, size_t eligible_rows, size_t percentage) {
    return static_cast<unsigned __int128>(rows) * 100 <
           static_cast<unsigned __int128>(eligible_rows) * percentage;
}

inline std::optional<StrictGroupPhase2Plan>
BuildStrictGroupPhase2Plan(size_t eligible_rows,
                           const std::vector<size_t>& group_row_counts) {
    if (eligible_rows == 0 || group_row_counts.empty()) {
        return std::nullopt;
    }

    size_t combined_rows = 0;
    for (auto count : group_row_counts) {
        combined_rows += count;
    }

    StrictGroupPhase2Plan plan;
    if (IsBelowSelectivity(combined_rows,
                           eligible_rows,
                           kStrictGroupPhase2MaxSelectivityPercent)) {
        std::vector<size_t> batch;
        batch.reserve(group_row_counts.size());
        for (size_t i = 0; i < group_row_counts.size(); ++i) {
            batch.emplace_back(i);
        }
        plan.batches.emplace_back(std::move(batch));
        return plan;
    }

    std::vector<size_t> current_small_batch;
    std::vector<size_t> large_batch;
    size_t current_small_rows = 0;
    for (size_t i = 0; i < group_row_counts.size(); ++i) {
        auto rows = group_row_counts[i];
        if (!IsBelowSelectivity(rows,
                                eligible_rows,
                                kStrictGroupSmallGroupSelectivityPercent)) {
            large_batch.emplace_back(i);
            continue;
        }

        if (!current_small_batch.empty() &&
            !IsBelowSelectivity(current_small_rows + rows,
                                eligible_rows,
                                kStrictGroupPhase2MaxSelectivityPercent)) {
            plan.batches.emplace_back(std::move(current_small_batch));
            current_small_batch.clear();
            current_small_rows = 0;
        }
        current_small_batch.emplace_back(i);
        current_small_rows += rows;
    }
    if (!current_small_batch.empty()) {
        plan.batches.emplace_back(std::move(current_small_batch));
    }
    if (!large_batch.empty()) {
        plan.batches.emplace_back(std::move(large_batch));
    }
    return plan;
}

}  // namespace milvus::exec
