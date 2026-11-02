// Licensed under the Apache License, Version 2.0.
#include "exec/expression/OffsetExpressionSampler.h"
#include <algorithm>
#include <limits>
#include <random>

namespace milvus::exec {
std::optional<double>
SampleOffsetFilterRatio(const expr::TypedExprPtr& expression,
                        ExecContext* context,
                        std::shared_ptr<const ChunkedColumnInterface> column) {
    auto* query = context->get_query_context();
    const auto active = query->get_active_count();
    const auto params = query->get_search_info().search_params_;
    const int requested = params.value("ann_fusing_sample_rows", 20);
    AssertInfo(requested == 10 || requested == 20,
               "ann_fusing_sample_rows must be 10 or 20");
    if (active <= 0 || active > std::numeric_limits<int32_t>::max()) {
        return std::nullopt;
    }
    OffsetExpressionOptions options;
    options.null_rejecting = true;
    PreparedOffsetExpressionEvaluator prepared(expression, options);
    auto workspace = prepared.CreateWorkspace(context);
    if (!workspace->SupportsOffsetInput()) {
        return std::nullopt;
    }
    if (!column || column->num_chunks() <= 0 || column->NumRows() < active) {
        return std::nullopt;
    }
    std::mt19937_64 random(
        static_cast<uint64_t>(query->get_segment()->get_segment_id()) ^
        query->get_query_timestamp());
    const auto last = column->GetChunkIDByOffset(active - 1).first;
    const auto chunk = std::uniform_int_distribution<int64_t>(0, last)(random);
    const auto first = column->GetNumRowsUntilChunk(chunk);
    const auto rows =
        std::min<int64_t>(column->chunk_row_nums(chunk), active - first);
    if (first < 0 || rows <= 0) {
        return std::nullopt;
    }
    const auto count = std::min<int64_t>(requested, rows);
    OffsetVector offsets;
    offsets.reserve(count);
    std::uniform_int_distribution<int64_t> choose(0, rows - 1);
    while (offsets.size() < count) {
        const auto offset = first + choose(random);
        if (std::find(offsets.begin(), offsets.end(), offset) ==
            offsets.end()) {
            offsets.push_back(offset);
        }
    }
    const auto result = workspace->EvalOffsets(offsets);
    TargetBitmapView truth(result->GetRawData(), result->size());
    TargetBitmapView known(result->GetValidRawData(), result->size());
    size_t accepted = 0;
    for (size_t i = 0; i < offsets.size(); ++i) {
        accepted += truth[i] && known[i];
    }
    return 1.0 - static_cast<double>(accepted) / offsets.size();
}
}  // namespace milvus::exec
