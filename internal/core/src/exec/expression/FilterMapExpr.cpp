// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include "exec/expression/Expr.h"

namespace milvus::exec {

FilterMap
Expr::EvalFilterMap(EvalCtx& context,
                    size_t universe,
                    size_t cap,
                    std::optional<FilterMap> input) {
    AssertInfo(context.get_offset_input() == nullptr &&
                   context.get_bitmap_input().empty(),
               "FilterMap evaluation requires a segment-level context");
    AssertInfo(!input || input->size() == universe,
               "FilterMap predicate input universe mismatch");
    auto result = FilterMap::Adaptive(universe, true, cap);
    auto* query = context.get_exec_context()->get_query_context();
    if (input && input->capability() == FilterMapCapability::EnumerateOnly &&
        SupportOffsetInput()) {
        const auto configured_batch =
            context.get_query_config()->get_expr_batch_size();
        AssertInfo(configured_batch > 0, "Invalid predicate batch size");
        const auto batch_size =
            std::min<size_t>(configured_batch, universe - input->count());
        if (batch_size == 0) {
            return result;
        }
        OffsetVector offsets(batch_size);
        std::vector<int32_t> accepted;
        accepted.reserve(batch_size);
        FilterMapCursor cursor;
        // Use a separate context: borrowed offset/mask state cannot leak to
        // another predicate or survive an exception.
        EvalCtx selected(context.get_exec_context(), &offsets);
        while (true) {
            checkCancellation(query);
            offsets.resize(batch_size);
            const auto count = input->ReadUnsetBatch(
                cursor, std::span<int32_t>(offsets.data(), offsets.size()));
            if (count == 0) {
                break;
            }
            offsets.resize(count);
            VectorPtr evaluated;
            Eval(selected, evaluated);
            auto values = GetColumnVector(evaluated);
            AssertInfo(values->size() == count,
                       "Offset predicate returned a different row count");
            TargetBitmapView truth(values->GetRawData(), count);
            TargetBitmapView valid(values->GetValidRawData(), count);
            accepted.clear();
            for (size_t i = 0; i < count; ++i) {
                if (truth[i] && valid[i]) {
                    accepted.push_back(offsets[i]);
                }
            }
            result.AppendUniqueBits(accepted, false);
        }
        return result;
    }

    // A legacy predicate without offset support is evaluated once normally.
    // Its result is still assembled batch-by-batch; no Dense-to-Sparse scan.
    const TargetBitmap* incoming = input ? &input->EnsureDense() : nullptr;
    if (CanExecuteAllAtOnce()) {
        // Unmigrated all-at-once index producers already return an N-bit
        // bitmap. Preserve that Dense result; never scan it to rebuild Sparse.
        SetExecuteAllAtOnce();
        VectorPtr evaluated;
        Eval(context, evaluated);
        auto values = GetColumnVector(evaluated);
        AssertInfo(values->size() == universe,
                   "All-at-once predicate universe mismatch");
        TargetBitmap invalid(
            TargetBitmapView(values->GetValidRawData(), universe));
        invalid.flip();
        auto dense_result = values->GetFilterMap();
        values.reset();
        evaluated.reset();
        dense_result.flip();
        dense_result.InplaceOr(TargetBitmapView(invalid));
        if (incoming) {
            // InplaceOr accepts a view and does not mutate its mask.
            TargetBitmapView mask(
                const_cast<TargetBitmap::data_type*>(incoming->data()),
                universe);
            dense_result.InplaceOr(mask);
        }
        return dense_result;
    }
    for (size_t offset = 0; offset < universe;) {
        checkCancellation(query);
        VectorPtr evaluated;
        Eval(context, evaluated);
        auto values = GetColumnVector(evaluated);
        const auto count = values->size();
        AssertInfo(count > 0 && count <= universe - offset,
                   "Invalid full-segment predicate batch size");
        TargetBitmapView truth(values->GetRawData(), count);
        TargetBitmapView valid(values->GetValidRawData(), count);
        if (incoming != nullptr) {
            auto excluded = incoming->view(offset, count);
            truth.inplace_sub(excluded, count);
        }
        result.AssignBitmapBatch(truth, &valid, offset, true);
        offset += count;
    }
    return result;
}

}  // namespace milvus::exec
