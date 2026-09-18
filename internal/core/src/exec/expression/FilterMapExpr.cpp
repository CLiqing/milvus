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
        const auto candidate_count = universe - input->count();
        if (candidate_count == 0) {
            return result;
        }
        OffsetVector offsets(candidate_count);
        FilterMapCursor cursor;
        checkCancellation(query);
        const auto count = input->ReadUnsetBatch(
            cursor, std::span<int32_t>(offsets.data(), offsets.size()));
        AssertInfo(count == candidate_count,
                   "FilterMap candidate count does not match its enumeration");
        // Use a separate context: borrowed offset/mask state cannot leak to
        // another predicate or survive an exception.
        EvalCtx selected(context.get_exec_context(), &offsets);
        VectorPtr evaluated;
        Eval(selected, evaluated);
        auto values = GetColumnVector(evaluated);
        AssertInfo(values->size() == count,
                   "Offset predicate returned a different row count");
        TargetBitmapView truth(values->GetRawData(), count);
        TargetBitmapView valid(values->GetValidRawData(), count);
        // Compact the owned input staging after Eval; do not allocate a second
        // V-sized accepted-ID buffer. Physical chunk reads remain kernel-owned.
        size_t written = 0;
        for (size_t i = 0; i < count; ++i) {
            if (truth[i] && valid[i]) {
                offsets[written++] = offsets[i];
            }
        }
        result.AppendUniqueBits(
            std::span<const int32_t>(offsets.data(), written), false);
        return result;
    }

    // A legacy predicate without offset support is evaluated once normally.
    // Its result is still assembled batch-by-batch; no Dense-to-Sparse scan.
    const TargetBitmap* incoming = input ? &input->EnsureDense() : nullptr;
    EvalCtx selected(context.get_exec_context());
    const auto prepare_active = [&](size_t offset, size_t count) {
        if (incoming) {
            TargetBitmap active(incoming->view(offset, count));
            active.flip();  // FilterMap 1=excluded; evaluator 1=active.
            selected.set_bitmap_input(std::move(active));
        }
    };
    if (CanExecuteAllAtOnce()) {
        // Unmigrated all-at-once index producers already return an N-bit
        // bitmap. Preserve that Dense result; never scan it to rebuild Sparse.
        SetExecuteAllAtOnce();
        VectorPtr evaluated;
        prepare_active(0, universe);
        Eval(selected, evaluated);
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
    const auto configured_batch =
        context.get_query_config()->get_expr_batch_size();
    AssertInfo(configured_batch > 0, "Invalid predicate batch size");
    for (size_t offset = 0; offset < universe;) {
        checkCancellation(query);
        VectorPtr evaluated;
        prepare_active(offset,
                       std::min<size_t>(configured_batch, universe - offset));
        Eval(selected, evaluated);
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
