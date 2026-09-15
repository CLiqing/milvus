// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include "exec/expression/Expr.h"

namespace milvus::exec {

void
Expr::Eval(EvalCtx& context, VectorPtr& result) {
    if (!context.filter_rows()) {
        if (type_ == DataType::BOOL && context.filter_input() &&
            context.filter_input()->none()) {
            const auto count = context.get_offset_input()
                                   ? context.get_offset_input()->size()
                                   : context.filter_input()->size();
            if (!context.get_offset_input()) {
                MoveCursor();
            }
            result = std::make_shared<ColumnVector>(TargetBitmap(count),
                                                    TargetBitmap(count, true));
            return;
        }
        EvalImpl(context, result);
        return;
    }

    const auto universe = *context.filter_rows();
    AssertInfo(type_ == DataType::BOOL,
               "Only predicates can produce a filter range");
    const auto cap =
        context.get_query_config()->filter_map_config().ExceptionCap(universe);
    AssertInfo(cap.has_value(), "Filter evaluation range requires a policy");
    AssertInfo(!context.get_offset_input(),
               "Filter range cannot be mixed with legacy offset input");
    auto input = context.filter_input();
    AssertInfo(!input || input->size() == universe,
               "Filter candidate universe mismatch");
    if (input) {
        input->flip();  // active evaluator input -> excluded-row enumeration
    }
    auto* query = context.get_exec_context()->get_query_context();
    checkCancellation(query);

    // Keep the existing ColumnVector truth/validity contract. UNKNOWN may be
    // folded only at this null-rejecting boundary, never inside a NOT/OR kernel.
    auto output = FilterMap::Adaptive(universe, false, *cap);
    auto publish = [&] {
        result = std::make_shared<ColumnVector>(std::move(output),
                                                TargetBitmap(universe, true));
    };

    // Representation- and kernel-independent short circuit, including an
    // empty Dense input or a successor without offset support.
    if (universe == 0 || (input && input->all())) {
        publish();
        return;
    }

    if (PropagatesFilterRange()) {
        EvalCtx selected(context.get_exec_context());
        selected.set_filter_rows(universe);
        if (input) {
            input->flip();
            selected.set_filter_input(std::move(*input));
        }
        // Same conjunction loop, ordering, and short-circuit as batch Eval.
        EvalImpl(selected, result);
        return;
    }

    if (input && input->capability() == FilterMapCapability::EnumerateOnly &&
        SupportOffsetInput()) {
        const auto candidates = universe - input->count();
        OffsetVector offsets(candidates);
        FilterMapCursor cursor;
        const auto count = input->ReadUnsetBatch(
            cursor, std::span<int32_t>(offsets.data(), offsets.size()));
        AssertInfo(count == candidates, "Candidate count/enumeration mismatch");
        EvalCtx selected(context.get_exec_context(), &offsets);
        VectorPtr evaluated;
        EvalImpl(selected, evaluated);
        auto values = GetColumnVector(evaluated);
        AssertInfo(values->size() == count, "Offset result size mismatch");
        TargetBitmapView truth(values->GetRawData(), count);
        TargetBitmapView valid(values->GetValidRawData(), count);
        size_t written = 0;
        for (size_t i = 0; i < count; ++i) {
            if (truth[i] && valid[i]) {
                offsets[written++] = offsets[i];
            }
        }
        output.AppendUniqueBits(
            std::span<const int32_t>(offsets.data(), written), true);
        publish();
        return;
    }

    // Legacy kernels continue producing their native batches. The input is
    // installed before EvalImpl, and the output sink also enforces it for
    // kernels which cannot use the active mask themselves.
    const TargetBitmap* incoming = input ? &input->EnsureDense() : nullptr;
    EvalCtx selected(context.get_exec_context());
    const bool all_at_once = CanExecuteAllAtOnce();
    if (all_at_once) {
        SetExecuteAllAtOnce();
    }
    const auto batch = context.get_query_config()->get_expr_batch_size();
    AssertInfo(batch > 0, "Invalid expression batch size");
    for (size_t offset = 0; offset < universe;) {
        checkCancellation(query);
        if (incoming) {
            const auto count = all_at_once
                                   ? universe
                                   : std::min<size_t>(batch, universe - offset);
            TargetBitmap active(incoming->view(offset, count));
            active.flip();
            selected.set_bitmap_input(std::move(active));
        }
        VectorPtr evaluated;
        EvalImpl(selected, evaluated);
        auto values = GetColumnVector(evaluated);
        const auto count = values->size();
        AssertInfo(count > 0 && count <= universe - offset,
                   "Invalid predicate result range");
        TargetBitmapView truth(values->GetRawData(), count);
        TargetBitmapView valid(values->GetValidRawData(), count);
        if (incoming) {
            auto excluded = incoming->view(offset, count);
            truth.inplace_sub(excluded, count);
        }
        if (all_at_once) {
            AssertInfo(count == universe, "All-at-once result size mismatch");
            // Preserve existing full bitmap index results. Do not rebuild
            // Sparse by scanning a completed segment-sized Dense result.
            truth.inplace_and(valid, count);
            output = values->GetFilterMap();
        } else {
            output.AssignBitmapBatch(truth, &valid, offset, false);
        }
        offset += count;
    }
    publish();
}

}  // namespace milvus::exec
