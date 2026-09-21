// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#include "exec/expression/PreparedBitmapExpr.h"

#include "exec/expression/EvalCtx.h"

namespace milvus::exec {

void
PhyPreparedBitmapExpr::MoveCursor() {
    if (!has_offset_input_) {
        cursor_ +=
            std::min<int64_t>(batch_size_, source_->truth().size() - cursor_);
    }
}

void
PhyPreparedBitmapExpr::Eval(EvalCtx& context, VectorPtr& result) {
    const auto* offsets = context.get_offset_input();
    SetHasOffsetInput(offsets != nullptr);
    TargetBitmap truth;
    TargetBitmap valid;
    if (offsets) {
        truth.resize(offsets->size());
        valid.resize(offsets->size());
        for (size_t lane = 0; lane < offsets->size(); ++lane) {
            const auto offset = offsets->at(lane);
            AssertInfo(offset >= 0 && static_cast<size_t>(offset) <
                                          source_->truth().size(),
                       "prepared bitmap offset outside segment");
            truth[lane] = source_->truth()[offset];
            valid[lane] = source_->valid()[offset];
        }
    } else {
        const auto count =
            std::min<int64_t>(batch_size_, source_->truth().size() - cursor_);
        truth.append(source_->truth(), cursor_, count);
        valid.append(source_->valid(), cursor_, count);
        MoveCursor();
    }
    // Conjunct/NOT may modify this result in-place: never return source_'s
    // shared storage, even for an all-at-once sequential evaluation.
    result = std::make_shared<ColumnVector>(std::move(truth), std::move(valid));
}

}  // namespace milvus::exec
