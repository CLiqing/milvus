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

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "common/Vector.h"
#include "exec/QueryContext.h"

namespace milvus {
namespace exec {

using OffsetVector = FixedVector<int32_t>;
class EvalCtx {
 public:
    EvalCtx(ExecContext* exec_ctx, OffsetVector* offset_input)
        : exec_ctx_(exec_ctx), offset_input_(offset_input) {
        assert(exec_ctx_ != nullptr);
    }

    explicit EvalCtx(ExecContext* exec_ctx) : exec_ctx_(exec_ctx) {
    }

    ExecContext*
    get_exec_context() {
        return exec_ctx_;
    }

    std::shared_ptr<QueryConfig>
    get_query_config() {
        return exec_ctx_->get_query_config();
    }

    inline OffsetVector*
    get_offset_input() {
        return offset_input_;
    }

    inline void
    set_offset_input(OffsetVector* offset_input) {
        offset_input_ = offset_input;
    }

    inline void
    set_bitmap_input(TargetBitmap&& bitmap_input) {
        if (bitmap_input.empty()) {
            filter_input_.reset();
            return;
        }
        filter_input_ = FilterMap::FromDense(
            std::make_shared<TargetBitmap>(std::move(bitmap_input)));
    }

    inline const TargetBitmap&
    get_bitmap_input() const {
        static const TargetBitmap empty;
        return filter_input_ ? filter_input_->EnsureDense() : empty;
    }

    void
    clear_bitmap_input() {
        filter_input_.reset();
    }

    // Set once by a null-rejecting filter consumer. Policy is read from this
    // query's immutable configuration, not passed through expression calls.
    void
    EnableFilterOutput() {
        const auto rows = exec_ctx_->get_query_context()->get_active_count();
        if (get_query_config()->filter_map_config().ExceptionCap(rows)) {
            filter_rows_ = rows;
        }
    }

    const std::optional<size_t>&
    filter_rows() const {
        return filter_rows_;
    }

    void
    set_filter_input(FilterMap input) {
        filter_input_ = std::move(input);
    }

    const std::optional<FilterMap>&
    filter_input() const {
        return filter_input_;
    }

    void
    clear_filter_input() {
        filter_input_.reset();
    }

    void
    set_filter_rows(std::optional<size_t> rows) {
        filter_rows_ = rows;
    }

 private:
    ExecContext* exec_ctx_ = nullptr;
    // we may accept offsets array as input and do expr filtering on these data
    OffsetVector* offset_input_ = nullptr;

    std::optional<size_t> filter_rows_;
    // One prefilter representation: logical one is active, matching the
    // existing evaluator mask. Absence is unrestricted, not an empty set.
    // Legacy bitmap reads materialize this same map at the compatibility edge.
    mutable std::optional<FilterMap> filter_input_;
};

}  // namespace exec
}  // namespace milvus
