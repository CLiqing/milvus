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

#include "VectorSearchNode.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <ratio>
#include <utility>
#include <vector>

#include "bitset/bitset.h"
#include "common/ArrayOffsets.h"
#include "common/BitsetView.h"
#include "common/EasyAssert.h"
#include "common/QueryResult.h"
#include "common/Span.h"
#include "common/Tracer.h"
#include "common/Utils.h"
#include "exec/QueryContext.h"
#include "exec/expression/Utils.h"
#include "exec/operator/Utils.h"
#include "monitor/Monitor.h"
#include "opentelemetry/trace/span.h"
#include "plan/PlanNode.h"
#include "prometheus/histogram.h"
#include "query/PlanImpl.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

namespace {

struct CardinalExprDownpushSearchContext {
    std::vector<PinWrapper<Span<int64_t>>> pins_;
    std::vector<const int64_t*> chunk_data_;
    std::vector<int64_t> chunk_offsets_;
    int64_t modulus_{0};
    int64_t threshold_{0};

    bool
    FilteredOut(int64_t seg_offset) const {
        if (seg_offset < 0 || chunk_offsets_.empty() ||
            seg_offset >= chunk_offsets_.back()) {
            return true;
        }
        auto it = std::upper_bound(
            chunk_offsets_.begin(), chunk_offsets_.end(), seg_offset);
        auto chunk_idx = static_cast<size_t>(
            std::distance(chunk_offsets_.begin(), it) - 1);
        auto inner_offset = seg_offset - chunk_offsets_[chunk_idx];
        auto pk = chunk_data_[chunk_idx][inner_offset];
        return pk % modulus_ >= threshold_;
    }
};

bool
CardinalExprDownpushFilteredOut(void* ctx, int64_t seg_offset) {
    auto* downpush_ctx =
        static_cast<CardinalExprDownpushSearchContext*>(ctx);
    return downpush_ctx->FilteredOut(seg_offset);
}

std::shared_ptr<CardinalExprDownpushSearchContext>
BuildCardinalExprDownpushSearchContext(
    const segcore::SegmentInternalInterface* segment,
    milvus::OpContext* op_context,
    const CardinalExprDownpushInfo& info) {
    if (segment == nullptr || segment->type() != SegmentType::Sealed) {
        return nullptr;
    }
    auto& schema = segment->get_schema();
    auto& field_meta = schema[info.field_id_];
    if (field_meta.get_data_type() != DataType::INT64 ||
        !segment->HasFieldData(info.field_id_)) {
        return nullptr;
    }

    auto num_chunks = segment->num_chunk(info.field_id_);
    if (num_chunks <= 0) {
        return nullptr;
    }

    auto ctx = std::make_shared<CardinalExprDownpushSearchContext>();
    ctx->modulus_ = info.modulus_;
    ctx->threshold_ = info.threshold_;
    ctx->pins_.reserve(num_chunks);
    ctx->chunk_data_.reserve(num_chunks);
    ctx->chunk_offsets_.reserve(num_chunks + 1);
    ctx->chunk_offsets_.push_back(0);

    for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
        auto pin =
            segment->chunk_data<int64_t>(op_context, info.field_id_, chunk_id);
        const auto& span = pin.get();
        ctx->chunk_data_.push_back(span.data());
        ctx->chunk_offsets_.push_back(ctx->chunk_offsets_.back() +
                                      span.row_count());
        ctx->pins_.push_back(std::move(pin));
    }
    return ctx;
}

}  // namespace

static milvus::SearchResult
empty_search_result(int64_t num_queries) {
    milvus::SearchResult final_result;
    final_result.total_nq_ = num_queries;
    final_result.unity_topK_ = 0;  // no result
    final_result.total_data_cnt_ = 0;
    return final_result;
}

PhyVectorSearchNode::PhyVectorSearchNode(
    int32_t operator_id,
    DriverContext* driverctx,
    const std::shared_ptr<const plan::VectorSearchNode>& search_node)
    : Operator(driverctx,
               search_node->output_type(),
               operator_id,
               search_node->id(),
               "PhyVectorSearchNode") {
    ExecContext* exec_context = operator_context_->get_exec_context();
    query_context_ = exec_context->get_query_context();
    segment_ = query_context_->get_segment();
    query_timestamp_ = query_context_->get_query_timestamp();
    active_count_ = query_context_->get_active_count();
    placeholder_group_ = query_context_->get_placeholder_group();
    search_info_ = query_context_->get_search_info();
}

void
PhyVectorSearchNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

RowVectorPtr
PhyVectorSearchNode::GetOutput() {
    milvus::exec::checkCancellation(query_context_);

    if (is_finished_ || !no_more_input_) {
        return nullptr;
    }

    tracer::AutoSpan span(
        "PhyVectorSearchNode::Execute", tracer::GetRootSpan(), true);

    DeferLambda([&]() { is_finished_ = true; });
    if (input_ == nullptr) {
        return nullptr;
    }

    span.GetSpan()->SetAttribute("search_type", search_info_.metric_type_);
    span.GetSpan()->SetAttribute("topk", search_info_.topk_);

    std::chrono::high_resolution_clock::time_point vector_start =
        std::chrono::high_resolution_clock::now();

    auto& ph = placeholder_group_->at(0);
    auto src_data = ph.get_blob();
    auto src_offsets = ph.get_offsets();
    auto num_queries = ph.num_of_queries_;
    std::shared_ptr<const IArrayOffsets> array_offsets = nullptr;
    if (ph.element_level_) {
        array_offsets = segment_->GetArrayOffsets(search_info_.field_id_);
        AssertInfo(array_offsets != nullptr, "Array offsets not available");
        query_context_->set_array_offsets(array_offsets);
        search_info_.array_offsets_ = array_offsets;
    }

    // Prepare BitsetView for search.
    // Fast path: all_rows_visible + non-element-level → empty BitsetView
    //            (IDSelectorAll in Knowhere, skips per-vector bit test).
    // Normal path: build BitsetView from the bitmap produced upstream.
    milvus::BitsetView search_view;
    int64_t data_cnt = active_count_;

    if (query_context_->get_all_rows_visible() && !ph.element_level_) {
        // search_view stays default-constructed (empty)
    } else {
        // There are two types of execution: pre-filter and iterative filter
        // For **pre-filter**: FilterBitsNode -> MvccNode -> ElementFilterBitsNode -> VectorSearchNode -> ...
        // For **iterative filter**: MvccNode -> VectorSearchNode -> IterativeElementFilterNode -> IterativeFilterNode -> ...
        //
        // When element_level_ is true, we need to transform doc-level bitset
        // to element-level bitset.  In pre-filter path, ElementFilterBitsNode
        // already does this.  We only need to do it here for the iterative
        // path or when ElementFilterBitsNode is not present.
        if (ph.element_level_ && !query_context_->bitset_is_element_level()) {
            auto col_input = GetColumnVector(input_);
            TargetBitmapView view(col_input->GetRawData(), col_input->size());
            TargetBitmapView valid_view(col_input->GetValidRawData(),
                                        col_input->size());

            auto [element_bitset, valid_element_bitset] =
                array_offsets->RowBitsetToElementBitset(view, valid_view, 0);

            query_context_->set_active_element_count(element_bitset.size());
            if (element_bitset.empty()) {
                query_context_->set_search_result(
                    empty_search_result(num_queries));
                return input_;
            }

            std::vector<VectorPtr> col_res;
            col_res.push_back(std::make_shared<ColumnVector>(
                std::move(element_bitset), std::move(valid_element_bitset)));
            input_ = std::make_shared<RowVector>(col_res);
        }

        auto col_input = GetColumnVector(input_);
        TargetBitmapView view(col_input->GetRawData(), col_input->size());

        if (view.all()) {
            query_context_->set_search_result(empty_search_result(num_queries));
            return input_;
        }

        // TODO: uniform knowhere BitsetView and milvus BitsetView
        search_view = milvus::BitsetView((uint8_t*)col_input->GetRawData(),
                                         col_input->size());
        data_cnt = search_view.size();
    }

    auto op_context = query_context_->get_op_context();
    std::shared_ptr<CardinalExprDownpushSearchContext> downpush_ctx;
    const auto& downpush_info =
        query_context_->get_cardinal_expr_downpush_info();
    if (downpush_info.has_value() && !ph.element_level_) {
        downpush_ctx = BuildCardinalExprDownpushSearchContext(
            segment_, op_context, downpush_info.value());
        if (downpush_ctx != nullptr) {
            search_view.set_extra_filter(downpush_ctx.get(),
                                         CardinalExprDownpushFilteredOut,
                                         downpush_info->filtered_out_count_);
        }
    }

    // Single search + metrics path
    milvus::SearchResult search_result;
    segment_->vector_search(search_info_,
                            src_data,
                            src_offsets,
                            num_queries,
                            query_timestamp_,
                            search_view,
                            op_context,
                            search_result);

    search_result.total_data_cnt_ = data_cnt;
    search_result.element_level_ = ph.element_level_;

    span.GetSpan()->SetAttribute(
        "result_count", static_cast<int>(search_result.seg_offsets_.size()));
    query_context_->set_search_result(std::move(search_result));

    std::chrono::high_resolution_clock::time_point vector_end =
        std::chrono::high_resolution_clock::now();
    double vector_cost =
        std::chrono::duration<double, std::micro>(vector_end - vector_start)
            .count();
    milvus::monitor::internal_core_search_latency_vector.Observe(vector_cost /
                                                                 1000);
    // vector search stores result in query_context;
    // this node returns the bitset for downstream operators
    return input_;
}

bool
PhyVectorSearchNode::IsFinished() {
    return is_finished_;
}

}  // namespace exec
}  // namespace milvus
