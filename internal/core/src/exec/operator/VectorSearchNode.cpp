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
#include <mutex>
#include <optional>
#include <ratio>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bitset/bitset.h"
#include "common/ArrayOffsets.h"
#include "common/BitsetView.h"
#include "common/EasyAssert.h"
#include "common/QueryResult.h"
#include "common/Tracer.h"
#include "common/Utils.h"
#include "exec/QueryContext.h"
#include "exec/expression/Utils.h"
#include "exec/operator/Utils.h"
#include "index/ScalarIndex.h"
#include "monitor/Monitor.h"
#include "opentelemetry/trace/span.h"
#include "plan/PlanNode.h"
#include "prometheus/histogram.h"
#include "query/PlanImpl.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

namespace {

struct CardinalDownpushSearchContext {
    std::vector<milvus::cachinglayer::PinWrapper<Span<int64_t>>> pins_;
    std::vector<milvus::cachinglayer::PinWrapper<const index::IndexBase*>>
        index_pins_;
    std::shared_ptr<std::vector<int64_t>> row_values_;
    std::vector<const int64_t*> chunk_values_;
    std::vector<int64_t> chunk_offsets_;
};

struct ScalarRowValuesCacheKey {
    int64_t segment_id;
    int64_t field_id;
    int64_t row_count;

    bool
    operator==(const ScalarRowValuesCacheKey& other) const {
        return segment_id == other.segment_id && field_id == other.field_id &&
               row_count == other.row_count;
    }
};

struct ScalarRowValuesCacheKeyHash {
    size_t
    operator()(const ScalarRowValuesCacheKey& key) const {
        size_t h = std::hash<int64_t>{}(key.segment_id);
        h ^= std::hash<int64_t>{}(key.field_id) + 0x9e3779b97f4a7c15ULL +
             (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.row_count) + 0x9e3779b97f4a7c15ULL +
             (h << 6) + (h >> 2);
        return h;
    }
};

std::mutex g_scalar_row_values_cache_mutex;
std::unordered_map<ScalarRowValuesCacheKey,
                   std::shared_ptr<std::vector<int64_t>>,
                   ScalarRowValuesCacheKeyHash>
    g_scalar_row_values_cache;

std::optional<knowhere::BitsetView::ExtraScalarInt64PredicateOp>
ToKnowherePredicateOp(CardinalDownpushPredicateOp op) {
    using KnowhereOp = knowhere::BitsetView::ExtraScalarInt64PredicateOp;
    switch (op) {
        case CardinalDownpushPredicateOp::Int64GreaterEqual:
            return KnowhereOp::kGreaterEqual;
        case CardinalDownpushPredicateOp::Int64ModLessThan:
            return KnowhereOp::kModLessThan;
        case CardinalDownpushPredicateOp::Int64GreaterThan:
            return KnowhereOp::kGreaterThan;
        case CardinalDownpushPredicateOp::Int64LessEqual:
            return KnowhereOp::kLessEqual;
        case CardinalDownpushPredicateOp::Int64LessThan:
            return KnowhereOp::kLessThan;
        case CardinalDownpushPredicateOp::Int64Equal:
            return KnowhereOp::kEqual;
        case CardinalDownpushPredicateOp::Int64NotEqual:
            return KnowhereOp::kNotEqual;
    }
    return std::nullopt;
}

std::shared_ptr<CardinalDownpushSearchContext>
BuildCardinalDownpushSearchContext(
    const segcore::SegmentInternalInterface* segment,
    milvus::OpContext* op_context,
    const CardinalDownpushPredicate& predicate) {
    if (segment == nullptr || segment->type() != SegmentType::Sealed) {
        return nullptr;
    }

    auto ctx = std::make_shared<CardinalDownpushSearchContext>();
    if (segment->HasFieldData(predicate.field_id_)) {
        auto num_chunks = segment->num_chunk_data(predicate.field_id_);
        if (num_chunks <= 0) {
            return nullptr;
        }
        ctx->pins_.reserve(num_chunks);
        ctx->chunk_values_.reserve(num_chunks);
        ctx->chunk_offsets_.reserve(num_chunks + 1);
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            ctx->chunk_offsets_.push_back(
                segment->num_rows_until_chunk(predicate.field_id_, chunk_id));
            auto pin = segment->chunk_data<int64_t>(
                op_context, predicate.field_id_, chunk_id);
            auto chunk = pin.get();
            ctx->chunk_values_.push_back(chunk.data());
            ctx->pins_.push_back(std::move(pin));
        }
        ctx->chunk_offsets_.push_back(segment->get_row_count());
        return ctx;
    }

    auto scalar_indexes = segment->PinIndex(op_context, predicate.field_id_);
    if (scalar_indexes.empty()) {
        return nullptr;
    }
    ctx->index_pins_.push_back(std::move(scalar_indexes[0]));

    auto scalar_index =
        dynamic_cast<const index::ScalarIndex<int64_t>*>(
            ctx->index_pins_[0].get());
    if (scalar_index == nullptr || !scalar_index->HasRawData()) {
        return nullptr;
    }

    auto row_count = segment->get_row_count();
    ScalarRowValuesCacheKey cache_key{
        segment->get_segment_id(), predicate.field_id_.get(), row_count};
    {
        std::lock_guard<std::mutex> lock(g_scalar_row_values_cache_mutex);
        if (auto iter = g_scalar_row_values_cache.find(cache_key);
            iter != g_scalar_row_values_cache.end()) {
            ctx->row_values_ = iter->second;
            return ctx;
        }
    }

    auto row_values = std::make_shared<std::vector<int64_t>>();
    row_values->resize(row_count);
    for (int64_t row = 0; row < row_count; ++row) {
        auto value = scalar_index->Reverse_Lookup(row);
        if (!value.has_value()) {
            return nullptr;
        }
        (*row_values)[row] = value.value();
    }
    {
        std::lock_guard<std::mutex> lock(g_scalar_row_values_cache_mutex);
        if (auto iter = g_scalar_row_values_cache.find(cache_key);
            iter != g_scalar_row_values_cache.end()) {
            ctx->row_values_ = iter->second;
        } else {
            ctx->row_values_ = row_values;
            g_scalar_row_values_cache.emplace(cache_key, std::move(row_values));
        }
    }
    return ctx;
}

}  // namespace

static milvus::SearchResult
empty_search_result(int64_t num_queries, bool element_level = false) {
    milvus::SearchResult final_result;
    final_result.total_nq_ = num_queries;
    final_result.unity_topK_ = 0;  // no result
    final_result.total_data_cnt_ = 0;
    final_result.element_level_ = element_level;
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

    if (!ph.element_level_ && query_context_->bitset_is_element_level()) {
        ThrowInfo(ExprInvalid,
                  "element-level filter bitset cannot be used for row-level "
                  "vector search; use MATCH_ANY/MATCH_* for row-level struct "
                  "array filtering");
    }

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
                    empty_search_result(num_queries, ph.element_level_));
                return input_;
            }

            std::vector<VectorPtr> col_res;
            col_res.push_back(std::make_shared<ColumnVector>(
                std::move(element_bitset), std::move(valid_element_bitset)));
            input_ = std::make_shared<RowVector>(col_res);
            query_context_->set_bitset_is_element_level(true);
        }

        auto col_input = GetColumnVector(input_);
        TargetBitmapView view(col_input->GetRawData(), col_input->size());

        if (view.all()) {
            auto search_result = empty_search_result(num_queries);
            search_result.total_data_cnt_ = data_cnt;
            search_result.element_level_ = ph.element_level_;
            query_context_->set_search_result(std::move(search_result));
            return input_;
        }

        // TODO: uniform knowhere BitsetView and milvus BitsetView
        search_view = milvus::BitsetView((uint8_t*)col_input->GetRawData(),
                                         col_input->size());
        data_cnt = search_view.size();
    }

    auto op_context = query_context_->get_op_context();
    std::shared_ptr<CardinalDownpushSearchContext> downpush_ctx;
    if (auto predicate = query_context_->get_cardinal_downpush_predicate();
        predicate.has_value()) {
        if (ph.element_level_) {
            ThrowInfo(
                UnexpectedError,
                "downpush hint does not support element-level vector search");
        }
        auto op = ToKnowherePredicateOp(predicate->op_);
        downpush_ctx = BuildCardinalDownpushSearchContext(
            segment_, op_context, predicate.value());
        if (!op.has_value() || downpush_ctx == nullptr) {
            ThrowInfo(UnexpectedError,
                      "failed to build Cardinal downpush search context");
        }
        knowhere::BitsetView::ExtraScalarInt64PredicateFilter filter;
        filter.chunk_values = downpush_ctx->chunk_values_.data();
        filter.chunk_offsets = downpush_ctx->chunk_offsets_.data();
        filter.num_chunks = downpush_ctx->chunk_values_.size();
        filter.row_values =
            downpush_ctx->row_values_ == nullptr
                ? nullptr
                : downpush_ctx->row_values_->data();
        filter.row_count = static_cast<size_t>(segment_->get_row_count());
        filter.op = op.value();
        filter.arg0 = predicate->arg0_;
        filter.arg1 = predicate->arg1_;
        search_view.set_extra_scalar_int64_predicate_filter(
            filter,
            static_cast<size_t>(predicate->estimated_filtered_out_count_));
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
