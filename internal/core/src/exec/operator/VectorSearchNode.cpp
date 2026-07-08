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
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <ratio>
#include <string>
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

constexpr size_t kDownpushRowValuesCacheMaxEntries = 16;

struct CardinalDownpushSearchContext {
    std::vector<milvus::cachinglayer::PinWrapper<Span<int64_t>>> int64_pins_;
    std::vector<milvus::cachinglayer::PinWrapper<Span<float>>> float_pins_;
    std::shared_ptr<std::vector<int64_t>> int64_row_values_;
    std::shared_ptr<std::vector<float>> float_row_values_;
    std::shared_ptr<std::vector<std::string>> string_row_values_;
    std::vector<const char*> string_row_value_ptrs_;
    std::vector<uint32_t> string_row_value_sizes_;
    std::vector<const char*> string_term_value_ptrs_;
    std::vector<uint32_t> string_term_value_sizes_;
    std::vector<const int64_t*> int64_chunk_values_;
    std::vector<const float*> float_chunk_values_;
    std::vector<int64_t> chunk_offsets_;
};

struct DownpushRowValuesCacheKey {
    int64_t segment_id;
    int64_t field_id;
    int64_t row_count;

    bool
    operator==(const DownpushRowValuesCacheKey& other) const {
        return segment_id == other.segment_id && field_id == other.field_id &&
               row_count == other.row_count;
    }
};

struct DownpushRowValuesCacheKeyHash {
    size_t
    operator()(const DownpushRowValuesCacheKey& key) const {
        size_t seed = std::hash<int64_t>{}(key.segment_id);
        seed ^= std::hash<int64_t>{}(key.field_id) + 0x9e3779b97f4a7c15ULL +
                (seed << 6) + (seed >> 2);
        seed ^= std::hash<int64_t>{}(key.row_count) + 0x9e3779b97f4a7c15ULL +
                (seed << 6) + (seed >> 2);
        return seed;
    }
};

std::mutex g_downpush_row_values_cache_mutex;
std::deque<DownpushRowValuesCacheKey> g_downpush_row_values_cache_order;
std::unordered_map<DownpushRowValuesCacheKey,
                   std::shared_ptr<std::vector<int64_t>>,
                   DownpushRowValuesCacheKeyHash>
    g_downpush_row_values_cache;

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
        case CardinalDownpushPredicateOp::ScalarRange:
            return KnowhereOp::kRange;
        case CardinalDownpushPredicateOp::ScalarAddLessThan:
            return KnowhereOp::kAddLessThan;
        case CardinalDownpushPredicateOp::ScalarTerm:
            return KnowhereOp::kTerm;
        case CardinalDownpushPredicateOp::ScalarSubLessThan:
            return KnowhereOp::kSubLessThan;
        case CardinalDownpushPredicateOp::ScalarMulLessThan:
            return KnowhereOp::kMulLessThan;
        case CardinalDownpushPredicateOp::ScalarDivLessThan:
            return KnowhereOp::kDivLessThan;
    }
    return std::nullopt;
}

bool
IsDownpushIntField(DataType data_type) {
    return data_type == DataType::INT8 || data_type == DataType::INT16 ||
           data_type == DataType::INT32 || data_type == DataType::INT64 ||
           data_type == DataType::TIMESTAMPTZ;
}

bool
IsDownpushFloatField(DataType data_type) {
    return data_type == DataType::FLOAT;
}

bool
IsDownpushStringField(DataType data_type) {
    return data_type == DataType::VARCHAR || data_type == DataType::STRING;
}

DataType
GetFieldDataType(const segcore::SegmentInternalInterface* segment,
                 FieldId field_id) {
    if (segment == nullptr) {
        return DataType::NONE;
    }
    return segment->get_schema()[field_id].get_data_type();
}

std::shared_ptr<std::vector<int64_t>>
BuildInt64RowValuesFromBulkSubscript(
    const segcore::SegmentInternalInterface* segment,
    milvus::OpContext* op_context,
    FieldId field_id,
    int64_t row_count) {
    std::vector<int64_t> offsets(row_count);
    std::iota(offsets.begin(), offsets.end(), 0);

    auto field_data = segment->bulk_subscript(
        op_context, field_id, offsets.data(), row_count);
    if (field_data == nullptr || !field_data->has_scalars()) {
        return nullptr;
    }

    auto row_values = std::make_shared<std::vector<int64_t>>();
    row_values->reserve(row_count);
    const auto& scalars = field_data->scalars();
    if (scalars.has_long_data() &&
        scalars.long_data().data_size() == row_count) {
        const auto& data = scalars.long_data().data();
        row_values->assign(data.begin(), data.end());
        return row_values;
    }
    if (scalars.has_int_data() && scalars.int_data().data_size() == row_count) {
        const auto& data = scalars.int_data().data();
        for (const auto value : data) {
            row_values->push_back(static_cast<int64_t>(value));
        }
        return row_values;
    }
    if (scalars.has_timestamptz_data() &&
        scalars.timestamptz_data().data_size() == row_count) {
        const auto& data = scalars.timestamptz_data().data();
        row_values->assign(data.begin(), data.end());
        return row_values;
    }
    return nullptr;
}

std::shared_ptr<std::vector<int64_t>>
GetCachedInt64RowValuesFromBulkSubscript(
    const segcore::SegmentInternalInterface* segment,
    milvus::OpContext* op_context,
    FieldId field_id) {
    auto row_count = segment->get_row_count();
    DownpushRowValuesCacheKey key{
        segment->get_segment_id(), field_id.get(), row_count};

    {
        std::lock_guard<std::mutex> lock(g_downpush_row_values_cache_mutex);
        auto it = g_downpush_row_values_cache.find(key);
        if (it != g_downpush_row_values_cache.end()) {
            return it->second;
        }
    }

    auto row_values = BuildInt64RowValuesFromBulkSubscript(
        segment, op_context, field_id, row_count);
    if (row_values == nullptr) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_downpush_row_values_cache_mutex);
    auto it = g_downpush_row_values_cache.find(key);
    if (it != g_downpush_row_values_cache.end()) {
        return it->second;
    }
    g_downpush_row_values_cache.emplace(key, row_values);
    g_downpush_row_values_cache_order.push_back(key);
    while (g_downpush_row_values_cache_order.size() >
           kDownpushRowValuesCacheMaxEntries) {
        g_downpush_row_values_cache.erase(
            g_downpush_row_values_cache_order.front());
        g_downpush_row_values_cache_order.pop_front();
    }
    return row_values;
}

std::shared_ptr<std::vector<float>>
BuildFloatRowValuesFromBulkSubscript(
    const segcore::SegmentInternalInterface* segment,
    milvus::OpContext* op_context,
    FieldId field_id,
    int64_t row_count) {
    std::vector<int64_t> offsets(row_count);
    std::iota(offsets.begin(), offsets.end(), 0);

    auto field_data = segment->bulk_subscript(
        op_context, field_id, offsets.data(), row_count);
    if (field_data == nullptr || !field_data->has_scalars() ||
        !field_data->scalars().has_float_data() ||
        field_data->scalars().float_data().data_size() != row_count) {
        return nullptr;
    }

    const auto& data = field_data->scalars().float_data().data();
    return std::make_shared<std::vector<float>>(data.begin(), data.end());
}

std::shared_ptr<std::vector<std::string>>
BuildStringRowValuesFromBulkSubscript(
    const segcore::SegmentInternalInterface* segment,
    milvus::OpContext* op_context,
    FieldId field_id,
    int64_t row_count) {
    std::vector<int64_t> offsets(row_count);
    std::iota(offsets.begin(), offsets.end(), 0);

    auto field_data = segment->bulk_subscript(
        op_context, field_id, offsets.data(), row_count);
    if (field_data == nullptr || !field_data->has_scalars() ||
        !field_data->scalars().has_string_data() ||
        field_data->scalars().string_data().data_size() != row_count) {
        return nullptr;
    }

    const auto& data = field_data->scalars().string_data().data();
    return std::make_shared<std::vector<std::string>>(data.begin(), data.end());
}

std::shared_ptr<CardinalDownpushSearchContext>
BuildCardinalDownpushSearchContext(
    const segcore::SegmentInternalInterface* segment,
    milvus::OpContext* op_context,
    const CardinalDownpushPredicate& predicate) {
    if (segment == nullptr || segment->type() != SegmentType::Sealed) {
        return nullptr;
    }
    auto field_data_type = GetFieldDataType(segment, predicate.field_id_);
    if (field_data_type != predicate.field_data_type_) {
        return nullptr;
    }

    auto ctx = std::make_shared<CardinalDownpushSearchContext>();
    if (predicate.value_type_ == CardinalDownpushPredicateValueType::Int64 &&
        segment->HasFieldData(predicate.field_id_) &&
        (field_data_type == DataType::INT64 ||
         field_data_type == DataType::TIMESTAMPTZ)) {
        auto num_chunks = segment->num_chunk_data(predicate.field_id_);
        if (num_chunks <= 0) {
            return nullptr;
        }
        ctx->int64_pins_.reserve(num_chunks);
        ctx->int64_chunk_values_.reserve(num_chunks);
        ctx->chunk_offsets_.reserve(num_chunks + 1);
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            ctx->chunk_offsets_.push_back(
                segment->num_rows_until_chunk(predicate.field_id_, chunk_id));
            auto pin = segment->chunk_data<int64_t>(
                op_context, predicate.field_id_, chunk_id);
            auto chunk = pin.get();
            ctx->int64_chunk_values_.push_back(chunk.data());
            ctx->int64_pins_.push_back(std::move(pin));
        }
        ctx->chunk_offsets_.push_back(segment->get_row_count());
        return ctx;
    }

    if (predicate.value_type_ == CardinalDownpushPredicateValueType::Float &&
        segment->HasFieldData(predicate.field_id_) &&
        field_data_type == DataType::FLOAT) {
        auto num_chunks = segment->num_chunk_data(predicate.field_id_);
        if (num_chunks <= 0) {
            return nullptr;
        }
        ctx->float_pins_.reserve(num_chunks);
        ctx->float_chunk_values_.reserve(num_chunks);
        ctx->chunk_offsets_.reserve(num_chunks + 1);
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            ctx->chunk_offsets_.push_back(
                segment->num_rows_until_chunk(predicate.field_id_, chunk_id));
            auto pin = segment->chunk_data<float>(
                op_context, predicate.field_id_, chunk_id);
            auto chunk = pin.get();
            ctx->float_chunk_values_.push_back(chunk.data());
            ctx->float_pins_.push_back(std::move(pin));
        }
        ctx->chunk_offsets_.push_back(segment->get_row_count());
        return ctx;
    }

    auto row_count = segment->get_row_count();
    if (predicate.value_type_ == CardinalDownpushPredicateValueType::Int64 &&
        IsDownpushIntField(field_data_type)) {
        ctx->int64_row_values_ = GetCachedInt64RowValuesFromBulkSubscript(
            segment, op_context, predicate.field_id_);
        if (ctx->int64_row_values_ == nullptr) {
            return nullptr;
        }
        return ctx;
    }

    if (predicate.value_type_ == CardinalDownpushPredicateValueType::Float &&
        IsDownpushFloatField(field_data_type)) {
        ctx->float_row_values_ = BuildFloatRowValuesFromBulkSubscript(
            segment, op_context, predicate.field_id_, row_count);
        if (ctx->float_row_values_ == nullptr) {
            return nullptr;
        }
        return ctx;
    }

    if (predicate.value_type_ == CardinalDownpushPredicateValueType::String &&
        IsDownpushStringField(field_data_type)) {
        ctx->string_row_values_ = BuildStringRowValuesFromBulkSubscript(
            segment, op_context, predicate.field_id_, row_count);
        if (ctx->string_row_values_ == nullptr) {
            return nullptr;
        }
        ctx->string_row_value_ptrs_.reserve(ctx->string_row_values_->size());
        ctx->string_row_value_sizes_.reserve(ctx->string_row_values_->size());
        for (const auto& value : *ctx->string_row_values_) {
            ctx->string_row_value_ptrs_.push_back(value.data());
            ctx->string_row_value_sizes_.push_back(
                static_cast<uint32_t>(value.size()));
        }
        return ctx;
    }

    return nullptr;
}

std::optional<knowhere::BitsetView::ExtraScalarPredicateValueType>
ToKnowherePredicateValueType(CardinalDownpushPredicateValueType value_type) {
    using KnowhereValueType =
        knowhere::BitsetView::ExtraScalarPredicateValueType;
    switch (value_type) {
        case CardinalDownpushPredicateValueType::Int64:
            return KnowhereValueType::kInt64;
        case CardinalDownpushPredicateValueType::Float:
            return KnowhereValueType::kFloat;
        case CardinalDownpushPredicateValueType::String:
            return KnowhereValueType::kString;
    }
    return std::nullopt;
}

void
FillKnowhereDownpushValueSource(
    knowhere::BitsetView::ExtraScalarInt64PredicateFilter& filter,
    const CardinalDownpushSearchContext& ctx) {
    filter.row_values = ctx.int64_row_values_ == nullptr
                            ? nullptr
                            : ctx.int64_row_values_->data();
    filter.chunk_values = ctx.int64_chunk_values_.data();
    filter.row_float_values = ctx.float_row_values_ == nullptr
                                  ? nullptr
                                  : ctx.float_row_values_->data();
    filter.chunk_float_values = ctx.float_chunk_values_.data();
    filter.row_string_values = ctx.string_row_value_ptrs_.data();
    filter.row_string_sizes = ctx.string_row_value_sizes_.data();
    filter.chunk_offsets = ctx.chunk_offsets_.data();
    if (!ctx.int64_chunk_values_.empty()) {
        filter.num_chunks = ctx.int64_chunk_values_.size();
    } else if (!ctx.float_chunk_values_.empty()) {
        filter.num_chunks = ctx.float_chunk_values_.size();
    }
}

void
FillKnowhereDownpushArgs(
    knowhere::BitsetView::ExtraScalarInt64PredicateFilter& filter,
    const CardinalDownpushPredicate& predicate,
    CardinalDownpushSearchContext& ctx) {
    filter.arg0 = predicate.arg0_;
    filter.arg1 = predicate.arg1_;
    filter.double_arg0 = predicate.double_arg0_;
    filter.double_arg1 = predicate.double_arg1_;
    filter.string_arg0_data = predicate.string_arg0_.data();
    filter.string_arg0_size =
        static_cast<uint32_t>(predicate.string_arg0_.size());
    filter.string_arg1_data = predicate.string_arg1_.data();
    filter.string_arg1_size =
        static_cast<uint32_t>(predicate.string_arg1_.size());
    filter.int64_terms = predicate.int64_terms_.data();
    filter.int64_term_count = predicate.int64_terms_.size();
    filter.double_terms = predicate.double_terms_.data();
    filter.double_term_count = predicate.double_terms_.size();
    if (!predicate.string_terms_.empty()) {
        ctx.string_term_value_ptrs_.reserve(predicate.string_terms_.size());
        ctx.string_term_value_sizes_.reserve(predicate.string_terms_.size());
        for (const auto& term : predicate.string_terms_) {
            ctx.string_term_value_ptrs_.push_back(term.data());
            ctx.string_term_value_sizes_.push_back(
                static_cast<uint32_t>(term.size()));
        }
        filter.string_term_values = ctx.string_term_value_ptrs_.data();
        filter.string_term_sizes = ctx.string_term_value_sizes_.data();
        filter.string_term_count = ctx.string_term_value_ptrs_.size();
    }
    filter.lower_inclusive = predicate.lower_inclusive_;
    filter.upper_inclusive = predicate.upper_inclusive_;
}

bool
IsDownpushPredicateSourceReady(const CardinalDownpushSearchContext& ctx,
                               CardinalDownpushPredicateValueType value_type) {
    switch (value_type) {
        case CardinalDownpushPredicateValueType::Int64:
            return ctx.int64_row_values_ != nullptr ||
                   !ctx.int64_chunk_values_.empty();
        case CardinalDownpushPredicateValueType::Float:
            return ctx.float_row_values_ != nullptr ||
                   !ctx.float_chunk_values_.empty();
        case CardinalDownpushPredicateValueType::String:
            return ctx.string_row_values_ != nullptr &&
                   !ctx.string_row_value_ptrs_.empty();
    }
    return false;
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
        auto value_type = ToKnowherePredicateValueType(predicate->value_type_);
        downpush_ctx = BuildCardinalDownpushSearchContext(
            segment_, op_context, predicate.value());
        if (!op.has_value() || !value_type.has_value() ||
            downpush_ctx == nullptr ||
            !IsDownpushPredicateSourceReady(*downpush_ctx,
                                            predicate->value_type_)) {
            ThrowInfo(UnexpectedError,
                      "failed to build Cardinal downpush search context");
        }
        knowhere::BitsetView::ExtraScalarInt64PredicateFilter filter;
        filter.value_type = value_type.value();
        FillKnowhereDownpushValueSource(filter, *downpush_ctx);
        filter.row_count = static_cast<size_t>(segment_->get_row_count());
        filter.op = op.value();
        FillKnowhereDownpushArgs(filter, predicate.value(), *downpush_ctx);
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
