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
#include <functional>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <ratio>
#include <string>
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
#include "exec/expression/CandidateEvaluator.h"
#include "exec/expression/Utils.h"
#include "exec/operator/Utils.h"
#include "exec/operator/DownpushSearchContext.h"
#include "index/ScalarIndex.h"
#include "index/StringIndex.h"
#include "log/Log.h"
#include "monitor/Monitor.h"
#include "opentelemetry/trace/span.h"
#include "plan/PlanNode.h"
#include "prometheus/histogram.h"
#include "query/PlanImpl.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

struct CardinalDownpushSearchContext {
    std::vector<milvus::cachinglayer::PinWrapper<Span<int64_t>>> int64_pins_;
    std::vector<milvus::cachinglayer::PinWrapper<Span<float>>> float_pins_;
    std::vector<milvus::cachinglayer::PinWrapper<RawStringChunkView>>
        string_pins_;
    std::vector<milvus::cachinglayer::PinWrapper<const index::IndexBase*>>
        scalar_index_pins_;
    std::shared_ptr<std::vector<int64_t>> int64_row_values_;
    std::shared_ptr<std::vector<float>> float_row_values_;
    std::vector<const char*> string_chunk_bases_;
    std::vector<const uint32_t*> string_chunk_value_offsets_;
    std::vector<const bool*> string_chunk_valid_data_;
    std::vector<size_t> string_chunk_row_counts_;
    size_t string_uniform_chunk_rows_{0};
    std::vector<const int64_t*> int64_chunk_values_;
    std::vector<const float*> float_chunk_values_;
    std::vector<int64_t> chunk_offsets_;
    const int32_t* row_dictionary_ids_{nullptr};
    size_t dictionary_row_count_{0};
    int32_t target_dictionary_id_{-1};
    bool target_dictionary_id_found_{false};
    std::optional<PreparedCandidateEvaluator> candidate_evaluator_;
    std::vector<std::shared_ptr<CardinalDownpushSearchContext>> child_contexts_;
    std::weak_ptr<CardinalDownpushSearchContext> self_;
};

namespace {

void*
AcquireCardinalDownpushLease(const void* opaque) noexcept {
    if (opaque == nullptr) {
        return nullptr;
    }
    try {
        const auto& weak =
            *static_cast<const std::weak_ptr<CardinalDownpushSearchContext>*>(
                opaque);
        auto owner = weak.lock();
        if (owner == nullptr) {
            return nullptr;
        }
        return new (std::nothrow)
            std::shared_ptr<CardinalDownpushSearchContext>(std::move(owner));
    } catch (...) {
        return nullptr;
    }
}

void
ReleaseCardinalDownpushLease(void* opaque) noexcept {
    delete static_cast<std::shared_ptr<CardinalDownpushSearchContext>*>(opaque);
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
        ctx->int64_row_values_ = BuildInt64RowValuesFromBulkSubscript(
            segment, op_context, predicate.field_id_, row_count);
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
        IsDownpushStringField(field_data_type) &&
        segment->HasFieldData(predicate.field_id_)) {
        auto num_chunks = segment->num_chunk_data(predicate.field_id_);
        if (num_chunks <= 0) {
            return nullptr;
        }
        ctx->string_pins_.reserve(num_chunks);
        ctx->string_chunk_bases_.reserve(num_chunks);
        ctx->string_chunk_value_offsets_.reserve(num_chunks);
        ctx->string_chunk_valid_data_.reserve(num_chunks);
        ctx->string_chunk_row_counts_.reserve(num_chunks);
        ctx->chunk_offsets_.reserve(num_chunks + 1);
        int64_t expected_row_offset = 0;
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            const auto chunk_row_offset =
                segment->num_rows_until_chunk(predicate.field_id_, chunk_id);
            if (chunk_row_offset != expected_row_offset) {
                return nullptr;
            }
            auto pin = segment->raw_string_chunk_view(
                op_context, predicate.field_id_, chunk_id);
            const auto view = pin.get();
            if (view.base == nullptr || view.offsets == nullptr ||
                view.row_count == 0 || expected_row_offset > row_count ||
                static_cast<int64_t>(view.row_count) >
                    row_count - expected_row_offset) {
                return nullptr;
            }
            ctx->chunk_offsets_.push_back(chunk_row_offset);
            ctx->string_chunk_bases_.push_back(view.base);
            ctx->string_chunk_value_offsets_.push_back(view.offsets);
            ctx->string_chunk_valid_data_.push_back(view.valid_data);
            ctx->string_chunk_row_counts_.push_back(view.row_count);
            ctx->string_pins_.push_back(std::move(pin));
            expected_row_offset += view.row_count;
        }
        if (expected_row_offset != row_count) {
            return nullptr;
        }
        ctx->chunk_offsets_.push_back(row_count);
        const auto uniform_rows = ctx->string_chunk_row_counts_.front();
        bool has_uniform_chunks = uniform_rows > 0;
        for (size_t i = 1; i + 1 < ctx->string_chunk_row_counts_.size(); ++i) {
            has_uniform_chunks =
                has_uniform_chunks &&
                ctx->string_chunk_row_counts_[i] == uniform_rows;
        }
        if (ctx->string_chunk_row_counts_.back() > uniform_rows) {
            has_uniform_chunks = false;
        }
        if (has_uniform_chunks) {
            ctx->string_uniform_chunk_rows_ = uniform_rows;
        }
        return ctx;
    }

    if (predicate.value_type_ == CardinalDownpushPredicateValueType::String &&
        IsDownpushStringField(field_data_type) &&
        (predicate.op_ == CardinalDownpushPredicateOp::Int64Equal ||
         predicate.op_ == CardinalDownpushPredicateOp::Int64NotEqual) &&
        segment->HasIndex(predicate.field_id_)) {
        auto pins = segment->PinIndex(op_context, predicate.field_id_);
        if (pins.size() != 1) {
            return nullptr;
        }
        const auto* string_index =
            dynamic_cast<const index::StringIndex*>(pins.front().get());
        if (string_index == nullptr) {
            return nullptr;
        }
        auto view =
            string_index->GetDictionaryIdColumnView(predicate.string_arg0_);
        if (!view.has_value() || view->row_value_ids == nullptr ||
            view->row_count != static_cast<size_t>(row_count)) {
            return nullptr;
        }
        ctx->row_dictionary_ids_ = view->row_value_ids;
        ctx->dictionary_row_count_ = view->row_count;
        ctx->target_dictionary_id_ = view->target_dictionary_id;
        ctx->target_dictionary_id_found_ = view->target_dictionary_id_found;
        ctx->scalar_index_pins_ = std::move(pins);
        LOG_INFO(
            "Cardinal downpush scalar source selected: "
            "source=stl_sort_dictionary_id, field_id={}, rows={}, "
            "target_found={}",
            predicate.field_id_.get(),
            ctx->dictionary_row_count_,
            ctx->target_dictionary_id_found_);
        return ctx;
    }

    return nullptr;
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
            return (!ctx.string_pins_.empty() &&
                    ctx.string_chunk_bases_.size() ==
                        ctx.string_pins_.size()) ||
                   (ctx.row_dictionary_ids_ != nullptr &&
                    ctx.dictionary_row_count_ > 0 &&
                    !ctx.scalar_index_pins_.empty());
    }
    return false;
}

}  // namespace

std::shared_ptr<CardinalDownpushSearchContext>
PrepareCardinalDownpushSearchContext(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    const CardinalDownpushPredicate& predicate) {
    auto context =
        BuildCardinalDownpushSearchContext(segment, op_context, predicate);
    if (context == nullptr ||
        !IsDownpushPredicateSourceReady(*context, predicate.value_type_)) {
        return nullptr;
    }
    if (predicate.value_type_ == CardinalDownpushPredicateValueType::Int64) {
        Int64CandidateSourceView source;
        source.row_values = context->int64_row_values_ == nullptr
                                ? nullptr
                                : context->int64_row_values_->data();
        source.row_count = static_cast<size_t>(segment->get_row_count());
        source.chunk_values = context->int64_chunk_values_.empty()
                                  ? nullptr
                                  : context->int64_chunk_values_.data();
        source.chunk_offsets = context->chunk_offsets_.empty()
                                   ? nullptr
                                   : context->chunk_offsets_.data();
        source.num_chunks = context->int64_chunk_values_.size();
        auto evaluator = PrepareInt64CandidateEvaluator(source, predicate);
        if (evaluator.has_value()) {
            context->candidate_evaluator_ = std::move(evaluator.value());
        }
    } else if (predicate.value_type_ ==
               CardinalDownpushPredicateValueType::Float) {
        FloatCandidateSourceView source;
        source.row_values = context->float_row_values_ == nullptr
                                ? nullptr
                                : context->float_row_values_->data();
        source.row_count = static_cast<size_t>(segment->get_row_count());
        source.chunk_values = context->float_chunk_values_.empty()
                                  ? nullptr
                                  : context->float_chunk_values_.data();
        source.chunk_offsets = context->chunk_offsets_.empty()
                                   ? nullptr
                                   : context->chunk_offsets_.data();
        source.num_chunks = context->float_chunk_values_.size();
        auto evaluator = PrepareFloatCandidateEvaluator(source, predicate);
        if (evaluator.has_value()) {
            context->candidate_evaluator_ = std::move(evaluator.value());
        }
    } else if (predicate.value_type_ ==
               CardinalDownpushPredicateValueType::String) {
        StringCandidateSourceView source;
        source.chunk_bases = context->string_chunk_bases_.empty()
                                 ? nullptr
                                 : context->string_chunk_bases_.data();
        source.chunk_value_offsets =
            context->string_chunk_value_offsets_.empty()
                ? nullptr
                : context->string_chunk_value_offsets_.data();
        source.chunk_valid_data =
            context->string_chunk_valid_data_.empty()
                ? nullptr
                : context->string_chunk_valid_data_.data();
        source.chunk_row_counts =
            context->string_chunk_row_counts_.empty()
                ? nullptr
                : context->string_chunk_row_counts_.data();
        source.chunk_row_offsets = context->chunk_offsets_.empty()
                                       ? nullptr
                                       : context->chunk_offsets_.data();
        source.num_chunks = context->string_pins_.size();
        source.row_count = static_cast<size_t>(segment->get_row_count());
        source.uniform_chunk_rows = context->string_uniform_chunk_rows_;
        source.row_dictionary_ids = context->row_dictionary_ids_;
        source.target_dictionary_id = context->target_dictionary_id_;
        source.target_dictionary_id_found =
            context->target_dictionary_id_found_;
        auto evaluator = PrepareStringCandidateEvaluator(source, predicate);
        if (evaluator.has_value()) {
            context->candidate_evaluator_ = std::move(evaluator.value());
        }
    }
    return context;
}

std::shared_ptr<CardinalDownpushSearchContext>
PrepareCardinalDownpushSearchContext(
    const segcore::SegmentInternalInterface* segment,
    OpContext* op_context,
    const CardinalDownpushPredicateProgram& program) {
    if (program.leaves.empty() || program.nodes.empty() ||
        program.root >= program.nodes.size()) {
        return nullptr;
    }
    auto context = std::make_shared<CardinalDownpushSearchContext>();
    context->self_ = context;
    std::vector<PreparedCandidateEvaluator> evaluators;
    evaluators.reserve(program.leaves.size());
    context->child_contexts_.reserve(program.leaves.size());
    for (const auto& predicate : program.leaves) {
        auto child = PrepareCardinalDownpushSearchContext(
            segment, op_context, predicate);
        if (child == nullptr || !child->candidate_evaluator_.has_value() ||
            !static_cast<bool>(*child->candidate_evaluator_)) {
            return nullptr;
        }
        evaluators.push_back(*child->candidate_evaluator_);
        context->child_contexts_.push_back(std::move(child));
    }
    auto composite = ComposeCandidateEvaluators(
        std::move(evaluators), program.nodes, program.root);
    if (!composite.has_value()) {
        return nullptr;
    }
    context->candidate_evaluator_ = std::move(*composite);
    return context;
}

const char*
CardinalDownpushSourceName(const CardinalDownpushSearchContext& context) {
    if (!context.child_contexts_.empty()) {
        return context.child_contexts_.size() == 1 ? "candidate_evaluator"
                                                   : "composite_evaluator";
    }
    if (context.row_dictionary_ids_ != nullptr) {
        return "stl_sort_dictionary_id";
    }
    if (!context.string_pins_.empty()) {
        return "raw_string_chunks";
    }
    if (!context.int64_chunk_values_.empty() ||
        !context.float_chunk_values_.empty()) {
        return "raw_numeric_chunks";
    }
    if (context.int64_row_values_ != nullptr ||
        context.float_row_values_ != nullptr) {
        return "bulk_subscript";
    }
    return "unknown";
}

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
    if (const auto& program = query_context_->get_cardinal_downpush_program();
        program.has_value()) {
        // element-level (array-of-vectors) search is rejected by the
        // FilterBitsNode gate before the predicate is ever set; this assert is
        // purely defensive against future regressions.
        AssertInfo(!ph.element_level_,
                   "downpush hint does not support element-level vector "
                   "search");
        downpush_ctx = query_context_->get_cardinal_downpush_search_context();
        if (downpush_ctx == nullptr) {
            ThrowInfo(UnexpectedError,
                      "failed to build Cardinal downpush search context");
        }
        const bool has_candidate_evaluator =
            downpush_ctx->candidate_evaluator_.has_value() &&
            static_cast<bool>(downpush_ctx->candidate_evaluator_.value());
        AssertInfo(has_candidate_evaluator,
                   "candidate evaluator was not prepared before downpush "
                   "commit");
        const auto& prepared = downpush_ctx->candidate_evaluator_.value();
        knowhere::BitsetView::CandidateEvaluatorV1 evaluator;
        evaluator.abi_major = prepared.view.abi_major;
        evaluator.struct_size = sizeof(evaluator);
        evaluator.abi_capabilities = prepared.view.abi_capabilities;
        evaluator.abi_capabilities |=
            knowhere::BitsetView::kCandidateEvaluatorCapabilityLease;
        evaluator.context = prepared.view.context;
        evaluator.eval_batch = prepared.view.eval_batch;
        evaluator.eval_contiguous = prepared.view.eval_contiguous;
        evaluator.lease_factory_context = &downpush_ctx->self_;
        evaluator.acquire_lease = &AcquireCardinalDownpushLease;
        evaluator.release_lease = &ReleaseCardinalDownpushLease;
        search_view.set_candidate_evaluator(
            evaluator,
            static_cast<size_t>(segment_->get_row_count()),
            static_cast<size_t>(program->estimated_filtered_out_count));
        const auto& first_predicate = program->leaves.front();
        const char* value_type_name =
            program->leaves.size() > 1 ? "composite"
            : first_predicate.value_type_ ==
                    CardinalDownpushPredicateValueType::Int64
                ? "int64"
            : first_predicate.value_type_ ==
                    CardinalDownpushPredicateValueType::Float
                ? "float"
                : "string";
        milvus::monitor::internal_core_downpush_execution_count_family
            .Add({{"source", CardinalDownpushSourceName(*downpush_ctx)},
                  {"value_type", value_type_name},
                  {"iterator",
                   search_info_.iterator_v2_info_.has_value() ? "true"
                                                              : "false"}})
            .Increment();
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
