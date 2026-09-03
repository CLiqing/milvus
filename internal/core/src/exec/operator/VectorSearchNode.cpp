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
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <ratio>
#include <string>
#include <utility>
#include <variant>
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
#include "exec/expression/OffsetExpressionEvaluator.h"
#include "exec/expression/Utils.h"
#include "exec/operator/Utils.h"
#include "exec/operator/DownpushSearchContext.h"
#include "index/ScalarIndex.h"
#include "log/Log.h"
#include "monitor/Monitor.h"
#include "opentelemetry/trace/span.h"
#include "plan/PlanNode.h"
#include "prometheus/histogram.h"
#include "query/PlanImpl.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

struct PreparedFusingBundle {
    std::optional<PreparedCandidateEvaluator> candidate_evaluator_;
    bool uses_shared_offset_evaluator_{false};
    std::vector<std::shared_ptr<const void>> opaque_resource_owners_;
    std::weak_ptr<PreparedFusingBundle> self_;
};

namespace {

void*
AcquireAnnFilterFusingLease(const void* opaque) noexcept {
    if (opaque == nullptr) {
        return nullptr;
    }
    try {
        const auto& weak =
            *static_cast<const std::weak_ptr<PreparedFusingBundle>*>(opaque);
        auto owner = weak.lock();
        if (owner == nullptr) {
            return nullptr;
        }
        return new (std::nothrow)
            std::shared_ptr<PreparedFusingBundle>(std::move(owner));
    } catch (...) {
        return nullptr;
    }
}

void
ReleaseAnnFilterFusingLease(void* opaque) noexcept {
    delete static_cast<std::shared_ptr<PreparedFusingBundle>*>(opaque);
}

}  // namespace

std::shared_ptr<PreparedFusingBundle>
PrepareAnnFilterFusingBundle(const segcore::SegmentInternalInterface* segment,
                             OpContext* op_context,
                             const AnnFilterFusingProgram& program) {
    if (program.leaves.empty() || program.nodes.empty() ||
        program.root >= program.nodes.size()) {
        return nullptr;
    }
    auto context = std::make_shared<PreparedFusingBundle>();
    context->self_ = context;
    std::vector<PreparedCandidateEvaluator> evaluators;
    evaluators.reserve(program.leaves.size());
    std::vector<std::shared_ptr<const PreparedOffsetExpressionEvaluator>>
        offset_evaluators;
    offset_evaluators.reserve(program.leaves.size());
    bool all_leaves_use_shared_offset_evaluator = true;
    for (const auto& plan : program.leaves) {
        if (!static_cast<bool>(plan)) {
            return nullptr;
        }
        std::optional<PreparedCandidateLeaf> prepared;
        try {
            prepared =
                plan.prepare(segment, op_context, plan.typed_state.get());
        } catch (const std::exception& error) {
            LOG_WARN("typed fusing leaf preparation failed: {}", error.what());
            return nullptr;
        } catch (...) {
            LOG_WARN("typed fusing leaf preparation failed with unknown error");
            return nullptr;
        }
        if (!prepared.has_value() || !static_cast<bool>(prepared->evaluator)) {
            return nullptr;
        }
        all_leaves_use_shared_offset_evaluator &=
            prepared->offset_evaluator != nullptr;
        offset_evaluators.push_back(prepared->offset_evaluator);
        evaluators.push_back(std::move(prepared->evaluator));
        context->opaque_resource_owners_.insert(
            context->opaque_resource_owners_.end(),
            std::make_move_iterator(prepared->resource_owners.begin()),
            std::make_move_iterator(prepared->resource_owners.end()));
    }
    if (all_leaves_use_shared_offset_evaluator) {
        std::vector<OffsetExpressionNode> nodes;
        nodes.reserve(program.nodes.size());
        for (const auto& node : program.nodes) {
            OffsetExpressionNodeType type;
            switch (node.type) {
                case CandidatePredicateNodeType::Leaf:
                    type = OffsetExpressionNodeType::Leaf;
                    break;
                case CandidatePredicateNodeType::Not:
                    type = OffsetExpressionNodeType::Not;
                    break;
                case CandidatePredicateNodeType::And:
                    type = OffsetExpressionNodeType::And;
                    break;
                case CandidatePredicateNodeType::Or:
                    type = OffsetExpressionNodeType::Or;
                    break;
            }
            nodes.push_back(OffsetExpressionNode{type, node.left, node.right});
        }
        auto shared = ComposeOffsetExpressionEvaluators(
            std::move(offset_evaluators), std::move(nodes), program.root);
        auto adapted = AdaptOffsetExpressionEvaluator(std::move(shared));
        if (!adapted.has_value()) {
            return nullptr;
        }
        context->candidate_evaluator_ = std::move(*adapted);
        context->uses_shared_offset_evaluator_ = true;
    } else {
        auto composite = ComposeCandidateEvaluators(
            std::move(evaluators), program.nodes, program.root);
        if (!composite.has_value()) {
            return nullptr;
        }
        context->candidate_evaluator_ = std::move(*composite);
    }
    return context;
}

const char*
AnnFilterFusingSourceName(const PreparedFusingBundle& context) {
    if (context.uses_shared_offset_evaluator_) {
        return "shared_offset_evaluator";
    }
    if (!context.opaque_resource_owners_.empty()) {
        return "typed_candidate_evaluator";
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
    std::shared_ptr<PreparedFusingBundle> downpush_ctx;
    if (const auto& program = query_context_->get_ann_filter_fusing_program();
        program.has_value()) {
        // element-level (array-of-vectors) search is rejected by the
        // FilterBitsNode gate before the predicate is ever set; this assert is
        // purely defensive against future regressions.
        AssertInfo(!ph.element_level_,
                   "downpush hint does not support element-level vector "
                   "search");
        downpush_ctx = query_context_->get_ann_filter_fusing_bundle();
        if (downpush_ctx == nullptr) {
            ThrowInfo(UnexpectedError,
                      "failed to obtain prepared ANN filter fusing bundle");
        }
        const bool has_candidate_evaluator =
            downpush_ctx->candidate_evaluator_.has_value() &&
            static_cast<bool>(downpush_ctx->candidate_evaluator_.value());
        AssertInfo(has_candidate_evaluator,
                   "candidate evaluator was not prepared before downpush "
                   "commit");
        const auto& prepared = downpush_ctx->candidate_evaluator_.value();
        auto evaluator = prepared.view;
        evaluator.struct_size = sizeof(knowhere::CandidateEvaluatorV1);
        evaluator.abi_capabilities |=
            knowhere::kCandidateEvaluatorCapabilityLease;
        evaluator.lease_factory_context = &downpush_ctx->self_;
        evaluator.acquire_lease = &AcquireAnnFilterFusingLease;
        evaluator.release_lease = &ReleaseAnnFilterFusingLease;
        search_view.set_candidate_evaluator(
            evaluator,
            static_cast<size_t>(segment_->get_row_count()),
            static_cast<size_t>(program->estimated_filtered_out_count));
        const char* value_type_name =
            program->leaves.size() == 1 ? "typed_leaf" : "composite";
        milvus::monitor::internal_core_downpush_execution_count_family
            .Add({{"source", AnnFilterFusingSourceName(*downpush_ctx)},
                  {"value_type", value_type_name},
                  {"iterator",
                   search_info_.iterator_v2_info_.has_value() ? "true"
                                                              : "false"}})
            .Increment();
    }

    // Single search + metrics path
    milvus::SearchResult search_result;
    if (downpush_ctx != nullptr) {
        search_info_.ann_filter_vector_index_lease_ =
            query_context_->get_ann_filter_vector_index_lease();
        search_info_.ann_filter_vector_index_ =
            query_context_->get_ann_filter_vector_index();
    }
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
