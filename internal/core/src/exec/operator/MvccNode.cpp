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

#include "MvccNode.h"

#include <utility>
#include <vector>

#include "common/Tracer.h"
#include "exec/QueryContext.h"
#include "exec/expression/Utils.h"
#include "fmt/core.h"
#include "plan/PlanNode.h"
#include "segcore/SegcoreConfig.h"
#include "segcore/SegmentInterface.h"

namespace milvus {
namespace exec {

PhyMvccNode::PhyMvccNode(int32_t operator_id,
                         DriverContext* driverctx,
                         const std::shared_ptr<const plan::MvccNode>& mvcc_node)
    : Operator(driverctx,
               mvcc_node->output_type(),
               operator_id,
               mvcc_node->id(),
               "PhyIterativeFilterNode") {
    ExecContext* exec_context = operator_context_->get_exec_context();
    QueryContext* query_context = exec_context->get_query_context();
    segment_ = query_context->get_segment();
    query_timestamp_ = query_context->get_query_timestamp();
    active_count_ = query_context->get_active_count();
    is_source_node_ = mvcc_node->sources().size() == 0;
    collection_ttl_timestamp_ = query_context->get_collection_ttl();
}

void
PhyMvccNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

RowVectorPtr
PhyMvccNode::GetOutput() {
    auto* query_context =
        operator_context_->get_exec_context()->get_query_context();
    milvus::exec::checkCancellation(query_context);

    if (is_finished_) {
        return nullptr;
    }

    tracer::AutoSpan span("PhyMvccNode::Execute", tracer::GetRootSpan(), true);

    if (active_count_ == 0) {
        is_finished_ = true;
        return nullptr;
    }

    if (!is_source_node_ && input_ == nullptr) {
        return nullptr;
    }

    tracer::AddEvent(fmt::format("input_rows: {}", active_count_));
    WaitPrefetch();

    // Native valid IDs are only the scalar predicate result. Respect the same
    // global visibility switch as the Dense path before applying any MVCC
    // overlay; when visibility filtering is disabled, VectorSearch consumes
    // the candidate list directly.
    if (!segcore::SegcoreConfig::default_config()
             .get_visibility_filter_enabled()) {
        if (query_context->get_valid_id_payload() != nullptr) {
            is_finished_ = true;
            return input_;
        }
        auto col_input = is_source_node_ ? std::make_shared<ColumnVector>(
                                               TargetBitmap(active_count_),
                                               TargetBitmap(active_count_))
                                         : GetColumnVector(input_);
        if (is_source_node_) {
            query_context->set_all_rows_visible(true);
        }
        is_finished_ = true;
        return std::make_shared<RowVector>(std::vector<VectorPtr>{col_input});
    }

    // A native valid-ID list is the scalar predicate's candidate set, not an
    // assertion that every candidate is visible at this snapshot.  Preserve
    // the regular MVCC semantics by compacting it against the same invalid
    // timestamp/delete mask used by the Dense path.  In the immutable sealed
    // case no mask is needed and the original query-owned list is forwarded.
    if (auto payload = query_context->get_valid_id_payload(); payload != nullptr) {
        AssertInfo(!is_source_node_ && segment_->type() == SegmentType::Sealed,
                   "native valid IDs require a sealed FilterBits input");
        AssertInfo(payload->universe == active_count_,
                   "valid-ID payload universe {} does not match active row count {}",
                   payload->universe,
                   active_count_);
        const auto& native_ids = payload->ids;
        if (native_ids->empty()) {
            is_finished_ = true;
            return input_;
        }

        const bool all_rows_visible =
            collection_ttl_timestamp_ == 0 &&
            query_timestamp_ >= segment_->get_max_timestamp() &&
            segment_->get_deleted_count() == 0;
        if (!all_rows_visible) {
            TargetBitmap invalid(active_count_);
            TargetBitmapView invalid_view(invalid.data(), invalid.size());
            segment_->mask_with_timestamps(
                invalid_view, query_timestamp_, collection_ttl_timestamp_);
            segment_->mask_with_delete(
                invalid_view, active_count_, query_timestamp_);

            if (!invalid_view.none()) {
                auto surviving_ids =
                    std::make_shared<std::vector<int32_t>>();
                surviving_ids->reserve(native_ids->size());
                for (const auto id : *native_ids) {
                    AssertInfo(id >= 0 && id < active_count_,
                               "native valid ID {} is outside active row "
                               "range {}",
                               id,
                               active_count_);
                    if (!invalid_view[id]) {
                        surviving_ids->push_back(id);
                    }
                }
                query_context->set_valid_id_payload(
                    std::shared_ptr<const std::vector<int32_t>>(
                        std::move(surviving_ids)),
                    payload->universe);
            }
        }
        is_finished_ = true;
        return input_;
    }

    // ── Sealed-segment fast path (skip timestamp mask) ──
    // On a sealed segment without TTL, when query_ts covers all inserts,
    // mask_with_timestamps is redundant (all rows pass).  Only apply the
    // delete mask — which is a no-op when no deletes exist.  Then decide
    // all_rows_visible from the actual bitmap instead of a racy
    // get_deleted_count() check.
    if (is_source_node_ && segment_->type() == SegmentType::Sealed &&
        collection_ttl_timestamp_ == 0 &&
        query_timestamp_ >= segment_->get_max_timestamp()) {
        auto col_input = std::make_shared<ColumnVector>(
            TargetBitmap(active_count_), TargetBitmap(active_count_));
        TargetBitmapView data(col_input->GetRawData(), col_input->size());
        segment_->mask_with_delete(data, active_count_, query_timestamp_);

        if (data.none()) {
            query_context->set_all_rows_visible(true);
        }

        is_finished_ = true;
        return std::make_shared<RowVector>(std::vector<VectorPtr>{col_input});
    }

    // Default path (has filter / growing / TTL)
    // the first vector is filtering result and second bitset is a valid bitset
    // if valid_bitset[i]==false, means result[i] is null
    auto col_input = is_source_node_ ? std::make_shared<ColumnVector>(
                                           TargetBitmap(active_count_),
                                           TargetBitmap(active_count_))
                                     : GetColumnVector(input_);

    TargetBitmapView data(col_input->GetRawData(), col_input->size());
    segment_->mask_with_timestamps(
        data, query_timestamp_, collection_ttl_timestamp_);
    segment_->mask_with_delete(data, active_count_, query_timestamp_);
    is_finished_ = true;

    // input_ have already been updated
    return std::make_shared<RowVector>(std::vector<VectorPtr>{col_input});
}

bool
PhyMvccNode::IsFinished() {
    return is_finished_;
}

}  // namespace exec
}  // namespace milvus
