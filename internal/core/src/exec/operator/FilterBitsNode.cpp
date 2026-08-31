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

#include "FilterBitsNode.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <ratio>
#include <utility>
#include <vector>

#include "common/EasyAssert.h"
#include "common/Tracer.h"
#include "common/Types.h"
#include "exec/QueryContext.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/ExprCache.h"
#include "exec/expression/DownpushPredicateProvider.h"
#include "exec/operator/DownpushSearchContext.h"
#include "expr/ITypeExpr.h"
#include "fmt/core.h"
#include "log/Log.h"
#include "monitor/Monitor.h"
#include "plan/PlanNode.h"
#include "prometheus/histogram.h"

namespace milvus {
namespace exec {

namespace {

constexpr int64_t kDownpushEstimatorSampleSize = 256;

std::string
BuildExprCacheKey(const plan::FilterBitsNode& filter,
                  QueryContext* query_context) {
    auto key = filter.ToString();
    auto* segment =
        query_context != nullptr ? query_context->get_segment() : nullptr;
    if (segment != nullptr &&
        segment->get_schema().get_ttl_field_id().has_value()) {
        key += fmt::format("|entity_ttl_physical_time_us:{}",
                           query_context->get_entity_ttl_physical_time_us());
    }
    return key;
}

uint64_t
SplitMix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

OffsetVector
BuildSampleOffsets(int64_t active_count,
                   int64_t sample_size,
                   int64_t segment_id) {
    auto real_sample_size = std::min(active_count, sample_size);
    OffsetVector offsets;
    offsets.reserve(real_sample_size);
    for (int64_t i = 0; i < real_sample_size; ++i) {
        auto hash =
            SplitMix64(static_cast<uint64_t>(segment_id) ^
                       (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL));
        offsets.push_back(static_cast<int32_t>(hash % active_count));
    }
    return offsets;
}

std::optional<CardinalDownpushPredicateOp>
ToDownpushRangeOp(proto::plan::OpType op_type) {
    switch (op_type) {
        case proto::plan::OpType::GreaterEqual:
            return CardinalDownpushPredicateOp::Int64GreaterEqual;
        case proto::plan::OpType::GreaterThan:
            return CardinalDownpushPredicateOp::Int64GreaterThan;
        case proto::plan::OpType::LessEqual:
            return CardinalDownpushPredicateOp::Int64LessEqual;
        case proto::plan::OpType::LessThan:
            return CardinalDownpushPredicateOp::Int64LessThan;
        case proto::plan::OpType::Equal:
            return CardinalDownpushPredicateOp::Int64Equal;
        case proto::plan::OpType::NotEqual:
            return CardinalDownpushPredicateOp::Int64NotEqual;
        case proto::plan::OpType::PrefixMatch:
            return CardinalDownpushPredicateOp::StringPrefixMatch;
        case proto::plan::OpType::PostfixMatch:
            return CardinalDownpushPredicateOp::StringPostfixMatch;
        case proto::plan::OpType::InnerMatch:
            return CardinalDownpushPredicateOp::StringInnerMatch;
        case proto::plan::OpType::Match:
            return CardinalDownpushPredicateOp::StringLikeMatch;
        default:
            return std::nullopt;
    }
}

bool
IsStringMatchOp(CardinalDownpushPredicateOp op) {
    return op == CardinalDownpushPredicateOp::StringPrefixMatch ||
           op == CardinalDownpushPredicateOp::StringPostfixMatch ||
           op == CardinalDownpushPredicateOp::StringInnerMatch ||
           op == CardinalDownpushPredicateOp::StringLikeMatch;
}

bool
TryFoldInt64TermsToRange(CardinalDownpushPredicate& predicate) {
    if (predicate.value_type_ != CardinalDownpushPredicateValueType::Int64 ||
        predicate.int64_terms_.empty()) {
        return false;
    }

    const auto first = predicate.int64_terms_.front();
    const auto last = predicate.int64_terms_.back();
    if (last < first) {
        return false;
    }

    const auto expected_size =
        static_cast<__int128>(last) - static_cast<__int128>(first) + 1;
    if (expected_size <= 0 ||
        expected_size != static_cast<__int128>(predicate.int64_terms_.size())) {
        return false;
    }

    predicate.arg0_ = first;
    predicate.arg1_ = last;
    predicate.lower_inclusive_ = true;
    predicate.upper_inclusive_ = true;
    predicate.int64_terms_.clear();
    predicate.op_ = CardinalDownpushPredicateOp::ScalarRange;
    return true;
}

std::optional<CardinalDownpushPredicate>
TryCompileCardinalDownpushLeaf(const expr::TypedExprPtr& filter,
                               QueryContext* query_context) {
    if (query_context == nullptr || filter == nullptr) {
        return std::nullopt;
    }
    auto* segment = query_context->get_segment();
    if (segment == nullptr || segment->type() != SegmentType::Sealed) {
        return std::nullopt;
    }

    struct ProviderPredicate {
        CardinalDownpushPredicate predicate;
        const DownpushPredicateProvider* provider;
    };
    auto try_field = [&](const expr::ColumnInfo& column)
        -> std::optional<ProviderPredicate> {
        auto field_id = column.field_id_;
        const auto* provider = FindDownpushPredicateProvider(column);
        if (provider == nullptr) {
            return std::nullopt;
        }
        if (!segment->HasFieldData(field_id) && !segment->HasIndex(field_id)) {
            return std::nullopt;
        }
        return ProviderPredicate{provider->NewPredicate(column), provider};
    };

    if (auto unary =
            std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(
                filter)) {
        auto predicate = try_field(unary->column_);
        auto op = ToDownpushRangeOp(unary->op_type_);
        if (!predicate.has_value() || !op.has_value() ||
            !predicate->provider->SupportsRangeOp(*op) ||
            !predicate->provider->FillArg(
                predicate->predicate, unary->val_, false)) {
            return std::nullopt;
        }
        predicate->predicate.op_ = op.value();
        if (!predicate->provider->FinalizeUnary(predicate->predicate)) {
            return std::nullopt;
        }
        return std::move(predicate->predicate);
    }

    if (auto binary =
            std::dynamic_pointer_cast<const expr::BinaryRangeFilterExpr>(
                filter)) {
        auto predicate = try_field(binary->column_);
        if (!predicate.has_value() ||
            !predicate->provider->FillArg(
                predicate->predicate, binary->lower_val_, false) ||
            !predicate->provider->FillArg(
                predicate->predicate, binary->upper_val_, true)) {
            return std::nullopt;
        }
        predicate->predicate.op_ = CardinalDownpushPredicateOp::ScalarRange;
        predicate->predicate.lower_inclusive_ = binary->lower_inclusive_;
        predicate->predicate.upper_inclusive_ = binary->upper_inclusive_;
        return std::move(predicate->predicate);
    }

    if (auto term =
            std::dynamic_pointer_cast<const expr::TermFilterExpr>(filter)) {
        auto predicate = try_field(term->column_);
        if (!predicate.has_value() || !predicate->provider->FillTerms(
                                          predicate->predicate, term->vals_)) {
            return std::nullopt;
        }
        if (TryFoldInt64TermsToRange(predicate->predicate)) {
            return std::move(predicate->predicate);
        }
        predicate->predicate.op_ = CardinalDownpushPredicateOp::ScalarTerm;
        return std::move(predicate->predicate);
    }

    if (auto arith =
            std::dynamic_pointer_cast<const expr::BinaryArithOpEvalRangeExpr>(
                filter)) {
        auto predicate = try_field(arith->column_);
        if (!predicate.has_value() || !predicate->provider->FillArithmetic(
                                          predicate->predicate, *arith)) {
            return std::nullopt;
        }
        return std::move(predicate->predicate);
    }

    return std::nullopt;
}

std::optional<size_t>
CompileCandidatePredicateNode(const expr::TypedExprPtr& filter,
                              QueryContext* query_context,
                              CardinalDownpushPredicateProgram& program) {
    if (auto logical =
            std::dynamic_pointer_cast<const expr::LogicalBinaryExpr>(filter)) {
        if (logical->inputs().size() != 2 ||
            (logical->op_type_ != expr::LogicalBinaryExpr::OpType::And &&
             logical->op_type_ != expr::LogicalBinaryExpr::OpType::Or)) {
            return std::nullopt;
        }
        auto left = CompileCandidatePredicateNode(
            logical->inputs()[0], query_context, program);
        auto right = CompileCandidatePredicateNode(
            logical->inputs()[1], query_context, program);
        if (!left.has_value() || !right.has_value()) {
            return std::nullopt;
        }
        CandidatePredicateNode node;
        node.type = logical->op_type_ == expr::LogicalBinaryExpr::OpType::And
                        ? CandidatePredicateNodeType::And
                        : CandidatePredicateNodeType::Or;
        node.left = *left;
        node.right = *right;
        program.nodes.push_back(node);
        return program.nodes.size() - 1;
    }
    if (auto logical =
            std::dynamic_pointer_cast<const expr::LogicalUnaryExpr>(filter)) {
        if (logical->op_type_ != expr::LogicalUnaryExpr::OpType::LogicalNot ||
            logical->inputs().size() != 1) {
            return std::nullopt;
        }
        auto child = CompileCandidatePredicateNode(
            logical->inputs()[0], query_context, program);
        if (!child.has_value()) {
            return std::nullopt;
        }
        program.nodes.push_back(
            CandidatePredicateNode{CandidatePredicateNodeType::Not, *child, 0});
        return program.nodes.size() - 1;
    }

    auto predicate = TryCompileCardinalDownpushLeaf(filter, query_context);
    if (!predicate.has_value()) {
        return std::nullopt;
    }
    // Numeric raw/bulk source views do not yet carry scalar validity. Avoid
    // claiming SQL three-valued semantics for nullable Numeric leaves.
    if (predicate->field_nullable_ &&
        predicate->value_type_ != CardinalDownpushPredicateValueType::String) {
        return std::nullopt;
    }
    const auto leaf = program.leaves.size();
    program.leaf_nullable.push_back(predicate->field_nullable_);
    program.leaves.push_back(std::move(*predicate));
    program.nodes.push_back(
        CandidatePredicateNode{CandidatePredicateNodeType::Leaf, leaf, 0});
    return program.nodes.size() - 1;
}

std::optional<CardinalDownpushPredicateProgram>
TryCompileCardinalDownpushProgram(const expr::TypedExprPtr& filter,
                                  QueryContext* query_context) {
    CardinalDownpushPredicateProgram program;
    auto root = CompileCandidatePredicateNode(filter, query_context, program);
    if (!root.has_value() || program.leaves.empty()) {
        return std::nullopt;
    }
    program.root = *root;
    return program;
}

struct CandidatePredicateSummary {
    uint32_t leaf_count{0};
    uint32_t logical_node_count{0};
};

std::optional<CandidatePredicateSummary>
AnalyzeCandidatePredicateShape(const expr::TypedExprPtr& filter) {
    if (filter == nullptr) {
        return std::nullopt;
    }
    if (auto logical =
            std::dynamic_pointer_cast<const expr::LogicalBinaryExpr>(filter)) {
        if (logical->inputs().size() != 2 ||
            (logical->op_type_ != expr::LogicalBinaryExpr::OpType::And &&
             logical->op_type_ != expr::LogicalBinaryExpr::OpType::Or)) {
            return std::nullopt;
        }
        auto left = AnalyzeCandidatePredicateShape(logical->inputs()[0]);
        auto right = AnalyzeCandidatePredicateShape(logical->inputs()[1]);
        if (!left.has_value() || !right.has_value()) {
            return std::nullopt;
        }
        return CandidatePredicateSummary{
            left->leaf_count + right->leaf_count,
            left->logical_node_count + right->logical_node_count + 1};
    }
    if (auto logical =
            std::dynamic_pointer_cast<const expr::LogicalUnaryExpr>(filter)) {
        if (logical->op_type_ != expr::LogicalUnaryExpr::OpType::LogicalNot ||
            logical->inputs().size() != 1) {
            return std::nullopt;
        }
        auto child = AnalyzeCandidatePredicateShape(logical->inputs()[0]);
        if (!child.has_value()) {
            return std::nullopt;
        }
        ++child->logical_node_count;
        return child;
    }
    if (std::dynamic_pointer_cast<const expr::UnaryRangeFilterExpr>(filter) ||
        std::dynamic_pointer_cast<const expr::BinaryRangeFilterExpr>(filter) ||
        std::dynamic_pointer_cast<const expr::TermFilterExpr>(filter) ||
        std::dynamic_pointer_cast<const expr::BinaryArithOpEvalRangeExpr>(
            filter)) {
        return CandidatePredicateSummary{1, 0};
    }
    return std::nullopt;
}

std::optional<int64_t>
EstimateFilteredOutCountBySample(QueryContext* query_context,
                                 ExprSet* exprs,
                                 ExecContext* exec_context) {
    if (query_context == nullptr || exprs == nullptr ||
        exec_context == nullptr) {
        return std::nullopt;
    }
    auto active_count = query_context->get_active_count();
    if (active_count <= 0) {
        return 0;
    }
    auto* segment = query_context->get_segment();
    auto segment_id = segment != nullptr ? segment->get_segment_id() : 0;
    auto offsets = BuildSampleOffsets(
        active_count, kDownpushEstimatorSampleSize, segment_id);
    if (offsets.empty()) {
        return 0;
    }

    EvalCtx eval_ctx(exec_context, &offsets);
    std::vector<VectorPtr> results;
    exprs->Eval(0, 1, true, eval_ctx, results);
    if (results.size() != 1 || results[0] == nullptr) {
        return std::nullopt;
    }
    auto col_vec = std::dynamic_pointer_cast<ColumnVector>(results[0]);
    if (!col_vec || !col_vec->IsBitmap()) {
        return std::nullopt;
    }
    TargetBitmapView passed(col_vec->GetRawData(), col_vec->size());
    auto passed_count = static_cast<int64_t>(passed.count());
    auto filtered_sample_count =
        static_cast<int64_t>(offsets.size()) - passed_count;
    auto estimate =
        static_cast<int64_t>((static_cast<double>(filtered_sample_count) /
                              static_cast<double>(offsets.size())) *
                             static_cast<double>(active_count));
    return std::clamp<int64_t>(estimate, 0, active_count);
}

}  // namespace

PhyFilterBitsNode::PhyFilterBitsNode(
    int32_t operator_id,
    DriverContext* driverctx,
    const std::shared_ptr<const plan::FilterBitsNode>& filter)
    : Operator(driverctx,
               filter->output_type(),
               operator_id,
               filter->id(),
               "PhyFilterBitsNode") {
    ExecContext* exec_context = operator_context_->get_exec_context();
    query_context_ = exec_context->get_query_context();
    std::vector<expr::TypedExprPtr> filters;
    filters.emplace_back(filter->filter());
    exprs_ = std::make_unique<ExprSet>(filters, exec_context);
    need_process_rows_ = query_context_->get_active_count();
    num_processed_rows_ = 0;

    const auto& search_info = query_context_->get_search_info();
    if (search_info.cardinal_downpush_fallback_reason.has_value()) {
        milvus::monitor::internal_core_downpush_fallback_count_family
            .Add({{"reason",
                   search_info.cardinal_downpush_fallback_reason.value()}})
            .Increment();
    } else if (search_info.cardinal_downpush_execution) {
        // downpush hint (ann filter fusing): attempt to fuse the scalar
        // predicate into the vector index. The hint is advisory — if any
        // precondition is unmet we silently fall back to the normal ExprSet
        // path below and never break correctness.
        TryEnableCardinalDownpush(*filter, exec_context);
    }

    enable_expr_cache_ = query_context_->get_enable_expr_cache();
    if (enable_expr_cache_ && !cardinal_downpush_enabled_) {
        expr_cache_key_ = BuildExprCacheKey(*filter, query_context_);
    }
}

void
PhyFilterBitsNode::TryEnableCardinalDownpush(const plan::FilterBitsNode& filter,
                                             ExecContext* exec_context) {
    const auto& search_info = query_context_->get_search_info();

    auto fallback = [](const char* reason) {
        milvus::monitor::internal_core_downpush_fallback_count_family
            .Add({{"reason", reason}})
            .Increment();
    };

    // Fusion is not implemented for element-level (array-of-vectors) search.
    if (search_info.element_level()) {
        LOG_DEBUG("downpush fallback: element-level vector search unsupported");
        fallback("element_level");
        return;
    }

    // Planning sees only a cheap expression-shape summary. Building the
    // Milvus-owned executable predicate program is deliberately deferred until
    // Cardinal commits FUSING, so BASELINE/AUTO rejection pays no IR cost.
    auto predicate_summary = AnalyzeCandidatePredicateShape(filter.filter());
    if (!predicate_summary.has_value()) {
        LOG_DEBUG("downpush fallback: unsupported predicate shape");
        fallback("unsupported_predicate");
        return;
    }

    std::vector<expr::TypedExprPtr> filters{filter.filter()};
    // Cost-model selectivity describes only the optional user predicate.
    // Mandatory entity TTL remains a separate upper-layer constraint.
    ExprSet sample_exprs(filters, exec_context, false);
    auto estimated_filtered_out_count = EstimateFilteredOutCountBySample(
        query_context_, &sample_exprs, exec_context);
    if (!estimated_filtered_out_count.has_value()) {
        LOG_DEBUG("downpush fallback: failed to estimate filter ratio");
        fallback("estimate_failed");
        return;
    }

    knowhere::AnnFilterPlanRequestV1 plan_request;
    plan_request.mode = knowhere::AnnFilterRequestMode::kExplicitFusing;
    plan_request.row_count = static_cast<uint64_t>(
        std::max<int64_t>(0, need_process_rows_));
    plan_request.estimated_filtered_out_count = static_cast<uint64_t>(
        std::max<int64_t>(0, estimated_filtered_out_count.value()));
    plan_request.topk = static_cast<uint64_t>(
        std::max<int64_t>(0, search_info.topk_));
    plan_request.predicate_leaf_count = predicate_summary->leaf_count;
    plan_request.predicate_logical_node_count =
        predicate_summary->logical_node_count;

    std::shared_ptr<void> vector_index_lease;
    void* planned_vector_index = nullptr;
    const auto plan = query_context_->get_segment()->PlanAnnFilter(
        query_context_->get_op_context(),
        search_info.field_id_,
        plan_request,
        &vector_index_lease,
        &planned_vector_index);
    if (plan.abi_major != knowhere::kAnnFilterPlannerAbiMajor ||
        plan.struct_size < sizeof(knowhere::AnnFilterPlanResultV1) ||
        plan.policy != knowhere::AnnFilterPolicy::kFusing) {
        const char* reason = "planner_unavailable";
        if (plan.reason == knowhere::AnnFilterPlanReason::kGraphUnavailable) {
            reason = "graph_unavailable";
        } else if (plan.reason ==
                   knowhere::AnnFilterPlanReason::kCostBaseline) {
            reason = "cost_baseline";
        } else if (plan.reason ==
                   knowhere::AnnFilterPlanReason::kIncompatibleRequest) {
            reason = "plan_version_mismatch";
        }
        LOG_DEBUG(
            "downpush fallback: Cardinal pre-plan rejected request, reason={}",
            reason);
        fallback(reason);
        return;
    }
    AssertInfo(vector_index_lease != nullptr && planned_vector_index != nullptr,
               "fusing planner accepted without an index lease");

    auto program =
        TryCompileCardinalDownpushProgram(filter.filter(), query_context_);
    if (!program.has_value()) {
        LOG_DEBUG(
            "downpush fallback: unsupported predicate operation or scalar source");
        fallback("unsupported_predicate");
        return;
    }

    // A LIKE predicate requires raw varchar values. A sealed segment backed by
    // STL_SORT may expose dictionary IDs without the raw string chunk required
    // by LIKE. This remains inside the pre-commit Prepare transaction.
    if (std::any_of(
            program->leaves.begin(),
            program->leaves.end(),
            [](const auto& leaf) { return IsStringMatchOp(leaf.op_); })) {
        LOG_DEBUG(
            "downpush fallback: varchar match needs a raw value source that "
            "is unavailable for some sealed scalar-index layouts");
        fallback("unsupported_predicate");
        return;
    }

    // Only FUSING needs a separate mandatory TTL bitmap; baseline ExprSet
    // already owns its normal TTL injection.
    auto ttl_expr = CreateTTLFieldFilterExpression(query_context_);
    if (ttl_expr != nullptr) {
        ttl_exprs_ = std::make_unique<ExprSet>(
            std::vector<expr::TypedExprPtr>{ttl_expr}, exec_context, false);
    }

    auto downpush_search_context =
        PrepareCardinalDownpushSearchContext(query_context_->get_segment(),
                                             query_context_->get_op_context(),
                                             program.value());
    if (downpush_search_context == nullptr) {
        LOG_DEBUG("downpush fallback: scalar value source unavailable");
        fallback("source_unavailable");
        return;
    }

    program->estimated_filtered_out_count =
        need_process_rows_ > 0
            ? std::max<int64_t>(1, estimated_filtered_out_count.value())
            : 0;
    cardinal_downpush_program_ = std::move(program);
    cardinal_downpush_search_context_ = std::move(downpush_search_context);
    query_context_->set_ann_filter_vector_index_lease(
        std::move(vector_index_lease), planned_vector_index);
    cardinal_downpush_enabled_ = true;
}

void
PhyFilterBitsNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

bool
PhyFilterBitsNode::AllInputProcessed() {
    if (num_processed_rows_ == need_process_rows_) {
        input_ = nullptr;
        return true;
    }
    return false;
}

bool
PhyFilterBitsNode::IsFinished() {
    return AllInputProcessed();
}

RowVectorPtr
PhyFilterBitsNode::GetOutput() {
    milvus::exec::checkCancellation(query_context_);

    if (AllInputProcessed()) {
        return nullptr;
    }

    // Cache read: Stage 2 of two-stage search reuses the bitset cached by Stage 1.
    // Cache lives in the process-level ExprResCacheManager keyed by
    // (segment_id, FilterBitsNode signature + dynamic filter context), so
    // cross-query reuse is automatic only when the effective predicate matches.
    auto* cache_segment = query_context_->get_segment();
    const bool can_use_cache = enable_expr_cache_ && !expr_cache_key_.empty() &&
                               cache_segment != nullptr &&
                               cache_segment->type() == SegmentType::Sealed &&
                               ExprResCacheManager::IsEnabled();
    if (can_use_cache) {
        ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                     expr_cache_key_};
        ExprResCacheManager::Value cached;
        cached.active_count = need_process_rows_;
        if (ExprResCacheManager::Instance().Get(key, cached) &&
            cached.result != nullptr &&
            cached.result->size() == need_process_rows_) {
            num_processed_rows_ = need_process_rows_;
            std::vector<VectorPtr> col_res;
            col_res.push_back(std::make_shared<ColumnVector>(
                cached.result->clone(),
                cached.valid_result ? cached.valid_result->clone()
                                    : TargetBitmap(need_process_rows_, true)));
            return std::make_shared<RowVector>(col_res);
        }
    }

    tracer::AutoSpan span(
        "PhyFilterBitsNode::Execute", tracer::GetRootSpan(), true);
    tracer::AddEvent(fmt::format("input_rows: {}", need_process_rows_));

    if (cardinal_downpush_enabled_) {
        query_context_->set_cardinal_downpush_program(
            cardinal_downpush_program_.value());
        query_context_->set_cardinal_downpush_search_context(
            cardinal_downpush_search_context_);
        num_processed_rows_ = need_process_rows_;
        std::vector<VectorPtr> col_res;
        if (ttl_exprs_ != nullptr) {
            // entity TTL stays a normal logical-space bitset (`1` = exclude),
            // while the user predicate is deferred into the vector index.
            EvalCtx ttl_eval_ctx(operator_context_->get_exec_context());
            std::vector<VectorPtr> ttl_results;
            ttl_exprs_->Eval(0, 1, true, ttl_eval_ctx, ttl_results);
            AssertInfo(ttl_results.size() == 1 && ttl_results[0] != nullptr,
                       "TTL filter should produce a single bitmap result");
            auto ttl_col =
                std::dynamic_pointer_cast<ColumnVector>(ttl_results[0]);
            AssertInfo(ttl_col && ttl_col->IsBitmap(),
                       "TTL filter result should be a bitmap ColumnVector");
            AssertInfo(
                static_cast<int64_t>(ttl_col->size()) == need_process_rows_,
                "TTL filter result size {} != need_process_rows_ {}",
                ttl_col->size(),
                need_process_rows_);
            // Eval yields `1` = keep (TTL not expired); flip to `1` = exclude
            // to match the exclude-bitset convention consumed downstream.
            TargetBitmapView ttl_view(ttl_col->GetRawData(), ttl_col->size());
            ttl_view.flip();
            col_res.push_back(std::move(ttl_results[0]));
        } else {
            col_res.push_back(std::make_shared<ColumnVector>(
                TargetBitmap(need_process_rows_, false),
                TargetBitmap(need_process_rows_, true)));
        }
        return std::make_shared<RowVector>(col_res);
    }

    std::chrono::high_resolution_clock::time_point scalar_start =
        std::chrono::high_resolution_clock::now();

    EvalCtx eval_ctx(operator_context_->get_exec_context());

    TargetBitmap bitset;
    TargetBitmap valid_bitset;

    // optimization: if all expressions can be executed at once,
    // execute in a single pass and flip in-place to avoid bitmap copies.
    if (exprs_->CanExecuteAllAtOnce()) {
        tracer::AddEvent("expr_execute_all_at_once");
        exprs_->SetExecuteAllAtOnce();

        exprs_->Eval(0, 1, true, eval_ctx, results_);
        AssertInfo(results_.size() == 1 && results_[0] != nullptr,
                   "PhyFilterBitsNode result size should be size one and not "
                   "be nullptr");
        auto col_vec = std::dynamic_pointer_cast<ColumnVector>(results_[0]);
        AssertInfo(col_vec && col_vec->IsBitmap(),
                   "PhyFilterBitsNode result should be bitmap ColumnVector");

        auto col_vec_size = col_vec->size();
        // flip in-place on the result bitmap, no extra copy
        TargetBitmapView view(col_vec->GetRawData(), col_vec_size);
        view.flip();
        num_processed_rows_ = col_vec_size;

        AssertInfo(col_vec_size == need_process_rows_,
                   "bitset size: {}, need_process_rows_: {}",
                   col_vec_size,
                   need_process_rows_);

        if (can_use_cache) {
            TargetBitmapView valid_view(col_vec->GetValidRawData(),
                                        col_vec_size);
            ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                         expr_cache_key_};
            ExprResCacheManager::Value v;
            v.result = std::make_shared<TargetBitmap>(view);
            v.valid_result = std::make_shared<TargetBitmap>(valid_view);
            v.active_count = need_process_rows_;
            ExprResCacheManager::Instance().Put(key, v);
        }

        std::vector<VectorPtr> col_res;
        col_res.push_back(std::move(results_[0]));

        std::chrono::high_resolution_clock::time_point scalar_end =
            std::chrono::high_resolution_clock::now();
        double scalar_cost =
            std::chrono::duration<double, std::micro>(scalar_end - scalar_start)
                .count();
        milvus::monitor::internal_core_search_latency_scalar.Observe(
            scalar_cost / 1000);

        return std::make_shared<RowVector>(col_res);
    }

    while (num_processed_rows_ < need_process_rows_) {
        exprs_->Eval(0, 1, true, eval_ctx, results_);

        AssertInfo(results_.size() == 1 && results_[0] != nullptr,
                   "PhyFilterBitsNode result size should be size one and not "
                   "be nullptr");

        if (auto col_vec =
                std::dynamic_pointer_cast<ColumnVector>(results_[0])) {
            if (col_vec->IsBitmap()) {
                auto col_vec_size = col_vec->size();
                TargetBitmapView view(col_vec->GetRawData(), col_vec_size);
                bitset.append(view);
                TargetBitmapView valid_view(col_vec->GetValidRawData(),
                                            col_vec_size);
                valid_bitset.append(valid_view);
                num_processed_rows_ += col_vec_size;
            } else {
                ThrowInfo(ExprInvalid,
                          "PhyFilterBitsNode result should be bitmap");
            }
        } else {
            ThrowInfo(ExprInvalid,
                      "PhyFilterBitsNode result should be ColumnVector");
        }
    }
    bitset.flip();

    AssertInfo(bitset.size() == need_process_rows_,
               "bitset size: {}, need_process_rows_: {}",
               bitset.size(),
               need_process_rows_);
    Assert(valid_bitset.size() == need_process_rows_);

    // Cache write: clone bitset into ExprResCacheManager — Stage 1 of two-stage
    // search. Must clone before move since Stage 1 still owns the bitset for
    // the ColumnVector return value below.
    if (can_use_cache) {
        ExprResCacheManager::Key key{cache_segment->get_segment_id(),
                                     expr_cache_key_};
        ExprResCacheManager::Value v;
        v.result = std::make_shared<TargetBitmap>(bitset.clone());
        v.valid_result = std::make_shared<TargetBitmap>(valid_bitset.clone());
        v.active_count = need_process_rows_;
        ExprResCacheManager::Instance().Put(key, v);
    }

    // num_processed_rows_ = need_process_rows_;
    std::vector<VectorPtr> col_res;
    col_res.push_back(std::make_shared<ColumnVector>(std::move(bitset),
                                                     std::move(valid_bitset)));
    std::chrono::high_resolution_clock::time_point scalar_end =
        std::chrono::high_resolution_clock::now();
    double scalar_cost =
        std::chrono::duration<double, std::micro>(scalar_end - scalar_start)
            .count();
    milvus::monitor::internal_core_search_latency_scalar.Observe(scalar_cost /
                                                                 1000);

    return std::make_shared<RowVector>(col_res);
}

}  // namespace exec
}  // namespace milvus
