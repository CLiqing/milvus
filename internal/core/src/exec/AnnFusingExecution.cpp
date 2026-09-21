// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#include "exec/AnnFusingPlan.h"

#include "exec/AnnFusingPolicy.h"
#include "exec/expression/EvalCtx.h"
#include "exec/expression/PreparedBitmapExpr.h"
#include "log/Log.h"

namespace milvus::exec {
namespace {
using Logical = expr::LogicalBinaryExpr;

expr::TypedExprPtr
And(expr::TypedExprPtr left, expr::TypedExprPtr right) {
    if (!left)
        return right;
    if (!right)
        return left;
    return std::make_shared<Logical>(Logical::OpType::And, left, right);
}

void
Conjuncts(const expr::TypedExprPtr& expression,
          std::vector<expr::TypedExprPtr>& out) {
    const auto* logical = dynamic_cast<const Logical*>(expression.get());
    if (logical && logical->op_type_ == Logical::OpType::And) {
        for (const auto& child : logical->inputs()) Conjuncts(child, out);
    } else {
        // OR/NOT are boundaries: never lift one of their branches into the
        // mandatory bitmap. No distribution/CNF/DNF rewriting.
        out.push_back(expression);
    }
}

const expr::BinaryArithOpEvalRangeExpr*
SupportedMod(const expr::TypedExprPtr& expression) {
    const auto* mod =
        dynamic_cast<const expr::BinaryArithOpEvalRangeExpr*>(expression.get());
    return mod && mod->column_.data_type_ == DataType::INT64 &&
                   !mod->column_.nullable_ && !mod->column_.element_level_ &&
                   mod->arith_op_type_ == proto::plan::ArithOpType::Mod &&
                   mod->right_operand_.has_int64_val() &&
                   mod->right_operand_.int64_val() > 0
               ? mod
               : nullptr;
}

// The scoped first OR shape is cached indexed range OR an eligible MOD.
// Other OR/NOT subtrees keep their original baseline execution.
struct Candidate {
    expr::TypedExprPtr original;
    expr::TypedExprPtr predicate;
    expr::TypedExprPtr indexed_or_operand;
};

std::shared_ptr<const PreparedBitmapExpr>
PrepareIndexedResult(const expr::TypedExprPtr& expression,
                     ExecContext* context) {
    auto* query = context->get_query_context();
    const auto facts = DescribeAnnFusingLeaf(
        expression, *query->get_segment(), query->get_op_context());
    if (!facts || facts->operation != "range" ||
        facts->index_type != "STL_SORT")
        return nullptr;
    // Compile original operators, preserving truth + validity. No synthetic
    // leaf truth or per-operator fusing implementation is introduced here.
    ExprSet source({expression}, context, false);
    if (!source.CanExecuteAllAtOnce())
        return nullptr;
    source.SetExecuteAllAtOnce();
    EvalCtx evaluation(context);
    std::vector<VectorPtr> results;
    source.Eval(evaluation, results);
    AssertInfo(results.size() == 1 && results[0],
               "indexed baseline result missing");
    auto result = std::dynamic_pointer_cast<ColumnVector>(results[0]);
    AssertInfo(result && result->IsBitmap() &&
                   result->size() == query->get_active_count(),
               "prepared indexed result must cover the entire segment");
    LOG_DEBUG("ann_fusing prepared_bitmap field={} rows={} index={}",
              facts->field_id.get(),
              result->size(),
              facts->index_type);
    return std::make_shared<PreparedBitmapExpr>(expression, result);
}
}  // namespace

AnnFusingExecutionPlan
PlanAnnFusingExpression(const expr::TypedExprPtr& expression,
                        ExecContext* context,
                        AnnFilterFusingRequest request) {
    if (request == AnnFilterFusingRequest::Baseline)
        return {expression, nullptr};
    const auto* policy = request == AnnFilterFusingRequest::Auto
                             ? &AnnFusingPolicy::Instance()
                             : nullptr;
    if (policy && !policy->available())
        return {expression, nullptr};
    auto* query = context->get_query_context();
    const auto* segment = query->get_segment();
    auto consider = [&](const expr::TypedExprPtr& leaf) {
        if (!policy)
            return true;
        const auto facts =
            DescribeAnnFusingLeaf(leaf, *segment, query->get_op_context());
        if (!facts)
            return false;
        const bool result = policy->Consider(facts->view());
        LOG_DEBUG(
            "ann_fusing auto rule type={} op={} index={} consider={} reason={}",
            facts->data_type,
            facts->operation,
            facts->index_type,
            result,
            result ? "consider_sample" : "baseline_rule");
        return result;
    };

    std::vector<expr::TypedExprPtr> terms, baseline_terms;
    std::vector<Candidate> candidates;
    Conjuncts(expression, terms);
    for (const auto& term : terms) {
        const bool eligible = consider(term);
        if (SupportedMod(term) && eligible) {
            candidates.push_back({term, term, nullptr});
            continue;
        }
        const auto* logical = dynamic_cast<const Logical*>(term.get());
        bool planned_or = false;
        if (logical && logical->op_type_ == Logical::OpType::Or &&
            logical->inputs().size() == 2) {
            for (size_t side = 0; side < 2; ++side) {
                const auto& mod = logical->inputs()[side];
                const auto& other = logical->inputs()[1 - side];
                if (!SupportedMod(mod))
                    continue;
                const auto facts = DescribeAnnFusingLeaf(
                    other, *segment, query->get_op_context());
                if (facts && facts->operation == "range" &&
                    facts->index_type == "STL_SORT" && consider(mod)) {
                    // Keep policy observation of the indexed operand without
                    // pretending its FALSE result rejects the whole OR.
                    consider(other);
                    candidates.push_back({term, mod, other});
                    planned_or = true;
                    break;
                }
            }
        }
        if (!planned_or)
            baseline_terms.push_back(term);
    }
    if (candidates.empty() ||
        !segment->SupportsAnnFusingDemo(query->get_op_context(),
                                        query->get_search_info().field_id_)) {
        return {expression, nullptr};
    }

    AnnFusingExecutionPlan plan;
    std::optional<TargetBitmap> known_mandatory;
    for (const auto& term : baseline_terms) {
        auto cached = PrepareIndexedResult(term, context);
        plan.baseline = And(plan.baseline, cached ? cached : term);
        if (cached) {
            auto accepted = cached->truth().clone();
            accepted.inplace_and(cached->valid(), accepted.size());
            if (known_mandatory)
                known_mandatory->inplace_and(accepted, accepted.size());
            else
                known_mandatory = std::move(accepted);
        }
    }
    const double mandatory_ratio =
        known_mandatory && known_mandatory->size() != 0
            ? 1.0 - static_cast<double>(known_mandatory->count()) /
                        known_mandatory->size()
            : -1;

    for (const auto& candidate : candidates) {
        expr::TypedExprPtr complete = candidate.predicate;
        if (candidate.indexed_or_operand) {
            const auto cached =
                PrepareIndexedResult(candidate.indexed_or_operand, context);
            if (!cached) {
                plan.baseline = And(plan.baseline, candidate.original);
                continue;
            }
            // Existing OR implementation combines a worker-local gather of A
            // with the original B evaluator. A is NOT part of mandatory_ratio.
            complete = std::make_shared<Logical>(
                Logical::OpType::Or, cached, candidate.predicate);
        }
        bool fusing = true;
        if (policy) {
            const auto* mod = SupportedMod(candidate.predicate);
            const auto ratio = SampleAnnFusingRejection(
                candidate.predicate, mod->column_.field_id_, context);
            fusing = ratio && policy->Choose({sizeof(MilvusAnnFusingSampleV1),
                                              ratio.value_or(1),
                                              mandatory_ratio});
            LOG_DEBUG(
                "ann_fusing auto sample decision={} rejection_ratio={} "
                "mandatory_rejection_ratio={}",
                fusing ? "fusing" : "baseline",
                ratio.value_or(-1),
                mandatory_ratio);
        }
        if (fusing)
            plan.residual = And(plan.residual, complete);
        else
            plan.baseline = And(plan.baseline, complete);
    }
    return plan;
}

}  // namespace milvus::exec
