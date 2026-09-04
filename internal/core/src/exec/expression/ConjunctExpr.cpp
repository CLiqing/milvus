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

#include "ConjunctExpr.h"

#include <algorithm>
#include <numeric>

#include "LikeConjunctExpr.h"
#include "UnaryExpr.h"
#include "common/Common.h"
#include "common/EasyAssert.h"
#include "common/Tracer.h"
#include "common/ValueOp.h"
#include "exec/QueryContext.h"
#include "exec/expression/Utils.h"
#include "fmt/core.h"
#include "opentelemetry/trace/span.h"

namespace milvus {
namespace exec {

namespace {

bool
ApplyPredicateToDenseResult(const ExprPtr& predicate,
                            EvalCtx& context,
                            FilterMap& result) {
    AssertInfo(result.capability() == FilterCapability::RandomMembership,
               "expected a random-membership FilterMap");
    TargetBitmap predicate_filtered;
    int64_t processed = 0;
    while (processed < static_cast<int64_t>(result.size())) {
        VectorPtr value;
        predicate->Eval(context, value);
        auto column = std::dynamic_pointer_cast<ColumnVector>(value);
        if (column == nullptr || !column->IsBitmap() || column->size() == 0) {
            return false;
        }
        const auto rows = static_cast<int64_t>(column->size());
        if (processed + rows > static_cast<int64_t>(result.size())) {
            return false;
        }
        TargetBitmap accepted(
            TargetBitmapView(column->GetRawData(), column->size()));
        TargetBitmapView valid(column->GetValidRawData(), column->size());
        accepted.inplace_and(valid, column->size());
        accepted.flip();
        predicate_filtered.append(accepted);
        processed += rows;
    }
    result.InplaceOr(TargetBitmapView(predicate_filtered));
    return true;
}

std::optional<FilterMap>
IntersectDenseWithAdaptivePredicate(const ExprPtr& predicate,
                                    EvalCtx& context,
                                    int64_t max_cardinality,
                                    FilterMap current) {
    AssertInfo(current.capability() == FilterCapability::RandomMembership,
               "expected a random-membership FilterMap");

    // Do not speculatively call a native producer and then re-run the same
    // predicate through Eval when it declines. Capability is a side-effect-
    // free commitment; unsupported predicates enter the ordinary batched path
    // directly and execute exactly once.
    if (!predicate->CanApplySparseFilter(
            context, /*has_sparse_input=*/false, max_cardinality)) {
        if (!ApplyPredicateToDenseResult(predicate, context, current)) {
            return std::nullopt;
        }
        return current;
    }

    auto next =
        predicate->TryApplySparseFilter(context, std::nullopt, max_cardinality);
    AssertInfo(next.has_value(),
               "Sparse capability preflight succeeded but predicate {} "
               "declined execution",
               predicate->name());
    if (next->size() != current.size()) {
        return std::nullopt;
    }
    if (next->capability() == FilterCapability::RandomMembership) {
        const auto next_dense = next->DenseOwner();
        if (next_dense == nullptr || next_dense->size() != current.size()) {
            return std::nullopt;
        }
        current.InplaceOr(*next_dense);
        return current;
    }

    auto accepted = std::make_shared<std::vector<int32_t>>();
    const auto next_ids = next->TryGetUnsetIdsView();
    if (!next_ids.has_value()) {
        return std::nullopt;
    }
    accepted->reserve(next_ids->size());
    for (const auto id : *next_ids) {
        if (id < 0 || static_cast<size_t>(id) >= current.size()) {
            return std::nullopt;
        }
        if (!current.test(static_cast<size_t>(id))) {
            accepted->push_back(id);
        }
    }
    return FilterMap::AdoptUnsetIds(current.size(),
                                    std::move(accepted),
                                    static_cast<size_t>(max_cardinality));
}

}  // namespace

DataType
PhyConjunctFilterExpr::ResolveType(const std::vector<DataType>& inputs) {
    AssertInfo(
        inputs.size() > 0,
        fmt::format(
            "Conjunct expressions expect at least one argument, received: {}",
            inputs.size()));

    for (const auto& type : inputs) {
        AssertInfo(
            type == DataType::BOOL,
            fmt::format("Conjunct expressions expect BOOLEAN, received: {}",
                        type));
    }
    return DataType::BOOL;
}

TargetBitmap
PhyConjunctFilterExpr::BuildActiveBitmap(const ColumnVectorPtr& vec) {
    // Rows that still need the following expressions.
    //
    // For AND: TRUE or NULL rows still need evaluation; only definite FALSE
    //   can stop (NULL AND FALSE = FALSE, so a NULL row can still change).
    //   With a null-rejecting consumer, NULL is already equivalent to FALSE
    //   downstream, so NULL rows are dropped too and only TRUE rows stay
    //   active — restoring the pre-3VL early exit for UNKNOWN-heavy batches.
    // For OR: FALSE or NULL rows still need evaluation; only definite TRUE
    //   can stop. A NULL row can still become TRUE (NULL OR TRUE = TRUE),
    //   so null-rejection does not shrink the active set for OR.
    const size_t size = vec->size();
    TargetBitmapView data(vec->GetRawData(), size);
    TargetBitmapView valid(vec->GetValidRawData(), size);
    if (is_and_) {
        if (null_rejecting_) {
            TargetBitmap active_rows(data);
            active_rows.inplace_and(valid, size);  // data & valid
            return active_rows;
        }
        TargetBitmap active_rows(valid);
        active_rows.inplace_sub(data, size);  // valid & ~data
        active_rows.flip();                   // data | ~valid
        return active_rows;
    }
    TargetBitmap active_rows(data);
    active_rows.inplace_and(valid, size);  // data & valid
    active_rows.flip();                    // ~data | ~valid
    return active_rows;
}

std::shared_ptr<std::vector<int32_t>>
PhyConjunctFilterExpr::TryGetNativeValidIds(EvalCtx& context) {
    auto max_cardinality = SPARSE_FILTER_RESULT_MAX_CARDINALITY.load();
    if (auto* exec_context = context.get_exec_context();
        exec_context != nullptr &&
        exec_context->get_query_context() != nullptr) {
        max_cardinality = exec_context->get_query_context()
                              ->get_search_info()
                              .SparseResultMaxCardinality(max_cardinality);
    }
    const auto result =
        TryApplySparseFilter(context, std::nullopt, max_cardinality);
    if (!result.has_value() ||
        result->capability() != FilterCapability::EnumerateOnly) {
        return nullptr;
    }
    const auto ids = result->TryGetUnsetIdsView();
    return ids.has_value() ? std::make_shared<std::vector<int32_t>>(
                                 ids->begin(), ids->end())
                           : nullptr;
}

bool
PhyConjunctFilterExpr::CanApplySparseFilter(EvalCtx& context,
                                            bool has_sparse_input,
                                            int64_t max_cardinality) {
    if (!is_and_ || inputs_.empty() || context.get_offset_input() != nullptr ||
        max_cardinality < 0) {
        return false;
    }

    size_t next_child = 0;
    if (!has_sparse_input) {
        if (!inputs_[0]->CanApplySparseFilter(
                context, /*has_sparse_input=*/false, max_cardinality)) {
            return false;
        }
        next_child = 1;
    }

    // A producer may choose Sparse or threshold-Dense at runtime. Dense can
    // always consume the next ordinary bitmap once, while the possible Sparse
    // branch requires every remaining child to commit to candidate filtering.
    // This is deliberately conservative and never changes predicate order.
    for (size_t i = next_child; i < inputs_.size(); ++i) {
        if (!inputs_[i]->CanApplySparseFilter(
                context, /*has_sparse_input=*/true, max_cardinality)) {
            return false;
        }
    }
    return true;
}

std::optional<FilterMap>
PhyConjunctFilterExpr::TryApplySparseFilter(EvalCtx& context,
                                            std::optional<FilterMap> input,
                                            int64_t max_cardinality) {
    // AND is safe to evaluate as a chain of definite-TRUE candidate sets:
    // every step only removes rows.  Do not use input_order_ here.  That is a
    // Dense executor scheduling detail whose runtime LIKE/optimization slots
    // need not be a complete Sparse-safe order.  `inputs_` is the actual
    // expression tree and retains SQL conjunction semantics regardless of
    // order.  OR/NOT/null-sensitive cases still use the Dense evaluator.
    if (input.has_value() &&
        input->capability() != FilterCapability::EnumerateOnly) {
        return std::nullopt;
    }

    if (!CanApplySparseFilter(context,
                              /*has_sparse_input=*/input.has_value(),
                              max_cardinality)) {
        return std::nullopt;
    }

    std::optional<FilterMap> result = std::move(input);
    size_t next_child = 0;
    if (!result.has_value()) {
        // Predicate order is an executor decision, not a request-by-request
        // producer hunt. Evaluate the first child once as the first stage of
        // the chain. If it cannot return an Adaptive result, let the
        // established Dense conjunction execute instead.
        result = inputs_[0]->TryApplySparseFilter(
            context, std::nullopt, max_cardinality);
        if (!result.has_value()) {
            return std::nullopt;
        }
        next_child = 1;
    }

    for (size_t i = next_child; i < inputs_.size(); ++i) {
        if (result->capability() == FilterCapability::EnumerateOnly &&
            result->size() == result->count()) {
            break;
        }
        if (result->capability() == FilterCapability::RandomMembership) {
            result = IntersectDenseWithAdaptivePredicate(
                inputs_[i], context, max_cardinality, std::move(*result));
            if (!result.has_value()) {
                return std::nullopt;
            }
            continue;
        }
        result = inputs_[i]->TryApplySparseFilter(
            context, std::move(result), max_cardinality);
        if (!result.has_value()) {
            return std::nullopt;
        }
    }
    return result;
}

void
PhyConjunctFilterExpr::SkipFollowingExprs(int start) {
    for (int i = start; i < input_order_.size(); ++i) {
        inputs_[input_order_[i]]->MoveCursor();
    }
}

void
PhyConjunctFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    tracer::AutoSpan span(
        "PhyConjunctFilterExpr::Eval", tracer::GetRootSpan(), true);
    span.GetSpan()->SetAttribute("is_and", is_and_);

    if (input_order_.empty()) {
        input_order_.resize(inputs_.size());
        for (size_t i = 0; i < inputs_.size(); i++) {
            input_order_[i] = i;
        }
    }

    auto has_input_offset = context.get_offset_input() != nullptr;
    if (!like_batch_initialized_ && is_and_ && like_indices_.size() > 1) {
        like_batch_initialized_ = true;
        // Collect LIKE expressions that can use ngram index at runtime.
        // The batch-ngram path evaluates whole batches, so it cannot be
        // used with an offset input; ngram_exprs then stays empty and the
        // else branch below erases the input_order_ slot reserved at
        // compile time. Deciding here — once, in every mode — keeps the
        // invariant that each input_order_ entry has a backing expression
        // in inputs_ before anything indexes inputs_ with it. (An
        // instance's offset mode is fixed for its lifetime, so no later
        // call could have used the batch path anyway.)
        std::vector<std::shared_ptr<PhyUnaryRangeFilterExpr>> ngram_exprs;
        if (!has_input_offset) {
            for (size_t idx : like_indices_) {
                auto unary_expr =
                    std::dynamic_pointer_cast<PhyUnaryRangeFilterExpr>(
                        inputs_[idx]);
                if (unary_expr && unary_expr->CanUseNgramIndex()) {
                    ngram_exprs.push_back(unary_expr);
                    batch_ngram_indices_.insert(idx);
                }
            }
        }

        // Create PhyLikeConjunctExpr and add to inputs_ if we have >= 2 eligible
        if (ngram_exprs.size() >= 2) {
            auto active_count = ngram_exprs[0]->GetActiveCount();
            auto like_conjunct = std::make_shared<PhyLikeConjunctExpr>(
                std::move(ngram_exprs),
                op_ctx_,
                active_count,
                context.get_query_config()->get_expr_batch_size());
            inputs_.push_back(like_conjunct);
        } else {
            batch_ngram_indices_.clear();
            // Remove the like_conjunct index from input_order_ since we're not
            // creating the batch expression. The index was reserved at compile
            // time but the PhyLikeConjunctExpr is not being created at runtime.
            auto original_size = inputs_.size();
            input_order_.erase(std::remove_if(input_order_.begin(),
                                              input_order_.end(),
                                              [original_size](size_t idx) {
                                                  return idx >= original_size;
                                              }),
                               input_order_.end());
        }
    }

    // Position of the last entry that will actually be evaluated: trailing
    // batch-ngram entries are skipped in the loop and must not force a
    // useless active-bitmap build after the real last input.
    size_t last_eval_pos = input_order_.size();
    for (size_t i = input_order_.size(); i-- > 0;) {
        if (batch_ngram_indices_.count(input_order_[i]) == 0) {
            last_eval_pos = i;
            break;
        }
    }

    bool has_result = false;
    for (size_t i = 0; i < input_order_.size(); ++i) {
        size_t idx = input_order_[i];

        // Skip expressions already executed via batch ngram
        if (batch_ngram_indices_.count(idx)) {
            continue;
        }

        VectorPtr input_result;
        inputs_[idx]->Eval(context, input_result);

        ColumnVectorPtr all_flat_result;
        if (!has_result) {
            result = input_result;
            has_result = true;
            all_flat_result = GetColumnVector(result);
        } else {
            auto input_flat_result = GetColumnVector(input_result);
            all_flat_result = GetColumnVector(result);
            if (is_and_) {
                common::ThreeValuedLogicOp::And(all_flat_result,
                                                input_flat_result);
            } else {
                common::ThreeValuedLogicOp::Or(all_flat_result,
                                               input_flat_result);
            }
        }

        // The last evaluated expression needs neither a skip decision nor a
        // bitmap input for a successor.
        if (i == last_eval_pos) {
            break;
        }

        // Build the active-row bitmap once per input: it decides the
        // batch-level early exit, and the same bitmap becomes the row-level
        // input of the next expression.
        auto active_rows = BuildActiveBitmap(all_flat_result);
        if (active_rows.none()) {
            SkipFollowingExprs(i + 1);
            ClearBitmapInput(context);
            return;
        }
        context.set_bitmap_input(std::move(active_rows));
    }
    ClearBitmapInput(context);
}

}  //namespace exec
}  // namespace milvus
