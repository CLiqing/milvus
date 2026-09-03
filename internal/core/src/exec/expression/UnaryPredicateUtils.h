// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#pragma once

#include "pb/plan.pb.h"

namespace milvus::exec {

// Scalar comparison primitive shared by traditional random-offset filtering
// and prepared candidate evaluation. Batch callers remain responsible for
// gathering values, NULL validity and active-lane handling.
template <proto::plan::OpType Op, typename Left, typename Right>
inline bool
EvaluateUnaryPredicate(const Left& left, const Right& right) {
    static_assert(Op == proto::plan::OpType::GreaterThan ||
                  Op == proto::plan::OpType::GreaterEqual ||
                  Op == proto::plan::OpType::LessThan ||
                  Op == proto::plan::OpType::LessEqual ||
                  Op == proto::plan::OpType::Equal ||
                  Op == proto::plan::OpType::NotEqual);
    if constexpr (Op == proto::plan::OpType::GreaterThan) {
        return left > right;
    } else if constexpr (Op == proto::plan::OpType::GreaterEqual) {
        return left >= right;
    } else if constexpr (Op == proto::plan::OpType::LessThan) {
        return left < right;
    } else if constexpr (Op == proto::plan::OpType::LessEqual) {
        return left <= right;
    } else if constexpr (Op == proto::plan::OpType::Equal) {
        return left == right;
    } else if constexpr (Op == proto::plan::OpType::NotEqual) {
        return left != right;
    }
}

}  // namespace milvus::exec
