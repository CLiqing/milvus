// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. Licensed under the Apache License,
// Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <type_traits>

#include "pb/plan.pb.h"

namespace milvus::exec {

// Pure scalar kernel shared by the traditional random-offset arithmetic
// expression and the ANN candidate evaluator. Integral Add/Sub/Mul use a wide
// intermediate so their result does not depend on signed-overflow UB.
template <proto::plan::ArithOpType Op, typename T>
bool
EvaluateNumericArithmeticLessThan(T value, T operand, T target) noexcept {
    if constexpr (std::is_integral_v<T>) {
        const auto wide_value = static_cast<__int128>(value);
        const auto wide_operand = static_cast<__int128>(operand);
        const auto wide_target = static_cast<__int128>(target);
        if constexpr (Op == proto::plan::ArithOpType::Add) {
            return wide_value + wide_operand < wide_target;
        } else if constexpr (Op == proto::plan::ArithOpType::Sub) {
            return wide_value - wide_operand < wide_target;
        } else if constexpr (Op == proto::plan::ArithOpType::Mul) {
            return wide_value * wide_operand < wide_target;
        } else if constexpr (Op == proto::plan::ArithOpType::Div) {
            return operand != 0 && wide_value / wide_operand < wide_target;
        } else if constexpr (Op == proto::plan::ArithOpType::Mod) {
            return operand != 0 && wide_value % wide_operand < wide_target;
        }
    } else {
        if constexpr (Op == proto::plan::ArithOpType::Add) {
            return value + operand < target;
        } else if constexpr (Op == proto::plan::ArithOpType::Sub) {
            return value - operand < target;
        } else if constexpr (Op == proto::plan::ArithOpType::Mul) {
            return value * operand < target;
        } else if constexpr (Op == proto::plan::ArithOpType::Div) {
            return operand != T{0} && value / operand < target;
        }
    }
    return false;
}

}  // namespace milvus::exec
