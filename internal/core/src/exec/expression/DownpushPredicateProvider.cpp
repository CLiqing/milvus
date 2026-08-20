// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include "exec/expression/DownpushPredicateProvider.h"

#include <array>

namespace milvus::exec {

const DownpushPredicateProvider*
FindDownpushPredicateProvider(const expr::ColumnInfo& column) {
    static const std::array<const DownpushPredicateProvider*, 2> providers = {
        &NumericDownpushPredicateProvider(),
        &StringDownpushPredicateProvider(),
    };
    for (const auto* provider : providers) {
        if (provider->Supports(column)) {
            return provider;
        }
    }
    return nullptr;
}

}  // namespace milvus::exec
