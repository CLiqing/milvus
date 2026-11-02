// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#pragma once

#include <stdexcept>

namespace milvus {
// Only a pre-search bitmap requirement permits baseline materialization.
// Callback/storage/cancellation failures retain their original exception.
class DeferredFilterUnsupported : public std::runtime_error {
 public:
    using std::runtime_error::runtime_error;
};
}  // namespace milvus
