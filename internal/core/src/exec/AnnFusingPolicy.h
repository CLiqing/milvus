// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#pragma once

#include "common/AnnFusingPlugin.h"

namespace milvus::exec {

// Process-owned immutable native policy. Loading is lazy so explicit baseline
// and existing iterative requests never pay plugin initialization costs.
class AnnFusingPolicy final {
 public:
    static const AnnFusingPolicy&
    Instance();

    bool
    available() const {
        return api_.context != nullptr;
    }

    bool
    Consider(const MilvusAnnFusingRuleV1& request) const;

    bool
    Choose(const MilvusAnnFusingSampleV1& request) const;

 private:
    AnnFusingPolicy();
    ~AnnFusingPolicy();
    AnnFusingPolicy(const AnnFusingPolicy&) = delete;
    AnnFusingPolicy&
    operator=(const AnnFusingPolicy&) = delete;
    void* library_{nullptr};
    MilvusAnnFusingPluginV1 api_{};
};

}  // namespace milvus::exec
