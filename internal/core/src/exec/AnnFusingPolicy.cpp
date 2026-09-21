// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#include "exec/AnnFusingPolicy.h"

#include <cstdlib>
#include <dlfcn.h>

#include "log/Log.h"

namespace milvus::exec {

const AnnFusingPolicy&
AnnFusingPolicy::Instance() {
    static const AnnFusingPolicy policy;
    return policy;
}

AnnFusingPolicy::AnnFusingPolicy() {
    const char* path = std::getenv("MILVUS_ANN_FUSING_PLUGIN");
    const char* config = std::getenv("MILVUS_ANN_FUSING_CONFIG");
    path = path ? path : "/milvus/lib/libAnnFusingPlugin.so";
    config = config ? config : "/milvus/configs/ann_fusing.yaml";
    library_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
        LOG_INFO("ann_fusing native policy unavailable; AUTO keeps baseline");
        return;
    }
    auto create = reinterpret_cast<MilvusCreateAnnFusingPluginV1Fn>(
        dlsym(library_, "MilvusCreateAnnFusingPluginV1"));
    MilvusAnnFusingPluginV1 candidate{};
    if (!create || !create(config, sizeof(candidate), &candidate) ||
        candidate.abi_major != 1 ||
        candidate.struct_size != sizeof(candidate) || !candidate.context ||
        !candidate.consider || !candidate.choose || !candidate.destroy) {
        // A conforming factory leaves output untouched on failure. Do not
        // dereference callbacks from an incompatible layout.
        dlclose(library_);
        library_ = nullptr;
        LOG_WARN("ann_fusing native policy rejected; AUTO keeps baseline");
        return;
    }
    api_ = candidate;
    LOG_INFO("ann_fusing native policy loaded abi=1 path={} config={}",
             path,
             config);
}

AnnFusingPolicy::~AnnFusingPolicy() {
    if (api_.context) {
        api_.destroy(api_.context);
    }
    if (library_) {
        dlclose(library_);
    }
}

bool
AnnFusingPolicy::Consider(const MilvusAnnFusingRuleV1& request) const {
    return available() && api_.consider(api_.context, &request);
}

bool
AnnFusingPolicy::Choose(const MilvusAnnFusingSampleV1& request) const {
    return available() && api_.choose(api_.context, &request);
}

}  // namespace milvus::exec
