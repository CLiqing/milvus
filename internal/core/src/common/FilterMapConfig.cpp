// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#include "common/FilterMapConfig.h"

#include <cmath>
#include <mutex>
#include <stdexcept>

namespace milvus {
namespace {
std::mutex config_mutex;
FilterMapConfig default_config;
}  // namespace

bool
FilterMapConfig::IsValid() const {
    return min_rows >= 0 && std::isfinite(max_ratio) && max_ratio >= 0 &&
           max_ratio <= 1;
}

std::optional<size_t>
FilterMapConfig::ExceptionCap(size_t universe) const {
    if (!IsValid()) {
        throw std::invalid_argument("Invalid FilterMap producer policy");
    }
    if (!enabled || universe < static_cast<size_t>(min_rows)) {
        return std::nullopt;
    }
    return static_cast<size_t>(
        std::floor(static_cast<long double>(universe) * max_ratio));
}

FilterMapConfig
GetDefaultFilterMapConfig() {
    std::lock_guard lock(config_mutex);
    return default_config;
}

bool
SetDefaultFilterMapConfig(const FilterMapConfig& config) {
    if (!config.IsValid()) {
        return false;
    }
    std::lock_guard lock(config_mutex);
    default_config = config;
    return true;
}
}  // namespace milvus
