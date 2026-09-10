// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace milvus {

// Producer policy only. Consumers use FilterMap capabilities independently of
// this switch. A QueryConfig snapshots these startup defaults once per query.
struct FilterMapConfig {
    bool enabled = false;
    int64_t min_rows = 50000;
    double max_ratio = 0.004;

    bool
    IsValid() const;

    // No absolute V cap: T = floor(N * max_ratio), independently per segment.
    // nullopt means keep the existing Dense producer, not an empty Sparse map.
    std::optional<size_t>
    ExceptionCap(size_t universe) const;
};

FilterMapConfig
GetDefaultFilterMapConfig();

// Reject invalid startup settings without changing any of the old settings.
bool
SetDefaultFilterMapConfig(const FilterMapConfig& config);

}  // namespace milvus
