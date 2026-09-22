// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#pragma once

#include <stdbool.h>
#include <stdint.h>

// Frozen native C ABI. V1 sizes/layouts never grow: incompatible evolution
// requires a new versioned entry symbol. Strings/requests are borrowed only
// for the duration of a call. No STL ownership or exceptions cross the DSO.
typedef struct MilvusAnnFusingRuleV1 {
    uint32_t struct_size;
    const char* data_type;   // schema spelling, e.g. Int64
    const char* operation;   // range, Mod, ...; metadata, not executable IR
    const char* index_type;  // actual loaded scalar index, or NONE
} MilvusAnnFusingRuleV1;

typedef struct MilvusAnnFusingSampleV1 {
    uint32_t struct_size;
    double filter_ratio;            // estimate, never an exact row count
    double mandatory_filter_ratio;  // -1 if not known; only necessary terms
} MilvusAnnFusingSampleV1;

typedef struct MilvusAnnFusingPluginV1 {
    uint32_t struct_size;
    uint32_t abi_major;
    void* context;  // immutable after creation; both decisions are thread safe
    bool (*consider)(const void*, const MilvusAnnFusingRuleV1*);
    bool (*choose)(const void*, const MilvusAnnFusingSampleV1*);
    void (*destroy)(void*);
} MilvusAnnFusingPluginV1;

// Factory symbol: MilvusCreateAnnFusingPluginV1. Caller owns the output buffer;
// size must equal sizeof(V1). Returns false on invalid config/ABI, leaving out
// untouched. Destroy context before unloading the DSO. All callbacks must
// contain their own exceptions, including configuration errors at creation.
typedef bool (*MilvusCreateAnnFusingPluginV1Fn)(const char* yaml_path,
                                                uint32_t output_size,
                                                MilvusAnnFusingPluginV1* out);
