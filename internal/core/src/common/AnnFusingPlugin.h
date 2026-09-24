// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.
#pragma once

#include <stdbool.h>
#include <stdint.h>

// Frozen native C ABI: V3 transmits original plan.proto operation codes.
// V1/V2 factories are intentionally not provided. Incompatible versions fail
// symbol resolution, never reinterpret an older request. No protobuf objects,
// STL ownership or exceptions cross the DSO boundary.
#ifdef __cplusplus
#define MILVUS_AF_ENUM(name) enum name : uint32_t
#else
#define MILVUS_AF_ENUM(name) enum name
#endif
MILVUS_AF_ENUM(MilvusAnnFusingDataType){
    kAnnFusingDataTypeUnknown = 0,
    kAnnFusingDataTypeBool = 1,
    kAnnFusingDataTypeInt8 = 2,
    kAnnFusingDataTypeInt16 = 3,
    kAnnFusingDataTypeInt32 = 4,
    kAnnFusingDataTypeInt64 = 5,
    kAnnFusingDataTypeFloat = 10,
    kAnnFusingDataTypeDouble = 11,
    kAnnFusingDataTypeString = 20,
    kAnnFusingDataTypeVarChar = 21,
    kAnnFusingDataTypeArray = 22,
    kAnnFusingDataTypeJSON = 23,
    kAnnFusingDataTypeGeometry = 24,
    kAnnFusingDataTypeText = 25,
    kAnnFusingDataTypeTimestamptz = 26,
};
MILVUS_AF_ENUM(MilvusAnnFusingAccessPath){
    kAnnFusingAccessPathUnknown = 0,
    kAnnFusingAccessPathRawData = 1,
    kAnnFusingAccessPathScalarIndex = 2,
    kAnnFusingAccessPathPkIndex = 3,
    kAnnFusingAccessPathTextIndex = 4,
    kAnnFusingAccessPathJsonStats = 5,
};
MILVUS_AF_ENUM(MilvusAnnFusingIndexType){
    kAnnFusingIndexTypeUnknown = 0,
    kAnnFusingIndexTypeNone = 1,
    kAnnFusingIndexTypeStlSort = 2,
    kAnnFusingIndexTypeBitmap = 3,
    kAnnFusingIndexTypeInverted = 4,
    kAnnFusingIndexTypeTrie = 5,
    kAnnFusingIndexTypeHybrid = 6,
};
#undef MILVUS_AF_ENUM

typedef struct MilvusAnnFusingRuleV3 {
    uint32_t struct_size;
    uint32_t data_type;  // MilvusAnnFusingDataType
    uint32_t expr_type;  // proto::plan::Expr::ExprCase
    uint32_t operation;  // proto::plan::OpType (comparison); Invalid if absent
    uint32_t arith_operation;  // proto::plan::ArithOpType; Unknown if absent
    uint32_t
        access_path;  // MilvusAnnFusingAccessPath, separate from index kind
    uint32_t
        index_type;  // MilvusAnnFusingIndexType; None when not a scalar index
} MilvusAnnFusingRuleV3;

typedef struct MilvusAnnFusingSampleV3 {
    uint32_t struct_size;
    double filter_ratio;            // whole user predicate estimate, not count
    double mandatory_filter_ratio;  // system visibility only; -1 if unknown
} MilvusAnnFusingSampleV3;

typedef struct MilvusAnnFusingPluginV3 {
    uint32_t struct_size;
    uint32_t abi_major;
    void* context;  // immutable, thread-safe callbacks; valid until destroy
    bool (*consider)(const void*, const MilvusAnnFusingRuleV3*);
    bool (*choose)(const void*, const MilvusAnnFusingSampleV3*);
    void (*destroy)(void*);
} MilvusAnnFusingPluginV3;

// No in-place ABI upgrade. Invalid ABI/config leaves the caller's out untouched.
typedef bool (*MilvusCreateAnnFusingPluginV3Fn)(const char* yaml_path,
                                                uint32_t output_size,
                                                MilvusAnnFusingPluginV3* out);
