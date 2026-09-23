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

// V2 uses compact metadata codes. V1 above remains frozen for old producers;
// consumers MUST resolve the versioned V2 symbol before passing a V2 buffer.
// Fixed-width storage in requests is independent of C compiler enum sizing.
// These are policy facts, not an executable expression/IR.
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
MILVUS_AF_ENUM(MilvusAnnFusingOperation){
    kAnnFusingOperationUnknown = 0,      kAnnFusingOperationRange = 1,
    kAnnFusingOperationEqual = 2,        kAnnFusingOperationNotEqual = 3,
    kAnnFusingOperationPrefixMatch = 4,  kAnnFusingOperationPostfixMatch = 5,
    kAnnFusingOperationMatch = 6,        kAnnFusingOperationInnerMatch = 7,
    kAnnFusingOperationRegexMatch = 8,   kAnnFusingOperationAdd = 16,
    kAnnFusingOperationSub = 17,         kAnnFusingOperationMul = 18,
    kAnnFusingOperationDiv = 19,         kAnnFusingOperationMod = 20,
    kAnnFusingOperationArrayLength = 21, kAnnFusingOperationBitAnd = 22,
    kAnnFusingOperationBitOr = 23,       kAnnFusingOperationBitXor = 24,
    kAnnFusingOperationShl = 25,         kAnnFusingOperationShr = 26,
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

typedef struct MilvusAnnFusingRuleV2 {
    uint32_t struct_size;
    uint32_t data_type;  // MilvusAnnFusingDataType
    uint32_t operation;  // MilvusAnnFusingOperation
    uint32_t
        access_path;  // MilvusAnnFusingAccessPath, separate from index kind
    uint32_t
        index_type;  // MilvusAnnFusingIndexType; None when not a scalar index
} MilvusAnnFusingRuleV2;

typedef struct MilvusAnnFusingSampleV2 {
    uint32_t struct_size;
    double filter_ratio;            // whole user predicate estimate, not count
    double mandatory_filter_ratio;  // system visibility only; -1 if unknown
} MilvusAnnFusingSampleV2;

typedef struct MilvusAnnFusingPluginV2 {
    uint32_t struct_size;
    uint32_t abi_major;
    void* context;  // immutable, thread-safe callbacks; same lifetime as V1
    bool (*consider)(const void*, const MilvusAnnFusingRuleV2*);
    bool (*choose)(const void*, const MilvusAnnFusingSampleV2*);
    void (*destroy)(void*);
} MilvusAnnFusingPluginV2;

// No in-place V1 upgrade. Invalid ABI/config leaves the caller's out untouched.
typedef bool (*MilvusCreateAnnFusingPluginV2Fn)(const char* yaml_path,
                                                uint32_t output_size,
                                                MilvusAnnFusingPluginV2* out);
