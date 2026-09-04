// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "common/Types.h"

namespace milvus {

// Describes the efficient access pattern offered by a FilterMap. Consumers
// must depend on this contract rather than its physical representation.
enum class FilterMapCapability : uint8_t {
    RandomMembership,
    EnumerateOnly,
};

struct FilterMapCursor {
    size_t position = 0;
};

// A bitmap-compatible filter whose physical storage can be either a dense
// TargetBitmap or a sparse collection of bits that differ from a default bit.
// set/reset/test retain ordinary bitmap semantics; in particular, set(id)
// always stores logical one. Sparse storage promotes itself to Dense when the
// exception cap would be exceeded.
class FilterMap {
 public:
    FilterMap() = default;
    FilterMap(const FilterMap&) = default;
    FilterMap&
    operator=(const FilterMap&) = default;
    FilterMap(FilterMap&&) noexcept = default;
    FilterMap&
    operator=(FilterMap&&) noexcept = default;
    ~FilterMap() = default;

    static FilterMap
    FromDense(std::shared_ptr<TargetBitmap> dense);

    static FilterMap
    Adaptive(size_t universe, bool default_bit, size_t exception_cap);

    bool
    IsInitialized() const noexcept;

    size_t
    size() const;

    size_t
    count() const;

    bool
    test(size_t id) const;

    void
    set(size_t id);

    void
    reset(size_t id);

    void
    set(size_t id, bool value);

    FilterMapCapability
    capability() const;

    // Assigns one logical batch at offset. source is caller-owned scratch and
    // may be modified when the final representation is Dense. With validity,
    // logical bits are source AND validity; invert is applied afterwards.
    // Producer batches must cover disjoint ranges.
    void
    AssignBatch(TargetBitmapView source,
                const TargetBitmapView* validity,
                size_t offset,
                bool invert);

    // Enumerates logical zero bits without exposing the backing storage.
    size_t
    ReadUnsetBatch(FilterMapCursor& cursor, std::span<int32_t> output) const;

    // One-way, idempotent compatibility boundary. A shared map detaches before
    // returning a mutable Dense bitmap, so mutations cannot affect its source.
    TargetBitmap&
    EnsureDense();

 private:
    struct Storage;

    explicit FilterMap(std::shared_ptr<Storage> storage);

    const Storage&
    GetStorage() const;

    Storage&
    GetMutableStorage();

    bool
    IsDense() const;

    TargetBitmap&
    GetMutableDense();

    void
    PromoteToDense();

    void
    WriteDenseBatch(TargetBitmapView source,
                    const TargetBitmapView* validity,
                    size_t offset,
                    bool invert);

    void
    CheckId(size_t id) const;

    std::shared_ptr<Storage> storage_;
};

}  // namespace milvus
