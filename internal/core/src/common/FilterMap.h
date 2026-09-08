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

class FilterMap;

struct FilterMapCursor {
    size_t position = 0;

 private:
    friend class FilterMap;
    const FilterMap* owner_ = nullptr;
    uint64_t revision_ = 0;
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
    operator=(const FilterMap&);
    FilterMap(FilterMap&&) noexcept;
    FilterMap&
    operator=(FilterMap&&) noexcept;
    ~FilterMap() = default;

    // The shared owner must not be mutated externally while retained by a map.
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

    // Logical bitwise complement, not SQL NOT. O(1), including shared copies:
    // only map-local polarity changes. Ends producer construction.
    void
    flip();

    FilterMapCapability
    capability() const;

    // Assigns one bitmap-producing kernel batch at offset. source is
    // caller-owned scratch and may be modified when the final representation
    // is Dense. With validity, logical bits are source AND validity; invert is
    // applied afterwards. Initial construction only: ranges must be increasing
    // and disjoint. Cannot mix with AppendUniqueBits, or follow set/reset or
    // EnsureDense. The producer checks final coverage/completion.
    void
    AssignBitmapBatch(TargetBitmapView source,
                      const TargetBitmapView* validity,
                      size_t offset,
                      bool invert);

    // Appends a producer-owned batch of unique bit positions without bitmap
    // materialization or per-ID duplicate lookup. IDs may be unordered, but
    // must be unique across this and all previous producer batches. This is a
    // trusted initial-construction boundary, not general set/reset. Cannot mix
    // with AssignBitmapBatch, or follow set/reset or EnsureDense.
    void
    AppendUniqueBits(std::span<const int32_t> ids, bool value);

    // Enumerates logical zero bits without exposing the backing storage.
    // Order is unspecified. Writes, assignment, flip and EnsureDense invalidate
    // cursors, even if logical bits are unchanged. Invalid cursors are rejected;
    // start a new enumeration with a new cursor after conversion. A cursor must
    // not outlive its map. Default-zero Sparse requires EnsureDense.
    size_t
    ReadUnsetBatch(FilterMapCursor& cursor, std::span<int32_t> output) const;

    // One-way, idempotent compatibility boundary; ends producer construction.
    // Read-only borrow, invalidated by mutation/destruction of this map. Writes
    // must go through FilterMap to preserve COW and its fixed universe.
    const TargetBitmap&
    EnsureDense();

    // Transfers ownership to a legacy mutable consumer. Consumes this map;
    // shared snapshots are detached, uniquely owned Dense bytes are moved.
    TargetBitmap
    TakeDense() &&;

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
    bool inverted_ = false;
    bool construction_finished_ = false;
    uint64_t revision_ = 0;
};

}  // namespace milvus
