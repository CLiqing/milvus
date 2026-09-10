// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file distributed
// with this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0.

#pragma once

#include <mutex>

#include "common/FilterMap.h"
#include "knowhere/bitsetview.h"

namespace milvus::query {

// The only Milvus-to-Knowhere representation adapter. The immutable map is
// shared across NQ tasks; each reader owns its cursor. Legacy consumers request
// one separately owned Dense snapshot, without invalidating any active cursor.
inline knowhere::BitsetView
MakeFilterMapView(const FilterMap& map) {
    struct Owner {
        explicit Owner(const FilterMap& value) : map(value) {
        }
        FilterMap map;
        mutable std::once_flag dense_once;
        mutable std::optional<TargetBitmap> dense;

        const uint8_t*
        Dense() const {
            std::call_once(dense_once, [this] {
                auto snapshot = map;
                dense.emplace(std::move(snapshot).TakeDense());
            });
            return reinterpret_cast<const uint8_t*>(dense->data());
        }
    };
    auto owner = std::make_shared<Owner>(map);
    const auto* context = owner.get();
    return knowhere::BitsetView::FromEnumerable(
        std::move(owner),
        context,
        map.size(),
        map.count(),
        [](const void* ptr, size_t& position, int32_t* ids, size_t capacity) {
            const auto& map = static_cast<const Owner*>(ptr)->map;
            FilterMapCursor cursor;
            cursor.position = position;
            const auto count =
                map.ReadUnsetBatch(cursor, std::span<int32_t>(ids, capacity));
            position = cursor.position;
            return count;
        },
        [](const void* ptr) {
            return static_cast<const Owner*>(ptr)->Dense();
        });
}

}  // namespace milvus::query
