// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. See the NOTICE file for additional information.
#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <optional>

#include "mmap/ChunkedColumnInterface.h"

namespace milvus::exec {

// Worker-local raw-column reader, independent of expression operators. The
// column owns its metadata; each retained span owns its backing cache cell.
// No per-row shared_ptr copies, segment snapshot loads or shared cache lock.
class OffsetColumnReader {
 public:
    static constexpr size_t kMaxPins = 128;
    static constexpr size_t kMaxBytes = 32 * 1024 * 1024;

    explicit OffsetColumnReader(
        std::shared_ptr<const ChunkedColumnInterface> column)
        : column_(std::move(column)) {
    }

    SpanBase
    Read(milvus::OpContext* context, int64_t row, int64_t& offset) {
        AssertInfo(row >= 0 && row < column_->NumRows(),
                   "offset reader row outside retained column");
        auto resolved = column_->GetChunkIDByOffset(row);
        offset = resolved.second;
        const auto chunk_id = static_cast<int64_t>(resolved.first);
        ++reads_;
        // Direct-mapped, fixed-capacity cache: bounded even for very large
        // columns. Collisions affect performance only, never row semantics.
        auto& entry = entries_[resolved.first % kMaxPins];
        if (entry && entry->chunk_id == chunk_id) {
            return entry->pin.get();
        }
        if (entry) {
            retained_bytes_ -= entry->bytes;
            --retained_pins_;
            entry.reset();
        }
        transient_.reset();
        auto [pin, bytes] = column_->PinOffsetSpan(context, chunk_id);
        ++misses_;
        if (bytes > kMaxBytes) {
            ClearPins();
            // An individual oversized cell may be required for this read,
            // but must not be retained across batches. It stays alive until
            // the caller consumes the returned span / the next Read call.
            transient_.emplace(std::move(pin));
            return transient_->get();
        }
        if (retained_bytes_ + bytes > kMaxBytes) {
            ClearPins();
        }
        entry.emplace(Entry{chunk_id, bytes, std::move(pin)});
        retained_bytes_ += bytes;
        ++retained_pins_;
        peak_bytes_ = std::max(peak_bytes_, retained_bytes_);
        peak_pins_ = std::max(peak_pins_, retained_pins_);
        return entry->pin.get();
    }

    void
    EndBatch() {
        transient_.reset();
    }

    size_t reads() const { return reads_; }
    size_t misses() const { return misses_; }
    size_t peak_bytes() const { return peak_bytes_; }
    size_t peak_pins() const { return peak_pins_; }

 private:
    void ClearPins() {
        for (auto& entry : entries_) {
            entry.reset();
        }
        retained_bytes_ = 0;
        retained_pins_ = 0;
    }
    struct Entry {
        int64_t chunk_id;
        size_t bytes;
        PinWrapper<SpanBase> pin;
    };
    std::shared_ptr<const ChunkedColumnInterface> column_;
    std::array<std::optional<Entry>, kMaxPins> entries_;
    std::optional<PinWrapper<SpanBase>> transient_;
    size_t retained_bytes_{0}, retained_pins_{0};
    size_t peak_bytes_{0}, peak_pins_{0}, reads_{0}, misses_{0};
};

}  // namespace milvus::exec
