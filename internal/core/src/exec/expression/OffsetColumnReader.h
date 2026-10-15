// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. See the NOTICE file for additional information.
#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "mmap/ChunkedColumnInterface.h"
#include "exec/FilterDiagnostics.h"

namespace milvus::exec {

// DEBUG ONLY: local budget experiments, not a fusing-policy/configuration API.
// Read once per process; ordinary deployments retain the 32 MiB default.
inline size_t
DebugOffsetReaderCacheBudgetBytes() {
    static const size_t bytes = [] {
        const auto* value =
            std::getenv("MILVUS_DEBUG_OFFSET_READER_CACHE_MIB");
        if (value == nullptr || std::string_view(value) == "32") {
            return size_t{32 * 1024 * 1024};
        }
        AssertInfo(std::string_view(value) == "64" ||
                       std::string_view(value) == "128",
                   "debug offset reader budget must be 32, 64 or 128 MiB");
        return (std::string_view(value) == "64" ? size_t{64} : size_t{128}) *
               1024 * 1024;
    }();
    return bytes;
}

// Worker-local raw-column reader, independent of expression operators. The
// column owns its metadata; each retained span owns its backing cache cell.
// Cache hits need no ownership copies or shared cache locks. The cache is
// associative: columns with more chunks do not alias a fixed number of slots.
template <typename PinnedValue>
class OffsetColumnReaderImpl {
 public:
    static constexpr size_t kMaxBytes = 32 * 1024 * 1024;
    static constexpr size_t kBatchRows = 64;

    OffsetColumnReaderImpl(const OffsetColumnReaderImpl&) = delete;
    OffsetColumnReaderImpl&
    operator=(const OffsetColumnReaderImpl&) = delete;

    explicit OffsetColumnReaderImpl(
        std::shared_ptr<const ChunkedColumnInterface> column,
        size_t budget_bytes = DebugOffsetReaderCacheBudgetBytes())
        : budget_bytes_(budget_bytes),
          column_(std::move(column)), row_count_(column_->NumRows()) {
        AssertInfo(budget_bytes_ > 0, "offset reader budget must be positive");
        // Bound lookup metadata independently of retained cell bytes. Large
        // layouts use the column's own mapping and the associative cache.
        const auto chunks = column_->num_chunks();
        if (chunks > 0 && static_cast<size_t>(chunks) <= kMaxLookupChunks) {
            if constexpr (std::is_same_v<PinnedValue, Chunk*>) {
                view_direct_.resize(chunks, nullptr);
            } else {
                direct_.resize(chunks, nullptr);
            }
            offsets_.reserve(chunks + 1);
            offsets_.push_back(0);
            for (int64_t i = 0; i < chunks; ++i) {
                const auto rows = column_->chunk_row_nums(i);
                AssertInfo(rows >= 0 && static_cast<size_t>(rows) <=
                                            row_count_ - offsets_.back(),
                           "invalid retained column chunk lengths");
                offsets_.push_back(offsets_.back() + rows);
            }
            AssertInfo(offsets_.back() == row_count_,
                       "retained column chunk lengths do not cover rows");
            // Uniform full chunks with a short final chunk are common, but
            // never assume that layout for irregular/empty chunks.
            uniform_rows_ = offsets_[1];
            for (size_t i = 1; i < static_cast<size_t>(chunks); ++i) {
                if (offsets_[i + 1] - offsets_[i] != uniform_rows_ &&
                    (i + 1 != static_cast<size_t>(chunks) ||
                     offsets_[i + 1] - offsets_[i] > uniform_rows_)) {
                    uniform_rows_ = 0;
                    break;
                }
            }
        }
    }

    PinnedValue
    Read(milvus::OpContext* context, int64_t row, int64_t& offset) {
        static_assert(std::is_same_v<PinnedValue, SpanBase>);
        auto resolved = Resolve(row);
        offset = resolved.second;
        ++reads_;
        const auto* entry = Find(resolved.first);
        return entry ? entry->pin.get() : Load(context, resolved.first);
    }

    void
    EndBatch() {
        transient_.reset();
        transient_chunk_.reset();
    }

    // Resolve/prefetch independent addresses before consuming their values.
    // Flush pending loads BEFORE a cache miss can evict a backing owner.
    // Scratch is bounded, including for large iterative batches. Values and
    // validity retain caller order; predicate semantics stay entirely in Expr.
    // Optional chunk IDs let the caller apply its existing SkipIndex decisions
    // without resolving every offset a second time. The reader knows no ops.
    template <typename T>
    bool
    Gather(milvus::OpContext* context,
           const int32_t* rows,
           size_t count,
           T* values,
           bool* valid,
           size_t* chunk_ids = nullptr) {
        static_assert(std::is_same_v<PinnedValue, SpanBase>);
        bool all_valid = true;
        std::array<const T*, kBatchRows> addresses;
        std::array<ValidityView, kBatchRows> validity;
        std::array<size_t, kBatchRows> local_offsets;
        for (size_t base = 0; base < count; base += kBatchRows) {
            const auto n = std::min(kBatchRows, count - base);
            size_t consumed = 0;
            auto consume = [&](size_t end) {
                for (; consumed < end; ++consumed) {
                    values[base + consumed] = *addresses[consumed];
                    const auto& mask = validity[consumed];
                    const bool is_valid =
                        !mask || mask[local_offsets[consumed]];
                    valid[base + consumed] = is_valid;
                    all_valid &= is_valid;
                }
            };
            for (size_t i = 0; i < n; ++i) {
                const auto [chunk_id, offset] = Resolve(rows[base + i]);
                if (chunk_ids != nullptr) {
                    chunk_ids[base + i] = chunk_id;
                }
                ++reads_;
                const auto* entry = Find(chunk_id);
                // No pin copies per lane: consume borrowed addresses before
                // Load can release any owner, including an oversized cell.
                if (!entry) {
                    consume(i);
                }
                const auto span =
                    entry ? entry->pin.get() : Load(context, chunk_id);
                AssertInfo(span.element_sizeof() == sizeof(T),
                           "offset reader scalar width mismatch");
                addresses[i] = static_cast<const T*>(span.data()) + offset;
                validity[i] = span.validity();
                local_offsets[i] = offset;
                __builtin_prefetch(addresses[i], 0, 1);
            }
            consume(n);
        }
        return all_valid;
    }

    // Variable-width values borrow their chunk backing. Consume a bounded
    // batch BEFORE a miss can evict any backing, preserving caller order.
    // The getter reads the existing storage representation; the consumer is
    // the existing Expr kernel, never a second predicate implementation.
    template <typename Getter, typename Consumer>
    void
    VisitViews(milvus::OpContext* context,
               const int32_t* rows,
               size_t count,
               Getter get_view,
               Consumer consume) {
        static_assert(std::is_same_v<PinnedValue, Chunk*>);
        using View = decltype(get_view(static_cast<Chunk*>(nullptr), size_t{}));
        std::array<View, kBatchRows> values;
        std::array<bool, kBatchRows> validity;
        std::array<size_t, kBatchRows> chunks;
        std::array<size_t, kBatchRows> offsets, pin_slots;
        std::array<Chunk*, kBatchRows> cached;
        auto* attribution = active_filter_diagnostics;
        if (attribution && !attribution->attribution) attribution = nullptr;
        for (size_t first = 0; first < count;) {
            FilterDiagnosticTimer resolve_timer(attribution ? &attribution->reader_resolve_ns : nullptr);
            if (transient_ && transient_chunk_ != Resolve(rows[first]).first) {
                EndBatch();
            }
            std::vector<int64_t> missing;
            missing.reserve(kBatchRows);
            size_t n = 0, estimated_bytes = 0;
            for (; first + n < count && n < kBatchRows; ++n) {
                const auto [id, offset] = Resolve(rows[first + n]);
                auto* hit = FindView(id);
                auto found = std::find(missing.begin(), missing.end(), id);
                const bool new_cell = !hit && found == missing.end();
                if (new_cell) {
                    const auto bytes = column_->OffsetChunkByteSize(id);
                    max_cell_bytes_ = std::max(max_cell_bytes_, bytes);
                    // Consume before releasing an oversized transient, or
                    // before the temporary miss cohort exceeds its budget.
                    if (n && (transient_ || bytes > budget_bytes_ - estimated_bytes)) {
                        break;
                    }
                    estimated_bytes = bytes >= budget_bytes_ - estimated_bytes
                                          ? budget_bytes_ : estimated_bytes + bytes;
                    missing.push_back(id);
                }
                pin_slots[n] = new_cell ? missing.size() - 1
                                       : found - missing.begin();
                chunks[n] = id;
                offsets[n] = offset;
                cached[n] = hit;
            }
            resolve_timer.Stop();
            // Crucially, do not evict cached hit owners until their borrowed
            // views have been consumed. One batch pin owns all missing cells.
            FilterDiagnosticTimer pin_timer(attribution ? &attribution->reader_pin_ns : nullptr);
            auto pinned = missing.empty()
                              ? std::pair{PinWrapper<std::vector<Chunk*>>{}, size_t{0}}
                              : column_->PinOffsetChunks(context, missing);
            pin_timer.Stop();
            reads_ += n;
            misses_ += missing.size();
            bulk_pin_calls_ += !missing.empty();
            peak_batch_bytes_ = std::max(peak_batch_bytes_, pinned.second);
            bool all_valid = true;
            FilterDiagnosticTimer views_timer(attribution ? &attribution->reader_views_ns : nullptr);
            for (size_t i = 0; i < n; ++i) {
                auto* chunk = cached[i] ? cached[i] : pinned.first.get()[pin_slots[i]];
                validity[i] = chunk->isValid(offsets[i]);
                all_valid &= validity[i];
                values[i] = validity[i] ? get_view(chunk, offsets[i]) : View{};
                if constexpr (std::is_same_v<View, std::string_view>) {
                    // Like numeric Gather, overlap independent value loads.
                    // Only the first cache line of a requested, valid value;
                    // no predicate knowledge, owning copy or full-string scan.
                    if (!values[i].empty()) {
                        __builtin_prefetch(values[i].data(), 0, 1);
                    }
                }
            }
            views_timer.Stop();
            consume(values.data(), validity.data(), chunks.data(), n, first,
                    all_valid);
            FilterDiagnosticTimer owners_timer(attribution ? &attribution->reader_owners_ns : nullptr);
            first += n;
            if (missing.empty()) {
                continue;
            }
            if (pinned.second > budget_bytes_) {
                ++oversized_pins_;
                if (missing.size() == 1) {
                    auto pin = std::move(pinned.first).template transform<Chunk*>(
                        [](std::vector<Chunk*>&& cells) {
                            return cells.front();
                        });
                    transient_.emplace(Entry{pinned.second, std::move(pin)});
                    transient_chunk_ = missing.front();
                }
                continue;
            }
            while (retained_bytes_ + pinned.second > budget_bytes_ ||
                   view_entries_ + missing.size() > max_entries_) {
                auto& old = view_cache_.front();
                for (auto id : old.chunks) {
                    if (id < view_direct_.size()) {
                        view_direct_[id] = nullptr;
                    } else {
                        view_lookup_.erase(id);
                    }
                    ++evictions_;
                }
                view_entries_ -= old.chunks.size();
                retained_bytes_ -= old.bytes;
                view_cache_.pop_front();
            }
            // Charge the entire shared owner ONCE. Charging each visible
            // field/cell separately would hide other cells retained by it.
            view_cache_.push_back({pinned.second, std::move(pinned.first),
                                  std::move(missing)});
            auto& entry = view_cache_.back();
            try {
                for (size_t i = 0; i < entry.chunks.size(); ++i) {
                    const auto id = entry.chunks[i];
                    auto* chunk = entry.pin.get()[i];
                    if (id < view_direct_.size()) {
                        view_direct_[id] = chunk;
                    } else {
                        // The bounded dense table already supplies this lookup.
                        // Only sparse layouts need a hash node; maintaining
                        // both duplicated per-miss allocation/erase work.
                        view_lookup_[id] = chunk;
                    }
                }
            } catch (...) {
                for (auto id : entry.chunks) {
                    if (id < view_direct_.size()) {
                        view_direct_[id] = nullptr;
                    } else {
                        view_lookup_.erase(id);
                    }
                }
                view_cache_.pop_back();
                throw;
            }
            view_entries_ += entry.chunks.size();
            retained_bytes_ += entry.bytes;
            peak_bytes_ = std::max(peak_bytes_, retained_bytes_);
            peak_pins_ = std::max(peak_pins_, view_entries_);
        }
        EndBatch();
    }

    size_t
    reads() const {
        return reads_;
    }
    size_t budget_bytes() const { return budget_bytes_; }
    size_t
    misses() const {
        return misses_;
    }
    size_t
    peak_bytes() const {
        return peak_bytes_;
    }
    size_t
    peak_pins() const {
        return peak_pins_;
    }
    size_t evictions() const { return evictions_; }
    size_t oversized_pins() const { return oversized_pins_; }
    size_t max_cell_bytes() const { return max_cell_bytes_; }
    size_t source_chunks() const { return column_->num_chunks(); }
    size_t bulk_pin_calls() const { return bulk_pin_calls_; }
    size_t peak_batch_bytes() const { return peak_batch_bytes_; }

 private:
    struct Entry {
        size_t bytes;
        PinWrapper<PinnedValue> pin;
    };
    struct ViewCohort {
        size_t bytes;
        PinWrapper<std::vector<Chunk*>> pin;
        std::vector<int64_t> chunks;
    };

    Chunk*
    FindView(size_t chunk) const {
        if (transient_ && transient_chunk_ == chunk) {
            return transient_->pin.get();
        }
        if (chunk < view_direct_.size()) {
            return view_direct_[chunk];
        }
        auto it = view_lookup_.find(chunk);
        return it == view_lookup_.end() ? nullptr : it->second;
    }
    // At most 1 MiB of dense pointer/offset metadata; fallback is sparse.
    static constexpr size_t kMaxLookupChunks =
        (1024 * 1024 - sizeof(size_t)) / (sizeof(Entry*) + sizeof(size_t));

    std::pair<size_t, size_t>
    Resolve(int64_t row) const {
        AssertInfo(row >= 0 && static_cast<size_t>(row) < row_count_,
                   "offset reader row outside retained column");
        if (uniform_rows_ != 0) {
            return {row / uniform_rows_, row % uniform_rows_};
        }
        if (!offsets_.empty()) {
            auto upper =
                std::upper_bound(offsets_.begin(), offsets_.end(), row);
            const size_t chunk = upper - offsets_.begin() - 1;
            return {chunk, row - offsets_[chunk]};
        }
        return column_->GetChunkIDByOffset(row);
    }

    const Entry*
    Find(size_t chunk) const {
        if (transient_ && transient_chunk_ == chunk) {
            return &*transient_;
        }
        if (chunk < direct_.size()) {
            return direct_[chunk];
        }
        auto it = entries_.find(chunk);
        return it == entries_.end() ? nullptr : &it->second;
    }

    void
    EvictFirst() {
        const auto chunk = fifo_.front();
        fifo_.pop_front();
        auto it = entries_.find(chunk);
        retained_bytes_ -= it->second.bytes;
        if (chunk < direct_.size()) {
            direct_[chunk] = nullptr;
        }
        entries_.erase(it);
        ++evictions_;
    }

    PinnedValue
    Load(milvus::OpContext* context, size_t chunk) {
        EndBatch();
        auto [pin, bytes] = [&] {
            if constexpr (std::is_same_v<PinnedValue, SpanBase>) {
                return column_->PinOffsetSpan(context, chunk);
            } else {
                return column_->PinOffsetChunk(context, chunk);
            }
        }();
        ++misses_;
        max_cell_bytes_ = std::max(max_cell_bytes_, bytes);
        if (bytes > budget_bytes_) {
            ++oversized_pins_;
            while (!fifo_.empty()) {
                EvictFirst();
            }
            // One required oversized cell is transient, never retained past
            // EndBatch. Prior borrowed addresses must already be consumed.
            transient_.emplace(Entry{bytes, std::move(pin)});
            transient_chunk_ = chunk;
            return transient_->pin.get();
        }
        while (retained_bytes_ + bytes > budget_bytes_ ||
               entries_.size() >= max_entries_) {
            EvictFirst();
        }
        // unordered_map references remain stable across rehash. FIFO eviction
        // only touches ownership on misses; a hit needs no LRU list updates.
        fifo_.push_back(chunk);
        try {
            auto [it, inserted] =
                entries_.emplace(chunk, Entry{bytes, std::move(pin)});
            if (chunk < direct_.size()) {
                direct_[chunk] = &it->second;
            }
            retained_bytes_ += bytes;
            peak_bytes_ = std::max(peak_bytes_, retained_bytes_);
            peak_pins_ = std::max(peak_pins_, entries_.size());
            return it->second.pin.get();
        } catch (...) {
            fifo_.pop_back();
            throw;
        }
    }

    const size_t budget_bytes_;
    // Also bound metadata for zero/tiny cells, independently of cell bytes.
    const size_t max_entries_ =
        std::max(size_t{1}, budget_bytes_ / (sizeof(Entry) + 64));
    std::shared_ptr<const ChunkedColumnInterface> column_;
    size_t row_count_;
    std::vector<Entry*> direct_;
    std::vector<Chunk*> view_direct_;
    std::unordered_map<size_t, Chunk*> view_lookup_;
    size_t view_entries_{0};
    std::deque<ViewCohort> view_cache_;
    std::vector<size_t> offsets_;
    size_t uniform_rows_{0};
    std::unordered_map<size_t, Entry> entries_;
    std::deque<size_t> fifo_;
    std::optional<Entry> transient_;
    std::optional<size_t> transient_chunk_;
    size_t retained_bytes_{0};
    size_t peak_bytes_{0}, peak_pins_{0}, reads_{0}, misses_{0};
    size_t evictions_{0}, oversized_pins_{0}, max_cell_bytes_{0};
    size_t bulk_pin_calls_{0}, peak_batch_bytes_{0};
};

// Both physical representations share the same metadata and bounded owner
// cache. Only reading a value differs; no expression/operator is stored here.
using OffsetColumnReader = OffsetColumnReaderImpl<SpanBase>;
using OffsetChunkReader = OffsetColumnReaderImpl<Chunk*>;

}  // namespace milvus::exec
