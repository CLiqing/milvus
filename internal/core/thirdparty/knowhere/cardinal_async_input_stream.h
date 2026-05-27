#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "filemanager/AsyncInputStream.h"
#include "filemanager/InputStream.h"

namespace cardinalv2::storage {

inline bool
IsEnvEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return false;
    }
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text == "1" || text == "true" || text == "on" || text == "yes";
}

inline uint64_t
GetUnsignedEnv(const char* name, uint64_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    try {
        size_t parsed = 0;
        auto result = std::stoull(value, &parsed, 10);
        return parsed == std::string(value).size() ? result : default_value;
    } catch (...) {
        return default_value;
    }
}

inline void
UpdateMax(std::atomic<uint64_t>& target, uint64_t value) {
    auto current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current,
                                         value,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

inline uint64_t
GetCardinalAsyncMaxInflight() {
    return std::max<uint64_t>(
        1, GetUnsignedEnv("MILVUS_S3_ASYNC_MAX_INFLIGHT", 100));
}

inline uint64_t
NowMicros() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline std::shared_ptr<milvus::AsyncInputStream>
GetAsyncInputStream(const std::shared_ptr<milvus::InputStream>& stream) {
    auto async_stream =
        std::dynamic_pointer_cast<milvus::AsyncInputStream>(stream);
    if (async_stream == nullptr || !async_stream->SupportsAsyncReadAt()) {
        return nullptr;
    }
    constexpr size_t kCardinalFetchLimiterId = 3;
    auto max_inflight = GetCardinalAsyncMaxInflight();
    async_stream->ConfigureAsyncReadAtLimiter(kCardinalFetchLimiterId,
                                              max_inflight);
    return async_stream;
}

struct CardinalAsyncReadStats {
    std::atomic<uint64_t> stream_batches{0};
    std::atomic<uint64_t> chunk_batches{0};
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> failed{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> last_max_inflight{0};
    std::atomic<uint64_t> last_reported_done{0};
    std::atomic<uint64_t> stream_batch_reads_sum{0};
    std::atomic<uint64_t> stream_batch_reads_max{0};
    std::atomic<uint64_t> stream_batch_bytes_sum{0};
    std::atomic<uint64_t> chunk_batch_reads_sum{0};
    std::atomic<uint64_t> chunk_batch_reads_max{0};
    std::atomic<uint64_t> chunk_batch_bytes_sum{0};
    std::atomic<uint64_t> batch_reads_le1{0};
    std::atomic<uint64_t> batch_reads_le2{0};
    std::atomic<uint64_t> batch_reads_le4{0};
    std::atomic<uint64_t> batch_reads_le8{0};
    std::atomic<uint64_t> batch_reads_le16{0};
    std::atomic<uint64_t> batch_reads_le32{0};
    std::atomic<uint64_t> batch_reads_le64{0};
    std::atomic<uint64_t> batch_reads_gt64{0};
    std::atomic<uint64_t> sync_fetch_batches{0};
    std::atomic<uint64_t> sync_fetch_tasks_sum{0};
    std::atomic<uint64_t> sync_fetch_tasks_max{0};
    std::atomic<uint64_t> sync_fetch_bytes_sum{0};
    std::atomic<uint64_t> sync_fetch_wait_us_sum{0};
    std::atomic<uint64_t> sync_fetch_wait_us_max{0};
    std::atomic<uint64_t> async_submit_us_sum{0};
    std::atomic<uint64_t> async_submit_us_max{0};
    std::atomic<uint64_t> async_wait_us_sum{0};
    std::atomic<uint64_t> async_wait_us_max{0};
    std::atomic<uint64_t> async_batch_total_us_sum{0};
    std::atomic<uint64_t> async_batch_total_us_max{0};
    std::atomic<uint64_t> inflight_after_submit_sum{0};
    std::atomic<uint64_t> inflight_after_submit_max{0};
    std::atomic<uint64_t> trace_batch_id{0};
    std::atomic<uint64_t> window_last_us{0};
    std::atomic<uint64_t> window_last_stream_batches{0};
    std::atomic<uint64_t> window_last_chunk_batches{0};
    std::atomic<uint64_t> window_last_submitted{0};
    std::atomic<uint64_t> window_last_completed{0};
    std::atomic<uint64_t> window_last_failed{0};
    std::atomic<uint64_t> window_last_bytes{0};
    std::atomic<uint64_t> window_last_stream_reads{0};
    std::atomic<uint64_t> window_last_chunk_reads{0};
    std::atomic<uint64_t> window_last_wait_us{0};

    ~CardinalAsyncReadStats() {
        if (IsEnvEnabled("MILVUS_CARDINAL_ASYNC_STATS")) {
            Print("process_exit");
        }
    }

    void
    Print(const char* reason) const {
        std::cerr << "[MILVUS_CARDINAL_ASYNC_STATS]"
                  << " reason=" << reason
                  << " stream_batches="
                  << stream_batches.load(std::memory_order_relaxed)
                  << " chunk_batches="
                  << chunk_batches.load(std::memory_order_relaxed)
                  << " submitted=" << submitted.load(std::memory_order_relaxed)
                  << " completed=" << completed.load(std::memory_order_relaxed)
                  << " failed=" << failed.load(std::memory_order_relaxed)
                  << " bytes=" << bytes.load(std::memory_order_relaxed)
                  << " max_inflight="
                  << last_max_inflight.load(std::memory_order_relaxed)
                  << " stream_batch_reads_sum="
                  << stream_batch_reads_sum.load(std::memory_order_relaxed)
                  << " stream_batch_reads_max="
                  << stream_batch_reads_max.load(std::memory_order_relaxed)
                  << " stream_batch_bytes_sum="
                  << stream_batch_bytes_sum.load(std::memory_order_relaxed)
                  << " chunk_batch_reads_sum="
                  << chunk_batch_reads_sum.load(std::memory_order_relaxed)
                  << " chunk_batch_reads_max="
                  << chunk_batch_reads_max.load(std::memory_order_relaxed)
                  << " chunk_batch_bytes_sum="
                  << chunk_batch_bytes_sum.load(std::memory_order_relaxed)
                  << " batch_reads_le1="
                  << batch_reads_le1.load(std::memory_order_relaxed)
                  << " batch_reads_le2="
                  << batch_reads_le2.load(std::memory_order_relaxed)
                  << " batch_reads_le4="
                  << batch_reads_le4.load(std::memory_order_relaxed)
                  << " batch_reads_le8="
                  << batch_reads_le8.load(std::memory_order_relaxed)
                  << " batch_reads_le16="
                  << batch_reads_le16.load(std::memory_order_relaxed)
                  << " batch_reads_le32="
                  << batch_reads_le32.load(std::memory_order_relaxed)
                  << " batch_reads_le64="
                  << batch_reads_le64.load(std::memory_order_relaxed)
                  << " batch_reads_gt64="
                  << batch_reads_gt64.load(std::memory_order_relaxed)
                  << " sync_fetch_batches="
                  << sync_fetch_batches.load(std::memory_order_relaxed)
                  << " sync_fetch_tasks_sum="
                  << sync_fetch_tasks_sum.load(std::memory_order_relaxed)
                  << " sync_fetch_tasks_max="
                  << sync_fetch_tasks_max.load(std::memory_order_relaxed)
                  << " sync_fetch_bytes_sum="
                  << sync_fetch_bytes_sum.load(std::memory_order_relaxed)
                  << " sync_fetch_wait_us_sum="
                  << sync_fetch_wait_us_sum.load(std::memory_order_relaxed)
                  << " sync_fetch_wait_us_max="
                  << sync_fetch_wait_us_max.load(std::memory_order_relaxed)
                  << " async_submit_us_sum="
                  << async_submit_us_sum.load(std::memory_order_relaxed)
                  << " async_submit_us_max="
                  << async_submit_us_max.load(std::memory_order_relaxed)
                  << " async_wait_us_sum="
                  << async_wait_us_sum.load(std::memory_order_relaxed)
                  << " async_wait_us_max="
                  << async_wait_us_max.load(std::memory_order_relaxed)
                  << " async_batch_total_us_sum="
                  << async_batch_total_us_sum.load(std::memory_order_relaxed)
                  << " async_batch_total_us_max="
                  << async_batch_total_us_max.load(std::memory_order_relaxed)
                  << " inflight_after_submit_sum="
                  << inflight_after_submit_sum.load(std::memory_order_relaxed)
                  << " inflight_after_submit_max="
                  << inflight_after_submit_max.load(std::memory_order_relaxed)
                  << std::endl;
    }

    void
    ObserveBatch(bool is_chunk_batch, uint64_t reads, uint64_t batch_bytes) {
        if (is_chunk_batch) {
            chunk_batch_reads_sum.fetch_add(reads, std::memory_order_relaxed);
            chunk_batch_bytes_sum.fetch_add(batch_bytes,
                                            std::memory_order_relaxed);
            UpdateMax(chunk_batch_reads_max, reads);
        } else {
            stream_batch_reads_sum.fetch_add(reads, std::memory_order_relaxed);
            stream_batch_bytes_sum.fetch_add(batch_bytes,
                                             std::memory_order_relaxed);
            UpdateMax(stream_batch_reads_max, reads);
        }
        if (reads <= 1) {
            batch_reads_le1.fetch_add(1, std::memory_order_relaxed);
        } else if (reads <= 2) {
            batch_reads_le2.fetch_add(1, std::memory_order_relaxed);
        } else if (reads <= 4) {
            batch_reads_le4.fetch_add(1, std::memory_order_relaxed);
        } else if (reads <= 8) {
            batch_reads_le8.fetch_add(1, std::memory_order_relaxed);
        } else if (reads <= 16) {
            batch_reads_le16.fetch_add(1, std::memory_order_relaxed);
        } else if (reads <= 32) {
            batch_reads_le32.fetch_add(1, std::memory_order_relaxed);
        } else if (reads <= 64) {
            batch_reads_le64.fetch_add(1, std::memory_order_relaxed);
        } else {
            batch_reads_gt64.fetch_add(1, std::memory_order_relaxed);
        }
        MaybePrintWindow("observe_batch");
    }

    uint64_t
    NextTraceBatchId() {
        return trace_batch_id.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    void
    ObserveAsyncTiming(uint64_t submit_us,
                       uint64_t wait_us,
                       uint64_t total_us,
                       uint64_t inflight_after_submit) {
        async_submit_us_sum.fetch_add(submit_us, std::memory_order_relaxed);
        async_wait_us_sum.fetch_add(wait_us, std::memory_order_relaxed);
        async_batch_total_us_sum.fetch_add(total_us, std::memory_order_relaxed);
        inflight_after_submit_sum.fetch_add(inflight_after_submit,
                                            std::memory_order_relaxed);
        UpdateMax(async_submit_us_max, submit_us);
        UpdateMax(async_wait_us_max, wait_us);
        UpdateMax(async_batch_total_us_max, total_us);
        UpdateMax(inflight_after_submit_max, inflight_after_submit);
        MaybePrintWindow("observe_timing");
    }

    void
    ObserveSyncFetchBatch(uint64_t tasks,
                          uint64_t batch_bytes,
                          uint64_t wait_us) {
        sync_fetch_batches.fetch_add(1, std::memory_order_relaxed);
        sync_fetch_tasks_sum.fetch_add(tasks, std::memory_order_relaxed);
        sync_fetch_bytes_sum.fetch_add(batch_bytes, std::memory_order_relaxed);
        sync_fetch_wait_us_sum.fetch_add(wait_us, std::memory_order_relaxed);
        UpdateMax(sync_fetch_tasks_max, tasks);
        UpdateMax(sync_fetch_wait_us_max, wait_us);
        MaybePrintWindow("observe_sync_fetch");
    }

    void
    TraceBatch(const char* path,
               uint64_t batch_id,
               uint64_t reads,
               uint64_t bytes,
               uint64_t submit_us,
               uint64_t wait_us,
               uint64_t total_us,
               uint64_t inflight_after_submit) {
        if (!IsEnvEnabled("MILVUS_CARDINAL_ASYNC_TRACE_BATCHES")) {
            return;
        }
        std::cerr << "[MILVUS_CARDINAL_BATCH_TRACE]"
                  << " path=" << path
                  << " batch_id=" << batch_id
                  << " reads=" << reads
                  << " bytes=" << bytes
                  << " submit_us=" << submit_us
                  << " wait_us=" << wait_us
                  << " total_us=" << total_us
                  << " inflight_after_submit=" << inflight_after_submit
                  << std::endl;
    }

    void
    MaybePrintWindow(const char* reason) {
        if (!IsEnvEnabled("MILVUS_READ_PATH_WINDOW_STATS")) {
            return;
        }
        const auto now_us = NowMicros();
        const auto interval_us =
            std::max<uint64_t>(1, GetUnsignedEnv("MILVUS_READ_PATH_WINDOW_MS", 250)) *
            1000;
        auto last_us = window_last_us.load(std::memory_order_relaxed);
        if (last_us == 0) {
            window_last_us.compare_exchange_strong(last_us,
                                                   now_us,
                                                   std::memory_order_relaxed,
                                                   std::memory_order_relaxed);
            return;
        }
        if (now_us <= last_us || now_us - last_us < interval_us) {
            return;
        }
        if (!window_last_us.compare_exchange_strong(last_us,
                                                    now_us,
                                                    std::memory_order_relaxed,
                                                    std::memory_order_relaxed)) {
            return;
        }

        const auto stream_batches_now =
            stream_batches.load(std::memory_order_relaxed);
        const auto chunk_batches_now =
            chunk_batches.load(std::memory_order_relaxed);
        const auto submitted_now = submitted.load(std::memory_order_relaxed);
        const auto completed_now = completed.load(std::memory_order_relaxed);
        const auto failed_now = failed.load(std::memory_order_relaxed);
        const auto bytes_now = bytes.load(std::memory_order_relaxed);
        const auto stream_reads_now =
            stream_batch_reads_sum.load(std::memory_order_relaxed);
        const auto chunk_reads_now =
            chunk_batch_reads_sum.load(std::memory_order_relaxed);
        const auto wait_us_now = async_wait_us_sum.load(std::memory_order_relaxed);

        const auto stream_batches_prev =
            window_last_stream_batches.exchange(stream_batches_now,
                                                std::memory_order_relaxed);
        const auto chunk_batches_prev =
            window_last_chunk_batches.exchange(chunk_batches_now,
                                               std::memory_order_relaxed);
        const auto submitted_prev =
            window_last_submitted.exchange(submitted_now, std::memory_order_relaxed);
        const auto completed_prev =
            window_last_completed.exchange(completed_now, std::memory_order_relaxed);
        const auto failed_prev =
            window_last_failed.exchange(failed_now, std::memory_order_relaxed);
        const auto bytes_prev =
            window_last_bytes.exchange(bytes_now, std::memory_order_relaxed);
        const auto stream_reads_prev =
            window_last_stream_reads.exchange(stream_reads_now,
                                              std::memory_order_relaxed);
        const auto chunk_reads_prev =
            window_last_chunk_reads.exchange(chunk_reads_now,
                                             std::memory_order_relaxed);
        const auto wait_us_prev =
            window_last_wait_us.exchange(wait_us_now, std::memory_order_relaxed);
        const auto active_now =
            submitted_now >= completed_now + failed_now
                ? submitted_now - completed_now - failed_now
                : 0;

        std::cerr << "[MILVUS_READ_PATH_WINDOW]"
                  << " layer=cardinal"
                  << " reason=" << reason
                  << " ts_us=" << now_us
                  << " delta_us=" << (now_us - last_us)
                  << " active=" << active_now
                  << " submitted_total=" << submitted_now
                  << " completed_total=" << completed_now
                  << " failed_total=" << failed_now
                  << " bytes_total=" << bytes_now
                  << " stream_batches_total=" << stream_batches_now
                  << " chunk_batches_total=" << chunk_batches_now
                  << " submitted_delta=" << (submitted_now - submitted_prev)
                  << " completed_delta=" << (completed_now - completed_prev)
                  << " failed_delta=" << (failed_now - failed_prev)
                  << " bytes_delta=" << (bytes_now - bytes_prev)
                  << " stream_batches_delta="
                  << (stream_batches_now - stream_batches_prev)
                  << " chunk_batches_delta="
                  << (chunk_batches_now - chunk_batches_prev)
                  << " stream_reads_delta="
                  << (stream_reads_now - stream_reads_prev)
                  << " chunk_reads_delta=" << (chunk_reads_now - chunk_reads_prev)
                  << " wait_us_delta=" << (wait_us_now - wait_us_prev)
                  << " max_inflight="
                  << last_max_inflight.load(std::memory_order_relaxed)
                  << " inflight_after_submit_max="
                  << inflight_after_submit_max.load(std::memory_order_relaxed)
                  << std::endl;
    }
};

inline CardinalAsyncReadStats&
GetCardinalAsyncReadStats() {
    static CardinalAsyncReadStats stats;
    return stats;
}

inline void
MaybeReportCardinalAsyncStats() {
    if (!IsEnvEnabled("MILVUS_CARDINAL_ASYNC_STATS")) {
        return;
    }
    auto& stats = GetCardinalAsyncReadStats();
    const auto submitted = stats.submitted.load(std::memory_order_relaxed);
    const auto done = stats.completed.load(std::memory_order_relaxed) +
                      stats.failed.load(std::memory_order_relaxed);
    if (submitted == 0 || done != submitted) {
        return;
    }
    auto last = stats.last_reported_done.load(std::memory_order_relaxed);
    while (last < done &&
           !stats.last_reported_done.compare_exchange_weak(
               last,
               done,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    if (last < done) {
        stats.Print("all_submitted_done");
    }
}

inline void
MaybeReportCardinalAsyncWindow(const char* reason) {
    GetCardinalAsyncReadStats().MaybePrintWindow(reason);
}

}  // namespace cardinalv2::storage
