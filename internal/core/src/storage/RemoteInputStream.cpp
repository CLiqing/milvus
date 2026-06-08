#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "RemoteInputStream.h"
#include "arrow/buffer.h"
#include "arrow/io/interfaces.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/future.h"
#include "common/Consts.h"
#include "common/EasyAssert.h"

extern "C" void
milvus_storage_set_s3_read_path_context_for_file(void* file,
                                                 const char* mode,
                                                 uint64_t max_inflight,
                                                 uint64_t event_loops,
                                                 uint64_t crt_max_connections,
                                                 double crt_throughput_gbps,
                                                 bool has_crt_throughput_gbps)
    __attribute__((weak));

using MilvusStorageReadAsyncIntoCallback = void (*)(
    void* callback_ctx, int64_t bytes_read, const char* error_message);

extern "C" bool
milvus_storage_read_async_into_file(
    void* file,
    int64_t position,
    int64_t nbytes,
    void* output,
    MilvusStorageReadAsyncIntoCallback callback,
    void* callback_ctx) __attribute__((weak));

namespace milvus::storage {
namespace {

bool
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

uint64_t
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

bool
IsS3AsyncReadPathEnabled() {
    return IsEnvEnabled("MILVUS_S3_GETOBJECT_ASYNC");
}

bool
IsS3AsyncReadPathEnabled(const milvus::S3ReadPathConfig& config) {
    if (config.override_enabled) {
        return config.mode == "curl_multi" || config.mode == "crt";
    }
    return IsEnvEnabled("MILVUS_S3_GETOBJECT_ASYNC");
}

bool
ShouldPrintS3ReadPathLog() {
    static const bool enabled = IsEnvEnabled("MILVUS_S3_READ_PATH_LOG");
    if (!enabled) {
        return false;
    }
    static std::atomic<uint64_t> last_print_us{0};
    const auto now_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    auto last_us = last_print_us.load(std::memory_order_relaxed);
    if (last_us != 0 && now_us <= last_us + 1000000) {
        return false;
    }
    return last_print_us.compare_exchange_strong(
        last_us,
        now_us,
        std::memory_order_relaxed,
        std::memory_order_relaxed);
}

void
PrintS3ReadPathProbe(const char* operation,
                     const milvus::S3ReadPathConfig& config,
                     size_t offset,
                     size_t size) {
    if (!ShouldPrintS3ReadPathLog()) {
        return;
    }
    std::cerr << "[MILVUS_S3_READ_PATH]"
              << " layer=remote_input_stream_probe"
              << " operation=" << operation
              << " override_enabled=" << config.override_enabled
              << " mode=" << config.mode
              << " offset=" << offset
              << " size=" << size
              << std::endl;
}

bool
ShouldPrintS3ReadWorkloadTrace(uint64_t& seq) {
    static const bool enabled = IsEnvEnabled("MILVUS_S3_READ_WORKLOAD_TRACE");
    if (!enabled) {
        return false;
    }
    static const uint64_t limit =
        GetUnsignedEnv("MILVUS_S3_READ_WORKLOAD_TRACE_LIMIT", 100000);
    static std::atomic<uint64_t> counter{0};
    seq = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    return limit == 0 || seq <= limit;
}

uint64_t
UnixMicros() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

void
PrintS3ReadWorkloadTrace(const char* operation,
                         const milvus::S3ReadPathConfig& config,
                         size_t offset,
                         size_t size,
                         size_t limiter_id,
                         size_t limiter_max_inflight,
                         size_t limiter_current_inflight) {
    uint64_t seq = 0;
    if (!ShouldPrintS3ReadWorkloadTrace(seq)) {
        return;
    }
    std::ostringstream out;
    out << "[MILVUS_S3_READ_WORKLOAD]"
        << " seq=" << seq
        << " ts_us=" << UnixMicros()
        << " layer=remote_input_stream"
        << " operation=" << operation
        << " override_enabled=" << config.override_enabled
        << " mode=" << config.mode
        << " offset=" << offset
        << " size=" << size
        << " limiter_id=" << limiter_id
        << " limiter_max_inflight=" << limiter_max_inflight
        << " limiter_current_inflight=" << limiter_current_inflight
        << " thread_id=" << std::this_thread::get_id()
        << "\n";
    const auto line = out.str();
    (void)::write(STDERR_FILENO, line.data(), line.size());
}

struct DirectAsyncReadCallbackState {
    std::shared_ptr<std::promise<RemoteAsyncReadResult>> promise;
    size_t limiter_id;
};

void
ApplyS3ReadPathConfigForFile(arrow::io::RandomAccessFile* file,
                             const milvus::S3ReadPathConfig& config) {
    if (milvus_storage_set_s3_read_path_context_for_file == nullptr ||
        file == nullptr || !config.override_enabled) {
        return;
    }
    milvus_storage_set_s3_read_path_context_for_file(
        file,
        config.mode.c_str(),
        static_cast<uint64_t>(config.max_inflight.value_or(0)),
        static_cast<uint64_t>(config.event_loops.value_or(0)),
        static_cast<uint64_t>(config.crt_max_connections.value_or(0)),
        config.crt_throughput_gbps.value_or(0.0),
        config.crt_throughput_gbps.has_value());
}

uint64_t
NowMicros() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void
UpdateMax(std::atomic<uint64_t>& target, uint64_t value) {
    auto current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current,
                                         value,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

struct AsyncReadStats {
    std::atomic<uint64_t> configured{0};
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> started{0};
    std::atomic<uint64_t> queued{0};
    std::atomic<uint64_t> completed{0};
    std::atomic<uint64_t> failed{0};
    std::atomic<uint64_t> requested_bytes{0};
    std::atomic<uint64_t> completed_bytes{0};
    std::atomic<uint64_t> max_pending{0};
    std::atomic<uint64_t> max_inflight_observed{0};
    std::atomic<uint64_t> last_max_inflight{0};
    std::atomic<uint64_t> last_reported_done{0};
    std::atomic<uint64_t> window_last_us{0};
    std::atomic<uint64_t> window_last_submitted{0};
    std::atomic<uint64_t> window_last_started{0};
    std::atomic<uint64_t> window_last_completed{0};
    std::atomic<uint64_t> window_last_failed{0};
    std::atomic<uint64_t> window_last_requested_bytes{0};
    std::atomic<uint64_t> window_last_completed_bytes{0};

    ~AsyncReadStats() {
        if (!IsEnvEnabled("MILVUS_S3_ASYNC_STATS")) {
            return;
        }
        Print("process_exit");
    }

    void
    Print(const char* reason) const {
        std::cerr << "[MILVUS_S3_ASYNC_STATS]"
                  << " reason=" << reason
                  << " configured="
                  << configured.load(std::memory_order_relaxed)
                  << " submitted=" << submitted.load(std::memory_order_relaxed)
                  << " started=" << started.load(std::memory_order_relaxed)
                  << " queued=" << queued.load(std::memory_order_relaxed)
                  << " completed="
                  << completed.load(std::memory_order_relaxed)
                  << " failed=" << failed.load(std::memory_order_relaxed)
                  << " requested_bytes="
                  << requested_bytes.load(std::memory_order_relaxed)
                  << " completed_bytes="
                  << completed_bytes.load(std::memory_order_relaxed)
                  << " max_pending="
                  << max_pending.load(std::memory_order_relaxed)
                  << " max_inflight_observed="
                  << max_inflight_observed.load(std::memory_order_relaxed)
                  << " last_max_inflight="
                  << last_max_inflight.load(std::memory_order_relaxed)
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

        const auto submitted_now = submitted.load(std::memory_order_relaxed);
        const auto started_now = started.load(std::memory_order_relaxed);
        const auto completed_now = completed.load(std::memory_order_relaxed);
        const auto failed_now = failed.load(std::memory_order_relaxed);
        const auto requested_bytes_now =
            requested_bytes.load(std::memory_order_relaxed);
        const auto completed_bytes_now =
            completed_bytes.load(std::memory_order_relaxed);
        const auto submitted_prev =
            window_last_submitted.exchange(submitted_now, std::memory_order_relaxed);
        const auto started_prev =
            window_last_started.exchange(started_now, std::memory_order_relaxed);
        const auto completed_prev =
            window_last_completed.exchange(completed_now, std::memory_order_relaxed);
        const auto failed_prev =
            window_last_failed.exchange(failed_now, std::memory_order_relaxed);
        const auto requested_bytes_prev =
            window_last_requested_bytes.exchange(requested_bytes_now,
                                                 std::memory_order_relaxed);
        const auto completed_bytes_prev =
            window_last_completed_bytes.exchange(completed_bytes_now,
                                                 std::memory_order_relaxed);
        const auto active_now =
            started_now >= completed_now + failed_now
                ? started_now - completed_now - failed_now
                : 0;

        std::cerr << "[MILVUS_READ_PATH_WINDOW]"
                  << " layer=remote_input"
                  << " reason=" << reason
                  << " ts_us=" << now_us
                  << " delta_us=" << (now_us - last_us)
                  << " active=" << active_now
                  << " submitted_total=" << submitted_now
                  << " started_total=" << started_now
                  << " completed_total=" << completed_now
                  << " failed_total=" << failed_now
                  << " requested_bytes_total=" << requested_bytes_now
                  << " completed_bytes_total=" << completed_bytes_now
                  << " submitted_delta=" << (submitted_now - submitted_prev)
                  << " started_delta=" << (started_now - started_prev)
                  << " completed_delta=" << (completed_now - completed_prev)
                  << " failed_delta=" << (failed_now - failed_prev)
                  << " requested_bytes_delta="
                  << (requested_bytes_now - requested_bytes_prev)
                  << " completed_bytes_delta="
                  << (completed_bytes_now - completed_bytes_prev)
                  << " max_pending=" << max_pending.load(std::memory_order_relaxed)
                  << " max_inflight_observed="
                  << max_inflight_observed.load(std::memory_order_relaxed)
                  << " last_max_inflight="
                  << last_max_inflight.load(std::memory_order_relaxed)
                  << std::endl;
    }
};

AsyncReadStats&
GetAsyncReadStats() {
    static AsyncReadStats stats;
    return stats;
}

void
MaybeReportAsyncReadStats() {
    if (!IsEnvEnabled("MILVUS_S3_ASYNC_STATS")) {
        return;
    }
    auto& stats = GetAsyncReadStats();
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

}  // namespace

void
RemoteInputStream::CompleteDirectAsyncRead(void* callback_ctx,
                                           int64_t bytes_read,
                                           const char* error_message) {
    std::unique_ptr<DirectAsyncReadCallbackState> state(
        static_cast<DirectAsyncReadCallbackState*>(callback_ctx));
    try {
        if (error_message != nullptr && *error_message != '\0') {
            throw std::runtime_error(error_message);
        }
        if (bytes_read < 0) {
            throw std::runtime_error("direct async read returned negative byte count");
        }

        RemoteAsyncReadResult async_result;
        async_result.bytes_read = static_cast<size_t>(bytes_read);
        GetAsyncReadStats().completed.fetch_add(1, std::memory_order_relaxed);
        GetAsyncReadStats().completed_bytes.fetch_add(
            async_result.bytes_read, std::memory_order_relaxed);
        GetAsyncReadStats().MaybePrintWindow("complete_direct_into");
        state->promise->set_value(std::move(async_result));
    } catch (...) {
        GetAsyncReadStats().failed.fetch_add(1, std::memory_order_relaxed);
        GetAsyncReadStats().MaybePrintWindow("fail_direct_into");
        state->promise->set_exception(std::current_exception());
    }
    FinishAsyncReadAndStartNext(state->limiter_id);
    MaybeReportAsyncReadStats();
}

RemoteInputStream::RemoteInputStream(
    std::shared_ptr<arrow::io::RandomAccessFile>&& remote_file,
    milvus::S3ReadPathConfig s3_read_path_config)
    : remote_file_(std::move(remote_file)),
      s3_read_path_config_(std::move(s3_read_path_config)),
      s3_read_path_context_mutex_(std::make_shared<std::mutex>()),
      async_read_at_limiter_id_(0) {
    auto status = remote_file_->GetSize();
    AssertInfo(status.ok(), "Failed to get size of remote file");
    file_size_ = static_cast<size_t>(status.ValueOrDie());
    std::lock_guard<std::mutex> lock(*s3_read_path_context_mutex_);
    ApplyS3ReadPathConfigForFile(remote_file_.get(), s3_read_path_config_);
}

size_t
RemoteInputStream::Read(void* data, size_t size) {
    auto status = remote_file_->Read(size, data);
    AssertInfo(status.ok(), "Failed to read from input stream");
    return static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::ReadAt(void* data, size_t offset, size_t size) {
    PrintS3ReadPathProbe(
        "ReadAt", s3_read_path_config_, offset, size);
    PrintS3ReadWorkloadTrace("ReadAt",
                             s3_read_path_config_,
                             offset,
                             size,
                             async_read_at_limiter_id_,
                             0,
                             0);
    auto status = remote_file_->ReadAt(offset, size, data);
    AssertInfo(status.ok(), "Failed to read from input stream");
    return static_cast<size_t>(status.ValueOrDie());
}

bool
RemoteInputStream::SupportsAsyncReadAt() const {
    return IsS3AsyncReadPathEnabled(s3_read_path_config_);
}

bool
RemoteInputStream::SupportsAsyncReadAt(
    const milvus::S3ReadPathConfig& config) const {
    return IsS3AsyncReadPathEnabled(config);
}

milvus::S3ReadPathConfig
RemoteInputStream::GetS3ReadPathConfig() const {
    return s3_read_path_config_;
}

std::future<RemoteAsyncReadResult>
RemoteInputStream::ReadAtAsync(size_t offset, size_t size) {
    return ReadAtAsync(offset, size, s3_read_path_config_);
}

std::future<RemoteAsyncReadResult>
RemoteInputStream::ReadAtAsync(size_t offset,
                               size_t size,
                               const milvus::S3ReadPathConfig& config) {
    const auto effective_config =
        config.override_enabled ? config : s3_read_path_config_;
    PrintS3ReadPathProbe(
        "ReadAtAsync", effective_config, offset, size);
    PrintS3ReadWorkloadTrace("ReadAtAsync",
                             effective_config,
                             offset,
                             size,
                             async_read_at_limiter_id_,
                             GetAsyncReadAtMaxInflight(),
                             GetAsyncReadAtCurrentInflight());
    auto& stats = GetAsyncReadStats();
    stats.submitted.fetch_add(1, std::memory_order_relaxed);
    stats.requested_bytes.fetch_add(size, std::memory_order_relaxed);
    stats.MaybePrintWindow("submit");

    auto promise = std::make_shared<std::promise<RemoteAsyncReadResult>>();
    auto future = promise->get_future();
    PendingAsyncRead request{
        remote_file_,
        offset,
        size,
        nullptr,
        promise,
        async_read_at_limiter_id_,
        effective_config,
        s3_read_path_context_mutex_};
    bool start_now = false;
    auto& limiter = GetAsyncReadLimiter(request.limiter_id);
    {
        std::lock_guard<std::mutex> lock(limiter.mutex);
        if (limiter.inflight < limiter.max_inflight) {
            limiter.inflight++;
            UpdateMax(stats.max_inflight_observed, limiter.inflight);
            start_now = true;
        } else {
            limiter.pending.push_back(std::move(request));
            stats.queued.fetch_add(1, std::memory_order_relaxed);
            UpdateMax(stats.max_pending, limiter.pending.size());
        }
    }

    if (start_now) {
        StartAsyncRead(std::move(request));
    }
    return future;
}

std::future<RemoteAsyncReadResult>
RemoteInputStream::ReadAtAsyncInto(size_t offset, size_t size, void* data) {
    return ReadAtAsyncInto(offset, size, data, s3_read_path_config_);
}

std::future<RemoteAsyncReadResult>
RemoteInputStream::ReadAtAsyncInto(size_t offset,
                                   size_t size,
                                   void* data,
                                   const milvus::S3ReadPathConfig& config) {
    const auto effective_config =
        config.override_enabled ? config : s3_read_path_config_;
    PrintS3ReadPathProbe("ReadAtAsyncInto", effective_config, offset, size);
    PrintS3ReadWorkloadTrace("ReadAtAsyncInto",
                             effective_config,
                             offset,
                             size,
                             async_read_at_limiter_id_,
                             GetAsyncReadAtMaxInflight(),
                             GetAsyncReadAtCurrentInflight());
    auto& stats = GetAsyncReadStats();
    stats.submitted.fetch_add(1, std::memory_order_relaxed);
    stats.requested_bytes.fetch_add(size, std::memory_order_relaxed);
    stats.MaybePrintWindow("submit_direct_into");

    auto promise = std::make_shared<std::promise<RemoteAsyncReadResult>>();
    auto future = promise->get_future();
    PendingAsyncRead request{
        remote_file_,
        offset,
        size,
        data,
        promise,
        async_read_at_limiter_id_,
        effective_config,
        s3_read_path_context_mutex_};
    bool start_now = false;
    auto& limiter = GetAsyncReadLimiter(request.limiter_id);
    {
        std::lock_guard<std::mutex> lock(limiter.mutex);
        if (limiter.inflight < limiter.max_inflight) {
            limiter.inflight++;
            UpdateMax(stats.max_inflight_observed, limiter.inflight);
            start_now = true;
        } else {
            limiter.pending.push_back(std::move(request));
            stats.queued.fetch_add(1, std::memory_order_relaxed);
            UpdateMax(stats.max_pending, limiter.pending.size());
        }
    }

    if (start_now) {
        StartAsyncRead(std::move(request));
    }
    return future;
}

void
RemoteInputStream::ConfigureAsyncReadAtLimiter(size_t limiter_id,
                                               size_t max_inflight) {
    if (max_inflight == 0) {
        max_inflight = 1;
    }
    auto& stats = GetAsyncReadStats();
    stats.configured.fetch_add(1, std::memory_order_relaxed);
    stats.last_max_inflight.store(max_inflight, std::memory_order_relaxed);
    async_read_at_limiter_id_ = limiter_id;

    std::vector<PendingAsyncRead> ready;
    auto& limiter = GetAsyncReadLimiter(limiter_id);
    {
        std::lock_guard<std::mutex> lock(limiter.mutex);
        limiter.max_inflight = max_inflight;
        while (!limiter.pending.empty() &&
               limiter.inflight < limiter.max_inflight) {
            ready.push_back(std::move(limiter.pending.front()));
            limiter.pending.pop_front();
            limiter.inflight++;
            UpdateMax(stats.max_inflight_observed, limiter.inflight);
        }
    }

    for (auto& request : ready) {
        StartAsyncRead(std::move(request));
    }
}

size_t
RemoteInputStream::GetAsyncReadAtMaxInflight() const {
    const auto& limiter = GetAsyncReadLimiter(async_read_at_limiter_id_);
    std::lock_guard<std::mutex> lock(limiter.mutex);
    return limiter.max_inflight;
}

size_t
RemoteInputStream::GetAsyncReadAtCurrentInflight() const {
    const auto& limiter = GetAsyncReadLimiter(async_read_at_limiter_id_);
    std::lock_guard<std::mutex> lock(limiter.mutex);
    return limiter.inflight;
}

RemoteInputStream::AsyncReadLimiter&
RemoteInputStream::GetAsyncReadLimiter(size_t limiter_id) {
    static std::array<AsyncReadLimiter, 4> limiters;
    if (limiter_id >= limiters.size()) {
        limiter_id = 0;
    }
    return limiters[limiter_id];
}

void
RemoteInputStream::StartAsyncRead(PendingAsyncRead request) {
    GetAsyncReadStats().started.fetch_add(1, std::memory_order_relaxed);
    GetAsyncReadStats().MaybePrintWindow("start");
    try {
        if (request.output != nullptr &&
            milvus_storage_read_async_into_file != nullptr) {
            auto* callback_state = new DirectAsyncReadCallbackState{
                request.promise,
                request.limiter_id};
            bool accepted = false;
            {
                std::lock_guard<std::mutex> lock(
                    *request.s3_read_path_context_mutex);
                ApplyS3ReadPathConfigForFile(request.remote_file.get(),
                                             request.s3_read_path_config);
                accepted = milvus_storage_read_async_into_file(
                    request.remote_file.get(),
                    static_cast<int64_t>(request.offset),
                    static_cast<int64_t>(request.size),
                    request.output,
                    &RemoteInputStream::CompleteDirectAsyncRead,
                    callback_state);
            }
            if (accepted) {
                return;
            }
            delete callback_state;
        }

        arrow::Future<std::shared_ptr<arrow::Buffer>> arrow_future;
        {
            std::lock_guard<std::mutex> lock(
                *request.s3_read_path_context_mutex);
            ApplyS3ReadPathConfigForFile(request.remote_file.get(),
                                         request.s3_read_path_config);
            arrow_future = request.remote_file->ReadAsync(
                static_cast<int64_t>(request.offset),
                static_cast<int64_t>(request.size));
        }
        arrow_future.AddCallback(
            [promise = request.promise,
             limiter_id = request.limiter_id,
             output = request.output](
                const arrow::Result<std::shared_ptr<arrow::Buffer>>& result) {
                try {
                    if (!result.ok()) {
                        throw std::runtime_error(result.status().ToString());
                    }
                    auto buffer = result.ValueOrDie();
                    RemoteAsyncReadResult async_result;
                    async_result.bytes_read =
                        buffer ? static_cast<size_t>(buffer->size()) : 0;
                    GetAsyncReadStats().completed.fetch_add(
                        1, std::memory_order_relaxed);
                    GetAsyncReadStats().completed_bytes.fetch_add(
                        async_result.bytes_read, std::memory_order_relaxed);
                    GetAsyncReadStats().MaybePrintWindow("complete");
                    if (output != nullptr) {
                        if (async_result.bytes_read > 0) {
                            std::memcpy(output,
                                        buffer->data(),
                                        async_result.bytes_read);
                        }
                    } else {
                        async_result.data.resize(async_result.bytes_read);
                        if (async_result.bytes_read > 0) {
                            std::memcpy(async_result.data.data(),
                                        buffer->data(),
                                        async_result.bytes_read);
                        }
                    }
                    promise->set_value(std::move(async_result));
                } catch (...) {
                    GetAsyncReadStats().failed.fetch_add(
                        1, std::memory_order_relaxed);
                    GetAsyncReadStats().MaybePrintWindow("fail");
                    promise->set_exception(std::current_exception());
                }
                FinishAsyncReadAndStartNext(limiter_id);
                MaybeReportAsyncReadStats();
            });
    } catch (...) {
        GetAsyncReadStats().failed.fetch_add(1, std::memory_order_relaxed);
        GetAsyncReadStats().MaybePrintWindow("fail_start");
        request.promise->set_exception(std::current_exception());
        FinishAsyncReadAndStartNext(request.limiter_id);
        MaybeReportAsyncReadStats();
    }
}

void
RemoteInputStream::FinishAsyncReadAndStartNext(size_t limiter_id) {
    PendingAsyncRead next;
    bool has_next = false;
    auto& limiter = GetAsyncReadLimiter(limiter_id);
    {
        std::lock_guard<std::mutex> lock(limiter.mutex);
        AssertInfo(limiter.inflight > 0,
                   "Invalid async read in-flight count");
        limiter.inflight--;
        if (!limiter.pending.empty() &&
            limiter.inflight < limiter.max_inflight) {
            next = std::move(limiter.pending.front());
            limiter.pending.pop_front();
            limiter.inflight++;
            UpdateMax(GetAsyncReadStats().max_inflight_observed,
                      limiter.inflight);
            has_next = true;
        }
    }

    if (has_next) {
        StartAsyncRead(std::move(next));
    }
}

size_t
RemoteInputStream::Read(int fd, size_t size) {
    size_t read_batch_size =
        std::min(size, static_cast<size_t>(DEFAULT_INDEX_FILE_SLICE_SIZE));
    size_t rest_size = size;
    std::vector<uint8_t> data(read_batch_size);

    while (rest_size > 0) {
        size_t read_size = std::min(rest_size, read_batch_size);
        auto status = remote_file_->Read(read_size, data.data());
        AssertInfo(status.ok(), "Failed to read from input stream");
        ssize_t ret = ::write(fd, data.data(), read_size);
        AssertInfo(ret == static_cast<ssize_t>(read_size),
                   "Failed to write to file");
        rest_size -= read_size;
    }
    ::fsync(fd);
    return size;
}

size_t
RemoteInputStream::Tell() const {
    auto status = remote_file_->Tell();
    AssertInfo(status.ok(), "Failed to tell input stream");
    return static_cast<size_t>(status.ValueOrDie());
}

bool
RemoteInputStream::Eof() const {
    return Tell() >= file_size_;
}

bool
RemoteInputStream::Seek(int64_t offset) {
    auto status = remote_file_->Seek(offset);
    return status.ok();
}

size_t
RemoteInputStream::Size() const {
    return file_size_;
}

}  // namespace milvus::storage
