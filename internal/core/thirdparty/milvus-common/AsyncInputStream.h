#pragma once

#include <cstddef>
#include <future>
#include <string>
#include <vector>
#include <optional>

namespace milvus {

struct S3ReadPathConfig {
    bool override_enabled = false;
    std::string mode;
    std::optional<size_t> max_inflight;
    std::optional<size_t> event_loops;
    std::optional<size_t> crt_max_connections;
    std::optional<double> crt_throughput_gbps;
};

struct AsyncReadResult {
    size_t bytes_read = 0;
    std::vector<std::byte> data;
};

class AsyncInputStream {
 public:
    virtual ~AsyncInputStream() = default;

    virtual bool
    SupportsAsyncReadAt() const = 0;

    virtual bool
    SupportsAsyncReadAt(const S3ReadPathConfig& config) const {
        (void)config;
        return SupportsAsyncReadAt();
    }

    virtual S3ReadPathConfig
    GetS3ReadPathConfig() const = 0;

    virtual std::future<AsyncReadResult>
    ReadAtAsync(size_t offset, size_t size) = 0;

    virtual std::future<AsyncReadResult>
    ReadAtAsync(size_t offset,
                size_t size,
                const S3ReadPathConfig& config) {
        (void)config;
        return ReadAtAsync(offset, size);
    }

    virtual void
    ConfigureAsyncReadAtLimiter(size_t limiter_id, size_t max_inflight) = 0;

    virtual size_t
    GetAsyncReadAtMaxInflight() const = 0;

    virtual size_t
    GetAsyncReadAtCurrentInflight() const = 0;
};

}  // namespace milvus
