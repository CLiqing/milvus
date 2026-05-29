#pragma once

#include <cstddef>
#include <future>
#include <optional>
#include <string>
#include <vector>

namespace milvus {

struct S3ReadPathConfig {
    bool override_enabled = false;
    std::string mode;
    std::optional<size_t> max_inflight;
    std::optional<size_t> event_loops;
    std::optional<size_t> crt_max_connections;
    std::optional<double> crt_throughput_gbps;
};

inline thread_local S3ReadPathConfig s3_read_path_config;

inline const S3ReadPathConfig&
GetS3ReadPathConfig() {
    return s3_read_path_config;
}

class ScopedS3ReadPathConfig {
 public:
    explicit ScopedS3ReadPathConfig(const S3ReadPathConfig& config)
        : previous_(s3_read_path_config) {
        s3_read_path_config = config;
    }

    ~ScopedS3ReadPathConfig() {
        s3_read_path_config = previous_;
    }

 private:
    S3ReadPathConfig previous_;
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

    virtual std::future<AsyncReadResult>
    ReadAtAsync(size_t offset, size_t size) = 0;

    virtual void
    ConfigureAsyncReadAtLimiter(size_t limiter_id, size_t max_inflight) = 0;

    virtual size_t
    GetAsyncReadAtMaxInflight() const = 0;

    virtual size_t
    GetAsyncReadAtCurrentInflight() const = 0;
};

}  // namespace milvus
