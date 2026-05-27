#pragma once

#include <cstddef>
#include <future>
#include <vector>

namespace milvus {

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
