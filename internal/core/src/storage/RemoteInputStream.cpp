#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
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

}  // namespace

RemoteInputStream::RemoteInputStream(
    std::shared_ptr<arrow::io::RandomAccessFile>&& remote_file)
    : remote_file_(std::move(remote_file)) {
    auto status = remote_file_->GetSize();
    AssertInfo(status.ok(), "Failed to get size of remote file");
    file_size_ = static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::Read(void* data, size_t size) {
    auto status = remote_file_->Read(size, data);
    AssertInfo(status.ok(), "Failed to read from input stream");
    return static_cast<size_t>(status.ValueOrDie());
}

size_t
RemoteInputStream::ReadAt(void* data, size_t offset, size_t size) {
    auto status = remote_file_->ReadAt(offset, size, data);
    AssertInfo(status.ok(), "Failed to read from input stream");
    return static_cast<size_t>(status.ValueOrDie());
}

bool
RemoteInputStream::SupportsAsyncReadAt() const {
    return IsEnvEnabled("MILVUS_S3_GETOBJECT_ASYNC");
}

std::future<RemoteAsyncReadResult>
RemoteInputStream::ReadAtAsync(size_t offset, size_t size) {
    auto promise = std::make_shared<std::promise<RemoteAsyncReadResult>>();
    auto future = promise->get_future();
    try {
        auto arrow_future =
            remote_file_->ReadAsync(static_cast<int64_t>(offset),
                                    static_cast<int64_t>(size));
        arrow_future.AddCallback(
            [promise](const arrow::Result<std::shared_ptr<arrow::Buffer>>& result) {
                try {
                    if (!result.ok()) {
                        throw std::runtime_error(result.status().ToString());
                    }
                    auto buffer = result.ValueOrDie();
                    RemoteAsyncReadResult async_result;
                    async_result.bytes_read =
                        buffer ? static_cast<size_t>(buffer->size()) : 0;
                    async_result.data.resize(async_result.bytes_read);
                    if (async_result.bytes_read > 0) {
                        std::memcpy(async_result.data.data(),
                                    buffer->data(),
                                    async_result.bytes_read);
                    }
                    promise->set_value(std::move(async_result));
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            });
    } catch (...) {
        promise->set_exception(std::current_exception());
    }
    return future;
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
