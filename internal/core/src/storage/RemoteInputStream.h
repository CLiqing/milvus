// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <deque>
#include <future>
#include <memory>
#include <mutex>

#include "arrow/io/interfaces.h"
#include "filemanager/AsyncInputStream.h"
#include "filemanager/InputStream.h"
#include "storage/RemoteAsyncReadResult.h"

namespace milvus::storage {

class RemoteInputStream : public milvus::InputStream,
                          public milvus::AsyncInputStream {
 public:
    explicit RemoteInputStream(
        std::shared_ptr<arrow::io::RandomAccessFile>&& remote_file);

    ~RemoteInputStream() override = default;

    size_t
    Size() const override;

    size_t
    Read(void* data, size_t size) override;

    size_t
    ReadAt(void* data, size_t offset, size_t size) override;

    bool
    SupportsAsyncReadAt() const override;

    std::future<RemoteAsyncReadResult>
    ReadAtAsync(size_t offset, size_t size) override;

    void
    ConfigureAsyncReadAtLimiter(size_t limiter_id, size_t max_inflight) override;

    size_t
    GetAsyncReadAtMaxInflight() const override;

    size_t
    GetAsyncReadAtCurrentInflight() const override;

    size_t
    Read(int fd, size_t size) override;

    size_t
    Tell() const override;

    bool
    Eof() const override;

    bool
    Seek(int64_t offset) override;

 private:
    struct PendingAsyncRead {
        std::shared_ptr<arrow::io::RandomAccessFile> remote_file;
        size_t offset;
        size_t size;
        std::shared_ptr<std::promise<RemoteAsyncReadResult>> promise;
        size_t limiter_id;
        milvus::S3ReadPathConfig s3_read_path_config;
    };

    struct AsyncReadLimiter {
        mutable std::mutex mutex;
        size_t max_inflight = static_cast<size_t>(-1);
        size_t inflight = 0;
        std::deque<PendingAsyncRead> pending;
    };

    static AsyncReadLimiter&
    GetAsyncReadLimiter(size_t limiter_id);

    static void
    StartAsyncRead(PendingAsyncRead request);

    static void
    FinishAsyncReadAndStartNext(size_t limiter_id);

    size_t file_size_;
    std::shared_ptr<arrow::io::RandomAccessFile> remote_file_;

    size_t async_read_at_limiter_id_;
};

}  // namespace milvus::storage
