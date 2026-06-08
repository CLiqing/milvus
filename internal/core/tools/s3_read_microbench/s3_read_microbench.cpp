#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "cachinglayer/Manager.h"
#include <cardinal/load_config.h>
#include <cardinal/serialization/serialization.h>
#include <cardinal/storage/memory_storage.h>
#include <cardinal/storage/translator/chunk_translator.h>
#include <cardinal/thread/task_pool.h>
#include "common/OpContext.h"
#include "knowhere/thread_pool.h"
#include "milvus-storage/filesystem/fs.h"
#include "storage/RemoteInputStream.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string object_path;
    std::string workload_file;
    std::string mode = "baseline";
    uint64_t duration_sec = 120;
    size_t concurrency = 100;
    size_t range_size = 4 * 1024 * 1024;
    size_t max_inflight = 100;
    size_t fetch_pool_size = 100;
    size_t event_loops = 8;
    size_t crt_max_connections = 100;
    double crt_throughput_gbps = 30.0;
    std::string pattern = "sequential";
    std::string scheduler = "batch";
    uint64_t warmup_sec = 0;
    bool repeat_workload_until_deadline = false;

    milvus_storage::ArrowFileSystemConfig storage;
};

struct RangeTiming {
    uint64_t open_us = 0;
    uint64_t alloc_us = 0;
    uint64_t submit_us = 0;
    uint64_t wait_us = 0;
    uint64_t post_us = 0;
    uint64_t total_us = 0;
};

struct Stats {
    std::atomic<uint64_t> batches{0};
    std::atomic<uint64_t> requests{0};
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<uint64_t> latency_us_sum{0};
    std::atomic<uint64_t> latency_us_min{std::numeric_limits<uint64_t>::max()};
    std::atomic<uint64_t> latency_us_max{0};

    mutable std::mutex phase_mutex;
    std::vector<uint64_t> open_us;
    std::vector<uint64_t> alloc_us;
    std::vector<uint64_t> submit_us;
    std::vector<uint64_t> wait_us;
    std::vector<uint64_t> post_us;
    std::vector<uint64_t> total_us;
};

struct ChunkSpec {
    std::string object_path;
    size_t offset = 0;
    size_t size = 0;
};

class RangedRemoteInputStream : public milvus::InputStream,
                                public milvus::AsyncInputStream {
 public:
    RangedRemoteInputStream(
        std::shared_ptr<milvus::storage::RemoteInputStream> stream,
        size_t start_offset,
        size_t range_size)
        : stream_(std::move(stream)),
          start_offset_(start_offset),
          range_size_(range_size) {
    }

    size_t
    Size() const override {
        return range_size_;
    }

    bool
    Seek(int64_t offset) override {
        if (offset < 0 || static_cast<size_t>(offset) > range_size_) {
            return false;
        }
        current_pos_ = static_cast<size_t>(offset);
        return true;
    }

    size_t
    Tell() const override {
        return current_pos_;
    }

    bool
    Eof() const override {
        return current_pos_ >= range_size_;
    }

    size_t
    Read(void* ptr, size_t size) override {
        if (current_pos_ >= range_size_) {
            return 0;
        }
        const auto to_read = std::min(size, range_size_ - current_pos_);
        const auto bytes_read = ReadAt(ptr, current_pos_, to_read);
        current_pos_ += bytes_read;
        return bytes_read;
    }

    size_t
    ReadAt(void* ptr, size_t offset, size_t size) override {
        if (offset >= range_size_) {
            return 0;
        }
        const auto to_read = std::min(size, range_size_ - offset);
        return stream_->ReadAt(ptr, start_offset_ + offset, to_read);
    }

    size_t
    Read(int fd, size_t size) override {
        std::vector<std::byte> buffer(size);
        const auto bytes_read = Read(buffer.data(), size);
        if (bytes_read == 0) {
            return 0;
        }
        const auto bytes_written = ::write(fd, buffer.data(), bytes_read);
        if (bytes_written < 0) {
            return 0;
        }
        return static_cast<size_t>(bytes_written);
    }

    bool
    SupportsAsyncReadAt() const override {
        return stream_->SupportsAsyncReadAt();
    }

    bool
    SupportsAsyncReadAt(const milvus::S3ReadPathConfig& config) const override {
        return stream_->SupportsAsyncReadAt(config);
    }

    milvus::S3ReadPathConfig
    GetS3ReadPathConfig() const override {
        return stream_->GetS3ReadPathConfig();
    }

    std::future<milvus::AsyncReadResult>
    ReadAtAsync(size_t offset, size_t size) override {
        if (offset >= range_size_) {
            return EmptyFuture();
        }
        const auto to_read = std::min(size, range_size_ - offset);
        return stream_->ReadAtAsync(start_offset_ + offset, to_read);
    }

    std::future<milvus::AsyncReadResult>
    ReadAtAsync(size_t offset,
                size_t size,
                const milvus::S3ReadPathConfig& config) override {
        if (offset >= range_size_) {
            return EmptyFuture();
        }
        const auto to_read = std::min(size, range_size_ - offset);
        return stream_->ReadAtAsync(start_offset_ + offset, to_read, config);
    }

    std::future<milvus::AsyncReadResult>
    ReadAtAsyncInto(size_t offset, size_t size, void* data) override {
        if (offset >= range_size_) {
            return EmptyFuture();
        }
        const auto to_read = std::min(size, range_size_ - offset);
        return stream_->ReadAtAsyncInto(start_offset_ + offset, to_read, data);
    }

    std::future<milvus::AsyncReadResult>
    ReadAtAsyncInto(size_t offset,
                    size_t size,
                    void* data,
                    const milvus::S3ReadPathConfig& config) override {
        if (offset >= range_size_) {
            return EmptyFuture();
        }
        const auto to_read = std::min(size, range_size_ - offset);
        return stream_->ReadAtAsyncInto(
            start_offset_ + offset, to_read, data, config);
    }

    void
    ConfigureAsyncReadAtLimiter(size_t limiter_id,
                                size_t max_inflight) override {
        stream_->ConfigureAsyncReadAtLimiter(limiter_id, max_inflight);
    }

    size_t
    GetAsyncReadAtMaxInflight() const override {
        return stream_->GetAsyncReadAtMaxInflight();
    }

    size_t
    GetAsyncReadAtCurrentInflight() const override {
        return stream_->GetAsyncReadAtCurrentInflight();
    }

 private:
    static std::future<milvus::AsyncReadResult>
    EmptyFuture() {
        std::promise<milvus::AsyncReadResult> promise;
        promise.set_value({});
        return promise.get_future();
    }

    std::shared_ptr<milvus::storage::RemoteInputStream> stream_;
    size_t start_offset_;
    size_t range_size_;
    size_t current_pos_ = 0;
};

class ChunkTensorMeta : public cardinalv2::DeserializerInterface {
 public:
    ChunkTensorMeta(std::shared_ptr<cardinalv2::LoadConfig> load_config,
                    milvus_storage::ArrowFileSystemPtr fs,
                    ChunkSpec spec,
                    milvus::S3ReadPathConfig stream_config)
        : load_config_(std::move(load_config)),
          fs_(std::move(fs)),
          spec_(std::move(spec)),
          stream_config_(std::move(stream_config)) {
    }

    cardinalv2::LoadConfig&
    GetLoadConfig() const override {
        return *load_config_;
    }

    void
    SetLoadConfig(const cardinalv2::LoadConfig& load_config) override {
        *load_config_ = load_config;
    }

    std::shared_ptr<milvus::InputStream>
    GetInputStreamForTensor(const std::string& key) const override {
        if (key != "data") {
            throw std::runtime_error("unknown tensor key: " + key);
        }
        auto result = fs_->OpenInputFile(spec_.object_path);
        if (!result.ok()) {
            throw std::runtime_error("OpenInputFile failed: " +
                                     result.status().ToString());
        }
        auto remote = std::make_shared<milvus::storage::RemoteInputStream>(
            std::move(result.ValueOrDie()), stream_config_);
        return std::make_shared<RangedRemoteInputStream>(
            std::move(remote), spec_.offset, spec_.size);
    }

    bool
    HasKV(const std::string& key) const override {
        return key == "capacity" || key == "size" || key == "length";
    }

    bool
    HasChild(const std::string&) const override {
        return false;
    }

    bool
    HasTensor(const std::string& key) const override {
        return key == "data";
    }

    std::shared_ptr<cardinalv2::DeserializerInterface>
    GetChildPtr(const std::string& key) const override {
        throw std::runtime_error("child not found: " + key);
    }

    std::string
    GetKVString(const std::string& key) const override {
        throw std::runtime_error("string key not found: " + key);
    }

    int32_t
    GetKVI32(const std::string& key) const override {
        return static_cast<int32_t>(GetKVU64(key));
    }

    uint32_t
    GetKVU32(const std::string& key) const override {
        return static_cast<uint32_t>(GetKVU64(key));
    }

    int64_t
    GetKVI64(const std::string& key) const override {
        return static_cast<int64_t>(GetKVU64(key));
    }

    uint64_t
    GetKVU64(const std::string& key) const override {
        if (key == "capacity" || key == "length") {
            return spec_.size;
        }
        if (key == "size") {
            return 1;
        }
        throw std::runtime_error("uint64 key not found: " + key);
    }

    bool
    GetKVBool(const std::string& key) const override {
        throw std::runtime_error("bool key not found: " + key);
    }

    float
    GetKVF32(const std::string& key) const override {
        throw std::runtime_error("float key not found: " + key);
    }

    std::vector<uint8_t>
    GetKVVecU8(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }

    std::vector<uint16_t>
    GetKVVecU16(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }

    std::vector<int32_t>
    GetKVVecI32(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }

    std::vector<float>
    GetKVVecF32(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }

    std::vector<uint32_t>
    GetKVVecU32(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }

    std::vector<uint64_t>
    GetKVVecU64(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }

    std::unordered_map<int32_t, int32_t>
    GetKVMapI32I32(const std::string& key) const override {
        throw std::runtime_error("map key not found: " + key);
    }

    std::unordered_map<int64_t, uint64_t>
    GetKVMapI64U64(const std::string& key) const override {
        throw std::runtime_error("map key not found: " + key);
    }

    std::unordered_map<int64_t, uint32_t>
    GetKVMapI64U32(const std::string& key) const override {
        throw std::runtime_error("map key not found: " + key);
    }

    void
    GetKVTensor(const std::string& key, std::byte* data) const override {
        auto stream = GetInputStreamForTensor(key);
        const auto bytes_read = stream->ReadAt(data, 0, spec_.size);
        if (bytes_read != spec_.size) {
            throw std::runtime_error("short tensor read");
        }
    }

    void
    GetKVBlobWithoutSize(const std::string& key,
                         std::byte*,
                         const size_t) const override {
        throw std::runtime_error("blob key not found: " + key);
    }

 private:
    std::shared_ptr<cardinalv2::LoadConfig> load_config_;
    milvus_storage::ArrowFileSystemPtr fs_;
    ChunkSpec spec_;
    milvus::S3ReadPathConfig stream_config_;
};

class ChunkSetMeta : public cardinalv2::DeserializerInterface {
 public:
    ChunkSetMeta(std::shared_ptr<cardinalv2::LoadConfig> load_config,
                 milvus_storage::ArrowFileSystemPtr fs,
                 std::vector<ChunkSpec> specs,
                 milvus::S3ReadPathConfig stream_config)
        : load_config_(std::move(load_config)),
          fs_(std::move(fs)),
          specs_(std::move(specs)),
          stream_config_(std::move(stream_config)) {
    }

    cardinalv2::LoadConfig&
    GetLoadConfig() const override {
        return *load_config_;
    }

    void
    SetLoadConfig(const cardinalv2::LoadConfig& load_config) override {
        *load_config_ = load_config;
    }

    std::shared_ptr<milvus::InputStream>
    GetInputStreamForTensor(const std::string& key) const override {
        throw std::runtime_error("root has no tensor: " + key);
    }

    bool
    HasKV(const std::string&) const override {
        return false;
    }

    bool
    HasChild(const std::string& key) const override {
        const auto cid = ParseCid(key);
        return cid < specs_.size();
    }

    bool
    HasTensor(const std::string&) const override {
        return false;
    }

    std::shared_ptr<cardinalv2::DeserializerInterface>
    GetChildPtr(const std::string& key) const override {
        const auto cid = ParseCid(key);
        if (cid >= specs_.size()) {
            throw std::runtime_error("cid not found: " + key);
        }
        return std::make_shared<ChunkTensorMeta>(
            load_config_, fs_, specs_[cid], stream_config_);
    }

    std::string
    GetKVString(const std::string& key) const override {
        throw std::runtime_error("string key not found: " + key);
    }

    int32_t
    GetKVI32(const std::string& key) const override {
        throw std::runtime_error("int32 key not found: " + key);
    }
    uint32_t
    GetKVU32(const std::string& key) const override {
        throw std::runtime_error("uint32 key not found: " + key);
    }
    int64_t
    GetKVI64(const std::string& key) const override {
        throw std::runtime_error("int64 key not found: " + key);
    }
    uint64_t
    GetKVU64(const std::string& key) const override {
        throw std::runtime_error("uint64 key not found: " + key);
    }
    bool
    GetKVBool(const std::string& key) const override {
        throw std::runtime_error("bool key not found: " + key);
    }
    float
    GetKVF32(const std::string& key) const override {
        throw std::runtime_error("float key not found: " + key);
    }
    std::vector<uint8_t>
    GetKVVecU8(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }
    std::vector<uint16_t>
    GetKVVecU16(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }
    std::vector<int32_t>
    GetKVVecI32(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }
    std::vector<float>
    GetKVVecF32(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }
    std::vector<uint32_t>
    GetKVVecU32(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }
    std::vector<uint64_t>
    GetKVVecU64(const std::string& key) const override {
        throw std::runtime_error("vec key not found: " + key);
    }
    std::unordered_map<int32_t, int32_t>
    GetKVMapI32I32(const std::string& key) const override {
        throw std::runtime_error("map key not found: " + key);
    }
    std::unordered_map<int64_t, uint64_t>
    GetKVMapI64U64(const std::string& key) const override {
        throw std::runtime_error("map key not found: " + key);
    }
    std::unordered_map<int64_t, uint32_t>
    GetKVMapI64U32(const std::string& key) const override {
        throw std::runtime_error("map key not found: " + key);
    }
    void
    GetKVTensor(const std::string& key, std::byte*) const override {
        throw std::runtime_error("tensor key not found: " + key);
    }
    void
    GetKVBlobWithoutSize(const std::string& key,
                         std::byte*,
                         const size_t) const override {
        throw std::runtime_error("blob key not found: " + key);
    }

 private:
    static size_t
    ParseCid(const std::string& key) {
        try {
            size_t parsed = 0;
            const auto value = std::stoull(key, &parsed, 10);
            if (parsed != key.size()) {
                throw std::invalid_argument("trailing characters");
            }
            return static_cast<size_t>(value);
        } catch (const std::exception& e) {
            throw std::runtime_error("invalid cid: " + key + " (" + e.what() +
                                     ")");
        }
    }

    std::shared_ptr<cardinalv2::LoadConfig> load_config_;
    milvus_storage::ArrowFileSystemPtr fs_;
    std::vector<ChunkSpec> specs_;
    milvus::S3ReadPathConfig stream_config_;
};

const char*
GetEnvAny(std::initializer_list<const char*> names) {
    for (const auto* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return value;
        }
    }
    return nullptr;
}

bool
ParseBool(const std::string& value) {
    return value == "1" || value == "true" || value == "TRUE" ||
           value == "True" || value == "yes" || value == "YES";
}

uint64_t
ParseUint64(const std::string& value, const std::string& name) {
    try {
        size_t parsed = 0;
        auto result = std::stoull(value, &parsed, 10);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception& e) {
        throw std::invalid_argument("invalid unsigned integer for " + name +
                                    ": " + value + " (" + e.what() + ")");
    }
}

double
ParseDouble(const std::string& value, const std::string& name) {
    try {
        size_t parsed = 0;
        auto result = std::stod(value, &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return result;
    } catch (const std::exception& e) {
        throw std::invalid_argument("invalid double for " + name + ": " +
                                    value + " (" + e.what() + ")");
    }
}

void
SetFromEnv(Options& options) {
    options.storage.storage_type = "remote";
    options.storage.cloud_provider = "aws";
    options.storage.request_timeout_ms = 10000;
    options.storage.max_connections = 100;

    if (const char* value = GetEnvAny({"MINIO_ADDRESS", "ADDRESS"})) {
        options.storage.address = value;
    }
    if (const char* value =
            GetEnvAny({"MINIO_ACCESS_KEY_ID", "ACCESS_KEY_ID", "ACCESS_KEY"})) {
        options.storage.access_key_id = value;
    }
    if (const char* value = GetEnvAny(
            {"MINIO_SECRET_ACCESS_KEY", "SECRET_ACCESS_KEY", "SECRET_KEY"})) {
        options.storage.access_key_value = value;
    }
    if (const char* value =
            GetEnvAny({"MINIO_BUCKET_NAME", "BUCKET_NAME", "BUCKET"})) {
        options.storage.bucket_name = value;
    }
    if (const char* value = GetEnvAny({"MINIO_ROOT_PATH", "ROOT_PATH"})) {
        options.storage.root_path = value;
    }
    if (const char* value =
            GetEnvAny({"COMMON_STORAGE_TYPE", "STORAGE_TYPE"})) {
        options.storage.storage_type = value;
    }
    if (const char* value =
            GetEnvAny({"MINIO_CLOUD_PROVIDER", "CLOUD_PROVIDER"})) {
        options.storage.cloud_provider = value;
    }
    if (const char* value = GetEnvAny({"MINIO_IAM_ENDPOINT", "IAM_ENDPOINT"})) {
        options.storage.iam_endpoint = value;
    }
    if (const char* value = GetEnvAny({"MINIO_REGION", "REGION"})) {
        options.storage.region = value;
    }
    if (const char* value = GetEnvAny({"MINIO_USE_SSL", "USE_SSL"})) {
        options.storage.use_ssl = ParseBool(value);
    }
    if (const char* value = GetEnvAny({"MINIO_USE_IAM", "USE_IAM"})) {
        options.storage.use_iam = ParseBool(value);
    }
    if (const char* value =
            GetEnvAny({"MINIO_USE_VIRTUAL_HOST", "USE_VIRTUAL_HOST"})) {
        options.storage.use_virtual_host = ParseBool(value);
    }
    if (const char* value =
            GetEnvAny({"MINIO_REQUEST_TIMEOUT_MS", "REQUEST_TIMEOUT_MS"})) {
        options.storage.request_timeout_ms =
            static_cast<int64_t>(ParseUint64(value, "request-timeout-ms"));
    }
    if (const char* value =
            GetEnvAny({"MINIO_MAX_CONNECTIONS", "MAX_CONNECTIONS"})) {
        options.storage.max_connections =
            static_cast<uint32_t>(ParseUint64(value, "max-connections"));
    }
    if (const char* value =
            GetEnvAny({"MINIO_TLS_MIN_VERSION", "TLS_MIN_VERSION"})) {
        options.storage.tls_min_version = value;
    }
    if (const char* value =
            GetEnvAny({"MINIO_USE_CRC32C_CHECKSUM", "USE_CRC32C_CHECKSUM"})) {
        options.storage.use_crc32c_checksum = ParseBool(value);
    }
}

void
PrintUsage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --object-path PATH [options]\n"
              << "\nOptions:\n"
              << "  --workload-file PATH  CSV/TSV lines: object_path,offset,size\n"
              << "  --mode baseline|curl_multi|crt\n"
              << "  --duration-sec N\n"
              << "  --warmup-sec N\n"
              << "  --concurrency N\n"
              << "  --range-size BYTES\n"
              << "  --max-inflight N\n"
              << "  --fetch-pool-size N\n"
              << "  --eventloops N\n"
              << "  --crt-max-connections N\n"
              << "  --crt-throughput-gbps X\n"
              << "  --pattern sequential|random\n"
              << "  --scheduler batch|steady\n"
              << "  --repeat-workload-until-deadline true|false\n"
              << "  --address HOST:PORT\n"
              << "  --bucket-name NAME\n"
              << "  --access-key-id KEY\n"
              << "  --secret-access-key KEY\n"
              << "  --root-path PREFIX\n"
              << "  --storage-type remote|local\n"
              << "  --cloud-provider aws|gcp|aliyun|tencent|huawei\n"
              << "  --region REGION\n"
              << "  --use-ssl true|false\n"
              << "  --use-iam true|false\n"
              << "  --use-virtual-host true|false\n"
              << "  --request-timeout-ms N\n"
              << "  --max-connections N\n";
}

std::string
RequireValue(int& index, int argc, char** argv, const std::string& name) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + name);
    }
    return argv[++index];
}

Options
ParseArgs(int argc, char** argv) {
    Options options;
    SetFromEnv(options);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--object-path" || arg == "--path") {
            options.object_path = RequireValue(i, argc, argv, arg);
        } else if (arg == "--workload-file") {
            options.workload_file = RequireValue(i, argc, argv, arg);
        } else if (arg == "--mode") {
            options.mode = RequireValue(i, argc, argv, arg);
        } else if (arg == "--duration-sec") {
            options.duration_sec =
                ParseUint64(RequireValue(i, argc, argv, arg), arg);
        } else if (arg == "--warmup-sec") {
            options.warmup_sec =
                ParseUint64(RequireValue(i, argc, argv, arg), arg);
        } else if (arg == "--concurrency") {
            options.concurrency = static_cast<size_t>(
                ParseUint64(RequireValue(i, argc, argv, arg), arg));
        } else if (arg == "--range-size") {
            options.range_size = static_cast<size_t>(
                ParseUint64(RequireValue(i, argc, argv, arg), arg));
        } else if (arg == "--max-inflight") {
            options.max_inflight = static_cast<size_t>(
                ParseUint64(RequireValue(i, argc, argv, arg), arg));
        } else if (arg == "--fetch-pool-size") {
            options.fetch_pool_size = static_cast<size_t>(
                ParseUint64(RequireValue(i, argc, argv, arg), arg));
        } else if (arg == "--eventloops") {
            options.event_loops = static_cast<size_t>(
                ParseUint64(RequireValue(i, argc, argv, arg), arg));
        } else if (arg == "--crt-max-connections") {
            options.crt_max_connections = static_cast<size_t>(
                ParseUint64(RequireValue(i, argc, argv, arg), arg));
        } else if (arg == "--crt-throughput-gbps") {
            options.crt_throughput_gbps =
                ParseDouble(RequireValue(i, argc, argv, arg), arg);
        } else if (arg == "--pattern") {
            options.pattern = RequireValue(i, argc, argv, arg);
        } else if (arg == "--scheduler") {
            options.scheduler = RequireValue(i, argc, argv, arg);
        } else if (arg == "--repeat-workload-until-deadline") {
            options.repeat_workload_until_deadline =
                ParseBool(RequireValue(i, argc, argv, arg));
        } else if (arg == "--address") {
            options.storage.address = RequireValue(i, argc, argv, arg);
        } else if (arg == "--bucket-name") {
            options.storage.bucket_name = RequireValue(i, argc, argv, arg);
        } else if (arg == "--access-key-id") {
            options.storage.access_key_id = RequireValue(i, argc, argv, arg);
        } else if (arg == "--secret-access-key") {
            options.storage.access_key_value = RequireValue(i, argc, argv, arg);
        } else if (arg == "--root-path") {
            options.storage.root_path = RequireValue(i, argc, argv, arg);
        } else if (arg == "--storage-type") {
            options.storage.storage_type = RequireValue(i, argc, argv, arg);
        } else if (arg == "--cloud-provider") {
            options.storage.cloud_provider = RequireValue(i, argc, argv, arg);
        } else if (arg == "--iam-endpoint") {
            options.storage.iam_endpoint = RequireValue(i, argc, argv, arg);
        } else if (arg == "--region") {
            options.storage.region = RequireValue(i, argc, argv, arg);
        } else if (arg == "--use-ssl") {
            options.storage.use_ssl =
                ParseBool(RequireValue(i, argc, argv, arg));
        } else if (arg == "--use-iam") {
            options.storage.use_iam =
                ParseBool(RequireValue(i, argc, argv, arg));
        } else if (arg == "--use-virtual-host") {
            options.storage.use_virtual_host =
                ParseBool(RequireValue(i, argc, argv, arg));
        } else if (arg == "--request-timeout-ms") {
            options.storage.request_timeout_ms = static_cast<int64_t>(
                ParseUint64(RequireValue(i, argc, argv, arg), arg));
        } else if (arg == "--max-connections") {
            options.storage.max_connections = static_cast<uint32_t>(
                ParseUint64(RequireValue(i, argc, argv, arg), arg));
        } else if (arg == "--tls-min-version") {
            options.storage.tls_min_version = RequireValue(i, argc, argv, arg);
        } else if (arg == "--use-crc32c-checksum") {
            options.storage.use_crc32c_checksum =
                ParseBool(RequireValue(i, argc, argv, arg));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.object_path.empty() && options.workload_file.empty()) {
        throw std::invalid_argument(
            "--object-path or --workload-file is required");
    }
    if (options.mode != "baseline" && options.mode != "crt" &&
        options.mode != "curl_multi") {
        throw std::invalid_argument(
            "--mode must be baseline, crt, or curl_multi");
    }
    if (options.pattern != "sequential" && options.pattern != "random") {
        throw std::invalid_argument("--pattern must be sequential or random");
    }
    if (options.scheduler != "batch" && options.scheduler != "steady") {
        throw std::invalid_argument("--scheduler must be batch or steady");
    }
    if (options.duration_sec == 0 || options.concurrency == 0 ||
        options.range_size == 0 || options.max_inflight == 0 ||
        options.fetch_pool_size == 0) {
        throw std::invalid_argument(
            "duration, concurrency, range-size, max-inflight, and fetch-pool-size must be > 0");
    }
    if (options.warmup_sec >= options.duration_sec) {
        throw std::invalid_argument(
            "--warmup-sec must be smaller than --duration-sec");
    }
    return options;
}

void
UpdateMin(std::atomic<uint64_t>& target, uint64_t value) {
    auto current = target.load(std::memory_order_relaxed);
    while (value < current &&
           !target.compare_exchange_weak(current,
                                         value,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

void
UpdateMax(std::atomic<uint64_t>& target, uint64_t value) {
    auto current = target.load(std::memory_order_relaxed);
    while (value > current &&
           !target.compare_exchange_weak(current,
                                         value,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

void
RecordLatency(Stats& stats, uint64_t latency_us) {
    stats.latency_us_sum.fetch_add(latency_us, std::memory_order_relaxed);
    UpdateMin(stats.latency_us_min, latency_us);
    UpdateMax(stats.latency_us_max, latency_us);
}

void
RecordRangeTiming(Stats& stats, const RangeTiming& timing) {
    std::lock_guard<std::mutex> lock(stats.phase_mutex);
    stats.open_us.push_back(timing.open_us);
    stats.alloc_us.push_back(timing.alloc_us);
    stats.submit_us.push_back(timing.submit_us);
    stats.wait_us.push_back(timing.wait_us);
    stats.post_us.push_back(timing.post_us);
    stats.total_us.push_back(timing.total_us);
}

uint64_t
ElapsedMicros(Clock::time_point start, Clock::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());
}

class OffsetGenerator {
 public:
    OffsetGenerator(size_t file_size, size_t range_size, std::string pattern)
        : file_size_(file_size),
          range_size_(range_size),
          pattern_(std::move(pattern)),
          slot_count_((file_size - range_size) / range_size + 1) {
    }

    size_t
    Next(uint64_t seed) {
        if (pattern_ == "random") {
            thread_local std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
            std::uniform_int_distribution<size_t> dist(0, slot_count_ - 1);
            return dist(rng) * range_size_;
        }
        auto slot =
            cursor_.fetch_add(1, std::memory_order_relaxed) % slot_count_;
        return slot * range_size_;
    }

 private:
    size_t file_size_;
    size_t range_size_;
    std::string pattern_;
    size_t slot_count_;
    std::atomic<uint64_t> cursor_{0};
};

std::shared_ptr<milvus::storage::RemoteInputStream>
OpenStream(const milvus_storage::ArrowFileSystemPtr& fs,
           const std::string& object_path,
           const milvus::S3ReadPathConfig& config = {}) {
    auto result = fs->OpenInputFile(object_path);
    if (!result.ok()) {
        throw std::runtime_error("OpenInputFile failed: " +
                                 result.status().ToString());
    }
    return std::make_shared<milvus::storage::RemoteInputStream>(
        std::move(result.ValueOrDie()), config);
}

milvus::S3ReadPathConfig
MakeReadPathConfig(const Options& options) {
    milvus::S3ReadPathConfig config;
    if (options.mode == "baseline") {
        return config;
    }
    config.override_enabled = true;
    config.mode = options.mode;
    config.max_inflight = options.max_inflight;
    config.event_loops = options.event_loops;
    if (options.mode == "crt") {
        config.crt_max_connections = options.crt_max_connections;
        config.crt_throughput_gbps = options.crt_throughput_gbps;
    }
    return config;
}

std::vector<ChunkSpec>
LoadWorkloadSpecs(const std::string& workload_file) {
    std::ifstream input(workload_file);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open workload file: " +
                                 workload_file);
    }

    std::vector<ChunkSpec> specs;
    std::string line;
    size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream fields(line);
        ChunkSpec spec;
        if (!(fields >> spec.object_path >> spec.offset >> spec.size)) {
            throw std::runtime_error("invalid workload line " +
                                     std::to_string(line_no));
        }
        if (spec.object_path.empty() || spec.size == 0) {
            throw std::runtime_error("invalid workload values on line " +
                                     std::to_string(line_no));
        }
        specs.push_back(std::move(spec));
    }
    if (specs.empty()) {
        throw std::runtime_error("workload file has no records: " +
                                 workload_file);
    }
    return specs;
}

std::vector<ChunkSpec>
BuildChunkSpecs(const Options& options, size_t file_size) {
    if (!options.workload_file.empty()) {
        return LoadWorkloadSpecs(options.workload_file);
    }

    std::vector<ChunkSpec> specs;
    specs.reserve(options.concurrency);

    OffsetGenerator offsets(file_size, options.range_size, options.pattern);
    for (size_t i = 0; i < options.concurrency; ++i) {
        specs.push_back(ChunkSpec{
            .object_path = options.object_path,
            .offset = offsets.Next(i + file_size),
            .size = options.range_size,
        });
    }
    return specs;
}

void
RunOneSpecBatch(const Options& options,
                const milvus_storage::ArrowFileSystemPtr& fs,
                const std::shared_ptr<cardinalv2::LoadConfig>& load_config,
                const milvus::S3ReadPathConfig& s3_config,
                const std::vector<ChunkSpec>& specs,
                size_t begin,
                size_t end,
                Stats& stats) {
    std::vector<ChunkSpec> batch_specs;
    batch_specs.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        batch_specs.push_back(specs[i]);
    }

    auto chunks_meta =
        std::make_shared<ChunkSetMeta>(load_config, fs, batch_specs, s3_config);
    cardinalv2::storage::ChunkTranslator<
        cardinalv2::storage::FixedLengthMemStorage>
        translator(batch_specs.size(), chunks_meta);

    std::vector<milvus::cachinglayer::cid_t> cids;
    cids.reserve(batch_specs.size());
    for (size_t i = 0; i < batch_specs.size(); ++i) {
        cids.push_back(static_cast<milvus::cachinglayer::cid_t>(i));
    }

    milvus::OpContext op_context;
    op_context.s3_read_path_config = s3_config;
    milvus::OpContext* op_context_ptr =
        options.mode == "baseline" ? nullptr : &op_context;

    const auto start = Clock::now();
    auto cells = translator.get_cells(op_context_ptr, cids);
    const auto end_time = Clock::now();
    if (cells.size() != batch_specs.size()) {
        throw std::runtime_error("unexpected cell count");
    }
    uint64_t bytes_read = 0;
    for (size_t i = 0; i < cells.size(); ++i) {
        if (cells[i].second == nullptr ||
            cells[i].second->Capacity() != batch_specs[i].size) {
            throw std::runtime_error("unexpected cell capacity");
        }
        bytes_read += cells[i].second->Capacity();
    }
    stats.batches.fetch_add(1, std::memory_order_relaxed);
    stats.requests.fetch_add(cells.size(), std::memory_order_relaxed);
    stats.bytes.fetch_add(bytes_read, std::memory_order_relaxed);
    RecordLatency(stats, ElapsedMicros(start, end_time));
}

void
RunCardinalChunkTranslator(const Options& options,
                           const milvus_storage::ArrowFileSystemPtr& fs,
                           size_t file_size,
                           Stats& stats,
                           Clock::time_point deadline) {
    auto load_config = std::make_shared<cardinalv2::LoadConfig>();
    load_config->index_prefix = "/tmp/s3_read_microbench";
    load_config->tenant_id = 0;
    load_config->enable_mmap = false;
    load_config->warmup = "disable";

    auto s3_config = MakeReadPathConfig(options);
    auto specs = BuildChunkSpecs(options, file_size);

    if (!options.workload_file.empty()) {
        for (size_t begin = 0; begin < specs.size();
             begin += options.concurrency) {
            const auto end = std::min(begin + options.concurrency,
                                      specs.size());
            try {
                RunOneSpecBatch(options,
                                fs,
                                load_config,
                                s3_config,
                                specs,
                                begin,
                                end,
                                stats);
            } catch (const std::exception& e) {
                stats.errors.fetch_add(1, std::memory_order_relaxed);
                if (stats.errors.load(std::memory_order_relaxed) <= 10) {
                    std::cerr << "cardinal get_cells failed: " << e.what()
                              << std::endl;
                }
            }
        }
        return;
    }

    while (Clock::now() < deadline) {
        try {
            RunOneSpecBatch(options,
                            fs,
                            load_config,
                            s3_config,
                            specs,
                            0,
                            specs.size(),
                            stats);
        } catch (const std::exception& e) {
            stats.errors.fetch_add(1, std::memory_order_relaxed);
            if (stats.errors.load(std::memory_order_relaxed) <= 10) {
                std::cerr << "cardinal get_cells failed: " << e.what()
                          << std::endl;
            }
        }
    }
}

RangeTiming
RunOneRangeSteady(const Options& options,
                  const milvus_storage::ArrowFileSystemPtr& fs,
                  const milvus::S3ReadPathConfig& s3_config,
                  const ChunkSpec& spec,
                  uint64_t& bytes_read) {
    RangeTiming timing;
    const auto total_start = Clock::now();

    auto open_start = Clock::now();
    auto stream = OpenStream(fs, spec.object_path, s3_config);
    if (options.mode != "baseline") {
        stream->ConfigureAsyncReadAtLimiter(0, options.max_inflight);
    }
    auto open_done = Clock::now();
    timing.open_us = ElapsedMicros(open_start, open_done);

    auto alloc_start = Clock::now();
    std::vector<std::byte> buffer(spec.size);
    auto alloc_done = Clock::now();
    timing.alloc_us = ElapsedMicros(alloc_start, alloc_done);

    if (options.mode == "baseline") {
        auto submit_start = Clock::now();
        size_t read_result = 0;
        std::vector<std::function<void()>> tasks;
        tasks.emplace_back([&]() {
            read_result = stream->ReadAt(buffer.data(), spec.offset, spec.size);
        });
        auto submit_done = Clock::now();
        timing.submit_us = ElapsedMicros(submit_start, submit_done);

        cardinalv2::ParallelFetchTask(tasks);
        auto wait_done = Clock::now();
        timing.wait_us = ElapsedMicros(submit_done, wait_done);
        bytes_read = read_result;
    } else {
        auto submit_start = Clock::now();
        auto future = stream->ReadAtAsyncInto(
            spec.offset, spec.size, buffer.data(), s3_config);
        auto submit_done = Clock::now();
        timing.submit_us = ElapsedMicros(submit_start, submit_done);

        auto result = future.get();
        auto wait_done = Clock::now();
        timing.wait_us = ElapsedMicros(submit_done, wait_done);
        bytes_read = result.bytes_read;
    }

    auto post_start = Clock::now();
    if (bytes_read != spec.size) {
        throw std::runtime_error("short range read: expected " +
                                 std::to_string(spec.size) + ", got " +
                                 std::to_string(bytes_read));
    }
    auto post_done = Clock::now();
    timing.post_us = ElapsedMicros(post_start, post_done);
    timing.total_us = ElapsedMicros(total_start, post_done);
    return timing;
}

void
RunSteadyReplay(const Options& options,
                const milvus_storage::ArrowFileSystemPtr& fs,
                size_t file_size,
                Stats& stats,
                Clock::time_point deadline,
                Clock::time_point measure_start) {
    auto s3_config = MakeReadPathConfig(options);
    std::vector<ChunkSpec> workload_specs;
    if (!options.workload_file.empty()) {
        workload_specs = LoadWorkloadSpecs(options.workload_file);
    }

    std::atomic<size_t> next_index{0};
    std::unique_ptr<OffsetGenerator> offsets;
    if (workload_specs.empty()) {
        offsets = std::make_unique<OffsetGenerator>(
            file_size, options.range_size, options.pattern);
    }

    auto next_spec = [&](ChunkSpec& spec) -> bool {
        if (!workload_specs.empty()) {
            if (options.repeat_workload_until_deadline &&
                Clock::now() >= deadline) {
                return false;
            }
            auto index = next_index.fetch_add(1, std::memory_order_relaxed);
            if (!options.repeat_workload_until_deadline &&
                index >= workload_specs.size()) {
                return false;
            }
            spec = workload_specs[index % workload_specs.size()];
            return true;
        }

        if (Clock::now() >= deadline) {
            return false;
        }
        auto index = next_index.fetch_add(1, std::memory_order_relaxed);
        spec = ChunkSpec{
            .object_path = options.object_path,
            .offset = offsets->Next(index + file_size),
            .size = options.range_size,
        };
        return true;
    };

    std::vector<std::thread> workers;
    workers.reserve(options.concurrency);
    for (size_t i = 0; i < options.concurrency; ++i) {
        workers.emplace_back([&, i]() {
            ChunkSpec spec;
            while (next_spec(spec)) {
                try {
                    uint64_t bytes_read = 0;
                    auto timing =
                        RunOneRangeSteady(options, fs, s3_config, spec, bytes_read);
                    if (Clock::now() >= measure_start) {
                        stats.requests.fetch_add(1, std::memory_order_relaxed);
                        stats.bytes.fetch_add(bytes_read, std::memory_order_relaxed);
                        RecordLatency(stats, timing.total_us);
                        RecordRangeTiming(stats, timing);
                    }
                } catch (const std::exception& e) {
                    auto errors =
                        stats.errors.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (errors <= 10) {
                        std::cerr << "steady range worker " << i
                                  << " failed: " << e.what() << std::endl;
                    }
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }
    stats.batches.store(1, std::memory_order_relaxed);
}

void
PrintResult(const Options& options,
            size_t file_size,
            double elapsed_sec,
            const Stats& stats) {
    const auto batches = stats.batches.load(std::memory_order_relaxed);
    const auto requests = stats.requests.load(std::memory_order_relaxed);
    const auto bytes = stats.bytes.load(std::memory_order_relaxed);
    const auto errors = stats.errors.load(std::memory_order_relaxed);
    const auto min_latency =
        stats.latency_us_min.load(std::memory_order_relaxed);
    const auto max_latency =
        stats.latency_us_max.load(std::memory_order_relaxed);
    const auto latency_count =
        options.scheduler == "steady" ? requests : batches;
    const auto avg_latency =
        latency_count == 0 ? 0.0
                           : static_cast<double>(stats.latency_us_sum.load(
                                 std::memory_order_relaxed)) /
                                 static_cast<double>(latency_count);
    const auto mib_per_sec =
        elapsed_sec <= 0.0
            ? 0.0
            : static_cast<double>(bytes) / 1024.0 / 1024.0 / elapsed_sec;

    auto json_escape = [](const std::string& value) {
        std::string out;
        out.reserve(value.size());
        for (char c : value) {
            switch (c) {
                case '\\':
                    out += "\\\\";
                    break;
                case '"':
                    out += "\\\"";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    out += c;
                    break;
            }
        }
        return out;
    };

    struct PhaseCopies {
        std::vector<uint64_t> open_us;
        std::vector<uint64_t> alloc_us;
        std::vector<uint64_t> submit_us;
        std::vector<uint64_t> wait_us;
        std::vector<uint64_t> post_us;
        std::vector<uint64_t> total_us;
    } phases;
    {
        std::lock_guard<std::mutex> lock(stats.phase_mutex);
        phases.open_us = stats.open_us;
        phases.alloc_us = stats.alloc_us;
        phases.submit_us = stats.submit_us;
        phases.wait_us = stats.wait_us;
        phases.post_us = stats.post_us;
        phases.total_us = stats.total_us;
    }

    auto phase_json = [](std::vector<uint64_t> values) {
        std::ostringstream out;
        if (values.empty()) {
            out << "{\"count\":0}";
            return out.str();
        }
        std::sort(values.begin(), values.end());
        auto percentile = [&](size_t pct) {
            const auto index = ((values.size() - 1) * pct) / 100;
            return values[index];
        };
        const auto sum = std::accumulate(
            values.begin(), values.end(), static_cast<uint64_t>(0));
        const auto avg =
            static_cast<double>(sum) / static_cast<double>(values.size());
        out << "{\"count\":" << values.size()
            << ",\"sum\":" << sum
            << ",\"avg\":" << avg
            << ",\"p50\":" << percentile(50)
            << ",\"p95\":" << percentile(95)
            << ",\"p99\":" << percentile(99)
            << ",\"max\":" << values.back() << "}";
        return out.str();
    };
    const auto entry =
        options.scheduler == "batch"
            ? "ChunkTranslator::get_cells"
            : "steady_range_worker";

    std::cout << "{"
              << "\"mode\":\"" << options.mode << "\","
              << "\"object_path\":\"" << json_escape(options.object_path)
              << "\","
              << "\"workload_file\":\"" << json_escape(options.workload_file)
              << "\","
              << "\"file_size\":" << file_size << ","
              << "\"duration_sec\":" << elapsed_sec << ","
              << "\"warmup_sec\":" << options.warmup_sec << ","
              << "\"repeat_workload_until_deadline\":"
              << (options.repeat_workload_until_deadline ? "true" : "false")
              << ","
              << "\"concurrency\":" << options.concurrency << ","
              << "\"range_size\":" << options.range_size << ","
              << "\"max_inflight\":" << options.max_inflight << ","
              << "\"fetch_pool_size\":"
              << knowhere::ThreadPool::GetGlobalFetchThreadPoolSize() << ","
              << "\"eventloops\":" << options.event_loops << ","
              << "\"crt_max_connections\":" << options.crt_max_connections
              << ","
              << "\"crt_throughput_gbps\":" << options.crt_throughput_gbps
              << ","
              << "\"pattern\":\"" << options.pattern << "\","
              << "\"scheduler\":\"" << options.scheduler << "\","
              << "\"cardinal_entry\":\"" << entry << "\","
              << "\"batches_completed\":" << batches << ","
              << "\"requests_completed\":" << requests << ","
              << "\"bytes_read\":" << bytes << ","
              << "\"mib_per_sec\":" << mib_per_sec << ","
              << "\"errors\":" << errors << ","
              << "\"latency_us_min\":"
              << (latency_count == 0 ? 0 : min_latency)
              << ","
              << "\"latency_us_avg\":" << avg_latency << ","
              << "\"latency_us_max\":" << max_latency << ","
              << "\"phase_open_us\":" << phase_json(std::move(phases.open_us))
              << ","
              << "\"phase_alloc_us\":"
              << phase_json(std::move(phases.alloc_us)) << ","
              << "\"phase_submit_us\":"
              << phase_json(std::move(phases.submit_us)) << ","
              << "\"phase_wait_us\":" << phase_json(std::move(phases.wait_us))
              << ","
              << "\"phase_post_us\":" << phase_json(std::move(phases.post_us))
              << ","
              << "\"phase_total_us\":"
              << phase_json(std::move(phases.total_us)) << "}" << std::endl;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        auto options = ParseArgs(argc, argv);
        knowhere::ThreadPool::SetGlobalFetchThreadPoolSize(
            static_cast<uint32_t>(options.fetch_pool_size));
        auto fs_result = milvus_storage::CreateArrowFileSystem(options.storage);
        if (!fs_result.ok()) {
            throw std::runtime_error("CreateArrowFileSystem failed: " +
                                     fs_result.status().ToString());
        }
        auto fs = fs_result.ValueOrDie();
        size_t file_size = 0;
        if (options.workload_file.empty()) {
            auto probe_stream = OpenStream(fs, options.object_path);
            file_size = probe_stream->Size();
            if (file_size < options.range_size) {
                throw std::invalid_argument(
                    "range-size is larger than object size");
            }
            probe_stream.reset();
        }

        Stats stats;
        const auto start = Clock::now();
        const auto deadline =
            start + std::chrono::seconds(options.duration_sec);
        const auto measure_start =
            start + std::chrono::seconds(options.warmup_sec);

        if (options.scheduler == "steady") {
            RunSteadyReplay(options, fs, file_size, stats, deadline, measure_start);
        } else {
            RunCardinalChunkTranslator(options, fs, file_size, stats, deadline);
        }

        const auto end = Clock::now();
        const auto elapsed_sec =
            std::chrono::duration<double>(end - measure_start).count();
        PrintResult(options, file_size, elapsed_sec, stats);
        return stats.errors.load(std::memory_order_relaxed) == 0 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "s3_read_microbench failed: " << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }
}
