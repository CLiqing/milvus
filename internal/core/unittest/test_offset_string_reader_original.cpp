// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>
#include <fstream>
#include <thread>

#include "exec/expression/OffsetColumnReader.h"
#include "mmap/ChunkedColumnGroup.h"
#include "storage/LocalChunkManagerSingleton.h"
#include "test_utils/cachinglayer_test_utils.h"

namespace milvus::exec {
namespace {

// The runner supplies unchanged original NQ text. Only chunk layout and
// candidate schedules vary: no generated/truncated/padded scalar values.
class OriginalStringOffsetReaderTest : public testing::Test {
 protected:
    void
    SetUp() override {
        const auto* path = std::getenv("MILVUS_NQ_OFFSET_FIXTURE");
        if (!path) {
            GTEST_SKIP() << "verified original NQ fixture not configured";
        }
        std::ifstream stream(path);
        ASSERT_TRUE(stream.good());
        for (std::string line; std::getline(stream, line);) {
            texts_.push_back(
                nlohmann::json::parse(line).at("text").get<std::string>());
        }
        ASSERT_EQ(texts_.size(), 65536);
        char directory[] = "/tmp/milvus-string-offset-XXXXXX";
        ASSERT_NE(mkdtemp(directory), nullptr);
        storage::LocalChunkManagerSingleton::GetInstance().Init(directory);
        storage::MmapConfig config{};
        config.cache_read_ahead_policy = "random";
        config.mmap_path = directory;
        config.disk_limit = 512 * 1024 * 1024;
        config.fix_file_size = 4 * 1024 * 1024;
        storage::MmapManager::GetInstance().Init(config);
        constexpr int64_t mb = 1024 * 1024;
        cachinglayer::Manager::ConfigureTieredStorage(
            {CacheWarmupPolicy::CacheWarmupPolicy_Disable,
             CacheWarmupPolicy::CacheWarmupPolicy_Disable,
             CacheWarmupPolicy::CacheWarmupPolicy_Disable,
             CacheWarmupPolicy::CacheWarmupPolicy_Disable},
            {512 * mb, 512 * mb, 512 * mb, 512 * mb, 512 * mb, 512 * mb},
            true,
            false,
            {10, true, 30},
            std::chrono::milliseconds(0),
            std::chrono::milliseconds(-1));
    }

    std::shared_ptr<ProxyChunkColumn>
    Column(bool irregular, size_t regular_rows = 1024) {
        std::vector<std::unique_ptr<GroupChunk>> chunks;
        std::vector<int64_t> lengths;
        for (size_t first = 0; first < texts_.size();) {
            const auto count = std::min(
                texts_.size() - first,
                irregular ? size_t(997 + lengths.size() % 7) : regular_rows);
            size_t bytes = (count + 1) * sizeof(uint32_t);
            for (size_t i = 0; i < count; ++i)
                bytes += texts_[first + i].size();
            buffers_.emplace_back(bytes);
            auto& buffer = buffers_.back();
            auto* offsets = reinterpret_cast<uint32_t*>(buffer.data());
            size_t cursor = (count + 1) * sizeof(uint32_t);
            for (size_t i = 0; i < count; ++i) {
                offsets[i] = cursor;
                std::memcpy(buffer.data() + cursor,
                            texts_[first + i].data(),
                            texts_[first + i].size());
                cursor += texts_[first + i].size();
            }
            offsets[count] = cursor;
            std::unordered_map<FieldId, std::shared_ptr<Chunk>> fields;
            fields.emplace(FieldId(100),
                           std::make_shared<StringChunk>(
                               count, buffer.data(), bytes, false, nullptr));
            chunks.push_back(std::make_unique<GroupChunk>(fields));
            lengths.push_back(count);
            first += count;
        }
        auto group = std::make_shared<ChunkedColumnGroup>(
            std::make_unique<TestGroupChunkTranslator>(
                1,
                lengths,
                "original-nq-string-" + std::to_string(irregular),
                std::move(chunks)));
        return std::make_shared<ProxyChunkColumn>(group,
                                                  FieldId(100),
                                                  FieldMeta(FieldName("text"),
                                                            FieldId(100),
                                                            DataType::VARCHAR,
                                                            false,
                                                            std::nullopt));
    }

    void
    Check(OffsetChunkReader& reader, const std::vector<int32_t>& rows) {
        size_t seen = 0;
        reader.VisitViews(
            nullptr,
            rows.data(),
            rows.size(),
            [](Chunk* chunk, size_t offset) {
                return (*static_cast<StringChunk*>(chunk))[offset];
            },
            [&](const std::string_view* values,
                const bool* valid,
                const size_t*,
                size_t count,
                size_t first,
                bool all_valid) {
                EXPECT_EQ(first, seen);
                EXPECT_TRUE(all_valid);
                EXPECT_LE(count, OffsetChunkReader::kBatchRows);
                for (size_t i = 0; i < count; ++i) {
                    EXPECT_TRUE(valid[i]);
                    EXPECT_EQ(values[i], texts_[rows[first + i]]);
                }
                seen += count;
            });
        EXPECT_EQ(seen, rows.size());
        EXPECT_LE(reader.peak_bytes(), OffsetChunkReader::kMaxBytes);
    }

    std::vector<std::string> texts_;
    std::vector<std::vector<char>> buffers_;
};

TEST_F(OriginalStringOffsetReaderTest, OrderedIrregularAndRepeatedOffsets) {
    auto column = Column(true);
    OffsetChunkReader reader(column);
    Check(reader, {});
    std::vector<int32_t> rows;
    for (int32_t i = 0; i < 1025; ++i)
        rows.push_back((i * 997) % texts_.size());
    Check(reader, rows);
    std::reverse(rows.begin(), rows.end());
    Check(reader, rows);
    rows.assign(
        65, rows.front());  // repeated requests, never duplicate source rows
    Check(reader, rows);
}

TEST_F(OriginalStringOffsetReaderTest,
       BoundedPinsSurviveEvictionAndWorkerIsolation) {
    auto column = Column(false);
    std::vector<int32_t> rows;
    for (int32_t i = 0; i < 65536; ++i)
        rows.push_back((i * 997) % texts_.size());
    auto worker = [&] {
        OffsetChunkReader reader(column);
        Check(reader, rows);
        Check(reader, rows);
        EXPECT_GT(reader.misses(), 2 * column->num_chunks());
    };
    std::thread one(worker), two(worker);
    one.join();
    two.join();
}

TEST_F(OriginalStringOffsetReaderTest, OversizedOriginalCellIsBatchScoped) {
    // >=513 original bytes *65536 rows naturally exceeds the retained budget;
    // no padding/generated payload is used to manufacture the oversized cell.
    auto column = Column(false, texts_.size());
    OffsetChunkReader reader(column);
    std::vector<int32_t> rows;
    for (int32_t i = 0; i < 257; ++i) rows.push_back(i);
    Check(reader, rows);
    EXPECT_EQ(reader.misses(), 1);
    EXPECT_EQ(reader.peak_bytes(), 0);  // not retained beyond EndBatch
    Check(reader, rows);
    EXPECT_EQ(reader.misses(), 2);
}

}  // namespace
}  // namespace milvus::exec
