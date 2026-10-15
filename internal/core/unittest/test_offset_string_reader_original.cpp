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
            auto record = nlohmann::json::parse(line);
            texts_.push_back(record.at("text").get<std::string>());
            ids_.push_back(record.at("original_id").get<std::string>());
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
    Column(bool irregular, size_t regular_rows = 1024,
           bool known_backing_sizes = true) {
        std::vector<std::unique_ptr<GroupChunk>> chunks;
        std::vector<int64_t> lengths;
        std::vector<int64_t> backing_bytes;
        for (size_t first = 0; first < texts_.size();) {
            const auto count = std::min(
                texts_.size() - first,
                irregular ? size_t(997 + lengths.size() % 7) : regular_rows);
            auto make_chunk = [&](const std::vector<std::string>& values) {
                size_t bytes = (count + 1) * sizeof(uint32_t);
                for (size_t i = 0; i < count; ++i) bytes += values[first + i].size();
                buffers_.emplace_back(bytes);
                auto& buffer = buffers_.back();
                auto* offsets = reinterpret_cast<uint32_t*>(buffer.data());
                size_t cursor = (count + 1) * sizeof(uint32_t);
                for (size_t i = 0; i < count; ++i) {
                    offsets[i] = cursor;
                    std::memcpy(buffer.data() + cursor, values[first + i].data(),
                                values[first + i].size());
                    cursor += values[first + i].size();
                }
                offsets[count] = cursor;
                return std::make_shared<StringChunk>(count, buffer.data(), bytes,
                                                     false, nullptr);
            };
            std::unordered_map<FieldId, std::shared_ptr<Chunk>> fields;
            fields.emplace(FieldId(100), make_chunk(texts_));
            fields.emplace(FieldId(101), make_chunk(ids_));
            chunks.push_back(std::make_unique<GroupChunk>(fields));
            backing_bytes.push_back(chunks.back()->Size());
            lengths.push_back(count);
            first += count;
        }
        auto translator = std::make_unique<TestGroupChunkTranslator>(
                2,
                lengths,
                "original-nq-string-" + std::to_string(irregular),
                std::move(chunks));
        if (known_backing_sizes) {
            static_cast<segcore::storagev2translator::GroupCTMeta*>(translator->meta())
                ->chunk_memory_size_ = backing_bytes;
        }
        auto group = std::make_shared<ChunkedColumnGroup>(std::move(translator));
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
        EXPECT_LE(reader.peak_bytes(), reader.budget_bytes());
    }

    std::vector<std::string> texts_;
    std::vector<std::string> ids_;
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

TEST_F(OriginalStringOffsetReaderTest, SparseLookupBeyondDenseMetadataBudget) {
    // 65536 original rows, one per chunk: exceeds the bounded dense table.
    // This must exercise the sparse hash fallback, not manufacture new data.
    auto column = Column(false, 1);
    OffsetChunkReader reader(column);
    const std::vector<int32_t> rows{0, 65000, 1, 65535, 32768, 65000, 0};
    Check(reader, rows);
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

TEST_F(OriginalStringOffsetReaderTest, BatchMissesSharePinsAndKeepCallerOrder) {
    auto column = Column(false);
    OffsetChunkReader reader(column);
    std::vector<int32_t> rows;
    for (int32_t i = 0; i < 64; ++i) rows.push_back(i * 1024);
    Check(reader, rows);
    EXPECT_EQ(reader.reads(), 64);
    EXPECT_EQ(reader.misses(), 64);
    EXPECT_LT(reader.bulk_pin_calls(), 16);
    // Releasing one cache cohort never leaves pointers to its other cells.
    std::reverse(rows.begin(), rows.end());
    Check(reader, rows);
    EXPECT_LE(reader.peak_bytes(), OffsetChunkReader::kMaxBytes);
}

TEST_F(OriginalStringOffsetReaderTest, ConsumerFailureReleasesTemporaryPins) {
    auto column = Column(false);
    OffsetChunkReader reader(column);
    const int32_t rows[] = {0, 1024, 2048, 3072};
    EXPECT_THROW(reader.VisitViews(
        nullptr, rows, 4,
        [](Chunk* c, size_t i) { return (*static_cast<StringChunk*>(c))[i]; },
        [](auto&&...) { throw std::runtime_error("consumer failure"); }),
        std::runtime_error);
    Check(reader, {0, 1024, 2048, 3072});
}

TEST_F(OriginalStringOffsetReaderTest, UnknownBackingChargeIsConservative) {
    auto column = Column(false, 1024, false);
    OffsetChunkReader reader(column);
    std::vector<int32_t> rows;
    for (int32_t i = 0; i < 64; ++i) rows.push_back(i * 1024);
    Check(reader, rows);
    EXPECT_EQ(reader.bulk_pin_calls(), 64);
    EXPECT_LE(reader.peak_bytes(), OffsetChunkReader::kMaxBytes);
}

TEST_F(OriginalStringOffsetReaderTest, BulkOwnerChargesOriginalSiblingField) {
    auto column = Column(false);
    auto [owner, bytes] = column->PinOffsetChunks(nullptr, {0, 1});
    ASSERT_EQ(owner.get().size(), 2);
    EXPECT_GT(bytes, owner.get()[0]->Size() + owner.get()[1]->Size());
    EXPECT_EQ(bytes, column->OffsetChunkByteSize(0) + column->OffsetChunkByteSize(1));
}

TEST_F(OriginalStringOffsetReaderTest, ExplicitBudgetsKeepOriginalValues) {
    auto column = Column(false);
    std::vector<int32_t> rows;
    for (int32_t i = 0; i < 65536; i += 17) rows.push_back(i);
    for (size_t mib : {32, 64, 128}) {
        OffsetChunkReader reader(column, mib * 1024 * 1024);
        EXPECT_EQ(reader.budget_bytes(), mib * 1024 * 1024);
        Check(reader, rows);
        auto reverse = rows;
        std::reverse(reverse.begin(), reverse.end());
        Check(reader, reverse);
        EXPECT_EQ(reader.reads(), 2 * rows.size());
    }
}

}  // namespace
}  // namespace milvus::exec
