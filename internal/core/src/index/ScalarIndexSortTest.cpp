#include <gtest/gtest.h>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "bitset/bitset.h"
#include "common/Tracer.h"
#include "common/TracerBase.h"
#include "common/Types.h"
#include "gtest/gtest.h"
#include "index/Meta.h"
#include "index/ScalarIndexSort.h"
#include "milvus-storage/filesystem/fs.h"
#include "pb/common.pb.h"
#include "storage/ChunkManager.h"
#include "storage/FileManager.h"
#include "storage/ThreadPools.h"
#include "storage/Types.h"
#include "storage/Util.h"
#include "test_utils/Constants.h"
#include "test_utils/TmpPath.h"
#include "test_utils/storage_test_utils.h"

using namespace milvus;
using namespace milvus::index;

static storage::FileManagerContext
CreateScalarSortTestFileManagerContext() {
    storage::StorageConfig storage_config;
    storage_config.storage_type = "local";
    storage_config.root_path = TestLocalPath;
    auto chunk_manager = storage::CreateChunkManager(storage_config);
    auto fs = storage::InitArrowFileSystem(storage_config);
    storage::FieldDataMeta field_meta{1, 2, 3, 101};
    field_meta.field_schema.set_data_type(proto::schema::DataType::Int64);
    storage::IndexMeta index_meta{3, 101, 1000, 10000};
    storage::FileManagerContext ctx(field_meta, index_meta, chunk_manager, fs);
    return ctx;
}

void
test_stlsort_for_range(
    const std::vector<int64_t>& data,
    DataType data_type,
    bool enable_mmap,
    std::function<TargetBitmap(
        const std::shared_ptr<ScalarIndexSort<int64_t>>&)> exec_expr,
    const std::vector<bool>& expected_result) {
    size_t nb = data.size();
    std::vector<std::string> index_files;
    {
        Config config;

        auto index = std::make_shared<index::ScalarIndexSort<int64_t>>(
            CreateScalarSortTestFileManagerContext());
        index->Build(nb, data.data());

        auto create_index_result = index->UploadUnified({});
        index_files = create_index_result->GetIndexFiles();
    }
    {
        Config config;
        config[milvus::index::ENABLE_MMAP] = enable_mmap;
        config[milvus::LOAD_PRIORITY] =
            milvus::proto::common::LoadPriority::HIGH;
        config["index_files"] = index_files;

        auto index = std::make_shared<index::ScalarIndexSort<int64_t>>(
            CreateScalarSortTestFileManagerContext());
        index->LoadUnified(config);

        auto cnt = index->Count();
        ASSERT_EQ(cnt, nb);
        auto bitset = exec_expr(index);
        for (size_t i = 0; i < nb; i++) {
            ASSERT_EQ(bitset[i], expected_result[i]);
        }
    }
}
TEST(StlSortIndexTest, TestRange) {
    std::vector<int64_t> data = {10, 2, 6, 5, 9, 3, 7, 8, 4, 1};
    {
        std::vector<bool> expected_result = {
            false, false, true, true, false, true, true, false, true, false};
        auto exec_expr =
            [](const std::shared_ptr<ScalarIndexSort<int64_t>>& index) {
                return index->Range(3, true, 7, true);
            };

        test_stlsort_for_range(
            data, DataType::INT64, false, exec_expr, expected_result);

        test_stlsort_for_range(
            data, DataType::INT64, true, exec_expr, expected_result);
    }

    {
        std::vector<bool> expected_result(data.size(), false);
        auto exec_expr =
            [](const std::shared_ptr<ScalarIndexSort<int64_t>>& index) {
                return index->Range(10, false, 70, true);
            };

        test_stlsort_for_range(
            data, DataType::INT64, false, exec_expr, expected_result);

        test_stlsort_for_range(
            data, DataType::INT64, true, exec_expr, expected_result);
    }
}

TEST(StlSortIndexTest, NativeValidIdRangeMatchesDenseRange) {
    // Values deliberately do not follow row offsets: a range's value-order
    // interval must still yield row-offset membership equivalent to Range().
    const std::vector<int64_t> data = {50, 10, 80, 20, 70, 30, 60, 40};
    const bool valid[] = {true, true, false, true, true, false, true, true};
    ScalarIndexSort<int64_t> index;
    index.Build(data.size(), data.data(), valid);

    const auto assert_matches_dense = [](const TargetBitmap& dense,
                                         const std::vector<int32_t>* ids) {
        ASSERT_NE(ids, nullptr);
        ASSERT_EQ(ids->size(), dense.count());
        TargetBitmap observed(dense.size(), false);
        for (const auto id : *ids) {
            ASSERT_GE(id, 0);
            ASSERT_LT(static_cast<size_t>(id), dense.size());
            ASSERT_FALSE(observed[id]);
            observed.set(id);
        }
        for (size_t offset = 0; offset < dense.size(); ++offset) {
            ASSERT_EQ(observed[offset], dense[offset]) << "offset=" << offset;
        }
    };

    const auto greater = index.TryGetValidIdRange(30, OpType::GreaterThan);
    assert_matches_dense(index.Range(30, OpType::GreaterThan), greater.get());
    const auto greater_cardinality =
        index.Range(30, OpType::GreaterThan).count();
    ASSERT_GT(greater_cardinality, 0);
    EXPECT_EQ(index.PreflightValidIdRange(30,
                                         OpType::GreaterThan,
                                         greater_cardinality),
              NativeValidIdPreflight::Fits);
    EXPECT_EQ(index.PreflightValidIdRange(30,
                                         OpType::GreaterThan,
                                         greater_cardinality - 1),
              NativeValidIdPreflight::Exceeds);

    const auto closed = index.TryGetValidIdRange(20, true, 60, true);
    assert_matches_dense(index.Range(20, true, 60, true), closed.get());
    const auto closed_cardinality = index.Range(20, true, 60, true).count();
    EXPECT_EQ(index.PreflightValidIdRange(
                  20, true, 60, true, closed_cardinality),
              NativeValidIdPreflight::Fits);
    EXPECT_EQ(index.PreflightValidIdRange(
                  20, true, 60, true, closed_cardinality - 1),
              NativeValidIdPreflight::Exceeds);

    // A supported but empty range must not be reported as an unsupported
    // producer, because the vector path can short-circuit it safely.
    const auto empty = index.TryGetValidIdRange(80, false, 90, true);
    assert_matches_dense(index.Range(80, false, 90, true), empty.get());
}

TEST(StlSortIndexTest, TestIn) {
    std::vector<int64_t> data = {10, 2, 6, 5, 9, 3, 7, 8, 4, 1};
    std::vector<bool> expected_result = {
        false, false, false, true, false, true, true, false, false, false};

    std::vector<int64_t> values = {3, 5, 7};

    auto exec_expr =
        [&values](const std::shared_ptr<ScalarIndexSort<int64_t>>& index) {
            return index->In(values.size(), values.data());
        };
    test_stlsort_for_range(
        data, DataType::INT64, false, exec_expr, expected_result);

    test_stlsort_for_range(
        data, DataType::INT64, true, exec_expr, expected_result);
}

TEST(StlSortIndexTest, MmapByteSizeCountsValidBitsetOnce) {
    constexpr size_t kAlignment = 32;
    constexpr uint64_t kMmapIndexPadding = 1;
    const std::vector<int64_t> data = {
        10, 2, 6, 5, 9, 3, 7, 8, 4, 1, 11, 12, 13};

    std::vector<std::string> index_files;
    {
        auto index = std::make_shared<index::ScalarIndexSort<int64_t>>(
            CreateScalarSortTestFileManagerContext());
        index->Build(data.size(), data.data());

        auto create_index_result = index->UploadUnified({});
        index_files = create_index_result->GetIndexFiles();
    }

    auto index = std::make_shared<index::ScalarIndexSort<int64_t>>(
        CreateScalarSortTestFileManagerContext());
    Config config;
    config[milvus::index::ENABLE_MMAP] = true;
    config[milvus::LOAD_PRIORITY] = milvus::proto::common::LoadPriority::HIGH;
    config["index_files"] = index_files;
    index->LoadUnified(config);

    auto index_data_bytes = data.size() * sizeof(IndexStructure<int64_t>);
    auto aligned_data_bytes =
        ((index_data_bytes + kAlignment - 1) / kAlignment) * kAlignment;
    TargetBitmap valid_bitset(data.size(), true);
    auto expected_byte_size = aligned_data_bytes + kMmapIndexPadding +
                              data.size() * sizeof(int32_t) +
                              valid_bitset.size_in_bytes();

    ASSERT_EQ(index->ByteSize(), static_cast<int64_t>(expected_byte_size));
}

// V2 compat test removed: kScalarIndexUseV3 flag deleted,
// Upload()/Load() now always route to V3 paths.
