#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <numeric>
#include <random>
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
#include "roaring/roaring.hh"
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

TEST(StlSortIndexTest, NativeRoaringRangeMatchesDenseRange) {
    // Values deliberately do not follow row offsets: a range's value-order
    // interval must still yield row-offset membership equivalent to Range().
    const std::vector<int64_t> data = {50, 10, 80, 20, 70, 30, 60, 40};
    const bool valid[] = {true, true, false, true, true, false, true, true};
    ScalarIndexSort<int64_t> index;
    index.Build(data.size(), data.data(), valid);

    const auto assert_matches_dense = [&index](
                                          const TargetBitmap& dense,
                                          const roaring_bitmap_t* roaring) {
        ASSERT_NE(roaring, nullptr);
        ASSERT_EQ(roaring_bitmap_get_cardinality(roaring), dense.count());
        for (size_t offset = 0; offset < dense.size(); ++offset) {
            ASSERT_EQ(roaring_bitmap_contains(roaring, offset), dense[offset])
                << "offset=" << offset;
        }
    };

    const auto greater = index.TryGetRoaringRange(30, OpType::GreaterThan);
    assert_matches_dense(index.Range(30, OpType::GreaterThan), greater.get());

    const auto closed = index.TryGetRoaringRange(20, true, 60, true);
    assert_matches_dense(index.Range(20, true, 60, true), closed.get());

    const auto native_list = index.TryGetValidIdRange(20, true, 60, true);
    ASSERT_NE(native_list, nullptr);
    const auto dense_closed = index.Range(20, true, 60, true);
    ASSERT_EQ(native_list->size(), dense_closed.count());
    for (const auto id : *native_list) {
        ASSERT_GE(id, 0);
        ASSERT_LT(static_cast<size_t>(id), dense_closed.size());
        ASSERT_TRUE(dense_closed[id]);
    }

    // A supported but empty range must not be reported as an unsupported
    // producer, because the vector path can short-circuit it safely.
    const auto empty = index.TryGetRoaringRange(80, false, 90, true);
    ASSERT_NE(empty, nullptr);
    assert_matches_dense(index.Range(80, false, 90, true), empty.get());
}

TEST(StlSortIndexBenchmark, DISABLED_ProducerDenseVsRoaringRandom1M) {
    // Producer-only microbenchmark.  Scalar values are a permutation of row
    // offsets so the value-ordered range span has random row IDs; this avoids
    // overestimating Roaring construction from a value=row-id layout.
    constexpr size_t kRows = 1'000'000;
    constexpr size_t kWarmups = 5;
    constexpr size_t kSamples = 20;
    std::vector<int64_t> values(kRows);
    std::iota(values.begin(), values.end(), 0);
    std::mt19937_64 rng(1732);
    std::shuffle(values.begin(), values.end(), rng);

    ScalarIndexSort<int64_t> index;
    index.Build(values.size(), values.data());

    for (const double ratio : {0.001, 0.01, 0.10, 0.50}) {
        const auto upper = static_cast<int64_t>(kRows * ratio);
        const auto dense = index.Range(upper, OpType::LessThan);
        const auto roaring = index.TryGetRoaringRange(upper, OpType::LessThan);
        ASSERT_NE(roaring, nullptr);
        ASSERT_EQ(dense.count(), roaring_bitmap_get_cardinality(roaring.get()));

        auto run_dense = [&]() {
            const auto result = index.Range(upper, OpType::LessThan);
            return result.count();
        };
        auto run_roaring = [&]() {
            const auto result = index.TryGetRoaringRange(upper, OpType::LessThan);
            return result == nullptr ? size_t{0}
                                     : roaring_bitmap_get_cardinality(result.get());
        };
        auto run_streaming_bulk = [&]() {
            roaring::Roaring result;
            roaring::BulkContext context;
            const auto* data = index.GetData();
            for (int64_t i = 0; i < upper; ++i) {
                result.addBulk(context, static_cast<uint32_t>(data[i].idx_));
            }
            return result.cardinality();
        };
        for (size_t i = 0; i < kWarmups; ++i) {
            ASSERT_EQ(run_dense(), dense.count());
            ASSERT_EQ(run_roaring(), dense.count());
            ASSERT_EQ(run_streaming_bulk(), dense.count());
        }

        std::vector<double> dense_us;
        std::vector<double> roaring_us;
        std::vector<double> streaming_bulk_us;
        dense_us.reserve(kSamples);
        roaring_us.reserve(kSamples);
        streaming_bulk_us.reserve(kSamples);
        for (size_t sample = 0; sample < kSamples; ++sample) {
            const auto measure = [](auto&& function) {
                const auto start = std::chrono::steady_clock::now();
                const auto cardinality = function();
                const auto end = std::chrono::steady_clock::now();
                return std::make_pair(
                    cardinality,
                    std::chrono::duration<double, std::micro>(end - start)
                        .count());
            };
            const bool dense_first = sample % 2 == 0;
            const auto first = dense_first ? measure(run_dense) : measure(run_roaring);
            const auto second = dense_first ? measure(run_roaring) : measure(run_dense);
            const auto bulk = measure(run_streaming_bulk);
            ASSERT_EQ(first.first, dense.count());
            ASSERT_EQ(second.first, dense.count());
            ASSERT_EQ(bulk.first, dense.count());
            dense_us.push_back(dense_first ? first.second : second.second);
            roaring_us.push_back(dense_first ? second.second : first.second);
            streaming_bulk_us.push_back(bulk.second);
        }
        std::sort(dense_us.begin(), dense_us.end());
        std::sort(roaring_us.begin(), roaring_us.end());
        std::sort(streaming_bulk_us.begin(), streaming_bulk_us.end());
        const auto dense_median = dense_us[dense_us.size() / 2];
        const auto roaring_median = roaring_us[roaring_us.size() / 2];
        const auto streaming_bulk_median =
            streaming_bulk_us[streaming_bulk_us.size() / 2];
        std::cout << "STLSORT producer N=" << kRows << " ratio=" << ratio
                  << " dense_median_us=" << dense_median
                  << " roaring_median_us=" << roaring_median
                  << " roaring_vs_dense_delta="
                  << (dense_median - roaring_median) / dense_median
                  << " streaming_bulk_median_us=" << streaming_bulk_median
                  << " streaming_bulk_vs_dense_delta="
                  << (dense_median - streaming_bulk_median) / dense_median
                  << std::endl;
    }
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
