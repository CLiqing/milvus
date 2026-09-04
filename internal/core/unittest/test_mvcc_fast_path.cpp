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

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/Common.h"
#include "common/Schema.h"
#include "exec/QueryContext.h"
#include "exec/Task.h"
#include "expr/ITypeExpr.h"
#include "index/ScalarIndexSort.h"
#include "pb/plan.pb.h"
#include "plan/PlanNode.h"
#include "segcore/ChunkedSegmentSealedImpl.h"
#include "segcore/SegcoreConfig.h"
#include "segcore/SegmentSealed.h"
#include "segcore/SegmentGrowingImpl.h"
#include "segcore/Types.h"
#include "test_utils/DataGen.h"
#include "test_utils/storage_test_utils.h"

using namespace milvus;
using namespace milvus::exec;
using namespace milvus::segcore;

namespace {

// Deterministic regression seam for the delete-publication race: the segment
// owns a real DeletedRecord and mask_with_delete() sees it, while the earlier
// count observation reports zero.  Sparse MVCC must still apply the mask.
class ZeroReportedDeleteCountSegment final : public ChunkedSegmentSealedImpl {
 public:
    explicit ZeroReportedDeleteCountSegment(SchemaPtr schema)
        : ChunkedSegmentSealedImpl(std::move(schema),
                                   empty_index_meta,
                                   SegcoreConfig::default_config(),
                                   /*segment_id=*/0) {
    }

    int64_t
    get_deleted_count() const override {
        return 0;
    }
};

}  // namespace

class MvccFastPathTest : public ::testing::Test {
 protected:
    void
    SetUp() override {
        original_sparse_filter_enabled_ =
            ENABLE_SPARSE_FILTER_RESULT.exchange(true);
        original_sparse_min_segment_rows_ =
            SPARSE_FILTER_RESULT_MIN_SEGMENT_ROWS.exchange(1);
        original_sparse_max_ratio_ =
            SPARSE_FILTER_RESULT_MAX_RATIO.exchange(1.0);
        schema_ = std::make_shared<Schema>();
        vec_fid_ = schema_->AddDebugField(
            "fakevec", DataType::VECTOR_FLOAT, 16, knowhere::metric::L2);
        int64_fid_ = schema_->AddDebugField("counter", DataType::INT64);
        schema_->set_primary_field_id(int64_fid_);
    }

    void
    TearDown() override {
        ENABLE_SPARSE_FILTER_RESULT.store(original_sparse_filter_enabled_);
        SPARSE_FILTER_RESULT_MIN_SEGMENT_ROWS.store(
            original_sparse_min_segment_rows_);
        SPARSE_FILTER_RESULT_MAX_RATIO.store(original_sparse_max_ratio_);
    }

    // Helper: create sealed segment with no deletes
    SegmentSealedSPtr
    CreateSealedSegment() {
        auto raw_data = DataGen(schema_, N_);
        auto segment = CreateSealedWithFieldDataLoaded(schema_, raw_data);
        return SegmentSealedSPtr(segment.release());
    }

    // Helper: create sealed segment with some rows deleted
    SegmentSealedSPtr
    CreateSealedSegmentWithDeletes(int64_t num_deletes) {
        auto raw_data = DataGen(schema_, N_);
        auto segment = CreateSealedWithFieldDataLoaded(schema_, raw_data);

        std::vector<idx_t> pks;
        for (int64_t i = 0; i < num_deletes; i++) {
            pks.push_back(i);
        }
        auto ids = std::make_unique<IdArray>();
        ids->mutable_int_id()->mutable_data()->Add(pks.begin(), pks.end());
        std::vector<Timestamp> timestamps(num_deletes, 10);

        LoadDeletedRecordInfo info = {
            timestamps.data(), ids.get(), num_deletes};
        segment->LoadDeletedRecord(info);

        return SegmentSealedSPtr(segment.release());
    }

    // Helper: execute MvccNode-only plan and return results
    // MvccNode as source (no upstream FilterBitsNode) = no scalar filter
    struct MvccResult {
        int64_t num_rows;
        bool all_rows_visible;
        RowVectorPtr output;
    };

    MvccResult
    RunMvccPlan(const SegmentInternalInterface* segment,
                Timestamp collection_ttl = 0,
                Timestamp query_timestamp = MAX_TIMESTAMP) {
        // Build plan: just MvccNode as source (empty sources)
        auto mvcc_node = std::make_shared<plan::MvccNode>("mvcc_1");
        auto plan = plan::PlanFragment(mvcc_node);

        auto query_context = std::make_shared<QueryContext>(
            "test_mvcc",
            segment,
            N_,
            query_timestamp,
            collection_ttl,
            0,
            query::PlanOptions{false},
            std::make_shared<QueryConfig>(
                std::unordered_map<std::string, std::string>{}));

        auto task = Task::Create("task_mvcc", plan, 0, query_context);
        MvccResult result{0, false, nullptr};
        for (;;) {
            auto output = task->Next();
            if (!output) {
                break;
            }
            result.num_rows += output->size();
            result.output = output;
        }
        result.all_rows_visible = query_context->get_all_rows_visible();
        return result;
    }

    // Execute the real FilterBitsNode -> MvccNode topology with a native
    // STLSORT range producer.  This ensures Mvcc sees the exact list emitted
    // by FilterBits, while timestamp/delete visibility stays identical to the
    // Dense route.
    std::shared_ptr<const std::vector<int32_t>>
    RunNativeListMvccPlan(const SegmentInternalInterface* segment,
                          int64_t upper_bound,
                          Timestamp collection_ttl = 0,
                          Timestamp query_timestamp = MAX_TIMESTAMP) {
        proto::plan::GenericValue value;
        value.set_int64_val(upper_bound);
        auto expr = std::make_shared<expr::UnaryRangeFilterExpr>(
            expr::ColumnInfo(int64_fid_, DataType::INT64),
            proto::plan::OpType::LessThan,
            value,
            std::vector<proto::plan::GenericValue>{});
        auto filter_node =
            std::make_shared<plan::FilterBitsNode>("filter_native_list", expr);
        auto mvcc_node = std::make_shared<plan::MvccNode>(
            "mvcc_native_list", std::vector<plan::PlanNodePtr>{filter_node});
        auto query_context = std::make_shared<QueryContext>(
            "test_native_list_mvcc",
            segment,
            N_,
            query_timestamp,
            collection_ttl,
            0,
            query::PlanOptions{false},
            std::make_shared<QueryConfig>(
                std::unordered_map<std::string, std::string>{}));
        SearchInfo search_info;
        search_info.search_params_ = knowhere::Json{
            {"bf_filter_scan_mode", "valid_ids_per_query"},
        };
        query_context->set_search_info(search_info);
        auto task = Task::Create("task_native_list_mvcc",
                                 plan::PlanFragment(mvcc_node),
                                 0,
                                 query_context);
        while (task->Next()) {
        }
        auto filter_map = query_context->get_filter_map();
        return filter_map ? filter_map->SnapshotUnsetIds() : nullptr;
    }

    SchemaPtr schema_;
    FieldId vec_fid_;
    FieldId int64_fid_;
    int64_t N_ = 1000;
    bool original_sparse_filter_enabled_ = false;
    int64_t original_sparse_min_segment_rows_ = 0;
    double original_sparse_max_ratio_ = 0.0;
};

// ---------------------------------------------------------------------------
// Level 1: Sealed + no deletes + no TTL + source node
// Expected: all_rows_visible = true, output is all-zero bitmap
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, Level1_SealedNoDeletes_SkipFilter) {
    auto segment = CreateSealedSegment();
    auto result = RunMvccPlan(segment.get());

    EXPECT_EQ(result.num_rows, N_);
    EXPECT_TRUE(result.all_rows_visible)
        << "Level 1: sealed + no deletes should set all_rows_visible=true";

    // Verify output bitmap is all zeros (no rows filtered out)
    ASSERT_NE(result.output, nullptr);
    auto col = std::static_pointer_cast<ColumnVector>(result.output->child(0));
    ASSERT_NE(col, nullptr);
    TargetBitmapView view(col->GetRawData(), col->size());
    EXPECT_EQ(view.count(), 0)
        << "Level 1: bitmap should be all zeros (no filtering)";
}

// ---------------------------------------------------------------------------
// Level 2: Sealed + has deletes + no TTL + source node
// Expected: all_rows_visible = false, output has delete mask applied
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, Level2_SealedWithDeletes_DeleteMaskOnly) {
    int64_t num_deletes = 5;
    auto segment = CreateSealedSegmentWithDeletes(num_deletes);
    auto result = RunMvccPlan(segment.get());

    EXPECT_EQ(result.num_rows, N_);
    EXPECT_FALSE(result.all_rows_visible)
        << "Level 2: sealed + has deletes should NOT set all_rows_visible";

    // Verify output bitmap has some bits set (deleted rows marked)
    ASSERT_NE(result.output, nullptr);
    auto col = std::static_pointer_cast<ColumnVector>(result.output->child(0));
    ASSERT_NE(col, nullptr);
    TargetBitmapView view(col->GetRawData(), col->size());
    EXPECT_GT(view.count(), 0)
        << "Level 2: bitmap should have deleted rows marked";
}

// ---------------------------------------------------------------------------
// Level 3: Sealed + TTL set -> falls through to default path
// Expected: all_rows_visible = false
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, Level3_SealedWithTTL_DefaultPath) {
    auto segment = CreateSealedSegment();
    // Pass non-zero collection_ttl to force Level 3
    auto result = RunMvccPlan(segment.get(), /*collection_ttl=*/100);

    EXPECT_EQ(result.num_rows, N_);
    EXPECT_FALSE(result.all_rows_visible)
        << "Level 3: TTL set should NOT trigger fast path";
}

// Native valid-ID lists are scalar candidates, not an MVCC bypass.  Verify
// that delete visibility is applied by compacting only the candidate IDs.
TEST_F(MvccFastPathTest, NativeValidIds_CompactsDeletedCandidates) {
    auto segment = CreateSealedSegmentWithDeletes(/*num_deletes=*/5);
    auto survivors = RunNativeListMvccPlan(segment.get(), /*upper_bound=*/10);

    ASSERT_NE(survivors, nullptr);
    EXPECT_EQ(*survivors, (std::vector<int32_t>{5, 6, 7, 8, 9}));
}

TEST_F(MvccFastPathTest, LegacySparseKnobRespectsDisabledFeatureGate) {
    auto segment = CreateSealedSegment();
    ENABLE_SPARSE_FILTER_RESULT.store(false);

    EXPECT_ANY_THROW(RunNativeListMvccPlan(segment.get(), /*upper_bound=*/10));
}

TEST_F(MvccFastPathTest, SparseDeleteMaskDoesNotTrustReportedDeleteCount) {
    auto raw_data = DataGen(schema_, N_);
    auto segment = std::make_shared<ZeroReportedDeleteCountSegment>(schema_);
    LoadGeneratedDataIntoSegment(raw_data, segment.get());

    constexpr int64_t kDeleted = 5;
    std::vector<idx_t> pks;
    for (int64_t i = 0; i < kDeleted; ++i) {
        pks.push_back(i);
    }
    auto ids = std::make_unique<IdArray>();
    ids->mutable_int_id()->mutable_data()->Add(pks.begin(), pks.end());
    std::vector<Timestamp> timestamps(kDeleted, 10);
    LoadDeletedRecordInfo info = {timestamps.data(), ids.get(), kDeleted};
    segment->LoadDeletedRecord(info);

    // This is the exact stale observation that previously bypassed the mask.
    ASSERT_EQ(segment->get_deleted_count(), 0);
    TargetBitmap expected_delete_mask(N_, false);
    TargetBitmapView expected_delete_view(expected_delete_mask);
    const SegmentInternalInterface* segment_interface = segment.get();
    segment_interface->mask_with_delete(
        expected_delete_view, N_, MAX_TIMESTAMP);
    ASSERT_EQ(expected_delete_view.count(), kDeleted);

    auto survivors = RunNativeListMvccPlan(segment.get(), /*upper_bound=*/10);

    ASSERT_NE(survivors, nullptr);
    EXPECT_EQ(*survivors, (std::vector<int32_t>{5, 6, 7, 8, 9}));
}

// The same native list must observe delete visibility at the query snapshot,
// not merely the segment's latest delete state.  Deletes are loaded at ts=10
// by CreateSealedSegmentWithDeletes: a snapshot at ts=5 must retain IDs 0..5,
// while the current snapshot above removes the deleted prefix.  DataGen assigns
// later insert timestamps to later rows, so IDs 6..9 are intentionally excluded
// from this historical snapshot.
TEST_F(MvccFastPathTest, NativeValidIds_HistoricalSnapshotPrecedesDelete) {
    auto segment = CreateSealedSegmentWithDeletes(/*num_deletes=*/5);
    auto survivors = RunNativeListMvccPlan(segment.get(),
                                           /*upper_bound=*/10,
                                           /*collection_ttl=*/0,
                                           /*query_timestamp=*/5);

    ASSERT_NE(survivors, nullptr);
    EXPECT_EQ(*survivors, (std::vector<int32_t>{0, 1, 2, 3, 4, 5}));
}

// TTL uses the same invalid mask as the Dense route.  The test deliberately
// includes the boundary so a future list-only shortcut cannot silently change
// the <= collection_ttl expiration rule.
TEST_F(MvccFastPathTest, NativeValidIds_CompactsTtlExpiredCandidates) {
    auto segment = CreateSealedSegment();
    auto survivors = RunNativeListMvccPlan(segment.get(),
                                           /*upper_bound=*/600,
                                           /*collection_ttl=*/100);

    ASSERT_NE(survivors, nullptr);
    ASSERT_EQ(survivors->size(), 499);
    EXPECT_EQ(survivors->front(), 101);
    EXPECT_EQ(survivors->back(), 599);
}

// Historical reads also use the regular timestamp invalid mask.  Rows whose
// insert timestamp is newer than the snapshot must not reach Cardinal.
TEST_F(MvccFastPathTest, NativeValidIds_CompactsFutureCandidatesAtSnapshot) {
    auto segment = CreateSealedSegment();
    auto survivors = RunNativeListMvccPlan(segment.get(),
                                           /*upper_bound=*/600,
                                           /*collection_ttl=*/0,
                                           /*query_timestamp=*/100);

    ASSERT_NE(survivors, nullptr);
    ASSERT_EQ(survivors->size(), 101);
    EXPECT_EQ(survivors->front(), 0);
    EXPECT_EQ(survivors->back(), 100);
}

// Sparse-ID payloads preserve producer order. STLSORT is value ordered, so a
// range posting must not pay an unrelated row-ID sort before handoff.
TEST_F(MvccFastPathTest, NativeValidIds_StlSortRangePreservesProducerOrder) {
    const std::vector<int64_t> values{50, 5, 100, 7, 75};
    auto sort_index = index::CreateScalarIndexSort<int64_t>();
    sort_index->Build(values.size(), values.data());

    auto ids = sort_index->TryGetValidIdRange(
        /*lower_bound_value=*/7,
        /*lb_inclusive=*/true,
        /*upper_bound_value=*/75,
        /*ub_inclusive=*/true);

    ASSERT_NE(ids, nullptr);
    EXPECT_EQ(*ids, (std::vector<int32_t>{3, 0, 4}));
}

TEST_F(MvccFastPathTest, SparseIdPayloadAcceptsUnorderedUniqueIds) {
    QueryContext query_context("unordered_sparse_payload", nullptr, 5, 0);
    auto ids = std::make_shared<const std::vector<int32_t>>(
        std::vector<int32_t>{3, 0, 4});

    EXPECT_NO_THROW(query_context.set_filter_map(std::make_shared<FilterMap>(
        FilterMap::FromUnsetIds(/*universe=*/5, ids))));
    ASSERT_NE(query_context.get_filter_map(), nullptr);
    EXPECT_EQ(*query_context.get_filter_map()->SnapshotUnsetIds(), *ids);
}

TEST_F(MvccFastPathTest, SparseIdPayloadRejectsOutOfRangeIds) {
    QueryContext query_context("invalid_sparse_payload", nullptr, 5, 0);
    auto out_of_range = std::make_shared<const std::vector<int32_t>>(
        std::vector<int32_t>{3, 5});

    EXPECT_ANY_THROW(FilterMap::FromUnsetIds(/*universe=*/5, out_of_range));
}

// Sparse is an output contract, not a sealed-index-only fast path.  Growing
// segments deliberately cannot take FilterBits' native scalar-index branch;
// this verifies its normal Dense evaluator is converted at the boundary and
// that Mvcc can consume that Sparse result without changing visibility.
TEST_F(MvccFastPathTest, SparseOutput_GrowingFallsBackToDenseEvaluator) {
    auto raw_data = DataGen(schema_, N_);
    auto segment = CreateGrowingSegment(schema_, empty_index_meta);
    segment->PreInsert(N_);
    segment->Insert(0,
                    N_,
                    raw_data.row_ids_.data(),
                    raw_data.timestamps_.data(),
                    raw_data.raw_);

    auto survivors = RunNativeListMvccPlan(segment.get(), /*upper_bound=*/10);

    ASSERT_NE(survivors, nullptr);
    EXPECT_EQ(*survivors, (std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// ---------------------------------------------------------------------------
// Level 3: Growing segment -> falls through to default path
// Expected: all_rows_visible = false
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, Level3_GrowingSegment_DefaultPath) {
    auto raw_data = DataGen(schema_, N_);
    auto segment = CreateGrowingSegment(schema_, empty_index_meta);
    segment->PreInsert(N_);
    segment->Insert(0,
                    N_,
                    raw_data.row_ids_.data(),
                    raw_data.timestamps_.data(),
                    raw_data.raw_);

    // Build plan manually for growing segment
    auto mvcc_node = std::make_shared<plan::MvccNode>("mvcc_1");
    auto plan = plan::PlanFragment(mvcc_node);

    auto query_context = std::make_shared<QueryContext>(
        "test_mvcc_growing",
        segment.get(),
        N_,
        MAX_TIMESTAMP,
        0,
        0,
        query::PlanOptions{false},
        std::make_shared<QueryConfig>(
            std::unordered_map<std::string, std::string>{}));

    auto task = Task::Create("task_mvcc_growing", plan, 0, query_context);
    int64_t num_rows = 0;
    for (;;) {
        auto output = task->Next();
        if (!output) {
            break;
        }
        num_rows += output->size();
    }
    EXPECT_EQ(num_rows, N_);
    EXPECT_FALSE(query_context->get_all_rows_visible())
        << "Growing segment should NOT trigger fast path";
}

// ---------------------------------------------------------------------------
// QueryContext all_rows_visible flag: default is false
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, QueryContext_SkipFilter_DefaultFalse) {
    auto segment = CreateSealedSegment();
    auto query_context = std::make_shared<QueryContext>(
        "test_default", segment.get(), N_, MAX_TIMESTAMP);
    EXPECT_FALSE(query_context->get_all_rows_visible())
        << "all_rows_visible should default to false";
}

// ---------------------------------------------------------------------------
// QueryContext all_rows_visible flag: set/get round-trip
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, QueryContext_SkipFilter_SetGet) {
    auto segment = CreateSealedSegment();
    auto query_context = std::make_shared<QueryContext>(
        "test_setget", segment.get(), N_, MAX_TIMESTAMP);
    query_context->set_all_rows_visible(true);
    EXPECT_TRUE(query_context->get_all_rows_visible());
    query_context->set_all_rows_visible(false);
    EXPECT_FALSE(query_context->get_all_rows_visible());
}

// ---------------------------------------------------------------------------
// Timestamp guard: query_timestamp < max_insert_timestamp -> Level 3 fallback
// Expected: all_rows_visible = false (fast path NOT taken)
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, TimestampGuard_OldQueryTs_FallsBackToLevel3) {
    auto segment = CreateSealedSegment();

    // Use query_timestamp = 1, which is less than any insert timestamp
    // generated by DataGen (timestamps start from 0 and go up to N_-1)
    // With the guard, this should fall through to Level 3 (default path)
    auto result = RunMvccPlan(segment.get(),
                              /*collection_ttl=*/0,
                              /*query_timestamp=*/1);

    EXPECT_EQ(result.num_rows, N_);
    EXPECT_FALSE(result.all_rows_visible)
        << "query_timestamp < max_insert_timestamp should NOT trigger fast "
           "path";
}

// ---------------------------------------------------------------------------
// Timestamp guard: query_timestamp >= max_insert_timestamp -> Level 1
// Expected: all_rows_visible = true (fast path taken)
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, TimestampGuard_CurrentQueryTs_TakesFastPath) {
    auto segment = CreateSealedSegment();

    // Use MAX_TIMESTAMP, which is always >= max_insert_timestamp
    auto result = RunMvccPlan(segment.get(),
                              /*collection_ttl=*/0,
                              /*query_timestamp=*/MAX_TIMESTAMP);

    EXPECT_EQ(result.num_rows, N_);
    EXPECT_TRUE(result.all_rows_visible)
        << "query_timestamp >= max_insert_timestamp should trigger fast path";
}

// ---------------------------------------------------------------------------
// Regression: sequential Level 1 queries must not share mutable bitmap state.
// A downstream operator (e.g. ElementFilterBitsNode) may flip() the bitmap
// in-place.  The second query must still see a clean all-zero bitmap.
// ---------------------------------------------------------------------------
TEST_F(MvccFastPathTest, Level1_NoCachePollution_SequentialQueries) {
    auto segment = CreateSealedSegment();

    // First query – Level 1
    auto result1 = RunMvccPlan(segment.get());
    ASSERT_TRUE(result1.all_rows_visible);
    auto col1 =
        std::static_pointer_cast<ColumnVector>(result1.output->child(0));
    ASSERT_NE(col1, nullptr);
    TargetBitmapView view1(col1->GetRawData(), col1->size());
    EXPECT_EQ(view1.count(), 0);

    // Simulate downstream mutation (ElementFilterBitsNode does doc_bitset.flip)
    view1.flip();
    EXPECT_EQ(view1.count(), N_);

    // Second query on the same thread – must NOT see the flipped bits
    auto result2 = RunMvccPlan(segment.get());
    ASSERT_TRUE(result2.all_rows_visible);
    auto col2 =
        std::static_pointer_cast<ColumnVector>(result2.output->child(0));
    ASSERT_NE(col2, nullptr);
    TargetBitmapView view2(col2->GetRawData(), col2->size());
    EXPECT_EQ(view2.count(), 0)
        << "Second query must return clean bitmap, not polluted cache";
}
