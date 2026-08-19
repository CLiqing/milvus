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

#include <folly/ExceptionWrapper.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cachinglayer/CacheSlot.h"
#include "cachinglayer/Utils.h"
#include "common/ArrayOffsets.h"
#include "common/BitsetView.h"
#include "common/Chunk.h"
#include "common/Consts.h"
#include "common/Downpush.h"
#include "common/EasyAssert.h"
#include "common/FieldMeta.h"
#include "common/OffsetMapping.h"
#include "common/QueryInfo.h"
#include "common/QueryResult.h"
#include "common/Schema.h"
#include "common/Types.h"
#include "common/Utils.h"
#include "exec/operator/Utils.h"
#include "index/Index.h"
#include "index/VectorIndex.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "mmap/ChunkedColumnInterface.h"
#include "query/CachedSearchIterator.h"
#include "query/SearchBruteForce.h"
#include "query/SearchOnSealed.h"
#include "query/SubSearchResult.h"
#include "query/Utils.h"
#include "query/helper.h"
#include "segcore/SealedIndexingRecord.h"

namespace milvus::query {

namespace {

// Buffers holding the physical-order downpush value source gathered from the
// logical-order source. Must outlive the vector search call below.
struct DownpushGatheredValueSource {
    std::vector<int64_t> int64_values;
    std::vector<float> float_values;
    std::vector<int32_t> dictionary_ids;
    std::vector<char> string_bytes;
    std::vector<uint32_t> string_offsets;
    // RawStringColumnView requires bool storage, not a byte buffer cast to
    // bool*. Keep the actual bool objects alive through VectorIndex::Query.
    std::unique_ptr<bool[]> string_valid;
    std::vector<const char*> string_chunk_bases;
    std::vector<const uint32_t*> string_chunk_value_offsets;
    std::vector<const bool*> string_chunk_valid_data;
    std::vector<size_t> string_chunk_row_counts;
    std::vector<int64_t> string_chunk_row_offsets;
};

// Gather the downpush scalar value source from logical to physical order using
// the vector index's offset mapping (p2l). The vector index stores only valid
// rows (physical order), so the fused predicate must read scalar values in that
// same order. Returns a new filter whose value-source pointers reference
// `gathered` and whose row_count is the physical valid count. Returns false on
// an unexpected source layout (defensive).
bool
GatherDownpushValueSource(
    const knowhere::BitsetView::ExtraScalarInt64PredicateFilter& in,
    const milvus::OffsetMapping& offset_mapping,
    knowhere::BitsetView::ExtraScalarInt64PredicateFilter& out,
    DownpushGatheredValueSource& gathered) {
    using ValueType = knowhere::BitsetView::ExtraScalarPredicateValueType;
    const int64_t valid_count = offset_mapping.GetValidCount();
    if (valid_count <= 0) {
        return false;
    }
    out = in;
    out.row_count = static_cast<size_t>(valid_count);

    // p2l is monotonically increasing, so a running chunk cursor is sufficient.
    auto logical_at = [&](int64_t physical) -> int64_t {
        return offset_mapping.GetLogicalOffset(physical);
    };

    switch (in.value_type) {
        case ValueType::kInt64: {
            if (in.row_values == nullptr && in.chunk_values == nullptr) {
                return false;
            }
            gathered.int64_values.resize(valid_count);
            size_t chunk_idx = 0;
            for (int64_t p = 0; p < valid_count; ++p) {
                const int64_t l = logical_at(p);
                if (l < 0) {
                    return false;
                }
                if (in.row_values != nullptr) {
                    gathered.int64_values[p] = in.row_values[l];
                } else {
                    while (chunk_idx + 1 < in.num_chunks &&
                           in.chunk_offsets[chunk_idx + 1] <= l) {
                        ++chunk_idx;
                    }
                    gathered.int64_values[p] =
                        in.chunk_values[chunk_idx][l -
                                                   in.chunk_offsets[chunk_idx]];
                }
            }
            out.row_values = gathered.int64_values.data();
            out.chunk_values = nullptr;
            out.chunk_offsets = nullptr;
            out.num_chunks = 0;
            return true;
        }
        case ValueType::kFloat: {
            if (in.row_float_values == nullptr &&
                in.chunk_float_values == nullptr) {
                return false;
            }
            gathered.float_values.resize(valid_count);
            size_t chunk_idx = 0;
            for (int64_t p = 0; p < valid_count; ++p) {
                const int64_t l = logical_at(p);
                if (l < 0) {
                    return false;
                }
                if (in.row_float_values != nullptr) {
                    gathered.float_values[p] = in.row_float_values[l];
                } else {
                    while (chunk_idx + 1 < in.num_chunks &&
                           in.chunk_offsets[chunk_idx + 1] <= l) {
                        ++chunk_idx;
                    }
                    gathered.float_values[p] =
                        in.chunk_float_values[chunk_idx][l -
                                                         in.chunk_offsets[chunk_idx]];
                }
            }
            out.row_float_values = gathered.float_values.data();
            out.chunk_float_values = nullptr;
            out.chunk_offsets = nullptr;
            out.num_chunks = 0;
            return true;
        }
        case ValueType::kDictionaryId: {
            if (in.row_dictionary_ids == nullptr) {
                return false;
            }
            gathered.dictionary_ids.resize(valid_count);
            for (int64_t p = 0; p < valid_count; ++p) {
                const int64_t l = logical_at(p);
                if (l < 0) {
                    return false;
                }
                gathered.dictionary_ids[p] = in.row_dictionary_ids[l];
            }
            out.row_dictionary_ids = gathered.dictionary_ids.data();
            return true;
        }
        case ValueType::kString: {
            const auto& col = in.string_column;
            if (col.chunk_bases == nullptr ||
                col.chunk_value_offsets == nullptr ||
                col.chunk_row_counts == nullptr ||
                col.chunk_row_offsets == nullptr || col.num_chunks == 0) {
                return false;
            }
            gathered.string_offsets.assign(valid_count + 1, 0);
            if (col.chunk_valid_data != nullptr) {
                gathered.string_valid = std::make_unique<bool[]>(valid_count);
                std::fill_n(gathered.string_valid.get(), valid_count, false);
            }
            size_t chunk_idx = 0;
            size_t total_bytes = 0;
            for (int64_t p = 0; p < valid_count; ++p) {
                const int64_t l = logical_at(p);
                if (l < 0) {
                    return false;
                }
                while (chunk_idx + 1 < col.num_chunks &&
                       col.chunk_row_offsets[chunk_idx + 1] <= l) {
                    ++chunk_idx;
                }
                const int64_t local = l - col.chunk_row_offsets[chunk_idx];
                if (local < 0 ||
                    static_cast<size_t>(local) >=
                        col.chunk_row_counts[chunk_idx]) {
                    return false;
                }
                const auto* offsets = col.chunk_value_offsets[chunk_idx];
                const auto* valid_data =
                    col.chunk_valid_data == nullptr
                        ? nullptr
                        : col.chunk_valid_data[chunk_idx];
                if (valid_data != nullptr && !valid_data[local]) {
                    gathered.string_valid[p] = 0;
                    gathered.string_offsets[p + 1] = total_bytes;
                    continue;
                }
                if (valid_data != nullptr) {
                    gathered.string_valid[p] = 1;
                }
                total_bytes += offsets[local + 1] - offsets[local];
                gathered.string_offsets[p + 1] = total_bytes;
            }

            // Knowhere treats a null chunk base as an invalid value source.
            // Preserve a non-null base even when every gathered value is the
            // valid empty string (whose concatenated byte buffer is empty).
            gathered.string_bytes.resize(std::max<size_t>(total_bytes, 1));
            chunk_idx = 0;
            for (int64_t p = 0; p < valid_count; ++p) {
                const int64_t l = logical_at(p);
                while (chunk_idx + 1 < col.num_chunks &&
                       col.chunk_row_offsets[chunk_idx + 1] <= l) {
                    ++chunk_idx;
                }
                const int64_t local = l - col.chunk_row_offsets[chunk_idx];
                const auto* offsets = col.chunk_value_offsets[chunk_idx];
                const auto* valid_data =
                    col.chunk_valid_data == nullptr
                        ? nullptr
                        : col.chunk_valid_data[chunk_idx];
                if (valid_data != nullptr && !valid_data[local]) {
                    continue;
                }
                const auto* base = col.chunk_bases[chunk_idx];
                const auto begin = offsets[local];
                const auto end = offsets[local + 1];
                std::copy(base + begin,
                          base + end,
                          gathered.string_bytes.data() +
                              gathered.string_offsets[p]);
            }

            // Rebuild as a single gathered chunk in physical order.
            gathered.string_chunk_bases = {gathered.string_bytes.data()};
            gathered.string_chunk_value_offsets = {
                gathered.string_offsets.data()};
            gathered.string_chunk_row_counts = {
                static_cast<size_t>(valid_count)};
            gathered.string_chunk_row_offsets = {0};
            out.string_column.chunk_bases =
                gathered.string_chunk_bases.data();
            out.string_column.chunk_value_offsets =
                gathered.string_chunk_value_offsets.data();
            out.string_column.chunk_row_counts =
                gathered.string_chunk_row_counts.data();
            out.string_column.chunk_row_offsets =
                gathered.string_chunk_row_offsets.data();
            if (col.chunk_valid_data != nullptr) {
                gathered.string_chunk_valid_data = {
                    gathered.string_valid.get()};
                out.string_column.chunk_valid_data =
                    gathered.string_chunk_valid_data.data();
            } else {
                out.string_column.chunk_valid_data = nullptr;
            }
            out.string_column.num_chunks = 1;
            out.string_column.row_count = static_cast<size_t>(valid_count);
            out.string_column.uniform_chunk_rows =
                static_cast<size_t>(valid_count);
            return true;
        }
    }
    return false;
}

}  // namespace

void
SearchOnSealedIndex(const Schema& schema,
                    const segcore::SealedIndexingRecord& record,
                    const SearchInfo& search_info,
                    const void* query_data,
                    const size_t* query_offsets,
                    int64_t num_queries,
                    const BitsetView& bitset,
                    milvus::OpContext* op_context,
                    SearchResult& search_result) {
    auto topK = search_info.topk_;
    auto round_decimal = search_info.round_decimal_;

    auto field_id = search_info.field_id_;
    auto& field = schema[field_id];
    auto is_sparse = field.get_data_type() == DataType::VECTOR_SPARSE_U32_F32;
    // TODO(SPARSE): see todo in PlanImpl.h::PlaceHolder.
    auto dim = is_sparse ? 0 : field.get_dim();

    AssertInfo(record.is_ready(field_id), "[SearchOnSealed]Record isn't ready");
    // Keep the field_indexing smart pointer, until all reference by raw dropped.
    auto field_indexing = record.get_field_indexing(field_id);
    AssertInfo(field_indexing->metric_type_ == search_info.metric_type_,
               "Metric type of field index isn't the same with search info,"
               "field index: {}, search info: {}",
               field_indexing->metric_type_,
               search_info.metric_type_);

    knowhere::DataSetPtr dataset;
    if (query_offsets == nullptr) {
        dataset = knowhere::GenDataSet(num_queries, dim, query_data);
    } else {
        // Rather than non-embedding list search where num_queries equals to the number of vectors,
        // in embedding list search, multiple vectors form an embedding list and the last element of query_offsets
        // stands for the total number of vectors.
        auto num_vectors = query_offsets[num_queries];
        dataset = knowhere::GenDataSet(num_vectors, dim, query_data);
        dataset->Set(knowhere::meta::EMB_LIST_OFFSET, query_offsets);
        dataset->Set(knowhere::meta::NQ, num_queries);
    }

    dataset->SetIsSparse(is_sparse);
    auto accessor =
        SemiInlineGet(field_indexing->indexing_->PinCells(op_context, {0}));
    auto vec_index =
        dynamic_cast<index::VectorIndex*>(accessor->get_cell_of(0));
    AssertInfo(vec_index != nullptr, "invalid vector index");
    if (bitset.has_extra_scalar_int64_predicate_filter()) {
        auto index_type = vec_index->GetIndexType();
        AssertInfo(IsDownpushSupportedIndexType(index_type),
                   "downpush hint is only supported by Cardinal index/backend "
                   "in v1, actual index type: {}",
                   index_type);
    }

    const auto& offset_mapping = vec_index->GetOffsetMapping();
    const bool is_element_level_search = search_info.array_offsets_ != nullptr;
    search_result.element_level_ = is_element_level_search;
    TargetBitmap transformed_bitset;
    BitsetView search_bitset = bitset;
    const auto has_offset_mapping =
        offset_mapping.IsEnabled() && !is_element_level_search;
    const bool has_downpush = bitset.has_extra_scalar_int64_predicate_filter();

    DownpushGatheredValueSource downpush_gathered;
    if (has_offset_mapping) {
        if (offset_mapping.GetValidCount() == 0) {
            FillEmptySearchResult(search_result, num_queries, topK);
            return;
        }
        if (!bitset.empty()) {
            auto status =
                offset_mapping.TransformBitset(bitset, transformed_bitset);
            if (status == OffsetMapping::BitsetTransformStatus::AllFiltered) {
                FillEmptySearchResult(search_result, num_queries, topK);
                return;
            }
            search_bitset =
                status == OffsetMapping::BitsetTransformStatus::NoFilter
                    ? BitsetView{}
                    : search_result.PinBitset(std::move(transformed_bitset));
        }
    }

    if (has_downpush && has_offset_mapping) {
        // The vector index stores only valid rows (physical order), so the
        // fused predicate must read scalar values in that same order. Gather
        // the value source from logical to physical using the p2l mapping.
        knowhere::BitsetView::ExtraScalarInt64PredicateFilter gathered_filter;
        if (!GatherDownpushValueSource(
                bitset.extra_scalar_int64_predicate_filter(),
                offset_mapping,
                gathered_filter,
                downpush_gathered)) {
            ThrowInfo(UnexpectedError,
                      "downpush failed to gather value source to physical "
                      "order");
        }
        search_bitset.set_extra_scalar_int64_predicate_filter(
            gathered_filter, bitset.extra_filtered_out_count());
    }

    if (search_info.iterator_v2_info_.has_value()) {
        CachedSearchIterator cached_iter(
            *vec_index, dataset, search_info, search_bitset);
        cached_iter.NextBatch(search_info, search_result);
        FinalizeVectorSearchOffsets(
            search_result, offset_mapping, search_info.array_offsets_.get());
        return;
    }

    bool use_iterator =
        milvus::exec::PrepareVectorIteratorsFromIndex(search_info,
                                                      num_queries,
                                                      dataset,
                                                      search_result,
                                                      search_bitset,
                                                      *vec_index);
    if (!use_iterator) {
        vec_index->Query(
            dataset, search_info, search_bitset, op_context, search_result);
        float* distances = search_result.distances_.data();
        auto total_num = num_queries * topK;
        if (round_decimal != -1) {
            const float multiplier = pow(10.0, round_decimal);
            for (int i = 0; i < total_num; i++) {
                distances[i] =
                    std::round(distances[i] * multiplier) / multiplier;
            }
        }
    }
    FinalizeVectorSearchOffsets(
        search_result,
        offset_mapping,
        use_iterator ? nullptr : search_info.array_offsets_.get());
    search_result.total_nq_ = num_queries;
    search_result.unity_topK_ = topK;
}

void
SearchOnSealedColumn(const Schema& schema,
                     ChunkedColumnInterface* column,
                     const SearchInfo& search_info,
                     const std::map<std::string, std::string>& index_info,
                     const void* query_data,
                     const size_t* query_offsets,
                     int64_t num_queries,
                     int64_t row_count,
                     const BitsetView& bitview,
                     milvus::OpContext* op_context,
                     SearchResult& result) {
    AssertInfo(!bitview.has_extra_scalar_int64_predicate_filter(),
               "downpush hint is only supported by Cardinal index/backend");

    auto field_id = search_info.field_id_;
    auto& field = schema[field_id];

    auto data_type = field.get_data_type();
    auto element_type = field.get_element_type();
    // TODO(SPARSE): see todo in PlanImpl.h::PlaceHolder.
    auto dim =
        data_type == DataType::VECTOR_SPARSE_U32_F32 ? 0 : field.get_dim();

    query::dataset::SearchDataset query_dataset{search_info.metric_type_,
                                                num_queries,
                                                search_info.topk_,
                                                search_info.round_decimal_,
                                                dim,
                                                query_data,
                                                query_offsets};

    CheckBruteForceSearchParam(field, search_info);

    if (column->IsNullable()) {
        column->BuildValidRowIds(op_context);
    }

    // Check for nullable vector field with all null values - must be done before creating iterators
    const auto& offset_mapping = column->GetOffsetMapping();
    // Element-level VECTOR_ARRAY search has already expanded the row bitset
    // to element IDs. OffsetMapping is row-level, so only use it for row-level
    // vector searches.
    bool is_element_level_search =
        field.get_data_type() == DataType::VECTOR_ARRAY &&
        search_info.array_offsets_ != nullptr;
    result.element_level_ = is_element_level_search;
    TargetBitmap transformed_bitset;
    BitsetView search_bitview = bitview;
    const auto has_offset_mapping =
        offset_mapping.IsEnabled() && !is_element_level_search;
    if (has_offset_mapping) {
        if (offset_mapping.GetValidCount() == 0) {
            // All vectors are null, return empty result
            FillEmptySearchResult(result, num_queries, search_info.topk_);
            return;
        }
        if (!bitview.empty()) {
            auto status =
                offset_mapping.TransformBitset(bitview, transformed_bitset);
            if (status == OffsetMapping::BitsetTransformStatus::AllFiltered) {
                FillEmptySearchResult(result, num_queries, search_info.topk_);
                return;
            }
            search_bitview =
                status == OffsetMapping::BitsetTransformStatus::NoFilter
                    ? BitsetView{}
                    : result.PinBitset(std::move(transformed_bitset));
        }
    }

    // For element-level search (embedding-search-embedding), the underlying
    // knowhere search is keyed by the scalar element type rather than
    // VECTOR_ARRAY, and per-chunk sizes must be counted in elements.
    if (is_element_level_search) {
        data_type = element_type;
    }

    if (search_info.iterator_v2_info_.has_value()) {
        // Element-level search has already replaced data_type with
        // element_type above. This assert still guards emb-list
        // (multi-search-multi) iterator, which proxy should have rejected
        // — keep it as a defense-in-depth check.
        AssertInfo(data_type != DataType::VECTOR_ARRAY,
                   "embedding list (multi-search-multi) iterator is not "
                   "supported on vector array fields");

        CachedSearchIterator cached_iter(column,
                                         query_dataset,
                                         search_info,
                                         index_info,
                                         search_bitview,
                                         data_type);
        cached_iter.NextBatch(search_info, result);
        FinalizeVectorSearchOffsets(
            result, offset_mapping, search_info.array_offsets_.get());
        return;
    }

    const bool use_vector_iterator =
        milvus::exec::UseVectorIterator(search_info);
    auto num_chunk = column->num_chunks();

    SubSearchResult final_qr(num_queries,
                             search_info.topk_,
                             search_info.metric_type_,
                             search_info.round_decimal_);

    auto offset = 0;
    auto vector_chunks = column->GetAllChunks(op_context);
    for (int i = 0; i < num_chunk; ++i) {
        const auto& pw = vector_chunks[i];
        auto vec_data = pw.get()->Data();
        auto chunk_size = column->chunk_row_nums(i);
        if (has_offset_mapping) {
            chunk_size = column->GetValidCountInChunk(i);
        }

        // For element-level search, get element count from VectorArrayOffsets
        if (is_element_level_search) {
            auto elem_offsets_pw = column->VectorArrayOffsets(op_context, i);
            // offsets[row_count] gives total element count in this chunk
            chunk_size = elem_offsets_pw.get()[chunk_size];
        }

        auto raw_dataset =
            query::dataset::RawDataset{offset, dim, chunk_size, vec_data};

        PinWrapper<const size_t*> offsets_pw;
        if (data_type == DataType::VECTOR_ARRAY) {
            AssertInfo(
                query_offsets != nullptr,
                "query_offsets is nullptr, but data_type is vector array");

            offsets_pw = column->VectorArrayOffsets(op_context, i);
            raw_dataset.raw_data_offsets = offsets_pw.get();
        }

        if (use_vector_iterator) {
            AssertInfo(data_type != DataType::VECTOR_ARRAY,
                       "vector array(embedding list) is not supported for "
                       "vector iterator");
            auto sub_qr =
                PackBruteForceSearchIteratorsIntoSubResult(query_dataset,
                                                           raw_dataset,
                                                           search_info,
                                                           index_info,
                                                           search_bitview,
                                                           data_type);
            final_qr.merge(sub_qr);
        } else {
            auto sub_qr = BruteForceSearch(query_dataset,
                                           raw_dataset,
                                           search_info,
                                           index_info,
                                           search_bitview,
                                           data_type,
                                           element_type,
                                           op_context);
            final_qr.merge(sub_qr);
        }
        offset += chunk_size;
    }
    if (use_vector_iterator) {
        bool larger_is_closer = PositivelyRelated(search_info.metric_type_);
        // Element-level search skips row-level mapping (element IDs are
        // not row-aligned); see ChunkMergeIterator ctor.
        const milvus::OffsetMapping* iter_offset_mapping =
            (search_info.array_offsets_ != nullptr || !has_offset_mapping)
                ? nullptr
                : &offset_mapping;
        result.AssembleChunkVectorIterators(num_queries,
                                            num_chunk,
                                            final_qr.chunk_iterators(),
                                            iter_offset_mapping,
                                            larger_is_closer);
    } else {
        // See FinalizeVectorSearchOffsets for the rationale: element-level
        // and row-level remapping are mutually exclusive.
        if (search_info.array_offsets_ != nullptr) {
            auto [seg_offsets, elem_indicies] =
                final_qr.convert_to_element_offsets(
                    search_info.array_offsets_.get());
            result.seg_offsets_ = std::move(seg_offsets);
            result.element_indices_ = std::move(elem_indicies);
            result.element_level_ = true;
        } else {
            if (has_offset_mapping) {
                offset_mapping.TransformOffsets(final_qr.mutable_offsets());
            }
            result.seg_offsets_ = std::move(final_qr.mutable_offsets());
        }
        result.distances_ = std::move(final_qr.mutable_distances());
    }
    result.unity_topK_ = query_dataset.topk;
    result.total_nq_ = query_dataset.num_queries;
}

}  // namespace milvus::query
