// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "exec/operator/groupby/GroupMembership.h"

#include <algorithm>
#include <memory>
#include <unordered_map>

#include "index/ScalarIndex.h"
#include "segcore/SegmentChunkReader.h"

namespace milvus::exec {
namespace {

template <typename T>
using GroupKey = std::optional<T>;

bool
IsEligible(const TargetBitmap* eligible_rows, size_t offset) {
    return eligible_rows == nullptr || (*eligible_rows)[offset];
}

void
ApplyBaseFilter(TargetBitmap& membership, const TargetBitmap* eligible_rows) {
    if (eligible_rows == nullptr) {
        return;
    }
    membership &= *eligible_rows;
}

template <typename T>
std::optional<TargetBitmap>
BuildIndexMembership(milvus::OpContext* op_ctx,
                     const segcore::SegmentInternalInterface& segment,
                     FieldId field_id,
                     size_t row_count,
                     const std::vector<GroupKey<T>>& groups,
                     const TargetBitmap* eligible_rows) {
    auto pinned_indexes = segment.PinIndex(op_ctx, field_id);
    if (pinned_indexes.empty()) {
        return std::nullopt;
    }

    TargetBitmap membership;
    membership.reserve(row_count);
    size_t remaining = row_count;
    for (auto& pinned_index : pinned_indexes) {
        auto scalar_index =
            dynamic_cast<const index::ScalarIndex<T>*>(pinned_index.get());
        if (scalar_index == nullptr) {
            return std::nullopt;
        }
        auto* mutable_index = const_cast<index::ScalarIndex<T>*>(scalar_index);
        TargetBitmap chunk_membership(mutable_index->Count(), false);
        for (const auto& group : groups) {
            auto matches = group.has_value()
                               ? mutable_index->In(1, &group.value())
                               : mutable_index->IsNull();
            if (matches.size() != chunk_membership.size()) {
                return std::nullopt;
            }
            chunk_membership |= matches;
        }

        auto append_size = std::min(remaining, chunk_membership.size());
        membership.append(chunk_membership, 0, append_size);
        remaining -= append_size;
        if (remaining == 0) {
            break;
        }
    }
    if (membership.size() != row_count) {
        return std::nullopt;
    }
    ApplyBaseFilter(membership, eligible_rows);
    return membership;
}

template <typename T, typename Visitor>
bool
ScanRawField(milvus::OpContext* op_ctx,
             const segcore::SegmentInternalInterface& segment,
             FieldId field_id,
             size_t row_count,
             Visitor&& visitor) {
    if (!segment.HasFieldData(field_id)) {
        return false;
    }
    if (row_count == 0) {
        return true;
    }
    auto raw_chunk_count = segment.num_chunk_data(field_id);
    if (raw_chunk_count == 0 ||
        segment.num_rows_until_chunk(field_id, 0) != 0) {
        return false;
    }
    size_t raw_row_count = 0;
    for (int64_t chunk = 0; chunk < raw_chunk_count; ++chunk) {
        raw_row_count += segment.chunk_size(field_id, chunk);
    }
    if (raw_row_count < row_count) {
        // A partially indexed field may only retain raw data for a suffix of
        // the segment. Do not reinterpret that suffix as logical offset zero.
        return false;
    }

    int64_t chunk_id = 0;
    int64_t chunk_pos = 0;
    segcore::SegmentChunkReader reader(op_ctx, &segment, row_count);
    auto accessor =
        reader.GetMultipleChunkDataAccessor(segment.GetFieldDataType(field_id),
                                            field_id,
                                            chunk_id,
                                            chunk_pos,
                                            segcore::PinnedIndexView{});
    for (size_t offset = 0; offset < row_count; ++offset) {
        auto value = accessor();
        if (value.has_value()) {
            visitor(offset, GroupKey<T>(segcore::get_from_variant<T>(value)));
        } else {
            visitor(offset, GroupKey<T>(std::nullopt));
        }
    }
    return true;
}

template <typename T>
std::optional<TargetBitmap>
BuildRawMembership(milvus::OpContext* op_ctx,
                   const segcore::SegmentInternalInterface& segment,
                   FieldId field_id,
                   size_t row_count,
                   const std::vector<GroupKey<T>>& groups,
                   const TargetBitmap* eligible_rows) {
    std::unordered_map<GroupKey<T>, bool> requested_groups;
    requested_groups.reserve(groups.size());
    for (const auto& group : groups) {
        requested_groups.emplace(group, true);
    }

    TargetBitmap membership(row_count, false);
    auto scanned = ScanRawField<T>(
        op_ctx, segment, field_id, row_count, [&](size_t offset, auto group) {
            if (IsEligible(eligible_rows, offset) &&
                requested_groups.find(group) != requested_groups.end()) {
                membership[offset] = true;
            }
        });
    if (!scanned) {
        return std::nullopt;
    }
    return membership;
}

std::shared_ptr<TargetBitmap>
BuildEligibleRows(const TargetBitmap* base_filter) {
    if (base_filter == nullptr) {
        return nullptr;
    }
    auto eligible_rows = std::make_shared<TargetBitmap>(base_filter->clone());
    eligible_rows->flip();
    return eligible_rows;
}

}  // namespace

template <typename T>
std::optional<GroupMembership<T>>
BuildGroupMembership(milvus::OpContext* op_ctx,
                     const segcore::SegmentInternalInterface& segment,
                     FieldId field_id,
                     int64_t row_count,
                     const std::vector<GroupKey<T>>& groups,
                     const TargetBitmap* base_filter) {
    if (row_count < 0 ||
        (base_filter != nullptr &&
         base_filter->size() != static_cast<size_t>(row_count))) {
        return std::nullopt;
    }
    auto count = static_cast<size_t>(row_count);
    auto eligible_rows = BuildEligibleRows(base_filter);
    auto eligible_rows_ptr = eligible_rows.get();
    auto eligible_row_count =
        eligible_rows == nullptr ? count : eligible_rows->count();

    // Prefer the scalar index even when raw field data is also loaded. In()
    // obtains all requested offsets without walking the scalar column.
    auto index_probe = BuildIndexMembership<T>(
        op_ctx, segment, field_id, count, {}, eligible_rows_ptr);
    if (index_probe.has_value()) {
        std::vector<size_t> group_row_counts;
        group_row_counts.reserve(groups.size());
        for (const auto& group : groups) {
            auto membership = BuildIndexMembership<T>(
                op_ctx, segment, field_id, count, {group}, eligible_rows_ptr);
            if (!membership.has_value()) {
                return std::nullopt;
            }
            group_row_counts.emplace_back(membership->count());
        }
        auto membership_builder =
            [op_ctx, &segment, field_id, count, eligible_rows](
                const std::vector<GroupKey<T>>& batch_groups) {
                return BuildIndexMembership<T>(op_ctx,
                                               segment,
                                               field_id,
                                               count,
                                               batch_groups,
                                               eligible_rows.get());
            };
        return GroupMembership<T>(eligible_row_count,
                                  groups,
                                  std::move(group_row_counts),
                                  GroupMembershipSource::ScalarIndex,
                                  std::move(membership_builder));
    }

    std::unordered_map<GroupKey<T>, size_t> group_ordinals;
    group_ordinals.reserve(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) {
        group_ordinals.emplace(groups[i], i);
    }
    std::vector<size_t> group_row_counts(groups.size(), 0);
    auto scanned = ScanRawField<T>(
        op_ctx, segment, field_id, count, [&](size_t offset, auto group) {
            if (!IsEligible(eligible_rows_ptr, offset)) {
                return;
            }
            auto found = group_ordinals.find(group);
            if (found != group_ordinals.end()) {
                ++group_row_counts[found->second];
            }
        });
    if (!scanned) {
        return std::nullopt;
    }

    auto membership_builder =
        [op_ctx, &segment, field_id, count, eligible_rows](
            const std::vector<GroupKey<T>>& batch_groups) {
            return BuildRawMembership<T>(op_ctx,
                                         segment,
                                         field_id,
                                         count,
                                         batch_groups,
                                         eligible_rows.get());
        };
    return GroupMembership<T>(eligible_row_count,
                              groups,
                              std::move(group_row_counts),
                              GroupMembershipSource::RawField,
                              std::move(membership_builder));
}

template std::optional<GroupMembership<bool>>
BuildGroupMembership<bool>(milvus::OpContext*,
                           const segcore::SegmentInternalInterface&,
                           FieldId,
                           int64_t,
                           const std::vector<std::optional<bool>>&,
                           const TargetBitmap*);
template std::optional<GroupMembership<int8_t>>
BuildGroupMembership<int8_t>(milvus::OpContext*,
                             const segcore::SegmentInternalInterface&,
                             FieldId,
                             int64_t,
                             const std::vector<std::optional<int8_t>>&,
                             const TargetBitmap*);
template std::optional<GroupMembership<int16_t>>
BuildGroupMembership<int16_t>(milvus::OpContext*,
                              const segcore::SegmentInternalInterface&,
                              FieldId,
                              int64_t,
                              const std::vector<std::optional<int16_t>>&,
                              const TargetBitmap*);
template std::optional<GroupMembership<int32_t>>
BuildGroupMembership<int32_t>(milvus::OpContext*,
                              const segcore::SegmentInternalInterface&,
                              FieldId,
                              int64_t,
                              const std::vector<std::optional<int32_t>>&,
                              const TargetBitmap*);
template std::optional<GroupMembership<int64_t>>
BuildGroupMembership<int64_t>(milvus::OpContext*,
                              const segcore::SegmentInternalInterface&,
                              FieldId,
                              int64_t,
                              const std::vector<std::optional<int64_t>>&,
                              const TargetBitmap*);
template std::optional<GroupMembership<std::string>>
BuildGroupMembership<std::string>(
    milvus::OpContext*,
    const segcore::SegmentInternalInterface&,
    FieldId,
    int64_t,
    const std::vector<std::optional<std::string>>&,
    const TargetBitmap*);

}  // namespace milvus::exec
