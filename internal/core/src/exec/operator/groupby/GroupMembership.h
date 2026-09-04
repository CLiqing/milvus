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

#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "common/OpContext.h"
#include "common/Types.h"
#include "segcore/SegmentInterface.h"

namespace milvus::exec {

enum class GroupMembershipSource {
    ScalarIndex,
    RawField,
};

// Holds the cardinality of each requested group after the original search
// filter, plus a provider that can materialize one union bitmap at a time.
// Delayed materialization avoids retaining topK full-segment bitmaps.
template <typename T>
class GroupMembership {
 public:
    using GroupKey = std::optional<T>;
    using MembershipBuilder = std::function<std::optional<TargetBitmap>(
        const std::vector<GroupKey>&)>;

    GroupMembership(size_t eligible_row_count,
                    std::vector<GroupKey> groups,
                    std::vector<size_t> group_row_counts,
                    GroupMembershipSource source,
                    MembershipBuilder membership_builder)
        : eligible_row_count_(eligible_row_count),
          groups_(std::move(groups)),
          group_row_counts_(std::move(group_row_counts)),
          source_(source),
          membership_builder_(std::move(membership_builder)) {
    }

    size_t
    EligibleRowCount() const {
        return eligible_row_count_;
    }

    const std::vector<GroupKey>&
    Groups() const {
        return groups_;
    }

    const std::vector<size_t>&
    GroupRowCounts() const {
        return group_row_counts_;
    }

    GroupMembershipSource
    Source() const {
        return source_;
    }

    std::optional<TargetBitmap>
    BuildMembership(const std::vector<GroupKey>& groups) const {
        return membership_builder_(groups);
    }

 private:
    size_t eligible_row_count_;
    std::vector<GroupKey> groups_;
    std::vector<size_t> group_row_counts_;
    GroupMembershipSource source_;
    MembershipBuilder membership_builder_;
};

// `base_filter` uses vector-search semantics: one means invalid. The returned
// membership bitmap uses scalar-index semantics: one means that the eligible
// row belongs to one of the requested groups.
template <typename T>
std::optional<GroupMembership<T>>
BuildGroupMembership(milvus::OpContext* op_ctx,
                     const segcore::SegmentInternalInterface& segment,
                     FieldId field_id,
                     int64_t row_count,
                     const std::vector<std::optional<T>>& groups,
                     const TargetBitmap* base_filter);

}  // namespace milvus::exec
