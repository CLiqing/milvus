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

#include <fmt/core.h>

#include <boost_ext/dynamic_bitset_ext.hpp>
#include <deque>
#include <memory>

#include "bitset/detail/element_wise.h"
#include "common/Types.h"
#include "common/EasyAssert.h"
#include "knowhere/bitsetview.h"

namespace milvus {

class BitsetView : public knowhere::BitsetView {
 public:
    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const std::nullptr_t value)  // NOLINT
        : knowhere::BitsetView(value) {     // NOLINT
    }

    BitsetView(const uint8_t* data, size_t num_bits)
        : knowhere::BitsetView(data, num_bits) {  // NOLINT
    }

    BitsetView(const uint8_t* data,
               size_t num_bits,
               size_t num_filtered_out_bits)
        : knowhere::BitsetView(data,
                               num_bits,
                               num_filtered_out_bits) {  // NOLINT
    }

    BitsetView(std::shared_ptr<const roaring_bitmap_t> bitmap,
               size_t num_bits,
               size_t num_filtered_out_bits)
        : knowhere::BitsetView(knowhere::BitsetView::FromOwnedRoaring(
              std::move(bitmap), num_bits, num_filtered_out_bits)) {  // NOLINT
    }

    static BitsetView
    FromOwnedRoaringValid(std::shared_ptr<const roaring_bitmap_t> bitmap,
                          size_t num_bits,
                          size_t num_filtered_out_bits) {
        return BitsetView(knowhere::BitsetView::FromOwnedRoaringValid(
            std::move(bitmap), num_bits, num_filtered_out_bits));
    }

    static BitsetView
    FromOwnedValidIdList(std::shared_ptr<const std::vector<int32_t>> ids,
                         size_t num_bits,
                         size_t num_filtered_out_bits) {
        return BitsetView(knowhere::BitsetView::FromOwnedValidIdList(
            std::move(ids), num_bits, num_filtered_out_bits));
    }

 private:
    explicit BitsetView(knowhere::BitsetView bitset)
        : knowhere::BitsetView(std::move(bitset)) {
    }

 public:
    BitsetView(const BitsetType& bitset)  // NOLINT
        : BitsetView((uint8_t*)(bitset.data()), bitset.size()) {
    }

    BitsetView(const BitsetTypePtr& bitset_ptr) {  // NOLINT
        if (bitset_ptr) {
            *this = BitsetView(*bitset_ptr);
        }
    }

    // Return whether all bits are set (vacuously true when empty).
    bool
    all() const {
        if (empty()) {
            return true;
        }
        if (has_out_ids()) {
            for (size_t i = 0; i < size(); ++i) {
                if (!test(i)) {
                    return false;
                }
            }
            return true;
        }
        return bitset::detail::ElementWiseBitsetPolicy<uint8_t>::op_all(
            data(), 0, size());
    }

    // Return whether no bit is set (vacuously true when empty).
    bool
    none() const {
        if (empty()) {
            return true;
        }
        if (has_out_ids()) {
            for (size_t i = 0; i < size(); ++i) {
                if (test(i)) {
                    return false;
                }
            }
            return true;
        }
        return bitset::detail::ElementWiseBitsetPolicy<uint8_t>::op_none(
            data(), 0, size());
    }

    BitsetView
    subview(size_t offset, size_t size) const {
        if (empty()) {
            return {};
        }

        AssertInfo(
            (offset & 0x7) == 0, "offset {} is not divisible by 8", offset);
        AssertInfo(offset + size <= this->size(),
                   "index out of range, offset={}, size={}, bitset.size={}",
                   offset,
                   size,
                   this->size());
        return {data() + (offset >> 3), size};
    }
};

}  // namespace milvus
