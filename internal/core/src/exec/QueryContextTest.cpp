// Licensed to the LF AI & Data foundation under one or more contributor
// license agreements. Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include <cstring>
#include <new>

#include "exec/QueryContext.h"

namespace milvus::exec {

TEST(QueryContextTest, RetrieveHasNoUninitializedVectorPlaceholders) {
    // Test object initialization, not dataset values. A zero-filled allocation
    // would hide the bug: retrieve does not call set_placeholder_group(), but
    // FilterBitsNode also serves retrieve plans when the search default is AUTO.
    alignas(QueryContext) unsigned char storage[sizeof(QueryContext)];
    for (int fill : {0x00, 0xff, 0xa5}) {
        std::memset(storage, fill, sizeof(storage));
        auto* context =
            new (storage) QueryContext("retrieve-context", nullptr, 0, 0);
        EXPECT_EQ(context->get_placeholder_group(), nullptr);
        context->~QueryContext();
    }
}

}  // namespace milvus::exec
