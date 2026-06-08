add_library(s3_read_microbench_remote_input_stream OBJECT
    ${MILVUS_ENGINE_SRC}/storage/RemoteInputStream.cpp
)

set_target_properties(s3_read_microbench_remote_input_stream PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(s3_read_microbench_remote_input_stream PRIVATE
    ${MILVUS_ENGINE_SRC}
    ${MILVUS_THIRDPARTY_SRC}
    ${KNOWHERE_INCLUDE_DIR}
    ${MILVUS_COMMON_INCLUDE_DIR}
    ${MILVUS_STORAGE_INCLUDE_DIR}
    ${SIMDJSON_INCLUDE_DIR}
    ${CONAN_INCLUDE_DIRS}
)

add_executable(s3_read_microbench
    ${MILVUS_BINARY_DIR}/thirdparty/knowhere/knowhere-src/thirdparty/cardinalv2/cardinal/thread/task_pool.cpp
    ${MILVUS_COMMON_INCLUDE_DIR}/../src/common/EasyAssert.cpp
    $<TARGET_OBJECTS:s3_read_microbench_remote_input_stream>
    s3_read_microbench.cpp
)

set_target_properties(s3_read_microbench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(s3_read_microbench PRIVATE
    ${MILVUS_ENGINE_SRC}
    ${MILVUS_THIRDPARTY_SRC}
    ${MILVUS_BINARY_DIR}/thirdparty/knowhere/knowhere-src/thirdparty/cardinalv2
    ${KNOWHERE_INCLUDE_DIR}
    ${MILVUS_COMMON_INCLUDE_DIR}
    ${MILVUS_STORAGE_INCLUDE_DIR}
    ${SIMDJSON_INCLUDE_DIR}
    ${CONAN_INCLUDE_DIRS}
)

target_link_options(s3_read_microbench PRIVATE
    "LINKER:--allow-shlib-undefined"
)

target_link_libraries(s3_read_microbench PRIVATE
    cardinalv2
    knowhere
    milvus-storage
    milvus-common
)
