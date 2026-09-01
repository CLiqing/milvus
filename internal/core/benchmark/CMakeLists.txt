# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(fastmem_benchmark FastMemBenchmark.cpp)
target_include_directories(fastmem_benchmark PRIVATE ${CMAKE_HOME_DIRECTORY}/src)
target_link_libraries(fastmem_benchmark PRIVATE benchmark::benchmark)
install(TARGETS fastmem_benchmark DESTINATION benchmark)

add_executable(adaptive_filter_sink_benchmark AdaptiveFilterSinkBenchmark.cpp)
add_dependencies(adaptive_filter_sink_benchmark milvus_core)
target_include_directories(adaptive_filter_sink_benchmark PRIVATE
    ${CMAKE_HOME_DIRECTORY}/src
    ${CMAKE_HOME_DIRECTORY}/output/include
)
target_link_libraries(adaptive_filter_sink_benchmark PRIVATE
    benchmark::benchmark
    "$<TARGET_FILE:milvus_core>"
    milvus-common::milvus-common
    fmt::fmt
    Folly::folly
)
install(TARGETS adaptive_filter_sink_benchmark DESTINATION benchmark)

add_executable(adaptive_filter_producer_benchmark
    AdaptiveFilterProducerBenchmark.cpp)
set(PLANPARSER_LIB_DIR ${CMAKE_HOME_DIRECTORY}/output/lib)
target_include_directories(adaptive_filter_producer_benchmark PRIVATE
    ${CMAKE_HOME_DIRECTORY}/src
    ${CMAKE_HOME_DIRECTORY}/unittest
    ${CMAKE_HOME_DIRECTORY}/output/include
)
target_link_libraries(adaptive_filter_producer_benchmark PRIVATE
    benchmark::benchmark
    GTest::gtest
    milvus_core
    milvus_conan_deps
    knowhere
    milvus-storage
    ${PLANPARSER_LIB_DIR}/libmilvus-planparser-cpp.so
)
if (LINUX)
    target_link_options(adaptive_filter_producer_benchmark PRIVATE
        "LINKER:--allow-multiple-definition"
        "-L${PLANPARSER_LIB_DIR}")
endif()
set_target_properties(adaptive_filter_producer_benchmark PROPERTIES
    BUILD_RPATH "${PLANPARSER_LIB_DIR}"
    INSTALL_RPATH "${PLANPARSER_LIB_DIR}"
)
install(TARGETS adaptive_filter_producer_benchmark DESTINATION benchmark)
