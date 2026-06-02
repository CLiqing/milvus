#-------------------------------------------------------------------------------
# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.
#-------------------------------------------------------------------------------

# Update KNOWHERE_VERSION for the first occurrence
milvus_add_pkg_config("knowhere")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES "")
set( KNOWHERE_VERSION v2.6.14 )
set( GIT_REPOSITORY  "https://github.com/zilliztech/knowhere.git")

message(STATUS "Knowhere repo: ${GIT_REPOSITORY}")
message(STATUS "Knowhere version: ${KNOWHERE_VERSION}")

message(STATUS "Building knowhere-${KNOWHERE_SOURCE_VER} from source")
message(STATUS ${CMAKE_BUILD_TYPE})

if ( BUILD_DISK_ANN STREQUAL "ON" )
    set(WITH_DISKANN ON CACHE BOOL "" FORCE )
else ()
    set(WITH_DISKANN OFF CACHE BOOL "" FORCE )
endif ()

if ( INDEX_ENGINE STREQUAL "cardinal" )
    set(WITH_CARDINAL ON CACHE BOOL "" FORCE )
else ()
    set(WITH_CARDINAL OFF CACHE BOOL "" FORCE )
endif()

if ( MILVUS_GPU_VERSION STREQUAL "ON" )
    set(WITH_CUVS ON CACHE BOOL "" FORCE )
endif ()

set( CMAKE_PREFIX_PATH ${CONAN_BOOST_ROOT} )
FetchContent_Declare(
        knowhere
        GIT_REPOSITORY  ${GIT_REPOSITORY}
        GIT_TAG         ${KNOWHERE_VERSION}
        SOURCE_DIR      ${CMAKE_CURRENT_BINARY_DIR}/knowhere-src
        BINARY_DIR      ${CMAKE_CURRENT_BINARY_DIR}/knowhere-build
        DOWNLOAD_DIR    ${THIRDPARTY_DOWNLOAD_PATH} )

FetchContent_GetProperties( knowhere )
if ( NOT knowhere_POPULATED )
    FetchContent_Populate( knowhere )

    if ( INDEX_ENGINE STREQUAL "cardinal" )
        set( KNOWHERE_CARDINAL_V2_CMAKE_DIR ${knowhere_SOURCE_DIR}/cmake/libs/cardinal/v2 )

        execute_process(
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${CMAKE_CURRENT_SOURCE_DIR}/cardinal_async_input_stream.h
                        ${KNOWHERE_CARDINAL_V2_CMAKE_DIR}/cardinal_async_input_stream.h
                RESULT_VARIABLE CARDINAL_ASYNC_HEADER_COPY_RESULT )
        if ( NOT CARDINAL_ASYNC_HEADER_COPY_RESULT EQUAL 0 )
            message( FATAL_ERROR "Failed to stage Cardinal async_input_stream.h" )
        endif()

        execute_process(
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${CMAKE_CURRENT_SOURCE_DIR}/cardinal_async_no_fetch_pool.patch
                        ${KNOWHERE_CARDINAL_V2_CMAKE_DIR}/cardinal_async_no_fetch_pool.patch
                RESULT_VARIABLE CARDINAL_ASYNC_PATCH_COPY_RESULT )
        if ( NOT CARDINAL_ASYNC_PATCH_COPY_RESULT EQUAL 0 )
            message( FATAL_ERROR "Failed to stage Cardinal async no-fetch-pool patch" )
        endif()

        set( CARDINAL_ASYNC_PATCH_HOOK ${CMAKE_CURRENT_SOURCE_DIR}/cardinal_async_patch_hook.patch )
        execute_process(
                COMMAND git apply --check ${CARDINAL_ASYNC_PATCH_HOOK}
                WORKING_DIRECTORY ${knowhere_SOURCE_DIR}
                RESULT_VARIABLE CARDINAL_ASYNC_PATCH_HOOK_CHECK_RESULT
                OUTPUT_QUIET
                ERROR_QUIET )
        if ( CARDINAL_ASYNC_PATCH_HOOK_CHECK_RESULT EQUAL 0 )
            execute_process(
                    COMMAND git apply ${CARDINAL_ASYNC_PATCH_HOOK}
                    WORKING_DIRECTORY ${knowhere_SOURCE_DIR}
                    RESULT_VARIABLE CARDINAL_ASYNC_PATCH_HOOK_APPLY_RESULT )
            if ( NOT CARDINAL_ASYNC_PATCH_HOOK_APPLY_RESULT EQUAL 0 )
                message( FATAL_ERROR "Failed to apply Knowhere Cardinal async patch hook" )
            endif()
        else()
            execute_process(
                    COMMAND git apply --reverse --check ${CARDINAL_ASYNC_PATCH_HOOK}
                    WORKING_DIRECTORY ${knowhere_SOURCE_DIR}
                    RESULT_VARIABLE CARDINAL_ASYNC_PATCH_HOOK_REVERSE_CHECK_RESULT
                    OUTPUT_QUIET
                    ERROR_QUIET )
            if ( NOT CARDINAL_ASYNC_PATCH_HOOK_REVERSE_CHECK_RESULT EQUAL 0 )
                message( FATAL_ERROR "Knowhere Cardinal async patch hook is neither applicable nor already applied" )
            endif()
        endif()
    endif()

    set( KNOWHERE_S3_READ_PATH_CONTEXT_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/knowhere_s3_read_path_context.patch )
    execute_process(
            COMMAND git apply --unidiff-zero --check ${KNOWHERE_S3_READ_PATH_CONTEXT_PATCH}
            WORKING_DIRECTORY ${knowhere_SOURCE_DIR}
            RESULT_VARIABLE KNOWHERE_S3_READ_PATH_CONTEXT_PATCH_CHECK_RESULT
            OUTPUT_QUIET
            ERROR_QUIET )
    if ( KNOWHERE_S3_READ_PATH_CONTEXT_PATCH_CHECK_RESULT EQUAL 0 )
        execute_process(
                COMMAND git apply --unidiff-zero ${KNOWHERE_S3_READ_PATH_CONTEXT_PATCH}
                WORKING_DIRECTORY ${knowhere_SOURCE_DIR}
                RESULT_VARIABLE KNOWHERE_S3_READ_PATH_CONTEXT_PATCH_APPLY_RESULT )
        if ( NOT KNOWHERE_S3_READ_PATH_CONTEXT_PATCH_APPLY_RESULT EQUAL 0 )
            message( FATAL_ERROR "Failed to apply Knowhere S3 read path context patch" )
        endif()
    else()
        execute_process(
                COMMAND git apply --unidiff-zero --reverse --check ${KNOWHERE_S3_READ_PATH_CONTEXT_PATCH}
                WORKING_DIRECTORY ${knowhere_SOURCE_DIR}
                RESULT_VARIABLE KNOWHERE_S3_READ_PATH_CONTEXT_PATCH_REVERSE_CHECK_RESULT
                OUTPUT_QUIET
                ERROR_QUIET )
        if ( NOT KNOWHERE_S3_READ_PATH_CONTEXT_PATCH_REVERSE_CHECK_RESULT EQUAL 0 )
            message( FATAL_ERROR "Knowhere S3 read path context patch is neither applicable nor already applied" )
        endif()
    endif()

    # Adding the following target:
    # knowhere
    add_subdirectory( ${knowhere_SOURCE_DIR}
                      ${knowhere_BINARY_DIR} )
endif()

# get prometheus COMPILE_OPTIONS
get_property( var DIRECTORY "${knowhere_SOURCE_DIR}" PROPERTY COMPILE_OPTIONS )
message( STATUS "knowhere src compile options: ${var}" )

set( KNOWHERE_INCLUDE_DIR ${knowhere_SOURCE_DIR}/include CACHE INTERNAL "Path to knowhere include directory" )
