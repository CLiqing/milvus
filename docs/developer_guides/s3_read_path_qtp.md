# S3 Read Path QTP Build Notes

This branch carries the Milvus/Cardinal S3 read-path demo as a single Milvus
build entry for QTP.

## Build Entry

Use the Milvus repository and commit as the QTP entry point:

```text
Repo owner: CLiqing
Repo: milvus
Branch: feat/milvus-s3-async-no-fetch-pool-demo
Build command: make jobs=4 index_engine=cardinal install
Dockerfile: ./build/docker/milvus/ubuntu22.04/Dockerfile
OS: ubuntu22.04
Arch: amd64
```

The Milvus commit owns the dependency closure through CMake. QTP does not need
local worktrees or `MILVUS_STORAGE_LOCAL_SOURCE_DIR`. The `index_engine=cardinal`
argument is required; plain `make jobs=4 install` builds `INDEX_ENGINE=knowhere`
and does not exercise the Cardinal S3 read path.

## Dependency Closure

The build intentionally keeps dependency changes in the Milvus repo:

```text
Milvus
  -> knowhere@b20270cf507d6d3fb943a9c1d89a4093762a7b53
       -> cardinal@8c664848bb052cd857ee6de0adbc6ed97345a099
          plus internal/core/thirdparty/knowhere/cardinal_async_*.patch/header
  -> milvus-common@e16257c
       plus internal/core/thirdparty/milvus-common/AsyncInputStream.h
  -> milvus-storage@9b817b1
       plus internal/core/thirdparty/milvus-storage/s3_async_read_path.patch
```

This avoids depending on local dirty worktrees during QTP image builds.

## Runtime Path Selection

The Milvus binary accepts one flag to select the S3 read path:

```bash
milvus run standalone --s3-read-path=baseline
milvus run standalone --s3-read-path=curl_multi
milvus run standalone --s3-read-path=crt
```

Optional tuning flags:

```bash
--s3-read-max-inflight=100
--s3-read-eventloops=8
--s3-read-crt-max-connections=100
--s3-read-crt-throughput-gbps=30
```

Path mapping:

```text
baseline:
  unset async S3 read-path env
  Cardinal/Knowhere fetch pool + sync GetObject

curl_multi:
  MILVUS_S3_GETOBJECT_ASYNC=1
  MILVUS_S3_CLIENT_COROUTINE=1
  MILVUS_S3_ASYNC_MAX_INFLIGHT=<flag>
  MILVUS_S3_CLIENT_COROUTINE_EVENTLOOPS=<flag>

crt:
  MILVUS_S3_GETOBJECT_ASYNC=1
  MILVUS_S3_CLIENT_CRT=1
  MILVUS_S3_ASYNC_MAX_INFLIGHT=<flag>
  MILVUS_S3_CLIENT_CRT_EVENTLOOPS=<flag>
  MILVUS_S3_CLIENT_CRT_MAX_CONNECTIONS=<flag>
```

If `--s3-read-path` is omitted, existing environment-variable behavior is kept.

## QTP Runtime Config

For cold S3 reads, keep vector index warmup disabled:

```yaml
queryNode:
  segcore:
    tieredStorage:
      warmup:
        vectorIndex: disable
```

S3 endpoint, bucket, root path, credentials, region, and virtual-host settings
can still be passed through normal Milvus configuration or environment
variables on the QTP instance.

## Validation Rules

Before comparing QTP performance metrics:

```text
1. Recall and first10 result IDs must match baseline.
2. curl_multi/CRT must report async path evidence in logs or metrics.
3. Async path max in-flight must be capped at the same value as baseline fetch pool pressure, normally 100.
4. Compare QPS, latency, process CPU, thread count, context switches, S3 bandwidth, and page faults under the same dataset and cold-load procedure.
```
