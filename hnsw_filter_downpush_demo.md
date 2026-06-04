# HNSW 过滤条件下沉 Demo

## 目标

快速验证一个固定 50% 过滤场景下的收益：避免在上层构造大规模 scalar
过滤 bitmap，把判断下沉到 HNSW 图搜索过程中。

固定过滤条件：

```text
id >= 5000000
```

这是一个非正式本地 demo。Milvus 和 knowhere 都允许做侵入式源码改动。正确性范围只覆盖这个固定场景。

## 当前理解

- baseline 路径保持现有行为：
  `FilterBitsNode -> MvccNode -> VectorSearchNode -> knowhere Search`。
- 当前 scalar range index 路径会在 vector search 前构造完整过滤 bitmap。
  对严格 50% 命中率，`ScalarIndexSort::Range()` 会对约一半 row offset 执行
  `bitset[lb->idx_] = true`。
- demo 路径需要在 Milvus 侧跳过这段昂贵的命中 bitmap 写入，并在 knowhere 的
  HNSW 图搜索里增加硬编码判断：
  `id >= 5000000`。
- MVCC/delete/TTL 过滤继续走现有 bitset 路径。
- 路径切换由简单环境变量控制。

## TODO

- [x] 确认查询字段 `id` 在该 benchmark 路径中对应 HNSW internal vector id、
      segment offset 还是 primary key。
- [x] 确定环境变量名称和默认行为。
- [x] 在 Milvus 侧增加 demo 开关，仅对固定 `>= 5000000` 场景跳过 scalar range
      命中 bitmap 写入。
- [x] 在 knowhere 侧增加 HNSW demo predicate，把低于 `5000000` 的候选判定为无效。
- [x] 编译修改后的本地代码。
- [x] 做最小代码/构建验证，确认 baseline 和 demo 路径可切换。
- [x] 记录实现细节、构建命令和限制。

## 实现

环境变量：

```bash
MILVUS_DEMO_HNSW_FILTER_DOWNPUSH=1
MILVUS_DEMO_HNSW_FILTER_THRESHOLD=5000000  # 可选，默认 5000000
```

当 `MILVUS_DEMO_HNSW_FILTER_DOWNPUSH` 未设置或不等于 `1` 时，默认走 baseline。

Milvus 侧改动：

- 文件：
  - `internal/core/src/index/ScalarIndexSort.h`
  - `internal/core/src/index/ScalarIndexSort.cpp`
  - `internal/core/src/exec/operator/VectorSearchNode.cpp`
- 当 demo env 开启，并且 scalar range 恰好是 `GreaterEqual 5000000` 时，
  `ScalarIndexSort<T>::Range()` 在逐命中写 bitmap 循环之前直接返回
  `valid_bitset_.clone()`。
- 这样 scalar predicate 会先产生一个 all-pass match bitmap，再交给
  `FilterBitsNode` 翻转，从而避开对每个命中 row 执行
  `bitset[lb->idx_] = true` 的昂贵循环。
- 对 `ScalarIndexSort<int64_t>`，`Range()` 还会保存一个临时
  `DemoHnswScalarFilterView`，包含：
  - 已排序 scalar index entries
  - `seg_offset -> sorted-rank` 映射
  - valid-bitset 指针
  - row count
  - threshold
- `VectorSearchNode` 把该 view 复制到 search-local context，并给 `BitsetView`
  挂上 knowhere extra filter callback。

knowhere 侧改动：

- Milvus 分支中附带完整 patch：
  [`knowhere_hnsw_filter_downpush.patch`](knowhere_hnsw_filter_downpush.patch)。
- 文件：`/home/ubuntu/workspace/TmpWorker/knowhere-demo/include/knowhere/bitsetview.h`
- `BitsetView` 增加 demo extra filter callback，用来把 Milvus 传入的
  scalar index view 应用到底层候选 id 判断。
- 文件：`/home/ubuntu/workspace/TmpWorker/knowhere-demo/thirdparty/hnswlib/hnswlib/hnswalg.h`
- 实际 HNSW 搜索路径直接使用 `BitsetView`，不只经过 `BitsetViewIDSelector`。
  因此在以下路径里都做了 demo-aware HNSW 过滤：
  graph traversal、BF fallback、range fallback、iterator count check、effective
  filter ratio 计算。
- 文件：`/home/ubuntu/workspace/TmpWorker/knowhere-demo/include/knowhere/bitsetview_idselector.h`
- `BitsetViewIDSelector::is_member()` 也应用 demo predicate，使相关 Faiss selector
  路径行为一致。

HNSW 图搜索中的 demo predicate：

```cpp
rank = idx_to_offsets[seg_offset]
value = sorted_data[rank].a_
value >= MILVUS_DEMO_HNSW_FILTER_THRESHOLD
```

当 env 开启时，scalar index value 低于 threshold 的候选会在 knowhere 中被视为
filtered out。这避免了依赖 HNSW internal id 或 segment offset 恰好等于 primary key。

## 构建

命令：

```bash
jobs=8 CMAKE_EXTRA_ARGS='-DFETCHCONTENT_SOURCE_DIR_KNOWHERE=/home/ubuntu/workspace/TmpWorker/knowhere-demo' make build-cpp
```

结果：

- 构建成功。
- 产物包含 `internal/core/output/lib/libknowhere.so`。
- 产物包含 `internal/core/output/lib/libmilvus_core.so`。

## 验证记录

- 当前官方 master 的 baseline pre-filter search 仍使用
  `FilterBitsNode -> MvccNode -> VectorSearchNode -> knowhere Search`。
- 当前 `ScalarIndexSort::Range()` 对 50% 场景仍存在逐 offset bitmap 写入循环。
  历史 bulkset 临时实验不在当前本地源码中。
- HNSW+SQ4U/refine 相关 knowhere target 编译成功，包括
  `IndexSQ4Uniform.cpp`、`IndexHNSW.cpp` 和 `src/index/hnsw/*`。
- 已在 10M collection 上跑过端到端 benchmark。默认 threshold `5000000`
  对应本需求的 10M rows 50% filter。若测试 100M rows 50% 场景，需要设置
  `MILVUS_DEMO_HNSW_FILTER_THRESHOLD=50000000`。

## 早期本机 QPS 结果

公共配置：

- collection：`VDBBench`
- rows：`10,000,000`
- expr：`id >= 5000000`
- vector index：`HNSW_SQ`、`SQ4U`、`refine=FP16`
- metric：`COSINE`
- dim：`768`
- topK：`100`
- ef：`100`
- client/server：本机，`127.0.0.1:19530`
- concurrency：`8`
- duration：`60s`
- storage：EBS root disk，不是实例可用的 NVMe 盘

正确性检查：

- 查询向量选择 `pk == 5000000`。
- 使用 `expr='id >= 5000000'`、`topK=100` 搜索，输出 `pk,id`。
- 结果：`bad_count=0`，`min_id=5000000`，`max_id=9980689`。

Benchmark：

| 路径 | QPS | Avg RT | TP99 | Fails |
| --- | ---: | ---: | ---: | ---: |
| baseline, env off | 246.1336 | 32.492 ms | 51.893 ms | 0 |
| scalar-index downpush, env on | 253.2867 | 31.572 ms | 49.814 ms | 0 |

观测收益：QPS 约 `+2.9%`，收益不明显。

当时收益弱的可能原因：当前加载数据的 `id` 基本按插入顺序排列，因此也大致按 sealed
segment 排列。全局谓词 `id >= 5000000` 在每个 segment 内并不是均匀的 50% 过滤；
很多 segment 要么几乎全被过滤，要么几乎全有效。baseline 在构造 bitmap 后可以对
完全过滤的 segment 很快返回，而 downpush demo 会给 HNSW 一个 all-pass bitset，
再在图搜索 callback 中拒绝候选。这不代表每个 segment 内 scalar 分布近似随机 50%
的理想情况。

## 本地加载状态

- 删除旧 `/tmp/threadpool-milvus-volumes` backing store 后重建 runtime。旧的 8M
  partial collection 不可恢复，因为其 etcd metadata 位于被删除的 legacy store 中。
- 启动干净本地 etcd：
  `/home/ubuntu/workspace/TmpWorker/milvus-runtime/etcd`。
- 启动干净 MinIO container：`hnsw-demo-minio`，数据目录：
  `/home/ubuntu/workspace/TmpWorker/milvus-runtime/minio`。
- 使用 streamed train parquet 下载加载 VDBBench `Performance768D10M`，
  索引为 HNSW_SQ/SQ4U/FP16-refine。
- 最终 collection 状态：
  - collection：`VDBBench`
  - rows：`10,000,000`
  - vector index：`HNSW_SQ`、`SQ4U`、`refine=FP16`、`state=Finished`
  - load state：`Loaded`
- 当前 master 在 `state=Finished` 且 `indexed_rows == total_rows` 时仍可能显示
  `pending_index_rows` 非零；本次以 finished state 作为准确信号。
- 初始阶段未强制 compact，因为 index 完成后 root disk 只剩约 34G，compact
  可能需要大量临时空间。同一 collection 会复用于 baseline 和 downpush QPS 对比。
- 修复了临时 streamed parquet iterator，避免到达 EOF 后重新打开最后一个 train parquet。

## 限制

- demo 现在通过 `ScalarIndexSort<int64_t>` mapping 读取 scalar value，因此不再假设
  primary key 等于 HNSW internal id。
- 仍假设 knowhere `BitsetView::test()` 的输入可通过现有 `out_ids_` 机制映射到
  segment offset，再调用 extra filter callback。
- Milvus 侧 skip 为了快速 demo 故意做得比较宽：env 开启时，任何 numeric scalar
  index 的 `>= threshold` range 都可能命中，不限于正式识别出来的 `id` 字段。
- Env var 会被 knowhere helper 代码读取并缓存，因此必须在启动被测进程前切换。
- MVCC/delete/TTL 仍走现有 bitset 路径。该 demo 只去掉固定 range 条件下用户 scalar
  predicate 的 bitmap 构造成本。

## 注意事项

- 不要泛化到其他 selectivity。
- 不实现正式 expression callback API。
- 不依赖历史 bulkset 代码；那是之前的临时实验，不属于当前源码。

## Segment 聚合尝试

目标：把本地 `VDBBench` collection 从 40 个 sealed segments 降到约 8 个 segments，
使本地 perf profile 更接近之前的对比环境。

目标形态：

- 总行数 10M。
- 约 8 个 segments。
- 每个 segment 约 125 万 rows。
- retry compaction 前将 `dataCoord.segment.maxSize` 设置为 `6144` MB。

观察：

- 直接 `compact("VDBBench")` 只产生了单输入 `MixCompaction` 任务，没有合并 40 个 segments。
- `compact("VDBBench", target_size=6144)` 进入 ForceMerge 路径。
- ForceMerge 创建了两个 pending compaction tasks：
  - 约 155 万 rows，来自 5 个 input segments。
  - 约 845 万 rows，来自 35 个 input segments。
- 这不必然表示最终只输出 2 个 segments；ForceMerge executor 可以根据 `MaxSize`
  把一个大任务切成多个 result segments。
- 任务最初被前一批单输入 compaction 的 index work 阻塞。
- root disk 被打满，原因是旧 dropped segment 对象被 Milvus GC 保留。runtime object
  storage 主要占用：
  - `insert_log`：约 116G。
  - `index_v1`：约 52G。
- GC 配置会保留 dropped segment binlog 数小时
  （`dataCoord.gc.dropTolerance=10800`），自然清理对本 demo 太慢。

待办：

- [x] 确认 runtime config 使用 `dataCoord.segment.maxSize=6144`。
- [x] 尝试 `target_size=6144` ForceMerge。
- [x] 估算 dropped segments 的旧对象可回收空间。
- [x] 释放足够本地磁盘空间，支撑 ForceMerge/index output。
- [x] 等待 index tasks drain，并让 ForceMerge 执行。
- [x] 检查最终 persistent segment 数量和 row 分布。
- [x] 等待 ForceMerge 后续 SortCompaction/index 完成。
- [ ] 用 6-segment collection 重新跑 baseline/downpush QPS。

ForceMerge 后最终状态：

- Persistent segments：`6`。
- 每段 rows：
  - `148302`
  - `1547416`
  - `2075712`
  - `2076096`
  - `2076202`
  - `2076272`
- Total rows：`10,000,000`。
- Load state：`Loaded`。
- `vector_idx`：`Finished`，`indexed_rows=10000000`，`pending_index_rows=0`。
- `id_sort_idx`：`Finished`，`indexed_rows=10000000`，`pending_index_rows=0`。
- Remaining build index tasks：全部 `Finished`。
- Remaining compaction tasks：无。
- cleanup 和 compaction 后 root disk 剩余约 `60G`。

本次手动清理：

- 当前实验 collection 下的旧 dropped segment 对象被 GC 保留并打满 root disk。
- 只删除当前 persistent segments 和未完成 compaction input/output 集合之外的 segment
  object directories，范围限制在 `VDBBench` 的 `index_v1`、`insert_log`、`stats_log`
  和 `delta_log`。
- 该清理释放了足够空间，使 ForceMerge 和 index output 能够完成。

## 本机 6-Segment QPS 与 Perf

配置：

- Collection：`VDBBench`，10M rows，6 persistent segments。
- Segment rows：`148302`、`1547416`、`2075712`、`2076096`、`2076202`、`2076272`。
- Client 和 server 均在本机。
- Client：`/home/ubuntu/workspace/RemoteClient/benchmark`。
- Server：`127.0.0.1:19530`。
- Config：`/home/ubuntu/workspace/RemoteClient/config_filter_50pct.yaml`。
- Filter：`id >= 5000000`。
- Concurrency：`C60`。

QPS：

| 路径 | QPS | Avg RT | TP99 | 备注 |
| --- | ---: | ---: | ---: | --- |
| baseline, env off | about `1275` | about `47 ms` | about `72-73 ms` | 未采 perf |
| downpush, env on | about `2350-2390` | about `25 ms` | about `38-40 ms` | 未采 perf |

Perf 采集：

- Baseline Milvus child pid：`2285122`。
- 命令：
  `sudo perf record -F 99 -g --call-graph dwarf,65528 -p 2285122 -o /tmp/milvus_baseline_6seg_c60.perf.data -- sleep 60`
- Perf data：`/tmp/milvus_baseline_6seg_c60.perf.data`。
- Perf report：`/tmp/milvus_baseline_6seg_c60.report.symbols.txt`。
- Flamegraph SVG：`/tmp/milvus_baseline_6seg_c60.svg`。
- Folded stacks：`/tmp/milvus_baseline_6seg_c60.folded`。

baseline perf 观察：

- `milvus::index::ScalarIndexSort<long>::Range(...)` self time 在
  `MILVUS_SEARCH_*` threads 中合计约 `45.2%`。
- 生成 flamegraph 的 folded stack summary 显示，包含 `ScalarIndexSort`/`Range`
  的采样权重约 `47.5%`。
- `perf record` 使用 dwarf callgraph 会扰动本机 QPS：perf 期间 baseline QPS
  下降到约 `1110-1130`；perf 停止后恢复到约 `1240+`。路径对比应使用未采 perf
  的 QPS。

## 远端 Client Baseline Perf

配置：

- Server：本机 Milvus，`10.15.10.217:19530`。
- Client：远端机器 `10.15.13.40`。
- 路径：baseline，`MILVUS_DEMO_HNSW_FILTER_DOWNPUSH` 未设置。
- Collection 状态：`VDBBench` loaded，6 persistent segments，两个 index 都 finished。
- 采集时 Milvus child pid：`2292015`。
- perf 采集前远端压力已稳定。

Perf 采集：

- 命令：
  `sudo perf record -F 99 -g --call-graph dwarf,65528 -p 2292015 -o /tmp/milvus_remote_baseline_6seg_c60.perf.data -- sleep 60`
- Perf data：`/tmp/milvus_remote_baseline_6seg_c60.perf.data`。
- Perf report：`/tmp/milvus_remote_baseline_6seg_c60.report.symbols.txt`。
- Perf script：`/tmp/milvus_remote_baseline_6seg_c60.perf.script`。
- Folded stacks：`/tmp/milvus_remote_baseline_6seg_c60.folded`。
- Flamegraph SVG：
  [`milvus_remote_baseline_6seg_c60.svg`](milvus_remote_baseline_6seg_c60.svg)。

观察：

- `milvus::index::ScalarIndexSort<long>::Range(...)` self time 在
  `MILVUS_SEARCH_*` threads 中合计约 `45.05%`。
- folded stack summary 显示包含 `ScalarIndexSort`/`Range` 的采样权重约 `47.36%`。
- 采集报告有少量丢样：
  `Processed 106066 events and lost 14 chunks`。

## 远端 Client Downpush Perf

配置：

- Server：本机 Milvus，`10.15.10.217:19530`。
- Client：远端机器发压。
- 路径：downpush，`MILVUS_DEMO_HNSW_FILTER_DOWNPUSH=1`。
- 阈值：`MILVUS_DEMO_HNSW_FILTER_THRESHOLD=5000000`。
- Collection 状态：`VDBBench` loaded，6 persistent segments，两个 index 都 finished。
- 采集时 Milvus child pid：`2295654`。
- perf 采集前远端压力已稳定。

Perf 采集：

- 命令：
  `sudo perf record -F 99 -g --call-graph dwarf,65528 -p 2295654 -o /tmp/milvus_remote_downpush_6seg_c60.perf.data -- sleep 60`
- Perf data：`/tmp/milvus_remote_downpush_6seg_c60.perf.data`。
- Perf report：`/tmp/milvus_remote_downpush_6seg_c60.report.symbols.txt`。
- Perf script：`/tmp/milvus_remote_downpush_6seg_c60.perf.script`。
- Folded stacks：`/tmp/milvus_remote_downpush_6seg_c60.folded`。
- Flamegraph SVG：
  [`milvus_remote_downpush_6seg_c60.svg`](milvus_remote_downpush_6seg_c60.svg)。

观察：

- `milvus::index::ScalarIndexSort<long>::Range(...)` 没有出现在 perf report 中。
- folded stack summary 显示包含 `ScalarIndexSort`/`Range` 的采样权重约 `2.29%`；
  这部分主要来自 HNSW 的 `neighbor_range` 等非 scalar range 符号，不是原来的
  `ScalarIndexSort::Range`。
- 新增的 demo predicate
  `milvus::exec::(anonymous namespace)::DemoHnswScalarIndexFilteredOut(void*, long)`
  self time 约 `15.75%`，成为当前下沉路径的主要额外开销。
- HNSW search 主体
  `faiss::cppcontrib::knowhere::v2_hnsw_searcher<...>::search(...)` self time
  约 `14.75%`。
- SQ4U batch distance 计算
  `DistanceComputerSQ4UByte_avx512<...>::query_to_codes_batch_4_vnni(...)`
  self time 约 `11.69%`。
- 采集报告有少量丢样：
  `Processed 108840 events and lost 32 chunks`。

## 远端 Client QPS 对比

配置同上：Server 为本机 `10.15.10.217:19530`，远端 client 持续发压，
collection 为 `VDBBench`，10M rows，6 persistent segments，过滤条件
`id >= 5000000`，并发 `C60`。

| 路径 | 稳定窗口 | Avg RT | Median | TP99 | QPS |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline, env off | `2026-06-04 03:26:36` - `03:27:16` | `45.68 ms` | `45.24 ms` | `70.48 ms` | `1313.57` |
| downpush, env on | `2026-06-04 03:40:10` - `03:41:00` | `24.01 ms` | `23.80 ms` | `37.12 ms` | `2499.35` |

对比结果：

- QPS：`2499.35 / 1313.57 = 1.90x`，提升约 `+90.3%`。
- Avg RT：`45.68 ms -> 24.01 ms`。
- TP99：`70.48 ms -> 37.12 ms`。
