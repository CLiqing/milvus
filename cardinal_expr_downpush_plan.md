# Cardinal 过滤表达式下沉实验计划

更新时间：2026-06-04

## 背景

当前过滤搜索 baseline 会在上层执行 scalar range/filter，并遍历命中结果设置 bitset。大数据量、50% 过滤一类场景下，这一步会扫描大量 scalar 命中并写入大 bitset，成为主要开销。

新的实验方向不再以 knowhere HNSW 作为最终验证路径，而是面向 cardinal 图搜索路径：上层跳过逐点设置过滤 bit，底层 cardinal graph search 在访问候选点时根据下沉表达式即时判断候选是否合法。

## 当前目标

先做一个 UAT/demo 级验证：

- 只覆盖 cardinal graph search 路径。
- 过滤表达式暂只支持 `pk % P < T`。
- `P`、`T`、是否开启下沉、过滤率都通过 per-query search param 传递。
- baseline 和 downpush 必须走一致的搜索策略，以便对比 recall 和性能。
- 正确性验证优先：对比两种路径 topK id 集合/recall，理论上不应有差异。
- 性能验证其次：对比 baseline 与 downpush 的 QPS、延迟、perf/flamegraph。

## 非目标

第一版暂不处理以下内容：

- 高过滤率下 cardinal 自动切换搜索策略后的完整覆盖。
- 通用表达式解析和任意 scalar 表达式执行。
- 删除数据、动态 segment、复杂 MV/tenant 场景下的过滤率精确维护。
- 正式 PR 级别的 API 设计和长期兼容性。

这些限制是为了先验证核心收益：避免上层大规模扫描和设置 bitset。

## 方案理解

### Baseline

baseline 维持现状：

1. Milvus 上层执行过滤表达式。
2. 遍历命中/未命中集合并设置 bitset。
3. knowhere/cardinal 根据 bitset 进行过滤搜索。

### Downpush

downpush 通过 search param 开启：

1. Milvus 上层识别 `pk % P < T` 这类实验表达式。
2. 上层跳过逐点 `bitset[...] = true` 的大循环。
3. 仍然向 cardinal 传递等价过滤率或 filtered count，保证搜索策略和 baseline 一致。
4. 将 PK/scalar 只读访问能力、`P`、`T` 下沉到 cardinal。
5. cardinal graph search 判断候选时：
   - 拿到 cardinal internal id；
   - 通过已有 reorder map 转成外部 row offset；
   - 读取对应 PK；
   - 执行 `pk % P < T`；
   - 不满足表达式则视为 filtered out。

## 代码路径初步定位

### Milvus

旧 demo 改动仍在本地，但不再作为最终方向：

- `internal/core/src/exec/QueryContext.h`
- `internal/core/src/exec/operator/FilterBitsNode.cpp`
- `internal/core/src/exec/operator/FilterBitsNode.h`
- `internal/core/src/exec/operator/VectorSearchNode.cpp`
- `internal/core/thirdparty/knowhere/CMakeLists.txt`

新的 Milvus 侧仍需要承担：

- 解析/读取 search param。
- 判断是否开启 cardinal 表达式下沉。
- 跳过上层 bitset 设置。
- 提供 PK/scalar value 访问上下文。
- 传递过滤率/filtered count，保持 cardinal 搜索策略一致。

### Knowhere cardinal adapter

cardinal adapter 当前只把 knowhere bitset 转成 cardinal bitset：

- `cardinal/know/cardinal.cc`
- `CardinalIndexNode::Search(...)`
- `CardinalIterator(...)`

需要在这里把下沉上下文从 knowhere/milvus 桥接到 `cardinalv2::FilterCheckerView`。

### Cardinal

核心过滤入口：

- `cardinal/cardinal/algo/filter_checker.h`
- `FilterCheckerView::Valid()`
- `FilterCheckerView::Test()`

graph search 调用点：

- `cardinal/cardinal/operator/graph_operator.h`
- `filter_checker.Valid(...)`
- `bitset.Valid(...)`

搜索策略依赖过滤率：

- `cardinal/cardinal/searcher/switch_strategy.h`
- `GetFilterRatio()`
- `GetFilteredCount()`

## 实现计划

- [x] 确认 cardinal index 在 Milvus 镜像中的编译链路：knowhere 的 `WITH_CARDINAL`、cardinal 版本来源、是否需要指向个人分支/commit。
- [x] 定义 per-query search param 名称：
  - 是否启用下沉；
  - `mod_p`；
  - `mod_t`；
  - filtered ratio 或 filtered count。
- [x] Milvus 侧接入 search param，并把参数保存到当前 query/search context。
- [x] Milvus 侧在下沉开启时跳过上层逐点设置 bitset 的路径。
- [x] Milvus 侧提供 PK/scalar value 的只读访问上下文，生命周期覆盖一次 search。
- [x] knowhere/cardinal adapter 桥接下沉上下文到 cardinal `FilterCheckerView`。
- [x] cardinal `FilterCheckerView` 增加 demo 级 extra filter 能力。
- [x] cardinal graph search 中通过 `Valid/Test` 执行 `pk % P < T` 判断。
- [x] cardinal `BitCompressBatch64()` 批量 bitset 解码路径在 extra filter 开启时合并 callback 判断。
- [x] 保证 `GetFilterRatio()` / `GetFilteredCount()` 与 baseline 一致，必要时直接使用 search param 传入值。
- [x] 构建本地或镜像版本，确认 patch 进入实际运行二进制。
- [x] 正确性验证：baseline 与 downpush 对比 topK id 集合/recall。
- [x] 性能验证：baseline 与 downpush 对比 QPS、延迟、perf/flamegraph。

## 当前实现状态

### 分支与依赖

- Milvus：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush`
  - 分支：`demo/cardinal-expr-downpush`
  - 基线：当前仓库 `master` 快照，未继续 rebase 到最新 upstream。
  - upstream/master 对比：之前确认落后 16 个 commit，用户确认继续基于当前快照开发。
- Knowhere：`/home/ubuntu/workspace/TmpWorker/knowhere-expr-downpush`
  - 分支：`demo/cardinal-expr-downpush-filter`
  - commit：`2f029c80d95c23e3845d85cffc6638eac2cc7dc5`
  - Milvus `internal/core/thirdparty/knowhere/CMakeLists.txt` 已指向 `https://github.com/CLiqing/knowhere.git` 的该 commit。
- Cardinal：`/home/ubuntu/workspace/TmpWorker/cardinal`
  - 分支：`demo/cardinal-expr-downpush-filter`
  - commit：`a7a3c58cc66459950cf58c8d50aa8e361aa3ba4d`
  - Knowhere cardinal 依赖已指向 `https://github.com/CLiqing/cardinal.git` 的该 commit。

### Search Param

当前 demo 支持以下 per-query 参数：

- `cardinal_expr_downpush`：开启下沉，支持 bool 或字符串 `"true"`/`"1"`。
- `cardinal_expr_mod_p`：表达式 `pk % P < T` 中的 `P`。
- `cardinal_expr_mod_t`：表达式 `pk % P < T` 中的 `T`。
- `cardinal_expr_filtered_out_count`：显式传入 filtered out 数量，优先级最高。
- `cardinal_expr_filter_ratio`：显式传入过滤率，未传 count 时使用。
- `hnsw_expr_downpush`：保留旧 demo alias，仅用于兼容本地已有脚本。

Milvus 侧仍会校验实际表达式必须是主键 `INT64 % P < T`，且 search param 中的 `P/T` 需要与表达式一致。

### 当前代码行为

- baseline：不开启 `cardinal_expr_downpush` 时走原路径，上层执行表达式并设置 bitset。
- downpush：开启后，`FilterBitsNode` 返回全通过 bitset，跳过上层逐点设置 bitset。
- `VectorSearchNode` 在搜索前 pin 住 sealed segment 的 PK chunks，并向 knowhere `BitsetView` 设置 extra filter callback。
- knowhere cardinal adapter 把 extra filter callback 桥接到 cardinal `FilterCheckerView`。
- cardinal `FilterCheckerView::Test()` 先执行原 bitset 判断，再把 internal id 映射回外部 row offset，用 callback 判断 `pk % P < T`。
- cardinal `FilterCheckerView::BitCompressBatch64()` 在 extra filter 开启时不再只按 bitmap 批量解码，而是在 64 个 id 的小窗口内逐个合并 bitmap 与 callback 判断，避免 BF/scan 类路径绕过下沉表达式。
- `filtered_out_count` 会被传给 cardinal，用于保持搜索策略和 baseline 尽量一致。

### 编译验证

本地新建构建目录：

- `/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/cmake_build_cardinal`

配置方式：

- `INDEX_ENGINE=cardinal`
- 复用已有 conan toolchain：`cmake_build/conan/conan_toolchain.cmake`

已通过的目标：

- `cmake --build cmake_build_cardinal --target knowhere -j 16`
- `cmake --build cmake_build_cardinal --target milvus_exec -j 16`
- `cmake --build cmake_build_cardinal --target milvus_core -j 16`

补充检查：

- `git diff --check` 通过。

### 正确性验证

#### 当前 HNSW / Cardinal v9 验证

脚本：

- `/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/cardinal_hnsw_correctness_check.py`

结果文件：

- `/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/cardinal_hnsw_correctness_results_20260604.json`

本地 standalone，`HNSW`，`COSINE`，20,000 rows，50 queries，topK=20。启动时设置：

- `MILVUS_CONF_DATACOORD_TARGETVECINDEXVERSION=9`
- 独立 etcd rootPath：`by-dev-cardinal-hnsw-v9d`
- 独立 minio rootPath：`files-cardinal-hnsw-v9d`

日志确认当前路径：

- `use key HNSW_fp32 to create knowhere index HNSW with version 9`
- `Index.Deserialize with cardinal, index_file_path_ = files-cardinal-hnsw-v9d/index_v1/.../1/_mem.index.bin`
- `Cardinal success load with size = 7340646`

对比方式为 baseline 与 downpush 的 topK id 顺序完全一致。

结果：

- 过滤 1%，表达式 `id % 100 < 99`：`mismatch_count=0`
- 过滤 20%，表达式 `id % 100 < 80`：`mismatch_count=0`
- 过滤 50%，表达式 `id % 4 < 2`：`mismatch_count=0`
- 过滤 99%，表达式 `id % 100 < 1`：`mismatch_count=0`

本次额外修复：

- `VectorDiskIndex::update_load_json()` 原本只对 `CARDINAL_TIERED` 使用真实 `_mem.index.bin` remote prefix；`HNSW` version 9 走 Cardinal 时也需要同样处理，否则 Cardinal 会从本地 cache prefix 推导 `_mem.index.bin`，导致 load 失败。
- `cmake_build_cardinal` 的新产物需要同步到 `internal/core/output/lib` 后本地 `./bin/milvus` 才会加载到修改后的 `libmilvus_core.so` / `libknowhere.so` / `libcardinalv2.so`。

#### 历史 CARDINAL_TIERED 验证

脚本：

- `/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/cardinal_correctness_check.py`

结果文件：

- `/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/cardinal_correctness_results_20260604.json`

本地 standalone，`CARDINAL_TIERED`，`COSINE`，20,000 rows，50 queries，topK=20。对比方式为 baseline 与 downpush 的 topK id 顺序完全一致。

结果：

- 过滤 1%，表达式 `id % 100 < 99`：`mismatch_count=0`
- 过滤 20%，表达式 `id % 100 < 80`：`mismatch_count=0`
- 过滤 50%，表达式 `id % 4 < 2`：`mismatch_count=0`
- 过滤 99%，表达式 `id % 100 < 1`：`mismatch_count=0`

## 过滤率说明

本实验中，“1% 过滤”表示过滤掉 1%，保留 99%。

示例表达式：

- 过滤 1%，保留 99%：`pk % 100 < 99`
- 过滤 20%，保留 80%：`pk % 100 < 80`
- 过滤 50%，保留 50%：`pk % 4 < 2`
- 过滤 99%，保留 1%：`pk % 100 < 1`

为了保证 baseline/downpush 的搜索策略一致，第一版可以允许 search param 显式传入过滤率或 filtered count。

## 已知风险

- cardinal internal id 不能当作 PK 使用，必须通过已有 reorder map 转为外部 row offset 后读取 PK。
- cardinal 自动切到 BF/IVF 或 batch bitset scan 路径时，当前 demo 已在 `BitCompressBatch64()` extra filter 分支补充正确性覆盖；性能表现仍需单独评估。
- iterator/range search 路径会重新构造 filter view，如后续测试覆盖 iterator/range，需要补充桥接。
- 静态无删除场景下过滤率可以由 search param 保证；正式 PR 需要考虑删除、segment 变化和表达式统计。
- 当前旧 knowhere demo patch 不等同于 cardinal 方案，只能作为桥接思路参考。

## 代码检视注意事项

以下问题不影响当前 demo 的 happy path 验证，但后续扩大测试范围或走正式 PR 时需要处理：

- `BuildCardinalExprDownpushSearchContext()` 如果构建失败，当前代码不会 fail fast，而是继续以全通过 bitset 搜索；当前测试需确保 sealed segment、INT64 PK 字段可 pin、普通 search 路径正常。
- extra filter context 当前由 `VectorSearchNode::GetOutput()` 局部 `shared_ptr` 持有，并以裸指针传入 knowhere/cardinal；普通同步 search 路径生命周期足够，iterator、group-by、range 等会延后消费 iterator/filter view 的路径暂不支持。
- `FilterCheckerView::Valid()` 的空过滤判断依赖 `filtered_out_count`；当前测试需保证显式传入或默认计算出的 filtered count 与表达式一致，避免 count 为 0 时绕过 callback。
- 未显式传入 `filtered_out_count` 时，默认 count 按 row offset 分布估算；当前测试数据使用连续 PK 或显式传 count，非连续 PK、delete/TTL、组合 bitset 场景不依赖默认估算。
- `CARDINAL_TIERED` load 兼容 patch 当前使用 `INDEX_FILES[0]` 推导 mem index prefix；当前测试产物顺序满足该假设，后续正式化应按 `_mem.index.bin` suffix 查找目标文件。

## UAT/VDC 10M Collection 构建记录

时间：2026-06-04

### 镜像与实例

- VDC buildRecordId：`4474`
- repoOwner：`CLiqing`
- branch：`demo/cardinal-expr-downpush`
- imageTag：`2.6-20260604-305ef57d-d5f9299`
- fullImageUrl：`harbor-us-vdc.zilliz.cc/milvus/milvus:2.6-20260604-305ef57d-d5f9299`
- dbVersion：`v3.0.0-305ef57d-4474`
- env：`awswest` / UAT3 AWS
- create instance task：`10185`
- QTP 页面：`https://qtp.zilliz.cc/run/customizeTable?id=10185`
- instanceId：`in01-7223f29479a9597`
- instanceUri：`https://in01-7223f29479a9597.aws-us-west-2.vectordb-uat3.zillizcloud.com:19542`
- CU：`class-4-enterprise`
- useHours：`240`

### Segment 参数

修改任务：

- QTP task：`10186`
- 参数：`dataCoord.segment.maxSize=6144`
- `needRestart=true`
- 结果：`currentValue=6144`

### VectorDBBench 调整

本次只为构建数据集做了本地 VectorDBBench 调整：

- 已存在用户改动：`VDBBENCH_STREAM_TRAIN_DOWNLOAD=1` 时训练 parquet 按分片下载，插完即删，避免一次落盘 10 个训练文件。
- 本次新增临时改动：`MilvusHNSW` 默认传 `metric_type=MetricType.COSINE`，否则 Cohere HNSW 建索引会因为空 metric 被 Milvus 拒绝。

本地验证：

- `MilvusHNSW --dry-run` 显示 `metric_type=<MetricType.COSINE: 'COSINE'>`
- 临时小 collection 使用 `index_type=HNSW, metric_type=COSINE` 建索引成功。

### 构建命令

执行目录：

- `/home/ubuntu/workspace/TmpWorker/VectorDBBench`

命令形态：

```bash
DATASET_SOURCE=S3 \
VDBBENCH_STREAM_TRAIN_DOWNLOAD=1 \
NUM_PER_BATCH=2000 \
LOG_FILE=logs/cohere10m_cardinal_hnsw_vdc_load_20260604T164350Z.app.log \
python3.11 -m vectordb_bench.cli.vectordbbench milvushnsw \
  --case-type Performance768D10M \
  --db-label cardinal-hnsw-downpush-10m \
  --uri '<instance-uri>' \
  --user-name db_admin \
  --password '<redacted>' \
  --num-shards 1 \
  --replica-number 1 \
  --m 48 \
  --ef-construction 400 \
  --ef-search 300 \
  --k 100 \
  --num-concurrency 1 \
  --concurrency-duration 10 \
  --load-concurrency 8 \
  --skip-search-serial \
  --skip-search-concurrent
```

### 构建结果

- collection：`VDBBench`
- 数据集：`Performance768D10M` / Cohere 10M / 768 dim
- schema：
  - `pk INT64 primary`
  - `id INT64`
  - `vector FLOAT_VECTOR dim=768`
- vector index：`HNSW`
  - `metric_type=COSINE`
  - `M=48`
  - `efConstruction=400`
- scalar index：`id STL_SORT`
- 插入结果：`count=10000000`
- 插入耗时：`1639.70s`
- 最终 row count：`10000000`
- `vector_idx`：`state=Finished, total_rows=10000000, indexed_rows=10000000, pending_index_rows=0`
- `id_sort_idx`：`state=Finished, total_rows=10000000, indexed_rows=10000000, pending_index_rows=0`
- load state：`Loaded`

一次 smoke search 已通过，第一条 query 的 top10 PK：

```text
9004060, 4870944, 3147147, 8024994, 2941370, 843044, 7937392, 7828776, 8362330, 4143237
```

### 注意事项

- VDC 禁止 `GetPersistentSegmentInfo`，VectorDBBench 的 `Milvus._wait_for_segments_sorted()` 会失败；本次在插入完成后手动执行 `flush`，再通过 `describe_index()` 等待 `state=Finished/indexed_rows=total_rows`。
- `pending_index_rows` 曾短暂出现滞后和波动；最终已归零。
- 本次未依赖 `list_persistent_segments()` 统计 segment 数。后续如果必须确认 segment 数，需要走 Cloud/metrics/内部接口。
- `refresh_load()` 在一次手动脚本里长时间未返回，但另一个客户端确认 collection 已是 `Loaded`，因此终止了本地阻塞脚本。

## VDC 1M 静态构建 smoke

### 实例

- create task：`10191`
- modify segment size task：`10195`
- instance：`in01-c5e4489c88bc122`
- URI：`https://in01-c5e4489c88bc122.aws-us-west-2.vectordb-uat3.zillizcloud.com:19541`
- dbVersion：`v3.0.0-305ef57d-4474`
- segment size：`dataCoord.segment.maxSize=6144`

### 构建脚本

- 脚本：`/home/ubuntu/workspace/TmpWorker/VectorDBBench/cardinal_hnsw_1m_static_build.py`
- 日志：`/home/ubuntu/workspace/TmpWorker/qtp_records/cardinal_hnsw_1m_static_build_20260605T021448Z.log`
- 数据集：Cohere 1M / 768 dim，目录 `cohere_medium_1m`
- collection：`VDBBenchStatic1M`
- vector index：`HNSW, COSINE, M=48, efConstruction=400`
- scalar index：`id STL_SORT`

### 结果

- 数据集下载成功。
- collection/index 创建成功。
- 插入成功：`1,000,000` rows，耗时约 `240.57s`。
- flush 成功。
- HNSW index 完成：`total_rows=1000000, indexed_rows=1000000, state=Finished`。
- `id_sort_idx` 完成：`total_rows=1000000, indexed_rows=1000000, state=Finished`。
- compaction 完成：job id `466780305945301312`。
- load 失败，当前 load state 为 `NotLoad`。

load 错误：

```text
failed to Deserialize index, cardinal inner error at /root/milvus/internal/core/src/index/VectorDiskIndex.cpp:356
: collection not loaded[collection=466780305935828705]
```

### 初步判断

这次 1M smoke 已经证明 create/insert/flush/index/compact 流程能跑通，但还不能进入查询验证。失败点在 querynode load vector index 时 Cardinal `Deserialize()` 返回错误。

当前最可疑原因是镜像中的 `VectorDiskIndex::update_load_json()` 只对 `index_type=CARDINAL_TIERED` 使用真实 `_mem.index.bin` remote prefix；本次使用的是 `index_type=HNSW`，但 HNSW version 9 也走 Cardinal load，因此仍然需要同样的 prefix 修正。本地工作区已有未提交改动把 `HNSW` 纳入该兼容逻辑，但它不在当前镜像 commit `305ef57d20` 中。

本地验证：

- 使用本地编译出的 `libmilvus_core.so` 运行 `cardinal_hnsw_correctness_check.py`。
- 5K rows / 10 queries / topK=10 下，1%/20%/50%/99% case 的 topK 对比均为 `mismatch_count=0`。
- 本地日志确认 `index_type=HNSW` 走 Cardinal v9 load，且 `_mem.index.bin` 路径正确：
  - `Index.Deserialize with cardinal`
  - `Cardinal success load`
  - `Index.Deserialize with cardinal done`

### UAT 热补丁验证

没有重新构建镜像和实例，改用 k8s 连接现有 pod 替换 `.so` 验证：

- namespace：`milvus-in01-c5e4489c88bc122`
- pod：`in01-c5e4489c88bc122-milvus-standalone-6cd95899-xknqt`
- 原始尝试：直接覆盖 `/milvus/lib/libmilvus_core.so`、`libknowhere.so`、`libcardinalv2.so` 后重启 container。
  - 结果：container restart 会恢复镜像层，`/milvus/lib` 覆盖丢失。
  - 额外问题：本地 `libknowhere.so` 依赖镜像中不存在的 `libsimdjson.so.25`。
- 最终方案：
  - 只热补丁 `libmilvus_core.so`，保留镜像自带 `libknowhere.so` 和 `libcardinalv2.so`。
  - 额外放入本地 `libsimdjson.so.25`，满足本地 `libmilvus_core.so` 的新增依赖。
  - 将文件放在 pod 级 `emptyDir`：`/milvus/data/codex_hotfix_core/`。
  - patch `/milvus/tools/run.sh`，启动时把 `/milvus/data/codex_hotfix_core` 加到 `LD_LIBRARY_PATH` 最前面。
- 新进程加载路径已确认：
  - `libmilvus_core.so` 来自 `/milvus/data/codex_hotfix_core/libmilvus_core.so`
  - `libsimdjson.so.25` 来自 `/milvus/data/codex_hotfix_core/libsimdjson.so.25`
  - `libknowhere.so` 和 `libcardinalv2.so` 仍来自 `/milvus/lib`

热补丁后重新 load `VDBBenchStatic1M`：

- load state：`Loaded`
- row count：`1,000,000`
- vector index：`HNSW, COSINE, M=48, efConstruction=400`
- scalar index：`id STL_SORT`
- smoke search：使用 `id % 4 < 2` filter 查询 top5 成功返回，load state 保持 `Loaded`。
- smoke 只验证正确性和链路可用性，未做 QPS/latency 对比，因此不能据此判断性能收益。
- 日志确认：
  - `index_type":"HNSW"`
  - `index_files":[".../_mem.index.bin"]`
  - `use key HNSW_fp32 to create knowhere index HNSW with version 10`
  - `Index.Deserialize with cardinal`
  - `Cardinal success load with size = 1292748766`
  - `Index.Deserialize with cardinal done, row = 1000000 , dim = 768`
- Cardinal 量化/数据组件日志：
  - graph：`graph_Matrix_dataset_data0_I32_vector_DenseVec_quant_None_None_storage_FixedLengthMemory`
  - low precision dataset：`data0_U8_vector_DenseVec_quant_RBQ3_Cosine_storage_FixedLengthMemory`
  - high precision/refiner dataset：`data0_U8_vector_DenseVec_quant_RBQ8_Cosine_storage_FixedLengthMemory`
  - ivf centroid：`ivf_centroid_data0_Fp32_vector_DenseVec_quant_None_None_storage_FixedLengthMemory`

segment 情况：

- 配置层面：`dataCoord.segment.maxSize=6144` MB。
- collection 本身没有单独的 segment size 配置。
- load 日志确认 compact 后实际加载为 1 个 sealed segment：
  - `segmentNum=1`
  - `segmentID=466780305945301315`
  - `rowNum=1000000`
  - resource estimate：memory `2537.75 MiB`，disk `5395.53 MiB`

### UAT smoke perf

采样目标：

- instance：`in01-c5e4489c88bc122`
- pod：`in01-c5e4489c88bc122-milvus-standalone-6cd95899-xknqt`
- node：`ip-10-15-87-208.us-west-2.compute.internal`
- workload：`cardinal_perf_smoke_search.py`
- collection：`VDBBenchStatic1M`
- search：`pk % 4 < 2`，`topK=100`，`ef=300`，`concurrency=16`
- search params：`cardinal_expr_downpush=true`，`cardinal_expr_mod_p=4`，`cardinal_expr_mod_t=2`，`cardinal_expr_filtered_out_count=500000`

采样方式：

- 容器和 node host 默认没有 `perf`。
- 创建临时 privileged debug pod，`hostPID=true`，在 Ubuntu debug pod 内安装 `linux-tools-6.8.0-94-generic`。
- 对 Milvus host PID `971695` 采样：`perf record -F 99 --call-graph dwarf,4096 -p 971695 -- sleep 30`
- debug pod 已删除。

第一次采样误用 `id % 4 < 2`：

- 当前 collection 的 primary key 是 `pk`，`id` 只是 scalar field。
- demo 代码只对 primary field 下沉，因此第一次没有触发 downpush。
- perf 热点集中在 `ScalarIndexSort<long>::Reverse_Lookup` / `ArithOpIndexFunc<long, Mod, LessThan>`，这个结果只作为误用记录。
- 目录：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-smoke-20260605T030619Z`

正确 downpush 采样结果：

- workload result：`ok=91382 err=0 qps=2030.71`
- latency：avg `7.878 ms`，median `7.171 ms`，p99 `15.566 ms`
- samples：`12674`
- DSO 聚合：
  - `libcardinalv2.so`：children `32.91%`，self `23.29%`
  - `libmilvus_core.so`：children `16.80%`，self `11.88%`
  - `milvus` Go binary：children `40.41%`，self `31.25%`
  - kernel：self `14.53%`
- top Cardinal symbols 包括：
  - `cardinalv2::search::MemGraphSearcherImpl<...>::Search(...)`
  - `rabitqv2::rbq_distance_4<3ul, ...>`
  - `cardinalv2::algo::NeighborSetDoublePopList::Insert(...)`

产物：

- perf data：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-smoke-20260605T0310-downpush-pk/perf.data`
- flamegraph：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-smoke-20260605T0310-downpush-pk/flamegraph.svg`
- folded stacks：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-smoke-20260605T0310-downpush-pk/perf.folded`
- symbol report：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-smoke-20260605T0310-downpush-pk/perf-report-symbols.txt`
- DSO flat report：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-smoke-20260605T0310-downpush-pk/perf-dso-flat.txt`
- 用同一 1M 脚本重跑 smoke，优先确认 collection 能 load，再查 sealed/growing segment 状态。

### UAT 100M 正式构建

目标实例：

- instance：`in01-c5e4489c88bc122`
- endpoint：`https://in01-c5e4489c88bc122.aws-us-west-2.vectordb-uat3.zillizcloud.com:19541`
- namespace：`milvus-in01-c5e4489c88bc122`
- pod：`in01-c5e4489c88bc122-milvus-standalone-6cd95899-xknqt`

构建前状态：

- pod 曾重新创建，`/milvus/data` emptyDir 中的 hotpatch 丢失。
- 已重新拷贝：
  - `/milvus/data/codex_hotfix_core/libmilvus_core.so`
  - `/milvus/data/codex_hotfix_core/libsimdjson.so.25`
- `/milvus/tools/run.sh` 仍包含 hotpatch 逻辑。
- 重启 standalone container 后确认：
  - Milvus parent/child 进程的 `LD_LIBRARY_PATH` 包含 `/milvus/data/codex_hotfix_core`
  - `libmilvus_core.so` 和 `libsimdjson.so.25` 来自 `/milvus/data/codex_hotfix_core`
  - `libcardinalv2.so` 仍来自镜像内 `/milvus/lib`
- 已 release `VDBBenchStatic1M`，释放 1M smoke collection 的 loaded 资源。

构建脚本：

- `/home/ubuntu/workspace/TmpWorker/VectorDBBench/cardinal_hnsw_100m_static_build.py`

参数：

- dataset：`LAION 100M`
- collection：`VDBBenchStatic100M`
- vector：`FLOAT_VECTOR dim=768`
- primary key：`pk INT64`
- scalar field：`id INT64`
- vector index：`HNSW`
- metric：`L2`
- HNSW params：`M=48, efConstruction=400`
- scalar index：`id STL_SORT`
- `NUM_PER_BATCH=2000`
- insert batch size：`2000`
- periodic flush：每 `1,000,000` rows
- compaction target：`6144 MB`
- train parquet：`VDBBENCH_STREAM_TRAIN_DOWNLOAD=1`，逐 shard 下载，读完删除。

启动记录：

- 误启动的普通 `nohup` 会被本地执行器清理，未进入构建；改用 `setsid -f` 启动后台进程。
- dryrun 曾创建 `VDBBenchStatic100M_dryrun` 并验证到首个 train shard 下载，已删除该 collection。
- 正式启动时间：`2026-06-05T03:27:06Z`
- 进程：`2678121`
- log：`/home/ubuntu/workspace/TmpWorker/qtp_records/cardinal_hnsw_100m_static_build_20260605T032706Z.log`
- summary：`/home/ubuntu/workspace/TmpWorker/qtp_records/cardinal_hnsw_100m_static_build_20260605T032706Z.summary.json`

当前进度：

- `prepare_dataset` 完成。
- `create_collection` 完成。
- 已进入 `train-00-of-100.parquet` 插入。
- `2026-06-05 03:27:24`：已插入 `100,000` rows，约 `6360 rows/s`。
- `2026-06-05 03:29:39`：已插入 `1,000,000` rows，约 `6631 rows/s`。
- `2026-06-05 03:29:40`：第一次 periodic flush 完成。
- `train-00-of-100.parquet` 已删除，开始下载/插入 `train-01-of-100.parquet`。
- Milvus collection stats 观察值：`row_count=1,288,000`。
- vector index 状态：`HNSW, L2, InProgress`，`indexed_rows=1,000,000`。
- 本机数据集目录仅保留当前 train shard，加 test/gt 文件，约 `2.0G`。
- `2026-06-05 03:32:20`：已插入 `2,000,000` rows，约 `6406 rows/s`。
- `2026-06-05 03:32:22`：第二次 periodic flush 完成。
- `train-01-of-100.parquet` 已删除，`train-02-of-100.parquet` 下载完成并继续插入。

### UAT 10M 正式矩阵

用户确认原 `100M` 是误输入，本轮修正为 `10M`。误建的 `VDBBenchStatic100M` 已删除。

目标实例：

- instance：`in01-c5e4489c88bc122`
- endpoint：`https://in01-c5e4489c88bc122.aws-us-west-2.vectordb-uat3.zillizcloud.com:19541`
- namespace：`milvus-in01-c5e4489c88bc122`
- pod：`in01-c5e4489c88bc122-milvus-standalone-6cd95899-xknqt`
- collection：`VDBBenchStatic10M`
- dataset：Cohere 10M，dim 768，COSINE
- vector index：HNSW，`M=48`，`efConstruction=400`
- scalar index：`id STL_SORT`
- collection rows：`10,000,000`
- load state：`Loaded`

构建记录：

- 构建脚本：`/home/ubuntu/workspace/TmpWorker/VectorDBBench/cardinal_hnsw_10m_concurrent_build.py`
- build log：`/home/ubuntu/workspace/TmpWorker/qtp_records/cardinal_hnsw_10m_concurrent_build_20260605T033940Z.log`
- summary：`/home/ubuntu/workspace/TmpWorker/qtp_records/cardinal_hnsw_10m_concurrent_build_20260605T033940Z.summary.json`
- 并发插入：`load_concurrency=16`，`NUM_PER_BATCH=2000`
- 插入完成：`10,000,000` rows，`1611.9s`，约 `6203.8 rows/s`
- compact job：`466781408636563590`，状态 `Completed`
- compact 后早期加载观察：`totalSealedSegmentNum=12`，`growingSegmentNum=0`，`loadedSealedRowCount=10000000`
- 后续自动 compaction / query view 更新后，最终测试结束时日志确认：`totalSealedSegmentNum=5`，`growingSegmentNum=0`，`loadedSealedRowCount=10000000`

参数修正：

- 一开始 smoke 使用了 collection 级 `cardinal_expr_filtered_out_count=5000000`。
- Milvus 的 `FilterBitsNode` 按 segment 执行，C++ 会把该值 clamp 到当前 segment `active_count`，导致每个 segment 被认为接近 100% filtered，downpush 返回空结果。
- 修正为 per-query 传 `cardinal_expr_filter_ratio`，让每个 segment 使用同一过滤率估算搜索策略；50% smoke 随后 `mismatch=0`。

正式矩阵：

- 脚本：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/cardinal_filter_matrix_benchmark.py`
- 原始结果：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/filter-matrix-uat10m-20260605T042709Z.json`
- search：`topK=100`，`ef=300`，`concurrency=60`，每轮 `30s`
- correctness：每个过滤率取 `20` 条 query，对比 baseline/downpush ordered topK id。
- 表达式：`pk % 100 < T`，其中 `T=100-filter_rate`。

| 过滤率 | baseline QPS avg/min/max | downpush QPS avg/min/max | speedup | correctness |
|---:|---:|---:|---:|---:|
| 1% | 267.46 / 266.83 / 267.83 | 675.26 / 674.37 / 676.37 | 2.52x | mismatch=0 |
| 5% | 239.56 / 182.43 / 268.20 | 472.81 / 352.80 / 674.60 | 1.97x | mismatch=0 |
| 20% | 297.44 / 247.53 / 322.77 | 1226.02 / 1218.93 / 1234.47 | 4.12x | mismatch=0 |
| 40% | 321.91 / 321.33 / 322.23 | 1197.88 / 1194.87 / 1200.80 | 3.72x | mismatch=0 |
| 50% | 319.53 / 319.30 / 319.73 | 1139.48 / 1134.33 / 1142.43 | 3.57x | mismatch=0 |
| 60% | 314.73 / 314.50 / 314.97 | 1048.14 / 1048.00 / 1048.33 | 3.33x | mismatch=0 |
| 80% | 289.04 / 288.53 / 289.73 | 737.43 / 736.20 / 738.17 | 2.55x | mismatch=0 |

注意：

- 5% 第 2/3 轮存在明显抖动，baseline 和 downpush 都受影响；原始三轮未剔除。
- 其余过滤率三轮稳定。
- 当前 correctness 只覆盖本轮 demo 的静态、无删除、PK 连续场景。

50% 火焰图：

- workload 脚本：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/cardinal_filter_workload.py`
- debug pod：`perf-debug`，`hostPID=true`，同节点 `ip-10-15-87-208.us-west-2.compute.internal`
- Milvus host PID：`986170`
- perf command：`perf record -F 99 --call-graph dwarf,4096 -p 986170 -- sleep 45`
- 内核符号：本机缺少节点 `kallsyms/vmlinux`，SVG/报告保留用户态栈，内核符号细分不完整。

baseline：

- workload result：`ok=28635 err=0 qps=318.17`
- latency：avg `188.795 ms`，median `194.241 ms`，p99 `260.478 ms`
- perf data：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-uat10m-50pct-baseline-20260605T045001Z/perf.data`
- flamegraph：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-uat10m-50pct-baseline-20260605T045001Z/flamegraph.svg`
- folded stacks：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-uat10m-50pct-baseline-20260605T045001Z/perf.folded`
- symbol report：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-uat10m-50pct-baseline-20260605T045001Z/perf-report-symbols.txt`

downpush：

- workload result：`ok=101599 err=0 qps=1128.88`
- latency：avg `53.160 ms`，median `62.098 ms`，p99 `78.794 ms`
- perf data：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-uat10m-50pct-downpush-20260605T045404Z/perf.data`
- flamegraph：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-uat10m-50pct-downpush-20260605T045404Z/flamegraph.svg`
- folded stacks：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-uat10m-50pct-downpush-20260605T045404Z/perf.folded`
- symbol report：`/home/ubuntu/workspace/TmpWorker/milvus-expr-downpush/perf-results/cardinal-uat10m-50pct-downpush-20260605T045404Z/perf-report-symbols.txt`
