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
- [ ] 性能验证：baseline 与 downpush 对比 QPS、延迟、perf/flamegraph。

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
