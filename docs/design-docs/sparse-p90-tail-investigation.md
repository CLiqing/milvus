# Sparse 复合过滤 P90 劣化定位实验

日期：2026-08-17
目标：定位 Direct Sparse 复合过滤（A→B→Cardinal BF）相对 Dense 基线在 P90 上劣化（mean/median 显著更优、P90 却 +15%~+38%）的根本原因。

> **状态更新（2026-08-21）**：第 1--7 节保留的是归因逐步收敛的实验记录，
> 其中“1,000 个随机读取发生 TLB/EPT 重尾”的结论已被请求级操作量观测否定。
> 当前有效根因、修复与回归结果以第 8.6--8.7 节为准。

## 1. 背景与现象

已接受的 E2E 结论（见 `sparse-compound-filter-milvus-e2e-plan.md`）：

- 1M×128D 受控单 segment（rebase 后）：Dense mean 3.889ms / Sparse 3.164ms（-18.65%），median -29.04%，但 **P90 +38.13%**。
- Cohere 10M×768D：mean -11.15%、median -17.57%、**P90 +17.02%**。

即 Sparse 是"双峰/重尾"分布：median 显著更优，但最慢的 10% 反而比 Dense 的 P90 更慢。此前 flamegraph 聚合归因无法定位尾部阶段，本实验通过零重建的分阶段测量逐步收缩根因。

## 2. 已完成的实验与结论

复用已加载的 `cardinal_sparse_compound_1m`（1M×128、谓词 `a < 1000 and b < 500000`，A 命中 0.1%，B 条件命中约 50%，最终 ~500 valid）。

### 2.1 基线复现（20 窗口 ABBA，2000 req/mode）

| 指标 | Dense | Sparse | Sparse 相对 Dense |
|---|---:|---:|---:|
| mean | 4.208 ms | 3.572 ms | -15.1% |
| median | 4.157 ms | 3.143 ms | -24.4% |
| P90 | 4.527 ms | 6.013 ms | **+32.8%** |

P90 劣化稳定复现。

### 2.2 慢请求相关性（`--log-slow-ms 4.5`，20 窗口）

慢请求均匀散布在全部 50 个 query 上（每个 query 均出现过慢样本，计数 2~11 次不等），**非固定 query 依赖**。结合"50 个 query 的过滤结果完全一致"，指向系统级/代码级事件，而非数据依赖。

### 2.3 单核 pinning（`taskset -apc 3`）

Pinned 后 Dense P90 4.666ms / Sparse 5.863ms，差仍 **+25.7%**。排除 core 迁移 / L1-L2 cache 热度作为主因。

### 2.4 Go GC

metrics 端点 `go_gc_duration_seconds`：STW 最大 0.26ms。排除 GC 停摆。

### 2.5 阶段拆分（`internal_core_search_latency` scalar 直方图差分）

| 阶段 | Dense | Sparse |
|---|---:|---:|
| scalar(A+B) p50 / p90 / p99 | 1.50 / 1.90 / 1.99 ms | 0.57 / 2.31 / 3.90 ms |
| 其余（向量+RPC+Go）mean | 2.73 ms | 2.22 ms |

尾部在 **scalar 过滤阶段**，不在向量搜索或 RPC。

### 2.6 逐步定位到 B 消费者（scalar 阶段内 A vs B）

| 谓词（sparse 模式） | scalar p50 | p90 | p99 |
|---|---:|---:|---:|
| `a < 1000`（仅 A 生产者） | 0.50 | 0.90 | 0.99 ms（紧）|
| `a < 1000 and b < 500000`（A+B） | 0.57 | **2.32** | **4.36 ms** |

A 生产者本身无尾部，**尾部由 B 消费者（第二个谓词 `FilterSortedNativeIdsByRawData`）引入**。

### 2.7 V 缩放（A 候选数 100 / 1000 / 10000）

scalar 阶段 p90 ~2.4~2.8ms、p99 ~5.3~5.8ms，**不随 V 增长**。排除逐 ID 的散布访问成本。

### 2.8 连续 vs 散布（新建 `a[i]=i` 连续集合，B 仅碰 1 个 chunk）

scalar p90 ~2.4ms、p99 ~4.0ms，尾部仍在。**排除散布内存访问与 chunk pin 数量**。

## 3. 当前结论

P90 劣化完全由 B 消费者的执行引入，且是一个**固定延迟（~2–5ms）、概率约 10% 的偶发事件**，与候选数 V、ID 散布方式、核迁移、GC 均无关。

B 消费者代码路径（`internal/core/src/exec/expression/Expr.h:496` `FilterSortedNativeIdsByRawData`）内唯一的阻塞原语：

- `chunk_data` → `Span` → `CacheSlot::PinCells` → `SemiInlineGet(folly::SemiFuture)`（`cachinglayer/Utils.h:55`，`Future::get()` 在 cell 未命中时同步阻塞等待异步加载）；
- skip-index 的 `CanSkipBinaryRange` → `GetFieldChunkMetrics` → 同样 `PinCells` → `SemiInlineGet`（`index/SkipIndex.cpp:24`）。

这是之前 225ms 尾巴（per-ID `PinCells`）的同一家族，现已 amortize 到 chunk 级。

尚未钉死的一点：进程 RSS 仅 1.47GB（2GB cache 上限内），且"连续 ID 也复现"，因此不能 100% 断定阻塞来自 cell eviction，也可能是 B 每次请求的 `EnsureExecPathDetermined`/skip-index 快照/输出 vector 分配等固定开销的偶发慢路径。

## 4. 打点 + 重建定位（已完成）

在 `FilterSortedNativeIdsByRawData`（`Expr.h`）内对 5 个子阶段分别打点（`steady_clock` + 新注册的 `internal_core_search_latency` 家族直方图，单位 us）：

- `b_validate`：单调性校验遍历
- `b_group`：chunk 分组 + `get_chunk_by_offset` / `chunk_size`
- `b_skip`：skip-index `can_skip` 判定
- `b_pin`：`chunk_data` pin（含 `PinCells` 阻塞）
- `b_read`：逐 ID 读 + `match` + `push_back`

打点代码：`Monitor.h`/`Monitor.cpp`（新增 5 个直方图声明与定义）+ `Expr.h`（`FilterSortedNativeIdsByRawData` 内分阶段计时）。重建 `libmilvus_core.so`（ninja 增量，38 个对象）后重启隔离实例，指标已生效。

### 4.1 子阶段结果（1M×128，`a<1000 and b<500000`，20 窗口）

| phase | mean(us) | p50(us) | p90(us) | p99(us) |
|---|---:|---:|---:|---:|
| b_validate | 46.0 | 0.6 | 302 | 491 |
| b_group | 26.8 | 1.9 | 151 | 245 |
| b_skip | 0.4 | 0.5 | 0.9 | 1.1 |
| b_pin | 2.9 | 3.0 | 5.1 | 12.4 |
| b_read | 157 | 25.5 | **1209** | **1964** |

**结论：尾部在 `b_read`（逐 ID 读 + match 循环），不在 pin 或 skip-index。** 之前"`PinCells` 阻塞"的假设被否定（`b_pin` p99 仅 12us）。

### 4.2 b_read 呈双峰分布（2000 请求）

| 区间 | 请求数 |
|---|---:|
| 16–32 us | 1711（86%）|
| 32–64 us | 48 |
| 64–1024 us | 0（空档）|
| 1024–2048 us | 241（12%）|

`b_read` 是干净的**二值状态**：要么 ~25us（命中 cache），要么 ~1.5ms（未命中），中间无过渡。这正是"数据是否驻留 cache 层级"的典型特征。

### 4.3 子阶段随 V 缩放（100/1000/10000）

`b_read` 的 p90（~1.25ms）与 p99（~2ms）**不随 V 增长**，只有 p50/mean 随 V 缩放。与 2.7 的"scalar 尾部恒定"一致。

### 4.4 缓存与页故障证据

- `internal_cache_cell_access_miss_bytes_total{scalar_field,memory}` 在 1500 请求稳态 run 中 **delta = 0**：标量 cell（列 B）在搜索期**没有被 eviction**，40.6MB 的 miss 全部来自初始加载。→ 排除"chunk cache eviction 导致重载"。
- `internal_cache_cell_count{scalar_field,memory}=10`、`{vector_field,disk}=123`：标量列在 RAM（10 个 4MB cell），向量列走 mmap（123 cell 在 disk）。
- `perf stat`（30s 稀疏 run）：`minor-faults=10616`、`dTLB-load-misses=54M（1.91%）`。标量列虽在 RAM，但散布读触发 dTLB miss；在 8 核 VM（1MB L2 / 32MB L3、嵌套页表）上 TLB/EPT miss 代价被放大。

## 5. 最终结论

P90 劣化的根因是 **Sparse B 消费者的 `b_read` 阶段——逐 ID 的散布读取（`data[local_offset]` + `valid[local_offset]` + 分支 `match`）对 cache/TLB 驻留高度敏感**，呈二值分布（86% ~25us / 14% ~1.5ms）。

- Dense 的 B 是全列 SIMD 顺序扫描（带宽受限、可预取），cache 友好 → P90 紧。
- Sparse 的 B 是对列 B 的散布随机读（延迟受限），偶尔命中昂贵的 DRAM + TLB/EPT miss → P90 重尾。

这是"用散布读换取更少比较次数"的固有 trade-off：Sparse 省掉了 O(N) 的顺序扫描，但把 B 的访问模式从"顺序"变成了"随机"，引入了延迟方差。pin（`b_pin`）与 skip-index（`b_skip`）都不是问题；向量搜索阶段经 6.4 验证也是紧的（p99 ~1ms），不背锅。

### 修复结论汇总

| 方向 | 结果 |
|---|---|
| 1. b_read 软件 prefetch | **有效**：b_read p99 2ms→1ms（-50%）；endpoint P90 改善有限（-4%），因有 A 全列扫描 + RPC/Go 固定开销 |
| 2. huge page | **受阻**：内核仅 madvise 模式 THP，jemalloc `thp:always` 无效，AnonHugePages 恒 0 |
| 3. 极端 V 评估 | Sparse P90 在所有 V 劣于 Dense，极端 V（10/100）最差（-40%）；mean 收益 V≈1000 最大（+23%） |

综合判断：**Sparse 的 mean 收益真实存在（中等 V 最优），但 P90 重尾是散布读的固有代价，prefetch 只能减半、无法消除**（残留 ~1ms 疑为 TLB/EPT miss，prefetch 不填充 TLB）。落地时应对 P90 敏感场景做取舍，或在 milvus-storage 侧显式 `madvise(MADV_HUGEPAGE)` 后再评估 huge page 能否消除残留尾。

## 6. 修复方向实验（依次尝试）

### 6.1 方向 1：b_read 软件 prefetch（已实现，有效）

在 `b_read` 循环内对 `data[local_offset]` 和 `valid[local_offset]` 做 `__builtin_prefetch`（前瞻 `kPrefetchAhead=16` 个候选），重叠散布读的 DRAM/TLB 延迟。代码见 `Expr.h` `FilterSortedNativeIdsByRawData`。

| b_read | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| mean | 157 us | 119 us | -24% |
| p90 | 1208 us | 575 us | -52% |
| p99 | 1964 us | 979 us | -50% |

`b_read` 尾部减半。endpoint P90（V=1000）从 5.81ms → 5.56ms（-4.3%），改善有限——因为 endpoint P90 还包含 RPC/Go 固定开销与 A 全列扫描（见 6.4 的三段拆分；向量搜索阶段经确认是紧的，非尾部来源）。

### 6.1.1 prefetch 距离调优（k=16 vs 64）

| kPrefetchAhead | b_read mean | p90 | p99 | max |
|---|---:|---:|---:|---:|
| 16 | 125 us | 597 us | 981 us | 1020 us |
| 64 | 150 us | 639 us | 1008 us | **1877 us** |

`k=16` 优于 `k=64`：更大的前瞻距离反而使 mean 上升、max 尾巴拉长（过度 prefetch 污染 cache）。**软件 prefetch 已到收益上限**：残留 ~1ms 尾部是 TLB/EPT miss（`__builtin_prefetch` 不填充 TLB，见 4.4 的 `dTLB-load-misses 1.91%` 证据），无法靠继续调 prefetch 消除。

### 6.2 方向 2：huge page（受阻，环境不支持）

- 内核配置 `CONFIG_TRANSPARENT_HUGEPAGE_MADVISE=y`（仅 madvise 模式，无 `_ALWAYS`）。
- `echo always > /sys/kernel/mm/transparent_hugepage/enabled` 后，系统 `AnonHugePages` 仍为 0。
- milvus 用 jemalloc（`libjemalloc.so.2`）；`MALLOC_CONF=thp:always` 重启后进程 `AnonHugePages` 仍为 0。

结论：本环境（VM）无法通过 THP "always" 或 jemalloc `thp:always` 获得 2MB 大页。要落地需在 milvus-storage 的标量 cell 分配处显式 `madvise(MADV_HUGEPAGE)` 并验证 VM 实际支持 2MB 页。**本方向暂缓，记为环境受阻。**

### 6.3 方向 3：极端稀疏 V 的 Sparse 收益评估（已完成）

Dense vs Sparse endpoint 对比（prefetch 后，12 窗口 ABBA）：

| case | dense mean | sparse mean | mean Δ | dense p90 | sparse p90 | p90 Δ |
|---|---:|---:|---:|---:|---:|---:|
| V=10 | 3.549 | 3.343 | +5.8% | 3.965 | 5.619 | **-41.7%** |
| V=100 | 3.728 | 3.175 | +14.8% | 3.933 | 5.508 | **-40.0%** |
| V=1000 | 4.316 | 3.315 | +23.2% | 4.549 | 5.489 | -20.7% |
| V=10000 | 4.781 | 4.199 | +12.2% | 5.036 | 6.237 | -23.8% |

结论：

- Sparse 的 **mean 收益在 V=1000 附近最大（+23%）**；V 极小（V=10）时收益只有 +5.8%，因为总耗时被 A 全列扫描 + RPC 固定成本摊薄。
- Sparse 的 **P90 在所有 V 都劣于 Dense，且极端 V 时最差（V=10/100 达 -40%）**：`b_read` 的双峰慢模式（固定 ~1ms）在小 V 下成为主导，把很小的总耗时拉出重尾。
- 即 **"极端稀疏 V 更该用 Sparse" 的直觉是错的**：mean 上极端 V 收益反而不高，P90 上极端 V 劣化最严重。Sparse 最划算的是中等 V（V≈1000）。

### 6.4 向量搜索阶段验证（已有 `vector_latency` 直方图，无需新增打点）

`VectorSearchNode` 已有 `internal_core_search_latency_vector`（`vector_latency`）打点。稀疏 run（prefetch 后，25 窗口）三段拆分：

| 阶段 | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| endpoint | 3.526 ms | 3.199 | 5.755 | 6.628 ms |
| scalar (A+B) | 0.885 ms | 0.570 | 2.378 | 3.905 ms |
| vector (Cardinal BF) | 0.414 ms | 0.500 | 0.901 | 0.991 ms |

**向量搜索阶段是紧的（p99 ~1ms），没有 mmap 页故障尾。** 之前"endpoint 尾部转移到向量搜索"的猜测**被否定**。endpoint P90 剩余的尾由 scalar 阶段（A 全列扫描 + `b_read`）与 RPC/Go 固定开销共同构成。

进一步拆分 A-only vs A+B（prefetch 后）：

| 谓词 | scalar p50 | p90 | p99 |
|---|---:|---:|---:|
| `a<1000`（仅 A） | 0.500 | 0.900 | 0.991 ms（紧）|
| `a<1000 and b<500000`（A+B） | 0.563 | 2.222 | 3.901 ms |

A 生产者仍无尾，scalar 的 p99 尾部（~3.9ms）依然由 B 消费者引入，其中 `b_read`（prefetch 后 p99 ~1ms）是最大单项。

## 7. 遗留与未定

- `b_read` 慢模式的精确触发层（L3 eviction vs TLB/EPT miss vs minor fault）尚未用 per-request trace 钉死；`perf stat` 的 TLB/页故障证据指向内存层级延迟，但未定位到具体页/缓存行。
- 连续 ID（`a[i]=i`）集合的早期实验存在 skip-index 使 A 跳 chunk 的混淆，未能在同一打点版本下复测；如需彻底排除"散布 vs 顺序"的对比，应在打点版本下重建连续集合重测 `b_read`。
- Fix 2（huge page）受阻于环境（内核仅 madvise 模式 THP、jemalloc `thp:always` 无效），未落地。
- `b_read` 慢模式仍有 ~1ms（prefetch 后），是 endpoint P90 残留尾的主要可修项；若要继续收敛，可尝试更大的 prefetch 距离、或按 chunk 预取整块列 B 数据。

## 8. 2026-08-21 归因修正与下一轮验证计划

### 8.1 对现有结论的修正

现有打点可以确认：Sparse 的慢请求发生在第二个 predicate 的
`FilterSortedNativeIdsByRawData` 调用期间，且 wall-clock 尾部主要落在
`b_read` 区间；但它**尚不能证明** 1--2ms 全部来自 scattered read 的 CPU/TLB
成本。当前 `steady_clock` 会把线程被抢占、阻塞和 page fault 等 off-CPU 时间一并
记入所在区间，进程级 `perf stat` 的 dTLB 数据也混合了 A 全列扫描、向量搜索和
其他线程，不能与单个慢请求建立因果关系。

下列现象与“纯 O(V) 随机读取成本”不完全一致：

- `b_read` 呈约 25us / 1--1.5ms 的双峰，而不是随 V 平滑增长；
- V=100/1000/10000 时慢模式延迟基本固定；
- 只遍历几十至一千个整数的 `b_validate` / `b_group` 也能记录数百 us 尾部；
- 旧单核实验将整个多线程 Milvus 进程限制在一个 CPU，不能排除调度竞争。

因此第 5 节“TLB/EPT miss 是最终根因”降级为**待验证假设**。目前可靠结论仅为：
尾部发生在 Sparse B consumer 的 wall-clock 区间，pin、skip-index 和 vector search
不是已观测到的主要尾部来源。

### 8.2 固定复现场景

首轮使用原始稳定复现点，不以 V=64、B 全通过的特殊 case 作为主归因样本：

| 项 | 固定值 |
|---|---|
| 数据 | 1M x 128D synthetic，单 sealed segment，collection 稳定且无 build/compaction |
| predicate | `a < 1000 and b < 500000`；B 输入 V=1000，最终约 500 valid |
| 路由 | Cardinal Tiered explicit BF；由 route counter / perf 栈确认 BF |
| 请求 | NQ=1、C1、topK=10、ef=64；固定 50 query |
| 运行 | 30s warmup；Sparse A-only、Sparse A+B、Dense；每模式至少 5,000 请求 |
| closure | 计时前比较 Dense/Sparse topK `(ID,distance)`；binary、index 和 segment 状态固定 |

### 8.3 请求级联合观测

在同一次 `b_read` 调用中记录：

| 指标 | 用途 |
|---|---|
| `wall_us` | 保留现有 endpoint 可见延迟 |
| `thread_cpu_us` | `CLOCK_THREAD_CPUTIME_ID`，只统计该线程实际运行时间 |
| `wall_us - thread_cpu_us` | 近似 off-CPU / 阻塞时间 |
| `ru_nvcsw` / `ru_nivcsw` | 主动切换 / 被抢占证据 |
| `ru_minflt` / `ru_majflt` | minor / major page-fault 证据 |
| V、chunk 数、近似 unique 4K pages、TID | 工作量闭环和调度 trace 关联 |

慢于 300us 的调用输出一条结构化诊断记录；日志发生在被测区间之后，只用于归因 run，
不将该 run 的 endpoint latency 当作正式性能结果。另以 histogram/counter 汇总所有请求，
避免只观察慢样本。

### 8.4 决策分流

| 观测 | 判断 | 后续实验 / 优化 |
|---|---|---|
| wall 高、thread CPU 低，context switch 增加 | 调度/off-CPU | `perf sched timehist`/`sched_switch` 对齐 TID；检查线程池和 CPU oversubscription |
| wall 高、thread CPU 低，page fault 增加 | 缺页 | page-fault trace；prefault / `MADV_WILLNEED` A/B |
| wall 约等于 thread CPU，且无 fault/switch | 真 CPU/内存层级成本 | 分开采集 cycles、instructions、cache miss、dTLB miss，并按 V/unique pages 归一化 |
| `b_read` 正常但 endpoint 仍慢 | 残差在其他阶段 | 扩展 FilterBits、MVCC、payload handoff、vector result 收尾的请求级打点 |

若确认是 CPU/内存成本，再执行 V=64/100/1000/10000 与
random/clustered/continuous 的 scale sweep，拟合：

```text
thread CPU cost = fixed cost + V * per-ID cost
wall cost = thread CPU cost + off-CPU stall
```

### 8.5 优化候选与准入条件

1. **Chunk all-match**：min/max 与 null 信息证明 predicate 对整个 chunk 全通过时，
   直接复用输入 IDs；优先覆盖 V=64、`b < 1000000` case。
2. **可信 Sparse 不变量**：payload 携带 ascending/unique/universe/chunk spans，移除
   consumer 重复 validation/grouping；先以 thread CPU 数据确认实际收益上限。
3. **Index/cache membership**：B 已有 Bitmap posting、scalar index 或 cached Dense
   result 时，对 V IDs 做 membership，避免 raw scalar scattered read。
4. **内存层级优化**：只有请求级数据证明 TLB/page residency 是根因后，才尝试
   huge page、prefault、page-aware grouping 或 gather kernel。

最终优化验收回到完整 Milvus E2E：30s warmup、固定 query、12-window ABBA，报告
mean/median/P90，并保持正确性和 Dense/Sparse route closure。

### 8.6 首轮联合观测结果与计划收敛

按 8.2 的固定场景完成 30s 预热并连续执行 8,908 个 Sparse 请求后，新加入的
`CLOCK_THREAD_CPUTIME_ID`、thread `rusage` 与慢调用日志得到以下结果：

| 观测 | 结果 |
|---|---:|
| `b_read` 调用数 | 8,908 |
| `b_read` mean wall | 148.9 us |
| `b_validate` mean | 48.6 us |
| `b_group` mean | 27.3 us |
| `b_read` 快簇 | 7,826 次，`<=64us` |
| `b_read` 慢簇 | 约 1,070 次，`512--2048us` |
| 慢簇输入规模 | **全部为 V=500,000** |
| 慢簇 thread state | wall 约等于 thread CPU；通常 0 context switch、0 major fault，minor fault 0--2 |

这组证据否定了“同样遍历约 1,000 IDs，但约 12% 请求因调度/TLB 变慢”的当前主假设。
慢簇实际执行了约 500 倍的逻辑工作：它符合先由 `b < 500000` 生产 500,000 IDs、
再由 `a < 1000` 消费的反向顺序；预期快路径则是 A 先产出 1,000 IDs、B 再消费。
因而旧实验中看似固定概率的 1ms 尾部，首先应按**复合谓词物理执行顺序分叉**调查，
不能继续归因为 1,000 次 scattered read 的固有重尾。

当前 CPU/off-CPU 双时钟的包围顺序还会给 `thread_cpu_us` 多计约 3--6us，使很小的
`offcpu_us` 被裁成 0；下一轮会同时修正时钟顺序。该测量误差不影响上述结论，因为
V=500,000、无 context switch/page fault 且 wall 约等于 CPU 的数量级证据已经闭合。

下一轮按下列顺序执行：

1. 低扰动统计全部 `b_read` 的 `V<=1000` / `V>1000` 调用数，并对两类各采样少量
   expression-chain 日志，记录 child index、child `ToString()`、输入/输出 cardinality
   和 expression 实例标识；不使用全请求日志污染计时。
2. 用同一 collection、同一 filter 连续执行至少 2,000 请求，证明快簇是否严格对应
   `A -> B, V=1000`，慢簇是否严格对应 `B -> A, V=500000`，并确认是否与不同物理
   expression 实例、plan cache 或构建顺序相关。
3. 检查 `PhyConjunctFilterExpr::TryApplySparseFilter` 与 Dense `input_order_` 的关系。
   当前 Sparse 明确沿 `inputs_`，而 optimizer 只写 `input_order_`；若该差异是根因，
   修复为一套稳定且 Sparse-safe 的完整执行顺序，并增加 `a<1000 AND b<500000`
   单测，禁止选择 500k producer。
4. 修复后重建并执行至少 5,000 个请求：要求 B consumer 输入恒为约 1,000、1ms
   双峰消失、Dense/Sparse topK 一致、Cardinal BF route 不变；随后再按 12-window
   ABBA 报告 endpoint mean/median/P90。

只有第 2 步否定执行顺序分叉时，才回到 8.4 的调度/page-fault/PMU 分流；避免用
聚合 PMU 数据解释一个已经改变了 500 倍操作量的混合样本。

### 8.7 根因闭环、修复与回归结果

#### 8.7.1 根因

低扰动 cardinality counter 与每档前 16 条 expression-chain 采样证明，同一逻辑
filter 在运行时存在两条物理链：

```text
快路径：a < 1000 生产 1,000 IDs
        -> b < 500000 消费 1,000 IDs，输出约 503 IDs

慢路径：b < 500000 生产 500,000 IDs
        -> a < 1000 消费 500,000 IDs，输出约 503 IDs
```

修复前一次 7,687 次 `b_read` 的复现中，6,717 次输入 `V<=1000`，970 次输入
`V>1000`；后者约占 12.6%，并与 512--2048us 慢簇一一对应。慢样本中
`wall≈thread CPU≈1ms`，context switch、major fault 均为 0。由此可确认：旧 P90
双峰来自谓词物理顺序变化造成约 500 倍的逻辑工作，而非 1,000 个 scattered IDs
发生概率性调度、缺页或 TLB/EPT 停顿。

原始联合观测日志：

- `/tmp/milvus-sparse-fix-20260820/p90-order-diagnostic-workload.log`
- `/tmp/milvus.ip-10-15-6-115.ubuntu.log.INFO.20260821-074154.3216245`

#### 8.7.2 修复

修复包含两个缺一不可的部分：

1. raw-data range producer 与 BitmapIndex、ScalarIndexSort producer 使用相同的
   `DEFAULT_SPARSE_LIST_CAP=1000`；发现第 1,001 个命中后立即返回 `nullptr`，禁止
   `b < 500000` 生成 500k-ID payload。
2. conjunction 不再只消费所选 producer 后方的 child。它先寻找任意可用 producer，
   再将 producer 之外的**全部** conjunct 作为 consumer；因此 B 位于 producer 前方且
   因 cap 放弃生产时，后续选择 A 也不会漏掉 B 条件。

新增 `SparseAndAppliesConsumerBeforeSelectedProducer` 单测固化“consumer first、producer
second”语义，同时与已有 chained/nested/fallback Sparse conjunction 测试共同回归。

#### 8.7.3 验证配置

| 项 | 固定值 |
|---|---|
| collection | `cardinal_sparse_param_perf_1m_20260820` |
| 数据与 topology | 1M x 128D synthetic，单 sealed segment |
| predicate | `a < 1000 and b < 500000`，最终约 500 valid |
| route | legacy `valid_ids_per_query` + explicit Cardinal BF |
| query | NQ=1、C1、topK=10、ef=64、固定 50 query |
| warmup | 正式 ABBA 前已连续执行 5,000 个 Sparse 请求；脚本另执行 10 query/mode |
| correctness | 计时前比较前 10 query 的 Dense/Sparse topK `(ID,distance)` |
| 正式统计 | 12-window ABBA；每个 mode 每窗口 100 请求，共 1,200 请求/mode |

修复后的 `libmilvus_core.so` 加载路径已由 `/proc/<pid>/maps` 核实；Cardinal BF metrics
的请求数与 endpoint 请求数闭合，排除 ANN route 混入。

#### 8.7.4 正确性、操作量与性能回归

- `ConjunctExprTest.Sparse*`：5/5 通过。
- Dense/Sparse 前 10 个固定 query 的 topK ID 与 distance 完全一致。
- 连续 5,000 个 Sparse 请求后，counter 增量为：`V<=1000` 5,010 次（含 10 次
  warmup），`V>1000` 0 次；不再出现 `V=500000` 的 `sparse_b_read_slow`。
- 5,000 请求的 Sparse 分布为 mean 2.753ms、median 2.706ms、P90 3.021ms。
  其中有 1 次 23.4ms endpoint 异常，但没有对应的 `b_read` 慢日志，属于本次根因之外
  的系统/endpoint 偶发残差，不形成旧有约 12% 的双峰。

正式 12-window ABBA 结果：

| 指标 | Dense | Sparse | Sparse 相对 Dense |
|---|---:|---:|---:|
| mean | 3.831 ms | 2.716 ms | **-29.09%** |
| median | 3.819 ms | 2.698 ms | **-29.36%** |
| P90 | 4.011 ms | 2.907 ms | **-27.54%** |

Sparse 在 12/12 个 paired windows 中更快，paired-window mean improvement 为 29.08%。
因此本复现场景原有“mean/median 改善但 P90 劣化”的矛盾已经消失；修复后的三项延迟
指标方向一致。

原始回归产物：

- `/tmp/milvus-sparse-fix-20260820/p90-order-fixed-smoke.log`
- `/tmp/milvus-sparse-fix-20260820/p90-order-fixed-5000.log`
- `/tmp/milvus-sparse-fix-20260820/p90-order-fixed-5000-before.metrics`
- `/tmp/milvus-sparse-fix-20260820/p90-order-fixed-5000-after.metrics`
- `/tmp/milvus-sparse-fix-20260820/p90-order-fixed-abba12.log`
- `/tmp/milvus-sparse-fix-20260820/launch-p90-order-fixed-20260821.log`

#### 8.7.5 当前结论与剩余边界

本轮已经完成从操作量、物理调用链、正确性到 endpoint 回归的因果闭环。旧 P90 问题
不需要继续投入 huge page、prefetch 或 TLB 调优；这些方向建立在错误的混合样本归因上。

当前修复解决的是“某个宽谓词意外成为 Sparse producer”及“晚出现 producer 漏消费前置
conjunct”两个问题。后续若扩大 cap 或增加新的 producer，必须继续满足：producer 在达到
cap 后可早停、任意 child 顺序语义等价、实际 consumer 操作量有 counter 闭环。单次
23.4ms endpoint 残差未与 `b_read` 相关，不应重新算入该根因；只有它形成可重复分布时，
才按 8.4 对其他 endpoint 阶段另立实验定位。

### 8.8 Adaptive Dense|Sparse 链复测与最终修正（2026-08-25）

> 本节采用当前确定的产品语义：开启 Adaptive 后，每个 AND predicate 都允许接收
> Dense/Sparse，并在本 predicate 后重新产生 Dense|Sparse。它取代 8.7 中“寻找另一个
> Sparse producer”的临时实现，但保留 8.7 关于物理 predicate 顺序会改变工作量的证据。

#### 8.8.1 Cache 假设被计数器推翻

5,000 请求诊断中，Adaptive output 与 cache path 均为 5,010 次（含 10 warmup），但
cache path 全部为 `disabled`，hit/miss/put 均为 0。因此约 12% 慢簇与 expression cache
无关。修复前 scalar histogram 少约 13%，代码唯一对应分支为 conjunction 先得到 Dense，
FilterBits 最后再扫描 N-bit Dense 转回 Sparse。

#### 8.8.2 Dense intermediate 可以被后续 Sparse predicate 收缩

AND 链现在按以下规则逐 predicate 组合，不固定 A/B 顺序，也不重复执行 predicate：

1. 当前为 Dense、下一 predicate 为 Sparse：枚举下一 predicate 的 IDs，并用当前 Dense
   filtered bit 做 membership，直接输出交集 Sparse；
2. 当前与下一 predicate 都为 Dense：两个 filtered bitmap 做 OR，继续 Dense；
3. 下一 predicate 不支持 Adaptive：才走既有 Dense Eval fallback。

`ConjunctExprTest.*` 18/18 通过；新增断言覆盖 Dense -> Sparse 收缩、每个 child 只执行一次，
以及 Dense/Sparse 混合链在 A -> B、B -> A 两种顺序下结果一致。生产 E2E 的前 10 个固定
query 也闭合了 Dense/Adaptive topK ID 与 distance。修复后另重跑 visibility E2E，
sealed multi-segment + nullable、current delete、TTL expiry、growing + nullable 均通过；
当前 pymilvus historical timestamp API 未恢复 deleted IDs，因此该项仍由既有 MVCC 单测固化。

#### 8.8.3 真正的尾部成本：超阈值后仍逐 ID 写 Dense

仅修 conjunction 后，V=64 的 5,010 请求已全部记录 scalar、全部输出 Sparse，但只有
4,368 次进入 `b_read`；其余约 12.8% 是宽谓词先产生 Dense、窄谓词后产生 Sparse 的合法
顺序。此时 P90 仍为 5.388ms。V=1001 更暴露出实现问题：raw producer 在第 1,001 个命中
后虽然创建 Dense，却仍枚举其余所有命中并逐 ID 清 bit；`b < N` 因而执行约 1M 次单 bit
写入，正式 ABBA 中 Adaptive 比 Dense 慢 34.5%。

修复后，producer 只枚举阈值内前缀；超阈值的当前 chunk 及后续 chunk 直接把 SIMD compare
mask 以 32K-row 块写入 Dense（accepted mask 与 `1=filtered` Dense 用批量 XOR 转换），只把
此前已完成 chunk 中至多 T 个 ID 回填。总过程保持单趟，不重扫 N，也不再对剩余命中做
O(matches) 单 bit 写入。

#### 8.8.4 计数闭环与 E2E 结果

固定 1M x 128D、单 loaded sealed segment、L2/topK10/NQ1/C1、50 queries、每模式 10 次
加 30 秒 warmup、12-window `Dense -> Adaptive -> Adaptive -> Dense`，每模式 1,200 timed
requests。predicate 为 `a < V AND b < 1000000`，Cardinal 保持 auto route。

| V | Route | Adaptive 最终表示 | 指标 | Dense | Adaptive | Delta |
|---:|---|---|---|---:|---:|---:|
| 64 | BF | Sparse | mean | 3.600ms | 2.992ms | **-16.90%** |
|  |  |  | median | 3.560ms | 2.951ms | **-17.09%** |
|  |  |  | P90 | 3.935ms | 3.356ms | **-14.72%** |
|  |  |  | QPS | 275.96 | 331.73 | **+20.21%** |
| 1001 | IVF | Dense threshold fallback | mean | 4.857ms | 4.470ms | **-7.97%** |
|  |  |  | median | 4.829ms | 4.442ms | **-8.02%** |
|  |  |  | P90 | 5.151ms | 4.716ms | **-8.45%** |
|  |  |  | QPS | 204.80 | 222.47 | **+8.63%** |

两个点均为 12/12 paired windows Adaptive 更快。另一次当前代码的 5,000-request V=64
run 得到 mean/median/P90 = 3.035/2.998/3.388ms；计数为 5,010 BF、5,010 Sparse output、
5,010 scalar、5,010 cache-disabled，说明路径闭合。`b_read` 仍为 4,366 次，证明执行顺序
分叉仍存在，但批量 Dense fallback 后不再形成旧的 P90 双峰；因此问题不是“必须固定
predicate 顺序”，而是宽 predicate 的 Dense fallback 实现不应退化为逐 ID 写入。

原始产物位于：

- `/home/ubuntu/workspace/SparseProject/artifacts/sparse-dense-intermediate-fix-20260825/`
- `v64-bulk-5000.jsonl` 与对应 before/after metrics；
- `v64-bulk-fallback-abba12.jsonl`；
- `v1001-bulk-fallback-abba12.jsonl`。
- `visibility-regression.jsonl`。

### 8.9 修复后高维与并发 P90 回归（2026-08-26）

使用当前 Adaptive 实现补测真实 Milvus E2E：Cohere 1M x 768D、两个 sealed segment
（330K + 670K）、COSINE/topK10/NQ1、最终总 V=64，Dense/Adaptive 均为 Cardinal
Tiered auto-BF。固定 50 query、每模式 10 次加 30 秒 warmup、12-window ABBA；C1/C8/C60
每模式分别完成 1,200/9,600/72,000 requests。route counter 和 Sparse output counter
均闭合，12/12 paired windows 在三个并发点全部为 Adaptive 更快。

| Concurrency | Dense P90 | Adaptive P90 | Delta |
|---:|---:|---:|---:|
| 1 | 3.729 ms | 3.316 ms | **-11.07%** |
| 8 | 9.151 ms | 7.989 ms | **-12.70%** |
| 60 | 73.671 ms | 66.311 ms | **-9.99%** |

另在 128D homogeneous multi-segment 补测 4x50K 与 8x50K，P90 分别改善 4.51% 和
2.76%，也均为 12/12 windows 更快。由此，8.1--8.7 所记录的旧 P90 重尾在修正
predicate 链执行和 bulk Dense fallback 后，没有在高维、高并发或 8 segment 范围内
重新出现。现阶段不再把 P90 视为 T=1000 的独立阻塞项；若未来扩大 cap、支持 OR/Graph
或改变 producer，再按请求级 route/representation/阶段 counter 重新做 closure，不能
沿用旧的 `b_read/TLB` 推断。

原始数据位于
`/home/ubuntu/workspace/SparseProject/artifacts/sparse-next-stage-20260826/`；完整配置和
mean/QPS 结果见 `sparse-filter-landing-roadmap.md` 13.7。
