# Sparse 复合过滤 P90 劣化定位实验

日期：2026-08-17
目标：定位 Direct Sparse 复合过滤（A→B→Cardinal BF）相对 Dense 基线在 P90 上劣化（mean/median 显著更优、P90 却 +15%~+38%）的根本原因。

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
