# Sparse 过滤表示落地路线图

日期：2026-08-18
关联：`sparse-p90-tail-investigation.md`（P90 定位实验）、`sparse-compound-filter-milvus-e2e-plan.md`（E2E 计划）。

## 1. 背景结论（来自 P90 定位实验）

- P90 劣化根因是 Sparse B 消费者 `b_read` 的散布读对 cache/TLB 驻留敏感（双峰 ~25us / ~1.5ms），prefetch 减半但无法消除（残留为 TLB/EPT miss）。
- 决策口径：**不看 P90，只看 mean/QPS**。
- 修复现状：prefetch 已落地（`kPrefetchAhead=16`）；huge page 受环境阻塞；向量搜索阶段确认无尾。

## 2. 落地设计

### 2.1 两套机制（核心拆解）

- **机制 A（生产者）**：过滤判定把原计划输出的 dense bitset 换成 sparse list，**不关心下游是谁**（下一个过滤判定 / 向量搜索）。
- **机制 B（消费者）**：过滤/搜索能消费 sparse list 输入（如 BF 按 list 枚举）。

机制 B（BF 消费）基本已有；本轮重点是把机制 A 做成可插拔、可控制、下游无关的生产。

### 2.2 sparse 只进 BF（阈值耦合，无 fallback）

Cardinal 路由（`switch_strategy.h:194`）：`filter_rate >= 0.985`（valid 占比 < 1.5%）时必走 BF。

因此：**把 sparse 决策阈值（比例 V/N）设在 1.5% 之下，sparse ⟹ BF 自动保证，无需 dense fallback**。当前实验阈值（~0.1%）远低于 1.5%，余量充足。不做"表示/调度彻底解耦"，而是用阈值让两者天然一致。

### 2.3 生产者分类

| producer | 能否精确知道 V | 产出是否 offset 升序 | 成本 |
|---|---|---|---|
| bitmap index 等值（Roaring posting）| ✅ `cardinality()` O(1) | ✅ | borrow（免费）|
| sort index range | ✅ `ub - lb` O(1) | ❌ value 序，需 sort | borrow + O(V log V) |
| raw-data 扫描 | ❌ 扫完才知道 | ✅ | scan（比 dense 略贵）|

单谓词与复合谓词都有收益（单谓词收益来自 BF 消费端省掉 O(N/64) 枚举）。

### 2.4 决策点（可插拔）

`ShouldUseSparse(producer_type, V, N, ...) -> bool`，输入优先用精确 V（bitmap/sort），raw-data 用 cap+fallback 或后续再议。策略（保守阈值 / cost model）做成可插拔，便于 A/B。

## 3. 与上游 roaring PR 的对照

| PR / Issue | 线 | 目标 | 状态 | 与我们 |
|---|---|---|---|---|
| milvus #51367 | A | 过滤结果用 Roaring 解内存带宽 | open（issue）| 同领域 |
| knowhere #1732 | A | BitsetView 支持 Roaring，搜索侧不重建 dense | **draft**，milvus 侧未发 | 同方向（list vs Roaring；Cardinal vs Faiss）|
| milvus #51140 bloom_match | B | 客户端下发近似 ID 集合谓词 | merged | 不重叠（输入端）|
| milvus #51968 roaring_match | B | 客户端下发精确 ID 集合谓词 | merged | 求值结果可作 producer |
| milvus #52586 | B | MRB1 codec 去重 | open | 无关 |

上游"线 A"（过滤结果传给搜索）尚未成型（knowhere #1732 仍是 draft、milvus 侧没发），我们做 sparse list 是并行/超前的方向；"线 B"（membership 输入谓词）已 merged，`roaring_match` 的求值结果天然可作机制 A 的一个 producer，是现成对接点。

## 4. 路线图

| 步骤 | 内容 | 状态 |
|---|---|---|
| **Step 1** | 表示与调度解耦：引入 `filter_result_representation` 控制（dense/sparse），与 `bf_filter_scan_mode` 拆分，加决策点骨架 | **已完成** |
| **Step 2** | producer 覆盖补齐：bitmap 等值（已通）、sort range（修 value-order）、raw-data（已通）| **已完成** |
| Step 3 | 决策策略 A/B：阈值 / cost model，跑完整收益矩阵 | 待办 |
| Step 4 | （可选）Graph/IVF 消费端（Bloom+Flat）| 待办 |

### Step 2 实现说明（已完成）

**改动 1：sort range 的 value-order 修复。**

`ScalarIndexSort::BuildValidIdsFromBounds`（`index/ScalarIndexSort.cpp`）：sort index 按 value 排序，`[lb, ub)` 迭代产出的是 value 序的 offset；sparse 消费者 `FilterSortedNativeIdsByRawData` 的"分 chunk 分组"优化要求 offset 升序（否则逐 ID pin，即之前的 225ms 尾巴）。加 `std::sort(ids->begin(), ids->end())` 把 offset 排序成升序（V 小时 O(V log V) 可接受）。

**改动 2：去掉消费者 `exec_path_ != RawData` 的顺序门槛。**

`UnaryExpr.cpp` / `BinaryRangeExpr.cpp` 的 `TryFilterNativeValidIds` 原先有 `EnsureExecPathDetermined(); if (exec_path_ != RawData) return nullptr;`。但消费者实际读的是 raw data（`FilterSortedNativeIdsByRawData` → `chunk_data`），根本不用索引；这个门槛导致"带索引的谓词被重排序成第二个谓词时，消费者拒绝消费 sparse"。删除后，消费者无条件读 raw data，谓词有没有索引、排什么位置都不影响（`FilterSortedNativeIdsByRawData` 内部有 `HasFieldData` 守卫兜底 fallback dense）。

**验证（隔离实例）：**

- sort 集合（`a` 有 STL_SORT 索引）：
  - `a<1000 and b<500000`：20 查询 dense/sparse topK 完全一致（之前间歇 10–20% 报错，现已 0 失败）；mean 2.94 vs 3.53ms（-16.6%）。
  - `a<1000` 单谓词：dense/sparse 一致；sparse mean 2.35 vs 2.52ms（-6.8%）、p90 2.70 vs 2.89ms——单谓词 sort range 有明确收益（省掉 dense 的 /64 枚举）。
- raw-data 集合（无索引）：回归验证 15 查询 0 失败 0 不一致，未破坏原有行为。

### Step 3 决策策略（proposal）

决策点：`ShouldUseSparse(producer_type, V, N) -> bool`，决定过滤结果用 sparse list 还是 dense bitmap。

**第一层（机制）：producer 能否产出 sparse**

| producer | V 是否精确已知 | 产出 sparse 的代价 |
|---|---|---|
| bitmap 等值 | ✅ `cardinality()` O(1) | borrow 现成 Roaring（升序），几乎免费 |
| sort range | ✅ `ub - lb` O(1) | borrow + O(V log V) 排序（升序）|
| raw-data 扫描 | ❌ 扫完才知道 | scan + ctz + push_back（比 dense 略贵）|

**第二层（策略）：是否该走 sparse**

- V 已知的 producer：`V/N <= threshold` → sparse，否则 dense。**精确阈值，无估计误差。**
- V 未知的 producer（raw-data）：默认 dense；若要支持，用 cap+fallback（前 X 个 valid 建 list，超 X 回退 dense），但单趟双输出成本偏高，先不做。

**threshold 的确定（两个约束 + cost model）**

1. 上界约束：`threshold < 1.5%`（Cardinal `filter_rate >= 0.985` 的 BF 切换点），保证 sparse ⟹ BF，无 fallback。
2. 收益交叉点（cost model）：sparse 的收益是省掉 dense 的 N/8 bitmap 物化 + N/64 枚举；sparse 的代价是 V log V 排序（sort）或 V 次散布读。对 borrow producer，交叉点约 `V/N ≈ c_enum / (64 · c_read) ≈ 0.1%`。
3. 实测调优：默认 `threshold = 0.1%`（与当前实验 sweet spot V=1000/1M 一致），跑收益矩阵（producer × V × 谓词形态）微调。

**cost model（单谓词 borrow producer 的粗略式）**

- dense ≈ N·c_scan + (N/8) 物化 + (N/64)·c_enum
- sparse ≈ borrow(≈0) + V·logV 排序 + V·c_read
- 交叉点 ≈ c_enum / (64·c_read)，代入量级得 ~0.1%。

**P90 备注**：决策只看 mean/QPS（既定口径），但 sparse 的 `b_read` 重尾（prefetch 后 ~1ms）会让 P90 劣化，尤其小 V 时更明显——落地时作为已知副作用记录，不作为决策项。

### Step 3 统一 cap+fallback（选定方案）

在"精确阈值（V 已知）/ 保守 dense（V 未知）"之外，采纳统一方案：**所有 producer 先走 sparse，list 一旦超过 cap（默认 1000）就回退 dense**。这样 raw-data（V 未知）也能统一参与，不必预先知道 V。

**三种实现方式的开销对比**（决定实现形态）：

| 方式 | 描述 | 问题 |
|---|---|---|
| (a) 天真双输出 | 同时建 list + dense，最后二选一 | low-V 场景也付 dense 成本，吃掉 sparse 收益 |
| (b) 天真 lazy-switch | 先 ctz+push_back，超 cap 改 set_bit | ctz 是 O(V)，fallback 时 V 大则不可忽略 |
| (c) **lazy-switch + early-break** | 到 cap 立即停止 ctz、改 direct-SIMD-store | 额外 ~1000 ctz/push_back/set，几 us，可忽略 |

选 (c)。实现要点：

- **sort/bitmap（V 已知）**：`if (V > cap) return nullptr`（O(1) 精确，无重扫）。
- **raw-data（V 未知）**：scan + ctz + push_back，`ids->size() >= cap` 时立即 `return nullptr`（fallback dense，重扫）。

注：raw-data 的 `return nullptr` 会触发 dense 重扫，代价 = sparse 扫到溢出点为止（early-break 使其只扫到第 cap 个 valid ID 处）+ 一次完整 dense 扫。对 V≫cap 溢出点在很前面，重扫代价≈完整 dense 扫（≈0）；对 V≈cap 溢出点在后面，重扫代价≈一次完整 scan（~0.3–0.5ms）。**这是当前 re-scan 版的固有代价，用 A/B 实测；若不可接受再升级为单趟 switch（不重扫）**。

**cap 参数**：默认 1000，先做成常量 `kSparseListCap`（`common/Consts.h`），后续接 querynode config 可调。

### Step 3 实现结果（已完成，cap+fallback + soft 语义）

**改动：**

- `common/Consts.h`：`DEFAULT_SPARSE_LIST_CAP = 1000`。
- `ScalarIndexSort::BuildValidIdsFromBounds` / `BitmapIndex::MaterializeValidIds`：`if (V > cap) return nullptr`（V 免费已知，O(1) 精确）。
- `UnaryExpr` RawData 生产者：ctz+push_back 循环里 `ids->size() > cap` 时立即 `return nullptr`（early-break）。
- **soft 语义**：`filter_result_representation=sparse` 从"硬性强制"改为"偏好"——`PlanProto.cpp` 不再把它翻译成 `bf_filter_scan_mode=valid_ids_per_query`；`VectorSearchNode` 改为**看实际 payload**（有 payload → sparse + 把 Cardinal scan mode 设成 `valid_ids_per_query`；无 payload → dense + `auto`）。legacy `bf_filter_scan_mode=valid_ids_per_query` 仍是硬性要求，无 payload 时抛错。

**A/B 结果（隔离实例，cap=1000）：**

| 场景 | 正确性 | overhead（vs 对应 baseline）|
|---|---|---|
| sort V=100（sparse）| dense==sparse ✓ | +7.3% mean |
| sort V=1000（sparse）| dense==sparse ✓ | +6.2% mean |
| sort V=5000（fallback dense）| ==auto ✓ | **-0.0%** |
| sort V=10000（fallback dense）| ==auto ✓ | **+2.6%** |
| raw V=100/1000（sparse）| dense==sparse ✓ | +12~15% mean |
| raw V=5000/10000（fallback dense）| ==auto ✓ | **-1.7~-3.5%** |

**结论**：cap+fallback 的**回退开销实测 ≈ 0**（fallback 场景 delta 在 -3.5% ~ +2.6%，即噪声范围），印证了"early-break 使 wasted work 被 cap 限制在 1000 量级"的判断。raw-data 也因此能统一走"先 sparse、超 cap 回退"，不必预先知道 V。

**附带发现（独立于本改动）**：`bf_filter_scan_mode=auto`（dense，Cardinal `ScanRangeFilter` 全扫+bitset 判定）与 `dense_per_query`/sparse（按 valid-ID 枚举）在 V=1000 时 topK 不一致（auto 漏了 `a=949` 的 `id=163650`，其 dist 15.54 本应 rank2）。即 `auto` 的 BF scan-filter 与 valid-ID 枚举结果不同，疑似 Cardinal 既有 bug 或扫描模式差异，需单独排查，与 cap+fallback 无关。

### Step 1 实现说明（已完成）

改动文件：

- `common/QueryInfo.h`：`SearchInfo::UseSparseFilterRepresentation()` —— 集中决策 helper，读 `filter_result_representation`（`sparse`），并保留 `bf_filter_scan_mode=="valid_ids_per_query"` 作为 legacy 兼容。
- `query/PlanProto.cpp` `ParseSearchInfo`：`filter_result_representation=sparse` → 翻译成 `bf_filter_scan_mode=valid_ids_per_query`（Cardinal 只认 BF scan mode，保持"表示决策在 Milvus、扫描调度在 Cardinal"的边界）。
- `exec/operator/FilterBitsNode.cpp` / `VectorSearchNode.cpp` / `query/ExecPlanNodeVisitor.cpp`：三处读点统一改用 helper，不再各自硬编码 `valid_ids_per_query` 字符串。

验证（隔离实例，1M×128，`a<1000 and b<500000`）：

- 正确性：`dense_per_query` / `filter_result_representation=sparse` / `valid_ids_per_query` 三者 topK(10) 的 (ID, distance) 完全一致。
- 性能：`filter_result_representation=sparse`（mean 3.49ms / p90 5.76ms）与 `valid_ids_per_query`（3.69 / 5.99ms）等价，无回归。

## 5. Step 1 详细范围

目标：把"过滤结果表示"从"BF 调度模式"里拆出来，成为独立的、可传参控制的开关，并留出决策点。

### 5.1 改动点

1. **新增 search param `filter_result_representation`**（`auto`/`dense`/`sparse`，默认 `auto`=dense），区别于现有 `bf_filter_scan_mode`（后者继续只表示 BF 的扫描方式）。
2. **`FilterBitsNode`**：读 `filter_result_representation`，`sparse` 时尝试走 native valid-ID 生产（现有 `valid_ids_per_query` 路径复用），不再依赖 `bf_filter_scan_mode`。
3. **`VectorSearchNode`**：读 `filter_result_representation`，`sparse` 时用 `FromOwnedValidIdList`。
4. **Cardinal**：`sparse` 表示 → BF 消费（复用 `ValidIdsPerQuery` 的扫描逻辑）。
5. **决策点**：`ShouldUseSparse` 骨架（Step 1 先固定"sparse 参数显式开启"，阈值决策留给 Step 3）。

### 5.2 语义

- `filter_result_representation=dense`：现行为不变。
- `filter_result_representation=sparse`：过滤产出 sparse list，下游必走 BF（阈值保证），等价于现在的 `valid_ids_per_query`，但作为"表示"而非"BF 模式"触发。

### 5.3 验证

- 正确性：`sparse` 与 `dense` 的 topK/ID 一致（复用 E2E closure）。
- 性能：`sparse` 与现有 `valid_ids_per_query` 等价（应无回归）。
