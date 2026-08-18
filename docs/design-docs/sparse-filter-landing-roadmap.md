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
| Step 2 | producer 覆盖补齐：bitmap 等值（已通）、sort range（修 value-order）、raw-data（已通）| 待办 |
| Step 3 | 决策策略 A/B：阈值 / cost model，跑完整收益矩阵 | 待办 |
| Step 4 | （可选）Graph/IVF 消费端（Bloom+Flat）| 待办 |

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
