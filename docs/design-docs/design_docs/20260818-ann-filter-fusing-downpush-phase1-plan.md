# ann filter fusing (downpush) — Phase 1 计划

> 状态：进行中（2026-08-18）
> 分支：`feature/cardinal-varchar-stlsort-eqne-downpush-20260730`

## 1. 背景与目标

把标量谓词下推（downpush，正式特性名 **ann filter fusing**）进向量索引（Cardinal 等）做
filter-fusing。当前实现（本分支）已打通 hint → Cardinal 的基础链路，但对"不支持的操作"是
**直接抛错**（`ThrowInfo` / `AssertInfo`），且 entity TTL、nullable 向量等场景尚不支持。

**Phase 1 目标**：打通 downpush hint 的模板链路，使其对任何请求都**安全**——支持的谓词/索引
走 fusing，不支持的**自动回退到 baseline**（普通 filter + 普通搜索），绝不破坏正确性。
自动决策 / 白名单放量放到 Phase 2。

## 2. 范围

### In scope（Phase 1）
1. 统一 gate 前置：把"能否 downpush"的判定集中到 `FilterBitsNode` 决策点。
2. 不支持 → **注释 + 静默 fallback + 日志 + metric**（不再 throw）。
3. entity TTL 直接支持（拆成普通 bitset，仅 fuse 用户谓词）。
4. nullable 向量直接支持（复用 p2l gather 把标量值源重排到物理空间）。
5. 移除搜索期的 throw（`SearchOnSealed` / `VectorSearchNode`），改为前置 gate + 防御断言。

### Out of scope（Phase 2 / 后续）
- 自动决策 / 白名单（autoindex 式 `tuning.*` 配置或 hook 注入）。
- 正式改名 ann filter fusing（wire 值 + 标识符 + 注释 + 设计文档）。
- 依赖原始 varchar 值源的 `LIKE`。当 sealed + STL_SORT 布局只暴露字典
  ID 而不暴露 raw-string chunk view 时，Phase 1 必须回退；等值、不等值和
  term 仍可使用字典 ID 下推。
- Cardinal 之外的其它 backend。

## 3. 命名约定

- 特性名 **ann filter fusing**；机制名 **downpush**。
- Phase 1 内 wire 值、标识符、注释统一沿用 `downpush`，最终 PR 一次性改名，避免现在改
  破坏现有 QTP 实验/脚本（collection 名 `downpush_v1_*`、任务名等）。

## 4. 已确认设计决策

| 决策 | 结论 |
|---|---|
| 能力方法命名 | `SegmentInterface::SupportsDownpush(FieldId)`，后端无关（不用 `SupportsCardinalDownpush`）|
| 值源构建位置 | 下沉到 `SearchOnSealed`（选项 A），拿到 offset_mapping 后再构建 |
| nullable 向量 | 复用 p2l gather 把标量值源重排到物理空间，直接支持（不走 fallback）|
| index-type 判定 | 前置 gate 用 `SupportsDownpush`，判定 index type ∈ {CARDINAL_TIERED, HNSW, HNSW_SQ, HNSW_PQ, HNSW_PRQ} |
| entity TTL | 拆成普通逻辑 bitset，仅 fuse 用户谓词 |
| 命名 | downpush 保留到最终 PR |

## 5. 关键实现点

### 5.1 统一 gate（FilterBitsNode）
把 `cardinal_downpush_execution` 分支的判定收敛为一个 gate，覆盖：
- sealed segment；
- 谓词形状可编译（`TryCompileCardinalDownpushPredicate`）；
- 数据型 / field 存在 / nullable-scalar 规则（`try_field`）；
- entity TTL（不再 throw，见 5.3）；
- element-level 向量搜索（`search_info.element_level()`）→ 不支持，fallback；
- index-type（`SupportsDownpush`）→ 不支持，fallback；
- 采样估算过滤比（`EstimateFilteredOutCountBySample`）→ 失败或 ratio ≥ 0.90，fallback。

任一不满足 → `cardinal_downpush_enabled_ = false`，走正常 ExprSet 求值（baseline），并记
注释 + 日志 + metric。

### 5.2 不支持 → fallback（替换现有 throw）
现有 throw 点 → 注释 + 静默 fallback：

| 位置 | 现状 | 改后 |
|---|---|---|
| `FilterBitsNode.cpp:611` TTL | throw | 直接支持（5.3）|
| `FilterBitsNode.cpp:617` 谓词形状 | throw | 注释 + fallback |
| `FilterBitsNode.cpp:629` 估算失败 | throw | 注释 + fallback |
| `VectorSearchNode.cpp:714` element-level | throw | gate 前置拦截（不可达，降为防御）|
| `VectorSearchNode.cpp:728` 构建 ctx 失败 | throw | 防御 error（内部不一致）|
| `SearchOnSealed.cpp:118` index 类型 | assert | gate 前置拦截（不可达，降为防御）|
| `SearchOnSealed.cpp:133` offset mapping | throw | 直接支持（5.4）|

### 5.3 entity TTL 直接支持
TTL 谓词 `(ttl IS NULL) OR (ttl > physical_us)` 由 `CompileExpressions` 自动 AND 进用户过滤。
downpush 路径需把 TTL **拆出来单独求值**（普通逻辑 bitset），仅把用户谓词 fuse 进 Cardinal。
最终 knowhere 侧同时吃「常规 bitset（TTL + MVCC delete）」+「extra scalar predicate filter（用户谓词）」。

### 5.4 nullable 向量（offset mapping）直接支持
baseline 对 nullable 的处理：filter 在逻辑空间求值，`SealedOffsetMapping::TransformBitset`
用 `p2l` 把 bitset gather 到物理空间（`result[physical] = bitset[p2l[physical]]`），索引在物理空间搜索。

复用：对 downpush 的标量值源做**同样的 p2l gather**——`physical_scalar[physical] = logical_scalar[p2l[physical]]`，
得到物理空间值源再交给 Cardinal。这是 `TransformBitset` 的标量版，复用同一份 `p2l` 数据。
代价 = 每查询一次 O(valid_count) 的 gather（int/float 直接重排；varchar 重建 offset 表、共享 byte base）。

#### 现状（值源构建在 VectorSearchNode）
`VectorSearchNode::GetOutput`（709-744）里：读 predicate → `BuildCardinalDownpushSearchContext`
（从 segment 取 int/float/string chunk 或 row 值源）→ `FillKnowhereDownpushValueSource` +
`FillKnowhereDownpushArgs` → `search_view.set_extra_scalar_int64_predicate_filter`。此时 offset
mapping 还不知道（要 pin 索引后才知）。

#### 改造（Option A：值源下沉到 SearchOnSealed）
1. `VectorSearchNode::GetOutput` 只保留 op/args/value_type 的填充（`ToKnowherePredicateOp`、
   `FillKnowhereDownpushArgs`），**不再**在这里 `BuildCardinalDownpushSearchContext` + fill 值源；
   值源数据留到 SearchOnSealed 填。
2. `SearchOnSealedIndex` 拿到 `vec_index` 与 `offset_mapping` 后，若
   `has_extra_scalar_int64_predicate_filter()`：
   - `offset_mapping.IsEnabled() == false` → 传原始逻辑值源（现状）；
   - `IsEnabled() == true` → 按 `p2l` 把三类值源 gather 到物理空间：
     - int64/float：`physical[i] = logical[p2l[i]]`，产出单一物理数组；
     - string：按 `p2l` 重建 `chunk_value_offsets`（前缀和），共享 `chunk_bases` 的 byte 数据；
     - string dictionary（stl_sort dictionary id）：per-row 的 `row_dictionary_ids` 按 `p2l` gather。
3. 生命周期：gather 产生的临时数组（int64/float 物理数组、string 物理 offset 表）必须在
   `vec_index->Query(...)` 期间存活 → 用 `std::vector` 在 `SearchOnSealedIndex` 栈上持有，
   并在调用前填好指针。

#### 需要解决的签名/接口问题
- `SearchOnSealedIndex` 当前签名只有 `schema` + `record` + `search_info` + `op_context`，
  **没有 segment**；但 `BuildCardinalDownpushSearchContext` 需要 `SegmentInternalInterface*`。
  需把 segment（或 downpush 构建所需的最小接口）传进 `SearchOnSealedIndex`，或在
  `ChunkedSegmentSealedImpl::vector_search` 里先构建值源再传。
- `set_extra_scalar_int64_predicate_filter` 目前对 `search_view`（mutable）调用；SearchOnSealed
  里的 `bitset` 是 `const BitsetView&`，需在局部 `search_bitset` 上重建/补填 filter。

### 5.5 可观测性（fallback metric）
在 C++ monitor 里新增一个带 reason label 的 counter：

- `Monitor.h`：`DECLARE_PROMETHEUS_COUNTER_FAMILY(internal_core_downpush_fallback_count);`
- `Monitor.cpp`：`DEFINE_PROMETHEUS_COUNTER_FAMILY(internal_core_downpush_fallback_count, {{"reason", "fallback reason"}});`
- `FilterBitsNode::TryEnableCardinalDownpush` 各 fallback 分支：
  `internal_core_downpush_fallback_count.Add({{"reason", "..."}});`

reason 取值：`unsupported_index` / `unsupported_predicate` / `ratio_threshold` / `estimate_failed` / `element_level`。

## 6. 涉及文件

- `internal/core/src/segcore/SegmentInterface.h` — 加 `SupportsDownpush` 虚方法
- `internal/core/src/segcore/ChunkedSegmentSealedImpl.{h,cpp}` — 实现 `SupportsDownpush`
- `internal/core/src/exec/operator/FilterBitsNode.{h,cpp}` — gate + fallback + TTL 拆分
- `internal/core/src/exec/operator/VectorSearchNode.cpp` — 移除 element-level throw / 值源构建下沉
- `internal/core/src/query/SearchOnSealed.cpp` — 移除 throw、值源构建 + p2l gather
- `internal/core/src/exec/QueryContext.h` — 如需要存 TTL bitset / 值源
- 测试：`internal/core/unittest/test_*`（FilterBitsNode / SearchOnSealed 相关）

## 7. 测试

### C++ 单测（`internal/core/unittest/`，`make test-cpp`）
- gate 各 fallback 分支：
  - unsupported index type（DISKANN/IVF → `SupportsDownpush=false`）；
  - 谓词形状不支持（如 JSON 谓词 / 复合 AND-OR → `TryCompile...` nullopt）；
  - ratio ≥ 0.90（构造低选择性谓词）；
  - element-level 搜索；
  - estimate 失败（构造 `Eval` 产出非 bitmap 的边界表达式）。
- entity TTL + downpush：结果与 baseline（普通 filter）逐行一致（含 null TTL / 已过期 / 未过期）。
- nullable 向量 + downpush：物理空间 gather 后结果与 baseline 一致（重点验证 p2l 重排后谓词作用在正确行）。
- `SupportsDownpush`：无索引（brute force）/ binlog 临时索引 / DISKANN / HNSW 家族 各返回预期。
- TTL bitset flip 语义：`1` = exclude 约定，与 MvccNode delete mask 合并正确。

### Go 侧（如涉及，`-tags dynamic,test -gcflags="all=-N -l"`）
- 暂无直接 Go 改动（hint 透传链路已在现状支持），回归 `make test-querycoord`/`test-proxy` 即可。

### 验证方式
- 构建：`./scripts/install_deps.sh` → `make`（或 `make SKIP_3RDPARTY=1`）。
- 仅 C++ 单测：`make test-cpp`。
- 目标单测过滤：`./bin/... --gtest_filter=*Downpush*`。

## 8. 微决策默认值（无异议即执行）

- metric 名 `QueryNodeDownpushFallbackCount` + reason label。
- gate 通过后值源构建仍失败 → 防御性 error（内部不一致，非 fallback）。
- varchar 值源 gather = 重建 offset 表（共享 byte base），三类值源统一 gather。
- element-level → fallback（Phase 1 不支持）。
- index-type 判定源：用 `col_index_meta_`（FieldIndexMeta::GetIndexType，具体类型），测试中与
  `vec_index->GetIndexType()` 交叉验证。

## 9. 后续（Phase 2 / 最终 PR）

- 自动决策 / 白名单（autoindex 式 tuning 配置或 hook 注入 hint）。
- 正式改名 ann filter fusing。
- 正式设计文档 + issue 链接（PR 规范要求）。
