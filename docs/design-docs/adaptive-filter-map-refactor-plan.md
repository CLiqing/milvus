# Adaptive FilterMap 重构需求与实施计划

状态：已确认，进入开发与验证  
日期：2026-09-01

## 1. 背景与目标

当前 Sparse 验证实现通过 `SparseFilterResult{accepted_ids, filtered}` 在 Milvus
过滤执行、MVCC、cache 和向量搜索节点之间传递具体的 list/Dense 表示。这使各节点
必须判断 `IsSparse()` / `IsDense()`，并把具体存储类型、生命周期和 fallback 逻辑
扩散到 Knowhere/Cardinal。

本阶段将过滤结果收敛为统一的 `FilterMap`：调用方继续使用与现有 Dense bitmap
一致的逻辑 bit 语义，底层表示、阈值判断和 Sparse→Dense 升级完全由 FilterMap
内部维护。完整性能结论以 Milvus endpoint E2E 为准，计时包含过滤结果构造、表示
升级、后续 predicate/MVCC、Knowhere/Cardinal handoff 和向量搜索；禁止把 payload
构造移到计时外。

目标：

1. `set(id)`、`reset(id)`、`test(id)` 与当前 `TargetBitmap` 语义一致；FilterMap
   不解释 bit 1 的业务含义。
2. 上层节点不感知 list、Roaring 或 Dense；表示切换不改变调用接口。
3. Sparse 内部保存相对 `default_bit` 的少量 exception IDs；exception 超过阈值时
   一次升级为现有 Dense baseline，不重新扫描已经处理的数据。
4. Cardinal 感知 filter capability。枚举型 FilterMap 标记为 BF-only 时，Cardinal
   必须强制选择 BF；支持随机 membership 时才允许现有 auto BF/IVF/Graph。
5. `sparseEnable=false` 为默认值，并保持当前纯 Dense 行为和性能。
6. 在同一接口下对比 Adaptive list、Adaptive Roaring 和 Dense 的完整 E2E，决定
   最终内部表示；BitmapIndex 已有 Roaring posting 不得先转 Dense 再重建。

## 2. 语义与接口合同

### 2.1 逻辑 bit 合同

FilterMap 对外只提供普通 bitmap 语义：

```cpp
class FilterMap {
 public:
    size_t size() const;
    void set(RowId id);
    void set(RowId id, bool value);
    void reset(RowId id);
    bool test(RowId id) const;
    size_t count() const;
    void ForEachSet(Consumer) const;
    void ForEachUnset(Consumer) const;
    FilterCapability capability() const;
};
```

最终 vector-search filter 延续当前 `1=filtered` / `0=valid` 约定，但 FilterMap 本身
不硬编码该解释。为避免历史 polarity 问题，跨层 metadata 必须携带 universe、逻辑
count、owner 和 capability；不得用具体 storage kind 推导语义。

### 2.2 自适应存储

Sparse backend 的逻辑模型为：

```text
SparseStorage {
    default_bit;
    exception_ids;  // logical bit != default_bit
}
```

因此既可表达“少量 1”，也可表达“少量 0”。阈值依据 exception cardinality，而不是
机械统计 `set()` 调用次数。初版阈值由配置提供，不能写死。

当第 `T+1` 个 exception 出现时：

1. 分配一个长度为 universe 的现有 `TargetBitmap`，全部初始化为 `default_bit`；
2. 回填此前最多 T 个 exception；
3. 应用触发切换的当前操作；
4. 后续操作直接写 Dense。

不得重新执行 predicate，也不得从 0 重新扫描已处理的 row universe。

### 2.3 batch-local Dense 边界

Milvus predicate kernel 内部已有大量 `TargetBitmapView::data()` SIMD/bulk 实现。
本阶段保留 batch-local scratch bitmap；predicate 最终结果、跨 predicate 传递、
MVCC、cache 和 vector-search filter 使用 FilterMap。producer 通过 bulk API 把 batch
结果提交给 FilterMap，FilterMap 内部决定追加 exceptions 或批量写 Dense。

### 2.4 consumer capability

```cpp
enum class FilterCapability {
    RandomMembership,
    EnumerateOnly,
};
```

- `RandomMembership`：`test()` 可高效随机访问，允许 Cardinal 保持 auto selector；
- `EnumerateOnly`：只保证高效枚举 valid/unset IDs，Cardinal 强制 BF；显式 IVF/Graph
  请求也以实际 BF route 执行并记录 route counter。

Knowhere 只传递稳定的 type-erased `FilterMapView`/function table、owner 和 capability；
Cardinal 不感知 list/Roaring/Dense enum。具体 implementation、promotion 和配置均在
Milvus。

## 3. 开发阶段

### Phase A：Dense FilterMap 等价迁移

1. 定义统一 FilterMap owner/view 和 Dense backend；
2. 先用 Dense backend 包装当前 `TargetBitmap`，不改变算法或 route；
3. 迁移最终 FilterBits、MVCC、cache、VectorSearch 及 SearchOnGrowing/Sealed 边界；
4. 对 Dense backend 做逐 bit differential、route、distance attempts 和 topK 闭环；
5. 单独验证 endpoint 不存在稳定性能回退。

### Phase B：Adaptive Sparse→Dense

1. 实现 `default_bit + exception IDs`；
2. 覆盖 `set/reset/test/count`、重复 set、toggle、无序 ID；
3. 实现 batch consume 和 T/T+1 原地 promotion；
4. 配置 `sparseEnable=false` 默认关闭，阈值独立配置；
5. 支持 AND 链、MVCC、delete、TTL、historical read、nullable、growing、sealed、
   multi-segment 和 expression cache。

第一阶段不实现 Sparse OR；需要 OR 或其它不支持的 bulk 运算时，由 FilterMap 内部
`EnsureDense()`，调用方不判断 storage kind。

### Phase C：删除具体 Sparse sidecar

删除或收敛 `SparseFilterResult{accepted_ids, filtered}`、QueryContext 中具体 list/
Roaring 字段以及节点间 `IsSparse()` / `IsDense()` 控制流。内部 debug/metrics 可以
报告 representation，但不得成为业务分支合同。

### Phase D：Knowhere/Cardinal view 与 route

1. 定义稳定的 FilterMap view、owner、universe/count/capability；
2. BF 通过 unset/valid iterator 枚举；Graph/IVF 仅消费 RandomMembership；
3. EnumerateOnly 在 Cardinal dispatcher 中强制 BF；
4. 增加 requested route、actual route、capability 和 distance-attempt counters。

### Phase E：Roaring backend

1. 在相同 FilterMap 接口下增加 Adaptive Roaring；
2. raw/STLSORT producer 按真实执行顺序逐次 set/reset，全部构造进入 endpoint timer；
3. BitmapIndex 按 row offset 递增 `add()` 形成的既有 posting 直接借用/持有，不执行
   Dense→Roaring；
4. 对 Dense、Adaptive list、Adaptive Roaring 做相同 E2E workload 对照。

## 4. 基线保存

开发前必须：

1. 分别提交 Milvus、Cardinal、vecTool 当前任务相关源码/测试/文档 checkpoint；
2. 创建本地 pre-refactor tag；
3. 保存当前 `libmilvus_core.so`、`libcardinalv2.so`、Milvus/QueryNode binary；
4. 保存 SHA256、`ldd`、repo commit、启动配置及 `/proc/<pid>/maps`；
5. 以当前 SO 重跑代表性 case，不能只引用历史结果。

build 目录、perf data 和大型 raw artifacts 不进入 Git。

## 5. 正确性矩阵

- bit contract：default 0/1、set/reset/test/count、空/全量、重复操作；
- promotion：T-1/T/T+1、触发 batch 边界、回填数量、Dense words；
- expression：单 predicate、两个及更长 AND；OR 触发内部 Dense；
- visibility：latest/historical、delete、TTL、nullable；
- topology：sealed/growing、single/multi-segment；
- cache：miss/hit 和 lifetime；
- search：BF-only route、RandomMembership auto route、完整 topK `(ID,distance)`、
  result hash 和 distance attempts。

## 6. E2E 复测

计时边界固定为客户端 endpoint：

```text
predicate evaluation
→ FilterMap construction/set/reset
→ optional Sparse→Dense promotion
→ predicate B / MVCC / cache
→ Knowhere/Cardinal handoff
→ vector search
→ RPC return
```

首轮代表 case：

| Case | 并发 | 目的 |
|---|---:|---|
| Cohere 1M×768D，V=500 | C1/C60 | Sparse/BF 主要收益点 |
| Cohere 1M×768D，V=1K | C1/C60 | 阈值附近 |
| Cohere 1M×768D，V=5K 或 10K | C1/C60 | promotion→Dense 无损 |
| 1M/10M×128D，V=1K | C1 | 放大 filter/enumeration 差异 |
| 一个 natural auto-IVF Dense case | C1/C60 | Dense route 与性能不变 |
| 两个 AND predicate 的 query/search | C1 | 下游继续消费统一 FilterMap |

正式比较使用保存的旧二进制 A 和新二进制 B，按 A→B→B→A 重启顺序；每次核对
实际 SO hash，collection/index 稳定后 warmup 30 秒，再运行固定 query 集合。报告
mean、median、P90、QPS、实际 Cardinal route 和 representation/promotion counters。

## 7. 验收条件

1. `sparseEnable=false`：bit/result/route 完全一致；C1/C60 不出现稳定超过 3% 的
   QPS 回退；
2. 超阈值 promotion：结果一致、不重扫，最差 E2E 回退不超过 5%；
3. Sparse 保留：MVCC/cache/search 结果一致，EnumerateOnly 确认实际走 BF；
4. List/Roaring 最终取舍只依据 construction-inclusive Milvus E2E；
5. mean 改善不能掩盖 P90 重尾，异常需结合 route、promotion、on/off-CPU 定位。

