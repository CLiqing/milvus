# Sparse 过滤表示落地路线图

日期：2026-08-18
关联：`sparse-p90-tail-investigation.md`（P90 定位实验）、`sparse-compound-filter-milvus-e2e-plan.md`（E2E 计划）。

## 0. 第一阶段正式方案（2026-08-27，覆盖后文历史原型）

本阶段把 Sparse 做成**显式开启的自适应过滤结果表示**，目标是验证机制与
正确性，不在这一阶段确定最终自动选择策略。后文中“开启后每个 predicate
必须硬输出 Sparse”“寻找一个可产出 Sparse 的 child 再重排 AND”以及
“Sparse 与 BF 路由绑定”等描述均为历史原型，不再作为实现依据。

### 0.1 开关与参数语义

建议全局配置：

```text
queryNode.segcore.enableSparseFilterResult = false
queryNode.segcore.sparseResultMaxCardinality = 6000
queryNode.segcore.sparseResultMinSegmentRows = 50000
queryNode.segcore.sparseResultMaxRatio = 0.006
```

请求级实验参数：

```text
filter_result_representation = dense | adaptive
sparse_result_max_cardinality = <positive integer>  # 实验 override
```

现有 `sparse` 值在迁移期作为 `adaptive` 的兼容别名。正式行为如下：

| 全局开关 | 请求参数 | 行为 |
|---|---|---|
| `false` | 缺省或 `dense` | 原始 Dense baseline；不创建临时 list、不计数、不承担切换成本 |
| `false` | `adaptive` / `sparse` | 明确返回 feature-disabled 错误，不能静默改变基线 |
| `true` | 缺省或 `dense` | 原始 Dense baseline，支持同实例请求级 ABBA |
| `true` | `adaptive` / `sparse` | 启用本节的 Dense/Sparse 自适应输出 |

当前 selector 候选由绝对上限、最小 segment 行数和比例上限共同约束；请求级
override 只允许收紧绝对上限，不能绕过全局 `min-N/ratio`。表示参数控制过滤结果的物理表示；当前第一阶段约定 Memory/Tiered
收到 Sparse payload 后直接进入 BF，Dense 仍保留 Cardinal 原有 auto selector。

### 0.2 统一结果与 AND 执行模型

第一阶段使用统一中间结果：

```cpp
FilterResult {
    representation: Dense | SparseIncluded;
    universe;
    dense bitmap or unordered unique sparse IDs;
}
```

`SparseIncluded` 的 canonical 契约只要求 ID 精确、唯一、位于 `[0, universe)`
并且生命周期安全；producer order 是合法顺序。不得为了某个下游 consumer 在
producer 中强制排序。BF 可直接枚举；需要 locality、chunk 或 bucket 组织的
consumer 在自己的边界按需做 `O(V)` grouping。

每个 predicate 恰好执行一次，只接收上一步的 `FilterResult` 并施加自己的
判定；不再遍历 child 寻找“最合适 producer”，也不依赖 parser 偶然产生的
物理顺序：

```text
Dense/Universe input -> predicate -> AdaptiveResult
Sparse input         -> 只检查候选 IDs -> Sparse output
```

AND 只会缩小候选集，因此 Sparse 输入经下一 predicate 后仍为 Sparse。对
Dense/Universe 输入，所有进入第一阶段支持面的 predicate 都必须把 batch-local
accepted mask 或 ID stream 直接交给统一 `AdaptiveFilterSink`；不得先构造完整
N-bit Dense 再从中枚举 Sparse。Bitmap/STL_SORT 等能预知 cardinality 的 producer
可以把该信息交给 sink 提前选择表示，但不再拥有另一套排序、转换或阈值语义。
V 未知的 producer 采用**单趟 lazy switch**：

```text
扫描并累计最多 T 个 accepted IDs
发现第 T+1 个 accepted ID：
    只分配一次 Dense 结果
    回填此前保留的、最多 T 个 accepted-ID 前缀
    当前触发 batch 整体写入 Dense，随后余段继续直接写 Dense
```

触发 batch 中为判断超限而临时追加的 ID 会被丢弃，由该 batch 的一次 bulk write
统一落入 Dense，避免同一批结果重复写入。总成本为 `O(N)` 单次 predicate 执行，
外加至多 `O(T)` 暂存/回填和一次 Dense 分配；已经执行的前缀不再求值，剩余输入
从当前位置继续。因此禁止 `return nullptr` 后重新执行一次完整 Dense evaluator，
也禁止在 FilterBits 边界对完整 Dense 做事后 Sparse 转换。Sparse 直接保留 producer
order，不增加排序；阈值边界按 `T-1/T/T+1` 固化单测。

MVCC、delete、TTL、nullable、growing 与 multi-segment 仍属于过滤正确性的一
部分；自适应表示不能绕过这些 mask。Expression cache 也不能靠禁用规避，cache
key/entry 至少需要区分 `dense/adaptive`、阈值和表示契约版本，或缓存逻辑结果后
按请求 materialize。

### 0.3 第一阶段范围：仅 AND

第一阶段只开发和验收 AND。OR 继续走现有 Dense 正确性路径，不因开启
Adaptive 而尝试 Sparse 快捷路径；这不是 OR 不重要，而是其集合语义和测试量
需要独立阶段处理。

OR 后续设计必须特别注意：

- 不能把左 branch 的 Sparse IDs 当作右 branch 的 candidate；两个 branch 必须
  在同一输入域分别求值后做 union；
- `Sparse ∪ Sparse` 超过阈值时需要切换为 Dense，`Sparse ∪ Dense` 与
  `Dense ∪ Dense` 需要正确的 Dense merge；
- nullable、NOT 与 OR 组合遵循 SQL 三值逻辑，不能用简单补集替代；
- 后续需单独覆盖嵌套 AND/OR、NULL、阈值切换、cache 和多 segment 正确性。

### 0.4 诊断与验收

功能测试除结果等价外，还必须闭合以下计数，避免把重复求值或隐藏 fallback
误判成算法收益：

```text
predicate evaluation count
rows/chunks scanned per predicate
full Dense allocation on Sparse-success path
Sparse IDs appended and Dense words written
Sparse -> Dense switch count
prefix IDs backfilled
final representation and cardinality
search route (BF/Graph/IVF)
```

单测至少覆盖：feature disabled 的 untouched Dense；`T-1/T/T+1`；单趟回填；
A→B 与 B→A；2/4/8 个 AND child 每个只执行一次；OR Dense fallback；以及
已有 visibility/nullable/growing 回归。构建和定向单测通过后，再在同实例、固定
collection/query/route 下按 ABBA 比较 Dense/Adaptive。

### 0.5 本轮开发进度（更新于 2026-08-27）

已落地的基础链路如下。本轮 unordered payload、统一 sink 和 capability preflight
重构已经按 0.6 的门禁完成重新编译、回归与 Milvus E2E；最终证据见 13.12，不能用
下方重构前的旧二进制记录替代：

- 新增全局配置 `enableSparseFilterResult=false` 与
  `sparseResultMaxCardinality=1000`，支持动态刷新；默认关闭；
- 请求参数接受 `adaptive`，并保留 `sparse` 兼容别名；支持请求级
  `sparse_result_max_cardinality`；显式请求 Adaptive 但全局未开启会报错；
- FilterBits 最终表示按阈值选择：不超过阈值交付 Sparse，超过阈值保留 Dense；
  native Bitmap/STL_SORT/raw producer 也受可调全局阈值与请求阈值约束；
- OR 在第一阶段被显式识别并完整保留原 Dense evaluator/cache 路径；
- expression cache 不再因 Adaptive 被禁用：Dense、Sparse 与 threshold-Dense
  使用表示和阈值隔离的 entry，cache hit 不通过完整 Dense 扫描构造 Sparse；
- 移除了 AND 的“遍历 child 直到找到 Sparse producer”运行时搜索。native AND
  只从执行顺序中的第一个 predicate 开始；不能形成 native fast chain 时，由
  普通 batched evaluator 执行每个 predicate 一次，并把 batch-local 结果直接交给
  `AdaptiveFilterSink`，不再先形成完整 Dense 再转 Sparse；
- 增加最终 Sparse、阈值 Dense、第一阶段 OR Dense 三类 counter，供 E2E
  闭合实际表示选择。
- 引入临时 `Dense|Sparse` predicate 中间结果；AND 的下一 predicate 可以直接
  消费 Sparse candidate，也可以在 Dense filtered bitmap 上继续合并，不需要
  为寻找 Sparse producer 改写 predicate 顺序；
- raw INT64 range 与普通 batched producer 已接入单趟 lazy switch：先累计最多 T 个
  accepted IDs，第 T+1 个命中时只分配一次 Dense、回填最多 T 个前缀，并从当前
  batch/当前位置继续写 Dense，不重新扫描前半段；
- 新增契约测试覆盖 Dense 中间结果合并且下一 predicate 只执行一次、A→B/B→A
  结果等价，以及 OR 不进入 Sparse 执行。第一阶段不会继续扩展 OR；其 union、
  阈值切换和 SQL 三值逻辑仅保留为下一阶段注意事项。

本轮重构前的验证基线：

- `milvus_core`、完整 `all_tests` 与 Milvus Go 可执行文件均已构建成功；Go 配置
  定向测试和 `internal/util/initcore` 编译检查通过；
- 最新二进制执行 16 项定向 C++ 测试全部通过：新增 Dense 中间结果、A→B/B→A、
  2/4/8 child 各 predicate 仅执行一次、OR fallback 契约，以及既有 Sparse AND、
  MVCC、delete、TTL、history 和 STL_SORT 回归；
- 新建 100K×128D Cardinal collection 的参数级 Milvus E2E 通过：复合条件最终
  50 valid rows，Dense/Adaptive 的 10 条固定 query topK/distance 完全一致；
  ABBA 2 windows、每 slot 10 query 的 smoke 中，Dense/Adaptive mean 为
  2.842/2.762 ms（该短测仅作链路 smoke，不作为性能结论）；
- 阈值 1000 的精确边界已用最终 cardinality 999/1000/1001 验证：前两者计入
  Sparse，1001 计入 threshold-Dense，三点 Dense/Adaptive topK 均一致；同一逻辑
  filter 交替使用阈值 2000/1000 时能分别 materialize Sparse/Dense；
- OR Adaptive 请求保持 Dense，E2E 与 Dense baseline 结果一致，并由
  `dense_or_phase1` counter 闭合；
- 新前缀 visibility E2E 完整通过 sealed multi-segment、nullable、delete、TTL
  expiry 与 growing；历史 snapshot 仍受当前 pymilvus API 能力限制，由 C++ MVCC
  定向测试闭合。

独立发现：同步上游后的 Cardinal auto-IVF 对部分约 1000 valid-row 的请求在
`partial_read.h:72` 触发 SIGSEGV；旧 collection 和本轮新建 collection 均可
复现。显式 `bf_filter_scan_mode=dense_per_query` 下 Dense/Adaptive 阈值测试稳定
通过，因此该崩溃不属于 Sparse 表示链路，但在恢复 auto route 性能测试前必须
单独定位。

### 0.6 非目标实现清理与 switch-point 执行计划（2026-08-27）

#### 0.6.1 目标实现与清理边界

当前目标只有两种 canonical 输出：Dense filtered bitmap 与 unordered、unique 的
SparseIncluded IDs。Roaring 可以继续作为 BitmapIndex 内部 posting 或 expression-cache
压缩格式，但不再是 Milvus 过滤执行器向下游传播的第三种 payload。

| 分类 | 处理 | 原因 |
|---|---|---|
| STL_SORT accepted IDs 的全局排序、ascending 断言、依赖有序输入的 predicate consumer | 删除；consumer 在自身边界按需做 `O(V)` chunk grouping | canonical Sparse 不承诺顺序，producer 不承担消费端 locality 成本 |
| `FilterBitsNode::ConvertFilteredBitsetToSparseIds` 及 generic/growing 的完整 Dense→Sparse 调用 | 删除，由统一 sink 直接消费 batch-local accepted mask/ID stream | 避免完整 Dense materialization 后再扫描 N-bit universe |
| `ScalarIndex::TryGetRoaringRange`、STL_SORT `Build/TryGetRoaringRange`、Hybrid 转发、`BitmapIndex::TryGetRoaringEqual`、Milvus `BitsetView::FromOwnedRoaring*` | 删除其跨层/公开 API 及仅验证这些 API 的测试和 disabled benchmark | 最终 handoff 统一为 ID list；STL_SORT 构建 Roaring 还会引入非目标排序/编码成本 |
| Cardinal Sparse→Bloom+Flat membership、Sparse-IVF bucket directory/grouping、Graph/IVF replay 实验实现 | 删除生产分支及只服务该分支的类型、测试和构建入口；保留历史结果文档 | 第一阶段 Sparse 统一 direct-BF，避免不可达 adapter、额外 metadata 和错误能力声明继续扩大 |
| 未使用的固定 `DEFAULT_SPARSE_LIST_CAP` | 删除 | 阈值来自全局配置或请求级实验 override，不能存在第二套写死语义 |
| P90 定位期的逐请求 wall/CPU timer、`getrusage` 和慢请求日志 | 从性能版本移除或默认关闭 | 诊断完成后不应污染热路径；保留逻辑 access counter |
| BitmapIndex 内部 Roaring posting/低 distinct sidecar，以及从 posting 解码 `V` 个 IDs | **保留** | 这是无需 Dense 扫描的 native producer；其 build/load 内存成本需单独计量 |
| expression-cache 内部 Roaring 压缩 | **保留且与本方案分开** | 它是 cache 存储编码，不是 canonical Sparse filter payload |
| Knowhere/Cardinal 既有 filtered-Roaring consumer | **保留为独立能力**；只清理本项目新增的 Milvus valid-Roaring producer/handoff 和第一阶段不用的 adapter | 不以 Adaptive Sparse 收口为由回退 PR #1732 或其他既有 Roaring 功能 |
| `Try/CanGetValidId*`、STL bounds、`AdaptiveFilterSink`、visibility/cache、legacy Dense consumer fallback | **保留** | 构成 native producer、统一表示选择和功能兼容闭环 |

清理后的执行约束是：所有表达式只执行一次；所有未知 V 的输出都进入同一个
`AdaptiveFilterSink`；Sparse success 不分配完整 Dense；T+1 fallback 只分配一次
Dense、回填最多 T 个前缀并继续余段；任何路径都不做 Dense→Sparse、不重跑
`O(N)` predicate、不为 Sparse 排序。

本轮按以下顺序执行；前一阶段未通过正确性和路径计数闭环时，不进入下一阶段：

| 阶段 | 内容 | 验收条件 | 状态 |
|---|---|---|---|
| A | unordered+unique payload、移除排序、consumer 按需 chunk grouping | 无序/跨 chunk 的 STL_SORT、raw、Bitmap 结果与 Dense 一致；生产路径无 Sparse 排序 | 完成；见 13.12 |
| B | 统一 `AdaptiveFilterSink`，删除完整 Dense→Sparse，接入 raw/generic/growing/AND | Sparse success 的 full-Dense allocation 为 0；T+1 只有一次 Dense 分配；每个 predicate 只执行一次 | 完成；见 13.12 |
| C | 非目标 valid-Roaring API、Cardinal Sparse-IVF/Graph/Bloom+Flat adapter 与诊断代码清理；`T-1/T/T+1`、visibility/cache 回归 | 生产 Sparse consumer 只剩 direct-BF；无废弃 API 调用；fallback 无二次 N 扫描；nullable、MVCC、delete、TTL、growing、multi-segment 等价 | 完成；见 13.12 |
| D | 三层 switch-point discovery/validation | 单项成本、真实 producer 和 endpoint 证据闭合，不能由一层结果替代另一层 | 完成首轮；Sparse B batch 修复后需刷新 endpoint 曲线，见 13.12.1 |
| E | 收敛候选 policy 与 hold-out | 已测准入点 QPS 劣化不超过 5%；目标 workload 提升 20%--30%；正确性、表示、route 一致 | `0.4% && V<=4000` 降级为修复前候选；重测与 hold-out 后再收敛 |

#### 0.6.2 输出成本模型

Dense 与 Sparse 共享 predicate 本身的读取、比较和 nullable 判定，但输出成本并不
完全相同，也不能把排序或 Dense→Sparse 计入目标 Sparse 成本：

```text
Dense   = C_predicate
        + C_dense_init/write(N / word)

Sparse  = C_predicate
        + C_mask_scan(N / word)       # producer 已直接给 ID stream 时可省
        + C_append(V)
        + C_handoff/visibility(V)

T+1 fallback
        = C_predicate                  # 只执行一次
        + C_probe_and_append(<= T+1)
        + C_dense_alloc_once(N / word)
        + C_prefix_backfill(<= T)
        + C_trigger_batch_and_rest_write
```

其中 raw/SIMD 与 generic evaluator 通常先产生 batch-local mask，sink 对该小批结果做
word scan；这不是完整 N-bit Dense payload。STL_SORT range 可直接迭代 bounds 内的 V
个 row offsets；Bitmap posting 可直接解码 V 个 IDs，二者都不排序。QueryContext 的
range/universe 校验、visibility compact、vector ownership 与 cache put 若仍在生产热
路径，也必须作为 `O(V)` 成本显式记录，不能归入“免费 handoff”。

Dense baseline 可能使用 all-at-once SIMD 和原位 flip，而 Adaptive generic 路径可能按
batch 执行。两侧 eval rows 相同并不代表调用/batch 固定成本相同；该差异必须独立报告，
不能错误归因于 prefix backfill 或 Sparse consumer。

性能采集前增加以下硬门禁，避免把过渡实现或正确性缺口计入 switch-point：

- Sparse MVCC 不得以 `get_deleted_count()==0` 作为跳过 delete mask 的条件；
  并发 delete 必须与 Dense 路径保持同一可见性语义；
- native Adaptive 链只能在无副作用 capability preflight 成功后执行；不允许先执行
  部分 predicate、发现后续不支持，再从头运行 Dense/generic evaluator；
- 默认关闭时所有兼容参数也必须受 feature gate 约束；Dense 返回路径必须清除旧的
  Sparse payload；
- 正式 A/B 前移除或默认关闭 Sparse-only 的逐请求时钟、`getrusage` 与慢请求日志，
  保留不改变热路径结构的逻辑 access counters；
- Dense 与 Adaptive 必须记录并闭合相同的 predicate eval rows/batches。若 Adaptive
  fallback 因失去 all-at-once 优化而多出 batch 开销，需要作为独立成本项报告，不能
  归入 `O(T)` prefix backfill。

#### 0.6.3 Switch policy 候选

第一轮不再只用固定绝对 `V`，而使用 per-segment `N` 与 accepted cardinality `V`：

```text
N >= N_min
&& V <= min(floor(ratio_cap * N), V_abs_safe)
```

其中 `V/N` 是表示选择主轴；当前 Sparse 进入 direct BF，因此在更大 segment、
更高维度或 NQ 下仍保留绝对 `V`/BF work safety cap。首轮实验使用的起点是：

```text
N_min      = 50K
ratio_cap  = 0.1%
V_abs_safe = 1000
```

完成 50K--1M、768D、C1/C60 的 Milvus E2E discovery 与 hold-out 后，下一轮候选
收敛为：

```text
N_min      = 50K
ratio_cap  = 0.4%
V_abs_safe = 4000
```

这是**已测范围内的实验候选，不是产品默认值或理论 switch point**。`0.4%` 的含义是
在当前矩阵中守住“准入点 QPS 劣化不超过 5%”的下界；稳定进入 20% 以上收益区间的
代表点仍是 `0.1%`。`N_min=50K` 只是当前验证下界，50K 以下尚未测试；超过 1M 的
segment、不同维度/NQ/并发以及其他硬件仍需独立 hold-out。由于相同 `V/N` 下的 absolute
V 和并发会改变 BF 饱和点，当前不能去掉 `V_abs_safe=4000` 的保护。

该 policy 尚未写入生产 selector。discovery 阶段使用请求级 cap override 模拟每个
候选阈值，不能把早期固定 `T=1000` 的行为当作 ratio policy 结论。除了规模 sweep，
还必须单独覆盖 `N_min-1/N_min/N_min+1`，并在每个 N 上覆盖有效阈值
`T_eff-1/T_eff/T_eff+1`，证明最小 N gate 和表示切换边界都没有 off-by-one。

#### 0.6.4 三层 switch-point 实验

**Layer 1：Adaptive sink 单元成本。** 使用预先生成的 batch-local truth mask，只比较
Dense sink 与 Adaptive sink 的表示构造、T+1 switch 和 backfill。该层固定并排除
predicate 生成，回答的是 `N/word` 写入、`V` append 与 `T` 回填的单位成本；它不是
raw/STL_SORT/Bitmap producer，也不能单独决定产品 switch point。

**Layer 2：真实 producer controlled A/B。** 分别执行生产实现的 raw/SIMD、STL_SORT
range、Bitmap posting 和至少一条 generic expression。每个 paired case 固定数据、
predicate、输入访问量和 batch width，只切换 Dense/Adaptive sink，记录 cycles、
instructions、task-clock、allocation、mask words、IDs appended、Dense words、switch
和 backfill。先用足够大的实验 cap 让每个 V 都能产出 Sparse，得到完整成本曲线；再按
候选 effective threshold 做 `T-1/T/T+1` fallback 验证。固定 `T=1000` 的微基准不能
用来推断 10M 上大于 0.01% 的 Sparse 成本，也不能验证 50K 上 ratio cap=50 的策略。

**Layer 3：Milvus endpoint validation。**

- Search discovery：Cohere 1M x 768D、C1，扫描 `V/N`，按 ABBA 对比
  Dense/Adaptive，并闭合最终 representation、Cardinal route、distance/candidate
  work 与 TopK；
- Search hold-out：保持 768D，以 C60 验证并发边界，并在 50K/100K/250K/1M
  单 segment 上检查 N-scale；Memory/Tiered、NQ>1、multi-segment 与更大 N 留作下一轮；
- Query 独立验收：`count(*)`、limit retrieve、宽字段 projection 与两层 AND。
  Query consumer 完成前，不用 Search direct-BF 数据外推 Query 收益或阈值。

为避免 producer、backend、并发与数据规模形成无解释力的大笛卡尔积，矩阵按三阶段执行：

1. **D1 sink discovery**：完整扫描 `N=50K/100K/250K/1M/10M` ×
   `V/N=0.01%/0.05%/0.1%/0.2%/0.5%/1%` × random/clustered；另补
   `N_min±1` 与 `T_eff±1`。这里只比较相同 batch-local mask 下的输出 sink。
2. **D2 producer validation**：raw/SIMD、STL_SORT、Bitmap posting、generic expression
   先测 ratio/绝对 V 锚点，再只在 D1 发现的 crossover 邻域加密；不重复整个 D1
   笛卡尔积。
3. **D3 endpoint hold-out**：先用 128D 做 route/correctness discovery；最终采用
   Cohere 1M×768D、C1/C60 验证 ratio 边界和 `T_eff±1`。Memory/Tiered、NQ>1、
   multi-segment 与 Query 分成独立 hold-out，不混入同一主表。

cache off/on、all-visible/visibility-on 分层执行，不聚合为一个区间。性能准入使用
paired-window QPS：所有计划准入点的 paired 95% CI 下界不得低于 -5%，并同时报告
同配置 A/A 噪声；目标 workload 的 +20%～30% 是期望收益，不是要求每个边界点都达到
的硬门槛。

每次 A/B 只允许一个独立变量。必须固定并记录：source/binary SHA、编译选项、dataset
和 seed、scalar/vector index、segment 数与每 segment N、predicate 与 V、query 集和
顺序、NQ、topK、metric、线程/CPU affinity、NUMA placement、CPU frequency policy、
并发度、batch width、cache state、预热、运行时长及搜索 route。每点同时保存
predicate rows/chunks、sink access counters、最终表示、原始请求样本和 A/A 噪声带。
cycles、instructions 与 task-clock 分开采集，避免 multiplex 后把缩放值当作最终单位
成本。endpoint QPS 只做独立验证，不反向替代未测的 producer/consumer 单项成本。

#### 0.6.5 Query 隔离与重型 predicate 补充实验（2026-08-28）

本轮 Pure Query **不是独立判断 Sparse 在 Query 产品场景是否值得采用**，而是移除
vector search、Cardinal route 和 distance work，减少变量干扰，隔离观察：predicate
执行、Dense/Sparse 输出、MVCC/visibility handoff，以及最终结果消费。正式计时前须先
让 Query consumer 原生消费 MVCC compact 后的 Sparse payload；不得把占位 bitmap
交给现有 `count/find_first_n` 路径后计时。

第一组使用 Cohere 1M collection 的随机分布 INT64 列、单 sealed segment、cache-off：

| Case | Predicate | 最终 V | 观察目标 |
|---|---|---:|---|
| Single | `a < 1000` | 约 1,000 | 去掉 search 后比较单 predicate 的表示构造与 handoff |
| AND-pass | `a < 1000 AND b < N` | 约 1,000 | 观察 Sparse input 经第二个 predicate 的固定成本 |
| AND-reduce | `a < 4000 AND b < 25%` | 约 1,000 | 观察 A→Sparse→B→Sparse 的候选缩减 |

主 endpoint 为 `count(*)`，避免 projection 和大结果传输污染隔离实验；limit/ID retrieve
仅作 offset、顺序、PK 去重与结果一致性闭环。先测 `V/N=0.1%/0.4%/0.5%`、C1/C60，
固定 predicate schedule、30 秒 warmup、12 个 ABBA windows，并闭合 predicate rows、
IDs appended、Dense words、最终表示、MVCC surviving IDs 和返回 count。cache-on 作为
独立 hold-out，不与 cache-off 聚合；Query 的 switch point 不从 Search 结果继承。

第二组保持 Cohere 1M×768D、最终 accepted ID 集合、向量、NQ/topK、Cardinal route 和
search work 一致，只改变 scalar predicate 类型。新增字段由同一随机 hit mask 编码，
至少覆盖 raw INT64、算术表达式和 VARCHAR prefix/LIKE；先测 `V/N=0.1%/0.4%`、C1/C60。
raw numeric 是 heavy predicate 的因果基线，已有 scalar-index INT64 结果只作实际 E2E
参考，不能与 raw VARCHAR 的差值直接解释成算子单位成本。

重型 predicate 的判定分两层：若 Dense/Adaptive scalar 绝对差基本不变而 endpoint
百分比缩小，说明共同 predicate 成本只是在稀释收益；若 scalar 差值随表达式复杂度
扩大，则继续拆分 generic Adaptive 的 batch 调用、batch-local mask scan、ID append、
变长字段 chunk pin/decode 和 cache 行为。当前非 INT64 range 表达式通常只能经 generic
evaluator 形成最终 Sparse，不能默认宣称已实现 A→Sparse→heavy B 的 candidate-only
消费；该链路需单独闭合 capability、eval-row counter 和结果后才进入性能结论。

##### 0.6.5.1 执行结果：Query 隔离

执行环境固定为 Cohere 1M collection、随机分布 `a/b` INT64、单 sealed segment、
cache-off、`count(*)`、30 秒预热、12 个 DS/SD ABBA paired windows。C1 每 slot
50 requests；C60 为 60 个 closed-loop worker、每 slot 合计 300 requests。这里的目的
是剥离 vector search 后定位差异所在，**不是**独立判断 Sparse 作为 Query 产品结果
表示是否值得采用。

| Case | C | 返回 count | Dense mean | Sparse mean | Dense QPS | Sparse QPS | paired QPS delta | 一致窗口 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | 1 | 1,000 | 2.750 ms | 2.547 ms | 361.5 | 390.4 | **+8.00%** | 12/12 |
| Single | 60 | 1,000 | 29.286 ms | 27.748 ms | 1,622.4 | 1,716.0 | **+5.80%** | 12/12 |
| AND-pass | 1 | 1,000 | 3.560 ms | 2.835 ms | 279.4 | 350.8 | **+25.54%** | 12/12 |
| AND-pass | 60 | 1,000 | 34.085 ms | 29.194 ms | 1,401.6 | 1,635.7 | **+16.80%** | 12/12 |
| AND-reduce | 1 | 1,015 | 3.504 ms | 3.057 ms | 283.9 | 325.5 | **+14.62%** | 12/12 |
| AND-reduce | 60 | 1,015 | 34.181 ms | 47.407 ms | 1,398.6 | 1,022.9 | **-26.80%** | 0/12 |

所有 case 的 Dense/Adaptive count 相同；Adaptive 每请求精确增加一次 Sparse output
counter，Dense 不增加，未发生 placeholder bitmap 误消费。Single 隔离的是输出与
handoff；AND-pass 证明第二谓词直接消费约 1,000 IDs 时可删除大部分全段工作；
AND-reduce 还需对 A 的约 4,000 candidates 执行 B。后者在 C1 为正收益、C60 却稳定
劣化，且没有 vector search 可作为稀释项，说明并发放大来自 Sparse B consumer 或其
调度/内存访问路径。该点作为独立性能缺口保留，不能用其它正收益 case 平均掉。

异常定位按以下顺序执行，不先预设为随机读取或调度问题：

1. 固定同一 collection、query、C60、count(*) 和请求数，分别采 Dense AND-reduce、
   Sparse AND-pass、Sparse AND-reduce，闭合 A 输出 V、B 输入/输出 V 与调用次数；
2. 对三条路径分别采 CPU `perf record`，按 scalar producer、Sparse candidate resolve、
   chunk pin、B value read/match、ID append、MVCC、allocator 和 scheduler 大模块归类；
3. 单独采 task-clock/context-switch/page-fault，并结合进程 CPU 利用率区分 CPU work、
   off-CPU 停顿和内存并发放大；PMU events 不做 multiplex 后直接比较；
4. 做 `V_A=1000/2000/4000/8000` 与 `C=1/15/30/60` 的最小 sweep，判断成本是
   `O(V)` 平滑增长还是跨并发/working-set 阈值突变；
5. 只有 access count、hotspot 和 endpoint 方向闭合后，才决定是复用 scratch buffer、
   减少 per-request allocation、调整 chunk-local traversal/prefetch，还是处理线程调度。

首轮 C60 定位已固定同一 collection、`count(*)`、30 秒预热、55 秒
continuous workload 和每个 slot 300 requests；`perf stat` 均在 workload 稳定后
采集 15 秒：

| 路径 | A 输出 / B 输入 | QPS | task-clock / 15 s | 平均 CPU | context switch | context switch/s | page fault |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense AND-reduce | Dense universe | 1,369.53 | 27,977 ms | 1.87 cores | 192,846 | 6.89K | 7,003 |
| Sparse AND-pass | 约 1,000 IDs | 1,582.42 | 45,848 ms | 3.06 cores | 388,387 | 8.47K | 14,078 |
| Sparse AND-reduce | 约 4,000 IDs | 1,104.58 | 63,799 ms | 4.25 cores | 1,661,485 | 26.04K | 12,760 |

Sparse AND-reduce 相对 Sparse AND-pass 不仅增加了 B 的 candidate reads：平均
CPU 从 3.06 cores 增至 4.25 cores，context switch 从 8.47K/s 增至
26.04K/s，QPS 却降至 1,104.58。page fault 没有同向暴涨，因而当前不支持
将 page fault 视为主因。CPU profile 显示 Sparse B consumer 经过
`FilterNativeIdsByRawData -> get_chunk_by_offset -> CapturePublishedState`，并出现
shared-pointer atomic/lock、mutex 与 futex 热点。这是“逐 ID chunk 定位导致锁/唤醒
放大”的强相关证据，但在实际 chunk 数、每请求定位次数与 batch 替代
路径闭合前，不作最终因果结论。

原始产物：

```text
artifacts/query-and-reduce-c60-investigation-20260828/
```

代码核对发现，storage-v2 已有 `Segment::bulk_subscript` 批量读取路径：它只捕获
一次 published state，通过 `GetChunkIDsByOffsets` 批量完成 offset→chunk 映射，
并对 touched chunks 去重后一次 pin。异常路径没有复用该 API，而是对 B 的
每个 candidate 调用 `get_chunk_by_offset`，使同一请求反复执行
`atomic_load(shared_ptr)` 及其 `_Sp_locker`。

为验证因果，做了一个单变量修复：仅当 skip index 无法跳过任一整块
chunk 时，将逐 ID resolve/read 替换为现有 `bulk_subscript`；可剪枝的 case 仍保留
原 per-chunk 路径。predicate、V、输出 Sparse IDs、MVCC/visibility、`count(*)` 和
C60 workload 均未变。

| Sparse AND-reduce | 修复前 | batch 修复后 | 变化 |
|---|---:|---:|---:|
| QPS | 1,104.58 | 1,645.67 | **+48.99%** |
| task-clock / 15 s | 63,799 ms | 44,587 ms | **-30.11%** |
| 平均 CPU | 4.25 cores | 2.97 cores | **-30.12%** |
| context switch/s | 26.04K | 7.58K | **-70.89%** |
| page fault / 15 s | 12,760 | 13,576 | +6.39% |

修复后再执行 12 个 DS/SD ABBA paired windows：Dense/Sparse `count(*)` 均为
1,015，Sparse 在 12/12 windows 全部更快；Dense/Sparse 平均 QPS 为
1,403.05/1,661.49，paired QPS 改善 **18.50%**；平均 endpoint latency 为
33.957/28.690 ms，P90 为 58.096/48.448 ms。因此异常已闭合为：
逐 ID published-state/shared-pointer 锁竞争在 C60×V 下引起 context-switch/唤醒
放大，而非 O(V) 随机读取本身或 page fault。

修复验证产物：

```text
artifacts/query-and-reduce-c60-investigation-20260828/
  sparse-reduce-batch-fix-workload.jsonl
  sparse-reduce-batch-fix.perf.stat
  and-reduce-batch-fix-abba.jsonl
```

##### 0.6.5.2 执行结果：等命中集合的重型 predicate Search

第一轮使用同一列与同一命中集合做严格控制变量：light 为 `a < V`，arithmetic 为
`(a % 1000000) < V`；由于 `a` 是 `[0,N)` permutation，两者最终 IDs 完全相同。
固定 Cohere 1M×768D、单 sealed segment、CARDINAL_TIERED、COSINE、NQ=1、topK=10、
50 个固定 queries、30 秒预热和 12 个 ABBA windows。每个 case 均通过 strict closure：
Dense 实际走 auto-IVF，Sparse 实际走 direct-BF；50/50 query TopK 完全一致，Sparse
BF distance attempts 精确为 `V/query`。

| Predicate | V/N | C | Dense QPS | Sparse QPS | paired QPS delta | paired latency improvement | 一致窗口 |
|---|---:|---:|---:|---:|---:|---:|---:|
| light | 0.1% | 1 | 226.8 | 285.3 | **+25.75%** | 20.52% | 12/12 |
| arithmetic | 0.1% | 1 | 157.2 | 178.3 | **+13.43%** | 11.86% | 12/12 |
| light | 0.4% | 1 | 242.0 | 265.0 | **+9.50%** | 8.70% | 12/12 |
| arithmetic | 0.4% | 1 | 161.2 | 166.9 | **+3.51%** | 3.38% | 12/12 |
| light | 0.1% | 60 | 1,352.8 | 1,537.8 | **+13.67%** | 11.88% | 12/12 |
| arithmetic | 0.1% | 60 | 1,141.2 | 1,265.5 | **+10.88%** | 11.47% | 12/12 |
| light | 0.4% | 60 | 1,417.1 | 1,530.9 | **+8.04%** | 7.20% | 12/12 |
| arithmetic | 0.4% | 60 | 1,171.5 | 1,233.7 | **+5.30%** | 5.38% | 12/12 |

scalar evaluator counter 给出的每次 evaluation 均值进一步闭合了变量：C1 light 约
`0.72/0.57 ms`（Dense/Sparse），arithmetic 约 `2.72/2.71 ms`；C60 light 约
`0.90/0.64 ms`，arithmetic 约 `3.37/3.34 ms`。算术表达式增加的是两侧共同成本，
Sparse 没有出现随算子复杂度扩大的额外负成本；表现为绝对收益仍在、endpoint 百分比
被共同 predicate work 稀释。V 从 1,000 增到 4,000 后 search work 上升，收益也按
预期继续收缩。

原始产物：

```text
artifacts/query-isolation-e2e-20260828/
  *-c{1,60}.jsonl                       # Query isolation
  search-{light,arithmetic}-v{1000,4000}-c{1,60}.jsonl
  targeted-cpp-regression.log
```

## 1. 背景结论（来自 P90 定位实验）

- P90 劣化根因是 Sparse B 消费者 `b_read` 的散布读对 cache/TLB 驻留敏感（双峰 ~25us / ~1.5ms），prefetch 减半但无法消除（残留为 TLB/EPT miss）。
- 决策口径：**不看 P90，只看 mean/QPS**。
- 修复现状：prefetch 已落地（`kPrefetchAhead=16`）；huge page 受环境阻塞；向量搜索阶段确认无尾。

> **历史原型说明。** 本节至第 6 节记录早期“Sparse 与 BF 路由耦合、cap 后
> Dense fallback”的实验和结论；当前实现契约以第 7 节（predicate 输出必须为
> Sparse）及第 9 节（consumer capability）为准。它们取代了本节中“sparse 只进
> BF”及第 4 节 Step 4 尚未实现的描述。

## 2. 落地设计（历史原型）

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

**附带发现（已于 2026-08-19 更正，见第 6 节）**：早期把 `auto` 与显式 per-query 模式的 topK 差异归因为同一 BF 内的 `ScanRangeFilter`/valid-ID traversal 差异；请求级 route 诊断证明该前提不成立：`auto` 实际走 IVF，显式模式刻意强制 BF。因此该差异不能作为 Cardinal filter membership 或 BF scan 正确性问题的证据。

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

## 6. Auto vs 显式模式路由确认（2026-08-19）

### 目的

核对早期观察到的 `auto` 与 `dense_per_query` / `valid_ids_per_query`
topK 不同，究竟是 Sparse/过滤正确性问题，还是两条请求实际走了不同
Cardinal 搜索路径。

### 固定实验条件

| 项 | 值 |
|---|---|
| 服务 | 本机隔离 standalone，`:19532` |
| collection | `cardinal_sparse_sort_1m`（1M rows，`a` 为 STL_SORT） |
| 向量 / 搜索 | fp32 128D，L2，topK=10，ef=64 |
| filter | `a < 1000`；实际 valid IDs=901 |
| query | `np.random.default_rng(1733)` 的第 0 条固定 query |
| 唯一变量 | `bf_filter_scan_mode`: `auto` / `dense_per_query` / `valid_ids_per_query` |

在实际被 Milvus 链接的内嵌 Cardinal 库中临时加入 request-level
diagnostic，记录每个 segment query 的搜索器计数、scan 类别、distance
attempts 与 refine 数。诊断代码仅用于本机验证，已撤销，未提交或推送。

### 结果

| 模式 | 实际搜索路径 | 计算 / scan 计数 | 结果关系 |
|---|---|---|---|
| `auto` | IVF + refine | IVF compute=596；refine=64；BF=0 | 未返回 `163650` |
| `dense_per_query` | BF + refine | BF attempts=901；valid-ID scan=1；refine=64 | 返回 `163650`（rank 2） |
| `valid_ids_per_query` | BF + refine | BF attempts=901；valid-ID scan=1；refine=64 | 与显式 Dense 完全一致 |

`dense_per_query` 与 `valid_ids_per_query` 此前在 50 个固定 query 上均
得到完全相同的 topK ID 和 distance；本次 request-level 计数也确认两者
走的是同一 BF valid-ID 枚举路线。因此当前没有 Sparse membership 漏判
证据。

### 结论与实验约束

1. `auto` 是 Cardinal 自适应调度，不是 explicit BF 的 Dense baseline；
   它在此点选择 IVF，而两个 explicit modes 在 dispatcher 中被设计为
   强制 BF。
2. 因 route、candidate 数及 approximate/refine 行为都不同，`auto` 与
   explicit modes 的 endpoint latency/topK 均不可用于 Sparse A/B 归因。
3. 尝试通过请求参数 `index_algo=BF` 强制 `auto` 未改变该实际 IVF route；
   该参数在此链路不是有效的 route override。
4. 后续性能 A/B 必须固定为 `dense_per_query` vs
   `valid_ids_per_query`（同 BF、同 valid-ID enumeration），或先实现并
   验证真正的 representation/route 解耦后再使用 auto 作为对照。

## 7. 链路落地待办（按阶段重排，2026-08-19）

核心目标不是先挑选阈值，而是让 sparse 成为 Dense 并列、可沿执行计划传递的
filter-result 表示：第一个 predicate 可产出它，第二个 predicate 与向量搜索
可消费它。只有链路完整后，阈值和性能口径才有意义。

### 7.1 已校正的参数语义（2026-08-19）

`filter_result_representation` 不是 auto-tuning hint，也不是 Cardinal BF 的
调度开关。验证阶段采用如下**硬契约**：

| 项 | `dense`（基线） | `sparse`（验证特性开启） |
|---|---|---|
| 每个过滤判定的输出 | Dense filter result | **必须为 Sparse filter result** |
| 判定的输入 | Dense 或无上游结果 | Dense、Sparse 或无上游结果均可 |
| 不支持 native sparse 的 predicate | 现有 Dense 执行 | 可在判定内部临时使用现有 Dense kernel，但在节点边界必须转为 Sparse 输出；不得把 Dense 作为对外 fallback |
| 向量搜索的输入 | 接受 Dense | 接受统一 `FilterResult`；实际收到 Sparse 时走相应 consumer，实际收到 Dense 时仍可走现有路径 |
| 搜索算法 / auto 路由 | 由既有 planner/dispatcher 决定 | **不由此参数改变**；BF、Graph、IVF 的选择是独立问题 |

因此正确的数据流是：

```text
FilterResult(Dense | Sparse | initial universe)
    -> predicate A: Apply(input) -> Sparse
    -> predicate B: Apply(input) -> Sparse
    -> ...
    -> vector search: Consume(Dense | Sparse)
```

这里的“必须为 Sparse”约束的是**判定对外可见的结果表示**，而不是要求每个
predicate 的内部实现都放弃现有 SIMD / scalar-index Dense kernel。对于尚无
direct sparse producer 的表达式，正确的过渡实现是“Dense evaluate -> 枚举
accepted IDs -> Sparse output”；对于收到 Sparse input 的表达式，应只判断这些
candidate IDs 并继续输出 Sparse。这样不会遗漏任何条件，也不会因 child 数量、
cap 或当前 consumer 而静默改变表示。

`Sparse` 的完整契约还需包含 polarity：`Included IDs`（valid 很少）与
`Excluded IDs`（invalid 很少）。第一阶段可先只落地 `Included IDs`，但类型
和 API 不得再以 `ValidIdPayload` 命名或假设为唯一最终形态。

**当前原型与该契约的差距**：它仅在 sealed/受支持 producer 时产生 accepted
list，unsupported/cap/growing 时回退 Dense；并且 `VectorSearchNode` 看见 payload
就改写为 `valid_ids_per_query`，等价于强制 BF。这些都只是原型行为，不能作为
最终接口或测试结论的前提。

### 7.2 本轮实现进度（2026-08-19，进行中）

已开始收敛原型到第 7.1 节契约：

1. `FilterBitsNode` 在 `sparse` 下优先使用 direct/native producer；若表达式
   不支持该 producer，则运行原有 Dense evaluator，并在 FilterBits 边界枚举
   accepted IDs 后交付 Sparse。cap 不再决定对外回退到 Dense；growing segment
   也走这条 Dense-evaluate-to-Sparse 的正确性路径。
2. `MvccNode` 对 Sparse list 在 sealed 与 growing 均施加 timestamp/delete/TTL
   visibility compact；不能以 segment 类型作为 Sparse 的拒绝条件。
3. `VectorSearchNode` 不再因为 Sparse payload 改写 `bf_filter_scan_mode`。
   Cardinal 侧将 list 在 BF 直接枚举，而在非 BF route 转换为 Bloom+Flat exact
   membership adapter。

尚未完成的是把目前 `PhyConjunctFilterExpr` 的两个 AND child 专项优化，替换为
真正的 predicate-level `Apply(FilterResult)`。因此本轮代码已经保证**整个
FilterBits predicate 的对外输出**为 Sparse，但尚未做到逻辑树中每个中间 child
均通过统一 variant 传递；该项仍是下一项开发与测试重点。

### 7.3 当前开发：predicate-level Sparse apply（2026-08-20）

第一步以低风险迁移方式引入 `SparseFilterResult`（immutable accepted IDs +
universe）与 `Expr::TryApplySparseFilter(EvalCtx&, optional<input>)`：

1. 默认 adapter 复用既有 leaf 的 producer/consumer hooks；无 input 时调用
   `TryGetNativeValidIds`，有 input 时调用 `TryFilterNativeValidIds`，因此不改变
   已验证的 Unary/Binary range 实现。
2. `PhyConjunctFilterExpr` 改为只编排 child 的 `TryApplySparseFilter`，不再直接
   调用 leaf-specific hook。嵌套 AND 因而可作为普通 predicate 消费上游 Sparse
   结果。
3. `FilterBitsNode` 的 native path 改为只消费 `SparseFilterResult`，并在进入
   QueryContext 前验证 predicate universe 与当前 FilterBits universe 相同。
4. 本提交只覆盖 Included IDs + AND；OR/NOT/nullable 不采用错误的“逐 ID
   相交”捷径，仍通过 Dense evaluator 后在 FilterBits 边界转为 Sparse，待显式
   集合语义设计完成后再迁移。

验证（2026-08-20）：`milvus_core` 与 `all_tests` 均完成构建；下列定向单测
4/4 通过：

```text
ConjunctExprTest.SparseAndPassesOnlyAcceptedOffsetsToSecondPredicate
ConjunctExprTest.SparseAndChainsAcceptedOffsetsAcrossPredicates
ConjunctExprTest.SparseApplyPreservesUniverseAcrossNestedAnd
ConjunctExprTest.SparseAndFallsBackWhenSecondPredicateHasNoSparseConsumer
```

本机构建测试的二进制 RPATH 会优先命中 `internal/core/output/lib` 的旧
`libmilvus_core.so`；执行时将 `cmake_build/src` 置于 `LD_LIBRARY_PATH` 首位，
以确保加载本次构建产物。该问题仅影响本机测试启动，不影响编译或代码语义。

### 7.4 参数级 Milvus E2E（2026-08-20）

`cardinal_sparse_visibility_e2e.py` 已改为只切换
`filter_result_representation=dense/sparse`，不再把 Dense 强制为
`dense_per_query`。全套 visibility closure 通过：sealed multi-segment +
nullable、current delete、TTL（过期前/后）和 growing + nullable 的 topK
`(ID, distance)` 均一致；historical MVCC public API probe 仍受已知 Proxy
限制。

随后在同一隔离 standalone、同一 1M×128 synthetic collection、单 segment、
L2/topK10/NQ1/C1、固定 50 query、10 warmup、12 个 ABBA window 下，过滤为
`a < 1000 and b < 500000`（首 predicate 约 0.1%，最终约 500 valid IDs）。
唯一变量为表示参数：

| 对照 | Dense mean | Sparse mean | mean delta | Sparse 更快 window |
|---|---:|---:|---:|---:|
| `filter_result_representation=dense/sparse`（auto routing） | 3.575 ms | 3.468 ms | -2.99% | 11/12 |
| legacy `dense_per_query/valid_ids_per_query`（explicit BF） | 3.518 ms | 3.400 ms | -3.35% | 12/12 |

> **路由更正（2026-08-20）**：这张早期 endpoint sanity 表不能再被解读为
> “两个同为 BF 的对照”。随后以同 collection 的 `perf` 调用栈确认，auto
> 的 Dense 请求实际进入 IVF；故其 -2.99% 仅是 IVF endpoint 结果。这里的
> legacy 数字也不是当前规范的 BF 复现口径。可靠的 BF 结果、ABBA 配置和
> Dense/Sparse route closure 见 10.5；Memory/Disk 的 auto-BF direct-Index
> 对照见 10.4.2。该表保留仅用于记录当时的参数链路 sanity，不作 route-level
> 性能归因。

| 阶段 | 优先级 | 事项 | 完成标准 |
|---|---|---|---|
| 1. 表示契约与 producer | 已实现（过渡实现） | 第一个 predicate 输出 sparse | sealed native producer 直接交付 accepted-ID list；unsupported/cap/growing 在 FilterBits 内复用 Dense evaluator，但节点边界一律交付 Sparse，满足第 7.1 节的硬输出契约 |
| 1. 表示契约与 producer | 高 | 泛化并固化 sparse 契约 | 引入 `FilterResult(Dense | SparseIncluded [| SparseExcluded])`，明确 universe、cardinality、生命周期、row-ID order 与 MVCC/TTL/delete 责任；`sparse` 参数下所有判定边界必须输出 Sparse |
| 1. 表示契约与 producer | 已实现 | 固化 row-ID order | Sparse payload 明确为 ascending、unique row IDs；`NativeValidIds_StlSortRangeUsesAscendingRowIdOrder` 验证 STL_SORT range 返回 `{0,3,4}` |
| 1. 表示契约与 producer | 中 | 补齐 producer 覆盖 | bitmap posting、STL_SORT range、raw-data scan 已有原型；补 producer × 表达式 × visibility 的正确性覆盖，并决定 raw-data 的 cap+fallback/单趟切换 |
| 2. 复合过滤消费 | 已实现（过渡实现） | `A → sparse IDs → B` | 支持任意长度的 AND：首个 native producer 的 IDs 依次传给其余 child；任一 child 不支持 sparse consumer 时，可内部回到 Dense evaluator，但 FilterBits 边界仍输出 Sparse |
| 2. 复合过滤消费 | 高 | predicate-level Apply | 将当前 AND 专项链路收敛为 `Apply(FilterResult)`：每个 predicate 接收 Dense/Sparse input 并产生 Sparse output；OR/NOT/nullable 采用显式语义正确的集合操作或内部 Dense 计算后再转 Sparse |
| 2. 复合过滤消费 | 高 | visibility 正确性闭环 | delete/TTL/future-insert/native 历史 MVCC、growing Dense-evaluator-to-Sparse 单测已通过；补历史 MVCC 的产品级 E2E（依赖 public time-travel API），并维持 multi-segment/nullable 的 Dense/Sparse 等价 |
| 3. 向量搜索消费 | 高 | 统一 consumer 输入 | VectorSearch 接收 `FilterResult(Dense | Sparse)`，而非由参数/是否有 payload 改写 dispatcher；Graph 用其 membership adapter（例如 Bloom+Flat），BF 枚举 Included IDs，IVF 明确直接消费或 adapter 策略 |
| 3. 向量搜索消费 | 中 | 搜索器覆盖 | 对 BF/Graph/IVF 分别验证 Dense/Sparse 的 topK/distance 等价和 representation conversion 成本；Sparse 不得隐式把 auto dispatcher 强制为 BF |
| 4. 自动决策与性能产品化 | 中 | `ShouldUseSparse` 策略 | 基于 producer 类型、V/N、consumer 与 cap 做阈值/cost-model A/B，不再依赖实验参数手动选择 |
| 4. 自动决策与性能产品化 | 中 | 统一收益矩阵和 tail 分析 | 对 producer × consumer × V/N × N（至少 1M/10M）用统一 ABBA/A-A 统计 mean/QPS/P90；定位 sparse 散布读的 P90 重尾并形成使用边界 |
| 诊断项 | 低 | 同 BF 内 scan-path 对照 | 仅在提供真正可强制 auto 到 BF 的诊断开关后，比较 Dense scan 与 valid-ID traversal；当前不能以 `index_algo=BF` 实现 |

## 8. Visibility correctness closure（2026-08-19）

新增可重复的本机 E2E 脚本：
`scripts/cardinal_sparse_visibility_e2e.py`。这是正确性闭环，不统计
性能；每个场景均以相同 predicate 比较 explicit Dense BF
(`dense_per_query`) 与 `filter_result_representation=sparse`，要求完整
topK `(ID, distance)` 相等。数据使用 16D L2 / `CARDINAL_TIERED`，每个
sealed segment 20K rows；`a < 3 and b < 128` 每段约保留 117 行，低于
Sparse cap 并足以避开小 segment 的非-Cardinal fallback。

| 场景 | 覆盖点 | 结果 |
|---|---|---|
| sealed multi-segment + nullable | 两个独立 flush 的 sealed segment；`a` 可空；NULL 不应通过 range predicate | Dense/Sparse topK 完全相等（通过） |
| current delete | 删除低距离、且满足 predicate 的 IDs 1/2；验证其不出现在结果 | Dense/Sparse 相等且均不返回删除 ID（通过） |
| collection TTL | `collection.ttl.seconds=10`；验证过期前非空、轮询至过期后为空 | Dense/Sparse 在两阶段均相等（通过） |
| growing + nullable | indexed sealed base 后插入未 flush rows | Dense/Sparse 相等（通过）；该历史 E2E 记录产生时仍为 Dense 回退，现实现已改为内部 Dense evaluator 后在 FilterBits 边界输出 Sparse，见下方新增单测 |

### Historical MVCC 的限制

脚本也发送了带 `travel_timestamp=insert_ts` 的原始 SearchRequest（不是
仅传 pymilvus keyword），但当前分支仍返回 delete 后视图，未恢复 IDs
1/2。代码核对显示 proxy 的 search task 将 public travel timestamp 覆盖为
`BeginTs`（`internal/proxy/task_statistic.go`），因此这不是 Sparse 与
Dense 的差异，也不能作为本轮 Sparse MVCC 回归结论。

MvccNode 的 native payload 路径已实际使用
`mask_with_timestamps` 与 `mask_with_delete` 对 list 做 compact；当前
delete/TTL E2E 证明这两类可用 visibility mask 的 Dense/Sparse 等价。

已新增直接针对 `PhyMvccNode` native payload 的 SegCore 单测
`NativeValidIds_HistoricalSnapshotPrecedesDelete`：delete timestamp 为 10
时，timestamp=5 的历史快照必须保留候选 IDs 0/2；晚于该 timestamp
插入的候选行仍须排除。构建产物同步后，以下 native-list MVCC 组已实际
通过（4/4）：current delete、历史 snapshot precedes delete、TTL expiry、
future insert at snapshot。历史快照的**产品级 E2E**仍依赖恢复/提供受支持的
search time-travel API。

### 单测执行记录

本轮重新构建后，以 `cmake_build/unittest/all_tests` 实际执行：

```text
MvccFastPathTest.NativeValidIds_CompactsDeletedCandidates
MvccFastPathTest.NativeValidIds_HistoricalSnapshotPrecedesDelete
MvccFastPathTest.NativeValidIds_CompactsTtlExpiredCandidates
MvccFastPathTest.NativeValidIds_CompactsFutureCandidatesAtSnapshot
MvccFastPathTest.NativeValidIds_StlSortRangeUsesAscendingRowIdOrder
MvccFastPathTest.SparseOutput_GrowingFallsBackToDenseEvaluator
ConjunctExprTest.SparseAndPassesOnlyAcceptedOffsetsToSecondPredicate
ConjunctExprTest.SparseAndChainsAcceptedOffsetsAcrossPredicates
ConjunctExprTest.SparseAndFallsBackWhenSecondPredicateHasNoSparseConsumer
```

结果为 **9/9 PASS**，总计 24 ms。该组同时覆盖 native list 的 delete、历史
snapshot、TTL、future insert，STL_SORT 的 row-ID order，growing 下 Dense
evaluator 的 Sparse 输出，以及三种 AND sparse 链式行为。

### 2026-08-20：Growing raw-BF compatibility fix 与重跑

严格 E2E 首次进入 unflushed growing rows 时曾稳定触发 SIGSEGV。gdb 的调用栈
确认路径为 `knowhere::brute_force_dense_impl -> FAISS BitsetViewSelectorHelper
-> BitsetView::test()`；`ValidIdList` 的 `bits_` 按设计为空，但 generic
Knowhere BF 仍将它作为 Dense bitset 读取。这不是 MVCC mask、ID list 生命周期
或 Cardinal BF 的问题，而是一个未声明的 consumer capability 缺口。

修复将转换限制在两个明确的 legacy boundary：`SearchOnGrowing` 的 raw growing
chunks、`SearchOnSealedColumn` 的未建索引 sealed raw column。两处将 accepted
Sparse IDs 还原为 legacy Dense filtered-bitset；已建 Cardinal index 的路径仍直接
接收 `ValidIdList`，不引入该转换。

随后在全新 standalone 数据目录重建并重跑脚本，完整输出为
`/tmp/milvus-sparse-fix-20260820/visibility.log`：

| 场景 | 结果 |
|---|---|
| sealed multi-segment + nullable | Dense/Sparse topK 相同 |
| current delete | Dense/Sparse 相同，IDs 1/2 均被排除 |
| TTL | 过期前相同且非空；轮询后相同且为空 |
| growing + nullable | Dense/Sparse 相同；结果实际含 unflushed IDs `300001`、`300002` |

脚本输出 `visibility_closure_passed`，服务日志未出现新的 SIGSEGV。public
time-travel API 的限制仍如上一节所述，因此历史 MVCC 的产品级 E2E 不在本次
通过范围内，继续由上述 SegCore 单测覆盖。

## 9. Consumer capability 收敛计划（2026-08-20，进行中）

本轮确认 Sparse 不能仅以 `BitsetView` 的物理形态隐式传给任意向量搜索器：
`ValidIdList` 是 accepted-ID enumeration，既不是 Dense filtered bitmap，也不是
任意点查询可直接使用的 membership structure。最终策略不是让每个 index 自行
实现 Dense fallback，而是在 search dispatcher 明确声明 consumer capability：

| capability | consumer | Sparse 的处理 |
|---|---|---|
| `EnumerateSparse` | BF | 直接枚举 accepted ID list |
| `SparseMembership` | Cardinal Graph / IVF | 一次性转换为共享、精确的 Bloom+Flat membership adapter |
| `DenseOnly` | legacy raw Knowhere BF | **仅在该 compatibility boundary** materialize 为 Dense filtered bitmap |

实现要求：

1. Cardinal dispatcher 将 “list 保留给 BF、list 转 Bloom+Flat 给 Graph/IVF”
   收敛为显式 capability-to-adapter 映射，不再把构造细节散落在单一路由分支。
2. Milvus 的 `SearchOnGrowing` 与 `SearchOnSealedColumn` 保持为当前唯一
   `DenseOnly` boundary；其他 Cardinal indexed path 不得因收到 Sparse 而
   materialize Dense。
3. 新增 capability-level 单测：验证同一个 immutable list 在 BF 仍为 list、在
   Graph/IVF 为 exact Bloom+Flat，并保持 universe、filtered count 和 owner
   lifetime。后续新 index 必须先声明 capability；未适配的 consumer 不得静默
   接收 list。

实施进度（2026-08-20）：已在 Cardinal 定义 `FilterConsumerCapability`，并将
dispatcher 中的隐式 list 分支改为 `AdaptForConsumer()` 映射：BF 选
`EnumerateSparse`，Graph/IVF 选 `SparseMembership`。`DenseOnly` 已具备
list-to-Dense materialization，只供 legacy compatibility boundary 或未来显式
声明的 Dense-only consumer 使用；Cardinal dispatcher 不选择该能力。对应单测与
构建验证已通过：独立 Cardinal Release 构建成功；启用 `WITH_UT=ON` 后，
`test_cardinal "Valid-ID list adapts only through declared consumer capabilities"`
通过（1 test case、29 assertions）。

这一步只收敛 Cardinal 与 legacy raw-BF 的既有支持面；通用 Knowhere 非-Cardinal
index 的完整 Sparse 支持仍需要单独引入同一 capability contract，不能据此宣称已
覆盖全部 index。

## 10. BF 收益复现与路由确认（2026-08-20，当前优先级）

### 10.1 重新收敛的目标

先不修改 Tiered/ObjectStore 的 auto selector，也不把 Memory/Disk 的矩阵与当前
结论混在一起。当前优先目标是：**在同一条、被直接证明为 BF 的 Cardinal 路由上，
重新建立 Dense 与 Sparse 的可归因性能对照。**

此前参数级 E2E 的 `filter_result_representation=dense/sparse` 默认使用 auto；对
其 Dense 请求的 clean perf 已证实实际为 IVF（`IvfSearchImpl` 栈约 24%，无 BF
搜索栈）。因此该表中的约 -3% 只能描述 Sparse 在 IVF 路径的 endpoint 表现，不能
与历史 explicit-BF 的约 -18.65% 并列解释为“BF 收益下降”。

Tiered 映射为 Cardinal `IndexType::ObjectStore`。它的
`SelectObjectStoreSearcher()` 只在以下任一条件成立时选择 BF：

```text
search_candidates_limit (ef) >= V
topK >= V
max_codes >= V
```

它**不**调用 Memory/Disk 路径使用的 `ShouldSwitchBitsetCheck(filter_rate)`；故
`V/N≈0.1%` 或更低本身不能保证 Tiered auto 选择 BF。在当前 1M case，A 约保留
1,000 行、B 再保留约一半，最终 `V≈500`，而 `ef=64`、`topK=10` 均小于 V，故选择
IVF 符合现有实现。

### 10.2 复现实验设计

| 项 | 固定值 / 要求 |
|---|---|
| 目标路径 | Cardinal BF；不以请求参数标签或预期过滤率推断，必须由真实计数确认 |
| 数据与过滤 | 现有 1M × 128D synthetic 单 sealed segment；L2；`a < 1000 and b < 500000`；最终 V 约 500 |
| 两个表示 | `dense_per_query` vs `valid_ids_per_query`；这是现有可靠的 BF 诊断模式，不使用 auto 作为 BF 基线 |
| 搜索参数 | topK=10、ef=64、NQ=1、C=1；固定 query 集与相同 binary/index/segment 状态 |
| 正确性闭环 | 计时前逐 query 对比 topK 的 `(ID, distance)`；不一致即停止性能归因 |
| 预热与统计 | 每模式 10 warmups；12 个 ABBA window；每 slot 50 个固定 query；报告每请求 mean/median/P90、窗口均值及 Sparse 胜出窗口数 |
| 路由闭环 | 每个 timed mode 导出 `bf_search_cnt/ivf_search_cnt/graph_search_cnt`；要求 BF>0、IVF=0、Graph=0，且两表示计数一致 |
| 归因验证 | 在稳定 collection 上对两模式分别 `perf record`；检查 BF 栈、valid-ID scan/dense filter preparation 计数以及无 build/compaction 干扰 |

`index_algo=BF` 目前不是 Tiered 的可靠 route override：ObjectStore selector 未读取
该字段。若要测试“正常 representation 参数 + 非实验 BF 路由”，应另行实现一个
明确、仅诊断用途的 route override；在此之前不得把它用作 BF 证据。

### 10.3 route counter 落地

Cardinal 已在 `CardinalSearchRecord` 中维护 `bf_search_cnt`、`ivf_search_cnt` 与
`graph_search_cnt`，并由 Knowhere 记录为 metrics；当前缺口是 E2E harness 没有将它
们按本次请求/测试窗口导出，且 `actual_payload_route` 只是脚本标签。

下一步在不改变 selector 的前提下，补可读取的诊断输出，至少记录：

```text
selected route count (BF / IVF / Graph)
V, filtered count, ef, topK, max_codes
filter representation before/after consumer adaptation
```

同时修正脚本中“default modes 已走 BF”的错误 help/报告文案。route counter 先用于
上述 BF 复现的 closure；确认稳定后，再作为统一口径扩展到 Memory、Disk、Tiered 的
auto 路由矩阵。

### 10.4 后续拓展（不阻塞本轮）

BF 复现成功后，增加 Memory、Disk、Tiered × Dense/Sparse × 高过滤的 route-first
矩阵。Memory/Disk 的 selector 会把 filter rate 纳入切换（包括
`ShouldSwitchBitsetCheck`），故应验证它们在实际配置下是否自动进入 BF，并只在
Dense/Sparse 实际走同一路由时比较性能。Tiered 是否也应采用高过滤 BF 策略是独立
产品决策，需基于 BF 收益、recall、tail latency 与资源成本再评估，不能先由本次
Sparse 实验隐式改变。

#### 10.4.1 route-first selector 验证（2026-08-20）

已新增并运行 Cardinal 单测
`High filter ratio selects BF for Memory and Disk but not Tiered ObjectStore`。
固定 `N=1,000,000`、`V=500`、`ef=64`、`topK=10`、`nbuckets=1024`，结果：

| `IndexType` | selector 结果 | 结论 |
|---|---|---|
| `Memory` | `LowBitBruteForceSearcher` | 高 filter rate 的 normal selector 进入 BF |
| `Disk` | `LowBitBruteForceSearcher` | 同上 |
| `ObjectStore`（Tiered） | `LowBitIVFSearcher` | 不因 filter rate 进入 BF |

命令使用 Cardinal 的 `build/sparse-milvus` 配置，测试实际通过（3 assertions）。
这一步是 selector-level closure，不是 endpoint 性能结果。下一层是为 Memory/Disk
分别以真实 `Index::Search` 验证 record 的 BF/IVF/Graph 计数和 Dense/Sparse 结果
等价；只有 route 一致且均为 BF 后才测 ABBA 性能。

当前开源 Milvus/Knowhere bridge 只注册 `CARDINAL_TIERED` 公共 index type；故
Memory/Disk 不能在这个 standalone 上仅替换 `index_type` 做同构 Milvus E2E。它们
需先通过 Cardinal direct-Index harness 验证；若需要完整 Milvus E2E，则需要 VDC
内部对应 index registration/构建环境，不能把 Tiered E2E 冒充为 Memory/Disk 结果。

#### 10.4.2 Memory / Disk direct-Index auto-BF 对照（2026-08-20）

为避免把 selector 单测当成真实执行证据，新增 test-only 工具
`cardinal/tools/sparse_index_type_benchmark.cpp`。它通过真实 `Index::Build` 和
`Index::Search` 运行 Memory/Disk；构建采用 `IndexAlgo=Ivf`（避免将 graph build
时间混入本次 BF consumer 对照），搜索仍使用**正常 auto selector**，并从每个
`CardinalSearchStatistics` 累计 BF/IVF/Graph 次数。不是 explicit-BF 诊断模式。

固定条件：随机生成的 1M × 128D FP32、L2、`V=500`（0.05% valid）、topK=10、
ef=64、NQ=1；固定 50 query；先逐 query 对比 Dense/Sparse topK `(ID,distance)`；
再各预热 20 请求；6 个 ABBA window（每 window 为 `Dense → Sparse → Sparse → Dense`，
每 slot 50 请求）。所以每种表示 600 个 timed `Index::Search`。Dense 是 native
filtered bitset；Sparse 是 native accepted-ID list；没有 Dense materialization。

| direct `IndexType` | Dense mean | Sparse mean | Sparse delta | Dense route (BF / IVF / Graph) | Sparse route (BF / IVF / Graph) |
|---|---:|---:|---:|---:|---:|
| Memory | 0.1208 ms | 0.0419 ms | **-65.33%** | 600 / 0 / 0 | 600 / 0 / 0 |
| Disk | 0.1220 ms | 0.0424 ms | **-65.25%** | 600 / 0 / 0 | 600 / 0 / 0 |

该结果闭合了三件事：Memory/Disk 的高 filter auto route 的确为 BF；两种表示在实际
`Index::Search` 下结果相同；同 route 下 Sparse 保留了明显的 BF consumer 收益。
它是 Cardinal direct-Index benchmark，**不等价于**包含 proxy/SegCore/expression
execution 的 Milvus endpoint E2E，也不应与 10.5 的 Tiered E2E latency 横向比较。
原始输出：

```text
/tmp/cardinal-sparse-index-type-20260820-memory.log
/tmp/cardinal-sparse-index-type-20260820-disk.log
```

#### 10.4.3 Tiered auto-BF 的最小-V Milvus E2E（2026-08-20）

为补齐 Tiered 的自然 auto-BF 缺口，新增 collection
`cardinal_sparse_tiered_autobf_v64_20260820`。仅改变 10.2 case 的 scalar
predicate：`a < 64 and b < 1000000`，使最终 `V=64`，满足 ObjectStore selector
的 `ef=64 >= V` 条件；不传 `index_algo=BF`。其余为 1M × 128D synthetic、单
sealed segment、L2、topK=10、ef=64、NQ=1/C=1、固定 50 query。Dense/Sparse
计时前逐 query 比对前 10 个 topK `(ID,distance)`；20 warmups 后执行 12-window
ABBA（每 slot 50 query），每模式 1,200 请求。

| 指标 | Dense | Sparse | Sparse 相对 Dense |
|---|---:|---:|---:|
| mean | 3.829 ms | 4.020 ms | **+5.00%** |
| median | 3.807 ms | 3.155 ms | -17.12% |
| P90 | 4.071 ms | 8.842 ms | +117.21% |
| Sparse 更快 paired window | - | 3 / 12 | - |

两侧各另以 6,000 请求的连续 workload（每侧 20 次预热，NQ=1）配合 20 秒
`sudo perf record -F 199 -g -p <milvus-pid>` 复核。Dense 与 Sparse 均出现
`BruteForceSearcher` / `BruteForceSearchImpl`，且两份 report 都没有
`IvfSearchImpl`、`IvfSearcherImpl`、`GraphSearcher` 或 `GraphSearchImpl`；故这是
真实 Tiered auto-BF，而非 route 混杂。Dense 代表性调用栈还包含
`FilterCheckerView::BitCompressBatch64`（约 1.02% sampled stack）。

该结果**不支持**“只要 Tiered auto 选 BF，Sparse endpoint 必然更快”的结论。本 case
没有 scalar index，A 由 raw scalar scan 产生 64 个 IDs；Sparse 虽省去 Dense 的
BF filter 枚举，但仍须完成整段 predicate 求值和 Milvus payload handoff，而 vector
distance 工作仅有 64 次。因而“producer/hand-off 固定成本及尾部波动大于所省 BF
枚举”是当前最符合路径结构的假设，尚需 predicate、payload handoff、Cardinal search
三段的请求级计时才能作为归因结论。它与 10.4.2 的 direct consumer（不含 Milvus
producer/endpoint）以及 10.5 的 explicit-BF E2E 是不同层级的测量，不能混合成
单一收益百分比。

原始产物：

```text
/tmp/milvus-sparse-fix-20260820/tiered-autobf-v64-abba.log
/tmp/milvus-sparse-fix-20260820/tiered-autobf-v64-{dense,sparse}-perf-workload.log
/tmp/milvus-sparse-fix-20260820/tiered-autobf-v64-{dense,sparse}.perf.{data,txt}
```

### 10.5 本轮 BF 复现结果（2026-08-20）

按 10.2 的配置，在已加载且无 `knowhere_build*` / compaction CPU 的隔离
standalone 上完成。正确性 closure 对前 10 个固定 query 比较 `(ID, distance)`，
Dense 与 Sparse 一致；随后进行 12-window ABBA。

| 指标 | Dense per-query BF | Sparse valid-ID BF | Sparse 相对 Dense |
|---|---:|---:|---:|
| endpoint mean | 4.437 ms | 3.833 ms | **-13.62%** |
| endpoint median | 4.385 ms | 3.353 ms | -23.54% |
| endpoint P90 | 4.563 ms | 6.217 ms | +36.23% |
| timed requests | 1,200 | 1,200 | - |
| Sparse 更快的 paired window | - | 12 / 12 | - |

这是一条明确的 BF-vs-BF 对照，因而可确认当前 Cardinal BF consumer 对 Sparse
accepted-ID list 仍有稳定的 mean 收益；它与此前 auto/IVF 的约 -3% endpoint
结果属于不同 route，不能互相替代。Sparse 的 P90 仍较高，符合此前已记录的稀疏
后续 predicate 散布访问尾延迟现象；本轮不将其混入 mean 收益归因。

route closure 不再仅依赖 `legacy_explicit_BF` 标签：两侧各以相同的 20 秒、199 Hz
`sudo perf record -g -p <milvus-pid>` 采样。Dense 与 Sparse 报告均出现
`BruteForceSearcher` / `BruteForceSearchImpl`，而 `IvfSearchImpl`、
`IvfSearcherImpl`、`GraphSearcher` / `GraphSearchImpl` 的匹配数均为 0。Dense
profile 还可见 `FilterCheckerView::BitCompressBatch64`（代表性调用栈 1.28%）；
Sparse profile 未见该符号。这是调用栈证据，按请求导出 BF/IVF/Graph counter 的
诊断项仍按 10.3 保留，后续用于 auto 路由矩阵。

原始产物：

```text
/tmp/milvus-sparse-fix-20260820/bf-repro-abba-20260820.log
/tmp/milvus-sparse-fix-20260820/bf-repro-dense.perf.{data,txt}
/tmp/milvus-sparse-fix-20260820/bf-repro-sparse.perf.{data,txt}
/tmp/milvus-sparse-fix-20260820/bf-repro-{dense,sparse}-perf-workload.log
```

## 11. IVF Sparse consumer 可行性微实验（2026-08-24）

### 11.1 动机与安全背景

Adaptive 输出 `ValidIdList`、Cardinal auto route 选择 IVF 时，当前原型会先把
list 转为 `BloomFlatValid`，随后 IVF 的 `/64` consumer 又把它当成 Dense bits
读取，最终因 `bits_ == nullptr` 崩溃。无论最终是否支持 Sparse IVF，这一输入都不能
再进入未声明 capability 的 consumer。

在直接将第一阶段收紧为 BF-only/`NotSupported` 前，先隔离 IVF filter consumer 做
一次受控实验，回答两个不同问题：

1. 当前 Bloom+Flat 逐 candidate membership 是否值得修通；
2. Sparse 是否存在更适合 IVF bucket 布局的专用消费方式。

### 11.2 方法与约束

固定同一组 internal valid IDs、同一组 64-row 对齐的 IVF bucket ranges、相同 bucket
probe 顺序和相同输出集合；单线程固定 CPU 2，Release `-O3 -march=native`，先 warmup
20 次，再以 `ABC/CBA` 交替顺序测 9 轮。比较：

| 路径 | 实际动作 |
|---|---|
| Dense `/64` | 调用生产 `FilterCheckerView::BitCompressBatch64` 扫描 probe ranges |
| Bloom+Flat | probe range 中每个 candidate 调用 exact `Contains` |
| Sorted intersection | 将 internal Sparse IDs 排序；每个 bucket 用 `lower_bound` 求交并直接枚举命中 IDs |

`N=100K/1M`，`V=50/500/1000/5000`；分别 probe 12.5% 和 100% buckets。三条路径
逐 ID 验证输出完全一致后才计时。consumer 和构造分别计时；排序构造使用随机 producer
order，避免把“输入已经有序”误当成免费。该实验不含 centroid search、distance、heap、
Milvus handoff，因此只能证明 consumer 的因果成本，不能直接声称 endpoint 收益。

### 11.3 Consumer 结果

下表列 1M 的完整结果；100K 的 Sorted intersection 相对 Dense 改善范围为
`-37.3%～-66.7%`，趋势一致。

| V | Probe rows | Dense `/64` | Bloom+Flat | Bloom delta | Sorted intersection | Sorted delta |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 125,504 | 8.170 us | 196.380 us | +2303.6% | 1.050 us | **-87.1%** |
| 500 | 125,504 | 8.936 us | 206.408 us | +2209.7% | 1.772 us | **-80.2%** |
| 1,000 | 125,504 | 9.802 us | 208.419 us | +2026.2% | 2.122 us | **-78.4%** |
| 5,000 | 125,504 | 13.546 us | 202.292 us | +1393.4% | 3.569 us | **-73.7%** |
| 50 | 1,000,000 | 66.239 us | 1,567.762 us | +2266.8% | 12.757 us | **-80.7%** |
| 500 | 1,000,000 | 72.282 us | 1,668.531 us | +2208.4% | 32.991 us | **-54.4%** |
| 1,000 | 1,000,000 | 79.724 us | 1,688.899 us | +2018.4% | 44.549 us | **-44.1%** |
| 5,000 | 1,000,000 | 129.066 us | 1,619.742 us | +1155.0% | 73.622 us | **-43.0%** |

代表点 `N=1M、V=1000、probe rows=125,504` 另用 `sudo perf stat` 分开采集
cycles/instructions，按一次 consumer 归一化：

| 路径 | cycles / consume | instructions / consume |
|---|---:|---:|
| Dense `/64` | 25,530 | 128,234 |
| Bloom+Flat | 544,374 | 2,435,433 |
| Sorted intersection | 5,471 | 28,978 |

cycles 与 instructions 均闭合 wall-time 方向，说明 Sorted improvement 不是 AB/BA
顺序或短计时噪声。

### 11.4 构造成本与当前决策

1M、随机 producer order 的单次表示构造成本如下：

| V | Dense materialization | Bloom+Flat build | Sorted-list build |
|---:|---:|---:|---:|
| 50 | 6.637 us | 0.295 us | 0.423 us |
| 500 | 7.013 us | 2.456 us | 8.272 us |
| 1,000 | 7.402 us | 4.947 us | 23.325 us |
| 5,000 | 14.364 us | 22.331 us | 184.548 us |

因此结论不是“Sparse IVF 已经值得直接落地”，而是：

- 当前 `BloomFlatValid -> IVF /64` 方向明确没有性能价值，也不能继续保留会崩溃的
  隐式适配；
- 不能据此断言 Sparse 只适合 BF。按 bucket range 直接消费 Sparse IDs 有明显的
  consumer 潜力；
- 是否有 endpoint 价值取决于 `V`、probe rows/buckets、ID 分组/排序成本和 distance
  占比。`V=1000、probe 12.5%` 中完整排序会吃掉 consumer 节省，而 `V=500` 或扫描
  更多 buckets 时仍有净空间；
- IVF 专用的 bucket-grouped Sparse prototype 已在 11.5 补测；它避免全局排序，按
  bucket 组织 producer IDs 后可直接把对应 span 交给已有 valid-ID distance scan。
  完成真实 IVF route 的结果闭环和 endpoint ABBA 后，再决定正式支持面。

在该决策完成前，不静默强制 BF，也不把本微实验数字作为 Milvus E2E 收益。

实验产物：

```text
Cardinal benchmark source:
  review-worktrees/cardinal/tools/sparse_ivf_consumer_benchmark.cpp
Raw AB/BA output:
  review-worktrees/cardinal/docs/research/results/sparse-ivf-consumer-micro-20260824-v2.csv
Non-multiplexed perf output:
  review-worktrees/cardinal/docs/research/results/sparse-ivf-consumer-perf-20260824.txt
Source SHA256 (benchmark):
  13d0493865efcfd24c7df2e06196bf717f8e9eb066eeed0c478605be0a2664bf
Binary SHA256:
  22a8249e04f11c16ba46d8ef0d666632829545444a568e6b8e81b420f50f2697
```

### 11.5 无序 Sparse list 与 bucket-grouped consumer（2026-08-24）

#### 设计原则

基础 Sparse list 的契约明确收紧为：**精确、可枚举、生命周期安全，但不要求有序**。
producer order 是合法顺序；不得为了尚未确定的下游 consumer 在 producer 中强制
排序。BF 直接枚举；IVF 若确认值得支持，由 IVF adapter 按需组织 bucket spans。

第一版 adapter 对每个 internal ID 在 IVF bucket offsets 上二分，虽然 consumer 很快，
但 `N=1M、V=1000、1024 buckets` 的分组构造达到 42.198 us，说明把 bucket 查找
结构完全留给每次 query 不合理。

第二版在 IVF index 上常驻 coarse bucket directory：每 256 个 internal rows 保存一个
32-bit bucket hint，query 时从 hint 开始向后校正边界。它适用于任意 bucket size，
不要求均匀分桶；`N=1M` 常驻开销 15,628 bytes，`N=10M` 约 156 KB。directory 在
index 生命周期构造，不计入 query adapter 构造；query adapter 只做两遍 O(V+B)：

```text
producer-order IDs
  -> bucket hint + exact boundary correction
  -> count/prefix offsets
  -> scatter IDs to flat bucket spans (bucket 内仍保持 producer order)
```

#### 1M 结果

同 11.2 的数据、probe trace、CPU、warmup 与 ABCD/DCBA 规则。`Total` 只合并兼容的
表示构造和 consumer，不包含 centroid、distance、heap 或 endpoint 固定成本。

| V | Probe rows | Dense consumer | Unsorted grouped consumer | Dense build | Group build | Dense total | Group total | Total delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 125,504 | 8.122 us | 0.214 us | 9.176 us | 1.240 us | 17.298 us | 1.454 us | **-91.6%** |
| 500 | 125,504 | 8.835 us | 0.360 us | 9.124 us | 3.443 us | 17.959 us | 3.803 us | **-78.8%** |
| 1,000 | 125,504 | 9.757 us | 0.537 us | 9.901 us | 6.025 us | 19.658 us | 6.562 us | **-66.6%** |
| 5,000 | 125,504 | 13.742 us | 1.457 us | 13.937 us | 25.969 us | 27.679 us | 27.426 us | -0.9% |
| 50 | 1,000,000 | 66.146 us | 1.628 us | 9.160 us | 1.248 us | 75.306 us | 2.876 us | **-96.2%** |
| 500 | 1,000,000 | 72.684 us | 3.115 us | 9.447 us | 3.544 us | 82.131 us | 6.659 us | **-91.9%** |
| 1,000 | 1,000,000 | 79.465 us | 4.561 us | 9.068 us | 5.974 us | 88.533 us | 10.535 us | **-88.1%** |
| 5,000 | 1,000,000 | 128.958 us | 11.563 us | 8.088 us | 26.372 us | 137.046 us | 37.935 us | **-72.3%** |

`N=100K` 暴露了必要边界：只 probe 12.5% 时，V=500/1000 的 total 分别为
`+53.9%/+137.2%`；probe 100% 时则为 `-57.5%/-33.2%`。因此收益仍取决于 N、V
和 probe work，不能仅凭 Sparse 表示就假设 IVF endpoint 必然更快。

代表点 `N=1M、V=1000、probe rows=125,504` 的 non-multiplexed PMU：

| 路径 | cycles / consume | instructions / consume | task-clock / consume |
|---|---:|---:|---:|
| Dense `/64` | 25,348 | 128,017 | 9.753 us |
| Unsorted bucket grouped | 1,368 | 6,366 | 0.526 us |

PMU 与 steady-clock 的 consumer 降幅闭合。当前结论是：不应给 Sparse list 增加有序
要求，也不能再使用 Bloom+Flat 逐 candidate 作为 IVF 适配；无序 bucket-grouped
值得进入真实 Cardinal IVF prototype。下一层必须复用真实 bucket offsets/scan order，
验证 topK/distance、max_codes/early-stop 语义和完整 endpoint ABBA，才能决定产品支持。

原始产物：

```text
Naive binary bucket grouping:
  review-worktrees/cardinal/docs/research/results/sparse-ivf-consumer-micro-20260824-v3-unsorted.csv
Coarse-directory grouping:
  review-worktrees/cardinal/docs/research/results/sparse-ivf-consumer-micro-20260824-v4-unsorted-directory.csv
PMU:
  review-worktrees/cardinal/docs/research/results/sparse-ivf-consumer-perf-20260824-v4-unsorted-directory.txt
Benchmark source SHA256:
  2f8aa5ba8154ee90862753e547dff04ea7176b4c054f557afecb0c89c51b5d9a
Binary SHA256:
  811ad1503d0844917d73f460becaa211713920a95fda5b891b80e0aa8b711123
```

### 11.6 真实 Milvus/Cardinal IVF E2E（2026-08-24）

#### 实现与单一变量

本轮把 11.5 的无序 bucket-grouped prototype 接入真实 Cardinal IVF。基础 Sparse
payload 仍是不要求有序的 canonical accepted-ID list；IVF consumer 在 query 内完成
两遍 `O(V+B)` 的 count/prefix/scatter，并将每个 bucket 的 span 交给既有
`ScanRangeByValidIds`。没有构造 Dense，也没有为了 IVF 全局排序 Sparse IDs。

为了避免每个 ID 在 `B` 个 bucket offset 上二分，IVF index 常驻 `4*B` 个 32-bit
bucket hint。其空间为 `O(B)`、与 `N` 无关；constructor/load 后随 `initChunkInfo()`
重建。query lookup 从按 internal-ID 比例定位的 hint 开始，仅向前修正到精确 bucket。
Graph 仍只在实际 Graph searcher 边界临时创建 point-membership view，退出后恢复
canonical list，因而 Graph -> IVF/BF fallback 不会丢失可枚举表示。

对照只改变 `filter_result_representation=dense/sparse`。collection、index、查询及顺序、
线程/并发、NQ、topK、ef、warmup 和二进制保持不变；两侧均由 Cardinal auto selector
选择 IVF，而不是通过 Sparse 强制改路由。

#### 正式配置

| 项 | 配置 |
|---|---|
| dataset | 1,000,000 rows, 128D FP32 synthetic |
| segment/index | 1 个 sealed segment；`CARDINAL_TIERED` |
| predicate | `a < 1000 and b < 500000` |
| final cardinality | 约 500 accepted IDs（`V/N ~= 0.05%`） |
| vector search | L2, topK=10, ef=64, auto route |
| load shape | NQ=1, concurrency=1, 固定 50 queries |
| correctness | 计时前逐 query 比较 topK `(ID,distance)` |
| timing | 每模式 10 warmups；12 个 ABBA windows；每 slot 50 queries |
| timed samples | 每模式 1,200 requests |

动态库曾因 `LD_LIBRARY_PATH` 中旧 `internal/core/output/lib` 优先而错误加载 8 月 20 日
的 Knowhere/Cardinal，触发与本实现无关的 SIGSEGV。正式数据采集前通过
`/proc/<pid>/maps` 确认加载的是当前 `cmake_build` 下的 `libmilvus_core.so`、
`libknowhere.so` 和 `libcardinalv2.so`。

#### Endpoint 结果

计时前的 topK/distance correctness closure 全部通过。

| 指标 | Dense | Sparse bucket-grouped | Sparse 相对 Dense |
|---|---:|---:|---:|
| mean | 4.8161 ms | 2.9792 ms | **-38.14%** |
| median | 4.8101 ms | 2.7769 ms | **-42.27%** |
| P90 | 4.9694 ms | 4.3142 ms | **-13.18%** |
| timed requests | 1,200 | 1,200 | - |
| Sparse 更快 paired windows | - | 12 / 12 | - |

这组结果证明：在本测试点，IVF 专用的无序 Sparse consumer 不仅有微基准潜力，
也能转化成完整 Milvus endpoint 收益。P90 的改善明显小于 mean/median，不能据此把
固定 endpoint、线程调度和其他未分类 residual 归入 IVF consumer；本节不建立可复用
的 per-ID 成本模型。

#### Route 与访问模式闭环

两种模式分别在同一 steady workload 下执行 6,000 requests 并用 `sudo perf record`
采样。采样期间的 endpoint 分布仍与 ABBA 同方向：Dense mean/median/P90 为
`5.0883/5.0323/5.4613 ms`，Sparse 为 `3.1680/2.9608/4.5303 ms`。

| 模式 | 实际 route | filter consumer 的可见大块 | 结论 |
|---|---|---|---|
| Dense | `IvfSearcherImpl -> IvfSearchImpl` | `ScanRangeFilter` 约 20.1%；其中代表性 `BitCompressBatch64` stack 约 18.6% | IVF probe range 上按 `/64` 枚举 Dense valid IDs 是主要 CPU 热点 |
| Sparse | `IvfSearcherImpl -> IvfSearchImpl` | `ScanRangeByValidIdsBatch4` 代表性 stack 约 1.3%；未见 Dense `BitCompressBatch64` 热点 | query-local grouping 后只枚举 bucket 内 accepted IDs |

这里的百分比是整台 Milvus 进程的 perf sample share，且调用栈 children 会嵌套，不能
彼此相加成 endpoint 时间分解；它们只用于证明 route 和 access pattern。因果链为：

```text
相同 producer / collection / queries / auto-IVF route
  Dense  -> IVF -> ScanRangeFilter -> BitCompressBatch64
  Sparse -> IVF -> bucket-grouped spans -> ScanRangeByValidIdsBatch4
```

因此本轮收益可以归因于 IVF filter enumeration 的访问模式改变，而不是路由变化；
但结果目前只覆盖 `1M x 128D, V ~= 500, Tiered auto-IVF, NQ1/C1`，不能外推阈值、
10M/768D 或高并发。

#### IVF 与 BF 的 2x2 补充对照

为判断 centroid pruning 是否与 Sparse 形成额外组合收益，在同一当前二进制、collection、
predicate 和 50 queries 上补测 `Dense/Sparse x auto-IVF/explicit-BF`。每个模式预热
10 queries，按正序/逆序交替 6 windows，每模式 300 timed requests；explicit-BF 的
top10 作为 reference，四种模式的 50/50 queries 均与 reference 完全一致。

| 表示 | auto-IVF mean | explicit-BF mean | IVF 相对 BF |
|---|---:|---:|---:|
| Dense | 4.8617 ms | 4.7487 ms | +2.38% |
| Sparse | 2.9151 ms | 2.8401 ms | +2.64% |

从另一个方向看，Sparse 在 IVF 内改善 40.04%，在 BF 内改善 40.19%，幅度几乎相同。
因此本 case 的收益来自 Sparse valid-ID enumeration，而不是 Sparse 与 IVF centroid
pruning 的协同；IVF 还多付了 coarse centroid search/grouped-bucket 调度成本。50 条
query 与 BF top10 完全一致也表明本点 IVF 没有观察到可见的结果剪枝收益，但要严格
证明扫描了多少 bucket/valid IDs，仍需增加请求级 bucket/distance-attempt counter。

当前更合理的 selector 行为是：当 `V` 已足够小、BF 可直接枚举全部 Sparse IDs 时，
优先 BF；只有当 IVF 实际 probe 的 valid IDs 显著小于 `V`，并且节省的 distance work
超过 centroid/grouping 固定成本时，Sparse-IVF 才可能有组合价值。

原始产物：

```text
ABBA:
  /tmp/milvus-sparse-ivf-abba-1m128-20260824.log
Dense perf/workload:
  /tmp/milvus-sparse-ivf-dense-1m128.perf.{data,txt}
  /tmp/milvus-sparse-ivf-dense-perf-workload.log
Sparse perf/workload:
  /tmp/milvus-sparse-ivf-sparse-1m128.perf.{data,txt}
  /tmp/milvus-sparse-ivf-sparse-perf-workload.log
Harness:
  scripts/cardinal_sparse_compound_filter_e2e.py
```

#### 落地前剩余验证

1. directory：uniform/skewed/empty buckets、越界 ID，以及 load 后重建；逐 ID 与
   `upper_bound(offset)` 的精确 bucket 结果对照。
2. grouped plan：输出 multiset 与输入一致，明确 duplicate 语义，并覆盖无序输入。
3. IVF：iterator、chunked initial/final bucket expansion、`max_codes`/early-stop，以及
   Graph primary -> IVF fallback 的 topK/distance 等价。
4. 性能扩展：先做 `V=500/1000/5000` sweep，再做 1M x 768D 和 10M x 128D；若采
   cycles/instructions/task-clock，分别采集，禁止 PMU multiplex 后直接比较。

## 12. 三类 auto consumer 的统一验证计划（2026-08-24）

### 12.1 目标与表示契约

Milvus predicate 层只交付一份 query-owned、精确、生命周期安全、无序且可枚举的
Sparse valid-ID list。表示决策与搜索路由解耦：Dense/Sparse 两种模式必须向 Cardinal
auto selector 提供相同的 `N`、filtered count、topK、search limit 和 index/build
configuration；Sparse 不能隐式强制 BF，也不因 consumer 改写逻辑 cardinality。

auto selector 完成后，consumer 在实际调用边界按 route 适配：

```text
canonical Sparse valid-ID list
  -> BF:    直接枚举 V 个 IDs
  -> IVF:   index-persistent O(B) bucket hints
            + query-local O(V+B) count/prefix/scatter
            + ScanRangeByValidIds
  -> Graph: query-local exact Blocked Bloom + FlatHash membership
```

Graph membership 采用 Bloom negative reject + FlatHash exact confirmation，不允许近似
filter 造成 SQL filter violation。当前阶段只研究 valid-ID polarity；invalid-ID Sparse、
OR/NOT 的双极性集合传播及 IVF exclude-list gap scan 不进入本阶段。

### 12.2 Auto 与 backend 边界

正式性能对照禁止用 `index_algo=BF/IVF/Graph` 强制 route。每个 Dense/Sparse paired
case 必须记录实际 `bf_search_cnt/ivf_search_cnt/graph_search_cnt`；若两侧 route 不同，
该 case 只能作为 selector 缺陷，不能作为表示性能结论。

本机 Milvus `CARDINAL_TIERED` 对应 ObjectStore selector，只会自然选择 IVF 或 BF，
不会选择 Graph。因此测试分两层：

| 层级 | 目的 | 可覆盖 route |
|---|---|---|
| Cardinal/vecTool direct consumer endpoint | 固定相同 dataset/query/config，建立 Memory、Disk、Tiered 的完整 route map，并隔离分析 Cardinal consumer | BF / IVF / Graph |
| Milvus native E2E | 复核 predicate -> payload -> Cardinal endpoint 的生产链路 | 当前本机 Tiered 的 BF / IVF |

Graph 的 Milvus native auto E2E 需要正式接入 Memory/Disk Cardinal index；不能用强制
Graph 结果冒充 auto route。

正式性能结论统一使用以下两类端到端指标之一：

1. Milvus client request latency：覆盖 predicate evaluation、Dense/Sparse producer、payload
   handoff、segment dispatch 和 vector search；
2. 固定并发、固定持续时间下的 Milvus QPS，并同时报告 latency distribution。

Cardinal/vecTool direct harness 的 `Index::Search` latency/throughput 只作为 consumer endpoint
证据，用于隔离 route 和表示成本，不能替代 Milvus E2E 结论。`payload_us`、`b_read`、
route/access counters、perf hotspot 等阶段指标只用于正确性闭环和性能归因，不再作为方案
收益基准。

所有性能结果必须同时展示 `N`、`V` 和执行单元数量。Milvus E2E 展示 collection 总行数、
sealed/growing segment 数及各 segment 行数分布；Cardinal direct harness 展示 Cardinal index
数量，并将 Milvus segment 标为 N/A，禁止把单 Cardinal index 直接表述成真实单 segment。

### 12.3 正确性与访问计数闭环

每个 case 先做 correctness，再允许进入 timing：

1. Dense/Sparse predicate 最终 cardinality、universe 和 visibility 结果一致；
2. auto selected route 一致；
3. topK IDs/order 一致，distance 在明确浮点容差内；
4. 所有结果满足 scalar filter，且不存在 delete/MVCC/TTL/nullable violation；
5. 记录与 route 对应的访问计数：
   - BF：distance attempts；
   - IVF：coarse buckets、scanned buckets、valid-ID distance attempts；
   - Graph：membership probes、distance attempts；
   - 三路共同记录 representation prepare/build time。

显式 BF 或离线 exact distance 只允许作为小样本 correctness oracle，不参与 auto 性能
统计。

### 12.4 第一阶段：1M x 128D route discovery

第一阶段只建立 route/access-count map，不输出正式收益结论。固定 synthetic FP32、L2、
topK10、NQ1/C1、单线程查询、同一 index binary 和固定 query vectors；scalar 值在每个
Cardinal index 的完整 row universe 内独立随机，避免 valid IDs 集中到少数 IVF bucket。
该 direct harness 不经过 Milvus segment 层。

| 变量 | discovery 取值 |
|---|---|
| rows / dim | 1M / 128D |
| Cardinal index / Milvus segment | 1 / N/A |
| valid cardinality V | 500、1K、5K、10K、1%、5%、10%、50% |
| backend | Memory、Disk、Tiered/ObjectStore |
| representation | Dense、Sparse valid-ID list |
| route | auto only |
| query | 固定 50 queries；每模式先 warmup 10 |

对每个点输出：Dense/Sparse correctness、两侧实际 route、route access counts、payload
prepare time，以及仅供 smoke 的 endpoint latency。若某 backend 的相邻 V 点发生 route
切换，在阈值两侧增加点位，确定自然 route regime。

### 12.5 第二阶段：consumer ABBA 与 Milvus E2E 扩展矩阵

从第一阶段每个自然 route regime 选代表点，执行固定 query、12-window ABBA、每 slot
50 queries。Cardinal direct 层报告 consumer endpoint mean/median/P90、paired-window
胜率并保留 route/access-count closure，只用于隔离归因。最终采用结论必须在 Milvus native
层报告 client request E2E latency，或 completed requests/elapsed QPS 及对应 latency
distribution。

扩展顺序：

1. 1M x 128D：三种 route 的代表点；
2. 10M x 128D：验证 N、Dense cache footprint 与常驻 Sparse 结构；
3. 1M x 768D Cohere：验证 distance 占比上升后的收益稀释；
4. C1 全矩阵后，仅对代表点补 C60 closed-loop throughput。

PMU 的 cycles、instructions、task-clock 分开采集；不使用 multiplex 后的读数建立单位
成本。不同 route、backend 或向量维度不得聚合成单一收益百分比。

### 12.6 第一阶段执行结果（2026-08-24）

已使用与当前 Milvus `cmake_build/conan` 相同依赖重新构建 Cardinal Release harness，
并完成 1M x 128D 主矩阵。固定 FP32、L2、topK=10、search limit=64、NQ=1/C=1、
50 个固定 query、warmup 10、auto route；每个点在计时前逐 query 校验 Dense/Sparse
topK `(ID, distance)` 和实际 BF/IVF/Graph route。21 个主矩阵点全部通过。

| Valid V | Valid ratio | Memory auto | Disk auto | ObjectStore/Tiered auto |
|---:|---:|---|---|---|
| 500 | 0.05% | BF | BF | IVF |
| 1,000 | 0.10% | BF | BF | IVF |
| 5,000 | 0.50% | BF | BF | IVF |
| 10,000 | 1.00% | BF | BF | IVF |
| 50,000 | 5.00% | IVF | IVF | IVF |
| 100,000 | 10.00% | Graph | IVF | IVF |
| 500,000 | 50.00% | Graph | Graph | IVF |

阈值补点结果：

| Backend | 实测自然切换区间 | 与 selector 实现的对应关系 |
|---|---|---|
| Memory | BF at 1.50% valid；IVF at 1.75%；IVF at 7.75%；Graph at 8.00% | BF 边界来自 98.5% filtered；IVF/Graph 边界同时受 `ShouldSwitchMemIVF`、N、level 和 search limit 影响 |
| Disk | BF at 1.50% valid；IVF at 1.75%；IVF at 12.50%；Graph at 15.00% | topK=10 时 Disk IVF/Graph 公式给出的 valid 边界约为 14.9767% |
| ObjectStore/Tiered | 1M 主矩阵全部 IVF；100K smoke 主矩阵全部 BF | selector 由 `ef >= V`、`topK >= V` 或 `max_codes >= V` 决定，故 route 不只取决于 valid ratio |

ObjectStore direct harness 必须使用其支持的量化和生命周期。`None`-quantized chunked BF
没有注册 searcher；直接搜索 build-time chunked storage 也不是合法用法。本轮改为现有
level-0 配置的 RBQ2，并执行 caching-layer 初始化和 `Build -> Save -> Load -> Prepare
-> Search`，随后 100K/1M correctness 均通过。Memory/Disk 继续使用 unquantized
FP32、Hybrid build。

访问计数的当前闭环状态：

- BF：Dense/Sparse route 相同，`bf_distance_attempt_cnt` 完全相同；
- Graph：route 相同，现有 `search_compute_cnt` 相同；
- IVF：route/topK 相同，但 `search_compute_cnt` 仍有小幅差异，尚未拆出 coarse buckets、
  scanned buckets 和 valid distance attempts，因此不能把 endpoint delta 单独归因于
  Sparse membership；
- 当前 `sparse_payload_us` 包含 synthetic `MakeValidIds` 对完整 universe 的 shuffle，
  不是 Milvus Sparse producer 成本，禁止用于 Dense/Sparse producer 结论。

因此本阶段输出的是 **route/correctness map**。harness 打印的 `smoke_delta_pct` 不进入
正式性能结论；第二阶段 ABBA 的门槛仍是先补 IVF scanned-bucket/valid-distance 和
Graph membership-probe/distance 细分计数，并证明 paired case 的算法访问量可解释。

复现命令形态：

```bash
build/route-discovery-milvus/tools/sparse_index_type_benchmark \
  --index-type memory|disk|object_store --index-algo auto \
  --rows 1000000 --dim 128 \
  --valid-values 500,1000,5000,10000,50000,100000,500000 \
  --queries 50 --warmup 10 --windows 1 \
  --index-prefix /tmp/cardinal-route-1m-<backend>
```

### 12.7 访问计数闭环与 ABBA 准入（完成，2026-08-25）

第一阶段已经完成 auto-route/correctness map，但现有通用 `search_compute_cnt` 不能区分
Graph membership、Graph distance、IVF bucket traversal 与 IVF distance，因此还不足以解释
Dense/Sparse endpoint 差异。consumer ABBA 前先补齐生产路径访问计数，且计数本身不参与
route 选择或结果计算。

| Route | 已有闭环 | 本阶段补充 | 计数语义 |
|---|---|---|---|
| BF | `bf_distance_attempt_cnt` | 保持并回归 | 实际进入 distance computer 的候选数 |
| IVF | 通用 `search_compute_cnt` | scanned buckets、valid distance attempts；必要时补 coarse buckets | 只在实际进入 bucket scan/执行 distance 的位置计数，不以 selector 预估代替 |
| Graph | 通用 `search_compute_cnt` | membership probes、distance attempts | 只在实际 point filter check/执行 distance 的位置计数 |

执行顺序：

1. 在 `CardinalSearchRecord/Statistics` 增加上述独立计数并接入聚合；
2. 在 IVF/Graph 的 Dense 与 Sparse 公共执行点插桩，保证两种表示使用相同计数语义；
3. 扩展 `sparse_index_type_benchmark` 输出；
4. 先跑 100K smoke，再在 1M x 128D 上验证 BF、IVF、Graph 的代表点；要求 route、
   topK/distance 正确且访问计数非零、可解释；
5. 只有闭环成立后，才进入 12-window ABBA。若访问数量本身不同，则先解释算法路径差异，
   不把 endpoint delta 直接归因为 Dense/Sparse 单点判定成本。

实验固定项继续沿用 12.4：相同 dataset/index/query 顺序、NQ1/C1、topK10、单线程、
相同 cache 状态与 Release binary。计数闭环是因果归因准入项，不替代后续独立的 endpoint
latency/QPS 验证。

#### 12.7.1 实现与 smoke 结果

已在 `CardinalSearchRecord/Statistics` 和 production search path 增加并聚合以下计数：

- IVF：`ivf_coarse_bucket_cnt`、`ivf_scanned_bucket_cnt`、
  `ivf_valid_distance_attempt_cnt`；
- Graph：`graph_membership_probe_cnt`、`graph_distance_attempt_cnt`；
- BF 继续使用已有 `bf_distance_attempt_cnt`。

Release harness 完整编译通过。100K x 128D Memory smoke 覆盖了 BF/IVF/Graph，所有点
Dense/Sparse topK、distance 和 auto route 一致；只在当前 route 上出现对应计数。代表性
结果如下（计数为一个 ABBA smoke window 的合计，endpoint delta 仍不作为正式结论）：

| Route | V | Dense access | Sparse access | 闭环判断 |
|---|---:|---:|---:|---|
| BF | 100 | distance 1,000 | distance 1,000 | 相同 |
| IVF | 5K | scanned buckets 2,730；distance 18,134 | scanned buckets 2,708；distance 18,134 | distance work 闭合；Sparse 省去 22 个无 valid ID 的空 bucket scan 调用 |
| Graph | 50K | membership 33,520；distance 20,034 | membership 33,520；distance 20,034 | 完全相同 |

#### 12.7.2 1M x 128D 三 backend 代表点

固定 synthetic FP32、L2、topK10、NQ1/C1、auto route、相同 query/order；本轮只验证
访问计数准入，窗口较短，不报告 smoke latency 为正式性能结果。

| Backend | Natural route | V | Dense access | Sparse access | 准入判断 |
|---|---|---:|---:|---:|---|
| Memory | BF | 1K | distance 20,000 | distance 20,000 | 可进入 ABBA |
| Memory | IVF | 50K | coarse 10,240；scanned 1,080；distance 59,304 | coarse 10,240；scanned 1,080；distance 59,304 | 完全闭合，可进入 ABBA |
| Memory | Graph | 100K | membership 347,022；distance 212,162 | membership 347,022；distance 212,162 | 可进入 ABBA |
| Disk | BF | 1K | distance 10,000 | distance 10,000 | 可进入 ABBA |
| Disk | IVF | 50K | coarse 10,240；scanned 1,080；distance 59,304 | coarse 10,240；scanned 1,080；distance 59,304 | 完全闭合，可进入 ABBA |
| Disk | Graph | 500K | membership 37,188；distance 20,602 | membership 37,188；distance 20,602 | 可进入 ABBA |
| ObjectStore | IVF | 1K | coarse 230；scanned 170；distance 7,476 | coarse 230；scanned 170；distance 7,476 | 完全闭合，可进入 ABBA |
| ObjectStore | IVF | 50K | coarse 230；scanned 50；distance 108,350 | coarse 230；scanned 50；distance 108,350 | 完全闭合，可进入 ABBA |

此前 IVF distance raw counter 的差异已确认为 Dense 统计 bug，而非实际 candidate 数差异：
Dense `ScanRangeFilterBatch4/Default` 已在尾部循环中执行完 `vs_counter` 个 distance，但只把
`vs_counter_back` 累加到 `search_compute_cnt`，因而按 bucket 漏记预取 ring buffer 的最后
一小批。现已将两处统计修正为 `vs_counter` 并完成 Release 全量重编。100K 与上述 1M
三 backend 复测均确认 Dense/Sparse distance attempts 完全一致。

`ivf_scanned_bucket_cnt` 的语义需区分表示：Dense 对选中的 coarse bucket 均进入
`ScanRangeFilter`，即使 bucket 与 filter 无交集也计数；Sparse 仅对含 valid ID 的 bucket
调用 `ScanRangeByValidIds`。所以准入闭环要求 coarse bucket work 和 distance attempts
一致；Sparse scanned buckets 更少时，差值是省去的空 bucket 调用，不是 route confound。

运行时需使用该 build 精确的 Conan revision；不可把所有 Conan `p/lib` 目录全量加入
`LD_LIBRARY_PATH`，否则会混入另一套 `milvus-common/grpc` ABI。当前可复现的最小前缀为
本 build 对应的 grpc、gflags、OpenSSL、milvus-common 和 Cardinal build lib 目录。

### 12.8 Cardinal direct consumer ABBA：访问量闭合的 BF/Graph（2026-08-25）

在 12.7 访问计数闭合后，对 Memory/Disk 的自然 BF 与自然 Graph 代表点执行 ABBA。
固定 1M x 128D synthetic FP32、L2、topK10、NQ1/C1、50 个固定 query、warmup 10、
12 个 window；每个 window 顺序为 Dense -> Sparse -> Sparse -> Dense，每个 slot 完整执行
50 queries。表中 median/P90 是 **12 个 paired window mean** 的分位数，不是单请求分位数。
每个 query 在计时前已完成 topK/distance/route correctness closure。

本节是 Cardinal direct consumer endpoint，不是 Milvus E2E。所有 case 均为 `N=1M`、
`Cardinal index count=1`、`Milvus segment=N/A`；producer、handoff 和 segment dispatch 不在
计时范围内，结果只用于 consumer 归因。

| Backend / natural route | V | Metric | Dense | Sparse | Delta / win |
|---|---:|---|---:|---:|---:|
| Memory / BF | 1K | mean | 0.1625 ms | 0.0655 ms | -59.71% |
|  |  | median | 0.1642 ms | 0.0654 ms | -60.17% |
|  |  | P90 | 0.1653 ms | 0.0673 ms | -59.29% |
|  |  | paired-window wins | - | - | 12/12 |
| Disk / BF | 1K | mean | 0.1577 ms | 0.0631 ms | -59.98% |
|  |  | median | 0.1576 ms | 0.0627 ms | -60.22% |
|  |  | P90 | 0.1612 ms | 0.0648 ms | -59.80% |
|  |  | paired-window wins | - | - | 12/12 |
| Memory / Graph | 100K | mean | 1.5941 ms | 3.1392 ms | +96.93% |
|  |  | median | 1.5865 ms | 3.1189 ms | +96.65% |
|  |  | P90 | 1.6278 ms | 3.2675 ms | +100.73% |
|  |  | paired-window wins | - | - | 0/12 |
| Disk / Graph | 500K | mean | 0.4229 ms | 9.5879 ms | +2167.32% |
|  |  | median | 0.4228 ms | 9.6075 ms | +2172.35% |
|  |  | P90 | 0.4356 ms | 9.9630 ms | +2187.19% |
|  |  | paired-window wins | - | - | 0/12 |

访问计数在每个 paired case 内完全相同：

| Backend / route | Dense = Sparse access count |
|---|---:|
| Memory / BF | 1,200,000 BF distance attempts |
| Disk / BF | 1,200,000 BF distance attempts |
| Memory / Graph | 20,820,768 membership probes；12,687,192 distance attempts |
| Disk / Graph | 4,570,392 membership probes；2,528,520 distance attempts |

因此 BF 的收益可以归因于相同 distance work 之外，Sparse 枚举避免 Dense universe scan；
Graph 的劣化也不能归因于 route/probe/distance 数变化，而来自高 V 下的 Sparse membership
运行成本，其中包含每请求 valid-ID list 到 Bloom+Flat 的适配。Memory V=100K、Disk
V=500K 均远高于当前拟议的 tiny-Sparse 阈值，这些点的作用是证明 **Sparse 不能在高 V
时无条件替代 Dense**，而不是建议在这些点启用 Sparse。

第一批正式结果支持以下阶段性决策：

1. BF + tiny valid set 是明确有效区间；
2. **Graph 当前阶段拒绝 Sparse**：自然 Graph route 的 V 已过大，Memory/Disk 分别稳定
   劣化约 97%/2167%；不继续扩展 Graph Sparse 矩阵，Graph 输入保持 Dense；
3. IVF Dense per-bucket distance counter 的尾部漏记已修正，计数复测和 consumer ABBA 见
   12.9；
4. 不继续投入 Graph Sparse；IVF 必须按 tiny-V 与中等-V 两个 regime 分开决策。

### 12.9 IVF 计数修复后的 Cardinal direct consumer ABBA（2026-08-25）

固定 1M x 128D synthetic FP32、L2、topK10、NQ1/C1、50 个固定 query、warmup 10、
12 个 paired window；每个 window 为 Dense -> Sparse -> Sparse -> Dense。使用 auto route，
所有点自然选择 IVF，且每个 paired case 的 topK、distance、route、coarse bucket work 和
distance attempts 完全一致。median/P90 仍是 12 个 paired-window mean 的分位数。

本节计时覆盖 `Index::Search`，包含 IVF query 内将 canonical unordered Sparse IDs 组织为
bucket-grouped spans 的 consumer 成本；不包含 producer 构造 Dense/Sparse payload 的一次性
成本。ObjectStore harness 通过 Cardinal ObjectStore selector、序列化后 local stream/mmap
加载来验证该 index regime，不包含真实 S3 网络延迟。

| N | Cardinal index count | Milvus segment count | Segment row distribution |
|---:|---:|---|---|
| 1,000,000 | 1 | N/A | N/A；本 harness 不经过 Milvus segment 层 |

| Backend / natural route | V | Metric | Dense | Sparse | Delta / win |
|---|---:|---|---:|---:|---:|
| ObjectStore / IVF | 1K | mean | 0.8368 ms | 0.0809 ms | **-90.34%** |
|  |  | median | 0.8371 ms | 0.0804 ms | -90.39% |
|  |  | P90 | 0.8411 ms | 0.0832 ms | -90.11% |
|  |  | paired-window wins | - | - | **12/12** |
| Memory / IVF | 50K | mean | 0.8557 ms | 1.5300 ms | **+78.81%** |
|  |  | median | 0.8547 ms | 1.5251 ms | +78.44% |
|  |  | P90 | 0.8631 ms | 1.5866 ms | +83.83% |
|  |  | paired-window wins | - | - | **0/12** |
| Disk / IVF | 50K | mean | 0.8644 ms | 1.6495 ms | **+90.82%** |
|  |  | median | 0.8625 ms | 1.6284 ms | +88.80% |
|  |  | P90 | 0.8841 ms | 1.7896 ms | +102.42% |
|  |  | paired-window wins | - | - | **0/12** |

访问计数闭环：

| Backend / V | Dense = Sparse coarse buckets | Dense = Sparse distance attempts |
|---|---:|---:|
| ObjectStore / 1K | 27,600 | 889,272 |
| Memory / 50K | 1,228,800 | 7,123,944 |
| Disk / 50K | 1,228,800 | 7,123,944 |

该 consumer endpoint 结果不能聚合成单一“IVF Sparse 收益”，也不能单独作为落地收益：

- **tiny V（ObjectStore V=1K）**：Sparse 大幅避免 Dense bucket-range bitset traversal，在
  相同 IVF distance work 下稳定降低约 90% search latency；
- **中等 V（Memory/Disk V=50K）**：每请求 O(V) 的 bucket directory lookup、计数/分组及
  grouped-span materialization 已超过 Dense `/64` 扫描成本，稳定劣化约 79%～91%；
- 因而 IVF 支持在 tiny-Sparse consumer 中有潜力，但是否端到端启用仍需 Milvus E2E/QPS
  覆盖 producer 和 handoff 后验证；不能仅依据本节打点或在所有 Sparse payload 上无条件使用。

## 13. Milvus E2E 固定阈值验收计划（2026-08-25）

### 13.1 目标与准入结论

最终决策只采用 Milvus client 完整请求的 E2E latency，或固定并发下的 QPS 与对应
latency distribution。Cardinal standalone、producer timer、route/access counter 和 perf
仅用于正确性闭环与失败归因，不作为阈值收益结论。

本轮验证候选固定阈值 `T=1000`。产品策略暂只依赖每 segment 的绝对 accepted-ID
cardinality `V_i`，不依赖 `V_i/N_i`；报告仍记录 ratio。实践边界假设为每个 segment
`N_i >= 50K`，因此 50K 是最小且最容易被固定开销稀释的验收点。

### 13.2 两条模式

| 模式 | 配置 | 行为 |
|---|---|---|
| Dense baseline | `enableSparseFilterResult=false` 或请求 `dense` | 从 predicate 到 search 全程保持既有 Dense 路径，不创建 Sparse list |
| Adaptive | `enableSparseFilterResult=true`，请求 `adaptive`，`T=1000` | 每个支持的 producer 先尝试 Sparse；第 `T+1` 个命中时单趟切 Dense、回填已保存 IDs 并继续剩余扫描 |

`sparseEnable` 只控制过滤执行器能否产生 Sparse。BF、IVF、Graph 对实际 Dense/Sparse
payload 的消费能力独立于该开关；表示不得强制或改写 Cardinal auto route。第一阶段 OR
仍保留 Dense；legacy raw BF 只在其既有兼容边界 materialize Dense。

### 13.3 首轮单 sealed-segment 矩阵

固定相同 Milvus binary、collection/index、query vectors/order、topK、NQ、concurrency、
cache/compaction 状态和 auto route。每个 case 计时前闭合 Dense/Adaptive topK、distance、
实际最终表示、route 和 segment 数。

| 每 segment N | V | 目的 |
|---:|---:|---|
| 50K | 64、100、500、999、1000 | 最小实践 N 下验证 threshold 内是否存在稳定 E2E/QPS 劣化 |
| 50K | 1001、5000 | 验证第 T+1 个立即切 Dense，fallback 与 baseline 等价且开销落在噪声内 |
| 100K | 64、500、1000、1001 | 验证 N 增大后的固定阈值行为 |
| 1M | 64、500、1000、1001 | 复现历史 V=64 反例与 V≈500 收益，并在最新实现上重测 |

首轮以 128D FP32 为固定向量配置。通过后补 1M x 768D，以及 4x50K/8x50K 均匀
sealed segments；多 segment 报告每个 segment 的 `N_i/V_i/representation/route`，并覆盖
部分 segment Sparse、部分 segment threshold-Dense 的混合情况。

### 13.4 统计与决策

- C1：30 秒 warmup 后采集完整 client request latency，Dense/Adaptive 使用 AB/BA 对称顺序；
- throughput：代表点补固定并发 QPS，并同时报告 mean/median/P90；
- 每张结果表必须展示 N、V、V/N、sealed/growing segment 数与行数分布、实际 route、
  completed requests、elapsed time；
- `V<=1000` 若存在可复现 mean/QPS 劣化，则先降低阈值或削减固定成本，不引入 ratio
  规则；`V>1000` 必须闭合为 Dense fallback，并与 baseline 落在实验噪声范围；
- 历史 Cardinal standalone V=50K 只证明无阈值 Sparse 不安全，不参与本轮 T=1000
  的采用判断。

### 13.5 首轮单 segment Milvus E2E 结果（2026-08-25）

本节只报告 Milvus client 完整 `search` 请求。固定 128D FP32 synthetic、L2、topK10、
NQ1/C1、50 个固定 query、每种表示先做 10 次并继续 warmup 30 秒；正式阶段为 12 个
`Dense -> Sparse -> Sparse -> Dense` window，每个 slot 重放全部 50 queries，因此每种表示
均有 1,200 个 timed requests。每个 case 复用同一 collection、index 和 query 顺序，计时前
检查前 10 个 query 的 topK/distance 完全一致。所有 collection 均为一个 loaded sealed
segment，实际行数分别为 50K、100K、1M；没有 scalar index，A/B 为确定性的随机
permutation，谓词为 `a < V AND b < N`，所以最终每 segment 精确留下 V 行。

route preflight 不进入计时：每种表示另发 10 个请求，用 Prometheus
`bf_search_cnt/ivf_search_cnt/graph_search_cnt` 的 sum delta 验证实际 auto route，并用
`internal_core_adaptive_filter_output_total` 验证最终表示。表中 latency delta 为
`(Sparse - Dense) / Dense`，负数表示 Sparse 更快；QPS delta 为
`(Sparse / Dense) - 1`。

#### 13.5.1 N=50K，单 sealed segment

该 collection 的所有点均自然选择 BF。`V<=1000` 的 Sparse preflight 每 10 请求记录
10 个 Sparse output；`V=1001/5000` 则记录 10 个 `dense_threshold` output，证明阈值边界
按预期生效。

| V | V/N | 最终表示 | Dense mean | Sparse mean | Latency delta | QPS delta | Sparse 更快 window |
|---:|---:|---|---:|---:|---:|---:|---:|
| 64 | 0.128% | Sparse | 2.254 ms | 2.193 ms | -2.69% | +2.75% | 11/12 |
| 100 | 0.200% | Sparse | 2.191 ms | 2.152 ms | -1.79% | +1.81% | 9/12 |
| 500 | 1.000% | Sparse | 2.084 ms | 2.078 ms | -0.31% | +0.29% | 5/12 |
| 999 | 1.998% | Sparse | 2.157 ms | 2.102 ms | -2.56% | +2.61% | 10/12 |
| 1000 | 2.000% | Sparse | 2.385 ms | 2.336 ms | -2.04% | +2.08% | 6/12 |
| 1001 | 2.002% | Dense fallback | 2.201 ms | 2.223 ms | +1.01% | -0.99% | 4/12 |
| 5000 | 10.000% | Dense fallback | 2.468 ms | 2.501 ms | +1.34% | -1.32% | 3/12 |

50K 下阈值内没有观察到明显 mean/QPS 回退，但信号仅为 0.3%～2.7%，其中 V=500、
V=1000 的 paired-window 一致性较弱，不能宣称为稳定收益。阈值外约 1% 的差异目前按
fallback 固定成本与运行波动处理，仍需和代表点复测噪声带一起判断。

#### 13.5.2 N=100K，单 sealed segment

V=64 自然选择 BF；V=500/1000/1001 自然选择 IVF。Dense/Sparse 对照在每个 case 中进入
相同 route。V=1001 的 Sparse 请求最终产生 `dense_threshold`，没有把超阈值 list 交给 IVF。

| V | V/N | Route | 最终表示 | Dense mean | Sparse mean | Latency delta | QPS delta | Sparse 更快 window |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 64 | 0.064% | BF | Sparse | 2.204 ms | 2.104 ms | -4.51% | +4.70% | 12/12 |
| 500 | 0.500% | IVF | Sparse | 2.477 ms | 2.250 ms | -9.17% | +10.03% | 12/12 |
| 1000 | 1.000% | IVF | Sparse | 2.445 ms | 2.280 ms | -6.75% | +7.19% | 12/12 |
| 1001 | 1.001% | IVF | Dense fallback | 2.311 ms | 2.332 ms | +0.89% | -0.87% | 5/12 |

#### 13.5.3 N=1M，单 sealed segment

V=64 自然选择 BF；V=500/1000/1001 自然选择 IVF。最新实现已不再复现旧版 V=64 的
mean 回退：本轮 mean 改善 10.24%，12/12 paired windows 更快。V=500/1000 的 IVF
mean/QPS 信号更大；V=1001 正确产生 Dense fallback，mean/QPS 回到约 1% 回退。

| V | V/N | Route | 最终表示 | Dense mean | Sparse mean | Latency delta | QPS delta | Sparse 更快 window |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 64 | 0.0064% | BF | Sparse | 3.742 ms | 3.359 ms | -10.24% | +11.35% | 12/12 |
| 500 | 0.0500% | IVF | Sparse | 5.081 ms | 3.410 ms | -32.90% | +48.74% | 12/12 |
| 1000 | 0.1000% | IVF | Sparse | 4.998 ms | 3.554 ms | -28.89% | +40.42% | 12/12 |
| 1001 | 0.1001% | IVF | Dense fallback | 4.927 ms | 4.983 ms | +1.14% | -1.12% | 5/12 |

mean/QPS 不能掩盖尾延迟风险。1M 的单请求 P90 如下；Sparse 在所有代表点均高于 Dense，
包括已 fallback Dense 的 V=1001。该现象与此前 `b_read` 重尾调查一致，因此当前结果支持
“固定阈值在 mean/QPS 上有潜力”，但**尚不足以宣布 T=1000 可正式启用**。

| V | Dense request P90 | Sparse-request P90 | P90 delta |
|---:|---:|---:|---:|
| 64 | 3.998 ms | 5.569 ms | +39.30% |
| 500 | 5.269 ms | 5.619 ms | +6.64% |
| 1000 | 5.194 ms | 5.753 ms | +10.76% |
| 1001 | 5.167 ms | 6.518 ms | +26.16% |

#### 13.5.4 当前判断与下一验收门槛

1. `T=1000` 的 representation switch 已由 counter 直接闭合：阈值内交付 Sparse，
   第 T+1 个命中后交付 Dense；BF 与 IVF 都能消费对应输入且不改变 auto route。
2. `N>=50K,V<=1000` 未出现 mean/QPS 的稳定劣化；N 增大后收益明显放大，符合避免
   Dense universe 工作的方向，但具体归因仍以 access/perf 诊断为准。
3. 1M request P90 重尾仍是启用阻塞项。下一步应保存 request/query-level raw samples，
   确认慢点是否固定落在特定 query，并把 route、A/B filter wall/cpu/off-CPU、page fault、
   context switch 与线程迁移按 request 关联；不能只依赖 aggregate P90。
4. 重尾解释并修复后，复测 1M 代表点与 `V=1001` fallback；随后再进入 1M x 768D、
   固定并发 QPS 和 4x50K/8x50K 多 segment，而不是现在直接扩大矩阵。

原始 JSONL 位于
`/home/ubuntu/workspace/SparseProject/artifacts/sparse-threshold-e2e-20260825/`；正式文件名为
`n{50k,100k,1m}-v*.jsonl`，50K 的独立 path closure 为 `n50k-v*-route.jsonl`。

### 13.6 P90 修复后的 1M 代表点复测（2026-08-25）

13.5 的 P90 阻塞已定位到两处实现缺口，而不是 cache、page fault 或 1K scattered read：

1. Dense intermediate 过去不能被后续低基数 predicate 重新收缩为 Sparse，FilterBits
   只能在末尾扫描完整 N-bit Dense；现已支持 Dense 与后续 Sparse 直接做 membership 交集；
2. raw producer 超过 T 后仍逐 ID 写 Dense；现改为阈值前缀回填加后续 SIMD chunk mask
   批量落 Dense，保持单趟 O(N/word) fallback。

复测沿用 13.5 的全部 Milvus E2E 约束和 ABBA 统计方法：

| V | V/N | Route | 最终表示 | Dense mean | Adaptive mean | Mean delta | Dense P90 | Adaptive P90 | P90 delta | windows |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.0064% | BF | Sparse | 3.600ms | 2.992ms | **-16.90%** | 3.935ms | 3.356ms | **-14.72%** | 12/12 |
| 1001 | 0.1001% | IVF | Dense fallback | 4.857ms | 4.470ms | **-7.97%** | 5.151ms | 4.716ms | **-8.45%** | 12/12 |

V=64 的 5,000-request 独立 run 同时闭合为 5,010 BF、5,010 Sparse output、5,010 scalar
和 5,010 cache-disabled，P90 为 3.388ms；旧的约 12% P90 双峰不再出现。V=1001 的结果
说明当前 bulk fallback 没有“先 Sparse 再全量重扫 Dense”的重复 O(N)，但该点仍走了
Adaptive producer，不能把约 8% 改善外推为所有 Dense fallback 的固有收益。

功能回归同时覆盖 sealed multi-segment、nullable、current delete、TTL expiry 与 growing，
结果通过；historical timestamp 的客户端探针在当前 pymilvus API 下不支持恢复 deleted IDs，
MVCC 历史读语义继续由已完成的 core 单测固化。

当前第一阶段的核心链路与单 segment 代表点已经闭合。下一验收顺序更新为：

1. 补 1M x 768D，确认向量计算占比上升后的收益稀释；
2. 补固定并发吞吐与 latency distribution；
3. 补 4x50K/8x50K multi-segment 和混合 Sparse/Dense-threshold segment；
4. 在以上 Milvus E2E 结果后再确定默认阈值与正式启用范围。

详细根因、单测和原始产物见 `sparse-p90-tail-investigation.md` 8.8，以及
`/home/ubuntu/workspace/SparseProject/artifacts/sparse-dense-intermediate-fix-20260825/`。

### 13.7 下一阶段 Milvus E2E：高维、并发与多 segment（2026-08-26）

本节继续只采用 Milvus client 完整 `search` 请求。Dense/Adaptive 之间唯一变量仍是
filter representation；Cardinal 保持 auto route。每个 case 使用固定 query/order、
topK10、NQ1、每模式 10 次 warmup 加 30 秒持续 warmup、12 个
`Dense -> Adaptive -> Adaptive -> Dense` window，每 slot 50 queries。C1 每模式共
1,200 requests；C8/C60 分别为 9,600/72,000 requests。计时前验证 topK ID/distance，
并用 counter 闭合 route、最终表示和每个 loaded segment 的行数。自动 compaction 在
显式 multi-segment 实验中关闭。

#### 13.7.1 Cohere 1M x 768D 与固定并发

数据来自 Cohere 1M，COSINE；collection 稳定后为两个 sealed segment，行数
`330K + 670K`。谓词为 `a < 64 AND b < 1000000`，最终总 V=64。Dense 与 Adaptive
preflight 均为 2/2 auto-BF；Adaptive 每请求产生两个 Sparse segment output。

| Concurrency | 指标 | Dense | Adaptive | Delta | paired windows |
|---:|---|---:|---:|---:|---:|
| 1 | mean | 3.510 ms | 3.077 ms | **-12.33%** | 12/12 |
|  | P90 | 3.729 ms | 3.316 ms | **-11.07%** |  |
|  | QPS | 283.19 | 322.77 | **+13.98%** |  |
| 8 | mean | 6.858 ms | 5.915 ms | **-13.75%** | 12/12 |
|  | P90 | 9.151 ms | 7.989 ms | **-12.70%** |  |
|  | QPS | 1,123.27 | 1,303.72 | **+16.06%** |  |
| 60 | mean | 44.719 ms | 40.232 ms | **-10.03%** | 12/12 |
|  | P90 | 73.671 ms | 66.311 ms | **-9.99%** |  |
|  | QPS | 1,280.58 | 1,424.53 | **+11.24%** |  |

高维和并发升高后收益没有消失，三个并发点的 mean、P90、QPS 方向一致，且均为
12/12 paired windows 更快。C60 相对 C8 略有稀释，符合排队、RPC 和向量计算等共享
成本占比上升，但该表不是单独的维度因果实验：本点是 2 segment，而 13.6 的 128D
代表点是单 segment，不能把二者差异全部归因为维度。

#### 13.7.2 混合 Sparse / Dense-threshold segment

1M x 128D synthetic、L2、C1，固定为四个 250K sealed segment。相同谓词下每个
segment 的 V 精确为 `[64, 1000, 1001, 5000]`，T=1000，因此每请求产生两个 Sparse
和两个 Dense-threshold output。两种模式的 auto route 均为 1 BF + 3 IVF。

| 指标 | Dense | Adaptive mixed | Delta | paired windows |
|---|---:|---:|---:|---:|
| mean | 3.825 ms | 3.656 ms | **-4.42%** | 12/12 |
| P90 | 3.978 ms | 3.788 ms | **-4.79%** |  |
| QPS | 259.40 | 271.28 | **+4.58%** |  |

该结果闭合了 representation 是 per-segment 决策：同一请求可以同时携带阈值内 Sparse
和阈值外 Dense，且不改变各 segment 的 auto route；它不是“整个请求一旦超阈值就全局
回退 Dense”。

#### 13.7.3 homogeneous multi-segment

两个 case 均为 128D synthetic、L2、C1，每 segment N=50K、V=64，所有 segment
自然选择 BF。Adaptive preflight 对每个 segment 均产生 Sparse output。

| Topology | 总 N / 总 V | Dense mean | Adaptive mean | Mean delta | Dense P90 | Adaptive P90 | P90 delta | QPS delta | windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 x 50K | 200K / 256 | 2.698 ms | 2.586 ms | **-4.17%** | 2.860 ms | 2.731 ms | **-4.51%** | **+4.35%** | 12/12 |
| 8 x 50K | 400K / 512 | 3.618 ms | 3.525 ms | **-2.58%** | 3.806 ms | 3.701 ms | **-2.76%** | **+2.63%** | 12/12 |

随着 segment 数增多，每 segment dispatch、merge 和 endpoint 固定成本增长，Sparse
可消除的表示/枚举工作占比下降，因此百分比收益由 4.17% 收缩到 2.58%；两个点仍均为
12/12 windows 更快，没有观察到 multi-segment 功能或尾延迟回退。

#### 13.7.4 阶段性阈值决策

当前结果已满足 `T=1000` 作为**功能开启后的候选默认 cap**的第一轮门槛：在
`N_i >= 50K, V_i <= 1000`、BF/IVF、1--8 segment、128D/768D 和 C1/C8/C60 的已测
范围内未观察到可复现 mean、P90 或 QPS 劣化；`V_i > 1000` 能单趟切换 Dense，混合
segment 也正确交付。不过这不等于默认启用功能：`enableSparseFilterResult` 继续保持
false，以维持 baseline 行为；OR、Graph 以及更广 producer/consumer 矩阵未完成前，
不把 Adaptive 改成默认路径。阈值先保持绝对 V cap，不增加尚无证据需要的 V/N 规则。

原始 JSONL 与命令输出位于：

```text
/home/ubuntu/workspace/SparseProject/artifacts/sparse-next-stage-20260826/
  cohere1m-c{1,8,60}-abba-stable-2seg.jsonl
  mixed-4x250k-c1-abba-current-libs.jsonl
  multi-{4,8}x50k-c1-abba.jsonl
```

实验二进制闭合信息：Milvus commit `09a9415a0a0e104df716220b330c49498fa1e7be`；
`libmilvus_core.so` SHA256 `376a54fe6f7571ef1a455c12ab57c519ed885c6641f7462025986cf1f3218c55`；
`libknowhere.so` SHA256 `400aa1e6e3ee8d44372cc5584de26d83cae8dcffb9cc9f03619990115329af82`。
启动时必须优先加载当前 `cmake_build/src`、当前 Knowhere build 和其中 Cardinal v2/v1；
旧绝对 RPATH 曾错误加载 `KnowPR1732` 库并导致 IVF `partial_read.h` 崩溃，故动态库 maps
检查是本矩阵的前置 closure，而不是性能变量。

### 13.8 当前实现语义审计与遗留闭环（2026-08-26）

本轮针对“Sparse 收益是否来自跳过必要功能”做代码审计。阶段性结论是：
当前已测的普通 Cardinal `Search` 路径没有故意跳过 predicate、cache、MVCC、
delete、TTL、auto route、distance 或 TopK 语义；收益主要与物理表示和访问量
变化一致。但 Sparse 尚不是所有 `BitsetView` consumer 都能安全处理的通用表示。

#### 13.8.1 已闭环的必要语义

| 阶段 | 当前 Sparse 行为 | 审计结论 |
|---|---|---|
| Scalar predicate | 以 accepted IDs 代替 N-bit 结果；nullable scalar 仍遵守 SQL NULL 语义 | 物理表示替换，未跳过 predicate |
| AND 后续 predicate | 只处理上一阶段留下的 IDs | 由 `O(N)` 收缩为 `O(V)` 的算法工作量变化 |
| Expression cache | Dense/Adaptive 使用表示和阈值隔离的 cache signature；Adaptive 可缓存 Sparse 或 threshold-Dense | 未为 Sparse 故意关闭 cache |
| MVCC / history / TTL | 使用正常 timestamp invalid mask 压缩 accepted IDs | 未跳过快照和 TTL |
| Delete | 使用正常 delete mask 移除不可见 IDs | 未跳过 delete |
| Growing boundary | payload universe 与 query snapshot `active_count` 校验 | 未把快照后追加行纳入候选 |
| Cardinal route | Dense/Sparse 继续使用同一 auto selector | 未暗中强制 BF/IVF/Graph |
| Distance / TopK | direct 对照已闭环主要 candidate/probe/distance 计数和 TopK/distance | 未通过少做 distance 或改变 TopK 获益 |

Sparse 路径的 one-bit `ColumnVector` 只用于满足 Driver 的非空输出约束；语义结果由
query-owned `SparseIdPayload` 携带，`MvccNode` 和 `VectorSearchNode` 都显式识别它。
因此已测路径省去的是 N-bit data/valid bitmap materialization、后续 AND 的 N-row
处理和 BF/IVF 的 universe traversal，不是可见性或搜索正确性校验。

最新 Milvus E2E 尚未对每个 case 完成 producer、visibility、search、RPC/merge 的
绝对时间全分解，所以只能说 endpoint 变化与上述算法工作量一致，不将
其改善百分比 100% 归结为单一函数。

#### 13.8.2 正式启用前的遗留清单

| 优先级 | 遗留项 | 当前风险 | 后续闭环 |
|---|---|---|---|
| P0 | 非 Cardinal sealed indexed search capability gate | ValidIdList 可能被直接传入 Dense-only Knowhere index，其 `test()` 仍会读取空 `bits_` | 仅对声明 Sparse capability 的 index 传 Sparse；其他边界 materialize Dense，并补回归 |
| P0 | Nullable vector `OffsetMapping` | 已测 nullable 是 scalar field；nullable vector 仍调用 Dense-oriented `all()/none()/test()` | 实现 Sparse offset transform 或 mapping 前转 Dense，补 nullable-vector E2E |
| P0 | Iterator / iterator-v2 | Cardinal normal `Search` 能识别 ValidIdList，iterator workspace 仍以 `bitset.data()` 构造 Dense checker | 能力补齐前转 Dense 或显式 `NotSupported`，禁止静默下传 |
| P1 | Multi-tenant / reorder mapping | single-tenant reorder 已有专用 mapping；multi-tenant tenant route 尚未闭环 | 补 ID-domain mapping、capability 检查和 multi-tenant 正确性测试 |
| P1 | Emb-list / element-level / vector-array | element-level 当前显式拒绝 Sparse；emb-list row/vector mapping 未闭环 | 保持安全拒绝或 Dense fallback，不宣称已支持 |
| P1 | Graph Milvus E2E | Cardinal 已有精确 Bloom+Flat adapter，但端到端功能/性能边界未完整闭环 | 补 auto-route E2E；高 `V` 不得无条件使用 Sparse |
| 设计边界 | OR/NOT 与 invalid-ID polarity | 第一阶段 OR 显式保持 Dense，Sparse union/NOT/SQL 3VL 组合未定义 | 当前 Dense fallback 保持正确；后续单独设计 |

Raw sealed/growing BF 已在 legacy Knowhere 边界 materialize Dense，功能安全但不保留
Sparse consumer 收益。在上述 P0 闭环前，`enableSparseFilterResult` 继续默认为
`false`，不将当前实现宣称为全局通用 filter representation。

### 13.9 统一 Cohere 单 segment 决策矩阵（2026-08-26）

为避免把不同数据集、维度、segment topology 和并发度混在同一主结果图，
第二部分的主展示数据重新收敛为一个受控矩阵。八个性能对照全部复用同一
collection、Cardinal index、scalar data、query 集和当前二进制；只改变请求的
accepted cardinality `V`、Dense/Adaptive 表示以及显式的 client concurrency。

| 固定项 | 配置 |
|---|---|
| Dataset | Cohere 1M x 768D，固定 train shard 和前 50 条 test query |
| Vector/index | COSINE，`CARDINAL_TIERED`，topK=10，ef=64 |
| Scalar producer | raw INT64；`a`/`b` 为固定 seed 的 permutation，不建 scalar index |
| Segment topology | 1 个 loaded sealed segment，精确 1,000,000 rows |
| Request | NQ=1，Cardinal auto route，禁止强制 BF/IVF/Graph |
| Representation | Dense baseline vs Adaptive，`T=1000` |
| Timing | 每模式 10 次 warmup + 30s 持续 warmup；12 个 ABBA window；每 slot 50 queries |

`dataCoord.segment.maxSize` 在 collection 创建前提高到至少 4096MB，并在计时前
强制校验 `loaded_segments == 1` 与 `loaded_segment_rows == [1000000]`。不满足则终止，
不用 multi-segment 结果代替本矩阵。

C60 保持 60 个独立的 NQ=1 请求；测试期间关闭 QueryNode search-task grouping，避免把多个
请求合并成 NQ>1 task 后复用一次 filter payload preparation，从而把 batch 共享收益混入
Dense/Sparse 表示对照。该设置只是本轮实验隔离变量的约束，不是 Sparse 对 NQ>1 的能力限制。

| V | V/N | C1 | C60 | Adaptive 预期最终表示 |
|---:|---:|---|---|---|
| 64 | 0.0064% | Dense/Adaptive | Dense/Adaptive | Sparse |
| 500 | 0.0500% | Dense/Adaptive | Dense/Adaptive | Sparse |
| 1000 | 0.1000% | Dense/Adaptive | Dense/Adaptive | Sparse |
| 1001 | 0.1001% | Dense/Adaptive | Dense/Adaptive | threshold-Dense |

每个 V 在计时前对全部 50 queries 校验 TopK ID/distance，并使用 route 和
representation counter 闭环 Dense/Adaptive 的实际 auto route 与最终 payload。V=64/500/1000
是 Sparse 性能点；V=1001 是阈值 fallback 控制组。是否进入 BF 或 IVF 以 counter
实测为准，不按历史 128D 结果推测，也不为覆盖某个 route 强制搜索算法。

C1 与 C60 的绝对延迟量级不同，最终结果分组展示并使用独立坐标尺度；
不把两组的绝对 latency/QPS 当成同一条件下的横向对照。

代码审计同时确认 Cardinal Memory 不缺 Sparse consumer：Knowhere Cardinal `HNSW`
node 对应内部 `IndexType::Memory`，与 `CARDINAL_TIERED` 共用 ValidIdList handoff 和
BF/IVF/Graph consumer。当前不需要新增 Memory 支持实现；Memory Milvus E2E 是后续
独立的 backend 扩展矩阵，不与本次统一 Tiered 主矩阵混测。

#### 执行结果

2026-08-26 完成全部八组 Milvus E2E。第一次以 10k insert batch 构建时，虽然
`segment.maxSize=8192MB`，仍在 33 个 batch、约 330k rows 时因 binlog 数触发 sealing；
该构建被判为无效并删除。正式 collection 改为 40k/batch，共 25 个 batch，在全部
1M rows 写入后只 flush 一次。最终 topology 闭环为一个 persistent/loaded sealed
segment，精确 1,000,000 rows；自动 compaction 关闭。

**C1（每模式 1,200 个 timed requests）**

| V | Auto route | Adaptive 输出 | Dense mean | Adaptive mean | Mean delta | Dense QPS | Adaptive QPS | QPS delta |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 64 | BF | Sparse | 3.857 ms | 3.227 ms | **-16.33%** | 257.76 | 307.76 | **+19.40%** |
| 500 | IVF | Sparse | 5.359 ms | 3.484 ms | **-34.98%** | 185.78 | 285.13 | **+53.48%** |
| 1,000 | IVF | Sparse | 5.098 ms | 3.421 ms | **-32.89%** | 195.28 | 290.46 | **+48.74%** |
| 1,001 | IVF | Dense threshold fallback | 4.962 ms | 4.581 ms | -7.66% | 200.63 | 217.21 | +8.27% |

| V | Dense median | Adaptive median | Median delta | Dense P90 | Adaptive P90 | P90 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3.805 ms | 3.174 ms | -16.58% | 4.160 ms | 3.566 ms | -14.28% |
| 500 | 5.349 ms | 3.436 ms | -35.76% | 5.566 ms | 3.827 ms | -31.24% |
| 1,000 | 5.091 ms | 3.408 ms | -33.06% | 5.300 ms | 3.758 ms | -29.10% |
| 1,001 | 4.912 ms | 4.542 ms | -7.55% | 5.229 ms | 4.789 ms | -8.42% |

**C60（每模式 72,000 个 timed requests）**

| V | Auto route | Adaptive 输出 | Dense mean | Adaptive mean | Mean delta | Dense QPS | Adaptive QPS | QPS delta |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 64 | BF | Sparse | 41.559 ms | 36.886 ms | **-11.24%** | 1,379.90 | 1,555.86 | **+12.75%** |
| 500 | IVF | Sparse | 47.748 ms | 36.517 ms | **-23.52%** | 1,205.58 | 1,568.83 | **+30.13%** |
| 1,000 | IVF | Sparse | 46.797 ms | 36.692 ms | **-21.59%** | 1,224.76 | 1,558.10 | **+27.22%** |
| 1,001 | IVF | Dense threshold fallback | 47.026 ms | 44.515 ms | -5.34% | 1,221.06 | 1,286.74 | +5.38% |

| V | Dense median | Adaptive median | Median delta | Dense P90 | Adaptive P90 | P90 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 38.894 ms | 33.955 ms | -12.70% | 68.575 ms | 60.776 ms | -11.37% |
| 500 | 45.123 ms | 33.924 ms | -24.82% | 75.288 ms | 60.178 ms | -20.07% |
| 1,000 | 43.861 ms | 34.125 ms | -22.20% | 75.483 ms | 60.444 ms | -19.92% |
| 1,001 | 43.731 ms | 41.324 ms | -5.50% | 76.561 ms | 73.310 ms | -4.25% |

所有八组均满足：50/50 fixed queries 的 TopK ID/distance 完全一致、12/12 paired
windows 中 Adaptive 更快。route counter 证明 V=64 两模式均走 BF，其他三个 V
两模式均走 IVF；representation counter 证明 V=64/500/1000 只在 Adaptive 请求中
产出 Sparse，而 V=1001 只触发 Dense threshold fallback。

V=1000 与 V=1001 是最有解释力的阈值相邻对照：数据规模和 IVF route 几乎不变，
但开启 Sparse payload 时 mean 优势由 C1 的 7.66% 扩大到 32.89%，由 C60 的
5.34% 扩大到 21.59%。因此结果不能只归因于 Adaptive producer 的执行差异；
IVF 对 Sparse payload 的消费在本 workload 中贡献了额外、可观测的端到端收益。
同时，V=1001 仍有 5%--8% 差异，故该行只能作为 threshold fallback 控制，不应
称为 Dense A/A，也不能用它精确扣除 Sparse consumer 的独立成本。

复现实物位于：

- `/home/ubuntu/workspace/SparseProject/artifacts/cohere1m-single-segment-matrix-20260826/prepare-batch40k.log`
- 同目录 `v{64,500,1000,1001}-c{1,60}.log`

本轮 SHA：Milvus binary `6b476925...`，`libmilvus_core.so` `376a54fe...`，
Knowhere `400aa1e6...`，Cardinal v2 `70aa45b1...`；`/proc/<pid>/maps` 已证明运行时
加载的均为本 worktree 对应文件。

### 13.10 Cohere V=500 C1 的 perf 收益归因（2026-08-27）

针对 13.9 中 QPS 收益最高的 `V=500, C1` 点补充 CPU 归因。复用同一 Cohere
1M x 768D collection、单个 1M-row sealed segment、`CARDINAL_TIERED`、COSINE、
NQ=1、topK=10、ef=64 和前 50 条固定 query；predicate 仍为
`a < 500 AND b < 1000000`。Dense/Adaptive 各先执行 10 次请求预热和 30 秒持续预热，
再分别以 30 秒 steady workload 执行：

```text
sudo perf record -F 199 -g -p <milvus-pid> -- sleep 30
```

两侧 route preflight 均为 IVF，只有 Adaptive 输出 Sparse。相同二进制和 collection
上的新一轮 12-window ABBA endpoint closure 为：

| 指标 | Dense | Adaptive | Adaptive delta |
|---|---:|---:|---:|
| Mean latency | 5.138 ms | 3.236 ms | -37.03% |
| QPS | 193.81 | 307.11 | +58.46% |
| P90 | 5.392 ms | 3.577 ms | -33.67% |

12/12 paired windows 均为 Adaptive 更快。perf flat cycles 的关键模块为：

| 模块 / 代表符号 | Dense cycles share | Adaptive cycles share | 含义 |
|---|---:|---:|---|
| IVF Dense valid-ID extraction，`BitCompressBatch64` | 19.24% | 未出现 | Adaptive bucket-grouped IDs 删除了 Dense IVF 的 bitmap 枚举/membership 工作 |
| Scalar SIMD compare，`OpCompareValImpl<int64>` | 16.86% | 15.58% | Dense 符号合并 A/B 全段比较；Adaptive 保留 A 全段扫描，B 只消费 A 的 500 个 IDs |
| Sparse B read/evaluate | 无 | 至多 1.57% | `TryApplySparseFilter` 加 `FilterSortedNativeIdsByRawData`；前者还含公共 Sparse 框架工作，因此该值是 B 的上界 |
| Sparse IVF count/prefix/scatter | 无 | 0.22% | query-local bucket grouping plan |
| IVF coarse search | 0.43% | 0.60% | 两侧公共工作；删除其他工作后 Adaptive 的相对占比上升 |

用同一时段的 process-wide cycles 和 ABBA QPS 近似归一化，Dense/Adaptive 分别约为
11.41M/6.86M cycles/request，下降 39.9%，与 endpoint mean 的 37.0% 下降接近。
对每请求约 4.55M saved cycles 的估算分解如下：

| 被删除或减少的工作 | 估算 cycles/request | saved cycles 占比 |
|---|---:|---:|
| IVF `BitCompressBatch64` | 2.19M | 48.3% |
| 第二 predicate 的全段 SIMD compare（尚未扣除 Sparse B read） | 0.85M | 18.8% |
| 其他 Dense materialization/handoff/framework 差异及 residual | 1.50M | 32.9% |

因此当前证据表明，最大单项收益来自 IVF 搜索侧删除 Dense valid-ID 枚举/过滤提取，
其可归因 cycles 约为第二 predicate compare 降幅的 2.6 倍。A/B 在 Dense 中共享同一个
SIMD symbol，若需要精确拆出 B-only cycles，仍需增加 predicate-specific counter/timer；
本轮以 Adaptive 保留的全段 compare 作为 A baseline，不把 residual 强行分摊给 B。

原始数据与独立分析记录：

- [归因摘要](/home/ubuntu/workspace/SparseProject/artifacts/cohere1m-v500-c1-perf-20260827/README.md)
- [Dense perf.data](/home/ubuntu/workspace/SparseProject/artifacts/cohere1m-v500-c1-perf-20260827/dense.data)
- [Adaptive perf.data](/home/ubuntu/workspace/SparseProject/artifacts/cohere1m-v500-c1-perf-20260827/sparse.data)

### 13.11 Sparse payload 直接路由 BF 的快速验证（2026-08-27）

为判断 13.9/13.10 的 Sparse-IVF 收益是否值得保留 IVF consumer，本轮只改变 Cardinal
Tiered selector：Dense payload 保持原 auto route，ValidIdList payload 直接选择 BF。
数据、collection、predicate、query 顺序和计时协议均复用 13.10 的 Cohere 1M x 768D、
单个 1M-row sealed segment、COSINE、`a < 500 AND b < 1000000`、V=500、NQ=1、
topK=10、ef=64、C1、前 50 条 query、30s warmup 和 12-window ABBA。

Route counter 闭环为：Dense 10/10 IVF、Sparse 10/10 BF。计时前逐条比较全部 50 条
query，TopK IDs 和 distances 50/50 完全一致。

| 指标 | Dense auto-IVF | Sparse direct-BF | Delta |
|---|---:|---:|---:|
| Mean latency | 5.237 ms | 3.347 ms | **-36.10%** |
| Median latency | 5.172 ms | 3.402 ms | **-34.22%** |
| P90 latency | 5.630 ms | 3.776 ms | **-32.92%** |
| QPS | 190.10 | 296.87 | **+56.17%** |

12/12 paired windows 均为 Sparse direct-BF 更快。与 13.10 同配置但 Sparse 仍走 IVF
的结果（3.236 ms、307.11 QPS）相比，direct-BF 的 mean latency 高约 3.4%、QPS 低约
3.3%；该比较跨两轮运行，不是 BF/IVF 同轮 ABBA，因此只用于判断量级。当前证据支持：
在 V=500 点直接 BF 可以保留几乎全部 Sparse E2E 收益并获得相同 TopK/distance；虽然没有
证明它比 Sparse-IVF 更快，但差异不大，而 BF 无需 IVF grouping adapter，并完整扫描 accepted
set。方案据此收敛为：Cardinal Memory/Tiered 收到 ValidIdList 后统一走 BF；Dense 继续使用
原 auto selector。IVF/Graph Sparse adapter 不进入第一阶段正式路径。后续矩阵用于校验该策略
在 V≤T、NQ 和并发范围内的稳定性，不再把 Sparse-IVF 作为第一阶段候选方案。

本决策替代 7.2 中“Sparse 不影响 auto route”以及 12.9 中继续评估 Sparse-IVF 准入的旧计划；
这些章节保留为实验演进记录，不再代表当前第一阶段方案。

### 13.12 非目标实现清理与 switch-point E2E（2026-08-27）

本轮按 0.6 的计划完成第一阶段实现收口：canonical payload 只保留 Dense filtered
bitmap 与 unordered/unique Sparse included IDs；删除 Milvus Dense→Sparse 全量转换、
跨层 valid-Roaring API、Sparse 排序契约，以及本项目新增的 Sparse-IVF/Graph/Bloom+Flat
adapter。BitmapIndex 内部 Roaring posting、expression-cache 内部压缩和 Knowhere/Cardinal
既有 filtered-Roaring 能力不受影响。当前 Cardinal Memory/Tiered 收到 Sparse payload
统一走 BF，Dense 仍使用原 auto selector。源码检索确认生产与测试目录中已无上述废弃
符号或旧 `ValidIdsPerQuery` gate。

#### 构建与正确性门禁

本轮首先修正了构建来源：Knowhere 嵌套 Cardinal checkout 曾使 Milvus 在运行时加载旧
dispatcher。重新配置后，`compile_commands.json` 与 `/proc/<pid>/maps` 均指向当前
SparseProject Cardinal worktree；新 `libcardinalv2.so` 中旧 gate 字符串不存在。

| 门禁 | 结果 |
|---|---:|
| Sparse/AND/MVCC/delete/TTL/nullable/index 定向 C++ | 85 / 85 PASS |
| Go 配置与动态 callback | 2 / 2 PASS |
| Cohere 1M strict smoke | 10 / 10 query TopK IDs 与 distances 完全一致 |
| smoke 表示与 route | Dense=IVF；Sparse=valid-ID BF；500 distance attempts/query |

固定产物 SHA：

```text
libmilvus_core.so  2624d0ad76bd35147e05b1df476d5aa8a1135cdbea60de5bb6b94762e429d946
libknowhere.so     c0325c0646dbfa456eb532a801dbc85868fe72df16ae47e17bf35c232e4e045e
libcardinalv1.so   aac1bd0986efaaf4aacd1a79853174b0830542deaaf7ddfbdfaaf67ccab6a790
libcardinalv2.so   3109c2ce403181f79fe14c1cf316668d2bc5fb3fd969f310861c79e8037da66a
```

#### Producer 输出成本 ABBA

在停止 Milvus、固定 CPU 4 后，使用当前清理版 producer benchmark 做
Dense→Adaptive→Adaptive→Dense；每格为两半轮各 5 repetitions 的 mean 再平均。
该表只度量过滤结果生产，不含 search，单位为微秒：

| Producer | V/T | Dense | Adaptive | Adaptive delta |
|---|---:|---:|---:|---:|
| Raw | 1,000/1,000 | 1323.29 | 710.03 | -46.34% |
| Raw | 1,001/1,000 | 1353.18 | 712.95 | -47.31% |
| Raw | 5,000/5,000 | 1360.63 | 737.24 | -45.82% |
| STL_SORT | 1,000/1,000 | 5.00 | 2.66 | -46.79% |
| STL_SORT | 1,001/1,000 | 5.01 | 71.52 | +1328.63% |
| STL_SORT | 5,000/5,000 | 9.36 | 12.49 | +33.46% |
| Bitmap | 1,000/1,000 | 8.42 | 3.33 | -60.40% |
| Bitmap | 1,001/1,000 | 8.35 | 76.15 | +811.64% |
| Bitmap | 5,000/5,000 | 7.56 | 14.17 | +87.42% |

这说明 Sparse/Dense 的 predicate 成本可以共享，但输出成本并不相等。native
STL_SORT/Bitmap 在 V≤T 时直接枚举 posting/bounds，Sparse 更便宜；V 较大时 append
IDs 超过 Dense word write。T+1 fallback 没有重跑 predicate 或重新扫描 N，但仍需回填
前 T 个 IDs 并建立 Dense payload，约 66--68 us 的绝对增量在该微基准中被较小的 native
Dense baseline 放大成高百分比。Raw 使用统一 batched sink，当前已测点仍快于原 Dense
all-at-once 路径；这属于实现 regime，而不是“Sparse 输出天然免费”的证据。

原始 ABBA：`artifacts/switchpoint-milvus-e2e-20260827/producer-abba-clean/`。该层只用于
成本归因，产品阈值仍以下面的 Milvus E2E 为准。

#### 1M×768D、C1 discovery

共同配置为 Cohere 1M×768D、COSINE、单个 1M-row sealed segment、
`CARDINAL_TIERED`、`a < V AND b < N`、NQ=1、topK=10、ef=64。每点使用前 50 条
固定 query、每模式 10 次加 30 秒 warmup、12 个 Dense→Sparse→Sparse→Dense
paired windows，共 1,200 timed requests/mode。每点设置 `T=V`，用于测量“允许 Sparse”
的真实 E2E，而不是测试固定阈值 fallback。所有点均通过 representation、route、work
counter 和结果闭环。

| V | V/N | Dense→Sparse route | Dense mean | Sparse mean | QPS delta（95% CI） | Sparse wins |
|---:|---:|---|---:|---:|---:|---:|
| 100 | 0.01% | BF→BF | 4.392 ms | 3.762 ms | +16.69%（+15.35%, +18.10%） | 12/12 |
| 500 | 0.05% | IVF→BF | 5.337 ms | 3.503 ms | +53.48%（+47.00%, +59.96%） | 12/12 |
| 1,000 | 0.10% | IVF→BF | 5.451 ms | 4.019 ms | +35.49%（+33.85%, +37.41%） | 12/12 |
| 2,000 | 0.20% | IVF→BF | 5.019 ms | 4.003 ms | +25.27%（+23.82%, +26.65%） | 12/12 |
| 5,000 | 0.50% | IVF→BF | 4.738 ms | 4.278 ms | +10.77%（+8.91%, +12.83%） | 12/12 |
| 6,000 | 0.60% | IVF→BF | 4.785 ms | 4.579 ms | +4.78%（+2.17%, +7.42%） | 11/12 |
| 7,000 | 0.70% | IVF→BF | 4.889 ms | 4.964 ms | -1.46%（-2.20%, -0.57%） | 2/12 |
| 8,000 | 0.80% | IVF→BF | 4.787 ms | 5.095 ms | -5.98%（-7.16%, -4.77%） | 0/12 |
| 9,000 | 0.90% | IVF→BF | 4.789 ms | 5.396 ms | -11.18%（-12.15%, -10.34%） | 0/12 |
| 10,000 | 1.00% | IVF→BF | 4.959 ms | 5.830 ms | -14.89%（-15.33%, -14.40%） | 0/12 |

C1 的 crossover 位于 0.6%--0.8%，但不能据此单独确定生产阈值。

#### 1M×768D、C60 hold-out

C60 保持相同 collection、predicate、NQ=1 和 ABBA 规则；60 个 closed-loop worker
各重放 5 条固定 query，每点 7,200 timed requests/mode。`grouping.maxNQ=1` 明确禁止
把独立请求合并成 NQ>1 task，因此变量只有并发度。结果表明并发下 BF 的 O(V)
distance work 更早饱和，switch point 明显左移：

| V | V/N | Dense mean | Sparse mean | QPS delta（95% CI） | Sparse wins |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.10% | 44.286 ms | 35.362 ms | +22.82%（+20.78%, +24.70%） | 12/12 |
| 2,000 | 0.20% | 42.263 ms | 36.202 ms | +16.02%（+13.32%, +18.97%） | 12/12 |
| 3,000 | 0.30% | 42.696 ms | 38.805 ms | +11.17%（+8.97%, +13.60%） | 12/12 |
| 4,000 | 0.40% | 41.794 ms | 43.408 ms | -1.60%（-3.95%, +0.82%） | 3/12 |
| 5,000 | 0.50% | 41.510 ms | 51.291 ms | -17.47%（-19.00%, -15.78%） | 0/12 |
| 7,000 | 0.70% | 42.903 ms | 68.293 ms | -35.18%（-36.72%, -33.51%） | 0/12 |
| 8,000 | 0.80% | 42.971 ms | 78.702 ms | -43.88%（-45.74%, -41.86%） | 0/12 |

因此 1M segment 上，满足“最差场景 QPS 劣化不超过 5%”的当前候选上界是
`V/N <= 0.4%`；`0.1%` 点达到目标收益区间（+22.82%），而 0.5% 已不能准入。

#### N-scale hold-out

为检查比例阈值是否只是 1M 特例，另建 Cohere 50K/100K/250K、768D、单 sealed
segment collection；其余协议相同。下表直接列 paired-window QPS delta：

| N | V/N=0.1%，C1 | V/N=0.4%，C1 / C60 | V/N=0.5%，C1 / C60 |
|---:|---:|---:|---:|
| 50K | +1.68% | +1.65% / +0.68% | +2.64% / -1.13% |
| 100K | +3.69% | +8.27% / +1.50% | +9.51% / +2.46% |
| 250K | +11.35% | +17.08% / +4.60% | +13.72% / +4.18% |
| 1M | +35.49%（C1）/+22.82%（C60） | +4.78% 至 -1.46% 的 C1 邻域 / -1.60% | +10.77% / -17.47% |

50K--1M 的已测范围支持 `N >= 50K && V/N <= 0.4%` 作为下一轮候选 policy，且所有
已测准入点的 QPS 劣化均小于 5%。但结果也证明性能不只由比例决定：相同比例下，
absolute V 与并发会改变 BF distance 饱和点。当前最大已验证准入 V 为 4,000；在更大
segment/不同维度完成 hold-out 前，建议保留 `V <= 4000` 安全上限，而不是把 0.4%
无界外推。`N_min=50K` 是当前测试边界，不是已证明的理论拐点；50K 以下尚未测试。

原始实物：

- `artifacts/switchpoint-milvus-e2e-20260827/targeted-cpp-regression.log`
- `artifacts/switchpoint-milvus-e2e-20260827/smoke-v500-c1-fixed-build.log`
- `artifacts/switchpoint-milvus-e2e-20260827/discovery-c1/`
- `artifacts/switchpoint-milvus-e2e-20260827/holdout-c60/`
- `artifacts/switchpoint-milvus-e2e-20260827/n-scale/`

以上结论只使用 Milvus E2E；producer 微基准仍仅用于解释 Dense/Sparse output 和
T+1 fallback 的局部成本，不参与产品阈值判定。

#### 13.12.1 batch 修复后的阈值验证状态（2026-08-28）

13.12 的 Search 主表使用 `a < V AND b < N`，因此 Sparse 会把 A 的 V 个 IDs
交给 B。2026-08-28 已确认修复 B 的逐 ID
`get_chunk_by_offset -> CapturePublishedState` 锁竞争；同一 AND-reduce C60
的 Sparse QPS 由 1,104.58 恢复到 1,645.67，context switch 由 26.04K/s
降到 7.58K/s。该修复会改变旧 switch-point 曲线，因而 13.12 的
`ratio_cap=0.4% && V_abs_safe=4000` 暂作修复前历史候选，不是最终阈值。

主曲线已在 batch 修复后的同一二进制上刷新。共同配置仍为 Cohere
1M×768D、COSINE、单个 1M-row sealed segment、`CARDINAL_TIERED`、
`a < V AND b < N`、NQ=1、topK=10；每模式 30 秒 warmup，12 个
Dense→Sparse→Sparse→Dense paired windows。C1 每模式 1,200 requests，C60
每模式 7,200 requests。所有 16 点均通过 strict closure：Dense 表示走 IVF，
Sparse 表示走 direct-BF，Sparse distance attempts 精确为 V/query，单 segment
拓扑稳定，Sparse BF 结果作为 exact reference。C1 的 Dense IVF 在 V≤4K 时与
Sparse BF TopK 完全一致；V=5K--8K 时 Dense recall@10 为 99.4%--99.0%。C60
使用 5 条固定 query，已测 query 的 TopK 均一致。

| V | V/N | C1 Dense / Sparse mean | C1 QPS delta（95% CI） | C1 wins | C60 Dense / Sparse mean | C60 QPS delta（95% CI） | C60 wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.10% | 4.996 / 3.527 ms | +41.69%（+38.44%, +45.07%） | 12/12 | 43.078 / 33.319 ms | +26.78%（+23.33%, +30.66%） | 12/12 |
| 2,000 | 0.20% | 4.979 / 3.752 ms | +32.73%（+30.17%, +35.83%） | 12/12 | 43.606 / 35.708 ms | +20.58%（+17.48%, +23.79%） | 12/12 |
| 3,000 | 0.30% | 4.864 / 3.792 ms | +28.27%（+25.57%, +30.55%） | 12/12 | 43.100 / 36.442 ms | +16.95%（+13.89%, +19.79%） | 12/12 |
| 4,000 | 0.40% | 5.029 / 4.184 ms | +20.08%（+18.93%, +21.14%） | 12/12 | 42.167 / 35.350 ms | +16.65%（+14.51%, +18.76%） | 12/12 |
| 5,000 | 0.50% | 4.568 / 3.726 ms | +22.62%（+20.50%, +24.69%） | 12/12 | 42.291 / 36.928 ms | +12.86%（+10.75%, +15.42%） | 12/12 |
| 6,000 | 0.60% | 5.044 / 4.414 ms | +14.58%（+11.60%, +17.87%） | 12/12 | 41.880 / 36.719 ms | +12.93%（+10.70%, +14.95%） | 12/12 |
| 7,000 | 0.70% | 4.643 / 4.162 ms | +11.73%（+9.66%, +13.80%） | 12/12 | 41.770 / 37.925 ms | +7.90%（+5.30%, +10.48%） | 12/12 |
| 8,000 | 0.80% | 4.658 / 4.397 ms | +6.02%（+4.46%, +7.55%） | 12/12 | 42.076 / 38.674 ms | +8.27%（+6.38%, +10.22%） | 12/12 |

该结果证明旧曲线的提前 crossover 主要受 B 的 per-ID chunk/state resolve 实现
影响：修复后 1M 上至 0.8% 仍未出现负收益，且 scalar evaluation 的 Sparse/Dense
差距在 C60 下由 V=1K 的 0.770/2.096 ms 逐步收敛到 V=8K 的
1.036/2.115 ms。它还不能单独证明新 ratio cap：本表同时改变 V 与 V/N，且 Sparse
BF distance work 随 V 增加；必须以 N-scale hold-out 分离 absolute V、ratio 与并发。

原始产物：

- `artifacts/switchpoint-post-batch-fix-20260828/search-main-c1/`
- `artifacts/switchpoint-post-batch-fix-20260828/search-main-c60/`

待完成项按优先级排列：

1. **P1：单项成本与真实 producer 边界。** Layer 1 尚未完成原计划的
   `N=50K--10M × ratio × random/clustered` sink 成本曲线；Layer 2 还需补 generic
   expression，并在候选 `T_eff` 上对 raw/STL_SORT/Bitmap/generic 执行
   `T-1/T/T+1`。已有 T=1000 正确性边界，但尚未覆盖 ratio policy 下各 N 的
   有效边界。
2. **P1：重 predicate hold-out。** 算术 INT64 已覆盖 V=1000/4000、
   C1/C60；VARCHAR prefix/LIKE 和变长字段 projection 尚未执行。
3. **P1：policy 本身的边界验证。** 当 ratio/最小 N/绝对 V 候选重新收敛后，
   实现 selector，再测 `N_min-1/N_min/N_min+1` 和
   `T_eff-1/T_eff/T_eff+1`，避免只用请求级 cap 模拟 policy。
4. **P2：外推 hold-out。** 10M/更大 segment、NQ>1、Cardinal Memory、
   multi-segment mixed payload、cache-on/visibility-on 和不同硬件尚未用新实现闭合；
   这些不阻塞首轮阈值 discovery，但在产品默认开启前必须补齐。

已完成、不需重复的内容：单趟 T+1 fallback 无二次 N 扫描、
T=1000 的 999/1000/1001 表示边界正确性、Dense/Sparse 结果等价、
MVCC/delete/TTL/nullable/growing/multi-segment 正确性，以及 Sparse direct-BF route。

##### N-scale hold-out 刷新结果

同一修复后二进制继续完成 24 点 Milvus E2E：Cohere 768D、单 sealed segment、
`N=50K/100K/250K/1M`、`V/N=0.1%/0.4%/0.5%`、C1/C60。除 N、V 与并发外，
predicate、query set/order、warmup、ABBA、Cardinal 配置和 strict closure 与主曲线
一致。下表为 paired-window QPS delta；括号内是 95% CI。所有点 representation、
route、V/query distance attempts、topology 与结果校验均通过。

| N | 0.1% C1 / C60 | 0.4% C1 / C60 | 0.5% C1 / C60 | Dense auto route（0.1% / 0.4% / 0.5%） |
|---:|---:|---:|---:|---|
| 50K | +2.55%（+1.42,+3.50） / +0.78%（-1.17,+2.86） | +2.83%（+1.08,+4.49） / +0.26%（-1.54,+2.17） | +0.62%（-0.51,+1.69） / -0.15%（-2.07,+1.66） | BF / BF / BF |
| 100K | +4.98%（+2.99,+7.21） / +1.30%（-0.60,+3.17） | +9.23%（+7.52,+10.78） / +1.35%（-0.21,+2.93） | +8.13%（+7.02,+9.46） / +1.51%（-0.28,+3.67） | BF / IVF / IVF |
| 250K | +10.60%（+9.17,+12.00） / +4.24%（+1.20,+7.54） | +15.92%（+14.25,+17.40） / +3.69%（+1.47,+5.74） | +12.60%（+10.89,+14.22） / +3.82%（+1.41,+5.95） | BF / IVF / IVF |
| 1M | +49.78%（+47.71,+51.85） / +25.62%（+23.34,+28.32） | +27.20%（+25.11,+29.34） / +15.19%（+13.14,+17.10） | +18.66%（+16.73,+20.70） / +14.20%（+11.88,+16.55） | IVF / IVF / IVF |

该 hold-out 给出三点约束：

1. 相同 V/N 下收益随 N 增长，不存在可脱离 N 与 route 使用的单一比例常数；
2. 小 N/低 V 时 Dense auto 本身也走 BF，两种表示接近，收益趋近于零；
3. 本轮所有点的 QPS delta 下界均高于 -5%，且最大实测 V=5,000 仍为正收益，
   因此修复前 `V<=4,000 && V/N<=0.4%` 已过严，但尚未找到修复后的负向 crossover。

1M 三点是独立于主曲线的复测。C60 与主曲线相近；C1 的增益更高，说明低并发
仍存在跨轮基线漂移，因此最终 selector 不应直接贴着某一轮 C1 crossover 设置。
在继续向更高 V 扩展 Search 曲线并完成 Query 曲线前，不冻结新 policy。

原始产物：

- `artifacts/switchpoint-post-batch-fix-20260828/n-scale/c1/`
- `artifacts/switchpoint-post-batch-fix-20260828/n-scale/c60/`

##### Query threshold 刷新结果

随后在同一 1M collection 上移除 vector search，以 `count(*)` 单独验证 Query
consumer。这里的 V 是进入后续阶段前的最大 Sparse 中间集合：Single 为
`a < V`；AND-pass 为 `a < V AND b < N`；AND-reduce 为
`a < V AND b < N/4`，其最终 count 约为 V/4，但第一层仍产生 V 个 IDs。
三类均使用 `threshold=V`，其余保持 cache-off、30 秒/模式 warmup、12 个
DS/SD paired windows；C1 每 slot 50 requests，C60 每 slot 300 requests。

| Case | V | 返回 count | C1 QPS delta（95% CI） | C1 wins | C60 QPS delta（95% CI） | C60 wins |
|---|---:|---:|---:|---:|---:|---:|
| Single | 1,000 | 1,000 | +8.25%（+6.63,+9.85） | 12/12 | +6.37%（+4.55,+8.29） | 12/12 |
| Single | 4,000 | 4,000 | +6.42%（+4.95,+7.83） | 12/12 | +3.48%（+1.75,+5.33） | 11/12 |
| Single | 5,000 | 5,000 | +6.10%（+4.25,+8.32） | 12/12 | +3.61%（+1.51,+5.89） | 9/12 |
| AND-pass | 1,000 | 1,000 | +33.31%（+31.81,+34.97） | 12/12 | +17.57%（+15.38,+19.65） | 12/12 |
| AND-pass | 4,000 | 4,000 | +30.09%（+28.77,+31.34） | 12/12 | +16.47%（+15.17,+17.75） | 12/12 |
| AND-pass | 5,000 | 5,000 | +23.09%（+21.26,+25.23） | 12/12 | +13.02%（+10.01,+15.78） | 12/12 |
| AND-reduce | 1,000 | 247 | +34.42%（+32.56,+36.25） | 12/12 | +17.21%（+15.34,+18.98） | 12/12 |
| AND-reduce | 4,000 | 1,015 | +28.24%（+26.17,+30.21） | 12/12 | +16.00%（+14.10,+18.21） | 12/12 |
| AND-reduce | 5,000 | 1,239 | +24.81%（+21.93,+28.17） | 12/12 | +10.87%（+8.12,+13.26） | 11/12 |

95% CI 对 12 个 paired QPS delta 使用固定 seed=1732 的 percentile bootstrap。
18/18 点的 Dense/Sparse count 完全一致；每个 Sparse timed request 精确增加一次
Sparse-output counter，Dense 为零。修复后的 AND-reduce 已不再复现旧的 C60 负收益，
说明 batch candidate read 修复同时改变 Query switch curve。Single 没有后续 predicate
可减少工作，因而只保留输出/handoff 差异，收益稳定但小；两层 AND 可以让第二层直接
消费 V 个 IDs，收益更显著。Search 与 Query 的 consumer 工作不同，仍不合并成一个
endpoint 阈值结论。

原始产物：

- `artifacts/switchpoint-post-batch-fix-20260828/query-threshold/c1/`
- `artifacts/switchpoint-post-batch-fix-20260828/query-threshold/c60/`

##### Search crossover 加密

由于刷新后的 1M 主曲线到 0.8% 仍为正收益，又补测 V=9K/10K/12K/16K，
其余配置和 strict closure 不变：

| V | V/N | C1 QPS delta（95% CI） | C1 wins | C60 QPS delta（95% CI） | C60 wins |
|---:|---:|---:|---:|---:|---:|
| 9,000 | 0.90% | -1.20%（-1.56,-0.86） | 0/12 | +4.54%（+1.25,+7.49） | 11/12 |
| 10,000 | 1.00% | -3.84%（-5.39,-1.69） | 1/12 | +6.15%（+4.06,+8.30） | 12/12 |
| 12,000 | 1.20% | -8.60%（-9.64,-7.53） | 0/12 | -0.56%（-2.54,+1.13） | 5/12 |
| 16,000 | 1.60% | -18.77%（-20.00,-17.49） | 0/12 | -4.89%（-6.80,-2.74） | 0/12 |

所有点均为 Dense auto-IVF、Sparse direct-BF，Sparse distance attempts 精确等于
V/query，strict closure 通过。C1 crossover 位于 0.8%--0.9%；C60 的饱和点更晚，
位于约 1.0%--1.2%。按“准入点 paired 95% CI 下界不得低于 -5%”的门禁，1M 上
V=10K 的 C1 下界 -5.39% 已越界，而 V=9K 仍满足。因此当前可进入跨 N hold-out 的
候选是 `ratio_cap=0.9%`、`V_abs_safe=9000`，尚不是最终 policy。下一步需在
N=50K/100K/250K/1M 的 0.8%/0.9%/1.0%、C1/C60 验证该边界，尤其确认小 N 下
Dense auto-BF 与 Sparse BF 的等价成本以及 absolute V 的交互。

原始产物：

- `artifacts/switchpoint-post-batch-fix-20260828/search-crossover-extension/c1/`
- `artifacts/switchpoint-post-batch-fix-20260828/search-crossover-extension/c60/`

##### 跨 N crossover hold-out

为验证 1M crossover 能否外推，继续固定所有其它变量，在 N=50K/100K/250K
补测 0.8%/0.9%/1.0%、C1/C60；1M 直接复用同二进制的主曲线与 crossover
extension，不重复压测。下表为 paired QPS delta（95% CI）：

| N | 0.8% C1 / C60 | 0.9% C1 / C60 | 1.0% C1 / C60 |
|---:|---:|---:|---:|
| 50K | +3.58%（+1.00,+6.14） / +1.81%（+0.24,+3.57） | +4.91%（+2.70,+7.07） / -0.78%（-3.15,+1.14） | +4.30%（+2.67,+5.86） / +2.08%（+0.25,+4.16） |
| 100K | +6.32%（+4.88,+7.76） / +2.43%（+0.74,+4.45） | +5.95%（+4.36,+7.52） / +1.18%（-0.62,+3.04） | +5.69%（+4.02,+7.37） / +1.15%（-0.05,+2.46） |
| 250K | +6.99%（+4.92,+9.26） / +3.77%（+1.16,+6.14） | +8.82%（+7.80,+9.95） / +2.01%（-0.66,+4.71） | +6.80%（+5.60,+8.00） / +3.83%（+1.65,+6.10） |
| 1M | +6.02%（+4.46,+7.55） / +8.27%（+6.38,+10.22） | -1.20%（-1.56,-0.86） / +4.54%（+1.25,+7.49） | -3.84%（-5.39,-1.69） / +6.15%（+4.06,+8.30） |

18 个新增点均通过 strict closure；该比例区间的 Dense auto route 已统一为 IVF，
Sparse 为 direct-BF。最差准入候选点是 N=50K、0.9%、C60 的 -0.78%，CI 下界
-3.15%，仍在 -5% 门禁内；1M、1.0%、C1 的 CI 下界 -5.39% 已越界。因此进入
selector 与 T±1 验证的当前候选为：

```text
sparse eligible iff N >= 50,000
                && V / N <= 0.9%
                && V <= 9,000
```

三项需同时成立。`N>=50K` 是已验证下界而非理论拐点；`V<=9000` 防止比例阈值在
更大 segment 上把尚未验证的 BF distance work 无界放大。该候选还需实现为真实
per-segment selector，并执行 N/T 边界正确性与 endpoint 门禁，当前 request-level
threshold 结果本身不等同于 policy 已落地。

原始产物：

- `artifacts/switchpoint-post-batch-fix-20260828/n-scale-boundary/c1/`
- `artifacts/switchpoint-post-batch-fix-20260828/n-scale-boundary/c60/`

##### Candidate T−1/T/T+1 producer 边界

将 benchmark-only candidate 更新为 `ratio=0.9%`、absolute cap=9,000 后，停止
Milvus 并固定 CPU4，使用 5 次 randomized-interleaving repetitions 测试
Raw/STL_SORT/Bitmap、random/clustered、N=50K/100K/250K/1M/10M 的有效
`T_eff−1/T_eff/T_eff+1`。这不改变生产 selector。构建目录的 CMake cache 仍引用旧
KnowPR 绝对路径，但本轮直接依赖的 `AdaptiveFilterSink.h`、`Expr.h`、
`FilterBitmap.h`、`FilterResult.h` 在旧/当前路径 SHA256 完全一致，benchmark 主源
明确从当前 SparseProject 编译；相关 SHA 已固化在产物目录。

以下列 Adaptive median − Dense median 的绝对时间范围（两种 ID 分布），避免用很小
的 native baseline 放大百分比：

| Producer | N / T_eff | V≤T 额外成本 | V=T+1 额外成本 |
|---|---:|---:|---:|
| Raw | 50K / 450 | -41.0～-39.9 us | -41.2～-37.5 us |
| Raw | 1M / 9K | -1,199～-1,099 us | -1,208～-1,094 us |
| Raw | 10M / 9K | -56.63～-56.05 ms | -56.51～-56.01 ms |
| STL_SORT | 50K / 450 | -0.06～+0.70 us | +1.81～+5.50 us |
| STL_SORT | 1M / 9K | -0.87～+4.52 us | +31.90～+131.54 us |
| STL_SORT | 10M / 9K | -51.14～-41.73 us | +158.57～+672.92 us |
| Bitmap | 50K / 450 | +0.75～+0.77 us | +1.83～+5.54 us |
| Bitmap | 1M / 9K | +16.93～+28.73 us | +33.38～+134.48 us |
| Bitmap | 10M / 9K | -63.60～-52.81 us | +168.51～+673.73 us |

Raw 的 Dense 与 Adaptive producer 使用不同 batch sink 实现，负差值只能说明当前
实现 regime，不能解释为 Sparse 输出的普适单位收益。STL_SORT/Bitmap 的 T+1 则暴露
了可操作的缺口：native preflight 已经知道 posting/range 超 cap，但当前 fallback 仍把
已生成的 accepted Dense bitmap 交给 AdaptiveSink 再发现 T+1，增加一次 Dense bitmap
消费。它没有重跑 scalar predicate 或重新扫描原始列，但 N=10M 的 0.16--0.67 ms
绝对成本应在真实 selector 落地前删除：preflight 拒绝后直接返回语义等价 Dense
filtered bitmap即可。

功能边界同时通过 `AdaptiveFilterSinkTest.*` 7/7：T−1/T 保持 Sparse、T+1 单次
切 Dense、multi-batch prefix backfill、nullable、word tail 与 all-true/all-false
均正确。generic expression producer 尚未进入这张表，仍是 P1 未完成项。

原始产物：

- `artifacts/switchpoint-post-batch-fix-20260828/producer-boundary/results.json`
- `artifacts/switchpoint-post-batch-fix-20260828/producer-boundary/console.log`
- `artifacts/switchpoint-post-batch-fix-20260828/producer-boundary/binary.sha256`
- `artifacts/switchpoint-post-batch-fix-20260828/producer-boundary/header.sha256`

##### Native direct-Dense fallback 与 selector 落地

STL_SORT/Bitmap 的 native producer 增加 tri-state preflight：`Fits` 继续直接产出
Sparse IDs，`Exceeds` 直接复用 native Dense 结果，`Unsupported` 才进入通用
Adaptive sink。这样，已知 `V>T` 的路径不再为了发现 T+1 而额外消费一次完整
accepted bitmap。相同 benchmark 矩阵复测后，T+1 的额外成本已降至噪声级：

| Producer | N 范围 | direct-Dense 后 Adaptive median − Dense median |
|---|---:|---:|
| STL_SORT | 50K～1M | 绝大多数约 ±0.03 us |
| STL_SORT | 10M | random +2.58 us；clustered -0.28 us |
| Bitmap | 50K～250K | +0.05～+0.14 us |
| Bitmap | 1M | -0.65～+0.48 us |
| Bitmap | 10M | +0.38～+3.67 us |

原始结果：

- `artifacts/switchpoint-post-batch-fix-20260828/producer-boundary-direct-dense/results.json`

候选策略随后落为真实 per-segment selector：

```text
if N < 50,000:
    Dense
else:
    T_eff = min(request_cap, 9,000, floor(N * 0.009))
    V <= T_eff ? Sparse : Dense threshold fallback
```

请求级 `sparse_result_max_cardinality` 只能收紧 `9,000`；默认开关仍为 false，
因此未启用实例完全保持 Dense baseline。`floor(N*ratio)` 对十进制配置的二进制
表示误差加了远小于一行的 relative epsilon，确保 50K×0.9%=450 不会误算为 449。

selector 与参数传播已通过以下闭环：

- C++ 20/20：覆盖 `Nmin-1/Nmin/Nmin+1`、50K/100K/250K/1M/10M、ratio cap、
  absolute cap、request cap 收紧及 Adaptive/Bitmap/STL_SORT 回归；
- Go 参数单测：`TestSparseFilterResultConfig` 与动态刷新 callback 通过；
- E2E 使用的 `bin/milvus` SHA256 为
  `c09b649c5d806a0e4e58c07584717c7d2890fe0e235d23e097f2885b19e22e00`。
  E2E 后代码复核发现 request cap 原先可能放大全局 absolute cap；现已改为
  `min(configured_default, requested)`，并用 `global=1000/request=4096` 与
  `global=9000/request=4096` 两个方向固化“请求只能收紧”语义。重新完成
  `all_tests` 链接、C++ 20/20、两项 Go 定向测试和 `make build-go`；修复后
  `bin/milvus` SHA256 为
  `4f84fe5daf11480358a767d72cf22a08b54978a0b9aaad39140d2f3683b2d4f8`。
  现有 E2E request cap 均小于或等于全局 9,000，selector 结果不受该修复影响；
  artifact 同时保留 E2E 与修复后二进制 SHA，避免混淆证据身份。

Milvus 参数级 E2E 使用 Synthetic 128D/L2、Cardinal Tiered auto、单 sealed
segment、NQ=1/topK=10，并以 representation、route、predicate evaluation、BF
distance attempts 等 counter 做 strict closure。该短跑只验证 selector 边界，不作为
QPS 结论：

| N | V | T_eff | Sparse 请求最终表示 | Route | Strict closure |
|---:|---:|---:|---|---|---|
| 50K | 449 | 450 | Sparse | BF | PASS |
| 50K | 450 | 450 | Sparse | BF | PASS |
| 50K | 451 | 450 | Dense threshold | BF | PASS |
| 1M | 8,999 | 9,000 | Sparse | BF | PASS |
| 1M | 9,000 | 9,000 | Sparse | BF | PASS |
| 1M | 9,001 | 9,000 | Dense threshold | IVF | PASS |

50K 三点的 Dense/Sparse TopK 完全一致。1M 的 Sparse 点走 exact BF，而 Dense
auto 走 approximate IVF，因此 hit list 不要求完全相同；过滤结果合法性、实际表示、
route 和访问计数均闭合。V=9,001 fallback 后两种模式均走 Dense/IVF 且结果一致。

E2E driver 同步改为从运行时配置计算真正的 per-segment `T_eff`，而不是只按请求
cap 推断路径；另用 request cap=9,000 复核 50K 的 ratio 边界和 1M 的 absolute
边界，450/451 与 9,000/9,001 四点均 strict PASS。

原始产物：

- `artifacts/selector-e2e-20260828/n50k-v449.log`
- `artifacts/selector-e2e-20260828/n50k-v450.log`
- `artifacts/selector-e2e-20260828/n50k-v451.log`
- `artifacts/selector-e2e-20260828/n1m-v8999.log`
- `artifacts/selector-e2e-20260828/n1m-v9000.log`
- `artifacts/selector-e2e-20260828/n1m-v9001.log`
- `artifacts/selector-e2e-20260828/strict-effective-ratio-v450.log`
- `artifacts/selector-e2e-20260828/strict-effective-ratio-v451.log`
- `artifacts/selector-e2e-20260828/strict-effective-cap-v9000.log`
- `artifacts/selector-e2e-20260828/strict-effective-cap-v9001.log`

至此，candidate selector 的实现、边界单测、构建和 Milvus 参数级路径闭环均完成。
下一阶段不再改 selector 语义，优先补真实性能门禁：在固定 Cohere 1M×768D、C1/C60、
12 个 ABBA windows 的正式矩阵中复测准入点与 T+1 fallback，并把 Query consumer 与
Search consumer 分开报告。

##### Selector 正式性能门禁与候选收紧（2026-08-28）

使用 Cohere 768D/COSINE、Cardinal Tiered auto、单 sealed segment、NQ=1/topK=10，
每个模式 30 秒 warmup，固定 50 条 query 和 12 个 ABBA paired windows。所有点使用
同一 binary/query SHA，且 representation、route、distance attempts、topology 与结果
均通过 strict closure。该轮首先复测已实现的 `0.9% && V<=9000`：

| N/V | C | Dense QPS/route | Adaptive QPS/route | Paired QPS delta（95% CI） | wins |
|---:|---:|---:|---:|---:|---:|
| 1M/9,000 | 1 | 259.76/IVF | 235.21/BF | -9.35%（-10.35,-8.26） | 0/12 |
| 1M/9,000 | 60 | 1,410.45/IVF | 1,389.73/BF | -1.40%（-3.15,+0.24） | 5/12 |
| 1M/9,001 | 1 | 257.44/IVF | 262.73/Dense-threshold IVF | +2.09%（+0.82,+3.44） | 10/12 |
| 1M/9,001 | 60 | 1,425.84/IVF | 1,471.98/Dense-threshold IVF | +3.25%（+2.26,+4.28） | 12/12 |

T+1 direct-Dense fallback 与 baseline 同 route/work，未出现额外负担；但 V=9,000/C1
明确越过 -5% 门禁，因此旧候选不能冻结。保持其它变量不变向下加密 C1：V=8,000
为 -4.85%（CI 下界 -6.10%），V=7,000 为 -1.38%（下界 -2.28%），V=6,000
为 +4.35%（+3.07,+5.87）。随后以 `V/N=0.6%` 做跨 N/C hold-out：

| N | V | C1 QPS delta（95% CI） | C60 QPS delta（95% CI） | Dense/Sparse route |
|---:|---:|---:|---:|---|
| 50K | 300 | +1.39%（-0.83,+3.64） | +0.06%（-0.67,+0.67） | BF/BF |
| 100K | 600 | +5.16%（+3.52,+6.70） | +1.20%（+0.31,+2.21） | IVF/BF |
| 250K | 1,500 | +8.06%（+5.83,+10.38） | +2.16%（+0.46,+3.87） | IVF/BF |
| 1M | 6,000 | +4.35%（+3.07,+5.87） | +3.27%（+1.85,+4.72） | IVF/BF |

8/8 点的 CI 下界均高于 -5%；小 N 上 Dense 自身也走 BF，结果趋于等价，100K
以上均为正收益。因此当前候选收紧为：

```text
sparse eligible iff N >= 50,000
                && V / N <= 0.6%
                && V <= 6,000
```

本轮 runtime 的全局上限仍为 9,000/0.9%，通过 request cap 单变量收紧到各测试点；
其有效 `T_eff` 与新候选完全相同。代码默认值随后同步改为 6,000/0.6%，feature
开关仍默认 false。收紧后重新完成 C++ 20/20、两项 Go 配置测试和 `make build-go`；
最终 `bin/milvus` SHA256 为
`0a7eb217a42df934eb8e3421256cad4da7545d6a31e44940e37bd1c530c9d33c`。
原始产物见 `artifacts/selector-performance-gate-20260828/`。
