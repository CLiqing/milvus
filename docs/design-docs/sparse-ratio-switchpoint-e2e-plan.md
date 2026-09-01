# Adaptive Sparse 比例阈值 Milvus E2E 实验计划

日期：2026-08-31

## 1. 目标

本实验重新确定 Adaptive Sparse 的切换阈值。此前先在 `N=1M` 找到候选点、
再仅以 `V/N=0.6%` 横跨多个 `N` 验证，不能证明不同数据规模上的切换点一致。
本轮在多个单 segment collection 上分别扫描 `V/N`，回答：

1. 在 `N >= N_min` 时，不同 `N` 的 Dense/Sparse QPS 交点是否落在接近的
   `V/N` 区间；
2. 能否仅使用 `N_min + max(V/N)` 作为 selector，而不再引入绝对 `V` 上限；
3. Dense 的 Cardinal Auto route 发生 BF/IVF 切换时，性能曲线如何变化。

本轮不把绝对 `V` 作为独立决策变量。若结果显示相同比例在不同 `N` 上呈现
系统性相反结论，再单独设计绝对 `V` 实验。

## 2. 变量与固定条件

### 2.1 主矩阵

| 变量 | 取值 |
|---|---|
| `N` | 50K、100K、250K、1M、3M |
| `V/N` | 0.05%、0.1%、0.2%、0.3%、0.4%、0.5%、0.6%、0.7%、0.8%、0.9%、1.0% |
| 客户端并发 | C1、C60 |
| 过滤输出 | Dense、Adaptive Sparse |

每个 `(N, V/N, C)` 测点只切换 `filter_result_representation=dense/sparse`。
`V=floor(N*ratio)`，scalar 值使用固定 seed 随机排列，使 accepted rows 在
segment 内均匀分布并得到精确 `V`。

### 2.2 固定条件

- Cohere 768D，COSINE；不同 `N` 使用 Cohere 10M 训练集的相同前缀；
- CARDINAL_TIERED，Cardinal Auto，NQ=1，topK=10；
- 相同的 50 个 query、顺序及 query-set SHA；
- scalar producer、predicate 形状、index 参数、二进制 SHA 保持一致；
- expression cache 关闭，`queryNode.grouping.maxNQ=1`；
- 同一机器、同一 Milvus 实例配置，测试期间不构建索引、不 compact；
- Dense 使用 Cardinal 原生 Auto selector；Sparse 按当前方案 direct BF。

当前生产候选阈值会使大比例测点回退到 Dense。实验实例临时放宽全局
`maxRatio >= 1%`，并将请求 cap 提高到覆盖 `3M * 1% = 30K`，但不修改正式
默认值。每个测点必须通过 counter 证明 Sparse 实际输出 Sparse，而不是
threshold-Dense fallback。

## 3. 单 segment 强制门禁

每个 `N` 使用独立 collection，目标拓扑为一个包含全部 `N` 行的 sealed segment。

1. 写入完成后 flush；
2. 如果 persistent/loaded sealed segment 不是一个，发起 manual compaction；
3. 等待 compaction 完成、旧 segment 被替换，并重新 load；
4. 等待 Cardinal index build、load 和后台 compaction 全部稳定；
5. 测量前后都断言：`loaded_segment_count=1`、`loaded_segment_rows=[N]`，且
   segment identity 未变化。

若 force merge 后仍无法形成一个 segment，该 `N` 不允许进入性能统计，不能把
多 segment 数据混入主图。为容纳 3M×768D，实验实例的内存/磁盘 segment size
均需高于原始向量约 9.2 GB，数据目录放在 `/data/nvme`。

## 4. 路径、正确性与测量闭环

每个测点记录并校验：

- 实际 `N`、`V`、`V/N`；
- 最终 Dense/Sparse representation；
- Dense/Sparse 的实际 BF/IVF/Graph route；
- BF distance attempts 及 Cardinal route/work counters；
- runtime config、collection/index 类型、segment 拓扑；
- TopK 合法性；同 route 时要求结果完全一致，不同 route 时以 Sparse BF 为
  reference 记录 Dense recall/overlap；
- 测量前后 topology 与 runtime config 稳定。

性能口径使用 Milvus endpoint QPS。正式测量采用每 mode 30 秒 warmup、12 个
ABBA paired windows；每个 slot 固定执行每 worker 50 个 query。主矩阵中的
`5 N × 11 ratios × 2 concurrencies = 110` 个测点必须全部执行这一正式口径；
每个测点内部都包含 Dense/Sparse 两种 mode 的 ABBA 对照，因此共覆盖 220 个
mode-cell。发现轮只可用于调试脚本和预览趋势，不得替代、删减或选择正式测点。

## 5. 展示方式与阈值判定

主图按 `N` 分为五张子图。横轴为 `V`，同时标注对应 `V/N`；每个横坐标放置
四根相邻柱：Dense C1、Sparse C1、Dense C60、Sparse C60，纵轴为 QPS。

Dense 柱按实际 Auto route 区分颜色或纹理，并直接标记 `BF`/`IVF`，避免把
`Dense-IVF vs Sparse-BF` 误读为同一路径的表示收益。另附一张 delta 图，分别
展示 C1/C60 的 `Sparse QPS / Dense QPS - 1`，标出 0% 与 -5% 参考线。

对每个 `N` 找到 Sparse 相对 Dense 越过允许劣化线的比例区间，并比较这些区间
是否一致。只有各 `N` 的结果和置信区间共同支持时，才给出统一比例阈值；否则
保留实验事实，不通过单点结果拟合阈值。

## 6. 执行阶段

| 阶段 | 内容 | 验收条件 | 状态 |
|---|---|---|---|
| A | 补齐跨 Cohere shard 的单 segment loader、manual compaction 与单 segment hard gate | 3M 可加载；非单 segment 会 force merge 或拒测 | 完成 |
| B | 运行 50K/100K/250K/1M/3M 全比例发现轮 | 每点 representation、route、拓扑与结果闭环 | 完成，110/110 点通过 |
| C | 对完整主矩阵执行正式 ABBA 测试 | 110/110 点均为 30s warmup、12 windows、95% CI；不允许选择性删点 | 完成，110/110 |
| D | 用完整 110 点正式结果绘制五子图 QPS 图和 delta 图 | 每个 `N` 展示全部 11 个比例；Dense route 标识完整 | 完成 |
| E | 基于完整矩阵更新 selector 结论 | 同时满足 C1/C60 的 -5% CI 门禁后才可推荐阈值 | 完成 |

## 7. 发现轮结果与正式复测选择

发现轮使用 2 个 ABBA window，仅用于定位曲线。110 个测点全部满足：同一 query
SHA、单 sealed segment、稳定拓扑、Dense/Sparse representation 正确及实际 route
counter 闭合。1M 写入后的 `330K*3+10K`、3M 写入后的 `330K*9+30K` 均由 manual
compaction 分别合并为一个 1M/3M segment 后才开始测量。

主要发现：

- 50K、100K、250K 在 0.05%～1% 范围内没有出现稳定的低于 -5% 边界；
- 1M 的 C1 从 0.6% 附近正收益转向 0.7%～0.8% 的劣化；
- 3M 的 C1 在 0.4% 接近持平、0.5% 开始低于 -5%，C60 的边界略靠后；
- Dense Auto 会随 `N/V` 从 BF 切到 IVF，Sparse 保持 direct BF，主图必须标记
  Dense route，不能把所有柱理解为同一搜索算法。

正式轮必须对所有 `N` 执行相同的完整比例列表，每点同时覆盖 C1/C60：

```text
0.05%, 0.1%, 0.2%, 0.3%, 0.4%, 0.5%,
0.6%, 0.7%, 0.8%, 0.9%, 1.0%
```

不得因为发现轮显示某些点远离交点而跳过正式测量。这样最终每张 `N` 子图都必须
有 11 组横坐标，每组包含 Dense C1、Sparse C1、Dense C60、Sparse C60 四根柱。

发现轮原始记录位于
`artifacts/sparse-ratio-switchpoint-20260831/discovery-summary.jsonl`，五子图位于
同目录的 `plots/`。正式轮不得使用发现轮的 2-window 数值替代置信区间结论。

## 8. 完整正式矩阵结果

正式轮完成全部 110 个 `(N, V/N, C)` 测点。每个测点采用 30 秒/mode warmup、12 个
ABBA paired windows 和固定 50-query workload，并对 paired-window QPS delta 做
10,000 次 bootstrap，报告 95% CI。

全量审计结果：

- 110/110 `strict_closure_passed=true`，无缺失、重复或额外测点；
- 每个 `N` 均为 22/22 点，即 11 ratios × C1/C60；
- 110/110 测量前后 topology 稳定，且均为 `[N]` 的单 sealed segment；
- 110/110 使用相同 query SHA：
  `3dbf59b372d816c2a260e68dbf49c899b1799d3777ce4648e00d9ae4fb1426de`；
- 110/110 均为 Dense representation 对 Sparse representation；
- Dense route 为 82 个 IVF、28 个 BF，Sparse 110 个点均为 BF；
- 不同 route 时结果闭环按 Sparse BF reference 检查 Dense IVF 的 recall/overlap，
  因而性能数字必须理解为 Cardinal Auto 的完整 E2E 比较，而非固定搜索算法微基准。

下表摘录统一候选和各规模边界附近的正式结果；完整 110 点见正式图和原始汇总。
数值为 `Sparse QPS / Dense QPS - 1`，方括号内为 95% CI：

| N | V/N | C1 delta [95% CI] | C60 delta [95% CI] | Auto route（Dense → Sparse） |
|---:|---:|---:|---:|---|
| 50,000 | 0.4% | -0.14% [-2.99%, +2.47%] | -0.51% [-2.99%, +1.94%] | BF → BF |
| 50,000 | 1.0% | +2.99% [+0.77%, +5.13%] | -0.41% [-2.53%, +1.64%] | IVF → BF |
| 100,000 | 0.4% | +6.98% [+5.36%, +8.73%] | +0.78% [-1.17%, +2.95%] | IVF → BF |
| 100,000 | 1.0% | +5.05% [+2.62%, +8.16%] | +0.69% [-0.37%, +1.75%] | IVF → BF |
| 250,000 | 0.4% | +11.22% [+8.67%, +13.63%] | +2.40% [+0.86%, +3.72%] | IVF → BF |
| 250,000 | 1.0% | +2.03% [+0.19%, +3.88%] | +1.99% [-0.02%, +4.19%] | IVF → BF |
| 1,000,000 | 0.4% | +11.35% [+9.13%, +13.02%] | +7.43% [+6.48%, +8.64%] | IVF → BF |
| 1,000,000 | 0.6% | +1.07% [-0.55%, +2.65%] | +3.31% [+1.28%, +5.54%] | IVF → BF |
| 1,000,000 | 0.7% | -2.93% [-3.66%, -2.16%] | +0.73% [-0.24%, +1.58%] | IVF → BF |
| 1,000,000 | 0.8% | -4.36% [-5.85%, -2.42%] | +0.20% [-1.53%, +1.55%] | IVF → BF |
| 3,000,000 | 0.3% | +8.76% [+7.88%, +9.74%] | +18.20% [+17.56%, +18.87%] | IVF → BF |
| 3,000,000 | 0.4% | -0.28% [-0.81%, +0.15%] | +9.05% [+8.49%, +9.60%] | IVF → BF |
| 3,000,000 | 0.5% | -7.16% [-8.02%, -6.30%] | +2.51% [+2.11%, +2.91%] | IVF → BF |
| 3,000,000 | 0.6% | -12.57% [-13.21%, -11.96%] | -3.16% [-3.50%, -2.81%] | IVF → BF |

正式图中每个 `N` 都包含全部 11 个比例，每个比例包含四根柱；Dense IVF 使用斜线标记：

![正式确认 QPS 对比 C1](../../artifacts/sparse-ratio-switchpoint-20260831/formal/plots/sparse-ratio-qps-c1-by-n.png)

![正式确认 QPS 对比 C60](../../artifacts/sparse-ratio-switchpoint-20260831/formal/plots/sparse-ratio-qps-c60-by-n.png)

![正式确认 QPS delta](../../artifacts/sparse-ratio-switchpoint-20260831/formal/plots/sparse-ratio-qps-delta-by-n.png)

## 9. Selector 结论

完整正式矩阵支持以下保守候选：

```text
Sparse eligible := N >= 50,000 && V/N <= 0.4%
```

依据如下：

1. `0.4%` 在全部五个 `N`、C1/C60 上，95% CI 下界均未越过 -5% 门禁；
2. 最严格的 3M 测点在 `0.4%` 为 C1 -0.28%、C60 +9.05%，但到 `0.5%`
   时 C1 已显著劣化至 -7.16%，CI 上界也只有 -6.30%；
3. 1M 的边界更晚，0.7% 仍通过、0.8% 的 C1 CI 下界才越过 -5%，因此统一
   阈值由 3M 决定；
4. 50K 的 0.4% 基本持平，说明 `N_min=50K` 不制造显著回退；小于 50K 未纳入
   本轮验证，应继续使用 Dense 默认路径。

完整 sweep 进一步确认：50K、100K、250K 在 1% 内均未越过 -5% CI 门禁；1M
从 0.8% 开始失败；3M 从 0.5% 开始失败。因此统一比例仍由 3M 的 0.4%/0.5%
边界决定。阈值决策只使用 `N_min + V/N`，本轮不引入绝对 `V`。

正式原始汇总位于
`artifacts/sparse-ratio-switchpoint-20260831/formal/formal-summary.jsonl`，各测点完整
日志位于同目录的 `formal-n*-r*pct-c*.log`。
