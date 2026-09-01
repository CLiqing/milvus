# Sparse Filter 本周工作简报

> 日期：2026-08-21  
> 本周目标：完善 Sparse filter 的 Milvus E2E 链路，并验证不同 Cardinal 路径下的收益与风险。

## 1. 主要工作

本周将 Sparse 从 Cardinal BF 的实验扫描模式推进为 Milvus 过滤链路中的独立结果表示，完成了
`predicate A -> Sparse IDs -> predicate B -> Sparse IDs -> vector search` 的基本贯通。
链路已覆盖复合 AND、MVCC、delete、TTL、nullable、growing 和 multi-segment；Cardinal BF
直接枚举 Sparse IDs，Graph/IVF 使用 exact Bloom+Flat membership，只有 legacy raw-BF
兼容边界转回 Dense。

正确性方面，Sparse 与 Dense 的 topK `(ID, distance)` 在 sealed multi-segment、delete、
TTL、nullable 和 growing E2E 中保持一致；针对历史 MVCC、future insert、STL_SORT row-ID
顺序和复合过滤的定向单测为 **9/9 PASS**。

本周还验证了一个轻量的阈值回退方案：producer 先保存最多 1,000 个 Sparse IDs，超过
阈值后立即停止 Sparse 枚举并 fallback 到 Dense。STL_SORT 与 raw-data 的 V=5,000/10,000
对照中，回退开销在 **-3.5%～+2.6%** 的运行波动范围内，说明 early-break 可以将试探
Sparse 的额外工作限制在较低水平。

## 2. 主要性能结果

| 测试层级 | 路径与配置 | Sparse 相对 Dense |
|---|---|---:|
| Milvus E2E | `CARDINAL_TIERED`，explicit BF，1M×128D，最终 V≈500 | mean **-13.62%**，median -23.54% |
| Cardinal direct `Index::Search` | Memory，normal auto selector 实际进入 BF，1M×128D，V=500 | **-65.33%** |
| Cardinal direct `Index::Search` | Disk，normal auto selector 实际进入 BF，1M×128D，V=500 | **-65.25%** |
| Producer 阈值回退 | 先保存 1,000 IDs，超过阈值 early-break 并 fallback Dense | **-3.5%～+2.6%**，无显著额外开销 |

Memory/Disk 是 Cardinal 的内存型和本地磁盘型索引路径，本轮结果为绕过 Milvus
filter producer、MVCC 和 RPC 的 direct-Index 测试；两组 Milvus E2E 仍使用
`CARDINAL_TIERED`，其内部对应 Cardinal `ObjectStore` 路径。因此 direct BF 的约 65%
收益不能直接等同于 Milvus endpoint 收益，但证明了 Sparse ID 枚举在 Cardinal BF
consumer 内的优化有效。

另外补测了 Tiered natural auto-BF：将 V 降至 64，使 `ef=64 >= V` 后由正常 selector
自动进入 BF。该点 Sparse mean +5.00%、median -17.12%、P90 +117.21%，说明极小 V 下
上层固定成本和尾部波动可能抵消 BF consumer 的收益，不能仅凭“已进入 BF”判断 endpoint
一定改善。

## 3. 路由与测试口径贡献

- 明确 Memory/Disk 高过滤时会通过 filter-rate 策略自动切换到 BF；Tiered/ObjectStore
  不使用同一规则，只在 `ef`、`topK` 或 `max_codes` 覆盖全部 valid IDs 时转 BF。
- 修正了此前把 Tiered auto/IVF 的约 -3% endpoint 结果解释成 BF 收益的问题；后续性能
  对照均要求通过 route counter 或 perf 调用栈证明 Dense/Sparse 进入同一路由。
- 统一采用固定 query、预热、ABBA、topK 正确性校验和稳定 collection 后采集 perf 的实验口径。

## 4. 当前问题与下一步

Sparse 仍存在明显的 P90 重尾，慢请求主要落在第二个过滤谓词的 `b_read` 执行区间。
目前只能确认尾部发生于该区间，尚不能确认是 O(V) 随机读取本身过慢，还是线程调度、
off-CPU 或 page-fault 停顿被 wall-clock 计时计入。下一步将联合记录 wall time、thread CPU
time、context switch 和 page fault，先闭合根因，再决定优化 all-match 快路径、重复
validation/grouping、index/cache membership，或运行时调度。
