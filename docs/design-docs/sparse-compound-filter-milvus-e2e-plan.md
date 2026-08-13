# Sparse Compound Filter to Cardinal: Milvus E2E Plan

## Objective

Validate one end-to-end product-path hypothesis: when the first conjunctive
scalar predicate produces a very small accepted-row set, Milvus can retain that
set as a query-owned Sparse valid-ID representation, pass it to the second
predicate, and finally pass the compacted result to Knowhere/Cardinal.  The
comparison is against the existing Dense `TargetBitmap` pipeline with identical
logical predicate and visibility semantics.

This is a **Milvus request E2E** experiment.  Its boundary is a client
`pymilvus.search()` call through QueryNode/SegCore/Knowhere/Cardinal, including
filter execution, MVCC/delete/TTL handling, vector search, and RPC.  vecTool
and direct Cardinal tools remain useful only for isolated data-structure and
consumer research; they are not evidence for this plan's endpoint result.

## Intended execution path

```text
Dense baseline:
  A predicate -> Dense bitmap -> B predicate with bitmap input
  -> Dense bitmap -> MVCC -> BitsetView -> auto vector route

Sparse candidate path:
  A predicate -> Sparse valid IDs -> B Sparse-to-Sparse consumer
  -> Sparse valid IDs -> MVCC compaction -> owned valid-ID BitsetView
  -> auto vector route
```

Only one representation is produced per request.  Sparse must not materialize
a Dense bitmap as an intermediate or fallback.  A Dense fallback is outside
this experiment; unsupported expression/segment/visibility cases retain the
existing Dense implementation.

## Scope and initial implementation

The current `valid_ids_per_query` bridge has native producers only at scalar
leaf expressions, while `PhyConjunctFilterExpr` passes a Dense bitmap to its
next child.  The initial extension is deliberately narrow:

- sealed, row-level segments;
- `AND` of two supported INT64 unary/binary range expressions;
- first child (A) produces accepted row IDs in ascending segment-offset order;
- second child (B) consumes those IDs directly, pins each involved scalar
  chunk once, and appends matching IDs directly to the output list;
- `PhyMvccNode` compacts that final list with the normal timestamp/delete/TTL
  invalid mask; and
- `PhyVectorSearchNode` passes the final owned list through the existing
  `BitsetView::FromOwnedValidIdList` bridge.

The Dense mode must keep the current executor unchanged.  The Sparse mode is
selected by an experiment-only filter-result representation parameter, not by
constructing both outputs.  The implementation must record/verify that A is
executed before B; query text order alone is insufficient because conjunctions
may be reordered.

## Dataset and workload

Use deterministic random fp32 vectors and two independent INT64 scalar
columns.  The main point is ten 1M-row sealed segments; the 1M point is a
smoke/debug scale point.

| Variable | 1M smoke | 10M main point |
|---|---:|---:|
| Rows / segments | 1M / 1 | 10M / 10 x 1M |
| Vector data | fp32, dim 128, deterministic seed 1732 | same |
| Vector index | `CARDINAL_TIERED`, L2, topK=10, ef=64 | same |
| A column / predicate | random-permuted INT64 rank; `a < 1000` | 1,000 valid per segment; `a < 1000` |
| A selectivity | 0.1%, 1,000 valid | 0.1%, 10,000 valid total |
| B column / predicate | independent rank; `b < 500000` | same |
| B conditional selectivity | about 50% of A candidates | same |
| Final valid count | about 500 | about 5,000 |
| Scalar indexes | none for the primary point; same raw/vectorized producer | none |
| Query set | 50 fixed seed-1732 queries | same |

The initial all-visible performance point has no delete and TTL=0 so it is
stable.  Separate correctness tests must cover delete, historical timestamp,
and TTL compaction before claiming the Sparse payload preserves visibility.

### Cohere 10M high-dimensional follow-up (scheduled)

The synthetic 128D point establishes the direct Sparse compound-filter
pipeline, but vector-distance work is artificially small.  The next E2E point
uses vdbbench's standard **Large Cohere (768D, 10M)** corpus: ten
`shuffle_train-XX-of-10.parquet` files and the corresponding `test.parquet`
queries, with `COSINE` metric.  It replaces only the vector base/query data;
the comparison, Cardinal index parameters, NQ=1/C1 ABBA protocol, and logical
filter stay fixed.

Each Cohere input file is an independently flushed 1M-row *logical ingestion
unit*.  Milvus may still split it into multiple physical sealed segments by
channel; it is not assumed that a Parquet file equals a physical segment.
Within every ingestion unit, `a` and `b` are independent deterministic
permutations of `[0, 1M)`: `a < 1000` gives exactly 1,000 A candidates, and
`b < 500000` retains about half of them.  Therefore every physical segment
receives an approximately uniform sample rather than a file-boundary or
segment-boundary run of matching rows.  Expected final candidates are about
500 per ingestion unit / 5,000 collection-wide.  The input is read as
streaming Parquet batches; all 10M 768D vectors are never resident in the
Python client at once.

This point is reported separately from the 1M x 128D synthetic result because
dataset, dimension, and metric differ.  Its purpose is to quantify how much
the Sparse pipeline's filtering benefit remains once realistic high-dimensional
COSINE distance work accounts for more of endpoint latency.

#### Cohere ingestion and route smoke (2026-08-12)

The official `test.parquet` was validated as 1,000 768D fp32 query vectors.
The first official training Parquet shard was streamed into the isolated
standalone as a 1M x 768D/COSINE collection; no synthetic vectors or queries
were used.  The standard topK/distance equality closure passed before timing.
A short one-window diagnostic (two 10-query slots per mode, NQ=1/C1,
topK10/ef64, explicit BF payload modes) gave:

| Metric | Dense per-query BF | Direct Sparse valid-ID BF | Sparse delta |
|---|---:|---:|---:|
| Mean endpoint latency | 3.427 ms | 2.914 ms | -14.95% |
| Median endpoint latency | 3.259 ms | 2.748 ms | -15.70% |
| P90 endpoint latency | 3.479 ms | 3.266 ms | -6.12% |

This is an ingestion/COSINE/route smoke, not the 12-window acceptance result.
All ten official training shards were subsequently downloaded and validated as
1M-row, 768D Parquet files (about 44GB total).  A disk preflight then found
the isolated standalone's accumulated historical experiment state occupied
about 78GB while the root filesystem had about 144GB free.  Full 10M creation
must start from reclaimed or fresh isolated storage; otherwise its raw binlogs
and Cardinal index have a material risk of exhausting the remaining space.

#### Full Cohere 10M acceptance run (2026-08-12)

After reclaiming obsolete local experiment collections, traces, logs, and
unrelated build trees, the complete official Cohere corpus was inserted in ten
streaming 1M-row logical units.  The per-unit disk guard was 40GiB; its lowest
observed post-flush free space was 107.96GiB, and the final completed run had
about 98GiB free.  No guard, OOM, or ingestion error fired.  The script built
and loaded `CARDINAL_TIERED` with COSINE, completed exact Dense/Sparse topK
and distance closure for the first ten fixed Cohere queries, warmed each mode
with ten requests, then completed the prescribed twelve
`Dense -> Sparse -> Sparse -> Dense` windows (50 fixed NQ=1/C1 requests per
slot, hence 1,200 timed requests per mode).

| Metric | Dense per-query BF | Direct Sparse valid-ID BF | Sparse delta |
|---|---:|---:|---:|
| Mean endpoint latency | 9.294 ms | 8.258 ms | -11.15% |
| Median endpoint latency | 8.980 ms | 7.402 ms | -17.57% |
| P90 endpoint latency | 10.236 ms | 11.978 ms | +17.02% |
| Paired ABBA windows faster | - | 12 / 12 | consistent direction |
| Mean paired-window delta | - | - | -11.08% |

This is the accepted high-dimensional product-path result: real Cohere vectors
and queries, COSINE, scalar A/B production, visibility processing,
Knowhere/Cardinal BF, RPC, and response are all included in the endpoint timer.
The P90 trade-off remains visible even though the mean/median and every paired
window favor Sparse.  It must be reported alongside the mean rather than
presented as a uniform tail-latency improvement.

## Route policy

The primary product question is whether normal route selection would choose BF
for the final high-filter result.  Before timing, run a **Dense auto-route
preflight** and record the actual BF/Graph counters and route marker.

### Current bridge constraint

Today `bf_filter_scan_mode=valid_ids_per_query` is not merely a result-format
selector: Cardinal defines it as an explicit per-query BF mode and dispatches
it directly to BF.  A Sparse valid-ID list therefore cannot yet participate in
the normal auto dispatcher.  The initial E2E is consequently valid only when
the Dense auto-route preflight proves the same final filter would naturally
select BF; the representation comparison then uses the existing explicit
Dense-per-query / valid-ID-per-query BF modes.  This is a verified-natural-BF
experiment, not a claim that Sparse itself has completed generic auto-route
support.

Supporting a true Sparse auto route is a separate required extension: split
filter-result representation from `BfFilterScanMode`, let Cardinal select the
normal searcher using the same cardinality, and provide a non-BF Sparse
consumer (for example list -> Bloom+Flat) when Graph/IVF is selected.  It is
not silently converted to Dense in this plan.

- If Dense auto naturally routes to BF, run the Sparse pipeline through the
  existing valid-ID BF bridge and report it as a *verified-natural-BF* result.
- If Dense auto routes to Graph, do not force BF as the headline experiment;
  stop and either implement Sparse auto-route support or report Graph as a
  separate compatibility result.
- Any `index_algo=BF` run before the preflight passes remains a diagnostic,
  not the headline E2E result.

## Measurement protocol

All endpoint measurements are local standalone Milvus calls.  Each RPC has
`NQ=1`; expression cache is disabled for both paths.  The primary point uses
C1.  C60 is a later, separate workload and must record QueryNode grouping
configuration rather than being mixed with C1.

1. Verify Dense and Sparse have identical A IDs, final IDs, and all returned
   `(topK ID, distance)` values for the fixed smoke query set.
2. Record per-segment A/B input-row and output-ID counts, final payload bytes,
   MVCC removals, selected route, and BF distance attempts or Graph probes.
3. Warm up 10 requests per mode.
4. Run 12 `Dense -> Sparse -> Sparse -> Dense` ABBA windows.  Each slot uses
   the same 50 fixed queries, so each mode receives 1,200 endpoint requests.
5. First run a Dense A/A baseline with the identical scheduling protocol.  Do
   not call an endpoint difference smaller than its A/A variation a result.

Endpoint latency is client-side `perf_counter_ns` around one
`pymilvus.search()` call and includes RPC, scheduler, scalar evaluation,
payload construction, MVCC, Knowhere/Cardinal, and response.  Aggregate QPS
at C>1 is completed requests divided by mode makespan, not the inverse of mean
endpoint latency.

Request/segment-scoped phase counters/timers are explanatory data, not terms
to be summed into endpoint latency: A time and rows evaluated, B time and rows
evaluated, payload bytes/ID count, MVCC removals, vector-search time, and BF
distance attempts or Graph probes.  This separation prevents an endpoint
delta from being incorrectly attributed to a single stage.

## Acceptance and non-goals

The implementation is correct only if Dense/Sparse final candidate sets and
search results are exact matches under the supported visibility cases.  A
performance claim requires same route, same query/configuration, closure
checks, A/A context, and repeatable ABBA direction.

This plan does not attempt to prove that Sparse exact membership improves
Graph/IVF.  If auto selects Graph, the result is reported faithfully; any
forced-Graph or forced-BF follow-up is a separate diagnostic.  It also does
not choose a production cardinality threshold or add Sparse expression-cache
support; those decisions wait for the measured producer-plus-consumer result.

## Execution log (2026-08-11)

The local standalone smoke environment was rebuilt from this worktree's Core
library and run on isolated ports and storage.  On the deterministic 1M x
128D dataset, a Dense request with `bf_filter_scan_mode=auto` issued 10 fixed
queries.  Cardinal recorded 50 BF searches and zero Graph searches: Milvus
sealed the inserted rows into five physical segments, so each request produced
five segment-level route records.  This meets the *verified-natural-BF*
preflight condition, although the physical segment count must be controlled
more tightly for the planned 1M/one-segment smoke point.

The first explicit Dense-per-query versus native Sparse valid-ID ABBA runs
are diagnostic only, not accepted endpoint data.  Dense was about 3.1 ms
mean; Sparse had a lower median (about 2.7--2.8 ms) but recurring 80 ms-class
tail samples, which dominated its mean.  Repeating after an additional
warm-up did not remove the tail.  Before recording any performance conclusion,
collect a request-level trace and Core/Cardinal phase counters to attribute
those tails, then rerun under an explicitly controlled sealed-segment layout.

### Perf attribution and corrected consumer design

`sudo perf record` and a system-wide `perf sched` trace attributed the
225 ms-class Sparse tail to the second scalar predicate, before Cardinal BF.
The requesting Search worker slept for about 225 ms while the worker executing
the predicate ran for about 223 ms; its maximum runqueue delay was only
0.109 ms.  Samples throughout that interval stayed in this call chain:

```text
ProcessDataByOffsets
  -> get_chunk_by_offset / CapturePublishedState
  -> ChunkedColumnGroup::GetGroupChunk
  -> CacheSlot::PinCells / Future::get
  -> GroupChunk::GetChunk
  -> shared_ptr release / allocator
```

The prototype reused the generic offset evaluator.  For every one of A's
roughly 1,000 candidate IDs it independently located, pinned, accessed, and
released the same scalar chunk, then materialized a V-bit result and scanned
that result to rebuild a list.  This defeats the intended Sparse cost model.

The corrected implementation adds a direct Sparse consumer contract.  Sparse
IDs are ascending, unique, segment-local row offsets.  A supported leaf:

1. walks the ordered IDs once and groups them by scalar chunk;
2. performs one skip-index decision and one pin per involved chunk;
3. evaluates the predicate directly against `chunk[local_offset]`, including
   nullable-row validity; and
4. appends surviving original row IDs directly to a reserved output vector.

No Dense or V-bit intermediate is constructed.  For the controlled one-chunk
case the expected access count changes from about 1,000 `PinCells` operations
to one.  Unsupported leaves, unordered/duplicate/out-of-range IDs, growing
segments, and execution paths without the required raw data return unsupported
and preserve the existing Dense executor.  The first implementation covers
sealed row-level INT64 unary and binary range leaves; broader types remain a
correctness-preserving fallback rather than entering the generic per-ID path.

The post-change E2E must hold dataset, query order, route, cache state, NQ,
concurrency, compiler, and segment layout fixed.  Acceptance requires exact
final IDs/topK closure, removal of the 225 ms tail, and repeatable ABBA results;
the lower Sparse median alone is not sufficient.

### Controlled-layout follow-up

Setting `dataCoord.segment.maxSize=8192`, `sealProportion=0.99`, and jitter to
zero produced exactly one sealed 1M-row segment.  This removed segment fan-out
as a confounder but did **not** remove the Sparse tail: in a one-window
request-level trace, Dense was 4.53 ms mean / 4.49 ms median, whereas native
Sparse was 3.41 ms median but 30.07 ms mean, with 225 ms-class requests only
on `valid_ids_per_query`.  The tail is therefore not a Graph route or
multi-segment aggregation effect.  The next attribution run must capture
per-request time around native producer, second-predicate offset evaluation,
MVCC compaction, `FromOwnedValidIdList`, and Cardinal BF, alongside the
client endpoint time.

### Direct Sparse consumer implementation and validation

The generic `Eval(offsets)` prototype was replaced with a leaf-level Sparse
consumer contract.  `PhyConjunctFilterExpr` now asks B to filter A's ordered
IDs directly.  The supported INT64 unary/binary range consumers validate that
the input is ascending, unique, in range, and segment-local; group IDs by raw
scalar chunk; make one skip-index decision and one `chunk_data` pin per
involved chunk; test `chunk[local_offset]` including nullable validity; and
append surviving original IDs to a reserved output vector.  No V-bit bitmap or
Dense intermediate is created.  Unsupported inputs and expression paths return
unsupported so the established Dense executor remains the fallback.

Build and correctness checks completed on 2026-08-11:

- `libmilvus_core.so` and `unittest/all_tests` linked successfully in Release;
- `ConjunctExprTest.*` passed 3/3, including proof that the second child does
  not call ordinary `Eval()` and that unsupported consumers fall back; and
- the Milvus E2E checked the first ten fixed queries for exact topK ID and
  distance equality before timing.

The first post-fix diagnostic retained five physical segments.  Even with
that uncontrolled fan-out, no request exceeded the 20 ms slow threshold and
Sparse reduced endpoint mean from 3.221 ms to 2.793 ms (-13.32%); Sparse was
faster in all 12 paired windows.  This result demonstrates that the former
80--225 ms tail was an implementation artifact, but it is not the final
controlled-layout number.

The final run added `dataCoord.segment.maxSize=8192`, seal proportion 0.99,
and zero jitter.  Query segment inspection confirmed exactly one sealed
1,000,000-row segment.  Configuration remained 1M x 128D, L2/topK10/ef64,
`a < 1000 and b < 500000`, NQ=1, C1, expression cache off, 10 warm-ups per
mode, and 12 Dense-Sparse-Sparse-Dense windows with 50 fixed queries per slot
(1,200 timed requests per mode).

| Metric | Dense per-query BF | Direct Sparse valid-ID BF | Delta |
|---|---:|---:|---:|
| Mean endpoint latency | 4.580 ms | 3.110 ms | -32.10% |
| Median endpoint latency | 4.554 ms | 2.726 ms | -40.14% |
| P90 endpoint latency | 4.817 ms | 5.546 ms | +15.13% |
| Requests >= 20 ms | 0 | 0 | no long tail |
| Paired windows won by Sparse | - | 12 / 12 | repeatable direction |

The P90 crossover is visible and should not be hidden: Sparse has more
short requests and a small upper-distribution cost around 5--6 ms, while the
removed generic offset evaluator previously produced 80--225 ms outliers.
The aggregate improvement is far above the Dense A/A run-to-run variation:
two otherwise identical 600-request Dense runs measured 4.749 ms and
4.781 ms mean (0.66% difference).

For this controlled high-filter BF point, the direct Sparse-to-Sparse
consumer therefore restores the intended cost model: A scans once to produce
about 1,000 IDs, B examines only those candidates using chunk-amortized data
access, MVCC/TTL/delete processing remains in the existing downstream
compaction path, and Cardinal receives about 500 final IDs.  The result does
not yet establish a production threshold or broader expression/type coverage,
but it rejects the previous conclusion that compound Sparse filtering is
intrinsically slower; the regression came from the per-ID generic evaluator.

### Rebase validation against upstream master

On 2026-08-11 the Direct Sparse branch was rebased onto Milvus upstream
`master` commit `5097a4246b30646485fd5635f9fb3a94ce54662e` (the resulting
branch head is recorded with the code change).  The rebase retained upstream
query-view state-machine, null-rejection, storage-v2 bulk-offset, and
chunk-access changes.  The obsolete experimental
`visibility_filter_enabled` branch was removed because current master always
applies the normal MVCC timestamp/delete/TTL mask; native IDs continue to be
compacted against that mask.

The associated Cardinal loading-overhead API had changed upstream.  The
experiment dependency chain was advanced to Cardinal `10c3cf8b` through
Knowhere `bc78bec9`, preserving the new loading-overhead group semantics.
`libmilvus_core.so` rebuilt successfully in Release.  Refreshing generated
protobuf C++ files was required because the updated schema includes
`ArithOpType::Shl/Shr` and `BloomFilterExpr`.

The isolated standalone server was then rebuilt and run with the same
one-segment, controlled layout and endpoint protocol above.  The rebase-run
result is independent of the earlier table:

| Metric | Dense per-query BF | Direct Sparse valid-ID BF | Delta |
|---|---:|---:|---:|
| Mean endpoint latency | 3.889 ms | 3.164 ms | -18.65% |
| Median endpoint latency | 3.865 ms | 2.743 ms | -29.04% |
| P90 endpoint latency | 4.111 ms | 5.679 ms | +38.13% |
| Timed requests / mode | 1,200 | 1,200 | - |
| Paired windows won by Sparse | - | 12 / 12 | repeatable direction |

The endpoint script completed its equality closure before timing, and reported
the explicit BF route.  The absolute numbers differ from the earlier run after
the master dependency/server refresh, but the direction remains stable.  A
full monolithic `unittest/all_tests` rebuild was intentionally not used as the
post-rebase gate: its protobuf refresh recompiles 184 unrelated sources.  The
targeted E2E exercises the changed producer, sparse consumer, MVCC compaction,
Knowhere/Cardinal bridge, and server binary together.

### Stable CPU flamegraph attribution (2026-08-13)

Earlier profiles that overlapped Cardinal index building/compaction were discarded.
The retained profiles were collected only after the Cohere 10M collection had
finished loading and compaction, with eight sealed serviceable segments and no
active `knowhere_build*` CPU. Dense and Sparse were profiled separately using
`sudo perf record -F 199 -g -p <milvus-pid>` for 30 seconds. Each mode first
received 20 warmups, followed by 60 windows of the same 50 fixed Cohere queries
(3,000 timed NQ=1/C1 requests per mode, topK=10, `ef=64`, explicit BF).
Artifacts are in `/tmp/knowpr1732-cohere-perf-clean-20260812/`.

The profile is a CPU-attribution experiment, not a replacement for the accepted
ABBA endpoint table above. Its single-mode endpoint means were 8.0066 ms
(Dense) and 6.9437 ms (Sparse); the corresponding median/P90 values were
7.9425/8.3916 ms and 5.6311/14.4934 ms. The profile confirms the functional
split below:

| Functional module | Dense profile | Sparse profile | Interpretation |
|---|---:|---:|---|
| Scalar filtering and result production | `PhyFilterBitsNode::GetOutput` 26.43% inclusive; `PhyConjunctFilterExpr::Eval` 24.14%; `PhyUnaryRangeFilterExpr::Eval` 21.26% | `PhyFilterBitsNode::GetOutput` about 27.12% inclusive; `TryGetNativeValidIds` 25.73%/18.86% in the conjunction/unary frames; `TryFilterNativeValidIds` about 6.65% self | Dense builds/evaluates the full bitmap; Sparse runs A's producer scan and B only on A's candidate list |
| Full-universe scalar scan / Dense materialization | `ProcessDataChunks*` about 18.5% and SVE `OpCompareValImpl` 14.72% self; `BitCompressBatch64` visible at about 2.11% | Not the primary B route; SVE compare 10.78% self is A's initial full scan | Sparse removes the final N-row bitmap materialization/enumeration, but A still has to inspect the column |
| Cardinal BF distance search | `IndexImpl::Search` 16.22%; `BruteForceSearcher` 14.58%; `BruteForceSearchImpl` 14.43% | `IndexImpl::Search` 13.70%; `BruteForceSearcher` 11.31%; `BruteForceSearchImpl` 11.29%; `ScanRangeByValidIdsBatch4` 6.91% | The same BF search remains on both paths; Sparse additionally exposes the valid-ID scan consumer |
| Execution/scheduling | Driver/task/segment scheduling frames | Same framework frames | Inclusive percentages overlap and are not additive |

The often-quoted `0.46 ms` for B is only a naive conversion of the roughly
6.65% `TryFilterNativeValidIds` CPU sample share multiplied by the 6.94 ms
Sparse endpoint mean. It is not a measured B wall-time timer and must not be
interpreted as a phase total: perf samples are nested/inclusive, the request is
multi-threaded, and CPU time from parallel segment/search workers can exceed the
critical-path endpoint time. The correct conclusion is therefore that the
profile shows a measurable B consumer block, not that B necessarily costs
0.46 ms of endpoint latency.

For this workload `a < 1000` produces about 0.1% candidates (roughly 10,000 IDs
over 10M); `b < 500000` examines those candidates and retains about half. The
implementation in `FilterSortedNativeIdsByRawData` is already chunk-amortized:
it validates that the input IDs are ascending/unique, groups them by raw-data
chunk, performs one skip-index check and one `chunk_data` pin per touched chunk,
then applies the scalar predicate and appends survivors to a reserved vector.
It does not call `get_chunk_by_offset`/`PinCells` once per ID and does not build a
second Dense bitmap. Consequently, the current code does not show the former
per-ID pin/release defect.

There are still bounded costs worth measuring before further optimization:

* the input list is walked once for monotonicity validation and again for
  chunk grouping/evaluation;
* random A IDs touch many scalar chunks, so pins scale with touched chunks
  rather than survivors; and
* B uses a branchy scalar comparison and nullable-validity check for each
  candidate, which is appropriate for an irregular sparse list but is not a
  SIMD full-column pass.

These effects can explain a nonzero B block, but the present profile alone does
not justify a code change. The next causal check should add request/segment
counters for input IDs, touched chunks, skip decisions, pins, output IDs, and
timers around A, B, MVCC, and Cardinal search. If B scales with touched chunks
rather than candidate count, a follow-up can cache chunk locations or combine
validation with the grouping pass; if it scales linearly with candidates while
remaining small in absolute time, the implementation is behaving as designed.
