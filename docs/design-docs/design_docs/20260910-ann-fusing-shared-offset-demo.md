# Explicit ann_fusing through the shared offset evaluator

This is the clean Stage 2 / Stage 3A development slice, not a production-ready
filter planner. `ann_fusing` is the public hint. Historical benchmark binaries
use the old spelling `downpush`; that is not an additional new hint alias.
No hint retains baseline behavior. Iterative filtering is not an AUTO choice.

## Pipeline and module responsibilities

Milvus parses the hint as intent. `FilterBitsNode` checks the narrow demo query
shape and asks the **loaded index** whether it supports this demo. The actual
Cardinal implementation answers the capability question and owns graph-path
selection. Milvus neither predicts graph/IVF/BF from an exact filter count nor
receives a `GRAPH_REQUIRED` enum. Both `VectorMemIndex` and
`VectorDiskAnnIndex` forward the capability: the latter also loads HNSW files,
so its class name is not evidence that the search algorithm is DiskANN.

After admission, Milvus retains the original typed expression in a
query-owned `OffsetExpressionCallback`. User-predicate dense bitmap evaluation
is skipped. The existing mandatory bitmap storage and MVCC/delete processing
remain; this does not claim to remove every O(N) operation or allocation.
`VectorSearchNode` attaches a borrowed descriptor to the search bitset view.

Knowhere carries that opaque descriptor separately from bitmap data and
counts. It has no MOD implementation, scalar-column view or expression IR.
Cardinal admits only a loaded memory graph, creates a private worker for each
query vector, translates graph IDs back to segment offsets, and applies
mandatory exclusions before invoking the callback. Candidate neighbor lists
are evaluated in batches of 32 plus the whole short tail. Entry points may
use a one-ID call. Batches do not span frontiers merely to fill up; existing
distance prefetch and valid/invalid connectivity handling are retained.

The Milvus callback calls `OffsetExpressionWorkspace::EvalAcceptedBatch`,
which uses the same physical `ExprSet` and `EvalCtx::offset_input` path as the
shared iterative evaluator. The existing arithmetic expression implementation
fetches the scalar values and evaluates MOD. There is no separate fusing MOD
kernel, second expression tree in Cardinal, or per-type parameter union.

## Ownership and callback contract

`CandidateEvaluatorViewV1` has an opaque immutable factory context and
create/destroy-worker and batch-evaluation function pointers. A worker owns
its mutable physical expression state; workers do not share an ExprSet.
IDs are signed 32-bit segment-row offsets. The callback accepts up to 64 IDs
and an active-lane mask, and returns an accepted-lane mask: only SQL TRUE is
accepted; FALSE and UNKNOWN are rejected. Inactive lanes are ignored. Callback
errors are returned as status values, not as “all rows rejected,” and C++
exceptions are contained by the producer thunk.

The query owns the factory and its execution context through synchronous
search completion. The descriptor is borrowed and does not own expression
data. Worker destruction calls back into the producer DSO. The demonstrated
Cardinal thread-pool path waits for all submitted futures before rethrowing
task exceptions, preserving lifetime on ordinary callback/underfill errors.

V1 layout and size are frozen and checked exactly. The three C++ repositories
are built together; neither this descriptor nor the changed C++ virtual
interfaces promise independent, append-only cross-DSO upgrades.

## Scope and failure policy

Admission is limited to ordinary sealed FP32 searches with a non-nullable
INT64 MOD leaf and a positive divisor. Unsupported shapes retain baseline
before any user filtering is skipped. Entity TTL, nullable/element-level
layouts, growing segments, range/grouping/iterator queries, routed/tiered/disk
graphs and general expression coverage are not added by this slice.

Once fusing is admitted, graph underfill requiring IVF/BF is an explicit demo
error. No fallback bitmap is materialized and no sampling estimate enters
exact-count/result-capacity logic. This is a development policy, not the final
fallback product contract. A future Cardinal cost model can remain private;
general planning and AUTO decisions are deferred until this call chain has
been reviewed.

## Verification and local dependency checkpoint

Native tests cover callback contracts, NULL truth, private-worker concurrency,
shared iterative regression, MOD graph results against a raw-distance oracle,
mandatory exclusions and unsupported underfill. Standalone tests must show
actual fusing admission, skipped user Eval and nonzero producer callbacks;
successful baseline results alone do not count as a fusing smoke.

The fixed historical performance anchor is 500K x 768D, two sealed segments,
memory HNSW, NQ=5, topK=20, `bucket % 5 < 3`, batch32/prefetch. Five ABBA rounds
compare whole historical/new builds on the same old-built index; matching Go
executables are required by their differing C ABIs. Binary hashes and a
round-level uncertainty estimate accompany the result. The historical
generator is periodic and its rebuilt graph has poor absolute recall, so a
separate non-periodic standalone oracle fixture is needed for quality checks.

Local source checkpoints: Knowhere `dda2fd570f5552734135f4f524444941fbbbcc7d`,
Cardinal `88cb0cef63116812e98c15e654197dfd3d6b23c7` (with the separate GCC 11
heap portability prerequisite `c830d932`). Remote publication/dependency pins
are not implied by these local commits. Development uses
`INDEX_ENGINE=cardinal` and `FETCHCONTENT_SOURCE_DIR_KNOWHERE` pointing at the
matching local Knowhere checkout.
