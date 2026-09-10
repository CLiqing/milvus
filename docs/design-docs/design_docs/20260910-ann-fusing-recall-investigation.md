# MOD demo: zero-recall investigation and preserved checkpoint

## Status and checkpoint

2026-09-10. Diagnosis only: no production-code fix, index rebuild, remote push,
VDC request or replacement of another service. At the start, all three working
trees were clean and the requested code state was already committed:

| Repository | Preserved commit |
| --- | --- |
| Milvus source | `25c9d2f92a3dcb09c1439f483daaee9c2618aa02` |
| Milvus documentation HEAD before diagnosis | `8d486e9b69b04e837c1ba48de4a3143460af1b4c` |
| Knowhere | `dda2fd570f5552734135f4f524444941fbbbcc7d` |
| Cardinal | `88cb0cef63116812e98c15e654197dfd3d6b23c7` |

This report is a documentation-only follow-up. The binaries used for the ABBA
and diagnosis remain unchanged. Their SHA-256 identities are in the artifact
JSON files; A uses the archived pre-migration libraries, not a mode toggle in B.

## What zero recall means

The fixture has 500,000 rows, 768 dimensions, two sealed 250,000-row segments,
and queries generated from IDs 0 through 4. The historical generator is
`(((row * 17 + dimension * 31) % 997) / 997).astype(float32)`.
It repeats every 997 IDs. After `bucket % 5 < 3`, each query has 301 or 302
valid identical vectors, so the exact twentieth-neighbor squared L2 is zero.

The oracle accepts ANY returned ID whose independently recomputed raw distance
is <= 1e-5; it does not demand an arbitrary set of twenty tied IDs. All twenty
results for each query fail this check in the performance GRAPH scenario.
These are nonempty results, not a query error or an empty segment.

For query 0, the returned distances are approximately 60.33235, despite valid
zero-distance neighbors. Across the five queries the returned graph distances
are roughly 15.66 to 60.33. Every returned ID is in the same primary-key residue
class `id % 997 == 533`. This is not a tiny tie-break or floating-point issue.

Read-only `get()` checks verified ten stored vectors against the generator with
zero elementwise error, including IDs 0, 997, 1994 and rows from both segments.
Actual stored vectors for the first query's returned IDs also reproduce their
large raw distances. These are targeted checks, not a full row-by-row audit.

## Controlled search results on the original frozen objects

No data or index was rebuilt during diagnosis. Both variants loaded the same
finished HNSW index metadata and the same two segment IDs as ABBA.

| Search | Old A recall@20, five queries | New B recall@20, five queries |
| --- | --- | --- |
| GRAPH ef=20, no filter | all 0 | all 0 |
| GRAPH ef=20, baseline MOD | all 0 | all 0 |
| GRAPH ef=20, fusing MOD | all 0 | all 0 |
| GRAPH ef=128 or 1024, each of the above | all 0 | all 0 |
| ef=4096, default routing, baseline | 100%,100%,100%,100%,50% | same |
| ef=4096, default routing, fusing | 100%,100%,100%,100%,50% | all 0 |

The last two rows are NOT an isolated evaluator regression: default Cardinal
routing can switch the baseline and legacy path away from GRAPH as ef rises.
`ShouldSwitchMemIVF` depends on ef, row count and filter ratio. For a graph-only
built index, `SelectMemSearcher` chooses its BF searcher when this gate fires.
The new demo deliberately bypasses that choice and requires GRAPH.

An extra A control sets `switch_ivf_ratio=1.0` and ef=4096. No-filter, baseline
and legacy fusing now ALL remain at zero recall. Its actual counters show
`graph_search_avg=1`, `ivf_search_avg=0`, `bf_search_avg=0`,
`predicate_test_avg=57`; unfiltered `search_compute_avg=56`. This verifies that
raising ef does not restore reachability when execution really stays on GRAPH.

The BF control with `switch_ivf_ratio=0.0` produces MOD recall
100%,100%,100%,75%,50%. It is not an exact raw-vector oracle: serialized index
metadata identifies RBQ3 search and RBQ8 refinement, not raw FP32 refinement.
Some raw-positive distances are reported as zero. Quantization/ranking is a
separate limitation from the graph's large-distance, zero-recall failure.
No fine-grained quantization-kernel root cause is claimed here.

## Direct graph evidence and probable build mechanism

The read-only inspector follows Cardinal's `DefaultSerializer` footer/tree,
`StorageBlock` layout and uncompressed `MatrixGraph` row layout. It checks
row degree and edge-ID bounds and traverses ALL reachable outgoing edges,
without a distance heap, filter, ef limit or callback.

| Segment suffix | Vertices | Entry internal ID | Degree | Reachable from entry |
| --- | ---: | ---: | ---: | ---: |
| 518471 | 250,000 | 533 | 56 at every vertex | 57 |
| 518890 | 250,000 | 780 | 56 at every vertex | 57 |

The native runtime's 57 visits agree independently with this file inspection.
Segment offsets/internal IDs are NOT primary keys; their modulo residues
must not be treated as vector identities. Graph traversal, not a guessed ID
mapping, establishes the reachability failure.

Both index SHA-256 values still match those recorded after ABBA:

- 518471: `dedfc7d339ed8a18f734b3d0626be2249cdfbfde449909a48f5e95699660ad94`
- 518890: `4497d1ef42e4896ce9fda38e854644a3998d4819a4d802729477322586998fdb`

The most directly implicated build code is `VamanaSelectNeighbors`, present
also in archived Cardinal source commit `94edaf38585e`:

- Candidates are sorted by distance and selection stops at graph width.
- The exclusion factor is updated for positive or negative candidate distance,
  but not for exactly zero candidate distance.
- Many zero-distance duplicate candidates can therefore consume the adjacency
  budget without preserving useful cross-group connections.
- `VamanaBuilder::SelectNeighbors` uses this function for graph construction.

This is a code-supported mechanism consistent with the observed duplicate-heavy
fixture and tiny reachable component. The disconnected stored graph is proven;
which exact build/prune operation first disconnected it has NOT been isolated
by a standalone builder reproduction or a controlled fix/rebuild.

## Consequences and next work

Baseline means conventional scalar filtering followed by ANN, not exhaustive
exact vector search. Its predicate can be correct while a defective graph
misses the nearest neighbors. The no-filter reproduction rules out requiring
the new fusing callback to trigger this particular failure.

Withdraw the earlier performance-acceptance conclusion. The 2.178/2.159 ms
old/new timings and 27.7% baseline-relative reduction remain measurements of
this abnormal workload only. Very limited graph work makes fixed/request and
bitmap costs disproportionately important. A separate 4096-row oracle at 100%
recall does not establish performance under good recall on the 500K fixture.

Before renewed performance acceptance: preserve this failing fixture, isolate
the builder's duplicate handling, and agree a fixed representative workload
with independent recall gating for both versions. Do not silently replace the
historical anchor, equate quantized BF with exact raw truth, or compare GRAPH
against an unnoticed BF switch. Whether to fix generic duplicate-graph behavior
is a separate implementation decision; no production fix is made in this task.

These findings concern the currently rebuilt local index. They do not establish
the recall of the original August index or of the 10M QTP/PDF experiments.

## Local evidence

Relative to the experiment workspace (parent of `milvus-clean`):

- `experiments/clean-stage3a/diagnose_recall.py`
- `experiments/clean-stage3a/inspect_anchor_graph.py`
- `artifacts/clean-stage3a-20260910/recall-diagnostic-A.json`
- `artifacts/clean-stage3a-20260910/recall-diagnostic-B.json`
- `artifacts/clean-stage3a-20260910/recall-diagnostic-A-route-control.json`
- `artifacts/clean-stage3a-20260910/graph-471-diagnostic.json`
- `artifacts/clean-stage3a-20260910/graph-890-diagnostic.json`
- Corresponding `logs/recall-diagnostic-*.log` files.

Diagnostic times include debug logging and cold-first-request effects; they
are not replacement benchmark numbers. Temporary startup channel-not-ready
responses were retried before search diagnostics began.
