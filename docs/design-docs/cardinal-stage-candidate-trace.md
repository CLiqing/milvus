# Opt-in Cardinal pre-refine and local-result tracing

This diagnostic records the actual candidate pool of a normal search. Disabling
refinement and requesting a different global topK does not establish that pool:
it can change local quotas and heap sizes. The diagnostic keeps the original
request topK, search budget, refiner, and ordinary output.

## Stages and ownership

- **C** is captured after primary/fallback search and before refinement.
- **L** is captured after local refinement, truncated to the ordinary local topK.
- **F** is the ordinary API result after the existing global reduction.

Cardinal owns per-request/per-query C/L buffers. Both captured phases use the
same reverse-ID map and distance mapper as normal output. Its Knowhere adapter
exports bounded JSON through DataSet JsonInfo. Milvus memory and disk vector
wrappers preserve that metadata in an internal SearchResult string. The normal
Segment.Search path maps offsets to actual INT64 primary keys using the same
request-pinned read snapshot; it does not assume PK equals row offset. It releases
the outer segment search mutex before snapshot-based diagnostic reads, avoiding
a recursive acquisition through result-filling helpers.

The trace string is internal runtime state. It is not part of protobuf output
or a persisted index, and does not change the physical-v11 index format.

## Activation and logging

`trace_visit=true` enables capture. `debug_stage_trace_tag` is a diagnostic-only
correlation token removed before Knowhere validates search parameters. Valid tags
contain 1–128 ASCII letters, digits, or `:_.-`. Tags do not choose a search path.

The segment emits `CARDINAL_EXACT_STAGE_TRACE` JSON chunks of at most128 rows,
plus completion manifests. Each candidate contains its original position,
offset, actual PK, and mapped float32 score. FNV1a64 checksums use decimal
`position:offset:pk:score_binary32_uint;` bytes rather than a JSON float spelling;
a separate checksum binds the ordered chunk checksums. Query hashes bind the
original native input bytes. Segment and field IDs bind logs to external
DataCoord collection/index metadata. A build ID is not treated as an index ID.

Limits are16 queries per capture and8192 candidates per phase/query. The wrapper
accepts at most32MiB of native JSON. Duplicate/invalid offsets, nonfinite scores,
incomplete batches, and L outside C invalidate the capture. All candidate records
are prepared before emission, so an unsuccessful PK lookup cannot produce a
successful-looking empty candidate pool. Missing or truncated logs are rejected
by the collector, never interpreted as zero recall.

## Failure behavior

Normal backend failures retain their existing propagation; this change does not
classify or rewrite them. Allocation/serialization failures in diagnostic code
are caught on that diagnostic path. PK/storage/cancellation failures that occur
while resolving diagnostic offsets invalidate the trace while retaining the
already completed ordinary search result. Fixed incomplete reason codes avoid
logging arbitrary storage exception text. Logging failure can leave missing
completion, which the collector must reject. This is not a claim that a process
can survive system-wide OOM.

Tracing is disabled by default and does not copy candidate arrays when disabled.
Enabled tracing adds copying, PK reads, JSON work and logging. Trace timings are
diagnostic observations and must not enter formal QPS comparisons.

## Validation requirements

1. Compare trace-off/on ordered IDs, score bits and original work counters on
   the real native IVF/refine path, including nonidentity reverse maps.
2. Check helper handling of missing/malformed/stale metadata, bounded chunks,
   checksums, query/candidate limits and invalid candidate values.
3. Run real sealed Segment.Search and pinned-snapshot PK mapping fixtures,
   including adversarial publication of newer segment state.
4. Exercise CARDINAL_TIERED SQ8/SAQ1 or SAQ2 with SAQM incremental refinement,
   upload/load a v11 index, and require expanded C plus identical ordinary output.
5. Load an existing pre-trace v11 artifact without rebuilding it. Report
   cross-compiler score tolerance separately from same-build trace toggle parity.
6. Before cloud analysis, verify actual deployed source/image, AutoIndex and
   target version, unchanged retained indexes, direct DataCoord segment/field
   bindings, complete correlated logs, native counters and API output closure.

The focused C++ fixtures are `test_cardinal_stage_trace.cpp` and
`test_cardinal_stage_trace_pk.cpp`. Full linked runtime validation, rather than
syntax checks alone, is required before publishing the diagnostic image.

## Analysis boundary

C coverage, local-quota bounds, L recall and F recall separate candidate
admission from local/global ranking losses. Independent coarse-only Top100 is
an auxiliary measurement, not C coverage. Actual scanned bucket/range coverage
is a separate future instrument; current counters measure work and do not
establish the scanned ID set. Query-hook and cloud log delivery remain cloud
validation steps even after local C++ fixtures pass.
