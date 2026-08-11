#!/usr/bin/env python3
"""Milvus E2E ABBA: Dense vs Sparse A -> B -> Cardinal BF pipeline.

Before using this script, a Dense-only request with ``bf_filter_scan_mode=auto``
must have verified that the same final filter naturally chooses BF.  The two
compared modes are explicit BF modes today: Cardinal's existing
``valid_ids_per_query`` couples valid-ID representation to BF dispatch.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
from pymilvus import DataType, MilvusClient


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def wait_for_load(client, collection, timeout_seconds=1800):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = client.get_load_state(collection_name=collection).get("state")
        if getattr(state, "name", None) == "Loaded":
            return
        time.sleep(5)
    raise TimeoutError(f"collection {collection} did not load")


def hits(result):
    return [(item["id"], float(item["distance"])) for item in result[0]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--token-env", default="LAB_MILVUS_TOKEN")
    parser.add_argument("--collection", default="cardinal_sparse_compound_1m")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--segment-rows", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch-rows", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--index-algo", choices=("auto", "BF"), default="auto",
                        help="optional diagnostic override; both payload modes already select BF")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="create/load the deterministic dataset then exit; used before route preflight")
    parser.add_argument("--log-slow-ms", type=float, default=0.0,
                        help="emit a JSON record for timed calls at or above this latency")
    parser.add_argument("--modes", nargs="+",
                        choices=("dense_per_query", "valid_ids_per_query"),
                        default=("dense_per_query", "valid_ids_per_query"),
                        help="one mode for perf attribution, or both for ABBA comparison")
    args = parser.parse_args()
    if args.rows % args.segment_rows:
        raise ValueError("rows must be divisible by segment-rows")
    if args.segment_rows < 1_000:
        raise ValueError("segment-rows must leave 1,000 A candidates")

    token = os.environ.get(args.token_env)
    if not token:
        raise ValueError(f"missing {args.token_env}")
    client = MilvusClient(uri=args.uri, token=token)
    rng = np.random.default_rng(1732)
    a_limit = 1_000
    b_limit = args.segment_rows // 2
    predicate = f"a < {a_limit} and b < {b_limit}"

    if not args.reuse_existing:
        if client.has_collection(args.collection):
            client.drop_collection(args.collection)
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("a", DataType.INT64)
        schema.add_field("b", DataType.INT64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=args.dim)
        client.create_collection(collection_name=args.collection, schema=schema)

        for segment_start in range(0, args.rows, args.segment_rows):
            # Independent random ranks make A precisely 0.1% and B ~50% of A
            # inside every sealed segment while keeping their matching IDs
            # uncorrelated with vector values and each other.
            a = rng.permutation(args.segment_rows).astype(np.int64)
            b = rng.permutation(args.segment_rows).astype(np.int64)
            for local_start in range(0, args.segment_rows, args.batch_rows):
                local_end = min(args.segment_rows, local_start + args.batch_rows)
                vectors = rng.random((local_end - local_start, args.dim), dtype=np.float32)
                ids = np.arange(segment_start + local_start, segment_start + local_end)
                client.insert(collection_name=args.collection, data=[
                    {"id": int(row_id), "a": int(a_value), "b": int(b_value),
                     "vector": vector.tolist()}
                    for row_id, a_value, b_value, vector in zip(
                        ids, a[local_start:local_end], b[local_start:local_end], vectors)
                ])
            # Preserve the requested multi-segment topology before the vector
            # index is built; no scalar index is created for this experiment.
            client.flush(collection_name=args.collection)
            print(json.dumps({"event": "segment_flushed", "rows": segment_start + args.segment_rows}),
                  flush=True)

        params = client.prepare_index_params()
        params.add_index(field_name="vector", index_type="CARDINAL_TIERED",
                         metric_type="L2", params={"M": 16, "efConstruction": 100})
        client.create_index(collection_name=args.collection, index_params=params)
        client.load_collection(collection_name=args.collection)
        wait_for_load(client, args.collection)

    if args.prepare_only:
        print(json.dumps({"event": "collection_ready", "collection": args.collection,
                          "rows": args.rows, "segment_rows": args.segment_rows,
                          "predicate": predicate}), flush=True)
        return

    query_rng = np.random.default_rng(1732 + 1)
    queries = query_rng.random((args.queries, args.dim), dtype=np.float32)

    def search(mode, query):
        params = {"metric_type": "L2", "params": {
            "ef": 64, "bf_filter_scan_mode": mode,
        }}
        if args.index_algo != "auto":
            params["params"]["index_algo"] = args.index_algo
        begin = time.perf_counter_ns()
        result = client.search(collection_name=args.collection, data=[query.tolist()],
                               anns_field="vector", limit=10, filter=predicate,
                               search_params=params)
        return (time.perf_counter_ns() - begin) / 1_000_000, hits(result)

    modes = tuple(args.modes)
    for mode in modes:
        for query in queries[:args.warmups]:
            search(mode, query)
    if len(modes) == 2:
        for number, query in enumerate(queries[:10]):
            dense = search("dense_per_query", query)[1]
            sparse = search("valid_ids_per_query", query)[1]
            if dense != sparse:
                raise AssertionError(f"topK/distance mismatch at query {number}")

    samples = {mode: [] for mode in modes}
    windows = {mode: [] for mode in modes}
    for window in range(args.windows):
        per_window = {mode: [] for mode in modes}
        order = (("dense_per_query", "valid_ids_per_query",
                  "valid_ids_per_query", "dense_per_query")
                 if len(modes) == 2 else modes)
        for mode in order:
            slot = []
            for query_number, query in enumerate(queries):
                latency, _ = search(mode, query)
                slot.append(latency)
                if args.log_slow_ms and latency >= args.log_slow_ms:
                    print(json.dumps({"event": "slow_request", "window": window + 1,
                                      "mode": mode, "query": query_number,
                                      "latency_ms": latency}), flush=True)
            samples[mode].extend(slot)
            per_window[mode].extend(slot)
        for mode in modes:
            windows[mode].append(statistics.mean(per_window[mode]))
        print(json.dumps({"event": "window_complete", "window": window + 1}), flush=True)

    report = {
        "event": "benchmark_complete", "collection": args.collection,
        "rows": args.rows, "segment_rows": args.segment_rows, "dim": args.dim,
        "predicate": predicate, "a_valid_per_segment": a_limit,
        "b_selectivity": "about 50pct conditional on A", "final_valid_expected":
            args.rows // args.segment_rows * a_limit // 2,
        "nq": 1, "concurrency": 1, "index_algo_request": args.index_algo,
        "actual_payload_route": "explicit_BF",
        "queries_per_slot": args.queries, "windows": args.windows,
        "modes": {
            mode: {"requests": len(samples[mode]), "mean_ms": statistics.mean(samples[mode]),
                   "median_ms": statistics.median(samples[mode]),
                   "p90_ms": percentile(samples[mode], 0.90)}
            for mode in modes
        },
    }
    if len(modes) == 2:
        dense_mean = statistics.mean(samples["dense_per_query"])
        sparse_mean = statistics.mean(samples["valid_ids_per_query"])
        paired = [(dense - sparse) / dense for dense, sparse in zip(
            windows["dense_per_query"], windows["valid_ids_per_query"])]
        report["sparse_vs_dense_mean_delta"] = (dense_mean - sparse_mean) / dense_mean
        report["paired_windows"] = {"count": len(paired), "mean_delta": statistics.mean(paired),
                                    "median_delta": statistics.median(paired),
                                    "sparse_faster_windows": sum(value > 0 for value in paired)}
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
