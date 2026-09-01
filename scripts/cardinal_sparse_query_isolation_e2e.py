#!/usr/bin/env python3
"""Milvus Query isolation ABBA for Dense versus Adaptive filter output.

This removes vector search from the timed endpoint.  It intentionally uses
count(*) so projection, ordering and result transfer do not obscure predicate,
filter-output, MVCC and Query-consumer costs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.request
from dataclasses import dataclass

from pymilvus import MilvusClient


COUNTERS = {
    'internal_core_adaptive_filter_output_total{representation="sparse"}':
        "sparse_outputs",
    'internal_core_adaptive_filter_output_total{representation="dense_threshold"}':
        "threshold_dense_outputs",
    'internal_core_adaptive_filter_output_total{representation="dense_or_phase1"}':
        "or_dense_outputs",
}


def emit(value: dict) -> None:
    print(json.dumps(value, sort_keys=True), flush=True)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def read_metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=10) as response:
        text = response.read().decode("utf-8")
    totals = {alias: 0.0 for alias in COUNTERS.values()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.rsplit(" ", 1)[0]
        if metric not in COUNTERS:
            continue
        try:
            totals[COUNTERS[metric]] += float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    return totals


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in after}


def extract_count(rows: list[dict]) -> int:
    if len(rows) != 1:
        raise RuntimeError(f"count(*) returned {len(rows)} rows: {rows!r}")
    for key, value in rows[0].items():
        if key.lower().replace(" ", "") in {"count(*)", "count"}:
            return int(value)
    if len(rows[0]) == 1:
        return int(next(iter(rows[0].values())))
    raise RuntimeError(f"cannot find count(*) in {rows!r}")


@dataclass(frozen=True)
class Mode:
    name: str
    representation: str


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.clients = [MilvusClient(uri=args.uri, token=args.token or None)
                        for _ in range(args.concurrency)]

    def query_once(self, worker: int, mode: Mode) -> tuple[float, int]:
        params = {
            "filter_result_representation": mode.representation,
            "sparse_result_max_cardinality": self.args.threshold,
        }
        start = time.perf_counter_ns()
        rows = self.clients[worker].query(
            collection_name=self.args.collection,
            filter=self.args.predicate,
            output_fields=["count(*)"],
            filter_params=params,
            consistency_level="Strong",
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        return elapsed_ms, extract_count(rows)

    def run_slot(self, mode: Mode, requests: int) -> dict:
        before = read_metrics(self.args.metrics_url)
        per_worker = [requests // self.args.concurrency] * self.args.concurrency
        for index in range(requests % self.args.concurrency):
            per_worker[index] += 1

        def run_worker(worker: int, count: int) -> list[tuple[float, int]]:
            return [self.query_once(worker, mode) for _ in range(count)]

        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.args.concurrency) as pool:
            futures = [pool.submit(run_worker, worker, count)
                       for worker, count in enumerate(per_worker) if count]
            samples = [sample for future in futures
                       for sample in future.result()]
        elapsed = time.perf_counter() - start
        after = read_metrics(self.args.metrics_url)
        counts = {count for _, count in samples}
        if len(counts) != 1:
            raise RuntimeError(f"non-deterministic counts for {mode.name}: {counts}")
        latencies = [latency for latency, _ in samples]
        return {
            "mode": mode.name,
            "requests": requests,
            "count": next(iter(counts)),
            "elapsed_s": elapsed,
            "qps": requests / elapsed,
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p90_ms": percentile(latencies, 0.90),
            "metrics_delta": counter_delta(before, after),
        }

    def warmup(self, mode: Mode) -> None:
        deadline = time.monotonic() + self.args.warmup_seconds
        index = 0
        while time.monotonic() < deadline:
            self.query_once(index % self.args.concurrency, mode)
            index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="http://127.0.0.1:19532")
    parser.add_argument("--token", default="")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9092/metrics")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--predicate", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--requests-per-slot", type=int, default=50)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument("--profile-mode", choices=("dense", "sparse"))
    parser.add_argument("--profile-seconds", type=float, default=0.0,
                        help="run one representation continuously for perf attachment")
    args = parser.parse_args()

    dense = Mode("dense", "dense")
    sparse = Mode("sparse", "adaptive")
    runner = Runner(args)

    dense_count = runner.query_once(0, dense)[1]
    sparse_count = runner.query_once(0, sparse)[1]
    if dense_count != sparse_count:
        raise RuntimeError(
            f"Dense/Sparse count mismatch: {dense_count} != {sparse_count}")
    emit({"event": "correctness", "case": args.case,
          "predicate": args.predicate, "count": dense_count})

    if args.profile_mode:
        if args.profile_seconds <= 0:
            raise ValueError("--profile-seconds must be positive with --profile-mode")
        mode = dense if args.profile_mode == "dense" else sparse
        runner.warmup(mode)
        deadline = time.monotonic() + args.profile_seconds
        requests = 0
        elapsed = 0.0
        while time.monotonic() < deadline:
            slot = runner.run_slot(mode, args.requests_per_slot)
            requests += slot["requests"]
            elapsed += slot["elapsed_s"]
        emit({
            "event": "profile_complete",
            "case": args.case,
            "predicate": args.predicate,
            "mode": mode.name,
            "count": dense_count,
            "requests": requests,
            "active_elapsed_s": elapsed,
            "qps": requests / elapsed,
        })
        return

    runner.warmup(dense)
    runner.warmup(sparse)

    windows = []
    for window in range(1, args.windows + 1):
        slots = []
        for mode in (dense, sparse, sparse, dense):
            slot = runner.run_slot(mode, args.requests_per_slot)
            slot.update({"event": "slot", "window": window})
            emit(slot)
            slots.append(slot)
        by_mode = {
            name: [slot for slot in slots if slot["mode"] == name]
            for name in ("dense", "sparse")
        }
        qps = {name: statistics.mean(slot["qps"] for slot in values)
               for name, values in by_mode.items()}
        mean_ms = {name: statistics.mean(slot["mean_ms"] for slot in values)
                   for name, values in by_mode.items()}
        windows.append({
            "window": window,
            "dense_qps": qps["dense"],
            "sparse_qps": qps["sparse"],
            "qps_delta": (qps["sparse"] - qps["dense"]) / qps["dense"],
            "dense_mean_ms": mean_ms["dense"],
            "sparse_mean_ms": mean_ms["sparse"],
        })

    emit({
        "event": "benchmark_complete",
        "case": args.case,
        "predicate": args.predicate,
        "threshold": args.threshold,
        "concurrency": args.concurrency,
        "windows": args.windows,
        "requests_per_slot": args.requests_per_slot,
        "count": dense_count,
        "qps_delta_mean": statistics.mean(w["qps_delta"] for w in windows),
        "sparse_faster_windows": sum(w["qps_delta"] > 0 for w in windows),
        "paired_windows": windows,
    })


if __name__ == "__main__":
    main()
