#!/usr/bin/env python3
"""Run the resumable single-segment Adaptive Sparse ratio discovery matrix."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pymilvus import MilvusClient


DEFAULT_N = (50_000, 100_000, 250_000, 1_000_000, 3_000_000)
DEFAULT_RATIOS = (
    0.0005, 0.001, 0.002, 0.003, 0.004, 0.005,
    0.006, 0.007, 0.008, 0.009, 0.010,
)


def final_record(path):
    if not path.exists():
        return None
    result = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "benchmark_complete":
                result = record
    return result


def run_logged(command, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def ratio_slug(ratio):
    return f"r{ratio * 100:.2f}pct".replace(".", "p")


def compact_record(record, log_path):
    return {
        "N": record["actual_n"], "V": record["actual_v"],
        "ratio": record["actual_v_over_n"],
        "concurrency": record["concurrency"],
        "dense_qps": record["modes"]["dense"]["qps"],
        "sparse_qps": record["modes"]["sparse"]["qps"],
        "qps_delta": record["paired_windows"]["qps_mean_delta"],
        "dense_route": record["modes"]["dense"]["route"],
        "sparse_route": record["modes"]["sparse"]["route"],
        "dense_representation": record["modes"]["dense"]["representation"],
        "sparse_representation": record["modes"]["sparse"]["representation"],
        "topology": record["loaded_segment_rows"],
        "query_set_sha256": record["query_set_sha256"],
        "strict_closure_passed": record["strict_closure_passed"],
        "source_log": str(log_path),
    }


def rebuild_summary(artifact_dir, summary_path):
    records = []
    for log_path in artifact_dir.glob("search-n*-r*pct-c*.log"):
        record = final_record(log_path)
        if record and record.get("strict_closure_passed"):
            records.append(compact_record(record, log_path))
    records.sort(key=lambda item: (item["N"], item["ratio"], item["concurrency"]))
    temporary = summary_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="http://127.0.0.1:19532")
    parser.add_argument("--artifact-dir", default="artifacts/sparse-ratio-switchpoint-20260831")
    parser.add_argument("--n-values", type=int, nargs="+", default=DEFAULT_N)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=(1, 60))
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--sparse-threshold", type=int, default=30_000)
    args = parser.parse_args()

    script = Path(__file__).with_name("cardinal_sparse_compound_filter_e2e.py")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "discovery-summary.jsonl"
    rebuild_summary(artifact_dir, summary_path)

    common = [
        sys.executable, str(script), "--uri", args.uri,
        "--vector-source", "cohere", "--dim", "768",
        "--metric-type", "COSINE", "--batch-rows", "10000",
        "--predicate-shape", "single", "--expected-segments", "1",
        "--strict-closure", "--queries", str(args.queries),
        "--sparse-threshold", str(args.sparse_threshold),
    ]

    for rows in args.n_values:
        collection = f"sparse_ratio_n{rows}"
        prepare_log = artifact_dir / f"prepare-n{rows}.log"
        if not prepare_log.exists() or "\"event\": \"collection_ready\"" not in prepare_log.read_text(
                encoding="utf-8", errors="replace"):
            prepare = common + [
                "--collection", collection, "--rows", str(rows),
                "--segment-rows", str(rows), "--valid-ratio", str(args.ratios[0]),
                "--force-single-segment", "--prepare-only",
            ]
            run_logged(prepare, prepare_log)

        for ratio in args.ratios:
            for concurrency in args.concurrencies:
                log_path = artifact_dir / (
                    f"search-n{rows}-{ratio_slug(ratio)}-c{concurrency}.log")
                existing = final_record(log_path)
                if existing and existing.get("strict_closure_passed"):
                    print(json.dumps({"event": "resume_skip", "path": str(log_path)}),
                          flush=True)
                    rebuild_summary(artifact_dir, summary_path)
                    continue
                command = common + [
                    "--collection", collection, "--rows", str(rows),
                    "--segment-rows", str(rows), "--valid-ratio", str(ratio),
                    "--concurrency", str(concurrency), "--windows", str(args.windows),
                    "--warmups", "5", "--warmup-seconds", str(args.warmup_seconds),
                    "--reuse-existing",
                ]
                run_logged(command, log_path)
                record = final_record(log_path)
                if not record or not record.get("strict_closure_passed"):
                    raise RuntimeError(f"missing closed benchmark record in {log_path}")
                compact = compact_record(record, log_path)
                rebuild_summary(artifact_dir, summary_path)
                print(json.dumps({"event": "matrix_point_complete", **compact}),
                      flush=True)

        # Keep the cross-N comparison from accumulating loaded Cardinal
        # indexes and changing the QueryNode memory/cache regime at every N.
        release_client = MilvusClient(uri=args.uri)
        release_client.release_collection(collection_name=collection)
        loaded_after_release = release_client.list_loaded_segments(collection)
        if loaded_after_release:
            raise RuntimeError(
                f"collection {collection} still has loaded segments after release: "
                f"{loaded_after_release}")
        print(json.dumps({"event": "collection_released", "collection": collection,
                          "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
