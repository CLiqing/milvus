#!/usr/bin/env python3
"""Run 12-window confirmation around the ratio sweep boundaries."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pymilvus import MilvusClient

from run_sparse_ratio_switchpoint_matrix import (
    compact_record,
    final_record,
    ratio_slug,
    run_logged,
)


RATIOS = (
    0.0005,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.010,
)

MATRIX = {
    50_000: RATIOS,
    100_000: RATIOS,
    250_000: RATIOS,
    1_000_000: RATIOS,
    3_000_000: RATIOS,
}


def rebuild_summary(artifact_dir, summary_path):
    records = []
    for log_path in artifact_dir.glob("formal-n*-r*pct-c*.log"):
        record = final_record(log_path)
        if record and record.get("strict_closure_passed"):
            compact = compact_record(record, log_path)
            compact["qps_delta_95pct_ci"] = record["paired_windows"][
                "qps_mean_delta_95pct_ci"]
            compact["qps_no_worse_than_minus_5pct"] = record[
                "paired_windows"]["qps_no_worse_than_minus_5pct"]
            compact["windows"] = record["windows"]
            records.append(compact)
    records.sort(key=lambda item: (item["N"], item["ratio"], item["concurrency"]))
    temporary = summary_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="http://127.0.0.1:19532")
    parser.add_argument("--artifact-dir", default="artifacts/sparse-ratio-switchpoint-20260831/formal")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=(1, 60))
    args = parser.parse_args()

    script = Path(__file__).with_name("cardinal_sparse_compound_filter_e2e.py")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "formal-summary.jsonl"
    rebuild_summary(artifact_dir, summary_path)
    client = MilvusClient(uri=args.uri)

    common = [
        sys.executable, str(script), "--uri", args.uri,
        "--vector-source", "cohere", "--dim", "768",
        "--metric-type", "COSINE", "--batch-rows", "10000",
        "--predicate-shape", "single", "--expected-segments", "1",
        "--strict-closure", "--queries", "50", "--sparse-threshold", "30000",
        "--windows", "12", "--warmups", "5", "--warmup-seconds", "30",
        "--reuse-existing",
    ]

    for rows, ratios in MATRIX.items():
        collection = f"sparse_ratio_n{rows}"
        client.load_collection(collection_name=collection)
        for ratio in ratios:
            for concurrency in args.concurrencies:
                log_path = artifact_dir / (
                    f"formal-n{rows}-{ratio_slug(ratio)}-c{concurrency}.log")
                existing = final_record(log_path)
                if existing and existing.get("strict_closure_passed"):
                    print(json.dumps({"event": "formal_resume_skip",
                                      "path": str(log_path)}), flush=True)
                    continue
                command = common + [
                    "--collection", collection, "--rows", str(rows),
                    "--segment-rows", str(rows), "--valid-ratio", str(ratio),
                    "--concurrency", str(concurrency),
                ]
                run_logged(command, log_path)
                record = final_record(log_path)
                if not record or not record.get("strict_closure_passed"):
                    raise RuntimeError(f"missing closed formal record in {log_path}")
                rebuild_summary(artifact_dir, summary_path)
                print(json.dumps({"event": "formal_point_complete",
                                  **compact_record(record, log_path),
                                  "qps_delta_95pct_ci": record["paired_windows"][
                                      "qps_mean_delta_95pct_ci"]}), flush=True)
        client.release_collection(collection_name=collection)
        if client.list_loaded_segments(collection):
            raise RuntimeError(f"{collection} remained loaded after formal run")
        print(json.dumps({"event": "formal_collection_released",
                          "collection": collection}), flush=True)


if __name__ == "__main__":
    main()
