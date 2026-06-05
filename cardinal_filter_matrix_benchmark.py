#!/usr/bin/env python3
import argparse
import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from pymilvus import MilvusClient


def utc_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(len(ordered) * pct) - 1)
    return ordered[idx]


def keep_threshold(filter_rate):
    return int(round(100 - filter_rate))


def filtered_out_count(row_count, mod_p, mod_t):
    keep = (row_count // mod_p) * mod_t + min(row_count % mod_p, mod_t)
    return row_count - keep


def make_search_params(args, downpush, mod_p, mod_t, filter_ratio):
    params = {"ef": args.ef}
    if downpush:
        params.update(
            {
                "cardinal_expr_downpush": True,
                "cardinal_expr_mod_p": mod_p,
                "cardinal_expr_mod_t": mod_t,
                "cardinal_expr_filter_ratio": filter_ratio,
            }
        )
    return {"metric_type": "COSINE", "params": params}


def extract_ids(search_result):
    ids = []
    for hit in search_result[0]:
        if isinstance(hit, dict):
            if "id" in hit:
                ids.append(hit["id"])
            elif "pk" in hit:
                ids.append(hit["pk"])
            elif "entity" in hit and "id" in hit["entity"]:
                ids.append(hit["entity"]["id"])
            elif "entity" in hit and "pk" in hit["entity"]:
                ids.append(hit["entity"]["pk"])
            else:
                ids.append(str(hit))
        else:
            ids.append(getattr(hit, "id", str(hit)))
    return ids


def fetch_query_vectors(args):
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    try:
        rows = client.query(
            args.collection,
            filter="pk >= 0",
            output_fields=["pk", "vector"],
            limit=args.query_count,
        )
        if not rows:
            raise RuntimeError("no query vectors fetched from collection")
        return [row["vector"] for row in rows]
    finally:
        client.close()


def collection_row_count(args):
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    try:
        stats = client.get_collection_stats(args.collection)
        return int(stats["row_count"])
    finally:
        client.close()


def search_once(args, vector, expr, search_params):
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    try:
        return extract_ids(
            client.search(
                args.collection,
                data=[vector],
                anns_field="vector",
                limit=args.topk,
                search_params=search_params,
                filter=expr,
                output_fields=["id"],
            )
        )
    finally:
        client.close()


def correctness_check(args, vectors, filter_rate, mod_p, mod_t, filter_ratio):
    expr = f"pk % {mod_p} < {mod_t}"
    baseline_params = make_search_params(args, False, mod_p, mod_t, filter_ratio)
    downpush_params = make_search_params(args, True, mod_p, mod_t, filter_ratio)
    mismatches = []
    query_count = min(args.correctness_queries, len(vectors))
    for idx, vector in enumerate(vectors[:query_count]):
        baseline = search_once(args, vector, expr, baseline_params)
        downpush = search_once(args, vector, expr, downpush_params)
        if baseline != downpush:
            mismatches.append(
                {
                    "query_index": idx,
                    "ordered_equal": False,
                    "set_equal": set(baseline) == set(downpush),
                    "baseline_head": baseline[:10],
                    "downpush_head": downpush[:10],
                }
            )
    return {
        "filter_rate": filter_rate,
        "query_count": query_count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[: args.max_mismatches],
    }


def perf_worker(args, vectors, expr, search_params, stop_at, stats, lock, worker_id):
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    latencies = []
    ok = 0
    err = 0
    idx = worker_id
    try:
        while time.time() < stop_at:
            vector = vectors[idx % len(vectors)]
            idx += args.concurrency
            begin = time.perf_counter()
            try:
                client.search(
                    args.collection,
                    data=[vector],
                    anns_field="vector",
                    limit=args.topk,
                    search_params=search_params,
                    filter=expr,
                    output_fields=["id"],
                )
                ok += 1
                latencies.append((time.perf_counter() - begin) * 1000.0)
            except Exception:
                err += 1
    finally:
        client.close()

    with lock:
        stats["ok"] += ok
        stats["err"] += err
        stats["latencies"].extend(latencies)


def run_perf(args, vectors, filter_rate, mode, mod_p, mod_t, filter_ratio, round_id):
    expr = f"pk % {mod_p} < {mod_t}"
    params = make_search_params(args, mode == "downpush", mod_p, mod_t, filter_ratio)
    stats = {"ok": 0, "err": 0, "latencies": []}
    lock = threading.Lock()

    stop_at = time.time() + args.duration
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(perf_worker, args, vectors, expr, params, stop_at, stats, lock, i)
            for i in range(args.concurrency)
        ]
        for future in as_completed(futures):
            future.result()

    lat = stats["latencies"]
    return {
        "filter_rate": filter_rate,
        "mode": mode,
        "round": round_id,
        "ok": stats["ok"],
        "err": stats["err"],
        "qps": stats["ok"] / args.duration,
        "lat_ms": {
            "avg": statistics.mean(lat) if lat else None,
            "min": min(lat) if lat else None,
            "max": max(lat) if lat else None,
            "median": statistics.median(lat) if lat else None,
            "p95": percentile(lat, 0.95),
            "p99": percentile(lat, 0.99),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="https://in01-c5e4489c88bc122.aws-us-west-2.vectordb-uat3.zillizcloud.com:19541")
    parser.add_argument("--token", default="db_admin:Milvus123")
    parser.add_argument("--collection", default="VDBBenchStatic10M")
    parser.add_argument("--row-count", type=int, default=0)
    parser.add_argument("--rates", default="1,5,20,40,50,60,80")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--concurrency", type=int, default=60)
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--correctness-queries", type=int, default=20)
    parser.add_argument("--max-mismatches", type=int, default=5)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--ef", type=int, default=300)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()

    row_count = args.row_count or collection_row_count(args)
    rates = [int(x.strip()) for x in args.rates.split(",") if x.strip()]
    vectors = fetch_query_vectors(args)

    result = {
        "started_at": utc_now(),
        "collection": args.collection,
        "row_count": row_count,
        "rates": rates,
        "rounds": args.rounds,
        "duration": args.duration,
        "concurrency": args.concurrency,
        "query_count": len(vectors),
        "topk": args.topk,
        "ef": args.ef,
        "correctness": [],
        "performance": [],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for rate in rates:
        mod_p = 100
        mod_t = keep_threshold(rate)
        filtered = filtered_out_count(row_count, mod_p, mod_t)
        filter_ratio = filtered / row_count if row_count else rate / 100.0
        if not args.skip_correctness:
            check = correctness_check(args, vectors, rate, mod_p, mod_t, filter_ratio)
            check["filtered_out_count_collection"] = filtered
            check["filter_ratio"] = filter_ratio
            result["correctness"].append(check)
            print(f"correctness rate={rate}% mismatch={check['mismatch_count']}")
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        for round_id in range(1, args.rounds + 1):
            for mode in ("baseline", "downpush"):
                perf = run_perf(args, vectors, rate, mode, mod_p, mod_t, filter_ratio, round_id)
                perf["filtered_out_count_collection"] = filtered
                perf["filter_ratio"] = filter_ratio
                result["performance"].append(perf)
                print(
                    "perf rate={} mode={} round={} qps={:.2f} avg={:.3f} p99={:.3f} err={}".format(
                        rate,
                        mode,
                        round_id,
                        perf["qps"],
                        perf["lat_ms"]["avg"] or 0,
                        perf["lat_ms"]["p99"] or 0,
                        perf["err"],
                    )
                )
                output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    result["finished_at"] = utc_now()
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
