#!/usr/bin/env python3
import argparse
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymilvus import MilvusClient


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]


def filtered_out_count(row_count, mod_p, mod_t):
    keep = (row_count // mod_p) * mod_t + min(row_count % mod_p, mod_t)
    return row_count - keep


def search_params(args, filter_ratio):
    params = {"ef": args.ef}
    if args.mode == "downpush":
        params.update(
            {
                "cardinal_expr_downpush": True,
                "cardinal_expr_mod_p": args.mod_p,
                "cardinal_expr_mod_t": args.mod_t,
                "cardinal_expr_filter_ratio": filter_ratio,
            }
        )
    return {"metric_type": "COSINE", "params": params}


def fetch_vectors(args):
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    try:
        rows = client.query(args.collection, filter="pk >= 0", output_fields=["vector"], limit=args.query_count)
        return [row["vector"] for row in rows]
    finally:
        client.close()


def row_count(args):
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    try:
        return int(client.get_collection_stats(args.collection)["row_count"])
    finally:
        client.close()


def worker(args, vectors, params, filtered_expr, stop_at, stats, lock, worker_id):
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    local_lat = []
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
                    search_params=params,
                    filter=filtered_expr,
                    output_fields=["id"],
                )
                ok += 1
                local_lat.append((time.perf_counter() - begin) * 1000.0)
            except Exception:
                err += 1
    finally:
        client.close()
    with lock:
        stats["ok"] += ok
        stats["err"] += err
        stats["lat"].extend(local_lat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="https://in01-c5e4489c88bc122.aws-us-west-2.vectordb-uat3.zillizcloud.com:19541")
    parser.add_argument("--token", default="db_admin:Milvus123")
    parser.add_argument("--collection", default="VDBBenchStatic10M")
    parser.add_argument("--mode", choices=["baseline", "downpush"], required=True)
    parser.add_argument("--row-count", type=int, default=0)
    parser.add_argument("--mod-p", type=int, default=100)
    parser.add_argument("--mod-t", type=int, default=50)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--concurrency", type=int, default=60)
    parser.add_argument("--query-count", type=int, default=100)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--ef", type=int, default=300)
    args = parser.parse_args()

    vectors = fetch_vectors(args)
    count = args.row_count or row_count(args)
    filtered = filtered_out_count(count, args.mod_p, args.mod_t)
    filter_ratio = filtered / count if count else 0.0
    params = search_params(args, filter_ratio)
    expr = f"pk % {args.mod_p} < {args.mod_t}"

    stats = {"ok": 0, "err": 0, "lat": []}
    lock = threading.Lock()
    stop_at = time.time() + args.duration
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(worker, args, vectors, params, expr, stop_at, stats, lock, i)
            for i in range(args.concurrency)
        ]
        for future in as_completed(futures):
            future.result()

    lat = stats["lat"]
    print(f"mode={args.mode} ok={stats['ok']} err={stats['err']} qps={stats['ok'] / args.duration:.2f}")
    if lat:
        print(
            "lat_ms avg={:.3f} min={:.3f} max={:.3f} median={:.3f} p95={:.3f} p99={:.3f}".format(
                statistics.mean(lat), min(lat), max(lat), statistics.median(lat), percentile(lat, 0.95), percentile(lat, 0.99)
            )
        )


if __name__ == "__main__":
    main()
