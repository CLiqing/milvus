#!/usr/bin/env python3
import argparse
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pymilvus import MilvusClient


def worker(args, vector, stop_at, stats, lock):
    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    search_params = {
        "metric_type": "COSINE",
        "params": {
            "ef": args.ef,
            "cardinal_expr_downpush": True,
            "cardinal_expr_mod_p": args.mod_p,
            "cardinal_expr_mod_t": args.mod_t,
            "cardinal_expr_filtered_out_count": args.filtered_out_count,
        },
    }
    local_lat = []
    local_ok = 0
    local_err = 0
    while time.time() < stop_at:
        begin = time.perf_counter()
        try:
            client.search(
                args.collection,
                data=[vector],
                anns_field="vector",
                limit=args.topk,
                search_params=search_params,
                filter=f"{args.filter_field} % {args.mod_p} < {args.mod_t}",
                output_fields=["id"],
            )
            local_ok += 1
            local_lat.append((time.perf_counter() - begin) * 1000.0)
        except Exception:
            local_err += 1
    with lock:
        stats["ok"] += local_ok
        stats["err"] += local_err
        stats["lat"].extend(local_lat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="https://in01-c5e4489c88bc122.aws-us-west-2.vectordb-uat3.zillizcloud.com:19541")
    parser.add_argument("--token", default="db_admin:Milvus123")
    parser.add_argument("--collection", default="VDBBenchStatic1M")
    parser.add_argument("--duration", type=float, default=45)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--ef", type=int, default=300)
    parser.add_argument("--filter-field", default="pk")
    parser.add_argument("--mod-p", type=int, default=4)
    parser.add_argument("--mod-t", type=int, default=2)
    parser.add_argument("--filtered-out-count", type=int, default=500000)
    args = parser.parse_args()

    client = MilvusClient(uri=args.uri, token=args.token, timeout=120)
    rows = client.query(args.collection, filter=f"{args.filter_field} >= 0", output_fields=["vector"], limit=1)
    vector = rows[0]["vector"]
    stop_at = time.time() + args.duration
    stats = {"ok": 0, "err": 0, "lat": []}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(worker, args, vector, stop_at, stats, lock) for _ in range(args.concurrency)]
        for future in as_completed(futures):
            future.result()

    elapsed = args.duration
    lat = stats["lat"]
    qps = stats["ok"] / elapsed if elapsed > 0 else 0
    print(f"ok={stats['ok']} err={stats['err']} qps={qps:.2f}")
    if lat:
        lat_sorted = sorted(lat)
        p99 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.99))]
        print(
            "lat_ms avg={:.3f} min={:.3f} max={:.3f} median={:.3f} p99={:.3f}".format(
                statistics.mean(lat), min(lat), max(lat), statistics.median(lat), p99
            )
        )


if __name__ == "__main__":
    main()
