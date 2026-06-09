import json
import os
import time

import pyarrow.parquet as pq
from pymilvus import MilvusClient


URI = os.getenv(
    "MILVUS_URI",
    "https://in01-86da8fd150a1408.aws-us-west-2.vectordb-uat3.zillizcloud.com:19541",
)
USER = os.getenv("MILVUS_USER", "db_admin")
PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus123")
COLLECTION = os.getenv("MILVUS_COLLECTION", "VDBBenchStd10MLevel1")
TEST_PARQUET = os.getenv(
    "VDBBENCH_TEST_PARQUET",
    "/home/ubuntu/data/vectordb_bench/dataset/cohere/cohere_large_10m/test.parquet",
)
OUTPUT_JSON = os.getenv(
    "OUTPUT_JSON",
    "/home/ubuntu/workspace/TmpWorker/qtp_records/smoke_threshold_downpush.json",
)

NQ = int(os.getenv("NQ", "20"))
TOPK = int(os.getenv("TOPK", "100"))
LEVEL = int(os.getenv("LEVEL", "1"))
THRESHOLD = int(os.getenv("THRESHOLD", "5000000"))
FILTERED_OUT_COUNT = int(os.getenv("FILTERED_OUT_COUNT", str(THRESHOLD)))
FILTER_RATIO = float(os.getenv("FILTER_RATIO", "0.5"))


def hit_ids(hits):
    return [int(hit["entity"]["id"]) for hit in hits]


def invalid_ids(ids):
    return [id_value for id_value in ids if id_value < THRESHOLD]


def main():
    started = time.time()
    table = pq.read_table(TEST_PARQUET, columns=["emb"])
    queries = table.slice(0, NQ).column("emb").to_pylist()

    client = MilvusClient(uri=URI, user=USER, password=PASSWORD, timeout=120)
    expr = f"id >= {THRESHOLD}"
    baseline_params = {
        "metric_type": "COSINE",
        "params": {
            "level": LEVEL,
        },
    }
    downpush_params = {
        "metric_type": "COSINE",
        "params": {
            "level": LEVEL,
            "cardinal_expr_downpush": True,
            "cardinal_expr_threshold": THRESHOLD,
            "cardinal_expr_filtered_out_count": FILTERED_OUT_COUNT,
            "cardinal_expr_filter_ratio": FILTER_RATIO,
        },
    }

    baseline = client.search(
        collection_name=COLLECTION,
        data=queries,
        anns_field="vector",
        search_params=baseline_params,
        limit=TOPK,
        filter=expr,
        output_fields=["id"],
        consistency_level="Strong",
    )
    downpush = client.search(
        collection_name=COLLECTION,
        data=queries,
        anns_field="vector",
        search_params=downpush_params,
        limit=TOPK,
        filter=expr,
        output_fields=["id"],
        consistency_level="Strong",
    )
    client.close()

    records = []
    same_order_count = 0
    same_set_count = 0
    baseline_invalid_total = 0
    downpush_invalid_total = 0
    empty_downpush_count = 0
    for idx, (base_hits, down_hits) in enumerate(zip(baseline, downpush)):
        base_ids = hit_ids(base_hits)
        down_ids = hit_ids(down_hits)
        base_invalid = invalid_ids(base_ids)
        down_invalid = invalid_ids(down_ids)
        same_order = base_ids == down_ids
        same_set = set(base_ids) == set(down_ids)
        same_order_count += int(same_order)
        same_set_count += int(same_set)
        baseline_invalid_total += len(base_invalid)
        downpush_invalid_total += len(down_invalid)
        empty_downpush_count += int(len(down_ids) == 0)
        records.append(
            {
                "query": idx,
                "same_order": same_order,
                "same_set": same_set,
                "baseline_count": len(base_ids),
                "downpush_count": len(down_ids),
                "baseline_invalid": len(base_invalid),
                "downpush_invalid": len(down_invalid),
                "baseline_head": base_ids[:10],
                "downpush_head": down_ids[:10],
            }
        )

    result = {
        "collection": COLLECTION,
        "expr": expr,
        "threshold": THRESHOLD,
        "filter_ratio": FILTER_RATIO,
        "filtered_out_count": FILTERED_OUT_COUNT,
        "level": LEVEL,
        "queries": NQ,
        "topk": TOPK,
        "same_order_count": same_order_count,
        "same_set_count": same_set_count,
        "baseline_invalid_total": baseline_invalid_total,
        "downpush_invalid_total": downpush_invalid_total,
        "empty_downpush_count": empty_downpush_count,
        "elapsed_sec": round(time.time() - started, 4),
        "records": records,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))

    if baseline_invalid_total or downpush_invalid_total or empty_downpush_count:
        raise SystemExit(1)
    if same_order_count != NQ:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
