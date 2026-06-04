import json
import random
import time

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


HOST = "10.15.10.217"
PORT = "19530"
COLLECTION = "cardinal_expr_downpush_correctness"
DIM = 64
NROWS = 20000
NQ = 50
TOPK = 20
CASES = [
    {"name": "filter_1pct", "mod_p": 100, "mod_t": 99},
    {"name": "filter_20pct", "mod_p": 100, "mod_t": 80},
    {"name": "filter_50pct", "mod_p": 4, "mod_t": 2},
    {"name": "filter_99pct", "mod_p": 100, "mod_t": 1},
]
OUTPUT_JSON = "cardinal_correctness_results_20260604.json"


def wait_loading(collection):
    for _ in range(120):
        progress = utility.loading_progress(collection.name)
        if progress.get("loading_progress") == "100%":
            return
        time.sleep(1)
    raise RuntimeError(f"load timeout: {progress}")


def make_vectors(n, dim):
    rng = np.random.default_rng(20260604)
    data = rng.random((n, dim), dtype=np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return (data / np.maximum(norms, 1e-12)).astype(np.float32)


def main():
    connections.connect(alias="default", host=HOST, port=PORT)
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)

    schema = CollectionSchema(
        fields=[
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        ],
        enable_dynamic_field=False,
    )
    collection = Collection(COLLECTION, schema=schema)

    vectors = make_vectors(NROWS, DIM)
    ids = list(range(NROWS))
    batch = 2000
    for start in range(0, NROWS, batch):
        end = min(start + batch, NROWS)
        collection.insert([ids[start:end], vectors[start:end].tolist()])
    collection.flush()

    index_params = {
        "index_type": "CARDINAL_TIERED",
        "metric_type": "COSINE",
        "params": {},
    }
    collection.create_index("vec", index_params)
    collection.load()
    wait_loading(collection)

    rng = random.Random(20260604)
    query_ids = [rng.randrange(NROWS) for _ in range(NQ)]
    queries = [vectors[i].tolist() for i in query_ids]

    case_results = []
    for case in CASES:
        mod_p = case["mod_p"]
        mod_t = case["mod_t"]
        keep_count = (NROWS // mod_p) * mod_t + min(NROWS % mod_p, mod_t)
        filtered_out_count = NROWS - keep_count
        expr = f"id % {mod_p} < {mod_t}"
        baseline_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        downpush_params = {
            "metric_type": "COSINE",
            "params": {
                "ef": 128,
                "cardinal_expr_downpush": True,
                "cardinal_expr_mod_p": mod_p,
                "cardinal_expr_mod_t": mod_t,
                "cardinal_expr_filtered_out_count": filtered_out_count,
            },
        }

        baseline = collection.search(
            data=queries,
            anns_field="vec",
            param=baseline_params,
            limit=TOPK,
            expr=expr,
            output_fields=["id"],
            consistency_level="Strong",
        )
        downpush = collection.search(
            data=queries,
            anns_field="vec",
            param=downpush_params,
            limit=TOPK,
            expr=expr,
            output_fields=["id"],
            consistency_level="Strong",
        )

        mismatches = []
        for qi, (base_hits, down_hits) in enumerate(zip(baseline, downpush)):
            base_ids = [hit.id for hit in base_hits]
            down_ids = [hit.id for hit in down_hits]
            if base_ids != down_ids:
                mismatches.append(
                    {
                        "query_index": qi,
                        "query_id": query_ids[qi],
                        "baseline": base_ids,
                        "downpush": down_ids,
                    }
                )

        case_results.append(
            {
                "name": case["name"],
                "expr": expr,
                "mod_p": mod_p,
                "mod_t": mod_t,
                "filtered_out_count": filtered_out_count,
                "filtered_out_ratio": filtered_out_count / NROWS,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches[:5],
            }
        )

    result = {
        "collection": COLLECTION,
        "nrows": NROWS,
        "nq": NQ,
        "topk": TOPK,
        "index_params": index_params,
        "cases": case_results,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))
    if any(case["mismatch_count"] for case in case_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
