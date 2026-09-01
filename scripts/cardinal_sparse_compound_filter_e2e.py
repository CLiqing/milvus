#!/usr/bin/env python3
"""Milvus E2E ABBA: filter-result representation A -> B -> vector search.

The default comparison changes only ``filter_result_representation`` between
``dense`` and ``sparse`` while leaving ``index_algo`` on auto.  Under the
current production policy Dense keeps Cardinal's auto selector and a native
Sparse accepted-ID payload routes directly to BF.  Legacy explicit BF modes
remain available only for historical diagnosis.
"""

import argparse
import concurrent.futures
import decimal
import hashlib
import json
import math
import os
import shutil
import statistics
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
from pymilvus import DataType, MilvusClient

try:
    import pyarrow.parquet as pq
except ImportError:  # synthetic mode does not require pyarrow
    pq = None


ROUTE_COUNTER_NAMES = ("bf_searches", "ivf_searches", "graph_searches")
CARDINAL_WORK_COUNTER_NAMES = (
    "quant_compute_cnt_sum",
    "raw_compute_cnt_sum",
    "re_search_cnt_sum",
)
ADAPTIVE_CACHE_DECISION_NAMES = (
    "adaptive_cache_disabled",
    "adaptive_cache_miss",
    "adaptive_cache_sparse_hit",
    "adaptive_cache_dense_hit",
)
ADAPTIVE_CACHE_PUT_NAMES = (
    "adaptive_cache_sparse_put",
    "adaptive_cache_dense_put",
)
RUNTIME_CONFIG_KEYS = (
    "queryNode.exprCache.enabled",
    "queryNode.grouping.maxNQ",
    "queryNode.segcore.enableSparseFilterResult",
)
RUNTIME_CONFIG_OPTIONAL_KEYS = (
    # These parameters are not registered in every standalone role's
    # management config manager.  Preserve them in the snapshot when the
    # endpoint exposes them, but do not turn an otherwise closed experiment
    # into a false failure when it returns "key not found".  The experiment
    # still closes the actual representation and Cardinal route with counters.
    "autoIndex.twoStageSearch.enabled",
    "queryNode.segcore.sparseResultMaxCardinality",
    "queryNode.segcore.sparseResultMinSegmentRows",
    "queryNode.segcore.sparseResultMaxRatio",
)
METRIC_NAME_MAP = {
    'bf_search_cnt_sum{module="cardinal"}': "bf_searches",
    'ivf_search_cnt_sum{module="cardinal"}': "ivf_searches",
    'graph_search_cnt_sum{module="cardinal"}': "graph_searches",
    'quant_compute_cnt_sum{module="cardinal"}': "quant_compute_cnt_sum",
    'raw_compute_cnt_sum{module="cardinal"}': "raw_compute_cnt_sum",
    're_search_cnt_sum{module="cardinal"}': "re_search_cnt_sum",
    'bf_scan_by_valid_ids_query_cnt_sum{module="cardinal"}': "bf_valid_id_queries",
    'bf_distance_attempt_cnt_sum{module="cardinal"}': "bf_distance_attempts",
    'internal_core_adaptive_filter_output_total{representation="sparse"}': "sparse_outputs",
    'internal_core_adaptive_filter_output_total{representation="dense_threshold"}': "dense_threshold_outputs",
    'internal_core_adaptive_filter_output_total{representation="dense_or_phase1"}': "dense_or_outputs",
    'internal_core_adaptive_filter_cache_total{path="disabled"}': "adaptive_cache_disabled",
    'internal_core_adaptive_filter_cache_total{path="miss"}': "adaptive_cache_miss",
    'internal_core_adaptive_filter_cache_total{path="sparse_hit"}': "adaptive_cache_sparse_hit",
    'internal_core_adaptive_filter_cache_total{path="dense_hit"}': "adaptive_cache_dense_hit",
    'internal_core_adaptive_filter_cache_total{path="sparse_put"}': "adaptive_cache_sparse_put",
    'internal_core_adaptive_filter_cache_total{path="dense_put"}': "adaptive_cache_dense_put",
    'internal_core_search_latency_sum{type="scalar_latency"}': "scalar_latency_ms_sum",
    'internal_core_search_latency_count{type="scalar_latency"}': "scalar_latency_count",
}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def bootstrap_mean_ci(values, confidence=0.95, resamples=10_000, seed=1732):
    """Deterministic percentile-bootstrap CI for paired window deltas."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap CI requires at least one scalar value")
    if not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if array.size == 1:
        value = float(array[0])
        return {"method": "percentile_bootstrap", "confidence": confidence,
                "resamples": 0, "seed": seed, "lower": value,
                "upper": value}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    return {"method": "percentile_bootstrap", "confidence": confidence,
            "resamples": resamples, "seed": seed, "lower": float(lower),
            "upper": float(upper)}


def derive_valid_count(segment_rows, a_limit, valid_ratio):
    """Map a ratio request to one deterministic integer V per segment."""
    if segment_rows <= 0:
        raise ValueError("segment-rows must be positive")
    if valid_ratio is not None:
        if not 0 < valid_ratio <= 1:
            raise ValueError("valid-ratio must be in (0, 1]")
        # CLI ratios are decimal experiment coordinates.  Do not let binary
        # float error turn 0.009 * 50_000 into 449.999... and floor it to 449.
        value = int(
            (decimal.Decimal(str(valid_ratio)) * segment_rows).to_integral_value(
                rounding=decimal.ROUND_FLOOR))
        if value <= 0:
            raise ValueError(
                "valid-ratio rounds to zero accepted rows; increase the ratio or segment size")
        return value
    if a_limit <= 0 or a_limit > segment_rows:
        raise ValueError("a-limit must be in [1, segment-rows]")
    return a_limit


def wait_for_load(client, collection, timeout_seconds=1800):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = client.get_load_state(collection_name=collection).get("state")
        if getattr(state, "name", None) == "Loaded":
            return
        time.sleep(5)
    raise TimeoutError(f"collection {collection} did not load")


def cohere_row_count(cohere_dir, total_shards):
    """Return the available row count without assuming one million rows/shard."""
    total = 0
    for shard in range(total_shards):
        path = os.path.join(
            cohere_dir, f"shuffle_train-{shard:02d}-of-{total_shards}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        total += pq.ParquetFile(path).metadata.num_rows
    return total


def iter_cohere_rows(cohere_dir, total_shards, start_row, row_count, batch_rows):
    """Stream a global Cohere row range, including ranges crossing shard files."""
    skip = start_row
    remaining = row_count
    for shard in range(total_shards):
        path = os.path.join(
            cohere_dir, f"shuffle_train-{shard:02d}-of-{total_shards}.parquet")
        parquet = pq.ParquetFile(path)
        shard_rows = parquet.metadata.num_rows
        if skip >= shard_rows:
            skip -= shard_rows
            continue
        local_skip = skip
        skip = 0
        for batch in parquet.iter_batches(
                batch_size=batch_rows, columns=["emb"], use_threads=True):
            batch_size = batch.num_rows
            if local_skip >= batch_size:
                local_skip -= batch_size
                continue
            take = min(batch_size - local_skip, remaining)
            yield batch.slice(local_skip, take).column("emb").to_pylist()
            remaining -= take
            local_skip = 0
            if remaining == 0:
                return
    raise RuntimeError(
        f"Cohere input ended with {remaining} rows missing from "
        f"range [{start_row}, {start_row + row_count})")


def wait_for_compaction(client, job_id, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        state = client.get_compaction_state(job_id)
        if state == "Completed":
            return
        if state in {"Failed", "Timeout"}:
            raise RuntimeError(f"compaction {job_id} ended in state {state}")
        time.sleep(5)
    raise TimeoutError(f"compaction {job_id} did not complete")


def persistent_segment_rows(client, collection):
    return sorted(
        int(segment.num_rows)
        for segment in client.list_persistent_segments(collection))


def force_single_persistent_segment(client, collection, expected_rows,
                                    target_gb, timeout_seconds):
    """Force-merge a collection and reject it unless one N-row segment remains."""
    rows = persistent_segment_rows(client, collection)
    if rows == [expected_rows]:
        return {"compacted": False, "job_id": None, "segment_rows": rows}
    job_id = client.compact(
        collection_name=collection, target_size=target_gb,
        target_size_unit="gb", timeout=timeout_seconds)
    print(json.dumps({"event": "force_merge_started", "job_id": job_id,
                      "before_segment_rows": rows, "target_gb": target_gb}),
          flush=True)
    wait_for_compaction(client, job_id, timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rows = persistent_segment_rows(client, collection)
        if rows == [expected_rows]:
            return {"compacted": True, "job_id": job_id,
                    "segment_rows": rows}
        time.sleep(5)
    raise RuntimeError(
        "force merge completed but single-segment topology did not converge: "
        f"expected [{expected_rows}], got {rows}")


def hits(result):
    return [(item["id"], float(item["distance"])) for item in result[0]]


def counter_delta(before, after):
    """Return a monotonic counter delta and reject resets/restarts."""
    if set(before) != set(after):
        raise RuntimeError(
            f"metric snapshots have different keys: {sorted(before)} vs {sorted(after)}")
    delta = {name: after[name] - before[name] for name in before}
    negative = {name: value for name, value in delta.items() if value < -1e-6}
    if negative:
        raise RuntimeError(f"metrics decreased during observation: {negative}")
    return delta


def add_counters(destination, source):
    for name, value in source.items():
        destination[name] = destination.get(name, 0.0) + value


def segment_record(segment):
    result = {
        "segment_id": int(segment.segment_id),
        "num_rows": int(segment.num_rows),
        "state": getattr(segment, "state_name", str(segment.state)),
        "level": getattr(segment, "level_name", str(segment.level)),
        "storage_version": int(segment.storage_version),
    }
    for name in ("partition_id", "index_name", "index_id", "node_ids", "mem_size"):
        if hasattr(segment, name):
            value = getattr(segment, name)
            result[name] = list(value) if isinstance(value, (list, tuple)) else value
    return result


def topology_snapshot(client, collection):
    persistent = sorted(
        (segment_record(segment) for segment in client.list_persistent_segments(collection)),
        key=lambda item: item["segment_id"],
    )
    loaded = sorted(
        (segment_record(segment) for segment in client.list_loaded_segments(collection)),
        key=lambda item: item["segment_id"],
    )
    return {
        "persistent_segments": persistent,
        "loaded_segments": loaded,
        "persistent_segment_count": len(persistent),
        "loaded_segment_count": len(loaded),
        "loaded_segment_rows": sorted(item["num_rows"] for item in loaded),
        "actual_n": sum(item["num_rows"] for item in loaded),
    }


def topology_identity(snapshot):
    """Fields whose change means the timed collection topology was unstable."""
    fields = ("segment_id", "num_rows", "state", "level", "storage_version")
    return {
        "persistent": [tuple(item.get(name) for name in fields)
                       for item in snapshot["persistent_segments"]],
        "loaded": [tuple(item.get(name) for name in fields)
                    for item in snapshot["loaded_segments"]],
    }


def management_base_url(management_url, metrics_url):
    if management_url:
        return management_url.rstrip("/")
    if not metrics_url:
        return None
    parsed = urllib.parse.urlsplit(metrics_url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def runtime_config_snapshot(base_url):
    if not base_url:
        raise ValueError("runtime configuration snapshot requires a management URL")
    requested_keys = RUNTIME_CONFIG_KEYS + RUNTIME_CONFIG_OPTIONAL_KEYS
    query = urllib.parse.urlencode({"keys": ",".join(requested_keys)})
    url = f"{base_url}/management/config/get?{query}"
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    entries = payload.get("configs", [])
    by_key = {entry.get("key"): entry for entry in entries}
    missing = [key for key in RUNTIME_CONFIG_KEYS if key not in by_key]
    errors = {key: by_key[key].get("error") for key in RUNTIME_CONFIG_KEYS
              if key in by_key and by_key[key].get("error")}
    if missing or errors:
        raise RuntimeError(
            f"runtime config snapshot incomplete: missing={missing}, errors={errors}")
    snapshot = {
        key: {"value": str(by_key[key].get("value", "")),
              "source": by_key[key].get("source")}
        for key in RUNTIME_CONFIG_KEYS
    }
    for key in RUNTIME_CONFIG_OPTIONAL_KEYS:
        entry = by_key.get(key, {})
        snapshot[key] = {
            "value": (None if not entry or entry.get("error") else
                      str(entry.get("value", ""))),
            "source": entry.get("source"),
            "error": entry.get("error") or ("key missing" if not entry else None),
        }
    return snapshot


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"not a boolean value: {value!r}")


def assert_runtime_config(snapshot, expected_expr_cache_enabled,
                          expected_grouping_max_nq,
                          expected_two_stage_enabled, modes, label):
    actual_expr_cache = parse_bool(
        snapshot["queryNode.exprCache.enabled"]["value"])
    actual_grouping_max_nq = int(
        snapshot["queryNode.grouping.maxNQ"]["value"])
    two_stage_value = snapshot["autoIndex.twoStageSearch.enabled"]["value"]
    if actual_expr_cache != expected_expr_cache_enabled:
        raise AssertionError(
            f"{label} expr cache is {actual_expr_cache}, expected "
            f"{expected_expr_cache_enabled}")
    if actual_grouping_max_nq != expected_grouping_max_nq:
        raise AssertionError(
            f"{label} grouping maxNQ is {actual_grouping_max_nq}, expected "
            f"{expected_grouping_max_nq}")
    if two_stage_value is not None:
        actual_two_stage = parse_bool(two_stage_value)
        if actual_two_stage != expected_two_stage_enabled:
            raise AssertionError(
                f"{label} two-stage search is {actual_two_stage}, expected "
                f"{expected_two_stage_enabled}")
    if "sparse" in modes and not parse_bool(
            snapshot["queryNode.segcore.enableSparseFilterResult"]["value"]):
        raise AssertionError(
            f"{label} adaptive Sparse feature is disabled while sparse mode is requested")


def effective_sparse_caps(snapshot, segment_rows, request_cap):
    """Mirror ComputeSparseFilterResultCap for strict E2E closure."""
    required = (
        "queryNode.segcore.sparseResultMaxCardinality",
        "queryNode.segcore.sparseResultMinSegmentRows",
        "queryNode.segcore.sparseResultMaxRatio",
    )
    missing = [key for key in required if snapshot[key]["value"] is None]
    if missing:
        raise RuntimeError(
            f"runtime config cannot derive effective Sparse caps: missing={missing}")
    global_cap = int(
        snapshot["queryNode.segcore.sparseResultMaxCardinality"]["value"])
    min_rows = int(
        snapshot["queryNode.segcore.sparseResultMinSegmentRows"]["value"])
    max_ratio = float(
        snapshot["queryNode.segcore.sparseResultMaxRatio"]["value"])
    absolute_cap = min(global_cap, request_cap)
    caps = []
    for rows in segment_rows:
        if rows < min_rows:
            caps.append(0)
            continue
        scaled = rows * max_ratio
        epsilon = max(1.0, abs(scaled)) * 1e-12
        caps.append(min(absolute_cap, math.floor(scaled + epsilon)))
    return {
        "global_max_cardinality": global_cap,
        "request_max_cardinality": request_cap,
        "effective_absolute_cap": absolute_cap,
        "min_segment_rows": min_rows,
        "max_ratio": max_ratio,
        "per_segment": caps,
    }


def query_set_sha256(queries):
    array = np.asarray(queries, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def parse_metric_snapshot(lines):
    """Parse the exact metrics required for one closure boundary."""
    values = {}
    for line in lines:
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        if name in METRIC_NAME_MAP:
            values[METRIC_NAME_MAP[name]] = float(value)
    missing = set(METRIC_NAME_MAP.values()) - set(values)
    if missing:
        raise RuntimeError(f"metrics preflight missing counters: {sorted(missing)}")
    return values


def metric_snapshot(url):
    """Read all route/output/work counters at one closure boundary."""
    with urllib.request.urlopen(url, timeout=10) as response:
        lines = response.read().decode("utf-8").splitlines()
    return parse_metric_snapshot(lines)


def count_matches(client, collection, predicate):
    """Return the exact collection-wide cardinality after timed work ends."""
    rows = client.query(collection_name=collection, filter=predicate,
                        output_fields=["count(*)"])
    if len(rows) != 1:
        raise RuntimeError(f"count query returned {len(rows)} rows: {rows}")
    for key in ("count(*)", "count"):
        if key in rows[0]:
            return int(rows[0][key])
    raise RuntimeError(f"count query did not return a count field: {rows[0]}")


def collection_runtime_snapshot(client, collection):
    description = client.describe_collection(collection_name=collection)
    fields = []
    for field in description.get("fields", []):
        field_type = field.get("type")
        try:
            field_type = int(field_type)
        except (TypeError, ValueError):
            field_type = str(field_type)
        params = field.get("params") or {}
        fields.append({
            "name": field.get("name"),
            "type": field_type,
            "is_primary": bool(field.get("is_primary", False)),
            "nullable": bool(field.get("nullable", False)),
            "dim": int(params["dim"]) if "dim" in params else None,
        })

    indexes = []
    for index_name in client.list_indexes(collection_name=collection):
        info = client.describe_index(collection_name=collection, index_name=index_name)
        if isinstance(info, list):
            if len(info) != 1:
                raise RuntimeError(
                    f"describe_index({index_name}) returned {len(info)} entries")
            info = info[0]
        indexes.append({
            "index_name": info.get("index_name", index_name),
            "field_name": info.get("field_name"),
            "index_type": info.get("index_type"),
            "metric_type": info.get("metric_type"),
            "state": str(info.get("state")) if info.get("state") is not None else None,
        })
    indexes.sort(key=lambda item: (str(item["field_name"]), str(item["index_name"])))
    fields.sort(key=lambda item: str(item["name"]))
    return {
        "collection_name": description.get("collection_name", collection),
        "collection_id": description.get("collection_id"),
        "enable_dynamic_field": bool(description.get("enable_dynamic_field", False)),
        "fields": fields,
        "indexes": indexes,
    }


def assert_collection_runtime(snapshot, dim, metric_type):
    fields = {field["name"]: field for field in snapshot["fields"]}
    expected_types = {
        "id": int(DataType.INT64),
        "a": int(DataType.INT64),
        "b": int(DataType.INT64),
        "vector": int(DataType.FLOAT_VECTOR),
    }
    for name, expected_type in expected_types.items():
        if name not in fields:
            raise AssertionError(f"collection is missing required field {name!r}")
        if fields[name]["type"] != expected_type:
            raise AssertionError(
                f"field {name!r} type is {fields[name]['type']}, expected {expected_type}")
    if fields["vector"]["dim"] != dim:
        raise AssertionError(
            f"vector dimension is {fields['vector']['dim']}, expected {dim}")
    if not fields["id"]["is_primary"]:
        raise AssertionError("field 'id' is not the primary key")

    vector_indexes = [index for index in snapshot["indexes"]
                      if index["field_name"] == "vector"]
    if len(vector_indexes) != 1:
        raise AssertionError(
            f"expected one vector index, found {len(vector_indexes)}: {vector_indexes}")
    vector_index = vector_indexes[0]
    if vector_index["index_type"] != "CARDINAL_TIERED":
        raise AssertionError(
            f"vector index is {vector_index['index_type']}, expected CARDINAL_TIERED")
    if str(vector_index["metric_type"]).upper() != metric_type.upper():
        raise AssertionError(
            f"vector index metric is {vector_index['metric_type']}, expected {metric_type}")
    scalar_indexes = [index for index in snapshot["indexes"]
                      if index["field_name"] in {"a", "b"}]
    if scalar_indexes:
        raise AssertionError(
            f"switch-point raw scalar workload unexpectedly has scalar indexes: {scalar_indexes}")


def validate_hit_ids(client, collection, predicate, ids, batch_size=512):
    unique_ids = sorted({int(row_id) for row_id in ids if int(row_id) >= 0})
    missing = []
    for start in range(0, len(unique_ids), batch_size):
        batch = unique_ids[start:start + batch_size]
        expression = f"({predicate}) and id in {json.dumps(batch)}"
        rows = client.query(collection_name=collection, filter=expression,
                            output_fields=["id"])
        accepted = {int(row["id"]) for row in rows}
        missing.extend(row_id for row_id in batch if row_id not in accepted)
    if missing:
        raise AssertionError(
            f"search returned {len(missing)} IDs rejected by predicate: {missing[:20]}")
    return len(unique_ids)


def preflight_summary(mode, counters):
    route_values = {
        "BF": counters.get("bf_searches", 0.0),
        "IVF": counters.get("ivf_searches", 0.0),
        "Graph": counters.get("graph_searches", 0.0),
    }
    active_routes = [name for name, value in route_values.items() if value > 0]
    route = active_routes[0] if len(active_routes) == 1 else (
        "+".join(active_routes) if active_routes else None)

    if mode == "dense":
        representation = "dense"
    elif mode == "sparse":
        sparse = counters.get("sparse_outputs", 0.0)
        fallback = counters.get("dense_threshold_outputs", 0.0)
        if sparse > 0 and fallback > 0:
            representation = "mixed_sparse_dense_threshold"
        elif sparse > 0:
            representation = "sparse"
        elif fallback > 0:
            representation = "dense_threshold"
        else:
            representation = "unknown"
    else:
        representation = "legacy_explicit_bf"

    scalar_count = counters.get("scalar_latency_count", 0.0)
    scalar_sum = counters.get("scalar_latency_ms_sum", 0.0)
    return {
        "representation": representation,
        "route": route,
        "route_counts": route_values,
        "cardinal_work": {
            name: counters.get(name, 0.0)
            for name in CARDINAL_WORK_COUNTER_NAMES
        },
        # Cardinal Tiered Dense-BF may also enumerate valid IDs for chunk
        # pinning.  This is a BF access counter, not proof of Sparse input.
        "bf_scan_by_valid_ids_queries": counters.get("bf_valid_id_queries", 0.0),
        "bf_distance_attempts": counters.get("bf_distance_attempts", 0.0),
        "adaptive_cache": {
            name: counters.get(name, 0.0)
            for name in ADAPTIVE_CACHE_DECISION_NAMES + ADAPTIVE_CACHE_PUT_NAMES
        },
        "scalar_latency_ms_sum": scalar_sum,
        "scalar_latency_count": scalar_count,
        "scalar_latency_ms_per_evaluation": (
            scalar_sum / scalar_count if scalar_count > 0 else None),
    }


def assert_counter(counters, name, expected, mode):
    actual = counters.get(name, 0.0)
    if abs(actual - expected) > 1e-6:
        raise AssertionError(
            f"strict closure failed for {mode}.{name}: expected {expected}, got {actual}")


def assert_mode_closure(counters, mode, request_count, valid_per_segment,
                        sparse_caps, expr_cache_enabled=False, label=None):
    label = label or mode
    segment_count = len(valid_per_segment)
    expected_searches = request_count * segment_count
    actual_searches = sum(counters[name] for name in ROUTE_COUNTER_NAMES)
    if abs(actual_searches - expected_searches) > 1e-6:
        raise AssertionError(
            f"strict closure failed for {label} routes: expected "
            f"{expected_searches}, got {actual_searches}")
    assert_counter(counters, "dense_or_outputs", 0, label)

    if not expr_cache_enabled:
        # With cache off, one scalar evaluation per request/segment also proves
        # that C>1 requests were not merged into a shared FilterBits task.
        assert_counter(counters, "scalar_latency_count", expected_searches, label)

    if mode == "dense":
        assert_counter(counters, "sparse_outputs", 0, label)
        assert_counter(counters, "dense_threshold_outputs", 0, label)
        for name in ADAPTIVE_CACHE_DECISION_NAMES + ADAPTIVE_CACHE_PUT_NAMES:
            assert_counter(counters, name, 0, label)
        # Do not assert bf_valid_id_queries == 0.  Chunked Cardinal Tiered BF
        # also enumerates valid IDs after receiving an ordinary Dense filter.
        return
    if mode != "sparse":
        raise ValueError(f"strict representation closure does not support mode {mode!r}")

    if len(sparse_caps) != segment_count:
        raise AssertionError(
            f"strict closure cap/segment mismatch: {len(sparse_caps)} != {segment_count}")
    sparse_segment_values = [value for value, cap in zip(valid_per_segment, sparse_caps)
                             if cap > 0 and value <= cap]
    fallback_segment_values = [value for value, cap in zip(valid_per_segment, sparse_caps)
                               if cap == 0 or value > cap]
    expected_sparse_outputs = request_count * len(sparse_segment_values)
    expected_fallback_outputs = request_count * len(fallback_segment_values)
    expected_adaptive_tasks = expected_sparse_outputs + expected_fallback_outputs
    assert_counter(counters, "sparse_outputs", expected_sparse_outputs, label)
    assert_counter(counters, "dense_threshold_outputs",
                   expected_fallback_outputs, label)

    cache_decisions = sum(counters[name] for name in ADAPTIVE_CACHE_DECISION_NAMES)
    if abs(cache_decisions - expected_adaptive_tasks) > 1e-6:
        raise AssertionError(
            f"strict closure failed for {label} adaptive cache decisions: "
            f"expected {expected_adaptive_tasks}, got {cache_decisions}")
    if not expr_cache_enabled:
        assert_counter(counters, "adaptive_cache_disabled",
                       expected_adaptive_tasks, label)
        for name in ADAPTIVE_CACHE_DECISION_NAMES[1:] + ADAPTIVE_CACHE_PUT_NAMES:
            assert_counter(counters, name, 0, label)

    sparse_distance_floor = request_count * sum(sparse_segment_values)
    if not fallback_segment_values:
        # Current policy: a canonical Sparse accepted-ID payload directly uses
        # BF even though index_algo remains auto.  Dense is intentionally not
        # required to choose the same route.
        assert_counter(counters, "bf_searches", expected_searches, label)
        assert_counter(counters, "ivf_searches", 0, label)
        assert_counter(counters, "graph_searches", 0, label)
        assert_counter(counters, "bf_valid_id_queries", expected_searches, label)
        assert_counter(counters, "bf_distance_attempts",
                       sparse_distance_floor, label)
    elif sparse_segment_values:
        # Aggregate Cardinal counters cannot map a route back to a physical
        # segment.  Sparse segments provide an exact lower bound; fallback
        # segments may add Dense-BF work to the same counters.
        if counters["bf_searches"] + 1e-6 < expected_sparse_outputs:
            raise AssertionError(
                f"strict closure found fewer BF searches than Sparse outputs "
                f"for {label}: {counters['bf_searches']} < {expected_sparse_outputs}")
        if counters["bf_valid_id_queries"] + 1e-6 < expected_sparse_outputs:
            raise AssertionError(
                f"strict closure found fewer BF valid-ID scans than Sparse "
                f"outputs for {label}: {counters['bf_valid_id_queries']} < "
                f"{expected_sparse_outputs}")
        if counters["bf_distance_attempts"] + 1e-6 < sparse_distance_floor:
            raise AssertionError(
                f"strict closure found fewer BF distance attempts than Sparse "
                f"accepted-ID work for {label}: "
                f"{counters['bf_distance_attempts']} < {sparse_distance_floor}")


def assert_strict_closure(observations, modes, request_counts,
                          valid_per_segment, actual_v, sparse_caps,
                          expr_cache_enabled=False, label="observation"):
    if len(modes) != 2 or set(modes) != {"dense", "sparse"}:
        raise ValueError(
            "--strict-closure requires exactly --modes dense sparse (in either order)")
    if set(observations) != {"dense", "sparse"}:
        raise AssertionError(
            f"strict closure requires {label} counters for dense and sparse")
    if isinstance(request_counts, int):
        request_counts = {mode: request_counts for mode in modes}
    if set(request_counts) != {"dense", "sparse"}:
        raise AssertionError(
            f"strict closure requires {label} request counts for dense and sparse")
    if actual_v is not None and actual_v != sum(valid_per_segment):
        raise AssertionError(
            f"strict closure expected V={sum(valid_per_segment)}, count query returned {actual_v}")

    for mode in ("dense", "sparse"):
        assert_mode_closure(
            observations[mode], mode, request_counts[mode], valid_per_segment,
            sparse_caps, expr_cache_enabled,
            label=f"{label}.{mode}")

    dense = observations["dense"]
    sparse = observations["sparse"]
    sparse_segment_values = [value for value, cap in zip(valid_per_segment, sparse_caps)
                             if cap > 0 and value <= cap]
    fallback_segment_values = [value for value, cap in zip(valid_per_segment, sparse_caps)
                               if cap == 0 or value > cap]
    if not sparse_segment_values and fallback_segment_values:
        if request_counts["dense"] != request_counts["sparse"]:
            raise AssertionError(
                f"cannot compare threshold-Dense routes with unequal request "
                f"counts: {request_counts}")
        # Adaptive exceeded T on every segment and handed an ordinary Dense
        # bitmap to the unchanged selector, so its route/work must match Dense.
        for name in ROUTE_COUNTER_NAMES + ("bf_distance_attempts",):
            assert_counter(sparse, name, dense[name], f"{label}.sparse_threshold")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--token-env", default="LAB_MILVUS_TOKEN")
    parser.add_argument("--collection", default="cardinal_sparse_compound_1m")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--segment-rows", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--metric-type", choices=("L2", "COSINE"), default="L2")
    parser.add_argument("--vector-source", choices=("synthetic", "cohere"), default="synthetic")
    parser.add_argument("--cohere-dir", default="/home/ubuntu/workspace/datasets/vdbbench/cohere_large_10m",
                        help="directory containing shuffle_train-XX-of-10.parquet and test.parquet")
    parser.add_argument("--cohere-total-shards", type=int, default=10,
                        help="total shard count encoded in Cohere filenames; permits a 1M smoke from shard 00")
    parser.add_argument("--batch-rows", type=int, default=10_000)
    parser.add_argument("--a-limit", type=int, default=1_000,
                        help="per-segment accepted rows from predicate A (a < value)")
    parser.add_argument(
        "--valid-ratio", type=float, default=None,
        help=("derive per-segment V as floor(valid-ratio * segment-rows); "
              "use a fraction, e.g. 0.001 for 0.1%%"))
    parser.add_argument("--segment-valid-counts", default=None,
                        help="comma-separated exact A matches per logical segment; used to test mixed Sparse/Dense-threshold output")
    parser.add_argument("--b-limit", type=int, default=None,
                        help="per-segment accepted rank range for B (b < value); default is half a segment")
    parser.add_argument("--predicate-shape", choices=("compound", "single"),
                        default="compound",
                        help="compound uses a < V AND b < limit; single uses only a < V")
    parser.add_argument(
        "--predicate-override", default=None,
        help=("execute this exact predicate while retaining --a-limit as the "
              "expected V for strict closure; intended for equivalent-hit-set "
              "operator comparisons such as '(a %% N) < V'"))
    parser.add_argument("--min-free-gb", type=float, default=0.0,
                        help="abort before a new logical segment when free space falls below this threshold")
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="closed-loop client workers; every worker replays the same fixed query set")
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=0.0,
                        help="per-mode time-based warmup after --warmups")
    parser.add_argument("--sparse-threshold", type=int, default=1_000,
                        help="request-level adaptive Sparse cardinality cap")
    parser.add_argument("--expected-segments", type=int, default=None,
                        help="abort unless Milvus reports this many loaded sealed segments")
    parser.add_argument("--force-single-segment", action="store_true",
                        help="manual-compact after flush unless one N-row persistent segment exists")
    parser.add_argument("--compaction-target-gb", type=int, default=16,
                        help="manual compaction target size used by --force-single-segment")
    parser.add_argument("--compaction-timeout-seconds", type=int, default=7200)
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9092/metrics",
                        help="Prometheus endpoint for route/representation closure")
    parser.add_argument("--management-url", default=None,
                        help="management API base URL; defaults to the metrics URL origin")
    parser.add_argument("--expected-expr-cache-enabled", choices=("false", "true"),
                        default="false",
                        help="runtime config required by strict closure")
    parser.add_argument("--expected-grouping-max-nq", type=int, default=1,
                        help="runtime queryNode.grouping.maxNQ required by strict closure")
    parser.add_argument("--expected-two-stage-enabled", choices=("false", "true"),
                        default="false",
                        help="runtime autoIndex.twoStageSearch.enabled required by strict closure")
    parser.add_argument("--route-preflight-queries", type=int, default=10,
                        help="untimed per-mode requests used to prove the actual Cardinal route")
    parser.add_argument("--strict-closure", action="store_true",
                        help="fail unless representation, route, and BF access counters match the expected path")
    parser.add_argument("--index-algo", choices=("auto", "BF"), default="auto",
                        help="optional diagnostic override; auto route depends on index type and V")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="create/load the deterministic dataset then exit; used before route preflight")
    parser.add_argument("--log-slow-ms", type=float, default=0.0,
                        help="emit a JSON record for timed calls at or above this latency")
    parser.add_argument("--emit-request-samples", action="store_true",
                        help="emit raw timed request records for tail-latency diagnosis")
    parser.add_argument("--modes", nargs="+",
                        choices=("dense", "sparse", "dense_per_query", "valid_ids_per_query"),
                        default=("dense", "sparse"),
                        help="representation modes (default) or legacy explicit BF modes")
    args = parser.parse_args()
    if args.rows % args.segment_rows:
        raise ValueError("rows must be divisible by segment-rows")
    if args.valid_ratio is not None and args.segment_valid_counts:
        raise ValueError("valid-ratio and segment-valid-counts are mutually exclusive")
    a_limit = derive_valid_count(
        args.segment_rows, args.a_limit, args.valid_ratio)
    if args.sparse_threshold <= 0:
        raise ValueError("sparse-threshold must be positive")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.queries <= 0 or args.windows <= 0 or args.warmups < 0:
        raise ValueError("queries/windows must be positive and warmups non-negative")
    if args.expected_grouping_max_nq <= 0:
        raise ValueError("expected-grouping-max-nq must be positive")
    if args.compaction_target_gb <= 0 or args.compaction_timeout_seconds <= 0:
        raise ValueError("compaction target and timeout must be positive")
    if args.strict_closure and (not args.metrics_url or
                                args.route_preflight_queries <= 0):
        raise ValueError(
            "--strict-closure requires --metrics-url and positive "
            "--route-preflight-queries")
    if args.strict_closure and args.index_algo != "auto":
        raise ValueError(
            "--strict-closure requires --index-algo auto so Dense uses the "
            "production selector and Sparse uses the direct-BF policy")

    token = os.environ.get(args.token_env)
    # The isolated local standalone used by the reproducible E2E tests has
    # authorization disabled.  QTP/Cloud callers still provide the token via
    # token-env; do not require a meaningless dummy credential locally.
    client = MilvusClient(uri=args.uri, token=token) if token else MilvusClient(uri=args.uri)
    rng = np.random.default_rng(1732)
    if args.vector_source == "cohere":
        if pq is None:
            raise RuntimeError("Cohere mode requires pyarrow")
        if args.dim != 768 or args.metric_type != "COSINE":
            raise ValueError("Cohere mode requires --dim 768 and --metric-type COSINE")
        available_rows = cohere_row_count(
            args.cohere_dir, args.cohere_total_shards)
        if args.rows > available_rows:
            raise ValueError(
                f"requested {args.rows} Cohere rows, only {available_rows} are available")
        query_path = os.path.join(args.cohere_dir, "test.parquet")
        if not os.path.exists(query_path):
            raise FileNotFoundError(query_path)
    logical_segment_count = args.rows // args.segment_rows
    segment_valid_counts = None
    if args.segment_valid_counts:
        segment_valid_counts = [int(value) for value in args.segment_valid_counts.split(",")]
        if len(segment_valid_counts) != logical_segment_count:
            raise ValueError("segment-valid-counts must contain one value per logical segment")
        if any(value < 0 or value > args.segment_rows for value in segment_valid_counts):
            raise ValueError("segment-valid-counts values must be in [0, segment-rows]")
    b_limit = args.segment_rows // 2 if args.b_limit is None else args.b_limit
    if b_limit < 1 or b_limit > args.segment_rows:
        raise ValueError("b-limit must be in [1, segment-rows]")
    if (args.strict_closure and args.predicate_shape == "compound" and
            b_limit != args.segment_rows):
        raise ValueError(
            "--strict-closure currently requires --predicate-shape single "
            "or a compound predicate with --b-limit equal to segment rows; "
            "otherwise per-segment final V is not observable from aggregate metrics")
    predicate = (args.predicate_override if args.predicate_override is not None
                 else (f"a < {a_limit} and b < {b_limit}"
                       if args.predicate_shape == "compound"
                       else f"a < {a_limit}"))

    if not args.reuse_existing:
        if client.has_collection(args.collection):
            client.drop_collection(args.collection)
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("a", DataType.INT64)
        schema.add_field("b", DataType.INT64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=args.dim)
        client.create_collection(collection_name=args.collection, schema=schema)

        for segment_no, segment_start in enumerate(range(0, args.rows, args.segment_rows)):
            free_gb = shutil.disk_usage("/").free / (1024 ** 3)
            if free_gb < args.min_free_gb:
                raise RuntimeError(
                    f"free space {free_gb:.1f}GiB below --min-free-gb {args.min_free_gb:.1f} before segment {segment_no}")
            # Independent random ranks make A precisely 0.1% and B ~50% of A
            # inside every sealed segment while keeping their matching IDs
            # uncorrelated with vector values and each other.
            if segment_valid_counts is None:
                a = rng.permutation(args.segment_rows).astype(np.int64)
            else:
                # Keep one query predicate while giving each physical segment
                # a different output cardinality.  Values need not be unique:
                # the shuffled positions, not scalar magnitude, determine the
                # exact match set.
                a = np.full(args.segment_rows, a_limit, dtype=np.int64)
                positions = rng.permutation(args.segment_rows)
                a[positions[:segment_valid_counts[segment_no]]] = a_limit - 1
            b = rng.permutation(args.segment_rows).astype(np.int64)
            if args.vector_source == "cohere":
                batches = iter_cohere_rows(
                    args.cohere_dir, args.cohere_total_shards,
                    segment_start, args.segment_rows, args.batch_rows)
            else:
                batches = None
            local_start = 0
            while local_start < args.segment_rows:
                if batches is None:
                    local_end = min(args.segment_rows, local_start + args.batch_rows)
                    vectors = rng.random((local_end - local_start, args.dim), dtype=np.float32)
                else:
                    try:
                        vectors = next(batches)
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"Cohere range ended at global row "
                            f"{segment_start + local_start}") from exc
                    local_end = local_start + len(vectors)
                ids = np.arange(segment_start + local_start, segment_start + local_end)
                client.insert(collection_name=args.collection, data=[
                    {"id": int(row_id), "a": int(a_value), "b": int(b_value),
                     "vector": vector.tolist() if hasattr(vector, "tolist") else vector}
                    for row_id, a_value, b_value, vector in zip(
                        ids, a[local_start:local_end], b[local_start:local_end], vectors)
                ])
                local_start = local_end
            # Preserve the requested multi-segment topology before the vector
            # index is built; no scalar index is created for this experiment.
            client.flush(collection_name=args.collection)
            print(json.dumps({"event": "segment_flushed", "rows": segment_start + args.segment_rows,
                              "free_gib": round(shutil.disk_usage("/").free / (1024 ** 3), 2)}),
                  flush=True)

        force_merge = None
        if args.force_single_segment:
            force_merge = force_single_persistent_segment(
                client, args.collection, args.rows,
                args.compaction_target_gb, args.compaction_timeout_seconds)
            print(json.dumps({"event": "force_merge_complete", **force_merge}),
                  flush=True)

        params = client.prepare_index_params()
        params.add_index(field_name="vector", index_type="CARDINAL_TIERED",
                         metric_type=args.metric_type, params={"M": 16, "efConstruction": 100})
        client.create_index(collection_name=args.collection, index_params=params)
        client.load_collection(collection_name=args.collection)
        wait_for_load(client, args.collection)

    collection_runtime = collection_runtime_snapshot(client, args.collection)
    if args.strict_closure:
        assert_collection_runtime(collection_runtime, args.dim, args.metric_type)

    if args.prepare_only:
        prepared_topology = topology_snapshot(client, args.collection)
        if (args.expected_segments is not None and
                prepared_topology["loaded_segment_count"] != args.expected_segments):
            raise RuntimeError(
                f"expected {args.expected_segments} loaded segments after prepare, got "
                f"{prepared_topology['loaded_segment_count']}: "
                f"{prepared_topology['loaded_segment_rows']}")
        if args.force_single_segment and (
                prepared_topology["loaded_segment_count"] != 1 or
                prepared_topology["loaded_segment_rows"] != [args.rows]):
            raise RuntimeError(
                "force-single-segment prepare gate failed: "
                f"{prepared_topology}")
        print(json.dumps({"event": "collection_ready", "collection": args.collection,
                          "rows": args.rows, "segment_rows": args.segment_rows,
                          "predicate": predicate,
                          "topology": prepared_topology,
                          "collection_runtime": collection_runtime}), flush=True)
        return

    topology_before = topology_snapshot(client, args.collection)
    segment_rows_actual = topology_before["loaded_segment_rows"]
    actual_n = topology_before["actual_n"]
    valid_per_segment = (segment_valid_counts if segment_valid_counts is not None
                         else [a_limit] * logical_segment_count)
    if (args.expected_segments is not None and
            topology_before["loaded_segment_count"] != args.expected_segments):
        raise RuntimeError(
            f"expected {args.expected_segments} loaded segments, got "
            f"{topology_before['loaded_segment_count']}: "
            f"{segment_rows_actual}")
    if args.strict_closure and (
            topology_before["loaded_segment_count"] != logical_segment_count or
            segment_rows_actual != [args.segment_rows] * logical_segment_count):
        raise RuntimeError(
            "--strict-closure requires the physical segment topology to match "
            f"the logical test topology; loaded rows are {segment_rows_actual}")
    print(json.dumps({"event": "segment_snapshot", "phase": "before_timed",
                      **topology_before}), flush=True)

    if args.vector_source == "cohere":
        query_table = pq.read_table(os.path.join(args.cohere_dir, "test.parquet"), columns=["emb"])
        all_queries = query_table["emb"].to_pylist()
        if args.queries > len(all_queries):
            raise ValueError(f"requested {args.queries} queries, Cohere test set has {len(all_queries)}")
        queries = all_queries[:args.queries]
    else:
        query_rng = np.random.default_rng(1732 + 1)
        queries = query_rng.random((args.queries, args.dim), dtype=np.float32)
    query_hash = query_set_sha256(queries)

    def search(mode, query):
        inner_params = {"ef": 64}
        if mode in {"dense", "sparse"}:
            inner_params["filter_result_representation"] = mode
            if mode == "sparse":
                inner_params["sparse_result_max_cardinality"] = args.sparse_threshold
        else:
            inner_params["bf_filter_scan_mode"] = mode
        params = {"metric_type": args.metric_type, "params": inner_params}
        if args.index_algo != "auto":
            params["params"]["index_algo"] = args.index_algo
        begin = time.perf_counter_ns()
        result = client.search(collection_name=args.collection,
                               data=[query.tolist() if hasattr(query, "tolist") else query],
                               anns_field="vector", limit=10, filter=predicate,
                               search_params=params)
        return (time.perf_counter_ns() - begin) / 1_000_000, hits(result)

    modes = tuple(args.modes)
    if args.strict_closure and (len(modes) != 2 or set(modes) != {"dense", "sparse"}):
        raise ValueError(
            "--strict-closure requires exactly --modes dense sparse (in either order)")
    expected_expr_cache_enabled = parse_bool(args.expected_expr_cache_enabled)
    expected_two_stage_enabled = parse_bool(args.expected_two_stage_enabled)
    runtime_management_url = management_base_url(args.management_url, args.metrics_url)
    runtime_config_before = None
    runtime_config_error = None
    sparse_cap_policy = None
    try:
        runtime_config_before = runtime_config_snapshot(runtime_management_url)
        if args.strict_closure:
            assert_runtime_config(
                runtime_config_before, expected_expr_cache_enabled,
                args.expected_grouping_max_nq, expected_two_stage_enabled,
                modes, "before_timed")
            sparse_cap_policy = effective_sparse_caps(
                runtime_config_before, segment_rows_actual,
                args.sparse_threshold)
    except Exception as exc:
        runtime_config_error = f"{type(exc).__name__}: {exc}"
        if args.strict_closure:
            raise RuntimeError(
                "strict closure requires a verified runtime config snapshot") from exc
    print(json.dumps({
        "event": "runtime_preflight",
        "management_url": runtime_management_url,
        "runtime_config": runtime_config_before,
        "runtime_config_error": runtime_config_error,
        "collection_runtime": collection_runtime,
        "query_set_sha256": query_hash,
    }), flush=True)

    # Use one long-lived executor so thread/channel setup is outside timed
    # slots.  Workers are closed-loop: after a response, each immediately
    # submits its next query.  The deterministic phase offset avoids making
    # all workers issue the same vector simultaneously, while Dense/Sparse
    # still see exactly the same per-worker sequence.
    executor = (concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
                if args.concurrency > 1 else None)

    def run_workers(mode, query_count, deadline=None, collect=True):
        if executor is None:
            values = []
            query_number = 0
            while ((deadline is not None and time.monotonic() < deadline) or
                   (deadline is None and query_number < query_count)):
                latency, _ = search(mode, queries[query_number % len(queries)])
                if collect:
                    values.append((0, query_number, latency))
                query_number += 1
            return values

        barrier = threading.Barrier(args.concurrency + 1)

        def worker(worker_number):
            values = []
            query_number = 0
            barrier.wait()
            while ((deadline is not None and time.monotonic() < deadline) or
                   (deadline is None and query_number < query_count)):
                query_index = (query_number + worker_number) % len(queries)
                latency, _ = search(mode, queries[query_index])
                if collect:
                    values.append((worker_number, query_index, latency))
                query_number += 1
            return values

        futures = [executor.submit(worker, worker_number)
                   for worker_number in range(args.concurrency)]
        barrier.wait()
        values = []
        for future in futures:
            values.extend(future.result())
        return values

    for mode in modes:
        run_workers(mode, args.warmups, collect=False)
        if args.warmup_seconds > 0:
            run_workers(mode, 0, deadline=time.monotonic() + args.warmup_seconds,
                        collect=False)
    route_preflight = {}
    route_preflight_summary = {}
    if args.metrics_url and args.route_preflight_queries > 0:
        for mode in modes:
            before = metric_snapshot(args.metrics_url)
            for number in range(args.route_preflight_queries):
                search(mode, queries[number % len(queries)])
            after = metric_snapshot(args.metrics_url)
            route_preflight[mode] = counter_delta(before, after)
            route_preflight_summary[mode] = preflight_summary(
                mode, route_preflight[mode])
        if args.strict_closure:
            assert_strict_closure(
                route_preflight, modes, args.route_preflight_queries,
                valid_per_segment, None, sparse_cap_policy["per_segment"],
                expected_expr_cache_enabled, label="preflight")
        print(json.dumps({"event": "route_preflight", "queries_per_mode":
                          args.route_preflight_queries, "modes": route_preflight,
                          "summary": route_preflight_summary}), flush=True)

    correctness_summary = None
    if len(modes) == 2:
        # Exact TopK equality is appropriate only when both representations run
        # the same search route.  Dense auto-IVF and Sparse direct-BF may
        # legitimately produce different approximate TopK sets; BF is then the
        # full accepted-set reference and overlap/recall is recorded explicitly.
        per_query = []
        all_hit_ids = []
        exact_queries = 0
        recall_values = []
        overlap_values = []
        first_mode, second_mode = modes
        first_route = route_preflight_summary.get(first_mode, {}).get("route")
        second_route = route_preflight_summary.get(second_mode, {}).get("route")
        routes_known = first_route is not None and second_route is not None
        same_route = routes_known and first_route == second_route
        sparse_direct_bf = (
            set(modes) == {"dense", "sparse"} and
            route_preflight_summary.get("sparse", {}).get("route") == "BF" and
            route_preflight_summary.get("sparse", {}).get("representation") == "sparse"
        )
        for number, query in enumerate(queries):
            results = {mode: search(mode, query)[1] for mode in modes}
            first = results[first_mode]
            second = results[second_mode]
            is_exact = first == second
            exact_queries += int(is_exact)
            for mode_hits in results.values():
                ids = [row_id for row_id, _ in mode_hits]
                if len(ids) != len(set(ids)):
                    raise AssertionError(
                        f"duplicate TopK ID at query {number}: {ids}")
                all_hit_ids.extend(ids)
            if set(modes) == {"dense", "sparse"}:
                dense_ids = {row_id for row_id, _ in results["dense"]}
                sparse_ids = {row_id for row_id, _ in results["sparse"]}
                intersection = len(dense_ids & sparse_ids)
                recall_values.append(
                    intersection / len(sparse_ids) if sparse_ids else 1.0)
                union = len(dense_ids | sparse_ids)
                overlap_values.append(intersection / union if union else 1.0)
            if not is_exact and (same_route or not routes_known):
                raise AssertionError(
                    f"topK/distance mismatch on the same or unverified route "
                    f"at query {number}: {first_mode}={first_route}, "
                    f"{second_mode}={second_route}")
            if not is_exact and args.strict_closure and not sparse_direct_bf:
                raise AssertionError(
                    f"topK differs at query {number}, but the observed route "
                    "is not Dense-auto versus Sparse-direct-BF")
            per_query.append({
                "query": number,
                "exact": is_exact,
                "topk_overlap": (overlap_values[-1]
                                  if set(modes) == {"dense", "sparse"} else None),
                "dense_recall_at_k_vs_sparse_bf": (
                    recall_values[-1]
                    if set(modes) == {"dense", "sparse"} else None),
            })
        validated_hit_ids = validate_hit_ids(
            client, args.collection, predicate, all_hit_ids)
        correctness_summary = {
            "queries": len(queries),
            "exact_queries": exact_queries,
            "same_route": same_route,
            "routes": {mode: route_preflight_summary.get(mode, {}).get("route")
                       for mode in modes},
            "validated_unique_hit_ids": validated_hit_ids,
            "mean_topk_overlap": (statistics.mean(overlap_values)
                                  if overlap_values else None),
            "mean_dense_recall_at_k_vs_sparse_bf": (
                statistics.mean(recall_values) if recall_values else None),
            "per_query": per_query,
        }
        print(json.dumps({"event": "correctness_closure",
                          **correctness_summary}), flush=True)

    samples = {mode: [] for mode in modes}
    windows = {mode: [] for mode in modes}
    qps_windows = {mode: [] for mode in modes}
    elapsed = {mode: 0.0 for mode in modes}
    timed_counters = {mode: {} for mode in modes}
    timed_request_counts = {mode: 0 for mode in modes}
    timed_slot_closure = []
    timed_window_closure = []
    request_samples = []
    for window in range(args.windows):
        per_window = {mode: [] for mode in modes}
        per_window_elapsed = {mode: 0.0 for mode in modes}
        per_window_counters = {mode: {} for mode in modes}
        per_window_requests = {mode: 0 for mode in modes}
        order = ((modes[0], modes[1], modes[1], modes[0])
                 if len(modes) == 2 else modes)
        for slot_number, mode in enumerate(order):
            metrics_before = (metric_snapshot(args.metrics_url)
                              if args.metrics_url else None)
            slot_begin = time.perf_counter()
            records = run_workers(mode, len(queries))
            slot_elapsed = time.perf_counter() - slot_begin
            metrics_after = (metric_snapshot(args.metrics_url)
                             if args.metrics_url else None)
            metrics_delta = (counter_delta(metrics_before, metrics_after)
                             if metrics_before is not None else None)
            request_count = len(records)
            if args.strict_closure:
                assert_mode_closure(
                    metrics_delta, mode, request_count, valid_per_segment,
                    sparse_cap_policy["per_segment"], expected_expr_cache_enabled,
                    label=f"timed.window{window + 1}.slot{slot_number + 1}.{mode}")
            if metrics_delta is not None:
                add_counters(timed_counters[mode], metrics_delta)
                add_counters(per_window_counters[mode], metrics_delta)
            timed_request_counts[mode] += request_count
            per_window_requests[mode] += request_count
            slot_closure = {
                "window": window + 1,
                "slot": slot_number + 1,
                "mode": mode,
                "requests": request_count,
                "elapsed_s": slot_elapsed,
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "metrics_delta": metrics_delta,
                "summary": (preflight_summary(mode, metrics_delta)
                            if metrics_delta is not None else None),
            }
            timed_slot_closure.append(slot_closure)
            print(json.dumps({"event": "timed_slot_complete", **slot_closure}),
                  flush=True)
            slot = [latency for _, _, latency in records]
            for worker_number, query_number, latency in records:
                if args.emit_request_samples:
                    request_samples.append({"window": window + 1, "slot": slot_number + 1,
                                            "mode": mode, "worker": worker_number,
                                            "query": query_number,
                                            "latency_ms": latency})
                if args.log_slow_ms and latency >= args.log_slow_ms:
                    print(json.dumps({"event": "slow_request", "window": window + 1,
                                      "mode": mode, "worker": worker_number,
                                      "query": query_number,
                                      "latency_ms": latency}), flush=True)
            samples[mode].extend(slot)
            per_window[mode].extend(slot)
            elapsed[mode] += slot_elapsed
            per_window_elapsed[mode] += slot_elapsed
        for mode in modes:
            windows[mode].append(statistics.mean(per_window[mode]))
            qps_windows[mode].append(
                per_window_requests[mode] / per_window_elapsed[mode])
        if args.strict_closure:
            assert_strict_closure(
                per_window_counters, modes, per_window_requests,
                valid_per_segment, None, sparse_cap_policy["per_segment"],
                expected_expr_cache_enabled,
                label=f"timed.window{window + 1}")
        window_closure = {
            "window": window + 1,
            "requests": per_window_requests,
            "metric_deltas": per_window_counters,
            "summaries": {
                mode: (preflight_summary(mode, per_window_counters[mode])
                       if per_window_counters[mode] else None)
                for mode in modes
            },
        }
        timed_window_closure.append(window_closure)
        print(json.dumps({"event": "window_complete", **window_closure}), flush=True)

    topology_after = topology_snapshot(client, args.collection)
    topology_stable = topology_identity(topology_before) == topology_identity(topology_after)
    if args.strict_closure and not topology_stable:
        raise RuntimeError(
            "segment topology changed during timed work: "
            f"before={topology_identity(topology_before)}, "
            f"after={topology_identity(topology_after)}")
    print(json.dumps({"event": "segment_snapshot", "phase": "after_timed",
                      "stable": topology_stable, **topology_after}), flush=True)

    runtime_config_after = None
    runtime_config_after_error = None
    try:
        runtime_config_after = runtime_config_snapshot(runtime_management_url)
        if args.strict_closure:
            assert_runtime_config(
                runtime_config_after, expected_expr_cache_enabled,
                args.expected_grouping_max_nq, expected_two_stage_enabled,
                modes, "after_timed")
            if runtime_config_after != runtime_config_before:
                raise AssertionError(
                    "runtime configuration or source changed during timed work")
    except Exception as exc:
        runtime_config_after_error = f"{type(exc).__name__}: {exc}"
        if args.strict_closure:
            raise RuntimeError(
                "strict closure requires a stable post-timed runtime config snapshot") from exc

    actual_v = None
    actual_v_error = None
    try:
        # Run this only after all timed slots so the exact count cannot warm or
        # populate expression state used by the benchmark itself.
        actual_v = count_matches(client, args.collection, predicate)
    except Exception as exc:  # keep historical non-strict invocations usable
        actual_v_error = f"{type(exc).__name__}: {exc}"
        if args.strict_closure:
            raise RuntimeError("strict closure requires an exact V count") from exc
        print(json.dumps({"event": "actual_v_unavailable",
                          "error": actual_v_error}), flush=True)

    strict_closure_passed = False
    if args.strict_closure:
        assert_strict_closure(
            route_preflight, modes, args.route_preflight_queries,
            valid_per_segment, actual_v, sparse_cap_policy["per_segment"],
            expected_expr_cache_enabled, label="preflight")
        assert_strict_closure(
            timed_counters, modes, timed_request_counts,
            valid_per_segment, actual_v, sparse_cap_policy["per_segment"],
            expected_expr_cache_enabled, label="timed")
        strict_closure_passed = True

    expected_final_valid = (
        sum(valid_per_segment)
        if args.predicate_shape == "single"
        else sum(valid_per_segment) * b_limit / args.segment_rows)
    mode_reports = {}
    for mode in modes:
        preflight = route_preflight_summary.get(mode, {})
        timed_summary = (preflight_summary(mode, timed_counters[mode])
                         if timed_counters[mode] else {})
        mode_reports[mode] = {
            "requests": len(samples[mode]),
            "timed_counter_requests": timed_request_counts[mode],
            "elapsed_s": elapsed[mode],
            "qps": len(samples[mode]) / elapsed[mode],
            "mean_ms": statistics.mean(samples[mode]),
            "median_ms": statistics.median(samples[mode]),
            "p90_ms": percentile(samples[mode], 0.90),
            "representation": timed_summary.get(
                "representation", preflight.get("representation")),
            "route": timed_summary.get("route", preflight.get("route")),
            "preflight_counters": route_preflight.get(mode),
            "preflight_summary": preflight or None,
            "timed_counters": timed_counters[mode] or None,
            "timed_summary": timed_summary or None,
            "window_mean_ms": windows[mode],
            "window_qps": qps_windows[mode],
        }

    report = {
        "event": "benchmark_complete", "collection": args.collection,
        "rows": args.rows, "segment_rows": args.segment_rows, "dim": args.dim,
        "vector_source": args.vector_source, "metric_type": args.metric_type,
        "predicate": predicate, "predicate_shape": args.predicate_shape,
        "actual_n": actual_n, "actual_v": actual_v,
        "actual_v_over_n": (actual_v / actual_n
                            if actual_v is not None and actual_n > 0 else None),
        "requested_valid_ratio": args.valid_ratio,
        "derived_a_limit": a_limit,
        "actual_v_error": actual_v_error,
        "threshold_t": args.sparse_threshold,
        "a_valid_per_segment": a_limit if segment_valid_counts is None else None,
        "a_valid_per_segment_values": valid_per_segment,
        "segment_valid_counts": segment_valid_counts,
        "sparse_threshold": args.sparse_threshold,
        "sparse_cap_policy": sparse_cap_policy,
        "b_selectivity": (b_limit / args.segment_rows
                          if args.predicate_shape == "compound" else None),
        "final_valid_expected": expected_final_valid,
        "persistent_segments": topology_before["persistent_segment_count"],
        "loaded_segments": topology_before["loaded_segment_count"],
        "loaded_segment_rows": segment_rows_actual,
        "topology_before": topology_before,
        "topology_after": topology_after,
        "topology_stable": topology_stable,
        "collection_runtime": collection_runtime,
        "query_set_sha256": query_hash,
        "query_schedule": {
            "fixed_query_count": len(queries),
            "per_worker_order": "(request_number + worker_number) % query_count",
            "abba_order": list((modes[0], modes[1], modes[1], modes[0]))
                          if len(modes) == 2 else list(modes),
        },
        "nq": 1, "concurrency": args.concurrency, "index_algo_request": args.index_algo,
        "actual_payload_route": (
            "dense_auto_sparse_direct_bf_or_threshold_dense"
            if set(modes).issubset({"dense", "sparse"}) else "legacy_explicit_BF"
        ),
        # Keep queries_per_slot for compatibility; actual request counts are
        # explicit because C>1 runs this many queries per worker.
        "queries_per_slot": args.queries,
        "queries_per_worker_per_slot": args.queries,
        "expected_requests_per_slot": args.queries * args.concurrency,
        "timing_scope": "run_workers_only; metrics/logging/closure excluded",
        "timed_request_counts": timed_request_counts,
        "windows": args.windows,
        "route_preflight": route_preflight,
        "route_preflight_summary": route_preflight_summary,
        "timed_slot_closure": timed_slot_closure,
        "timed_window_closure": timed_window_closure,
        "runtime_constraints": {
            "management_url": runtime_management_url,
            "expected_expr_cache_enabled": expected_expr_cache_enabled,
            "expected_grouping_max_nq": args.expected_grouping_max_nq,
            "expected_two_stage_enabled": expected_two_stage_enabled,
            "before": runtime_config_before,
            "before_error": runtime_config_error,
            "after": runtime_config_after,
            "after_error": runtime_config_after_error,
        },
        "correctness": correctness_summary,
        "strict_closure_requested": args.strict_closure,
        "strict_closure_passed": strict_closure_passed,
        "modes": mode_reports,
    }
    if set(modes) == {"dense", "sparse"}:
        dense_mode, sparse_mode = "dense", "sparse"
        dense_mean = statistics.mean(samples[dense_mode])
        sparse_mean = statistics.mean(samples[sparse_mode])
        paired = [(dense - sparse) / dense for dense, sparse in zip(
            windows[dense_mode], windows[sparse_mode])]
        paired_qps = [(sparse - dense) / dense for dense, sparse in zip(
            qps_windows[dense_mode], qps_windows[sparse_mode])]
        qps_ci = bootstrap_mean_ci(paired_qps)
        report["sparse_vs_dense_mean_delta"] = (dense_mean - sparse_mean) / dense_mean
        report["paired_windows"] = {
            "count": len(paired),
            # Compatibility keys retain the historical latency-improvement
            # meaning; explicit latency/QPS names remove ambiguity.
            "mean_delta": statistics.mean(paired),
            "median_delta": statistics.median(paired),
            "sparse_faster_windows": sum(value > 0 for value in paired),
            "latency_mean_improvement": statistics.mean(paired),
            "latency_median_improvement": statistics.median(paired),
            "latency_improvement_values": paired,
            "qps_mean_delta": statistics.mean(paired_qps),
            "qps_median_delta": statistics.median(paired_qps),
            "qps_delta_values": paired_qps,
            "qps_mean_delta_95pct_ci": qps_ci,
            "qps_no_worse_than_minus_5pct": qps_ci["lower"] >= -0.05,
            "aa_noise_collected": False,
        }
    elif len(modes) == 2:
        first_mode, second_mode = modes
        first_mean = statistics.mean(samples[first_mode])
        second_mean = statistics.mean(samples[second_mode])
        paired = [(first - second) / first for first, second in zip(
            windows[first_mode], windows[second_mode])]
        paired_qps = [(second - first) / first for first, second in zip(
            qps_windows[first_mode], qps_windows[second_mode])]
        report["second_vs_first_mean_delta"] = (first_mean - second_mean) / first_mean
        report["paired_windows"] = {
            "first_mode": first_mode,
            "second_mode": second_mode,
            "count": len(paired),
            "mean_delta": statistics.mean(paired),
            "median_delta": statistics.median(paired),
            "second_faster_windows": sum(value > 0 for value in paired),
            "latency_improvement_values": paired,
            "qps_delta_values": paired_qps,
            "qps_mean_delta_95pct_ci": bootstrap_mean_ci(paired_qps),
        }
    if args.emit_request_samples:
        print(json.dumps({"event": "request_samples", "samples": request_samples}), flush=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    if executor is not None:
        executor.shutdown()


if __name__ == "__main__":
    main()
