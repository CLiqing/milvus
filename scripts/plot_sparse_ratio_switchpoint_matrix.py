#!/usr/bin/env python3
"""Render route-aware QPS and Sparse-vs-Dense delta plots."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary = Path(args.summary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(summary)
    grouped = defaultdict(dict)
    for record in records:
        grouped[(record["N"], record["ratio"])][record["concurrency"]] = record

    n_values = sorted({record["N"] for record in records})
    ratios = sorted({record["ratio"] for record in records})
    if not n_values or not ratios:
        raise RuntimeError("summary contains no matrix records")

    colors = {
        "dense_bf": "#1565C0",
        "dense_ivf": "#64B5F6",
        "sparse_bf": "#EF6C00",
    }

    def render_qps_by_concurrency(concurrency):
        fig, axes = plt.subplots(3, 2, figsize=(24, 17), constrained_layout=True)
        axes = axes.ravel()
        width = 0.36
        offsets = (-width / 2, width / 2)
        for axis, rows in zip(axes, n_values):
            present_ratios = [ratio for ratio in ratios if (rows, ratio) in grouped]
            x = np.arange(len(present_ratios))
            for offset, mode in zip(offsets, ("dense", "sparse")):
                values = []
                routes = []
                for ratio in present_ratios:
                    record = grouped[(rows, ratio)].get(concurrency)
                    values.append(np.nan if record is None else record[f"{mode}_qps"])
                    routes.append(None if record is None else record[f"{mode}_route"])
                bar_colors = (
                    [colors["dense_ivf"] if route == "IVF" else colors["dense_bf"]
                     for route in routes]
                    if mode == "dense" else colors["sparse_bf"]
                )
                bars = axis.bar(x + offset, values, width, color=bar_colors,
                                edgecolor="#263238", linewidth=0.35)
                if mode == "dense":
                    for bar, route in zip(bars, routes):
                        if route == "IVF":
                            bar.set_hatch("///")
            tick_labels = []
            for ratio in present_ratios:
                record = next(iter(grouped[(rows, ratio)].values()))
                tick_labels.append(f"{record['V']:,}\n({ratio * 100:.2f}%)")
            axis.set_xticks(x, tick_labels, fontsize=8)
            axis.set_title(f"N = {rows:,} · one sealed segment", fontsize=13)
            axis.set_xlabel("V (V/N)")
            axis.set_ylabel("QPS")
            axis.grid(axis="y", alpha=0.22)

        for axis in axes[len(n_values):]:
            axis.axis("off")
        legend_handles = [
            Patch(facecolor=colors["dense_bf"], edgecolor="#263238",
                  label="Dense · BF"),
            Patch(facecolor=colors["dense_ivf"], edgecolor="#263238",
                  hatch="///", label="Dense · IVF"),
            Patch(facecolor=colors["sparse_bf"], edgecolor="#263238",
                  label="Sparse · BF"),
        ]
        fig.legend(legend_handles, [item.get_label() for item in legend_handles],
                   loc="outside lower center", ncol=3, frameon=False, fontsize=11)
        fig.suptitle(
            f"Adaptive Sparse vs Dense - Milvus E2E ratio sweep (C{concurrency})",
            fontsize=18,
        )
        fig.savefig(output / f"sparse-ratio-qps-c{concurrency}-by-n.png", dpi=180)
        plt.close(fig)

    render_qps_by_concurrency(1)
    render_qps_by_concurrency(60)

    fig, axes = plt.subplots(3, 2, figsize=(22, 15), constrained_layout=True)
    axes = axes.ravel()
    for axis, rows in zip(axes, n_values):
        present_ratios = [ratio for ratio in ratios if (rows, ratio) in grouped]
        x = np.arange(len(present_ratios))
        for concurrency, color, marker in ((1, "#EF6C00", "o"),
                                            (60, "#1565C0", "s")):
            values = []
            for ratio in present_ratios:
                record = grouped[(rows, ratio)].get(concurrency)
                values.append(np.nan if record is None else 100 * record["qps_delta"])
            axis.plot(x, values, color=color, marker=marker, linewidth=2,
                      label=f"C{concurrency}")
        axis.axhline(0, color="#263238", linewidth=1)
        axis.axhline(-5, color="#C62828", linewidth=1, linestyle="--")
        axis.set_xticks(x, [f"{ratio * 100:.2f}%" for ratio in present_ratios],
                        fontsize=8, rotation=35, ha="right")
        axis.set_title(f"N = {rows:,}", fontsize=13)
        axis.set_xlabel("V/N")
        axis.set_ylabel("Sparse QPS delta (%)")
        axis.grid(alpha=0.22)
    for axis in axes[len(n_values):]:
        axis.axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2,
               frameon=False, fontsize=11)
    fig.suptitle("Sparse QPS / Dense QPS - 1", fontsize=18)
    fig.savefig(output / "sparse-ratio-qps-delta-by-n.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
