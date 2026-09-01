#!/usr/bin/env python3
"""Render the unified Cohere 1M Dense/Adaptive Sparse Milvus E2E matrix."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "adaptive-sparse-filter-e2e-summary.png"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
regular = fm.FontProperties(fname=FONT_REGULAR)
bold = fm.FontProperties(fname=FONT_BOLD)

scenario_labels = ["V=64 · BF", "V=500 · IVF", "V=1,000 · IVF"]

results = {
    "C1": {
        "mean": ([3.857085, 5.358856, 5.097590], [3.227364, 3.484063, 3.420859]),
        "p90": ([4.159830, 5.565898, 5.299833], [3.565625, 3.826892, 3.757836]),
        "qps": ([257.763, 185.781, 195.282], [307.763, 285.134, 290.457]),
    },
    "C60": {
        "mean": ([41.558556, 47.748027, 46.797283], [36.886191, 36.516532, 36.692302]),
        "p90": ([68.575136, 75.287675, 75.482532], [60.775636, 60.177927, 60.443620]),
        "qps": ([1379.896, 1205.577, 1224.762], [1555.862, 1568.832, 1558.097]),
    },
}

columns = [
    ("mean", "Mean latency (ms)", True),
    ("p90", "P90 latency (ms)", True),
    ("qps", "Throughput (QPS)", False),
]

plt.rcParams.update(
    {
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#334155",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "xtick.color": "#475569",
        "ytick.color": "#334155",
    }
)

fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.8), gridspec_kw={"hspace": 0.43, "wspace": 0.13})
fig.subplots_adjust(left=0.16, right=0.975, top=0.80, bottom=0.115)

y = np.arange(len(scenario_labels))
bar_height = 0.31
legend_bars = None

for row, concurrency in enumerate(("C1", "C60")):
    for column, (metric, title, is_latency) in enumerate(columns):
        axis = axes[row, column]
        dense_values = np.asarray(results[concurrency][metric][0], dtype=float)
        sparse_values = np.asarray(results[concurrency][metric][1], dtype=float)
        deltas = (sparse_values / dense_values - 1.0) * 100.0
        x_max = max(dense_values.max(), sparse_values.max()) * 1.34

        dense_bars = axis.barh(
            y - bar_height / 2,
            dense_values,
            height=bar_height,
            color="#2563EB",
            label="Dense",
        )
        sparse_bars = axis.barh(
            y + bar_height / 2,
            sparse_values,
            height=bar_height,
            color="#F97316",
            label="Adaptive Sparse",
        )
        if legend_bars is None:
            legend_bars = (dense_bars, sparse_bars)

        axis.set_xlim(0, x_max)
        axis.grid(axis="x", color="#E2E8F0", linewidth=0.9)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.tick_params(axis="x", labelsize=9.5)
        axis.set_xlabel(title, fontproperties=regular, fontsize=11, labelpad=7)
        axis.invert_yaxis()
        if column == 0:
            axis.set_yticks(y, scenario_labels, fontproperties=regular, fontsize=10.5)
        else:
            axis.set_yticks(y, [""] * len(y))

        for bar, value in zip(dense_bars, dense_values):
            value_label = f"{value:.3f}" if is_latency else f"{value:,.1f}"
            axis.text(
                value + x_max * 0.012,
                bar.get_y() + bar.get_height() / 2,
                value_label,
                va="center",
                ha="left",
                fontsize=8.7,
                color="#1D4ED8",
                fontproperties=regular,
            )
        for bar, value, delta in zip(sparse_bars, sparse_values, deltas):
            value_label = f"{value:.3f}" if is_latency else f"{value:,.1f}"
            axis.text(
                value + x_max * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value_label}  ({delta:+.2f}%)",
                va="center",
                ha="left",
                fontsize=8.7,
                color="#C2410C",
                fontproperties=regular,
            )

    axes[row, 0].text(
        -0.34,
        1.13,
        concurrency,
        transform=axes[row, 0].transAxes,
        fontsize=15,
        color="#334155",
        fontproperties=bold,
        ha="left",
        va="center",
    )

fig.legend(
    [legend_bars[0], legend_bars[1]],
    ["Dense", "Adaptive Sparse"],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.865),
    ncol=2,
    frameon=False,
    prop=regular,
)
fig.suptitle(
    "Adaptive Sparse vs Dense - Milvus E2E",
    x=0.5,
    y=0.965,
    fontsize=21,
    fontproperties=bold,
    color="#172033",
)
fig.text(
    0.5,
    0.915,
    "Cohere 1M × 768D • 1 sealed segment • Cardinal Autoindex",
    ha="center",
    fontsize=11,
    color="#475569",
    fontproperties=regular,
)
fig.text(
    0.5,
    0.040,
    "NQ=1 · topK=10 · V is the final accepted cardinality per segment",
    ha="center",
    va="bottom",
    fontsize=10.5,
    color="#64748B",
    fontproperties=regular,
)

fig.savefig(OUTPUT, dpi=180, facecolor="white", bbox_inches="tight")
print(OUTPUT)
