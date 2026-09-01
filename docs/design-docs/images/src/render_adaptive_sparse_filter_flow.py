#!/usr/bin/env python3
"""Render the per-segment Adaptive Sparse pipeline on a strict orthogonal grid."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "adaptive-sparse-filter-flow.png"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
regular = fm.FontProperties(fname=FONT_REGULAR)
bold = fm.FontProperties(fname=FONT_BOLD)

fig, ax = plt.subplots(figsize=(22, 9.2))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 22)
ax.set_ylim(0, 9.2)
ax.axis("off")


def rounded_box(x, y, width, height, face, edge, *, radius=0.16, linewidth=1.9, dashed=False, zorder=2):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=(0, (5, 3)) if dashed else "solid",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def node(x, y, width, height, label, face, edge, text_color="#172033", *, fontsize=13, dashed=False):
    rounded_box(x, y, width, height, face, edge, dashed=dashed)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontproperties=regular,
        linespacing=1.18,
        zorder=5,
    )


def decision(cx, cy, rx, ry, label, *, fontsize=11.5):
    patch = Polygon(
        [(cx - rx, cy), (cx, cy + ry), (cx + rx, cy), (cx, cy - ry)],
        closed=True,
        facecolor="#FFF7ED",
        edgecolor="#F59E0B",
        linewidth=1.9,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        color="#9A3412",
        fontsize=fontsize,
        fontproperties=regular,
        linespacing=1.12,
        zorder=5,
    )


def line(points, *, color="#64748B", linewidth=1.9, dashed=False, zorder=4):
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        linestyle=(0, (5, 3)) if dashed else "solid",
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=zorder,
    )


def arrow(points, *, color="#64748B", linewidth=1.9, dashed=False):
    """Draw an orthogonal polyline and put the arrowhead on its final segment."""
    if len(points) > 2:
        line(points[:-1], color=color, linewidth=linewidth, dashed=dashed)
    start, end = points[-2], points[-1]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=linewidth,
        color=color,
        linestyle=(0, (5, 3)) if dashed else "solid",
        connectionstyle="arc3,rad=0",
        shrinkA=0,
        shrinkB=1,
        zorder=4,
    )
    ax.add_patch(patch)


def junction(x, y, color="#64748B"):
    ax.scatter([x], [y], s=22, color=color, zorder=6)


def edge_label(x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#475569",
        fontproperties=regular,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.95},
        zorder=7,
    )


# Whole pipeline is scoped to one segment.
rounded_box(0.18, 0.45, 21.64, 7.85, "#FFFFFF", "#94A3B8", radius=0.26, linewidth=2.2, zorder=0)
ax.text(11.0, 8.62, "Adaptive Sparse Filter Pipeline", ha="center", va="center", fontsize=24, fontproperties=bold)
ax.text(0.65, 8.05, "Segment k", ha="left", va="center", fontsize=16, color="#334155", fontproperties=bold)

# Inner framework boundaries.
rounded_box(0.50, 0.80, 12.55, 6.80, "#F8FBFF", "#BFDBFE", radius=0.24, linewidth=2.0, zorder=1)
rounded_box(13.35, 0.80, 8.12, 6.80, "#FCFAFF", "#DDD6FE", radius=0.24, linewidth=2.0, zorder=1)
ax.text(6.78, 7.24, "Milvus · Filter Execution", ha="center", fontsize=16.5, color="#1E3A5F", fontproperties=bold)
ax.text(17.41, 7.24, "Cardinal · Route & Search", ha="center", fontsize=16.5, color="#4C1D95", fontproperties=bold)

# Milvus flow, aligned to two horizontal lanes.
node(0.85, 4.30, 1.55, 1.00, "Scalar\nPredicate", "#EFF6FF", "#93C5FD", fontsize=12.5)
decision(3.45, 4.80, 0.82, 0.64, "Native Sparse\nproducer?", fontsize=9.8)
node(4.70, 4.30, 1.75, 1.00, "Try Sparse IDs", "#DBEAFE", "#60A5FA", fontsize=12.5)
decision(7.55, 4.80, 0.74, 0.64, "V ≤ T?")
node(8.70, 4.30, 1.60, 1.00, "Sparse IDs", "#DDF7EE", "#34B996", "#075E54", fontsize=12.8)
node(8.70, 2.55, 1.60, 1.00, "Dense Bitmap\nBuild", "#F1F5F9", "#94A3B8", fontsize=12.2)
node(
    11.10,
    3.53,
    1.55,
    1.15,
    "Visibility Merge\nFilter Payload",
    "#EDE9FE",
    "#A78BFA",
    "#4C1D95",
    fontsize=11.6,
)

# Main and decision branches: every segment is horizontal or vertical.
arrow([(2.40, 4.80), (2.63, 4.80)])
arrow([(4.27, 4.80), (4.70, 4.80)])
edge_label(4.48, 5.04, "Yes")
arrow([(6.45, 4.80), (6.81, 4.80)])
arrow([(8.29, 4.80), (8.70, 4.80)])
edge_label(8.49, 5.04, "Yes")

# Both No branches meet once, then enter the single Dense build node.
line([(3.45, 4.16), (3.45, 3.05), (7.55, 3.05)])
line([(7.55, 4.16), (7.55, 3.05)])
junction(7.55, 3.05)
arrow([(7.55, 3.05), (8.70, 3.05)])
edge_label(3.70, 3.72, "No")
edge_label(7.78, 3.72, "No")

# Sparse and Dense representations meet at one junction before visibility.
line([(10.30, 4.80), (10.75, 4.80), (10.75, 4.10)])
line([(10.30, 3.05), (10.75, 3.05), (10.75, 4.10)])
junction(10.75, 4.10, color="#7C3AED")
arrow([(10.75, 4.10), (11.10, 4.10)], color="#7C3AED", linewidth=2.2)

# Compact native-producer support table.
table_x, table_y, table_w, table_h = 0.85, 1.10, 3.90, 1.55
rounded_box(table_x, table_y, table_w, table_h, "#FFFFFF", "#CBD5E1", radius=0.09, linewidth=1.25)
ax.text(
    table_x + 0.16,
    table_y + table_h - 0.27,
    "Native Sparse producer support",
    ha="left",
    va="center",
    fontsize=9.7,
    color="#334155",
    fontproperties=bold,
)
header_rule = table_y + 1.02
divider_x = table_x + 1.32
ax.plot([table_x, table_x + table_w], [header_rule, header_rule], color="#E2E8F0", linewidth=1.0, zorder=3)
ax.plot([divider_x, divider_x], [table_y, header_rule], color="#E2E8F0", linewidth=1.0, zorder=3)
rows = [
    ("BitmapIndex", "==, …"),
    ("STL_SORT", "==, <, >, range, …"),
    ("Raw INT64", "<, >, range, …"),
]
for idx, (producer, operations) in enumerate(rows):
    row_y = table_y + 0.84 - idx * 0.30
    ax.text(table_x + 0.12, row_y, producer, ha="left", va="center", fontsize=8.9, color="#334155", fontproperties=regular)
    ax.text(divider_x + 0.12, row_y, operations, ha="left", va="center", fontsize=8.9, color="#475569", fontproperties=regular)

# Cardinal consumers use the same orthogonal grid.
node(13.75, 3.55, 1.70, 1.10, "Payload Router", "#EDE9FE", "#8B5CF6", "#4C1D95", fontsize=12.2)
node(16.10, 5.15, 1.95, 0.90, "BF\nDirect enumeration", "#ECFDF5", "#34D399", "#065F46", fontsize=11.6)
node(16.10, 3.65, 1.95, 0.90, "IVF\nDense auto path", "#EFF6FF", "#60A5FA", "#1E40AF", fontsize=11.3)
node(16.10, 2.15, 1.95, 0.90, "Graph\nDense auto path", "#FFF7ED", "#F59E0B", "#92400E", fontsize=11.3)
node(19.35, 3.55, 1.50, 1.10, "TopK\nMerge", "#F8FAFC", "#94A3B8", fontsize=12.5)

arrow([(12.65, 4.10), (13.75, 4.10)], color="#7C3AED", linewidth=2.2)

# Selector to BF / IVF / Graph.
arrow([(14.60, 4.65), (14.60, 5.60), (16.10, 5.60)])
edge_label(15.18, 5.82, "Sparse direct / Dense auto")
arrow([(15.45, 4.10), (16.10, 4.10)])
edge_label(15.77, 4.34, "Dense auto")
arrow([(14.60, 3.55), (14.60, 2.60), (16.10, 2.60)], color="#D97706")
edge_label(15.18, 2.82, "Dense auto")

# Three consumers join once, then enter TopK horizontally.
line([(18.05, 5.60), (18.80, 5.60), (18.80, 4.10)])
line([(18.05, 4.10), (18.80, 4.10)])
line([(18.05, 2.60), (18.80, 2.60), (18.80, 4.10)], color="#D97706")
junction(18.80, 4.10)
arrow([(18.80, 4.10), (19.35, 4.10)])

fig.savefig(OUTPUT, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.10)
print(OUTPUT)
