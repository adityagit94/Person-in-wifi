"""Generates the README figures.

The learning-curve numbers are transcribed from the milestone-1 probe logs
(peak-at-true-joint values printed every 250-300 steps); the final metrics are
from the committed 8000-step run of mw_vs_l2_toy.py. Nothing here retrains.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax):
    ax.grid(axis="x", visible=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASE)


# ---------------------------------------------------------------- fig 1: weight
fig, ax = plt.subplots(figsize=(7.2, 3.6))
y_pos = np.linspace(0, 1, 200)
y_neg = np.linspace(-1, -0.001, 200)
ax.plot(y_pos, np.abs(y_pos) + 1.0, color=BLUE, lw=2)
ax.text(0.02, 1.85, "heatmap weight\nw = |y| + 1", color=BLUE,
        fontsize=9.5, va="top")
yy_full = np.linspace(-1, 1, 400)
ax.plot(yy_full, np.abs(yy_full) + 0.3, color=AQUA, lw=2)
ax.text(-0.98, 1.62, "limb-map weight\nw = |y| + 0.3", color=AQUA,
        fontsize=9.5, va="top")
ax.plot(y_pos, y_pos + 0.3, color=RED, lw=2, ls=(0, (4, 3)))
ax.plot(y_neg, y_neg - 0.3, color=RED, lw=2, ls=(0, (4, 3)))
ax.text(-0.52, -0.92, "the paper's formula, read literally:\nw = y − 0.3  for negative targets",
        color=RED, fontsize=9.5, va="top")
ax.text(1.0, 0.52, "(dashed = same as solid for y ≥ 0)", color=INK2,
        fontsize=8.5, ha="right")
ax.axhspan(-1.4, 0, facecolor=RED, alpha=0.06)
ax.text(-0.98, -0.35, "below zero the loss REWARDS error", color=RED,
        fontsize=9.5, style="italic", va="top")
ax.axhline(0, color=BASE, lw=1)
ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.45, 2.15)
ax.set_xlabel("target value y at a pixel")
ax.set_ylabel("loss weight w")
ax.set_title("The Matthew Weight, and why the paper's formula can't be taken literally",
             fontsize=11, color=INK, pad=12)
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mw_weight.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- fig 2: sparsity
H, W, J = 46, 82, 18
rng = np.random.default_rng(3)
cx = rng.uniform(4, W - 4, J)
cy = rng.uniform(4, H - 4, J)
ys, xs = np.mgrid[0:H, 0:W]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
for ax, sigma in zip(axes, (1.5, 3.0)):
    d2 = (xs[None] - cx[:, None, None]) ** 2 + (ys[None] - cy[:, None, None]) ** 2
    t = np.exp(-d2 / (2 * sigma ** 2))
    fg = (t > 0.1).mean()
    ax.imshow(t.max(0), cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"σ = {sigma}   →   {fg:.1%} of pixels are foreground",
                 fontsize=10.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(GRID)
fig.suptitle("One training target (18 joints, all channels overlaid) at the two blob sizes",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sparsity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 3: learning curves
# measured peak-at-true-joint values from the investigation probes
runs_stuck = [  # sigma 1.5 -- every variant flat near zero
    ([250, 500, 750, 1000, 1250, 1500, 1750],
     [.005, .005, .006, .006, .007, .007, .007]),          # original decoder
    ([250, 500, 750, 1000, 1250, 1500],
     [.004, .004, .004, .005, .005, .005]),                # broadcast decoder
    ([250, 500, 750, 1000, 1250, 1500],
     [.004, .004, .004, .005, .005, .006]),                # fixed dataset
    ([300, 600, 900, 1200], [.004, .005, .005, .005]),     # sin/cos encoding
]
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for i, (s, p) in enumerate(runs_stuck):
    ax.plot(s, p, color=MUTED, lw=1.4, alpha=0.75,
            label="σ = 1.5 (four variants)" if i == 0 else None)
ax.plot([300, 600, 900, 1200], [.027, .032, .037, .044], color=BLUE, lw=2,
        label="σ = 3.0, original decoder")
ax.plot([250, 500, 750, 1000, 1250, 1500], [.016, .020, .022, .025, .027, .030],
        color=AQUA, lw=2, label="σ = 3.0, broadcast decoder")
ax.text(1770, 0.0075, "σ = 1.5: stuck,\nno matter what", color=MUTED,
        fontsize=9, va="center")
ax.text(1225, 0.0435, "learning", color=BLUE, fontsize=9.5, va="center")
ax.set_xlim(0, 2350); ax.set_ylim(0, 0.052)
ax.set_xlabel("training step")
ax.set_ylabel("predicted peak at true joints")
ax.set_title("Blob size decided everything (a perfect model scores about 0.96)",
             fontsize=11, color=INK, pad=12)
ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 4: final metrics
panels = [
    ("Foreground error\n(lower is better)", [1.65e-1, 1.59e-1], "{:.3f}"),
    ("Background error\n(lower is better)", [5.93e-4, 1.13e-3], "{:.4f}"),
    ("Peak at true joints\n(higher is better)", [0.104, 0.111], "{:.3f}"),
]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
for ax, (title, vals, fmt) in zip(axes, panels):
    bars = ax.bar([0, 1], vals, width=0.55, color=[MUTED, BLUE])
    for x, v in zip([0, 1], vals):
        ax.text(x, v * 1.02, fmt.format(v), ha="center", va="bottom",
                fontsize=9.5, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["plain L2", "Matthew\nWeight"], fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, color=INK2)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
fig.suptitle("After 8000 identical training steps (64 unseen samples)",
             fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "final_metrics.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------- fig 5: architecture diagram
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)


def _box(cx, cy, w, h, title, shape, fill, edge):
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.02",
        linewidth=1.5, edgecolor=edge, facecolor=fill))
    ax.text(cx, cy + 0.035, title, ha="center", va="center", fontsize=9.3,
            color=INK)
    ax.text(cx, cy - 0.05, shape, ha="center", va="center", fontsize=8.8,
            color=INK2, family="monospace")


def _arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.4, color=MUTED))


blue_fill = (0.165, 0.471, 0.839, 0.10)
aqua_fill = (0.106, 0.686, 0.478, 0.12)
gray_fill = (0.537, 0.529, 0.506, 0.12)

xs = [0.09, 0.30, 0.51, 0.72]
w, h = 0.16, 0.24
_box(xs[0], 0.5, w, h, "input CSI", "150 x 3 x 3", gray_fill, MUTED)
_box(xs[1], 0.5, w, h, "bilinear upsample", "150 x 96 x 96", blue_fill, BLUE)
_box(xs[2], 0.5, w, h, "residual block", "150 x 96 x 96", blue_fill, BLUE)
_box(xs[3], 0.5, w, h, "U-Net trunk", "64 x 96 x 96", blue_fill, BLUE)
for a, b in zip(xs[:-1], xs[1:]):
    _arrow(a + w / 2, 0.5, b - w / 2, 0.5)

_box(0.92, 0.75, 0.145, 0.2, "JHM head", "19 x 46 x 82", aqua_fill, AQUA)
_box(0.92, 0.25, 0.145, 0.2, "PAF head", "38 x 46 x 82", aqua_fill, AQUA)
_arrow(xs[3] + w / 2, 0.54, 0.92 - 0.0725, 0.72)
_arrow(xs[3] + w / 2, 0.46, 0.92 - 0.0725, 0.28)

ax.text(xs[3], 0.24, "encoder / decoder\n96 > 48 > 24 > 12 > 24 > 48 > 96",
        ha="center", va="center", fontsize=7.8, color=MUTED)
ax.text(0.985, 0.75, "18 joints + background", ha="right", va="center",
        fontsize=7.8, color=INK2)
ax.text(0.985, 0.25, "19 limbs x (x, y)", ha="right", va="center",
        fontsize=7.8, color=INK2)
ax.set_title("Network data flow (paper Figure 6), pose heads only",
             fontsize=11, color=INK, pad=6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "architecture.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- fig 6: parameter breakdown
import sys
sys.path.insert(0, os.path.dirname(OUT))   # repo root, to import piw
from piw.network import PersonInWiFi, count_parameters

_m = PersonInWiFi()
_total = count_parameters(_m)
comps = [
    ("U-Net trunk", count_parameters(_m.unet), BLUE),
    ("PAF head", count_parameters(_m.paf_head), AQUA),
    ("residual block", count_parameters(_m.res_block), MUTED),
    ("JHM head", count_parameters(_m.jhm_head), MUTED),
]
comps.sort(key=lambda c: c[1])   # ascending, so largest ends on top

fig, ax = plt.subplots(figsize=(7.6, 2.9))
names = [c[0] for c in comps]
vals = [c[1] / 1e6 for c in comps]
colors = [c[2] for c in comps]
ax.barh(names, vals, color=colors, height=0.62)
for i, (n, v, _) in enumerate(comps):
    ax.text(v / 1e6 + 0.12, i, f"{v/1e6:.2f}M  ({v/_total:.0%})",
            va="center", ha="left", fontsize=9, color=INK)
ax.set_xlim(0, max(vals) * 1.28)
ax.set_xlabel("trainable parameters (millions)")
ax.set_title(f"Where the {_total/1e6:.2f}M parameters live",
             fontsize=11, color=INK, pad=10)
ax.grid(axis="y", visible=False)
ax.spines["left"].set_color(BASE)
ax.spines["bottom"].set_color(BASE)
ax.tick_params(labelcolor=INK2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "param_breakdown.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

print("wrote 6 figures to", OUT)
