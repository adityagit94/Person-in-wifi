"""Stage 3 visual sanity check: render JHM and PAF targets for a few real
Wi-Pose frames and overlay the ground-truth skeleton, to confirm the labels
line up. Saves figs/stage3_targets.png and prints a small table.

Run: python stage3_data_check.py
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from piw.dataset import load_mat, parse_skeleton
from piw.skeleton import LIMBS, NUM_JOINTS
from piw.targets import (OUT_H, OUT_W, joint_valid, render_jhm, render_paf,
                         scale_keypoints)

SAMPLE_GLOB = os.path.join("data", "Wi-Pose_sample", "**", "Train", "*.mat")


def pick_one_per_action(n=6):
    """One file each for the first n actions (alphabetical), for variety."""
    files = sorted(glob.glob(SAMPLE_GLOB, recursive=True))
    chosen = {}
    for f in files:
        m = re.match(r"([^/\\]+?)_\d+-frame", os.path.basename(f))
        act = m.group(1) if m else "?"
        if act not in chosen:
            chosen[act] = f
        if len(chosen) >= n:
            break
    return list(chosen.items())


def overlay_skeleton(ax, gx, gy, valid):
    for a, b in LIMBS:
        if valid[a] and valid[b]:
            ax.plot([gx[a], gx[b]], [gy[a], gy[b]], color="#39d3ff", lw=1.3,
                    alpha=0.9)
    ax.scatter(gx[valid], gy[valid], s=14, color="#39ff9e",
               edgecolors="black", linewidths=0.4, zorder=3)
    inv = ~valid
    ax.scatter(gx[inv], gy[inv], s=14, color="#ff5a5a", edgecolors="black",
               linewidths=0.4, zorder=3)


def main():
    picks = pick_one_per_action(6)
    fig, axes = plt.subplots(len(picks), 2, figsize=(8.5, 2.05 * len(picks)))
    print(f"{'action':>10} {'valid joints':>13} {'jhm':>14} {'paf':>14}")
    for row, (act, path) in enumerate(picks):
        csi, sp = load_mat(path)
        px, py, conf = parse_skeleton(sp)
        valid = joint_valid(conf)
        gx, gy = scale_keypoints(px, py)
        jhm = render_jhm(gx, gy, valid)
        paf = render_paf(gx, gy, valid)
        paf_mag = np.hypot(paf[0::2], paf[1::2]).max(axis=0)

        ax = axes[row, 0]
        ax.imshow(jhm[:NUM_JOINTS].max(0), cmap="magma", vmin=0, vmax=1)
        overlay_skeleton(ax, gx, gy, valid)
        ax.set_ylabel(act, fontsize=10)
        if row == 0:
            ax.set_title("JHM (joints) + skeleton", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row, 1]
        ax.imshow(paf_mag, cmap="viridis", vmin=0, vmax=1)
        overlay_skeleton(ax, gx, gy, valid)
        if row == 0:
            ax.set_title("PAF (limbs) + skeleton", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        print(f"{act:>10} {int(valid.sum()):>10}/{NUM_JOINTS} "
              f"{str(tuple(jhm.shape)):>16} {str(tuple(paf.shape)):>14}")

    fig.suptitle("Wi-Pose targets rendered at 46 x 82, overlaid on the "
                 "ground-truth skeleton\n(green = confident joint, "
                 "red = low-confidence, masked out of the loss)", fontsize=10)
    fig.tight_layout()
    out = os.path.join("figs", "stage3_targets.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
