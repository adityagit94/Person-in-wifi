"""Render training targets from Wi-Pose keypoints (JHM and PAF).

The dataset gives 18 keypoints per frame as pixel coordinates plus a confidence.
This module turns them into the paper's supervision maps at 46 x 82:

  JHM  (joint heat maps): one Gaussian channel per joint plus a background
       channel (background = 1 - max over joints, the OpenPose convention).
  PAF  (part affinity fields): two channels per limb holding the (x, y) unit
       vector of the limb, painted in a thin band along the limb segment.

Low-confidence joints (conf < 0.2) come from a teacher network and are
unreliable, so they are excluded: their JHM channel is left empty and their
weight mask is 0, and any limb touching them is dropped from the PAF. The
weight masks are what the loss multiplies in to ignore those channels.

One deliberate exception: the background channel is computed from ALL joints,
including low-confidence ones. A low-confidence joint is usually somewhere
near its labeled spot, so background must not confidently claim "empty" right
where a joint probably is. Rendering the uncertain joint into the background
subtraction (while keeping its own channel masked) softens one channel instead
of biasing it.
"""

import numpy as np

from piw.constants import OUT_H, OUT_W
from piw.skeleton import LIMBS, NUM_JOINTS

SIGMA = 1.5                  # joint Gaussian std, in grid pixels
PAF_WIDTH = 1.0              # half-thickness of the painted limb band, in px
CONF_THRESH = 0.2            # joints below this are masked out of the loss

# Source camera resolution. Inferred from the sample: the vertical (head-to-feet)
# coordinate reaches ~608 and the horizontal ~464, so the frame is portrait
# 480 x 640 (width x height). Confirmed by the visual check: this orientation
# produces upright skeletons. Recompute over the full dataset before the final
# run if you want to be exact; the value only sets the coordinate scale.
SRC_W, SRC_H = 480, 640


def scale_keypoints(x, y, src_w=SRC_W, src_h=SRC_H, out_h=OUT_H, out_w=OUT_W):
    """Map pixel coordinates onto the 46 x 82 grid (nonuniform, fills the grid)."""
    gx = np.asarray(x, np.float32) * (out_w / src_w)
    gy = np.asarray(y, np.float32) * (out_h / src_h)
    return gx, gy


def joint_valid(conf, thresh=CONF_THRESH):
    return np.asarray(conf) >= thresh


def render_jhm(gx, gy, valid, out_h=OUT_H, out_w=OUT_W, sigma=SIGMA):
    """(NUM_JOINTS + 1, H, W) float32: a Gaussian per valid joint + background.

    Invalid joints get an empty channel (their mask excludes it from the loss
    anyway), but they still participate in the background subtraction; see the
    module docstring for why.
    """
    hm = np.zeros((NUM_JOINTS + 1, out_h, out_w), np.float32)
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    bg = np.zeros((out_h, out_w), np.float32)
    for j in range(NUM_JOINTS):
        if not (np.isfinite(gx[j]) and np.isfinite(gy[j])):
            continue
        d2 = (xx - gx[j]) ** 2 + (yy - gy[j]) ** 2
        g = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
        if valid[j]:
            hm[j] = g
        bg = np.maximum(bg, g)
    hm[NUM_JOINTS] = 1.0 - bg
    return hm


def render_paf(gx, gy, valid, out_h=OUT_H, out_w=OUT_W, width=PAF_WIDTH):
    """(2 * NUM_LIMBS, H, W) float32: (x, y) unit vector along each valid limb."""
    paf = np.zeros((2 * len(LIMBS), out_h, out_w), np.float32)
    yy, xx = np.mgrid[0:out_h, 0:out_w]
    for li, (a, b) in enumerate(LIMBS):
        if not (valid[a] and valid[b]):
            continue
        ax, ay, bx, by = gx[a], gy[a], gx[b], gy[b]
        vx, vy = bx - ax, by - ay
        norm = float(np.hypot(vx, vy))
        if norm < 1e-6:
            continue
        ux, uy = vx / norm, vy / norm
        dx, dy = xx - ax, yy - ay
        along = dx * ux + dy * uy               # projection onto the limb
        perp = np.abs(dx * uy - dy * ux)        # perpendicular distance
        on = (along >= 0) & (along <= norm) & (perp <= width)
        paf[2 * li][on] = ux
        paf[2 * li + 1][on] = uy
    return paf


def jhm_weight_mask(valid):
    """(NUM_JOINTS + 1,) per-channel loss mask; background is always kept."""
    m = np.ones(NUM_JOINTS + 1, np.float32)
    m[:NUM_JOINTS] = np.asarray(valid, np.float32)
    return m


def paf_weight_mask(valid):
    """(2 * NUM_LIMBS,) per-channel loss mask; a limb needs both endpoints."""
    m = np.zeros(2 * len(LIMBS), np.float32)
    for li, (a, b) in enumerate(LIMBS):
        v = 1.0 if (valid[a] and valid[b]) else 0.0
        m[2 * li] = v
        m[2 * li + 1] = v
    return m
