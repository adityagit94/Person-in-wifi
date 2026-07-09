"""Wi-Pose dataset loaders.

Two ways to read the data, producing identical items:

  WiPoseDataset   reads the original .mat files (MATLAB v7.3 / HDF5, so h5py,
                  not scipy.io.loadmat). Fine locally and for small samples.
  PackedWiPose    reads the contiguous arrays written by ``python -m piw.pack``.
                  This is what real training uses: one file open per split
                  instead of one per sample per epoch, and the whole training
                  split (~0.9 GB float32) fits in RAM.

Each .mat holds two variables:
  CSI              h5py shape (3, 3, 30, 5); this is MATLAB 5x30x3x3 with the
                   axes reversed. Already amplitude (real, non-negative).
  SkeletonPoints   shape (1, 54) = [vertical*18, horizontal*18, conf*18], 18
                   COCO-18 joints, pixel coordinates plus AlphaPose confidences.

An item is a normalized 150x3x3 CSI tensor plus the JHM / PAF targets and
their loss masks.
"""

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from piw.targets import (jhm_weight_mask, joint_valid, paf_weight_mask,
                         render_jhm, render_paf, scale_keypoints)


def load_mat(path):
    """Return (CSI, SkeletonPoints) as numpy arrays straight from the file."""
    with h5py.File(path, "r") as h:
        csi = np.asarray(h["CSI"][()], dtype=np.float64)
        sp = np.asarray(h["SkeletonPoints"][()], dtype=np.float64).ravel()
    return csi, sp


def reshape_csi(csi):
    """h5py (3,3,30,5) -> (150, 3, 3) float32, unnormalized.

    Undo the HDF5 axis reversal back to MATLAB order (5, 30, 3, 3), then
    flatten the 5 time samples x 30 subcarriers onto 150 channels, keeping the
    3x3 antenna grid.
    """
    m = np.transpose(csi, (3, 2, 1, 0))          # (5, 30, 3, 3)
    return m.reshape(150, 3, 3).astype(np.float32)


def normalize_csi(x):
    """Per-sample normalization: zero mean, unit std."""
    return (x - x.mean()) / (x.std() + 1e-6)


def csi_to_input(csi):
    """h5py CSI -> normalized (150, 3, 3) float32 network input."""
    return normalize_csi(reshape_csi(csi))


def parse_skeleton(sp):
    """(54,) -> (x[18], y[18], conf[18]) in image pixels.

    SkeletonPoints is laid out [block1*18, block2*18, conf*18]. Block1 is the
    vertical (head-to-feet) axis and block2 is horizontal, confirmed by the
    visual check: nose sits at a small block1 value and ankles at a large one,
    and only this ordering yields upright skeletons. So x (horizontal) is
    block2 and y (vertical) is block1.
    """
    y = sp[0:18]        # vertical, 0..~640
    x = sp[18:36]       # horizontal, 0..~480
    conf = sp[36:54]
    return x, y, conf


def make_item(x, sp):
    """Normalized CSI (150,3,3) + raw SkeletonPoints (54,) -> training item."""
    px, py, conf = parse_skeleton(sp)
    valid = joint_valid(conf)
    gx, gy = scale_keypoints(px, py)
    return {
        "csi": torch.from_numpy(x),
        "jhm": torch.from_numpy(render_jhm(gx, gy, valid)),
        "paf": torch.from_numpy(render_paf(gx, gy, valid)),
        "jhm_mask": torch.from_numpy(jhm_weight_mask(valid)),
        "paf_mask": torch.from_numpy(paf_weight_mask(valid)),
        # (18, 3) grid-space keypoints for visualization / evaluation
        "keypoints_grid": torch.from_numpy(
            np.stack([gx, gy, np.asarray(conf, np.float32)], axis=-1)),
    }


class WiPoseDataset(Dataset):
    """Loads Wi-Pose frames from ``<root>/<split>/*.mat``."""

    def __init__(self, root, split="Train"):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.mat")))
        if not self.files:
            raise FileNotFoundError(
                f"no .mat files under {os.path.join(root, split)}")
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        csi, sp = load_mat(self.files[i])
        return make_item(csi_to_input(csi), sp)


class PackedWiPose(Dataset):
    """Loads the packed arrays written by ``python -m piw.pack``.

    in_memory=True (default) loads both arrays into RAM; False memory-maps
    them instead (for machines where the split does not fit).
    """

    def __init__(self, root, split="Train", in_memory=True):
        d = os.path.join(root, split)
        mode = None if in_memory else "r"
        self.csi = np.load(os.path.join(d, "csi.npy"), mmap_mode=mode)
        self.skel = np.load(os.path.join(d, "skel.npy"), mmap_mode=mode)
        if len(self.csi) != len(self.skel):
            raise ValueError(f"csi/skel length mismatch under {d}")

    def __len__(self):
        return len(self.csi)

    def __getitem__(self, i):
        x = normalize_csi(np.array(self.csi[i], dtype=np.float32))
        sp = np.asarray(self.skel[i], dtype=np.float64)
        return make_item(x, sp)
