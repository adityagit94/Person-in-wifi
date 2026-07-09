"""Wi-Pose dataset loader.

Each .mat is MATLAB v7.3 (HDF5), so it is read with h5py, not
scipy.io.loadmat. Two variables per file:
  CSI              h5py shape (3, 3, 30, 5); this is MATLAB 5x30x3x3 with the
                   axes reversed. Already amplitude (real, non-negative).
  SkeletonPoints   shape (1, 54) = [x*18, y*18, conf*18], 18 COCO-18 joints,
                   pixel coordinates plus AlphaPose confidences.

The loader turns each file into network-ready tensors: a 150x3x3 CSI input and
the JHM / PAF targets with their loss masks.
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


def csi_to_input(csi):
    """h5py (3,3,30,5) -> normalized (150, 3, 3) float32 network input.

    Undo the HDF5 axis reversal to MATLAB order (5, 30, 3, 3), then flatten the
    5 time samples x 30 subcarriers onto 150 channels, keeping the 3x3 antenna
    grid. Normalize per sample (zero mean, unit std).
    """
    m = np.transpose(csi, (3, 2, 1, 0))          # (5, 30, 3, 3)
    x = m.reshape(150, 3, 3).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x


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
        px, py, conf = parse_skeleton(sp)
        valid = joint_valid(conf)
        gx, gy = scale_keypoints(px, py)
        return {
            "csi": torch.from_numpy(csi_to_input(csi)),
            "jhm": torch.from_numpy(render_jhm(gx, gy, valid)),
            "paf": torch.from_numpy(render_paf(gx, gy, valid)),
            "jhm_mask": torch.from_numpy(jhm_weight_mask(valid)),
            "paf_mask": torch.from_numpy(paf_weight_mask(valid)),
            # (18, 3) grid-space keypoints for visualization / evaluation
            "keypoints_grid": torch.from_numpy(
                np.stack([gx, gy, conf.astype(np.float32)], axis=-1)),
        }
