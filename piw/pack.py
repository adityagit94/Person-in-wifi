"""Pack extracted Wi-Pose .mat files into contiguous arrays.

Why: the dataset is 166,600 tiny HDF5 files, and opening each one costs more
than reading it. One training epoch means 166,600 file opens, which Colab
filesystems make brutal. Packing pays that cost exactly once: afterwards a
sample is an array slice, and the whole training split (~0.9 GB float32) fits
in RAM, so epochs after the first touch the disk zero times.

Usage:
    python -m piw.pack --src <dir containing Train/ and Test/> --dst <out dir>

Writes per split:
    <dst>/<split>/csi.npy    (N, 150, 3, 3) float32 amplitude, unnormalized
    <dst>/<split>/skel.npy   (N, 54) float32 raw SkeletonPoints
    <dst>/<split>/files.txt  source filename per row, for traceability

Also prints the keypoint coordinate maxima per split, as a check on the
480 x 640 source-frame estimate in piw/targets.py.
"""

import argparse
import glob
import os

import numpy as np

from piw.dataset import load_mat, reshape_csi


def pack_split(src, dst, split):
    files = sorted(glob.glob(os.path.join(src, split, "*.mat")))
    if not files:
        raise FileNotFoundError(f"no .mat files under {os.path.join(src, split)}")
    n = len(files)
    csi = np.empty((n, 150, 3, 3), np.float32)
    skel = np.empty((n, 54), np.float32)
    for i, f in enumerate(files):
        c, s = load_mat(f)
        csi[i] = reshape_csi(c)
        skel[i] = s.astype(np.float32)
        if (i + 1) % 10000 == 0:
            print(f"  [{split}] {i + 1}/{n}", flush=True)
    out = os.path.join(dst, split)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "csi.npy"), csi)
    np.save(os.path.join(out, "skel.npy"), skel)
    with open(os.path.join(out, "files.txt"), "w") as fh:
        fh.write("\n".join(os.path.basename(f) for f in files))
    vert, horiz = skel[:, 0:18], skel[:, 18:36]
    print(f"[{split}] packed {n} samples  "
          f"vertical max {vert.max():.1f}  horizontal max {horiz.max():.1f}  "
          f"(source-frame estimate is 480 x 640)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="directory containing Train/ and Test/")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--splits", nargs="+", default=["Train", "Test"])
    a = ap.parse_args()
    for split in a.splits:
        pack_split(a.src, a.dst, split)


if __name__ == "__main__":
    main()
