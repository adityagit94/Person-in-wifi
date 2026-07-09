"""Evaluation: decode joints from heatmaps and score PCK@0.2.

PCK@0.2: a predicted joint counts as correct if it lands within 0.2 of the
ground-truth bounding-box diagonal of the true joint. The box is computed from
the confident GT joints, matching the paper's convention. Only joints the
teacher was confident about (conf >= 0.2) are scored.

COCO-18 has no separate foot keypoints, so joints are grouped head / arms
(torso and arms) / legs, with ankles as the lowest joints in the leg group.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from piw.dataset import WiPoseDataset
from piw.network import PersonInWiFi
from piw.skeleton import JOINT_NAMES, NUM_JOINTS
from piw.targets import CONF_THRESH

JOINT_GROUPS = {
    "head": [0, 14, 15, 16, 17],        # nose, eyes, ears
    "arms": [1, 2, 3, 4, 5, 6, 7],      # neck, shoulders, elbows, wrists
    "legs": [8, 9, 10, 11, 12, 13],     # hips, knees, ankles
}


@torch.no_grad()
def decode_keypoints(jhm):
    """(B, J+1, H, W) -> (B, J, 2) grid coordinates (x=col, y=row) via argmax."""
    hm = jhm[:, :NUM_JOINTS]                       # drop background channel
    B, J, H, W = hm.shape
    idx = hm.reshape(B, J, -1).argmax(-1)
    xs = (idx % W).float()
    ys = (idx // W).float()
    return torch.stack([xs, ys], dim=-1)


@torch.no_grad()
def pck_correct(pred_xy, gt, thresh=0.2):
    """Per-joint correct/valid boolean masks, shape (B, J)."""
    gx, gy, conf = gt[..., 0], gt[..., 1], gt[..., 2]
    valid = conf >= CONF_THRESH
    dist = torch.hypot(pred_xy[..., 0] - gx, pred_xy[..., 1] - gy)
    correct = torch.zeros_like(valid)
    for b in range(gt.shape[0]):
        v = valid[b]
        if int(v.sum()) < 2:
            continue
        xs, ys = gx[b][v], gy[b][v]
        diag = torch.hypot(xs.max() - xs.min(), ys.max() - ys.min()).clamp_min(1e-6)
        correct[b] = (dist[b] <= thresh * diag) & v
    return correct, valid


@torch.no_grad()
def evaluate(model, root, split="Test", device="cpu", batch=32, workers=0,
             max_batches=None):
    dl = DataLoader(WiPoseDataset(root, split), batch_size=batch,
                    num_workers=workers)
    model.eval().to(device)
    hit = np.zeros(NUM_JOINTS)
    tot = np.zeros(NUM_JOINTS)
    for i, bd in enumerate(dl):
        jhm_p, _ = model(bd["csi"].to(device))
        pred = decode_keypoints(jhm_p).cpu()
        correct, valid = pck_correct(pred, bd["keypoints_grid"])
        hit += correct.sum(0).numpy()
        tot += valid.sum(0).numpy()
        if max_batches and i + 1 >= max_batches:
            break
    per_joint = hit / np.clip(tot, 1, None)
    overall = hit.sum() / max(tot.sum(), 1)
    groups = {g: hit[ix].sum() / max(tot[ix].sum(), 1)
              for g, ix in JOINT_GROUPS.items()}
    return {"overall": float(overall), "groups": groups,
            "per_joint": dict(zip(JOINT_NAMES, per_joint.tolist()))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--root",
                    default=os.path.join("data", "Wi-Pose_sample", "Wi-Pose"))
    ap.add_argument("--split", default="Test")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model = PersonInWiFi()
    model.load_state_dict(torch.load(a.checkpoint, map_location=a.device))
    res = evaluate(model, a.root, a.split, a.device)
    print(f"PCK@0.2 overall: {res['overall']:.3f}")
    for g, v in res["groups"].items():
        print(f"  {g:>5}: {v:.3f}")
