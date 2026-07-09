"""Loader contract tests. These need real Wi-Pose files, so they run only on
machines where the sample is extracted under data/ and skip cleanly elsewhere.
"""

from pathlib import Path

import pytest
import torch

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "Wi-Pose_sample" / "Wi-Pose"

pytestmark = pytest.mark.skipif(
    not (SAMPLE / "Train").is_dir(),
    reason="Wi-Pose sample not extracted under data/",
)


def test_item_contract():
    from piw.dataset import WiPoseDataset
    from piw.skeleton import JHM_CHANNELS, NUM_JOINTS, PAF_CHANNELS

    ds = WiPoseDataset(str(SAMPLE), "Train")
    assert len(ds) > 0
    it = ds[0]

    assert it["csi"].shape == (150, 3, 3) and it["csi"].dtype == torch.float32
    assert torch.isfinite(it["csi"]).all()

    assert it["jhm"].shape == (JHM_CHANNELS, 46, 82)
    assert 0.0 <= it["jhm"].min() and it["jhm"].max() <= 1.0

    assert it["paf"].shape == (PAF_CHANNELS, 46, 82)
    mag = torch.hypot(it["paf"][0::2], it["paf"][1::2])
    assert mag.max() <= 1.0 + 1e-6                # unit vectors or zero

    assert it["jhm_mask"].shape == (JHM_CHANNELS,)
    assert it["jhm_mask"][-1] == 1.0              # background always kept
    pm = it["paf_mask"]
    assert pm.shape == (PAF_CHANNELS,)
    assert torch.equal(pm[0::2], pm[1::2])        # limb x/y masked together

    kp = it["keypoints_grid"]
    assert kp.shape == (NUM_JOINTS, 3)
    assert (kp[:, 0] <= 82).all() and (kp[:, 1] <= 46).all()
