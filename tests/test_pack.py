"""Pack round-trip: the packed loader must produce items identical to the
.mat loader. Needs the extracted sample, skips cleanly elsewhere.
"""

from pathlib import Path

import pytest
import torch

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "Wi-Pose_sample" / "Wi-Pose"

pytestmark = pytest.mark.skipif(
    not (SAMPLE / "Train").is_dir(),
    reason="Wi-Pose sample not extracted under data/",
)


def test_pack_roundtrip(tmp_path):
    from piw.dataset import PackedWiPose, WiPoseDataset
    from piw.pack import pack_split

    pack_split(str(SAMPLE), str(tmp_path), "Train")
    packed = PackedWiPose(str(tmp_path), "Train")
    raw = WiPoseDataset(str(SAMPLE), "Train")
    assert len(packed) == len(raw)

    for i in (0, len(raw) // 2, len(raw) - 1):
        a, b = packed[i], raw[i]
        assert a.keys() == b.keys()
        for k in a:
            # skel is stored float32 in the pack, so tiny rounding is expected
            assert torch.allclose(a[k], b[k], atol=1e-3), (i, k)
