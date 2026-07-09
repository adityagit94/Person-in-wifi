import torch

from piw.eval import decode_keypoints, pck_correct
from piw.losses import masked_mw_loss, mw_loss
from piw.skeleton import NUM_JOINTS
from piw.targets import OUT_H, OUT_W


def test_masked_loss_matches_mw_when_all_kept():
    torch.manual_seed(0)
    pred = torch.randn(2, 3, 4, 5)
    target = torch.randn(2, 3, 4, 5)
    mask = torch.ones(2, 3)
    assert torch.isclose(masked_mw_loss(pred, target, mask), mw_loss(pred, target))


def test_masked_loss_ignores_masked_channels():
    torch.manual_seed(1)
    pred = torch.randn(2, 3, 4, 5)
    target = torch.randn(2, 3, 4, 5)
    mask = torch.ones(2, 3)
    mask[:, 1] = 0.0                              # drop channel 1
    base = masked_mw_loss(pred, target, mask)
    pred2 = pred.clone()
    pred2[:, 1] += 100.0                          # garbage in the masked channel
    assert torch.isclose(masked_mw_loss(pred2, target, mask), base)


def test_decode_recovers_peak():
    jhm = torch.zeros(1, NUM_JOINTS + 1, OUT_H, OUT_W)
    jhm[0, 0, 20, 40] = 1.0                       # joint 0 peak at row 20, col 40
    xy = decode_keypoints(jhm)
    assert xy[0, 0, 0].item() == 40 and xy[0, 0, 1].item() == 20


def test_pck_perfect_prediction():
    torch.manual_seed(2)
    gx = torch.rand(1, NUM_JOINTS) * OUT_W
    gy = torch.rand(1, NUM_JOINTS) * OUT_H
    conf = torch.ones(1, NUM_JOINTS)
    gt = torch.stack([gx, gy, conf], dim=-1)
    pred = gt[..., :2].clone()                    # exact predictions
    correct, valid = pck_correct(pred, gt)
    assert valid.all() and correct.all()
