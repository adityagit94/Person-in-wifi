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
    correct, scored = pck_correct(pred, gt)
    assert scored.all() and correct.all()


def test_pck_measures_in_source_pixel_space():
    # The grid squashes the portrait frame anisotropically: one grid step is
    # ~5.9 source px in x but ~13.9 in y. The SAME 10-cell grid offset must
    # therefore pass in x (58 px) and fail in y (139 px) against a ~109 px
    # threshold. Grid-space scoring would treat the two identically.
    gx = torch.full((1, NUM_JOINTS), 40.0)
    gy = torch.full((1, NUM_JOINTS), 25.0)
    gx[0, 0], gy[0, 0] = 10.0, 10.0               # bbox corners
    gx[0, 1], gy[0, 1] = 70.0, 40.0
    gt = torch.stack([gx, gy, torch.ones(1, NUM_JOINTS)], dim=-1)

    pred = gt[..., :2].clone()
    pred[0, 2, 0] += 10.0                         # offset joint 2 in x
    correct_x, _ = pck_correct(pred, gt)
    assert correct_x[0, 2]

    pred = gt[..., :2].clone()
    pred[0, 2, 1] += 10.0                         # same offset, in y
    correct_y, _ = pck_correct(pred, gt)
    assert not correct_y[0, 2]


def test_pck_excludes_unscoreable_samples():
    # fewer than 2 confident joints: no bbox, so nothing in the sample is
    # scored (rather than everything counting as a miss)
    gx = torch.rand(1, NUM_JOINTS) * OUT_W
    gy = torch.rand(1, NUM_JOINTS) * OUT_H
    conf = torch.full((1, NUM_JOINTS), 0.05)
    conf[0, 0] = 1.0                              # only one confident joint
    gt = torch.stack([gx, gy, conf], dim=-1)
    correct, scored = pck_correct(gt[..., :2].clone(), gt)
    assert scored.sum() == 0 and correct.sum() == 0
