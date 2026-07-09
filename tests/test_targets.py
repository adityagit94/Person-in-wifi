import numpy as np

from piw.skeleton import LIMBS, NUM_JOINTS, JHM_CHANNELS, PAF_CHANNELS
from piw.targets import (OUT_H, OUT_W, SIGMA, jhm_weight_mask,
                         paf_weight_mask, render_jhm, render_paf,
                         scale_keypoints)


def _gauss(cx, cy):
    yy, xx = np.mgrid[0:OUT_H, 0:OUT_W]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA ** 2))


def _one_valid(joint):
    valid = np.zeros(NUM_JOINTS, bool)
    valid[joint] = True
    return valid


def test_scale_keypoints_maps_corner():
    gx, gy = scale_keypoints([640], [480], src_w=640, src_h=480)
    assert np.isclose(gx[0], OUT_W) and np.isclose(gy[0], OUT_H)


def test_jhm_shape_and_peak_location():
    gx = np.zeros(NUM_JOINTS); gy = np.zeros(NUM_JOINTS)
    gx[0], gy[0] = 40.0, 20.0
    hm = render_jhm(gx, gy, _one_valid(0))
    assert hm.shape == (JHM_CHANNELS, OUT_H, OUT_W)
    row, col = np.unravel_index(hm[0].argmax(), hm[0].shape)
    assert (row, col) == (20, 40)                 # peak sits on the joint
    assert np.isclose(hm[0].max(), 1.0, atol=1e-3)


def test_jhm_background_is_complement_over_all_joints():
    # joint 0 valid at (40, 20); all other joints invalid, sitting at (0, 0).
    # Background subtracts ALL joints (valid or not), so it dips at (0, 0)
    # even though those joints' own channels stay empty.
    gx = np.zeros(NUM_JOINTS); gy = np.zeros(NUM_JOINTS)
    gx[0], gy[0] = 40.0, 20.0
    hm = render_jhm(gx, gy, _one_valid(0))
    expected_bg = 1.0 - np.maximum(_gauss(40.0, 20.0), _gauss(0.0, 0.0))
    assert np.allclose(hm[NUM_JOINTS], expected_bg, atol=1e-6)
    assert hm[NUM_JOINTS][0, 0] < 0.01      # not "confidently empty" there
    assert hm[1].max() == 0.0               # invalid channel still empty


def test_invalid_joint_channel_empty_and_masked():
    gx = np.full(NUM_JOINTS, 40.0); gy = np.full(NUM_JOINTS, 20.0)
    hm = render_jhm(gx, gy, _one_valid(0))
    assert hm[1].max() == 0.0                     # joint 1 invalid -> empty
    mask = jhm_weight_mask(_one_valid(0))
    assert mask[0] == 1.0 and mask[1] == 0.0 and mask[NUM_JOINTS] == 1.0


def test_paf_shape_unit_vectors_and_direction():
    a, b = LIMBS[0]                               # first limb
    gx = np.zeros(NUM_JOINTS); gy = np.zeros(NUM_JOINTS)
    gx[a], gy[a] = 10.0, 10.0
    gx[b], gy[b] = 30.0, 10.0                     # horizontal limb, +x direction
    valid = np.zeros(NUM_JOINTS, bool); valid[a] = valid[b] = True
    paf = render_paf(gx, gy, valid)
    assert paf.shape == (PAF_CHANNELS, OUT_H, OUT_W)
    on = paf[0] != 0
    assert on.any()
    mag = np.hypot(paf[0][on], paf[1][on])
    assert np.allclose(mag, 1.0, atol=1e-5)       # unit vectors
    assert np.allclose(paf[0][on], 1.0) and np.allclose(paf[1][on], 0.0)


def test_paf_masked_when_endpoint_invalid():
    a, b = LIMBS[0]
    gx = np.zeros(NUM_JOINTS); gy = np.zeros(NUM_JOINTS)
    gx[a], gy[a], gx[b], gy[b] = 10.0, 10.0, 30.0, 10.0
    valid = np.zeros(NUM_JOINTS, bool); valid[a] = True    # b invalid
    paf = render_paf(gx, gy, valid)
    assert paf[0].max() == 0.0 and paf[1].max() == 0.0
    mask = paf_weight_mask(valid)
    assert mask[0] == 0.0 and mask[1] == 0.0
