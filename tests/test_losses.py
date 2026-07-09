import torch
import torch.nn.functional as F

from piw.losses import JHM_B, JHM_K, PAF_B, PAF_K, matthew_weight, mw_loss


def test_known_value_hand_computed():
    # PAF constants (k=1, b=0.3); one negative target to exercise the |y| form.
    target = torch.tensor([[1.0, 0.0], [0.5, -0.5]])
    pred = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    # weights  w = 1*|y| + 0.3          -> [[1.3, 0.3], [0.8, 0.8]]
    # sq. err  (pred - target)^2        -> [[0.25, 0.25], [0.25, 0.25]]
    # weighted                          -> [[0.325, 0.075], [0.2, 0.2]]
    # mean = (0.325 + 0.075 + 0.2 + 0.2) / 4 = 0.8 / 4 = 0.2
    expected_w = torch.tensor([[1.3, 0.3], [0.8, 0.8]])
    assert torch.allclose(matthew_weight(target, k=1.0, b=0.3), expected_w)
    assert torch.isclose(mw_loss(pred, target, k=1.0, b=0.3),
                         torch.tensor(0.2), atol=1e-7)


def test_weights_strictly_positive_for_negative_targets():
    torch.manual_seed(0)
    # PAF-style targets: unit-vector components in [-1, 1]
    target = torch.rand(4, 2, 46, 82) * 2 - 1
    w = matthew_weight(target, k=PAF_K, b=PAF_B)
    assert (w > 0).all()
    # The paper's literal formula w = k*y + b*I(y) DOES go negative on this
    # data, the gotcha the magnitude form exists to avoid.
    literal = PAF_K * target + PAF_B * torch.where(target >= 0, 1.0, -1.0)
    assert (literal < 0).any()


def test_k0_b1_equals_plain_mse():
    torch.manual_seed(1)
    pred = torch.randn(3, 5, 7)
    target = torch.randn(3, 5, 7)
    assert torch.isclose(mw_loss(pred, target, k=0.0, b=1.0),
                         F.mse_loss(pred, target))


def test_defaults_are_jhm_constants():
    assert (JHM_K, JHM_B) == (1.0, 1.0)
    # y=0 -> w=b=1 (background), y=1 -> w=k+b=2 (peak)
    assert torch.allclose(matthew_weight(torch.tensor([0.0, 1.0])),
                          torch.tensor([1.0, 2.0]))
