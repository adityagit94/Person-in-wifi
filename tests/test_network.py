import pytest
import torch

from piw.network import (IN_CHANNELS, OUT_H, OUT_W, PersonInWiFi,
                         count_parameters)
from piw.skeleton import JHM_CHANNELS, PAF_CHANNELS


def make_input(batch):
    return torch.randn(batch, IN_CHANNELS, 3, 3)


def test_upsample_intermediate_shape():
    # the paper's 150 x 96 x 96 intermediate
    x = PersonInWiFi.upsample_input(make_input(2))
    assert x.shape == (2, IN_CHANNELS, 96, 96)


@pytest.mark.parametrize("batch", [1, 4])
def test_forward_head_shapes(batch):
    model = PersonInWiFi().eval()
    jhm, paf = model(make_input(batch))
    assert jhm.shape == (batch, JHM_CHANNELS, OUT_H, OUT_W)   # (b, 19, 46, 82)
    assert paf.shape == (batch, PAF_CHANNELS, OUT_H, OUT_W)   # (b, 38, 46, 82)


def test_outputs_are_finite():
    model = PersonInWiFi().eval()
    jhm, paf = model(make_input(2))
    assert torch.isfinite(jhm).all() and torch.isfinite(paf).all()


def test_backward_produces_finite_grads():
    # the network must be trainable: a backward pass fills every parameter
    # with a finite gradient
    model = PersonInWiFi().train()
    jhm, paf = model(make_input(2))
    loss = jhm.pow(2).mean() + paf.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_parameter_count_is_reported():
    n = count_parameters(PersonInWiFi())
    assert n > 0
    print(f"\nPersonInWiFi parameter count: {n:,}")


def test_configurable_head_channels():
    # e.g. MM-Fi (17 joints) would need different head sizes; the network
    # must accept them without touching anything else
    model = PersonInWiFi(jhm_channels=18, paf_channels=32).eval()
    jhm, paf = model(make_input(1))
    assert jhm.shape[1] == 18 and paf.shape[1] == 32
