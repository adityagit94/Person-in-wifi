"""
Matthew Weight (MW) loss vs plain L2 on sparse joint heatmaps.
Toy reproduction of the core trick in "Person-in-WiFi" (ICCV 2019, arXiv:1904.00276).

Idea being tested:
  Joint heatmaps are mostly background. Plain L2 averages the error over
  everything, so the easiest way for the network to reduce loss is to predict
  near-zero everywhere, washing out the joints. MW up-weights foreground
  pixels by up to 2x (k=1, b=1: w in [1, 2]), nudging capacity toward the
  joints. Honest framing: with these constants the aggregate loss is still
  background-dominated, so expect a modest but consistent edge, not a
  night-and-day difference.

The loss lives in piw/losses.py (w = k*|y| + b; see its docstring for why the
magnitude form and not the paper's literal k*y + b*I(y), which goes negative).

What changed from the original draft (empirical findings, 2026-07-09):
  1. The draft's fc decoder (Linear->ReLU->Linear->ReLU -> convs) suffered
     total dead-ReLU collapse within ~200 steps: output froze at a constant,
     gradients died, neither loss learned anything.
  2. With that fixed, NO variant learned at sigma 1.5 (0.9% foreground):
     not more steps (4000), not higher lr (3e-3, 1e-2), not a sinusoidal
     measurement encoding, not a fixed dataset iterated in epochs. A
     raw-coordinates control showed the bottleneck was decoder-side
     (painting Gaussians), not inverting the measurement.
  3. Foreground density was the one lever that mattered: at sigma 3.0
     (~3.5% fg per channel) every variant learns. So this toy runs at
     SIGMA = 3.0 -- the learnable regime where the loss comparison is
     meaningful. The paper-style sigma ~1.5 still applies to the real
     Wi-Pose targets (milestone 3), where a full U-Net and 20 epochs of
     real data face a related but not identical problem.
  4. Decoder is a spatial-broadcast design (embed measurement, tile over
     the grid with per-pixel (x,y) coords, shared 1x1 convs): suited to
     coordinate->heatmap rendering, ~18k params, fast on CPU.

Toy setup:
  - 18 joints (COCO-18, same as the Wi-Pose dataset), heatmaps at 46x82.
  - Input mimics CSI in spirit: a fixed random linear projection of the
    normalized joint coordinates plus noise (a scrambled, low-dimensional
    measurement the net must invert).
  - Two IDENTICAL networks. The RNG is reseeded to INIT_SEED before each
    model build (identical weights) and to DATA_SEED right after (identical
    training-batch sequence regardless of how many draws init consumed).
    Only the loss differs.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from piw.losses import mw_loss

torch.manual_seed(0)
np.random.seed(0)

H, W = 46, 82          # paper's output resolution
J = 18                 # COCO-18 joints (matches Wi-Pose annotations)
SIGMA = 3.0            # sigma 1.5 is unlearnable for this toy; see docstring
IN_DIM = 64            # dimension of the fake "CSI" measurement
STEPS = 8000
BATCH = 32
LR = 1e-3
DEV = "cpu"
INIT_SEED = 42         # weights: same for both models
DATA_SEED = 7          # batches: same sequence for both training runs

# fixed random projection: coords (J*2) -> measurement (IN_DIM), like CSI
# being a scrambled function of the pose
PROJ = torch.randn(J * 2, IN_DIM) / (J * 2) ** 0.5

yy, xx = torch.meshgrid(torch.arange(H).float(),
                        torch.arange(W).float(), indexing="ij")


def make_batch(n):
    """Random joint coords -> (noisy measurement, target heatmaps)."""
    cx = torch.rand(n, J) * (W - 8) + 4          # col in [4, 78]
    cy = torch.rand(n, J) * (H - 8) + 4          # row in [4, 42]
    coords = torch.stack([cx / W, cy / H], -1).reshape(n, J * 2)
    meas = coords @ PROJ + 0.05 * torch.randn(n, IN_DIM)
    d2 = (xx[None, None] - cx[:, :, None, None]) ** 2 \
       + (yy[None, None] - cy[:, :, None, None]) ** 2
    target = torch.exp(-d2 / (2 * SIGMA ** 2))    # (n, J, H, W) in [0, 1]
    return meas, target, (cx, cy)


def l2_loss(pred, target):
    return ((pred - target) ** 2).mean()


class Decoder(nn.Module):
    """Spatial-broadcast decoder: measurement -> J x 46 x 82 heatmaps.

    The measurement embedding is tiled over the output grid and concatenated
    with each pixel's own normalized (row, col); shared 1x1 convs then compute
    a per-pixel function of (embedding, position) -- the natural shape for
    "how close is this pixel to each decoded joint".
    """
    def __init__(self, width=64):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(IN_DIM, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())
        self.paint = nn.Sequential(nn.Conv2d(width + 2, width, 1), nn.ReLU(),
                                   nn.Conv2d(width, width, 1), nn.ReLU(),
                                   nn.Conv2d(width, J, 1))
        self.register_buffer("grid", torch.stack([yy / H, xx / W]))

    def forward(self, x):
        h = self.embed(x)
        n = h.shape[0]
        hmap = h[:, :, None, None].expand(n, h.shape[1], H, W)
        g = self.grid[None].expand(n, 2, H, W)
        return self.paint(torch.cat([hmap, g], 1))


def train(loss_fn, tag):
    torch.manual_seed(INIT_SEED)   # identical init for both models
    model = Decoder().to(DEV)
    torch.manual_seed(DATA_SEED)   # identical batch sequence for both runs
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           betas=(0.9, 0.999))   # paper's optimizer config
    for step in range(1, STEPS + 1):
        meas, target, _ = make_batch(BATCH)
        loss = loss_fn(model(meas), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"  [{tag}] step {step:5d}  train loss {loss.item():.5f}",
                  flush=True)
    return model


@torch.no_grad()
def evaluate(model, meas, target, cx, cy):
    pred = model(meas)
    fg = target > 0.1
    fg_mse = ((pred - target) ** 2)[fg].mean().item()
    bg_mse = ((pred - target) ** 2)[~fg].mean().item()
    n = meas.shape[0]
    bi = torch.arange(n)[:, None].expand(n, J)
    ji = torch.arange(J)[None, :].expand(n, J)
    rows, cols = cy.round().long(), cx.round().long()
    peaks = pred[bi, ji, rows, cols]
    target_peaks = target[bi, ji, rows, cols]
    return pred, fg_mse, bg_mse, peaks.mean().item(), target_peaks.mean().item()


if __name__ == "__main__":
    print("Training identical networks, only the loss differs...")
    model_l2 = train(l2_loss, "plain L2")
    model_mw = train(mw_loss, "MW k1 b1")   # defaults are the JHM constants

    torch.manual_seed(123)   # fresh unseen eval data, same for both models
    meas, target, (cx, cy) = make_batch(64)
    pred_l2, fg_l2, bg_l2, pk_l2, tp = evaluate(model_l2, meas, target, cx, cy)
    pred_mw, fg_mw, bg_mw, pk_mw, _ = evaluate(model_mw, meas, target, cx, cy)

    fg_frac = (target > 0.1).float().mean().item()
    print("\n=== Validation (64 unseen samples) ===")
    print(f"foreground (target > 0.1) is {fg_frac:.1%} of pixels")
    print(f"joints sit at continuous coords, so the target value at the "
          f"rounded joint pixel averages {tp:.3f} (upper bound for peaks)\n")
    # scientific notation: background MSE is where MW is expected to LOSE to
    # L2, and fixed-point %.4f would round that difference away to 0.0000
    print(f"{'':>18}{'plain L2':>12}{'Matthew W.':>12}")
    print(f"{'foreground MSE':>18}{fg_l2:>12.2e}{fg_mw:>12.2e}")
    print(f"{'background MSE':>18}{bg_l2:>12.2e}{bg_mw:>12.2e}")
    print(f"{'mean joint peak':>18}{pk_l2:>12.3f}{pk_mw:>12.3f}")

    # ---- figure: one sample, max over joint channels ----
    i = 0
    panels = [(target[i], "Target (18 joints)"),
              (pred_l2[i], f"Plain L2  (mean peak {pk_l2:.2f})"),
              (pred_mw[i], f"Matthew Weight  (mean peak {pk_mw:.2f})")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, (hm, title) in zip(axes, panels):
        ax.imshow(hm.max(0).values, vmin=0, vmax=1, cmap="magma")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Same network, same init, same data. Only the loss differs.",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("mw_vs_l2.png", dpi=130, bbox_inches="tight")
    print("\nSaved figure to mw_vs_l2.png")
