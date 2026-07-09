"""Matthew Weight (MW) loss from "Person-in-WiFi" (ICCV 2019, arXiv:1904.00276).

The paper trains joint heatmaps (JHMs) and part affinity fields (PAFs) with a
weighted L2 whose per-pixel weight grows with the target value, tilting the
loss toward the sparse foreground (joint peaks, limb vectors) that plain
averaging over the ~98-99% background pixels would wash out ("Matthew effect":
the rich get richer). With the paper's constants the tilt is capped at 2:1, a
nudge rather than a takeover; the stage-1 toy experiment measured exactly that.

The negative-weight gotcha
--------------------------
The paper writes the weight as

    w = k * y + b * I(y),    I(y) = +1 if y >= 0 else -1.

Taken literally this goes NEGATIVE for negative targets. PAF targets are unit
vector components in [-1, 1], so y = -0.8 with the paper's PAF constants
(k=1, b=0.3) gives w = -0.8 - 0.3 = -1.1. A negative weight multiplied into
the squared error *rewards increasing* the error on those pixels, which cannot
be intended. The only sane reading is that the weight is a magnitude:

    w = k * |y| + b

For JHMs (y >= 0 everywhere) the two forms are identical; for PAFs the
magnitude form keeps every weight strictly positive (>= b > 0). This module
implements the magnitude form.

Constants from the paper: JHM k=1, b=1; PAF k=1, b=0.3.
Head loss weights: lambda_JHM = lambda_PAF = 1.
"""

# Constants from the paper (defaults below are the JHM pair).
JHM_K, JHM_B = 1.0, 1.0
PAF_K, PAF_B = 1.0, 0.3


def matthew_weight(target, k=JHM_K, b=JHM_B):
    """Per-element weight w = k * |y| + b; strictly positive whenever b > 0."""
    return k * target.abs() + b


def mw_loss(pred, target, k=JHM_K, b=JHM_B):
    """Matthew-Weighted L2: mean( (k*|y| + b) * (pred - y)**2 ).

    With k=0, b=1 this reduces exactly to plain MSE.
    """
    return (matthew_weight(target, k, b) * (pred - target) ** 2).mean()


def masked_mw_loss(pred, target, channel_mask, k=JHM_K, b=JHM_B):
    """Matthew-Weighted L2 that ignores masked-out channels.

    pred, target : (B, C, H, W)
    channel_mask : (B, C) with 1.0 for channels to keep, 0.0 to ignore
                   (low-confidence joints, or limbs missing an endpoint).

    Averages the weighted squared error over the kept channels' elements, so it
    matches ``mw_loss`` when every channel is kept.
    """
    w = matthew_weight(target, k, b)
    m = channel_mask[..., None, None]                 # (B, C, 1, 1)
    num = (m * w * (pred - target) ** 2).sum()
    den = m.expand_as(pred).sum().clamp_min(1.0)
    return num / den


def pose_loss(jhm_pred, jhm_tgt, jhm_mask, paf_pred, paf_tgt, paf_mask,
              lam_jhm=1.0, lam_paf=1.0):
    """Combined JHM + PAF training loss (paper head weights lambda = 1 each).

    Returns (total, jhm_term, paf_term); the terms are detached for logging.
    """
    lj = masked_mw_loss(jhm_pred, jhm_tgt, jhm_mask, JHM_K, JHM_B)
    lp = masked_mw_loss(paf_pred, paf_tgt, paf_mask, PAF_K, PAF_B)
    return lam_jhm * lj + lam_paf * lp, lj.detach(), lp.detach()
