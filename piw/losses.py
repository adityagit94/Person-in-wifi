"""Matthew Weight (MW) loss from "Person-in-WiFi" (ICCV 2019, arXiv:1904.00276).

The paper trains joint heatmaps (JHMs) and part affinity fields (PAFs) with a
weighted L2 whose per-pixel weight grows with the target value, so the sparse
foreground (joint peaks, limb vectors) dominates the gradient instead of the
~98-99% background pixels ("Matthew effect": the rich get richer).

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
