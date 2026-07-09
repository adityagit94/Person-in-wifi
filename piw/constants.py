"""Shared spatial constants.

The paper's label/output grid is 46 x 82. The network heads and the target
renderer must always agree on this, so it is defined once here and imported
everywhere else.
"""

OUT_H, OUT_W = 46, 82
