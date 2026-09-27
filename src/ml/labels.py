"""Training labels from future returns (training pipeline only — not for live signals)."""

from __future__ import annotations

import numpy as np


def forward_return_binary(
    close_values: np.ndarray,
    *,
    index: int,
    horizon: int,
) -> int | None:
    """Return 1 if close[i+h] > close[i], 0 if not, None if horizon not available."""
    if horizon < 1:
        return None
    i = int(index)
    if i < 0 or i + horizon >= len(close_values):
        return None
    a = float(close_values[i])
    b = float(close_values[i + horizon])
    if a <= 0.0 or np.isnan(a) or np.isnan(b):
        return None
    return 1 if b > a else 0
