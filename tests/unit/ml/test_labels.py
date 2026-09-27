"""Training labels (numpy only)."""

from __future__ import annotations

import numpy as np

from src.ml.labels import forward_return_binary


def test_forward_return_binary_up_and_down() -> None:
    c = np.array([10.0, 10.0, 11.0, 10.5], dtype=float)
    assert forward_return_binary(c, index=0, horizon=2) == 1
    assert forward_return_binary(c, index=2, horizon=1) == 0


def test_forward_return_binary_none_when_horizon_missing() -> None:
    c = np.array([1.0, 2.0], dtype=float)
    assert forward_return_binary(c, index=0, horizon=5) is None
