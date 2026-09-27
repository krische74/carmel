"""ML feature vector construction (no sklearn)."""

from __future__ import annotations

from datetime import UTC, datetime, time

import numpy as np
import pandas as pd
import pytest

from src.ml.features import ML_FEATURE_NAMES, build_feature_vector, feature_dict_to_row


def _synthetic_ohlcv(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    c = np.maximum(1.0, 100.0 + np.cumsum(rng.normal(0, 0.3, size=n)))
    # Coherent bars so indicators stay finite (high/low bracket open/close).
    wiggle = np.abs(rng.normal(0, 0.004, size=n))
    high = c * (1.0 + wiggle)
    low = c * (1.0 - wiggle)
    open_ = np.r_[c[0], c[:-1]]
    high = np.maximum(high, np.maximum(open_, c))
    low = np.minimum(low, np.minimum(open_, c))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": c,
            "volume": rng.integers(1_000_000, 2_000_000, size=n),
        },
        index=pd.date_range("2024-01-02", periods=n, freq="B"),
    )


def test_build_feature_vector_returns_all_ml_feature_names() -> None:
    df = _synthetic_ohlcv(80)
    last = df.index[-1]
    as_of = datetime.combine(last.date(), time(16, 0), tzinfo=UTC)
    feats = build_feature_vector(df, as_of=as_of)
    assert feats is not None
    for name in ML_FEATURE_NAMES:
        assert name in feats
        assert isinstance(feats[name], float)


def test_build_feature_vector_rejects_wrong_case_ohlcv_columns() -> None:
    df = _synthetic_ohlcv(80).rename(columns={"open": "Open"})
    last = df.index[-1]
    as_of = datetime.combine(last.date(), time(16, 0), tzinfo=UTC)
    assert build_feature_vector(df, as_of=as_of) is None


def test_build_feature_vector_rejects_non_datetime_index() -> None:
    raw = _synthetic_ohlcv(80).reset_index(drop=True)
    as_of = datetime(2024, 5, 15, 12, 0, tzinfo=UTC)
    assert build_feature_vector(raw, as_of=as_of) is None


def test_feature_dict_to_row_order_matches_names() -> None:
    feats = {n: float(i) for i, n in enumerate(ML_FEATURE_NAMES)}
    row = feature_dict_to_row(feats, ML_FEATURE_NAMES)
    assert row.shape == (1, len(ML_FEATURE_NAMES))
    for i, _name in enumerate(ML_FEATURE_NAMES):
        assert row[0, i] == pytest.approx(float(i))
