"""Unit tests for OHLCV validation."""

import numpy as np
import pandas as pd

from src.data.validation import OHLCVValidationResult, validate_ohlcv


def test_validate_ohlcv_accepts_consistent_fixture(sample_ohlcv: pd.DataFrame) -> None:
    result = validate_ohlcv(sample_ohlcv)
    assert result.is_valid is True
    assert result.errors == []


def test_validate_ohlcv_rejects_invalid_fixture(sample_ohlcv_invalid: pd.DataFrame) -> None:
    result = validate_ohlcv(sample_ohlcv_invalid)
    assert result.is_valid is False
    assert len(result.errors) >= 2
    err_text = " ".join(result.errors)
    assert "high" in err_text.lower() or "ordering" in err_text.lower()
    assert "volume" in err_text.lower() or "negative" in err_text.lower()


def test_validate_ohlcv_returns_typed_result(sample_ohlcv: pd.DataFrame) -> None:
    result = validate_ohlcv(sample_ohlcv)
    assert isinstance(result, OHLCVValidationResult)


def test_validate_ohlcv_rejects_abs_daily_return_above_max() -> None:
    """Huge move vs prior close fails hard max_abs check (not z-score)."""
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0],
            "high": [101.0, 101.5, 102.0],
            "low": [99.5, 100.0, 100.5],
            "close": [100.5, 101.0, 500.0],
            "volume": [1e6, 1e6, 1e6],
        },
        index=idx,
    )
    result = validate_ohlcv(df, max_abs_daily_return=0.5)
    assert result.is_valid is False
    assert any("max_abs_daily_return" in e or "daily return" in e.lower() for e in result.errors)


def test_validate_ohlcv_zscore_outlier_on_close() -> None:
    """Large level jump vs prior mean is a warning when daily move stays under max_abs."""
    n = 30
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    closes = [100.0 + i * 0.01 for i in range(n - 1)] + [150.0]
    close_arr = np.asarray(closes, dtype=float)
    open_arr = np.r_[close_arr[0], close_arr[:-1]]
    high = np.maximum(open_arr, close_arr) + 0.5
    low = np.minimum(open_arr, close_arr) - 0.5
    df = pd.DataFrame(
        {
            "open": open_arr,
            "high": high,
            "low": low,
            "close": close_arr,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    result = validate_ohlcv(
        df,
        outlier_zscore_threshold=3.0,
        min_prior_for_zscore=20,
        max_abs_daily_return=0.5,
    )
    assert result.is_valid is True
    assert result.errors == []
    assert any("z-score" in w.lower() or "outlier" in w.lower() for w in result.warnings)


def test_validate_ohlcv_rejects_negative_close() -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, -10.0],
            "volume": [1e6, 1e6],
        },
        index=idx,
    )
    result = validate_ohlcv(df)
    assert result.is_valid is False
    assert any("non-positive" in e.lower() for e in result.errors)


def test_validate_ohlcv_rejects_zero_close() -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 0.0],
            "close": [100.0, 0.0],
            "volume": [1e6, 0.0],
        },
        index=idx,
    )
    result = validate_ohlcv(df)
    assert result.is_valid is False
    assert any("non-positive" in e.lower() for e in result.errors)


def test_validate_ohlcv_rejects_nan_in_required_column() -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, float("nan")],
            "volume": [1e6, 1e6],
        },
        index=idx,
    )
    result = validate_ohlcv(df)
    assert result.is_valid is False
    assert any("nan" in e.lower() for e in result.errors)


def test_validate_ohlcv_rejects_inf_in_required_column() -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, np.inf],
            "volume": [1e6, 1e6],
        },
        index=idx,
    )
    result = validate_ohlcv(df)
    assert result.is_valid is False
    assert any("inf" in e.lower() for e in result.errors)


def test_validate_ohlcv_rejects_daily_return_above_50pct() -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 201.0],
            "low": [99.0, 100.0],
            "close": [100.0, 200.0],
            "volume": [1e6, 1e6],
        },
        index=idx,
    )
    result = validate_ohlcv(df, max_abs_daily_return=0.5)
    assert result.is_valid is False
    assert any("daily return" in e.lower() for e in result.errors)


def test_validate_ohlcv_accepts_real_market_crash() -> None:
    """Single-day ~15% drop is valid (not a 50%+ feed error)."""
    n = 500
    idx = pd.bdate_range("2023-01-03", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = np.full(n, 100.0)
    for i in range(1, n):
        close[i] = close[i - 1] * (1.0 + rng.normal(0.0, 0.008))
    close[250] = close[249] * 0.85
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    result = validate_ohlcv(df, outlier_zscore_threshold=None)
    assert result.is_valid is True
    assert result.errors == []


def test_validate_ohlcv_outlier_check_logs_warning_not_error() -> None:
    """Extreme close vs prior mean logs warning; daily move stays under 50% cap."""
    n = 40
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    close = np.linspace(100.0, 115.0, n)
    close[-1] = 160.0
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    result = validate_ohlcv(df, outlier_zscore_threshold=3.0, min_prior_for_zscore=20)
    assert result.is_valid is True
    assert result.warnings


def test_validate_ohlcv_zero_close_with_volume_rejected() -> None:
    idx = pd.date_range("2024-01-02", periods=1, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [0.0],
            "volume": [1000.0],
        },
        index=idx,
    )
    result = validate_ohlcv(df)
    assert result.is_valid is False
    assert any("non-positive" in e.lower() for e in result.errors)


def test_validate_ohlcv_zero_close_with_zero_volume_rejected() -> None:
    """Zero close is rejected by strict positive-price rule (volume does not matter)."""
    idx = pd.date_range("2024-01-02", periods=1, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [0.0],
            "volume": [0.0],
        },
        index=idx,
    )
    result = validate_ohlcv(df)
    assert result.is_valid is False
    assert any("non-positive" in e.lower() for e in result.errors)
