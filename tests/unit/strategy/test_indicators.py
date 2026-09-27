"""Unit tests for technical indicators (known inputs / outputs)."""

import numpy as np
import pandas as pd

from src.strategy.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
)


def test_sma_matches_rolling_mean() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    got = sma(close, period=3)
    expected = close.rolling(3).mean()
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_ema_matches_pandas_ewm() -> None:
    close = pd.Series(np.linspace(10.0, 20.0, 15))
    span = 5
    got = ema(close, period=span)
    expected = close.ewm(span=span, adjust=False).mean()
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_rsi_bounded_0_100() -> None:
    rng = pd.date_range("2024-01-01", periods=40, freq="B")
    # Uptrend produces high RSI
    close = pd.Series(np.linspace(50.0, 100.0, len(rng)), index=rng)
    out = rsi(close, period=14)
    valid = out.dropna()
    assert valid.min() >= 0.0
    assert valid.max() <= 100.0


def test_rsi_pure_uptrend_reaches_100() -> None:
    rng = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.Series(np.linspace(10.0, 50.0, len(rng)), index=rng)
    out = rsi(close, period=14)
    assert out.dropna().iloc[-1] == 100.0


def test_rsi_pure_downtrend_near_zero() -> None:
    rng = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.Series(np.linspace(50.0, 10.0, len(rng)), index=rng)
    out = rsi(close, period=14)
    assert out.dropna().iloc[-1] < 5.0


def test_rsi_flat_series_is_50() -> None:
    rng = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.Series([100.0] * len(rng), index=rng)
    out = rsi(close, period=14)
    assert out.dropna().iloc[-1] == 50.0


def test_atr_positive_on_simple_path() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    high = pd.Series(np.linspace(11.0, 15.0, len(idx)), index=idx)
    low = pd.Series(np.linspace(10.0, 14.0, len(idx)), index=idx)
    close = pd.Series(np.linspace(10.5, 14.5, len(idx)), index=idx)
    out = atr(high, low, close, period=5)
    assert out.dropna().min() > 0.0


def test_bollinger_bands_symmetric_around_sma() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mid, upper, lower = bollinger_bands(close, period=5, num_std=2.0)
    assert len(mid) == len(close)
    m = mid.dropna().iloc[-1]
    u = upper.dropna().iloc[-1]
    ell = lower.dropna().iloc[-1]
    assert u > m > ell
    # Sample std (ddof=1) vs population: last window std matches pandas convention
    win = close.iloc[-5:]
    expected_std = win.std(ddof=1)
    assert abs((u - m) / 2.0 - expected_std) < 1e-9


def test_macd_histogram_is_macd_minus_signal() -> None:
    rng = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series(np.linspace(100.0, 120.0, len(rng)), index=rng)
    m, s, h = macd(close)
    valid = m.dropna().index.intersection(s.dropna().index).intersection(h.dropna().index)
    assert len(valid) > 0
    for i in valid:
        assert abs(m.loc[i] - s.loc[i] - h.loc[i]) < 1e-9


def test_macd_crossover_detectable_on_trending_data() -> None:
    rng = pd.date_range("2024-01-01", periods=100, freq="B")
    close = pd.Series(np.linspace(50.0, 150.0, len(rng)), index=rng)
    m, sig, _h = macd(close)
    diff = m - sig
    # Strong uptrend: MACD typically above signal at the end
    assert diff.dropna().iloc[-1] > 0


def test_macd_custom_periods() -> None:
    rng = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.Series(np.linspace(10.0, 20.0, len(rng)), index=rng)
    m_default, _, _ = macd(close)
    m_fast, _, _ = macd(close, fast_period=5, slow_period=10, signal_period=3)
    assert m_default.dropna().iloc[-1] != m_fast.dropna().iloc[-1]


def test_adx_rises_in_strong_trend() -> None:
    rng = pd.date_range("2024-01-01", periods=120, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, len(rng)), index=rng)
    high = close * 1.002
    low = close * 0.998
    adx_line, _, _ = adx(high, low, close, period=14)
    assert adx_line.dropna().iloc[-1] > 25.0


def test_adx_low_in_range_bound_market() -> None:
    rng = pd.date_range("2024-01-01", periods=120, freq="B")
    base = 100.0
    # Tight oscillation around a level (chop) — small range keeps ADX well below trend levels
    t = np.arange(len(rng))
    wobble = 0.08 * np.sin(t / 2.5)
    close = pd.Series(base + wobble, index=rng)
    high = close + 0.05
    low = close - 0.05
    adx_line, _, _ = adx(high, low, close, period=14)
    assert adx_line.dropna().iloc[-1] < 20.0


def test_adx_plus_di_exceeds_minus_di_in_uptrend() -> None:
    rng = pd.date_range("2024-01-01", periods=100, freq="B")
    close = pd.Series(np.linspace(50.0, 120.0, len(rng)), index=rng)
    high = close * 1.01
    low = close * 0.99
    _adx, pdi, mdi = adx(high, low, close, period=14)
    assert pdi.dropna().iloc[-1] > mdi.dropna().iloc[-1]


def test_stochastic_bounded_0_100() -> None:
    rng = pd.date_range("2024-01-01", periods=40, freq="B")
    close = pd.Series(np.linspace(90.0, 110.0, len(rng)), index=rng)
    high = close * 1.01
    low = close * 0.99
    k, d = stochastic(high, low, close, k_period=14, d_period=3)
    assert k.dropna().min() >= 0.0
    assert k.dropna().max() <= 100.0
    assert d.dropna().min() >= 0.0
    assert d.dropna().max() <= 100.0


def test_stochastic_oversold_in_downtrend() -> None:
    rng = pd.date_range("2024-01-01", periods=60, freq="B")
    close = pd.Series(np.linspace(120.0, 50.0, len(rng)), index=rng)
    high = close * 1.005
    low = close * 0.995
    k, _d = stochastic(high, low, close, k_period=14, d_period=3)
    assert k.dropna().iloc[-1] < 20.0
