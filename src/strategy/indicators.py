"""Technical indicators — pure pandas/numpy, point-in-time safe."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, *, period: int) -> pd.Series:
    """Simple moving average of ``close``."""
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, *, period: int) -> pd.Series:
    """Exponential moving average (``ewm(span=period, adjust=False)``)."""
    return close.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, *, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-style smoothing via EWM)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    out = 100.0 - (100.0 / (1.0 + rs))
    # When average loss is zero, RSI is 100 if there were gains, 50 if flat.
    out = out.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    out = out.mask(avg_loss.eq(0) & avg_gain.eq(0), 50.0)
    return out


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int = 14,
) -> pd.Series:
    """Average True Range (Wilder EMA of true range)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def bollinger_bands(
    close: pd.Series,
    *,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return middle band (SMA), upper, and lower Bollinger bands.

    Rolling volatility uses the **sample** standard deviation (``ddof=1``), matching
    common quant platforms; bands widen slightly vs. population std at small windows.
    """
    mid = sma(close, period=period)
    std = close.rolling(window=period, min_periods=period).std(ddof=1)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def macd(
    close: pd.Series,
    *,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram (MACD - signal).

    MACD line = EMA(fast) - EMA(slow); signal = EMA(MACD, signal_period);
    histogram = MACD - signal.
    """
    macd_line = ema(close, period=fast_period) - ema(close, period=slow_period)
    signal_line = ema(macd_line, period=signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index (Wilder-style smoothing).

    Returns ``(adx, plus_di, minus_di)``. +DM/-DM use the standard rule:
    +DM = up-move when it exceeds down-move and is positive; else 0.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=high.index, dtype=float)
    minus_dm_s = pd.Series(minus_dm, index=high.index, dtype=float)

    alpha = 1.0 / float(period)
    tr_smooth = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_smooth = plus_dm_s.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_smooth = minus_dm_s.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * plus_smooth / tr_smooth.replace(0.0, float("nan"))
    minus_di = 100.0 * minus_smooth / tr_smooth.replace(0.0, float("nan"))
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, float("nan"))
    dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    adx_line = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = plus_di.fillna(0.0)
    minus_di = minus_di.fillna(0.0)
    return adx_line, plus_di, minus_di


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator: %K and %D (SMA of %K).

    %K = 100 * (close - lowest low) / (highest high - lowest low) over ``k_period``.
    When the range is zero, %K is 50.0.
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    rng = highest_high - lowest_low
    pct_k = np.where(
        rng > 0,
        100.0 * (close - lowest_low) / rng,
        50.0,
    )
    pct_k_s = pd.Series(pct_k, index=close.index, dtype=float)
    pct_d = sma(pct_k_s, period=d_period)
    return pct_k_s, pct_d
