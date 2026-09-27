"""Portfolio return analytics from stored equity snapshots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


class ReturnMetrics(BaseModel):
    """Standard portfolio return analytics computed from a daily return series."""

    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    calmar_ratio: float | None
    annual_volatility_pct: float
    best_day_pct: float
    worst_day_pct: float
    trading_days: int


def _equity_basis(row: dict[str, Any]) -> tuple[float | None, str]:
    """Prior-day equity and the source used.

    Prefers ``broker_equity`` when present; otherwise ``total_market_value + cash``.
    ``None`` source means the pair cannot be priced (unknown cash / missing basis).
    """
    be = row.get("broker_equity")
    if be is not None:
        try:
            val = float(be)
        except (TypeError, ValueError):
            val = None
        else:
            if val == val:  # not NaN
                return val, "broker"
    cash = row.get("cash")
    if cash is None:
        return None, "none"
    try:
        return float(row.get("total_market_value", 0.0)) + float(cash), "mv_cash"
    except (TypeError, ValueError):
        return None, "none"


def compute_daily_returns(snapshots: list[dict[str, Any]]) -> pd.Series:
    """Time-weighted daily returns from equity snapshots.

    ``r_t = (total_pnl[t] - total_pnl[t-1]) / equity[t-1]``. Equity prefers
    ``broker_equity`` when present, else ``total_market_value + cash``. The first
    pair where the basis source changes (MV+cash → broker_equity cutover) is
    skipped so the accounting correction is not booked as a return.
    """
    if len(snapshots) < 2:
        return pd.Series(dtype=float)

    rows = sorted(snapshots, key=lambda x: str(x.get("date", "")))
    rets: list[float] = []
    idx: list[str] = []

    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur = rows[i]
        equity_prev, src_prev = _equity_basis(prev)
        _equity_cur, src_cur = _equity_basis(cur)
        if equity_prev is None or src_cur == "none":
            continue
        if src_prev != src_cur:
            continue
        if equity_prev <= 0.0:
            continue
        pnl_prev = float(prev.get("total_pnl", 0.0))
        pnl_cur = float(cur.get("total_pnl", 0.0))
        r = (pnl_cur - pnl_prev) / equity_prev
        rets.append(r)
        idx.append(str(cur.get("date", "")))

    if not rets:
        return pd.Series(dtype=float)
    return pd.Series(rets, index=pd.Index(idx, name="date"))


def compute_drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Cumulative wealth drawdown from peak: ``(1+r).cumprod() / cummax - 1`` (negative = drawdown)."""
    if daily_returns.empty:
        return pd.Series(dtype=float)
    r = daily_returns.astype(float)
    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def compute_rolling_sharpe(daily_returns: pd.Series, window: int = 30) -> pd.Series:
    """Annualized rolling Sharpe (risk-free = 0) using sample std (``ddof=1``)."""
    if len(daily_returns) < window:
        return pd.Series(dtype=float)
    r = daily_returns.astype(float)
    mu = r.rolling(window=window, min_periods=window).mean()
    sig = r.rolling(window=window, min_periods=window).std(ddof=1)
    sharpe = mu / sig * float(np.sqrt(252.0))
    out = sharpe.where(sig > 1e-18)
    return out.dropna()


def illustrative_vol_scaled_cagr(
    daily_returns: pd.Series,
    *,
    target_vol_pct: float,
) -> float | None:
    """CAGR after scaling daily returns to ``target_vol_pct`` (illustrative only).

    Does not model borrowing costs, leverage constraints, or path dependency.
    """
    r = daily_returns.astype(float).dropna()
    if r.empty or float(target_vol_pct) <= 0.0:
        return None
    raw = compute_return_metrics(r)
    vol = float(raw.annual_volatility_pct)
    if vol <= 1e-12:
        return None
    scale = float(target_vol_pct) / vol
    return compute_return_metrics(r * scale).cagr_pct


def compute_return_metrics(
    daily_returns: pd.Series,
    *,
    risk_free_rate_annual: float = 0.0,
    risk_free_daily: pd.Series | None = None,
) -> ReturnMetrics:
    """Standard portfolio return analytics from a daily return series.

    When ``risk_free_daily`` is provided, **Sharpe** and **Sortino** use excess
    return over that per-day rate (aligned to ``daily_returns.index``; missing
    days fall back to ``risk_free_rate_annual / 100 / 252``).

    When ``risk_free_rate_annual`` is positive and ``risk_free_daily`` is omitted,
    excess return uses the constant daily rate ``risk_free_rate_annual / 100 / 252``.

    **Sharpe** uses the sample standard deviation of returns (``ddof=1``), matching common
    implementations. **Sortino** uses the downside deviation ``sqrt(mean(min(0, r)^2))``
    without a Bessel correction on that root term — the usual convention
    (e.g. Investopedia / many quant libraries) and intentionally asymmetric with Sharpe.
    """
    r = daily_returns.astype(float).dropna()
    n = len(r)
    if n == 0:
        return ReturnMetrics(
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe_ratio=None,
            sortino_ratio=None,
            max_drawdown_pct=0.0,
            max_drawdown_duration_days=0,
            calmar_ratio=None,
            annual_volatility_pct=0.0,
            best_day_pct=0.0,
            worst_day_pct=0.0,
            trading_days=0,
        )

    arr = r.to_numpy(dtype=float)
    total_return = float(np.prod(1.0 + arr) - 1.0)
    total_return_pct = total_return * 100.0

    cagr_pct = float((1.0 + total_return) ** (252.0 / n) - 1.0) * 100.0

    flat_daily_rf = float(risk_free_rate_annual) / 100.0 / 252.0
    if risk_free_daily is not None and not risk_free_daily.empty:
        aligned = pd.to_numeric(risk_free_daily, errors="coerce").reindex(r.index)
        daily_rf_arr = aligned.fillna(flat_daily_rf).to_numpy(dtype=float)
    else:
        daily_rf_arr = np.full(n, flat_daily_rf, dtype=float)
    if n < 2:
        sharpe: float | None = None
        sortino = None
        annual_volatility_pct = 0.0
    else:
        std_r = float(np.std(arr, ddof=1))
        excess = arr - daily_rf_arr
        mean_excess = float(np.mean(excess))
        if std_r <= 1e-18:
            sharpe = 0.0 if abs(mean_excess) <= 1e-12 else None
        else:
            sharpe = float(mean_excess / std_r * np.sqrt(252.0))
        downside = np.minimum(0.0, excess)
        downside_var = float(np.mean(downside**2))
        downside_std = float(np.sqrt(downside_var)) if downside_var > 1e-36 else 0.0
        if downside_std <= 1e-18:
            sortino = 0.0 if abs(mean_excess) <= 1e-12 else None
        else:
            sortino = float(mean_excess / downside_std * np.sqrt(252.0))
        annual_volatility_pct = float(std_r * np.sqrt(252.0) * 100.0)

    wealth = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(wealth)
    dd_ratio = wealth / running_max - 1.0
    max_drawdown_pct = float(abs(np.min(dd_ratio)) * 100.0)

    max_dd_duration = 0
    below_peak = wealth < running_max
    run_len = 0
    for b in below_peak:
        if b:
            run_len += 1
            max_dd_duration = max(max_dd_duration, run_len)
        else:
            run_len = 0

    best_day_pct = float(np.max(arr) * 100.0)
    worst_day_pct = float(np.min(arr) * 100.0)

    if max_drawdown_pct <= 1e-12:
        calmar: float | None = None
    else:
        calmar = float(cagr_pct / max_drawdown_pct)

    return ReturnMetrics(
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_duration_days=max_dd_duration,
        calmar_ratio=calmar,
        annual_volatility_pct=annual_volatility_pct,
        best_day_pct=best_day_pct,
        worst_day_pct=worst_day_pct,
        trading_days=n,
    )
