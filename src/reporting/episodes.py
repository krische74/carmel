"""Drawdown-episode slicing, buy-and-hold, and a 75/25 SPY/cash static blend."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel

from src.reporting.returns import compute_return_metrics


class EpisodeSlice(BaseModel):
    """Named calendar window for conditional backtest analysis."""

    name: str
    start: date
    end: date
    label: str = ""


DRAWDOWN_EPISODES: tuple[EpisodeSlice, ...] = (
    EpisodeSlice(
        name="gfc_2008",
        start=date(2007, 10, 1),
        end=date(2009, 3, 31),
        label="2007-10 → 2009-03 GFC",
    ),
    EpisodeSlice(
        name="correction_2011",
        start=date(2011, 5, 1),
        end=date(2011, 10, 31),
        label="2011-05 → 2011-10 correction",
    ),
    EpisodeSlice(
        name="q4_2018",
        start=date(2018, 10, 1),
        end=date(2018, 12, 31),
        label="2018-10 → 2018-12 selloff",
    ),
    EpisodeSlice(
        name="covid_2020",
        start=date(2020, 2, 1),
        end=date(2020, 3, 31),
        label="2020-02 → 2020-03 crash",
    ),
    EpisodeSlice(
        name="bear_2022",
        start=date(2022, 1, 1),
        end=date(2022, 10, 31),
        label="2022-01 → 2022-10 bear",
    ),
)

BULL_EPISODES: tuple[EpisodeSlice, ...] = (
    EpisodeSlice(
        name="bull_2009_2011",
        start=date(2009, 4, 1),
        end=date(2011, 4, 30),
        label="2009-04 → 2011-04 advance",
    ),
    EpisodeSlice(
        name="bull_2011_2018",
        start=date(2011, 11, 1),
        end=date(2018, 9, 30),
        label="2011-11 → 2018-09 advance",
    ),
    EpisodeSlice(
        name="bull_2019_2020",
        start=date(2019, 1, 1),
        end=date(2020, 1, 31),
        label="2019-01 → 2020-01 advance",
    ),
    EpisodeSlice(
        name="bull_2020_2021",
        start=date(2020, 4, 1),
        end=date(2021, 12, 31),
        label="2020-04 → 2021-12 advance",
    ),
    EpisodeSlice(
        name="bull_2022_present",
        start=date(2022, 11, 1),
        end=date(2026, 8, 7),
        label="2022-11 → 2026-08 advance",
    ),
)


class EpisodeMetrics(BaseModel):
    """Return, max drawdown, and recovery for one named window."""

    name: str
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    calmar_ratio: float | None
    recovery_trading_days: int | None
    trading_days: int


def _curve_as_frame(equity_curve: list[dict[str, Any]]) -> pd.DataFrame:
    if not equity_curve:
        return pd.DataFrame(columns=["date", "equity"])
    frame = pd.DataFrame(equity_curve)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["equity"] = frame["equity"].astype(float)
    return frame.sort_values("date").reset_index(drop=True)


def _slice_curve(
    equity_curve: list[dict[str, Any]],
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    frame = _curve_as_frame(equity_curve)
    if frame.empty:
        return frame
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    return frame.loc[(frame["date"] >= lo) & (frame["date"] <= hi)].reset_index(drop=True)


def episode_slice_metrics(
    equity_curve: list[dict[str, Any]],
    episode: EpisodeSlice,
) -> EpisodeMetrics:
    """Return and max drawdown of ``equity_curve`` inside ``episode`` (inclusive)."""
    sl = _slice_curve(equity_curve, start=episode.start, end=episode.end)
    if len(sl) < 2:
        return EpisodeMetrics(
            name=episode.name,
            total_return_pct=0.0,
            cagr_pct=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=None,
            recovery_trading_days=recovery_trading_days(
                equity_curve, start=episode.start, end=episode.end
            ),
            trading_days=max(0, len(sl) - 1),
        )
    eqs = sl["equity"].to_numpy(dtype=float)
    rets = (eqs[1:] - eqs[:-1]) / eqs[:-1]
    idx = sl["date"].iloc[1:].dt.strftime("%Y-%m-%d")
    series = pd.Series(rets, index=idx)
    m = compute_return_metrics(series)
    return EpisodeMetrics(
        name=episode.name,
        total_return_pct=m.total_return_pct,
        cagr_pct=m.cagr_pct,
        max_drawdown_pct=m.max_drawdown_pct,
        calmar_ratio=m.calmar_ratio,
        recovery_trading_days=recovery_trading_days(
            equity_curve, start=episode.start, end=episode.end
        ),
        trading_days=m.trading_days,
    )


def recovery_trading_days(
    equity_curve: list[dict[str, Any]],
    *,
    start: date,
    end: date,
) -> int | None:
    """Trading days from the episode max-DD trough until equity restores that peak.

    Searches the **full** curve after the trough so recovery after the episode
    window still counts. Returns ``None`` if the peak is never restored.
    """
    frame = _curve_as_frame(equity_curve)
    if frame.empty:
        return None
    sl = _slice_curve(equity_curve, start=start, end=end)
    if sl.empty:
        return None
    wealth = sl["equity"].to_numpy(dtype=float)
    running_max = pd.Series(wealth).cummax().to_numpy()
    dd = wealth / running_max - 1.0
    trough_i = int(dd.argmin())
    peak_level = float(running_max[trough_i])
    trough_date = sl["date"].iloc[trough_i]
    after = frame.loc[frame["date"] > trough_date]
    if after.empty:
        return None
    restored = after.loc[after["equity"] >= peak_level - 1e-9]
    if restored.empty:
        return None
    recover_date = restored["date"].iloc[0]
    n = int((frame["date"] > trough_date).sum() - (frame["date"] > recover_date).sum())
    return n


def buy_and_hold_equity_curve(
    ohlcv: pd.DataFrame,
    *,
    initial_capital: float = 10_000.0,
) -> list[dict[str, Any]]:
    """Close-to-close buy-and-hold equity from the first bar in ``ohlcv``."""
    if ohlcv.empty or "close" not in ohlcv.columns:
        return []
    closes = ohlcv["close"].astype(float)
    first = float(closes.iloc[0])
    if first <= 0.0:
        return []
    out: list[dict[str, Any]] = []
    for ts, px in closes.items():
        d = pd.Timestamp(ts).normalize().date().isoformat()
        out.append({"date": d, "equity": initial_capital * (float(px) / first)})
    return out


def static_blend_equity_curve(
    spy_ohlcv: pd.DataFrame,
    *,
    initial_capital: float = 10_000.0,
    spy_weight: float = 0.75,
    cash_yield_annual_pct: float = 4.0,
    cash_yield_by_date: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Daily-rebalanced ``spy_weight`` / remainder-in-cash equity curve."""
    if spy_ohlcv.empty or "close" not in spy_ohlcv.columns:
        return []
    from src.backtesting.cash_yield import lookup_cash_yield_annual_pct

    closes = spy_ohlcv["close"].astype(float)
    cash_w = 1.0 - float(spy_weight)
    yield_series = None
    if cash_yield_by_date is not None:
        yield_series = pd.Series(
            {pd.Timestamp(k): float(v) for k, v in cash_yield_by_date.items()},
            dtype=float,
        )
        if not yield_series.empty:
            yield_series.index = pd.DatetimeIndex(yield_series.index).normalize()
    equity = float(initial_capital)
    out: list[dict[str, Any]] = []
    prev: float | None = None
    for ts, px in closes.items():
        px_f = float(px)
        d = pd.Timestamp(ts).normalize().date()
        if yield_series is not None:
            y_ann, _mode = lookup_cash_yield_annual_pct(
                yield_series,
                d,
                fallback_pct=float(cash_yield_annual_pct),
            )
            daily_cash = (y_ann / 100.0) / 252.0
        else:
            daily_cash = (float(cash_yield_annual_pct) / 100.0) / 252.0
        if prev is not None and prev > 0.0:
            spy_r = (px_f - prev) / prev
            equity *= 1.0 + float(spy_weight) * spy_r + cash_w * daily_cash
        out.append({"date": d.isoformat(), "equity": equity})
        prev = px_f
    return out


def strategy_usable_start_dates(data: dict[str, pd.DataFrame]) -> dict[str, date]:
    """Earliest date each strategy can be modelled given available OHLCV.

    Momentum and mean-reversion are SHV-bound (cash leg). DCA is VXUS-bound.
    """

    def _first(sym: str) -> date | None:
        df = data.get(sym.strip().upper())
        if df is None or df.empty:
            return None
        return pd.Timestamp(df.index.min()).normalize().date()

    shv = _first("SHV")
    vxus = _first("VXUS")
    out: dict[str, date] = {}
    if shv is not None:
        out["momentum"] = shv
        out["mean_reversion"] = shv
    if vxus is not None:
        out["dca"] = vxus
    return out
