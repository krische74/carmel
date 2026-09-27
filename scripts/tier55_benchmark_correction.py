"""Tier 55A — recompute adjusted-store benchmark series and repair 54F cache."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.backtesting.cash_yield import series_from_macro_rows
from src.config import get_settings, hub_sqlite_path
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.returns import compute_return_metrics

START = date(2011, 1, 28)
END = date(2026, 8, 19)
INITIAL_CAPITAL = 10_000.0
SERIES = ("SPY", "BND", "VOO", "VXUS")


def _close_series(store: ParquetStore, symbol: str) -> pd.Series:
    """Load one adjusted close series restricted to the benchmark window."""
    frame = store.read_ohlcv(symbol)
    if frame.empty:
        raise ValueError(f"missing parquet data for {symbol}")
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    closes = frame.loc[
        (frame.index.date >= START) & (frame.index.date <= END),
        "close",
    ].astype(float)
    return closes[~closes.index.duplicated()].sort_index()


def _risk_free_daily(settings) -> pd.Series:
    """Load the historical DTB3 annual-percent series as daily decimals."""
    rows = SQLiteStore(hub_sqlite_path(settings)).get_macro_indicator("DTB3")
    series = series_from_macro_rows(rows)
    if series.empty:
        raise ValueError("DTB3 series is empty")
    series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
    return (series.astype(float) / 100.0) / 252.0


def _metrics(equity: pd.Series, risk_free_daily: pd.Series) -> dict[str, float | int]:
    """Return calendar-year CAGR plus daily DTB3-excess risk metrics."""
    returns = equity.pct_change().dropna()
    metrics = compute_return_metrics(returns, risk_free_daily=risk_free_daily)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = ((float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0) * 100.0
    return {
        "cagr_pct": round(cagr, 4),
        "sharpe": (
            None
            if metrics.sharpe_ratio is None
            else round(float(metrics.sharpe_ratio), 4)
        ),
        "max_dd_pct": round(float(metrics.max_drawdown_pct), 4),
        "max_dd_pct_signed": round(-float(metrics.max_drawdown_pct), 4),
        "total_return_pct": round(float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0), 4),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "trading_days": len(returns),
    }


def _buy_and_hold(closes: pd.Series) -> pd.Series:
    """Build a close-to-close buy-and-hold equity curve."""
    return INITIAL_CAPITAL * closes / float(closes.iloc[0])


def _monthly_blend(
    spy: pd.Series,
    bnd: pd.Series,
    *,
    spy_weight: float,
) -> pd.Series:
    """Build a monthly-close-rebalanced SPY/BND equity curve."""
    prices = pd.concat([spy, bnd], axis=1, join="inner").dropna()
    prices.columns = ["spy", "bnd"]
    equity = INITIAL_CAPITAL
    shares: tuple[float, float] | None = None
    prior_month: tuple[int, int] | None = None
    values: list[float] = []
    for timestamp, row in prices.iterrows():
        month = (timestamp.year, timestamp.month)
        if shares is None:
            shares = (
                equity * spy_weight / float(row["spy"]),
                equity * (1.0 - spy_weight) / float(row["bnd"]),
            )
        else:
            equity = shares[0] * float(row["spy"]) + shares[1] * float(row["bnd"])
            if month != prior_month:
                shares = (
                    equity * spy_weight / float(row["spy"]),
                    equity * (1.0 - spy_weight) / float(row["bnd"]),
                )
        equity = shares[0] * float(row["spy"]) + shares[1] * float(row["bnd"])
        values.append(equity)
        prior_month = month
    return pd.Series(values, index=prices.index)


def _benchmark_rows(
    store: ParquetStore,
    risk_free_daily: pd.Series,
) -> dict[str, dict[str, float | int]]:
    """Compute the five pre-registered benchmark rows."""
    closes = {symbol: _close_series(store, symbol) for symbol in SERIES}
    rows = {
        "SPY": _metrics(_buy_and_hold(closes["SPY"]), risk_free_daily),
        "BND": _metrics(_buy_and_hold(closes["BND"]), risk_free_daily),
    }
    for label, weight in (("75_25_SPY_BND", 0.75), ("65_35_SPY_BND", 0.65), ("60_40_SPY_BND", 0.60)):
        rows[label] = _metrics(
            _monthly_blend(closes["SPY"], closes["BND"], spy_weight=weight),
            risk_free_daily,
        )
    return rows


def _repair_tier54f(
    benchmark_rows: dict[str, dict[str, float | int]],
    cache_path: Path,
) -> dict:
    """Amend the 54F cache while retaining its superseded benchmark values."""
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    old_headlines = dict(payload["headline_benchmarks"])
    old_headline_basis = {
        "spy_cagr_pct": old_headlines["spy_cagr_pct"],
        "blend_75_25_cagr_pct": old_headlines["blend_75_25_cagr_pct"],
    }
    current_spy = benchmark_rows["SPY"]
    current_blend = benchmark_rows["75_25_SPY_BND"]
    for row in payload["headline_rows"]:
        benchmarks = row["benchmarks"]
        for key, current in (("spy", current_spy), ("blend_75_25", current_blend)):
            old = dict(benchmarks[key])
            benchmarks[key] = dict(current)
            benchmarks[key]["superseded_price_basis"] = old
    payload["headline_benchmarks"] = {
        "spy_cagr_pct": current_spy["cagr_pct"],
        "blend_75_25_cagr_pct": current_blend["cagr_pct"],
        "superseded_price_basis": old_headline_basis,
    }
    payload["benchmark_correction"] = {
        "corrected_on": "2026-08-31",
        "basis": "dividend-adjusted close series from data/parquet",
        "superseded_price_basis": old_headline_basis,
        "artifact": "data/cache/tier54f_results.json is a cached artifact amended in place",
    }
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    """Write the 55A benchmark artifact and repair the 54F cached benchmarks."""
    settings = get_settings()
    store = ParquetStore("data/parquet")
    risk_free_daily = _risk_free_daily(settings)
    rows = _benchmark_rows(store, risk_free_daily)
    artifact = {
        "tier": "55A",
        "window": {"start": START.isoformat(), "end": END.isoformat()},
        "data_basis": "dividend-adjusted close series from data/parquet",
        "cagr_basis": "calendar years using 365.25 days",
        "blend_rebalance": "at the first available close of each calendar month",
        "risk_free": "DTB3 historical annual percent, converted to daily / 252",
        "max_dd_convention": "max_dd_pct is a positive magnitude; max_dd_pct_signed is also provided",
        "benchmarks": rows,
        "superseded_price_basis": {
            "SPY": {"cagr_pct": 12.27},
            "75_25_SPY_BND": {"cagr_pct": 9.80},
        },
    }
    artifact_path = Path("data/cache/tier55a_benchmarks.json")
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    _repair_tier54f(rows, Path("data/cache/tier54f_results.json"))
    print(f"wrote {artifact_path}")
    print("benchmarks=" + json.dumps(rows, sort_keys=True))
    print("repaired data/cache/tier54f_results.json")


if __name__ == "__main__":
    main()
