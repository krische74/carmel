"""Tier 55B — measured F0 attribution identity and satellite hurdle."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.tier54f_dca_ablation import _cash_map, _load_data, _run_arm
from src.backtesting.cash_yield import lookup_cash_yield_annual_pct
from src.config import get_settings

START = date(2011, 1, 28)
END = date(2026, 8, 19)
SLEEVE = ("VOO", "VXUS", "BND")
SATELLITE = ("SPY", "QQQ", "TLT", "GLD")
SLEEVE_WEIGHTS = {"VOO": 0.6, "VXUS": 0.3, "BND": 0.1}


def _normalized_closes(data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Return date-normalized close series for loaded symbols."""
    out: dict[str, pd.Series] = {}
    for symbol, frame in data.items():
        copy = frame.copy()
        copy.index = pd.to_datetime(copy.index).tz_localize(None).normalize()
        out[symbol.strip().upper()] = copy["close"].astype(float)
    return out


def _slice_to_window(series: pd.Series, *, start: date = START, end: date = END) -> pd.Series:
    """Restrict one close series to the contract window."""
    sliced = series.loc[(series.index.date >= start) & (series.index.date <= end)]
    sliced = sliced[~sliced.index.duplicated()].sort_index()
    if sliced.empty:
        raise ValueError(f"no data in contract window {start} -> {end}")
    first = pd.Timestamp(sliced.index[0]).date()
    assert first == start, f"{first} != {start}"
    return sliced.astype(float)


def _windowed_closes(closes: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Pin every loaded close series to the contract window."""
    return {symbol: _slice_to_window(series) for symbol, series in closes.items()}


def _calendar_cagr(equity: pd.Series) -> float:
    """Calculate CAGR using elapsed calendar years."""
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return ((float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0) * 100.0


def _monthly_blend(
    closes: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Build an equity curve rebalanced at each month's first close."""
    prices = pd.concat(
        [closes[symbol].rename(symbol) for symbol in weights],
        axis=1,
        join="inner",
    ).dropna()
    equity = 10_000.0
    shares: dict[str, float] | None = None
    prior_month: tuple[int, int] | None = None
    values: list[float] = []
    for timestamp, row in prices.iterrows():
        month = (timestamp.year, timestamp.month)
        if shares is None:
            shares = {
                symbol: equity * weight / float(row[symbol])
                for symbol, weight in weights.items()
            }
        else:
            equity = sum(shares[symbol] * float(row[symbol]) for symbol in shares)
            if month != prior_month:
                shares = {
                    symbol: equity * weight / float(row[symbol])
                    for symbol, weight in weights.items()
                }
        equity = sum(shares[symbol] * float(row[symbol]) for symbol in shares)
        values.append(equity)
        prior_month = month
    return pd.Series(values, index=prices.index)


def _cash_cagr(cash_map: dict[str, float], dates: pd.Index) -> float:
    """Calculate the calendar CAGR of a dollar held at realized DTB3 rates."""
    cash_series = pd.Series(
        {pd.Timestamp(key): float(value) for key, value in cash_map.items()},
        dtype=float,
    )
    cash_series.index = pd.DatetimeIndex(cash_series.index).normalize()
    value = 1.0
    for timestamp in dates[1:]:
        day = pd.Timestamp(timestamp).date()
        annual, _ = lookup_cash_yield_annual_pct(
            cash_series,
            day,
            fallback_pct=0.0,
        )
        value *= 1.0 + (float(annual) / 100.0) / 252.0
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    return (value ** (1.0 / years) - 1.0) * 100.0


def _measured_weights(
    result,
    closes: dict[str, pd.Series],
) -> tuple[dict[str, float], int]:
    """Reconstruct average F0 category weights from daily marks and fills."""
    trades_by_date: dict[str, list] = defaultdict(list)
    for trade in result.trades:
        trades_by_date[trade.date].append(trade)
    quantities: dict[str, float] = defaultdict(float)
    sums = {"sleeve": 0.0, "satellite": 0.0, "cash_and_sweep": 0.0}
    count = 0
    for point in result.equity_curve:
        day = point["date"]
        equity = float(point["equity"])
        for trade in trades_by_date.get(day, []):
            direction = 1.0 if trade.side == "buy" else -1.0
            quantities[trade.symbol.strip().upper()] += direction * float(trade.qty)
        values: dict[str, float] = {}
        timestamp = pd.Timestamp(day)
        for symbol, quantity in quantities.items():
            if abs(quantity) <= 1e-12 or symbol not in closes:
                continue
            price = closes[symbol].get(timestamp)
            if price is not None:
                values[symbol] = max(0.0, quantity * float(price))
        sleeve = sum(values.get(symbol, 0.0) for symbol in SLEEVE)
        satellite = sum(values.get(symbol, 0.0) for symbol in SATELLITE)
        sums["sleeve"] += sleeve / equity
        sums["satellite"] += satellite / equity
        sums["cash_and_sweep"] += (equity - sleeve - satellite) / equity
        count += 1
    return {key: round(value / count, 8) for key, value in sums.items()}, count


def main() -> None:
    """Run F0, write measured attribution, and solve the satellite hurdle."""
    settings = get_settings()
    data = _load_data(settings)
    closes = _windowed_closes(_normalized_closes(data))
    cash_map = _cash_map(settings)
    if cash_map is None:
        raise ValueError("DTB3 cash map is required")
    _, f0_result = _run_arm(
        settings=settings,
        data=data,
        cash_map=cash_map,
        start=START,
        mode="constrained",
        arm="F0",
        fixed_amount=None,
        persist=False,
    )
    weights, observations = _measured_weights(f0_result, closes)
    component_cagrs = {
        "sleeve_60_30_10": round(_calendar_cagr(_monthly_blend(closes, SLEEVE_WEIGHTS)), 4),
        "sleeve_voo_only": round(_calendar_cagr(_monthly_blend(closes, {"VOO": 1.0})), 4),
        "satellite_raw_f0": 8.0349,
        "cash_and_sweep_dtb3": round(_cash_cagr(cash_map, f0_result.equity_curve and pd.Index(
            [point["date"] for point in f0_result.equity_curve]
        )), 4),
    }
    measured_f0_cagr = float(f0_result.return_metrics.cagr_pct)
    predicted = (
        weights["sleeve"] * component_cagrs["sleeve_60_30_10"]
        + weights["satellite"] * component_cagrs["satellite_raw_f0"]
        + weights["cash_and_sweep"] * component_cagrs["cash_and_sweep_dtb3"]
    )
    spy_cagr = 14.2213
    satellite_hurdle = (
        spy_cagr
        - weights["sleeve"] * component_cagrs["sleeve_60_30_10"]
        - weights["cash_and_sweep"] * component_cagrs["cash_and_sweep_dtb3"]
    ) / weights["satellite"]
    voo_only_hurdle = (
        spy_cagr
        - weights["sleeve"] * component_cagrs["sleeve_voo_only"]
        - weights["cash_and_sweep"] * component_cagrs["cash_and_sweep_dtb3"]
    ) / weights["satellite"]
    path = Path("data/cache/tier55b_hurdle_attribution.json")
    superseded_window_drift: dict[str, dict[str, float]] = {}
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        old_components = prior.get("component_cagrs_pct", {})
        old_hurdle = prior.get("satellite_hurdle", {})
        if "sleeve_voo_only" in old_components:
            superseded_window_drift["sleeve_voo_only"] = {
                "cagr_pct": old_components["sleeve_voo_only"],
            }
        if "voo_only_sleeve_required_satellite_cagr_pct" in old_hurdle:
            superseded_window_drift["voo_only_sleeve_required_satellite_cagr_pct"] = {
                "cagr_pct": old_hurdle["voo_only_sleeve_required_satellite_cagr_pct"],
            }
    payload = {
        "tier": "55B",
        "window": {"start": START.isoformat(), "end": END.isoformat()},
        "source": "constrained F0 rerun from scripts/tier54f_dca_ablation.py, persist=False",
        "measured_f0": {
            "cagr_pct": round(measured_f0_cagr, 4),
            "final_equity": round(float(f0_result.final_equity), 2),
            "observations": observations,
        },
        "measured_average_weights": weights,
        "component_cagrs_pct": component_cagrs,
        "identity": {
            "formula": "w_sleeve*sleeve + w_satellite*satellite + w_cash_and_sweep*cash",
            "predicted_f0_cagr_pct": round(predicted, 4),
            "measured_f0_cagr_pct": round(measured_f0_cagr, 4),
            "residual_pp": round(measured_f0_cagr - predicted, 4),
            "acceptance_tolerance_pp": 0.6,
            "acceptance_pass": abs(measured_f0_cagr - predicted) <= 0.6,
        },
        "satellite_hurdle": {
            "spy_cagr_pct": spy_cagr,
            "required_satellite_cagr_pct": round(satellite_hurdle, 4),
            "voo_only_sleeve_required_satellite_cagr_pct": round(voo_only_hurdle, 4),
        },
        "method": {
            "data_basis": "dividend-adjusted close series from data/parquet",
            "cash_basis": "historical DTB3",
            "weight_basis": "daily post-fill category marks divided by F0 equity",
            "cagr_basis": "calendar years using 365.25 days",
            "component_note": "linear CAGR identity is an attribution diagnostic; residual captures timing, flows, and compounding",
        },
    }
    if superseded_window_drift:
        payload["superseded_window_drift"] = superseded_window_drift
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(json.dumps(payload["identity"], sort_keys=True))
    print(json.dumps(payload["satellite_hurdle"], sort_keys=True))


if __name__ == "__main__":
    main()
