"""Tier 54F — factorial DCA sleeve ablation.

Measurement only.  Momentum and cash-sweep settings remain unchanged.  The script
calibrates fixed-dollar arms to their paired percent-of-equity arm and records every
DCA fill in ``data/cache/tier54f_contributions.csv``.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from scripts.tier52_analysis import _load_data
from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.cash_yield import lookup_cash_yield_annual_pct, series_from_macro_rows
from src.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    _daily_returns_from_equity_curve,
    compute_benchmark_metrics,
)
from src.backtesting.persist import persist_backtest_run
from src.config import get_settings, hub_sqlite_path
from src.data.regime import build_market_regime_snapshot
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.episodes import strategy_usable_start_dates
from src.reporting.returns import compute_return_metrics

logger = logging.getLogger(__name__)

START = date(2011, 1, 28)
END = date(2026, 8, 19)
LATCH_START = date(2020, 5, 26)
LATCH_END = date(2021, 2, 12)
SPLIT_1_END = date(2021, 12, 31)
TARGETS = ("VOO", "VXUS", "BND")
STAGGER_WEEKS = range(5)
SPY_REFERENCE_CAGR_PCT = 12.27
BLEND_REFERENCE_CAGR_PCT = 9.80
MATERIALITY_FLOOR_PP = 0.10
PREREGISTERED_RULES_VERBATIM = """
**Pre-stated test:** re-run each pairwise comparison at **5 staggered start offsets** —
2011-01-28 plus 0, 1, 2, 3, 4 weeks. **If the sign of the CAGR delta is not identical across all
five offsets, the effect is NOT ESTABLISHED**, whatever its mean magnitude. Report the mean and the
full range.

**0.10 pp of CAGR.** Both |F1-F0| and |F2-F0| below 0.10 pp, both sign-stable → **"the DCA
sleeve's sizing rules are immaterial at this scale."** Neither goes to the design conversation as
a priority. This is a permitted and useful outcome.

- The multiplier is the **dominant** DCA distortion only if **|F1-F0| >= 2 x |F2-F0|**.
- If **|F2-F0| >= |F1-F0|**, the premise of the original 54F is **SUPERSEDED**: the budget basis
  is the larger distortion, it goes to the design conversation first, and the supersede is written
  into `BUILD_STATE.md` struck-through per guardrail 4.
- Anything between — **INCONCLUSIVE**. Report as inconclusive. Do not pick the larger and call it
  the finding.

Both arms must pass the sign-stability test in 3.2 before this rule is applied. If either fails, the
verdict is **NOT ESTABLISHED** and the dominance rule is not evaluated at all.
"""


def _cash_map(settings) -> dict[str, float] | None:
    """Load DTB3 as an ISO-date annual-percent map."""
    sid = str(settings.backtest.cash_yield_series_id or "").strip()
    if not sid:
        return None
    series = series_from_macro_rows(
        SQLiteStore(hub_sqlite_path(settings)).get_macro_indicator(sid)
    )
    if series.empty:
        return None
    return {ts.date().isoformat(): float(value) for ts, value in series.items()}


def _dca_trade_rows(result) -> list[dict]:
    """Aggregate DCA fills by date and target."""
    totals: dict[tuple[str, str], float] = {}
    for trade in result.trades:
        symbol = trade.symbol.strip().upper()
        if symbol in TARGETS and trade.side == "buy":
            key = (trade.date, symbol)
            totals[key] = totals.get(key, 0.0) + float(trade.qty) * float(trade.price)
    return [
        {"date": day, "target": target, "dollars": round(dollars, 8)}
        for (day, target), dollars in sorted(totals.items())
    ]


def _gross_contributions(result) -> float:
    """Return gross dollars deployed into configured DCA targets."""
    return sum(row["dollars"] for row in _dca_trade_rows(result))


def _cycle_count(result) -> int:
    """Count DCA dates with at least one non-zero DCA fill."""
    return len({row["date"] for row in _dca_trade_rows(result)})


def _units(result) -> dict[str, float]:
    """Return total units accumulated per DCA target."""
    out = {target: 0.0 for target in TARGETS}
    for trade in result.trades:
        symbol = trade.symbol.strip().upper()
        if symbol in out and trade.side == "buy":
            out[symbol] += float(trade.qty)
    return {symbol: round(value, 8) for symbol, value in out.items()}


def _curve_metrics(curve: list[dict], start: date, end: date) -> dict[str, float | None]:
    """Compute return metrics for an inclusive equity-curve slice."""
    sliced = [
        point
        for point in curve
        if start.isoformat() <= str(point["date"]) <= end.isoformat()
    ]
    metrics = compute_return_metrics(_daily_returns_from_equity_curve(sliced))
    return {
        "cagr_pct": round(metrics.cagr_pct, 4),
        "sharpe": None if metrics.sharpe_ratio is None else round(metrics.sharpe_ratio, 4),
        "max_dd_pct": round(metrics.max_drawdown_pct, 2),
        "final_equity": round(float(sliced[-1]["equity"]), 2) if sliced else None,
    }


def _benchmark_headlines(settings, data, cash_map, start: date, end: date) -> dict:
    """Return benchmark metrics shown beside each arm headline."""
    days = sorted(
        {
            ts.date()
            for symbol in ("SPY",)
            if symbol in data
            for ts in pd.DatetimeIndex(data[symbol].index)
            if start <= ts.date() <= end
        }
    )
    rf_daily = None
    if cash_map:
        rf_daily = pd.Series(
            {
                day: (float(value) / 100.0) / 252.0
                for day, value in cash_map.items()
                if start.isoformat() <= day <= end.isoformat()
            },
            dtype=float,
        )
    spy = compute_benchmark_metrics(
        data,
        "SPY",
        days,
        risk_free_rate_annual=float(settings.backtest.cash_yield_annual_pct),
        risk_free_daily=rf_daily,
    )
    result = {
        "spy": {
            "cagr_pct": SPY_REFERENCE_CAGR_PCT,
            "sharpe": None if spy is None else spy.sharpe_ratio,
            "max_dd_pct": None if spy is None else spy.max_drawdown_pct,
        },
        "blend_75_25": {
            "cagr_pct": BLEND_REFERENCE_CAGR_PCT,
            "sharpe": None,
            "max_dd_pct": None,
        },
    }
    return result


def _regime_for_cycle(settings, data, day: str):
    """Build the point-in-time regime used for a contribution-log row."""
    vix = data.get(settings.regime.vix_symbol.strip().upper())
    vix_close = None
    if vix is not None and not vix.empty:
        frame = vix.copy()
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index).tz_localize(None)).normalize()
        prior = frame.loc[frame.index <= pd.Timestamp(day)]
        if not prior.empty and "close" in prior:
            vix_close = float(prior["close"].iloc[-1])
    return build_market_regime_snapshot(
        settings,
        as_of=datetime.fromisoformat(day).replace(tzinfo=UTC),
        vix_close=vix_close,
        yield_spread=None,
    )


def _append_contribution_log(
    rows: list[dict],
    *,
    result,
    settings,
    data,
    arm: str,
    mode: str,
    start: date,
    fixed_amount: float | None,
) -> None:
    """Append inspectable per-target DCA fill rows."""
    by_key = {(row["date"], row["target"]): row["dollars"] for row in _dca_trade_rows(result)}
    dates = sorted({day for day, _ in by_key})
    for day in dates:
        regime = _regime_for_cycle(settings, data, day)
        multiplier = (
            1.0
            if arm in {"F1", "F3"}
            else float(regime.sizing_multiplier)
        )
        basis = "fixed_dollar" if fixed_amount is not None else "percent_of_equity"
        for target in TARGETS:
            rows.append(
                {
                    "date": day,
                    "target": target,
                    "dollars": round(by_key.get((day, target), 0.0), 8),
                    "regime": regime.overall.value,
                    "multiplier": round(multiplier, 6),
                    "budget_basis": basis,
                    "arm": arm,
                    "sizing_mode": mode,
                    "start": start.isoformat(),
                }
            )


def _run_arm(
    *,
    settings,
    data,
    cash_map,
    start: date,
    mode: str,
    arm: str,
    fixed_amount: float | None,
    persist: bool,
) -> tuple[dict, object]:
    """Run one factorial arm and return its headline plus raw result."""
    constrained = mode == "constrained"
    label = f"54F-{arm}-{mode}-start-{start.isoformat()}"
    config = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=constrained,
        include_dca=True,
        include_cash_sweep=True,
        cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
        cash_yield_by_date=cash_map,
        min_coverage_ratio=0.8,
        drift_band_pct=0.05,
        pin_dca_regime_multiplier=1.0 if arm in {"F1", "F3"} else None,
        dca_fixed_amount_usd=fixed_amount,
        experiment_label=label,
    )
    strategy = build_strategy_for_backtest(settings, "momentum")
    result = BacktestEngine().run(
        strategy,
        data,
        start=start,
        end=END,
        config=config,
        settings=settings,
    )
    if persist:
        persist_backtest_run(
            settings,
            strategy_name=type(strategy).__name__,
            config=config,
            result=result,
            start=start,
            end=END,
            benchmark_symbol="SPY",
            benchmark_metrics=None,
            experiment_label=label,
        )
    metrics = result.return_metrics
    row = {
        "arm": arm,
        "sizing_mode": mode,
        "start": start.isoformat(),
        "fixed_amount_usd": None if fixed_amount is None else round(fixed_amount, 8),
        "cagr_pct": round(metrics.cagr_pct, 4),
        "max_dd_pct": round(metrics.max_drawdown_pct, 2),
        "sharpe": None if metrics.sharpe_ratio is None else round(metrics.sharpe_ratio, 4),
        "final_equity": round(result.final_equity, 2),
        "avg_exposure_pct": round(result.avg_exposure_pct, 4),
        "max_exposure_pct": round(result.max_exposure_pct, 4),
        "total_dca_contributed": round(_gross_contributions(result), 8),
        "dca_cycle_count": _cycle_count(result),
        "dca_units": _units(result),
        "total_trades": len(result.trades),
        "discrete_state_changes": result.discrete_state_changes,
        "maintenance_band_breaches": result.maintenance_band_breaches,
        "maintenance_trades": result.maintenance_trades,
        "split_2011_2021": _curve_metrics(result.equity_curve, start, SPLIT_1_END),
        "split_2022_2026": _curve_metrics(result.equity_curve, date(2022, 1, 3), END),
    }
    return row, result


def _calibrated_amount(result) -> float:
    """Calibrate fixed dollars to average gross F0/F1 deployment."""
    cycles = _cycle_count(result)
    if cycles <= 0:
        return 0.0
    return _gross_contributions(result) / cycles


def _delta_sign_stability(rows: list[dict], *, arm: str, mode: str) -> dict:
    """Summarize the pre-registered five-start sign test."""
    deltas: list[float] = []
    starts = sorted(
        {
            row["start"]
            for row in rows
            if row["sizing_mode"] == mode and row["arm"] in {"F0", arm}
        }
    )
    for start in starts:
        pair = [row for row in rows if row["sizing_mode"] == mode and row["start"] == start]
        base = next((row for row in pair if row["arm"] == "F0"), None)
        test = next((row for row in pair if row["arm"] == arm), None)
        if base is not None and test is not None:
            deltas.append(round(float(test["cagr_pct"]) - float(base["cagr_pct"]), 4))
    signs = {1 if delta > 0 else -1 if delta < 0 else 0 for delta in deltas}
    return {
        "arm": arm,
        "sizing_mode": mode,
        "deltas_pp": deltas,
        "mean_delta_pp": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "range_pp": (
            round(max(deltas) - min(deltas), 4)
            if deltas
            else None
        ),
        "sign_stable": len(signs) == 1,
    }


def _dominance_verdict(sign_tests: list[dict]) -> dict:
    """Apply amended-contract floor and dominance rules to constrained deltas."""
    constrained = {row["arm"]: row for row in sign_tests if row["sizing_mode"] == "constrained"}
    f1 = constrained.get("F1")
    f2 = constrained.get("F2")
    if not f1 or not f2 or not f1["sign_stable"] or not f2["sign_stable"]:
        return {"verdict": "NOT ESTABLISHED", "reason": "sign instability"}
    d1 = abs(float(f1["mean_delta_pp"]))
    d2 = abs(float(f2["mean_delta_pp"]))
    if d1 < MATERIALITY_FLOOR_PP and d2 < MATERIALITY_FLOOR_PP:
        return {"verdict": "IMMATERIAL", "reason": "both effects below 0.10 pp"}
    if d1 >= 2.0 * d2:
        return {"verdict": "multiplier_dominant", "f1_abs_delta_pp": d1, "f2_abs_delta_pp": d2}
    if d2 >= d1:
        return {"verdict": "budget_basis_dominant", "f1_abs_delta_pp": d1, "f2_abs_delta_pp": d2}
    return {"verdict": "INCONCLUSIVE", "f1_abs_delta_pp": d1, "f2_abs_delta_pp": d2}


def _pct_delta(new: float, old: float) -> float | None:
    """Return percentage difference, or None when the denominator is zero."""
    if abs(old) <= 1e-12:
        return None
    return round(100.0 * (new - old) / old, 4)


def _latch_foregone_return(settings, data, cash_map, f0_result, f1_result) -> dict:
    """Compute 2020 withheld dollars' excess return over realised DTB3 cash."""
    f0 = {(row["date"], row["target"]): row["dollars"] for row in _dca_trade_rows(f0_result)}
    f1 = {(row["date"], row["target"]): row["dollars"] for row in _dca_trade_rows(f1_result)}
    terminal = pd.Timestamp(LATCH_END)
    withheld_total = 0.0
    invested_terminal = 0.0
    cash_terminal = 0.0
    target_data = {symbol: data[symbol].copy() for symbol in TARGETS if symbol in data}
    for (day, target), amount in f1.items():
        day_date = date.fromisoformat(day)
        if not LATCH_START <= day_date <= LATCH_END:
            continue
        withheld = max(0.0, amount - f0.get((day, target), 0.0))
        if withheld <= 0.0 or target not in target_data:
            continue
        prices = target_data[target]
        prices.index = pd.DatetimeIndex(pd.to_datetime(prices.index).tz_localize(None)).normalize()
        entry = prices.loc[prices.index >= pd.Timestamp(day_date), "close"]
        exit_ = prices.loc[prices.index <= terminal, "close"]
        if entry.empty or exit_.empty:
            continue
        risk_growth = float(exit_.iloc[-1] / entry.iloc[0])
        invested_terminal += withheld * risk_growth
        cash_value = withheld
        cursor = day_date
        while cursor <= LATCH_END:
            if cash_map:
                annual, _ = lookup_cash_yield_annual_pct(
                    cash_map,
                    cursor,
                    fallback_pct=float(settings.backtest.cash_yield_annual_pct),
                )
                cash_value *= 1.0 + (float(annual) / 100.0) / 252.0
            cursor += timedelta(days=1)
        cash_terminal += cash_value
        withheld_total += withheld
    return {
        "window": {"start": LATCH_START.isoformat(), "end": LATCH_END.isoformat()},
        "status": "measured" if withheld_total > 0.0 else "no_withheld_dollars_observed",
        "withheld_dollars": round(withheld_total, 8),
        "invested_terminal_value": round(invested_terminal, 8),
        "dtb3_cash_terminal_value": round(cash_terminal, 8),
        "excess_return_foregone": round(invested_terminal - cash_terminal, 8),
        "framing": "excess return on withheld DCA capital over realised DTB3 cash, not portfolio drag",
    }


def run_ablation(*, persist: bool = True) -> dict:
    """Run the calibrated 2x2 ablation, staggered sign tests, and latch audit."""
    settings = get_settings()
    data = _load_data(settings)
    starts = strategy_usable_start_dates(data)
    start = max(START, starts["dca"])
    cash_map = _cash_map(settings)
    log_rows: list[dict] = []
    headline_rows: list[dict] = []
    main_results: dict[tuple[str, str], object] = {}

    for mode in ("constrained", "raw"):
        f0_row, f0_result = _run_arm(
            settings=settings, data=data, cash_map=cash_map, start=start,
            mode=mode, arm="F0", fixed_amount=None, persist=persist,
        )
        f1_row, f1_result = _run_arm(
            settings=settings, data=data, cash_map=cash_map, start=start,
            mode=mode, arm="F1", fixed_amount=None, persist=persist,
        )
        f2_amount = _calibrated_amount(f0_result)
        f2_row, f2_result = _run_arm(
            settings=settings, data=data, cash_map=cash_map, start=start,
            mode=mode, arm="F2", fixed_amount=f2_amount, persist=persist,
        )
        f3_amount = _calibrated_amount(f1_result)
        f3_row, f3_result = _run_arm(
            settings=settings, data=data, cash_map=cash_map, start=start,
            mode=mode, arm="F3", fixed_amount=f3_amount, persist=persist,
        )
        for arm, row, result, fixed in (
            ("F0", f0_row, f0_result, None),
            ("F1", f1_row, f1_result, None),
            ("F2", f2_row, f2_result, f2_amount),
            ("F3", f3_row, f3_result, f3_amount),
        ):
            _append_contribution_log(
                log_rows,
                result=result,
                settings=settings,
                data=data,
                arm=arm,
                mode=mode,
                start=start,
                fixed_amount=fixed,
            )
            row["benchmarks"] = _benchmark_headlines(settings, data, cash_map, start, END)
            headline_rows.append(row)
            main_results[(mode, arm)] = result

    # Five-start sign tests. Offset 0 is represented by the main rows above.
    sign_rows: list[dict] = []
    for offset in STAGGER_WEEKS:
        offset_start = start + timedelta(days=7 * offset)
        for mode in ("constrained", "raw"):
            if offset == 0:
                for arm in ("F0", "F1", "F2"):
                    main = next(
                        row for row in headline_rows
                        if row["start"] == start.isoformat()
                        and row["sizing_mode"] == mode and row["arm"] == arm
                    )
                    sign_rows.append(main)
                continue
            f0_row, f0_result = _run_arm(
                settings=settings, data=data, cash_map=cash_map, start=offset_start,
                mode=mode, arm="F0", fixed_amount=None, persist=persist,
            )
            f1_row, f1_result = _run_arm(
                settings=settings, data=data, cash_map=cash_map, start=offset_start,
                mode=mode, arm="F1", fixed_amount=None, persist=persist,
            )
            f2_amount = _calibrated_amount(f0_result)
            f2_row, f2_result = _run_arm(
                settings=settings, data=data, cash_map=cash_map, start=offset_start,
                mode=mode, arm="F2", fixed_amount=f2_amount, persist=persist,
            )
            for arm, row, result, fixed in (
                ("F0", f0_row, f0_result, None),
                ("F1", f1_row, f1_result, None),
                ("F2", f2_row, f2_result, f2_amount),
            ):
                _append_contribution_log(
                    log_rows, result=result, settings=settings, data=data, arm=arm,
                    mode=mode, start=offset_start, fixed_amount=fixed,
                )
                row["benchmarks"] = _benchmark_headlines(
                    settings, data, cash_map, offset_start, END
                )
                sign_rows.append(row)

    sign_tests: list[dict] = []
    for arm in ("F1", "F2"):
        for mode in ("constrained", "raw"):
            sign_rows_for_arm = [
                row for row in sign_rows
                if row["arm"] in {"F0", arm} and row["sizing_mode"] == mode
            ]
            # Keep only one row for each arm/start; the helper consumes paired rows.
            sign_rows_for_arm = [
                row for row in sign_rows_for_arm
                if row["start"] in {
                    (start + timedelta(days=7 * offset)).isoformat()
                    for offset in STAGGER_WEEKS
                }
            ]
            sign_rows_for_arm_result = _delta_sign_stability(
                sign_rows_for_arm, arm=arm, mode=mode
            )
            sign_rows_for_arm_result["arm"] = arm
            sign_rows_for_arm_result["sizing_mode"] = mode
            sign_rows_for_arm_result["comparison"] = f"{arm}-F0"
            sign_tests.append(sign_rows_for_arm_result)

    dominance = _dominance_verdict(sign_tests)
    f0c = main_results[("constrained", "F0")]
    f1c = main_results[("constrained", "F1")]
    f2c = main_results[("constrained", "F2")]
    f3c = main_results[("constrained", "F3")]
    interaction = {
        "constrained_cagr_pp": round(
            f3c.return_metrics.cagr_pct
            - f1c.return_metrics.cagr_pct
            - f2c.return_metrics.cagr_pct
            + f0c.return_metrics.cagr_pct,
            4,
        ),
        "interpretation": "hypothesis-generating only; no pre-registered threshold",
    }
    f0c_row = next(row for row in headline_rows if row["arm"] == "F0" and row["sizing_mode"] == "constrained")
    f1c_row = next(row for row in headline_rows if row["arm"] == "F1" and row["sizing_mode"] == "constrained")
    f2c_row = next(row for row in headline_rows if row["arm"] == "F2" and row["sizing_mode"] == "constrained")
    f3c_row = next(row for row in headline_rows if row["arm"] == "F3" and row["sizing_mode"] == "constrained")
    contribution_comparisons = {
        "F1_vs_F0_pct": _pct_delta(
            f1c_row["total_dca_contributed"], f0c_row["total_dca_contributed"]
        ),
        "F2_vs_F0_pct": _pct_delta(
            f2c_row["total_dca_contributed"], f0c_row["total_dca_contributed"]
        ),
        "F3_vs_F1_pct": _pct_delta(
            f3c_row["total_dca_contributed"], f1c_row["total_dca_contributed"]
        ),
    }
    contribution_comparisons["over_2pct_confounded"] = any(
        value is not None and abs(value) > 2.0
        for key, value in contribution_comparisons.items()
        if key.endswith("_pct")
    )
    if contribution_comparisons["over_2pct_confounded"]:
        dominance = {
            "verdict": "CONFOUNDED",
            "reason": "at least one compared pair differs in total gross contribution by more than 2%",
            "unadjusted_dominance_signal": dominance,
        }
    payload = {
        "window": {"start": start.isoformat(), "end": END.isoformat()},
        "preregistered_rules": {
            "materiality_floor_pp": MATERIALITY_FLOOR_PP,
            "sign_stability_offsets": [0, 1, 2, 3, 4],
            "dominance": "F1 multiplier dominant iff |F1-F0| >= 2x|F2-F0|; "
            "F2 budget dominant iff |F2-F0| >= |F1-F0|; otherwise INCONCLUSIVE",
            "confound_limit_pct": 2.0,
        },
        "preregistered_rules_verbatim": PREREGISTERED_RULES_VERBATIM.strip(),
        "headline_rows": headline_rows,
        "sign_tests": sign_tests,
        "dominance_verdict": dominance,
        "interaction": interaction,
        "contribution_comparisons_constrained": contribution_comparisons,
        "latch_2020": _latch_foregone_return(
            settings, data, cash_map, f0c, f1c
        ),
        "latch_2009": {
            "status": "not_measurable",
            "reason": "VOO and VXUS, 90% of configured DCA weight, lack 2009 history; no proxy used.",
            "inception": {
                symbol: (
                    None
                    if symbol not in data
                    else str(pd.to_datetime(data[symbol].index).min())[:10]
                )
                for symbol in TARGETS
            },
        },
        "headline_benchmarks": {
            "spy_cagr_pct": SPY_REFERENCE_CAGR_PCT,
            "blend_75_25_cagr_pct": BLEND_REFERENCE_CAGR_PCT,
        },
    }
    return payload, log_rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    payload, log_rows = run_ablation(persist=True)
    cache = Path("data/cache")
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "tier54f_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (cache / "tier54f_contributions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "date", "target", "dollars", "regime", "multiplier", "budget_basis",
            "arm", "sizing_mode", "start",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"wrote {cache / 'tier54f_results.json'}", flush=True)
    print(f"wrote {cache / 'tier54f_contributions.csv'}", flush=True)
    print(
        f"54F dominance={payload['dominance_verdict']['verdict']} "
        f"F1_signs={[x for x in payload['sign_tests'] if x['arm'] == 'F1']} "
        f"F2_signs={[x for x in payload['sign_tests'] if x['arm'] == 'F2']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
