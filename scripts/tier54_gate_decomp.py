"""Step 0 — decompose post-54A backtest gating before the Tier 54F ablation.

Runs the real full-system baseline with each of the three post-54A gate bypasses
enabled individually, plus all three enabled together.  The bypasses are
temporary test-only ``BacktestConfig`` switches; live settings are untouched.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scripts.tier52_analysis import _cash_map, _load_data
from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.persist import persist_backtest_run
from src.config import get_settings
from src.reporting.episodes import strategy_usable_start_dates

START = date(2011, 1, 28)
END = date(2026, 8, 19)
DRIFT_BAND_PCT = 0.05

ROWS: tuple[tuple[str, dict[str, bool]], ...] = (
    ("post_54a", {}),
    ("bypass_rotation_gate", {"bypass_rotation_gate": True}),
    ("bypass_drift_gate", {"bypass_drift_gate": True}),
    ("bypass_target_refresh_gate", {"bypass_target_refresh_gate": True}),
    (
        "bypass_all_gates",
        {
            "bypass_rotation_gate": True,
            "bypass_drift_gate": True,
            "bypass_target_refresh_gate": True,
        },
    ),
)

EXPECTED_PRE_54A = {
    "constrained": {"trades": 756, "final_equity": 31_431.63, "cagr_pct": 7.65},
    "raw": {"trades": 415, "final_equity": 35_490.73, "cagr_pct": 8.50},
}


def _run_cell(
    *,
    settings,
    data,
    cash_map,
    label: str,
    constrained: bool,
    bypasses: dict[str, bool],
) -> dict:
    """Run one decomposition cell with an explicit drift-band value."""
    config = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=constrained,
        include_dca=True,
        include_cash_sweep=True,
        cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
        cash_yield_by_date=cash_map,
        min_coverage_ratio=0.8,
        drift_band_pct=DRIFT_BAND_PCT,
        experiment_label=label,
        **bypasses,
    )
    strategy = build_strategy_for_backtest(settings, "momentum")
    result = BacktestEngine().run(
        strategy,
        data,
        start=START,
        end=END,
        config=config,
        settings=settings,
    )
    persist_backtest_run(
        settings,
        strategy_name=type(strategy).__name__,
        config=config,
        result=result,
        start=START,
        end=END,
        benchmark_symbol="SPY",
        benchmark_metrics=None,
        experiment_label=label,
    )
    metrics = result.return_metrics
    row = {
        "label": label,
        "sizing_mode": result.sizing_mode,
        "trades": len(result.trades),
        "final_equity": round(result.final_equity, 2),
        "cagr_pct": round(metrics.cagr_pct, 4),
        "sharpe": None if metrics.sharpe_ratio is None else round(metrics.sharpe_ratio, 4),
        "max_dd_pct": round(metrics.max_drawdown_pct, 2),
        "avg_exposure_pct": round(result.avg_exposure_pct, 4),
        "max_exposure_pct": round(result.max_exposure_pct, 4),
        "discrete_state_changes": result.discrete_state_changes,
        "maintenance_band_breaches": result.maintenance_band_breaches,
        "maintenance_trades": result.maintenance_trades,
        "bypasses": bypasses,
    }
    print(
        f"{label} trades={row['trades']} final={row['final_equity']} "
        f"CAGR={row['cagr_pct']} Sharpe={row['sharpe']} "
        f"exp={row['avg_exposure_pct']}",
        flush=True,
    )
    return row


def run_gate_decomposition(*, persist: bool = True) -> dict:
    """Run the five-row post-54A gating decomposition."""
    settings = get_settings()
    data = _load_data(settings)
    start = max(START, strategy_usable_start_dates(data)["dca"])
    cash_map = _cash_map(settings)
    table: list[dict] = []

    for row_name, bypasses in ROWS:
        row: dict = {"row": row_name, "bypasses": bypasses}
        for constrained in (True, False):
            mode = "constrained" if constrained else "raw"
            label = f"54-step0-{row_name}-{mode}"
            cell = _run_cell(
                settings=settings,
                data=data,
                cash_map=cash_map,
                label=label,
                constrained=constrained,
                bypasses=bypasses,
            )
            row[mode] = cell
        table.append(row)

    full = next(row for row in table if row["row"] == "bypass_all_gates")
    decisive = {}
    for mode, expected in EXPECTED_PRE_54A.items():
        got = full[mode]
        decisive[mode] = {
            "matches_pre_54a": (
                got["trades"] == expected["trades"]
                and got["final_equity"] == expected["final_equity"]
                and got["cagr_pct"] == expected["cagr_pct"]
            ),
            "expected_pre_54a": expected,
            "observed_full_bypass": {
                key: got[key] for key in ("trades", "final_equity", "cagr_pct")
            },
        }

    return {
        "window": {"start": start.isoformat(), "end": END.isoformat()},
        "drift_band_pct": DRIFT_BAND_PCT,
        "table": table,
        "decisive_test": decisive,
        "all_modes_match_pre_54a": all(
            value["matches_pre_54a"] for value in decisive.values()
        ),
        "interpretation": (
            "Full bypass reproduces pre-54A figures; the post-54A delta is attributable "
            "to gating."
            if all(value["matches_pre_54a"] for value in decisive.values())
            else "Full bypass does not reproduce pre-54A figures; stop 54F and investigate a third change."
        ),
    }


def main() -> None:
    payload = run_gate_decomposition()
    out = Path("data/cache/tier54_gate_decomp.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(f"DECISIVE: {payload['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
