"""Tier 54D — ADX portability on primary vs alternative universe.

Measurement only. Reports exposure for every cell. Pre-registered Sharpe rule on alt universe.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.automation.strategy_wiring import build_strategy_for_backtest
from src.backtesting.cash_yield import series_from_macro_rows
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.persist import persist_backtest_run
from src.config import Settings, get_settings, hub_sqlite_path, parquet_dir
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.episodes import strategy_usable_start_dates
from src.strategy.indicators import adx

logger = logging.getLogger(__name__)

WINDOW_START = date(2007, 1, 11)
WINDOW_END = date(2021, 12, 31)
SUB1_END = date(2013, 12, 31)
SUB2_START = date(2014, 1, 2)
OOS_START = date(2022, 1, 3)
OOS_END = date(2025, 12, 31)

PRIMARY_RISK = ["SPY", "QQQ", "TLT", "GLD"]
PRIMARY_CASH = "SHV"
ALT_RISK = ["IWM", "EFA", "EEM", "DBC"]
ALT_CASH = "BIL"

PRIMARY_SHARPE_DELTA_REF = 0.279  # Tier 53 B-ADX minus B, primary raw


def _cash_map(settings: Settings) -> dict[str, float] | None:
    sid = str(settings.backtest.cash_yield_series_id or "").strip()
    if not sid:
        return None
    ser = series_from_macro_rows(SQLiteStore(hub_sqlite_path(settings)).get_macro_indicator(sid))
    if ser.empty:
        return None
    return {ts.date().isoformat(): float(v) for ts, v in ser.items()}


def _settings_for_universe(base: Settings, *, risk: list[str], cash: str) -> Settings:
    return base.model_copy(
        deep=True,
        update={
            "data": base.data.model_copy(update={"universe": list(risk)}),
            "strategy": base.strategy.model_copy(
                update={
                    "momentum": base.strategy.momentum.model_copy(update={"cash_symbol": cash}),
                }
            ),
        },
    )


def _inception_rows(pq: ParquetStore, symbols: list[str]) -> list[dict]:
    rows: list[dict] = []
    for sym in symbols:
        df = pq.read_ohlcv(sym)
        rows.append(
            {
                "symbol": sym,
                "first_date": None if df.empty else str(df.index.min())[:10],
                "rows": 0 if df.empty else len(df),
            }
        )
    return rows


def _load_data(settings: Settings, pq: ParquetStore) -> dict[str, pd.DataFrame]:
    strat = build_strategy_for_backtest(settings, "momentum")
    data: dict[str, pd.DataFrame] = {}
    for sym in [*strat.get_universe(), settings.regime.vix_symbol.strip().upper(), "SPY"]:
        df = pq.read_ohlcv(sym)
        if not df.empty:
            data[sym.strip().upper()] = df
    return data


def _window_start(data: dict[str, pd.DataFrame], floor: date) -> date:
    starts = strategy_usable_start_dates(data)
    if "momentum" in starts:
        return max(floor, starts["momentum"])
    return floor


def _pack(result, *, label: str) -> dict:
    m = result.return_metrics
    return {
        "label": label,
        "sizing_mode": result.sizing_mode,
        "trades": len(result.trades),
        "cagr_pct": round(m.cagr_pct, 4),
        "sharpe": None if m.sharpe_ratio is None else round(m.sharpe_ratio, 4),
        "max_dd_pct": round(m.max_drawdown_pct, 2),
        "avg_exposure_pct": round(result.avg_exposure_pct, 4),
        "max_exposure_pct": round(result.max_exposure_pct, 4),
        "turnover": round(result.turnover, 4),
    }


def _run_cell(
    eng: BacktestEngine,
    settings: Settings,
    data: dict[str, pd.DataFrame],
    cash_map: dict[str, float] | None,
    *,
    label: str,
    constrained: bool,
    disable_adx: bool,
    start: date,
    end: date,
    adx_threshold: float | None = None,
    persist: bool = True,
) -> dict:
    run_settings = settings
    if adx_threshold is not None:
        run_settings = settings.model_copy(
            deep=True,
            update={
                "strategy": settings.strategy.model_copy(
                    update={
                        "momentum": settings.strategy.momentum.model_copy(
                            update={"adx_threshold": float(adx_threshold)}
                        ),
                    }
                ),
            },
        )
    strat = build_strategy_for_backtest(run_settings, "momentum")
    cfg = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=constrained,
        cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
        cash_yield_by_date=cash_map,
        min_coverage_ratio=0.8,
        drift_band_pct=0.05 if constrained else 1.0,
        disable_adx=disable_adx,
        experiment_label=label,
    )
    result = eng.run(strat, data, start=start, end=end, config=cfg, settings=run_settings)
    if persist:
        persist_backtest_run(
            run_settings,
            strategy_name=type(strat).__name__,
            config=cfg,
            result=result,
            start=start,
            end=end,
            benchmark_symbol="SPY",
            benchmark_metrics=None,
            experiment_label=label,
        )
    row = _pack(result, label=label)
    print(
        f"{label} CAGR={row['cagr_pct']} Sharpe={row['sharpe']} "
        f"exp={row['avg_exposure_pct']} id={label}",
        flush=True,
    )
    return row


def _adx_portability_verdict(rows: list[dict]) -> dict:
    by_label = {r["label"]: r for r in rows}
    b = by_label.get("alt-B/raw")
    badx = by_label.get("alt-B-ADX/raw")
    if not b or not badx or b["sharpe"] is None or badx["sharpe"] is None:
        return {"verdict": "error", "reason": "missing alt raw cells"}
    delta_sharpe = float(badx["sharpe"]) - float(b["sharpe"])
    delta_cagr = float(badx["cagr_pct"]) - float(b["cagr_pct"])
    delta_exp_pp = (float(badx["avg_exposure_pct"]) - float(b["avg_exposure_pct"])) * 100.0
    if delta_sharpe >= PRIMARY_SHARPE_DELTA_REF / 2.0 and delta_cagr > 0:
        verdict = "confirms_structural_defect"
    elif delta_sharpe < 0 or delta_sharpe < 0.05:
        verdict = "refutes_curve_fitting"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "delta_sharpe": round(delta_sharpe, 4),
        "delta_cagr_pct": round(delta_cagr, 4),
        "delta_avg_exposure_pp": round(delta_exp_pp, 2),
        "threshold_confirms": PRIMARY_SHARPE_DELTA_REF / 2.0,
        "threshold_refutes": 0.05,
    }


def _subperiod_pass(rows: list[dict]) -> bool:
    by = {r["label"]: r for r in rows}
    ok = True
    for tag in ("2007-2013", "2014-2021"):
        b = by.get(f"alt-B/raw-{tag}")
        badx = by.get(f"alt-B-ADX/raw-{tag}")
        if not b or not badx or b["sharpe"] is None or badx["sharpe"] is None:
            return False
        if float(badx["sharpe"]) <= float(b["sharpe"]):
            ok = False
    return ok


def _adx_mechanism_tab(data: dict[str, pd.DataFrame]) -> dict:
    """Tabulate SPY ADX through 2009-Q2 and 2020-Q2 (V-bottom windows)."""
    spy = data.get("SPY")
    if spy is None or spy.empty:
        return {"error": "missing SPY"}
    frame = spy.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index).tz_localize(None)).normalize()
    closes = frame["close"].astype(float)
    highs = frame["high"].astype(float)
    lows = frame["low"].astype(float)
    adx_line, _, _ = adx(highs, lows, closes, period=14)
    adx_ser = adx_line.copy()
    adx_ser.index = pd.DatetimeIndex(pd.to_datetime(adx_ser.index).tz_localize(None)).normalize()
    windows = {
        "2009-Q2": (pd.Timestamp("2009-04-01"), pd.Timestamp("2009-06-30")),
        "2020-Q2": (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-06-30")),
    }
    out: dict[str, dict] = {}
    for name, (ws, we) in windows.items():
        mask = (adx_ser.index >= ws) & (adx_ser.index <= we)
        slice_ = adx_ser.loc[mask].dropna()
        if slice_.empty:
            out[name] = {"error": "no data"}
            continue
        min_d = slice_.idxmin()
        out[name] = {
            "min_adx": round(float(slice_.min()), 2),
            "min_adx_date": str(min_d.date()),
            "mean_adx": round(float(slice_.mean()), 2),
            "below_25_days": int((slice_ < 25.0).sum()),
            "n_days": len(slice_),
        }
    return out


def _adx_sweep_monotone(sharpe_by_threshold: list[tuple[int, float | None]]) -> bool:
    """Performance should decay as threshold rises (Sharpe non-increasing)."""
    vals = [s for _, s in sharpe_by_threshold if s is not None]
    if len(vals) < 2:
        return False
    return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def run_portability(*, persist: bool = True) -> dict:
    base = get_settings()
    pq = ParquetStore(parquet_dir(base))
    cash_map = _cash_map(base)
    eng = BacktestEngine()

    universes = {
        "primary": _settings_for_universe(base, risk=PRIMARY_RISK, cash=PRIMARY_CASH),
        "alt": _settings_for_universe(base, risk=ALT_RISK, cash=ALT_CASH),
    }

    matrix: list[dict] = []
    for uni_key, uni_settings in universes.items():
        data = _load_data(uni_settings, pq)
        ws = _window_start(data, WINDOW_START)
        for constrained in (False, True):
            mode = "constrained" if constrained else "raw"
            for disable_adx, adx_tag in ((False, "B"), (True, "B-ADX")):
                label = f"{uni_key}-{adx_tag}/{mode}"
                matrix.append(
                    _run_cell(
                        eng,
                        uni_settings,
                        data,
                        cash_map,
                        label=label,
                        constrained=constrained,
                        disable_adx=disable_adx,
                        start=ws,
                        end=WINDOW_END,
                        persist=persist,
                    )
                )

    alt_settings = universes["alt"]
    alt_data = _load_data(alt_settings, pq)
    ws = _window_start(alt_data, WINDOW_START)

    sub_rows: list[dict] = []
    for start, end, tag in (
        (ws, SUB1_END, "2007-2013"),
        (max(SUB2_START, ws), WINDOW_END, "2014-2021"),
    ):
        for disable_adx, adx_tag in ((False, "B"), (True, "B-ADX")):
            label = f"alt-{adx_tag}/raw-{tag}"
            sub_rows.append(
                _run_cell(
                    eng,
                    alt_settings,
                    alt_data,
                    cash_map,
                    label=label,
                    constrained=False,
                    disable_adx=disable_adx,
                    start=start,
                    end=end,
                    persist=persist,
                )
            )

    sweep_rows: list[dict] = []
    sharpe_pairs: list[tuple[int, float | None]] = []
    for thr in range(0, 42, 2):
        label = f"alt-ADXthr-{thr:02d}/raw"
        row = _run_cell(
            eng,
            alt_settings,
            alt_data,
            cash_map,
            label=label,
            constrained=False,
            disable_adx=False,
            start=ws,
            end=WINDOW_END,
            adx_threshold=float(thr),
            persist=False,
        )
        sweep_rows.append(row)
        sharpe_pairs.append((thr, row["sharpe"]))

    oos_rows: list[dict] = []
    oos_end = WINDOW_END
    if alt_data:
        last_dates = [df.index.max().date() for df in alt_data.values() if not df.empty]
        if last_dates:
            oos_end = min(OOS_END, max(last_dates))
    if oos_end >= OOS_START:
        for disable_adx, adx_tag in ((False, "B"), (True, "B-ADX")):
            label = f"alt-{adx_tag}/raw-OOS"
            oos_rows.append(
                _run_cell(
                    eng,
                    alt_settings,
                    alt_data,
                    cash_map,
                    label=label,
                    constrained=False,
                    disable_adx=disable_adx,
                    start=OOS_START,
                    end=oos_end,
                    persist=persist,
                )
            )

    mechanism = _adx_mechanism_tab(_load_data(universes["primary"], pq))
    verdict = _adx_portability_verdict(matrix)

    payload = {
        "window": {"start": ws.isoformat(), "end": WINDOW_END.isoformat()},
        "inception": {
            "primary": _inception_rows(pq, [*PRIMARY_RISK, PRIMARY_CASH]),
            "alt": _inception_rows(pq, [*ALT_RISK, ALT_CASH]),
        },
        "matrix": matrix,
        "subperiod_alt_raw": sub_rows,
        "subperiod_both_halves_adx_wins": _subperiod_pass(sub_rows),
        "adx_threshold_sweep_alt_raw": sweep_rows,
        "adx_sweep_monotone_sharpe": _adx_sweep_monotone(sharpe_pairs),
        "oos_alt_raw": oos_rows,
        "oos_window": {"start": OOS_START.isoformat(), "end": oos_end.isoformat()},
        "adx_mechanism_spy": mechanism,
        "portability_verdict": verdict,
        "execution_lag": {
            "status": "not_run",
            "note": "Skipped by design (Tier 54): ADX refuted on pre-registered gate plus three diagnostics.",
        },
    }
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    payload = run_portability(persist=True)
    out = Path("data/cache/tier54_adx_portability.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    v = payload["portability_verdict"]
    print(
        f"wrote {out} verdict={v.get('verdict')} dSharpe={v.get('delta_sharpe')} "
        f"dExp={v.get('delta_avg_exposure_pp')}pp",
        flush=True,
    )


if __name__ == "__main__":
    main()
