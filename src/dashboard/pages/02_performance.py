"""Performance summary, portfolio vs benchmark overlay, and risk charts (read-only)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import PROJECT_ROOT, get_settings
from src.dashboard.auth import require_auth
from src.dashboard.charts import (
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_SECONDARY,
    apply_hub_layout,
)
from src.dashboard.paths import hub_sqlite_path
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.portfolio.tax_lots import LotLedger
from src.reporting.attribution import compute_brinson_from_snapshots
from src.reporting.performance import compute_equity_curve, compute_performance_summary
from src.reporting.returns import (
    compute_daily_returns,
    compute_drawdown_series,
    compute_return_metrics,
    compute_rolling_sharpe,
)

settings = get_settings()
if not require_auth(settings):
    st.stop()

st.title("Performance")

parquet_dir = Path(settings.data.parquet_dir)
pq_root = parquet_dir if parquet_dir.is_absolute() else PROJECT_ROOT / parquet_dir
pq = ParquetStore(pq_root)
store = SQLiteStore(hub_sqlite_path())
executions = store.get_executions(limit=500)
summary = compute_performance_summary(executions)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total legs", summary.total_trades)
c2.metric("Buys", summary.buys)
c3.metric("Sells", summary.sells)
c4.metric("Submitted", summary.submitted_count)
c5.metric("Rejected", summary.rejected_count)

if summary.first_trade and summary.last_trade:
    st.caption(
        f"Log range (UTC): {summary.first_trade.isoformat()} — {summary.last_trade.isoformat()}",
    )

snap_rows = store.get_equity_snapshots()
choices = sorted(set(settings.data.universe) | {"SPY"})
bench_default = st.session_state.get("perf_bench_pick", "SPY")
if bench_default not in choices:
    bench_default = "SPY"

st.subheader("Portfolio vs Benchmark")
if len(snap_rows) >= 2:
    df_snap = pd.DataFrame(snap_rows)
    d0 = str(df_snap["date"].iloc[0])[:10]
    d1 = str(df_snap["date"].iloc[-1])[:10]
    cash_missing = int(df_snap["cash"].isna().sum()) if "cash" in df_snap.columns else len(df_snap)
    if cash_missing > 0:
        st.warning(
            f"{cash_missing} snapshot row(s) lack cash — equity curve is unreliable for those "
            "dates (total_market_value is positions-only). Run "
            "`carmel backfill-snapshot-cash --since 2026-05-16 --seed 3336` after a lot rebuild.",
        )
    # Equity prefers broker_equity (Tier 47C); else positions MV + cash.
    mv_cash = df_snap["total_market_value"].astype(float) + df_snap["cash"].astype(float)
    if "broker_equity" in df_snap.columns:
        be = pd.to_numeric(df_snap["broker_equity"], errors="coerce")
        equity = be.where(be.notna(), mv_cash)
    else:
        equity = mv_cash
    eq0 = (
        float(equity.iloc[0])
        if pd.notna(equity.iloc[0]) and float(equity.iloc[0]) > 0
        else float(
            "nan",
        )
    )
    if eq0 != eq0 or eq0 <= 0:  # NaN or non-positive
        st.caption("Cannot normalize portfolio: first-row equity (MV + cash) is missing or zero.")
        norm_port = equity  # unused path below may still plot raw
    else:
        norm_port = equity / eq0 * 100.0

    curve = compute_equity_curve(pq, bench_default)
    if not curve.empty:
        curve = curve.copy()
        curve["d"] = pd.to_datetime(curve["date"]).dt.normalize()
        c0 = pd.Timestamp(d0)
        c1_ts = pd.Timestamp(d1)
        curve = curve[(curve["d"] >= c0) & (curve["d"] <= c1_ts)]
    if curve.empty or len(curve) < 2:
        st.caption(f"No Parquet OHLCV for benchmark in range {d0}-{d1} (symbol {bench_default}).")
        fig_ov = go.Figure()
        fig_ov.add_trace(
            go.Scatter(
                x=df_snap["date"],
                y=norm_port,
                mode="lines",
                name="Portfolio (normalized)",
                line=dict(color=COLOR_NEUTRAL),
            ),
        )
        apply_hub_layout(fig_ov, title="Portfolio vs Benchmark", height=420)
        fig_ov.update_layout(yaxis_title="Normalized (100 = start)")
        st.plotly_chart(fig_ov, use_container_width=True)
    else:
        norm_bench = curve["close"].astype(float) / float(curve["close"].iloc[0]) * 100.0
        fig_ov = go.Figure()
        fig_ov.add_trace(
            go.Scatter(
                x=df_snap["date"],
                y=norm_port,
                mode="lines",
                name="Portfolio",
                line=dict(color=COLOR_NEUTRAL),
            ),
        )
        fig_ov.add_trace(
            go.Scatter(
                x=curve["date"],
                y=norm_bench,
                mode="lines",
                name=f"{bench_default} (normalized)",
                line=dict(color=COLOR_SECONDARY),
            ),
        )
        apply_hub_layout(fig_ov, title="Portfolio vs Benchmark", height=420)
        fig_ov.update_layout(yaxis_title="Normalized (100 = start)")
        st.plotly_chart(fig_ov, use_container_width=True)
    st.selectbox(
        "Benchmark symbol",
        choices,
        index=choices.index(bench_default) if bench_default in choices else 0,
        key="perf_bench_pick",
    )
else:
    st.caption(
        "Need at least two equity snapshots for the overlay — run a trading cycle with lot recording."
    )

daily = compute_daily_returns(snap_rows) if len(snap_rows) >= 2 else pd.Series(dtype=float)

st.subheader("Drawdown")
if len(daily) > 0:
    dd_series = compute_drawdown_series(daily)
    if not dd_series.empty:
        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=dd_series.index,
                y=dd_series * 100.0,
                mode="lines",
                name="Drawdown",
                line=dict(color=COLOR_NEGATIVE),
                fill="tozeroy",
                fillcolor="rgba(239,83,80,0.3)",
            ),
        )
        apply_hub_layout(fig_dd, height=360)
        fig_dd.update_layout(yaxis_title="Drawdown %")
        st.plotly_chart(fig_dd, use_container_width=True)
    else:
        st.caption("No drawdown series to display.")
else:
    st.caption("Insufficient return data for drawdown.")

st.subheader("Rolling 30-day Sharpe")
if len(daily) >= 30:
    rs = compute_rolling_sharpe(daily, window=30)
    if not rs.empty:
        fig_rs = go.Figure()
        fig_rs.add_trace(
            go.Scatter(
                x=rs.index,
                y=rs,
                mode="lines",
                name="Rolling Sharpe",
                line=dict(color=COLOR_NEUTRAL),
            ),
        )
        fig_rs.add_hline(y=0.0, line_dash="dash", line_color="#78909C")
        apply_hub_layout(fig_rs, title="Rolling 30-day Sharpe", height=360)
        st.plotly_chart(fig_rs, use_container_width=True)
    else:
        st.caption("Rolling Sharpe could not be computed.")
else:
    st.caption("Need at least 30 daily return observations for rolling Sharpe.")

st.subheader("Return metrics")
if len(snap_rows) < 2:
    st.caption("Insufficient data — need at least two equity snapshots.")
else:
    rm = compute_return_metrics(daily)

    def _pct(x: float, *, digits: int = 2) -> str:
        return f"{x:.{digits}f}%"

    st.caption("**Returns**")
    g1 = st.columns(4)
    g1[0].metric("Total return", _pct(rm.total_return_pct))
    g1[1].metric("CAGR", _pct(rm.cagr_pct))
    g1[2].metric(
        "Sharpe",
        f"{rm.sharpe_ratio:.2f}" if rm.sharpe_ratio is not None else "—",
    )
    g1[3].metric(
        "Sortino",
        f"{rm.sortino_ratio:.2f}" if rm.sortino_ratio is not None else "—",
    )
    st.caption("**Risk**")
    g2 = st.columns(4)
    g2[0].metric("Max drawdown", _pct(rm.max_drawdown_pct))
    g2[1].metric("Max DD duration (days)", str(rm.max_drawdown_duration_days))
    g2[2].metric("Annual volatility", _pct(rm.annual_volatility_pct))
    g2[3].metric(
        "Calmar",
        f"{rm.calmar_ratio:.2f}" if rm.calmar_ratio is not None else "—",
    )
    st.caption(
        f"From {rm.trading_days} trading days of snapshot-derived returns (risk-free rate = 0). "
        "Best day / worst day: "
        f"{rm.best_day_pct:.2f}% / {rm.worst_day_pct:.2f}%.",
    )

with st.expander("Benchmark attribution (Brinson)"):
    if len(snap_rows) < 2:
        st.caption("Need equity snapshots to anchor a period for Brinson attribution.")
    else:
        d0 = str(snap_rows[0].get("date", ""))[:10]
        d1 = str(snap_rows[-1].get("date", ""))[:10]
        try:
            start_d = date.fromisoformat(d0)
            end_d = date.fromisoformat(d1)
        except ValueError:
            st.caption("Could not parse snapshot dates for Brinson window.")
        else:
            bench = sorted(set(settings.data.universe) | {"SPY"})
            ledger = LotLedger(hub_sqlite_path())
            pq_br = ParquetStore(pq_root)
            br = compute_brinson_from_snapshots(
                ledger,
                pq_br,
                bench,
                start_date=start_d,
                end_date=end_d,
            )
            if not br.positions:
                st.caption("No overlapping portfolio/benchmark data for Brinson.")
            else:
                st.caption(
                    "Same per-symbol returns are used for portfolio and benchmark (shared "
                    "price history), so the selection column is always 0%; active return "
                    "shows up in allocation and interaction.",
                )
                rows = [
                    {
                        "Symbol": p.symbol,
                        "Port w": f"{p.portfolio_weight * 100:.2f}%",
                        "Bench w": f"{p.benchmark_weight * 100:.2f}%",
                        "Alloc": f"{p.allocation_effect * 100:.4f}%",
                        "Select": f"{p.selection_effect * 100:.4f}%",
                        "Interact": f"{p.interaction_effect * 100:.4f}%",
                    }
                    for p in br.positions
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.caption(
                    f"Totals — allocation {br.total_allocation * 100:.4f}%, "
                    f"selection {br.total_selection * 100:.4f}%, "
                    f"interaction {br.total_interaction * 100:.4f}%; "
                    f"active return {br.total_active_return * 100:.4f}% "
                    f"(period {d0} → {d1}, equal-weight benchmark).",
                )

st.caption("Portfolio P&L detail is on the Portfolio page.")
