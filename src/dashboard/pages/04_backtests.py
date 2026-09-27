"""Historical backtest runs stored in SQLite (read-only)."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_settings
from src.dashboard.auth import require_auth
from src.dashboard.charts import COLOR_NEUTRAL, apply_hub_layout
from src.dashboard.formatting import format_trade_side_display
from src.dashboard.paths import hub_sqlite_path
from src.data.storage.sqlite_store import SQLiteStore

if not require_auth(get_settings()):
    st.stop()

st.title("Backtests")

store = SQLiteStore(hub_sqlite_path())
runs = store.get_backtest_runs(limit=100)

if not runs:
    st.info("No backtest runs stored yet. Run `carmel backtest …` from the CLI.")
else:
    df = pd.DataFrame(runs)
    st.subheader("Recent runs")
    show = df[
        [
            "id",
            "run_at",
            "strategy_name",
            "start_date",
            "end_date",
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "trade_count",
        ]
    ]
    st.dataframe(show, use_container_width=True, hide_index=True)

    ids = [int(r["id"]) for r in runs]
    compare = st.multiselect(
        "Compare runs",
        ids,
        format_func=lambda i: f"Run #{i}",
        max_selections=3,
    )
    if len(compare) in (2, 3):
        fig_cmp = go.Figure()
        tbl: dict[str, list[str]] = {}
        for rid in compare:
            row = store.get_backtest_run(int(rid))
            if not row:
                continue
            try:
                curve = json.loads(str(row["equity_curve_json"]))
            except json.JSONDecodeError:
                continue
            ic = float(row["initial_capital"])
            if curve and "equity" in curve[0]:
                dates = [c.get("date") for c in curve]
                norm = [float(c["equity"]) / ic * 100.0 for c in curve]
                label = f"Run #{rid} — {row['strategy_name']}"
                fig_cmp.add_trace(
                    go.Scatter(
                        x=dates,
                        y=norm,
                        mode="lines",
                        name=label,
                    ),
                )
            col_key = f"Run #{rid}"
            tbl[col_key] = [
                f"{float(row['total_return_pct']):.2f}%",
                f"{float(row['sharpe_ratio']):.2f}"
                if row.get("sharpe_ratio") is not None
                else "—",
                f"{float(row['max_drawdown_pct']):.2f}%",
                str(int(row["trade_count"])),
                f"{float(row['slippage_bps']):.1f}",
            ]
        apply_hub_layout(fig_cmp, title="Normalized equity (100 = starting capital)", height=420)
        st.plotly_chart(fig_cmp, use_container_width=True)
        cmp_df = pd.DataFrame(
            tbl,
            index=[
                "Total return %",
                "Sharpe",
                "Max drawdown %",
                "Trade count",
                "Slippage bps",
            ],
        )
        st.dataframe(cmp_df, use_container_width=True)

    pick = st.selectbox("View run", ids, format_func=lambda i: f"Run #{i}")
    row = store.get_backtest_run(int(pick))
    if row:
        st.subheader(f"Run #{row['id']} — {row['strategy_name']}")
        g1 = st.columns(4)
        g1[0].metric("Total return %", f"{float(row['total_return_pct']):.2f}")
        g1[1].metric("CAGR %", f"{float(row['cagr_pct']):.2f}")
        g1[2].metric(
            "Sharpe",
            f"{float(row['sharpe_ratio']):.2f}" if row.get("sharpe_ratio") is not None else "—",
        )
        g1[3].metric("Max DD %", f"{float(row['max_drawdown_pct']):.2f}")
        g2 = st.columns(4)
        g2[0].metric("Final equity", f"${float(row['final_equity']):,.2f}")
        g2[1].metric("Trades", str(int(row["trade_count"])))
        g2[2].metric("Slippage bps", f"{float(row['slippage_bps']):.1f}")
        g2[3].metric("Rebalance", str(row["rebalance_frequency"]))

        try:
            curve = json.loads(str(row["equity_curve_json"]))
        except json.JSONDecodeError:
            st.caption("Could not parse stored equity curve.")
            curve = []
        if curve:
            cdf = pd.DataFrame(curve)
            if "date" in cdf.columns and "equity" in cdf.columns:
                fig_eq = go.Figure()
                fig_eq.add_trace(
                    go.Scatter(
                        x=cdf["date"],
                        y=cdf["equity"],
                        mode="lines",
                        name="Equity",
                        line=dict(color=COLOR_NEUTRAL),
                    ),
                )
                apply_hub_layout(fig_eq, title="Equity curve", height=420)
                st.plotly_chart(fig_eq, use_container_width=True)

        tc = int(row["trade_count"])
        with st.expander(f"Trades ({tc})"):
            raw_t = row.get("trades_json")
            if raw_t is None or str(raw_t).strip() in ("", "null"):
                st.caption("Trade details not available for runs saved before this feature.")
            else:
                try:
                    trades = json.loads(str(raw_t))
                except json.JSONDecodeError:
                    st.caption("Could not parse stored trades.")
                else:
                    if not trades:
                        st.caption("No trades in this run.")
                    else:
                        trows = [
                            {
                                "Date": t.get("date", ""),
                                "Symbol": t.get("symbol", ""),
                                "Side": format_trade_side_display(t.get("side", "")),
                                "Qty": t.get("qty"),
                                "Price": t.get("price"),
                                "Slippage cost": t.get("slippage_cost"),
                            }
                            for t in trades
                        ]
                        st.dataframe(
                            pd.DataFrame(trows),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Price": st.column_config.NumberColumn(format="$%.2f"),
                                "Slippage cost": st.column_config.NumberColumn(format="$%.2f"),
                                "Qty": st.column_config.NumberColumn(format="%.4f"),
                            },
                        )

st.caption(f"SQLite: {hub_sqlite_path()}")
