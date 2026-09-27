"""Portfolio valuation from tax lots and Parquet prices (read-only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import PROJECT_ROOT, get_settings
from src.dashboard.auth import require_auth
from src.dashboard.charts import (
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_SECONDARY,
    apply_hub_layout,
)
from src.dashboard.paths import hub_sqlite_path
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.portfolio.tax_lots import LotLedger
from src.reporting.attribution import compute_position_contributions
from src.reporting.portfolio_analytics import compute_portfolio_valuation, load_current_prices

if not require_auth(get_settings()):
    st.stop()

st.title("Portfolio")

POSITIONS_TABLE_COLUMN_CONFIG = {
    "Avg cost": st.column_config.NumberColumn(format="$%.2f"),
    "Current": st.column_config.NumberColumn(format="$%.2f"),
    "Market value": st.column_config.NumberColumn(format="$%.2f"),
    "Unrealized P&L": st.column_config.NumberColumn(format="$%.2f"),
    "Unrealized %": st.column_config.NumberColumn(format="%.2f%%"),
    "Qty": st.column_config.NumberColumn(format="%.4f"),
}

settings = get_settings()
parquet_dir = Path(settings.data.parquet_dir)
pq_root = parquet_dir if parquet_dir.is_absolute() else PROJECT_ROOT / parquet_dir
pq = ParquetStore(pq_root)
hub_db = hub_sqlite_path()
ledger = LotLedger(hub_db)
meta_store = SQLiteStore(hub_db)
snapshots = meta_store.get_equity_snapshots()
if snapshots:
    last = snapshots[-1]
    st.caption(
        f"Last equity snapshot (UTC date): **{last.get('date', '—')}** — "
        "stored after each trading cycle."
    )
else:
    st.caption("No equity snapshots recorded yet (run `carmel once` with lot tracking).")

open_lots = ledger.get_open_lots()
closed_lots = ledger.get_closed_lots()
symbols = sorted({lot.symbol for lot in open_lots} | {c.symbol for c in closed_lots})
prices = load_current_prices(pq, symbols) if symbols else {}
valuation = compute_portfolio_valuation(ledger, prices)
contributions = compute_position_contributions(ledger, prices)

r1 = st.columns(5)
r1[0].metric("Total market value", f"${valuation.total_market_value:,.2f}")
r1[1].metric("Total cost basis", f"${valuation.total_cost_basis:,.2f}")
r1[2].metric("Unrealized P&L", f"${valuation.total_unrealized_pnl:,.2f}")
r1[3].metric("Realized P&L", f"${valuation.total_realized_pnl:,.2f}")
r1[4].metric("Total P&L", f"${valuation.total_pnl:,.2f}")

st.caption(
    "Valuation uses FIFO tax lots in SQLite and latest closes from Parquet (no live broker calls).",
)

if not valuation.positions:
    st.info("No open lots recorded yet. Run `carmel once` after trades populate the ledger.")
else:
    rows = []
    for p in valuation.positions:
        rows.append(
            {
                "Symbol": p.symbol,
                "Lots": p.open_lots,
                "Qty": p.total_qty,
                "Avg cost": p.avg_cost_per_share,
                "Current": p.current_price,
                "Market value": p.market_value,
                "Unrealized P&L": p.unrealized_pnl,
                "Unrealized %": p.unrealized_pnl_pct * 100.0,
            },
        )
    df = pd.DataFrame(rows)
    st.subheader("Positions")
    pie_colors = [COLOR_NEUTRAL, COLOR_SECONDARY, COLOR_POSITIVE, COLOR_NEGATIVE]
    if len(valuation.positions) >= 2:
        c_pie, c_tbl = st.columns([1, 2])
        with c_pie:
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=[p.symbol for p in valuation.positions],
                        values=[p.market_value for p in valuation.positions],
                        hole=0.45,
                        marker=dict(
                            colors=[
                                pie_colors[i % len(pie_colors)]
                                for i in range(len(valuation.positions))
                            ],
                        ),
                    ),
                ],
            )
            apply_hub_layout(fig_pie, title="Allocation by market value", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c_tbl:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config=POSITIONS_TABLE_COLUMN_CONFIG,
            )
    else:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=POSITIONS_TABLE_COLUMN_CONFIG,
        )

    if any(p.unrealized_pnl != 0.0 for p in valuation.positions):
        pos_sorted = sorted(valuation.positions, key=lambda p: abs(p.unrealized_pnl), reverse=True)
        fig_bar = go.Figure(
            go.Bar(
                x=[p.unrealized_pnl for p in pos_sorted],
                y=[p.symbol for p in pos_sorted],
                orientation="h",
                marker_color=[
                    COLOR_POSITIVE if p.unrealized_pnl >= 0 else COLOR_NEGATIVE for p in pos_sorted
                ],
            ),
        )
        h_bar = max(200, 40 * len(pos_sorted))
        apply_hub_layout(fig_bar, title="Unrealized P&L by position", height=h_bar)
        fig_bar.update_layout(xaxis_title="Unrealized P&L ($)")
        st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("P&L contribution")
if not contributions:
    st.caption("No contribution data — no tax lots in the ledger.")
else:
    c_rows = [
        {
            "Symbol": c.symbol,
            "Realized P&L": c.realized_pnl,
            "Unrealized P&L": c.unrealized_pnl,
            "Total P&L": c.total_pnl,
            "Weight %": c.weight_pct,
            "Contribution %": c.contribution_pct,
        }
        for c in contributions
    ]
    st.dataframe(
        pd.DataFrame(c_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Realized P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Unrealized P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Total P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Weight %": st.column_config.NumberColumn(format="%.2f%%"),
            "Contribution %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.caption(
        "Contribution % is each symbol's total P&L divided by portfolio cost basis "
        "(open lots plus cost of closed tranches). Fully closed positions appear with weight 0%."
    )

with st.expander("Open lots (detail)"):
    if not open_lots:
        st.caption("No open lots.")
    else:
        ol = [
            {
                "Lot ID": lot.id[:8],
                "Symbol": lot.symbol,
                "Qty": lot.qty,
                "Cost/share": lot.cost_per_share,
                "Opened (UTC)": lot.opened_at.isoformat(),
            }
            for lot in open_lots
        ]
        st.dataframe(pd.DataFrame(ol), use_container_width=True, hide_index=True)

with st.expander("Closed lots (recent)"):
    closed = ledger.get_closed_lots(limit=150)
    if not closed:
        st.caption("No closed lots yet.")
    else:
        cl = [
            {
                "Lot ID": c.lot_id[:8],
                "Symbol": c.symbol,
                "Qty": c.qty,
                "Cost/share": c.cost_per_share,
                "Sell price": c.sell_price,
                "Realized P&L": c.realized_pnl,
                "Opened (UTC)": c.opened_at.isoformat(),
                "Closed (UTC)": c.closed_at.isoformat(),
            }
            for c in closed
        ]
        st.dataframe(pd.DataFrame(cl), use_container_width=True, hide_index=True)

with st.expander("Wash sales (IRS wash-sale window)"):
    _win = int(get_settings().tax.wash_sale_window_days)
    ws = ledger.detect_wash_sales(window_days=_win)
    if not ws:
        st.caption(
            f"No wash-sale patterns detected for current lots (loss sale plus replacement "
            f"within ±{_win} calendar days; configurable via tax.wash_sale_window_days)."
        )
    else:
        total_dl = sum(w.disallowed_loss for w in ws)
        st.warning(
            f"{len(ws)} potential wash sale(s) — total disallowed loss **${total_dl:,.2f}** "
            "(informational only; not tax advice)."
        )
        wrows = [
            {
                "Symbol": w.symbol,
                "Loss lot": w.closed_lot_id.split("|")[0][:8],
                "Replacement": (w.replacement_lot_id[:8] if w.replacement_lot_id else "---"),
                "Sell date": w.sell_date,
                "Buy date": w.buy_date,
                "Disallowed loss": w.disallowed_loss,
            }
            for w in ws
        ]
        st.dataframe(
            pd.DataFrame(wrows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Disallowed loss": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        st.caption(
            "If you sell at a loss and buy substantially identical shares within 30 days before "
            "or after the sale, the loss is generally disallowed and added to replacement basis "
            "(IRS wash sale rule)."
        )

with st.expander("Fundamental scores (Piotroski F-Score)"):
    frows = meta_store.get_fundamental_scores(limit=50)
    if not frows:
        st.caption(
            "No fundamental data available. F-Score applies to individual stocks; ETFs are not scored.",
        )
    else:
        disp = []
        for fr in frows:
            sc = int(fr.get("score", 0))
            if sc >= 7:
                tag = "🟢"
            elif sc >= 4:
                tag = "🟡"
            else:
                tag = "🔴"
            data = fr.get("data") if isinstance(fr.get("data"), dict) else {}
            p = data.get("profitability", "—")
            l_ = data.get("leverage", "—")
            e = data.get("efficiency", "—")
            br_s = f"P{p}/L{l_}/E{e}"
            disp.append(
                {
                    "Symbol": fr.get("symbol", ""),
                    "Period": fr.get("period", ""),
                    "Score": sc,
                    "Rating": tag,
                    "Categories (P/L/E)": br_s,
                },
            )
        st.dataframe(pd.DataFrame(disp), use_container_width=True, hide_index=True)
        st.caption("Informational only; not investment advice.")
