"""Recent signals and executions from the trade log (read-only)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.dashboard.auth import require_auth
from src.dashboard.formatting import format_trade_side_display, parse_iso_timestamp
from src.dashboard.paths import hub_sqlite_path
from src.data.storage.sqlite_store import SQLiteStore

if not require_auth(get_settings()):
    st.stop()

st.title("Trade log")

store = SQLiteStore(hub_sqlite_path())
execs = store.get_executions(limit=500)
sigs = store.get_signals(limit=500)

sym_set: set[str] = set()
for e in execs:
    sym_set.add(str(e.get("symbol", "")).strip().upper())
for s in sigs:
    sym_set.add(str(s.get("symbol", "")).strip().upper())
all_syms = sorted(sym for sym in sym_set if sym)
sym_opts = ["All", *all_syms]

df_e = pd.DataFrame(execs) if execs else pd.DataFrame()
times = [parse_iso_timestamp(x) for x in df_e.get("timestamp", [])] if not df_e.empty else []
valid_times = [t for t in times if t is not None]
if valid_times:
    dmin = min(valid_times).date()
    dmax = max(valid_times).date()
else:
    dmin = dmax = date.today()

c1, c2, c3 = st.columns(3)
fsym = c1.selectbox("Symbol", sym_opts)
dr = c2.date_input("Date range (executions)", value=(dmin, dmax))
if isinstance(dr, tuple) and len(dr) == 2:
    d_lo, d_hi = dr[0], dr[1]
else:
    d_lo = d_hi = dr if isinstance(dr, date) else dmin
status = c3.radio("Status", ["All", "Submitted", "Rejected"], horizontal=True)

st.subheader("Executions")
if df_e.empty:
    st.caption("No executions recorded yet.")
else:

    def _keep_row(row: pd.Series) -> bool:
        if fsym != "All" and str(row.get("symbol", "")).upper() != fsym:
            return False
        ts = parse_iso_timestamp(row.get("timestamp"))
        if ts is not None:
            td = ts.date()
            if td < d_lo or td > d_hi:
                return False
        sub = bool(row.get("submitted"))
        if status == "Submitted" and not sub:
            return False
        return not (status == "Rejected" and sub)

    filt = df_e.loc[df_e.apply(_keep_row, axis=1)].copy()
    if filt.empty:
        st.caption("No executions match the current filters.")
    else:
        filt["Side"] = (
            filt["side"].map(format_trade_side_display) if "side" in filt.columns else ""
        )
        want = [
            "timestamp",
            "symbol",
            "Side",
            "submitted",
            "qty",
            "filled_avg_price",
            "order_id",
            "reason",
            "cycle_id",
        ]
        disp = filt[[c for c in want if c in filt.columns]]
        if "submitted" in disp.columns:
            disp = disp.copy()
            disp["submitted"] = disp["submitted"].astype(bool)
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "qty": st.column_config.NumberColumn(format="%.4f"),
                "filled_avg_price": st.column_config.NumberColumn(format="$%.2f"),
                "Side": st.column_config.TextColumn("Side"),
            },
        )

with st.expander("Execution quality (slippage vs daily close)"):
    st.caption(
        "Reference price is the last Parquet daily close on or before the execution "
        "date (UTC). Positive bps = adverse vs that close — not live bid/ask microstructure.",
    )
    qrows = store.get_execution_quality_rows(limit=100)
    if not qrows:
        st.caption("No execution quality rows yet (recorded after fills on the next cycle).")
    else:
        df_q = pd.DataFrame(qrows)
        st.dataframe(df_q, use_container_width=True, hide_index=True)

st.subheader("Signals")
if not sigs:
    st.caption("No signals recorded yet.")
else:
    df_s = pd.DataFrame(sigs)
    if fsym != "All" and "symbol" in df_s.columns:
        df_s = df_s[df_s["symbol"].str.upper() == fsym]
    if df_s.empty:
        st.caption("No signals match the symbol filter.")
    else:
        st.dataframe(
            df_s,
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Experimental ML scores (latest cycle)", expanded=False):
    ml_rows = store.get_latest_ml_scores_batch()
    if ml_rows:
        st.caption(
            "Classifier scores from the most recent trading cycle (see `ml.enabled` and "
            "`carmel ml-train`). Diagnostic only; execution is unchanged.",
        )
        df_m = pd.DataFrame(ml_rows)
        if "top_features_json" in df_m.columns:
            df_m = df_m.rename(columns={"top_features_json": "top_features"})
        st.dataframe(df_m, use_container_width=True, hide_index=True)
    else:
        st.caption(
            "No ML scores in SQLite yet. Enable `ml.enabled`, train a model, then run a cycle.",
        )
