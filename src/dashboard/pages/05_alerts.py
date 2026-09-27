"""Operational alerts from SQLite (read-only)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.config import get_settings
from src.dashboard.auth import require_auth
from src.dashboard.formatting import parse_iso_timestamp
from src.dashboard.paths import hub_sqlite_path
from src.data.storage.sqlite_store import SQLiteStore

if not require_auth(get_settings()):
    st.stop()


def _fmt_ts_short(ts: object) -> str:
    dt = parse_iso_timestamp(ts)
    if dt is None:
        return str(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


st.title("Alerts")

store = SQLiteStore(hub_sqlite_path())
rows = store.get_alerts(limit=100)

if not rows:
    st.caption("No alerts recorded yet.")
else:
    df = pd.DataFrame(rows)
    valid_times = [parse_iso_timestamp(x) for x in df.get("timestamp", [])]
    valid_times = [t for t in valid_times if t is not None]
    if valid_times:
        dmin = min(valid_times).date()
        dmax = max(valid_times).date()
    else:
        dmin = dmax = date.today()

    levels = st.multiselect(
        "Filter by level",
        ["critical", "warning", "info"],
        default=["critical", "warning", "info"],
    )
    dr = st.date_input("Date range", value=(dmin, dmax))
    if isinstance(dr, tuple) and len(dr) == 2:
        d_lo, d_hi = dr[0], dr[1]
    else:
        d_lo = d_hi = dr if isinstance(dr, date) else dmin

    df_f = df.copy()
    if "level" in df_f.columns and levels:
        want = {x.lower() for x in levels}
        df_f = df_f[df_f["level"].str.lower().isin(want)]

    def _in_date(row: pd.Series) -> bool:
        ts = parse_iso_timestamp(row.get("timestamp"))
        if ts is None:
            return True
        td = ts.date()
        return d_lo <= td <= d_hi

    df_f = df_f.loc[df_f.apply(_in_date, axis=1)]

    crit = (
        int((df_f["level"].str.lower() == "critical").sum())
        if "level" in df_f.columns and not df_f.empty
        else 0
    )
    warn = (
        int((df_f["level"].str.lower() == "warning").sum())
        if "level" in df_f.columns and not df_f.empty
        else 0
    )
    info = (
        int((df_f["level"].str.lower() == "info").sum())
        if "level" in df_f.columns and not df_f.empty
        else 0
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Critical",
        crit,
        delta="active" if crit > 0 else None,
        delta_color="inverse",
    )
    m2.metric(
        "Warning",
        warn,
        delta="review" if warn > 0 else None,
        delta_color="off",
    )
    m3.metric("Info", info)

    if df_f.empty:
        st.caption("No alerts match the current filters.")
    else:
        show = df_f.copy()
        if "timestamp" in show.columns:
            show["timestamp"] = show["timestamp"].map(_fmt_ts_short)
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "level": st.column_config.TextColumn(
                    "Level",
                    help="critical = red flag; warning = review; info = FYI",
                ),
                "category": st.column_config.TextColumn("Category"),
                "message": st.column_config.TextColumn("Message", width="large"),
            },
        )
