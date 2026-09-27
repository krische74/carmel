"""Decision journal: signal explanations and execution traceability (read-only)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import streamlit as st

from src.ai.explainer import generate_cycle_summary
from src.config import get_settings
from src.dashboard.auth import require_auth
from src.dashboard.formatting import format_timestamp_short, parse_iso_timestamp
from src.dashboard.paths import hub_sqlite_path
from src.data.storage.sqlite_store import SQLiteStore
from src.models import OrderExecutionResult, Signal

if not require_auth(get_settings()):
    st.stop()

st.title("Decision Journal")
st.caption(
    "Plain-English explanations for each signal (template-enriched when available), "
    "with links to executions in the same cycle.",
)

store = SQLiteStore(hub_sqlite_path())
signals = store.get_signals(limit=200)
execs = store.get_executions(limit=500)

if not signals:
    st.info("No signals recorded yet. Run a trading cycle to populate this page.")
else:
    sym_opts = sorted(
        {str(s.get("symbol", "")).strip().upper() for s in signals if s.get("symbol")}
    )
    strat_opts = sorted(
        {str(s.get("strategy_name", "")).strip() for s in signals if s.get("strategy_name")},
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        sym_pick = st.selectbox("Symbol", ["All", *sym_opts])
    with f2:
        week_ago = date.today() - timedelta(days=7)
        dr = st.date_input(
            "Date range",
            value=(week_ago, date.today()),
            key="dj_dates",
        )
    with f3:
        strat_pick = st.multiselect("Strategy", strat_opts, default=strat_opts)

    start_d: date | None = None
    end_d: date | None = None
    if isinstance(dr, tuple) and len(dr) == 2:
        start_d, end_d = dr[0], dr[1]
    elif isinstance(dr, date):
        start_d = end_d = dr

    def _in_filters(row: dict[str, object]) -> bool:
        if sym_pick != "All" and str(row.get("symbol", "")).strip().upper() != sym_pick:
            return False
        sn = str(row.get("strategy_name", "")).strip()
        if strat_pick and sn not in strat_pick:
            return False
        ts = parse_iso_timestamp(row.get("timestamp"))
        if ts is not None and start_d is not None and end_d is not None:
            d = ts.date()
            if d < start_d or d > end_d:
                return False
        return True

    filtered = [s for s in signals if _in_filters(s)]

    st.subheader("Timeline")
    for row in filtered:
        ts_raw = row.get("timestamp", "")
        sym = str(row.get("symbol", ""))
        strat = str(row.get("strategy_name", ""))
        direction = str(row.get("direction", ""))
        expl = row.get("explanation") or row.get("rationale") or ""
        header = f"{ts_raw} — {strat} — {sym} — {direction}"
        with st.expander(header):
            st.markdown(expl)
            cid = str(row.get("cycle_id", ""))
            matches = [
                e
                for e in execs
                if str(e.get("cycle_id", "")) == cid
                and str(e.get("symbol", "")).upper() == sym.upper()
            ]
            if matches:
                for e in matches:
                    ok = e.get("submitted")
                    oid = e.get("order_id")
                    reason = e.get("reason")
                    if ok:
                        st.caption(f"Execution: submitted (order_id: {oid})")
                    else:
                        st.caption(f"Execution: rejected — {reason or 'unknown'}")
            else:
                st.caption("No matching execution in this window.")

    st.subheader("Trace a cycle")
    cycle_ids = sorted(
        {str(s.get("cycle_id", "")) for s in signals if s.get("cycle_id")}, reverse=True
    )
    pick_c = st.selectbox("Cycle ID", cycle_ids)
    if pick_c:
        sig_rows = [s for s in signals if str(s.get("cycle_id")) == pick_c]
        ex_rows = [e for e in execs if str(e.get("cycle_id")) == pick_c]

        st.markdown("**Signals**")
        for s in sig_rows:
            st.write(
                f"- {s.get('symbol')} {s.get('direction')} — "
                f"{(s.get('explanation') or s.get('rationale') or '')[:200]}",
            )

        st.markdown("**Executions**")
        for e in ex_rows:
            st.write(
                f"- {e.get('symbol')} {e.get('side')} submitted={e.get('submitted')} "
                f"id={e.get('order_id')}",
            )

        built_sigs: list[Signal] = []
        for s in sig_rows:
            try:
                raw_ts = str(s.get("timestamp", "")).replace("Z", "+00:00")
                ts = datetime.fromisoformat(raw_ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                built_sigs.append(
                    Signal(
                        symbol=str(s.get("symbol", "")),
                        direction=s.get("direction", "long"),  # type: ignore[arg-type]
                        weight=float(s.get("weight", 0.0)),
                        confidence=float(s.get("confidence", 0.0)),
                        rationale=str(s.get("rationale", "")),
                        timestamp=ts,
                        strategy_name=str(s.get("strategy_name"))
                        if s.get("strategy_name")
                        else None,
                    ),
                )
            except (TypeError, ValueError):
                continue

        built_ex: list[OrderExecutionResult] = []
        for e in ex_rows:
            try:
                raw_ts = str(e.get("timestamp", "")).replace("Z", "+00:00")
                ts = datetime.fromisoformat(raw_ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                built_ex.append(
                    OrderExecutionResult(
                        symbol=str(e.get("symbol", "")),
                        submitted=bool(e.get("submitted")),
                        order_id=str(e.get("order_id")) if e.get("order_id") else None,
                        reason=str(e.get("reason")) if e.get("reason") else None,
                        timestamp=ts,
                        side=str(e.get("side", "buy")),  # type: ignore[arg-type]
                    ),
                )
            except (TypeError, ValueError):
                continue

        st.markdown("**Cycle summary (reconstructed)**")
        summary = generate_cycle_summary(built_sigs, built_ex, None, None)
        st.code(summary)

st.caption(
    f"SQLite: {hub_sqlite_path()} — timestamps: {format_timestamp_short(datetime.now(UTC))} UTC"
)
