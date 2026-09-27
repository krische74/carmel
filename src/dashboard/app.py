"""Carmel Streamlit home — equity overview, activity, and navigation."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alpaca.common.exceptions import APIError

from src.automation.health import read_heartbeat
from src.config import get_settings
from src.dashboard.auth import require_auth
from src.dashboard.charts import COLOR_NEUTRAL, apply_hub_layout
from src.dashboard.paths import account_symbols, heartbeat_path, hub_sqlite_path
from src.data.storage.sqlite_store import SQLiteStore
from src.execution.alpaca_adapter import AlpacaBrokerAdapter
from src.portfolio.state import snapshot_from_broker
from src.reporting.returns import compute_daily_returns, compute_return_metrics

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Carmel", layout="wide")
settings = get_settings()
if not require_auth(settings):
    st.stop()
store = SQLiteStore(hub_sqlite_path())
all_snap_rows = store.get_equity_snapshots()


def _filter_snapshots_by_range(
    rows: list[dict[str, Any]], range_label: str
) -> list[dict[str, Any]]:
    if range_label == "All" or not rows:
        return list(rows)
    today = date.today()
    if range_label == "30d":
        start = today - timedelta(days=30)
    elif range_label == "90d":
        start = today - timedelta(days=90)
    elif range_label == "1y":
        start = today - timedelta(days=365)
    else:
        return list(rows)
    start_s = start.isoformat()
    end_s = today.isoformat()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = str(r.get("date", ""))[:10]
        if len(d) >= 10 and start_s <= d <= end_s:
            out.append(r)
    return out


def _render_trader_heartbeat_sidebar(hb: datetime | None, *, interval_seconds: int) -> None:
    """Show trader/scheduler status from heartbeat file age (dead-man / liveness)."""
    now = datetime.now(UTC)
    if hb is None:
        st.markdown("**Trader:** :gray[Not started]")
        st.caption("No heartbeat file — start the daemon with `carmel run` or Docker.")
        return
    hb_utc = hb.replace(tzinfo=UTC) if hb.tzinfo is None else hb.astimezone(UTC)
    age = now - hb_utc
    detail = f"Last heartbeat (UTC): {hb_utc.isoformat()}"
    fresh = timedelta(seconds=2 * max(1, interval_seconds))
    if age < fresh:
        st.markdown("**Trader:** :green[Active]")
        st.caption(detail)
    elif age < timedelta(minutes=10):
        st.warning("Trader heartbeat is stale. The scheduler may have stopped.")
        st.markdown("**Trader:** :orange[Stale]")
        st.caption(detail)
    else:
        st.error(
            f"Trader appears down. Last heartbeat: {hb_utc.isoformat()}. "
            "Restart with `carmel run` or `docker compose up`.",
        )
        st.markdown("**Trader:** :red[Down]")


st.title("Carmel")
st.caption(f"{settings.app.name}")

st.subheader("Portfolio equity")
range_choice = "All"
snap_for_chart: list[dict[str, Any]] = []
if all_snap_rows:
    range_choice = st.radio(
        "Time range",
        ("30d", "90d", "1y", "All"),
        horizontal=True,
    )
    snap_for_chart = _filter_snapshots_by_range(all_snap_rows, range_choice)
if not all_snap_rows:
    st.info("No equity data yet. Run `carmel once` to record your first snapshot.")
elif snap_for_chart:
    df_pe = pd.DataFrame(snap_for_chart)
    if "cash" in df_pe.columns:
        mv_cash = df_pe["total_market_value"].astype(float) + df_pe["cash"].astype(float)
        if "broker_equity" in df_pe.columns:
            be = pd.to_numeric(df_pe["broker_equity"], errors="coerce")
            equity = be.where(be.notna(), mv_cash)
            series_name = "Equity (broker when known)"
        else:
            equity = mv_cash
            series_name = "Equity (MV + cash)"
        y_vals = equity
        if int(df_pe["cash"].isna().sum()) > 0:
            st.warning(
                "Some snapshots lack cash — chart mixes equity with positions-only MV on those dates.",
            )
    else:
        y_vals = df_pe["total_market_value"]
        series_name = "Total market value"
    fig_pe = go.Figure()
    fig_pe.add_trace(
        go.Scatter(
            x=df_pe["date"],
            y=y_vals,
            mode="lines",
            name=series_name,
            line=dict(color=COLOR_NEUTRAL),
        ),
    )
    apply_hub_layout(fig_pe, title="Equity curve", height=420)
    st.plotly_chart(fig_pe, use_container_width=True)
    st.caption(
        f"Range: **{range_choice}** — daily snapshots from SQLite "
        "(equity = total_market_value + cash when cash is known).",
    )
else:
    st.caption(
        f"No snapshots in the selected range ({range_choice}). Try **All** or a wider window."
    )

st.subheader("Market conditions")
_reg = store.get_latest_regime()
if not _reg:
    st.info(
        "Regime data unavailable — run data ingest or configure **FRED_API_KEY** for yield curve series.",
    )
else:
    _yc = str(_reg.get("yield_curve", ""))
    _vol = str(_reg.get("volatility", ""))
    _ov = str(_reg.get("overall", ""))
    _vix = _reg.get("vix_close")
    _spr = _reg.get("yield_spread")
    _mult = float(_reg.get("sizing_multiplier", 1.0))
    _pill = {
        "risk_on": ":green[Risk-on]",
        "cautious": ":orange[Cautious]",
        "defensive": ":orange[Defensive]",
        "crisis": ":red[Crisis]",
    }.get(_ov, _ov)
    mc1, mc2, mc3 = st.columns(3)
    spr_s = f"{float(_spr):.2f}%" if _spr is not None else "—"
    with mc1:
        st.markdown("**Yield curve**")
        st.caption(f"Spread (10Y-2Y): **{spr_s}** — `{_yc}`")
    with mc2:
        st.markdown("**VIX**")
        vix_s = f"{float(_vix):.2f}" if _vix is not None else "—"
        st.caption(f"Close: **{vix_s}** — `{_vol}`")
    with mc3:
        st.markdown("**Overall**")
        st.markdown(_pill)
        st.caption(f"Sizing multiplier: **{_mult * 100:.0f}%**")

# Metrics row — prefer equity (MV + cash) when cash is known
latest_mv: float | None = None
prev_mv: float | None = None
if all_snap_rows:
    last_row = all_snap_rows[-1]
    last_mv = float(last_row.get("total_market_value", 0.0))
    last_cash = last_row.get("cash")
    latest_mv = last_mv + float(last_cash) if last_cash is not None else last_mv
    if len(all_snap_rows) >= 2:
        prev_row = all_snap_rows[-2]
        prev_only = float(prev_row.get("total_market_value", 0.0))
        prev_cash = prev_row.get("cash")
        prev_mv = prev_only + float(prev_cash) if prev_cash is not None else prev_only

key = (settings.alpaca_api_key or "").strip()
sec = (settings.alpaca_secret_key or "").strip()
broker_snap = None
if key and sec:
    try:
        broker = AlpacaBrokerAdapter.create(key, sec, paper=settings.broker.paper_trading)
        broker_snap = snapshot_from_broker(broker, account_symbols())
    except (APIError, ConnectionError, TimeoutError, OSError, ValueError) as exc:
        logger.exception("Broker snapshot failed for dashboard home")
        st.warning(f"Could not load broker snapshot: {exc}")

broker_equity: float | None = None
if latest_mv is None and broker_snap is not None:
    broker_equity = float(broker_snap.equity)

display_equity = latest_mv if latest_mv is not None else broker_equity

day_delta: float | None = None
if latest_mv is not None and prev_mv is not None:
    day_delta = latest_mv - prev_mv

ret_pct_s: str = "---"
sharpe_s: str = "---"
rows_for_returns = snap_for_chart if len(snap_for_chart) >= 2 else all_snap_rows
if len(rows_for_returns) >= 2:
    daily = compute_daily_returns(rows_for_returns)
    rm = compute_return_metrics(daily)
    ret_pct_s = f"{rm.total_return_pct:.2f}%"
    sharpe_s = f"{rm.sharpe_ratio:.2f}" if rm.sharpe_ratio is not None else "—"

strategies_unique = len({str(x).strip().lower() for x in settings.strategy.enabled})

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "Total equity",
    f"${display_equity:,.2f}" if display_equity is not None else "—",
)
m2.metric(
    "Day change",
    f"${day_delta:+,.2f}" if day_delta is not None else "—",
)
m3.metric("Total return %", ret_pct_s)
m4.metric("Sharpe", sharpe_s)
m5.metric("Strategies active", str(strategies_unique))

st.subheader("Recent activity")
c_left, c_right = st.columns(2)
with c_left:
    st.markdown("**Recent trades**")
    execs = store.get_executions(limit=5)
    if execs:
        tdf = pd.DataFrame(
            [
                {
                    "Symbol": e.get("symbol", ""),
                    "Side": e.get("side", ""),
                    "Time (UTC)": e.get("timestamp", ""),
                    "Submitted": bool(e.get("submitted")),
                }
                for e in execs
            ],
        )
        st.dataframe(tdf, use_container_width=True, hide_index=True)
    else:
        st.caption("No executions recorded yet.")

with c_right:
    st.markdown("**Recent alerts**")
    alerts = store.get_alerts(limit=3)
    if alerts:
        for a in alerts:
            lvl = str(a.get("level", "")).upper()
            msg = str(a.get("message", ""))
            st.markdown(f"**{lvl}** — {msg}")
    else:
        st.caption("No alerts recorded yet.")

with st.sidebar:
    st.subheader("System")
    st.caption(f"Version {settings.app.version}")
    hb = read_heartbeat(heartbeat_path())
    _render_trader_heartbeat_sidebar(
        hb, interval_seconds=settings.scheduler.heartbeat_interval_seconds
    )
    st.markdown(f"**Environment:** `{settings.app.environment}`")
    st.caption(
        "**Enabled strategies:** " + ", ".join(str(x) for x in settings.strategy.enabled),
    )
    st.caption(f"**Sizing method:** `{settings.risk.sizing_method}`")

    if key and sec:
        st.divider()
        st.subheader("Broker")
        if broker_snap is not None:
            snap = broker_snap
            non_zero = sum(1 for q in snap.positions.values() if q > 0.0)
            b1, b2, b3 = st.columns(3)
            b1.metric("Equity", f"${snap.equity:,.2f}")
            b2.metric("Cash", f"${snap.cash:,.2f}")
            b3.metric("Open positions", non_zero)
            st.caption(
                "Daily P&L % uses (equity - last_equity) / last_equity from Alpaca when available."
            )
        else:
            st.caption("Broker metrics unavailable — fix keys or network, then reload.")
    else:
        st.caption(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in `.env` for live broker metrics in the sidebar."
        )
