"""Aggregate recent hub activity for scheduled digest notifications."""

from __future__ import annotations

import html
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class DigestSummary(BaseModel):
    """Structured summary for digest email / webhook payloads."""

    trade_count: int = 0
    alert_count: int = 0
    net_pnl: float = 0.0
    top_movers: list[tuple[str, float]] = Field(
        default_factory=list,
        description="Symbol and approximate notional change from executions in window.",
    )
    regime_summary: str = ""
    open_positions_count: int = 0
    lookback_hours: int = 168
    account_id: str | None = None


def _parse_row_ts(raw: Any) -> datetime | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        s = str(raw).strip().replace("Z", "+00:00")
        ts = datetime.fromisoformat(s)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _in_window(ts: datetime, cutoff: datetime) -> bool:
    return ts >= cutoff


def generate_digest_summary(
    sqlite_store: SQLiteStore,
    *,
    lookback_hours: int = 168,
    account_id: str | None = None,
) -> DigestSummary:
    """Aggregate recent trades, alerts, P&L, regime, and open lots for a digest."""
    lb = max(0, int(lookback_hours))
    now = datetime.now(UTC)
    cutoff = now - timedelta(hours=lb)

    execs = sqlite_store.get_executions(limit=10_000, offset=0, account_id=account_id)
    trade_count = 0
    sym_notional: dict[str, float] = {}
    for row in execs:
        ts = _parse_row_ts(row.get("timestamp"))
        if ts is None or not _in_window(ts, cutoff):
            continue
        trade_count += 1
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        qty = float(row.get("filled_qty") or row.get("qty") or 0.0)
        px = float(row.get("filled_avg_price") or 0.0)
        n = abs(qty * px)
        sym_notional[sym] = sym_notional.get(sym, 0.0) + n

    alerts = sqlite_store.get_alerts(limit=10_000, offset=0)
    alert_count = 0
    for row in alerts:
        ts = _parse_row_ts(row.get("timestamp"))
        if ts is None or not _in_window(ts, cutoff):
            continue
        alert_count += 1

    net_pnl = 0.0
    start_d = cutoff.date().isoformat()
    end_d = now.date().isoformat()
    snaps = sqlite_store.get_equity_snapshots(
        start=start_d,
        end=end_d,
        account_id=account_id,
        limit=10_000,
        offset=0,
    )
    if len(snaps) >= 2:
        first = float(snaps[0].get("total_pnl") or 0.0)
        last = float(snaps[-1].get("total_pnl") or 0.0)
        net_pnl = last - first
    elif len(snaps) == 1:
        net_pnl = float(snaps[0].get("total_pnl") or 0.0)

    top = sorted(sym_notional.items(), key=lambda x: -x[1])[:5]

    regime_summary = ""
    try:
        reg = sqlite_store.get_latest_regime()
    except (OSError, ValueError) as exc:
        logger.warning("Digest: could not load latest regime: %s", exc)
        reg = None
    if reg:
        overall = str(reg.get("overall") or "")
        vol = str(reg.get("volatility") or "")
        vix = reg.get("vix_close")
        parts = [f"Regime: {overall}".strip()]
        if vol:
            parts.append(f"Volatility: {vol}")
        if vix is not None:
            try:
                parts.append(f"VIX {float(vix):.2f}")
            except (TypeError, ValueError):
                parts.append(f"VIX {vix!s}")
        regime_summary = " | ".join(p for p in parts if p)

    open_positions_count = 0
    try:
        open_positions_count = sqlite_store.count_open_tax_lots(account_id=account_id)
    except OSError as exc:
        logger.warning("Digest: open lot count failed: %s", exc)

    return DigestSummary(
        trade_count=trade_count,
        alert_count=alert_count,
        net_pnl=net_pnl,
        top_movers=top,
        regime_summary=regime_summary,
        open_positions_count=open_positions_count,
        lookback_hours=lb,
        account_id=account_id,
    )


def render_digest_plain_text(summary: DigestSummary) -> str:
    """Render digest as plain text for email / webhook body."""
    scope = "all accounts" if summary.account_id is None else f"account {summary.account_id}"
    lines = [
        f"Carmel digest ({scope}, last {summary.lookback_hours}h)",
        f"Trades: {summary.trade_count}",
        f"Alerts: {summary.alert_count}",
        f"Equity snapshot total P&L delta: ${summary.net_pnl:,.2f}",
        f"Open tax lots: {summary.open_positions_count}",
    ]
    if summary.regime_summary:
        lines.append(summary.regime_summary)
    if summary.top_movers:
        lines.append("Top symbols by execution notional (approx.):")
        for sym, amt in summary.top_movers:
            lines.append(f"  {sym}: ${amt:,.2f}")
    return "\n".join(lines)


def render_digest_html(summary: DigestSummary) -> str:
    """Render digest as a minimal HTML fragment."""
    scope = html.escape(
        "all accounts" if summary.account_id is None else f"account {summary.account_id}"
    )
    body_lines = [
        f"<h2>Carmel digest ({scope})</h2>",
        "<ul>",
        f"<li>Trades: <b>{summary.trade_count}</b></li>",
        f"<li>Alerts: <b>{summary.alert_count}</b></li>",
        f"<li>Equity total P&amp;L delta: <b>${summary.net_pnl:,.2f}</b></li>",
        f"<li>Open tax lots: <b>{summary.open_positions_count}</b></li>",
        "</ul>",
    ]
    if summary.regime_summary:
        body_lines.append(f"<p>{html.escape(summary.regime_summary)}</p>")
    if summary.top_movers:
        items = "".join(
            f"<li>{html.escape(sym)}: ${amt:,.2f}</li>" for sym, amt in summary.top_movers
        )
        body_lines.append(f"<p>Top symbols (approx. notional)</p><ul>{items}</ul>")
    return "\n".join(body_lines)
