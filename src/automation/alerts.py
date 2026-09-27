"""Rule-based operational alerts after a trading cycle (no external delivery)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.automation.health import read_heartbeat

if TYPE_CHECKING:
    from src.data.regime import MarketRegime


class AlertLevel(StrEnum):
    """Severity for dashboard coloring and operator triage."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    """One evaluated alert (typically persisted to SQLite)."""

    timestamp: datetime
    level: AlertLevel
    category: str
    message: str


def evaluate_cycle_alerts(
    *,
    kill_switch_halted: bool,
    daily_pnl_pct: float,
    daily_loss_limit_pct: float,
    ingest_failures: list[str],
    rejected_orders: list[tuple[str, str | None]],
    market_regime: MarketRegime | None = None,
) -> list[Alert]:
    """Evaluate alert rules for one completed trading cycle.

    Rules are evaluated in a fixed order; multiple alerts may be returned.

    Args:
        rejected_orders: Each entry is ``(symbol, reason)`` where ``reason`` is the
            persisted :attr:`~src.models.OrderExecutionResult.reason` when known.
    """
    now = datetime.now(UTC)
    out: list[Alert] = []

    if kill_switch_halted:
        out.append(
            Alert(
                timestamp=now,
                level=AlertLevel.CRITICAL,
                category="kill_switch",
                message="Kill switch is active: trading halt or daily loss limit breached.",
            ),
        )

    pnl = float(daily_pnl_pct)
    limit = float(daily_loss_limit_pct)
    if not kill_switch_halted and pnl < 0 and abs(pnl) >= limit * 0.8:
        out.append(
            Alert(
                timestamp=now,
                level=AlertLevel.WARNING,
                category="daily_loss_warning",
                message=(f"Daily P&L ({pnl:.2%}) is within 80% of the loss limit ({limit:.2%})."),
            ),
        )

    if ingest_failures:
        syms = ", ".join(sorted({s.strip().upper() for s in ingest_failures}))
        out.append(
            Alert(
                timestamp=now,
                level=AlertLevel.WARNING,
                category="ingest_failure",
                message=f"OHLCV ingest failed for: {syms}",
            ),
        )

    if rejected_orders:
        parts: list[str] = []
        for sym, reason in sorted(rejected_orders, key=lambda t: t[0].strip().upper()):
            su = sym.strip().upper()
            if reason and str(reason).strip():
                parts.append(f"{su}: {str(reason).strip()}")
            else:
                parts.append(su)
        detail = ", ".join(parts)
        out.append(
            Alert(
                timestamp=now,
                level=AlertLevel.INFO,
                category="order_rejected",
                message=f"Orders not submitted for: {detail}",
            ),
        )

    if market_regime is not None:
        from src.data.regime import OverallRegime

        if market_regime.overall in (OverallRegime.DEFENSIVE, OverallRegime.CRISIS):
            vix_s = (
                f"{market_regime.vix_close:.2f}" if market_regime.vix_close is not None else "—"
            )
            sp_s = (
                f"{market_regime.yield_spread:.2f}"
                if market_regime.yield_spread is not None
                else "—"
            )
            lvl = (
                AlertLevel.CRITICAL
                if market_regime.overall == OverallRegime.CRISIS
                else AlertLevel.WARNING
            )
            mult_pct = float(market_regime.sizing_multiplier) * 100.0
            out.append(
                Alert(
                    timestamp=now,
                    level=lvl,
                    category="market_regime",
                    message=(
                        f"Market regime: {market_regime.overall.value} "
                        f"(VIX: {vix_s}, yield spread: {sp_s}). "
                        f"Position sizing reduced to {mult_pct:.0f}%."
                    ),
                ),
            )

    return out


def check_heartbeat_staleness(
    heartbeat_path: Path,
    max_age_seconds: int = 300,
) -> Alert | None:
    """Return a CRITICAL alert when the heartbeat file is missing or older than ``max_age_seconds``."""
    now = datetime.now(UTC)
    hb = read_heartbeat(heartbeat_path)
    if hb is None:
        return Alert(
            timestamp=now,
            level=AlertLevel.CRITICAL,
            category="heartbeat_stale",
            message="Heartbeat file is missing or unreadable.",
        )
    hb_utc = hb.replace(tzinfo=UTC) if hb.tzinfo is None else hb.astimezone(UTC)
    age = (now - hb_utc).total_seconds()
    if age > float(max_age_seconds):
        return Alert(
            timestamp=now,
            level=AlertLevel.CRITICAL,
            category="heartbeat_stale",
            message=(
                f"Heartbeat is stale (age {age:.0f}s, max {max_age_seconds}s). "
                f"Last beat (UTC): {hb_utc.isoformat()}."
            ),
        )
    return None
