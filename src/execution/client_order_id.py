"""Broker ``client_order_id`` values for idempotent submits (Alpaca dedupes repeats)."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime


def _utc_minute_key(anchor: datetime) -> str:
    """UTC timestamp truncated to the minute (stable across retries in the same minute)."""
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    u = anchor.astimezone(UTC)
    return u.strftime("%Y-%m-%dT%H:%M") + "Z"


def sanitize_client_order_id(raw: str) -> str:
    """Keep Alpaca-safe characters only; cap length."""
    s = re.sub(r"[^A-Za-z0-9_-]", "", raw.strip())[:128]
    if not s:
        msg = "client_order_id empty after sanitization"
        raise ValueError(msg)
    return s


def build_exec_client_order_id(
    *,
    account_id: str,
    symbol: str,
    qty: float,
    side: str,
    intent: str,
    anchor: datetime,
    price_key: str | None = None,
    cycle_id: str | None = None,
) -> str:
    """Build a deterministic id for a single execution intent.

    Args:
        account_id: Hub / logical account partition.
        symbol: Ticker.
        qty: Order quantity.
        side: ``buy`` or ``sell``.
        intent: Short lane label, e.g. ``mkt``, ``lmt``, ``stp``.
        anchor: Time anchor for legacy minute-scoped keys (retries in the same chain).
        price_key: Optional disambiguator (limit or stop price string).
        cycle_id: When set, scopes dedupe to the trading cycle instead of the UTC minute
            (Tier 49C). Distinct cycles always get distinct keys; retries within one cycle
            stay stable even if the clock crosses a minute boundary.
    """
    sym = symbol.strip().upper()
    sd = side.strip().lower()
    acct = (account_id or "").strip() or "default"
    scope = (cycle_id or "").strip() or _utc_minute_key(anchor)
    qty_s = f"{float(qty):.8f}"
    extra = (price_key or "").strip()
    lane = intent.strip().lower()
    payload = f"{lane}|{acct}|{sym}|{sd}|{qty_s}|{scope}|{extra}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return sanitize_client_order_id(f"cml-{digest}")
