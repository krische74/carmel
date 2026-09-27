"""Shared display helpers for dashboard tables and timestamps."""

from __future__ import annotations

from datetime import UTC, datetime


def parse_iso_timestamp(ts: object) -> datetime | None:
    """Parse SQLite/ISO timestamps to UTC ``datetime``, or ``None`` if invalid."""
    if ts is None:
        return None
    raw = str(ts).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def format_trade_side_display(side: object) -> str:
    """Format execution side for compact columns (^ Buy / v Sell)."""
    s = str(side or "").strip().lower()
    if s == "sell":
        return "v Sell"
    return "^ Buy"


def format_timestamp_short(ts: object) -> str:
    """Format ISO/SQLite timestamps for compact table display."""
    dt = parse_iso_timestamp(ts)
    if dt is None:
        return str(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# Sprint contract aliases
parse_timestamp = parse_iso_timestamp
format_side = format_trade_side_display
format_timestamp = format_timestamp_short
