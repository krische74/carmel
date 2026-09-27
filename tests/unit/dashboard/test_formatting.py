"""Pure formatting helpers used by dashboard tables."""

from __future__ import annotations

from datetime import UTC, datetime

from src.dashboard.formatting import (
    format_side,
    format_timestamp,
    format_timestamp_short,
    format_trade_side_display,
    parse_iso_timestamp,
    parse_timestamp,
)


def test_parse_iso_timestamp_various_formats() -> None:
    assert parse_iso_timestamp("2026-04-07T12:30:00+00:00") == datetime(
        2026,
        4,
        7,
        12,
        30,
        0,
        tzinfo=UTC,
    )
    assert parse_iso_timestamp("2026-04-07T12:30:00Z") is not None
    naive = parse_iso_timestamp("2026-04-07T12:30:00")
    assert naive is not None
    assert naive.tzinfo == UTC


def test_parse_iso_timestamp_invalid() -> None:
    assert parse_iso_timestamp(None) is None
    assert parse_iso_timestamp("not-a-date") is None


def test_parse_timestamp_alias_matches_parse_iso() -> None:
    ts = "2026-01-15T10:00:00+00:00"
    assert parse_timestamp(ts) == parse_iso_timestamp(ts)


def test_format_timestamp_short_valid() -> None:
    s = format_timestamp_short("2026-03-01T08:09:10+00:00")
    assert s == "2026-03-01 08:09:10"


def test_format_timestamp_short_invalid_falls_back_to_string() -> None:
    assert format_timestamp_short("garbage") == "garbage"


def test_format_timestamp_alias() -> None:
    assert format_timestamp("2026-02-02T00:00:00+00:00") == "2026-02-02 00:00:00"


def test_format_trade_side_display_buy_vs_sell() -> None:
    assert "^" in format_trade_side_display("buy")
    assert "Buy" in format_trade_side_display("BUY")
    assert "v" in format_trade_side_display("sell")
    assert "Sell" in format_trade_side_display("SELL")


def test_format_side_alias() -> None:
    assert format_side("buy") == format_trade_side_display("buy")
