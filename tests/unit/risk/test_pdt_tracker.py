"""Tests for PDT day-trade counting and risk evaluation."""

from datetime import UTC, datetime

from src.risk.pdt_tracker import (
    PDTCheckResult,
    count_day_trades,
    evaluate_pdt_risk,
    projected_day_trades_after_sell,
    sell_adds_day_trade,
)
from src.risk.pre_trade_checks import evaluate_pdt_before_sell


def _row(
    *,
    ts: str,
    symbol: str,
    side: str,
    submitted: int = 1,
    order_status: str = "filled",
) -> dict:
    return {
        "timestamp": ts,
        "symbol": symbol,
        "side": side,
        "submitted": submitted,
        "order_status": order_status,
    }


def test_count_day_trades_same_day_buy_sell() -> None:
    d = "2026-04-06T15:00:00+00:00"  # Monday
    ex = [
        _row(ts=d, symbol="SPY", side="buy"),
        _row(ts=d, symbol="SPY", side="sell"),
    ]
    as_of = datetime(2026, 4, 6, 20, 0, tzinfo=UTC)
    assert count_day_trades(ex, as_of=as_of, lookback_days=5) == 1


def test_count_day_trades_multiple_symbols_same_day() -> None:
    """Each symbol-day round trip counts separately (e.g. SPY + QQQ same day = 2)."""
    d = "2026-04-06T15:00:00+00:00"
    ex = [
        _row(ts=d, symbol="SPY", side="buy"),
        _row(ts=d, symbol="SPY", side="sell"),
        _row(ts=d, symbol="QQQ", side="buy"),
        _row(ts=d, symbol="QQQ", side="sell"),
    ]
    as_of = datetime(2026, 4, 6, 20, 0, tzinfo=UTC)
    assert count_day_trades(ex, as_of=as_of, lookback_days=5) == 2


def test_count_day_trades_across_days_not_counted() -> None:
    buy = "2026-04-06T15:00:00+00:00"  # Mon
    sell = "2026-04-07T15:00:00+00:00"  # Tue
    ex = [
        _row(ts=buy, symbol="SPY", side="buy"),
        _row(ts=sell, symbol="SPY", side="sell"),
    ]
    as_of = datetime(2026, 4, 8, 20, 0, tzinfo=UTC)
    assert count_day_trades(ex, as_of=as_of, lookback_days=5) == 0


def test_pdt_blocks_at_threshold() -> None:
    r = evaluate_pdt_risk(4, 10_000.0, pdt_threshold=4, equity_floor=25_000.0)
    assert isinstance(r, PDTCheckResult)
    assert r.allowed is False
    assert r.day_trade_count == 4


def test_pdt_allows_above_equity_floor() -> None:
    r = evaluate_pdt_risk(10, 100_000.0, pdt_threshold=4, equity_floor=25_000.0)
    assert r.allowed is True
    assert r.warning is None


def test_pdt_warns_near_threshold() -> None:
    r = evaluate_pdt_risk(3, 10_000.0, pdt_threshold=4, equity_floor=25_000.0)
    assert r.allowed is True
    assert r.warning is not None
    assert "PDT" in r.warning or "day trade" in r.warning.lower()


def test_sell_adds_day_trade_when_unmatched_buy_today() -> None:
    d = "2026-04-06T14:00:00+00:00"
    ex = [_row(ts=d, symbol="SPY", side="buy")]
    as_of = datetime(2026, 4, 6, 16, 0, tzinfo=UTC)
    assert sell_adds_day_trade(ex, symbol="SPY", as_of=as_of) is True


def test_sell_does_not_add_after_balanced_day() -> None:
    d = "2026-04-06T14:00:00+00:00"
    ex = [
        _row(ts=d, symbol="SPY", side="buy"),
        _row(ts=d, symbol="SPY", side="sell"),
    ]
    as_of = datetime(2026, 4, 6, 16, 0, tzinfo=UTC)
    assert sell_adds_day_trade(ex, symbol="SPY", as_of=as_of) is False


def test_projected_day_trades_includes_pending_round_trip() -> None:
    """Two prior day trades (other days) + today's buy → sell projects +1."""
    ex = [
        _row(ts="2026-04-01T15:00:00+00:00", symbol="AAA", side="buy"),
        _row(ts="2026-04-01T15:30:00+00:00", symbol="AAA", side="sell"),
        _row(ts="2026-04-02T15:00:00+00:00", symbol="BBB", side="buy"),
        _row(ts="2026-04-02T15:30:00+00:00", symbol="BBB", side="sell"),
        _row(ts="2026-04-06T10:00:00+00:00", symbol="SPY", side="buy"),
    ]
    as_of = datetime(2026, 4, 6, 16, 0, tzinfo=UTC)
    assert count_day_trades(ex, as_of=as_of, lookback_days=5) == 2
    assert projected_day_trades_after_sell(ex, symbol="SPY", as_of=as_of, lookback_days=5) == 3


def test_pre_trade_pdt_disabled_always_ok() -> None:
    r = evaluate_pdt_before_sell(
        symbol="SPY",
        equity=5_000.0,
        as_of=datetime(2026, 4, 6, 12, 0, tzinfo=UTC),
        executions=[],
        pdt_protection=False,
    )
    assert r.ok is True
    assert r.reason is None
