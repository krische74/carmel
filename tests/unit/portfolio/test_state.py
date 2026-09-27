"""Tests for portfolio snapshot helpers."""

from unittest.mock import MagicMock

import pytest

from src.portfolio.state import PortfolioSnapshot, snapshot_from_broker


def test_snapshot_from_broker_collects_equity_cash_and_positions() -> None:
    broker = MagicMock()
    broker.get_account_equity.return_value = 12_500.0
    broker.get_cash.return_value = 2_000.0
    broker.get_last_equity.return_value = 12_500.0
    broker.get_position_qty.side_effect = lambda s: {"SPY": 10.0, "QQQ": 0.0}.get(
        s.strip().upper(), 0.0
    )

    snap = snapshot_from_broker(broker, ["SPY", "QQQ", "TLT"])

    assert isinstance(snap, PortfolioSnapshot)
    assert snap.equity == 12_500.0
    assert snap.cash == 2_000.0
    assert snap.positions == {"SPY": 10.0, "QQQ": 0.0, "TLT": 0.0}
    assert snap.daily_pnl_pct == 0.0
    broker.get_account_equity.assert_called_once()
    broker.get_cash.assert_called_once()
    broker.get_last_equity.assert_called_once()


def test_snapshot_from_broker_accepts_daily_pnl_override() -> None:
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.return_value = 0.0

    snap = snapshot_from_broker(broker, ["SPY"], daily_pnl_pct=-0.01)

    assert snap.daily_pnl_pct == -0.01
    broker.get_last_equity.assert_not_called()


def test_snapshot_computes_daily_pnl_from_last_equity() -> None:
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_200.0
    broker.get_cash.return_value = 5_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_position_qty.return_value = 0.0

    snap = snapshot_from_broker(broker, ["SPY"])

    assert snap.daily_pnl_pct == pytest.approx(0.02)


def test_snapshot_zero_last_equity_returns_zero_pnl() -> None:
    broker = MagicMock()
    broker.get_account_equity.return_value = 5_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_last_equity.return_value = 0.0
    broker.get_position_qty.return_value = 0.0

    snap = snapshot_from_broker(broker, ["SPY"])

    assert snap.daily_pnl_pct == 0.0


def test_snapshot_override_pnl_skips_last_equity() -> None:
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_cash.return_value = 10_000.0
    broker.get_position_qty.return_value = 0.0

    snap = snapshot_from_broker(broker, ["SPY"], daily_pnl_pct=0.05)

    assert snap.daily_pnl_pct == pytest.approx(0.05)
    broker.get_last_equity.assert_not_called()


def test_snapshot_from_broker_uses_one_get_all_positions_call() -> None:
    broker = MagicMock()
    broker.get_account_equity.return_value = 12_500.0
    broker.get_cash.return_value = 2_000.0
    broker.get_last_equity.return_value = 12_500.0
    broker.get_all_positions.return_value = {"SPY": 10.0, "QQQ": 2.0, "BIL": 15.0}

    snap = snapshot_from_broker(broker, ["SPY", "QQQ", "TLT"])

    assert snap.positions["SPY"] == pytest.approx(10.0)
    assert snap.positions["QQQ"] == pytest.approx(2.0)
    assert snap.positions["TLT"] == pytest.approx(0.0)
    assert snap.positions["BIL"] == pytest.approx(15.0)
    broker.get_all_positions.assert_called_once()
    broker.get_position_qty.assert_not_called()


def test_snapshot_from_broker_falls_back_when_get_all_positions_unimplemented() -> None:
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_cash.return_value = 1_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_all_positions.side_effect = NotImplementedError
    broker.get_position_qty.side_effect = lambda s: {"SPY": 3.0}.get(s.strip().upper(), 0.0)

    snap = snapshot_from_broker(broker, ["SPY", "QQQ"])

    assert snap.positions == {"SPY": 3.0, "QQQ": 0.0}
    assert broker.get_position_qty.call_count == 2
