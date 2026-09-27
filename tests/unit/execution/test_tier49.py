"""Tier 49B — fill price/qty capture and lot ledger semantics."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.config import RiskConfig, Settings
from src.execution.order_manager import OrderManager
from src.portfolio.tax_lots import LotLedger
from src.risk.kill_switch import KillSwitch


def _settings() -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        risk=RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            daily_loss_limit_pct=0.03,
        ),
    )


def test_close_position_captures_partial_fill_qty_and_price() -> None:
    broker = MagicMock()
    broker.submit_market_order.return_value = "ord-1"
    broker.get_order_status.return_value = {
        "status": "filled",
        "filled_qty": 6.0,
        "filled_avg_price": 110.25,
    }
    broker.get_order_fill_price.return_value = 110.25
    mgr = OrderManager(broker=broker, settings=_settings(), kill_switch=KillSwitch(0.03))
    res = mgr.close_position(
        "SHV",
        7.791,
        price=110.25,
        equity=3400.0,
        daily_pnl_pct=0.0,
    )
    assert res.submitted is True
    assert res.filled_qty == pytest.approx(6.0)
    assert res.fill_price == pytest.approx(110.25)
    assert res.qty == pytest.approx(7.791)


def test_lot_ledger_records_filled_qty_not_requested_on_partial_sell(tmp_path) -> None:
    ledger = LotLedger(tmp_path / "hub.sqlite")
    ledger.record_buy("SHV", 10.0, 100.0, datetime(2026, 8, 1, tzinfo=UTC))
    ledger.record_sell(
        "SHV",
        6.0,
        110.25,
        datetime(2026, 8, 19, 17, 30, tzinfo=UTC),
        method="fifo",
        account_id="default",
    )
    open_lots = ledger.get_open_lots(account_id="default")
    assert sum(lot.qty for lot in open_lots) == pytest.approx(4.0)
