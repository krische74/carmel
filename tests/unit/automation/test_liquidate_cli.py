"""``carmel liquidate SYMBOL`` CLI (Tier 48D)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.automation.runner import _run_liquidate_cli
from src.config import RiskConfig, Settings
from src.models import OrderExecutionResult


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


def test_liquidate_submits_and_logs_execution_row(tmp_path, monkeypatch) -> None:
    broker = MagicMock()
    broker.get_position_qty.return_value = 9.4
    broker.get_account_equity.return_value = 3436.33
    broker.get_last_equity.return_value = 3400.0
    broker.get_account_id.return_value = "default"

    store = MagicMock()
    store.log_execution.return_value = 42

    om = MagicMock()
    om.close_position.return_value = OrderExecutionResult(
        symbol="BIL",
        submitted=True,
        order_id="ord-1",
        side="sell",
        qty=9.4,
        fill_price=91.57,
    )

    monkeypatch.setattr("src.automation.runner.broker_from_settings", lambda _s: broker)
    monkeypatch.setattr("src.automation.runner.SQLiteStore", lambda _p: store)
    monkeypatch.setattr(
        "src.automation.runner.hub_sqlite_path", lambda _s: tmp_path / "hub.sqlite"
    )
    monkeypatch.setattr("src.automation.runner.parquet_dir", lambda _s: tmp_path / "parquet")
    monkeypatch.setattr("src.automation.runner.OrderManager", lambda **kw: om)
    monkeypatch.setattr(
        "src.reporting.portfolio_analytics.load_current_prices",
        lambda _pq, _syms: {"BIL": 91.57},
    )
    monkeypatch.setattr(
        "src.execution.account_factory.resolve_sqlite_account_id",
        lambda _b: "default",
    )

    rc = _run_liquidate_cli(_settings(), "BIL")
    assert rc == 0
    om.close_position.assert_called_once()
    store.log_execution.assert_called_once()
    call = store.log_execution.call_args
    res = call[0][1]
    assert res.symbol == "BIL"
    assert res.submitted is True
