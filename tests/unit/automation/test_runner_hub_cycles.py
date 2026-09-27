"""Hub multi-account trading cycle entry (scheduler / overlap guard)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from src.automation import runner as runner_mod
from src.automation.runner import run_hub_trading_cycles_for_all_accounts
from src.config import DataConfig, Settings
from src.data.storage.sqlite_store import SQLiteStore
from src.portfolio.tax_lots import LotLedger


def _settings(tmp_path: object) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )


@patch("src.automation.runner.create_trading_workflow")
def test_run_hub_trading_cycles_invokes_each_broker(
    mock_wf: MagicMock,
    tmp_path: object,
) -> None:
    wf = MagicMock()
    mock_wf.return_value = wf
    settings = _settings(tmp_path)
    db = tmp_path / "hub.db"
    sqlite = SQLiteStore(db)
    ledger = LotLedger(db)
    b1 = MagicMock()
    b2 = MagicMock()
    run_hub_trading_cycles_for_all_accounts(
        settings=settings,
        brokers={"a": b1, "b": b2},
        sqlite=sqlite,
        ledger=ledger,
        trigger="signal_generation",
    )
    assert mock_wf.call_count == 2
    assert wf.run_cycle.call_count == 2


@patch("src.automation.runner.create_trading_workflow")
def test_run_hub_trading_cycles_skips_when_lock_busy(
    mock_wf: MagicMock,
    tmp_path: object,
    monkeypatch: object,
) -> None:
    held = threading.Lock()
    assert held.acquire(blocking=False) is True
    monkeypatch.setattr(runner_mod, "HUB_TRADING_CYCLE_LOCK", held)
    settings = _settings(tmp_path)
    db = tmp_path / "hub2.db"
    sqlite = SQLiteStore(db)
    ledger = LotLedger(db)
    try:
        run_hub_trading_cycles_for_all_accounts(
            settings=settings,
            brokers={"a": MagicMock()},
            sqlite=sqlite,
            ledger=ledger,
            trigger="rebalance_check",
        )
        mock_wf.assert_not_called()
    finally:
        held.release()
