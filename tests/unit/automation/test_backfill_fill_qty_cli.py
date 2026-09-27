"""CLI wiring for ``carmel backfill-fill-qty``."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.automation import runner as runner_mod
from src.config import DataConfig, Settings
from src.data.storage.sqlite_store import SQLiteStore
from src.execution.errors import OrderNotFoundError
from src.models import OrderExecutionResult

if TYPE_CHECKING:
    from pathlib import Path

BIL_ORDER = "e93fe3f7-c8e2-48e3-b56c-cdc4b8cc4917"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
        ),
        alpaca_api_key="test_key",
        alpaca_secret_key="test_secret",
    )


def _seed_null(store: SQLiteStore) -> None:
    store.log_execution(
        "c-null",
        OrderExecutionResult(
            symbol="BIL",
            submitted=True,
            order_id=BIL_ORDER,
            side="buy",
            qty=15.533477,
            fill_price=None,
            filled_qty=None,
            timestamp=datetime(2026, 8, 10, 17, 30, 14, tzinfo=UTC),
            order_status="filled",
        ),
    )


def _broker_with_fill(
    *, filled_qty: float | None, price: float | None, missing: bool = False
) -> MagicMock:
    broker = MagicMock()

    def get_order_fill(order_id: str) -> tuple[float | None, float | None]:
        if missing:
            raise OrderNotFoundError(order_id)
        return filled_qty, price

    broker.get_order_fill.side_effect = get_order_fill
    return broker


def test_backfill_fill_qty_cli_dry_run_does_not_backup_or_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "hub_metadata.sqlite"
    store = SQLiteStore(db)
    _seed_null(store)
    monkeypatch.setattr(runner_mod, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(runner_mod, "hub_sqlite_path", lambda _settings: db)

    def _boom(_path: Path) -> Path:
        raise AssertionError("dry-run must not back up sqlite")

    monkeypatch.setattr(runner_mod, "_backup_hub_sqlite", _boom)
    monkeypatch.setattr(
        runner_mod,
        "broker_from_settings",
        lambda _settings: _broker_with_fill(filled_qty=6.133477, price=91.55),
    )

    code = runner_mod.main(
        ["backfill-fill-qty", "--since", "2026-05-16", "--dry-run", "--account-id", "default"],
    )
    assert code == 0
    row = store.get_executions(limit=5)[0]
    assert row["filled_qty"] is None
    assert float(row["qty"]) == pytest.approx(15.533477)
    out = capsys.readouterr().out
    assert "6.133477" in out
    assert BIL_ORDER in out


def test_backfill_fill_qty_cli_missing_order_leaves_null_and_backs_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db = tmp_path / "hub_metadata.sqlite"
    store = SQLiteStore(db)
    _seed_null(store)
    backups: list[Path] = []
    monkeypatch.setattr(runner_mod, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(runner_mod, "hub_sqlite_path", lambda _settings: db)
    monkeypatch.setattr(
        runner_mod, "_backup_hub_sqlite", lambda path: backups.append(path) or path
    )
    monkeypatch.setattr(
        runner_mod,
        "broker_from_settings",
        lambda _settings: _broker_with_fill(filled_qty=None, price=None, missing=True),
    )

    code = runner_mod.main(["backfill-fill-qty", "--since", "2026-05-16"])
    assert code == 0
    assert backups == [db]
    row = store.get_executions(limit=5)[0]
    assert row["filled_qty"] is None
    assert row["filled_avg_price"] is None
    assert float(row["qty"]) == pytest.approx(15.533477)
    assert row["side"] == "buy"
    assert row["symbol"] == "BIL"
    assert int(row["submitted"]) == 1
