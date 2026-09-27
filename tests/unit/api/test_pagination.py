"""API offset pagination metadata on list endpoints."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import DataConfig, Settings
from src.data.storage.sqlite_store import SQLiteStore
from src.models import OrderExecutionResult, Signal


@pytest.fixture()
def paginated_app(
    tmp_path: Path,
) -> Generator[tuple[TestClient, SQLiteStore], None, None]:
    db = tmp_path / "hub.sqlite"
    store = SQLiteStore(db)
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            cache_dir=str(tmp_path / "cache"),
            parquet_dir=str(tmp_path / "parquet"),
        ),
    )
    app = create_app(settings=settings, sqlite_store=store)
    with TestClient(app) as client:
        yield client, store


def test_api_trades_pagination_metadata(paginated_app: tuple[TestClient, SQLiteStore]) -> None:
    client, store = paginated_app
    from datetime import UTC, datetime

    for i in range(5):
        store.log_signal(
            "c1",
            Signal(
                symbol="SPY",
                direction="long",
                weight=0.1,
                confidence=1.0,
                rationale=f"r{i}",
                timestamp=datetime(2026, 1, i + 1, tzinfo=UTC),
                strategy_name="S",
            ),
        )
    r = client.get("/api/trades/signals", params={"limit": 2, "offset": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 5
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert len(body["items"]) == 2


def test_api_executions_offset_beyond_total_returns_empty_items(
    paginated_app: tuple[TestClient, SQLiteStore],
) -> None:
    client, store = paginated_app
    store.log_execution(
        "c",
        OrderExecutionResult(symbol="X", submitted=True, side="buy"),
    )
    r = client.get("/api/trades/executions", params={"limit": 10, "offset": 9999})
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 1
    assert body["offset"] == 9999


def test_api_pagination_defaults_offset_zero(
    paginated_app: tuple[TestClient, SQLiteStore],
) -> None:
    client, _store = paginated_app
    r = client.get("/api/alerts")
    assert r.status_code == 200
    assert r.json()["offset"] == 0
    assert r.json()["limit"] == 50
