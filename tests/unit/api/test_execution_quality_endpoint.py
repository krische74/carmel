"""GET /api/execution-quality."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import DataConfig, Settings
from src.data.storage.sqlite_store import SQLiteStore
from src.models import OrderExecutionResult


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            cache_dir=str(tmp_path / "cache"),
            parquet_dir=str(tmp_path / "parquet"),
        ),
    )


def test_execution_quality_endpoint_returns_rows(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    store = SQLiteStore(tmp_path / "hub.db")
    eid = store.log_execution(
        "c1",
        OrderExecutionResult(
            symbol="SPY",
            submitted=True,
            order_id="o1",
            side="buy",
            qty=1.0,
            fill_price=100.0,
        ),
    )
    store.write_execution_quality(
        {
            "execution_id": eid,
            "cycle_id": "c1",
            "execution_ts": "2026-01-01T16:00:00+00:00",
            "symbol": "SPY",
            "side": "buy",
            "qty": 1.0,
            "fill_price": 100.0,
            "reference_price": 99.0,
            "slippage_bps": 101.01,
            "note": "",
        },
    )
    app = create_app(settings=s, sqlite_store=store)
    with TestClient(app) as client:
        r = client.get("/api/execution-quality?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "SPY"
