"""Read-only API routes backed by stores."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_sqlite_store
from src.config import Settings


def test_portfolio_empty_ledger(client_open: TestClient) -> None:
    r = client_open.get("/api/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["positions"] == []
    assert body["total_market_value"] == 0.0


def test_portfolio_positions_matches_nested(client_open: TestClient) -> None:
    r1 = client_open.get("/api/portfolio")
    r2 = client_open.get("/api/portfolio/positions")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r2.json() == r1.json()["positions"]


def test_performance_snapshots_and_returns(tmp_api_settings: Settings) -> None:
    mock_sqlite = MagicMock()
    mock_sqlite.get_equity_snapshots.return_value = [
        {"snapshot_date": "2026-01-01", "total_pnl": 0.0, "total_market_value": 10000.0},
        {"snapshot_date": "2026-01-02", "total_pnl": 50.0, "total_market_value": 10050.0},
    ]
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)

    def override_sqlite() -> MagicMock:
        return mock_sqlite

    app.dependency_overrides[get_sqlite_store] = override_sqlite
    with TestClient(app) as client:
        r_snap = client.get("/api/performance/snapshots")
        r_ret = client.get("/api/performance/returns")
    assert r_snap.status_code == 200
    assert len(r_snap.json()) == 2
    assert r_ret.status_code == 200
    assert "sharpe_ratio" in r_ret.json()


def test_trades_signals_executions_limit(tmp_api_settings: Settings) -> None:
    mock_sqlite = MagicMock()
    mock_sqlite.get_signals.return_value = [{"id": 1}]
    mock_sqlite.get_executions.return_value = [{"id": 2}]
    mock_sqlite.count_signals.return_value = 42
    mock_sqlite.count_executions.return_value = 99
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)

    def override_sqlite() -> MagicMock:
        return mock_sqlite

    app.dependency_overrides[get_sqlite_store] = override_sqlite
    with TestClient(app) as client:
        r1 = client.get("/api/trades/signals", params={"limit": 5})
        r2 = client.get("/api/trades/executions", params={"limit": 5})
    assert r1.status_code == 200
    mock_sqlite.get_signals.assert_called_with(limit=5, offset=0)
    b1 = r1.json()
    assert b1["items"] == [{"id": 1}]
    assert b1["total"] == 42
    assert b1["limit"] == 5
    assert b1["offset"] == 0
    b2 = r2.json()
    assert b2["items"] == [{"id": 2}]
    assert b2["total"] == 99


def test_alerts_and_backtests(tmp_api_settings: Settings) -> None:
    mock_sqlite = MagicMock()
    mock_sqlite.get_alerts.return_value = []
    mock_sqlite.get_backtest_runs.return_value = [{"id": 7, "strategy_name": "dca"}]
    mock_sqlite.count_alerts.return_value = 0
    mock_sqlite.count_backtest_runs.return_value = 12
    mock_sqlite.get_backtest_run.return_value = None
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)

    def override_sqlite() -> MagicMock:
        return mock_sqlite

    app.dependency_overrides[get_sqlite_store] = override_sqlite
    with TestClient(app) as client:
        ra = client.get("/api/alerts")
        rb = client.get("/api/backtests")
        r404 = client.get("/api/backtests/999")
    assert ra.status_code == 200
    assert ra.json()["items"] == []
    assert ra.json()["total"] == 0
    assert rb.status_code == 200
    assert rb.json()["items"][0]["id"] == 7
    assert rb.json()["total"] == 12
    assert r404.status_code == 404
    err = r404.json()
    assert err.get("error") == "Not found"


def test_backtest_get_by_id_returns_row_when_present(tmp_api_settings: Settings) -> None:
    mock_sqlite = MagicMock()
    mock_sqlite.get_backtest_run.return_value = {
        "id": 42,
        "strategy_name": "dca",
        "equity_curve_json": "[]",
        "trades_json": "[]",
    }
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)

    def override_sqlite() -> MagicMock:
        return mock_sqlite

    app.dependency_overrides[get_sqlite_store] = override_sqlite
    with TestClient(app) as client:
        r = client.get("/api/backtests/42")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == 42
    assert body["strategy_name"] == "dca"
