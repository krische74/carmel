"""Action endpoints: Alpaca required vs mocked workflow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes import actions as actions_routes
from src.automation.workflows import TradingCycleResult
from src.config import Settings
from src.data.pipeline import IngestResult
from src.models import OrderExecutionResult, Signal


def test_actions_return_503_without_alpaca_keys(tmp_api_settings: Settings) -> None:
    tmp_api_settings.alpaca_api_key = ""
    tmp_api_settings.alpaca_secret_key = ""
    app = create_app(settings=tmp_api_settings)
    with TestClient(app) as client:
        r1 = client.post("/api/actions/ingest")
        r2 = client.post("/api/actions/cycle", json={})
        r3 = client.post("/api/actions/reconcile")
    assert r1.status_code == 503
    assert r2.status_code == 503
    assert r3.status_code == 503


def test_post_ingest_success_with_mock_workflow(tmp_api_settings: Settings) -> None:
    tmp_api_settings.alpaca_api_key = "k"
    tmp_api_settings.alpaca_secret_key = "s"
    wf = MagicMock()
    wf.run_ingest_only.return_value = [
        ("SPY", IngestResult(success=True, error=None)),
    ]
    app = create_app(settings=tmp_api_settings)

    def fake_wf() -> MagicMock:
        return wf

    app.dependency_overrides[actions_routes.get_trading_workflow] = fake_wf
    with TestClient(app) as client:
        r = client.post("/api/actions/ingest")
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert r.json()["results"][0]["symbol"] == "SPY"


def test_post_cycle_returns_cycle_id(tmp_api_settings: Settings) -> None:
    from datetime import UTC, datetime

    tmp_api_settings.alpaca_api_key = "k"
    tmp_api_settings.alpaca_secret_key = "s"
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.5,
        confidence=0.9,
        rationale="test",
        timestamp=ts,
    )
    ex = OrderExecutionResult(symbol="SPY", submitted=True)
    result = TradingCycleResult(
        cycle_id="abc123",
        ingest_results=[],
        signals=[sig],
        execution_results=[ex],
    )
    wf = MagicMock()
    wf.run_cycle.return_value = result
    app = create_app(settings=tmp_api_settings)

    def fake_wf() -> MagicMock:
        return wf

    app.dependency_overrides[actions_routes.get_trading_workflow] = fake_wf
    with TestClient(app) as client:
        r = client.post("/api/actions/cycle", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["cycle_id"] == "abc123"
    assert body["signals_count"] == 1
    assert body["executions_count"] == 1


def test_post_reconcile_success_returns_reconciliation_result(tmp_api_settings: Settings) -> None:
    tmp_api_settings.alpaca_api_key = "k"
    tmp_api_settings.alpaca_secret_key = "s"
    mock_sqlite = MagicMock()
    mock_sqlite.get_executions.return_value = []
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)
    broker = MagicMock()
    broker.list_recent_orders.return_value = []
    with (
        patch.object(actions_routes, "broker_from_settings", return_value=broker),
        TestClient(app) as client,
    ):
        r = client.post("/api/actions/reconcile")
    assert r.status_code == 200
    data = r.json()
    assert "matched" in data
    assert "discrepancies" in data
    assert "entries" in data


def test_post_ingest_returns_500_on_typed_workflow_failure(tmp_api_settings: Settings) -> None:
    tmp_api_settings.alpaca_api_key = "k"
    tmp_api_settings.alpaca_secret_key = "s"
    wf = MagicMock()
    wf.run_ingest_only.side_effect = ValueError("simulated ingest failure")
    app = create_app(settings=tmp_api_settings)

    def fake_wf() -> MagicMock:
        return wf

    app.dependency_overrides[actions_routes.get_trading_workflow] = fake_wf
    with TestClient(app) as client:
        r = client.post("/api/actions/ingest")
    assert r.status_code == 500
    assert r.json()["detail"] == "Ingest failed"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/openapi.json"),
        ("get", "/docs"),
    ],
)
def test_openapi_docs_available_without_auth(
    client_open: TestClient,
    method: str,
    path: str,
) -> None:
    """OpenAPI and Swagger UI live outside /api and are not bearer-gated."""
    r = client_open.request(method, path)
    assert r.status_code == 200
