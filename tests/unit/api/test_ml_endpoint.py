"""GET /api/ml/latest."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_sqlite_store
from src.config import Settings


def test_ml_latest_returns_batch(tmp_api_settings: Settings) -> None:
    mock_sqlite = MagicMock()
    mock_sqlite.get_latest_ml_scores_batch.return_value = [
        {
            "id": 1,
            "cycle_id": "c1",
            "symbol": "SPY",
            "score": 0.55,
            "model_version": "1",
            "computed_at": "2026-04-01T12:00:00+00:00",
            "top_features_json": "SHAP: ret_1: +0.02",
        },
    ]
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)
    app.dependency_overrides[get_sqlite_store] = lambda: mock_sqlite
    with TestClient(app) as client:
        r = client.get("/api/ml/latest")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["symbol"] == "SPY"
    assert body[0]["score"] == pytest.approx(0.55)
    assert body[0]["top_features"] == "SHAP: ret_1: +0.02"


def test_ml_latest_empty_list(tmp_api_settings: Settings) -> None:
    mock_sqlite = MagicMock()
    mock_sqlite.get_latest_ml_scores_batch.return_value = []
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)
    app.dependency_overrides[get_sqlite_store] = lambda: mock_sqlite
    with TestClient(app) as client:
        r = client.get("/api/ml/latest")
    assert r.status_code == 200
    assert r.json() == []


def test_ml_latest_skips_nan_and_inf_scores(tmp_api_settings: Settings) -> None:
    mock_sqlite = MagicMock()
    mock_sqlite.get_latest_ml_scores_batch.return_value = [
        {
            "id": 1,
            "cycle_id": "c1",
            "symbol": "SPY",
            "score": float("nan"),
            "model_version": "1",
            "computed_at": "2026-04-01T12:00:00+00:00",
            "top_features_json": None,
        },
        {
            "id": 2,
            "cycle_id": "c1",
            "symbol": "QQQ",
            "score": 0.7,
            "model_version": "1",
            "computed_at": "2026-04-01T12:00:00+00:00",
            "top_features_json": None,
        },
    ]
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)
    app.dependency_overrides[get_sqlite_store] = lambda: mock_sqlite
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/api/ml/latest")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["symbol"] == "QQQ"


def test_ml_latest_storage_error_returns_500(tmp_api_settings: Settings) -> None:
    mock_sqlite = MagicMock()
    mock_sqlite.get_latest_ml_scores_batch.side_effect = OSError("disk io")
    app = create_app(settings=tmp_api_settings, sqlite_store=mock_sqlite)
    app.dependency_overrides[get_sqlite_store] = lambda: mock_sqlite
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/api/ml/latest")
    assert r.status_code == 500
