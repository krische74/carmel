"""POST /api/ai/ask endpoint tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import DataConfig, LLMConfig, Settings


def _settings(tmp_path: Path, *, llm_enabled: bool = True, api_token: str = "") -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            cache_dir=str(tmp_path / "cache"),
            parquet_dir=str(tmp_path / "parquet"),
        ),
        llm=LLMConfig(enabled=llm_enabled),
        api_token=api_token,
    )


def test_ask_endpoint_returns_answer(tmp_path: Path) -> None:
    s = _settings(tmp_path, llm_enabled=True)
    mock_sqlite = MagicMock()
    mock_sqlite.get_equity_snapshots.return_value = []
    mock_sqlite.get_executions.return_value = []
    mock_sqlite.get_alerts.return_value = []
    mock_sqlite.get_latest_regime.return_value = None

    app = create_app(settings=s, sqlite_store=mock_sqlite)

    with (
        patch("src.ai.llm_client.OllamaClient") as mock_cls,
        TestClient(app) as client,
    ):
        inst = mock_cls.return_value
        inst.is_available.return_value = True
        inst.generate.return_value = "Your portfolio is doing great."

        r = client.post("/api/ai/ask", json={"question": "How is my portfolio?"})

    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "Your portfolio is doing great."
    assert data["llm_available"] is True


def test_ask_endpoint_returns_503_when_llm_disabled(tmp_path: Path) -> None:
    s = _settings(tmp_path, llm_enabled=False)
    app = create_app(settings=s)
    with TestClient(app) as client:
        r = client.post("/api/ai/ask", json={"question": "Hi?"})
    assert r.status_code == 503
    assert "disabled" in r.json()["detail"].lower()


def test_ask_endpoint_validates_empty_question(tmp_path: Path) -> None:
    s = _settings(tmp_path, llm_enabled=True)
    app = create_app(settings=s)
    with TestClient(app) as client:
        r = client.post("/api/ai/ask", json={"question": ""})
    assert r.status_code == 400
    assert r.json().get("error") == "Bad request"


def test_ask_endpoint_requires_auth_when_token_set(tmp_path: Path) -> None:
    s = _settings(tmp_path, llm_enabled=True, api_token="secret123")
    app = create_app(settings=s)
    with TestClient(app) as client:
        r = client.post("/api/ai/ask", json={"question": "test"})
    assert r.status_code == 401
