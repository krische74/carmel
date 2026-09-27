"""Bearer token behavior for /api routes."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import DataConfig, Settings


def _settings(tmp_path: Path, *, api_token: str) -> Settings:
    return Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        api_token=api_token,
        data=DataConfig(
            cache_dir=str(tmp_path / "cache"),
            parquet_dir=str(tmp_path / "parquet"),
        ),
    )


def test_api_open_when_api_token_empty(client_open: TestClient) -> None:
    """Empty api_token allows requests without Authorization header."""
    r = client_open.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_rejects_missing_bearer_when_token_configured(tmp_path: Path) -> None:
    """401 when api_token is set and Authorization is absent."""
    app = create_app(settings=_settings(tmp_path, api_token="secret-token"))
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 401
    assert r.json()["detail"] == "Missing bearer token"


def test_api_rejects_wrong_bearer_when_token_configured(tmp_path: Path) -> None:
    """401 when Bearer value does not match Settings.api_token."""
    app = create_app(settings=_settings(tmp_path, api_token="expected"))
    with TestClient(app) as client:
        r = client.get("/api/health", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid bearer token"


def test_api_accepts_matching_bearer(tmp_path: Path) -> None:
    """200 when Authorization Bearer matches api_token."""
    app = create_app(settings=_settings(tmp_path, api_token="good"))
    with TestClient(app) as client:
        r = client.get("/api/health", headers={"Authorization": "Bearer good"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
