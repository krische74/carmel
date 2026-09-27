"""Health and public config JSON endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_openapi_lists_config_tag_separate_from_health(client_open: TestClient) -> None:
    """``GET /api/config`` is grouped under ``config`` in OpenAPI, not ``health``."""
    r = client_open.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    cfg_tags = paths.get("/api/config", {}).get("get", {}).get("tags", [])
    health_tags = paths.get("/api/health", {}).get("get", {}).get("tags", [])
    assert cfg_tags == ["config"]
    assert "health" in health_tags


def test_health_returns_ok_shape(client_open: TestClient) -> None:
    r = client_open.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "heartbeat_utc" in body
    assert "environment" in body
    assert "version" in body
    assert "uptime_seconds" in body
    assert isinstance(body["uptime_seconds"], (int, float))


def test_config_excludes_secrets(client_open: TestClient) -> None:
    r = client_open.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert "alpaca_api_key" not in data
    assert "alpaca_secret_key" not in data
    assert "api_token" not in data
    assert "smtp_password" not in data
    assert "dashboard_password" not in data
    assert "app" in data
