"""Consistent JSON bodies for API errors (400 / 404 / 500)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import DataConfig, Settings


@pytest.fixture()
def err_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            cache_dir=str(tmp_path / "cache"),
            parquet_dir=str(tmp_path / "parquet"),
        ),
    )
    app = create_app(settings=s)

    def boom() -> None:
        raise RuntimeError("internal detail must not leak")

    app.add_api_route("/api/__boom", boom, methods=["GET"])
    # ``raise_server_exceptions=False`` so unhandled errors return HTTP 500 bodies.
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_api_400_returns_json_error(err_client: TestClient) -> None:
    r = err_client.get("/api/trades/signals", params={"offset": -1})
    assert r.status_code == 400
    body = r.json()
    assert body.get("error") == "Bad request"
    assert "detail" in body


def test_api_404_returns_json_error(err_client: TestClient) -> None:
    r = err_client.get("/api/this_route_does_not_exist_ever")
    assert r.status_code == 404
    body = r.json()
    assert body.get("error") == "Not found"


def test_api_500_returns_json_error_without_internal_detail(err_client: TestClient) -> None:
    r = err_client.get("/api/__boom")
    assert r.status_code == 500
    body = r.json()
    assert body == {"error": "Internal server error"}
