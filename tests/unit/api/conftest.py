"""Fixtures for FastAPI unit tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import DataConfig, Settings


@pytest.fixture()
def tmp_api_settings(tmp_path: Path) -> Settings:
    """Isolated cache/parquet dirs so SQLite and Parquet paths stay under tmp_path."""
    return Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            cache_dir=str(tmp_path / "cache"),
            parquet_dir=str(tmp_path / "parquet"),
        ),
    )


@pytest.fixture()
def client_open(tmp_api_settings: Settings) -> Generator[TestClient, None, None]:
    """Test client with API auth disabled (empty api_token)."""
    app = create_app(settings=tmp_api_settings)
    with TestClient(app) as c:
        yield c
