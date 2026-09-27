"""Shared fixtures for dashboard unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import DataConfig, Settings


@pytest.fixture()
def dashboard_settings(tmp_path) -> Settings:
    """Settings with hub cache under ``tmp_path`` (no real SQLite required for path tests)."""
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            cache_dir=str(tmp_path / "cache"),
            parquet_dir=str(tmp_path / "parquet"),
        ),
    )


@pytest.fixture()
def mock_sqlite_store() -> MagicMock:
    """Lightweight stand-in when pages are not executed."""
    return MagicMock()
