"""Shared test fixtures for Carmel.

All fixtures available to every test file via pytest's conftest mechanism.

File logging: ``setup_logging()`` normally appends to ``logs/carmel.log`` under
the project root. ``pytest_configure`` sets ``CARMEL_LOG_DIR`` to a temporary
directory so test runs (including ``setup_logging()`` without mocks) do not
pollute the operator's production log. Override ``CARMEL_LOG_DIR`` in the
environment before invoking pytest if you need a fixed path.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import Settings


def pytest_configure(config: pytest.Config) -> None:
    """Redirect rotating file logs away from ``logs/carmel.log`` for the session."""
    _ = config  # required pytest hook signature
    if os.environ.get("CARMEL_LOG_DIR"):
        return
    isolated = Path(tempfile.mkdtemp(prefix="carmel-pytest-logs-"))
    os.environ["CARMEL_LOG_DIR"] = str(isolated)


@pytest.fixture()
def settings() -> Settings:
    """Provide a fresh Settings instance for each test (not the singleton)."""
    return Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
    )


@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """Provide a small sample OHLCV DataFrame for testing data pipelines."""
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 101.5, 103.0],
            "high": [101.0, 102.5, 103.0, 102.0, 104.0],
            "low": [99.5, 100.5, 101.0, 100.0, 102.5],
            "close": [100.5, 102.0, 101.5, 101.0, 103.5],
            "volume": [1000000, 1100000, 950000, 1050000, 1200000],
        },
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )


@pytest.fixture()
def sample_ohlcv_invalid() -> pd.DataFrame:
    """Provide an OHLCV DataFrame with validation errors for testing."""
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [99.0, 102.5, 103.0],  # high < open on row 0
            "low": [99.5, 100.5, 101.0],
            "close": [100.5, 102.0, 101.5],
            "volume": [1000000, -100, 950000],  # negative volume on row 1
        },
        index=pd.date_range("2024-01-02", periods=3, freq="B"),
    )
