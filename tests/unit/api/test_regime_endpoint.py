"""GET /api/market/regime."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import Settings
from src.data.regime import (
    MarketRegime,
    OverallRegime,
    VolatilityRegime,
    YieldCurveRegime,
)
from src.data.storage.sqlite_store import SQLiteStore


def test_regime_endpoint_returns_latest(tmp_path, tmp_api_settings: Settings) -> None:
    store = SQLiteStore(tmp_path / "hub.db")
    mr = MarketRegime(
        timestamp=datetime(2026, 4, 1, 16, 0, tzinfo=UTC),
        vix_close=20.0,
        yield_spread=0.5,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.RISK_ON,
        sizing_multiplier=1.0,
    )
    store.write_regime_snapshot(mr)
    app = create_app(settings=tmp_api_settings, sqlite_store=store)
    with TestClient(app) as client:
        r = client.get("/api/market/regime")
    assert r.status_code == 200
    data = r.json()
    assert data["overall"] == "risk_on"
    assert float(data["sizing_multiplier"]) == pytest.approx(1.0)


def test_regime_endpoint_404_when_no_data(tmp_api_settings: Settings) -> None:
    from unittest.mock import MagicMock

    store = MagicMock()
    store.get_latest_regime.return_value = None
    app = create_app(settings=tmp_api_settings, sqlite_store=store)
    with TestClient(app) as client:
        r = client.get("/api/market/regime")
    assert r.status_code == 404
