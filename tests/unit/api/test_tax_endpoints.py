"""REST tax summary and CSV export."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.dependencies import get_lot_ledger
from src.config import Settings


def test_tax_summary_returns_200_with_year(tmp_api_settings: Settings) -> None:
    mock_led = MagicMock()
    mock_led.detect_wash_sales.return_value = []
    mock_led.get_closed_lots.return_value = []

    app = create_app(settings=tmp_api_settings, lot_ledger=mock_led)

    def override_ledger() -> MagicMock:
        return mock_led

    app.dependency_overrides[get_lot_ledger] = override_ledger
    with TestClient(app) as client:
        r = client.get("/api/tax/summary", params={"year": 2026})
    assert r.status_code == 200
    data = r.json()
    assert data["year"] == 2026
    assert "total_net" in data


def test_tax_export_returns_csv_content_type(tmp_api_settings: Settings) -> None:
    mock_led = MagicMock()
    mock_led.detect_wash_sales.return_value = []
    mock_led.get_closed_lots.return_value = []

    app = create_app(settings=tmp_api_settings, lot_ledger=mock_led)

    def override_ledger() -> MagicMock:
        return mock_led

    app.dependency_overrides[get_lot_ledger] = override_ledger
    with TestClient(app) as client:
        r = client.get("/api/tax/export", params={"year": 2025})
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")
    assert b"Description" in r.content


def test_tax_summary_returns_empty_for_no_data_year(tmp_path, tmp_api_settings: Settings) -> None:
    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub.sqlite"
    led = LotLedger(db)
    app = create_app(settings=tmp_api_settings, lot_ledger=led)
    with TestClient(app) as client:
        r = client.get("/api/tax/summary", params={"year": 2005})
    assert r.status_code == 200
    assert r.json()["total_net"] == 0.0


def test_tax_summary_with_account_filter_returns_subset(tmp_path, tmp_api_settings: Settings) -> None:
    from datetime import UTC, datetime

    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub_acct.sqlite"
    led = LotLedger(db)
    t = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="main")
    led.record_sell("SPY", 1.0, 110.0, t, account_id="main")
    led.record_buy("QQQ", 1.0, 50.0, t, account_id="ira")
    led.record_sell("QQQ", 1.0, 55.0, t, account_id="ira")

    app = create_app(settings=tmp_api_settings, lot_ledger=led)
    with TestClient(app) as client:
        r = client.get("/api/tax/summary", params={"year": 2026, "account_id": "main"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["lots"]) == 1
    assert data["lots"][0]["symbol"] == "SPY"
    assert data["lots"][0]["account_id"] == "main"


def test_tax_summary_account_id_whitespace_is_stripped(tmp_path, tmp_api_settings: Settings) -> None:
    """Leading/trailing spaces on ``account_id`` should still match the stored id."""
    from datetime import UTC, datetime

    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub_ws.sqlite"
    led = LotLedger(db)
    t = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="main")
    led.record_sell("SPY", 1.0, 110.0, t, account_id="main")

    app = create_app(settings=tmp_api_settings, lot_ledger=led)
    with TestClient(app) as client:
        r = client.get("/api/tax/summary", params={"year": 2026, "account_id": "  main  "})
    assert r.status_code == 200
    data = r.json()
    assert len(data["lots"]) == 1
    assert data["lots"][0]["symbol"] == "SPY"


def test_tax_summary_whitespace_only_account_id_means_consolidated(tmp_path, tmp_api_settings: Settings) -> None:
    """Blank/whitespace ``account_id`` normalizes to None → all accounts in summary."""
    from datetime import UTC, datetime

    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub_blank_acct.sqlite"
    led = LotLedger(db)
    t = datetime(2026, 3, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t, account_id="a")
    led.record_sell("SPY", 1.0, 110.0, t, account_id="a")
    led.record_buy("QQQ", 1.0, 50.0, t, account_id="b")
    led.record_sell("QQQ", 1.0, 55.0, t, account_id="b")

    app = create_app(settings=tmp_api_settings, lot_ledger=led)
    with TestClient(app) as client:
        r = client.get("/api/tax/summary", params={"year": 2026, "account_id": "   "})
    assert r.status_code == 200
    assert len(r.json()["lots"]) == 2


def test_tax_schedule_d_returns_json(tmp_path, tmp_api_settings: Settings) -> None:
    from datetime import UTC, datetime

    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub_sd.sqlite"
    led = LotLedger(db)
    t = datetime(2026, 8, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t)
    led.record_sell("SPY", 1.0, 115.0, t)
    app = create_app(settings=tmp_api_settings, lot_ledger=led)
    with TestClient(app) as client:
        r = client.get("/api/tax/schedule-d", params={"year": 2026})
    assert r.status_code == 200
    data = r.json()
    assert data["year"] == 2026
    assert "short_term_net" in data
    assert data["total_net_gain_or_loss"] == pytest.approx(15.0)


def test_tax_schedule_d_export_csv(tmp_path, tmp_api_settings: Settings) -> None:
    from datetime import UTC, datetime

    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub_sde.sqlite"
    led = LotLedger(db)
    t = datetime(2026, 9, 1, tzinfo=UTC)
    led.record_buy("Z", 1.0, 1.0, t)
    led.record_sell("Z", 1.0, 2.0, t)
    app = create_app(settings=tmp_api_settings, lot_ledger=led)
    with TestClient(app) as client:
        r = client.get("/api/tax/schedule-d/export", params={"year": 2026})
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")
    assert b"Part I" in r.content


def test_tax_reconcile_1099b_upload(tmp_path, tmp_api_settings: Settings) -> None:
    from datetime import UTC, datetime

    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub_r.sqlite"
    led = LotLedger(db)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ts = datetime(2026, 6, 1, tzinfo=UTC)
    led.record_buy("SPY", 1.0, 100.0, t0)
    led.record_sell("SPY", 1.0, 110.0, ts)
    csv_body = (
        b"Symbol,Qty,Date Acquired,Date Sold,Proceeds,Cost Basis\n"
        b"SPY,1.0,2026-01-01,2026-06-01,110.00,100.00\n"
    )
    app = create_app(settings=tmp_api_settings, lot_ledger=led)
    with TestClient(app) as client:
        r = client.post(
            "/api/tax/reconcile-1099b",
            params={"year": 2026},
            files={"file": ("b.csv", BytesIO(csv_body), "text/csv")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["matched"] >= 1


def test_tax_carryforward_endpoint_returns_model(tmp_path, tmp_api_settings: Settings) -> None:
    from src.data.storage.sqlite_store import SQLiteStore
    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub_cf_api.sqlite"
    led = LotLedger(db)
    store = SQLiteStore(db)
    app = create_app(settings=tmp_api_settings, lot_ledger=led, sqlite_store=store)
    with TestClient(app) as client:
        r = client.get("/api/tax/carryforward", params={"year": 2026})
    assert r.status_code == 200
    data = r.json()
    assert "short_term_carryforward" in data
    assert "long_term_carryforward" in data
    assert "total_carryforward" in data


def test_tax_cross_account_washes_endpoint(tmp_path, tmp_api_settings: Settings) -> None:
    from datetime import UTC, datetime

    from src.portfolio.tax_lots import LotLedger

    db = tmp_path / "hub_x.sqlite"
    led = LotLedger(db)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    loss_t = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    buy_t = datetime(2026, 2, 10, 12, 0, tzinfo=UTC)
    led.record_buy("SPY", 10.0, 100.0, t0, account_id="a")
    led.record_sell("SPY", 10.0, 90.0, loss_t, account_id="a")
    led.record_buy("SPY", 10.0, 88.0, buy_t, account_id="b")

    app = create_app(settings=tmp_api_settings, lot_ledger=led)
    with TestClient(app) as client:
        r = client.get("/api/tax/cross-account-washes", params={"year": 2026})
    assert r.status_code == 200
    body = r.json()
    assert len(body) >= 1
    assert body[0]["selling_account"] == "a"
    assert body[0]["buying_account"] == "b"
    assert body[0]["symbol"] == "SPY"
