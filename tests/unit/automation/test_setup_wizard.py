"""Tests for setup wizard helpers (non-interactive)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from unittest.mock import MagicMock, patch

import pytest

from src.automation.setup_wizard import (
    run_initial_ingest,
    run_setup_wizard,
    validate_alpaca_keys,
    write_env_file,
)
from src.config import (
    DataConfig,
    MeanReversionConfig,
    MomentumConfig,
    Settings,
    StrategyConfig,
    TaxConfig,
)
from src.execution.errors import ConfigurationError


def test_validate_alpaca_keys_success() -> None:
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    with patch("src.automation.setup_wizard.AlpacaBrokerAdapter.create", return_value=broker):
        ok, msg = validate_alpaca_keys("k", "s")
    assert ok is True
    assert float(msg) == pytest.approx(10_000.0)


def test_validate_alpaca_keys_failure() -> None:
    with (
        patch(
            "src.automation.setup_wizard.AlpacaBrokerAdapter.create",
            side_effect=ConfigurationError("Alpaca rejected API credentials (bad creds)"),
        ),
        patch("src.automation.runner.notify_operational_critical") as notify_mock,
    ):
        ok, msg = validate_alpaca_keys("k", "s")
    assert ok is False
    assert "bad creds" in msg
    notify_mock.assert_not_called()


def test_run_setup_wizard_retries_then_succeeds_without_credential_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.automation import setup_wizard

    monkeypatch.setattr(setup_wizard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        setup_wizard,
        "validate_alpaca_keys",
        MagicMock(side_effect=[(False, "bad 1"), (False, "bad 2"), (True, "1000.0")]),
    )
    monkeypatch.setattr(setup_wizard, "write_env_file", MagicMock(return_value=True))
    monkeypatch.setattr(setup_wizard, "run_initial_ingest", MagicMock(return_value={}))
    settings_obj = Settings(
        _yaml_path=tmp_path / "settings.yaml",
        _env_file=None,
        data=DataConfig(parquet_dir=str(tmp_path / "pq"), cache_dir=str(tmp_path / "cache")),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    get_settings_mock = MagicMock(return_value=settings_obj)
    get_settings_mock.cache_clear = MagicMock()
    monkeypatch.setattr(setup_wizard, "get_settings", get_settings_mock)
    workflow = MagicMock()

    with (
        patch(
            "builtins.input",
            side_effect=[
                "bad-ak-1",
                "bad-sk-1",
                "y",
                "bad-ak-2",
                "bad-sk-2",
                "y",
                "good-ak",
                "good-sk",
                "",
                "",
            ],
        ),
        patch("src.automation.runner.notify_operational_critical") as notify_mock,
        patch("src.automation.runner.create_trading_workflow", return_value=workflow),
    ):
        run_setup_wizard()

    notify_mock.assert_not_called()


def test_run_setup_wizard_notifies_once_on_credential_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.automation import setup_wizard

    monkeypatch.setattr(setup_wizard, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        setup_wizard,
        "validate_alpaca_keys",
        MagicMock(return_value=(False, "Alpaca rejected API credentials (bad creds)")),
    )
    monkeypatch.setattr(setup_wizard, "run_initial_ingest", MagicMock(return_value={}))
    settings_obj = Settings(
        _yaml_path=tmp_path / "settings.yaml",
        _env_file=None,
        data=DataConfig(parquet_dir=str(tmp_path / "pq"), cache_dir=str(tmp_path / "cache")),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    get_settings_mock = MagicMock(return_value=settings_obj)
    get_settings_mock.cache_clear = MagicMock()
    monkeypatch.setattr(setup_wizard, "get_settings", get_settings_mock)

    with (
        patch(
            "builtins.input",
            side_effect=["bad-ak", "bad-sk", "n", "", ""],
        ),
        patch("src.automation.runner.notify_operational_critical") as notify_mock,
    ):
        run_setup_wizard()

    notify_mock.assert_called_once()
    _, kwargs = notify_mock.call_args
    assert kwargs["category"] == "alpaca_credentials"
    assert "Alpaca API rejected credentials during setup validation" in kwargs["message"]
    assert "bad creds" in kwargs["message"]


def test_write_env_file_creates_new(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    written = write_env_file(
        path,
        {
            "alpaca_api_key": "KEY1",
            "alpaca_secret_key": "SEC1",
            "fred_api_key": "FRED1",
            "webhook_url": "https://example.com/hook",
        },
        overwrite=False,
    )
    assert written is True
    text = path.read_text(encoding="utf-8")
    assert "ALPACA_API_KEY=KEY1" in text
    assert "ALPACA_SECRET_KEY=SEC1" in text
    assert "ALPACA_PAPER=true" in text
    assert "FRED_API_KEY=FRED1" in text
    assert "NOTIFICATION__ENABLED=true" in text
    assert "NOTIFICATION__WEBHOOK_URL=https://example.com/hook" in text
    assert not text.endswith(" \n") and not text.rstrip().endswith(" ")


def test_write_env_file_does_not_overwrite_without_flag(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("ORIGINAL=1\n", encoding="utf-8")
    written = write_env_file(
        path,
        {"alpaca_api_key": "K", "alpaca_secret_key": "S"},
        overwrite=False,
    )
    assert written is False
    assert path.read_text(encoding="utf-8") == "ORIGINAL=1\n"


def test_write_env_file_overwrites_when_flag_true(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("ORIGINAL=1\n", encoding="utf-8")
    written = write_env_file(
        path,
        {
            "alpaca_api_key": "NEWK",
            "alpaca_secret_key": "NEWS",
            "fred_api_key": "",
            "webhook_url": "",
        },
        overwrite=True,
    )
    assert written is True
    text = path.read_text(encoding="utf-8")
    assert "ORIGINAL" not in text
    assert "ALPACA_API_KEY=NEWK" in text
    assert "ALPACA_SECRET_KEY=NEWS" in text


def test_run_initial_ingest_reports_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data.pipeline import DataPipeline, IngestResult

    def fake_ingest(self: DataPipeline, symbol: str) -> IngestResult:
        if symbol == "BAD":
            return IngestResult(success=False, error="fail")
        return IngestResult(success=True)

    monkeypatch.setattr(DataPipeline, "ingest_ohlcv", fake_ingest)

    yml = tmp_path / "no_settings.yaml"
    assert not yml.exists()
    settings = Settings(
        _yaml_path=yml,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "QQQ", "BAD", "TLT"],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    out = run_initial_ingest(settings)
    assert out["SPY"] is True
    assert out["QQQ"] is True
    assert out["BAD"] is False
    assert out["TLT"] is True
