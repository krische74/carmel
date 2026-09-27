"""Scheduler timezone wiring (Tier 25)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.automation import runner as runner_mod
from src.automation.scheduler import HubScheduler, scheduler_timezone_from_settings
from src.config import DataConfig, SchedulerConfig, Settings


@pytest.fixture(autouse=True)
def _bypass_daemon_dedupe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable cross-process daemon dedupe in unit tests (real heartbeat would trip it)."""
    monkeypatch.setattr(
        runner_mod,
        "_abort_if_other_daemon_running",
        lambda *a, **k: None,
    )


def test_scheduler_uses_configured_timezone() -> None:
    captured: dict[str, object] = {}

    def _spy(*args, **kwargs):
        captured["kwargs"] = kwargs
        return MagicMock()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        scheduler=SchedulerConfig(timezone="US/Eastern"),
    )
    with patch("src.automation.scheduler.BackgroundScheduler", _spy):
        HubScheduler(settings=settings)
    tz = captured["kwargs"].get("timezone")
    assert tz is not None
    assert str(tz) == "US/Eastern"


def test_zoneinfo_failure_falls_back_to_utc(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from zoneinfo import ZoneInfo as RealZI

    def _zi(name: str):
        if name == "US/Eastern":
            raise OSError("simulated ZoneInfo failure")
        return RealZI(name)

    monkeypatch.setattr("src.automation.scheduler.ZoneInfo", _zi)
    caplog.set_level(logging.WARNING)
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        scheduler=SchedulerConfig(timezone="US/Eastern"),
    )
    tz = scheduler_timezone_from_settings(settings)
    assert str(tz) == "UTC"
    assert any("ZoneInfo failed" in r.message for r in caplog.records)


def test_invalid_timezone_in_settings_yaml_falls_back_to_utc(caplog) -> None:
    caplog.set_level(logging.WARNING)
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        scheduler=SchedulerConfig(timezone="Mars/Olympus"),
    )
    assert s.scheduler.timezone == "UTC"
    assert any("Invalid scheduler.timezone" in r.message for r in caplog.records)


def test_timezone_in_startup_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hub = MagicMock()
    hub.scheduler = MagicMock()
    monkeypatch.setattr(runner_mod, "HubScheduler", lambda **kwargs: hub)

    def _raise_kb(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.time, "sleep", _raise_kb)

    wf = MagicMock()
    wf.run_ingest_only = MagicMock()
    monkeypatch.setattr(runner_mod, "create_trading_workflow", lambda *a, **k: wf)
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
        scheduler=SchedulerConfig(timezone="Europe/London", reconcile_cron=""),
    )
    caplog.set_level(logging.INFO)
    runner_mod._run_daemon(settings, {"default": MagicMock()})
    assert any("Scheduler timezone: Europe/London" in r.message for r in caplog.records)
