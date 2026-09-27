"""Tests for CLI and default wiring (broker/pipeline mocked or injected)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.automation import runner as runner_mod
from src.automation.strategy_wiring import build_strategies_from_enabled

# Captured before the autouse `_bypass_daemon_dedupe` fixture patches the
# module-level name out. The fixture is correct for _run_daemon tests
# (which would otherwise trip on a real local heartbeat) but the
# _abort_if_other_daemon_running tests below need the real implementation.
_real_abort_if_other_daemon_running = runner_mod._abort_if_other_daemon_running
# Keep these imports after capturing the unpatched runner helper.
from src.config import DataConfig, DCATarget, Settings  # noqa: E402
from src.strategy.base import Strategy  # noqa: E402
from src.strategy.dca import DCAStrategy  # noqa: E402
from src.strategy.mean_reversion import MeanReversionStrategy  # noqa: E402
from src.strategy.momentum import MomentumRotationStrategy  # noqa: E402


def _settings_with_local_paths(tmp_path: Path) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
        ),
        alpaca_api_key="test_key",
        alpaca_secret_key="test_secret",
    )


def test_create_trading_workflow_raises_when_alpaca_missing_and_no_broker() -> None:
    from src.automation.runner import create_trading_workflow

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        alpaca_api_key="",
        alpaca_secret_key="",
    )
    with pytest.raises(ValueError, match="Alpaca"):
        create_trading_workflow(settings)


def test_broker_from_settings_raises_configuration_error_when_alpaca_rejects_credentials(
    tmp_path: Path,
) -> None:
    """Invalid Alpaca keys must fail at broker construction (never a silent mock broker)."""
    from unittest.mock import patch

    from src.automation.runner import broker_from_settings
    from src.execution.errors import ConfigurationError

    settings = _settings_with_local_paths(tmp_path)
    with (
        patch(
            "src.automation.runner.AlpacaBrokerAdapter.create",
            side_effect=ConfigurationError(
                "Alpaca rejected API credentials (check ALPACA_API_KEY / ALPACA_SECRET_KEY)",
            ),
        ),
        pytest.raises(ConfigurationError, match="Alpaca rejected API credentials"),
    ):
        broker_from_settings(settings)


def test_email_notifier_instantiated_when_enabled(tmp_path: Path) -> None:
    from src.automation.notifier import EmailNotifier
    from src.automation.runner import create_trading_workflow

    settings = _settings_with_local_paths(tmp_path)
    settings.notification.enabled = False
    settings.notification.webhook_url = ""
    settings.notification.email_enabled = True
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    broker.list_recent_orders.return_value = []
    wf = create_trading_workflow(settings, broker=broker)
    assert isinstance(wf.notifier, EmailNotifier)


def test_create_trading_workflow_wires_momentum_and_dca_with_injected_broker(
    tmp_path: Path,
) -> None:
    from src.automation.runner import create_trading_workflow

    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    broker.list_recent_orders.return_value = []

    settings = _settings_with_local_paths(tmp_path)
    wf = create_trading_workflow(settings, broker=broker)

    assert wf._lot_ledger is not None

    names = [type(s).__name__ for s in wf.strategies]
    assert "MomentumRotationStrategy" in names
    assert "DCAStrategy" in names
    assert any(isinstance(s, MomentumRotationStrategy) for s in wf.strategies)
    assert any(isinstance(s, DCAStrategy) for s in wf.strategies)


def test_main_config_prints_non_secret_json(capsys: pytest.CaptureFixture[str]) -> None:
    from src.automation.runner import main

    with patch("src.automation.runner.get_settings") as gs:
        gs.return_value = Settings(_yaml_path=None, _env_file=None, alpaca_api_key="secret")
        code = main(["config"])
    assert code == 0
    out = capsys.readouterr().out
    assert "secret" not in out
    assert "Carmel" in out or "app" in out


def test_main_serve_invokes_uvicorn_with_host_port() -> None:
    """``serve`` loads the FastAPI app factory and passes host/port to uvicorn."""
    from src.automation import runner

    with (
        patch.object(runner, "setup_logging", lambda *a, **k: None),
        patch.object(
            runner,
            "get_settings",
            lambda: Settings(_yaml_path=None, _env_file=None),
        ),
        patch("uvicorn.run") as uv_run,
    ):
        code = runner.main(["serve", "--host", "127.0.0.1", "--port", "9000"])
    assert code == 0
    uv_run.assert_called_once()
    assert uv_run.call_args.kwargs["host"] == "127.0.0.1"
    assert uv_run.call_args.kwargs["port"] == 9000


def test_main_run_returns_1_when_alpaca_keys_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from src.automation import runner

    monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: Settings(
            _yaml_path=None,
            _env_file=None,
            alpaca_api_key="",
            alpaca_secret_key="",
        ),
    )
    caplog.set_level(logging.ERROR)
    code = runner.main(["run"])
    assert code == 1
    assert "Alpaca API credentials" in caplog.text
    assert ".env.example" in caplog.text


def test_main_once_calls_run_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.automation import runner

    wf = MagicMock()
    monkeypatch.setattr(
        runner,
        "build_brokers",
        lambda _s: {"default": MagicMock()},
    )
    monkeypatch.setattr(
        runner,
        "create_trading_workflow",
        lambda *_a, **_k: wf,
    )
    monkeypatch.setattr(runner, "SQLiteStore", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(runner, "LotLedger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: Settings(
            _yaml_path=None,
            _env_file=None,
            alpaca_api_key="k",
            alpaca_secret_key="s",
        ),
    )

    code = runner.main(["once"])
    assert code == 0
    wf.run_cycle.assert_called_once_with(as_of=None)


def test_parse_as_of_iso_naive_datetime_assumes_utc() -> None:
    from src.config import parse_as_of_iso

    got = parse_as_of_iso("2026-03-30T12:00:00")
    assert got is not None
    assert got.tzinfo == UTC
    assert got == datetime(2026, 3, 30, 12, 0, 0, tzinfo=UTC)


def test_rows_in_digest_window_keeps_only_rows_in_date_range() -> None:
    """Weekly digest should not count executions/alerts outside the digest window."""
    from datetime import date

    from src.automation.runner import _rows_in_digest_window

    rows = [
        {"timestamp": "2026-04-01T12:00:00+00:00", "id": 1},
        {"timestamp": "2025-01-01T12:00:00+00:00", "id": 2},
        {"timestamp": "", "id": 3},
    ]
    out = _rows_in_digest_window(rows, start=date(2026, 3, 30), end=date(2026, 4, 4))
    assert len(out) == 1
    assert out[0]["id"] == 1


def test_main_backtest_prints_metrics(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=120, freq="B")
    close = 100.0 + np.linspace(0, 20, len(idx))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )
    pq_store = MagicMock()
    pq_store.read_ohlcv.return_value = df
    sql_store = MagicMock()
    sql_store.save_backtest_run.return_value = 42

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "ParquetStore", MagicMock(return_value=pq_store))
    monkeypatch.setattr(runner_mod, "SQLiteStore", MagicMock(return_value=sql_store))

    code = runner_mod.main(
        [
            "backtest",
            "--strategy",
            "momentum",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-28",
        ],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Total return" in out
    assert "Run saved (ID: 42)" in out
    assert sql_store.save_backtest_run.called


def test_backtest_cli_prints_helpful_error_on_no_data(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Backtest window with no overlapping OHLCV dates exits 1 with ingest hint."""
    idx = pd.bdate_range("2024-01-02", periods=120, freq="B")
    close = 100.0 + np.linspace(0, 20, len(idx))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )
    pq_store = MagicMock()
    pq_store.read_ohlcv.return_value = df
    sql_store = MagicMock()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "ParquetStore", MagicMock(return_value=pq_store))
    monkeypatch.setattr(runner_mod, "SQLiteStore", MagicMock(return_value=sql_store))

    code = runner_mod.main(
        [
            "backtest",
            "--strategy",
            "dca",
            "--start",
            "2020-01-01",
            "--end",
            "2020-12-31",
        ],
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "ERROR:" in out
    assert "ingest" in out.lower()
    assert "SUGGESTION:" in out
    assert "Your data covers" in out
    assert "2020-01-01" in out and "2020-12-31" in out
    assert not sql_store.save_backtest_run.called


def test_backtest_cli_prints_zero_trades_warning(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI surfaces engine zero-trades warning after the metrics block."""

    class _QuietSpy(Strategy):
        def get_universe(self) -> list[str]:
            return ["SPY"]

        def generate_signals(
            self,
            data: dict[str, pd.DataFrame],
            *,
            as_of: datetime | None = None,
            market_regime: object | None = None,
        ) -> list[Any]:
            return []

    idx = pd.bdate_range("2024-01-02", periods=120, freq="B")
    close = 100.0 + np.linspace(0, 20, len(idx))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )
    pq_store = MagicMock()
    pq_store.read_ohlcv.return_value = df
    sql_store = MagicMock()
    sql_store.save_backtest_run.return_value = 43

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "ParquetStore", MagicMock(return_value=pq_store))
    monkeypatch.setattr(runner_mod, "SQLiteStore", MagicMock(return_value=sql_store))
    monkeypatch.setattr(runner_mod, "build_strategy_for_backtest", lambda *_a, **_k: _QuietSpy())

    code = runner_mod.main(
        [
            "backtest",
            "--strategy",
            "momentum",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-28",
        ],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "WARNING:" in out
    assert "0 trades" in out
    assert "Run saved (ID: 43)" in out


def test_backtest_cli_suggests_viable_range_on_data_missing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    idx = pd.bdate_range("2024-04-11", periods=80, freq="B")
    close = 100.0 + np.linspace(0, 10, len(idx))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )
    pq_store = MagicMock()
    pq_store.read_ohlcv.return_value = df
    sql_store = MagicMock()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "ParquetStore", MagicMock(return_value=pq_store))
    monkeypatch.setattr(runner_mod, "SQLiteStore", MagicMock(return_value=sql_store))

    code = runner_mod.main(
        [
            "backtest",
            "--strategy",
            "dca",
            "--start",
            "2023-06-01",
            "--end",
            "2024-06-01",
        ],
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "SUGGESTION:" in out
    assert "2024-04-11" in out
    assert "carmel backtest" in out


def test_backtest_cli_suggests_ingest_command_on_data_missing(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    idx = pd.bdate_range("2024-01-02", periods=50, freq="B")
    close = 100.0 + np.linspace(0, 5, len(idx))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )
    pq_store = MagicMock()
    pq_store.read_ohlcv.return_value = df
    sql_store = MagicMock()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "ParquetStore", MagicMock(return_value=pq_store))
    monkeypatch.setattr(runner_mod, "SQLiteStore", MagicMock(return_value=sql_store))

    code = runner_mod.main(
        [
            "backtest",
            "--strategy",
            "dca",
            "--start",
            "2019-01-01",
            "--end",
            "2020-01-01",
        ],
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "To extend coverage, run:" in out
    assert "carmel ingest --start 2019-01-01 --end 2020-01-01" in out


def test_main_backtest_missing_parquet_exits_1(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock()
    store.read_ohlcv.return_value = pd.DataFrame()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "ParquetStore", MagicMock(return_value=store))

    code = runner_mod.main(
        [
            "backtest",
            "--strategy",
            "momentum",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-28",
        ],
    )
    assert code == 1


def test_main_backtest_rejects_start_after_end(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)

    code = runner_mod.main(
        [
            "backtest",
            "--strategy",
            "momentum",
            "--start",
            "2025-12-31",
            "--end",
            "2024-01-01",
        ],
    )
    assert code == 1


def test_main_ml_train_missing_parquet_exits_1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = MagicMock()
    store.read_ohlcv.return_value = pd.DataFrame()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "parquet"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        alpaca_api_key="k",
        alpaca_secret_key="s",
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "ParquetStore", MagicMock(return_value=store))

    code = runner_mod.main(
        [
            "ml-train",
            "--start",
            "2024-01-02",
            "--end",
            "2024-06-28",
        ],
    )
    assert code == 1


def test_main_once_passes_as_of(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.automation import runner

    wf = MagicMock()
    monkeypatch.setattr(
        runner,
        "build_brokers",
        lambda _s: {"default": MagicMock()},
    )
    monkeypatch.setattr(runner, "create_trading_workflow", lambda *_a, **_k: wf)
    monkeypatch.setattr(runner, "SQLiteStore", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(runner, "LotLedger", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(runner, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "get_settings",
        lambda: Settings(
            _yaml_path=None,
            _env_file=None,
            alpaca_api_key="k",
            alpaca_secret_key="s",
        ),
    )

    code = runner.main(["once", "--as-of", "2026-03-30T12:00:00+00:00"])
    assert code == 0
    wf.run_cycle.assert_called_once()
    call_kw = wf.run_cycle.call_args.kwargs
    assert call_kw["as_of"] == datetime(2026, 3, 30, 12, 0, tzinfo=UTC)


def test_create_trading_workflow_respects_enabled_strategies(tmp_path: Path) -> None:
    from src.automation.runner import create_trading_workflow

    settings = _settings_with_local_paths(tmp_path)
    settings.strategy.enabled = ["dca"]
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    broker.list_recent_orders.return_value = []

    wf = create_trading_workflow(settings, broker=broker)
    assert len(wf.strategies) == 1
    assert isinstance(wf.strategies[0], DCAStrategy)


def test_create_trading_workflow_includes_mean_reversion_when_enabled(tmp_path: Path) -> None:
    from src.automation.runner import create_trading_workflow

    settings = _settings_with_local_paths(tmp_path)
    settings.strategy.enabled = ["momentum", "dca", "mean_reversion"]
    broker = MagicMock()
    broker.get_account_equity.return_value = 10_000.0
    broker.get_last_equity.return_value = 10_000.0
    broker.get_cash.return_value = 5_000.0
    broker.get_position_qty.return_value = 0.0
    broker.list_recent_orders.return_value = []

    wf = create_trading_workflow(settings, broker=broker)
    names = [type(s).__name__ for s in wf.strategies]
    assert names == ["MomentumRotationStrategy", "DCAStrategy", "MeanReversionStrategy"]
    assert isinstance(wf.strategies[2], MeanReversionStrategy)


def test_build_strategies_skips_unknown_name_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings_with_local_paths(tmp_path)
    settings.strategy.enabled = ["bogus", "dca"]
    with caplog.at_level("WARNING"):
        strategies = build_strategies_from_enabled(settings)
    assert len(strategies) == 1
    assert isinstance(strategies[0], DCAStrategy)
    assert any("Unknown strategy name" in rec.message for rec in caplog.records)


def test_build_strategies_raises_when_all_names_unknown(tmp_path: Path) -> None:
    settings = _settings_with_local_paths(tmp_path)
    settings.strategy.enabled = ["bogus", "invalid"]
    with pytest.raises(ValueError, match="No valid strategies"):
        build_strategies_from_enabled(settings)


def test_build_strategies_deduplicates_enabled(tmp_path: Path) -> None:
    settings = _settings_with_local_paths(tmp_path)
    settings.strategy.enabled = ["dca", "dca", "momentum"]
    strategies = build_strategies_from_enabled(settings)
    assert len(strategies) == 2
    assert [type(s).__name__ for s in strategies] == ["DCAStrategy", "MomentumRotationStrategy"]


def test_create_trading_workflow_raises_on_empty_enabled(tmp_path: Path) -> None:
    from src.automation.runner import create_trading_workflow

    settings = _settings_with_local_paths(tmp_path)
    settings.strategy.enabled = []
    with pytest.raises(ValueError, match=r"strategy\.enabled"):
        create_trading_workflow(settings, broker=MagicMock())


@pytest.fixture(autouse=True)
def _bypass_daemon_dedupe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable cross-process daemon dedupe in unit tests.

    The real check reads ``logs/hub.heartbeat``. When a developer's local
    container is running, the heartbeat is fresh and ``_run_daemon`` would
    refuse to start — correct in production, wrong inside unit tests.
    """
    from src.automation import runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "_abort_if_other_daemon_running",
        lambda *a, **k: None,
    )


def test_run_daemon_wires_signal_and_rebalance_as_distinct_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.automation import runner as runner_mod

    hub = MagicMock()
    hub.scheduler = MagicMock()
    monkeypatch.setattr(runner_mod, "HubScheduler", lambda **kwargs: hub)

    def _raise_kb(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.time, "sleep", _raise_kb)

    wf = MagicMock()
    wf.run_ingest_only = MagicMock()
    monkeypatch.setattr(runner_mod, "create_trading_workflow", lambda *a, **k: wf)
    settings = _settings_with_local_paths(tmp_path)
    settings.scheduler.reconcile_cron = ""
    broker = MagicMock()
    runner_mod._run_daemon(settings, {"default": broker})
    hub.register_trading_jobs.assert_called_once()
    kw = hub.register_trading_jobs.call_args.kwargs
    assert kw["run_signal_cycle"] is not kw["run_rebalance_check"]
    assert kw["run_data_ingestion"] is wf.run_ingest_only


def test_reconcile_cron_registered_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.automation import runner as runner_mod

    hub = MagicMock()
    hub.scheduler = MagicMock()
    monkeypatch.setattr(runner_mod, "HubScheduler", lambda **kwargs: hub)

    def _raise_kb(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.time, "sleep", _raise_kb)

    wf = MagicMock()
    wf.run_ingest_only = MagicMock()
    monkeypatch.setattr(runner_mod, "create_trading_workflow", lambda *a, **k: wf)
    settings = _settings_with_local_paths(tmp_path)
    settings.scheduler.reconcile_cron = "0 19 * * 5"
    runner_mod._run_daemon(settings, {"default": MagicMock()})
    ids = [
        c.kwargs.get("id")
        for c in hub.scheduler.add_job.call_args_list
        if c.kwargs and "id" in c.kwargs
    ]
    assert "reconciliation" in ids


def test_digest_cron_registered_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.automation import runner as runner_mod

    hub = MagicMock()
    hub.scheduler = MagicMock()
    monkeypatch.setattr(runner_mod, "HubScheduler", lambda **kwargs: hub)

    def _raise_kb(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.time, "sleep", _raise_kb)

    wf = MagicMock()
    wf.run_ingest_only = MagicMock()
    monkeypatch.setattr(runner_mod, "create_trading_workflow", lambda *a, **k: wf)
    settings = _settings_with_local_paths(tmp_path)
    settings.scheduler.digest_cron = "0 9 * * 1"
    runner_mod._run_daemon(settings, {"default": MagicMock()})
    ids = [
        c.kwargs.get("id")
        for c in hub.scheduler.add_job.call_args_list
        if c.kwargs and "id" in c.kwargs
    ]
    assert "weekly_digest" in ids


def test_ml_retraining_cron_registered_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.automation import runner as runner_mod

    hub = MagicMock()
    hub.scheduler = MagicMock()
    monkeypatch.setattr(runner_mod, "HubScheduler", lambda **kwargs: hub)

    def _raise_kb(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.time, "sleep", _raise_kb)

    wf = MagicMock()
    wf.run_ingest_only = MagicMock()
    monkeypatch.setattr(runner_mod, "create_trading_workflow", lambda *a, **k: wf)
    settings = _settings_with_local_paths(tmp_path)
    settings.ml.enabled = True
    settings.scheduler.ml_retraining_cron = "0 2 * * 0"
    runner_mod._run_daemon(settings, {"default": MagicMock()})
    ids = [
        c.kwargs.get("id")
        for c in hub.scheduler.add_job.call_args_list
        if c.kwargs and "id" in c.kwargs
    ]
    assert "ml_retraining" in ids


def test_digest_cron_skipped_when_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.automation import runner as runner_mod

    hub = MagicMock()
    hub.scheduler = MagicMock()
    monkeypatch.setattr(runner_mod, "HubScheduler", lambda **kwargs: hub)

    def _raise_kb(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.time, "sleep", _raise_kb)

    wf = MagicMock()
    wf.run_ingest_only = MagicMock()
    monkeypatch.setattr(runner_mod, "create_trading_workflow", lambda *a, **k: wf)
    settings = _settings_with_local_paths(tmp_path)
    settings.scheduler.digest_cron = ""
    runner_mod._run_daemon(settings, {"default": MagicMock()})
    ids = [
        c.kwargs.get("id")
        for c in hub.scheduler.add_job.call_args_list
        if c.kwargs and "id" in c.kwargs
    ]
    assert "weekly_digest" not in ids


def test_reconcile_cron_skipped_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.automation import runner as runner_mod

    hub = MagicMock()
    hub.scheduler = MagicMock()
    monkeypatch.setattr(runner_mod, "HubScheduler", lambda **kwargs: hub)

    def _raise_kb(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.time, "sleep", _raise_kb)

    wf = MagicMock()
    wf.run_ingest_only = MagicMock()
    monkeypatch.setattr(runner_mod, "create_trading_workflow", lambda *a, **k: wf)
    settings = _settings_with_local_paths(tmp_path)
    settings.scheduler.reconcile_cron = ""
    runner_mod._run_daemon(settings, {"default": MagicMock()})
    ids = [
        c.kwargs.get("id")
        for c in hub.scheduler.add_job.call_args_list
        if c.kwargs and "id" in c.kwargs
    ]
    assert "reconciliation" not in ids


# -------- _abort_if_other_daemon_running: observation-based detection --------


def _settings_with_heartbeat_interval(seconds: int = 60) -> Settings:
    s = Settings(_yaml_path=None, _env_file=None)
    s.scheduler.heartbeat_interval_seconds = seconds
    return s


def test_abort_proceeds_when_heartbeat_missing(tmp_path: Path) -> None:
    # No prior heartbeat -- cleanest possible startup, must not abort or sleep.
    sleep_calls: list[float] = []
    _real_abort_if_other_daemon_running(
        _settings_with_heartbeat_interval(),
        heartbeat_path=tmp_path / "absent",
        sleep_fn=sleep_calls.append,
    )
    assert sleep_calls == []  # never observed -- no heartbeat to observe


def test_abort_proceeds_when_heartbeat_already_stale_by_age(
    tmp_path: Path,
) -> None:
    # Heartbeat older than threshold (interval*2 + 30 = 150s default) needs no
    # observation: the previous writer is definitely gone.
    hb = tmp_path / "hb"
    old = datetime.now(UTC) - timedelta(seconds=1000)
    hb.write_text(old.isoformat() + "\n", encoding="utf-8")
    sleep_calls: list[float] = []
    _real_abort_if_other_daemon_running(
        _settings_with_heartbeat_interval(),
        heartbeat_path=hb,
        sleep_fn=sleep_calls.append,
    )
    assert sleep_calls == []


def test_abort_proceeds_when_heartbeat_frozen_during_observation(
    tmp_path: Path,
) -> None:
    # The post-`docker compose restart` case: previous daemon's heartbeat
    # is fresh-by-age but no one is writing it anymore. Must NOT abort.
    hb = tmp_path / "hb"
    recent = datetime.now(UTC) - timedelta(seconds=5)
    hb.write_text(recent.isoformat() + "\n", encoding="utf-8")
    sleep_calls: list[float] = []
    # sleep_fn is the seam where we'd normally wait for the heartbeat to
    # advance; we don't touch the file during the fake sleep, so the
    # observation will see a frozen timestamp and proceed.
    _real_abort_if_other_daemon_running(
        _settings_with_heartbeat_interval(),
        heartbeat_path=hb,
        sleep_fn=sleep_calls.append,
    )
    assert sleep_calls, "expected observation sleep to occur for a fresh heartbeat"


def test_abort_proceeds_when_heartbeat_disappears_during_observation(
    tmp_path: Path,
) -> None:
    # Another path back to a clean start: the heartbeat is removed mid-observation
    # (e.g. user manually deletes it, or the other daemon shuts down cleanly).
    hb = tmp_path / "hb"
    recent = datetime.now(UTC) - timedelta(seconds=5)
    hb.write_text(recent.isoformat() + "\n", encoding="utf-8")

    def _remove_during_sleep(_secs: float) -> None:
        hb.unlink()

    _real_abort_if_other_daemon_running(
        _settings_with_heartbeat_interval(),
        heartbeat_path=hb,
        sleep_fn=_remove_during_sleep,
    )


def test_abort_raises_when_heartbeat_advances_during_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The genuine duplicate-daemon case: heartbeat timestamp moves forward
    # during the observation window. Must exit(2) and fire the CRITICAL alert.
    hb = tmp_path / "hb"
    first = datetime.now(UTC) - timedelta(seconds=5)
    hb.write_text(first.isoformat() + "\n", encoding="utf-8")

    def _advance_heartbeat_during_sleep(_secs: float) -> None:
        # Simulate another writer updating the heartbeat while we observe.
        newer = datetime.now(UTC)
        hb.write_text(newer.isoformat() + "\n", encoding="utf-8")

    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        runner_mod,
        "notify_operational_critical",
        lambda settings, *, category, message: alerts.append((category, message)),
    )

    with pytest.raises(SystemExit) as excinfo:
        _real_abort_if_other_daemon_running(  # type: ignore[misc]
            _settings_with_heartbeat_interval(),
            heartbeat_path=hb,
            sleep_fn=_advance_heartbeat_during_sleep,
        )
    assert excinfo.value.code == 2
    assert len(alerts) == 1, "expected exactly one CRITICAL alert per abort"
    assert alerts[0][0] == "duplicate_scheduler"


def test_run_daemon_clears_heartbeat_on_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The matching write-side guarantee: a clean shutdown (Ctrl+C / SIGTERM)
    # must delete the heartbeat so the next start has nothing to observe.
    hub = MagicMock()
    hub.scheduler = MagicMock()
    monkeypatch.setattr(runner_mod, "HubScheduler", lambda **kwargs: hub)

    def _raise_kb(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.time, "sleep", _raise_kb)
    wf = MagicMock()
    wf.run_ingest_only = MagicMock()
    monkeypatch.setattr(runner_mod, "create_trading_workflow", lambda *a, **k: wf)
    hb = tmp_path / "hb"
    hb.write_text(datetime.now(UTC).isoformat() + "\n", encoding="utf-8")
    monkeypatch.setattr(runner_mod, "HEARTBEAT_FILE", hb)

    settings = _settings_with_local_paths(tmp_path)
    settings.scheduler.reconcile_cron = ""
    runner_mod._run_daemon(settings, {"default": MagicMock()})

    assert not hb.exists(), "graceful shutdown must remove the heartbeat file"
