"""Tests for ``carmel ingest`` CLI."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

from src.automation import runner as runner_mod
from src.config import (
    CashSweepConfig,
    DataConfig,
    DCATarget,
    MeanReversionConfig,
    MomentumConfig,
    Settings,
    StrategyConfig,
    TaxConfig,
)
from src.data.pipeline import IngestResult
from src.data.symbol_resolver import resolve_required_symbols


def test_ingest_cli_uses_resolved_symbol_set_when_no_symbols(monkeypatch, tmp_path) -> None:
    """Default ingest uses ``resolve_required_symbols`` (not only universe + DCA)."""
    missing = tmp_path / "no.yaml"
    assert not missing.exists()
    settings = Settings(
        _yaml_path=missing,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "QQQ"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol="SHV"),
            mean_reversion=MeanReversionConfig(universe=["TLT", "GLD"]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    mock_pipeline = MagicMock()
    mock_pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    monkeypatch.setattr(runner_mod, "DataPipeline", MagicMock(return_value=mock_pipeline))

    code = runner_mod.main(["ingest", "--years", "1", "--end", "2020-12-31"])
    assert code == 0
    expected = sorted(resolve_required_symbols(settings))
    got = [c[0][0] for c in mock_pipeline.ingest_ohlcv.call_args_list]
    assert sorted(got) == expected
    assert mock_pipeline.ingest_ohlcv.call_count == len(expected)


def test_ingest_cli_years_arg(monkeypatch, tmp_path) -> None:
    yml = tmp_path / "years.yaml"
    assert not yml.exists()
    settings = Settings(
        _yaml_path=yml,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
        cash_sweep=CashSweepConfig(enabled=False),  # single-symbol ingest; sweep off
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    mock_pipeline = MagicMock()
    mock_pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    monkeypatch.setattr(runner_mod, "DataPipeline", MagicMock(return_value=mock_pipeline))

    end = date(2020, 12, 31)
    code = runner_mod.main(["ingest", "--years", "5", "--end", end.isoformat()])
    assert code == 0
    assert mock_pipeline.ingest_ohlcv.call_count == 1
    c = mock_pipeline.ingest_ohlcv.call_args
    assert c[0][0] == "SPY"
    assert c[1]["start"] == end - timedelta(days=365 * 5)
    assert c[1]["end"] == end


def test_ingest_cli_explicit_symbols_overrides_resolver(monkeypatch, tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "QQQ", "TLT"],
            dca_targets=[],
        ),
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    mock_pipeline = MagicMock()
    mock_pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    monkeypatch.setattr(runner_mod, "DataPipeline", MagicMock(return_value=mock_pipeline))

    code = runner_mod.main(
        [
            "ingest",
            "--symbols",
            "SPY,QQQ",
            "--start",
            "2023-01-01",
            "--end",
            "2023-12-31",
        ],
    )
    assert code == 0
    got = [c[0][0] for c in mock_pipeline.ingest_ohlcv.call_args_list]
    assert got == ["QQQ", "SPY"]


def test_ingest_cli_handles_failures(monkeypatch, tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "QQQ"],
            dca_targets=[],
        ),
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    mock_pipeline = MagicMock()
    mock_pipeline.ingest_ohlcv.side_effect = [
        IngestResult(success=True),
        IngestResult(success=False, error="network"),
    ]
    monkeypatch.setattr(runner_mod, "DataPipeline", MagicMock(return_value=mock_pipeline))

    code = runner_mod.main(
        [
            "ingest",
            "--symbols",
            "SPY,QQQ",
            "--start",
            "2023-01-01",
            "--end",
            "2023-06-01",
        ],
    )
    assert code == 1


def test_ingest_cli_includes_momentum_cash_symbol(monkeypatch, tmp_path) -> None:
    """Regression: SHV must be ingested when configured only as momentum cash (not in universe)."""
    missing = tmp_path / "no2.yaml"
    assert not missing.exists()
    settings = Settings(
        _yaml_path=missing,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol="SHV"),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    monkeypatch.setattr(runner_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(runner_mod, "setup_logging", lambda *a, **k: None)
    mock_pipeline = MagicMock()
    mock_pipeline.ingest_ohlcv.return_value = IngestResult(success=True)
    monkeypatch.setattr(runner_mod, "DataPipeline", MagicMock(return_value=mock_pipeline))

    code = runner_mod.main(["ingest", "--years", "1", "--end", "2020-12-31"])
    assert code == 0
    got = {c[0][0] for c in mock_pipeline.ingest_ohlcv.call_args_list}
    assert "SHV" in got
    assert "SPY" in got
