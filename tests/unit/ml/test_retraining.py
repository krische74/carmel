"""Scheduled ML retraining (Tier 25)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.config import DataConfig, MLConfig, Settings, StrategyConfig
from src.data.storage.parquet_store import ParquetStore
from src.ml.retraining import retrain_ml_bundle
from tests.unit.ml.test_features import _synthetic_ohlcv


def test_retrain_ml_bundle_returns_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("sklearn")
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

    df = _synthetic_ohlcv(220)
    pq_root = tmp_path / "data" / "parquet"
    pq_root.mkdir(parents=True)
    store = ParquetStore(pq_root)
    store.write_ohlcv("SPY", df)
    store.write_ohlcv("QQQ", df)
    store.write_ohlcv("TLT", df)
    store.write_ohlcv("GLD", df)
    store.write_ohlcv("SHV", df)

    end_d = pd.Timestamp(df.index[-15]).date()

    class _FixedDate:
        @staticmethod
        def today() -> date:
            return end_d

    monkeypatch.setattr("src.ml.retraining.date", _FixedDate)

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(pq_root),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "QQQ", "TLT", "GLD", "SHV"],
        ),
        strategy=StrategyConfig(),
        ml=MLConfig(
            model_path="ml/cron_train.joblib",
            min_train_rows=30,
            label_horizon_days=3,
            retraining_lookback_days=400,
        ),
    )

    out = retrain_ml_bundle(settings)
    assert out is not None
    assert Path(out).is_file()


def test_retrain_insufficient_data_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("sklearn")
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

    df = _synthetic_ohlcv(80)
    pq_root = tmp_path / "data" / "parquet"
    pq_root.mkdir(parents=True)
    store = ParquetStore(pq_root)
    for sym in ("SPY", "QQQ", "TLT", "GLD", "SHV"):
        store.write_ohlcv(sym, df)

    end_d = pd.Timestamp(df.index[-5]).date()

    class _FixedDate2:
        @staticmethod
        def today() -> date:
            return end_d

    monkeypatch.setattr("src.ml.retraining.date", _FixedDate2)

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(pq_root),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "QQQ", "TLT", "GLD", "SHV"],
        ),
        strategy=StrategyConfig(),
        ml=MLConfig(
            model_path="ml/too_short.joblib",
            min_train_rows=500,
            label_horizon_days=3,
            retraining_lookback_days=30,
        ),
    )

    assert retrain_ml_bundle(settings) is None
