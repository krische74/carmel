"""Tests for market regime detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.config import RegimeConfig, Settings
from src.data.regime import RegimeDetector
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore

if TYPE_CHECKING:
    from pathlib import Path


def _settings(**kwargs: object) -> Settings:
    return Settings(_yaml_path=None, _env_file=None, regime=RegimeConfig(**kwargs))


def _ohlcv_close(last: float) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    c = float(last)
    return pd.DataFrame(
        {
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "volume": 1e6,
        },
        index=idx,
    )


def _detector(
    tmp_path: Path, **reg_kw: object
) -> tuple[RegimeDetector, ParquetStore, SQLiteStore]:
    s = _settings(**reg_kw)
    pq = ParquetStore(tmp_path / "pq")
    sql = SQLiteStore(tmp_path / "hub.db")
    return RegimeDetector(s), pq, sql


def test_regime_risk_on_normal_curve_low_vix(tmp_path: Path) -> None:
    from src.data.regime import VolatilityRegime

    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(13.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 4.5)
    sql.write_macro_indicator("2024-06-01", "DGS2", 2.0)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "risk_on"
    assert r.volatility == VolatilityRegime.LOW
    assert r.sizing_multiplier == pytest.approx(1.0)


def test_regime_cautious_elevated_vix(tmp_path: Path) -> None:
    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(25.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 4.5)
    sql.write_macro_indicator("2024-06-01", "DGS2", 2.0)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "cautious"
    assert r.sizing_multiplier == pytest.approx(0.75)


def test_regime_cautious_flat_curve(tmp_path: Path) -> None:
    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(16.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 2.5)
    sql.write_macro_indicator("2024-06-01", "DGS2", 2.2)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "cautious"


def test_regime_defensive_inverted_curve(tmp_path: Path) -> None:
    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(18.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 3.0)
    sql.write_macro_indicator("2024-06-01", "DGS2", 4.0)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "defensive"
    assert r.sizing_multiplier == pytest.approx(0.5)


def test_regime_defensive_flat_and_elevated(tmp_path: Path) -> None:
    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(28.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 2.6)
    sql.write_macro_indicator("2024-06-01", "DGS2", 2.4)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "defensive"


def test_regime_crisis_high_vix(tmp_path: Path) -> None:
    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(35.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 4.5)
    sql.write_macro_indicator("2024-06-01", "DGS2", 2.0)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "crisis"
    assert r.sizing_multiplier == pytest.approx(0.25)


def test_regime_defaults_normal_when_data_missing(tmp_path: Path) -> None:
    det, pq, sql = _detector(tmp_path)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "risk_on"
    assert r.vix_close is None
    assert r.yield_spread is None


def test_regime_handles_missing_vix_only(tmp_path: Path) -> None:
    det, pq, sql = _detector(tmp_path)
    sql.write_macro_indicator("2024-06-01", "DGS10", 3.0)
    sql.write_macro_indicator("2024-06-01", "DGS2", 4.0)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "defensive"


def test_regime_handles_missing_fred_only(tmp_path: Path) -> None:
    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(25.0))
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.overall.value == "cautious"


def test_regime_vix_exact_low_boundary(tmp_path: Path) -> None:
    """VIX == vix_low (15.0) is NORMAL, not LOW."""
    from src.data.regime import VolatilityRegime

    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(15.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 4.5)
    sql.write_macro_indicator("2024-06-01", "DGS2", 2.0)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.volatility == VolatilityRegime.NORMAL


def test_regime_vix_exact_elevated_boundary(tmp_path: Path) -> None:
    """VIX == vix_elevated (30.0) is CRISIS, not ELEVATED."""
    from src.data.regime import VolatilityRegime

    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(30.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 4.5)
    sql.write_macro_indicator("2024-06-01", "DGS2", 2.0)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.volatility == VolatilityRegime.CRISIS


def test_regime_yield_spread_exact_flat_boundary(tmp_path: Path) -> None:
    """Spread == flat_threshold (0.5) is FLAT, not NORMAL."""
    from src.data.regime import YieldCurveRegime

    det, pq, sql = _detector(tmp_path)
    pq.write_ohlcv("^VIX", _ohlcv_close(13.0))
    sql.write_macro_indicator("2024-06-01", "DGS10", 2.5)
    sql.write_macro_indicator("2024-06-01", "DGS2", 2.0)
    r = det.detect(parquet_store=pq, sqlite_store=sql)
    assert r.yield_curve == YieldCurveRegime.FLAT


def test_build_market_regime_snapshot_matches_spread_and_vix() -> None:
    """Snapshot helper agrees with classification used by RegimeDetector."""
    from datetime import UTC, datetime

    from src.data.regime import (
        OverallRegime,
        VolatilityRegime,
        YieldCurveRegime,
        build_market_regime_snapshot,
    )

    s = _settings()
    when = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
    r = build_market_regime_snapshot(
        s,
        as_of=when,
        vix_close=13.0,
        yield_spread=1.0,
    )
    assert r.vix_close == pytest.approx(13.0)
    assert r.yield_spread == pytest.approx(1.0)
    assert r.volatility == VolatilityRegime.LOW
    assert r.yield_curve == YieldCurveRegime.NORMAL
    assert r.overall == OverallRegime.RISK_ON


def test_build_market_regime_snapshot_vix_only_defaults_yield_normal() -> None:
    from datetime import UTC, datetime

    from src.data.regime import OverallRegime, VolatilityRegime, build_market_regime_snapshot

    s = _settings()
    when = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
    r = build_market_regime_snapshot(s, as_of=when, vix_close=35.0, yield_spread=None)
    assert r.volatility == VolatilityRegime.CRISIS
    assert r.yield_spread is None
    assert r.overall == OverallRegime.CRISIS


def test_build_market_regime_snapshot_rejects_non_settings() -> None:
    from datetime import UTC, datetime

    from src.data.regime import build_market_regime_snapshot

    when = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
    with pytest.raises(TypeError, match="settings must be"):
        build_market_regime_snapshot("not-settings", as_of=when, vix_close=20.0, yield_spread=None)  # type: ignore[arg-type]
