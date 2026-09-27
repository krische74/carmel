"""Unit tests for MeanReversionStrategy."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.config import DataConfig, MeanReversionConfig, Settings, StrategyConfig
from src.data.regime import (
    MarketRegime,
    OverallRegime,
    VolatilityRegime,
    YieldCurveRegime,
)
from src.strategy.mean_reversion import MeanReversionStrategy

if TYPE_CHECKING:
    from pathlib import Path


def _ohlcv_from_close(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
        },
        index=close.index,
    )


def _settings_mr(**kwargs: object) -> Settings:
    mcfg = MeanReversionConfig(**kwargs) if kwargs else MeanReversionConfig()
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=["SPY"]),
        strategy=StrategyConfig(mean_reversion=mcfg),
    )


def test_mean_reversion_emits_long_when_below_lower_band_and_rsi_oversold() -> None:
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 100.0, n)
    base[-10:] = np.linspace(100.0, 55.0, 10)
    close = pd.Series(base, index=idx)
    df = _ohlcv_from_close(close)
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        rsi_oversold=45.0,
        rsi_overbought=70.0,
        universe=["SPY"],
    )
    strat = MeanReversionStrategy(settings)
    sigs = strat.generate_signals({"SPY": df}, as_of=datetime(2024, 4, 15, tzinfo=UTC))
    longs = [s for s in sigs if s.direction == "long"]
    assert len(longs) >= 1
    assert longs[0].symbol == "SPY"
    assert longs[0].weight > 0


def test_mean_reversion_emits_flat_when_above_upper_band() -> None:
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 100.0, n)
    base[-1] = 130.0
    close = pd.Series(base, index=idx)
    df = _ohlcv_from_close(close)
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        universe=["SPY"],
    )
    strat = MeanReversionStrategy(settings)
    sigs = strat.generate_signals({"SPY": df}, as_of=datetime(2024, 4, 15, tzinfo=UTC))
    flats = [s for s in sigs if s.direction == "flat"]
    assert len(flats) >= 1


def test_mean_reversion_respects_sma_trend_guard() -> None:
    n = 120
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 50.0, n)
    close = pd.Series(base, index=idx)
    df = _ohlcv_from_close(close)
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=20,
        rsi_oversold=50.0,
        universe=["SPY"],
    )
    strat = MeanReversionStrategy(settings)
    sigs = strat.generate_signals({"SPY": df}, as_of=datetime(2024, 7, 1, tzinfo=UTC))
    longs = [s for s in sigs if s.direction == "long"]
    assert longs == []


def _crisis_regime_mr(as_of: datetime) -> MarketRegime:
    return MarketRegime(
        timestamp=as_of,
        vix_close=55.0,
        yield_spread=None,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.CRISIS,
        overall=OverallRegime.CRISIS,
        sizing_multiplier=0.25,
    )


def _overall_regime_mr(as_of: datetime, overall: OverallRegime) -> MarketRegime:
    return MarketRegime(
        timestamp=as_of,
        vix_close=20.0,
        yield_spread=None,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=overall,
        sizing_multiplier=1.0,
    )


def test_mean_reversion_signals_capped_cautious_regime() -> None:
    """Six dip patterns; cautious regime caps at 3 positions (defaults)."""
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 100.0, n)
    base[-10:] = np.linspace(100.0, 55.0, 10)
    close = pd.Series(base, index=idx)
    syms = [f"S{i}" for i in range(6)]
    data = {s: _ohlcv_from_close(close) for s in syms}
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        rsi_oversold=45.0,
        max_positions=4,
        universe=syms,
    )
    strat = MeanReversionStrategy(settings)
    as_of = datetime(2024, 4, 15, tzinfo=UTC)
    sigs = strat.generate_signals(
        data,
        as_of=as_of,
        market_regime=_overall_regime_mr(as_of, OverallRegime.CAUTIOUS),
    )
    longs = [s for s in sigs if s.direction == "long"]
    assert len(longs) == 3
    assert all(s.weight == pytest.approx(1.0 / 3.0) for s in longs)


def test_mean_reversion_signals_capped_defensive_regime() -> None:
    """Six dip patterns; defensive regime caps at 2 positions (defaults)."""
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 100.0, n)
    base[-10:] = np.linspace(100.0, 55.0, 10)
    close = pd.Series(base, index=idx)
    syms = [f"S{i}" for i in range(6)]
    data = {s: _ohlcv_from_close(close) for s in syms}
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        rsi_oversold=45.0,
        max_positions=4,
        universe=syms,
    )
    strat = MeanReversionStrategy(settings)
    as_of = datetime(2024, 4, 15, tzinfo=UTC)
    sigs = strat.generate_signals(
        data,
        as_of=as_of,
        market_regime=_overall_regime_mr(as_of, OverallRegime.DEFENSIVE),
    )
    longs = [s for s in sigs if s.direction == "long"]
    assert len(longs) == 2
    assert all(s.weight == pytest.approx(0.5) for s in longs)


def test_mean_reversion_signals_risk_on_matches_no_regime() -> None:
    """RISK_ON uses same caps as base ``max_positions`` — identical long picks."""
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 100.0, n)
    base[-10:] = np.linspace(100.0, 55.0, 10)
    close = pd.Series(base, index=idx)
    syms = [f"S{i}" for i in range(6)]
    data = {s: _ohlcv_from_close(close) for s in syms}
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        rsi_oversold=45.0,
        max_positions=4,
        universe=syms,
    )
    strat = MeanReversionStrategy(settings)
    as_of = datetime(2024, 4, 15, tzinfo=UTC)
    sigs_base = strat.generate_signals(data, as_of=as_of)
    sigs_risk = strat.generate_signals(
        data,
        as_of=as_of,
        market_regime=_overall_regime_mr(as_of, OverallRegime.RISK_ON),
    )
    longs_b = sorted((s.symbol, s.weight) for s in sigs_base if s.direction == "long")
    longs_r = sorted((s.symbol, s.weight) for s in sigs_risk if s.direction == "long")
    assert longs_b == longs_r


def test_mean_reversion_signals_capped_by_regime() -> None:
    """Six dip patterns; base max 4 but crisis regime caps at 1."""
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 100.0, n)
    base[-10:] = np.linspace(100.0, 55.0, 10)
    close = pd.Series(base, index=idx)
    syms = [f"S{i}" for i in range(6)]
    data = {s: _ohlcv_from_close(close) for s in syms}
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        rsi_oversold=45.0,
        max_positions=4,
        universe=syms,
    )
    strat = MeanReversionStrategy(settings)
    as_of = datetime(2024, 4, 15, tzinfo=UTC)
    sigs = strat.generate_signals(
        data,
        as_of=as_of,
        market_regime=_crisis_regime_mr(as_of),
    )
    longs = [s for s in sigs if s.direction == "long"]
    assert len(longs) == 1
    assert longs[0].weight == pytest.approx(1.0)
    assert "Regime cap" in longs[0].rationale


def test_mean_reversion_signals_without_regime_unchanged() -> None:
    """Same panel as regime cap test but no ``market_regime`` → base max_positions=4."""
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 100.0, n)
    base[-10:] = np.linspace(100.0, 55.0, 10)
    close = pd.Series(base, index=idx)
    syms = [f"S{i}" for i in range(6)]
    data = {s: _ohlcv_from_close(close) for s in syms}
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        rsi_oversold=45.0,
        max_positions=4,
        universe=syms,
    )
    strat = MeanReversionStrategy(settings)
    sigs = strat.generate_signals(data, as_of=datetime(2024, 4, 15, tzinfo=UTC))
    longs = [s for s in sigs if s.direction == "long"]
    assert len(longs) == 4
    assert all(s.weight == pytest.approx(0.25) for s in longs)
    assert all("Regime cap" not in s.rationale for s in longs)


def test_mean_reversion_caps_positions_to_max() -> None:
    """Six identical dip patterns: only ``max_positions`` longs with equal weight."""
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.linspace(100.0, 100.0, n)
    base[-10:] = np.linspace(100.0, 55.0, 10)
    close = pd.Series(base, index=idx)
    syms = [f"S{i}" for i in range(6)]
    data = {s: _ohlcv_from_close(close) for s in syms}
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        rsi_oversold=45.0,
        max_positions=4,
        universe=syms,
    )
    strat = MeanReversionStrategy(settings)
    sigs = strat.generate_signals(data, as_of=datetime(2024, 4, 15, tzinfo=UTC))
    longs = [s for s in sigs if s.direction == "long"]
    assert len(longs) == 4
    assert all(s.weight == pytest.approx(0.25) for s in longs)


def test_mean_reversion_returns_empty_when_no_candidates() -> None:
    n = 80
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    close = pd.Series(np.full(n, 100.0), index=idx)
    df = _ohlcv_from_close(close)
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        universe=["SPY"],
    )
    strat = MeanReversionStrategy(settings)
    sigs = strat.generate_signals({"SPY": df}, as_of=datetime(2024, 4, 15, tzinfo=UTC))
    assert [s for s in sigs if s.direction == "long"] == []


def test_mean_reversion_get_universe_from_config() -> None:
    settings = _settings_mr(universe=["AAA", "BBB"])
    strat = MeanReversionStrategy(settings)
    assert strat.get_universe() == ["AAA", "BBB"]


def test_mean_reversion_backtest_produces_trades(tmp_path: Path) -> None:
    settings = _settings_mr(
        bb_period=10,
        rsi_period=8,
        sma_trend_period=0,
        rsi_oversold=40.0,
        universe=["SPY"],
    )
    strat = MeanReversionStrategy(settings)
    n = 120
    idx = pd.bdate_range("2024-01-02", periods=n, freq="B")
    base = np.concatenate([np.linspace(100.0, 100.0, 80), np.linspace(100.0, 75.0, 40)])
    close = pd.Series(base, index=idx)
    df = _ohlcv_from_close(close)
    data = {"SPY": df}
    engine = BacktestEngine()
    result = engine.run(
        strat,
        data,
        start=date(2024, 1, 2),
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly", slippage_bps=2.0),
    )
    assert len(result.trades) > 0
    assert result.final_equity > 0
