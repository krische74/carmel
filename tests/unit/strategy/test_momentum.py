"""Unit tests for MomentumRotationStrategy."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.config import DataConfig, MomentumConfig, Settings, StrategyConfig
from src.strategy.as_of import slice_to_as_of
from src.strategy.momentum import MomentumRotationStrategy


def _make_trending_close(start: float, daily_return: float, n: int) -> pd.Series:
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    levels = [start]
    for _ in range(1, n):
        levels.append(levels[-1] * (1.0 + daily_return))
    return pd.Series(levels, index=idx)


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


def _synthetic_panel() -> dict[str, pd.DataFrame]:
    """SPY strongest momentum; QQQ second; TLT/GLD weaker — all above 200SMA."""
    # Need > 200 (SMA) + 252 (12m horizon) + buffer for stable scores.
    n = 500
    spy_close = _make_trending_close(100.0, 0.0015, n)
    qqq_close = _make_trending_close(100.0, 0.0012, n)
    tlt_close = _make_trending_close(100.0, 0.0004, n)
    gld_close = _make_trending_close(100.0, 0.0003, n)
    shv_close = _make_trending_close(50.0, 0.0001, n)
    return {
        "SPY": _ohlcv_from_close(spy_close),
        "QQQ": _ohlcv_from_close(qqq_close),
        "TLT": _ohlcv_from_close(tlt_close),
        "GLD": _ohlcv_from_close(gld_close),
        "SHV": _ohlcv_from_close(shv_close),
    }


def test_momentum_empty_data_returns_no_signals() -> None:
    settings = Settings(
        data=DataConfig(universe=["SPY", "QQQ", "TLT", "GLD"]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                lookback_months=[1, 3, 6, 12],
                sma_filter_period=200,
                cash_symbol="SHV",
            ),
        ),
    )
    strat = MomentumRotationStrategy(settings=settings)
    assert strat.generate_signals({}, as_of=datetime(2024, 1, 2, tzinfo=UTC)) == []


def test_momentum_slice_to_as_of_handles_tz_aware_index() -> None:
    n = 500
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    close = pd.Series(np.linspace(100.0, 200.0, n), index=idx)
    df = _ohlcv_from_close(close)
    as_of = datetime(2021, 6, 15, 16, 30, tzinfo=UTC)
    sliced = slice_to_as_of(df, as_of)
    assert not sliced.empty
    assert sliced.index.tz is None
    assert sliced.index[-1] <= pd.Timestamp("2021-06-15").normalize()


def test_momentum_ranks_spy_first_on_synthetic_data() -> None:
    settings = Settings(
        data=DataConfig(
            universe=["SPY", "QQQ", "TLT", "GLD"],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                lookback_months=[1, 3, 6, 12],
                sma_filter_period=200,
                cash_symbol="SHV",
            ),
        ),
    )
    strat = MomentumRotationStrategy(settings=settings)
    data = _synthetic_panel()
    as_of = data["SPY"].index[-1].to_pydatetime().replace(tzinfo=UTC)
    signals = strat.generate_signals(data, as_of=as_of)
    assert len(signals) >= 1
    top = signals[0]
    assert top.symbol == "SPY"
    assert top.direction == "long"
    assert top.strategy_name == "MomentumRotationStrategy"
    assert len(top.rationale) >= 30


def test_momentum_all_below_sma_goes_to_cash() -> None:
    n = 500
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    # Strong downtrend: last close is well below the 200-day SMA for each series.
    close = pd.Series(np.linspace(150.0, 40.0, n), index=idx)

    def panel_for(factor: float) -> pd.DataFrame:
        return _ohlcv_from_close(close * factor)

    data = {
        "SPY": panel_for(1.0),
        "QQQ": panel_for(0.99),
        "TLT": panel_for(1.01),
        "GLD": panel_for(0.98),
    }
    data["SHV"] = _ohlcv_from_close(_make_trending_close(50.0, 0.00005, n))

    settings = Settings(
        data=DataConfig(universe=["SPY", "QQQ", "TLT", "GLD"]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                sma_filter_period=200,
                cash_symbol="SHV",
                lookback_months=[1, 3, 6, 12],
            ),
        ),
    )
    strat = MomentumRotationStrategy(settings=settings)
    as_of = data["SPY"].index[-1].to_pydatetime().replace(tzinfo=UTC)
    signals = strat.generate_signals(data, as_of=as_of)
    assert any(s.symbol == "SHV" and s.direction == "cash" for s in signals)


def test_momentum_get_universe_includes_cash_symbol() -> None:
    settings = Settings(
        data=DataConfig(universe=["SPY", "QQQ", "TLT", "GLD"]),
        strategy=StrategyConfig(momentum=MomentumConfig(cash_symbol="SHV")),
    )
    strat = MomentumRotationStrategy(settings=settings)
    u = strat.get_universe()
    assert set(["SPY", "QQQ", "TLT", "GLD", "SHV"]).issubset(set(u))


def _panel_spy_choppy_high_score_qqq_trending() -> dict[str, pd.DataFrame]:
    """SPY ends in chop (low ADX) but 12m momentum can beat QQQ; QQQ stays trending (ADX > 25)."""
    n = 500
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    spy_early = np.linspace(100.0, 240.0, 360)
    t = np.arange(140)
    spy_late = 240.0 + 0.25 * np.sin(t / 3.0)
    spy_close = pd.Series(np.concatenate([spy_early, spy_late]), index=idx)
    qqq_close = pd.Series(np.linspace(100.0, 175.0, n), index=idx)
    shv_close = _make_trending_close(50.0, 0.0001, n)
    return {
        "SPY": _ohlcv_from_close(spy_close),
        "QQQ": _ohlcv_from_close(qqq_close),
        "SHV": _ohlcv_from_close(shv_close),
    }


def test_momentum_passes_symbol_with_high_adx() -> None:
    settings = Settings(
        data=DataConfig(universe=["SPY", "QQQ", "TLT", "GLD"]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                lookback_months=[1, 3, 6, 12],
                sma_filter_period=200,
                cash_symbol="SHV",
                adx_threshold=25.0,
                adx_filter_period=14,
            ),
        ),
    )
    strat = MomentumRotationStrategy(settings=settings)
    data = _synthetic_panel()
    as_of = data["SPY"].index[-1].to_pydatetime().replace(tzinfo=UTC)
    signals = strat.generate_signals(data, as_of=as_of)
    assert len(signals) >= 1
    assert signals[0].direction == "long"
    assert signals[0].symbol == "SPY"


def test_momentum_filters_symbol_with_low_adx() -> None:
    settings = Settings(
        data=DataConfig(universe=["SPY", "QQQ"]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                lookback_months=[1, 3, 6, 12],
                sma_filter_period=200,
                cash_symbol="SHV",
                adx_threshold=25.0,
                adx_filter_period=14,
            ),
        ),
    )
    strat = MomentumRotationStrategy(settings=settings)
    data = _panel_spy_choppy_high_score_qqq_trending()
    as_of = data["SPY"].index[-1].to_pydatetime().replace(tzinfo=UTC)
    signals = strat.generate_signals(data, as_of=as_of)
    assert len(signals) >= 1
    top = signals[0]
    assert top.direction == "long"
    assert top.symbol == "QQQ"


def test_momentum_all_below_adx_goes_to_cash() -> None:
    n = 500
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    ramp = np.linspace(70.0, 125.0, 240)
    t = np.arange(260)
    chop = 125.0 + 0.15 * np.sin(t / 2.8)
    close = pd.Series(np.concatenate([ramp, chop]), index=idx)

    def pan(f: float) -> pd.DataFrame:
        return _ohlcv_from_close(close * f)

    data = {
        "SPY": pan(1.0),
        "QQQ": pan(0.99),
        "TLT": pan(1.01),
        "GLD": pan(0.98),
    }
    data["SHV"] = _ohlcv_from_close(_make_trending_close(50.0, 0.00005, n))

    settings = Settings(
        data=DataConfig(universe=["SPY", "QQQ", "TLT", "GLD"]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                sma_filter_period=200,
                cash_symbol="SHV",
                lookback_months=[1, 3, 6, 12],
                adx_threshold=25.0,
                adx_filter_period=14,
            ),
        ),
    )
    strat = MomentumRotationStrategy(settings=settings)
    as_of = data["SPY"].index[-1].to_pydatetime().replace(tzinfo=UTC)
    signals = strat.generate_signals(data, as_of=as_of)
    assert any(s.symbol == "SHV" and s.direction == "cash" for s in signals)


def test_momentum_regime_can_raise_adx_gate_to_cash() -> None:
    """Very high effective ADX threshold forces cash when no name passes."""
    from unittest.mock import patch

    from src.data.regime import (
        MarketRegime,
        OverallRegime,
        VolatilityRegime,
        YieldCurveRegime,
    )

    settings = Settings(
        data=DataConfig(universe=["SPY", "QQQ", "TLT", "GLD"]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                sma_filter_period=200,
                cash_symbol="SHV",
                lookback_months=[1, 3, 6, 12],
                adx_threshold=25.0,
                adx_filter_period=14,
            ),
        ),
    )
    strat = MomentumRotationStrategy(settings=settings)
    data = _synthetic_panel()
    as_of = data["SPY"].index[-1].to_pydatetime().replace(tzinfo=UTC)
    mr = MarketRegime(
        timestamp=as_of,
        vix_close=20.0,
        yield_spread=0.5,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.CRISIS,
        sizing_multiplier=0.25,
    )
    with patch("src.strategy.regime_params.effective_adx_threshold", return_value=100.0):
        signals = strat.generate_signals(data, as_of=as_of, market_regime=mr)
    assert any(s.direction == "cash" for s in signals)
