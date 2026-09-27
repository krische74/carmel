"""Unit tests for the strategy backtesting engine."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path  # noqa: TC003

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import (
    BacktestConfig,
    BacktestDataMissingError,
    BacktestEngine,
    _close_price_from_normalized,
    _daily_returns_from_equity_curve,
    _normalize_ohlcv_index,
    _rebalance_dates,
    trading_days_in_range,
    vix_close_as_of,
)
from src.config import DataConfig, MomentumConfig, Settings, StrategyConfig
from src.models import Signal
from src.strategy.base import Strategy
from src.strategy.momentum import MomentumRotationStrategy


class _NoSignalsStrategy(Strategy):
    """Never emits signals (drives zero-trade backtests)."""

    def get_universe(self) -> list[str]:
        return ["SPY"]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: object | None = None,
    ) -> list[Signal]:
        return []


class _AlternatingLongStrategy(Strategy):
    """Alternates full long between two symbols on each signal call (drives turnover)."""

    def __init__(self) -> None:
        self._prefer_qqq = False

    def get_universe(self) -> list[str]:
        return ["SPY", "QQQ"]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: object | None = None,
    ) -> list[Signal]:
        when = as_of or datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        sym = "QQQ" if self._prefer_qqq else "SPY"
        self._prefer_qqq = not self._prefer_qqq
        return [
            Signal(
                symbol=sym,
                direction="long",
                weight=1.0,
                confidence=0.5,
                rationale="Synthetic alternating target for backtest frequency tests.",
                timestamp=when,
                strategy_name="AlternatingLongStrategy",
            ),
        ]


def _test_momentum_settings(tmp_path: Path) -> Settings:
    """Relaxed momentum params so synthetic series can pass filters with fewer bars."""
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "QQQ"],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                lookback_months=[1],
                sma_filter_period=10,
                cash_symbol="SHV",
                adx_filter_period=5,
                adx_threshold=0.0,
            ),
        ),
    )


def _trending_ohlcv(
    n: int,
    start: date,
    *,
    close_start: float = 100.0,
    close_drift: float = 0.3,
) -> pd.DataFrame:
    """Upward-trending OHLCV with enough bars for momentum + ADX."""
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = close_start + np.linspace(0.0, close_drift * n, n, dtype=float)
    high = close + 1.5
    low = close - 1.0
    open_ = np.r_[close[0], close[:-1]]
    vol = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_daily_returns_aligns_dates_when_prior_equity_zero() -> None:
    """Tier 49A: skipped zero-equity days must not shift later return labels."""
    curve = [
        {"date": "2024-01-02", "equity": 10_000.0},
        {"date": "2024-01-03", "equity": 0.0},
        {"date": "2024-01-04", "equity": 100.0},
        {"date": "2024-01-05", "equity": 110.0},
    ]
    rets = _daily_returns_from_equity_curve(curve)
    assert rets.index.tolist() == ["2024-01-03", "2024-01-05"]
    assert rets.loc["2024-01-05"] == pytest.approx(0.1)
    assert "2024-01-04" not in rets.index


def test_backtest_forward_fills_missing_marks_to_avoid_zero_equity(tmp_path: Path) -> None:
    """Held positions with a missing bar use the last mark — equity must not drop to cash-only zero."""
    settings = _test_momentum_settings(tmp_path)
    strat = MomentumRotationStrategy(settings)
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.5)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.1)
    shv = _trending_ohlcv(n, start_d, close_drift=0.01)
    # Drop one SHV bar mid-series to simulate a missing mark.
    shv = shv.drop(shv.index[40])
    data = {"SPY": spy, "QQQ": qqq, "SHV": shv}
    result = BacktestEngine().run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly", use_regime=False),
    )
    zero_days = [p for p in result.equity_curve if float(p["equity"]) <= 0.0]
    assert not zero_days


def test_backtest_engine_raises_on_no_data_in_range(tmp_path: Path) -> None:
    settings = _test_momentum_settings(tmp_path)
    strat = MomentumRotationStrategy(settings)
    engine = BacktestEngine()
    start = date(2024, 1, 2)
    end = date(2024, 6, 28)
    with pytest.raises(BacktestDataMissingError, match="No OHLCV data found"):
        engine.run(strat, {}, start=start, end=end, config=BacktestConfig())


def test_backtest_momentum_rotation_executes_trades(tmp_path: Path) -> None:
    settings = _test_momentum_settings(tmp_path)
    strat = MomentumRotationStrategy(settings)
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.5)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.1)
    shv = _trending_ohlcv(n, start_d, close_drift=0.01)
    data = {"SPY": spy, "QQQ": qqq, "SHV": shv}
    engine = BacktestEngine()
    result = engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly", slippage_bps=5.0),
    )
    assert len(result.trades) >= 1
    assert result.final_equity > 0.0
    assert len(result.equity_curve) >= 2
    assert result.return_metrics.trading_days > 0


def test_backtest_applies_slippage(tmp_path: Path) -> None:
    settings = _test_momentum_settings(tmp_path)
    strat = MomentumRotationStrategy(settings)
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.5)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.1)
    shv = _trending_ohlcv(n, start_d, close_drift=0.01)
    data = {"SPY": spy, "QQQ": qqq, "SHV": shv}
    engine = BacktestEngine()
    r0 = engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly", slippage_bps=0.0),
    )
    r1 = engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly", slippage_bps=10.0),
    )
    assert r1.final_equity < r0.final_equity


def test_backtest_cash_signal_holds_cash(tmp_path: Path) -> None:
    """Flat / weak series keeps price at or below SMA so strategy rotates to cash ETF."""
    settings = _test_momentum_settings(tmp_path)
    strat = MomentumRotationStrategy(settings)
    n = 120
    start_d = date(2024, 1, 2)
    idx = pd.bdate_range(start_d, periods=n, freq="B")
    flat = np.full(n, 100.0)
    df_spy = pd.DataFrame(
        {
            "open": flat,
            "high": flat + 0.5,
            "low": flat - 0.5,
            "close": flat,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )
    df_qqq = df_spy.copy()
    df_shv = df_spy.copy()
    data = {"SPY": df_spy, "QQQ": df_qqq, "SHV": df_shv}
    engine = BacktestEngine()
    result = engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="monthly", slippage_bps=2.0),
    )
    assert result.final_equity == pytest.approx(10_000.0, abs=250.0)


def test_backtest_engine_partial_data_still_works(tmp_path: Path) -> None:
    """Data that overlaps the requested window still completes with metrics."""
    settings = _test_momentum_settings(tmp_path)
    strat = MomentumRotationStrategy(settings)
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.5)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.1)
    shv = _trending_ohlcv(n, start_d, close_drift=0.01)
    data = {"SPY": spy, "QQQ": qqq, "SHV": shv}
    engine = BacktestEngine()
    result = engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 2, 15),
        config=BacktestConfig(rebalance_frequency="weekly", warmup_bars=0, use_regime=False),
    )
    assert len(result.equity_curve) >= 2
    assert result.return_metrics.trading_days > 0


def test_backtest_result_has_return_metrics(tmp_path: Path) -> None:
    settings = _test_momentum_settings(tmp_path)
    strat = MomentumRotationStrategy(settings)
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.4)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.05)
    shv = _trending_ohlcv(n, start_d, close_drift=0.01)
    data = {"SPY": spy, "QQQ": qqq, "SHV": shv}
    engine = BacktestEngine()
    result = engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly"),
    )
    assert result.return_metrics.trading_days > 0
    assert result.return_metrics.total_return_pct == pytest.approx(
        (result.final_equity / result.initial_capital - 1.0) * 100.0,
        rel=0.02,
    )


def test_backtest_rebalance_monthly_trades_less_than_daily(tmp_path: Path) -> None:
    """Alternating targets produce more round-trips when rebalancing every day vs monthly."""
    strat = _AlternatingLongStrategy()
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.4)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.05)
    data = {"SPY": spy, "QQQ": qqq}
    engine = BacktestEngine()
    daily = engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="daily"),
    )
    strat2 = _AlternatingLongStrategy()
    monthly = engine.run(
        strat2,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="monthly"),
    )
    assert len(monthly.trades) < len(daily.trades)


def test_close_price_uses_normalized_index_o1_lookup() -> None:
    """Pricing uses index ``.loc`` on normalized frames (not per-row scan)."""
    idx = pd.bdate_range("2024-01-02", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": 1e6,
        },
        index=idx,
    )
    norm = _normalize_ohlcv_index(df)
    d = idx[2].date()
    assert _close_price_from_normalized(norm, d) == pytest.approx(102.0)


def test_rebalance_dates_weekly_first_day_per_iso_week() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 8)]
    r = _rebalance_dates(days, "weekly")
    assert r == {date(2024, 1, 2), date(2024, 1, 8)}


def test_rebalance_dates_tranche_offset_staggered_weekly() -> None:
    days = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
        date(2024, 1, 9),
        date(2024, 1, 10),
    ]
    r0 = _rebalance_dates(days, "weekly", tranche_offset=0)
    r1 = _rebalance_dates(days, "weekly", tranche_offset=1)
    assert r0 == {days[0], days[5]}
    assert r1 == {days[1], days[6]}


def test_trading_days_in_range_unions_symbol_calendars_and_clips() -> None:
    """Public calendar helper matches engine walk-forward expectations."""
    idx_a = pd.bdate_range("2024-01-02", periods=3, freq="B")
    idx_b = pd.bdate_range("2024-01-03", periods=3, freq="B")
    spy = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx_a)
    qqq = pd.DataFrame({"close": [200.0, 201.0, 202.0]}, index=idx_b)
    data = {"SPY": spy, "QQQ": qqq}
    got = trading_days_in_range(data, date(2024, 1, 2), date(2024, 1, 5))
    assert got == sorted({d.date() for d in idx_a} | {d.date() for d in idx_b})


def test_trading_days_in_range_exported_from_backtesting_package() -> None:
    from src.backtesting import trading_days_in_range as public_td

    assert public_td is trading_days_in_range


def test_vix_close_as_of_returns_last_bar_on_or_before_as_of() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "open": 20.0,
            "high": 21.0,
            "low": 19.0,
            "close": [12.0, 13.0, 14.0, 15.0, 16.0],
            "volume": 1e6,
        },
        index=idx,
    )
    norm = _normalize_ohlcv_index(df)
    d_mid = datetime(2024, 1, 4, 16, 0, tzinfo=UTC)
    assert vix_close_as_of(norm, d_mid) == pytest.approx(14.0)
    d_before = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    assert vix_close_as_of(norm, d_before) is None


class _CaptureMomentum(MomentumRotationStrategy):
    """Records last ``market_regime`` passed to ``generate_signals``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.captured_regimes: list[object] = []

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: object | None = None,
    ) -> list[Signal]:
        self.captured_regimes.append(market_regime)
        return super().generate_signals(data, as_of=as_of, market_regime=market_regime)


def test_backtest_passes_regime_when_settings_and_vix(tmp_path: Path) -> None:
    """Momentum backtest with ``settings`` + VIX series passes non-None ``market_regime``."""
    settings = _test_momentum_settings(tmp_path)
    strat = _CaptureMomentum(settings)
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.5)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.1)
    shv = _trending_ohlcv(n, start_d, close_drift=0.01)
    vix = _trending_ohlcv(n, start_d, close_start=18.0, close_drift=0.02)
    data = {"SPY": spy, "QQQ": qqq, "SHV": shv, "^VIX": vix}
    engine = BacktestEngine()
    engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly"),
        settings=settings,
    )
    assert any(r is not None for r in strat.captured_regimes)


def test_backtest_config_accepts_legacy_use_regime_for_momentum_keyword() -> None:
    """Deprecated ``use_regime_for_momentum`` maps to ``use_regime``."""
    cfg = BacktestConfig(use_regime_for_momentum=False)
    assert cfg.use_regime is False


def test_backtest_skips_regime_when_use_regime_false(tmp_path: Path) -> None:
    settings = _test_momentum_settings(tmp_path)
    strat = _CaptureMomentum(settings)
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.5)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.1)
    shv = _trending_ohlcv(n, start_d, close_drift=0.01)
    vix = _trending_ohlcv(n, start_d, close_start=18.0, close_drift=0.02)
    data = {"SPY": spy, "QQQ": qqq, "SHV": shv, "^VIX": vix}
    engine = BacktestEngine()
    engine.run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly", use_regime=False),
        settings=settings,
    )
    assert all(r is None for r in strat.captured_regimes)


def test_trading_days_in_range_rejects_non_datetime_index() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0]}, index=pd.RangeIndex(2))
    with pytest.raises(TypeError, match="RangeIndex"):
        trading_days_in_range({"SPY": df}, date(2024, 1, 1), date(2024, 1, 31))


def test_trading_days_in_range_accepts_datetime_index() -> None:
    idx = pd.bdate_range("2024-01-02", periods=3, freq="B")
    df = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
    got = trading_days_in_range({"SPY": df}, date(2024, 1, 2), date(2024, 1, 5))
    assert len(got) == 3


def test_backtest_raises_on_partial_coverage_below_threshold() -> None:
    """Requested year-long window but only a short slice of data overlaps → fail loudly."""
    start = date(2024, 1, 2)
    end = date(2024, 12, 31)
    short = _trending_ohlcv(40, date(2024, 6, 3))
    data = {"SPY": short}
    with pytest.raises(BacktestDataMissingError, match="coverage"):
        BacktestEngine().run(
            _NoSignalsStrategy(),
            data,
            start=start,
            end=end,
            config=BacktestConfig(rebalance_frequency="weekly", min_coverage_ratio=0.8),
        )


def test_backtest_passes_on_partial_coverage_above_threshold() -> None:
    """Narrow request with nearly full data overlap clears the coverage gate."""
    start_d = date(2024, 1, 2)
    end = date(2024, 2, 15)
    n = 120
    spy = _trending_ohlcv(n, start_d)
    data = {"SPY": spy}
    result = BacktestEngine().run(
        _AlternatingLongStrategy(),
        data,
        start=start_d,
        end=end,
        config=BacktestConfig(rebalance_frequency="weekly", min_coverage_ratio=0.8),
    )
    assert len(result.equity_curve) >= 2


def test_backtest_min_coverage_zero_disables_check() -> None:
    start = date(2024, 1, 2)
    end = date(2024, 12, 31)
    short = _trending_ohlcv(40, date(2024, 6, 3))
    data = {"SPY": short}
    result = BacktestEngine().run(
        _NoSignalsStrategy(),
        data,
        start=start,
        end=end,
        config=BacktestConfig(
            rebalance_frequency="daily",
            min_coverage_ratio=0.0,
            use_regime=False,
        ),
    )
    assert result.zero_trades_warning is not None


def test_backtest_data_missing_error_includes_available_range() -> None:
    start = date(2024, 1, 2)
    end = date(2024, 12, 31)
    short = _trending_ohlcv(40, date(2024, 6, 3))
    data = {"SPY": short}
    with pytest.raises(BacktestDataMissingError) as excinfo:
        BacktestEngine().run(
            _NoSignalsStrategy(),
            data,
            start=start,
            end=end,
            config=BacktestConfig(rebalance_frequency="weekly"),
        )
    err = excinfo.value
    assert err.available_start == date(2024, 6, 3)
    assert err.available_end is not None
    assert "2024-06-03" in str(err)


def test_backtest_zero_trades_sets_warning() -> None:
    start_d = date(2024, 1, 2)
    n = 80
    spy = _trending_ohlcv(n, start_d)
    data = {"SPY": spy}
    result = BacktestEngine().run(
        _NoSignalsStrategy(),
        data,
        start=start_d,
        end=date(2024, 4, 30),
        config=BacktestConfig(rebalance_frequency="weekly", use_regime=False),
    )
    assert not result.trades
    assert result.zero_trades_warning
    assert "0 trades" in result.zero_trades_warning


def test_backtest_with_trades_no_warning(tmp_path: Path) -> None:
    settings = _test_momentum_settings(tmp_path)
    strat = MomentumRotationStrategy(settings)
    n = 120
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(n, start_d, close_drift=0.5)
    qqq = _trending_ohlcv(n, start_d, close_drift=0.1)
    shv = _trending_ohlcv(n, start_d, close_drift=0.01)
    data = {"SPY": spy, "QQQ": qqq, "SHV": shv}
    result = BacktestEngine().run(
        strat,
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=BacktestConfig(rebalance_frequency="weekly", use_regime=False),
    )
    assert len(result.trades) >= 1
    assert result.zero_trades_warning is None
