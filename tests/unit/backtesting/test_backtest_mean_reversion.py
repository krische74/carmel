"""Walk-forward backtest with MeanReversionStrategy."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.config import DataConfig, MeanReversionConfig, Settings, StrategyConfig
from src.strategy.mean_reversion import MeanReversionStrategy

if TYPE_CHECKING:
    from src.models import Signal


def _mean_reversion_settings() -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=["SPY"]),
        strategy=StrategyConfig(
            mean_reversion=MeanReversionConfig(
                bb_period=10,
                bb_num_std=1.5,
                rsi_period=10,
                rsi_oversold=40.0,
                rsi_overbought=75.0,
                sma_trend_period=0,
                max_positions=2,
                universe=["SPY"],
            ),
        ),
    )


def _oscillating_close(n: int, start: str) -> pd.DataFrame:
    """Strong swing so some rebalance days hit lower band + low RSI."""
    rng = np.random.default_rng(19)
    idx = pd.bdate_range(start, periods=n, freq="B")
    t = np.linspace(0.0, 6.0 * np.pi, n, dtype=float)
    close = 100.0 + 12.0 * np.sin(t) + rng.normal(0.0, 0.25, size=n)
    high = close + 1.5
    low = close - 1.5
    open_ = np.r_[close[0], close[:-1]]
    vol = np.full(n, 1e6)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_backtest_mean_reversion_equity_curve_walk_forward() -> None:
    settings = _mean_reversion_settings()
    df = _oscillating_close(160, "2022-06-01")
    data = {"SPY": df}
    eng = BacktestEngine()
    res = eng.run(
        MeanReversionStrategy(settings),
        data,
        start=date(2022, 8, 1),
        end=date(2023, 2, 28),
        config=BacktestConfig(rebalance_frequency="weekly"),
    )
    assert res.initial_capital == 10_000.0
    assert res.final_equity > 0
    assert all(float(p["equity"]) > 0 for p in res.equity_curve)
    assert len(res.equity_curve) >= 20
    assert len(res.trades) > 0
    symbols_traded = {t.symbol for t in res.trades}
    assert "SPY" in symbols_traded
    buy_trades = [t for t in res.trades if t.side == "buy"]
    assert buy_trades, "mean reversion should open long positions on dips"
    assert res.return_metrics.trading_days >= 1


def _vix_ohlcv_like(template: pd.DataFrame, level: float = 18.0) -> pd.DataFrame:
    close = np.full(len(template), level, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(template), 1e6),
        },
        index=template.index,
    )


class _CaptureMeanReversion(MeanReversionStrategy):
    """Records each ``market_regime`` passed to ``generate_signals``."""

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


def test_backtest_mean_reversion_passes_regime_when_settings_and_vix() -> None:
    settings = _mean_reversion_settings()
    df = _oscillating_close(160, "2022-06-01")
    vix = _vix_ohlcv_like(df)
    data = {"SPY": df, "^VIX": vix}
    strat = _CaptureMeanReversion(settings)
    eng = BacktestEngine()
    eng.run(
        strat,
        data,
        start=date(2022, 8, 1),
        end=date(2023, 2, 28),
        config=BacktestConfig(rebalance_frequency="weekly"),
        settings=settings,
    )
    assert any(r is not None for r in strat.captured_regimes)


def test_backtest_mean_reversion_skips_regime_when_use_regime_false() -> None:
    settings = _mean_reversion_settings()
    df = _oscillating_close(160, "2022-06-01")
    vix = _vix_ohlcv_like(df)
    data = {"SPY": df, "^VIX": vix}
    strat = _CaptureMeanReversion(settings)
    eng = BacktestEngine()
    eng.run(
        strat,
        data,
        start=date(2022, 8, 1),
        end=date(2023, 2, 28),
        config=BacktestConfig(rebalance_frequency="weekly", use_regime=False),
        settings=settings,
    )
    assert all(r is None for r in strat.captured_regimes)


def test_backtest_mean_reversion_high_vix_fewer_buys_than_calm_vix() -> None:
    """High VIX (crisis cap) takes fewer simultaneous MR entries than calm VIX (H2)."""
    syms = [f"S{i}" for i in range(6)]
    base_df = _oscillating_close(160, "2022-06-01")
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=syms),
        strategy=StrategyConfig(
            mean_reversion=MeanReversionConfig(
                bb_period=10,
                bb_num_std=1.5,
                rsi_period=10,
                rsi_oversold=40.0,
                rsi_overbought=75.0,
                sma_trend_period=0,
                max_positions=4,
                universe=syms,
            ),
        ),
    )
    data_calm = {s: base_df for s in syms}
    data_calm["^VIX"] = _vix_ohlcv_like(base_df, 14.0)
    data_stress = {s: base_df for s in syms}
    data_stress["^VIX"] = _vix_ohlcv_like(base_df, 55.0)
    eng = BacktestEngine()
    cfg = BacktestConfig(rebalance_frequency="weekly")
    strat = MeanReversionStrategy(settings)
    r_calm = eng.run(
        strat,
        data_calm,
        start=date(2022, 8, 1),
        end=date(2023, 2, 28),
        config=cfg,
        settings=settings,
    )
    r_stress = eng.run(
        MeanReversionStrategy(settings),
        data_stress,
        start=date(2022, 8, 1),
        end=date(2023, 2, 28),
        config=cfg,
        settings=settings,
    )
    buys_calm = sum(1 for t in r_calm.trades if t.side == "buy")
    buys_stress = sum(1 for t in r_stress.trades if t.side == "buy")
    assert buys_calm > 0 and buys_stress > 0
    assert buys_calm > buys_stress
