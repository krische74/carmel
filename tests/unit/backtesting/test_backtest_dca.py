"""Walk-forward backtest with DCAStrategy."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.config import DataConfig, DCAConfig, DCATarget, Settings, StrategyConfig
from src.strategy.dca import DCAStrategy

if TYPE_CHECKING:
    from src.models import Signal


def _ohlcv(n: int, start: str) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = 100.0 + np.linspace(0.0, 5.0, n, dtype=float) + rng.normal(0.0, 0.12, size=n)
    high = close + 1.0
    low = close - 0.5
    open_ = np.r_[close[0], close[:-1]]
    vol = np.full(n, 1e6)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_backtest_dca_daily_generates_trades() -> None:
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            dca_targets=[
                DCATarget(symbol="SPY", weight=0.6),
                DCATarget(symbol="QQQ", weight=0.4),
            ],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="daily", amount=200.0)),
    )
    spy = _ohlcv(90, "2023-01-03")
    qqq = _ohlcv(90, "2023-01-03")
    data = {"SPY": spy, "QQQ": qqq}
    eng = BacktestEngine()
    res = eng.run(
        DCAStrategy(settings),
        data,
        start=date(2023, 1, 10),
        end=date(2023, 5, 1),
        config=BacktestConfig(rebalance_frequency="daily"),
    )
    assert res.initial_capital == 10_000.0
    assert res.final_equity > 0
    assert all(float(p["equity"]) > 0 for p in res.equity_curve)
    assert len(res.equity_curve) >= 50
    assert len(res.trades) > 0
    symbols_traded = {t.symbol for t in res.trades}
    assert "SPY" in symbols_traded and "QQQ" in symbols_traded
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


class _CaptureDCA(DCAStrategy):
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


def test_backtest_dca_passes_regime_when_settings_and_vix() -> None:
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            dca_targets=[
                DCATarget(symbol="SPY", weight=0.6),
                DCATarget(symbol="QQQ", weight=0.4),
            ],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="daily", amount=200.0)),
    )
    spy = _ohlcv(90, "2023-01-03")
    qqq = _ohlcv(90, "2023-01-03")
    vix = _vix_ohlcv_like(spy)
    data = {"SPY": spy, "QQQ": qqq, "^VIX": vix}
    strat = _CaptureDCA(settings)
    eng = BacktestEngine()
    eng.run(
        strat,
        data,
        start=date(2023, 1, 10),
        end=date(2023, 5, 1),
        config=BacktestConfig(rebalance_frequency="daily"),
        settings=settings,
    )
    assert any(r is not None for r in strat.captured_regimes)


def test_backtest_dca_skips_regime_when_use_regime_false() -> None:
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            dca_targets=[
                DCATarget(symbol="SPY", weight=0.6),
                DCATarget(symbol="QQQ", weight=0.4),
            ],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="daily", amount=200.0)),
    )
    spy = _ohlcv(90, "2023-01-03")
    qqq = _ohlcv(90, "2023-01-03")
    vix = _vix_ohlcv_like(spy)
    data = {"SPY": spy, "QQQ": qqq, "^VIX": vix}
    strat = _CaptureDCA(settings)
    eng = BacktestEngine()
    eng.run(
        strat,
        data,
        start=date(2023, 1, 10),
        end=date(2023, 5, 1),
        config=BacktestConfig(rebalance_frequency="daily", use_regime=False),
        settings=settings,
    )
    assert all(r is None for r in strat.captured_regimes)


def test_backtest_dca_high_vix_reduces_final_equity_vs_calm_vix() -> None:
    """Crisis-scaled DCA deploys less capital in a rising tape than calm VIX (H2)."""
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            dca_targets=[
                DCATarget(symbol="SPY", weight=0.6),
                DCATarget(symbol="QQQ", weight=0.4),
            ],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="daily", amount=200.0)),
    )
    spy = _ohlcv(90, "2023-01-03")
    qqq = _ohlcv(90, "2023-01-03")
    vix_calm = _vix_ohlcv_like(spy, 14.0)
    vix_stress = _vix_ohlcv_like(spy, 55.0)
    eng = BacktestEngine()
    cfg = BacktestConfig(rebalance_frequency="daily")
    r_calm = eng.run(
        DCAStrategy(settings),
        {"SPY": spy, "QQQ": qqq, "^VIX": vix_calm},
        start=date(2023, 1, 10),
        end=date(2023, 5, 1),
        config=cfg,
        settings=settings,
    )
    r_stress = eng.run(
        DCAStrategy(settings),
        {"SPY": spy, "QQQ": qqq, "^VIX": vix_stress},
        start=date(2023, 1, 10),
        end=date(2023, 5, 1),
        config=cfg,
        settings=settings,
    )
    assert r_stress.final_equity < r_calm.final_equity
