"""Tier 32: backtest warm-up period skips initial bars."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, BacktestDataMissingError, BacktestEngine
from src.models import Signal
from src.strategy.base import Strategy

# Short synthetic windows use min_coverage_ratio=0.0 (defaults require ~80% of expected trading days).
_WARM = dict(rebalance_frequency="daily", warmup_bars=0, use_regime=False, min_coverage_ratio=0.0)


class _DailyLongSpy(Strategy):
    def get_universe(self) -> list[str]:
        return ["SPY"]

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
        return [
            Signal(
                symbol="SPY",
                direction="long",
                weight=1.0,
                confidence=0.5,
                rationale="always long",
                timestamp=when,
                strategy_name="DailyLongSpy",
            ),
        ]


def _ohlcv(n: int, start: date) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = 100.0 + np.linspace(0.0, 0.2 * n, n, dtype=float)
    high = close + 1.0
    low = close - 1.0
    open_ = np.r_[close[0], close[:-1]]
    vol = np.full(n, 1_000_000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_backtest_warmup_skips_initial_bars() -> None:
    data = {"SPY": _ohlcv(45, date(2024, 1, 2))}
    strat = _DailyLongSpy()
    start = date(2024, 1, 2)
    end = date(2024, 3, 1)
    r0 = BacktestEngine().run(
        strat,
        data,
        start=start,
        end=end,
        config=BacktestConfig(**_WARM),
    )
    r1 = BacktestEngine().run(
        strat,
        data,
        start=start,
        end=end,
        config=BacktestConfig(**{**_WARM, "warmup_bars": 12}),
    )
    assert len(r1.equity_curve) == len(r0.equity_curve) - 12


def test_backtest_warmup_zero_starts_immediately() -> None:
    data = {"SPY": _ohlcv(20, date(2024, 1, 2))}
    strat = _DailyLongSpy()
    r0 = BacktestEngine().run(
        strat,
        data,
        start=date(2024, 1, 2),
        end=date(2024, 2, 15),
        config=BacktestConfig(**_WARM),
    )
    r1 = BacktestEngine().run(
        strat,
        data,
        start=date(2024, 1, 2),
        end=date(2024, 2, 15),
        config=BacktestConfig(**_WARM),
    )
    assert len(r0.equity_curve) == len(r1.equity_curve)


def test_backtest_warmup_longer_than_data_raises() -> None:
    data = {"SPY": _ohlcv(5, date(2024, 1, 2))}
    strat = _DailyLongSpy()
    with pytest.raises(BacktestDataMissingError, match="warmup_bars"):
        BacktestEngine().run(
            strat,
            data,
            start=date(2024, 1, 2),
            end=date(2024, 1, 10),
            config=BacktestConfig(
                rebalance_frequency="daily",
                warmup_bars=100,
                use_regime=False,
                min_coverage_ratio=0.0,
            ),
        )
