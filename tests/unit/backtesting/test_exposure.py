"""Tier 54B — risk-asset exposure metric (excludes cash leg)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.sim_steps import compute_risk_exposure_pct
from src.config import DataConfig, MomentumConfig, Settings, StrategyConfig
from src.models import Signal
from src.strategy.base import Strategy


class _HoldSymbolStrategy(Strategy):
    def __init__(self, symbol: str) -> None:
        self._symbol = symbol.strip().upper()

    def get_universe(self) -> list[str]:
        return [self._symbol]

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
                symbol=self._symbol,
                direction="long",
                weight=1.0,
                confidence=0.5,
                rationale=f"Synthetic hold {self._symbol}.",
                timestamp=when,
                strategy_name="HoldSymbolStrategy",
            ),
        ]


def _flat_ohlcv(n: int, start: date) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = np.full(n, 100.0, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": np.ones(n)},
        index=idx,
    )


def test_compute_risk_exposure_excludes_cash_leg() -> None:
    exp = compute_risk_exposure_pct(
        mtm=10_000.0,
        positions={"SPY": 50.0, "SHV": 50.0},
        last_marks={"SPY": 100.0, "SHV": 100.0},
        cash_symbols=frozenset({"SHV"}),
        sweep_symbol="",
    )
    assert exp == 0.5


def test_raw_mode_exposure_differs_for_risk_vs_cash_leg(tmp_path) -> None:
    """SHV must not count as risk exposure; configs must not all read 0.9032."""
    n = 60
    start_d = date(2024, 1, 2)
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "SHV"],
        ),
        strategy=StrategyConfig(momentum=MomentumConfig(cash_symbol="SHV")),
    )
    data = {"SPY": _flat_ohlcv(n, start_d), "SHV": _flat_ohlcv(n, start_d)}
    cfg = BacktestConfig(
        slippage_bps=0.0,
        cash_yield_annual_pct=0.0,
        drift_band_pct=1.0,
        use_regime=False,
        apply_risk_layer=False,
        rebalance_frequency="weekly",
    )
    spy = BacktestEngine().run(
        _HoldSymbolStrategy("SPY"),
        data,
        start=start_d,
        end=date(2024, 3, 29),
        config=cfg,
        settings=settings,
    )
    shv = BacktestEngine().run(
        _HoldSymbolStrategy("SHV"),
        data,
        start=start_d,
        end=date(2024, 3, 29),
        config=cfg,
        settings=settings,
    )
    assert spy.avg_exposure_pct > 0.9
    assert shv.avg_exposure_pct < 0.05
    assert spy.avg_exposure_pct != shv.avg_exposure_pct
