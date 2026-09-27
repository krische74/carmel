"""Tier 54F — DCA sleeve ablation flags (pin multiplier, fixed-dollar budget)."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.config import DataConfig, DCAConfig, DCATarget, RiskConfig, Settings, StrategyConfig
from src.strategy.base import Strategy
from src.strategy.momentum import MomentumRotationStrategy

if TYPE_CHECKING:
    from src.models import Signal


def _ohlcv(n: int, start: date, *, drift: float = 0.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = [100.0 * (1.0 + drift) ** i for i in range(n)]
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000_000.0] * n,
        },
        index=idx,
    )


def _vix_elevated(n: int, start: date) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = [35.0] * n
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000_000.0] * n,
        },
        index=idx,
    )


class _MomentumOnly(Strategy):
    def get_universe(self) -> list[str]:
        return ["SPY", "SHV"]

    def generate_signals(self, data, *, as_of=None, market_regime=None) -> list[Signal]:
        return MomentumRotationStrategy(
            Settings(
                _yaml_path=None,
                _env_file=None,
                data=DataConfig(
                    parquet_dir=".",
                    cache_dir=".",
                    universe=["SPY"],
                ),
                strategy=StrategyConfig(
                    momentum={"cash_symbol": "SHV"},
                ),
            )
        ).generate_signals(data, as_of=as_of, market_regime=market_regime)


def _full_settings(tmp_path) -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[
                DCATarget(symbol="VOO", weight=0.6),
                DCATarget(symbol="VXUS", weight=0.3),
                DCATarget(symbol="BND", weight=0.1),
            ],
        ),
        strategy=StrategyConfig(
            dca=DCAConfig(
                frequency="weekly",
                percent_of_equity=0.025,
                regime_amount_cautious=0.708,
            ),
        ),
        risk=RiskConfig(
            max_position_pct=0.25,
            min_cash_reserve_pct=0.05,
            min_order_notional_usd=5.0,
        ),
    )


def _full_data(start: date, n: int = 120) -> dict[str, pd.DataFrame]:
    return {
        "SPY": _ohlcv(n, start, drift=0.0002),
        "SHV": _ohlcv(n, start),
        "VOO": _ohlcv(n, start, drift=0.0002),
        "VXUS": _ohlcv(n, start, drift=0.0001),
        "BND": _ohlcv(n, start),
        "^VIX": _vix_elevated(n, start),
        "BIL": _ohlcv(n, start),
    }


def test_pin_dca_regime_multiplier_isolated_from_momentum(tmp_path) -> None:
    """With include_dca=False, pin_dca_regime_multiplier must not change momentum path."""
    settings = _full_settings(tmp_path)
    start_d = date(2024, 1, 2)
    data = _full_data(start_d)
    eng = BacktestEngine()
    base_cfg = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=True,
        include_dca=False,
        use_regime=True,
        cash_yield_annual_pct=0.0,
    )
    pinned_cfg = base_cfg.model_copy(update={"pin_dca_regime_multiplier": 1.0})
    strat = _MomentumOnly()
    r0 = eng.run(strat, data, start=start_d, end=date(2024, 4, 30), config=base_cfg, settings=settings)
    r1 = eng.run(strat, data, start=start_d, end=date(2024, 4, 30), config=pinned_cfg, settings=settings)
    assert [(t.date, t.symbol, t.side, round(t.qty, 6)) for t in r0.trades] == [
        (t.date, t.symbol, t.side, round(t.qty, 6)) for t in r1.trades
    ]
    assert r0.equity_curve == r1.equity_curve


def test_pin_dca_and_pin_momentum_are_distinct_levers(tmp_path) -> None:
    settings = _full_settings(tmp_path)
    start_d = date(2024, 1, 2)
    data = _full_data(start_d)
    eng = BacktestEngine()
    common = dict(
        rebalance_frequency="weekly",
        apply_risk_layer=True,
        include_dca=True,
        use_regime=True,
        cash_yield_annual_pct=0.0,
    )
    r_dca = eng.run(
        _MomentumOnly(),
        data,
        start=start_d,
        end=date(2024, 4, 30),
        config=BacktestConfig(**common, pin_dca_regime_multiplier=1.0),
        settings=settings,
    )
    r_mom = eng.run(
        _MomentumOnly(),
        data,
        start=start_d,
        end=date(2024, 4, 30),
        config=BacktestConfig(**common, pin_regime_multiplier=1.0),
        settings=settings,
    )
    assert r_dca.final_equity != r_mom.final_equity


def test_pin_dca_regime_multiplier_increases_gross_dca_dollars(tmp_path) -> None:
    settings = _full_settings(tmp_path)
    start_d = date(2024, 1, 2)
    data = _full_data(start_d)
    eng = BacktestEngine()
    common = dict(
        rebalance_frequency="weekly",
        apply_risk_layer=True,
        include_dca=True,
        use_regime=True,
        cash_yield_annual_pct=0.0,
    )
    r0 = eng.run(
        _MomentumOnly(),
        data,
        start=start_d,
        end=date(2024, 4, 30),
        config=BacktestConfig(**common),
        settings=settings,
    )
    r1 = eng.run(
        _MomentumOnly(),
        data,
        start=start_d,
        end=date(2024, 4, 30),
        config=BacktestConfig(**common, pin_dca_regime_multiplier=1.0),
        settings=settings,
    )
    dca0 = sum(t.qty * t.price for t in r0.trades if t.symbol in {"VOO", "VXUS", "BND"})
    dca1 = sum(t.qty * t.price for t in r1.trades if t.symbol in {"VOO", "VXUS", "BND"})
    assert dca1 > dca0


def test_dca_fixed_amount_usd_deploys_constant_budget_before_caps(tmp_path) -> None:
    settings = _full_settings(tmp_path)
    start_d = date(2024, 1, 2)
    data = _full_data(start_d, n=80)
    fixed = 250.0
    result = BacktestEngine().run(
        _MomentumOnly(),
        data,
        start=start_d,
        end=date(2024, 4, 15),
        config=BacktestConfig(
            rebalance_frequency="weekly",
            apply_risk_layer=True,
            include_dca=True,
            use_regime=False,
            cash_yield_annual_pct=0.0,
            dca_fixed_amount_usd=fixed,
            pin_dca_regime_multiplier=1.0,
        ),
        settings=settings,
    )
    dca_buys = [t for t in result.trades if t.symbol in {"VOO", "VXUS", "BND"}]
    assert dca_buys
    per_cycle: dict[str, float] = {}
    for t in dca_buys:
        per_cycle[t.date] = per_cycle.get(t.date, 0.0) + t.qty * t.price
    for total in per_cycle.values():
        assert total == pytest.approx(fixed, rel=0.02)


@pytest.mark.slow
@pytest.mark.skipif(
    not any(__import__("pathlib").Path("data/parquet").glob("*.parquet")),
    reason="needs the local market-data cache (data/parquet), which is not in the repo",
)
def test_default_flags_reproduce_post_54a_full_system_regression() -> None:
    """Default 54F flags preserve the measured post-54A real-data baseline."""
    from scripts.tier52_analysis import _cash_map, _load_data
    from src.automation.strategy_wiring import build_strategy_for_backtest
    from src.config import get_settings
    from src.reporting.episodes import strategy_usable_start_dates

    settings = get_settings()
    data = _load_data(settings)
    start = strategy_usable_start_dates(data)["dca"]
    cash_map = _cash_map(settings)
    strategy = build_strategy_for_backtest(settings, "momentum")

    common = dict(
        rebalance_frequency="weekly",
        include_dca=True,
        include_cash_sweep=True,
        cash_yield_annual_pct=float(settings.backtest.cash_yield_annual_pct),
        cash_yield_by_date=cash_map,
        min_coverage_ratio=0.8,
        use_regime=True,
        # Expected values below are conditional on this explicitly pinned band.
        drift_band_pct=0.05,
    )
    constrained_cfg = BacktestConfig(**common, apply_risk_layer=True)
    raw_cfg = BacktestConfig(**common, apply_risk_layer=False)
    constrained = BacktestEngine().run(
        strategy,
        data,
        start=start,
        end=date(2026, 8, 19),
        config=constrained_cfg,
        settings=settings,
    )
    raw = BacktestEngine().run(
        strategy,
        data,
        start=start,
        end=date(2026, 8, 19),
        config=raw_cfg,
        settings=settings,
    )

    assert constrained_cfg.pin_dca_regime_multiplier is None
    assert constrained_cfg.dca_fixed_amount_usd is None
    assert raw_cfg.pin_dca_regime_multiplier is None
    assert raw_cfg.dca_fixed_amount_usd is None
    # The old 52B artifact predates 54A's state-gated execution and is superseded.
    assert constrained.final_equity == pytest.approx(39_361.555318380284, abs=0.01)
    assert constrained.return_metrics.cagr_pct == pytest.approx(9.222753538838834, abs=0.0001)
    assert len(constrained.trades) == 715
    assert raw.final_equity == pytest.approx(33_213.41370101835, abs=0.01)
    assert raw.return_metrics.cagr_pct == pytest.approx(8.034936077025812, abs=0.0001)
    assert len(raw.trades) == 445
