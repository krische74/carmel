"""Tier 52: weekday rebalance, historical cash yield, exposure, DCA contribution."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtesting.cash_yield import lookup_cash_yield_annual_pct
from src.backtesting.dca_contrib import contribution_notional
from src.backtesting.engine import BacktestConfig, BacktestEngine, _rebalance_dates
from src.config import DataConfig, DCAConfig, DCATarget, RiskConfig, Settings, StrategyConfig
from src.models import Signal
from src.reporting.returns import compute_return_metrics, illustrative_vol_scaled_cagr
from src.risk.position_sizing import compute_order_notional
from src.strategy.base import Strategy
from src.strategy.dca import DCAStrategy


class _NoSignalsStrategy(Strategy):
    def get_universe(self) -> list[str]:
        return ["VOO"]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: object | None = None,
    ) -> list[Signal]:
        return []


class _FullLongStrategy(Strategy):
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
        return [
            Signal(
                symbol="SPY",
                direction="long",
                weight=1.0,
                confidence=1.0,
                rationale="full long",
                timestamp=when,
                strategy_name="FullLong",
            ),
        ]


def _ohlcv(n: int, start: date, *, drift: float = 0.2) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = 100.0 + np.linspace(0.0, drift * n, n, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_weekly_rebalance_none_is_first_trading_day_of_iso_week() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 8)]
    r = _rebalance_dates(days, "weekly")
    assert r == {date(2024, 1, 2), date(2024, 1, 8)}


def test_weekly_rebalance_thursday_skips_earlier_weekdays() -> None:
    days = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
        date(2024, 1, 8),
    ]
    thu = _rebalance_dates(days, "weekly", weekday=3)
    assert date(2024, 1, 4) in thu
    assert date(2024, 1, 2) not in thu
    assert date(2024, 1, 8) not in thu


def test_monthly_phase_picks_later_day_of_month() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 2, 1)]
    first = _rebalance_dates(days, "monthly", month_phase=0)
    second = _rebalance_dates(days, "monthly", month_phase=1)
    assert first == {date(2024, 1, 2), date(2024, 2, 1)}
    assert date(2024, 1, 3) in second
    assert date(2024, 1, 2) not in second


def test_cash_yield_varies_by_date_and_fallbacks() -> None:
    series = pd.Series(
        {pd.Timestamp("2020-01-02"): 1.5, pd.Timestamp("2020-01-03"): 1.6},
        dtype=float,
    )
    series.index = pd.DatetimeIndex(series.index).normalize()
    y, mode = lookup_cash_yield_annual_pct(series, date(2020, 1, 3), fallback_pct=4.0)
    assert y == pytest.approx(1.6)
    assert mode == "historical"
    y2, mode2 = lookup_cash_yield_annual_pct(series, date(2019, 1, 1), fallback_pct=4.0)
    assert y2 == pytest.approx(4.0)
    assert mode2 == "fallback_flat"


def test_cash_yield_logs_fallback(caplog: pytest.LogCaptureFixture) -> None:
    series = pd.Series(dtype=float)
    with caplog.at_level("WARNING", logger="src.backtesting.cash_yield"):
        y, mode = lookup_cash_yield_annual_pct(series, date(2020, 1, 2), fallback_pct=4.0)
    assert y == pytest.approx(4.0)
    assert mode == "fallback_flat"
    assert any("fallback" in r.message.lower() for r in caplog.records)


def test_illustrative_vol_scaled_cagr() -> None:
    daily = pd.Series(0.001 + 0.0005 * np.linspace(-1.0, 1.0, 252))
    raw = compute_return_metrics(daily)
    scaled = illustrative_vol_scaled_cagr(daily, target_vol_pct=raw.annual_volatility_pct * 2.0)
    assert scaled is not None
    assert scaled > raw.cagr_pct


def test_dca_contribution_matches_live_sizer(tmp_path) -> None:
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly", percent_of_equity=0.025)),
        risk=RiskConfig(max_position_pct=0.25, min_order_notional_usd=5.0, min_cash_reserve_pct=0.05),
    )
    equity = 10_000.0
    budget = equity * 0.025
    sig = DCAStrategy(settings).generate_signals(
        {"VOO": _ohlcv(40, date(2024, 1, 2))},
        as_of=datetime(2024, 1, 8, tzinfo=UTC),
    )
    assert len(sig) == 1
    live = compute_order_notional(
        signal_weight=sig[0].weight,
        equity=equity,
        max_position_pct=0.25,
        dca_budget=budget,
        use_dca_budget=True,
        current_position_notional=0.0,
    )
    got = contribution_notional(
        signal_weight=sig[0].weight,
        dca_budget=budget,
        equity=equity,
        current_position_notional=0.0,
        max_position_pct=0.25,
        min_order_notional_usd=5.0,
        spendable_cash=equity,
    )
    assert got == pytest.approx(live)


def test_dca_ramp_reaches_cap_not_starts_there(tmp_path) -> None:
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly", percent_of_equity=0.10)),
        risk=RiskConfig(max_position_pct=0.25, min_order_notional_usd=5.0, min_cash_reserve_pct=0.05),
    )
    start_d = date(2024, 1, 2)
    voo = _ohlcv(80, start_d, drift=0.0)
    result = BacktestEngine().run(
        _NoSignalsStrategy(),
        {"VOO": voo},
        start=start_d,
        end=date(2024, 4, 15),
        config=BacktestConfig(
            rebalance_frequency="weekly",
            apply_risk_layer=True,
            include_dca=True,
            cash_yield_annual_pct=0.0,
            use_regime=False,
        ),
        settings=settings,
    )
    first_buy = next(t for t in result.trades if t.side == "buy")
    assert first_buy.qty * first_buy.price < 10_000.0 * 0.15
    assert result.peak_single_symbol_allocation_pct <= 0.25 + 1e-6
    assert result.peak_single_symbol_allocation_pct > 0.18


def test_constrained_run_reports_exposure(tmp_path) -> None:
    start_d = date(2024, 1, 2)
    spy = _ohlcv(40, start_d, drift=0.3)
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        risk=RiskConfig(max_position_pct=0.25, min_cash_reserve_pct=0.05, min_order_notional_usd=5.0),
    )
    result = BacktestEngine().run(
        _FullLongStrategy(),
        {"SPY": spy},
        start=start_d,
        end=date(2024, 2, 29),
        config=BacktestConfig(
            rebalance_frequency="weekly",
            apply_risk_layer=True,
            cash_yield_annual_pct=0.0,
            use_regime=False,
        ),
        settings=settings,
    )
    assert 0.15 < result.avg_exposure_pct <= 0.26
    # Tier 54A: state-gated engine does not drift-rebalance; peak can exceed cap while held.
    assert result.max_exposure_pct > result.avg_exposure_pct
