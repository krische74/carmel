"""Tier 54A — state-gated backtest execution identity tests."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from src.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    discrete_hold_state,
)
from src.backtesting.sim_steps import drift_band_exceeded, target_weights_changed
from src.config import DataConfig, RiskConfig, Settings
from src.models import Signal
from src.strategy.base import Strategy

# Identity accounting (Tier 54A):
# ``discrete_state_changes`` counts only symbol-set rotations after the first
# deployment (increment when ``last_hold_state is not None`` and the hold state
# changes). Initial cash→risk entry is one buy, uncounted. Hence the test asserts
# ``len(trades) == 2 * discrete_state_changes + 1``. The full-system Step 0 diagnostic
# reports flow and maintenance counters separately instead of applying this pure-rotation
# identity to mixed trade streams.
#
# Rotation identity tests set ``drift_band_pct=1.0`` (DR #5 "infinite drift
# tolerance" diagnostic — disables weight maintenance) and ``cash_yield_annual_pct=0``
# (no cash accrual; dividends N/A on synthetic flat series).


class _HoldSpyStrategy(Strategy):
    """Always targets 100% SPY (N=1 retain test)."""

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
        return [
            Signal(
                symbol="SPY",
                direction="long",
                weight=1.0,
                confidence=0.5,
                rationale="Synthetic hold SPY.",
                timestamp=when,
                strategy_name="HoldSpyStrategy",
            ),
        ]


class _AlternatingLongStrategy(Strategy):
    """Alternates full long between SPY and QQQ each rebalance."""

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
                rationale="Synthetic alternating target.",
                timestamp=when,
                strategy_name="AlternatingLongStrategy",
            ),
        ]


def _trending_ohlcv(n: int, start: date, *, drift: float = 0.3) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = 100.0 + np.linspace(0.0, drift * n, n, dtype=float)
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": close + 1.0,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _zero_cost_config(**kwargs: object) -> BacktestConfig:
    return BacktestConfig(
        slippage_bps=0.0,
        cash_yield_annual_pct=0.0,
        drift_band_pct=1.0,
        use_regime=False,
        apply_risk_layer=False,
        rebalance_frequency="weekly",
        **kwargs,
    )


def _flat_ohlcv(n: int, start: date, *, price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = np.full(n, price, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_discrete_hold_state_excludes_cash_leg() -> None:
    state = discrete_hold_state({"SHV": 1.0, "SPY": 0.0}, cash_symbols=frozenset({"SHV"}))
    assert state == ()


def test_state_gate_no_trades_when_incumbent_retained() -> None:
    """Retaining the incumbent must emit zero trades after initial entry."""
    n = 120
    start_d = date(2024, 1, 2)
    data = {
        "SPY": _trending_ohlcv(n, start_d, drift=0.4),
        "QQQ": _trending_ohlcv(n, start_d, drift=0.2),
    }
    result = BacktestEngine().run(
        _HoldSpyStrategy(),
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=_zero_cost_config(),
    )
    assert len(result.trades) >= 1
    first_date = result.trades[0].date
    follow_on = [t for t in result.trades if t.date > first_date]
    assert follow_on == []


def test_state_gate_exactly_two_trades_per_rotation() -> None:
    """Each discrete rotation between two risk assets emits exactly two trades."""
    n = 120
    start_d = date(2024, 1, 2)
    data = {
        "SPY": _trending_ohlcv(n, start_d, drift=0.4),
        "QQQ": _trending_ohlcv(n, start_d, drift=0.2),
    }
    result = BacktestEngine().run(
        _AlternatingLongStrategy(),
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=_zero_cost_config(),
    )
    assert result.discrete_state_changes >= 2
    rotation_trades = result.trades[1:]
    # After initial entry, every state change is a full rotation (sell + buy).
    for i in range(0, len(rotation_trades), 2):
        pair = rotation_trades[i : i + 2]
        assert len(pair) == 2
        sides = {t.side for t in pair}
        assert sides == {"buy", "sell"}


def test_state_gate_identity_total_trades_equals_twice_state_changes() -> None:
    """Tier 54A identity: each rotation after initial entry emits exactly two trades."""
    n = 120
    start_d = date(2024, 1, 2)
    data = {
        "SPY": _trending_ohlcv(n, start_d, drift=0.4),
        "QQQ": _trending_ohlcv(n, start_d, drift=0.2),
    }
    result = BacktestEngine().run(
        _AlternatingLongStrategy(),
        data,
        start=start_d,
        end=date(2024, 6, 28),
        config=_zero_cost_config(),
    )
    assert result.discrete_state_changes > 0
    # Initial cash deployment is one buy; every subsequent state change is a two-leg rotation.
    assert len(result.trades) == 2 * result.discrete_state_changes + 1


def test_drift_band_exceeded_within_and_outside_tolerance() -> None:
    assert not drift_band_exceeded(
        targets={"SPY": 0.25},
        positions={"SPY": 250.0},
        prices={"SPY": 10.0},
        equity=10_000.0,
        skip_symbols=set(),
        band_pct=0.05,
    )
    assert drift_band_exceeded(
        targets={"SPY": 0.25},
        positions={"SPY": 150.0},
        prices={"SPY": 10.0},
        equity=10_000.0,
        skip_symbols=set(),
        band_pct=0.05,
    )


def test_target_weights_changed_detects_regime_resize() -> None:
    prev = {"SPY": 0.25}
    assert target_weights_changed(prev, {"SPY": 0.1875})
    assert not target_weights_changed(prev, {"SPY": 0.25})


def test_drift_band_no_trades_within_band_constrained(tmp_path) -> None:
    """Flat price + zero yield: capped weight stays at target; no maintenance trades."""
    start_d = date(2024, 1, 2)
    n = 80
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        risk=RiskConfig(
            max_position_pct=0.25, min_cash_reserve_pct=0.0, min_order_notional_usd=1.0
        ),
    )
    result = BacktestEngine().run(
        _HoldSpyStrategy(),
        {"SPY": _flat_ohlcv(n, start_d)},
        start=start_d,
        end=date(2024, 4, 30),
        config=BacktestConfig(
            slippage_bps=0.0,
            cash_yield_annual_pct=0.0,
            drift_band_pct=0.05,
            use_regime=False,
            apply_risk_layer=True,
            rebalance_frequency="weekly",
        ),
        settings=settings,
    )
    assert len(result.trades) == 1


def test_drift_band_triggers_exactly_one_resize_past_band(tmp_path) -> None:
    """Cash yield grows equity while SPY is flat; one top-up once band is breached."""
    start_d = date(2024, 1, 2)
    n = 120
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        risk=RiskConfig(
            max_position_pct=0.25, min_cash_reserve_pct=0.0, min_order_notional_usd=1.0
        ),
    )
    result = BacktestEngine().run(
        _HoldSpyStrategy(),
        {"SPY": _flat_ohlcv(n, start_d)},
        start=start_d,
        end=date(2024, 2, 27),
        config=BacktestConfig(
            slippage_bps=0.0,
            cash_yield_annual_pct=200.0,
            drift_band_pct=0.05,
            use_regime=False,
            apply_risk_layer=True,
            rebalance_frequency="weekly",
        ),
        settings=settings,
    )
    first_date = result.trades[0].date
    maintenance = [t for t in result.trades if t.date > first_date]
    assert len(maintenance) == 1
    assert maintenance[0].side == "buy"
    assert maintenance[0].symbol == "SPY"
    assert result.maintenance_trades == 1
    assert result.maintenance_band_breaches >= 1


def test_constrained_run_records_date_of_max_exposure(tmp_path) -> None:
    """The exposure diagnostic identifies the bar that sets the maximum."""
    start_d = date(2024, 1, 2)
    n = 120
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        risk=RiskConfig(
            max_position_pct=0.25, min_cash_reserve_pct=0.05, min_order_notional_usd=5.0
        ),
    )
    result = BacktestEngine().run(
        _HoldSpyStrategy(),
        {"SPY": _flat_ohlcv(n, start_d)},
        start=start_d,
        end=date(2024, 4, 30),
        config=BacktestConfig(
            slippage_bps=0.0,
            cash_yield_annual_pct=200.0,
            drift_band_pct=0.05,
            use_regime=False,
            apply_risk_layer=True,
            rebalance_frequency="weekly",
        ),
        settings=settings,
    )

    assert result.max_exposure_date is not None
    assert result.max_exposure_date in {point["date"] for point in result.equity_curve}


def test_gate_bypass_switches_are_off_by_default_and_full_bypass_repeats_rebalances(
    tmp_path,
) -> None:
    """Test-only gate bypasses must expose the pre-54A repeated-rebalance path."""
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        risk=RiskConfig(
            max_position_pct=0.25, min_cash_reserve_pct=0.0, min_order_notional_usd=1.0
        ),
    )
    default_cfg = _zero_cost_config().model_copy(
        update={"apply_risk_layer": True, "cash_yield_annual_pct": 200.0}
    )
    assert default_cfg.bypass_rotation_gate is False
    assert default_cfg.bypass_drift_gate is False
    assert default_cfg.bypass_target_refresh_gate is False
    data = {"SPY": _flat_ohlcv(120, date(2024, 1, 2))}

    baseline = BacktestEngine().run(
        _HoldSpyStrategy(),
        data,
        start=date(2024, 1, 2),
        end=date(2024, 4, 30),
        config=default_cfg,
        settings=settings,
    )
    bypassed = BacktestEngine().run(
        _HoldSpyStrategy(),
        data,
        start=date(2024, 1, 2),
        end=date(2024, 4, 30),
        config=default_cfg.model_copy(
            update={
                "bypass_rotation_gate": True,
                "bypass_drift_gate": True,
                "bypass_target_refresh_gate": True,
            }
        ),
        settings=settings,
    )

    assert len(baseline.trades) == 1
    assert len(bypassed.trades) > len(baseline.trades)
