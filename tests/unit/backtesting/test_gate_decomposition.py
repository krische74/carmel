"""Discrimination test for the Tier 54F gate-bypass instrument."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.models import Signal
from src.strategy.base import Strategy


class _StableTargetStrategy(Strategy):
    """Always target the same synthetic SPY position."""

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
                rationale="Synthetic stable target.",
                timestamp=when,
                strategy_name="StableTargetStrategy",
            )
        ]


class _SequenceStrategy(Strategy):
    """Emit a position, then no signal, then the same position."""

    def __init__(self) -> None:
        self._calls = 0

    def get_universe(self) -> list[str]:
        return ["SPY"]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: object | None = None,
    ) -> list[Signal]:
        self._calls += 1
        if self._calls == 2:
            return []
        when = as_of or datetime.now(UTC)
        return [
            Signal(
                symbol="SPY",
                direction="long",
                weight=1.0,
                confidence=1.0,
                rationale="Synthetic sequenced target.",
                timestamp=when,
                strategy_name="SequenceStrategy",
            )
        ]


def _synthetic_data() -> dict[str, pd.DataFrame]:
    """Return a short, deterministic price path for the gate test."""
    idx = pd.bdate_range(date(2024, 1, 2), periods=40, freq="B")
    close = 100.0 + np.arange(len(idx), dtype=float)
    return {
        "SPY": pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.full(len(idx), 1_000_000.0),
            },
            index=idx,
        )
    }


def _trade_sequence(config_update: dict[str, bool]) -> list[tuple[str, str, str]]:
    """Run one single-bypass configuration and return its trade sequence."""
    config = BacktestConfig(
        rebalance_frequency="weekly",
        slippage_bps=0.0,
        cash_yield_annual_pct=0.0,
        drift_band_pct=1.0,
        apply_risk_layer=False,
        **config_update,
    )
    result = BacktestEngine().run(
        _StableTargetStrategy(),
        _synthetic_data(),
        start=date(2024, 1, 2),
        end=date(2024, 2, 27),
        config=config,
    )
    return [(trade.date, trade.symbol, trade.side) for trade in result.trades]


# Not a valid discriminator yet: a stable one-symbol target cannot diverge after its opening buy.
@pytest.mark.xfail(reason="synthetic fixture cannot distinguish the gate bypass paths")
def test_single_gate_bypasses_produce_distinct_trade_sequences() -> None:
    """Each gate bypass must isolate a distinct execution path."""
    rotation = _trade_sequence({"bypass_rotation_gate": True})
    drift = _trade_sequence({"bypass_drift_gate": True})
    target_refresh = _trade_sequence({"bypass_target_refresh_gate": True})

    assert len({tuple(rotation), tuple(drift), tuple(target_refresh)}) == 3


def test_legacy_empty_target_skip_preserves_position_and_counts_liquidation() -> None:
    """The legacy flag skips empty-target liquidation while recording the period."""
    common = {
        "rebalance_frequency": "weekly",
        "slippage_bps": 0.0,
        "cash_yield_annual_pct": 0.0,
        "drift_band_pct": 1.0,
        "apply_risk_layer": False,
    }
    current = BacktestEngine().run(
        _SequenceStrategy(),
        _synthetic_data(),
        start=date(2024, 1, 2),
        end=date(2024, 1, 19),
        config=BacktestConfig(**common),
    )
    legacy = BacktestEngine().run(
        _SequenceStrategy(),
        _synthetic_data(),
        start=date(2024, 1, 2),
        end=date(2024, 1, 19),
        config=BacktestConfig(**common, legacy_empty_target_skip=True),
    )

    assert current.empty_target_periods == 1
    assert current.empty_target_liquidation_trades == 1
    assert legacy.empty_target_periods == 1
    assert legacy.empty_target_liquidation_trades == 0
    assert len(current.trades) == 3
    assert len(legacy.trades) == 1
