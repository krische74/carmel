"""Unit tests for tactical signal ensemble merge."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.config import EnsembleConfig
from src.models import Signal
from src.strategy.ensemble import merge_signals

_TS = datetime(2026, 4, 1, 16, 0, tzinfo=UTC)


def _cfg(**kwargs: object) -> EnsembleConfig:
    return EnsembleConfig(**kwargs)


def test_ensemble_disabled_returns_signals_unchanged() -> None:
    inp = [
        Signal(
            symbol="SPY",
            direction="long",
            weight=0.5,
            confidence=0.5,
            rationale="m",
            timestamp=_TS,
            strategy_name="MomentumRotationStrategy",
        ),
    ]
    out = merge_signals(inp, ensemble_config=_cfg(enabled=False), timestamp=_TS)
    assert len(out) == 1
    assert out[0] is inp[0]


def test_dca_signals_pass_through_unchanged() -> None:
    dca = Signal(
        symbol="VOO",
        direction="long",
        weight=0.33,
        confidence=1.0,
        rationale="DCA",
        timestamp=_TS,
        strategy_name="DCAStrategy",
    )
    mom = Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=0.5,
        rationale="mom",
        timestamp=_TS,
        strategy_name="MomentumRotationStrategy",
    )
    out = merge_signals(
        [dca, mom],
        ensemble_config=_cfg(enabled=True),
        timestamp=_TS,
    )
    dca_out = [s for s in out if s.strategy_name == "DCAStrategy"]
    assert len(dca_out) == 1
    assert dca_out[0].symbol == "VOO"
    assert dca_out[0].weight == pytest.approx(0.33)


def test_single_strategy_long_becomes_ensemble_with_same_weight() -> None:
    mom = Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=0.6,
        rationale="target SPY",
        timestamp=_TS,
        strategy_name="MomentumRotationStrategy",
    )
    out = merge_signals([mom], ensemble_config=_cfg(enabled=True), timestamp=_TS)
    ens = [s for s in out if s.strategy_name == "Ensemble"]
    assert len(ens) == 1
    assert ens[0].symbol == "SPY"
    assert ens[0].direction == "long"
    assert ens[0].weight == pytest.approx(1.0)
    assert ens[0].confidence == pytest.approx(0.6)
    assert "Ensemble:" in ens[0].rationale


def test_two_strategies_same_symbol_blended() -> None:
    mom = Signal(
        symbol="SPY",
        direction="long",
        weight=0.6,
        confidence=0.5,
        rationale="m",
        timestamp=_TS,
        strategy_name="MomentumRotationStrategy",
    )
    mr = Signal(
        symbol="SPY",
        direction="long",
        weight=0.25,
        confidence=0.7,
        rationale="mr",
        timestamp=_TS,
        strategy_name="MeanReversionStrategy",
    )
    out = merge_signals([mom, mr], ensemble_config=_cfg(enabled=True), timestamp=_TS)
    ens = [s for s in out if s.strategy_name == "Ensemble"]
    assert len(ens) == 1
    # (0.6*0.5 + 0.25*0.5) / (0.5+0.5) = 0.425
    assert ens[0].weight == pytest.approx(0.425)
    assert ens[0].confidence == pytest.approx(0.7)
    assert "MomentumRotation" in ens[0].rationale
    assert "MeanReversion" in ens[0].rationale


def test_cash_signal_penalizes_tactical_longs() -> None:
    cash = Signal(
        symbol="SHV",
        direction="cash",
        weight=1.0,
        confidence=0.5,
        rationale="to cash",
        timestamp=_TS,
        strategy_name="MomentumRotationStrategy",
    )
    mr = Signal(
        symbol="QQQ",
        direction="long",
        weight=0.4,
        confidence=0.6,
        rationale="mr",
        timestamp=_TS,
        strategy_name="MeanReversionStrategy",
    )
    out = merge_signals(
        [cash, mr],
        ensemble_config=_cfg(enabled=True, cash_signal_penalty=0.5),
        timestamp=_TS,
    )
    ens = [s for s in out if s.strategy_name == "Ensemble"]
    assert len(ens) == 1
    # unpenalized blend = 0.4; * 0.5 = 0.2
    assert ens[0].weight == pytest.approx(0.2)
    assert not any(s.direction == "cash" for s in out)


def test_cash_signal_passes_through_when_no_tactical_longs_survive() -> None:
    cash = Signal(
        symbol="SHV",
        direction="cash",
        weight=1.0,
        confidence=0.5,
        rationale="to cash",
        timestamp=_TS,
        strategy_name="MomentumRotationStrategy",
    )
    # MR-only long: blend 0.08; with cash penalty 0.5 -> 0.04 < min_blended_weight 0.05
    mr = Signal(
        symbol="QQQ",
        direction="long",
        weight=0.08,
        confidence=0.6,
        rationale="mr",
        timestamp=_TS,
        strategy_name="MeanReversionStrategy",
    )
    out = merge_signals(
        [cash, mr],
        ensemble_config=_cfg(
            enabled=True,
            cash_signal_penalty=0.5,
            min_blended_weight=0.05,
        ),
        timestamp=_TS,
    )
    cash_out = [s for s in out if s.direction == "cash"]
    assert len(cash_out) == 1
    assert cash_out[0].symbol == "SHV"
    assert not any(s.strategy_name == "Ensemble" for s in out)


def test_signals_below_min_blended_weight_dropped() -> None:
    mom = Signal(
        symbol="IWM",
        direction="long",
        weight=0.04,
        confidence=0.5,
        rationale="m",
        timestamp=_TS,
        strategy_name="MomentumRotationStrategy",
    )
    out = merge_signals(
        [mom],
        ensemble_config=_cfg(enabled=True, min_blended_weight=0.05),
        timestamp=_TS,
    )
    assert not any(s.strategy_name == "Ensemble" for s in out)


def test_flat_signals_pass_through() -> None:
    flat = Signal(
        symbol="SPY",
        direction="flat",
        weight=0.0,
        confidence=0.65,
        rationale="flat info",
        timestamp=_TS,
        strategy_name="MeanReversionStrategy",
    )
    mom = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.5,
        rationale="m",
        timestamp=_TS,
        strategy_name="MomentumRotationStrategy",
    )
    out = merge_signals([flat, mom], ensemble_config=_cfg(enabled=True), timestamp=_TS)
    flats = [s for s in out if s.direction == "flat"]
    assert len(flats) == 1
    assert flats[0].rationale == "flat info"


def test_unknown_strategy_weight_zero_dropped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    weird = Signal(
        symbol="X",
        direction="long",
        weight=0.9,
        confidence=0.5,
        rationale="w",
        timestamp=_TS,
        strategy_name="UnknownStrategyXYZ",
    )
    with caplog.at_level(logging.WARNING):
        out = merge_signals([weird], ensemble_config=_cfg(enabled=True), timestamp=_TS)
    assert not any(s.strategy_name == "Ensemble" for s in out)
    assert any("UnknownStrategyXYZ" in r.message for r in caplog.records)


def test_only_dca_no_error() -> None:
    dca = Signal(
        symbol="VOO",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="d",
        timestamp=_TS,
        strategy_name="DCAStrategy",
    )
    out = merge_signals([dca], ensemble_config=_cfg(enabled=True), timestamp=_TS)
    assert len(out) == 1
    assert out[0].strategy_name == "DCAStrategy"
