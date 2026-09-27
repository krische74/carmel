"""Tests for shared Pydantic models (Signal)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.models import Signal


def test_signal_requires_non_empty_rationale() -> None:
    with pytest.raises(ValidationError):
        Signal(
            symbol="SPY",
            direction="long",
            weight=1.0,
            confidence=1.0,
            rationale="",
            timestamp=datetime.now(UTC),
        )


def test_signal_accepts_valid_payload() -> None:
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.25,
        confidence=0.9,
        rationale="Test allocation based on momentum score.",
        timestamp=datetime(2024, 1, 2, tzinfo=UTC),
    )
    assert sig.symbol == "SPY"
    assert sig.weight == 0.25
    assert sig.strategy_name is None


def test_signal_optional_strategy_name() -> None:
    sig = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="Attributed to a named strategy for downstream risk and logging.",
        timestamp=datetime(2024, 6, 1, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )
    assert sig.strategy_name == "MomentumRotationStrategy"
