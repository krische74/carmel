"""Tests for Strategy ABC."""

import pytest

from src.strategy.base import Strategy


def test_strategy_cannot_be_instantiated() -> None:
    class Incomplete(Strategy):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[misc]


def test_strategy_concrete_has_required_methods() -> None:
    class Minimal(Strategy):
        def generate_signals(self, data, *, as_of=None, market_regime=None):
            return []

        def get_universe(self):
            return []

    s = Minimal()
    assert s.get_universe() == []
    assert s.generate_signals({}) == []
