"""Tests for BrokerInterface ABC."""

import pytest

from src.execution.broker_interface import BrokerInterface


def test_broker_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BrokerInterface()  # type: ignore[misc]
