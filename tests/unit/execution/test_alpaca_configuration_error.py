"""Alpaca broker must fail loudly when credentials are rejected (no silent mock broker)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from alpaca.common.exceptions import APIError

from src.execution.alpaca_adapter import AlpacaBrokerAdapter
from src.execution.errors import ConfigurationError


def test_alpaca_adapter_create_raises_configuration_error_when_account_probe_fails() -> None:
    client = MagicMock()
    client.get_account.side_effect = APIError("bad creds")

    with (
        patch("src.execution.alpaca_adapter.TradingClient", return_value=client),
        pytest.raises(ConfigurationError, match="Alpaca rejected API credentials"),
    ):
        AlpacaBrokerAdapter.create("bad", "creds", paper=True)

    client.get_account.assert_called_once()
