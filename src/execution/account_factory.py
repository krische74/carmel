"""Build one or more broker adapters from settings (multi-account Alpaca)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.execution.alpaca_adapter import AlpacaBrokerAdapter

if TYPE_CHECKING:
    from src.config import Settings
    from src.execution.broker_interface import BrokerInterface


def resolve_sqlite_account_id(broker: BrokerInterface) -> str:
    """Return a non-empty hub partition id; non-string broker mocks map to ``default``."""
    try:
        raw = broker.get_account_id()
    except Exception:
        return "default"
    if not isinstance(raw, str):
        return "default"
    s = raw.strip()
    return s if s else "default"


def build_brokers(settings: Settings) -> dict[str, BrokerInterface]:
    """Map account label to broker.

    When ``broker.accounts`` is empty, uses legacy ``ALPACA_API_KEY`` /
    ``ALPACA_SECRET_KEY`` and a single logical id from ``broker.default_account``
    or ``\"default\"``.
    """
    accts = list(settings.broker.accounts)
    paper = bool(settings.broker.paper_trading)
    if not accts:
        key = (settings.broker.default_account or "").strip() or "default"
        api = (settings.alpaca_api_key or "").strip()
        sec = (settings.alpaca_secret_key or "").strip()
        if not api or not sec:
            msg = (
                "Alpaca API credentials are missing. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in the environment, or configure broker.accounts with "
                "per-account env vars."
            )
            raise ValueError(msg)
        br = AlpacaBrokerAdapter.create(
            api,
            sec,
            paper=paper,
            logical_account_id=key,
        )
        return {key: br}

    out: dict[str, BrokerInterface] = {}
    for ac in accts:
        label = (ac.name or "").strip()
        if not label:
            label = (settings.broker.default_account or "").strip() or "default"
        api = os.environ.get(ac.api_key_env, "").strip()
        sec = os.environ.get(ac.api_secret_env, "").strip()
        if not api or not sec:
            msg = (
                f"Missing credentials for broker account {label!r}: "
                f"set {ac.api_key_env} and {ac.api_secret_env}"
            )
            raise ValueError(msg)
        out[label] = AlpacaBrokerAdapter.create(
            api,
            sec,
            paper=paper,
            logical_account_id=label,
        )
    return out
