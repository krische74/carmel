"""Shared strategy factory map for live workflow and backtest CLI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.config import Settings
    from src.strategy.base import Strategy

logger = logging.getLogger(__name__)

# Recognized in ``strategy.enabled`` but not built as a ``Strategy`` (workflow-only TLH).
_WORKFLOW_ONLY_STRATEGY_KEYS = frozenset({"tax_loss_harvest"})


def strategy_factory_map(settings: Settings) -> dict[str, Callable[[], Strategy]]:
    """Map normalized strategy keys to zero-arg factories (fresh ``Settings`` snapshot)."""
    from src.strategy.dca import DCAStrategy
    from src.strategy.mean_reversion import MeanReversionStrategy
    from src.strategy.momentum import MomentumRotationStrategy

    return {
        "momentum": lambda: MomentumRotationStrategy(settings),
        "dca": lambda: DCAStrategy(settings),
        "mean_reversion": lambda: MeanReversionStrategy(settings),
    }


def build_strategies_from_enabled(settings: Settings) -> list[Strategy]:
    """Instantiate strategies from ``settings.strategy.enabled``, deduped by key (order preserved)."""
    factories = strategy_factory_map(settings)
    seen: set[str] = set()
    out: list[Strategy] = []
    for name in settings.strategy.enabled:
        key = str(name).strip().lower()
        if key in _WORKFLOW_ONLY_STRATEGY_KEYS:
            continue
        if key not in factories:
            logger.warning("Unknown strategy name in strategy.enabled: %r (skipped).", name)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(factories[key]())
    if not out:
        msg = "No valid strategies after resolving strategy.enabled; check configuration."
        raise ValueError(msg)
    return out


def build_strategy_for_backtest(settings: Settings, strategy_name: str) -> Strategy:
    """Construct a single strategy for ``carmel backtest``."""
    key = str(strategy_name).strip().lower()
    factories = strategy_factory_map(settings)
    if key not in factories:
        msg = f"Unknown strategy {strategy_name!r} (use momentum, dca, or mean_reversion)."
        raise ValueError(msg)
    return factories[key]()
