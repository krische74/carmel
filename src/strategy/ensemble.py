"""Merge tactical signals from multiple strategies into weighted ensemble outputs."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from src.models import Signal

if TYPE_CHECKING:
    from datetime import datetime

    from src.config import EnsembleConfig

logger = logging.getLogger(__name__)


def _strategy_label(strategy_name: str | None) -> str:
    if not strategy_name:
        return "Unknown"
    if strategy_name.endswith("Strategy"):
        return strategy_name[:-8]
    return strategy_name


def merge_signals(
    signals: list[Signal],
    *,
    ensemble_config: EnsembleConfig,
    timestamp: datetime,
) -> list[Signal]:
    """Partition DCA / flat / tactical longs; blend tactical longs by configured weights.

    DCA and flat signals are never modified. When ``ensemble_config.enabled`` is False,
    returns a shallow copy of ``signals`` unchanged.
    """
    if not ensemble_config.enabled:
        return list(signals)

    dca: list[Signal] = []
    flats: list[Signal] = []
    tactical: list[Signal] = []

    for s in signals:
        if s.strategy_name == "DCAStrategy":
            dca.append(s)
        elif s.direction == "flat":
            flats.append(s)
        else:
            tactical.append(s)

    cash_penalty = any(
        s.strategy_name == "MomentumRotationStrategy" and s.direction == "cash" for s in tactical
    )
    momentum_cash_signals = [
        s
        for s in tactical
        if s.strategy_name == "MomentumRotationStrategy" and s.direction == "cash"
    ]

    longs = [s for s in tactical if s.direction == "long"]
    by_sym: dict[str, list[Signal]] = defaultdict(list)
    for s in longs:
        by_sym[s.symbol.strip().upper()].append(s)

    sw_map = ensemble_config.strategy_weights
    min_w = float(ensemble_config.min_blended_weight)
    penalty = float(ensemble_config.cash_signal_penalty)

    merged_out: list[Signal] = []

    for sym in sorted(by_sym.keys()):
        group = by_sym[sym]
        num = 0.0
        den = 0.0
        parts: list[str] = []
        max_conf = 0.0
        for s in group:
            name = s.strategy_name or ""
            sw = float(sw_map.get(name, 0.0))
            if sw <= 0.0:
                if name and name not in sw_map:
                    logger.warning(
                        "Ensemble: unknown strategy_name %r; skipping for blend",
                        name,
                    )
                continue
            w = float(s.weight)
            num += w * sw
            den += sw
            max_conf = max(max_conf, float(s.confidence))
            label = _strategy_label(name)
            parts.append(f"{label}({w:g}*{sw:g})")

        if den <= 0.0:
            continue

        blended = num / den
        if cash_penalty:
            blended *= penalty

        if blended + 1e-12 < min_w:
            continue

        rationale = "Ensemble: " + " + ".join(parts) + f" = {blended:g}"
        merged_out.append(
            Signal(
                symbol=sym,
                direction="long",
                weight=round(blended, 6),
                confidence=max_conf,
                rationale=rationale,
                timestamp=timestamp,
                strategy_name="Ensemble",
            ),
        )

    if not merged_out and momentum_cash_signals:
        merged_out.extend(momentum_cash_signals)

    return [*merged_out, *dca, *flats]
