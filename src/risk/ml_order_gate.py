"""Filter strategy signals using ML scores (optional ``ml.affect_orders``)."""

from __future__ import annotations

import logging
import math
from typing import Literal

from src.models import Signal

logger = logging.getLogger(__name__)


def filter_signals_by_ml_scores(
    signals: list[Signal],
    score_by_symbol: dict[str, float],
    *,
    threshold: float,
    missing_score_action: Literal["pass", "block"],
) -> tuple[list[Signal], int, int]:
    """Drop signals whose symbol ML score is below ``threshold``.

    Missing or NaN scores follow ``missing_score_action`` (``pass`` keeps the signal,
    ``block`` drops it).

    Returns:
        Tuple of ``(filtered_signals, passed_count, original_count)``.
    """
    n = len(signals)
    out: list[Signal] = []
    for sig in signals:
        sym = sig.symbol.strip().upper()
        raw = score_by_symbol.get(sym)
        if raw is None:
            if missing_score_action == "block":
                logger.info(
                    "ML gate: dropped %s (no ML score in latest batch), policy=block",
                    sym,
                )
                continue
            out.append(sig)
            continue
        try:
            sc = float(raw)
        except (TypeError, ValueError):
            if missing_score_action == "block":
                logger.info("ML gate: dropped %s (non-numeric ML score), policy=block", sym)
                continue
            out.append(sig)
            continue
        if math.isnan(sc):
            if missing_score_action == "block":
                logger.info("ML gate: dropped %s (NaN ML score), policy=block", sym)
                continue
            out.append(sig)
            continue
        if sc < threshold:
            logger.info(
                "ML gate: dropped %s (score=%.4f < threshold=%.4f)",
                sym,
                sc,
                threshold,
            )
            continue
        out.append(sig)
    passed = len(out)
    logger.info("ML gate: %d of %d signals passed (threshold=%.4f)", passed, n, threshold)
    return out, passed, n
