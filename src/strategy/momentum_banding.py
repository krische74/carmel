"""Asymmetric momentum banding (Tier 53D) — relative switch banded; cash exit unbanded."""

from __future__ import annotations

import math


def cross_sectional_sigma(scores: dict[str, float]) -> float:
    """Population stdev of score values; 0 when fewer than two finite scores."""
    vals = [float(v) for v in scores.values() if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)


def select_with_asymmetric_band(
    *,
    ranked_risk: list[str],
    scores: dict[str, float],
    incumbent_risk: str | None,
    cash_symbol: str,
    band_k: float | None,
    top_n: int = 1,
) -> list[tuple[str, float]]:
    """Return ``(symbol, weight)`` targets after asymmetric banding.

    - If ``ranked_risk`` is empty → cash at weight 1 (absolute exit / stay in cash).
      **Never banded.**
    - Re-entry from cash (``incumbent_risk is None``) → take top_n unbanded.
    - Relative switch while holding a risk asset → challenger must beat incumbent
      by ``band_k * sigma``; otherwise keep incumbent (or fill top_n from held set).

    Returns cash as a single ``(cash_symbol, 1.0)`` entry when rotating to cash.
    """
    n = max(1, int(top_n))
    if not ranked_risk:
        return [(cash_symbol.strip().upper(), 1.0)]

    ranked = [s.strip().upper() for s in ranked_risk]
    if band_k is None or incumbent_risk is None:
        chosen = ranked[:n]
        w = 1.0 / len(chosen)
        return [(s, w) for s in chosen]

    inc = incumbent_risk.strip().upper()
    sigma = cross_sectional_sigma(scores)
    band = float(band_k) * sigma
    challenger = ranked[0]
    inc_score = float(scores.get(inc, float("-inf")))
    chal_score = float(scores.get(challenger, float("-inf")))

    if challenger == inc or chal_score >= inc_score + band:
        chosen = ranked[:n]
    else:
        chosen = [inc]
        for s in ranked:
            if s not in chosen:
                chosen.append(s)
            if len(chosen) >= n:
                break
    w = 1.0 / len(chosen)
    return [(s, w) for s in chosen]
