"""Regime-derived strategy parameters (no SQLite in strategies — caller passes regime)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings
    from src.data.regime import OverallRegime


def effective_dca_amount(
    settings: Settings,
    overall: OverallRegime | None,
    *,
    equity: float | None = None,
) -> float:
    """Regime-scaled DCA dollar amount (base amount or equity-percent, then regime multiplier).

    For live ``dca_budget`` in ``TradingWorkflow``, pass ``overall=None`` so regime is applied
    only via ``DCAStrategy`` signal weights (single source of truth). Use non-``None``
    ``overall`` when the caller should include regime scaling (e.g. rationale totals).
    """
    cfg = settings.strategy.dca
    pct = cfg.percent_of_equity
    if pct is not None and equity is not None:
        base = max(0.0, float(equity)) * float(pct)
    else:
        base = float(cfg.amount)
    if overall is None:
        return base
    if overall.value == "risk_on":
        return base * float(cfg.regime_amount_risk_on)
    if overall.value == "cautious":
        return base * float(cfg.regime_amount_cautious)
    if overall.value == "defensive":
        return base * float(cfg.regime_amount_defensive)
    if overall.value == "crisis":
        return base * float(cfg.regime_amount_crisis)
    return base


def effective_mean_reversion_max_positions(
    settings: Settings, overall: OverallRegime | None
) -> int:
    """Cap on simultaneous mean-reversion longs from regime-specific limits."""
    cfg = settings.strategy.mean_reversion
    base = int(cfg.max_positions)
    if overall is None:
        return base
    if overall.value == "risk_on":
        return int(cfg.regime_max_positions_risk_on)
    if overall.value == "cautious":
        return int(cfg.regime_max_positions_cautious)
    if overall.value == "defensive":
        return int(cfg.regime_max_positions_defensive)
    if overall.value == "crisis":
        return int(cfg.regime_max_positions_crisis)
    return base


def effective_adx_threshold(settings: Settings, overall: OverallRegime | None) -> float:
    """Return the ADX cutoff used by momentum trend-strength filtering.

    Policy (see ``config/settings.yaml`` comments):

    - ``risk_on``: allow a **weaker** trend (lower ADX) than the base threshold.
    - ``cautious``: use the base ``adx_threshold``.
    - ``defensive`` / ``crisis``: require a **stronger** trend (higher ADX) — fewer
      names pass, more defensive positioning in stressed regimes.

    Optional per-regime overrides on ``MomentumConfig`` take precedence when set.
    When ``overall`` is ``None`` (regime disabled or unknown), the base threshold is used.
    """
    cfg = settings.strategy.momentum
    base = float(cfg.adx_threshold)

    if overall is None:
        return base

    ov = cfg.adx_threshold_risk_on
    oc = cfg.adx_threshold_cautious
    od = cfg.adx_threshold_defensive
    ocr = cfg.adx_threshold_crisis

    if overall.value == "risk_on":
        if ov is not None:
            return float(ov)
        return max(5.0, base - float(cfg.adx_threshold_risk_on_delta))
    if overall.value == "cautious":
        if oc is not None:
            return float(oc)
        return base
    if overall.value == "defensive":
        if od is not None:
            return float(od)
        return base + float(cfg.adx_threshold_defensive_delta)
    if overall.value == "crisis":
        if ocr is not None:
            return float(ocr)
        return base + float(cfg.adx_threshold_crisis_delta)
    return base
