"""Union of all OHLCV symbols required by the current configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Settings


def _norm_symbol(s: str) -> str:
    """Normalize ticker: strip whitespace and uppercase."""
    return str(s).strip().upper()


def resolve_required_symbols_with_sources(settings: Settings) -> dict[str, set[str]]:
    """Return each configuration source mapped to the symbols it contributes.

    Keys are diagnostic labels (e.g. ``'momentum.cash_symbol'``). Only sources
    that contribute at least one non-empty symbol are included.

    See :func:`resolve_required_symbols` for which settings fields are scanned.
    """
    sources: dict[str, set[str]] = {}

    uni = {_norm_symbol(x) for x in settings.data.universe if str(x).strip()}
    if uni:
        sources["data.universe"] = uni

    dca = {_norm_symbol(t.symbol) for t in settings.data.dca_targets if str(t.symbol).strip()}
    if dca:
        sources["data.dca_targets"] = dca

    cash = _norm_symbol(settings.strategy.momentum.cash_symbol)
    if cash:
        sources["momentum.cash_symbol"] = {cash}

    mr = {_norm_symbol(x) for x in settings.strategy.mean_reversion.universe if str(x).strip()}
    if mr:
        sources["mean_reversion.universe"] = mr

    if settings.cash_sweep.enabled:
        sweep = _norm_symbol(settings.cash_sweep.symbol)
        if sweep:
            sources["cash_sweep.symbol"] = {sweep}

    tax_syms: set[str] = set()
    for k, v in settings.tax.replacement_map.items():
        if str(k).strip():
            tax_syms.add(_norm_symbol(k))
        if str(v).strip():
            tax_syms.add(_norm_symbol(v))
    if tax_syms:
        sources["tax.replacement_map"] = tax_syms

    return sources


def resolve_required_symbols(settings: Settings) -> set[str]:
    """Return the union of all OHLCV symbols any enabled configuration feature needs.

    Includes:

    - ``settings.data.universe``
    - ``settings.data.dca_targets`` symbols
    - ``settings.strategy.momentum.cash_symbol`` (e.g. SHV for momentum cash)
    - ``settings.strategy.mean_reversion.universe``
    - ``settings.cash_sweep.symbol`` when ``cash_sweep.enabled`` (needs a price each cycle)
    - keys and values of ``settings.tax.replacement_map`` (TLH substitutes)

    Does **not** include ``settings.regime.vix_symbol`` (handled on ingest paths
    that need regime), FRED series IDs, or SEC tickers.

    Returns:
        Normalized (``strip().upper()``) tickers as a set.
    """
    out: set[str] = set()
    for syms in resolve_required_symbols_with_sources(settings).values():
        out |= syms
    return out
