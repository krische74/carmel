"""Dollar-cost averaging strategy — scheduled allocation to configured targets."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.models import Signal
from src.strategy.base import Strategy
from src.strategy.regime_params import effective_dca_amount

if TYPE_CHECKING:
    import pandas as pd

    from src.config import Settings
    from src.data.regime import MarketRegime


def _is_dca_day(as_of: datetime, frequency: str) -> bool:
    """Return True when ``as_of`` falls on a contribution day for ``frequency``."""
    freq = frequency.strip().lower()
    if freq == "weekly":
        return as_of.weekday() == 0
    if freq == "monthly":
        return as_of.day <= 7 and as_of.weekday() == 0
    if freq == "daily":
        return True
    return as_of.weekday() == 0


class DCAStrategy(Strategy):
    """Periodic buys split across ``Settings.data.dca_targets`` by weight."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_universe(self) -> list[str]:
        return [t.symbol.strip().upper() for t in self._settings.data.dca_targets]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: MarketRegime | None = None,
    ) -> list[Signal]:
        when = as_of or datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)

        if not _is_dca_day(when, self._settings.strategy.dca.frequency):
            return []

        targets = self._settings.data.dca_targets
        if not targets:
            return []

        total_w = sum(float(t.weight) for t in targets)
        if total_w <= 0:
            return []

        overall = market_regime.overall if market_regime is not None else None
        base_amt = effective_dca_amount(self._settings, None)
        amount = effective_dca_amount(self._settings, overall)
        scale = amount / base_amt if base_amt > 1e-12 else 0.0
        if scale < 1e-12:
            return []
        regime_note = ""
        if overall is not None and abs(amount - base_amt) > 1e-9:
            regime_note = (
                f" Regime-adjusted total ({overall.value}) is ${amount:.0f} vs base ${base_amt:.0f}."
            )
        percent_note = ""
        pct = self._settings.strategy.dca.percent_of_equity
        if pct is not None:
            percent_note = (
                f" Base budget target is {float(pct) * 100.0:.2f}% of equity "
                "(resolved at execution)."
            )
        out: list[Signal] = []
        for t in targets:
            sym = t.symbol.strip().upper()
            w = (float(t.weight) / total_w) * scale
            rationale = (
                f"Dollar-cost averaging ({self._settings.strategy.dca.frequency}): "
                f"allocate {w:.0%} of this ${amount:.0f} contribution to {sym}. "
                "This follows the fixed policy weights in settings, not short-term market timing."
                f"{regime_note}{percent_note}"
            )
            out.append(
                Signal(
                    symbol=sym,
                    direction="long",
                    weight=round(w, 6),
                    confidence=1.0,
                    rationale=rationale,
                    timestamp=when,
                    strategy_name="DCAStrategy",
                )
            )
        return out
