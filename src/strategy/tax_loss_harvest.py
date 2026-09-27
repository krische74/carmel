"""Tax-loss harvesting: scan open lots for harvestable losses (wash-sale aware)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Settings
    from src.portfolio.tax_lots import ClosedLot, LotLedger

logger = logging.getLogger(__name__)

WASH_SALE_LOOKBACK_DAYS = 30


class HarvestCandidate(BaseModel):
    """A lot eligible for tax-loss harvesting."""

    lot_id: str
    symbol: str
    qty: float
    cost_per_share: float
    current_price: float
    unrealized_loss: float = Field(description="Negative when at a loss.")
    unrealized_loss_pct: float = Field(description="Negative fraction, e.g. -0.064 for -6.4%.")
    replacement_symbol: str | None = None
    reason: str = ""


class TaxLossHarvester:
    """Scan open lots for tax-loss harvest candidates, respecting wash-sale rules."""

    def __init__(
        self,
        *,
        lot_ledger: LotLedger,
        settings: Settings,
        account_id: str | None = None,
    ) -> None:
        self._ledger = lot_ledger
        self._settings = settings
        self._account_id = account_id

    def find_candidates(
        self,
        current_prices: dict[str, float],
        protected_symbols: set[str],
        as_of: datetime | None = None,
    ) -> list[HarvestCandidate]:
        """Return loss lots sorted by largest loss first (most tax benefit)."""
        if not self._settings.tax.harvest_enabled:
            return []

        when = as_of or datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        when_utc = when.astimezone(UTC)
        as_date = when_utc.date()

        cfg = self._settings.tax
        pct_thr = float(cfg.harvest_threshold_pct)
        min_usd = float(cfg.harvest_min_loss_dollars)
        repl = {k.strip().upper(): v.strip().upper() for k, v in cfg.replacement_map.items()}

        open_lots = self._ledger.get_open_lots(account_id=self._account_id)
        closed_all = self._ledger.get_closed_lots(limit=5_000, account_id=self._account_id)

        protected_upper = {p.strip().upper() for p in protected_symbols}
        out: list[HarvestCandidate] = []
        for lot in open_lots:
            sym = lot.symbol.strip().upper()
            if sym in protected_upper:
                logger.debug("Skipping harvest for protected symbol %s", sym)
                continue

            px = float(current_prices.get(sym, 0.0))
            if px <= 0.0:
                continue

            basis = lot.cost_per_share
            unrealized = lot.qty * (px - basis)
            if unrealized >= 0.0:
                continue

            loss_abs = abs(unrealized)
            loss_pct = abs((px - basis) / basis) if abs(basis) > 1e-12 else 0.0
            if loss_pct < pct_thr or loss_abs < min_usd:
                continue

            opened = lot.opened_at
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=UTC)
            opened_d = opened.astimezone(UTC).date()
            if (as_date - opened_d).days < WASH_SALE_LOOKBACK_DAYS:
                logger.debug(
                    "Skipping %s lot %s: opened within %d days",
                    sym,
                    lot.id,
                    WASH_SALE_LOOKBACK_DAYS,
                )
                continue

            if self._had_recent_sale(closed_all, sym, when_utc):
                logger.debug("Skipping %s: sale within past %d days", sym, WASH_SALE_LOOKBACK_DAYS)
                continue

            repl_sym = repl.get(sym)
            reason = (
                f"{sym} lot opened {opened_d.isoformat()} has unrealized loss of "
                f"-${loss_abs:,.2f} ({-loss_pct * 100:.1f}%). "
            )
            if repl_sym:
                reason += f"Replacement: {repl_sym}."
            else:
                reason += "No replacement configured (cash out position)."

            out.append(
                HarvestCandidate(
                    lot_id=lot.id,
                    symbol=sym,
                    qty=float(lot.qty),
                    cost_per_share=float(basis),
                    current_price=px,
                    unrealized_loss=unrealized,
                    unrealized_loss_pct=(px - basis) / basis if abs(basis) > 1e-12 else 0.0,
                    replacement_symbol=repl_sym,
                    reason=reason,
                ),
            )

        out.sort(key=lambda c: c.unrealized_loss)
        return out

    def _had_recent_sale(self, closed: list[ClosedLot], symbol: str, as_of: datetime) -> bool:
        """True if any closed lot for ``symbol`` was sold in the lookback window ending at ``as_of``."""
        sym_u = symbol.strip().upper()
        end_d = as_of.astimezone(UTC).date()
        start_d = end_d - timedelta(days=WASH_SALE_LOOKBACK_DAYS)
        for cl in closed:
            if str(cl.symbol).strip().upper() != sym_u:
                continue
            cd = (
                cl.closed_at.astimezone(UTC).date() if cl.closed_at.tzinfo else cl.closed_at.date()
            )
            if start_d <= cd <= end_d:
                return True
        return False
