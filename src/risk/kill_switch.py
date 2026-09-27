"""Daily loss kill switch — halts new risk until reset."""

from __future__ import annotations


class KillSwitch:
    """Latches halted state when daily P&L breaches the configured loss limit."""

    def __init__(self, daily_loss_limit_pct: float) -> None:
        self._limit = float(daily_loss_limit_pct)
        self._halted = False

    def arm_from_daily_pnl(self, daily_pnl_pct: float) -> None:
        """Set halted if ``daily_pnl_pct`` is at or below ``-daily_loss_limit_pct``."""
        if float(daily_pnl_pct) <= -self._limit:
            self._halted = True

    def reset(self) -> None:
        """Clear latched halt (e.g. new trading day after manual review)."""
        self._halted = False

    @property
    def halted(self) -> bool:
        """Whether trading is halted due to a prior breach."""
        return self._halted

    def is_halted(self, daily_pnl_pct: float) -> bool:
        """True if latched halted or today's P&L breaches the loss limit."""
        return self._halted or float(daily_pnl_pct) <= -self._limit
