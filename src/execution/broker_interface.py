"""Abstract broker — all live trading goes through this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from src.execution.leverage import LeverageSnapshot


class BrokerInterface(ABC):
    """Contract for account state and order submission (paper or live)."""

    @abstractmethod
    def get_account_id(self) -> str:
        """Stable hub partition id for this connection (configured label or broker account id)."""

    @abstractmethod
    def get_account_equity(self) -> float:
        """Total account equity (cash + positions, mark-to-market)."""

    @abstractmethod
    def get_last_equity(self) -> float:
        """Previous trading day's closing equity (for day-change calculation)."""

    @abstractmethod
    def get_cash(self) -> float:
        """Settled cash available for new buys."""

    @abstractmethod
    def get_position_qty(self, symbol: str) -> float:
        """Net position size in shares (positive long, negative short)."""

    def get_all_positions(self) -> dict[str, float]:
        """Return all broker positions as ``{SYMBOL: qty}``.

        Non-abstract: adapters that cannot batch-read raise ``NotImplementedError``.
        ``snapshot_from_broker`` falls back to per-symbol ``get_position_qty``.
        """
        raise NotImplementedError

    def get_leverage_snapshot(self) -> LeverageSnapshot:
        """Return cash and independent margin/leverage signals from the broker."""

        raise NotImplementedError

    @abstractmethod
    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        *,
        client_order_id: str | None = None,
    ) -> str:
        """Submit a market order; return broker order id.

        When ``client_order_id`` is set, brokers should treat it as an idempotency key
        (e.g. Alpaca ``client_order_id``) so duplicate submits collapse to one order.
        """

    @abstractmethod
    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        *,
        limit_price: float,
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> str:
        """Submit a limit order; return broker order id."""

    @abstractmethod
    def submit_stop_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        *,
        stop_price: float,
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> str:
        """Submit a stop order; return broker order id."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order; return True if cancelled, False if already filled/cancelled."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Return order status with at least: order_id, status, filled_qty, filled_avg_price."""

    @abstractmethod
    def refresh_account(self) -> None:
        """Refresh any cached account snapshot from the broker (no-op if not cached)."""

    @abstractmethod
    def get_order_fill_price(self, order_id: str) -> float | None:
        """Return the average fill price for a completed order, or None if not yet filled."""

    @abstractmethod
    def list_recent_orders(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent orders with at least: order_id, symbol, side, qty, filled_qty, status, filled_avg_price."""
