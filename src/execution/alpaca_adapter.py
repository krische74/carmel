"""Alpaca broker implementation of ``BrokerInterface`` (alpaca-py)."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from src.execution.leverage import LeverageSnapshot

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)

from src.execution.broker_interface import BrokerInterface
from src.execution.client_order_id import build_exec_client_order_id
from src.execution.errors import ConfigurationError, OrderNotFoundError

logger = logging.getLogger(__name__)


def _alpaca_order_missing(exc: APIError) -> bool:
    """True when Alpaca no longer has the order (paper history is not forever)."""
    code = getattr(exc, "status_code", None)
    if code == 404:
        return True
    low = str(exc).lower()
    return "not found" in low


def _alpaca_api_error_is_auth_failure(exc: APIError) -> bool:
    """Return True when Alpaca indicates invalid or unauthorized credentials."""
    code = getattr(exc, "status_code", None)
    if code in (401, 403):
        return True
    low = str(exc).lower()
    return any(
        s in low
        for s in (
            "unauthorized",
            "invalid api key",
            "access key",
            "forbidden",
            "bad creds",
        )
    )


def _time_in_force_from_str(raw: str) -> TimeInForce:
    key = raw.strip().lower()
    if key == "day":
        return TimeInForce.DAY
    if key in ("gtc",):
        return TimeInForce.GTC
    if key in ("ioc",):
        return TimeInForce.IOC
    if key in ("fok",):
        return TimeInForce.FOK
    return TimeInForce.DAY


def _alpaca_order_side(side_raw: object, *, order_id: str) -> str:
    if side_raw is None:
        logger.warning("Alpaca order %s has no side; defaulting to sell", order_id)
        return "sell"
    return "buy" if side_raw == OrderSide.BUY else "sell"


def _alpaca_order_timestamp(order: Any) -> str | None:
    for attr in ("submitted_at", "created_at", "filled_at", "updated_at"):
        raw = getattr(order, attr, None)
        if raw is None:
            continue
        if isinstance(raw, datetime):
            ts = raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
            return ts.isoformat()
        return str(raw)
    return None


def _order_to_recon_dict(order: Any) -> dict[str, Any]:
    oid = str(getattr(order, "id", "") or "")
    fq = getattr(order, "filled_qty", None)
    q = getattr(order, "qty", None)
    fap = getattr(order, "filled_avg_price", None)
    st = getattr(order, "status", None)
    cid = getattr(order, "client_order_id", None)
    return {
        "order_id": oid,
        "client_order_id": str(cid) if cid is not None else "",
        "symbol": str(getattr(order, "symbol", "") or ""),
        "side": _alpaca_order_side(getattr(order, "side", None), order_id=oid),
        "qty": float(q) if q is not None else None,
        "filled_qty": float(fq) if fq is not None else None,
        "status": str(st) if st is not None else "",
        "filled_avg_price": float(fap) if fap is not None else None,
        "timestamp": _alpaca_order_timestamp(order),
    }


def _as_float(raw: Any) -> float | None:
    """Best-effort float coercion; ``None`` when the field is absent or non-numeric."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class AlpacaBrokerAdapter(BrokerInterface):
    """Routes orders and account queries through Alpaca's trading API."""

    def __init__(
        self,
        client: TradingClient,
        *,
        logical_account_id: str | None = None,
    ) -> None:
        self._client = client
        self._cached_account: Any | None = None
        self._logical_account_id = logical_account_id

    @staticmethod
    def create(
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
        logical_account_id: str | None = None,
    ) -> AlpacaBrokerAdapter:
        """Build a trading client with explicit credentials (from env in production).

        Performs a live ``get_account`` probe so invalid keys fail immediately with
        :class:`ConfigurationError` instead of surfacing later during order submission.
        """
        client = TradingClient(api_key, secret_key, paper=paper)
        adapter = AlpacaBrokerAdapter(client, logical_account_id=logical_account_id)
        try:
            adapter.refresh_account()
        except APIError as exc:
            if _alpaca_api_error_is_auth_failure(exc):
                msg = (
                    "Alpaca rejected API credentials (check ALPACA_API_KEY / "
                    f"ALPACA_SECRET_KEY and paper vs live): {exc}"
                )
                raise ConfigurationError(msg) from exc
            msg = f"Alpaca API error while validating broker connection: {exc}"
            raise ConfigurationError(msg) from exc
        except (ConnectionError, TimeoutError, OSError) as exc:
            msg = f"Alpaca account probe failed (network): {exc}"
            raise ConfigurationError(msg) from exc
        return adapter

    def get_account_id(self) -> str:
        """Return configured logical id when set; otherwise Alpaca account ``id``."""
        if self._logical_account_id is not None:
            s = str(self._logical_account_id).strip()
            return s if s else "default"
        self._ensure_account()
        raw = getattr(self._cached_account, "id", None)
        if raw is None:
            return "default"
        s = str(raw).strip()
        return s if s else "default"

    def refresh_account(self) -> None:
        """Fetch a fresh account object and replace the cache."""
        self._cached_account = self._client.get_account()

    def _ensure_account(self) -> None:
        if self._cached_account is None:
            self._cached_account = self._client.get_account()

    def get_account_equity(self) -> float:
        self._ensure_account()
        return float(self._cached_account.equity)

    def get_last_equity(self) -> float:
        self._ensure_account()
        return float(self._cached_account.last_equity)

    def get_cash(self) -> float:
        self._ensure_account()
        return float(self._cached_account.cash)

    def get_all_positions(self) -> dict[str, float]:
        """One ``get_all_positions`` call indexed by symbol."""
        out: dict[str, float] = {}
        for pos in self._client.get_all_positions():
            out[str(pos.symbol).strip().upper()] = float(pos.qty)
        return out

    def get_position_qty(self, symbol: str) -> float:
        return self.get_all_positions().get(symbol.strip().upper(), 0.0)

    def get_leverage_snapshot(self) -> LeverageSnapshot:
        """Cash, buying power, and a borrowing check read off the broker account.

        ``maintenance_margin`` is carried for context only — see the note below
        on why it is not itself a leverage signal.
        """
        from src.execution.leverage import LeverageSnapshot

        self._ensure_account()
        acct = self._cached_account
        cash = float(acct.cash)
        bp_raw = getattr(acct, "buying_power", None)
        buying_power = float(bp_raw) if bp_raw is not None else None
        maintenance = _as_float(getattr(acct, "maintenance_margin", None))
        # Maintenance margin is NOT a borrowing signal. Alpaca posts a house
        # requirement (~30% of long market value) against fully-paid longs, so
        # `maintenance > 0` holds for any open position: the check that used to
        # live here fired on 100% of cycles from 2026-08-20 on an account that
        # had never borrowed. Real borrowing shows up as long market value
        # exceeding equity — the same invariant `cash < 0` expresses, read off
        # independent fields so the two can disagree and be caught.
        lmv = _as_float(getattr(acct, "long_market_value", None))
        equity = _as_float(getattr(acct, "equity", None))
        broker_lev = (
            lmv is not None
            and equity is not None
            and lmv > equity + max(1.0, abs(equity) * 1e-3)
        )
        return LeverageSnapshot(
            cash=cash,
            buying_power=buying_power,
            maintenance_margin=maintenance,
            broker_reports_leverage=broker_lev,
        )

    def get_order_fill(self, order_id: str) -> tuple[float | None, float | None]:
        """Return ``(filled_qty, filled_avg_price)`` from one ``get_order_by_id`` call.

        Raises :class:`OrderNotFoundError` when Alpaca no longer has the order.
        A present order with a null fill field returns ``None`` for that field —
        callers must not substitute the requested qty.
        """
        try:
            order = self._client.get_order_by_id(order_id)
        except APIError as exc:
            if _alpaca_order_missing(exc):
                raise OrderNotFoundError(order_id) from exc
            raise
        fq = getattr(order, "filled_qty", None)
        fap = getattr(order, "filled_avg_price", None)
        filled_qty = float(fq) if fq is not None else None
        filled_avg_price = float(fap) if fap is not None else None
        return filled_qty, filled_avg_price

    def get_order_fill_price(self, order_id: str) -> float | None:
        """Return Alpaca ``filled_avg_price`` when present."""
        _qty, price = self.get_order_fill(order_id)
        return price

    def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any]:
        """Look up one order by Alpaca ``client_order_id``.

        Raises :class:`OrderNotFoundError` when Alpaca no longer has the order.
        """
        cid = client_order_id.strip()
        try:
            order = self._client.get_order_by_client_id(cid)
        except APIError as exc:
            if _alpaca_order_missing(exc):
                raise OrderNotFoundError(cid) from exc
            raise
        return _order_to_recon_dict(order)

    def get_account_activities(
        self,
        *,
        after: datetime | date,
        until: datetime | date,
    ) -> list[dict[str, Any]]:
        """Return account activities in ``[after, until]`` with no type filter.

        Uses the raw REST ``GET /account/activities`` path — alpaca-py's
        ``TradingClient`` has no typed wrapper. Paginated until exhausted.
        """
        from datetime import date as date_cls

        def _as_param(v: datetime | date) -> str:
            if isinstance(v, datetime):
                ts = v if v.tzinfo is not None else v.replace(tzinfo=UTC)
                return ts.isoformat().replace("+00:00", "Z")
            if isinstance(v, date_cls):
                return v.isoformat()
            return str(v)

        out: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "after": _as_param(after),
                "until": _as_param(until),
                "direction": "asc",
                "page_size": 100,
            }
            if page_token:
                params["page_token"] = page_token
            batch = self._client.get("/account/activities", params)
            if not batch:
                break
            if isinstance(batch, dict):
                batch = [batch]
            for row in batch:
                if isinstance(row, dict):
                    out.append(dict(row))
                else:
                    # Defensive: typed model → dict
                    dumped = getattr(row, "model_dump", None)
                    out.append(dumped() if callable(dumped) else dict(row))
            if len(batch) < 100:
                break
            last = batch[-1]
            page_token = str(
                last.get("id") if isinstance(last, dict) else getattr(last, "id", "") or "",
            )
            if not page_token:
                break
        return out

    def list_recent_orders(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent orders mapped to a stable dict shape for reconciliation."""
        lim = max(1, min(int(limit), 500))
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=lim)
        orders = self._client.get_orders(filter=req)
        return [_order_to_recon_dict(o) for o in orders]

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        *,
        client_order_id: str | None = None,
    ) -> str:
        sym_u = symbol.strip().upper()
        cid = client_order_id
        if cid is None:
            cid = build_exec_client_order_id(
                account_id=self.get_account_id(),
                symbol=sym_u,
                qty=float(qty),
                side=side,
                intent="mkt",
                anchor=datetime.now(UTC),
            )
        req = MarketOrderRequest(
            symbol=sym_u,
            qty=float(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=cid,
        )
        order = self._client.submit_order(req)
        self._cached_account = None
        return str(order.id)

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
        sym_u = symbol.strip().upper()
        lp = float(limit_price)
        tif = _time_in_force_from_str(time_in_force)
        cid = client_order_id
        if cid is None:
            cid = build_exec_client_order_id(
                account_id=self.get_account_id(),
                symbol=sym_u,
                qty=float(qty),
                side=side,
                intent="lmt",
                anchor=datetime.now(UTC),
                price_key=f"{lp:.6f}",
            )
        req = LimitOrderRequest(
            symbol=sym_u,
            qty=float(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=tif,
            limit_price=lp,
            client_order_id=cid,
        )
        order = self._client.submit_order(req)
        self._cached_account = None
        return str(order.id)

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
        sym_u = symbol.strip().upper()
        sp = float(stop_price)
        tif = _time_in_force_from_str(time_in_force)
        cid = client_order_id
        if cid is None:
            cid = build_exec_client_order_id(
                account_id=self.get_account_id(),
                symbol=sym_u,
                qty=float(qty),
                side=side,
                intent="stp",
                anchor=datetime.now(UTC),
                price_key=f"{sp:.6f}",
            )
        req = StopOrderRequest(
            symbol=sym_u,
            qty=float(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.STOP,
            time_in_force=tif,
            stop_price=sp,
            client_order_id=cid,
        )
        order = self._client.submit_order(req)
        self._cached_account = None
        return str(order.id)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            logger.warning("cancel_order failed for %s: %s", order_id, exc)
            return False

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        order = self._client.get_order_by_id(order_id)
        fq = getattr(order, "filled_qty", None)
        fap = getattr(order, "filled_avg_price", None)
        st = getattr(order, "status", None)
        return {
            "order_id": str(getattr(order, "id", order_id)),
            "status": str(st) if st is not None else "",
            "filled_qty": float(fq) if fq is not None else 0.0,
            "filled_avg_price": float(fap) if fap is not None else None,
        }
