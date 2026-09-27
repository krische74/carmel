"""Apply risk checks and submit orders through ``BrokerInterface``."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from src.execution.account_factory import resolve_sqlite_account_id
from src.execution.client_order_id import build_exec_client_order_id
from src.execution.errors import DuplicateClientOrderIdError
from src.execution.retry_policy import backoff_seconds, should_retry_exception
from src.models import OrderExecutionResult, Signal
from src.risk.position_sizing import compute_atr_notional, compute_order_notional
from src.risk.pre_trade_checks import evaluate_pdt_before_sell, evaluate_pre_trade_checks

if TYPE_CHECKING:
    from src.config import Settings
    from src.data.storage.sqlite_store import SQLiteStore
    from src.execution.broker_interface import BrokerInterface
    from src.portfolio.tax_lots import LotLedger
    from src.risk.kill_switch import KillSwitch

logger = logging.getLogger(__name__)
_PENDING_EXPOSURE_STATUSES = frozenset({"new", "accepted", "pending_new", "partially_filled"})
FILL_WAIT_INTERVAL_SECONDS = 0.4
FILL_WAIT_BUDGET_SECONDS = 5.0
PARTIAL_UNSWEEP_REASON = "insufficient cash after partial unsweep"


@dataclass
class _PendingBuyDebit:
    """Same-cycle buy whose cash debit may lag ``filled`` status (Tier 48A)."""

    order_id: str
    notional: float
    cash_before: float
    debit_observed: bool = False


def _limit_buy_price(last_price: float, offset_bps: float) -> float:
    """Limit buy at last price plus offset; floor at $0.01."""
    px = float(last_price) * (1.0 + float(offset_bps) / 10000.0)
    return max(0.01, float(px))


def _stop_loss_price(fill_price: float, stop_loss_pct: float) -> float:
    """Stop-market trigger below fill price; floor at $0.01."""
    px = float(fill_price) * (1.0 - float(stop_loss_pct) / 100.0)
    return max(0.01, float(px))


def _parse_execution_ts(raw: object) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    if not raw:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _bucket_order_status(raw: str) -> Literal["filled", "pending", "cancelled", "unknown"]:
    """Normalize broker status strings (Alpaca-style) to internal buckets.

    Unknown strings are logged and returned as ``\"unknown\"`` so callers can
    persist accurate state instead of assuming ``pending``.
    """
    original = str(raw).strip()
    r = original.lower().replace("orderstatus.", "")
    r_norm = r.replace(" ", "_").replace("-", "_")

    filled_states = frozenset({"filled", "done_for_day"})
    pending_states = frozenset(
        {
            "new",
            "accepted",
            "partially_filled",
            "pending_cancel",
            "pending_replace",
            "queued",
            "held",
            "pending_new",
            "accepted_for_bidding",
            "calculated",
            "new_calculated",
            "pending",
        },
    )
    terminal_states = frozenset(
        {
            "canceled",
            "cancelled",
            "expired",
            "rejected",
            "replaced",
            "failed",
        },
    )

    if r_norm in filled_states:
        return "filled"
    if r_norm in pending_states:
        return "pending"
    if r_norm in terminal_states:
        return "cancelled"
    logger.warning(
        "Unknown broker order status %r — treating as unknown",
        original,
    )
    return "unknown"


class OrderManager:
    """Turns strategy signals into broker orders after pre-trade checks."""

    def __init__(
        self,
        *,
        broker: BrokerInterface,
        settings: Settings,
        kill_switch: KillSwitch,
        sqlite_store: SQLiteStore | None = None,
        lot_ledger: LotLedger | None = None,
    ) -> None:
        self._broker = broker
        self._settings = settings
        self._kill_switch = kill_switch
        self._sqlite = sqlite_store
        self._lot_ledger = lot_ledger
        self._account_id = resolve_sqlite_account_id(broker)
        self._pending_buy_debits: list[_PendingBuyDebit] = []
        self._cycle_id: str | None = None

    def set_cycle_id(self, cycle_id: str | None) -> None:
        """Scope client-order dedupe keys to this trading cycle (Tier 49C)."""
        self._cycle_id = (cycle_id or "").strip() or None

    def _check_and_reserve_cid(self, cid: str, sym: str, qty: float, side: str) -> None:
        """Raise :class:`DuplicateClientOrderIdError` when this cid was used recently."""
        if self._sqlite is None:
            return
        if self._sqlite.has_recent_client_order_id(cid):
            msg = (
                f"Refusing duplicate {side} {sym} qty={qty:.8f}: client_order_id "
                f"{cid} was already submitted within the dedupe window. This guards "
                f"against multi-scheduler races; investigate if seen unexpectedly."
            )
            logger.warning(msg)
            raise DuplicateClientOrderIdError(msg)

    def _submit_market_order_with_retry(self, sym: str, qty: float, side: str) -> str:
        """Submit a market order with retries on transient connection failures."""
        anchor = datetime.now(UTC)
        cid = build_exec_client_order_id(
            account_id=self._account_id,
            symbol=sym,
            qty=float(qty),
            side=side,
            intent="mkt",
            anchor=anchor,
            cycle_id=self._cycle_id,
        )
        self._check_and_reserve_cid(cid, sym, qty, side)
        cfg = self._settings.scheduler.execution_retry
        max_attempts = max(1, int(cfg.max_attempts))
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                order_id = self._broker.submit_market_order(
                    sym,
                    qty,
                    side,
                    client_order_id=cid,
                )
                if self._sqlite is not None:
                    self._sqlite.record_submitted_client_order_id(cid)
                return order_id
            except Exception as exc:
                last_exc = exc
                if not should_retry_exception(exc) or attempt >= max_attempts - 1:
                    raise
                delay = backoff_seconds(
                    attempt,
                    float(cfg.backoff_base_seconds),
                    use_jitter=bool(cfg.use_jitter),
                )
                logger.warning(
                    "Order retry %d/%d for %s: %s (backoff %.2fs)",
                    attempt + 1,
                    max_attempts,
                    sym,
                    exc,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _submit_limit_order_with_retry(
        self,
        sym: str,
        qty: float,
        side: str,
        *,
        limit_price: float,
        time_in_force: str = "day",
    ) -> str:
        anchor = datetime.now(UTC)
        cid = build_exec_client_order_id(
            account_id=self._account_id,
            symbol=sym,
            qty=float(qty),
            side=side,
            intent="lmt",
            anchor=anchor,
            cycle_id=self._cycle_id,
            price_key=f"{float(limit_price):.6f}",
        )
        self._check_and_reserve_cid(cid, sym, qty, side)
        cfg = self._settings.scheduler.execution_retry
        max_attempts = max(1, int(cfg.max_attempts))
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                order_id = self._broker.submit_limit_order(
                    sym,
                    qty,
                    side,
                    limit_price=limit_price,
                    time_in_force=time_in_force,
                    client_order_id=cid,
                )
                if self._sqlite is not None:
                    self._sqlite.record_submitted_client_order_id(cid)
                return order_id
            except Exception as exc:
                last_exc = exc
                if not should_retry_exception(exc) or attempt >= max_attempts - 1:
                    raise
                delay = backoff_seconds(
                    attempt,
                    float(cfg.backoff_base_seconds),
                    use_jitter=bool(cfg.use_jitter),
                )
                logger.warning(
                    "Limit order retry %d/%d for %s: %s (backoff %.2fs)",
                    attempt + 1,
                    max_attempts,
                    sym,
                    exc,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _submit_stop_order_with_retry(
        self,
        sym: str,
        qty: float,
        side: str,
        *,
        stop_price: float,
        time_in_force: str = "day",
    ) -> str:
        anchor = datetime.now(UTC)
        cid = build_exec_client_order_id(
            account_id=self._account_id,
            symbol=sym,
            qty=float(qty),
            side=side,
            intent="stp",
            anchor=anchor,
            cycle_id=self._cycle_id,
            price_key=f"{float(stop_price):.6f}",
        )
        self._check_and_reserve_cid(cid, sym, qty, side)
        cfg = self._settings.scheduler.execution_retry
        max_attempts = max(1, int(cfg.max_attempts))
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                order_id = self._broker.submit_stop_order(
                    sym,
                    qty,
                    side,
                    stop_price=stop_price,
                    time_in_force=time_in_force,
                    client_order_id=cid,
                )
                if self._sqlite is not None:
                    self._sqlite.record_submitted_client_order_id(cid)
                return order_id
            except Exception as exc:
                last_exc = exc
                if not should_retry_exception(exc) or attempt >= max_attempts - 1:
                    raise
                delay = backoff_seconds(
                    attempt,
                    float(cfg.backoff_base_seconds),
                    use_jitter=bool(cfg.use_jitter),
                )
                logger.warning(
                    "Stop order retry %d/%d for %s: %s (backoff %.2fs)",
                    attempt + 1,
                    max_attempts,
                    sym,
                    exc,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def cancel_stale_orders(self, *, timeout_minutes: int) -> list[str]:
        """Cancel broker orders still marked pending in SQLite past ``timeout_minutes``."""
        if self._sqlite is None:
            return []
        cutoff = datetime.now(UTC) - timedelta(minutes=max(1, int(timeout_minutes)))
        cutoff_iso = cutoff.isoformat()
        stale = self._sqlite.list_stale_pending_executions(
            before_timestamp_iso=cutoff_iso,
            account_id=self._account_id,
        )
        cancelled: list[str] = []
        for row in stale:
            oid = str(row.get("order_id") or "").strip()
            if not oid:
                continue
            if self._broker.cancel_order(oid):
                self._sqlite.update_trade_execution_status_by_order_id(
                    oid,
                    order_status="cancelled",
                )
                cancelled.append(oid)
            else:
                try:
                    st = self._broker.get_order_status(oid)
                except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
                    logger.warning(
                        "get_order_status failed after cancel miss for order_id=%s: %s",
                        oid,
                        exc,
                    )
                    continue
                raw_status = str(st.get("status") or "")
                bucket = _bucket_order_status(raw_status)
                db_status = bucket if bucket != "unknown" else "unknown"
                fq = st.get("filled_qty")
                fap = st.get("filled_avg_price")
                self._sqlite.update_trade_execution_status_by_order_id(
                    oid,
                    order_status=db_status,
                    filled_qty=float(fq) if fq is not None else None,
                    filled_avg_price=float(fap) if fap is not None else None,
                )
                if db_status == "filled":
                    logger.info("Order %s filled while cancel in flight", oid)
                else:
                    logger.info(
                        "Stale cancel skipped for order_id=%s; synced status=%s from broker.",
                        oid,
                        db_status,
                    )
                self._write_through_observed_fill(
                    row,
                    observed_qty=float(fq) if fq is not None else None,
                    fill_price=float(fap) if fap is not None else None,
                    order_status=db_status,
                )
        return cancelled

    def _write_through_observed_fill(
        self,
        row: dict[str, object],
        *,
        observed_qty: float | None,
        fill_price: float | None,
        order_status: str,
    ) -> None:
        """Catch the lot ledger up when a timed-out fill lands on a later cycle.

        The timeout path records lots at ``filled_qty=0``. The next cycle repairs
        ``trade_executions`` here; without this write-through the lots stay at zero.
        A NULL prior ``filled_qty`` means the live path already used requested qty —
        do not add the observed fill on top.
        """
        if self._lot_ledger is None:
            return
        if order_status not in ("filled", "cancelled"):
            return
        prior_raw = row.get("filled_qty")
        oid = str(row.get("order_id") or "")
        if prior_raw is None:
            logger.info(
                "Order %s fill observed but prior filled_qty is NULL; not writing lots",
                oid,
            )
            return
        if observed_qty is None:
            logger.warning(
                "Order %s terminal status %s has no filled_qty; lots not updated",
                oid,
                order_status,
            )
            return
        delta = float(observed_qty) - float(prior_raw)
        if delta <= 1e-9:
            return
        if fill_price is None or float(fill_price) <= 0.0:
            logger.warning(
                "Order %s observed fill %.6f but no fill price; lots not updated",
                oid,
                delta,
            )
            return
        side = str(row.get("side") or "").strip().lower()
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or side not in ("buy", "sell"):
            logger.warning("Order %s missing symbol/side; lots not updated", oid)
            return
        when = _parse_execution_ts(row.get("timestamp"))
        aid = str(row.get("account_id") or self._account_id or "default")
        if side == "buy":
            self._lot_ledger.record_buy(symbol, delta, float(fill_price), when, account_id=aid)
        else:
            try:
                self._lot_ledger.record_sell(
                    symbol,
                    delta,
                    float(fill_price),
                    when,
                    method=self._settings.tax.lot_selection_method,
                    account_id=aid,
                )
            except ValueError as exc:
                logger.warning("Lot sell mismatch for %s after late fill: %s", symbol, exc)
                return
        logger.info(
            "Order %s late fill wrote through %.6f %s %s to lot ledger",
            oid,
            delta,
            side,
            symbol,
        )

    def _size_buy_notional(
        self,
        signal: Signal,
        *,
        price: float,
        equity: float,
        current_qty: float,
        pending_notional: float,
        dca_budget: float | None,
        atr_values: dict[str, float] | None,
        regime_multiplier: float,
    ) -> float:
        """Dollar notional for one buy signal (same path as :meth:`execute_signals`)."""
        risk = self._settings.risk
        use_dca = signal.strategy_name == "DCAStrategy" and dca_budget is not None
        atr = (atr_values or {}).get(signal.symbol.strip().upper())
        use_atr = (
            not use_dca
            and risk.sizing_method == "atr_risk_parity"
            and atr is not None
            and float(atr) > 0.0
        )
        effective_current_qty = float(current_qty) + (float(pending_notional) / float(price))
        rm = float(regime_multiplier)
        if use_atr:
            return compute_atr_notional(
                atr_value=float(atr),
                price=float(price),
                equity=equity,
                risk_pct=risk.atr_risk_pct,
                max_position_pct=risk.max_position_pct,
                regime_multiplier=rm,
            )
        return compute_order_notional(
            signal_weight=signal.weight,
            equity=equity,
            max_position_pct=risk.max_position_pct,
            dca_budget=dca_budget,
            use_dca_budget=use_dca,
            regime_multiplier=rm,
            current_position_notional=effective_current_qty * float(price),
        )

    def intended_buy_notionals(
        self,
        signals: list[Signal],
        *,
        last_prices: dict[str, float],
        equity: float,
        positions: dict[str, float],
        dca_budget: float | None = None,
        atr_values: dict[str, float] | None = None,
        regime_multiplier: float = 1.0,
    ) -> dict[str, float]:
        """Map symbol → intended buy notional using the live execution sizing path.

        Skips flat/non-buy directions, missing prices, and below-floor notionals.
        Does not apply cash-reserve or kill-switch gates — those stay in
        :meth:`execute_signals`. Used by the unsweep step so funding and
        submission cannot drift.
        """
        risk = self._settings.risk
        positions = dict(positions)
        pending_by_symbol = self._pending_buy_notional_by_symbol(last_prices=last_prices)
        out: dict[str, float] = {}
        for s in signals:
            if s.direction == "flat" or s.direction not in ("long", "cash"):
                continue
            sym = s.symbol.strip().upper()
            price = last_prices.get(sym)
            if price is None or price <= 0.0:
                continue
            cur = float(positions.get(sym, 0.0))
            pending_notional = float(pending_by_symbol.get(sym, 0.0))
            notional = self._size_buy_notional(
                s,
                price=float(price),
                equity=equity,
                current_qty=cur,
                pending_notional=pending_notional,
                dca_budget=dca_budget,
                atr_values=atr_values,
                regime_multiplier=regime_multiplier,
            )
            if notional < risk.min_order_notional_usd:
                continue
            out[sym] = float(out.get(sym, 0.0)) + float(notional)
            positions[sym] = cur + (float(notional) / float(price))
        return out

    def execute_signals(
        self,
        signals: list[Signal],
        *,
        last_prices: dict[str, float],
        equity: float,
        cash: float,
        positions: dict[str, float],
        daily_pnl_pct: float,
        dca_budget: float | None = None,
        atr_values: dict[str, float] | None = None,
        regime_multiplier: float = 1.0,
        unsweep_attempted: bool = False,
    ) -> list[OrderExecutionResult]:
        """Execute buy orders for **long** and **cash** signals; skip **flat**."""
        self._pending_buy_debits = []
        risk = self._settings.risk
        ex = self._settings.execution
        use_limit = ex.default_order_type == "limit"
        halted = self._kill_switch.is_halted(daily_pnl_pct)
        remaining_cash = float(cash)
        positions = dict(positions)
        submitted_notional_by_symbol = self._pending_buy_notional_by_symbol(last_prices=last_prices)
        out: list[OrderExecutionResult] = []
        for s in signals:
            if s.direction == "flat":
                continue
            if s.direction not in ("long", "cash"):
                continue
            sym = s.symbol.strip().upper()
            price = last_prices.get(sym)
            if price is None or price <= 0.0:
                out.append(
                    OrderExecutionResult(
                        symbol=sym,
                        submitted=False,
                        reason="Missing or non-positive last price for sizing.",
                        side="buy",
                        qty=None,
                        fill_price=None,
                    ),
                )
                continue
            cur = float(positions.get(sym, 0.0))
            pending_notional = float(submitted_notional_by_symbol.get(sym, 0.0))
            effective_current_qty = cur + (pending_notional / float(price))
            notional = self._size_buy_notional(
                s,
                price=float(price),
                equity=equity,
                current_qty=cur,
                pending_notional=pending_notional,
                dca_budget=dca_budget,
                atr_values=atr_values,
                regime_multiplier=regime_multiplier,
            )
            if notional < risk.min_order_notional_usd:
                logger.debug(
                    "order_skipped_below_floor symbol=%s notional=%.2f floor=%.2f",
                    sym,
                    float(notional),
                    float(risk.min_order_notional_usd),
                )
                out.append(
                    OrderExecutionResult(
                        symbol=sym,
                        submitted=False,
                        reason=f"Notional below minimum order size (${risk.min_order_notional_usd:.2f}).",
                        side="buy",
                        qty=None,
                        fill_price=None,
                    ),
                )
                continue
            chk = evaluate_pre_trade_checks(
                notional=notional,
                price=price,
                equity=equity,
                cash=remaining_cash,
                current_qty=effective_current_qty,
                max_position_pct=risk.max_position_pct,
                min_cash_reserve_pct=risk.min_cash_reserve_pct,
                kill_switch_halted=halted,
            )
            if not chk.ok:
                reason = chk.reason
                if unsweep_attempted and reason is not None and "minimum cash reserve" in reason:
                    reason = PARTIAL_UNSWEEP_REASON
                logger.warning(
                    "pre_trade_reject symbol=%s proposed_notional=%.2f pending_notional=%.2f current_qty=%.6f reason=%s",
                    sym,
                    float(notional),
                    pending_notional,
                    effective_current_qty,
                    reason,
                )
                out.append(
                    OrderExecutionResult(
                        symbol=sym,
                        submitted=False,
                        reason=reason,
                        side="buy",
                        qty=None,
                        fill_price=None,
                    ),
                )
                continue
            qty = notional / price
            limit_px: float | None = None
            if use_limit:
                limit_px = _limit_buy_price(float(price), float(ex.limit_offset_bps))

            cash_before_buy: float | None = None
            if not use_limit:
                try:
                    cash_before_buy = float(self._broker.get_cash())
                except (
                    ConnectionError,
                    OSError,
                    ValueError,
                    RuntimeError,
                    TimeoutError,
                    TypeError,
                ):
                    cash_before_buy = None

            try:
                if use_limit:
                    oid = self._submit_limit_order_with_retry(
                        sym,
                        qty,
                        "buy",
                        limit_price=limit_px,  # set in block above when use_limit
                    )
                else:
                    oid = self._submit_market_order_with_retry(sym, qty, "buy")
            except DuplicateClientOrderIdError as exc:
                out.append(
                    OrderExecutionResult(
                        symbol=sym,
                        submitted=False,
                        reason=f"Duplicate suppressed: {exc}",
                        side="buy",
                        qty=None,
                        fill_price=None,
                        order_type="limit" if use_limit else "market",
                        limit_price=limit_px,
                    ),
                )
                continue
            except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
                logger.exception("Broker submit failed for %s", sym)
                out.append(
                    OrderExecutionResult(
                        symbol=sym,
                        submitted=False,
                        reason=f"Broker error: {exc}",
                        side="buy",
                        qty=None,
                        fill_price=None,
                        order_type="limit" if use_limit else "market",
                        limit_price=limit_px,
                    ),
                )
                continue

            fill_price: float | None = None
            filled_qty: float | None = None
            order_status = "filled"
            buy_debited = True
            order_type: Literal["market", "limit", "stop", "stop_limit"] = (
                "limit" if use_limit else "market"
            )

            if use_limit:
                try:
                    st = self._broker.get_order_status(oid)
                except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
                    logger.warning("get_order_status failed for %s: %s", oid, exc)
                    st = {}
                raw_status = str(st.get("status") or "")
                bucket = _bucket_order_status(raw_status)
                fq = st.get("filled_qty")
                fap = st.get("filled_avg_price")
                if bucket == "filled":
                    order_status = "filled"
                    if fap is not None:
                        fill_price = float(fap)
                    if fq is not None:
                        filled_qty = float(fq)
                    if fill_price is None:
                        try:
                            fill_price = self._broker.get_order_fill_price(oid)
                        except (
                            ConnectionError,
                            OSError,
                            ValueError,
                            RuntimeError,
                            TimeoutError,
                        ) as exc:
                            logger.warning("Fill price lookup failed for order %s: %s", oid, exc)
                    if filled_qty is None:
                        filled_qty = qty
                elif bucket == "cancelled":
                    order_status = "cancelled"
                else:
                    # pending or unknown — treat as pending until next reconciliation
                    order_status = "pending"
            else:
                pending_entry = _PendingBuyDebit(
                    oid,
                    float(notional),
                    float(cash_before_buy or 0.0),
                )
                self._pending_buy_debits.append(pending_entry)
                debited = self.wait_for_fill(
                    oid,
                    side="buy",
                    expected_cash_delta=float(notional),
                    cash_before=cash_before_buy,
                )
                buy_debited = debited
                if debited:
                    pending_entry.debit_observed = True
                else:
                    logger.info(
                        "buy fill wait timed out before cash debit observed for %s; "
                        "proceeding (sweep sizes via pending-notional accounting)",
                        sym,
                    )
                bucket, fq, fap = self._order_fill_fields(oid)
                fill_price = fap
                filled_qty = fq
                if fill_price is None:
                    try:
                        fill_price = self._broker.get_order_fill_price(oid)
                    except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError):
                        fill_price = None
                if bucket == "cancelled":
                    order_status = "cancelled"
                elif bucket != "filled" and not buy_debited:
                    order_status = "pending"
                elif fill_price is not None and filled_qty is None:
                    filled_qty = qty

            if order_status == "filled":
                if buy_debited:
                    remaining_cash -= float(notional)
                positions[sym] = float(positions.get(sym, 0.0)) + qty
                if ex.stop_loss_enabled and fill_price is not None and fill_price > 0.0:
                    stop_px = _stop_loss_price(float(fill_price), float(ex.stop_loss_pct))
                    sell_qty = filled_qty if filled_qty is not None else qty
                    try:
                        self._submit_stop_order_with_retry(
                            sym,
                            float(sell_qty),
                            "sell",
                            stop_price=stop_px,
                        )
                    except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError):
                        logger.exception("Stop-loss submit failed for %s", sym)
            if order_status != "filled" or not buy_debited:
                submitted_notional_by_symbol[sym] = (
                    float(submitted_notional_by_symbol.get(sym, 0.0)) + float(notional)
                )

            out.append(
                OrderExecutionResult(
                    symbol=sym,
                    submitted=True,
                    order_id=oid,
                    side="buy",
                    qty=qty,
                    fill_price=fill_price,
                    strategy_name=s.strategy_name,
                    order_type=order_type,
                    limit_price=limit_px,
                    stop_price=None,
                    order_status=order_status,
                    filled_qty=filled_qty,
                ),
            )
        return out

    def _pending_buy_notional_by_symbol(self, *, last_prices: dict[str, float]) -> dict[str, float]:
        """Outstanding buy exposure from broker open/pending orders by symbol notional.

        Uses ``list_recent_orders(limit=100)``; very busy accounts can tail-truncate older
        open orders from the snapshot. A follow-up is to query open/pending orders explicitly
        (e.g. Alpaca ``status=open``) so stale exposure is not missed.
        """
        out: dict[str, float] = {}
        for row in self._broker.list_recent_orders(limit=100):
            status = str(row.get("status") or "").strip().lower().replace("orderstatus.", "")
            if status not in _PENDING_EXPOSURE_STATUSES:
                continue
            side = str(row.get("side") or "").strip().lower()
            if side != "buy":
                continue
            sym = str(row.get("symbol") or "").strip().upper()
            if not sym:
                continue
            notional = self._order_notional(row, fallback_price=last_prices.get(sym))
            if notional <= 0.0:
                continue
            out[sym] = float(out.get(sym, 0.0)) + float(notional)
        return out

    def _order_notional(self, row: dict[str, object], *, fallback_price: float | None) -> float:
        """Best-effort order notional from broker order fields."""
        notional_raw = row.get("notional")
        if notional_raw is not None:
            try:
                n = float(notional_raw)
            except (TypeError, ValueError):
                n = 0.0
            if n > 0.0:
                return n

        qty = 0.0
        for key in ("qty", "filled_qty"):
            v = row.get(key)
            if v is None:
                continue
            try:
                q = float(v)
            except (TypeError, ValueError):
                continue
            if q > 0.0:
                qty = q
                break
        if qty <= 0.0:
            return 0.0

        price = 0.0
        for key in ("limit_price", "filled_avg_price", "stop_price"):
            v = row.get(key)
            if v is None:
                continue
            try:
                p = float(v)
            except (TypeError, ValueError):
                continue
            if p > 0.0:
                price = p
                break
        if price <= 0.0 and fallback_price is not None:
            try:
                price = float(fallback_price)
            except (TypeError, ValueError):
                price = 0.0
        if price <= 0.0:
            return 0.0
        return qty * price

    def _order_fill_fields(
        self,
        order_id: str,
    ) -> tuple[str, float | None, float | None]:
        """Read broker order status, filled quantity, and average fill price (Tier 49B)."""
        try:
            st = self._broker.get_order_status(order_id)
        except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError):
            st = None
        if not isinstance(st, dict):
            return "filled", None, None
        bucket = _bucket_order_status(str(st.get("status") or ""))
        fq = st.get("filled_qty")
        fap = st.get("filled_avg_price")
        filled_qty = float(fq) if fq is not None else None
        fill_price = float(fap) if fap is not None else None
        if fill_price is None:
            try:
                fill_price = self._broker.get_order_fill_price(order_id)
            except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError):
                fill_price = None
        return bucket, filled_qty, fill_price

    def wait_for_fill(
        self,
        order_id: str,
        *,
        side: Literal["buy", "sell"] = "sell",
        expected_cash_delta: float | None = None,
        cash_before: float | None = None,
        interval_seconds: float = FILL_WAIT_INTERVAL_SECONDS,
        budget_seconds: float = FILL_WAIT_BUDGET_SECONDS,
        expected_cash_increase: float | None = None,
    ) -> bool:
        """Poll until a sell's proceeds or a buy's cash debit is reflected.

        **Must** call ``refresh_account()`` on every iteration — invalidate-on-submit
        (Tier 47A) only clears the cache once; without a per-poll refresh the loop
        would reread a frozen account object and always time out.

        **Sells (``side=\"sell\"``, default):** primary exit ``get_order_status() ==
        filled``; cash having risen by ``expected_cash_delta`` is a fallback only.

        **Buys (Tier 48A-2 — inverts 47B-2 constraint 2):** ``filled`` is a
        *precondition*, not the trigger. Poll until cash has **decreased** by
        approximately ``expected_cash_delta``. ``filled`` while the debit is still
        in flight must **not** satisfy the wait — 8/19 proved status leads cash.

        Non-dict status (typical MagicMock brokers) returns True immediately.
        """
        delta = expected_cash_delta if expected_cash_delta is not None else expected_cash_increase
        deadline = time.monotonic() + float(budget_seconds)
        abs_tol = max(1.0, abs(float(delta)) * 0.01) if delta is not None else 1.0
        while True:
            self._broker.refresh_account()
            try:
                st = self._broker.get_order_status(order_id)
            except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError):
                st = None
            if not isinstance(st, dict):
                return True
            raw = str(st.get("status") or "")
            status_filled = _bucket_order_status(raw) == "filled"
            cash_now: float | None = None
            try:
                cash_now = float(self._broker.get_cash())
            except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError, TypeError):
                cash_now = None

            if side == "buy":
                if (
                    status_filled
                    and delta is not None
                    and cash_before is not None
                    and cash_now is not None
                    and cash_now <= float(cash_before) - float(delta) + abs_tol
                ):
                    return True
            elif status_filled or (
                delta is not None
                and cash_before is not None
                and cash_now is not None
                and cash_now + 1e-9 >= float(cash_before) + float(delta)
            ):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(float(interval_seconds), remaining))

    def unsettled_pending_buy_notional(self) -> float:
        """Sum of same-cycle buy notionals whose cash debits have not landed yet."""
        self._broker.refresh_account()
        try:
            cash_now = float(self._broker.get_cash())
        except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError, TypeError):
            cash_now = None
        total = 0.0
        debited_so_far = 0.0
        for entry in self._pending_buy_debits:
            if entry.debit_observed:
                debited_so_far += float(entry.notional)
                continue
            if cash_now is not None:
                tol = max(1.0, entry.notional * 0.01)
                cumulative = debited_so_far + float(entry.notional)
                if cash_now <= float(entry.cash_before) - cumulative + tol:
                    entry.debit_observed = True
                    debited_so_far = cumulative
                    continue
            total += float(entry.notional)
        return total

    def close_position(
        self,
        symbol: str,
        qty: float,
        *,
        price: float,
        equity: float,
        daily_pnl_pct: float,
        as_of: datetime | None = None,
        wait_for_fill: bool = False,
    ) -> OrderExecutionResult:
        """Submit a market sell to flatten ``qty`` shares (momentum rotation).

        Skips position-limit and cash-reserve checks; only the kill switch can block.
        ``equity`` is reserved for future margin-aware logic.
        When ``wait_for_fill`` is True, poll until the sell fills (or the wait
        budget expires) before returning — used when a later same-cycle buy
        depends on the proceeds.
        """
        sym = symbol.strip().upper()
        when = as_of if as_of is not None else datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)

        logger.debug(
            "close_position request sym=%s qty=%s price=%s equity=%s",
            sym,
            qty,
            price,
            equity,
        )
        halted = self._kill_switch.is_halted(daily_pnl_pct)
        if halted:
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason="Kill switch halted trading.",
                side="sell",
                qty=None,
                fill_price=None,
            )
        if qty <= 0.0 or price <= 0.0:
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason="Invalid qty or price for close.",
                side="sell",
                qty=None,
                fill_price=None,
            )

        risk = self._settings.risk
        if risk.pdt_protection and self._sqlite is not None:
            exec_rows = self._sqlite.get_executions(limit=5000, account_id=self._account_id)
            pdt_chk = evaluate_pdt_before_sell(
                symbol=sym,
                equity=float(equity),
                as_of=when,
                executions=exec_rows,
                pdt_protection=True,
                pdt_threshold=int(risk.pdt_threshold),
                equity_floor=float(risk.pdt_equity_floor),
            )
            if not pdt_chk.ok:
                return OrderExecutionResult(
                    symbol=sym,
                    submitted=False,
                    reason=pdt_chk.reason,
                    side="sell",
                    qty=None,
                    fill_price=None,
                )
            if pdt_chk.warning:
                logger.warning("PDT advisory for %s: %s", sym, pdt_chk.warning)

        cash_before: float | None = None
        if wait_for_fill:
            try:
                cash_before = float(self._broker.get_cash())
            except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError, TypeError):
                cash_before = None
        try:
            oid = self._submit_market_order_with_retry(sym, float(qty), "sell")
        except DuplicateClientOrderIdError as exc:
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason=f"Duplicate suppressed: {exc}",
                side="sell",
                qty=None,
                fill_price=None,
            )
        except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
            logger.exception("Broker submit_market_order (sell) failed for %s", sym)
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason=f"Broker error: {exc}",
                side="sell",
                qty=None,
                fill_price=None,
            )
        order_status = "filled"
        if wait_for_fill:
            filled = self.wait_for_fill(
                oid,
                expected_cash_increase=float(qty) * float(price),
                cash_before=cash_before,
            )
            if not filled:
                logger.info(
                    "rotation fill timeout — skip dependent buy this cycle "
                    "(recoverable: next cycle will unsweep if capital is parked "
                    "in the sweep vehicle)"
                )
                order_status = "pending"
        bucket, filled_qty, fill_price = self._order_fill_fields(oid)
        if bucket == "cancelled":
            order_status = "cancelled"
        elif bucket == "pending" and order_status == "filled":
            order_status = "pending"
        if filled_qty is None and order_status == "filled":
            filled_qty = float(qty)
        return OrderExecutionResult(
            symbol=sym,
            submitted=True,
            order_id=oid,
            side="sell",
            qty=float(qty),
            fill_price=fill_price,
            order_status=order_status,
            filled_qty=filled_qty,
        )

    def sweep_buy(
        self,
        symbol: str,
        notional: float,
        *,
        price: float,
        equity: float,
        cash: float,
        daily_pnl_pct: float,
    ) -> OrderExecutionResult:
        """Buy ``notional`` dollars of the cash-sweep vehicle (e.g. BIL).

        Deliberately **skips the max-position check** — a T-bill ETF may legitimately
        be 40-50%+ of equity, which is not the concentration risk the cap protects
        against. Kill switch, the Tier-43 ``min_order_notional_usd`` floor, and the
        minimum cash reserve still apply. Logs ``strategy_name="CashSweep"``.
        """
        sym = symbol.strip().upper()
        if self._kill_switch.is_halted(daily_pnl_pct):
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason="Kill switch active — new orders blocked.",
                side="buy",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        notional = float(notional)
        if price <= 0.0 or notional <= 0.0:
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason="Invalid price or notional for sweep buy.",
                side="buy",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        risk = self._settings.risk
        if notional < risk.min_order_notional_usd:
            logger.debug(
                "order_skipped_below_floor symbol=%s notional=%.2f floor=%.2f strategy=CashSweep",
                sym,
                notional,
                float(risk.min_order_notional_usd),
            )
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason=f"Notional below minimum order size (${risk.min_order_notional_usd:.2f}).",
                side="buy",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        min_cash = float(risk.min_cash_reserve_pct) * float(equity)
        pending = self.unsettled_pending_buy_notional()
        effective_cash = float(cash) - pending
        if effective_cash - notional + 1e-9 < min_cash:
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason="Sweep buy would violate minimum cash reserve after settlement.",
                side="buy",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        qty = notional / float(price)
        cash_before_buy: float | None = None
        try:
            cash_before_buy = float(self._broker.get_cash())
        except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError, TypeError):
            cash_before_buy = None
        try:
            oid = self._submit_market_order_with_retry(sym, qty, "buy")
        except DuplicateClientOrderIdError as exc:
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason=f"Duplicate suppressed: {exc}",
                side="buy",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
            logger.exception("Sweep buy submit failed for %s", sym)
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason=f"Broker error: {exc}",
                side="buy",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        pending_entry = _PendingBuyDebit(
            oid,
            float(notional),
            float(cash_before_buy or 0.0),
        )
        self._pending_buy_debits.append(pending_entry)
        debited = self.wait_for_fill(
            oid,
            side="buy",
            expected_cash_delta=float(notional),
            cash_before=cash_before_buy,
        )
        if debited:
            pending_entry.debit_observed = True
        else:
            logger.info(
                "sweep buy fill wait timed out before cash debit observed for %s; proceeding",
                sym,
            )
        bucket, filled_qty, fill_price = self._order_fill_fields(oid)
        if filled_qty is None:
            filled_qty = float(qty)
        return OrderExecutionResult(
            symbol=sym,
            submitted=True,
            order_id=oid,
            side="buy",
            qty=float(qty),
            fill_price=fill_price,
            strategy_name="CashSweep",
            filled_qty=filled_qty,
            order_status="cancelled" if bucket == "cancelled" else "filled",
        )

    def sweep_sell(
        self,
        symbol: str,
        qty: float,
        *,
        price: float,
        equity: float,
        daily_pnl_pct: float,
        wait_for_fill: bool = False,
    ) -> OrderExecutionResult:
        """Market-sell ``qty`` shares of the cash-sweep vehicle to restore the reserve.

        Kill-switch gate only (mirrors :meth:`close_position`). ``equity`` is accepted
        for signature symmetry / future margin-aware logic. Logs ``strategy_name="CashSweep"``.
        When ``wait_for_fill`` is True (unsweep-before-entry), poll until filled so
        the subsequent buy sees the proceeds.
        """
        sym = symbol.strip().upper()
        if self._kill_switch.is_halted(daily_pnl_pct):
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason="Kill switch active — new orders blocked.",
                side="sell",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        if qty <= 0.0 or price <= 0.0:
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason="Invalid qty or price for sweep sell.",
                side="sell",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        cash_before: float | None = None
        if wait_for_fill:
            try:
                cash_before = float(self._broker.get_cash())
            except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError, TypeError):
                cash_before = None
        try:
            oid = self._submit_market_order_with_retry(sym, float(qty), "sell")
        except DuplicateClientOrderIdError as exc:
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason=f"Duplicate suppressed: {exc}",
                side="sell",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
            logger.exception("Sweep sell submit failed for %s", sym)
            return OrderExecutionResult(
                symbol=sym,
                submitted=False,
                reason=f"Broker error: {exc}",
                side="sell",
                qty=None,
                fill_price=None,
                strategy_name="CashSweep",
            )
        order_status = "filled"
        if wait_for_fill:
            filled = self.wait_for_fill(
                oid,
                expected_cash_increase=float(qty) * float(price),
                cash_before=cash_before,
            )
            if not filled:
                logger.info(
                    "unsweep fill timeout — skip dependent buy this cycle "
                    "(recoverable: next cycle will retry unsweep)"
                )
                order_status = "pending"
        bucket, filled_qty, fill_price = self._order_fill_fields(oid)
        if bucket == "cancelled":
            order_status = "cancelled"
        elif bucket == "pending" and order_status == "filled":
            order_status = "pending"
        if filled_qty is None and order_status == "filled":
            filled_qty = float(qty)
        return OrderExecutionResult(
            symbol=sym,
            submitted=True,
            order_id=oid,
            side="sell",
            qty=float(qty),
            fill_price=fill_price,
            strategy_name="CashSweep",
            order_status=order_status,
            filled_qty=filled_qty,
        )
