"""End-to-end pipeline: ingest OHLCV, generate signals, execute via ``OrderManager``."""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd  # noqa: TC002

from src.automation.alerts import Alert, AlertLevel, evaluate_cycle_alerts
from src.automation.notifier import Notifier  # noqa: TC001
from src.execution.account_factory import resolve_sqlite_account_id
from src.models import OrderExecutionResult, Signal
from src.portfolio.state import snapshot_from_broker
from src.reporting.equity_curve import record_equity_snapshot
from src.risk.kill_switch import KillSwitch  # noqa: TC001
from src.strategy.indicators import atr
from src.strategy.regime_params import effective_dca_amount

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.ai.llm_client import OllamaClient
    from src.config import Settings
    from src.data.pipeline import DataPipeline, IngestResult
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore
    from src.execution.broker_interface import BrokerInterface
    from src.execution.order_manager import OrderManager
    from src.portfolio.tax_lots import LotLedger
    from src.strategy.base import Strategy

logger = logging.getLogger(__name__)


def _latest_atr_from_ohlcv(df: pd.DataFrame, period: int) -> float | None:
    """Last ATR value from stored OHLCV, or None if insufficient data."""
    if df is None or df.empty:
        return None
    for col in ("high", "low", "close"):
        if col not in df.columns:
            return None
    if len(df) < period + 1:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    s = atr(high, low, close, period=period)
    val = float(s.iloc[-1])
    if math.isnan(val) or val <= 0.0:
        return None
    return val


@dataclass(frozen=True)
class TradingCycleResult:
    """One scheduled pass: ingest outcomes, strategy outputs, and execution results."""

    cycle_id: str
    ingest_results: list[tuple[str, IngestResult]]
    signals: list[Signal]
    execution_results: list[OrderExecutionResult]


class TradingWorkflow:
    """Wires ``DataPipeline`` → strategies → ``OrderManager`` with broker snapshot."""

    def __init__(
        self,
        *,
        settings: Settings,
        data_pipeline: DataPipeline,
        parquet_store: ParquetStore,
        strategies: Sequence[Strategy],
        order_manager: OrderManager,
        broker: BrokerInterface,
        sqlite_store: SQLiteStore | None = None,
        lot_ledger: LotLedger | None = None,
        kill_switch: KillSwitch | None = None,
        notifier: Notifier | None = None,
        fred_pipeline: DataPipeline | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self._settings = settings
        self._pipeline = data_pipeline
        self._fred_pipeline = fred_pipeline
        self._parquet = parquet_store
        self._strategies = list(strategies)
        self._order_manager = order_manager
        self._broker = broker
        self._account_id = resolve_sqlite_account_id(broker)
        self._sqlite = sqlite_store
        self._lot_ledger = lot_ledger
        self._kill_switch = kill_switch
        self._notifier = notifier
        self._llm_client = llm_client

    @property
    def strategies(self) -> tuple[Strategy, ...]:
        """Configured strategies in execution order (read-only)."""
        return tuple(self._strategies)

    @property
    def notifier(self) -> Notifier | None:
        """Optional notifier for alerts and digests."""
        return self._notifier

    @property
    def lot_ledger(self) -> LotLedger | None:
        """Optional tax lot ledger when wired."""
        return self._lot_ledger

    def _universe_symbols(self) -> list[str]:
        """All OHLCV symbols config requires, plus VIX when regime uses it (sorted)."""
        from src.data.symbol_resolver import resolve_required_symbols

        syms = set(resolve_required_symbols(self._settings))
        vix = self._settings.regime.vix_symbol.strip().upper()
        if self._settings.regime.enabled and vix and vix not in syms:
            syms.add(vix)
        return sorted(syms)

    def run_ingest_only(self) -> list[tuple[str, IngestResult]]:
        """Fetch and store OHLCV for every symbol strategies need (no signals or orders)."""
        if self._settings.regime.enabled and self._fred_pipeline is not None:
            for sid in self._settings.regime.fred_series:
                self._fred_pipeline.ingest_macro(sid)
        symbols = self._universe_symbols()
        return [(sym, self._pipeline.ingest_ohlcv(sym)) for sym in symbols]

    def run_cycle(self, *, as_of: datetime | None = None) -> TradingCycleResult:
        """Ingest each symbol in the union of strategy universes, run strategies, execute."""
        when = as_of or datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)

        self._broker.refresh_account()

        if self._sqlite is not None:
            cancelled = self._order_manager.cancel_stale_orders(
                timeout_minutes=int(self._settings.execution.unfilled_timeout_minutes),
            )
            if cancelled:
                logger.info("Cancelled %d stale pending limit order(s).", len(cancelled))

        cycle_id = uuid.uuid4().hex[:12]
        self._order_manager.set_cycle_id(cycle_id)
        cycle_date = when.astimezone(UTC).date()
        cycle_alerts: list[Alert] = []  # stale ML, reconciliation, then rule-based cycle alerts

        macro_ingest_failures: list[str] = []
        if self._settings.regime.enabled and self._fred_pipeline is not None:
            for sid in self._settings.regime.fred_series:
                macro_res = self._fred_pipeline.ingest_macro(sid)
                if not macro_res.success:
                    logger.warning("FRED ingest failed for %s: %s", sid, macro_res.error)
                    macro_ingest_failures.append(sid)

        symbols = self._universe_symbols()
        ingest_results: list[tuple[str, IngestResult]] = []
        for sym in symbols:
            ingest_results.append((sym, self._pipeline.ingest_ohlcv(sym)))

        combined: dict[str, pd.DataFrame] = {sym: self._parquet.read_ohlcv(sym) for sym in symbols}

        regime_mult = 1.0
        cycle_regime = None
        if self._settings.regime.enabled and self._sqlite is not None:
            from src.data.regime import RegimeDetector

            cycle_regime = RegimeDetector(self._settings).detect(
                parquet_store=self._parquet,
                sqlite_store=self._sqlite,
            )
            self._sqlite.write_regime_snapshot(cycle_regime)
            regime_mult = float(cycle_regime.sizing_multiplier)

        if self._settings.fundamental.enabled and self._sqlite is not None:
            self._maybe_ingest_fundamentals(when)

        signals: list[Signal] = []
        for strat in self._strategies:
            uni = strat.get_universe()
            sub = {s: combined[s] for s in uni if s in combined}
            signals.extend(
                strat.generate_signals(sub, as_of=when, market_regime=cycle_regime),
            )

        if self._sqlite is not None:
            from src.ai.explainer import generate_signal_explanation

            for sig in signals:
                expl: str | None = None
                if self._llm_client is not None:
                    from src.ai.llm_explainer import generate_llm_explanation

                    expl = generate_llm_explanation(
                        self._llm_client,
                        sig,
                        combined,
                        self._settings,
                        market_regime=cycle_regime,
                    )
                if expl is None:
                    expl = generate_signal_explanation(sig, combined, self._settings)
                self._sqlite.log_signal(
                    cycle_id,
                    sig,
                    explanation=expl,
                    account_id=self._account_id,
                )

            self._maybe_log_ml_scores(cycle_id, combined, when)

        raw_signals = list(signals)
        if self._settings.strategy.ensemble.enabled:
            from src.strategy.ensemble import merge_signals

            signals = merge_signals(
                signals,
                ensemble_config=self._settings.strategy.ensemble,
                timestamp=when,
            )

        self._maybe_append_ml_model_stale_alert(cycle_alerts)
        signals = self._apply_ml_order_gate(signals)

        all_close: dict[str, float] = {}
        for sym, df in combined.items():
            if df is not None and not df.empty and "close" in df.columns:
                all_close[sym] = float(df["close"].astype(float).iloc[-1])

        last_prices: dict[str, float] = {}
        for sig in signals:
            sym = sig.symbol.strip().upper()
            if sym in all_close:
                last_prices[sym] = all_close[sym]

        snapshot = snapshot_from_broker(self._broker, symbols)

        rotation_results: list[OrderExecutionResult] = []
        momentum_universe = self._momentum_universe()
        mom = [s for s in raw_signals if s.strategy_name == "MomentumRotationStrategy"]
        if len(mom) > 1:
            logger.warning(
                "Multiple momentum signals (%d); using first only: %s",
                len(mom),
                mom[0].symbol,
            )
        target_sym = mom[0].symbol.strip().upper() if mom else None

        if target_sym and momentum_universe:
            for sym, qty in snapshot.positions.items():
                if sym not in momentum_universe or sym == target_sym or qty <= 0.0:
                    continue
                px = all_close.get(sym)
                if px is None or px <= 0.0:
                    continue
                logger.info(
                    "Rotating momentum: selling %s (qty=%s), target allocation %s",
                    sym,
                    qty,
                    target_sym,
                )
                res = self._order_manager.close_position(
                    sym,
                    float(qty),
                    price=px,
                    equity=snapshot.equity,
                    daily_pnl_pct=snapshot.daily_pnl_pct,
                    as_of=when,
                    wait_for_fill=True,
                )
                rotation_results.append(res)
                if self._sqlite is not None:
                    eid = self._sqlite.log_execution(
                        cycle_id,
                        res,
                        account_id=self._account_id,
                    )
                    self._persist_execution_quality(cycle_id, eid, res)

            snapshot = snapshot_from_broker(self._broker, symbols)

        harvest_results: list[OrderExecutionResult] = []
        replacement_signals: list[Signal] = []
        if self._settings.tax.harvest_enabled and self._lot_ledger is not None:
            from src.strategy.tax_loss_harvest import TaxLossHarvester

            harvester = TaxLossHarvester(
                lot_ledger=self._lot_ledger,
                settings=self._settings,
                account_id=self._account_id,
            )
            protected = self._protected_symbols(signals)
            candidates = harvester.find_candidates(all_close, protected, as_of=when)
            for cand in candidates:
                px = float(cand.current_price)
                res = self._order_manager.close_position(
                    cand.symbol,
                    float(cand.qty),
                    price=px,
                    equity=snapshot.equity,
                    daily_pnl_pct=snapshot.daily_pnl_pct,
                    as_of=when,
                )
                if res.submitted:
                    res = res.model_copy(update={"strategy_name": "TaxLossHarvest"})
                harvest_results.append(res)
                if self._sqlite is not None:
                    eid = self._sqlite.log_execution(
                        cycle_id,
                        res,
                        account_id=self._account_id,
                    )
                    self._persist_execution_quality(cycle_id, eid, res)
                if res.submitted and self._lot_ledger is not None:
                    fill = res.fill_price if res.fill_price and res.fill_price > 0 else px
                    try:
                        self._lot_ledger.record_sell_lot(
                            cand.lot_id,
                            float(fill),
                            res.timestamp,
                            is_tlh=True,
                            account_id=self._account_id,
                        )
                    except ValueError as exc:
                        logger.warning("TLH lot close mismatch for %s: %s", cand.symbol, exc)
                if res.submitted and cand.replacement_symbol:
                    repl = cand.replacement_symbol.strip().upper()
                    rpx = float(all_close.get(repl, 0.0))
                    if rpx <= 0.0:
                        logger.warning(
                            "TLH replacement %s not in OHLCV / no price; skipping replacement buy.",
                            repl,
                        )
                    else:
                        notional = float(cand.qty) * float(px)
                        eq = max(float(snapshot.equity), 1e-9)
                        w = min(notional / eq, float(self._settings.risk.max_position_pct))
                        replacement_signals.append(
                            Signal(
                                symbol=repl,
                                direction="long",
                                weight=w,
                                confidence=1.0,
                                rationale=f"TLH replacement for {cand.symbol}: {cand.reason}"[
                                    :2000
                                ],
                                timestamp=when,
                                strategy_name="TaxLossHarvest",
                            ),
                        )
            if harvest_results:
                snapshot = snapshot_from_broker(self._broker, symbols)

        signals.extend(replacement_signals)
        if self._sqlite is not None and replacement_signals:
            from src.ai.explainer import generate_signal_explanation

            for sig in replacement_signals:
                expl = generate_signal_explanation(sig, combined, self._settings)
                self._sqlite.log_signal(
                    cycle_id,
                    sig,
                    explanation=expl,
                    account_id=self._account_id,
                )

        signals = self._apply_ml_order_gate(signals)

        dca_budget: float | None = None
        if any(s.strategy_name == "DCAStrategy" for s in signals):
            # Regime scaling of DCA is applied only in DCAStrategy signal weights (scale vs
            # effective_dca_amount(..., None)). Budget here must be the unscaled base so we do
            # not multiply regime twice (workflow budget x per-signal weight already includes it).
            dca_budget = effective_dca_amount(
                self._settings,
                None,
                equity=float(snapshot.equity),
            )

        atr_values: dict[str, float] | None = None
        if self._settings.risk.sizing_method == "atr_risk_parity":
            period = int(self._settings.risk.atr_period)
            atr_map: dict[str, float] = {}
            for s in signals:
                if s.direction not in ("long", "cash"):
                    continue
                if s.strategy_name == "DCAStrategy":
                    continue
                sym_u = s.symbol.strip().upper()
                df = combined.get(sym_u)
                av = _latest_atr_from_ohlcv(df, period) if df is not None else None
                if av is not None:
                    atr_map[sym_u] = av
            atr_values = atr_map if atr_map else None

        rotation_fill_ok = all(
            (not r.submitted) or r.order_status == "filled" for r in rotation_results
        )
        unsweep_results: list[OrderExecutionResult] = []
        unsweep_attempted = False
        if not rotation_fill_ok:
            buy_results = []
        else:
            unsweep_results, snapshot, unsweep_attempted = self._unsweep_for_unfunded_buys(
                signals,
                snapshot=snapshot,
                last_prices=last_prices,
                all_close=all_close,
                symbols=symbols,
                dca_budget=dca_budget,
                atr_values=atr_values,
                regime_multiplier=regime_mult,
            )
            buy_results = self._order_manager.execute_signals(
                signals,
                last_prices=last_prices,
                equity=snapshot.equity,
                cash=snapshot.cash,
                positions=snapshot.positions,
                daily_pnl_pct=snapshot.daily_pnl_pct,
                dca_budget=dca_budget,
                atr_values=atr_values,
                regime_multiplier=regime_mult,
                unsweep_attempted=unsweep_attempted,
            )
        if unsweep_attempted:
            strategy_buy_ok = any(
                r.submitted and r.side == "buy" and r.strategy_name != "CashSweep"
                for r in buy_results
            )
            if not strategy_buy_ok:
                cycle_alerts.append(
                    Alert(
                        timestamp=datetime.now(UTC),
                        level=AlertLevel.WARNING,
                        category="unsweep_failed",
                        message=(
                            "Unsweep attempted but strategy buy still not submitted; "
                            "account may remain stuck until the next cycle or a manual BIL sale."
                        ),
                    ),
                )
        if self._sqlite is not None:
            for res in [*unsweep_results, *buy_results]:
                eid = self._sqlite.log_execution(
                    cycle_id,
                    res,
                    account_id=self._account_id,
                )
                self._persist_execution_quality(cycle_id, eid, res)

        sweep_results = self._run_cash_sweep(cycle_id, symbols, all_close, when)

        execution_results = [
            *rotation_results,
            *harvest_results,
            *unsweep_results,
            *buy_results,
            *sweep_results,
        ]
        self._append_leverage_alert(cycle_alerts, execution_results)
        if self._sqlite is not None:
            from src.automation.reconciliation_alerts import generate_reconciliation_alerts
            from src.execution.reconciliation import (
                broker_orders_in_cycle_window,
                reconcile_executions,
            )

            rows = [
                r
                for r in self._sqlite.get_executions(limit=5000, account_id=self._account_id)
                if r.get("cycle_id") == cycle_id
            ]
            broker_all = self._broker.list_recent_orders(limit=500)
            # Time-window scope (not order-id-only): otherwise missing_from_log is
            # unreachable for fills that never wrote a trade_executions row.
            broker_orders = broker_orders_in_cycle_window(
                broker_all,
                rows,
                as_of=when,
            )
            rec_result = reconcile_executions(rows, broker_orders)
            sch = self._settings.scheduler
            cycle_alerts.extend(
                generate_reconciliation_alerts(
                    rec_result,
                    warning_threshold=int(sch.reconciliation_warning_threshold),
                    critical_threshold=int(sch.reconciliation_critical_threshold),
                ),
            )

        if self._lot_ledger is not None:
            for res in execution_results:
                if not res.submitted or not res.qty:
                    continue
                sym_u = res.symbol.strip().upper()
                lot_qty = float(res.filled_qty if res.filled_qty is not None else res.qty)
                if (
                    res.filled_qty is not None
                    and res.qty is not None
                    and lot_qty + 1e-9 < float(res.qty)
                ):
                    logger.warning(
                        "partial fill %s %s requested=%.6f filled=%.6f — lot ledger uses filled qty",
                        res.side,
                        sym_u,
                        float(res.qty),
                        lot_qty,
                    )
                price = res.fill_price
                if price is None or price <= 0.0:
                    price = float(all_close.get(sym_u, 0.0)) or float(last_prices.get(sym_u, 0.0))
                if price <= 0.0:
                    continue
                if res.side == "buy":
                    self._lot_ledger.record_buy(
                        sym_u,
                        lot_qty,
                        price,
                        res.timestamp,
                        account_id=self._account_id,
                    )
                elif res.side == "sell":
                    if res.strategy_name == "TaxLossHarvest":
                        continue
                    try:
                        self._lot_ledger.record_sell(
                            sym_u,
                            lot_qty,
                            price,
                            res.timestamp,
                            method=self._settings.tax.lot_selection_method,
                            account_id=self._account_id,
                        )
                    except ValueError as exc:
                        logger.warning("Lot sell mismatch for %s: %s", sym_u, exc)

        if self._sqlite is not None and self._lot_ledger is not None:
            snap_cash: float | None = None
            snap_broker_equity: float | None = None
            try:
                self._broker.refresh_account()
                snap_for_cash = snapshot_from_broker(self._broker, symbols)
                snap_cash = float(snap_for_cash.cash)
                snap_broker_equity = float(snap_for_cash.equity)
            except (ConnectionError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
                logger.warning(
                    "Broker snapshot for equity cash failed; writing snapshot with cash=NULL: %s",
                    exc,
                )
                snap_cash = None
                snap_broker_equity = None
            record_equity_snapshot(
                self._sqlite,
                self._lot_ledger,
                self._parquet,
                as_of_date=cycle_date,
                account_id=self._account_id,
                cash=snap_cash,
                broker_equity=snap_broker_equity,
            )

        if self._sqlite is not None:
            self._broker.refresh_account()
            snap_alerts = snapshot_from_broker(self._broker, symbols)
            if self._lot_ledger is not None:
                from src.automation.reconciliation_alerts import generate_lot_ledger_alerts
                from src.execution.lot_reconciliation import reconcile_lot_ledger
                from src.reporting.portfolio_analytics import (
                    compute_portfolio_valuation,
                    load_current_prices,
                )

                open_lots = self._lot_ledger.get_open_lots(account_id=self._account_id)
                ledger_qty: dict[str, float] = {}
                for lot in open_lots:
                    ledger_qty[lot.symbol] = ledger_qty.get(lot.symbol, 0.0) + float(lot.qty)
                lot_symbols = sorted(set(symbols) | set(ledger_qty) | set(snap_alerts.positions))
                snap_lots = snapshot_from_broker(self._broker, lot_symbols)
                prices = load_current_prices(self._parquet, lot_symbols)
                valuation = compute_portfolio_valuation(
                    self._lot_ledger,
                    prices,
                    account_id=self._account_id,
                )
                lot_rec = reconcile_lot_ledger(
                    ledger_qty_by_symbol=ledger_qty,
                    broker_qty_by_symbol=dict(snap_lots.positions),
                    ledger_market_value=float(valuation.total_market_value),
                    cash=float(snap_lots.cash),
                    broker_equity=float(snap_lots.equity),
                )
                cycle_alerts.extend(generate_lot_ledger_alerts(lot_rec))
                snap_alerts = snap_lots
            ingest_failures = [
                sym for sym, res in ingest_results if not res.success
            ] + macro_ingest_failures
            rejected_orders = [
                (res.symbol.strip().upper(), res.reason)
                for res in execution_results
                if not res.submitted
            ]
            kill_halted = (
                self._kill_switch.is_halted(snap_alerts.daily_pnl_pct)
                if self._kill_switch is not None
                else False
            )
            rule_alerts = evaluate_cycle_alerts(
                kill_switch_halted=kill_halted,
                daily_pnl_pct=float(snap_alerts.daily_pnl_pct),
                daily_loss_limit_pct=float(self._settings.risk.daily_loss_limit_pct),
                ingest_failures=ingest_failures,
                rejected_orders=rejected_orders,
                market_regime=cycle_regime,
            )
            combined_alerts = [*cycle_alerts, *rule_alerts]
            for alt in combined_alerts:
                self._sqlite.log_alert(alt)
                if self._notifier is not None:
                    self._notifier.send(alt)
            cycle_alerts = combined_alerts

        if self._notifier is not None:
            from src.ai.explainer import generate_cycle_summary

            snap_final = snapshot_from_broker(self._broker, symbols)
            summary_body: str | None = None
            if self._llm_client is not None:
                from src.ai.llm_explainer import generate_llm_cycle_summary

                summary_body = generate_llm_cycle_summary(
                    self._llm_client,
                    signals,
                    execution_results,
                    snapshot=snap_final,
                    alerts=cycle_alerts or None,
                    market_regime=cycle_regime,
                )
            if summary_body is None:
                summary_body = generate_cycle_summary(
                    signals,
                    execution_results,
                    snapshot=snap_final,
                    alerts=cycle_alerts or None,
                    market_regime=cycle_regime,
                )
            cycle_alert = Alert(
                timestamp=datetime.now(UTC),
                level=AlertLevel.INFO,
                category="cycle_summary",
                message=summary_body,
            )
            self._sqlite.log_alert(cycle_alert)
            self._notifier.send(cycle_alert)

        return TradingCycleResult(
            cycle_id=cycle_id,
            ingest_results=ingest_results,
            signals=signals,
            execution_results=execution_results,
        )

    def _append_leverage_alert(
        self,
        cycle_alerts: list[Alert],
        execution_results: list[OrderExecutionResult],
    ) -> None:
        """CRITICAL when cash is negative or the broker reports margin borrowing (Tier 48B)."""
        from src.execution.leverage import (
            LeverageSnapshot,
            format_cycle_order_sequence,
            leverage_violation_message,
        )

        try:
            self._broker.refresh_account()
            lev = self._broker.get_leverage_snapshot()
        except (
            ConnectionError,
            OSError,
            ValueError,
            RuntimeError,
            TimeoutError,
            NotImplementedError,
            AttributeError,
        ):
            return
        if not isinstance(lev, LeverageSnapshot):
            return
        seq = format_cycle_order_sequence(execution_results)
        msg = leverage_violation_message(lev, seq)
        if msg is None:
            return
        cycle_alerts.append(
            Alert(
                timestamp=datetime.now(UTC),
                level=AlertLevel.CRITICAL,
                category="leverage",
                message=msg,
            ),
        )

    def _maybe_append_ml_model_stale_alert(self, cycle_alerts: list[Alert]) -> None:
        """Warn when the on-disk ML bundle is older than ``ml.max_model_age_days``."""
        if not self._settings.ml.affect_orders:
            return
        from src.config import ml_model_path

        path = ml_model_path(self._settings)
        if not path.is_file():
            return
        age_seconds = datetime.now(UTC).timestamp() - path.stat().st_mtime
        age_days = int(age_seconds / 86400.0)
        max_age = int(self._settings.ml.max_model_age_days)
        if age_days <= max_age:
            return
        logger.warning(
            "ML model is %d days old (max %d); scores may be unreliable",
            age_days,
            max_age,
        )
        cycle_alerts.append(
            Alert(
                timestamp=datetime.now(UTC),
                level=AlertLevel.WARNING,
                category="ml_model_stale",
                message=(
                    f"ML model is {age_days} days old (max {max_age}); scores may be unreliable"
                ),
            ),
        )

    def _apply_ml_order_gate(self, signals: list[Signal]) -> list[Signal]:
        """Drop signals below ML score threshold when ``ml.affect_orders`` is enabled."""
        if not self._settings.ml.affect_orders:
            return signals
        if self._sqlite is None:
            logger.warning(
                "ml.affect_orders is True but SQLite store is missing; skipping ML gate",
            )
            return signals

        thr = float(self._settings.ml.score_threshold)
        if thr > 0.9:
            logger.warning(
                "ML gate: score_threshold=%.4f is very high; most signals may be blocked",
                thr,
            )

        rows = self._sqlite.get_latest_ml_scores_batch()
        by_sym: dict[str, float] = {}
        for r in rows:
            try:
                sc = float(r["score"])
            except (TypeError, ValueError):
                continue
            by_sym[str(r["symbol"]).strip().upper()] = sc

        from src.risk.ml_order_gate import filter_signals_by_ml_scores

        filtered, _passed, _total = filter_signals_by_ml_scores(
            signals,
            by_sym,
            threshold=thr,
            missing_score_action=self._settings.ml.missing_score_action,
        )
        return filtered

    def _maybe_log_ml_scores(
        self,
        cycle_id: str,
        combined: dict[str, pd.DataFrame],
        when: datetime,
    ) -> None:
        """Persist diagnostic ML scores for momentum universe (no effect on orders)."""
        if not self._settings.ml.enabled or self._sqlite is None:
            return
        from src.config import ml_model_path
        from src.ml.explain import shap_summary_text
        from src.ml.features import build_feature_vector
        from src.ml.infer import load_ml_bundle, score_positive_proba

        path = ml_model_path(self._settings)
        bundle = load_ml_bundle(path)
        if bundle is None:
            logger.debug("ML scoring skipped: no bundle at %s", path)
            return

        vix_sym = ""
        if self._settings.regime.enabled:
            vix_sym = self._settings.regime.vix_symbol.strip().upper()
        model_ver = str(bundle.get("model_version") or self._settings.ml.model_version)
        x_background = bundle.get("X_bg")
        model = bundle.get("model")
        names: list[str] = list(bundle.get("feature_names") or [])

        computed = when.astimezone(UTC).isoformat()
        for sym in self._momentum_universe():
            su = sym.strip().upper()
            if vix_sym and su == vix_sym:
                continue
            df = combined.get(sym)
            if df is None or df.empty:
                continue
            feats = build_feature_vector(df, as_of=when)
            if feats is None:
                continue
            proba = score_positive_proba(bundle, feats)
            if proba is None:
                continue
            shap_txt = ""
            if model is not None and names and x_background is not None:
                try:
                    import numpy as np

                    row = np.array([[float(feats[n]) for n in names]], dtype=np.float64)
                    shap_txt = shap_summary_text(model, x_background, row, names)
                except (KeyError, TypeError, ValueError):
                    shap_txt = ""
            self._sqlite.write_ml_score(
                cycle_id=cycle_id,
                symbol=su,
                score=float(proba),
                model_version=model_ver,
                computed_at=computed,
                top_features_json=shap_txt or None,
            )

    def _persist_execution_quality(
        self,
        cycle_id: str,
        execution_id: int | None,
        res: OrderExecutionResult,
    ) -> None:
        """Store slippage vs daily close reference when a fill exists."""
        if self._sqlite is None or not isinstance(execution_id, int) or not res.submitted:
            return
        from src.reporting.execution_quality import (
            build_execution_quality_row,
            execution_quality_row_to_dict,
        )

        side = (res.side or "buy").strip().lower()
        row = build_execution_quality_row(
            execution_id=execution_id,
            cycle_id=cycle_id,
            result_side=side,
            result_qty=res.qty,
            fill_price=res.fill_price,
            execution_ts=res.timestamp,
            symbol=res.symbol,
            parquet_store=self._parquet,
        )
        if row is None:
            return
        self._sqlite.write_execution_quality(execution_quality_row_to_dict(row))

    def _maybe_ingest_fundamentals(self, when: datetime) -> None:
        """Refresh F-scores for a capped set of symbols past ``refresh_interval_days``."""
        if self._sqlite is None:
            return
        cfg = self._settings.fundamental
        skip = {s.strip().upper() for s in cfg.skip_symbols}
        interval = float(cfg.refresh_interval_days)
        max_n = int(cfg.max_symbols_per_cycle)

        symbols: set[str] = set()
        vix = (
            self._settings.regime.vix_symbol.strip().upper()
            if self._settings.regime.enabled
            else ""
        )
        for s in self._universe_symbols():
            su = s.strip().upper()
            if vix and su == vix:
                continue
            symbols.add(su)
        if self._lot_ledger is not None:
            for lot in self._lot_ledger.get_open_lots(account_id=self._account_id):
                symbols.add(lot.symbol.strip().upper())

        candidates = sorted(s for s in symbols if s not in skip)
        refreshed = 0
        skip_list = list(cfg.skip_symbols)
        for sym in candidates:
            if refreshed >= max_n:
                break
            row = self._sqlite.get_latest_fundamental_score(sym)
            need = True
            if row is not None:
                raw = row.get("fetched_at")
                if raw:
                    try:
                        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=UTC)
                        age_days = (when - ts).total_seconds() / 86400.0
                        need = age_days >= interval
                    except ValueError:
                        need = True
                else:
                    need = True
            if not need:
                continue
            result = self._pipeline.ingest_fundamentals(sym, skip_symbols=skip_list)
            if result.success:
                refreshed += 1

    def _protected_symbols(self, sigs: list[Signal]) -> set[str]:
        """Symbols actively held by momentum or mean reversion (do not harvest)."""
        protected: set[str] = set()
        for s in sigs:
            if s.direction == "long" and s.strategy_name in (
                "MomentumRotationStrategy",
                "MeanReversionStrategy",
                "Ensemble",
            ):
                protected.add(s.symbol.strip().upper())
        return protected

    def _momentum_universe(self) -> set[str]:
        """Symbols tracked by ``MomentumRotationStrategy`` (risk + cash ETF), if configured."""
        from src.strategy.momentum import MomentumRotationStrategy

        for strat in self._strategies:
            if isinstance(strat, MomentumRotationStrategy):
                return {s.strip().upper() for s in strat.get_universe()}
        return set()

    def _unsweep_for_unfunded_buys(
        self,
        signals: list[Signal],
        *,
        snapshot: object,
        last_prices: dict[str, float],
        all_close: dict[str, float],
        symbols: list[str],
        dca_budget: float | None,
        atr_values: dict[str, float] | None,
        regime_multiplier: float,
    ) -> tuple[list[OrderExecutionResult], object, bool]:
        """Sell sweep holdings to fund strategy buys that cash alone cannot cover.

        Sizing comes from :meth:`OrderManager.intended_buy_notionals` so unsweep
        and execution cannot drift. Returns ``(results, snapshot, attempted)``.
        """
        from src.portfolio.state import PortfolioSnapshot

        if not isinstance(snapshot, PortfolioSnapshot):
            return [], snapshot, False
        cfg = self._settings.cash_sweep
        if not cfg.enabled:
            return [], snapshot, False
        if self._kill_switch is not None and self._kill_switch.is_halted(
            float(snapshot.daily_pnl_pct),
        ):
            return [], snapshot, False
        intended = self._order_manager.intended_buy_notionals(
            signals,
            last_prices=last_prices,
            equity=float(snapshot.equity),
            positions=dict(snapshot.positions),
            dca_budget=dca_budget,
            atr_values=atr_values,
            regime_multiplier=regime_multiplier,
        )
        if not isinstance(intended, dict) or not intended:
            return [], snapshot, False
        required = sum(float(v) for v in intended.values())
        reserve = float(self._settings.risk.min_cash_reserve_pct) * float(snapshot.equity)
        available = float(snapshot.cash) - reserve
        if required <= available + 1e-9:
            return [], snapshot, False
        sweep_sym = cfg.symbol.strip().upper()
        px = all_close.get(sweep_sym)
        if px is None or float(px) <= 0.0:
            px = last_prices.get(sweep_sym)
        if px is None or float(px) <= 0.0:
            return [], snapshot, False
        holdings_qty = float(snapshot.positions.get(sweep_sym, 0.0))
        if holdings_qty <= 0.0:
            return [], snapshot, False
        shortfall = required - available
        margin = float(cfg.buffer_pct) * float(snapshot.equity)
        sell_notional = min(shortfall + margin, holdings_qty * float(px))
        if sell_notional < float(self._settings.risk.min_order_notional_usd):
            return [], snapshot, False
        sell_qty = sell_notional / float(px)
        fund_sym = "+".join(sorted(intended))
        logger.info(
            "cash_sweep unsweep symbol=%s notional=%.2f reason=funding %s entry",
            sweep_sym,
            float(sell_notional),
            fund_sym,
        )
        res = self._order_manager.sweep_sell(
            sweep_sym,
            sell_qty,
            price=float(px),
            equity=float(snapshot.equity),
            daily_pnl_pct=float(snapshot.daily_pnl_pct),
            wait_for_fill=True,
        )
        snapshot = snapshot_from_broker(self._broker, symbols)
        return [res], snapshot, True

    def _run_cash_sweep(
        self,
        cycle_id: str,
        symbols: list[str],
        all_close: dict[str, float],
        when: datetime,
    ) -> list[OrderExecutionResult]:
        """Sweep idle cash above reserve+buffer into the T-bill ETF (Tier 44B).

        Cash management, not a strategy: no signals, no regime multiplier. Runs after
        strategy buys so it sweeps whatever cash they left unclaimed. The sweep symbol
        is outside the momentum universe (enforced in config), so rotation never touches
        it. Same-cycle sell proceeds may not be reflected in this snapshot's cash yet;
        the thermostat self-corrects next cycle (no settlement tracking).
        """
        cfg = self._settings.cash_sweep
        if not cfg.enabled:
            return []
        from src.execution.cash_sweep import compute_sweep_action

        sym = cfg.symbol.strip().upper()
        price = all_close.get(sym)
        if price is None or price <= 0.0:
            logger.warning(
                "cash_sweep skipped: no price for sweep symbol %s this cycle (never trade unpriced)",
                sym,
            )
            return []

        self._broker.refresh_account()
        snap = snapshot_from_broker(self._broker, symbols)
        sweep_position_notional = float(snap.positions.get(sym, 0.0)) * float(price)
        pending_buy_notional = self._order_manager.unsettled_pending_buy_notional()
        action = compute_sweep_action(
            cash=float(snap.cash),
            equity=float(snap.equity),
            sweep_position_notional=sweep_position_notional,
            min_cash_reserve_pct=float(self._settings.risk.min_cash_reserve_pct),
            buffer_pct=float(cfg.buffer_pct),
            min_trade_usd=float(cfg.min_trade_usd),
            pending_buy_notional=float(pending_buy_notional),
        )

        if action.action == "hold":
            if "Negative cash/holdings" in action.reason:
                logger.warning("cash_sweep hold: %s", action.reason)
            else:
                logger.debug("cash_sweep hold: %s", action.reason)
            return []

        res: OrderExecutionResult
        if action.action == "buy":
            logger.info(
                "cash_sweep buy symbol=%s notional=%.2f cash=%.2f equity=%.2f",
                sym,
                float(action.notional),
                float(snap.cash),
                float(snap.equity),
            )
            res = self._order_manager.sweep_buy(
                sym,
                float(action.notional),
                price=float(price),
                equity=float(snap.equity),
                cash=float(snap.cash),
                daily_pnl_pct=float(snap.daily_pnl_pct),
            )
        else:  # sell
            sell_qty = float(action.notional) / float(price)
            logger.info(
                "cash_sweep sell symbol=%s notional=%.2f qty=%.6f cash=%.2f equity=%.2f",
                sym,
                float(action.notional),
                sell_qty,
                float(snap.cash),
                float(snap.equity),
            )
            res = self._order_manager.sweep_sell(
                sym,
                sell_qty,
                price=float(price),
                equity=float(snap.equity),
                daily_pnl_pct=float(snap.daily_pnl_pct),
            )

        if self._sqlite is not None:
            eid = self._sqlite.log_execution(cycle_id, res, account_id=self._account_id)
            self._persist_execution_quality(cycle_id, eid, res)
        return [res]
