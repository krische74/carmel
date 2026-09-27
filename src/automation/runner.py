"""Default wiring from ``Settings`` and CLI entrypoint (``carmel``)."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.automation.alerts import Alert, AlertLevel
from src.automation.health import clear_heartbeat, read_heartbeat, write_heartbeat
from src.automation.notifier import CompositeNotifier, EmailNotifier, Notifier, WebhookNotifier
from src.automation.scheduler import HubScheduler, cron_trigger_from_expression
from src.automation.strategy_wiring import (
    build_strategies_from_enabled,
    build_strategy_for_backtest,
)
from src.automation.workflows import TradingWorkflow
from src.backtesting.engine import (
    BacktestConfig,
    BacktestDataMissingError,
    BacktestEngine,
    compute_benchmark_metrics,
    trading_days_in_range,
)
from src.config import (
    PROJECT_ROOT,
    Settings,
    get_settings,
    hub_sqlite_path,
    parquet_dir,
    parse_as_of_iso,
    public_settings_dict,
    setup_logging,
)
from src.data.adapters.edgar_adapter import EdgarAdapter
from src.data.adapters.fred_adapter import FredAdapter
from src.data.adapters.yfinance_adapter import YFinanceAdapter
from src.data.pipeline import DataPipeline
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.execution.account_factory import build_brokers
from src.execution.alpaca_adapter import AlpacaBrokerAdapter
from src.execution.errors import ConfigurationError
from src.execution.order_manager import OrderManager
from src.execution.reconciliation import reconcile_executions
from src.portfolio.tax_lots import LotLedger
from src.risk.kill_switch import KillSwitch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.execution.broker_interface import BrokerInterface

logger = logging.getLogger(__name__)

HUB_TRADING_CYCLE_LOCK = threading.Lock()
HEARTBEAT_FILE = PROJECT_ROOT / "logs" / "hub.heartbeat"


def _abort_if_other_daemon_running(
    settings: Settings,
    *,
    heartbeat_path: Path = HEARTBEAT_FILE,
    now: datetime | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    observation_seconds: int | None = None,
) -> None:
    """Refuse startup only when another daemon is *actively* writing the heartbeat.

    A fresh-by-age heartbeat alone is no longer sufficient evidence — routine
    ``docker compose restart`` leaves a recent-but-frozen heartbeat that
    previously caused false-positive aborts and Discord spam. The check now
    observes the heartbeat for one heartbeat interval + buffer; only an
    advancing timestamp during observation triggers an abort.

    The shared ``logs/`` volume is still the cross-namespace signal (PIDs
    are not portable across container/host PID namespaces). Exits with
    status 2 (and a CRITICAL webhook, when configured) on confirmed duplicate.
    """
    stamp = read_heartbeat(heartbeat_path)
    if stamp is None:
        return
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    interval = max(1, int(settings.scheduler.heartbeat_interval_seconds))
    threshold_seconds = interval * 2 + 30
    current = now or datetime.now(UTC)
    age = (current - stamp).total_seconds()
    if age >= threshold_seconds:
        return  # already stale by age — previous writer is definitely gone
    obs = observation_seconds if observation_seconds is not None else interval + 10
    logger.info(
        "Heartbeat at %s is %.1fs old; observing %ds to verify a writer is active.",
        heartbeat_path,
        age,
        obs,
    )
    if obs > 0:
        sleep_fn(obs)
    later = read_heartbeat(heartbeat_path)
    if later is None:
        return  # heartbeat removed during observation — clean to proceed
    if later.tzinfo is None:
        later = later.replace(tzinfo=UTC)
    if later <= stamp:
        logger.info(
            "Heartbeat at %s did not advance during observation; treating as "
            "stale leftover from a prior process and proceeding.",
            heartbeat_path,
        )
        return
    later_age = (datetime.now(UTC) - later).total_seconds()
    msg = (
        f"Another carmel daemon is actively writing {heartbeat_path} "
        f"(timestamp advanced during a {obs}s observation; last update "
        f"{later_age:.1f}s ago). Refusing to start a second scheduler — "
        f"two schedulers submit duplicate orders. If you are sure no other "
        f"instance is running, delete the heartbeat file and retry."
    )
    logger.critical(msg)
    notify_operational_critical(
        settings,
        category="duplicate_scheduler",
        message=msg,
    )
    raise SystemExit(2)


def broker_from_settings(settings: Settings) -> AlpacaBrokerAdapter:
    """Build ``AlpacaBrokerAdapter`` from settings; raises ``ValueError`` if keys are missing."""
    key = (settings.alpaca_api_key or "").strip()
    secret = (settings.alpaca_secret_key or "").strip()
    if not key or not secret:
        msg = (
            "Alpaca API credentials are missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "in the environment, or pass broker= to create_trading_workflow()."
        )
        raise ValueError(msg)
    logical = (settings.broker.default_account or "").strip() or "default"
    return AlpacaBrokerAdapter.create(
        key,
        secret,
        paper=settings.broker.paper_trading,
        logical_account_id=logical,
    )


def notify_operational_critical(
    settings: Settings,
    *,
    category: str,
    message: str,
) -> None:
    """Send a CRITICAL alert when webhook/email is configured (no-op otherwise)."""
    notifier = _notifier_from_settings(settings)
    if notifier is None:
        return
    notifier.send(
        Alert(
            timestamp=datetime.now(UTC),
            level=AlertLevel.CRITICAL,
            category=category,
            message=message,
        ),
    )


def _notifier_from_settings(settings: Settings) -> Notifier | None:
    """Build webhook/email notifier(s) from notification settings."""
    ncfg = settings.notification
    notifiers: list[Notifier] = []
    if ncfg.enabled and (ncfg.webhook_url or "").strip():
        notifiers.append(
            WebhookNotifier(
                ncfg.webhook_url.strip(),
                telegram_chat_id=(ncfg.telegram_chat_id or "").strip(),
                timeout_seconds=ncfg.timeout_seconds,
                max_retries=ncfg.max_retries,
                backoff_seconds=ncfg.backoff_seconds,
            ),
        )
    if ncfg.email_enabled:
        smtp_host = (ncfg.smtp_host or "").strip() or "localhost"
        notifiers.append(
            EmailNotifier(
                smtp_host,
                ncfg.smtp_port,
                use_tls=ncfg.smtp_use_tls,
                username=(settings.smtp_username or "").strip(),
                password=settings.smtp_password or "",
                from_addr=(ncfg.email_from or "carmel@localhost").strip(),
                to_addrs=list(ncfg.email_to),
                timeout_seconds=ncfg.timeout_seconds,
            ),
        )
    if len(notifiers) == 1:
        return notifiers[0]
    if len(notifiers) > 1:
        return CompositeNotifier(notifiers)
    return None


def _run_weekly_digest(settings: Settings, workflow: TradingWorkflow) -> None:
    """Load recent activity, build structured digest + optional narrative, send via notifier."""
    from src.ai.explainer import generate_weekly_digest
    from src.automation.digest import generate_digest_summary, render_digest_plain_text
    from src.reporting.attribution import compute_position_contributions
    from src.reporting.portfolio_analytics import load_current_prices

    if workflow.notifier is None:
        logger.info("Weekly digest skipped: no notifier configured.")
        return

    store = SQLiteStore(hub_sqlite_path(settings))
    end = date.today()
    start = end - timedelta(days=7)
    snaps = store.get_equity_snapshots(start=start.isoformat(), end=end.isoformat())
    execs = _rows_in_digest_window(store.get_executions(limit=500), start=start, end=end)
    alts = _rows_in_digest_window(store.get_alerts(limit=200), start=start, end=end)

    contrib = None
    ledger = workflow.lot_ledger
    if ledger is not None:
        pq = ParquetStore(parquet_dir(settings))
        open_lots = ledger.get_open_lots()
        closed = ledger.get_closed_lots(limit=500)
        syms = sorted({o.symbol for o in open_lots} | {c.symbol for c in closed})
        prices = load_current_prices(pq, syms) if syms else {}
        contrib = compute_position_contributions(ledger, prices)

    digest_summary = generate_digest_summary(store, lookback_hours=168, account_id=None)
    digest_text = render_digest_plain_text(digest_summary)

    narrative: str | None = None
    if getattr(workflow, "_llm_client", None) is not None:
        from src.ai.llm_explainer import generate_llm_weekly_digest

        narrative = generate_llm_weekly_digest(workflow._llm_client, snaps, execs, alts, contrib)
    if narrative is None:
        narrative = generate_weekly_digest(snaps, execs, alts, contrib)

    body = f"{digest_text}\n\n---\n\n{narrative}"
    workflow.notifier.send(
        Alert(
            timestamp=datetime.now(UTC),
            level=AlertLevel.INFO,
            category="weekly_digest",
            message=body,
        ),
    )


def _rows_in_digest_window(
    rows: list[dict[str, Any]],
    *,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """Keep rows whose ``timestamp`` falls on a calendar day in ``[start, end]`` (inclusive)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        raw = row.get("timestamp")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            s = str(raw).strip().replace("Z", "+00:00")
            ts = datetime.fromisoformat(s)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        d = ts.date()
        if start <= d <= end:
            out.append(row)
    return out


def _create_workflow_or_log_alpaca_help(settings: Settings) -> TradingWorkflow | None:
    """Build a workflow for the first configured account, or log help if credentials are absent."""
    try:
        brokers = build_brokers(settings)
    except ValueError as exc:
        msg = str(exc)
        if "Alpaca API credentials are missing" not in msg and "Missing credentials for broker" not in msg:
            raise
        logger.error("%s", exc)
        logger.error(
            "To fix: copy .env.example to .env, set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY (paper keys from https://app.alpaca.markets/), "
            "or configure broker.accounts with per-account env vars, "
            "or run: carmel setup",
        )
        return None
    _name, br = next(iter(brokers.items()))
    return create_trading_workflow(settings, broker=br)


def create_trading_workflow(
    settings: Settings,
    *,
    broker: BrokerInterface | None = None,
    sqlite_store: SQLiteStore | None = None,
    lot_ledger: LotLedger | None = None,
) -> TradingWorkflow:
    """Wire YFinance → Parquet/SQLite, configured strategies, and ``OrderManager``.

    If ``broker`` is omitted, builds ``AlpacaBrokerAdapter`` from settings/env keys.
    Reuse ``sqlite_store`` / ``lot_ledger`` when running multiple account cycles in one process.
    """
    b = broker if broker is not None else broker_from_settings(settings)
    parquet = ParquetStore(parquet_dir(settings))
    db_path = hub_sqlite_path(settings)
    sqlite = sqlite_store if sqlite_store is not None else SQLiteStore(db_path)
    ledger = lot_ledger if lot_ledger is not None else LotLedger(db_path)
    adapter = YFinanceAdapter()
    fred_key = (settings.fred_api_key or "").strip()
    fred_adapter = FredAdapter(fred_key) if fred_key else None
    edgar_adapter = EdgarAdapter() if settings.fundamental.enabled else None
    dcfg = settings.data
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=parquet,
        sqlite_store=sqlite,
        fred_adapter=fred_adapter,
        edgar_adapter=edgar_adapter,
        ohlcv_outlier_zscore_threshold=dcfg.ohlcv_outlier_zscore_threshold,
        ohlcv_min_prior_for_zscore=dcfg.ohlcv_min_prior_for_zscore,
        ohlcv_max_abs_daily_return=dcfg.ohlcv_max_abs_daily_return,
        validation_exempt_symbols=dcfg.validation_exempt_symbols,
        initial_backfill_years=dcfg.initial_backfill_years,
    )
    fred_pipeline = pipeline if fred_adapter is not None else None
    strategies = build_strategies_from_enabled(settings)
    kill_switch = KillSwitch(settings.risk.daily_loss_limit_pct)
    order_manager = OrderManager(
        broker=b,
        settings=settings,
        kill_switch=kill_switch,
        sqlite_store=sqlite,
        lot_ledger=ledger,
    )
    notifier = _notifier_from_settings(settings)

    llm_client = None
    if settings.llm.enabled:
        from src.ai.llm_client import OllamaClient

        llm_client = OllamaClient(settings.llm)

    return TradingWorkflow(
        settings=settings,
        data_pipeline=pipeline,
        parquet_store=parquet,
        strategies=strategies,
        order_manager=order_manager,
        broker=b,
        sqlite_store=sqlite,
        lot_ledger=ledger,
        kill_switch=kill_switch,
        notifier=notifier,
        fred_pipeline=fred_pipeline,
        llm_client=llm_client,
    )


def run_hub_trading_cycles_for_all_accounts(
    *,
    settings: Settings,
    brokers: dict[str, BrokerInterface],
    sqlite: SQLiteStore,
    ledger: LotLedger,
    trigger: str,
) -> None:
    """Run ``TradingWorkflow.run_cycle`` for each broker (multi-account hub).

    Uses a process-wide non-reentrant lock so overlapping signal and rebalance
    scheduler callbacks (or API triggers) do not run two full passes concurrently.
    """
    if not HUB_TRADING_CYCLE_LOCK.acquire(blocking=False):
        logger.info(
            "Skipping trading cycle (%s): another invocation is already running",
            trigger,
        )
        return
    try:
        for name, br in brokers.items():
            logger.info("Running cycle for account: %s (%s)", name, trigger)
            try:
                wf = create_trading_workflow(
                    settings,
                    broker=br,
                    sqlite_store=sqlite,
                    lot_ledger=ledger,
                )
                wf.run_cycle()
            except Exception as exc:
                logger.exception(
                    "Cycle failed for account %s (%s)",
                    name,
                    trigger,
                )
                notify_operational_critical(
                    settings,
                    category="cycle_failed",
                    message=(
                        f"Cycle failed for account '{name}' (trigger {trigger}): "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
    finally:
        HUB_TRADING_CYCLE_LOCK.release()


def _run_ml_retraining_job(settings: Settings) -> None:
    """APScheduler callback: retrain momentum ML bundle from Parquet (best-effort)."""
    from src.ml.retraining import retrain_ml_bundle

    retrain_ml_bundle(settings)


def _emit_liveness_ping(settings: Settings, sqlite: SQLiteStore) -> None:
    """Send a periodic 'daemon is alive' alert with last-cycle context.

    Confirms the daemon process is running and reports when it last completed a
    trading cycle. Absence of this ping (alongside no ``cycle_summary``) is the
    signal that the daemon has died or is otherwise stuck.
    """
    notifier = _notifier_from_settings(settings)
    if notifier is None:
        return
    last_cycle: str = "never"
    try:
        for row in sqlite.get_alerts(limit=200):
            if str(row.get("category") or "").strip() == "cycle_summary":
                last_cycle = str(row.get("timestamp") or "unknown")
                break
    except Exception as exc:  # best-effort lookup; do not fail the ping
        logger.warning("Liveness ping: failed to read last cycle from alerts: %s", exc)

    hb = read_heartbeat(HEARTBEAT_FILE)
    if hb is None:
        hb_age = "missing"
    else:
        if hb.tzinfo is None:
            hb = hb.replace(tzinfo=UTC)
        hb_age = f"{(datetime.now(UTC) - hb).total_seconds():.0f}s ago"

    notifier.send(
        Alert(
            timestamp=datetime.now(UTC),
            level=AlertLevel.INFO,
            category="liveness",
            message=(
                f"Carmel daemon alive. Last cycle_summary: {last_cycle}. "
                f"Heartbeat: {hb_age}."
            ),
        ),
    )


def _run_liquidate_cli(settings: Settings, symbol: str) -> int:
    """Market-sell an entire long position and log an execution row (Tier 48D)."""
    sym = symbol.strip().upper()
    if not sym:
        logger.error("liquidate requires a non-empty SYMBOL")
        return 1
    broker = broker_from_settings(settings)
    store = SQLiteStore(hub_sqlite_path(settings))
    kill_switch = KillSwitch(settings.risk.daily_loss_limit_pct)
    order_manager = OrderManager(
        broker=broker,
        settings=settings,
        kill_switch=kill_switch,
        sqlite_store=store,
    )
    broker.refresh_account()
    qty = float(broker.get_position_qty(sym))
    if qty <= 0.0:
        logger.error("No long position in %s to liquidate (qty=%.6f)", sym, qty)
        return 1
    equity = float(broker.get_account_equity())
    last_eq = float(broker.get_last_equity())
    daily_pnl_pct = (equity - last_eq) / last_eq if last_eq > 0.0 else 0.0
    parquet = ParquetStore(parquet_dir(settings))
    from src.reporting.portfolio_analytics import load_current_prices

    prices = load_current_prices(parquet, [sym])
    price = float(prices.get(sym, 0.0))
    if price <= 0.0:
        logger.error("No OHLCV price for %s — run ingest first", sym)
        return 1
    cycle_id = f"liquidate-{sym}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    res = order_manager.close_position(
        sym,
        qty,
        price=price,
        equity=equity,
        daily_pnl_pct=daily_pnl_pct,
    )
    from src.execution.account_factory import resolve_sqlite_account_id

    account_id = resolve_sqlite_account_id(broker)
    eid = store.log_execution(cycle_id, res, account_id=account_id)
    if res.submitted:
        logger.info(
            "liquidate %s qty=%.6f order_id=%s execution_row=%s",
            sym,
            qty,
            res.order_id,
            eid,
        )
        return 0
    logger.error("liquidate %s failed: %s", sym, res.reason)
    return 1


def _run_reconcile_cli(settings: Settings) -> int:
    """Load SQLite executions and broker orders; log reconciliation summary."""
    broker = broker_from_settings(settings)
    store = SQLiteStore(hub_sqlite_path(settings))
    rows = store.get_executions(limit=500)
    broker_orders = broker.list_recent_orders(limit=500)
    result = reconcile_executions(rows, broker_orders)
    logger.info(
        "Reconciliation: matched=%s, discrepancies=%s",
        result.matched,
        result.discrepancies,
    )
    for e in result.entries:
        logger.info(
            "  %s %s %s — %s (local_qty=%s broker_qty=%s local_price=%s broker_price=%s)",
            e.order_id,
            e.symbol,
            e.side,
            e.status,
            e.local_qty,
            e.broker_qty,
            e.local_price,
            e.broker_price,
        )

    from src.execution.lot_reconciliation import reconcile_lot_ledger
    from src.reporting.portfolio_analytics import compute_portfolio_valuation, load_current_prices

    db_path = hub_sqlite_path(settings)
    ledger = LotLedger(db_path)
    parquet = ParquetStore(parquet_dir(settings))
    aid = "default"
    open_lots = ledger.get_open_lots(account_id=aid)
    ledger_qty: dict[str, float] = {}
    for lot in open_lots:
        ledger_qty[lot.symbol] = ledger_qty.get(lot.symbol, 0.0) + float(lot.qty)
    broker.refresh_account()
    syms = sorted(set(ledger_qty) | set(settings.data.universe) | {t.symbol for t in settings.data.dca_targets})
    if settings.cash_sweep.enabled:
        syms = sorted(set(syms) | {settings.cash_sweep.symbol.strip().upper()})
    from src.portfolio.state import snapshot_from_broker

    snap = snapshot_from_broker(broker, syms)
    prices = load_current_prices(parquet, syms)
    valuation = compute_portfolio_valuation(ledger, prices, account_id=aid)
    lot_rec = reconcile_lot_ledger(
        ledger_qty_by_symbol=ledger_qty,
        broker_qty_by_symbol=dict(snap.positions),
        ledger_market_value=float(valuation.total_market_value),
        cash=float(snap.cash),
        broker_equity=float(snap.equity),
    )
    logger.info(
        "Lot reconciliation: qty_mismatches=%s equity_mismatch=%s "
        "ledger_equity=%.2f broker_equity=%.2f",
        lot_rec.qty_mismatches,
        lot_rec.equity_mismatch,
        lot_rec.ledger_equity,
        lot_rec.broker_equity,
    )
    for e in lot_rec.entries:
        if e.status != "matched":
            logger.warning(
                "  lot %s — %s (ledger_qty=%s broker_qty=%s)",
                e.symbol,
                e.status,
                e.ledger_qty,
                e.broker_qty,
            )

    bad = result.discrepancies > 0 or lot_rec.qty_mismatches > 0 or lot_rec.equity_mismatch
    return 1 if bad else 0


def _backup_hub_sqlite(db_path: Path) -> Path:
    """Copy hub SQLite beside the original with a timestamped ``.bak.`` suffix."""
    import shutil

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_name(f"{db_path.name}.bak.{stamp}")
    shutil.copy2(db_path, backup)
    return backup


def _run_rebuild_lots_cli(
    settings: Settings,
    *,
    since: date,
    account_id: str,
    no_backup: bool,
    end: date | None,
) -> int:
    """Rebuild tax lots from trade_executions and rewrite equity_snapshots."""
    from src.portfolio.lot_ledger_rebuild import (
        rebuild_equity_snapshots_from_ledger,
        rebuild_lot_ledger_from_executions,
    )

    db_path = hub_sqlite_path(settings)
    if not db_path.is_file():
        logger.error("Hub SQLite not found: %s", db_path)
        return 1
    if not no_backup:
        backup = _backup_hub_sqlite(db_path)
        logger.info("Backed up hub SQLite to %s", backup)

    store = SQLiteStore(db_path)
    ledger = LotLedger(db_path)
    parquet = ParquetStore(parquet_dir(settings))
    since_dt = datetime(since.year, since.month, since.day, tzinfo=UTC)
    method = settings.tax.lot_selection_method
    result = rebuild_lot_ledger_from_executions(
        store,
        ledger,
        parquet_store=parquet,
        since=since_dt,
        account_id=account_id,
        method=method,
    )
    logger.info(
        "Lot rebuild: buys=%s sells=%s skipped=%s cost_basis=%.2f realized_pnl=%.2f open=%s",
        result.buys_replayed,
        result.sells_replayed,
        result.skipped,
        result.total_cost_basis,
        result.realized_pnl,
        result.open_symbols,
    )
    end_d = end or datetime.now(UTC).date()
    points = rebuild_equity_snapshots_from_ledger(
        store,
        ledger,
        parquet,
        start=since,
        end=end_d,
        account_id=account_id,
    )
    newest = points[-1] if points else None
    if newest is not None:
        logger.info(
            "Equity curve rebuilt %s -> %s (%s days); newest MV=%.2f cost=%.2f realized=%.2f",
            since.isoformat(),
            end_d.isoformat(),
            len(points),
            newest.total_market_value,
            newest.total_cost_basis,
            newest.realized_pnl,
        )
    return 0


def _run_backfill_snapshot_cash_cli(
    settings: Settings,
    *,
    since: date,
    seed: float,
    account_id: str,
    no_backup: bool,
    skip_broker_check: bool,
) -> int:
    """Replay execution cash flows onto equity_snapshots.cash."""
    from src.reporting.backfill_snapshot_cash import backfill_snapshot_cash

    db_path = hub_sqlite_path(settings)
    if not db_path.is_file():
        logger.error("Hub SQLite not found: %s", db_path)
        return 1
    if not no_backup:
        backup = _backup_hub_sqlite(db_path)
        logger.info("Backed up hub SQLite to %s", backup)

    store = SQLiteStore(db_path)
    broker_cash: float | None = None
    if not skip_broker_check:
        try:
            broker = broker_from_settings(settings)
            broker.refresh_account()
            broker_cash = float(broker.get_cash())
        except (ValueError, ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
            logger.warning(
                "Could not read broker cash for self-check (%s); continuing without delta.",
                exc,
            )

    parquet = ParquetStore(parquet_dir(settings))
    result = backfill_snapshot_cash(
        store,
        since=since,
        seed=float(seed),
        account_id=account_id,
        broker_cash=broker_cash,
        parquet_store=parquet,
    )
    logger.info(
        "Snapshot cash backfill: rows=%s final_cash=%.2f seed=%.2f delta_vs_broker=%s",
        result.rows_updated,
        result.final_cash,
        result.seed,
        f"{result.delta_vs_broker:.2f}" if result.delta_vs_broker is not None else "n/a",
    )
    return 0


def _fetch_fill_quote(broker: BrokerInterface, order_id: str) -> Any:
    """One broker lookup for filled_qty and filled_avg_price. Missing order stays missing."""
    from src.execution.errors import OrderNotFoundError
    from src.reporting.backfill_fill_qty import BrokerOrderMissingError, FillQuote

    get_fill = getattr(broker, "get_order_fill", None)
    if not callable(get_fill):
        msg = "broker does not implement get_order_fill; refusing to guess filled_qty from requested qty"
        raise RuntimeError(msg)
    try:
        filled_qty, filled_avg_price = get_fill(order_id)
    except OrderNotFoundError as exc:
        raise BrokerOrderMissingError(order_id) from exc
    return FillQuote(filled_qty=filled_qty, filled_avg_price=filled_avg_price)


def _run_backfill_fill_qty_cli(
    settings: Settings,
    *,
    since: date,
    account_id: str,
    dry_run: bool,
    no_backup: bool,
) -> int:
    """Write broker filled_qty onto NULL trade_executions rows. Never rewrites qty."""
    from src.reporting.backfill_fill_qty import (
        backfill_filled_qty,
        format_backfill_fill_qty_summary,
    )

    db_path = hub_sqlite_path(settings)
    if not db_path.is_file():
        logger.error("Hub SQLite not found: %s", db_path)
        return 1
    if not dry_run and not no_backup:
        backup = _backup_hub_sqlite(db_path)
        logger.info("Backed up hub SQLite to %s", backup)

    broker = broker_from_settings(settings)
    store = SQLiteStore(db_path)
    since_dt = datetime(since.year, since.month, since.day, tzinfo=UTC)
    result = backfill_filled_qty(
        store,
        since=since_dt,
        account_id=account_id,
        fetch_fill=lambda order_id: _fetch_fill_quote(broker, order_id),
        dry_run=dry_run,
    )
    summary = format_backfill_fill_qty_summary(result)
    logger.info("\n%s", summary)
    print(summary, flush=True)
    return 0


def _run_daemon(settings: Settings, brokers: dict[str, BrokerInterface]) -> None:
    """Start scheduler; ingest uses the first broker, signal/rebalance run all accounts sequentially."""
    _abort_if_other_daemon_running(settings)
    db_path = hub_sqlite_path(settings)
    sqlite = SQLiteStore(db_path)
    ledger = LotLedger(db_path)
    first_br = next(iter(brokers.values()))
    workflow = create_trading_workflow(
        settings,
        broker=first_br,
        sqlite_store=sqlite,
        lot_ledger=ledger,
    )
    hub = HubScheduler(settings=settings)
    logger.info("Scheduler timezone: %s", settings.scheduler.timezone)
    heartbeat_path = HEARTBEAT_FILE

    def run_signal_account_cycles() -> None:
        run_hub_trading_cycles_for_all_accounts(
            settings=settings,
            brokers=brokers,
            sqlite=sqlite,
            ledger=ledger,
            trigger="signal_generation",
        )

    def run_rebalance_account_cycles() -> None:
        run_hub_trading_cycles_for_all_accounts(
            settings=settings,
            brokers=brokers,
            sqlite=sqlite,
            ledger=ledger,
            trigger="rebalance_check",
        )

    hub.register_trading_jobs(
        run_data_ingestion=workflow.run_ingest_only,
        run_signal_cycle=run_signal_account_cycles,
        run_rebalance_check=run_rebalance_account_cycles,
    )
    logger.info(
        "Signal and rebalance crons invoke full account cycles; "
        "concurrent invocations are serialized via a process lock.",
    )
    hub.scheduler.add_job(
        lambda: write_heartbeat(heartbeat_path),
        "interval",
        seconds=max(1, settings.scheduler.heartbeat_interval_seconds),
        id="heartbeat",
        replace_existing=True,
    )
    rc = (settings.scheduler.reconcile_cron or "").strip()
    if rc:
        hub.scheduler.add_job(
            lambda: _run_reconcile_cli(settings),
            cron_trigger_from_expression(rc),
            id="reconciliation",
            replace_existing=True,
        )
        logger.info("Reconciliation cron: %s", rc)
    dg = (settings.scheduler.digest_cron or "").strip()
    if dg:
        hub.scheduler.add_job(
            lambda: _run_weekly_digest(settings, workflow),
            cron_trigger_from_expression(dg),
            id="weekly_digest",
            replace_existing=True,
        )
        logger.info("Weekly digest cron: %s", dg)
    ml_cron = (settings.scheduler.ml_retraining_cron or "").strip()
    if settings.ml.enabled and ml_cron:
        hub.scheduler.add_job(
            lambda s=settings: _run_ml_retraining_job(s),
            cron_trigger_from_expression(ml_cron),
            id="ml_retraining",
            replace_existing=True,
        )
        logger.info("ML retraining cron: %s", ml_cron)
    lv_cron = (settings.scheduler.liveness_cron or "").strip()
    if lv_cron:
        hub.scheduler.add_job(
            lambda: _emit_liveness_ping(settings, sqlite),
            cron_trigger_from_expression(lv_cron),
            id="liveness",
            replace_existing=True,
        )
        logger.info("Liveness cron: %s", lv_cron)
    # Translate Docker's SIGTERM into KeyboardInterrupt so the finally block
    # runs and the heartbeat file is removed — otherwise the next container
    # start sees a fresh-but-frozen heartbeat and pays the observation cost.
    def _raise_keyboard_interrupt(*_args: object) -> None:
        raise KeyboardInterrupt

    prev_sigterm = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    hub.scheduler.start()
    logger.info("scheduler started (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("interrupt received, shutting down")
    finally:
        hub.scheduler.shutdown(wait=True)
        clear_heartbeat(heartbeat_path)
        signal.signal(signal.SIGTERM, prev_sigterm)


def load_backtest_data(
    settings: Settings,
    strategy_name: str,
    *,
    parquet: ParquetStore | None = None,
) -> tuple[object, dict[str, pd.DataFrame]]:
    """Load strategy + OHLCV frames the same way ``carmel backtest`` does."""
    strat = build_strategy_for_backtest(settings, strategy_name)
    store = parquet or ParquetStore(parquet_dir(settings))
    symbols = strat.get_universe()
    data: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in symbols:
        df = store.read_ohlcv(sym)
        if df.empty:
            missing.append(sym)
        else:
            data[sym] = df
    if missing:
        msg = f"No Parquet OHLCV data for: {', '.join(missing)}."
        raise BacktestDataMissingError(
            start=date.min,
            end=date.max,
            symbols=missing,
            detail=msg,
        )
    vix_sym = settings.regime.vix_symbol.strip().upper()
    if vix_sym not in data:
        vix_df = store.read_ohlcv(vix_sym)
        if not vix_df.empty:
            data[vix_sym] = vix_df
    return strat, data


def _run_backtest_cli(
    settings: Settings,
    *,
    strategy_name: str,
    start: date,
    end: date,
    capital: float,
    slippage_bps: float,
    rebalance: str,
    raw_signal: bool = False,
    benchmark_symbol: str | None = None,
) -> int:
    """Load Parquet OHLCV, run ``BacktestEngine``, print summary. Return exit code."""
    if start > end:
        logger.error(
            "--start must be on or before --end (got %s and %s)",
            start.isoformat(),
            end.isoformat(),
        )
        return 1

    parquet = ParquetStore(parquet_dir(settings))
    try:
        strat, data = load_backtest_data(settings, strategy_name, parquet=parquet)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    except BacktestDataMissingError as exc:
        logger.error("%s", exc.detail or exc)
        return 1

    bench_sym = (benchmark_symbol or settings.backtest.benchmark_symbol).strip().upper()
    store = SQLiteStore(hub_sqlite_path(settings))
    fallback_rf = float(settings.backtest.cash_yield_annual_pct)
    cash_map: dict[str, float] | None = None
    sid = str(settings.backtest.cash_yield_series_id or "").strip()
    if sid:
        from src.backtesting.cash_yield import series_from_macro_rows

        rows = store.get_macro_indicator(sid)
        ser = series_from_macro_rows(rows)
        if ser.empty:
            logger.warning(
                "cash_yield_series_id=%s missing in sqlite; using fallback_flat %.2f%%",
                sid,
                fallback_rf,
            )
        else:
            cash_map = {ts.date().isoformat(): float(v) for ts, v in ser.items()}
    cfg_bt = BacktestConfig(
        initial_capital=float(capital),
        slippage_bps=float(slippage_bps),
        rebalance_frequency=rebalance,
        warmup_bars=int(settings.backtest.warmup_bars),
        min_coverage_ratio=float(settings.backtest.min_coverage_ratio),
        apply_risk_layer=not raw_signal,
        cash_yield_annual_pct=fallback_rf,
        cash_yield_by_date=cash_map,
    )
    if cfg_bt.use_regime:
        vix_sym = settings.regime.vix_symbol.strip().upper()
        if vix_sym not in data:
            vix_df = parquet.read_ohlcv(vix_sym)
            if vix_df.empty:
                logger.warning(
                    "No Parquet OHLCV for %s; backtest uses VIX=None for regime.",
                    vix_sym,
                )
            else:
                data[vix_sym] = vix_df

    try:
        result = BacktestEngine().run(
            strat,
            data,
            start=start,
            end=end,
            config=cfg_bt,
            settings=settings,
        )
    except BacktestDataMissingError as exc:
        syms_csv = ",".join(sorted({s.strip().upper() for s in strat.get_universe()}))
        print(f"ERROR: {exc}", flush=True)
        lo, hi = exc.available_start, exc.available_end
        if lo is not None and hi is not None and lo <= hi:
            overlap_start = max(start, lo)
            overlap_end = min(end, hi)
            print("", flush=True)
            print(
                f"SUGGESTION: Your data covers {lo.isoformat()} to {hi.isoformat()}. Try:",
                flush=True,
            )
            if overlap_start <= overlap_end:
                print(
                    f"    carmel backtest --strategy {strategy_name} "
                    f"--start {overlap_start.isoformat()} --end {overlap_end.isoformat()}",
                    flush=True,
                )
            else:
                print(
                    f"    carmel backtest --strategy {strategy_name} "
                    f"--start {lo.isoformat()} --end {hi.isoformat()}",
                    flush=True,
                )
        print("", flush=True)
        print(
            "To extend coverage, run:",
            flush=True,
        )
        print(
            f"    carmel ingest --start {start.isoformat()} --end {end.isoformat()}",
            flush=True,
        )
        print(
            f"(or narrow symbols: carmel ingest --symbols {syms_csv} --start {start.isoformat()} "
            f"--end {end.isoformat()})",
            flush=True,
        )
        return 1
    trading_days = trading_days_in_range(data, start, end)
    warm = max(0, min(int(cfg_bt.warmup_bars), 500))
    if warm > 0 and len(trading_days) > warm:
        trading_days = trading_days[warm:]
    rf_daily = None
    if cash_map:
        rf_daily = pd.Series(
            {k: (float(v) / 100.0) / 252.0 for k, v in cash_map.items()},
            dtype=float,
        )
    bench_metrics = compute_benchmark_metrics(
        data,
        bench_sym,
        trading_days,
        risk_free_rate_annual=cfg_bt.cash_yield_annual_pct,
        risk_free_daily=rf_daily,
    )
    result.benchmark_symbol = bench_sym
    result.benchmark_metrics = bench_metrics
    m = result.return_metrics
    strat_label = type(strat).__name__
    run_id = store.save_backtest_run(
        strat_label,
        cfg_bt,
        result,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        benchmark_symbol=bench_sym,
        benchmark_metrics=bench_metrics,
    )
    mode_label = "raw signal (no risk layer)" if raw_signal else "constrained (production-faithful)"
    print(f"Backtest: {strat_label} ({start.isoformat()} to {end.isoformat()})", flush=True)
    print(f"Sizing mode:        {mode_label}", flush=True)
    print(
        f"Cash yield:         mode={result.cash_yield_mode} "
        f"realized={result.realized_cash_yield_annual_pct:.2f}% "
        f"fallback={cfg_bt.cash_yield_annual_pct:.2f}%",
        flush=True,
    )
    print(
        f"Equity exposure:    avg={result.avg_exposure_pct:.1%} max={result.max_exposure_pct:.1%}",
        flush=True,
    )
    print(f"Initial capital:    ${result.initial_capital:,.2f}", flush=True)
    print(f"Final equity:       ${result.final_equity:,.2f}", flush=True)
    print(f"Total return:       {m.total_return_pct:.2f}%", flush=True)
    print(f"CAGR:               {m.cagr_pct:.2f}%", flush=True)
    sharpe = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None else "N/A"
    sortino = f"{m.sortino_ratio:.2f}" if m.sortino_ratio is not None else "N/A"
    calmar = f"{m.calmar_ratio:.2f}" if m.calmar_ratio is not None else "N/A"
    print(f"Sharpe ratio:       {sharpe}", flush=True)
    print(f"Sortino ratio:      {sortino}", flush=True)
    print(f"Max drawdown:       {m.max_drawdown_pct:.2f}%", flush=True)
    print(f"Calmar ratio:       {calmar}", flush=True)
    print(f"Annual volatility:  {m.annual_volatility_pct:.2f}%", flush=True)
    print(f"Trades executed:    {len(result.trades)}", flush=True)
    if bench_metrics is not None:
        print("", flush=True)
        print(f"Benchmark ({bench_sym}):", flush=True)
        print(f"  Total return:       {bench_metrics.total_return_pct:.2f}%", flush=True)
        print(f"  CAGR:               {bench_metrics.cagr_pct:.2f}%", flush=True)
        b_sharpe = (
            f"{bench_metrics.sharpe_ratio:.2f}" if bench_metrics.sharpe_ratio is not None else "N/A"
        )
        b_calmar = (
            f"{bench_metrics.calmar_ratio:.2f}" if bench_metrics.calmar_ratio is not None else "N/A"
        )
        print(f"  Sharpe ratio:       {b_sharpe}", flush=True)
        print(f"  Max drawdown:       {bench_metrics.max_drawdown_pct:.2f}%", flush=True)
        print(f"  Calmar ratio:       {b_calmar}", flush=True)
        print(
            f"  Annual volatility:  {bench_metrics.annual_volatility_pct:.2f}%",
            flush=True,
        )
    print(f"Run saved (ID: {run_id})", flush=True)
    if result.zero_trades_warning:
        print("", flush=True)
        print(f"WARNING: {result.zero_trades_warning}", flush=True)
    return 0


def _ingest_symbols_from_settings(settings: Settings, symbols_csv: str | None) -> list[str]:
    """Resolve ticker list from ``--symbols`` or the canonical required-symbol set."""
    from src.data.symbol_resolver import resolve_required_symbols

    if symbols_csv and str(symbols_csv).strip():
        return sorted({s.strip().upper() for s in str(symbols_csv).split(",") if s.strip()})
    return sorted(resolve_required_symbols(settings))


def _run_ingest_cli(settings: Settings, args: argparse.Namespace) -> int:
    """Backfill OHLCV for ``--symbols`` (or configured universe) over a date range."""
    from datetime import date, timedelta

    symbols = _ingest_symbols_from_settings(settings, getattr(args, "symbols", None))
    if not symbols:
        logger.error("No symbols to ingest (empty universe and DCA targets).")
        return 1

    end_d = date.fromisoformat(str(args.end).strip()) if args.end else datetime.now(UTC).date()
    if args.years is not None:
        start_d = end_d - timedelta(days=365 * int(args.years))
    else:
        start_d = date.fromisoformat(str(args.start).strip())

    if start_d > end_d:
        logger.error(
            "--start must be on or before --end (got %s and %s)",
            start_d.isoformat(),
            end_d.isoformat(),
        )
        return 1

    parquet = ParquetStore(parquet_dir(settings))
    sqlite = SQLiteStore(hub_sqlite_path(settings))
    adapter = YFinanceAdapter()
    fred_key = (settings.fred_api_key or "").strip()
    fred_adapter = FredAdapter(fred_key) if fred_key else None
    edgar_adapter = EdgarAdapter() if settings.fundamental.enabled else None
    dcfg = settings.data
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=parquet,
        sqlite_store=sqlite,
        fred_adapter=fred_adapter,
        edgar_adapter=edgar_adapter,
        ohlcv_outlier_zscore_threshold=dcfg.ohlcv_outlier_zscore_threshold,
        ohlcv_min_prior_for_zscore=dcfg.ohlcv_min_prior_for_zscore,
        ohlcv_max_abs_daily_return=dcfg.ohlcv_max_abs_daily_return,
        validation_exempt_symbols=dcfg.validation_exempt_symbols,
        initial_backfill_years=dcfg.initial_backfill_years,
    )

    failed = False
    for sym in symbols:
        res = pipeline.ingest_ohlcv(sym, start=start_d, end=end_d)
        if res.success:
            n = len(parquet.read_ohlcv(sym))
            logger.info("Ingest %s: OK (%s rows in parquet)", sym, n)
        else:
            failed = True
            logger.error("Ingest %s: FAILED (%s)", sym, res.error or res.validation)
    return 1 if failed else 0


def _run_ml_train_cli(
    settings: Settings,
    *,
    strategy_name: str,
    start: date,
    end: date,
) -> int:
    """Train experimental classifier from Parquet OHLCV; requires ``uv pip install '.[ml]'``."""
    from src.ml.train import train_momentum_model_bundle
    from src.strategy.momentum import MomentumRotationStrategy

    try:
        strat = build_strategy_for_backtest(settings, strategy_name)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    if not isinstance(strat, MomentumRotationStrategy):
        logger.error("ml-train supports --strategy momentum only.")
        return 1

    parquet = ParquetStore(parquet_dir(settings))
    symbols_all = strat.get_universe()
    vix_u = settings.regime.vix_symbol.strip().upper()
    train_syms = [s for s in symbols_all if s.strip().upper() != vix_u]

    data: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for sym in symbols_all:
        df = parquet.read_ohlcv(sym)
        if df.empty:
            missing.append(sym)
        else:
            data[sym] = df

    if missing:
        logger.error(
            "No Parquet OHLCV for: %s. Run ingest first.",
            ", ".join(missing),
        )
        return 1

    try:
        out = train_momentum_model_bundle(
            settings,
            data,
            symbols=train_syms,
            start=start,
            end=end,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    except ImportError:
        logger.error('Install ML extras: uv pip install ".[ml]"')
        return 1

    print(f"ML model saved to {out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: ``config`` | ``once`` | ``run`` | ``backtest``."""
    parser = argparse.ArgumentParser(
        prog="carmel",
        description="Carmel — data ingest, signals, and broker execution.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="Print non-secret configuration as JSON.")
    p_once = sub.add_parser("once", help="Run one full trading cycle and exit.")
    p_once.add_argument(
        "--as-of",
        default=None,
        metavar="ISO8601",
        help="Simulation timestamp (UTC), e.g. 2026-03-30T12:00:00+00:00",
    )
    sub.add_parser("run", help="Start the APScheduler daemon (ingest + signal crons + heartbeat).")
    p_serve = sub.add_parser("serve", help="Run FastAPI REST API (uvicorn).")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0).")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default 8000).")
    sub.add_parser(
        "index-rag",
        help="Rebuild document RAG chunks in SQLite (requires llm.enabled and running Ollama).",
    )
    sub.add_parser(
        "setup",
        help="Interactive first-time setup: configure keys, ingest data, run first cycle.",
    )
    sub.add_parser(
        "reconcile",
        help="Compare SQLite execution log and lot ledger to broker state (exit 1 if discrepancies).",
    )
    p_liquidate = sub.add_parser(
        "liquidate",
        help="Market-sell entire long position in SYMBOL via OrderManager (logs execution row).",
    )
    p_liquidate.add_argument(
        "symbol",
        metavar="SYMBOL",
        help="Ticker to liquidate (e.g. BIL).",
    )
    p_rebuild = sub.add_parser(
        "rebuild-lots",
        help=(
            "Rebuild tax_lots_* from trade_executions since a cutoff and rewrite equity_snapshots "
            "(backs up hub SQLite first)."
        ),
    )
    p_rebuild.add_argument(
        "--since",
        required=True,
        metavar="YYYY-MM-DD",
        help="Replay fills on/after this date (e.g. paper-account reset day).",
    )
    p_rebuild.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Last equity-snapshot date to rewrite (default: today UTC).",
    )
    p_rebuild.add_argument(
        "--account-id",
        default="default",
        help="Hub account_id partition (default: default).",
    )
    p_rebuild.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip copying hub_metadata.sqlite before mutating (not recommended).",
    )
    p_bf_cash = sub.add_parser(
        "backfill-snapshot-cash",
        help=(
            "Replay trade_executions cash flows onto equity_snapshots.cash "
            "(backs up hub SQLite first)."
        ),
    )
    p_bf_cash.add_argument(
        "--since",
        required=True,
        metavar="YYYY-MM-DD",
        help="Start date (e.g. paper-account reset day).",
    )
    p_bf_cash.add_argument(
        "--seed",
        type=float,
        required=True,
        metavar="USD",
        help="Starting cash at --since (e.g. 3336 after the 2026-05-16 reset).",
    )
    p_bf_cash.add_argument(
        "--account-id",
        default="default",
        help="Hub account_id partition (default: default).",
    )
    p_bf_cash.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip copying hub_metadata.sqlite before mutating (not recommended).",
    )
    p_bf_cash.add_argument(
        "--skip-broker-check",
        action="store_true",
        help="Do not compare final cash to the live broker (offline / tests).",
    )
    p_bf_fill = sub.add_parser(
        "backfill-fill-qty",
        help=(
            "Backfill trade_executions.filled_qty from Alpaca for submitted rows "
            "where filled_qty is NULL (backs up hub SQLite first unless --dry-run)."
        ),
    )
    p_bf_fill.add_argument(
        "--since",
        required=True,
        metavar="YYYY-MM-DD",
        help="Only rows with timestamp on/after this date (e.g. paper-account reset).",
    )
    p_bf_fill.add_argument(
        "--account-id",
        default="default",
        help="Hub account_id partition (default: default).",
    )
    p_bf_fill.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fill diff and write nothing (no backup).",
    )
    p_bf_fill.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip copying hub_metadata.sqlite before mutating (not recommended).",
    )
    sub.add_parser(
        "preflight",
        help="Validate configuration and broker connectivity before live trading.",
    )
    sub.add_parser(
        "notify-test",
        help="Send a test alert through configured notification channels (webhook/email).",
    )
    p_ingest = sub.add_parser("ingest", help="Backfill OHLCV history for symbols.")
    p_ingest.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated tickers (default: universe + DCA targets from settings).",
    )
    p_ingest.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date (default: today, UTC).",
    )
    ing = p_ingest.add_mutually_exclusive_group(required=True)
    ing.add_argument("--start", default=None, metavar="YYYY-MM-DD", help="Start date.")
    ing.add_argument(
        "--years",
        type=int,
        default=None,
        metavar="N",
        help="Backfill N years ending at --end (or today); mutually exclusive with --start.",
    )
    p_bt = sub.add_parser("backtest", help="Run a historical backtest using stored Parquet OHLCV.")
    p_bt.add_argument(
        "--strategy",
        required=True,
        choices=("momentum", "dca", "mean_reversion"),
        help="Strategy to simulate (momentum, dca, or mean_reversion).",
    )
    p_bt.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    p_bt.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    p_bt.add_argument(
        "--capital", type=float, default=10_000.0, help="Initial capital (default 10000)."
    )
    p_bt.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        dest="slippage_bps",
        help="Slippage in basis points (default 5).",
    )
    p_bt.add_argument(
        "--rebalance",
        default="monthly",
        choices=("daily", "weekly", "monthly"),
        help="Rebalance cadence for the backtester (default monthly).",
    )
    p_bt.add_argument(
        "--raw-signal",
        action="store_true",
        dest="raw_signal",
        help="Disable risk-layer sizing (100%% signal weight; not production-faithful).",
    )
    p_bt.add_argument(
        "--benchmark",
        default=None,
        metavar="SYMBOL",
        help="Buy-and-hold benchmark ticker (default: settings.backtest.benchmark_symbol).",
    )
    p_ml = sub.add_parser(
        "ml-train",
        help="Train experimental momentum classifier from Parquet (requires .[ml] extra).",
    )
    p_ml.add_argument(
        "--strategy",
        default="momentum",
        choices=("momentum",),
        help="Strategy universe for training (default momentum).",
    )
    p_ml.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    p_ml.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(log_rotation=settings.scheduler.log_rotation)

    if args.command == "config":
        print(json.dumps(public_settings_dict(settings), indent=2, default=str))
        return 0

    if args.command == "index-rag":
        from src.ai.llm_client import OllamaClient
        from src.ai.rag_index import index_docs

        store = SQLiteStore(hub_sqlite_path(settings))
        client = OllamaClient(settings.llm)
        n = index_docs(settings, store, client)
        print(f"Indexed {n} RAG chunk(s).", flush=True)
        return 0

    if args.command == "setup":
        from src.automation.setup_wizard import run_setup_wizard

        run_setup_wizard()
        return 0

    if args.command == "once":
        try:
            brokers = build_brokers(settings)
        except ConfigurationError as exc:
            logger.critical("Broker configuration failed: %s", exc)
            notify_operational_critical(
                settings,
                category="alpaca_credentials",
                message=str(exc),
            )
            return 1
        except ValueError as exc:
            msg = str(exc)
            if "Alpaca API credentials are missing" not in msg and "Missing credentials for broker" not in msg:
                raise
            logger.error("%s", exc)
            logger.error(
                "To fix: copy .env.example to .env, set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY (paper keys from https://app.alpaca.markets/), "
                "or configure broker.accounts with per-account env vars, "
                "or run: carmel setup",
            )
            return 1
        as_of = parse_as_of_iso(args.as_of)
        db_path = hub_sqlite_path(settings)
        sqlite = SQLiteStore(db_path)
        ledger = LotLedger(db_path)
        for name, br in brokers.items():
            logger.info("Running cycle for account: %s", name)
            wf = create_trading_workflow(
                settings,
                broker=br,
                sqlite_store=sqlite,
                lot_ledger=ledger,
            )
            wf.run_cycle(as_of=as_of)
        return 0

    if args.command == "run":
        try:
            brokers = build_brokers(settings)
        except ConfigurationError as exc:
            logger.critical("Broker configuration failed: %s", exc)
            notify_operational_critical(
                settings,
                category="alpaca_credentials",
                message=str(exc),
            )
            return 1
        except ValueError as exc:
            msg = str(exc)
            if "Alpaca API credentials are missing" not in msg and "Missing credentials for broker" not in msg:
                raise
            logger.error("%s", exc)
            logger.error(
                "To fix: copy .env.example to .env, set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY (paper keys from https://app.alpaca.markets/), "
                "or configure broker.accounts with per-account env vars, "
                "or run: carmel setup",
            )
            return 1
        _run_daemon(settings, brokers)
        return 0

    if args.command == "serve":
        import uvicorn

        from src.api.app import create_app

        uvicorn.run(create_app(), host=str(args.host), port=int(args.port))
        return 0

    if args.command == "reconcile":
        try:
            return _run_reconcile_cli(settings)
        except ConfigurationError as exc:
            logger.critical("Broker configuration failed: %s", exc)
            notify_operational_critical(
                settings,
                category="alpaca_credentials",
                message=str(exc),
            )
            return 1

    if args.command == "liquidate":
        try:
            return _run_liquidate_cli(settings, str(args.symbol))
        except ConfigurationError as exc:
            logger.critical("Broker configuration failed: %s", exc)
            notify_operational_critical(
                settings,
                category="alpaca_credentials",
                message=str(exc),
            )
            return 1

    if args.command == "rebuild-lots":
        try:
            since_d = date.fromisoformat(str(args.since).strip())
            end_d = date.fromisoformat(str(args.end).strip()) if args.end else None
            return _run_rebuild_lots_cli(
                settings,
                since=since_d,
                account_id=str(args.account_id),
                no_backup=bool(args.no_backup),
                end=end_d,
            )
        except ValueError as exc:
            logger.error("rebuild-lots invalid args: %s", exc)
            return 1
        except (OSError, RuntimeError) as exc:
            logger.exception("rebuild-lots failed: %s", exc)
            return 1

    if args.command == "backfill-snapshot-cash":
        try:
            since_d = date.fromisoformat(str(args.since).strip())
            return _run_backfill_snapshot_cash_cli(
                settings,
                since=since_d,
                seed=float(args.seed),
                account_id=str(args.account_id),
                no_backup=bool(args.no_backup),
                skip_broker_check=bool(args.skip_broker_check),
            )
        except ValueError as exc:
            logger.error("backfill-snapshot-cash invalid args: %s", exc)
            return 1
        except (OSError, RuntimeError) as exc:
            logger.exception("backfill-snapshot-cash failed: %s", exc)
            return 1

    if args.command == "backfill-fill-qty":
        try:
            since_d = date.fromisoformat(str(args.since).strip())
            return _run_backfill_fill_qty_cli(
                settings,
                since=since_d,
                account_id=str(args.account_id),
                dry_run=bool(args.dry_run),
                no_backup=bool(args.no_backup),
            )
        except ValueError as exc:
            logger.error("backfill-fill-qty invalid args: %s", exc)
            return 1
        except (OSError, RuntimeError) as exc:
            logger.exception("backfill-fill-qty failed: %s", exc)
            return 1

    if args.command == "preflight":
        from src.automation.preflight import preflight_report_as_text, run_preflight_checks

        try:
            broker = broker_from_settings(settings)
        except ConfigurationError as exc:
            logger.critical("Broker configuration failed: %s", exc)
            notify_operational_critical(
                settings,
                category="alpaca_credentials",
                message=str(exc),
            )
            return 1
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        store = SQLiteStore(hub_sqlite_path(settings))
        pq = ParquetStore(parquet_dir(settings))
        report = run_preflight_checks(settings, broker, sqlite_store=store, parquet_store=pq)
        print(preflight_report_as_text(report), flush=True)
        return 1 if report.failed else 0

    if args.command == "notify-test":
        from src.automation.notify_test import run_notify_test

        return run_notify_test(settings)

    if args.command == "ingest":
        return _run_ingest_cli(settings, args)

    if args.command == "backtest":
        start = date.fromisoformat(str(args.start).strip())
        end = date.fromisoformat(str(args.end).strip())
        return _run_backtest_cli(
            settings,
            strategy_name=str(args.strategy),
            start=start,
            end=end,
            capital=float(args.capital),
            slippage_bps=float(args.slippage_bps),
            rebalance=str(args.rebalance),
            raw_signal=bool(getattr(args, "raw_signal", False)),
            benchmark_symbol=getattr(args, "benchmark", None),
        )

    if args.command == "ml-train":
        start = date.fromisoformat(str(args.start).strip())
        end = date.fromisoformat(str(args.end).strip())
        return _run_ml_train_cli(
            settings,
            strategy_name=str(args.strategy),
            start=start,
            end=end,
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
