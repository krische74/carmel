"""Pre-flight validation before switching from paper to live trading."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Settings
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore
    from src.execution.broker_interface import BrokerInterface

logger = logging.getLogger(__name__)


class PreflightStatus(StrEnum):
    """Outcome for one preflight check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class PreflightCheck(BaseModel):
    """Single validation row."""

    name: str
    status: PreflightStatus
    detail: str = ""


class PreflightReport(BaseModel):
    """Aggregate preflight outcome."""

    checks: list[PreflightCheck] = Field(default_factory=list)
    failed: bool = False
    warned: bool = False


def run_preflight_checks(
    settings: Settings,
    broker: BrokerInterface,
    *,
    sqlite_store: SQLiteStore | None = None,
    parquet_store: ParquetStore | None = None,
) -> PreflightReport:
    """Validate configuration and broker connectivity before live trading."""
    from src.data.symbol_resolver import resolve_required_symbols

    checks: list[PreflightCheck] = []

    def add(name: str, status: PreflightStatus, detail: str = "") -> None:
        checks.append(PreflightCheck(name=name, status=status, detail=detail))

    connected = False
    eq = 0.0
    try:
        broker.refresh_account()
        eq = float(broker.get_account_equity())
        connected = True
        add("alpaca_connectivity", PreflightStatus.PASS, "Broker reachable")
    except (ConnectionError, OSError, TimeoutError, ValueError, RuntimeError) as exc:
        add("alpaca_connectivity", PreflightStatus.FAIL, str(exc))

    if connected:
        if eq <= 0.0:
            add("account_funded", PreflightStatus.FAIL, "Account equity is zero or negative")
        else:
            add("account_funded", PreflightStatus.PASS, f"equity=${eq:,.2f}")
    else:
        add("account_funded", PreflightStatus.FAIL, "Skipped after connectivity failure")

    # 3 Kill switch
    lim = float(settings.risk.daily_loss_limit_pct)
    if lim >= 1.0:
        add(
            "kill_switch_limit",
            PreflightStatus.FAIL,
            f"daily_loss_limit_pct={lim} is effectively disabled (>= 100%)",
        )
    elif lim <= 0.0:
        add("kill_switch_limit", PreflightStatus.WARN, "daily_loss_limit_pct is zero or negative")
    else:
        add("kill_switch_limit", PreflightStatus.PASS, f"daily_loss_limit_pct={lim:.4f}")

    # 4 Strategies
    if not settings.strategy.enabled:
        add("strategies_enabled", PreflightStatus.FAIL, "strategy.enabled is empty")
    else:
        add(
            "strategies_enabled",
            PreflightStatus.PASS,
            ", ".join(settings.strategy.enabled),
        )

    # 5 Paper flag alignment
    paper_env = bool(settings.alpaca_paper)
    paper_broker = bool(settings.broker.paper_trading)
    if paper_env != paper_broker:
        add(
            "paper_flag_consistency",
            PreflightStatus.WARN,
            f"ALPACA_PAPER={paper_env} vs broker.paper_trading={paper_broker}",
        )
    else:
        add(
            "paper_flag_consistency",
            PreflightStatus.PASS,
            f"paper={paper_broker}",
        )

    # 6 PDT for sub-25k
    if not connected:
        add("pdt_protection", PreflightStatus.WARN, "Skipped — broker equity unknown")
    elif eq < float(settings.risk.pdt_equity_floor) and not settings.risk.pdt_protection:
        add(
            "pdt_protection",
            PreflightStatus.FAIL,
            f"Equity ${eq:,.0f} < ${settings.risk.pdt_equity_floor:,.0f} but pdt_protection is off",
        )
    else:
        add(
            "pdt_protection",
            PreflightStatus.PASS,
            "enabled" if settings.risk.pdt_protection else "not required (protection off)",
        )

    # 7 Notifications
    ncfg = settings.notification
    webhook_enabled = ncfg.enabled
    webhook_url = (ncfg.webhook_url or "").strip()
    email_enabled = ncfg.email_enabled

    if webhook_enabled and not webhook_url:
        add(
            "notification_channel",
            PreflightStatus.FAIL,
            "notification.enabled=true but webhook_url is empty (set NOTIFICATION__WEBHOOK_URL)",
        )
    elif not webhook_enabled and not email_enabled:
        add(
            "notification_channel",
            PreflightStatus.WARN,
            "No webhook (enabled+URL) and no email_enabled — alerts will only go to logs",
        )
    else:
        add(
            "notification_channel",
            PreflightStatus.PASS,
            f"webhook={webhook_enabled and bool(webhook_url)}, email={email_enabled}",
        )

    # 7b Cash sweep vehicle must stay outside the momentum universe (rotation safety).
    if settings.cash_sweep.enabled:
        from src.config import momentum_universe_symbols

        sweep_sym = settings.cash_sweep.symbol.strip().upper()
        mom_uni = momentum_universe_symbols(settings)
        if sweep_sym in mom_uni:
            add(
                "cash_sweep_symbol",
                PreflightStatus.FAIL,
                f"cash_sweep.symbol {sweep_sym} is in the momentum universe {sorted(mom_uni)}; "
                "rotation would liquidate the swept position. Use a symbol outside it (e.g. BIL).",
            )
        else:
            add(
                "cash_sweep_symbol",
                PreflightStatus.PASS,
                f"sweep vehicle {sweep_sym} is outside the momentum universe",
            )

    # 8 Recent backtest (warn)
    store = sqlite_store
    recent = False
    if store is not None:
        try:
            runs = store.get_backtest_runs(limit=20, offset=0)
            cutoff = datetime.now(UTC) - timedelta(days=30)
            for r in runs:
                raw = r.get("run_at")
                if not raw:
                    continue
                try:
                    ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= cutoff:
                    recent = True
                    break
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Preflight backtest scan failed: %s", exc)
    if not recent:
        add(
            "recent_backtest",
            PreflightStatus.WARN,
            "No backtest run recorded in SQLite in the last 30 days",
        )
    else:
        add("recent_backtest", PreflightStatus.PASS, "Found a run within 30 days")

    # 9 Required OHLCV symbols present in Parquet (optional caller supplies store)
    if parquet_store is not None:
        required = sorted(resolve_required_symbols(settings))
        missing = [s for s in required if not parquet_store.has_ohlcv_file(s)]
        if missing:
            miss_csv = ",".join(missing)
            add(
                "ohlcv_symbol_coverage",
                PreflightStatus.WARN,
                f"Missing Parquet OHLCV for {len(missing)} required symbol(s): {miss_csv}. "
                f"Run: carmel ingest --symbols {miss_csv} --years 2",
            )
        else:
            add(
                "ohlcv_symbol_coverage",
                PreflightStatus.PASS,
                f"{len(required)} required symbol(s) have Parquet data",
            )
            short_hist: list[str] = []
            for sym in required:
                df = parquet_store.read_ohlcv(sym)
                if df.empty:
                    continue
                idx = df.index
                if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
                    continue
                earliest = pd.Timestamp(idx.min()).normalize()
                latest = pd.Timestamp(idx.max()).normalize()
                days_of_history = (latest - earliest).days
                if days_of_history < 365:
                    short_hist.append(f"{sym} ({days_of_history}d)")
            if short_hist:
                add(
                    "ohlcv_backtest_history",
                    PreflightStatus.WARN,
                    "Symbol(s) have less than 365 calendar days of OHLCV history: "
                    + ", ".join(short_hist)
                    + ". Momentum and long lookbacks need more data. "
                    "Run: carmel ingest --years 3",
                )
            else:
                add(
                    "ohlcv_backtest_history",
                    PreflightStatus.PASS,
                    "Required symbols have at least 365 calendar days of OHLCV span",
                )

    failed = any(c.status == PreflightStatus.FAIL for c in checks)
    warned = any(c.status == PreflightStatus.WARN for c in checks)
    return PreflightReport(checks=checks, failed=failed, warned=warned)


def preflight_report_as_text(report: PreflightReport) -> str:
    """Plain-text rendering for CLI output."""
    lines = []
    for c in report.checks:
        lines.append(f"[{c.status.upper()}] {c.name}: {c.detail}".strip())
    if report.failed:
        lines.append("\nRESULT: FAIL — fix failing checks before live trading.")
    elif report.warned:
        lines.append("\nRESULT: PASS with warnings — review before live trading.")
    else:
        lines.append("\nRESULT: PASS")
    return "\n".join(lines)
