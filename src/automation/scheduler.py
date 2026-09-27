"""APScheduler wiring for data ingest, signal generation, and rebalance checks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.config import Settings

logger = logging.getLogger(__name__)


def scheduler_timezone_from_settings(settings: Settings) -> ZoneInfo:
    """Resolve ``settings.scheduler.timezone`` to :class:`zoneinfo.ZoneInfo`, or UTC on failure."""
    name = (settings.scheduler.timezone or "UTC").strip() or "UTC"
    try:
        return ZoneInfo(name)
    except Exception:
        logger.warning(
            "ZoneInfo failed for scheduler.timezone %r; using UTC",
            name,
        )
        return ZoneInfo("UTC")


# Unix crontab uses 0=Sun..6=Sat (with 7 as an alias for Sun); APScheduler's
# numeric day_of_week is 0=Mon..6=Sun, and CronTrigger.from_crontab does NOT
# translate between them. Rewriting numeric tokens to day names is the only
# unambiguous way to preserve Unix semantics in config like "30 17 * * 1-5".
_UNIX_DOW_TO_NAME = {
    "0": "sun", "1": "mon", "2": "tue", "3": "wed",
    "4": "thu", "5": "fri", "6": "sat", "7": "sun",
}


def _translate_unix_day_of_week(field: str) -> str:
    """Rewrite numeric Unix-cron dow tokens to APScheduler-safe day names."""
    def xlate(tok: str) -> str:
        tok = tok.strip()
        return _UNIX_DOW_TO_NAME.get(tok, tok)

    parts: list[str] = []
    for chunk in field.split(","):
        head, _, step = chunk.partition("/")
        head = "-".join(xlate(b) for b in head.split("-")) if "-" in head else xlate(head)
        parts.append(f"{head}/{step}" if step else head)
    return ",".join(parts)


def cron_trigger_from_expression(expr: str) -> CronTrigger:
    """Parse a five-field cron string (``min hour dom mon dow``) into a trigger.

    Unix-cron day-of-week (0=Sun, 7=Sun) is preserved by rewriting numeric
    tokens to day names — APScheduler's native numbering is 0=Mon and its
    ``from_crontab`` does not translate, so a literal ``1-5`` would otherwise
    fire Tue-Sat instead of Mon-Fri.
    """
    parts = expr.strip().split()
    if len(parts) != 5:
        msg = f"cron expression must have 5 fields (min hour dom mon dow); got {len(parts)}: {expr!r}"
        raise ValueError(msg)
    parts[4] = _translate_unix_day_of_week(parts[4])
    return CronTrigger.from_crontab(" ".join(parts))


class HubScheduler:
    """Registers cron jobs from ``Settings.scheduler`` against callables."""

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings
        tz = scheduler_timezone_from_settings(settings)
        self._scheduler = BackgroundScheduler(timezone=tz)

    @property
    def scheduler(self) -> BackgroundScheduler:
        """Underlying APScheduler instance (start/shutdown from the app entrypoint)."""
        return self._scheduler

    def register_trading_jobs(
        self,
        *,
        run_data_ingestion: Callable[[], None],
        run_signal_cycle: Callable[[], None],
        run_rebalance_check: Callable[[], None] | None = None,
    ) -> None:
        """Attach ingest, signal, and optional rebalance jobs using configured crons."""
        cfg = self._settings.scheduler
        rebalance_fn = run_rebalance_check or (lambda: None)

        self._scheduler.add_job(
            run_data_ingestion,
            cron_trigger_from_expression(cfg.data_ingestion_cron),
            id="data_ingestion",
            replace_existing=True,
        )
        self._scheduler.add_job(
            run_signal_cycle,
            cron_trigger_from_expression(cfg.signal_generation_cron),
            id="signal_generation",
            replace_existing=True,
        )
        self._scheduler.add_job(
            rebalance_fn,
            cron_trigger_from_expression(cfg.rebalance_check_cron),
            id="rebalance_check",
            replace_existing=True,
        )
        logger.info(
            "Registered trading jobs: data=%s signal=%s rebalance=%s",
            cfg.data_ingestion_cron,
            cfg.signal_generation_cron,
            cfg.rebalance_check_cron,
        )
