"""Tests for APScheduler wiring (no real clock)."""

from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.cron import CronTrigger

from src.automation.scheduler import (
    HubScheduler,
    _translate_unix_day_of_week,
    cron_trigger_from_expression,
)
from src.config import Settings


def test_cron_trigger_from_expression_parses_five_field_crontab() -> None:
    trig = cron_trigger_from_expression("0 17 * * 1-5")
    s = str(trig)
    assert "hour='17'" in s
    assert "minute='0'" in s


# 2026-05-09 is a Saturday; 5/10 Sun; 5/11 Mon; ... 5/15 Fri; 5/16 Sat.
# These dates ground the day-of-week assertions below in real calendar days.
_UTC = ZoneInfo("UTC")
_SAT_2026_05_09 = datetime(2026, 5, 9, 0, 0, tzinfo=_UTC)
_SUN_2026_05_10 = datetime(2026, 5, 10, 0, 0, tzinfo=_UTC)
_MON_2026_05_11 = datetime(2026, 5, 11, 0, 0, tzinfo=_UTC)
_FRI_2026_05_15 = datetime(2026, 5, 15, 0, 0, tzinfo=_UTC)
_SAT_2026_05_16 = datetime(2026, 5, 16, 0, 0, tzinfo=_UTC)


def _utc_trigger(expr: str) -> CronTrigger:
    """Build a trigger via the same translation path but pinned to UTC for tests.

    ``cron_trigger_from_expression`` doesn't take a timezone (the scheduler
    supplies one at registration). For deterministic weekday assertions
    independent of the host's local zone, rebuild here in UTC.
    """
    parts = expr.strip().split()
    assert len(parts) == 5, expr
    parts[4] = _translate_unix_day_of_week(parts[4])
    return CronTrigger.from_crontab(" ".join(parts), timezone=_UTC)


def _next_fire_weekdays(expr: str, start: datetime, count: int) -> list[int]:
    """Return ``count`` consecutive next-fire weekdays (0=Mon..6=Sun) from ``start``."""
    trig = _utc_trigger(expr)
    out: list[int] = []
    cursor = start
    for _ in range(count):
        nxt = trig.get_next_fire_time(None, cursor)
        assert nxt is not None
        out.append(nxt.weekday())
        # Advance one minute past the fire so the next get_next_fire_time
        # returns the *following* firing rather than the same one.
        cursor = nxt.astimezone(_UTC).replace(second=0, microsecond=0)
        cursor = cursor.replace(minute=(cursor.minute + 1) % 60)
        if cursor.minute == 0:
            cursor = cursor.replace(hour=(cursor.hour + 1) % 24)
    return out


def test_unix_cron_dow_1_to_5_fires_mon_through_fri_not_tue_through_sat() -> None:
    # Regression for the APScheduler day_of_week footgun: "1-5" in Unix cron
    # is Mon-Fri; APScheduler's native numeric dow would read it as Tue-Sat.
    # Empirically observed in carmel.log (5/9 Sat fired, 5/11 Mon skipped).
    weekdays = _next_fire_weekdays("30 17 * * 1-5", _SAT_2026_05_09, 5)
    # Starting Saturday morning, the next five firings must be Mon-Fri.
    assert weekdays == [0, 1, 2, 3, 4]


def test_unix_cron_dow_5_fires_friday_only_not_saturday() -> None:
    # rebalance_check_cron uses "0 18 * * 5" intending Friday only.
    weekdays = _next_fire_weekdays("0 18 * * 5", _SAT_2026_05_09, 3)
    assert weekdays == [4, 4, 4]


def test_unix_cron_dow_0_and_7_both_mean_sunday() -> None:
    # Both 0 and 7 are Sunday in standard crontab; APScheduler numeric would
    # treat 0 as Monday and reject 7. The translation must map both to 'sun'.
    sun_via_0 = _next_fire_weekdays("0 2 * * 0", _MON_2026_05_11, 2)
    sun_via_7 = _next_fire_weekdays("0 2 * * 7", _MON_2026_05_11, 2)
    assert sun_via_0 == [6, 6]
    assert sun_via_7 == [6, 6]


def test_unix_cron_dow_named_days_pass_through_unchanged() -> None:
    # Name-based dow ("mon-fri") is already unambiguous in APScheduler;
    # the translator must leave it alone.
    weekdays = _next_fire_weekdays("30 17 * * mon-fri", _SAT_2026_05_09, 5)
    assert weekdays == [0, 1, 2, 3, 4]


def test_unix_cron_dow_list_fires_each_listed_day() -> None:
    # Mon and Fri only — exercises comma-list translation.
    weekdays = _next_fire_weekdays("0 12 * * 1,5", _SAT_2026_05_09, 4)
    assert weekdays == [0, 4, 0, 4]


def test_cron_expression_with_wrong_field_count_raises() -> None:
    with pytest.raises(ValueError, match="5 fields"):
        cron_trigger_from_expression("0 17 * * 1-5 extra")


def test_hub_scheduler_registers_data_and_signal_jobs() -> None:
    settings = Settings(_yaml_path=None, _env_file=None)
    hub = HubScheduler(settings=settings)
    data_fn = MagicMock()
    signal_fn = MagicMock()
    hub.register_trading_jobs(run_data_ingestion=data_fn, run_signal_cycle=signal_fn)
    jobs = hub.scheduler.get_jobs()
    assert len(jobs) == 3
    ids = {j.id for j in jobs}
    assert ids == {"data_ingestion", "signal_generation", "rebalance_check"}
