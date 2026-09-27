"""Unit tests for cycle alert evaluation."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.automation.alerts import AlertLevel, check_heartbeat_staleness, evaluate_cycle_alerts


def test_evaluate_alerts_kill_switch_returns_critical() -> None:
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=True,
        daily_pnl_pct=-0.01,
        daily_loss_limit_pct=0.03,
        ingest_failures=[],
        rejected_orders=[],
    )
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.CRITICAL
    assert alerts[0].category == "kill_switch"


def test_evaluate_alerts_approaching_loss_limit_returns_warning() -> None:
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=False,
        daily_pnl_pct=-0.024,
        daily_loss_limit_pct=0.03,
        ingest_failures=[],
        rejected_orders=[],
    )
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.WARNING
    assert alerts[0].category == "daily_loss_warning"


def test_evaluate_alerts_ingest_failure_returns_warning() -> None:
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=False,
        daily_pnl_pct=0.0,
        daily_loss_limit_pct=0.03,
        ingest_failures=["SPY"],
        rejected_orders=[],
    )
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.WARNING
    assert alerts[0].category == "ingest_failure"
    assert "SPY" in alerts[0].message


def test_evaluate_alerts_rejected_orders_returns_info() -> None:
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=False,
        daily_pnl_pct=0.0,
        daily_loss_limit_pct=0.03,
        ingest_failures=[],
        rejected_orders=[("QQQ", None)],
    )
    assert len(alerts) == 1
    assert alerts[0].level == AlertLevel.INFO
    assert alerts[0].category == "order_rejected"
    assert "QQQ" in alerts[0].message


def test_evaluate_alerts_rejected_orders_includes_reason() -> None:
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=False,
        daily_pnl_pct=0.0,
        daily_loss_limit_pct=0.03,
        ingest_failures=[],
        rejected_orders=[("QQQ", "min_cash_reserve_pct")],
    )
    assert len(alerts) == 1
    assert alerts[0].category == "order_rejected"
    assert "QQQ" in alerts[0].message
    assert "min_cash_reserve_pct" in alerts[0].message


def test_evaluate_alerts_clean_cycle_returns_empty() -> None:
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=False,
        daily_pnl_pct=0.0,
        daily_loss_limit_pct=0.03,
        ingest_failures=[],
        rejected_orders=[],
    )
    assert alerts == []


def test_check_heartbeat_staleness_returns_none_when_fresh(tmp_path: Path) -> None:
    p = tmp_path / "hub.heartbeat"
    p.write_text(datetime.now(UTC).isoformat() + "\n", encoding="utf-8")
    assert check_heartbeat_staleness(p, max_age_seconds=300) is None


def test_check_heartbeat_staleness_returns_alert_when_stale(tmp_path: Path) -> None:
    old = datetime.now(UTC) - timedelta(seconds=600)
    p = tmp_path / "hub.heartbeat"
    p.write_text(old.isoformat() + "\n", encoding="utf-8")
    a = check_heartbeat_staleness(p, max_age_seconds=300)
    assert a is not None
    assert a.level == AlertLevel.CRITICAL
    assert a.category == "heartbeat_stale"


def test_check_heartbeat_staleness_returns_alert_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "missing.heartbeat"
    a = check_heartbeat_staleness(p, max_age_seconds=300)
    assert a is not None
    assert a.level == AlertLevel.CRITICAL
    assert a.category == "heartbeat_stale"


def test_alert_timestamp_is_utc() -> None:
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=True,
        daily_pnl_pct=0.0,
        daily_loss_limit_pct=0.03,
        ingest_failures=[],
        rejected_orders=[],
    )
    assert alerts[0].timestamp.tzinfo == UTC


def test_cycle_alerts_include_regime_critical_when_crisis() -> None:
    from datetime import UTC, datetime

    from src.data.regime import (
        MarketRegime,
        OverallRegime,
        VolatilityRegime,
        YieldCurveRegime,
    )

    mr = MarketRegime(
        timestamp=datetime(2026, 4, 1, 16, 0, tzinfo=UTC),
        vix_close=40.0,
        yield_spread=1.5,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.CRISIS,
        overall=OverallRegime.CRISIS,
        sizing_multiplier=0.25,
    )
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=False,
        daily_pnl_pct=0.0,
        daily_loss_limit_pct=0.03,
        ingest_failures=[],
        rejected_orders=[],
        market_regime=mr,
    )
    reg = [a for a in alerts if a.category == "market_regime"]
    assert len(reg) == 1
    assert reg[0].level == AlertLevel.CRITICAL
    assert "crisis" in reg[0].message.lower()
    assert "25%" in reg[0].message


def test_cycle_alerts_include_regime_warning_when_defensive() -> None:
    from datetime import UTC, datetime

    from src.data.regime import (
        MarketRegime,
        OverallRegime,
        VolatilityRegime,
        YieldCurveRegime,
    )

    mr = MarketRegime(
        timestamp=datetime(2026, 4, 1, 16, 0, tzinfo=UTC),
        vix_close=22.0,
        yield_spread=-0.6,
        yield_curve=YieldCurveRegime.INVERTED,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.DEFENSIVE,
        sizing_multiplier=0.5,
    )
    alerts = evaluate_cycle_alerts(
        kill_switch_halted=False,
        daily_pnl_pct=0.0,
        daily_loss_limit_pct=0.03,
        ingest_failures=[],
        rejected_orders=[],
        market_regime=mr,
    )
    reg = [a for a in alerts if a.category == "market_regime"]
    assert len(reg) == 1
    assert reg[0].level.value == "warning"
    assert "defensive" in reg[0].message.lower()
    assert "50%" in reg[0].message
