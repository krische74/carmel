"""Tests for kill switch (daily loss halt)."""

from src.risk.kill_switch import KillSwitch


def test_kill_switch_halts_on_daily_loss_limit() -> None:
    ks = KillSwitch(daily_loss_limit_pct=0.03)
    assert ks.is_halted(daily_pnl_pct=-0.04)
    assert not ks.is_halted(daily_pnl_pct=-0.02)


def test_kill_switch_reset() -> None:
    ks = KillSwitch(daily_loss_limit_pct=0.03)
    ks.arm_from_daily_pnl(-0.05)
    assert ks.halted
    ks.reset()
    assert not ks.halted
