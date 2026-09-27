"""Tier 32: capital loss carryforward (informational, not tax advice)."""

from __future__ import annotations

import pytest

from src.reporting.tax_report import LossCarryforward, TaxSummary, compute_loss_carryforward


def _ts(
    year: int,
    *,
    st_net: float,
    lt_net: float,
) -> TaxSummary:
    st_g = max(0.0, st_net)
    st_l = min(0.0, st_net)
    lt_g = max(0.0, lt_net)
    lt_l = min(0.0, lt_net)
    return TaxSummary(
        year=year,
        short_term_gains=st_g,
        short_term_losses=st_l,
        short_term_net=st_net,
        long_term_gains=lt_g,
        long_term_losses=lt_l,
        long_term_net=lt_net,
        total_net=st_net + lt_net,
        lots=[],
    )


def test_carryforward_no_losses() -> None:
    s = _ts(2026, st_net=5_000.0, lt_net=3_000.0)
    out = compute_loss_carryforward(s, None, account_id="default")
    assert out.short_term_carryforward == 0.0
    assert out.long_term_carryforward == 0.0


def test_carryforward_losses_exceed_gains() -> None:
    """10k ST loss and 5k ST gain → net -5k; 3k ordinary absorption → 2k carries forward."""
    s = _ts(2026, st_net=-5_000.0, lt_net=0.0)
    out = compute_loss_carryforward(s, None, annual_deduction_limit=3_000.0, account_id="default")
    assert out.short_term_carryforward == pytest.approx(2_000.0)
    assert out.long_term_carryforward == 0.0


def test_carryforward_with_prior_year() -> None:
    """Prior ST carryforward stacks with current ST loss; 3k rule absorbs combined net loss."""
    s = _ts(2026, st_net=-1_000.0, lt_net=0.0)
    prior = LossCarryforward(
        year=2025,
        account_id="default",
        short_term_carryforward=2_000.0,
        long_term_carryforward=0.0,
        computed_at="",
    )
    out = compute_loss_carryforward(s, prior, annual_deduction_limit=3_000.0, account_id="default")
    assert out.total_carryforward == pytest.approx(0.0)


def test_carryforward_cross_term_netting() -> None:
    """ST loss offsets LT gain after same-term pools; net -2k then full ordinary offset."""
    s = _ts(2026, st_net=-5_000.0, lt_net=3_000.0)
    out = compute_loss_carryforward(s, None, annual_deduction_limit=3_000.0, account_id="default")
    assert out.short_term_carryforward == pytest.approx(0.0)
    assert out.long_term_carryforward == 0.0


def test_carryforward_per_account_isolation_in_model() -> None:
    """LossCarryforward carries account_id for hub partitioning."""
    out = compute_loss_carryforward(_ts(2026, st_net=0.0, lt_net=0.0), None, account_id="Main")
    assert out.account_id == "Main"


def test_carryforward_per_account_sqlite_isolation(tmp_path) -> None:
    """SQLite rows partition carryforward by hub ``account_id``."""
    from src.data.storage.sqlite_store import SQLiteStore

    store = SQLiteStore(tmp_path / "cf.sqlite")
    store.upsert_loss_carryforward(
        2025,
        account_id="Main",
        short_term_carryforward=100.0,
        long_term_carryforward=0.0,
    )
    store.upsert_loss_carryforward(
        2025,
        account_id="IRA",
        short_term_carryforward=250.0,
        long_term_carryforward=0.0,
    )
    a = store.get_loss_carryforward_row(2025, account_id="Main")
    b = store.get_loss_carryforward_row(2025, account_id="IRA")
    assert a is not None and float(a["short_term_carryforward"]) == pytest.approx(100.0)
    assert b is not None and float(b["short_term_carryforward"]) == pytest.approx(250.0)
