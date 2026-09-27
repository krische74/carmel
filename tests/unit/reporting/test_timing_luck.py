"""Unit tests for Tier 54C timing-luck metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.timing_luck import compute_timing_luck_metrics, timing_luck_headlines


def test_identical_tranches_have_zero_te_and_spread() -> None:
    idx = [f"2024-01-{d:02d}" for d in range(2, 12)]
    ser = pd.Series([0.001] * len(idx), index=idx)
    tranches = [ser.copy() for _ in range(5)]
    m = compute_timing_luck_metrics(tranches, tranche_cagrs_pct=[5.0] * 5, years=2.0)
    assert m["tranche_cagr_spread_pp"] == 0.0
    assert m["mean_timing_luck_te_annual_pct"] == 0.0
    assert m["observed_vs_predicted_sd_ratio"] == 0.0
    assert m["observed_sd_below_predicted"] is True


def test_baseline_arithmetic_matches_matched_units() -> None:
    """Regression: 54C baseline TE/years/CAGR SD ratio (Keith verified 2026-08-23)."""
    years = 14.97
    mean_te = 4.5106
    cagrs = [3.3564, 4.0211, 4.7667, 4.1629, 4.8018] * 4  # 20 tranches, 5 unique
    h = timing_luck_headlines(
        mean_te_annual_pct=mean_te,
        tranche_cagrs_pct=cagrs,
        years=years,
    )
    assert h["predicted_tranche_cagr_std_pp"] == pytest.approx(1.166, abs=0.01)
    assert h["observed_tranche_cagr_std_pp"] == pytest.approx(0.548, abs=0.01)
    assert h["observed_vs_predicted_sd_ratio"] == pytest.approx(0.47, abs=0.02)
    assert h["timing_luck_ci_95_cagr_pp"] == pytest.approx(1.96 * 1.166, abs=0.05)
    assert h["observed_sd_below_predicted"] is True


def test_observed_sd_below_predicted_when_ratio_under_one() -> None:
    h = timing_luck_headlines(
        mean_te_annual_pct=5.0,
        tranche_cagrs_pct=[4.0, 4.3, 4.1, 4.2],
        years=10.0,
    )
    assert h["observed_vs_predicted_sd_ratio"] < 1.0
    assert h["observed_sd_below_predicted"] is True


def test_large_schedule_divergence_can_exceed_predicted_sd() -> None:
    h = timing_luck_headlines(
        mean_te_annual_pct=2.0,
        tranche_cagrs_pct=[2.0, 8.0],
        years=5.0,
    )
    assert h["tranche_cagr_spread_pp"] == 6.0
    assert h["observed_sd_below_predicted"] is False
