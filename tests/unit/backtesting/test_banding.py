"""Tier 53D: asymmetric banding — absolute cash exit carries no hysteresis."""

from __future__ import annotations

from src.strategy.momentum_banding import select_with_asymmetric_band


def test_absolute_exit_to_cash_is_unbanded() -> None:
    """Even with a large band_k, empty eligible list liquidates to cash immediately."""
    out = select_with_asymmetric_band(
        ranked_risk=[],
        scores={"SPY": 0.1, "QQQ": 0.05},
        incumbent_risk="SPY",
        cash_symbol="SHV",
        band_k=10.0,
        top_n=1,
    )
    assert out == [("SHV", 1.0)]


def test_relative_switch_requires_band() -> None:
    scores = {"SPY": 0.10, "QQQ": 0.11, "TLT": 0.00, "GLD": -0.01}
    # sigma of scores ~ nontrivial; QQQ barely ahead of SPY — may fail band at k=1
    hold = select_with_asymmetric_band(
        ranked_risk=["QQQ", "SPY", "TLT", "GLD"],
        scores=scores,
        incumbent_risk="SPY",
        cash_symbol="SHV",
        band_k=5.0,
        top_n=1,
    )
    assert hold[0][0] == "SPY"

    switch = select_with_asymmetric_band(
        ranked_risk=["QQQ", "SPY", "TLT", "GLD"],
        scores=scores,
        incumbent_risk="SPY",
        cash_symbol="SHV",
        band_k=0.0,
        top_n=1,
    )
    assert switch[0][0] == "QQQ"


def test_reentry_from_cash_is_unbanded() -> None:
    out = select_with_asymmetric_band(
        ranked_risk=["QQQ", "SPY"],
        scores={"QQQ": 0.01, "SPY": 0.009},
        incumbent_risk=None,
        cash_symbol="SHV",
        band_k=100.0,
        top_n=1,
    )
    assert out[0][0] == "QQQ"


def test_top_two_equal_weight() -> None:
    out = select_with_asymmetric_band(
        ranked_risk=["QQQ", "SPY", "TLT"],
        scores={"QQQ": 0.2, "SPY": 0.1, "TLT": 0.0},
        incumbent_risk=None,
        cash_symbol="SHV",
        band_k=None,
        top_n=2,
    )
    assert out == [("QQQ", 0.5), ("SPY", 0.5)]
