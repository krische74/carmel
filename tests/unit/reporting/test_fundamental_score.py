"""Piotroski F-Score tests."""

from __future__ import annotations

from src.reporting.fundamental_score import compute_piotroski_f_score


def _full_good() -> dict:
    return {
        "symbol": "TEST",
        "period": "2025-12-31",
        "net_income": 100.0,
        "total_assets": 1000.0,
        "operating_cash_flow": 120.0,
        "total_debt": 50.0,
        "total_debt_prior": 80.0,
        "current_ratio": 2.0,
        "current_ratio_prior": 1.5,
        "shares_outstanding": 100.0,
        "shares_outstanding_prior": 110.0,
        "gross_margin": 0.4,
        "gross_margin_prior": 0.35,
        "asset_turnover": 1.2,
        "asset_turnover_prior": 1.0,
        "roa": 0.12,
        "roa_prior": 0.08,
    }


def test_f_score_perfect_9() -> None:
    r = compute_piotroski_f_score(_full_good())
    assert r is not None
    assert r.score == 9
    assert r.profitability == 4
    assert r.leverage == 3
    assert r.efficiency == 2


def test_f_score_zero() -> None:
    d = _full_good()
    d["net_income"] = -10.0
    d["operating_cash_flow"] = -15.0
    d["roa"] = 0.05
    d["roa_prior"] = 0.08
    d["total_debt"] = 100.0
    d["total_debt_prior"] = 50.0
    d["current_ratio"] = 1.0
    d["current_ratio_prior"] = 1.5
    d["shares_outstanding"] = 120.0
    d["shares_outstanding_prior"] = 100.0
    d["gross_margin"] = 0.2
    d["gross_margin_prior"] = 0.35
    d["asset_turnover"] = 0.8
    d["asset_turnover_prior"] = 1.0
    r = compute_piotroski_f_score(d)
    assert r is not None
    assert r.score == 0


def test_f_score_mixed() -> None:
    d = _full_good()
    d["roa"] = 0.05
    d["roa_prior"] = 0.08
    d["operating_cash_flow"] = 50.0
    d["total_debt"] = 100.0
    d["total_debt_prior"] = 50.0
    r = compute_piotroski_f_score(d)
    assert r is not None
    assert r.score == 6
    assert r.criteria["improving_roa"] is False
    assert r.criteria["lower_long_term_debt"] is False


def test_f_score_returns_none_when_data_missing() -> None:
    assert compute_piotroski_f_score({}) is None
    assert compute_piotroski_f_score({"symbol": "X"}) is None


def test_f_score_handles_zero_assets() -> None:
    d = _full_good()
    d["total_assets"] = 0.0
    assert compute_piotroski_f_score(d) is None


def test_f_score_category_breakdown() -> None:
    r = compute_piotroski_f_score(_full_good())
    assert r is not None
    assert r.profitability + r.leverage + r.efficiency == r.score


def test_f_score_returns_none_on_nan_total_assets() -> None:
    d = _full_good()
    d["total_assets"] = float("nan")
    assert compute_piotroski_f_score(d) is None


def test_f_score_returns_none_on_inf_net_income() -> None:
    d = _full_good()
    d["net_income"] = float("inf")
    assert compute_piotroski_f_score(d) is None


def test_f_score_returns_none_on_nan_operating_cash_flow() -> None:
    d = _full_good()
    d["operating_cash_flow"] = float("nan")
    assert compute_piotroski_f_score(d) is None
