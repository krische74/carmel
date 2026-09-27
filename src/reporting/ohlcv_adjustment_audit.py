"""Detect whether stored OHLCV closes are dividend-adjusted (Tier 53A / Q014)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class AdjustmentAuditResult(BaseModel):
    """Outcome of comparing a stored close series to Yahoo Close vs Adj Close."""

    matches_nominal_close: bool
    matches_adj_close: bool
    verdict: str
    sample_date: str | None = None
    stored_close: float | None = None
    yahoo_close: float | None = None
    yahoo_adj_close: float | None = None
    relative_gap_nominal_vs_adj: float | None = None


class DividendGapSummary(BaseModel):
    """Rough magnitude of omitted dividend yield by symbol (CAGR total - price)."""

    gaps_pp: dict[str, float] = Field(default_factory=dict)
    spy_gap_pp: float = 0.0
    max_income_etf_gap_pp: float = 0.0
    direction_note: str = ""


def _normalize_index(idx: pd.DatetimeIndex | pd.Index) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(pd.to_datetime(idx))
    if di.tz is not None:
        di = di.tz_convert("UTC").tz_localize(None)
    return di.normalize()


def audit_spy_adjustment(
    stored_ohlcv: pd.DataFrame,
    *,
    yahoo_close: pd.Series,
    yahoo_adj_close: pd.Series,
    atol: float = 0.05,
) -> AdjustmentAuditResult:
    """Classify stored ``close`` against Yahoo nominal Close and Adj Close.

    Uses the earliest overlapping date present in all three series.
    """
    if stored_ohlcv.empty or "close" not in stored_ohlcv.columns:
        return AdjustmentAuditResult(
            matches_nominal_close=False,
            matches_adj_close=False,
            verdict="no_data",
        )
    s = stored_ohlcv.copy()
    s.index = _normalize_index(s.index)
    yc = yahoo_close.copy()
    yc.index = _normalize_index(yc.index)
    ya = yahoo_adj_close.copy()
    ya.index = _normalize_index(ya.index)
    common = s.index.intersection(yc.index).intersection(ya.index)
    if len(common) == 0:
        return AdjustmentAuditResult(
            matches_nominal_close=False,
            matches_adj_close=False,
            verdict="no_overlap",
        )
    d = common.min()
    stored = float(s.loc[d, "close"])
    if isinstance(stored, pd.Series):
        stored = float(stored.iloc[-1])
    close_v = float(yc.loc[d])
    adj_v = float(ya.loc[d])
    match_nom = abs(stored - close_v) <= atol
    match_adj = abs(stored - adj_v) <= atol
    gap = None if close_v == 0.0 else (close_v - adj_v) / close_v
    if match_adj and not match_nom:
        verdict = "dividend_adjusted"
    elif match_nom and not match_adj:
        verdict = "unadjusted_nominal"
    elif match_nom and match_adj:
        verdict = "ambiguous_equal"
    else:
        verdict = "matches_neither"
    return AdjustmentAuditResult(
        matches_nominal_close=match_nom,
        matches_adj_close=match_adj,
        verdict=verdict,
        sample_date=pd.Timestamp(d).date().isoformat(),
        stored_close=stored,
        yahoo_close=close_v,
        yahoo_adj_close=adj_v,
        relative_gap_nominal_vs_adj=gap,
    )


def summarize_dividend_gap(gaps_pp: dict[str, float]) -> DividendGapSummary:
    """Describe who is hurt more when dividends are omitted from returns."""
    spy = float(gaps_pp.get("SPY", 0.0))
    income = {k: float(v) for k, v in gaps_pp.items() if k in ("TLT", "SHV", "BIL", "BND")}
    max_inc = max(income.values()) if income else 0.0
    note = (
        "Omitting dividends understates total return on both sides, but income ETFs "
        f"(max gap {max_inc:.2f} pp CAGR) lose more than SPY ({spy:.2f} pp). "
        "A strategy that rotates into TLT/SHV is therefore penalised relative to SPY "
        "on an unadjusted instrument. Separate from DTB3 cash-on-cash yield."
    )
    return DividendGapSummary(
        gaps_pp={k: float(v) for k, v in gaps_pp.items()},
        spy_gap_pp=spy,
        max_income_etf_gap_pp=max_inc,
        direction_note=note,
    )


def audit_result_as_dict(result: AdjustmentAuditResult) -> dict[str, Any]:
    """JSON-friendly dump."""
    return result.model_dump()
