"""Tier 53A: audit whether Parquet OHLCV is dividend-adjusted (Q014)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.reporting.ohlcv_adjustment_audit import (
    audit_spy_adjustment,
    summarize_dividend_gap,
)


def test_audit_detects_nominal_close_matches_unadjusted_yahoo() -> None:
    """Stored Close == Yahoo Close (auto_adjust=False) => unadjusted for dividends."""
    idx = pd.DatetimeIndex([pd.Timestamp("2007-01-03")])
    stored = pd.DataFrame({"close": [141.37]}, index=idx)
    yahoo_close = pd.Series([141.37], index=idx)
    yahoo_adj = pd.Series([98.87], index=idx)
    result = audit_spy_adjustment(stored, yahoo_close=yahoo_close, yahoo_adj_close=yahoo_adj)
    assert result.matches_nominal_close is True
    assert result.matches_adj_close is False
    assert result.verdict == "unadjusted_nominal"


def test_audit_detects_adj_close_series() -> None:
    idx = pd.DatetimeIndex([pd.Timestamp("2007-01-03")])
    stored = pd.DataFrame({"close": [98.87]}, index=idx)
    yahoo_close = pd.Series([141.37], index=idx)
    yahoo_adj = pd.Series([98.87], index=idx)
    result = audit_spy_adjustment(stored, yahoo_close=yahoo_close, yahoo_adj_close=yahoo_adj)
    assert result.matches_adj_close is True
    assert result.verdict == "dividend_adjusted"


def test_summarize_dividend_gap_direction() -> None:
    """Bond/cash income ETFs lose more omitted yield than SPY — strategy is penalised."""
    gaps = {
        "SPY": 2.0,
        "TLT": 3.16,
        "SHV": 1.52,
        "QQQ": 0.91,
        "GLD": 0.0,
    }
    s = summarize_dividend_gap(gaps)
    assert s.spy_gap_pp == pytest.approx(2.0)
    assert s.max_income_etf_gap_pp == pytest.approx(3.16)
    assert "penalis" in s.direction_note.lower() or "understate" in s.direction_note.lower()
