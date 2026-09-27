"""Unit tests for stored-OHLCV audit used by deep-history ingest (Tier 51B)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.reporting.ohlcv_audit import audit_ohlcv_frame


def test_audit_reports_max_abs_daily_return_and_no_gaps() -> None:
    idx = pd.bdate_range("2024-01-02", periods=5, freq="B")
    close = [100.0, 101.0, 90.0, 91.0, 92.0]
    df = pd.DataFrame(
        {
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1e6] * 5,
        },
        index=idx,
    )
    report = audit_ohlcv_frame("SPY", df)
    assert report.symbol == "SPY"
    assert report.start == date(2024, 1, 2)
    assert report.end == date(2024, 1, 8)
    assert report.max_abs_daily_return == pytest.approx(11.0 / 101.0, rel=1e-9)
    assert report.gap_count == 0
    assert report.rows == 5


def test_audit_counts_business_day_gaps() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-08"])
    close = [100.0, 101.0, 102.0]
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": [1e6] * 3},
        index=idx,
    )
    report = audit_ohlcv_frame("SPY", df)
    assert report.gap_count >= 1
    assert any("2024-01-04" in g or "2024-01-05" in g for g in report.gap_ranges)
