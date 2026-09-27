"""Trading calendar helpers used by the backtester."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.backtesting.engine import trading_days_in_range


def _df(days: int, start: str) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=days, freq="B")
    return pd.DataFrame({"close": range(days, 0, -1)}, index=idx)


def test_trading_days_in_range_union_two_symbols() -> None:
    data = {"A": _df(5, "2024-01-02"), "B": _df(5, "2024-01-02")}
    days = trading_days_in_range(data, date(2024, 1, 1), date(2024, 12, 31))
    assert len(days) == 5
    assert days[0] == date(2024, 1, 2)


def test_trading_days_in_range_clips_start_end() -> None:
    data = {"X": _df(10, "2024-01-02")}
    days = trading_days_in_range(data, date(2024, 1, 8), date(2024, 1, 19))
    assert all(date(2024, 1, 8) <= d <= date(2024, 1, 19) for d in days)


def test_trading_days_in_range_empty_data() -> None:
    assert trading_days_in_range({}, date(2024, 1, 1), date(2024, 1, 31)) == []


def test_trading_days_in_range_skips_empty_frame() -> None:
    data = {"A": pd.DataFrame(), "B": _df(3, "2024-02-01")}
    days = trading_days_in_range(data, date(2024, 1, 1), date(2024, 12, 31))
    assert len(days) == 3


def test_trading_days_sorted() -> None:
    a = _df(3, "2024-03-01")
    b = _df(3, "2024-03-04")
    data = {"A": a, "B": b}
    days = trading_days_in_range(data, date(2024, 1, 1), date(2024, 12, 31))
    assert days == sorted(days)
