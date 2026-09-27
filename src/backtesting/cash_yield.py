from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from datetime import date

logger = logging.getLogger(__name__)

MAX_FFILL_CALENDAR_DAYS = 10


def series_from_macro_rows(rows: list[dict[str, Any]]) -> pd.Series:
    """Build a date-indexed annual-percent series from ``macro_indicators`` rows."""
    if not rows:
        return pd.Series(dtype=float)
    idx: list[pd.Timestamp] = []
    vals: list[float] = []
    for row in rows:
        d = str(row.get("date", "")).strip()
        if not d:
            continue
        try:
            val = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        idx.append(pd.Timestamp(d).normalize())
        vals.append(val)
    if not idx:
        return pd.Series(dtype=float)
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def lookup_cash_yield_annual_pct(
    series: pd.Series,
    d: date,
    *,
    fallback_pct: float,
) -> tuple[float, str]:
    """Return (annual percent, mode) for calendar date ``d``.

    Forward-fills the last observation within ``MAX_FFILL_CALENDAR_DAYS``.
    Longer gaps, empty series, or no prior observation use ``fallback_pct`` and
    log a WARNING (never a silent 4%).
    """
    fallback = float(fallback_pct)
    if series is None or series.empty:
        logger.warning(
            "cash yield series empty on %s; using fallback_flat %.2f%%",
            d.isoformat(),
            fallback,
        )
        return fallback, "fallback_flat"
    idx = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
    s = pd.Series(series.to_numpy(dtype=float), index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    ts = pd.Timestamp(d).normalize()
    prior = s.loc[:ts]
    if prior.empty:
        logger.warning(
            "cash yield has no observation on or before %s; using fallback_flat %.2f%%",
            d.isoformat(),
            fallback,
        )
        return fallback, "fallback_flat"
    last_ts = prior.index[-1]
    gap = int((ts - last_ts).days)
    if gap > MAX_FFILL_CALENDAR_DAYS:
        logger.warning(
            "cash yield gap of %s days at %s (last %s); using fallback_flat %.2f%%",
            gap,
            d.isoformat(),
            last_ts.date().isoformat(),
            fallback,
        )
        return fallback, "fallback_flat"
    return float(prior.iloc[-1]), "historical"
