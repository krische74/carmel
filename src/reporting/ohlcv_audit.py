"""Inspect stored OHLCV for gaps and extreme daily returns (Tier 51B)."""

from __future__ import annotations

from datetime import date  # noqa: TC003

import pandas as pd
from pydantic import BaseModel


class OhlcvAuditReport(BaseModel):
    """Coverage and quality summary for one symbol's stored bars."""

    symbol: str
    rows: int
    start: date | None
    end: date | None
    max_abs_daily_return: float
    max_abs_daily_return_date: str | None
    gap_count: int
    gap_ranges: list[str]
    zscore_outlier_count: int = 0


def audit_ohlcv_frame(
    symbol: str,
    df: pd.DataFrame,
    *,
    zscore_threshold: float = 4.0,
    min_prior_for_zscore: int = 20,
) -> OhlcvAuditReport:
    """Compute date span, max |daily close return|, and missing business days."""
    if df is None or df.empty or "close" not in df.columns:
        return OhlcvAuditReport(
            symbol=symbol.strip().upper(),
            rows=0,
            start=None,
            end=None,
            max_abs_daily_return=0.0,
            max_abs_daily_return_date=None,
            gap_count=0,
            gap_ranges=[],
        )
    frame = df.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None).normalize()
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    closes = frame["close"].astype(float)
    rets = closes.pct_change().dropna()
    max_abs = 0.0
    max_date: str | None = None
    if not rets.empty:
        abs_r = rets.abs()
        i = int(abs_r.values.argmax())
        max_abs = float(abs_r.iloc[i])
        max_date = pd.Timestamp(abs_r.index[i]).date().isoformat()

    bdays = pd.bdate_range(frame.index.min(), frame.index.max(), freq="B")
    present = set(frame.index.normalize())
    missing = [d for d in bdays if d.normalize() not in present]
    gap_ranges: list[str] = []
    gap_count = 0
    if missing:
        run_start = missing[0]
        prev = missing[0]
        for d in missing[1:]:
            if (d - prev).days > 3:
                gap_ranges.append(f"{run_start.date().isoformat()}->{prev.date().isoformat()}")
                gap_count += 1
                run_start = d
            prev = d
        gap_ranges.append(f"{run_start.date().isoformat()}->{prev.date().isoformat()}")
        gap_count += 1

    z_count = 0
    if len(rets) > min_prior_for_zscore:
        for i in range(min_prior_for_zscore, len(rets)):
            prior = rets.iloc[i - min_prior_for_zscore : i]
            std = float(prior.std(ddof=1))
            if std <= 1e-18:
                continue
            z = abs(float(rets.iloc[i] - float(prior.mean())) / std)
            if z >= zscore_threshold:
                z_count += 1

    return OhlcvAuditReport(
        symbol=symbol.strip().upper(),
        rows=len(frame),
        start=pd.Timestamp(frame.index.min()).date(),
        end=pd.Timestamp(frame.index.max()).date(),
        max_abs_daily_return=max_abs,
        max_abs_daily_return_date=max_date,
        gap_count=gap_count,
        gap_ranges=gap_ranges,
        zscore_outlier_count=z_count,
    )
