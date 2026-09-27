"""Point-in-time slicing of OHLCV frames (no lookahead)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime


def slice_to_as_of(df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Return rows with index <= ``as_of`` (date-normalized, tz-stripped)."""
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ts = ts.normalize()
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    trimmed = df.copy()
    trimmed.index = idx
    return trimmed.loc[:ts]
