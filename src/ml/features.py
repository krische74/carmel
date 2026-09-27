"""Point-in-time OHLCV features for ML (no lookahead in live use)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.strategy.as_of import slice_to_as_of
from src.strategy.indicators import adx, macd, rsi, sma

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

logger = logging.getLogger(__name__)

ML_FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_10",
    "rsi_14",
    "macd_hist",
    "adx_14",
    "sma50_dist",
)

_MIN_BARS = 55  # 50-bar SMA window + 5-bar buffer

_REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


def build_feature_vector(df: pd.DataFrame, *, as_of: datetime) -> dict[str, float] | None:
    """Build a feature dict from OHLCV using only bars up to and including ``as_of``."""
    if df is None or df.empty:
        logger.debug("build_feature_vector: empty_or_none_dataframe")
        return None

    for col in _REQUIRED_OHLCV:
        if col not in df.columns:
            wrong_case = next((c for c in df.columns if str(c).lower() == col), None)
            if wrong_case is not None:
                logger.debug(
                    "build_feature_vector: column %r must be lowercase; found %r",
                    col,
                    wrong_case,
                )
            else:
                logger.debug("build_feature_vector: missing_column %r", col)
            return None

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.debug(
            "build_feature_vector: index_must_be_datetimeindex got_type=%s",
            type(df.index).__name__,
        )
        return None

    frame = slice_to_as_of(df, as_of)
    if len(frame) < _MIN_BARS:
        logger.debug(
            "build_feature_vector: insufficient_bars %s < %s",
            len(frame),
            _MIN_BARS,
        )
        return None
    for col in _REQUIRED_OHLCV:
        if col not in frame.columns:
            logger.debug("build_feature_vector: missing_column_after_slice %r", col)
            return None

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)

    last = float(close.iloc[-1])
    ret_1 = float(last / float(close.iloc[-2]) - 1.0) if len(close) >= 2 else 0.0
    ret_5 = float(last / float(close.iloc[-6]) - 1.0) if len(close) >= 6 else 0.0
    ret_10 = float(last / float(close.iloc[-11]) - 1.0) if len(close) >= 11 else 0.0

    rsi_s = rsi(close, period=14)
    rsi_v = float(rsi_s.iloc[-1])
    if np.isnan(rsi_v):
        logger.debug("build_feature_vector: rsi_14_nan")
        return None

    _m, _s, hist = macd(close)
    macd_hist = float(hist.iloc[-1])
    if np.isnan(macd_hist):
        logger.debug("build_feature_vector: macd_hist_nan")
        return None

    adx_s, _, _ = adx(high, low, close, period=14)
    adx_v = float(adx_s.iloc[-1])
    if np.isnan(adx_v):
        logger.debug("build_feature_vector: adx_14_nan")
        return None

    sma50 = sma(close, period=50)
    s50 = float(sma50.iloc[-1])
    if s50 <= 0.0 or np.isnan(s50):
        logger.debug("build_feature_vector: sma50_invalid value=%s", s50)
        return None
    sma50_dist = float(last / s50 - 1.0)

    return {
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "rsi_14": rsi_v,
        "macd_hist": macd_hist,
        "adx_14": adx_v,
        "sma50_dist": sma50_dist,
    }


def feature_dict_to_row(
    feats: dict[str, float],
    names: Sequence[str],
) -> np.ndarray:
    """Single sample shape (1, n_features) in ``names`` order."""
    return np.array([[float(feats[n]) for n in names]], dtype=np.float64)
