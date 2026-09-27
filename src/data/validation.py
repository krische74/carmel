"""OHLCV consistency checks before indicators or storage."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class OHLCVValidationResult:
    """Outcome of validating a price/volume frame.

    ``is_valid`` is False only when hard errors are present (impossible data).
    ``warnings`` may be non-empty even when ``is_valid`` is True (e.g. statistical
    outlier notes); operators should review warnings in logs without blocking ingest.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase column names."""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    outlier_zscore_threshold: float | None = 4.0,
    min_prior_for_zscore: int = 20,
    max_abs_daily_return: float = 0.5,
) -> OHLCVValidationResult:
    """Check OHLCV for impossible data and log soft outlier notes.

    Hard errors (``is_valid`` False): broken OHLC relationships, negative volume,
    non-positive OHLC prices (each must be strictly > 0), NaN/Inf in required
    columns, or a daily return whose absolute value exceeds ``max_abs_daily_return``.

    Soft warnings: close level z-score vs prior closes when ``outlier_zscore_threshold``
    is set — fat-tailed real returns can trigger this without blocking ingest.

    Args:
        df: DataFrame with OHLCV columns (case-insensitive).
        outlier_zscore_threshold: If set, append a warning when ``close`` is extreme
            vs prior closes (point-in-time). None disables.
        min_prior_for_zscore: Minimum prior closes before the outlier z-score warning.
        max_abs_daily_return: Reject if ``|close[t]/close[t-1] - 1|`` exceeds this
            (default 0.5 = 50%). Skips the first row (no prior close).

    Returns:
        ``OHLCVValidationResult`` with ``is_valid`` False if any hard error exists.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return OHLCVValidationResult(is_valid=False, errors=["empty or missing OHLCV frame"])

    frame = _normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        return OHLCVValidationResult(
            is_valid=False,
            errors=[f"missing required columns: {', '.join(missing)}"],
        )

    o = frame["open"].astype(float)
    h = frame["high"].astype(float)
    low = frame["low"].astype(float)
    c = frame["close"].astype(float)
    v = frame["volume"].astype(float)

    for col_name, series in (
        ("open", o),
        ("high", h),
        ("low", low),
        ("close", c),
        ("volume", v),
    ):
        arr = series.to_numpy(dtype=float)
        if np.isnan(arr).any():
            errors.append(f"NaN in column {col_name!r}")
        if np.isinf(arr).any():
            errors.append(f"Inf in column {col_name!r}")

    if errors:
        return OHLCVValidationResult(is_valid=False, errors=errors, warnings=[])

    for i in range(len(frame)):
        lo, hi = float(low.iloc[i]), float(h.iloc[i])
        oc, cc = float(o.iloc[i]), float(c.iloc[i])
        if oc <= 0.0 or hi <= 0.0 or lo <= 0.0 or cc <= 0.0:
            errors.append(
                f"row {i}: non-positive price (open/high/low/close must be > 0; "
                f"got open={oc}, high={hi}, low={lo}, close={cc})",
            )
        if lo > hi:
            errors.append(f"row {i}: low ({lo}) > high ({hi})")
        if oc < lo or oc > hi:
            errors.append(f"row {i}: open ({oc}) outside [low, high] [{lo}, {hi}]")
        if not (lo <= cc <= hi):
            errors.append(f"row {i}: close ({cc}) outside [low, high] [{lo}, {hi}]")

    if (v < 0).any():
        bad = v[v < 0]
        errors.append(f"negative volume on {len(bad)} row(s)")

    closes = c.to_numpy(dtype=float)
    if len(closes) > 1 and max_abs_daily_return > 0.0:
        for t in range(1, len(closes)):
            prev = closes[t - 1]
            if prev <= 0.0:
                continue
            r_abs = abs(closes[t] / prev - 1.0)
            if r_abs > max_abs_daily_return:
                errors.append(
                    f"row {t}: daily return |{r_abs:.4f}| exceeds max_abs_daily_return "
                    f"({max_abs_daily_return}) — likely corrupt feed",
                )

    if len(errors) == 0 and outlier_zscore_threshold is not None and len(closes) > 1:
        for i in range(1, len(closes)):
            prior = closes[:i]
            mu = float(np.mean(prior))
            sigma = float(np.std(prior, ddof=1)) if len(prior) > 1 else 0.0
            sigma = max(sigma, 1e-12)
            z = (closes[i] - mu) / sigma
            if len(prior) >= min_prior_for_zscore and abs(z) > outlier_zscore_threshold:
                warnings.append(
                    f"row {i}: close z-score vs prior levels {z:.2f} exceeds "
                    f"outlier threshold {outlier_zscore_threshold} (informational)",
                )

    is_valid = len(errors) == 0
    return OHLCVValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
