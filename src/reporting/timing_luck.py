"""Timing-luck metrics from overlapping tranche backtests (Tier 54C)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def align_tranche_return_matrix(tranche_returns: list[pd.Series]) -> pd.DataFrame:
    """Outer-join tranche daily return series on date index; NaN where missing."""
    if not tranche_returns:
        return pd.DataFrame()
    frames = []
    for i, ser in enumerate(tranche_returns):
        s = ser.copy()
        s.index = pd.Index(s.index.astype(str), name="date")
        frames.append(s.rename(f"t{i}"))
    return pd.concat(frames, axis=1, join="outer").sort_index()


def timing_luck_headlines(
    *,
    mean_te_annual_pct: float,
    tranche_cagrs_pct: list[float],
    years: float,
) -> dict[str, float | bool]:
    """Headline timing-luck test in matched CAGR units (pp).

    Predicted cross-tranche CAGR dispersion from timing luck alone:
    ``sigma_CAGR ≈ TE / sqrt(T)`` where TE is annualized tracking error (%).

    95% CI in CAGR space: ``1.96 * TE / sqrt(T)`` (per-tranche, not cumulative).
    """
    cagrs = np.array(tranche_cagrs_pct, dtype=float)
    observed_std = float(cagrs.std(ddof=1)) if len(cagrs) > 1 else 0.0
    cagr_spread = float(cagrs.max() - cagrs.min()) if len(cagrs) else 0.0
    sqrt_t = math.sqrt(max(years, 1e-9))
    predicted_std = float(mean_te_annual_pct / sqrt_t)
    ratio = observed_std / predicted_std if predicted_std > 1e-12 else 0.0
    ci_95_cagr = 1.96 * predicted_std
    return {
        "ensemble_cagr_pct": round(float(np.mean(cagrs)), 4),
        "tranche_cagr_std_pp": round(observed_std, 4),
        "tranche_cagr_spread_pp": round(cagr_spread, 4),
        "predicted_tranche_cagr_std_pp": round(predicted_std, 4),
        "observed_tranche_cagr_std_pp": round(observed_std, 4),
        "observed_vs_predicted_sd_ratio": round(ratio, 4),
        "timing_luck_ci_95_cagr_pp": round(ci_95_cagr, 4),
        "observed_sd_below_predicted": bool(ratio < 1.0),
    }


def compute_timing_luck_metrics(
    tranche_returns: list[pd.Series],
    *,
    tranche_cagrs_pct: list[float],
    years: float,
) -> dict[str, float | int | bool | None]:
    """Headline timing-luck statistics for N staggered tranches.

    ``tranche_cagrs_pct`` supplies per-tranche CAGRs (percent). Daily returns feed
    tracking-error vs the equal-weight ensemble.
    """
    n = len(tranche_returns)
    empty: dict[str, float | int | bool | None] = {
        "n_tranches": n,
        "ensemble_cagr_pct": None,
        "tranche_cagr_std_pp": None,
        "tranche_cagr_spread_pp": None,
        "mean_timing_luck_te_annual_pct": None,
        "predicted_tranche_cagr_std_pp": None,
        "observed_tranche_cagr_std_pp": None,
        "observed_vs_predicted_sd_ratio": None,
        "timing_luck_ci_95_cagr_pp": None,
        "observed_sd_below_predicted": None,
    }
    if n < 2 or years <= 0.0:
        return empty

    mat = align_tranche_return_matrix(tranche_returns).dropna(how="any")
    if mat.shape[0] < 2:
        out = dict(empty)
        out["ensemble_cagr_pct"] = float(np.mean(tranche_cagrs_pct))
        return out

    ensemble = mat.mean(axis=1)
    tes: list[float] = []
    for col in mat.columns:
        diff = mat[col] - ensemble
        te = float(diff.std(ddof=1) * math.sqrt(252.0) * 100.0)
        tes.append(te)
    mean_te = float(np.mean(tes))

    headlines = timing_luck_headlines(
        mean_te_annual_pct=mean_te,
        tranche_cagrs_pct=tranche_cagrs_pct,
        years=years,
    )
    return {
        "n_tranches": n,
        "mean_timing_luck_te_annual_pct": round(mean_te, 4),
        **headlines,
    }
