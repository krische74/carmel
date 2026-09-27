"""SHAP summaries for tree models (optional ``[ml]`` extra)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def shap_summary_text(
    model: Any,
    background: np.ndarray,
    x_one: np.ndarray,
    feature_names: list[str],
    *,
    max_features: int = 5,
) -> str:
    """Short text listing largest |SHAP| contributions for one row."""
    try:
        import shap
    except ImportError:
        return ""

    if background.size == 0 or x_one.size == 0:
        return ""

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x_one.reshape(1, -1))
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return ""

    if sv is None:
        return ""
    vec = np.asarray(sv).ravel()
    if vec.shape[0] != len(feature_names):
        return ""

    order = np.argsort(np.abs(vec))[::-1][:max_features]
    parts = [f"{feature_names[i]}: {vec[i]:+.4f}" for i in order]
    return "SHAP: " + "; ".join(parts)
