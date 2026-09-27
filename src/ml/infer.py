"""Load persisted model and score one feature dict."""

from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def load_ml_bundle(path: Path) -> dict[str, Any] | None:
    """Load joblib bundle; return None on failure."""
    if not path.is_file():
        return None
    try:
        import joblib

        return joblib.load(path)
    except (OSError, EOFError, KeyError, pickle.UnpicklingError, ValueError) as exc:
        logger.warning("ML bundle load failed: %s", exc)
        return None


def score_positive_proba(bundle: dict[str, Any], feats: dict[str, float]) -> float | None:
    """Return P(class=1) for binary classifier, or None."""
    model = bundle.get("model")
    names: list[str] = list(bundle.get("feature_names") or [])
    if model is None or not names:
        return None
    try:
        row = np.array([[float(feats[n]) for n in names]], dtype=np.float64)
    except KeyError:
        return None
    try:
        proba = model.predict_proba(row)
    except (TypeError, ValueError, AttributeError) as exc:
        logger.warning("ML predict_proba failed: %s", exc)
        return None
    if proba is None or proba.size < 2:
        return None
    return float(proba[0, 1])
