"""Train a small classifier on historical features (walk-forward friendly inputs)."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.backtesting.engine import _normalize_ohlcv_index
from src.ml.features import ML_FEATURE_NAMES, build_feature_vector, feature_dict_to_row
from src.ml.labels import forward_return_binary

if TYPE_CHECKING:
    from pathlib import Path

    from src.config import Settings

logger = logging.getLogger(__name__)


def train_momentum_model_bundle(
    settings: Settings,
    data: dict[str, pd.DataFrame],
    *,
    symbols: list[str],
    start: date,
    end: date,
) -> Path:
    """Fit :class:`sklearn.ensemble.HistGradientBoostingClassifier` and save joblib bundle.

    Returns path written (``settings.ml.model_path`` resolved).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    from src.config import PROJECT_ROOT, ml_model_path

    out = ml_model_path(settings)
    out_resolved = out.resolve()
    root_resolved = PROJECT_ROOT.resolve()
    if not out_resolved.is_relative_to(root_resolved):
        msg = (
            f"ml.model_path must resolve under project root ({root_resolved}), got {out_resolved}"
        )
        raise ValueError(msg)

    horizon = int(settings.ml.label_horizon_days)
    names = list(ML_FEATURE_NAMES)
    feat_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for sym in symbols:
        raw = data.get(sym)
        if raw is None or raw.empty:
            continue
        norm = _normalize_ohlcv_index(raw.copy())
        if len(norm) < horizon + 60:
            continue
        closes = norm["close"].astype(float).to_numpy()
        for i in range(60, len(norm) - horizon):
            ts = norm.index[i]
            d = pd.Timestamp(ts).date()
            if not (start <= d <= end):
                continue
            as_of = datetime.combine(d, time.min, tzinfo=UTC)
            fv = build_feature_vector(raw, as_of=as_of)
            if fv is None:
                continue
            lab = forward_return_binary(closes, index=i, horizon=horizon)
            if lab is None:
                continue
            feat_rows.append(feature_dict_to_row(fv, names).ravel())
            y_rows.append(int(lab))

    if len(feat_rows) < int(settings.ml.min_train_rows):
        msg = (
            f"Not enough training rows: {len(feat_rows)} "
            f"(min_train_rows={settings.ml.min_train_rows})"
        )
        raise ValueError(msg)

    x_fit = np.vstack(feat_rows)
    y = np.array(y_rows, dtype=np.int64)
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        random_state=42,
        max_depth=6,
        learning_rate=0.06,
    )
    clf.fit(x_fit, y)

    bg_n = min(200, len(x_fit))
    x_background = x_fit[-bg_n:] if bg_n > 0 else x_fit

    bundle = {
        "model": clf,
        "feature_names": names,
        "X_bg": x_background,
        "model_version": str(settings.ml.model_version),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(bundle, out)
    logger.info("Saved ML bundle to %s (%d rows)", out, len(y_rows))
    return out
