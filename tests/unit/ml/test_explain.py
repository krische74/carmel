"""SHAP summary text (requires ``.[ml]`` extra)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.explain import shap_summary_text
from src.ml.features import ML_FEATURE_NAMES


def test_shap_summary_text_with_gradient_boosting() -> None:
    pytest.importorskip("shap")
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(7)
    x_bg = rng.normal(size=(60, len(ML_FEATURE_NAMES)))
    y = (x_bg[:, 0] + x_bg[:, 1] > 0).astype(np.int64)
    clf = HistGradientBoostingClassifier(
        max_iter=40,
        max_depth=4,
        random_state=0,
        learning_rate=0.1,
    )
    clf.fit(x_bg, y)
    x_one = x_bg[-1:]
    names = list(ML_FEATURE_NAMES)
    text = shap_summary_text(clf, x_bg[:40], x_one, names)
    assert text.startswith("SHAP:")
    assert any(n in text for n in ("rsi_14", "sma50_dist", "ret_1"))
    assert text.count(";") >= 1
    assert "+" in text or "-" in text
    # Largest |SHAP| first (explainer orders by np.argsort(np.abs(vec))[::-1])
    body = text.removeprefix("SHAP: ").strip()
    parts = [p.strip() for p in body.split(";") if p.strip()]
    assert len(parts) >= 2
    vals = [float(p.split(":", 1)[1].strip()) for p in parts[:2]]
    assert abs(vals[0]) >= abs(vals[1])
