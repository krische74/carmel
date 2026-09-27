"""ML bundle load and scoring (requires ``.[ml]`` for some tests)."""

from __future__ import annotations

from datetime import UTC, datetime, time
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.ml.features import ML_FEATURE_NAMES, build_feature_vector
from src.ml.infer import load_ml_bundle, score_positive_proba
from tests.unit.ml.test_features import _synthetic_ohlcv

if TYPE_CHECKING:
    from pathlib import Path


def test_load_ml_bundle_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_ml_bundle(tmp_path / "missing.joblib") is None


def test_load_ml_bundle_corrupt_file_returns_none(tmp_path: Path) -> None:
    pytest.importorskip("joblib")
    p = tmp_path / "bad.joblib"
    p.write_bytes(b"\x00not a valid joblib")
    assert load_ml_bundle(p) is None


def test_score_positive_proba_requires_model_and_features() -> None:
    assert score_positive_proba({}, {"ret_1": 0.1}) is None
    assert score_positive_proba({"model": object(), "feature_names": []}, {"ret_1": 0.1}) is None


def test_score_positive_proba_with_sklearn_classifier() -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, len(ML_FEATURE_NAMES)))
    y = (x[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=200)
    clf.fit(x, y)
    bundle = {"model": clf, "feature_names": list(ML_FEATURE_NAMES)}
    feats = {n: float(x[-1, i]) for i, n in enumerate(ML_FEATURE_NAMES)}
    proba = score_positive_proba(bundle, feats)
    assert proba is not None
    assert 0.0 <= proba <= 1.0


def test_score_positive_proba_predict_proba_error_returns_none() -> None:
    model = Mock()
    model.predict_proba.side_effect = ValueError("invalid feature matrix")
    bundle = {"model": model, "feature_names": list(ML_FEATURE_NAMES)}
    feats = {n: 0.0 for n in ML_FEATURE_NAMES}
    assert score_positive_proba(bundle, feats) is None


def test_score_positive_proba_missing_required_feature_key_returns_none() -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, len(ML_FEATURE_NAMES)))
    y = (x[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=200)
    clf.fit(x, y)
    bundle = {"model": clf, "feature_names": list(ML_FEATURE_NAMES)}
    incomplete = {n: 0.0 for n in list(ML_FEATURE_NAMES)[:-1]}
    assert score_positive_proba(bundle, incomplete) is None


def test_score_round_trip_after_train(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from src.config import DataConfig, MLConfig, Settings
    from src.ml.train import train_momentum_model_bundle

    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

    df = _synthetic_ohlcv(220)
    data = {"SPY": df}
    start = pd.Timestamp(df.index[80]).date()
    end = pd.Timestamp(df.index[-15]).date()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=["SPY"]),
        ml=MLConfig(model_path="ml/infer_rt.joblib", min_train_rows=30, label_horizon_days=3),
    )

    import joblib

    bundles_at_dump: list[dict[str, Any]] = []
    orig_dump = joblib.dump

    def dump_capture(bundle: dict[str, Any], path: Any, **kwargs: Any) -> Any:
        bundles_at_dump.append(dict(bundle))
        return orig_dump(bundle, path, **kwargs)

    monkeypatch.setattr(joblib, "dump", dump_capture)

    out = train_momentum_model_bundle(
        settings,
        data,
        symbols=["SPY"],
        start=start,
        end=end,
    )
    assert bundles_at_dump, "train should persist bundle via joblib.dump"
    bundle_disk = load_ml_bundle(out)
    assert bundle_disk is not None

    as_of = datetime.combine(pd.Timestamp(df.index[-5]).date(), time(12, 0), tzinfo=UTC)
    feats = build_feature_vector(df, as_of=as_of)
    assert feats is not None
    proba_disk = score_positive_proba(bundle_disk, feats)
    assert proba_disk is not None
    assert 0.0 <= proba_disk <= 1.0
    proba_memory = score_positive_proba(bundles_at_dump[-1], feats)
    assert proba_memory is not None
    assert proba_memory == pytest.approx(proba_disk, rel=0.0, abs=1e-9)
