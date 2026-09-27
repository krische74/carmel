"""Training momentum ML bundle (requires ``.[ml]`` extra)."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from src.config import DataConfig, MLConfig, Settings
from src.ml.features import ML_FEATURE_NAMES
from src.ml.train import train_momentum_model_bundle
from tests.unit.ml.test_features import _synthetic_ohlcv

if TYPE_CHECKING:
    from pathlib import Path


def test_train_rejects_model_path_outside_project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("sklearn")
    project_root = tmp_path / "repo"
    project_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_model = outside_dir / "model.joblib"
    monkeypatch.setattr("src.config.PROJECT_ROOT", project_root)

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(),
        ml=MLConfig(model_path=str(outside_model.resolve())),
    )
    with pytest.raises(ValueError, match="project root"):
        train_momentum_model_bundle(
            settings,
            {},
            symbols=["SPY"],
            start=date(2024, 1, 1),
            end=date(2024, 6, 1),
        )


def test_train_raises_when_insufficient_feature_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("sklearn")
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=["SPY"]),
        ml=MLConfig(
            model_path="ml/too_few_rows.joblib",
            min_train_rows=500,
            label_horizon_days=3,
        ),
    )
    df = _synthetic_ohlcv(220)
    data = {"SPY": df}
    start = pd.Timestamp(df.index[80]).date()
    end = pd.Timestamp(df.index[-15]).date()

    with pytest.raises(ValueError, match="Not enough training rows"):
        train_momentum_model_bundle(
            settings,
            data,
            symbols=["SPY"],
            start=start,
            end=end,
        )


def test_train_momentum_model_bundle_writes_expected_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hgb_mod = pytest.importorskip("sklearn.ensemble")
    hgb_cls = hgb_mod.HistGradientBoostingClassifier

    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

    df = _synthetic_ohlcv(220)
    data = {"SPY": df}
    start = pd.Timestamp(df.index[80]).date()
    end = pd.Timestamp(df.index[-15]).date()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=["SPY"]),
        ml=MLConfig(model_path="ml/train_test.joblib", min_train_rows=30, label_horizon_days=3),
    )

    out = train_momentum_model_bundle(
        settings,
        data,
        symbols=["SPY"],
        start=start,
        end=end,
    )
    assert out.is_file()
    import joblib

    bundle = joblib.load(out)
    assert isinstance(bundle.get("model"), hgb_cls)
    assert list(bundle.get("feature_names") or []) == list(ML_FEATURE_NAMES)
    assert bundle.get("X_bg") is not None
    assert bundle.get("model_version") == settings.ml.model_version


def test_train_x_bg_is_last_up_to_200_rows_of_training_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier

    x_fit_captured: list[np.ndarray] = []
    orig_fit = HistGradientBoostingClassifier.fit

    def fit_capture(self, x, y, *args: Any, **kwargs: Any) -> Any:
        x_fit_captured.append(np.asarray(x, dtype=np.float64))
        return orig_fit(self, x, y, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", fit_capture)
    monkeypatch.setattr("src.config.PROJECT_ROOT", tmp_path)

    df = _synthetic_ohlcv(220)
    data = {"SPY": df}
    start = pd.Timestamp(df.index[80]).date()
    end = pd.Timestamp(df.index[-15]).date()

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(universe=["SPY"]),
        ml=MLConfig(model_path="ml/xbg_test.joblib", min_train_rows=30, label_horizon_days=3),
    )

    out = train_momentum_model_bundle(
        settings,
        data,
        symbols=["SPY"],
        start=start,
        end=end,
    )
    import joblib

    bundle = joblib.load(out)
    x_fit = x_fit_captured[0]
    bg_n = min(200, len(x_fit))
    expected = x_fit[-bg_n:] if bg_n > 0 else x_fit
    np.testing.assert_array_equal(bundle["X_bg"], expected)
