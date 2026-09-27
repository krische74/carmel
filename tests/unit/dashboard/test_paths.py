"""Dashboard path resolution (mirrors hub layout)."""

from __future__ import annotations

import pytest  # noqa: TC002

from src.config import DataConfig, Settings
from src.dashboard.paths import hub_parquet_path, hub_sqlite_path, parquet_root


def test_hub_sqlite_path_resolves_under_cache(
    dashboard_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.dashboard.paths.get_settings", lambda: dashboard_settings)
    p = hub_sqlite_path()
    assert p.name == "hub_metadata.sqlite"
    assert "cache" in p.parts


def test_hub_parquet_path_matches_parquet_root(
    dashboard_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.dashboard.paths.get_settings", lambda: dashboard_settings)
    assert hub_parquet_path() == parquet_root()
    assert hub_parquet_path().name == "parquet"


def test_parquet_root_absolute_passthrough(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    abs_parq = tmp_path / "abs_pq"
    abs_parq.mkdir()
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(parquet_dir=str(abs_parq)),
    )
    monkeypatch.setattr("src.dashboard.paths.get_settings", lambda: s)
    assert parquet_root() == abs_parq
