"""Strategy factory wiring from settings."""

from __future__ import annotations

import logging

import pytest

from src.automation.strategy_wiring import build_strategies_from_enabled, strategy_factory_map
from src.config import DataConfig, DCATarget, Settings, StrategyConfig
from src.strategy.dca import DCAStrategy
from src.strategy.mean_reversion import MeanReversionStrategy
from src.strategy.momentum import MomentumRotationStrategy


def test_strategy_factory_map_has_core_strategies() -> None:
    s = Settings(_yaml_path=None, _env_file=None)
    m = strategy_factory_map(s)
    assert set(m) >= {"momentum", "dca", "mean_reversion"}
    assert isinstance(m["momentum"](), MomentumRotationStrategy)
    assert isinstance(m["dca"](), DCAStrategy)
    assert isinstance(m["mean_reversion"](), MeanReversionStrategy)


def test_build_strategies_from_enabled_all_three() -> None:
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        strategy=StrategyConfig(enabled=["momentum", "dca", "mean_reversion"]),
        data=DataConfig(
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
    )
    strats = build_strategies_from_enabled(s)
    types = [type(x).__name__ for x in strats]
    assert types == [
        "MomentumRotationStrategy",
        "DCAStrategy",
        "MeanReversionStrategy",
    ]


def test_build_strategies_from_default_yaml_excludes_mean_reversion() -> None:
    """Tier 44A: mean reversion is disabled in settings.yaml (config kept, not enabled)."""
    from pathlib import Path

    s = Settings(_yaml_path=Path("config/settings.yaml"), _env_file=None)
    assert "mean_reversion" not in s.strategy.enabled
    assert s.strategy.enabled == ["momentum", "dca"]
    strats = build_strategies_from_enabled(s)
    assert not any(isinstance(x, MeanReversionStrategy) for x in strats)
    assert [type(x).__name__ for x in strats] == ["MomentumRotationStrategy", "DCAStrategy"]


def test_build_strategies_empty_enabled_raises() -> None:
    # Normal ``Settings`` rejects ``enabled=[]`` at validation time; use model_construct
    # to exercise the wiring guard when the resolved list is empty.
    s = Settings.model_construct(strategy=StrategyConfig.model_construct(enabled=[]))
    with pytest.raises(ValueError, match="No valid strategies"):
        build_strategies_from_enabled(s)


def test_build_strategies_only_workflow_only_keys_raises() -> None:
    """``tax_loss_harvest`` is skipped in the factory loop — no strategies left."""
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        strategy=StrategyConfig(enabled=["tax_loss_harvest"]),
    )
    with pytest.raises(ValueError, match="No valid strategies"):
        build_strategies_from_enabled(s)


def test_build_strategies_unknown_name_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        strategy=StrategyConfig(enabled=["momentum", "not_a_strategy"]),
        data=DataConfig(universe=["SPY"]),
    )
    out = build_strategies_from_enabled(s)
    assert len(out) == 1
    assert isinstance(out[0], MomentumRotationStrategy)
    assert any("Unknown strategy" in r.message for r in caplog.records)


def test_build_strategies_deduplicates_repeated_names() -> None:
    s = Settings(
        _yaml_path=None,
        _env_file=None,
        strategy=StrategyConfig(enabled=["momentum", "dca", "momentum"]),
        data=DataConfig(
            universe=["SPY"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
    )
    strats = build_strategies_from_enabled(s)
    assert [type(x).__name__ for x in strats] == [
        "MomentumRotationStrategy",
        "DCAStrategy",
    ]
