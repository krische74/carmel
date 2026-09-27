"""Strategy interface is polymorphic for downstream orchestration."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import DataConfig, DCAConfig, DCATarget, Settings, StrategyConfig
from src.models import Signal
from src.strategy.base import Strategy
from src.strategy.dca import DCAStrategy
from src.strategy.momentum import MomentumRotationStrategy


def _empty_panel() -> dict[str, pd.DataFrame]:
    return {}


def _run_any(strategy: Strategy, data: dict[str, pd.DataFrame]) -> list[Signal]:
    return strategy.generate_signals(data, as_of=datetime(2024, 1, 8, 16, 0, tzinfo=UTC))


def test_pipeline_accepts_any_strategy_subclass() -> None:
    dca = DCAStrategy(
        settings=Settings(
            _yaml_path=Path("config/settings.yaml"),
            _env_file=None,
            data=DataConfig(
                dca_targets=[
                    DCATarget(symbol="VOO", weight=1.0),
                ],
            ),
            strategy=StrategyConfig(dca=DCAConfig(frequency="weekly")),
        ),
    )
    mom = MomentumRotationStrategy(
        settings=Settings(
            data=DataConfig(universe=["SPY", "QQQ", "TLT", "GLD"]),
            strategy=StrategyConfig(),
        ),
    )
    for strat in (dca, mom):
        sigs = _run_any(strat, _empty_panel())
        assert isinstance(sigs, list)
        for s in sigs:
            assert isinstance(s, Signal)
