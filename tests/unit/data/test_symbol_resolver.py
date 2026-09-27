"""Unit tests for canonical OHLCV symbol resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import (
    CashSweepConfig,
    DataConfig,
    DCATarget,
    MeanReversionConfig,
    MomentumConfig,
    Settings,
    StrategyConfig,
    TaxConfig,
)
from src.data.symbol_resolver import (
    resolve_required_symbols,
    resolve_required_symbols_with_sources,
)

if TYPE_CHECKING:
    from pathlib import Path


def _empty_yaml_settings(tmp_path: Path, **kwargs) -> Settings:
    """Settings with no merged YAML (isolated unit tests)."""
    p = tmp_path / "missing.yaml"
    assert not p.exists()
    # Sweep is off by default here so per-source tests stay isolated; sweep-specific
    # tests opt in with cash_sweep=CashSweepConfig(enabled=True, ...).
    base = dict(_yaml_path=p, _env_file=None, cash_sweep=CashSweepConfig(enabled=False))
    base.update(kwargs)
    return Settings(**base)


def test_resolve_includes_universe(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(
            universe=["SPY", "QQQ", "TLT", "GLD"],
            dca_targets=[],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    assert resolve_required_symbols(s) == {"SPY", "QQQ", "TLT", "GLD"}


def test_resolve_includes_dca_targets(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(
            universe=[],
            dca_targets=[
                DCATarget(symbol="VOO", weight=0.6),
                DCATarget(symbol="VXUS", weight=0.3),
                DCATarget(symbol="BND", weight=0.1),
            ],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    assert resolve_required_symbols(s) == {"VOO", "VXUS", "BND"}


def test_resolve_includes_momentum_cash_symbol(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=[], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol="SHV"),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    assert resolve_required_symbols(s) == {"SHV"}


def test_resolve_includes_mean_reversion_universe(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=["AAA"], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=["BBB", "CCC"]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    assert resolve_required_symbols(s) == {"AAA", "BBB", "CCC"}


def test_resolve_includes_tax_replacement_map(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=[], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={"SPY": "IVV", "QQQ": "QQQM", "TLT": "VGLT"}),
    )
    assert resolve_required_symbols(s) == {"SPY", "IVV", "QQQ", "QQQM", "TLT", "VGLT"}


def test_resolve_normalizes_case_and_whitespace(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=[" spy "], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    assert resolve_required_symbols(s) == {"SPY"}


def test_resolve_dedupes_overlapping_sources(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=["SPY", "QQQ"], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=["SPY", "TLT"]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    assert resolve_required_symbols(s) == {"SPY", "QQQ", "TLT"}
    assert len(resolve_required_symbols(s)) == 3


def test_resolve_returns_set_not_list(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=["SPY"], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    out = resolve_required_symbols(s)
    assert isinstance(out, set)


def test_resolve_handles_empty_sections(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=[], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
    )
    assert resolve_required_symbols(s) == set()
    assert resolve_required_symbols_with_sources(s) == {}


def test_resolve_with_sources_returns_diagnostic_mapping(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(
            universe=["SPY", "QQQ"],
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol="SHV"),
            mean_reversion=MeanReversionConfig(universe=["TLT", "GLD"]),
        ),
        tax=TaxConfig(replacement_map={"SPY": "IVV"}),
    )
    m = resolve_required_symbols_with_sources(s)
    assert m["data.universe"] == {"SPY", "QQQ"}
    assert m["data.dca_targets"] == {"VOO"}
    assert m["momentum.cash_symbol"] == {"SHV"}
    assert m["mean_reversion.universe"] == {"TLT", "GLD"}
    assert m["tax.replacement_map"] == {"SPY", "IVV"}
    assert resolve_required_symbols(s) == {"SPY", "QQQ", "VOO", "SHV", "TLT", "GLD", "IVV"}


def test_resolve_includes_sweep_symbol_when_enabled(tmp_path: Path) -> None:
    """Tier 44B: the sweep vehicle needs OHLCV every cycle, so it is a required symbol."""
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=["SPY"], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
        cash_sweep=CashSweepConfig(enabled=True, symbol="BIL"),
    )
    m = resolve_required_symbols_with_sources(s)
    assert m["cash_sweep.symbol"] == {"BIL"}
    assert "BIL" in resolve_required_symbols(s)


def test_resolve_excludes_sweep_symbol_when_disabled(tmp_path: Path) -> None:
    s = _empty_yaml_settings(
        tmp_path,
        data=DataConfig(universe=["SPY"], dca_targets=[]),
        strategy=StrategyConfig(
            momentum=MomentumConfig(cash_symbol=""),
            mean_reversion=MeanReversionConfig(universe=[]),
        ),
        tax=TaxConfig(replacement_map={}),
        cash_sweep=CashSweepConfig(enabled=False, symbol="BIL"),
    )
    assert "BIL" not in resolve_required_symbols(s)
    assert "cash_sweep.symbol" not in resolve_required_symbols_with_sources(s)
