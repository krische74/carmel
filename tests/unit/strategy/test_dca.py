"""Unit tests for DCAStrategy."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.config import DataConfig, DCAConfig, DCATarget, Settings, StrategyConfig
from src.data.regime import (
    MarketRegime,
    OverallRegime,
    VolatilityRegime,
    YieldCurveRegime,
)
from src.strategy.dca import DCAStrategy
from src.strategy.regime_params import effective_dca_amount


def _minimal_ohlcv(rows: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.0] * rows,
            "volume": [1e6] * rows,
        },
        index=idx,
    )


def test_dca_equal_weight_signals_on_scheduled_day() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            parquet_dir="data/parquet",
            dca_targets=[
                DCATarget(symbol="VOO", weight=1.0 / 3.0),
                DCATarget(symbol="VXUS", weight=1.0 / 3.0),
                DCATarget(symbol="BND", weight=1.0 / 3.0),
            ],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly", amount=100.0)),
    )
    strat = DCAStrategy(settings=settings)
    data = {
        "VOO": _minimal_ohlcv(),
        "VXUS": _minimal_ohlcv(),
        "BND": _minimal_ohlcv(),
    }
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)  # Monday
    signals = strat.generate_signals(data, as_of=monday)
    assert len(signals) == 3
    weights = sorted(s.weight for s in signals)
    for w in weights:
        assert abs(w - 1.0 / 3.0) < 1e-5
    for s in signals:
        assert s.direction == "long"
        assert s.strategy_name == "DCAStrategy"
        assert len(s.rationale) >= 20
        assert "DCA" in s.rationale or "dollar" in s.rationale.lower()


def test_dca_no_signals_off_schedule() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[
                DCATarget(symbol="VOO", weight=0.6),
                DCATarget(symbol="VXUS", weight=0.3),
                DCATarget(symbol="BND", weight=0.1),
            ],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly")),
    )
    strat = DCAStrategy(settings=settings)
    data = {s: _minimal_ohlcv() for s in ("VOO", "VXUS", "BND")}
    tuesday = datetime(2024, 1, 9, 16, 0, tzinfo=UTC)
    assert strat.generate_signals(data, as_of=tuesday) == []


def test_dca_get_universe_lists_targets() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[
                DCATarget(symbol="VOO", weight=0.5),
                DCATarget(symbol="BND", weight=0.5),
            ],
        ),
    )
    strat = DCAStrategy(settings=settings)
    assert strat.get_universe() == ["VOO", "BND"]


def test_dca_fires_on_daily_schedule() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="daily", amount=50.0)),
    )
    strat = DCAStrategy(settings=settings)
    data = {"VOO": _minimal_ohlcv()}
    tuesday = datetime(2024, 1, 9, 12, 0, tzinfo=UTC)
    sigs = strat.generate_signals(data, as_of=tuesday)
    assert len(sigs) == 1
    assert sigs[0].strategy_name == "DCAStrategy"


def test_dca_fires_on_monthly_first_monday() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="monthly", amount=200.0)),
    )
    strat = DCAStrategy(settings=settings)
    data = {"VOO": _minimal_ohlcv()}
    first_monday_jan_2024 = datetime(2024, 1, 1, 15, 0, tzinfo=UTC)
    sigs = strat.generate_signals(data, as_of=first_monday_jan_2024)
    assert len(sigs) == 1
    assert sigs[0].symbol == "VOO"


def test_dca_skips_monthly_when_not_first_week_monday() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="monthly")),
    )
    strat = DCAStrategy(settings=settings)
    data = {"VOO": _minimal_ohlcv()}
    second_monday_jan_2024 = datetime(2024, 1, 8, 15, 0, tzinfo=UTC)
    assert strat.generate_signals(data, as_of=second_monday_jan_2024) == []


def test_dca_empty_targets_returns_empty() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(dca_targets=[]),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly")),
    )
    strat = DCAStrategy(settings=settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    assert strat.generate_signals({}, as_of=monday) == []


def test_dca_naive_as_of_is_treated_as_utc() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly")),
    )
    strat = DCAStrategy(settings=settings)
    monday_naive = datetime(2024, 1, 8, 16, 0)
    sigs = strat.generate_signals({"VOO": _minimal_ohlcv()}, as_of=monday_naive)
    assert len(sigs) == 1
    assert sigs[0].timestamp.tzinfo is not None


def test_dca_unknown_frequency_falls_back_to_monday() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[DCATarget(symbol="VOO", weight=1.0)],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="custom_quarterly")),
    )
    strat = DCAStrategy(settings=settings)
    data = {"VOO": _minimal_ohlcv()}
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    tuesday = datetime(2024, 1, 9, 16, 0, tzinfo=UTC)
    assert len(strat.generate_signals(data, as_of=monday)) == 1
    assert strat.generate_signals(data, as_of=tuesday) == []


def test_dca_zero_total_weight_returns_empty() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[
                DCATarget(symbol="VOO", weight=0.0),
                DCATarget(symbol="VXUS", weight=0.0),
            ],
        ),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly")),
    )
    strat = DCAStrategy(settings=settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    assert strat.generate_signals({"VOO": _minimal_ohlcv()}, as_of=monday) == []


def _market_regime(overall: OverallRegime, *, vix: float = 22.0) -> MarketRegime:
    return MarketRegime(
        timestamp=datetime(2024, 1, 8, 16, 0, tzinfo=UTC),
        vix_close=vix,
        yield_spread=None,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=overall,
        sizing_multiplier=1.0,
    )


def _crisis_market_regime() -> MarketRegime:
    return MarketRegime(
        timestamp=datetime(2024, 1, 8, 16, 0, tzinfo=UTC),
        vix_close=55.0,
        yield_spread=None,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.CRISIS,
        overall=OverallRegime.CRISIS,
        sizing_multiplier=0.25,
    )


def test_dca_signals_with_crisis_regime() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(dca_targets=[DCATarget(symbol="VOO", weight=1.0)]),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly", amount=100.0)),
    )
    strat = DCAStrategy(settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    sigs = strat.generate_signals(
        {"VOO": _minimal_ohlcv()},
        as_of=monday,
        market_regime=_crisis_market_regime(),
    )
    assert len(sigs) == 1
    assert sigs[0].weight == pytest.approx(0.25)
    assert "$25" in sigs[0].rationale
    assert "Regime-adjusted" in sigs[0].rationale


def test_dca_signals_cautious_regime() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(dca_targets=[DCATarget(symbol="VOO", weight=1.0)]),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly", amount=100.0)),
    )
    strat = DCAStrategy(settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    sigs = strat.generate_signals(
        {"VOO": _minimal_ohlcv()},
        as_of=monday,
        market_regime=_market_regime(OverallRegime.CAUTIOUS),
    )
    assert len(sigs) == 1
    assert sigs[0].weight == pytest.approx(0.75)
    assert "$75" in sigs[0].rationale


def test_dca_signals_defensive_regime() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(dca_targets=[DCATarget(symbol="VOO", weight=1.0)]),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly", amount=100.0)),
    )
    strat = DCAStrategy(settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    sigs = strat.generate_signals(
        {"VOO": _minimal_ohlcv()},
        as_of=monday,
        market_regime=_market_regime(OverallRegime.DEFENSIVE),
    )
    assert len(sigs) == 1
    assert sigs[0].weight == pytest.approx(0.5)
    assert "$50" in sigs[0].rationale


def test_dca_signals_risk_on_matches_no_regime() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(dca_targets=[DCATarget(symbol="VOO", weight=1.0)]),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly", amount=100.0)),
    )
    strat = DCAStrategy(settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    data = {"VOO": _minimal_ohlcv()}
    sigs_base = strat.generate_signals(data, as_of=monday)
    sigs_risk = strat.generate_signals(
        data,
        as_of=monday,
        market_regime=_market_regime(OverallRegime.RISK_ON),
    )
    assert len(sigs_base) == len(sigs_risk) == 1
    assert sigs_base[0].weight == sigs_risk[0].weight
    assert sigs_base[0].rationale == sigs_risk[0].rationale


def test_dca_zero_crisis_multiplier_emits_no_signals() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(dca_targets=[DCATarget(symbol="VOO", weight=1.0)]),
        strategy=StrategyConfig(
            dca=DCAConfig(
                frequency="weekly",
                amount=100.0,
                regime_amount_crisis=0.0,
            ),
        ),
    )
    strat = DCAStrategy(settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    assert (
        strat.generate_signals(
            {"VOO": _minimal_ohlcv()},
            as_of=monday,
            market_regime=_crisis_market_regime(),
        )
        == []
    )


def test_dca_signals_without_regime_unchanged() -> None:
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(dca_targets=[DCATarget(symbol="VOO", weight=1.0)]),
        strategy=StrategyConfig(dca=DCAConfig(frequency="weekly", amount=100.0)),
    )
    strat = DCAStrategy(settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    sigs = strat.generate_signals({"VOO": _minimal_ohlcv()}, as_of=monday)
    assert len(sigs) == 1
    assert sigs[0].weight == pytest.approx(1.0)
    assert "$100" in sigs[0].rationale
    assert "Regime-adjusted" not in sigs[0].rationale


def test_dca_percent_of_equity_budget_splits_notionals_by_weights() -> None:
    from src.risk.position_sizing import compute_order_notional

    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(
            dca_targets=[
                DCATarget(symbol="VOO", weight=0.6),
                DCATarget(symbol="VXUS", weight=0.3),
                DCATarget(symbol="BND", weight=0.1),
            ],
        ),
        strategy=StrategyConfig(
            dca=DCAConfig(
                frequency="weekly",
                amount=100.0,
                percent_of_equity=0.005,
            ),
        ),
    )
    strat = DCAStrategy(settings=settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    signals = strat.generate_signals(
        {"VOO": _minimal_ohlcv(), "VXUS": _minimal_ohlcv(), "BND": _minimal_ohlcv()},
        as_of=monday,
    )
    assert len(signals) == 3
    dca_budget = effective_dca_amount(settings, None, equity=100_000.0)
    notional_by_symbol = {
        s.symbol: compute_order_notional(
            signal_weight=s.weight,
            equity=100_000.0,
            max_position_pct=0.25,
            dca_budget=dca_budget,
            use_dca_budget=True,
        )
        for s in signals
    }
    assert notional_by_symbol["VOO"] == pytest.approx(300.0)
    assert notional_by_symbol["VXUS"] == pytest.approx(150.0)
    assert notional_by_symbol["BND"] == pytest.approx(50.0)


def test_dca_amount_fallback_when_percent_of_equity_not_set() -> None:
    from src.risk.position_sizing import compute_order_notional

    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
        data=DataConfig(dca_targets=[DCATarget(symbol="VOO", weight=1.0)]),
        strategy=StrategyConfig(
            dca=DCAConfig(
                frequency="weekly",
                amount=100.0,
                percent_of_equity=None,
            ),
        ),
    )
    strat = DCAStrategy(settings=settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    signals = strat.generate_signals({"VOO": _minimal_ohlcv()}, as_of=monday)
    assert len(signals) == 1
    dca_budget = effective_dca_amount(settings, None, equity=100_000.0)
    assert dca_budget == pytest.approx(100.0)
    notional = compute_order_notional(
        signal_weight=signals[0].weight,
        equity=100_000.0,
        max_position_pct=0.25,
        dca_budget=dca_budget,
        use_dca_budget=True,
    )
    assert notional == pytest.approx(100.0)


def test_default_yaml_dca_is_monday_only_at_2_5_pct() -> None:
    """Tier 45B: default settings.yaml → weekly Mondays at 2.5% of equity."""
    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
    )
    assert settings.strategy.dca.frequency == "weekly"
    assert settings.strategy.dca.percent_of_equity == pytest.approx(0.025)

    strat = DCAStrategy(settings=settings)
    data = {t.symbol: _minimal_ohlcv() for t in settings.data.dca_targets}
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    tuesday = datetime(2024, 1, 9, 16, 0, tzinfo=UTC)
    assert len(strat.generate_signals(data, as_of=monday)) == len(settings.data.dca_targets)
    assert strat.generate_signals(data, as_of=tuesday) == []


def test_weekly_dca_legs_clear_min_order_floor_under_cautious() -> None:
    """At a small account equity, weekly 2.5% * cautious 0.75 -> all three legs clear $5.

    Budget is equity*pct (no regime); strategy weights apply the cautious scale once
    (same pattern as test_dca_percent_of_equity_with_cautious_regime_no_double_scaling).
    """
    from src.risk.position_sizing import compute_order_notional

    settings = Settings(
        _yaml_path=Path("config/settings.yaml"),
        _env_file=None,
    )
    equity = 3_437.0
    floor = float(settings.risk.min_order_notional_usd)
    regime = MarketRegime(
        timestamp=datetime(2024, 1, 8, 16, 0, tzinfo=UTC),
        vix_close=18.0,
        yield_spread=None,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.CAUTIOUS,
        sizing_multiplier=0.75,
    )
    strat = DCAStrategy(settings=settings)
    monday = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    data = {t.symbol: _minimal_ohlcv() for t in settings.data.dca_targets}
    signals = strat.generate_signals(data, as_of=monday, market_regime=regime)
    assert len(signals) == 3

    # Base budget (regime applied via signal weights, not here).
    budget = effective_dca_amount(settings, None, equity=equity)
    assert budget == pytest.approx(85.925)
    # Effective weekly outlay under cautious ≈ 85.925 * 0.75 ≈ 64.44
    notionals = {
        s.symbol: compute_order_notional(
            signal_weight=s.weight,
            equity=equity,
            max_position_pct=float(settings.risk.max_position_pct),
            dca_budget=budget,
            use_dca_budget=True,
        )
        for s in signals
    }
    assert notionals["VOO"] == pytest.approx(budget * 0.6 * 0.75)
    assert notionals["VXUS"] == pytest.approx(budget * 0.3 * 0.75)
    assert notionals["BND"] == pytest.approx(budget * 0.1 * 0.75)
    assert all(n >= floor for n in notionals.values())
