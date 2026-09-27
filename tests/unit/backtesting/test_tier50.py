"""Tier 50: production-faithful backtest (risk layer, cash yield, benchmark)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path  # noqa: TC003

import numpy as np
import pandas as pd
import pytest

from src.automation.runner import load_backtest_data
from src.backtesting.engine import BacktestConfig, BacktestEngine, compute_benchmark_metrics
from src.config import DataConfig, MomentumConfig, RiskConfig, Settings, StrategyConfig
from src.models import Signal
from src.risk.position_sizing import compute_order_notional, compute_target_position_notional
from src.strategy.base import Strategy


class _NoSignalsStrategy(Strategy):
    def get_universe(self) -> list[str]:
        return ["SPY"]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: object | None = None,
    ) -> list[Signal]:
        return []


class _FullLongStrategy(Strategy):
    """Always 100% long one symbol (drives allocation-cap tests)."""

    def get_universe(self) -> list[str]:
        return ["SPY"]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: object | None = None,
    ) -> list[Signal]:
        when = as_of or datetime.now(UTC)
        return [
            Signal(
                symbol="SPY",
                direction="long",
                weight=1.0,
                confidence=1.0,
                rationale="Full long for cap test.",
                timestamp=when,
                strategy_name="FullLong",
            ),
        ]


class _AlternatingLongStrategy(Strategy):
    """Alternates full long between SPY and QQQ each rebalance."""

    def __init__(self) -> None:
        self._prefer_qqq = False

    def get_universe(self) -> list[str]:
        return ["SPY", "QQQ"]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: object | None = None,
    ) -> list[Signal]:
        when = as_of or datetime.now(UTC)
        sym = "QQQ" if self._prefer_qqq else "SPY"
        self._prefer_qqq = not self._prefer_qqq
        return [
            Signal(
                symbol=sym,
                direction="long",
                weight=1.0,
                confidence=0.5,
                rationale="Alternating for CLI pin test.",
                timestamp=when,
                strategy_name="AlternatingLong",
            ),
        ]


def _trending_ohlcv(n: int, start: date, *, drift: float = 0.2) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n, freq="B")
    close = 100.0 + np.linspace(0.0, drift * n, n, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_target_position_notional_matches_order_notional_at_zero_current() -> None:
    """Backtest rebalance targets must match the live sizing function at zero current."""
    equity = 10_000.0
    kwargs = dict(
        signal_weight=1.0,
        equity=equity,
        max_position_pct=0.25,
        regime_multiplier=0.75,
    )
    target = compute_target_position_notional(**kwargs)
    delta = compute_order_notional(
        **kwargs,
        dca_budget=None,
        use_dca_budget=False,
        current_position_notional=0.0,
    )
    assert target == pytest.approx(delta)
    assert target == pytest.approx(0.25 * equity)


def test_cli_loader_pins_trades_on_fixed_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tier 50D: CLI data-loading path is deterministic on a fixed Parquet fixture."""
    import src.automation.runner as runner_mod
    from src.data.storage.parquet_store import ParquetStore

    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY", "QQQ"],
        ),
        strategy=StrategyConfig(
            momentum=MomentumConfig(
                lookback_months=[1],
                sma_filter_period=10,
                cash_symbol="SHV",
                adx_filter_period=5,
                adx_threshold=0.0,
            ),
        ),
    )
    pq = ParquetStore(settings.data.parquet_dir)
    start_d = date(2024, 1, 2)
    n = 80
    for sym, drift in (("SPY", 0.4), ("QQQ", 0.05)):
        pq.write_ohlcv(sym, _trending_ohlcv(n, start_d, drift=drift))

    monkeypatch.setattr(
        runner_mod,
        "build_strategy_for_backtest",
        lambda _settings, _name: _AlternatingLongStrategy(),
    )
    strat, data = load_backtest_data(settings, "momentum", parquet=pq)
    cfg = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=False,
        cash_yield_annual_pct=0.0,
        use_regime=False,
    )
    result = BacktestEngine().run(
        strat,
        data,
        start=start_d,
        end=date(2024, 4, 30),
        config=cfg,
    )
    assert len(result.trades) == 33
    assert result.final_equity == pytest.approx(11_530.070_343_699_887, rel=1e-9)


def test_tier49_partial_universe_missing_tlt_gld() -> None:
    """Tier 50D: document Tier 49 in-process divergence — wrong symbol set."""
    settings = Settings(_yaml_path=None, _env_file=None)
    from src.automation.strategy_wiring import build_strategy_for_backtest

    full = build_strategy_for_backtest(settings, "momentum").get_universe()
    tier49_ad_hoc = {"SPY", "QQQ", "SHV", "^VIX"}
    assert {"TLT", "GLD"}.issubset(set(full))
    assert not {"TLT", "GLD"}.issubset(tier49_ad_hoc)
    assert len(full) > len(tier49_ad_hoc)


def test_cash_yield_on_idle_cash(tmp_path: Path) -> None:
    """100% cash backtest accrues configured yield instead of returning 0%."""
    start_d = date(2024, 1, 2)
    n = 252
    spy = _trending_ohlcv(n, start_d, drift=0.0)
    data = {"SPY": spy}
    cfg = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=False,
        cash_yield_annual_pct=4.0,
        use_regime=False,
        min_coverage_ratio=0.0,
    )
    result = BacktestEngine().run(
        _NoSignalsStrategy(),
        data,
        start=start_d,
        end=date(2024, 12, 31),
        config=cfg,
    )
    assert not result.trades
    n_days = len(result.equity_curve) - 1
    expected = 10_000.0 * (1.0 + 0.04 / 252.0) ** n_days
    assert result.final_equity == pytest.approx(expected, rel=0.02)


def test_constrained_backtest_caps_peak_allocation(tmp_path: Path) -> None:
    """Peak single-symbol target weight must not exceed max_position_pct under risk layer."""
    start_d = date(2024, 1, 2)
    n = 60
    spy = _trending_ohlcv(n, start_d, drift=0.3)
    data = {"SPY": spy}
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        risk=RiskConfig(max_position_pct=0.25, min_cash_reserve_pct=0.05, min_order_notional_usd=5.0),
    )
    cfg = BacktestConfig(
        rebalance_frequency="weekly",
        apply_risk_layer=True,
        cash_yield_annual_pct=0.0,
        use_regime=False,
    )
    result = BacktestEngine().run(
        _FullLongStrategy(),
        data,
        start=start_d,
        end=date(2024, 3, 29),
        config=cfg,
        settings=settings,
    )
    assert result.peak_single_symbol_allocation_pct <= settings.risk.max_position_pct + 1e-6
    assert result.peak_single_symbol_allocation_pct > 0.20


def test_raw_signal_mode_matches_unconstrained_weight(tmp_path: Path) -> None:
    """--raw-signal (apply_risk_layer=False) keeps full weight-to-equity mapping."""
    start_d = date(2024, 1, 2)
    n = 60
    spy = _trending_ohlcv(n, start_d, drift=0.3)
    data = {"SPY": spy}
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        risk=RiskConfig(max_position_pct=0.25),
    )
    raw_cfg = BacktestConfig(rebalance_frequency="weekly", apply_risk_layer=False, cash_yield_annual_pct=0.0)
    raw = BacktestEngine().run(
        _FullLongStrategy(),
        data,
        start=start_d,
        end=date(2024, 3, 29),
        config=raw_cfg,
        settings=settings,
    )
    assert raw.peak_single_symbol_allocation_pct > 0.90


def test_backtest_result_records_sizing_mode(tmp_path: Path) -> None:
    start_d = date(2024, 1, 2)
    spy = _trending_ohlcv(40, start_d)
    data = {"SPY": spy}
    settings = Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            parquet_dir=str(tmp_path / "pq"),
            cache_dir=str(tmp_path / "cache"),
            universe=["SPY"],
        ),
        risk=RiskConfig(max_position_pct=0.25),
    )
    cfg = BacktestConfig(
        apply_risk_layer=True,
        cash_yield_annual_pct=3.0,
        min_coverage_ratio=0.0,
    )
    result = BacktestEngine().run(
        _NoSignalsStrategy(),
        data,
        start=start_d,
        end=date(2024, 2, 29),
        config=cfg,
        settings=settings,
    )
    assert result.sizing_mode == "constrained"
    assert result.cash_yield_annual_pct == pytest.approx(3.0)


def test_compute_benchmark_metrics_buy_and_hold() -> None:
    start_d = date(2024, 1, 2)
    n = 40
    spy = _trending_ohlcv(n, start_d, drift=0.5)
    days = [d.date() for d in spy.index]
    m = compute_benchmark_metrics({"SPY": spy}, "SPY", days, risk_free_rate_annual=4.0)
    assert m is not None
    assert m.total_return_pct > 0.0
    assert m.trading_days == len(days) - 1
