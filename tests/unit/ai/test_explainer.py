"""Tests for template-based decision explanations."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.automation.alerts import Alert, AlertLevel
from src.config import DataConfig, DCATarget, Settings
from src.models import OrderExecutionResult, Signal
from src.portfolio.state import PortfolioSnapshot
from src.reporting.attribution import PositionContribution


def _settings_minimal() -> Settings:
    return Settings(
        _yaml_path=None,
        _env_file=None,
        data=DataConfig(
            universe=["SPY", "QQQ", "TLT", "GLD"],
            dca_targets=[
                DCATarget(symbol="VOO", weight=0.6),
                DCATarget(symbol="VXUS", weight=0.3),
                DCATarget(symbol="BND", weight=0.1),
            ],
        ),
    )


def _ohlcv_rows(
    n: int = 500,
    *,
    close_start: float = 100.0,
    end: datetime | None = None,
) -> pd.DataFrame:
    end_ts = end or datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    idx = pd.date_range(end=end_ts, periods=n, freq="B", tz="UTC")
    close = np.linspace(close_start, close_start * 1.15, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
        },
        index=idx,
    )


def test_explain_momentum_long_signal_includes_return_and_sma() -> None:
    from src.ai.explainer import generate_signal_explanation

    settings = _settings_minimal()
    when = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    data = {
        s: _ohlcv_rows(500, close_start=120.0 + i * 5, end=when)
        for i, s in enumerate(["SPY", "QQQ", "TLT", "GLD"])
    }
    sig = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.75,
        rationale="stub",
        timestamp=when,
        strategy_name="MomentumRotationStrategy",
    )
    text = generate_signal_explanation(sig, data, settings)
    assert "QQQ" in text
    assert "SMA" in text or "day" in text.lower()
    assert "ADX" in text


def test_explain_momentum_cash_signal_explains_why() -> None:
    from src.ai.explainer import generate_signal_explanation

    settings = _settings_minimal()
    when = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    data = {s: _ohlcv_rows(500, close_start=50.0, end=when) for s in ["SPY", "QQQ", "TLT", "GLD"]}
    sig = Signal(
        symbol="SHV",
        direction="cash",
        weight=1.0,
        confidence=0.85,
        rationale="stub",
        timestamp=when,
        strategy_name="MomentumRotationStrategy",
    )
    text = generate_signal_explanation(sig, data, settings)
    lowered = text.lower()
    assert "etf" in lowered or "sma" in lowered or "adx" in lowered or "cash" in lowered


def test_explain_dca_signal_includes_amount_and_schedule() -> None:
    from src.ai.explainer import generate_signal_explanation

    settings = _settings_minimal()
    settings.strategy.dca.percent_of_equity = None  # exercise the fixed-amount fallback
    when = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    sig = Signal(
        symbol="VOO",
        direction="long",
        weight=0.6,
        confidence=1.0,
        rationale="stub",
        timestamp=when,
        strategy_name="DCAStrategy",
    )
    text = generate_signal_explanation(sig, {}, settings)
    assert "DCA" in text or "contribution" in text.lower()
    assert (
        str(int(settings.strategy.dca.amount)) in text
        or f"{settings.strategy.dca.amount:.2f}" in text
    )
    assert settings.strategy.dca.frequency in text


def test_explain_dca_signal_mentions_percent_of_equity_when_configured() -> None:
    from src.ai.explainer import generate_signal_explanation

    settings = _settings_minimal()
    settings.strategy.dca.percent_of_equity = 0.005
    when = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    sig = Signal(
        symbol="VOO",
        direction="long",
        weight=0.6,
        confidence=1.0,
        rationale="stub",
        timestamp=when,
        strategy_name="DCAStrategy",
    )
    text = generate_signal_explanation(sig, {}, settings)
    assert "0.50%" in text
    assert "equity" in text.lower()
    assert "$100" not in text


def test_explain_mean_reversion_long_includes_bollinger_and_rsi() -> None:
    from src.ai.explainer import generate_signal_explanation

    settings = _settings_minimal()
    when = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    df = _ohlcv_rows(500, close_start=400.0, end=when)
    df["close"] = df["close"] * 0.5
    df["low"] = df["low"] * 0.5
    df["high"] = df["high"] * 0.5
    df["open"] = df["open"] * 0.5
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.25,
        confidence=0.7,
        rationale="stub",
        timestamp=when,
        strategy_name="MeanReversionStrategy",
    )
    text = generate_signal_explanation(sig, {"SPY": df}, settings)
    assert "Bollinger" in text or "bollinger" in text.lower()
    assert "RSI" in text


def test_explain_signal_falls_back_to_rationale_when_data_missing() -> None:
    from src.ai.explainer import generate_signal_explanation

    settings = _settings_minimal()
    when = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    sig = Signal(
        symbol="QQQ",
        direction="long",
        weight=1.0,
        confidence=0.75,
        rationale="only this rationale",
        timestamp=when,
        strategy_name="MomentumRotationStrategy",
    )
    text = generate_signal_explanation(sig, {}, settings)
    assert text == "only this rationale"


def test_cycle_summary_includes_signal_and_execution_counts() -> None:
    from src.ai.explainer import generate_cycle_summary

    when = datetime(2026, 4, 4, 17, 30, tzinfo=UTC)
    sigs = [
        Signal(
            symbol="QQQ",
            direction="long",
            weight=1.0,
            confidence=0.8,
            rationale="a",
            timestamp=when,
            strategy_name="MomentumRotationStrategy",
        ),
        Signal(
            symbol="VOO",
            direction="long",
            weight=0.5,
            confidence=1.0,
            rationale="b",
            timestamp=when,
            strategy_name="DCAStrategy",
        ),
    ]
    exes = [
        OrderExecutionResult(symbol="QQQ", submitted=True, order_id="1", side="buy"),
        OrderExecutionResult(symbol="VOO", submitted=False, reason="risk", side="buy"),
    ]
    text = generate_cycle_summary(sigs, exes, None, None)
    assert "Signals: 2" in text
    assert "Executed: 1" in text
    assert "Rejected: 1" in text


def test_cycle_summary_includes_equity_when_snapshot_provided() -> None:
    from src.ai.explainer import generate_cycle_summary

    when = datetime(2026, 4, 4, 17, 30, tzinfo=UTC)
    sig = Signal(
        symbol="X",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="a",
        timestamp=when,
        strategy_name="DCAStrategy",
    )
    snap = PortfolioSnapshot(equity=10_000.0, cash=2_000.0)
    text = generate_cycle_summary([sig], [], snap, None)
    assert "$10,000.00" in text


def test_cycle_summary_mentions_alerts_when_present() -> None:
    from src.ai.explainer import generate_cycle_summary

    when = datetime(2026, 4, 4, 17, 30, tzinfo=UTC)
    sig = Signal(
        symbol="X",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="a",
        timestamp=when,
        strategy_name="DCAStrategy",
    )
    alerts = [
        Alert(
            timestamp=when,
            level=AlertLevel.WARNING,
            category="test",
            message="near limit",
        ),
    ]
    text = generate_cycle_summary([sig], [], None, alerts)
    assert "warning" in text.lower()


def test_cycle_summary_handles_empty_cycle() -> None:
    from src.ai.explainer import generate_cycle_summary

    text = generate_cycle_summary([], [], None, None)
    assert "Signals: 0" in text
    assert "no activity" in text.lower()


def test_enrich_signal_rationale_replaces_field() -> None:
    from src.ai.explainer import enrich_signal_rationale

    when = datetime(2026, 4, 4, 17, 30, tzinfo=UTC)
    sig = Signal(
        symbol="X",
        direction="long",
        weight=1.0,
        confidence=1.0,
        rationale="above SMA",
        timestamp=when,
        strategy_name="MomentumRotationStrategy",
    )
    enriched = enrich_signal_rationale(sig, "Full template text with details.")
    assert enriched.rationale == "Full template text with details."
    assert sig.rationale == "above SMA"


def test_weekly_digest_includes_return_and_top_contributor() -> None:
    from src.ai.explainer import generate_weekly_digest

    snaps = [
        {"date": "2026-03-29", "total_market_value": 10_000.0, "total_pnl": 0.0},
        {"date": "2026-04-04", "total_market_value": 10_230.0, "total_pnl": 230.0},
    ]
    execs = [{"side": "buy", "submitted": True}, {"side": "sell", "submitted": True}]
    alts: list[dict[str, str]] = []
    contribs = [
        PositionContribution(
            symbol="TLT",
            total_qty=5.0,
            realized_pnl=0.0,
            unrealized_pnl=-20.0,
            total_pnl=-20.0,
            weight_pct=10.0,
            contribution_pct=-0.2,
        ),
        PositionContribution(
            symbol="QQQ",
            total_qty=10.0,
            realized_pnl=0.0,
            unrealized_pnl=180.0,
            total_pnl=180.0,
            weight_pct=40.0,
            contribution_pct=1.8,
        ),
    ]
    text = generate_weekly_digest(snaps, execs, alts, contribs)
    assert "Top contributor:" in text
    assert "QQQ" in text
    assert "+$180" in text
    assert "%" in text


def test_weekly_digest_handles_empty_week() -> None:
    from src.ai.explainer import generate_weekly_digest

    text = generate_weekly_digest([], [], [], None)
    assert "No data" in text


def test_explain_harvest_replacement_includes_original_symbol() -> None:
    """TLH template parses rationale for harvested symbol and names replacement."""
    from src.ai.explainer import generate_signal_explanation

    settings = _settings_minimal()
    when = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    sig = Signal(
        symbol="VOO",
        direction="long",
        weight=0.1,
        confidence=1.0,
        rationale="TLH replacement for SPY: test. Estimated unrealized loss of $200.00.",
        timestamp=when,
        strategy_name="TaxLossHarvest",
    )
    text = generate_signal_explanation(sig, {}, settings)
    assert "VOO" in text
    assert "SPY" in text
    assert "harvested SPY" in text
    assert "wash-sale" in text.lower()


def test_cycle_summary_includes_regime_when_provided() -> None:
    from src.ai.explainer import generate_cycle_summary
    from src.data.regime import (
        MarketRegime,
        OverallRegime,
        VolatilityRegime,
        YieldCurveRegime,
    )

    when = datetime(2026, 3, 30, 16, 0, tzinfo=UTC)
    sig = Signal(
        symbol="SPY",
        direction="long",
        weight=0.2,
        confidence=0.8,
        rationale="r",
        timestamp=when,
        strategy_name="DCAStrategy",
    )
    mr = MarketRegime(
        timestamp=when,
        vix_close=24.5,
        yield_spread=0.35,
        yield_curve=YieldCurveRegime.FLAT,
        volatility=VolatilityRegime.ELEVATED,
        overall=OverallRegime.CAUTIOUS,
        sizing_multiplier=0.75,
    )
    text = generate_cycle_summary([sig], [], market_regime=mr)
    assert "cautious" in text.lower()
    assert "24.5" in text
    assert "0.35" in text
    assert "75%" in text
