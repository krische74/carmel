"""LLM explainer tests (OllamaClient mocked)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd

from src.ai.llm_explainer import (
    build_signal_context,
    generate_llm_cycle_summary,
    generate_llm_explanation,
    generate_llm_weekly_digest,
)
from src.config import Settings
from src.data.regime import MarketRegime, OverallRegime, VolatilityRegime, YieldCurveRegime
from src.models import OrderExecutionResult, Signal
from src.portfolio.state import PortfolioSnapshot


def _signal() -> Signal:
    return Signal(
        symbol="SPY",
        direction="long",
        weight=1.0,
        confidence=0.8,
        rationale="Momentum top pick",
        timestamp=datetime(2026, 4, 1, 16, 0, tzinfo=UTC),
        strategy_name="MomentumRotationStrategy",
    )


def _settings() -> Settings:
    return Settings(_yaml_path=None, _env_file=None)


def _ohlcv() -> dict[str, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    return {
        "SPY": pd.DataFrame(
            {"open": 500.0, "high": 510.0, "low": 490.0, "close": 505.0, "volume": 1e6},
            index=idx,
        ),
    }


def _regime() -> MarketRegime:
    return MarketRegime(
        timestamp=datetime(2026, 4, 1, 16, 0, tzinfo=UTC),
        vix_close=18.0,
        yield_spread=1.2,
        yield_curve=YieldCurveRegime.NORMAL,
        volatility=VolatilityRegime.NORMAL,
        overall=OverallRegime.RISK_ON,
        sizing_multiplier=1.0,
    )


def _snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(equity=50_000.0, cash=10_000.0, positions={"SPY": 50.0})


def test_build_signal_context_includes_strategy_and_symbol() -> None:
    ctx = build_signal_context(_signal(), _ohlcv(), _settings())
    assert "MomentumRotationStrategy" in ctx
    assert "SPY" in ctx
    assert "long" in ctx


def test_build_signal_context_includes_regime_when_present() -> None:
    ctx = build_signal_context(_signal(), _ohlcv(), _settings(), market_regime=_regime())
    assert "risk_on" in ctx
    assert "VIX" in ctx


def test_build_signal_context_omits_regime_when_none() -> None:
    ctx = build_signal_context(_signal(), _ohlcv(), _settings(), market_regime=None)
    assert "regime" not in ctx.lower() or "Market regime" not in ctx


def test_build_signal_context_includes_snapshot() -> None:
    ctx = build_signal_context(_signal(), _ohlcv(), _settings(), snapshot=_snapshot())
    assert "50,000" in ctx
    assert "SPY" in ctx


def test_generate_llm_explanation_returns_response() -> None:
    client = MagicMock()
    client.generate.return_value = "LLM explanation for SPY buy."
    result = generate_llm_explanation(client, _signal(), _ohlcv(), _settings())
    assert result == "LLM explanation for SPY buy."
    client.generate.assert_called_once()


def test_generate_llm_explanation_returns_none_on_failure() -> None:
    client = MagicMock()
    client.generate.return_value = None
    result = generate_llm_explanation(client, _signal(), _ohlcv(), _settings())
    assert result is None


def test_generate_llm_cycle_summary_includes_all_sections() -> None:
    client = MagicMock()
    client.generate.return_value = "Cycle went well."

    signals = [_signal()]
    execs = [
        OrderExecutionResult(
            symbol="SPY", submitted=True, order_id="o1", side="buy", qty=10.0, fill_price=505.0
        ),
    ]
    from src.automation.alerts import Alert, AlertLevel

    alerts = [
        Alert(timestamp=datetime.now(UTC), level=AlertLevel.INFO, category="test", message="OK")
    ]

    result = generate_llm_cycle_summary(client, signals, execs, _snapshot(), alerts, _regime())
    assert result == "Cycle went well."
    prompt_arg = client.generate.call_args[0][0]
    assert "SPY" in prompt_arg
    assert "Executed" in prompt_arg
    assert "Regime" in prompt_arg


def test_generate_llm_cycle_summary_returns_none_when_client_unavailable() -> None:
    client = MagicMock()
    client.generate.return_value = None
    result = generate_llm_cycle_summary(client, [], [])
    assert result is None


def test_generate_llm_weekly_digest_returns_narrative() -> None:
    client = MagicMock()
    client.generate.return_value = "Great week for the portfolio."
    snaps = [
        {"date": "2026-03-28", "total_market_value": 50000.0, "total_pnl": 500.0},
        {"date": "2026-04-04", "total_market_value": 51000.0, "total_pnl": 1000.0},
    ]
    result = generate_llm_weekly_digest(client, snaps, [], [])
    assert result == "Great week for the portfolio."


def test_weekly_digest_falls_back_to_none_when_no_snapshots() -> None:
    client = MagicMock()
    result = generate_llm_weekly_digest(client, [], [], [])
    assert result is None
    client.generate.assert_not_called()
