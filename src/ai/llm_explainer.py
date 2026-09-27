"""LLM-powered explanations with graceful template fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from src.ai.llm_client import OllamaClient
    from src.automation.alerts import Alert
    from src.config import Settings
    from src.data.regime import MarketRegime
    from src.models import OrderExecutionResult, Signal
    from src.portfolio.state import PortfolioSnapshot
    from src.reporting.attribution import PositionContribution

logger = logging.getLogger(__name__)

SIGNAL_SYSTEM_PROMPT = (
    "You are a financial analyst assistant for a married couple's personal portfolio. "
    "Explain trading signals in plain English. Be concise (2-4 sentences). "
    "Reference specific data points (prices, indicator values, thresholds). "
    "Never give advice—only explain what the system did and why."
)

CYCLE_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Summarize the trading cycle clearly "
    "for a non-expert couple. Include key numbers: equity, cash, trades executed, "
    "and any alerts. Be concise (one short paragraph)."
)

DIGEST_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Write a brief weekly performance digest "
    "for a married couple's personal portfolio. Highlight the key takeaway, notable "
    "winners/losers, and any concerns. Keep it conversational and under 150 words."
)


def build_signal_context(
    signal: Signal,
    ohlcv_data: dict[str, pd.DataFrame],
    settings: Settings,
    market_regime: MarketRegime | None = None,
    snapshot: PortfolioSnapshot | None = None,
) -> str:
    """Build a structured context block for an LLM signal explanation."""
    sym = signal.symbol.strip().upper()
    parts = [
        f"Strategy: {signal.strategy_name or 'unknown'}",
        f"Symbol: {sym}",
        f"Direction: {signal.direction}",
        f"Weight: {signal.weight:.4f}",
        f"Confidence: {signal.confidence:.2f}",
        f"Rationale: {signal.rationale}",
    ]

    df = ohlcv_data.get(sym)
    if df is not None and not df.empty and "close" in df.columns:
        close = df["close"].astype(float)
        last_px = float(close.iloc[-1])
        parts.append(f"Last close: ${last_px:.2f}")
        if len(close) >= 5:
            prev5 = float(close.iloc[-5])
            chg = ((last_px / prev5) - 1.0) * 100.0 if prev5 > 0 else 0.0
            parts.append(f"5-day change: {chg:+.1f}%")

    if market_regime is not None:
        vix_s = f"{market_regime.vix_close:.1f}" if market_regime.vix_close is not None else "N/A"
        parts.append(
            f"Market regime: {market_regime.overall.value} (VIX: {vix_s}, "
            f"sizing multiplier: {market_regime.sizing_multiplier:.2f})"
        )

    if snapshot is not None:
        parts.append(f"Portfolio equity: ${snapshot.equity:,.2f}, cash: ${snapshot.cash:,.2f}")
        if snapshot.positions:
            held = ", ".join(
                f"{s}={q:.1f}" for s, q in sorted(snapshot.positions.items()) if q > 0
            )
            if held:
                parts.append(f"Current positions: {held}")

    return "\n".join(parts)


def generate_llm_explanation(
    client: OllamaClient,
    signal: Signal,
    ohlcv_data: dict[str, pd.DataFrame],
    settings: Settings,
    market_regime: MarketRegime | None = None,
    snapshot: PortfolioSnapshot | None = None,
) -> str | None:
    """Generate an LLM-powered signal explanation, or ``None`` on failure."""
    ctx = build_signal_context(signal, ohlcv_data, settings, market_regime, snapshot)
    prompt = f"Explain this trading signal to a non-expert investor:\n\n{ctx}"
    result = client.generate(prompt, system=SIGNAL_SYSTEM_PROMPT)
    if result:
        logger.debug("LLM explanation generated for %s", signal.symbol)
    return result


def generate_llm_cycle_summary(
    client: OllamaClient,
    signals: list[Signal],
    executions: list[OrderExecutionResult],
    snapshot: PortfolioSnapshot | None = None,
    alerts: list[Alert] | None = None,
    market_regime: MarketRegime | None = None,
) -> str | None:
    """Generate an LLM-powered cycle summary, or ``None`` on failure."""
    parts = [f"Signals generated: {len(signals)}"]
    for s in signals[:10]:
        parts.append(f"  {s.strategy_name}: {s.symbol} {s.direction} w={s.weight:.3f}")

    submitted = [e for e in executions if e.submitted]
    rejected = [e for e in executions if not e.submitted]
    parts.append(f"Executed: {len(submitted)}, Rejected: {len(rejected)}")
    for e in submitted[:10]:
        px_s = f" ${e.fill_price:.2f}" if e.fill_price else ""
        parts.append(f"  {e.side} {e.symbol}{px_s}")

    if snapshot is not None:
        parts.append(f"Portfolio: ${snapshot.equity:,.2f} equity, ${snapshot.cash:,.2f} cash")

    if alerts:
        parts.append(f"Alerts: {len(alerts)}")
        for a in alerts[:5]:
            parts.append(f"  [{a.level.value}] {a.message[:120]}")

    if market_regime is not None:
        parts.append(
            f"Regime: {market_regime.overall.value} (mult {market_regime.sizing_multiplier:.2f})"
        )

    prompt = "Summarize this trading cycle:\n\n" + "\n".join(parts)
    return client.generate(prompt, system=CYCLE_SYSTEM_PROMPT)


def generate_llm_weekly_digest(
    client: OllamaClient,
    snapshots: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    contributions: list[PositionContribution] | None = None,
) -> str | None:
    """Generate an LLM-powered weekly digest narrative, or ``None`` on failure."""
    if not snapshots:
        return None

    first_mv = float(snapshots[0].get("total_market_value", 0.0))
    last_mv = float(snapshots[-1].get("total_market_value", 0.0))
    first_pnl = float(snapshots[0].get("total_pnl", 0.0))
    last_pnl = float(snapshots[-1].get("total_pnl", 0.0))
    ret_pct = ((last_mv - first_mv) / first_mv * 100.0) if first_mv > 1e-9 else 0.0

    parts = [
        f"Period: {snapshots[0].get('date', '?')} to {snapshots[-1].get('date', '?')}",
        f"Portfolio value: ${last_mv:,.0f} ({ret_pct:+.1f}% change)",
        f"P&L delta: ${last_pnl - first_pnl:+,.0f}",
        f"Trades: {len(executions)} execution records",
        f"Alerts: {len(alerts)} alerts",
    ]

    if contributions:
        top = max(contributions, key=lambda c: c.total_pnl)
        worst = min(contributions, key=lambda c: c.total_pnl)
        parts.append(f"Top contributor: {top.symbol} (${top.total_pnl:+,.0f})")
        parts.append(f"Worst performer: {worst.symbol} (${worst.total_pnl:+,.0f})")

    prompt = "Write a brief weekly portfolio digest:\n\n" + "\n".join(parts)
    return client.generate(prompt, system=DIGEST_SYSTEM_PROMPT)
