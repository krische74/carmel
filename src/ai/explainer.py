"""Template-based plain-English explanations for signals, cycles, and digests."""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.strategy.as_of import slice_to_as_of
from src.strategy.indicators import adx, bollinger_bands, rsi, sma

if TYPE_CHECKING:
    import pandas as pd

    from src.automation.alerts import Alert
    from src.config import Settings
    from src.data.regime import MarketRegime
    from src.models import OrderExecutionResult, Signal
    from src.portfolio.state import PortfolioSnapshot
    from src.reporting.attribution import PositionContribution

logger = logging.getLogger(__name__)


def _trading_days_for_month_label(months: int) -> int:
    return max(1, round(21 * months))


def _momentum_score(close: pd.Series, lookback_months: list[int]) -> float | None:
    if close.empty:
        return None
    rets: list[float] = []
    last = float(close.iloc[-1])
    for m in lookback_months:
        days = _trading_days_for_month_label(m)
        if len(close) <= days:
            return None
        prior = float(close.iloc[-1 - days])
        if prior == 0.0:
            return None
        rets.append(last / prior - 1.0)
    return float(sum(rets) / len(rets))


def _explain_momentum(
    signal: Signal,
    market_data: dict[str, pd.DataFrame],
    settings: Settings,
) -> str | None:
    when = signal.timestamp
    when = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)

    cfg = settings.strategy.momentum
    risk_syms = [s.strip().upper() for s in settings.data.universe]
    lookbacks = cfg.lookback_months
    sma_period = int(cfg.sma_filter_period)
    adx_period = int(cfg.adx_filter_period)
    adx_threshold = float(cfg.adx_threshold)
    max_lookback_days = max(_trading_days_for_month_label(m) for m in lookbacks)
    min_bars = max(
        sma_period + max_lookback_days + 5,
        adx_period * 2 + 5,
    )

    scores: dict[str, float] = {}
    sliced_by_sym: dict[str, pd.DataFrame] = {}
    sma_eligible: list[str] = []
    any_valid_risk = False

    for sym in risk_syms:
        raw = market_data.get(sym)
        if raw is None or raw.empty:
            continue
        frame = slice_to_as_of(raw, when)
        if len(frame) < min_bars:
            continue
        close = frame["close"].astype(float)
        last = float(close.iloc[-1])
        trend = sma(close, period=sma_period)
        last_sma = float(trend.iloc[-1])
        sc = _momentum_score(close, lookbacks)
        if sc is None:
            continue
        any_valid_risk = True
        scores[sym] = sc
        sliced_by_sym[sym] = frame
        if last > last_sma:
            sma_eligible.append(sym)

    if not any_valid_risk:
        return None

    sym_list = ", ".join(risk_syms)
    if not sma_eligible:
        return (
            f"Rotated to cash: no ETF in the universe [{sym_list}] is above its "
            f"{sma_period}-day SMA. Defensive positioning until trend resumes."
        )

    eligible: list[str] = []
    adx_by_sym: dict[str, float] = {}
    for sym in sma_eligible:
        frame = sliced_by_sym[sym]
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        adx_series, _, _ = adx(high, low, close, period=adx_period)
        last_adx = float(adx_series.iloc[-1])
        adx_by_sym[sym] = last_adx
        if math.isnan(last_adx) or last_adx <= adx_threshold:
            continue
        eligible.append(sym)

    if not eligible:
        pick = max(sma_eligible, key=lambda s: scores.get(s, float("-inf")))
        adx_val = adx_by_sym.get(pick, float("nan"))
        return (
            f"Rotated to cash: {pick} is above its SMA but ADX ({adx_val:.1f}) is below "
            f"threshold ({adx_threshold}), indicating a weak trend."
        )

    eligible.sort(key=lambda s: scores.get(s, float("-inf")), reverse=True)
    best = eligible[0]
    best_score = scores[best] * 100.0
    if len(eligible) >= 2:
        second_sym = eligible[1]
        second_ret = scores.get(second_sym, 0.0) * 100.0
        runner_line = f" Runner-up: {second_sym} ({second_ret:.1f}%)."
    else:
        runner_line = ""
    frame = sliced_by_sym[best]
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    adx_series, _, _ = adx(high, low, close, period=adx_period)
    last_adx = float(adx_series.iloc[-1])

    max_lb = max(_trading_days_for_month_label(m) for m in lookbacks)
    return (
        f"Momentum rotated to {best}: highest {max_lb}-day blended return ({best_score:.1f}%) "
        f"among ETFs above their {sma_period}-day SMA with ADX {last_adx:.1f} > {adx_threshold} "
        f"(strong trend).{runner_line}"
    )


def _explain_dca(signal: Signal, settings: Settings) -> str:
    when = signal.timestamp
    when = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)
    weekday = when.strftime("%A")
    amount = float(settings.strategy.dca.amount)
    pct = settings.strategy.dca.percent_of_equity
    freq = settings.strategy.dca.frequency
    syms = ", ".join(t.symbol.strip().upper() for t in settings.data.dca_targets)
    if pct is not None:
        return (
            f"DCA contribution: ~{float(pct) * 100.0:.2f}% of account equity per {freq} cycle, "
            f"split across [{syms}] ({weekday}). Dollar size is resolved at execution from "
            "equity; see the signal rationale for this cycle's slice."
        )
    return (
        f"DCA contribution: ${amount:.2f} allocated across [{syms}] per {freq} schedule ({weekday})."
    )


def _explain_mean_reversion(
    signal: Signal,
    market_data: dict[str, pd.DataFrame],
    settings: Settings,
) -> str | None:
    when = signal.timestamp
    when = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)

    cfg = settings.strategy.mean_reversion
    sym = signal.symbol.strip().upper()
    raw = market_data.get(sym)
    if raw is None or raw.empty:
        return None
    frame = slice_to_as_of(raw, when)
    min_bars = max(cfg.bb_period, cfg.rsi_period, cfg.sma_trend_period) + 5
    if len(frame) < min_bars:
        return None
    close = frame["close"].astype(float)
    last = float(close.iloc[-1])
    _mid, upper, lower = bollinger_bands(close, period=cfg.bb_period, num_std=cfg.bb_num_std)
    rsi_s = rsi(close, period=cfg.rsi_period)
    last_upper = float(upper.iloc[-1])
    last_lower = float(lower.iloc[-1])
    last_rsi = float(rsi_s.iloc[-1])

    if signal.direction == "flat":
        return (
            f"Mean reversion exit on {sym}: price ${last:.2f} is above upper Bollinger Band "
            f"(${last_upper:.2f}), suggesting overbought conditions."
        )

    trend_note = ""
    if cfg.sma_trend_period > 0:
        trend_note = f" SMA trend guard ({cfg.sma_trend_period}-day) satisfied."
    return (
        f"Mean reversion entry on {sym}: price ${last:.2f} is below lower Bollinger Band "
        f"(${last_lower:.2f}) with RSI {last_rsi:.1f} (oversold < {cfg.rsi_oversold}){trend_note}"
    )


def generate_signal_explanation(
    signal: Signal,
    market_data: dict[str, pd.DataFrame],
    settings: Settings,
) -> str:
    """Produce a template-based explanation, or fall back to ``signal.rationale``."""
    sn = (signal.strategy_name or "").strip()
    try:
        if sn == "DCAStrategy":
            return _explain_dca(signal, settings)
        if sn == "MeanReversionStrategy":
            mr = _explain_mean_reversion(signal, market_data, settings)
            return mr if mr is not None else signal.rationale
        if sn == "MomentumRotationStrategy":
            mom = _explain_momentum(signal, market_data, settings)
            return mom if mom is not None else signal.rationale
        if sn == "TaxLossHarvest":
            return _explain_harvest(signal)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        logger.warning("Signal explanation failed, using rationale: %s", exc)
    return signal.rationale


def _explain_harvest(signal: Signal) -> str:
    """Plain-language TLH replacement buy (not tax advice)."""
    sym = signal.symbol.strip().upper()
    original = "the harvested symbol"
    rationale = signal.rationale
    if "TLH replacement for " in rationale:
        tail = rationale.split("TLH replacement for ", 1)[1]
        original = tail.split(":", 1)[0].strip()
    loss_line = ""
    if "unrealized loss of" in rationale.lower():
        loss_line = " See rationale for loss context."
    return (
        f"TLH replacement buy: {sym} purchased to replace harvested {original} position."
        f"{loss_line} Replacement avoids wash-sale trigger on the original security."
    )


def _format_regime_line(market_regime: MarketRegime | None) -> str:
    """Build the market-regime line for cycle summaries (empty string when no regime)."""
    if market_regime is None:
        return ""
    vix_s = f"{market_regime.vix_close:.1f}" if market_regime.vix_close is not None else "—"
    sp_s = f"{market_regime.yield_spread:.2f}" if market_regime.yield_spread is not None else "—"
    mult_pct = float(market_regime.sizing_multiplier) * 100.0
    return (
        f"\n  Market regime: {market_regime.overall.value} "
        f"(VIX {vix_s}, yield spread {sp_s}). Sizing at {mult_pct:.0f}%."
    )


def _strategy_label(name: str | None) -> str:
    n = (name or "").strip()
    if n == "MomentumRotationStrategy":
        return "momentum"
    if n == "DCAStrategy":
        return "DCA"
    if n == "MeanReversionStrategy":
        return "mean reversion"
    return n or "unknown"


def generate_cycle_summary(
    signals: list[Signal],
    executions: list[OrderExecutionResult],
    snapshot: PortfolioSnapshot | None = None,
    alerts: list[Alert] | None = None,
    market_regime: MarketRegime | None = None,
) -> str:
    """Multi-line summary of one trading cycle."""
    ts = datetime.now(UTC)
    if signals:
        ts = signals[0].timestamp
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)

    regime_line = _format_regime_line(market_regime)

    if not signals and not executions:
        return (
            f"Cycle summary ({ts.strftime('%Y-%m-%d %H:%M')} UTC):\n"
            "  Signals: 0\n"
            "  No activity this cycle.\n"
            "  Executed: 0\n"
            "  Rejected: 0\n"
            "  Portfolio: —\n"
            f"  Alerts: none{regime_line}"
        )

    by_strat = Counter(_strategy_label(s.strategy_name) for s in signals)
    strat_bits = [f"{n} {label}" for label, n in sorted(by_strat.items())]
    sig_line = f"Signals: {len(signals)}"
    if strat_bits:
        sig_line += f" ({', '.join(strat_bits)})"

    submitted = [e for e in executions if e.submitted]
    rejected = [e for e in executions if not e.submitted]
    ex_parts: list[str] = []
    for e in submitted:
        px = f" ${e.fill_price:.2f}" if e.fill_price is not None else ""
        ex_parts.append(f"{e.side} ({e.symbol}{px})")
    ex_line = (
        f"Executed: {len(submitted)} ({'; '.join(ex_parts)})"
        if ex_parts
        else f"Executed: {len(submitted)}"
    )
    rej_line = f"Rejected: {len(rejected)}"

    port_line = "Portfolio: —"
    if snapshot is not None:
        port_line = f"Portfolio: ${snapshot.equity:,.2f} equity, ${snapshot.cash:,.2f} cash"

    alt_line = "Alerts: none"
    if alerts:
        by_lvl = Counter(a.level.value for a in alerts)
        alt_line = "Alerts: " + ", ".join(f"{n} {lvl}" for lvl, n in sorted(by_lvl.items()))

    return (
        f"Cycle summary ({ts.strftime('%Y-%m-%d %H:%M')} UTC):\n"
        f"  {sig_line}\n"
        f"  {ex_line}\n"
        f"  {rej_line}\n"
        f"  {port_line}\n"
        f"  {alt_line}{regime_line}"
    )


def enrich_signal_rationale(signal: Signal, explanation: str) -> Signal:
    """Return a copy with ``rationale`` set to ``explanation`` (does not mutate ``signal``)."""
    return signal.model_copy(update={"rationale": explanation})


def generate_weekly_digest(
    snapshots: list[dict[str, Any]],
    executions: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    contributions: list[PositionContribution] | None = None,
) -> str:
    """Plain-text weekly performance and activity digest."""
    if not snapshots:
        return "No data for this period."

    start_d = str(snapshots[0].get("date", ""))[:10]
    end_d = str(snapshots[-1].get("date", ""))[:10]

    first_mv = float(snapshots[0].get("total_market_value", 0.0))
    last_mv = float(snapshots[-1].get("total_market_value", 0.0))
    first_pnl = float(snapshots[0].get("total_pnl", 0.0))
    last_pnl = float(snapshots[-1].get("total_pnl", 0.0))

    ret_pct = ((last_mv - first_mv) / first_mv * 100.0) if first_mv > 1e-9 else 0.0
    pnl_delta = last_pnl - first_pnl

    top_sym = "—"
    worst_sym = "—"
    if contributions:
        top = max(contributions, key=lambda c: c.total_pnl)
        top_sym = f"{top.symbol} (+${top.total_pnl:,.0f}, {top.contribution_pct:+.2f}%)"
        worst = min(contributions, key=lambda c: c.total_pnl)
        worst_sym = f"{worst.symbol} (${worst.total_pnl:,.0f}, {worst.contribution_pct:+.2f}%)"

    buys = sum(
        1 for e in executions if str(e.get("side", "")).lower() == "buy" and e.get("submitted")
    )
    sells = sum(
        1 for e in executions if str(e.get("side", "")).lower() == "sell" and e.get("submitted")
    )

    crit = sum(1 for a in alerts if str(a.get("level", "")).lower() == "critical")
    warn = sum(1 for a in alerts if str(a.get("level", "")).lower() == "warning")

    return (
        f"Weekly digest ({start_d} to {end_d}):\n"
        f"  Portfolio: ${last_mv:,.0f} ({ret_pct:+.1f}%, {pnl_delta:+,.0f} $ P&L)\n"
        f"  Top contributor: {top_sym}\n"
        f"  Worst performer: {worst_sym}\n"
        f"  Trades: {buys} buys, {sells} sells\n"
        f"  Alerts: {crit} critical, {warn} warning"
    )
