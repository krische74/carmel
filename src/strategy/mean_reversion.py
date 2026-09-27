"""Mean reversion using Bollinger Bands + RSI with optional SMA trend guard.

**Rebalance semantics (backtester integration):** On each rebalance day the engine maps
``long`` / ``cash`` signals to target weights; ``flat`` signals are ignored (see
``BacktestEngine``). If a position was opened while oversold but price has since moved
back inside the bands (no longer oversold), this strategy emits **no** ``long`` signal for
that symbol, so the target weight is zero and the backtester **sells**—even if the symbol
is not yet overbought. In other words, exits are driven by the *absence* of a long
signal, not by the presence of a ``flat`` overbought signal. The overbought ``flat``
signals below are **informational** (rationale for the dashboard) and do not
themselves set targets. A "hold until overbought" style would require emitting
continued ``long`` signals while between bands (not implemented here).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from src.models import Signal
from src.strategy.as_of import slice_to_as_of
from src.strategy.base import Strategy
from src.strategy.indicators import bollinger_bands, rsi, sma
from src.strategy.regime_params import effective_mean_reversion_max_positions

if TYPE_CHECKING:
    import pandas as pd

    from src.config import MeanReversionConfig, Settings
    from src.data.regime import MarketRegime


def _min_bars(cfg: MeanReversionConfig) -> int:
    """Minimum bars needed for indicators plus a small buffer."""
    p = max(cfg.bb_period, cfg.rsi_period)
    if cfg.sma_trend_period > 0:
        p = max(p, cfg.sma_trend_period)
    return p + 5


class MeanReversionStrategy(Strategy):
    """Buy near lower Bollinger band when RSI is oversold.

    Positions are held only while the oversold entry conditions are **active** on
    rebalance days; when conditions clear, the lack of a ``long`` signal unwinds the
    position (see module docstring). Overbought ``flat`` signals document the upper-band
    story but do not drive sizing.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_universe(self) -> list[str]:
        return [s.strip().upper() for s in self._settings.strategy.mean_reversion.universe]

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        *,
        as_of: datetime | None = None,
        market_regime: MarketRegime | None = None,
    ) -> list[Signal]:
        when = as_of or datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)

        cfg = self._settings.strategy.mean_reversion
        overall = market_regime.overall if market_regime is not None else None
        max_pos = effective_mean_reversion_max_positions(self._settings, overall)
        base_max = int(cfg.max_positions)
        regime_note = ""
        if overall is not None and max_pos != base_max:
            regime_note = (
                f" Regime cap ({overall.value}): at most {max_pos} positions "
                f"(base max_positions={base_max})."
            )
        min_bars = _min_bars(cfg)
        flat_signals: list[Signal] = []
        candidates: list[dict[str, float | str | None]] = []

        for sym in self.get_universe():
            raw = data.get(sym)
            if raw is None or raw.empty:
                continue
            frame = slice_to_as_of(raw, when)
            if len(frame) < min_bars:
                continue
            close = frame["close"].astype(float)
            last = float(close.iloc[-1])
            _mid, upper, lower = bollinger_bands(
                close,
                period=cfg.bb_period,
                num_std=cfg.bb_num_std,
            )
            rsi_s = rsi(close, period=cfg.rsi_period)
            last_upper = float(upper.iloc[-1])
            last_lower = float(lower.iloc[-1])
            last_rsi = float(rsi_s.iloc[-1])

            if np.isnan(last_upper) or np.isnan(last_lower) or np.isnan(last_rsi):
                continue

            # Informational only: backtester ignores `flat`; true exit is target weight 0
            # when no long signal is emitted after oversold clears.
            if last >= last_upper or last_rsi >= cfg.rsi_overbought:
                flat_signals.append(
                    Signal(
                        symbol=sym,
                        direction="flat",
                        weight=0.0,
                        confidence=0.65,
                        rationale=(
                            f"Mean reversion exit: {sym} at/above upper band ({last_upper:.2f}) "
                            f"or RSI overbought ({last_rsi:.1f} >= {cfg.rsi_overbought:.0f})."
                        ),
                        timestamp=when,
                        strategy_name="MeanReversionStrategy",
                    ),
                )
                continue

            sma_last: float | None = None
            if cfg.sma_trend_period > 0:
                trend = sma(close, period=cfg.sma_trend_period)
                sma_last = float(trend.iloc[-1])
                if last < sma_last:
                    continue

            if last <= last_lower and last_rsi <= cfg.rsi_oversold:
                depth = float(last_lower - last)
                candidates.append(
                    {
                        "symbol": sym,
                        "depth": depth,
                        "last": last,
                        "lower": last_lower,
                        "rsi": last_rsi,
                        "sma": sma_last,
                    },
                )

        out: list[Signal] = list(flat_signals)
        if not candidates:
            return out

        candidates.sort(key=lambda c: float(c["depth"]), reverse=True)
        k = min(len(candidates), max_pos)
        if k <= 0:
            return out
        chosen = candidates[:k]
        w = 1.0 / float(k)
        guard_note = (
            f"{cfg.sma_trend_period}-day SMA trend filter active"
            if cfg.sma_trend_period > 0
            else "SMA trend filter disabled"
        )
        for c in chosen:
            sym = str(c["symbol"])
            last = float(c["last"])
            lo = float(c["lower"])
            rv = float(c["rsi"])
            rationale = (
                f"Mean reversion entry: {sym} close {last:.2f} at/below lower band {lo:.2f}, "
                f"RSI {rv:.1f} <= oversold {cfg.rsi_oversold:.0f}; ranked by depth below band; "
                f"{guard_note}.{regime_note}"
            )
            out.append(
                Signal(
                    symbol=sym,
                    direction="long",
                    weight=w,
                    confidence=0.7,
                    rationale=rationale,
                    timestamp=when,
                    strategy_name="MeanReversionStrategy",
                ),
            )
        return out
