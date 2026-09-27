"""Absolute-momentum rotation across a risk-asset universe with a cash ETF fallback."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from src.models import Signal
from src.strategy.as_of import slice_to_as_of
from src.strategy.base import Strategy
from src.strategy.indicators import adx, sma
from src.strategy.momentum_banding import select_with_asymmetric_band

if TYPE_CHECKING:
    import pandas as pd

    from src.config import Settings
    from src.data.regime import MarketRegime


def _trading_days_for_month_label(months: int) -> int:
    """Approximate calendar months as 21 trading days per month."""
    return max(1, round(21 * months))


def _momentum_score(close: pd.Series, lookback_months: list[int]) -> float | None:
    """Average of total returns over each lookback horizon (trading-day approx)."""
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


class MomentumRotationStrategy(Strategy):
    """Rank by blended momentum; require price above long SMA; else hold cash ETF."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._disable_adx = False
        self._top_n = 1
        self._band_k: float | None = None
        self._incumbent_risk: str | None = None

    def configure_ablation(
        self,
        *,
        disable_adx: bool = False,
        top_n: int = 1,
        band_k: float | None = None,
    ) -> None:
        """Opt-in ablation knobs (Tier 53). Defaults match production behaviour."""
        self._disable_adx = bool(disable_adx)
        self._top_n = max(1, int(top_n))
        self._band_k = band_k
        self._incumbent_risk = None

    def get_universe(self) -> list[str]:
        risk = [s.strip().upper() for s in self._settings.data.universe]
        cash = self._settings.strategy.momentum.cash_symbol.strip().upper()
        out = list(dict.fromkeys([*risk, cash]))
        return out

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

        from src.strategy.regime_params import effective_adx_threshold

        cfg = self._settings.strategy.momentum
        risk_syms = [s.strip().upper() for s in self._settings.data.universe]
        cash_sym = cfg.cash_symbol.strip().upper()
        lookbacks = cfg.lookback_months
        sma_period = int(cfg.sma_filter_period)
        adx_period = int(cfg.adx_filter_period)
        overall = market_regime.overall if market_regime is not None else None
        adx_threshold = effective_adx_threshold(self._settings, overall)
        max_lookback_days = max(_trading_days_for_month_label(m) for m in lookbacks)
        min_bars = max(
            sma_period + max_lookback_days + 5,
            adx_period * 2 + 5,
        )

        scores: dict[str, float] = {}
        sliced_by_sym: dict[str, pd.DataFrame] = {}
        sma_eligible: list[str] = []
        any_valid_risk: bool = False

        for sym in risk_syms:
            raw = data.get(sym)
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
            return []

        if not sma_eligible:
            self._incumbent_risk = None
            rationale = (
                "Absolute momentum filter: every risk asset in the universe is at or below its "
                f"{sma_period}-day moving average as of this date. "
                f"Rotating to short-term cash ({cash_sym}) until trend improves."
            )
            return [
                Signal(
                    symbol=cash_sym,
                    direction="cash",
                    weight=1.0,
                    confidence=0.85,
                    rationale=rationale,
                    timestamp=when,
                    strategy_name="MomentumRotationStrategy",
                )
            ]

        eligible: list[str] = []
        for sym in sma_eligible:
            if self._disable_adx:
                eligible.append(sym)
                continue
            frame = sliced_by_sym[sym]
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            close = frame["close"].astype(float)
            adx_series, _, _ = adx(high, low, close, period=adx_period)
            last_adx = float(adx_series.iloc[-1])
            if np.isnan(last_adx) or last_adx <= adx_threshold:
                continue
            eligible.append(sym)

        if not eligible:
            self._incumbent_risk = None
            rationale = (
                f"ADX trend-strength filter: no risk asset shows ADX above {adx_threshold:.0f} "
                f"({adx_period}-day) while also trading above its {sma_period}-day average. "
                f"Rotating to short-term cash ({cash_sym}) until a stronger trend emerges."
            )
            return [
                Signal(
                    symbol=cash_sym,
                    direction="cash",
                    weight=1.0,
                    confidence=0.82,
                    rationale=rationale,
                    timestamp=when,
                    strategy_name="MomentumRotationStrategy",
                )
            ]

        ranked = sorted(eligible, key=lambda s: scores.get(s, float("-inf")), reverse=True)
        targets = select_with_asymmetric_band(
            ranked_risk=ranked,
            scores=scores,
            incumbent_risk=self._incumbent_risk,
            cash_symbol=cash_sym,
            band_k=self._band_k,
            top_n=self._top_n,
        )
        if len(targets) == 1 and targets[0][0] == cash_sym:
            self._incumbent_risk = None
            return [
                Signal(
                    symbol=cash_sym,
                    direction="cash",
                    weight=1.0,
                    confidence=0.82,
                    rationale="Banding/selection rotated to cash.",
                    timestamp=when,
                    strategy_name="MomentumRotationStrategy",
                )
            ]

        risk_targets = [(s, w) for s, w in targets if s != cash_sym]
        if not risk_targets:
            self._incumbent_risk = None
            return [
                Signal(
                    symbol=cash_sym,
                    direction="cash",
                    weight=1.0,
                    confidence=0.82,
                    rationale="No risk target after selection.",
                    timestamp=when,
                    strategy_name="MomentumRotationStrategy",
                )
            ]

        # Incumbent for next rebalance: highest-weight risk leg (equal-weight → first).
        self._incumbent_risk = risk_targets[0][0]
        adx_note = (
            "ADX filter disabled (ablation)."
            if self._disable_adx
            else f"ADX above {adx_threshold:.1f} ({adx_period}-day)."
        )
        band_note = (
            f" Band k={self._band_k}." if self._band_k is not None else ""
        )
        out: list[Signal] = []
        for sym, weight in risk_targets:
            sc = scores.get(sym, 0.0)
            rationale = (
                f"Momentum rotation: {sym} weight={weight:.0%} (score {sc:.2%}) among "
                f"names above {sma_period}-day SMA. {adx_note} top_n={self._top_n}.{band_note}"
            )
            out.append(
                Signal(
                    symbol=sym,
                    direction="long",
                    weight=float(weight),
                    confidence=0.75,
                    rationale=rationale,
                    timestamp=when,
                    strategy_name="MomentumRotationStrategy",
                )
            )
        return out
