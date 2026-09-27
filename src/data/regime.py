"""Market regime classification from stored VIX (Parquet) and FRED yields (SQLite)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Settings
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


def build_market_regime_snapshot(
    settings: Settings,
    *,
    as_of: datetime,
    vix_close: float | None,
    yield_spread: float | None,
) -> MarketRegime:
    """Classify VIX and yield spread into a point-in-time :class:`MarketRegime`.

    Used by :class:`RegimeDetector` and by walk-forward backtests. When
    ``yield_spread`` is ``None`` (typical in backtests without macro series),
    the yield curve is treated as NORMAL and classification is VIX-driven only.

    Args:
        settings: Hub settings (``regime`` thresholds and sizing map).
        as_of: Snapshot timestamp (normalized to UTC on the model).
        vix_close: Latest VIX close on or before ``as_of``, or ``None``.
        yield_spread: 10Y-2Y in percent points, or ``None``.
    """
    from src.config import Settings as SettingsCls

    if not isinstance(settings, SettingsCls):
        msg = "settings must be a Settings instance"
        raise TypeError(msg)

    cfg = settings.regime
    yc = _classify_yield_curve(
        yield_spread,
        flat_th=cfg.yield_curve_flat_threshold,
        inv_th=cfg.yield_curve_inverted_threshold,
    )
    vol = _classify_volatility(vix_close, settings)
    overall = _overall_regime(yc, vol)
    mult = float(cfg.sizing_adjustment.get(overall.value, 1.0))

    when = as_of.replace(tzinfo=UTC) if as_of.tzinfo is None else as_of.astimezone(UTC)

    return MarketRegime(
        timestamp=when,
        vix_close=vix_close,
        yield_spread=yield_spread,
        yield_curve=yc,
        volatility=vol,
        overall=overall,
        sizing_multiplier=mult,
    )


class YieldCurveRegime(StrEnum):
    """10Y-2Y spread classification."""

    NORMAL = "normal"
    FLAT = "flat"
    INVERTED = "inverted"


class VolatilityRegime(StrEnum):
    """VIX level bucket."""

    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


class OverallRegime(StrEnum):
    """Combined macro + vol regime for sizing."""

    RISK_ON = "risk_on"
    CAUTIOUS = "cautious"
    DEFENSIVE = "defensive"
    CRISIS = "crisis"


class MarketRegime(BaseModel):
    """Point-in-time market regime snapshot."""

    timestamp: datetime
    vix_close: float | None
    yield_spread: float | None
    yield_curve: YieldCurveRegime
    volatility: VolatilityRegime
    overall: OverallRegime
    sizing_multiplier: float = Field(ge=0.0, le=2.0)


def _classify_yield_curve(
    spread: float | None,
    *,
    flat_th: float,
    inv_th: float,
) -> YieldCurveRegime:
    if spread is None:
        return YieldCurveRegime.NORMAL
    if spread < inv_th:
        return YieldCurveRegime.INVERTED
    if spread <= flat_th:
        return YieldCurveRegime.FLAT
    return YieldCurveRegime.NORMAL


def _classify_volatility(vix: float | None, cfg: Settings) -> VolatilityRegime:
    """When VIX is missing, return NORMAL (neutral assumption)."""
    rc = cfg.regime
    if vix is None:
        return VolatilityRegime.NORMAL
    v = float(vix)
    if v < rc.vix_low:
        return VolatilityRegime.LOW
    if v < rc.vix_normal:
        return VolatilityRegime.NORMAL
    if v < rc.vix_elevated:
        return VolatilityRegime.ELEVATED
    return VolatilityRegime.CRISIS


def _overall_regime(
    yc: YieldCurveRegime,
    vol: VolatilityRegime,
) -> OverallRegime:
    if vol == VolatilityRegime.CRISIS:
        return OverallRegime.CRISIS
    if yc == YieldCurveRegime.INVERTED:
        return OverallRegime.DEFENSIVE
    flat = yc == YieldCurveRegime.FLAT
    elevated = vol == VolatilityRegime.ELEVATED
    if flat and elevated:
        return OverallRegime.DEFENSIVE
    if flat or elevated:
        return OverallRegime.CAUTIOUS
    return OverallRegime.RISK_ON


class RegimeDetector:
    """Classify yield curve and VIX into a market regime from stored data."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def detect(
        self,
        *,
        parquet_store: ParquetStore,
        sqlite_store: SQLiteStore,
    ) -> MarketRegime:
        """Read latest VIX from Parquet and FRED spreads from SQLite."""
        cfg = self._settings.regime
        when = datetime.now(UTC)

        vix_sym = cfg.vix_symbol.strip().upper()
        vix_close: float | None = None
        df = parquet_store.read_ohlcv(vix_sym)
        if df is not None and not df.empty and "close" in df.columns:
            df = df.sort_index()
            vix_close = float(df["close"].astype(float).iloc[-1])

        d10 = sqlite_store.get_latest_macro("DGS10")
        d2 = sqlite_store.get_latest_macro("DGS2")
        yield_spread: float | None = None
        if d10 is not None and d2 is not None:
            yield_spread = float(d10["value"]) - float(d2["value"])

        regime = build_market_regime_snapshot(
            self._settings,
            as_of=when,
            vix_close=vix_close,
            yield_spread=yield_spread,
        )
        logger.info(
            "Regime detected: overall=%s vol=%s yc=%s vix=%s spread=%s mult=%.2f",
            regime.overall.value,
            regime.volatility.value,
            regime.yield_curve.value,
            vix_close,
            yield_spread,
            regime.sizing_multiplier,
        )

        return regime
