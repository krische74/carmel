"""Walk-forward backtesting for ``Strategy`` implementations (close-to-close, no lookahead)."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.backtesting.cash_yield import lookup_cash_yield_annual_pct
from src.data.regime import build_market_regime_snapshot
from src.reporting.returns import ReturnMetrics, compute_return_metrics
from src.risk.position_sizing import compute_target_position_notional

if TYPE_CHECKING:
    from src.config import Settings
    from src.models import Signal
    from src.strategy.base import Strategy

logger = logging.getLogger(__name__)

MIN_REBALANCE_TRADE_USD = 1.0


class BacktestDataMissingError(ValueError):
    """Raised when no OHLCV rows fall in the requested backtest window (or warm-up consumes all)."""

    def __init__(
        self,
        *,
        start: date,
        end: date,
        symbols: list[str],
        detail: str | None = None,
        available_start: date | None = None,
        available_end: date | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.symbols = symbols
        self.detail = detail
        self.available_start = available_start
        self.available_end = available_end
        sym_str = ", ".join(symbols) if symbols else "(no symbols loaded)"
        msg = (
            f"No OHLCV data found for [{sym_str}] in date range "
            f"{start.isoformat()} to {end.isoformat()}."
        )
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)


class BacktestConfig(BaseModel):
    """Parameters for one backtest run."""

    model_config = ConfigDict(populate_by_name=True)

    initial_capital: float = 10_000.0
    slippage_bps: float = 5.0
    rebalance_frequency: str = "monthly"  # daily | weekly | monthly
    use_regime: bool = Field(
        default=True,
        description=(
            "When True and ``settings`` is passed, each rebalance passes VIX-based "
            "``market_regime`` to every strategy."
        ),
    )
    warmup_bars: int = Field(
        default=0,
        ge=0,
        le=500,
        description="Skip first N trading days in the backtest window (indicator warm-up).",
    )
    min_coverage_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum overlap fraction between requested range and available trading days. "
            "Below this, raise BacktestDataMissingError. Set 0.0 to disable."
        ),
    )
    apply_risk_layer: bool = Field(
        default=True,
        description=(
            "When True and ``settings`` is provided, apply live sizing constraints "
            "(max_position_pct, regime multiplier, cash reserve, min order floor)."
        ),
    )
    cash_yield_annual_pct: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Fallback annualized cash yield when no historical series is supplied. "
            "0 disables flat accrual."
        ),
    )
    cash_yield_by_date: dict[str, float] | None = Field(
        default=None,
        description=(
            "Optional ISO-date -> annual percent map (e.g. FRED DTB3). When set, "
            "each day looks up the series (bounded ffill); missing data logs fallback."
        ),
    )
    rebalance_weekday: int | None = Field(
        default=None,
        description="Weekly: 0=Mon .. 4=Fri. None = first trading day of the ISO week.",
    )
    rebalance_month_phase: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Monthly: skip this many trading days, then rebalance (0 = first day).",
    )
    include_dca: bool = False
    include_cash_sweep: bool = False
    # Tier 53 ablation flags — all default to current production behaviour.
    disable_adx: bool = Field(
        default=False,
        description="When True, skip the ADX trend-strength filter (ablation only).",
    )
    top_n: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Hold top-N equal-weight risk assets (1 = winner-take-all).",
    )
    band_k: float | None = Field(
        default=None,
        description=(
            "Dispersion-scaled relative-switch band: challenger must beat incumbent "
            "by k * cross-sectional sigma of momentum scores. None disables banding. "
            "Absolute exit to cash and re-entry from cash are never banded."
        ),
    )
    pin_regime_multiplier: float | None = Field(
        default=None,
        description=(
            "When set, use this sizing multiplier instead of the VIX regime multiplier "
            "(ablation B-Regime uses 1.0)."
        ),
    )
    pin_dca_regime_multiplier: float | None = Field(
        default=None,
        description=(
            "When set, pin the DCA sleeve regime scale to this value (Tier 54F). "
            "Does not alter momentum sizing; distinct from pin_regime_multiplier."
        ),
    )
    dca_fixed_amount_usd: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "When set, DCA budget per cycle is this fixed dollar amount instead of "
            "percent_of_equity * equity (Tier 54F)."
        ),
    )
    # Temporary decomposition switches for the post-54A baseline (Step 0).
    bypass_rotation_gate: bool = Field(
        default=False,
        description="Test-only: bypass the discrete hold-state rotation gate.",
    )
    bypass_drift_gate: bool = Field(
        default=False,
        description="Test-only: bypass the drift-band maintenance gate.",
    )
    bypass_target_refresh_gate: bool = Field(
        default=False,
        description="Test-only: bypass the target-weight refresh gate.",
    )
    legacy_empty_target_skip: bool = Field(
        default=False,
        description="Test-only: skip rebalances when a scheduled target set is empty.",
    )
    experiment_label: str | None = Field(
        default=None,
        description="Optional label persisted with the run (e.g. B-ADX).",
    )
    drift_band_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Absolute portfolio-weight tolerance for threshold rebalancing while hold "
            "state is unchanged (Tier 54A). Rebalance only when any leg's actual "
            "weight deviates from its target by more than this many percentage points "
            "(e.g. 0.05 = 5 pp). Standard practitioner default when DR #5 does not "
            "export Doc thresholds; use 1.0 to disable drift maintenance. "
            "Backtest-only: allows positions above max_position_pct by up to band pp "
            "before trim; live sizing would need a one-sided trim-only rule."
        ),
    )
    tranche_offset: int | None = Field(
        default=None,
        ge=0,
        description=(
            "When set with weekly frequency, rebalance every 5 trading days starting "
            "at this day index (Tier 54C tranche study). None uses calendar weekly logic."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_use_regime_for_momentum(cls, data: Any) -> Any:
        """Accept deprecated ``use_regime_for_momentum`` (same meaning as ``use_regime``)."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "use_regime_for_momentum" in out and "use_regime" not in out:
            out["use_regime"] = out.pop("use_regime_for_momentum")
        else:
            out.pop("use_regime_for_momentum", None)
        return out

    def ablation_config_dict(self) -> dict[str, Any]:
        """Serializable knobs needed to interpret a persisted run (Tier 53B)."""
        return {
            "apply_risk_layer": self.apply_risk_layer,
            "rebalance_frequency": self.rebalance_frequency,
            "rebalance_weekday": self.rebalance_weekday,
            "rebalance_month_phase": self.rebalance_month_phase,
            "disable_adx": self.disable_adx,
            "top_n": self.top_n,
            "band_k": self.band_k,
            "pin_regime_multiplier": self.pin_regime_multiplier,
            "pin_dca_regime_multiplier": self.pin_dca_regime_multiplier,
            "dca_fixed_amount_usd": self.dca_fixed_amount_usd,
            "bypass_rotation_gate": self.bypass_rotation_gate,
            "bypass_drift_gate": self.bypass_drift_gate,
            "bypass_target_refresh_gate": self.bypass_target_refresh_gate,
            "legacy_empty_target_skip": self.legacy_empty_target_skip,
            "use_regime": self.use_regime,
            "include_dca": self.include_dca,
            "include_cash_sweep": self.include_cash_sweep,
            "cash_yield_annual_pct": self.cash_yield_annual_pct,
            "experiment_label": self.experiment_label,
            "drift_band_pct": self.drift_band_pct,
            "tranche_offset": self.tranche_offset,
        }


class BacktestTrade(BaseModel):
    """One simulated trade."""

    date: str
    symbol: str
    side: str  # buy | sell
    qty: float
    price: float
    slippage_cost: float


class BacktestResult(BaseModel):
    """Output of a backtest run."""

    equity_curve: list[dict[str, Any]] = Field(default_factory=list)
    trades: list[BacktestTrade] = Field(default_factory=list)
    return_metrics: ReturnMetrics
    initial_capital: float
    final_equity: float
    zero_trades_warning: str | None = None
    sizing_mode: str = "raw_signal"
    cash_yield_annual_pct: float = 0.0
    realized_cash_yield_annual_pct: float = 0.0
    cash_yield_mode: str = "flat"
    peak_single_symbol_allocation_pct: float = 0.0
    avg_exposure_pct: float = 0.0
    max_exposure_pct: float = 0.0
    max_exposure_date: str | None = None
    run_kind: str = "momentum"
    turnover: float = 0.0
    experiment_label: str | None = None
    discrete_state_changes: int = 0
    maintenance_band_breaches: int = 0
    maintenance_trades: int = 0
    empty_target_periods: int = 0
    empty_target_liquidation_trades: int = 0
    benchmark_symbol: str | None = None
    benchmark_metrics: ReturnMetrics | None = None


def _rebalance_dates(
    trading_days: list[date],
    frequency: str,
    *,
    weekday: int | None = None,
    month_phase: int = 0,
    tranche_offset: int | None = None,
) -> set[date]:
    """Period rebalance dates; daily returns all trading days.

    Weekly with ``tranche_offset``: every 5 trading days from that index (54C tranches).
    Weekly with ``weekday`` None: first trading day of each ISO week (legacy).
    Weekly with ``weekday`` 0-4: first trading day in the ISO week with
    ``date.weekday() >= weekday``.
    Monthly: the ``month_phase``-th trading day of each month (0 = first).
    """
    if not trading_days:
        return set()
    if frequency == "daily":
        return set(trading_days)
    if frequency == "weekly" and tranche_offset is not None:
        off = int(tranche_offset)
        period = 5
        return {trading_days[i] for i in range(off, len(trading_days), period)}
    if frequency == "weekly":
        groups: dict[tuple[int, int], list[date]] = {}
        for d in trading_days:
            key = (d.isocalendar()[0], d.isocalendar()[1])
            groups.setdefault(key, []).append(d)
        out: set[date] = set()
        for days in groups.values():
            days.sort()
            if weekday is None:
                out.add(days[0])
                continue
            wd = int(weekday)
            for d in days:
                if d.weekday() >= wd:
                    out.add(d)
                    break
        return out
    if frequency == "monthly":
        groups_m: dict[tuple[int, int], list[date]] = {}
        for d in trading_days:
            groups_m.setdefault((d.year, d.month), []).append(d)
        out_m: set[date] = set()
        phase = max(0, int(month_phase))
        for days in groups_m.values():
            days.sort()
            if phase < len(days):
                out_m.add(days[phase])
        return out_m
    msg = f"unsupported rebalance_frequency: {frequency!r}"
    raise ValueError(msg)


def _normalize_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with timezone-naive, midnight-normalized DatetimeIndex (dedupe keep last)."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.normalize()
    out = df.copy()
    out.index = idx
    if out.index.duplicated().any():
        out = out[~out.index.duplicated(keep="last")]
    return out


def _close_price_from_normalized(df: pd.DataFrame, d: date) -> float | None:
    """O(1) ``close`` lookup on a frame whose index is normalized to calendar dates."""
    if df.empty or "close" not in df.columns:
        return None
    key = pd.Timestamp(d).normalize()
    if key not in df.index:
        return None
    row = df.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return float(row["close"])


def vix_close_as_of(normalized_vix: pd.DataFrame, as_of: datetime) -> float | None:
    """Last VIX ``close`` on or before ``as_of`` (no lookahead).

    ``normalized_vix`` must use :func:`_normalize_ohlcv_index` (naive midnight index).
    """
    if normalized_vix.empty or "close" not in normalized_vix.columns:
        return None
    ts = pd.Timestamp(as_of)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ts = ts.normalize()
    sub = normalized_vix.loc[normalized_vix.index <= ts]
    if sub.empty:
        return None
    return float(sub["close"].astype(float).iloc[-1])


def _approx_trading_days_between(start: date, end: date) -> int:
    """Heuristic expected trading days in ``[start, end]`` (~252/year)."""
    if end < start:
        return 1
    return max(1, int((end - start).days * 252 / 365))


def _available_data_date_bounds(data: dict[str, pd.DataFrame]) -> tuple[date | None, date | None]:
    """Earliest and latest calendar dates across non-empty frames (union of symbol ranges)."""
    lows: list[date] = []
    highs: list[date] = []
    for df in data.values():
        if df is None or df.empty:
            continue
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            continue
        lows.append(pd.Timestamp(idx.min()).normalize().date())
        highs.append(pd.Timestamp(idx.max()).normalize().date())
    if not lows:
        return None, None
    return min(lows), max(highs)


def trading_days_in_range(
    data: dict[str, pd.DataFrame],
    start: date,
    end: date,
) -> list[date]:
    """Sorted union of calendar dates present in ``data`` indices, clipped to ``[start, end]``.

    Used by :class:`BacktestEngine` and walk-forward helpers so all backtesting code
    shares one definition of the trading calendar.
    """
    days: set[date] = set()
    for sym, df in data.items():
        if df is None or df.empty:
            continue
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError(
                f"trading_days_in_range: data[{sym!r}] has {type(idx).__name__}, "
                f"expected DatetimeIndex. Use ParquetStore.read_ohlcv() to load.",
            )
        for ts in idx:
            t = pd.Timestamp(ts).normalize().date()
            if start <= t <= end:
                days.add(t)
    return sorted(days)


def _signals_to_target_weights(signals: list[Signal]) -> dict[str, float]:
    """Map long/cash signals to target weights; ignore ``flat``.

    If multiple signals reference the same symbol, weights are **summed** (not replaced).
    Callers should ensure totals stay meaningful; the backtester does not normalize weights
    to sum to 1. Rebalancing uses dollar targets ``weight * equity`` and may skip buys when
    cash is insufficient rather than scaling positions proportionally.
    """
    out: dict[str, float] = {}
    for s in signals:
        if s.direction == "flat":
            continue
        if s.direction in ("long", "cash"):
            sym = s.symbol.strip().upper()
            out[sym] = out.get(sym, 0.0) + float(s.weight)
    return out


def discrete_hold_state(
    targets: dict[str, float],
    *,
    cash_symbols: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Risk-asset hold state for state-gated rebalancing (Tier 54A).

    Cash legs (e.g. SHV) are excluded. Empty tuple means cash-only / no risk asset.
    """
    cash = cash_symbols or frozenset()
    risk = {
        sym.strip().upper(): w
        for sym, w in targets.items()
        if w > 1e-12 and sym.strip().upper() not in cash
    }
    if not risk:
        return ()
    return tuple(sorted(risk.keys()))


def _states_equal(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    return a == b


def _signals_to_constrained_target_weights(
    signals: list[Signal],
    *,
    equity: float,
    max_position_pct: float,
    regime_multiplier: float,
) -> dict[str, float]:
    """Target weights after live position-cap and regime multiplier (absolute, not delta)."""
    if equity <= 0.0:
        return {}
    out: dict[str, float] = {}
    for s in signals:
        if s.direction == "flat":
            continue
        if s.direction not in ("long", "cash"):
            continue
        sym = s.symbol.strip().upper()
        tgt_notional = compute_target_position_notional(
            signal_weight=float(s.weight),
            equity=equity,
            max_position_pct=max_position_pct,
            regime_multiplier=regime_multiplier,
        )
        out[sym] = out.get(sym, 0.0) + tgt_notional / equity
    return out


def compute_benchmark_metrics(
    data: dict[str, pd.DataFrame],
    symbol: str,
    trading_days: list[date],
    *,
    risk_free_rate_annual: float = 0.0,
    risk_free_daily: pd.Series | None = None,
) -> ReturnMetrics | None:
    """Buy-and-hold daily returns for ``symbol`` over ``trading_days`` (close-to-close)."""
    sym = symbol.strip().upper()
    df = data.get(sym)
    if df is None or df.empty or len(trading_days) < 2:
        return None
    norm = _normalize_ohlcv_index(df)
    closes: list[float] = []
    idx_dates: list[str] = []
    for d in trading_days:
        px = _close_price_from_normalized(norm, d)
        if px is None:
            return None
        closes.append(px)
        idx_dates.append(d.isoformat())
    rets: list[float] = []
    ridx: list[str] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev <= 0.0:
            continue
        rets.append((closes[i] - prev) / prev)
        ridx.append(idx_dates[i])
    if not rets:
        return None
    series = pd.Series(rets, index=pd.Index(ridx, name="date"))
    return compute_return_metrics(
        series,
        risk_free_rate_annual=risk_free_rate_annual,
        risk_free_daily=risk_free_daily,
    )


class BacktestEngine:
    """Walk-forward backtester for any ``Strategy`` subclass."""

    def run(
        self,
        strategy: Strategy,
        data: dict[str, pd.DataFrame],
        *,
        start: date,
        end: date,
        config: BacktestConfig | None = None,
        settings: Settings | None = None,
    ) -> BacktestResult:
        """Simulate daily mark-to-market and periodic rebalancing using ``generate_signals``.

        When ``settings`` is provided and ``use_regime`` is True, each rebalance passes
        ``market_regime`` built from VIX in ``data`` (``settings.regime.vix_symbol``) with
        ``yield_spread=None`` (VIX-only regime in backtests unless you add macro series
        later). All strategies receive the same snapshot.
        """
        cfg = config or BacktestConfig()
        initial = float(cfg.initial_capital)
        bps = float(cfg.slippage_bps) / 10_000.0
        sym_keys = sorted(data.keys())
        avail_lo, avail_hi = _available_data_date_bounds(data)
        trading_days = trading_days_in_range(data, start, end)
        if not trading_days:
            raise BacktestDataMissingError(
                start=start,
                end=end,
                symbols=sym_keys,
                available_start=avail_lo,
                available_end=avail_hi,
            )
        expected_trading_days = _approx_trading_days_between(start, end)
        coverage_ratio = len(trading_days) / max(1, expected_trading_days)
        if float(cfg.min_coverage_ratio) > 0.0 and coverage_ratio < float(cfg.min_coverage_ratio):
            detail = (
                f"Only {len(trading_days)} of ~{expected_trading_days} expected trading "
                f"days available ({coverage_ratio:.0%} coverage). Available data: "
                f"{avail_lo} to {avail_hi}. Requested: {start} to {end}."
            )
            raise BacktestDataMissingError(
                start=start,
                end=end,
                symbols=sym_keys,
                detail=detail,
                available_start=avail_lo,
                available_end=avail_hi,
            )
        warm = max(0, min(int(cfg.warmup_bars), 500))
        if warm > 0:
            if len(trading_days) <= warm:
                raise BacktestDataMissingError(
                    start=start,
                    end=end,
                    symbols=sym_keys,
                    available_start=avail_lo,
                    available_end=avail_hi,
                    detail=(
                        f"Not enough trading days in range for warmup_bars={warm} "
                        f"(have {len(trading_days)}, need more than {warm})."
                    ),
                )
            trading_days = trading_days[warm:]
        rebal_on = _rebalance_dates(
            trading_days,
            cfg.rebalance_frequency,
            weekday=cfg.rebalance_weekday,
            month_phase=int(cfg.rebalance_month_phase),
            tranche_offset=cfg.tranche_offset,
        )
        normalized: dict[str, pd.DataFrame] = {
            sym: _normalize_ohlcv_index(df) for sym, df in data.items()
        }

        from src.backtesting.sim_steps import (
            apply_cash_sweep,
            apply_dca_contributions,
            compute_risk_exposure_pct,
            drift_band_exceeded,
            mtm_now,
            rebalance_to_targets,
            target_weights_changed,
        )

        cash = initial
        positions: dict[str, float] = {}
        last_marks: dict[str, float] = {}
        trades: list[BacktestTrade] = []
        equity_curve: list[dict[str, Any]] = []
        use_risk = bool(cfg.apply_risk_layer and settings is not None)
        sizing_mode = "constrained" if use_risk else "raw_signal"
        cash_yield = float(cfg.cash_yield_annual_pct)
        yield_series: pd.Series | None = None
        if cfg.cash_yield_by_date is not None:
            yield_series = pd.Series(
                {pd.Timestamp(k): float(v) for k, v in cfg.cash_yield_by_date.items()},
                dtype=float,
            )
            if not yield_series.empty:
                yield_series.index = pd.DatetimeIndex(yield_series.index).normalize()
        peak_alloc_pct = 0.0
        exposures: list[float] = []
        exposure_dates: list[str] = []
        applied_yields: list[float] = []
        yield_modes: set[str] = set()
        rf_by_date: dict[str, float] = {}
        min_trade_usd = MIN_REBALANCE_TRADE_USD
        min_reserve_pct = 0.0
        max_position_pct = 1.0
        min_order_floor = MIN_REBALANCE_TRADE_USD
        sweep_sym = ""
        if settings is not None:
            min_trade_usd = float(settings.risk.min_order_notional_usd)
            min_order_floor = float(settings.risk.min_order_notional_usd)
            min_reserve_pct = float(settings.risk.min_cash_reserve_pct)
            max_position_pct = float(settings.risk.max_position_pct)
            if cfg.include_cash_sweep:
                sweep_sym = settings.cash_sweep.symbol.strip().upper()
        if not use_risk:
            min_reserve_pct = 0.0
            if not cfg.include_dca:
                min_trade_usd = MIN_REBALANCE_TRADE_USD
                min_order_floor = MIN_REBALANCE_TRADE_USD
                max_position_pct = 1.0

        dca_strategy = None
        if cfg.include_dca and settings is not None:
            from src.strategy.dca import DCAStrategy

            dca_strategy = DCAStrategy(settings)

        cash_symbols: frozenset[str] = frozenset()
        if settings is not None:
            cash_symbols = frozenset(
                {settings.strategy.momentum.cash_symbol.strip().upper()},
            )

        run_kind = "momentum"
        if cfg.include_dca and cfg.include_cash_sweep:
            run_kind = "full_system"
        elif cfg.include_dca:
            run_kind = "momentum_plus_dca"

        configure = getattr(strategy, "configure_ablation", None)
        if callable(configure):
            configure(
                disable_adx=bool(cfg.disable_adx),
                top_n=int(cfg.top_n),
                band_k=cfg.band_k,
            )

        last_hold_state: tuple[str, ...] | None = None
        last_targets: dict[str, float] | None = None
        discrete_state_changes = 0
        maintenance_band_breaches = 0
        empty_target_periods = 0
        empty_target_liquidation_trades = 0

        for d in trading_days:
            if yield_series is not None:
                y_ann, y_mode = lookup_cash_yield_annual_pct(
                    yield_series,
                    d,
                    fallback_pct=cash_yield,
                )
                yield_modes.add(y_mode)
                applied_yields.append(y_ann)
                daily_yield = (y_ann / 100.0) / 252.0 if y_ann > 0.0 else 0.0
            else:
                daily_yield = (cash_yield / 100.0) / 252.0 if cash_yield > 0.0 else 0.0
                applied_yields.append(cash_yield)
                yield_modes.add("flat")
            rf_by_date[d.isoformat()] = daily_yield
            if daily_yield > 0.0 and cash > 0.0:
                cash *= 1.0 + daily_yield

            mtm = mtm_now(cash, positions, last_marks, normalized, d)
            equity_curve.append({"date": d.isoformat(), "equity": mtm})

            if mtm > 0.0:
                exp = compute_risk_exposure_pct(
                    mtm=mtm,
                    positions=positions,
                    last_marks=last_marks,
                    cash_symbols=cash_symbols,
                    sweep_symbol=sweep_sym,
                )
                exposures.append(exp)
                exposure_dates.append(d.isoformat())
                if (not use_risk) or cfg.include_dca:
                    for sym, qty in positions.items():
                        if sweep_sym and sym == sweep_sym:
                            continue
                        if sym.strip().upper() in cash_symbols:
                            continue
                        px = last_marks.get(sym)
                        if px is not None and px > 0.0:
                            peak_alloc_pct = max(peak_alloc_pct, (qty * px) / mtm)

            as_of = datetime.combine(d, time.min, tzinfo=UTC)
            regime = None
            regime_multiplier = 1.0
            if settings is not None and cfg.use_regime:
                vix_sym = settings.regime.vix_symbol.strip().upper()
                vix_df = normalized.get(vix_sym)
                vx = vix_close_as_of(vix_df, as_of) if vix_df is not None else None
                regime = build_market_regime_snapshot(
                    settings,
                    as_of=as_of,
                    vix_close=vx,
                    yield_spread=None,
                )
                regime_multiplier = float(regime.sizing_multiplier)
            if cfg.pin_regime_multiplier is not None:
                regime_multiplier = float(cfg.pin_regime_multiplier)

            if d in rebal_on:
                if settings is not None and cfg.use_regime and regime is not None:
                    signals = strategy.generate_signals(data, as_of=as_of, market_regime=regime)
                else:
                    signals = strategy.generate_signals(data, as_of=as_of)
                if use_risk and settings is not None:
                    targets = _signals_to_constrained_target_weights(
                        signals,
                        equity=mtm,
                        max_position_pct=max_position_pct,
                        regime_multiplier=regime_multiplier,
                    )
                    for w in targets.values():
                        peak_alloc_pct = max(peak_alloc_pct, w)
                else:
                    targets = _signals_to_target_weights(signals)
                if not targets:
                    empty_target_periods += 1
                if not (cfg.legacy_empty_target_skip and not targets):
                    new_hold_state = discrete_hold_state(targets, cash_symbols=cash_symbols)
                    rotation_needed = cfg.bypass_rotation_gate or (
                        last_hold_state is None
                        or not _states_equal(new_hold_state, last_hold_state)
                    )
                    skip_syms: set[str] = set()
                    if sweep_sym:
                        skip_syms.add(sweep_sym)
                    if cfg.include_dca and settings is not None:
                        skip_syms.update(
                            t.symbol.strip().upper() for t in settings.data.dca_targets
                        )
                    prices: dict[str, float] = {}
                    for sym in set(positions) | set(targets):
                        if sym in skip_syms:
                            continue
                        df = normalized.get(sym)
                        if df is None:
                            continue
                        px = _close_price_from_normalized(df, d)
                        if px is not None:
                            prices[sym] = px
                    drift_breach = (
                        not rotation_needed
                        and last_hold_state is not None
                        and drift_band_exceeded(
                            targets=targets,
                            positions=positions,
                            prices=prices,
                            equity=mtm,
                            skip_symbols=skip_syms,
                            band_pct=float(cfg.drift_band_pct),
                        )
                    )
                    drift_needed = cfg.bypass_drift_gate or drift_breach
                    target_refresh_needed = cfg.bypass_target_refresh_gate or (
                        not rotation_needed
                        and last_targets is not None
                        # Inert for momentum under bound dominance (min(w*m,cap) == cap when
                        # m >= cap); live if cap scales as c_t = m_t * c (DR #6).
                        and target_weights_changed(last_targets, targets)
                    )
                    if rotation_needed or drift_needed or target_refresh_needed:
                        if rotation_needed and last_hold_state is not None:
                            discrete_state_changes += 1
                        if drift_breach:
                            maintenance_band_breaches += 1
                        equity = mtm
                        reserve_cash = equity * min_reserve_pct if use_risk else 0.0
                        trades_before_rebalance = len(trades)
                        cash, positions = rebalance_to_targets(
                            d=d,
                            targets=targets,
                            equity=equity,
                            cash=cash,
                            positions=positions,
                            prices=prices,
                            bps=bps,
                            min_trade_usd=min_trade_usd,
                            min_order_floor=min_order_floor,
                            reserve_cash=reserve_cash,
                            use_risk=use_risk,
                            trades=trades,
                            skip_symbols=skip_syms,
                        )
                        if not targets:
                            empty_target_liquidation_trades += (
                                len(trades) - trades_before_rebalance
                            )
                        last_hold_state = new_hold_state
                        last_targets = dict(targets)

            if dca_strategy is not None and settings is not None:
                if cfg.use_regime and regime is not None and cfg.pin_dca_regime_multiplier is None:
                    dca_sigs = dca_strategy.generate_signals(
                        data,
                        as_of=as_of,
                        market_regime=regime,
                    )
                else:
                    dca_sigs = dca_strategy.generate_signals(data, as_of=as_of)
                if cfg.pin_dca_regime_multiplier is not None:
                    pin = float(cfg.pin_dca_regime_multiplier)
                    if abs(pin - 1.0) > 1e-12:
                        dca_sigs = [
                            s.model_copy(update={"weight": round(float(s.weight) * pin, 6)})
                            for s in dca_sigs
                        ]
                if dca_sigs:
                    from src.strategy.regime_params import effective_dca_amount

                    equity = mtm_now(cash, positions, last_marks, normalized, d)
                    if cfg.dca_fixed_amount_usd is not None:
                        budget = float(cfg.dca_fixed_amount_usd)
                    else:
                        budget = effective_dca_amount(settings, None, equity=equity)
                    reserve_cash = equity * min_reserve_pct if use_risk else 0.0
                    cash, positions = apply_dca_contributions(
                        d=d,
                        signals=dca_sigs,
                        dca_budget=budget,
                        equity=equity,
                        cash=cash,
                        positions=positions,
                        last_marks=last_marks,
                        normalized=normalized,
                        bps=bps,
                        max_position_pct=max_position_pct,
                        min_order_floor=min_order_floor,
                        reserve_cash=reserve_cash,
                        use_risk=use_risk,
                        trades=trades,
                    )

            if cfg.include_cash_sweep and settings is not None and sweep_sym:
                cash, positions = apply_cash_sweep(
                    d=d,
                    cash=cash,
                    positions=positions,
                    last_marks=last_marks,
                    normalized=normalized,
                    sweep_sym=sweep_sym,
                    settings=settings,
                    bps=bps,
                    trades=trades,
                )

        final_equity = initial
        if equity_curve:
            final_equity = float(equity_curve[-1]["equity"])

        rets = _daily_returns_from_equity_curve(equity_curve)
        rf_daily = pd.Series(rf_by_date, dtype=float)
        metrics = compute_return_metrics(
            rets,
            risk_free_rate_annual=cash_yield,
            risk_free_daily=rf_daily,
        )

        zero_trades_warning: str | None = None
        if not trades:
            zero_trades_warning = (
                "Backtest completed with 0 trades. This usually means the strategy "
                "could not generate signals over the requested window — common causes: "
                "insufficient lookback (e.g. momentum needs 252+ bars), restrictive filters, "
                "or no rebalance dates fell in the trading days. Available trading days: "
                f"{len(trading_days)}; rebalance dates: {len(rebal_on)}."
            )
            logger.warning("%s", zero_trades_warning)

        if yield_series is None:
            cash_yield_mode = "flat"
        elif yield_modes <= {"historical"}:
            cash_yield_mode = "historical"
        elif yield_modes <= {"fallback_flat"}:
            cash_yield_mode = "fallback_flat"
        else:
            cash_yield_mode = "mixed"
        realized = float(sum(applied_yields) / len(applied_yields)) if applied_yields else 0.0
        avg_exp = float(sum(exposures) / len(exposures)) if exposures else 0.0
        max_exp = float(max(exposures)) if exposures else 0.0
        max_exp_date = (
            exposure_dates[exposures.index(max_exp)] if exposures and exposure_dates else None
        )
        mean_eq = (
            float(sum(float(p["equity"]) for p in equity_curve) / len(equity_curve))
            if equity_curve
            else initial
        )
        traded = sum(float(t.qty) * float(t.price) for t in trades)
        n_years = max(len(trading_days) / 252.0, 1e-9)
        turnover = (traded / (2.0 * mean_eq) / n_years) if mean_eq > 0.0 else 0.0

        maintenance_trades = max(0, len(trades) - (2 * discrete_state_changes + 1))

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            return_metrics=metrics,
            initial_capital=initial,
            final_equity=final_equity,
            zero_trades_warning=zero_trades_warning,
            sizing_mode=sizing_mode,
            cash_yield_annual_pct=cash_yield,
            realized_cash_yield_annual_pct=realized,
            cash_yield_mode=cash_yield_mode,
            peak_single_symbol_allocation_pct=peak_alloc_pct,
            avg_exposure_pct=avg_exp,
            max_exposure_pct=max_exp,
            max_exposure_date=max_exp_date,
            run_kind=run_kind,
            turnover=float(turnover),
            experiment_label=cfg.experiment_label,
            discrete_state_changes=discrete_state_changes,
            maintenance_band_breaches=maintenance_band_breaches,
            maintenance_trades=maintenance_trades,
            empty_target_periods=empty_target_periods,
            empty_target_liquidation_trades=empty_target_liquidation_trades,
        )


def _daily_returns_from_equity_curve(equity_curve: list[dict[str, Any]]) -> pd.Series:
    """Build a date-aligned daily return series, skipping non-positive prior equity.

    Skipped days are omitted from **both** values and index so a protective length
    mismatch cannot silently attach returns to the wrong date (Tier 49A).
    """
    if len(equity_curve) < 2:
        return pd.Series(dtype=float)
    eqs = [float(p["equity"]) for p in equity_curve]
    idx = [p["date"] for p in equity_curve]
    rlist: list[float] = []
    ridx: list[str] = []
    for i in range(1, len(eqs)):
        prev = eqs[i - 1]
        if prev <= 0.0:
            continue
        rlist.append((eqs[i] - prev) / prev)
        ridx.append(idx[i])
    return pd.Series(rlist, index=pd.Index(ridx, name="date"))
