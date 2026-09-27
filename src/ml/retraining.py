"""Scheduled retraining of the momentum ML bundle from Parquet OHLCV."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from src.config import Settings

logger = logging.getLogger(__name__)


def retrain_ml_bundle(settings: Settings) -> Path | None:
    """Load recent OHLCV from Parquet, train bundle, persist to ``ml.model_path``.

    Returns the written path on success, or ``None`` on skipped / recoverable failure.
    """
    from src.automation.strategy_wiring import build_strategy_for_backtest
    from src.config import parquet_dir
    from src.data.storage.parquet_store import ParquetStore
    from src.ml.train import train_momentum_model_bundle
    from src.strategy.momentum import MomentumRotationStrategy

    try:
        strat = build_strategy_for_backtest(settings, "momentum")
    except ValueError as exc:
        logger.warning("ML retraining skipped (strategy config): %s", exc)
        return None

    if not isinstance(strat, MomentumRotationStrategy):
        logger.warning("ML retraining supports momentum strategy only; skipping.")
        return None

    parquet = ParquetStore(parquet_dir(settings))
    symbols_all = strat.get_universe()
    vix_u = settings.regime.vix_symbol.strip().upper()
    train_syms = [s for s in symbols_all if s.strip().upper() != vix_u]

    data: dict[str, Any] = {}
    missing: list[str] = []
    for sym in symbols_all:
        df = parquet.read_ohlcv(sym)
        if df.empty:
            missing.append(sym)
        else:
            data[sym] = df

    if missing:
        logger.warning(
            "ML retraining skipped: missing Parquet OHLCV for %s",
            ", ".join(missing),
        )
        return None

    end = date.today()
    start = end - timedelta(days=int(settings.ml.retraining_lookback_days))

    try:
        out = train_momentum_model_bundle(
            settings,
            data,
            symbols=train_syms,
            start=start,
            end=end,
        )
    except ValueError as exc:
        logger.warning("ML retraining skipped (insufficient data): %s", exc)
        return None
    except OSError as exc:
        logger.error("ML retraining failed (OS error): %s", exc)
        return None
    except Exception as exc:
        mod = getattr(type(exc), "__module__", "") or ""
        if mod.startswith("joblib"):
            logger.error("ML retraining failed (joblib): %s", exc)
            return None
        raise

    logger.info("ML retraining saved bundle to %s", out)
    return out
