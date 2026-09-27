"""Orchestrates fetch -> validate -> store with retries and structured logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from src.data.validation import OHLCVValidationResult, validate_ohlcv

if TYPE_CHECKING:
    from src.data.adapters.base import MarketDataAdapter
    from src.data.storage.parquet_store import ParquetStore
    from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """Outcome of a single-symbol OHLCV ingest run."""

    success: bool
    validation: OHLCVValidationResult | None = None
    error: str | None = None


class DataPipeline:
    """Run a ``MarketDataAdapter`` through validation into ``ParquetStore``."""

    def __init__(
        self,
        *,
        adapter: MarketDataAdapter,
        parquet_store: ParquetStore,
        sqlite_store: SQLiteStore | None,
        fred_adapter: MarketDataAdapter | None = None,
        edgar_adapter: MarketDataAdapter | None = None,
        max_retries: int = 3,
        ohlcv_outlier_zscore_threshold: float | None = 4.0,
        ohlcv_min_prior_for_zscore: int = 20,
        ohlcv_max_abs_daily_return: float = 0.5,
        validation_exempt_symbols: list[str] | None = None,
        initial_backfill_years: int = 3,
    ) -> None:
        self._adapter = adapter
        self._fred_adapter = fred_adapter
        self._edgar_adapter = edgar_adapter
        self._parquet = parquet_store
        self._sqlite = sqlite_store
        self._max_retries = max(1, max_retries)
        self._ohlcv_outlier_zscore_threshold = ohlcv_outlier_zscore_threshold
        self._ohlcv_min_prior_for_zscore = ohlcv_min_prior_for_zscore
        self._ohlcv_max_abs_daily_return = float(ohlcv_max_abs_daily_return)
        raw_exempt = validation_exempt_symbols if validation_exempt_symbols is not None else []
        self._validation_exempt_symbols = frozenset(
            s.strip().upper() for s in raw_exempt if str(s).strip() != ""
        )
        self._initial_backfill_years = max(0, int(initial_backfill_years))

    def ingest_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> IngestResult:
        """Fetch OHLCV for ``symbol``, validate, and persist on success."""
        sym = symbol.strip().upper()
        fetch_start = start
        if (
            fetch_start is None
            and self._initial_backfill_years > 0
            and not self._parquet.has_ohlcv_file(sym)
        ):
            fetch_start = datetime.now(UTC) - timedelta(days=365 * self._initial_backfill_years)
        last_error: str | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                raw = self._adapter.fetch_ohlcv(sym, start=fetch_start, end=end)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "fetch failed for %s (attempt %s/%s): %s",
                    sym,
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt >= self._max_retries:
                    logger.error(
                        "ingest aborted for %s after %s failed fetch attempts: %s",
                        sym,
                        self._max_retries,
                        last_error,
                    )
                    return IngestResult(success=False, error=last_error)
                continue

            effective_max_return = self._ohlcv_max_abs_daily_return
            if sym in self._validation_exempt_symbols:
                effective_max_return = 0.0  # disable daily return check (validator skips if <= 0)

            validation = validate_ohlcv(
                raw,
                outlier_zscore_threshold=self._ohlcv_outlier_zscore_threshold,
                min_prior_for_zscore=self._ohlcv_min_prior_for_zscore,
                max_abs_daily_return=effective_max_return,
            )
            if not validation.is_valid:
                logger.error(
                    "validation failed for %s: %s",
                    sym,
                    "; ".join(validation.errors),
                )
                return IngestResult(success=False, validation=validation)
            if validation.warnings:
                logger.warning(
                    "validation warnings for %s: %s",
                    sym,
                    "; ".join(validation.warnings),
                )

            self._parquet.append_ohlcv(sym, raw)
            if self._sqlite is not None:
                self._sqlite.register_symbol(sym, source=self._adapter.__class__.__name__)
                self._sqlite.record_last_fetch(sym, datetime.now(UTC).isoformat())
            return IngestResult(success=True, validation=validation)

        return IngestResult(success=False, error=last_error or "unknown error")

    def ingest_macro(self, series_id: str) -> IngestResult:
        """Fetch a FRED macro series and persist to SQLite (with retries)."""
        if self._fred_adapter is None:
            return IngestResult(success=True, error="fred_adapter not configured -- skipped")
        if self._sqlite is None:
            return IngestResult(success=False, error="SQLite store not configured")
        sid = series_id.strip().upper()
        last_error: str | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                df = self._fred_adapter.fetch_macro(sid)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "FRED fetch failed for %s (attempt %s/%s): %s",
                    sid,
                    attempt,
                    self._max_retries,
                    exc,
                )
                if attempt >= self._max_retries:
                    logger.error(
                        "FRED ingest aborted for %s after %s failed attempts: %s",
                        sid,
                        self._max_retries,
                        last_error,
                    )
                    return IngestResult(success=False, error=last_error)
                continue
            if df is None or df.empty:
                return IngestResult(success=True)
            if "date" not in df.columns or "value" not in df.columns:
                logger.error("FRED macro frame for %s missing date/value columns", sid)
                return IngestResult(success=False, error="Invalid FRED macro frame")
            rows_tuples = list(df.itertuples(index=False))
            with self._sqlite._connect() as conn:
                fetched = pd.Timestamp.now(tz="UTC").isoformat()
                for row in rows_tuples:
                    if pd.isna(row.value):
                        continue
                    dt = pd.Timestamp(row.date)
                    d_str = dt.date().isoformat()
                    conn.execute(
                        """
                        INSERT INTO macro_indicators(date, series_id, value, fetched_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(date, series_id) DO UPDATE SET
                            value = excluded.value,
                            fetched_at = excluded.fetched_at
                        """,
                        (d_str, sid, float(row.value), fetched),
                    )
                conn.commit()
            return IngestResult(success=True)
        return IngestResult(success=False, error=last_error or "unknown error")

    def ingest_fundamentals(
        self,
        symbol: str,
        *,
        skip_symbols: list[str] | None = None,
    ) -> IngestResult:
        """Fetch fundamentals via Edgar adapter, compute F-score, persist to SQLite."""
        if self._edgar_adapter is None:
            return IngestResult(success=False, error="edgar_adapter not configured")
        if self._sqlite is None:
            return IngestResult(success=False, error="SQLite store not configured")
        sym = symbol.strip().upper()
        raw = self._edgar_adapter.fetch_fundamentals(sym, skip_symbols=skip_symbols)
        if not raw:
            return IngestResult(success=False, error="no fundamentals returned")
        from src.reporting.fundamental_score import compute_piotroski_f_score

        fs = compute_piotroski_f_score(raw)
        if fs is None:
            return IngestResult(success=False, error="F-score not computable")
        self._sqlite.write_fundamental_score(
            fs.symbol,
            fs.period,
            fs.score,
            fs.model_dump(),
        )
        return IngestResult(success=True)
