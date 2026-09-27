"""Unit tests for DataPipeline orchestration (adapters mocked)."""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import DataConfig
from src.data.adapters.base import MarketDataAdapter
from src.data.pipeline import DataPipeline
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore


class FakeAdapter(MarketDataAdapter):
    """Deterministic adapter for pipeline tests."""

    def __init__(self, ohlcv: pd.DataFrame | None = None, *, fail_times: int = 0) -> None:
        self._ohlcv = ohlcv
        self._fail_times = fail_times
        self._calls = 0
        self._last_start: str | pd.Timestamp | None = None
        self._last_end: str | pd.Timestamp | None = None

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        self._calls += 1
        self._last_start = start
        self._last_end = end
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ConnectionError("network down")
        assert self._ohlcv is not None
        return self._ohlcv.copy()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()


def test_pipeline_stores_valid_ohlcv(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    adapter = FakeAdapter(sample_ohlcv)
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "meta.db")
    pipeline = DataPipeline(adapter=adapter, parquet_store=pq, sqlite_store=meta)
    result = pipeline.ingest_ohlcv("SPY")
    assert result.success is True
    loaded = pq.read_ohlcv("SPY")
    pd.testing.assert_frame_equal(
        loaded.sort_index(),
        sample_ohlcv.sort_index(),
        check_freq=False,
        check_dtype=False,
    )


def test_pipeline_rejects_invalid_ohlcv(
    tmp_path: Path,
    sample_ohlcv_invalid: pd.DataFrame,
) -> None:
    adapter = FakeAdapter(sample_ohlcv_invalid)
    pq = ParquetStore(tmp_path / "pq")
    pipeline = DataPipeline(adapter=adapter, parquet_store=pq, sqlite_store=None)
    result = pipeline.ingest_ohlcv("BAD")
    assert result.success is False
    assert result.validation is not None
    assert result.validation.is_valid is False
    assert pq.read_ohlcv("BAD").empty


def test_pipeline_retries_fetch_then_succeeds(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    adapter = FakeAdapter(sample_ohlcv, fail_times=1)
    pq = ParquetStore(tmp_path / "pq")
    pipeline = DataPipeline(adapter=adapter, parquet_store=pq, sqlite_store=None, max_retries=3)
    result = pipeline.ingest_ohlcv("SPY")
    assert result.success is True
    assert adapter._calls == 2


def test_pipeline_logs_after_exhausting_retries(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = FakeAdapter(sample_ohlcv, fail_times=5)
    pq = ParquetStore(tmp_path / "pq")
    pipeline = DataPipeline(adapter=adapter, parquet_store=pq, sqlite_store=None, max_retries=2)
    caplog.set_level(logging.ERROR)
    result = pipeline.ingest_ohlcv("SPY")
    assert result.success is False
    assert any(
        "ingest" in r.message.lower() or "fetch" in r.message.lower() for r in caplog.records
    )


def test_pipeline_same_store_with_different_adapter_types(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    """Downstream storage behavior does not depend on which adapter produced the frame."""
    pq = ParquetStore(tmp_path / "pq")
    a1 = FakeAdapter(sample_ohlcv)
    a2 = FakeAdapter(sample_ohlcv)
    p1 = DataPipeline(adapter=a1, parquet_store=pq, sqlite_store=None)
    p2 = DataPipeline(adapter=a2, parquet_store=pq, sqlite_store=None)
    assert p1.ingest_ohlcv("X").success is True
    assert p2.ingest_ohlcv("Y").success is True
    assert len(pq.read_ohlcv("X")) == len(pq.read_ohlcv("Y"))


class FredMacroAdapter(MarketDataAdapter):
    """FRED-like macro fetch for pipeline tests."""

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "value": [4.0, 4.1, 4.2, 4.3, 4.4],
            },
        )


class RaisingMacroAdapter(MarketDataAdapter):
    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        raise ConnectionError("FRED down")


def test_ingest_macro_stores_observations_in_sqlite(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "m.db")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
        fred_adapter=FredMacroAdapter(),
    )
    res = pipeline.ingest_macro("DGS10")
    assert res.success is True
    rows = meta.get_macro_indicator("DGS10")
    assert len(rows) == 5
    assert rows[-1]["value"] == pytest.approx(4.4)


def test_ingest_macro_returns_failure_when_fred_raises(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "m2.db")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
        fred_adapter=RaisingMacroAdapter(),
    )
    res = pipeline.ingest_macro("DGS10")
    assert res.success is False
    assert res.error is not None


def test_ingest_macro_skipped_when_no_fred_adapter(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "m3.db")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
        fred_adapter=None,
    )
    res = pipeline.ingest_macro("DGS10")
    assert res.success is True
    assert res.error is not None
    assert "skipped" in res.error.lower()
    assert meta.get_latest_macro("DGS10") is None


def test_ingest_macro_fails_when_no_sqlite_store(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    """M17: ingest_macro with valid fred_adapter but sqlite_store=None returns failure."""
    pq = ParquetStore(tmp_path / "pq")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=None,
        fred_adapter=FredMacroAdapter(),
    )
    res = pipeline.ingest_macro("DGS10")
    assert res.success is False
    assert "sqlite" in (res.error or "").lower()


class BadColumnMacroAdapter(MarketDataAdapter):
    """Returns a DataFrame with wrong column names."""

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"dt": ["2024-01-01"], "val": [1.0]})


def test_ingest_macro_fails_on_malformed_columns(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    """M18: FRED adapter returns frame missing 'date'/'value' columns."""
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "m4.db")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
        fred_adapter=BadColumnMacroAdapter(),
    )
    res = pipeline.ingest_macro("DGS10")
    assert res.success is False
    assert res.error is not None
    assert "Invalid" in res.error


def test_ingest_macro_retries_on_transient_failure(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    """M5: ingest_macro retries like ingest_ohlcv."""

    class FailOnceMacroAdapter(MarketDataAdapter):
        def __init__(self) -> None:
            self._calls = 0

        def fetch_ohlcv(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {}

        def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
            self._calls += 1
            if self._calls < 2:
                raise ConnectionError("FRED transient failure")
            return pd.DataFrame(
                {"date": pd.date_range("2024-01-01", periods=2, freq="D"), "value": [1.0, 2.0]}
            )

    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "m5.db")
    adapter = FailOnceMacroAdapter()
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
        fred_adapter=adapter,
        max_retries=3,
    )
    res = pipeline.ingest_macro("DGS10")
    assert res.success is True
    assert adapter._calls == 2
    rows = meta.get_macro_indicator("DGS10")
    assert len(rows) == 2


def test_ingest_fundamentals_writes_score(tmp_path: Path, sample_ohlcv: pd.DataFrame) -> None:
    class FundAdapter(MarketDataAdapter):
        def fetch_ohlcv(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

        def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "symbol": "AAPL",
                "period": "2024",
                "total_assets": 100.0,
                "net_income": 10.0,
                "operating_cash_flow": 12.0,
            }

    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "fund.db")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
        edgar_adapter=FundAdapter(),
    )
    res = pipeline.ingest_fundamentals("AAPL")
    assert res.success is True
    row = meta.get_latest_fundamental_score("AAPL")
    assert row is not None
    assert row["symbol"] == "AAPL"
    assert int(row["score"]) >= 0


def test_ingest_fundamentals_skips_without_edgar(
    tmp_path: Path, sample_ohlcv: pd.DataFrame
) -> None:
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "nf.db")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
    )
    res = pipeline.ingest_fundamentals("AAPL")
    assert res.success is False
    assert "edgar" in (res.error or "").lower()


def _shv_like_ohlcv(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1e-4, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0001
    low = np.minimum(open_, close) * 0.9999
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_pipeline_uses_configured_validation_thresholds(
    tmp_path: Path,
) -> None:
    """30-bar low-volatility frame passes ingest with Tier-33 default thresholds."""
    low_vol = _shv_like_ohlcv(30)
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "meta.db")
    dcfg = DataConfig()
    pipeline = DataPipeline(
        adapter=FakeAdapter(low_vol),
        parquet_store=pq,
        sqlite_store=meta,
        ohlcv_outlier_zscore_threshold=dcfg.ohlcv_outlier_zscore_threshold,
        ohlcv_min_prior_for_zscore=dcfg.ohlcv_min_prior_for_zscore,
        ohlcv_max_abs_daily_return=dcfg.ohlcv_max_abs_daily_return,
    )
    result = pipeline.ingest_ohlcv("SHV")
    assert result.success is True
    assert not pq.read_ohlcv("SHV").empty


def test_pipeline_validation_can_be_disabled(
    tmp_path: Path,
) -> None:
    """When z-score thresholds are None, pathological gaps do not fail validation."""
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    close = [100.0, 100.0, 100.0, 100.0, 160.0]
    wild = pd.DataFrame(
        {
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [1e6] * 5,
        },
        index=idx,
    )
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "meta2.db")
    dc = DataConfig(
        ohlcv_outlier_zscore_threshold=None,
        ohlcv_max_abs_daily_return=2.0,
    )
    pipeline = DataPipeline(
        adapter=FakeAdapter(wild),
        parquet_store=pq,
        sqlite_store=meta,
        ohlcv_outlier_zscore_threshold=dc.ohlcv_outlier_zscore_threshold,
        ohlcv_min_prior_for_zscore=dc.ohlcv_min_prior_for_zscore,
        ohlcv_max_abs_daily_return=dc.ohlcv_max_abs_daily_return,
    )
    assert pipeline.ingest_ohlcv("WILD").success is True


def test_pipeline_strict_thresholds_still_reject_real_outliers(
    tmp_path: Path,
) -> None:
    """With enough history, a 50%+ close-to-close gap is still rejected."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    close[-1] = 40.0
    df = pd.DataFrame(
        {
            "open": close,
            "high": np.maximum.accumulate(close) * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "meta3.db")
    dcfg = DataConfig()
    pipeline = DataPipeline(
        adapter=FakeAdapter(df),
        parquet_store=pq,
        sqlite_store=meta,
        ohlcv_outlier_zscore_threshold=dcfg.ohlcv_outlier_zscore_threshold,
        ohlcv_min_prior_for_zscore=dcfg.ohlcv_min_prior_for_zscore,
        ohlcv_max_abs_daily_return=dcfg.ohlcv_max_abs_daily_return,
    )
    result = pipeline.ingest_ohlcv("CRASH")
    assert result.success is False
    assert result.validation is not None
    assert result.validation.is_valid is False


def test_pipeline_accepts_data_with_real_crash(tmp_path: Path) -> None:
    """Single-day ~12% drop passes default max_abs_daily_return."""
    n = 500
    idx = pd.bdate_range("2023-01-03", periods=n, freq="B")
    rng = np.random.default_rng(1)
    close = np.full(n, 100.0)
    for i in range(1, n):
        close[i] = close[i - 1] * (1.0 + rng.normal(0.0, 0.008))
    close[250] = close[249] * 0.88
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=idx,
    )
    pq = ParquetStore(tmp_path / "pq_crash_ok")
    meta = SQLiteStore(tmp_path / "meta_crash_ok.db")
    dcfg = DataConfig()
    pipeline = DataPipeline(
        adapter=FakeAdapter(df),
        parquet_store=pq,
        sqlite_store=meta,
        ohlcv_outlier_zscore_threshold=dcfg.ohlcv_outlier_zscore_threshold,
        ohlcv_min_prior_for_zscore=dcfg.ohlcv_min_prior_for_zscore,
        ohlcv_max_abs_daily_return=dcfg.ohlcv_max_abs_daily_return,
    )
    assert pipeline.ingest_ohlcv("CRASHOK").success is True


def test_pipeline_rejects_data_with_50pct_daily_return(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-02", periods=2, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 201.0],
            "low": [99.0, 100.0],
            "close": [100.0, 200.0],
            "volume": [1e6, 1e6],
        },
        index=idx,
    )
    pq = ParquetStore(tmp_path / "pq_50")
    meta = SQLiteStore(tmp_path / "meta_50.db")
    dcfg = DataConfig()
    pipeline = DataPipeline(
        adapter=FakeAdapter(df),
        parquet_store=pq,
        sqlite_store=meta,
        ohlcv_outlier_zscore_threshold=dcfg.ohlcv_outlier_zscore_threshold,
        ohlcv_min_prior_for_zscore=dcfg.ohlcv_min_prior_for_zscore,
        ohlcv_max_abs_daily_return=dcfg.ohlcv_max_abs_daily_return,
    )
    result = pipeline.ingest_ohlcv("JUMP")
    assert result.success is False
    assert result.validation is not None
    assert result.validation.is_valid is False


class NanMacroAdapter(MarketDataAdapter):
    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                "value": [4.0, float("nan"), 4.2],
            },
        )


def test_ingest_macro_skips_nan_values(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "nan_macro.db")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
        fred_adapter=NanMacroAdapter(),
    )
    assert pipeline.ingest_macro("DGS10").success is True
    rows = meta.get_macro_indicator("DGS10")
    assert len(rows) == 2
    vals = sorted(float(r["value"]) for r in rows)
    assert vals == pytest.approx([4.0, 4.2])


class AllNanMacroAdapter(MarketDataAdapter):
    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "value": [float("nan"), float("nan")],
            },
        )


def test_ingest_macro_all_nan_returns_success(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "allnan_macro.db")
    pipeline = DataPipeline(
        adapter=FakeAdapter(sample_ohlcv),
        parquet_store=pq,
        sqlite_store=meta,
        fred_adapter=AllNanMacroAdapter(),
    )
    res = pipeline.ingest_macro("DGS2")
    assert res.success is True
    assert meta.get_macro_indicator("DGS2") == []


def _sample_frame(idx: pd.DatetimeIndex) -> pd.DataFrame:
    close = 100.0 + np.linspace(0, 0.5, len(idx), dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = np.r_[close[0], close[:-1]]
    open_ = np.minimum(np.maximum(open_, low), high)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(len(idx), 1e6),
        },
        index=idx,
    )


class _SeqOhlcvAdapter(MarketDataAdapter):
    """Returns successive frames from ``frames`` for each fetch."""

    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self._frames = frames
        self._i = 0

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        f = self._frames[self._i]
        self._i += 1
        return f.copy()

    def fetch_fundamentals(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return {}

    def fetch_macro(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()


def test_pipeline_ingest_ohlcv_passes_date_range_to_adapter(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    adapter = FakeAdapter(sample_ohlcv)
    pq = ParquetStore(tmp_path / "pq")
    meta = SQLiteStore(tmp_path / "dr.db")
    pipeline = DataPipeline(adapter=adapter, parquet_store=pq, sqlite_store=meta)
    pipeline.ingest_ohlcv("SPY", start="2023-01-01", end="2023-06-01")
    assert adapter._last_start is not None
    assert pd.Timestamp(adapter._last_start).date() == date(2023, 1, 1)
    assert adapter._last_end is not None
    assert pd.Timestamp(adapter._last_end).date() == date(2023, 6, 1)


def test_pipeline_ingest_ohlcv_no_dates_uses_adapter_default(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    adapter = FakeAdapter(sample_ohlcv)
    pq = ParquetStore(tmp_path / "pq2")
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=pq,
        sqlite_store=None,
        initial_backfill_years=0,
    )
    pipeline.ingest_ohlcv("SPY")
    assert adapter._last_start is None
    assert adapter._last_end is None


def test_pipeline_uses_append_not_overwrite(tmp_path: Path) -> None:
    idx1 = pd.bdate_range("2024-01-01", periods=30, freq="B")
    idx2 = pd.bdate_range("2024-02-15", periods=5, freq="B")
    df1 = _sample_frame(idx1)
    df2 = _sample_frame(idx2)
    adapter = _SeqOhlcvAdapter([df1, df2])
    pq = ParquetStore(tmp_path / "pq3")
    pipeline = DataPipeline(adapter=adapter, parquet_store=pq, sqlite_store=None, initial_backfill_years=0)
    assert pipeline.ingest_ohlcv("ZZZ").success is True
    assert pipeline.ingest_ohlcv("ZZZ").success is True
    loaded = pq.read_ohlcv("ZZZ")
    assert len(loaded) == 35


def test_pipeline_dedup_overlapping_dates(tmp_path: Path) -> None:
    idx1 = pd.bdate_range("2024-01-01", periods=30, freq="B")
    idx2 = pd.bdate_range("2024-01-10", periods=20, freq="B")
    df1 = _sample_frame(idx1)
    df2 = _sample_frame(idx2)
    overlap = len(set(idx1.normalize()) & set(idx2.normalize()))
    adapter = _SeqOhlcvAdapter([df1, df2])
    pq = ParquetStore(tmp_path / "pq4")
    pipeline = DataPipeline(adapter=adapter, parquet_store=pq, sqlite_store=None, initial_backfill_years=0)
    assert pipeline.ingest_ohlcv("OVL").success is True
    assert pipeline.ingest_ohlcv("OVL").success is True
    loaded = pq.read_ohlcv("OVL")
    assert len(loaded) == 30 + 20 - overlap
    assert not loaded.index.duplicated().any()


def test_pipeline_first_ingest_backfills_history(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    adapter = FakeAdapter(sample_ohlcv)
    pq = ParquetStore(tmp_path / "pq5")
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=pq,
        sqlite_store=None,
        initial_backfill_years=3,
    )
    pipeline.ingest_ohlcv("NEW")
    assert adapter._last_start is not None
    ts = pd.Timestamp(adapter._last_start)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    age = pd.Timestamp.now(tz="UTC") - ts
    assert timedelta(days=365 * 2) < age < timedelta(days=365 * 4)


def test_pipeline_subsequent_ingest_uses_default_window(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    pq = ParquetStore(tmp_path / "pq6")
    pq.write_ohlcv("OLD", sample_ohlcv)
    adapter = FakeAdapter(sample_ohlcv)
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=pq,
        sqlite_store=None,
        initial_backfill_years=3,
    )
    pipeline.ingest_ohlcv("OLD")
    assert adapter._last_start is None
    assert adapter._last_end is None


def test_pipeline_backfill_disabled_when_zero(
    tmp_path: Path,
    sample_ohlcv: pd.DataFrame,
) -> None:
    adapter = FakeAdapter(sample_ohlcv)
    pq = ParquetStore(tmp_path / "pq7")
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=pq,
        sqlite_store=None,
        initial_backfill_years=0,
    )
    pipeline.ingest_ohlcv("X")
    assert adapter._last_start is None


def _ohlcv_seventy_pct_close_spike() -> pd.DataFrame:
    """Three rows with a 70% single-day close jump (rejected for equities at max_abs 0.5)."""
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 170.0],
            "high": [102.0, 175.0, 176.0],
            "low": [99.0, 99.0, 160.0],
            "close": [100.0, 170.0, 165.0],
            "volume": [1e6, 1e6, 1e6],
        },
        index=idx,
    )


def test_pipeline_exempt_symbol_skips_daily_return_check(tmp_path: Path) -> None:
    """^VIX-like exempt symbols allow large daily moves without failing validation."""
    df = _ohlcv_seventy_pct_close_spike()
    adapter = FakeAdapter(df)
    pq = ParquetStore(tmp_path / "pq_exempt_ok")
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=pq,
        sqlite_store=None,
        ohlcv_max_abs_daily_return=0.5,
        validation_exempt_symbols=["^VIX"],
    )
    result = pipeline.ingest_ohlcv("^VIX")
    assert result.success is True
    assert not pq.read_ohlcv("^VIX").empty


def test_pipeline_non_exempt_symbol_still_rejects_large_return(tmp_path: Path) -> None:
    """Same 70% move data fails for a non-exempt symbol (regression guard)."""
    df = _ohlcv_seventy_pct_close_spike()
    adapter = FakeAdapter(df)
    pq = ParquetStore(tmp_path / "pq_exempt_fail")
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=pq,
        sqlite_store=None,
        ohlcv_max_abs_daily_return=0.5,
        validation_exempt_symbols=["^VIX"],
    )
    result = pipeline.ingest_ohlcv("SPY")
    assert result.success is False
    assert result.validation is not None
    assert result.validation.is_valid is False


def test_pipeline_exempt_symbol_still_checks_nan(tmp_path: Path) -> None:
    """Exempt symbols still fail on NaN in required columns."""
    df = _ohlcv_seventy_pct_close_spike()
    df.loc[df.index[2], "close"] = float("nan")
    adapter = FakeAdapter(df)
    pq = ParquetStore(tmp_path / "pq_exempt_nan")
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=pq,
        sqlite_store=None,
        ohlcv_max_abs_daily_return=0.5,
        validation_exempt_symbols=["^VIX"],
    )
    result = pipeline.ingest_ohlcv("^VIX")
    assert result.success is False
    assert result.validation is not None
    assert any("NaN" in e for e in result.validation.errors)


def test_pipeline_exempt_symbol_still_checks_ohlc_consistency(tmp_path: Path) -> None:
    """Exempt symbols still fail when low > high."""
    df = _ohlcv_seventy_pct_close_spike()
    df.loc[df.index[1], "low"] = 200.0
    adapter = FakeAdapter(df)
    pq = ParquetStore(tmp_path / "pq_exempt_ohlc")
    pipeline = DataPipeline(
        adapter=adapter,
        parquet_store=pq,
        sqlite_store=None,
        ohlcv_max_abs_daily_return=0.5,
        validation_exempt_symbols=["^VIX"],
    )
    result = pipeline.ingest_ohlcv("^VIX")
    assert result.success is False
    assert result.validation is not None
    assert any("low" in e.lower() and "high" in e.lower() for e in result.validation.errors)
