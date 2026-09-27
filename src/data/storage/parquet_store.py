"""Read/write OHLCV to Parquet files partitioned by symbol (one file per symbol)."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class ParquetStore:
    """Store OHLCV bars in ``{base_path}/{SYMBOL}.parquet``."""

    def __init__(self, base_path: str | Path) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path_for(self, symbol: str) -> Path:
        safe = symbol.strip().upper().replace("/", "_")
        return self._base / f"{safe}.parquet"

    def has_ohlcv_file(self, symbol: str) -> bool:
        """Return True if a Parquet file exists for ``symbol`` (may be empty)."""
        return self._path_for(symbol).exists()

    def write_ohlcv(self, symbol: str, df: pd.DataFrame) -> None:
        """Overwrite ``symbol`` file with ``df`` (DatetimeIndex)."""
        path = self._path_for(symbol)
        normalized = self._normalize_ohlcv_frame(df)
        table = self._dataframe_to_table(normalized)
        pq.write_table(table, path)

    def append_ohlcv(self, symbol: str, df: pd.DataFrame) -> None:
        """Append rows to ``symbol``, de-duplicating by index."""
        path = self._path_for(symbol)
        incoming = self._normalize_ohlcv_frame(df)
        if not path.exists():
            self.write_ohlcv(symbol, incoming)
            return
        existing = self.read_ohlcv(symbol)
        merged = pd.concat([existing, incoming])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged = merged.sort_index()
        self.write_ohlcv(symbol, merged)

    def read_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV for ``symbol`` (DatetimeIndex, OHLCV columns only)."""
        path = self._path_for(symbol)
        if not path.exists():
            return pd.DataFrame()
        table = pq.read_table(path)
        raw = table.to_pandas()
        if raw.empty:
            return pd.DataFrame()
        date_col = "date" if "date" in raw.columns else raw.columns[0]
        idx = pd.to_datetime(raw[date_col], utc=False)
        out = raw.drop(columns=[date_col], errors="ignore")
        for c in OHLCV_COLUMNS:
            if c not in out.columns:
                msg = f"expected column {c!r} in stored OHLCV"
                raise ValueError(msg)
        out = out[list(OHLCV_COLUMNS)].astype(float)
        out.index = pd.DatetimeIndex(idx)
        out.index.name = None
        return out

    @staticmethod
    def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)
        for c in OHLCV_COLUMNS:
            if c not in frame.columns:
                msg = f"missing OHLCV column {c!r}"
                raise ValueError(msg)
        frame = frame[list(OHLCV_COLUMNS)].astype(float)
        frame = frame.sort_index()
        frame.index.name = None
        return frame

    @staticmethod
    def _dataframe_to_table(df: pd.DataFrame) -> pa.Table:
        frame = df.reset_index()
        first = frame.columns[0]
        frame = frame.rename(columns={first: "date"})
        return pa.Table.from_pandas(frame, preserve_index=False)
