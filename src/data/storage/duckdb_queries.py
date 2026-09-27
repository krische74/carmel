"""Analytical queries over Parquet OHLCV files (multi-symbol, date windows)."""

from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd


class DuckDBQueries:
    """Run DuckDB SQL over per-symbol Parquet files produced by ``ParquetStore``."""

    def __init__(self, parquet_dir: str | Path) -> None:
        self._dir = Path(parquet_dir)

    def query_ohlcv(
        self,
        symbols: Sequence[str],
        *,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """Return stacked OHLCV rows with a ``symbol`` column.

        Args:
            symbols: Ticker symbols (files ``{SYMBOL}.parquet`` under ``parquet_dir``).
            start: Inclusive lower bound on ``date`` (or ``None`` for open-ended).
            end: Inclusive upper bound on ``date`` (or ``None`` for open-ended).

        Returns:
            DataFrame with columns ``symbol``, ``date``, OHLCV columns.
        """
        parts: list[str] = []
        for sym in symbols:
            safe = sym.strip().upper().replace("/", "_")
            path = self._dir / f"{safe}.parquet"
            if not path.exists():
                continue
            p = path.as_posix().replace("'", "''")
            parts.append(f"SELECT '{safe}' AS symbol, * FROM read_parquet('{p}')")
        if not parts:
            return pd.DataFrame(
                columns=["symbol", "date", "open", "high", "low", "close", "volume"]
            )

        union_sql = " UNION ALL BY NAME ".join(parts)
        where_clauses: list[str] = []
        if start is not None:
            where_clauses.append(f"date >= TIMESTAMP '{pd.Timestamp(start).isoformat()}'")
        if end is not None:
            where_clauses.append(f"date <= TIMESTAMP '{pd.Timestamp(end).isoformat()}'")
        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"SELECT * FROM ({union_sql}) AS u{where_sql} ORDER BY symbol, date"
        con = duckdb.connect(database=":memory:")
        try:
            return con.execute(sql).df()
        finally:
            con.close()
