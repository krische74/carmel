"""Path helpers for dashboard (read-only; mirrors runner layout)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import PROJECT_ROOT, get_settings
from src.config import hub_sqlite_path as hub_sqlite_path_for_settings
from src.config import parquet_dir as parquet_dir_for_settings

if TYPE_CHECKING:
    from pathlib import Path


def hub_sqlite_path() -> Path:
    """SQLite file shared with the trading runner and data pipeline."""
    return hub_sqlite_path_for_settings(get_settings())


def parquet_root() -> Path:
    """Parquet OHLCV root directory."""
    return parquet_dir_for_settings(get_settings())


def hub_parquet_path() -> Path:
    """Alias for :func:`parquet_root` (docs / sprint naming)."""
    return parquet_root()


def heartbeat_path() -> Path:
    """Process liveness file written by ``carmel run``."""
    return PROJECT_ROOT / "logs" / "hub.heartbeat"


def account_symbols() -> list[str]:
    """Symbols to query for position rows in portfolio snapshot."""
    s = get_settings()
    out: set[str] = {x.strip().upper() for x in s.data.universe}
    for t in s.data.dca_targets:
        out.add(t.symbol.strip().upper())
    out.add(s.strategy.momentum.cash_symbol.strip().upper())
    return sorted(out)
