"""Equity snapshots and return metrics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_sqlite_store
from src.data.storage.sqlite_store import SQLiteStore
from src.reporting.returns import ReturnMetrics, compute_daily_returns, compute_return_metrics

router = APIRouter(tags=["performance"])


@router.get("/performance/snapshots")
def get_snapshots(
    sqlite: SQLiteStore = Depends(get_sqlite_store),
    start: str | None = Query(default=None, description="YYYY-MM-DD inclusive"),
    end: str | None = Query(default=None, description="YYYY-MM-DD inclusive"),
) -> list[dict[str, Any]]:
    """Daily equity snapshot rows from SQLite."""
    return sqlite.get_equity_snapshots(start=start, end=end)


@router.get("/performance/returns", response_model=ReturnMetrics)
def get_returns(sqlite: SQLiteStore = Depends(get_sqlite_store)) -> ReturnMetrics:
    """Return analytics from stored equity snapshots (zeroed if fewer than 2 rows)."""
    snaps = sqlite.get_equity_snapshots()
    daily = compute_daily_returns(snaps)
    return compute_return_metrics(daily)
