"""Saved backtest runs (read-only)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_sqlite_store
from src.api.pagination import PaginatedResponse
from src.data.storage.sqlite_store import SQLiteStore

router = APIRouter(tags=["backtests"])


def _clamp_offset(offset: int | None) -> int:
    if offset is None:
        return 0
    return max(0, min(int(offset), 1_000_000))


@router.get("/backtests", response_model=PaginatedResponse)
def list_backtests(
    sqlite: SQLiteStore = Depends(get_sqlite_store),
    limit: int | None = Query(default=50, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
) -> PaginatedResponse:
    """Recent backtest run summaries (no large JSON columns)."""
    lim = max(1, min(int(limit or 50), 10_000))
    off = _clamp_offset(offset)
    items = sqlite.get_backtest_runs(limit=lim, offset=off)
    total = sqlite.count_backtest_runs()
    return PaginatedResponse(items=items, total=total, limit=lim, offset=off)


@router.get("/backtests/{run_id}")
def get_backtest(
    run_id: int,
    sqlite: SQLiteStore = Depends(get_sqlite_store),
) -> dict[str, Any]:
    """One backtest run including equity and trades JSON."""
    row = sqlite.get_backtest_run(run_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest run not found")
    return row
