"""Operational alerts (read-only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_sqlite_store
from src.api.pagination import PaginatedResponse
from src.data.storage.sqlite_store import SQLiteStore

router = APIRouter(tags=["alerts"])


def _clamp_offset(offset: int | None) -> int:
    if offset is None:
        return 0
    return max(0, min(int(offset), 1_000_000))


@router.get("/alerts", response_model=PaginatedResponse)
def get_alerts(
    sqlite: SQLiteStore = Depends(get_sqlite_store),
    limit: int | None = Query(default=50, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
) -> PaginatedResponse:
    """Recent persisted alerts."""
    lim = max(1, min(int(limit or 50), 10_000))
    off = _clamp_offset(offset)
    items = sqlite.get_alerts(limit=lim, offset=off)
    total = sqlite.count_alerts()
    return PaginatedResponse(items=items, total=total, limit=lim, offset=off)
