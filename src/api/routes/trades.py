"""Trade signals and executions (read-only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_sqlite_store
from src.api.pagination import PaginatedResponse
from src.data.storage.sqlite_store import SQLiteStore

router = APIRouter(tags=["trades"])


def _clamp_limit(limit: int | None, *, default: int = 100, max_lim: int = 10_000) -> int:
    if limit is None:
        return default
    return max(1, min(int(limit), max_lim))


def _clamp_offset(offset: int | None) -> int:
    if offset is None:
        return 0
    return max(0, min(int(offset), 1_000_000))


@router.get("/trades/signals", response_model=PaginatedResponse)
def get_signals(
    sqlite: SQLiteStore = Depends(get_sqlite_store),
    limit: int | None = Query(default=100, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
) -> PaginatedResponse:
    """Recent strategy signals (newest first)."""
    lim = _clamp_limit(limit, default=100)
    off = _clamp_offset(offset)
    items = sqlite.get_signals(limit=lim, offset=off)
    total = sqlite.count_signals()
    return PaginatedResponse(items=items, total=total, limit=lim, offset=off)


@router.get("/trades/executions", response_model=PaginatedResponse)
def get_executions(
    sqlite: SQLiteStore = Depends(get_sqlite_store),
    limit: int | None = Query(default=100, ge=1, le=10_000),
    offset: int = Query(default=0, ge=0),
) -> PaginatedResponse:
    """Recent order execution outcomes (newest first)."""
    lim = _clamp_limit(limit, default=100)
    off = _clamp_offset(offset)
    items = sqlite.get_executions(limit=lim, offset=off)
    total = sqlite.count_executions()
    return PaginatedResponse(items=items, total=total, limit=lim, offset=off)
