"""Execution quality (slippage vs daily close reference)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from src.data.storage.sqlite_store import SQLiteStore

router = APIRouter(tags=["trades"])


def _get_sqlite(request: Request) -> SQLiteStore:
    return request.app.state.sqlite_store


@router.get(
    "/execution-quality",
    summary="Recent execution quality",
    description="Fills vs prior/same-day daily close from Parquet (signed bps; positive = adverse).",
)
def list_execution_quality(
    sqlite: SQLiteStore = Depends(_get_sqlite),
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent slippage rows."""
    lim = max(1, min(int(limit), 500))
    return sqlite.get_execution_quality_rows(limit=lim)
