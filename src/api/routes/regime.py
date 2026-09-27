"""Market regime snapshot API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_sqlite_store
from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["market"])


@router.get(
    "/market/regime",
    summary="Latest market regime",
    description="Return the latest stored market regime snapshot (VIX, yield curve, overall classification).",
)
def get_regime(sqlite: SQLiteStore = Depends(get_sqlite_store)) -> dict[str, Any]:
    """Return the latest stored market regime snapshot."""
    row = sqlite.get_latest_regime()
    if row is None:
        raise HTTPException(status_code=404, detail="No regime data available")
    logger.debug("Returning regime snapshot: %s", row.get("overall"))
    return row
