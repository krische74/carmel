"""Experimental ML scores (read-only, diagnostic)."""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.api.dependencies import get_sqlite_store
from src.data.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ml"])


class MlScoreRow(BaseModel):
    """One symbol's score from the latest recorded cycle."""

    id: int
    cycle_id: str
    symbol: str
    score: float = Field(description="Estimated P(positive forward return) from classifier.")
    model_version: str
    computed_at: str
    top_features: str | None = Field(
        default=None,
        description="Optional SHAP-style summary text when shap is installed.",
    )


@router.get("/ml/latest", response_model=list[MlScoreRow])
def get_ml_latest(sqlite: SQLiteStore = Depends(get_sqlite_store)) -> list[MlScoreRow]:
    """Latest batch of ML scores (same cycle_id as the newest row)."""
    rows = sqlite.get_latest_ml_scores_batch()
    out: list[MlScoreRow] = []
    for r in rows:
        try:
            raw = float(r["score"])
        except (TypeError, ValueError):
            logger.warning("ML API: skipping row id=%s with non-numeric score", r.get("id"))
            continue
        if math.isnan(raw) or math.isinf(raw):
            logger.warning("ML API: skipping row id=%s with non-finite score", r.get("id"))
            continue
        out.append(
            MlScoreRow(
                id=int(r["id"]),
                cycle_id=str(r["cycle_id"]),
                symbol=str(r["symbol"]),
                score=raw,
                model_version=str(r["model_version"]),
                computed_at=str(r["computed_at"]),
                top_features=r.get("top_features_json"),
            ),
        )
    return out
