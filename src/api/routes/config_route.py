"""Public (non-secret) configuration JSON."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from src.api.dependencies import get_settings
from src.config import Settings, public_settings_dict

router = APIRouter(tags=["config"])


@router.get("/config")
def api_config(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Non-secret configuration (same shape as ``carmel config``)."""
    return public_settings_dict(settings)
