"""Health and readiness endpoint."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_settings
from src.automation.health import read_heartbeat
from src.config import PROJECT_ROOT, Settings

router = APIRouter(tags=["health"])


@router.get("/health")
def api_health(request: Request, settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Liveness: heartbeat age, environment, version, uptime."""
    hb_path = PROJECT_ROOT / "logs" / "hub.heartbeat"
    hb = read_heartbeat(hb_path)
    started = float(getattr(request.app.state, "started_perf", time.perf_counter()))
    uptime = time.perf_counter() - started
    return {
        "status": "ok",
        "heartbeat_utc": hb.isoformat() if hb else None,
        "environment": settings.app.environment,
        "version": settings.app.version,
        "uptime_seconds": round(uptime, 3),
    }
