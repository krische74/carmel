"""CLI-equivalent actions (ingest, cycle, reconcile)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_settings as get_settings_dep
from src.api.dependencies import get_sqlite_store
from src.automation.runner import broker_from_settings, create_trading_workflow
from src.automation.workflows import TradingWorkflow
from src.config import Settings, parse_as_of_iso
from src.data.storage.sqlite_store import SQLiteStore
from src.execution.errors import ConfigurationError
from src.execution.reconciliation import ReconciliationResult, reconcile_executions

logger = logging.getLogger(__name__)

router = APIRouter(tags=["actions"])

_ACTION_FAILURES = (ValueError, ConnectionError, OSError, RuntimeError)


def get_trading_workflow(request: Request) -> TradingWorkflow:
    """Lazily construct a wired workflow; requires Alpaca credentials."""
    settings = request.app.state.settings
    key = (settings.alpaca_api_key or "").strip()
    sec = (settings.alpaca_secret_key or "").strip()
    if not key or not sec:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Alpaca API credentials are not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.",
        )
    wf = getattr(request.app.state, "trading_workflow", None)
    if wf is None:
        try:
            wf = create_trading_workflow(settings)
        except ConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Alpaca broker configuration or connectivity failed: {exc}",
            ) from exc
        request.app.state.trading_workflow = wf
    return wf


class CycleBody(BaseModel):
    """Optional simulation timestamp for ``run_cycle``."""

    as_of: str | None = Field(default=None, description="ISO-8601 UTC, optional")


@router.post("/actions/ingest")
def post_ingest(workflow: TradingWorkflow = Depends(get_trading_workflow)) -> dict[str, Any]:
    """Run data ingest for all strategy universe symbols (no trading)."""
    try:
        results = workflow.run_ingest_only()
    except _ACTION_FAILURES as exc:
        logger.exception("API ingest failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ingest failed",
        ) from exc
    payload = [
        {"symbol": sym, "success": bool(res.success), "error": res.error} for sym, res in results
    ]
    return {"success": all(r["success"] for r in payload), "results": payload}


@router.post("/actions/cycle")
def post_cycle(
    body: CycleBody | None = None,
    workflow: TradingWorkflow = Depends(get_trading_workflow),
) -> dict[str, Any]:
    """Run one full trading cycle (signals + execution)."""
    as_of = parse_as_of_iso(body.as_of if body else None)
    try:
        result = workflow.run_cycle(as_of=as_of)
    except _ACTION_FAILURES as exc:
        logger.exception("API cycle failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trading cycle failed",
        ) from exc
    return {
        "cycle_id": result.cycle_id,
        "signals_count": len(result.signals),
        "executions_count": len(result.execution_results),
    }


@router.post("/actions/reconcile", response_model=ReconciliationResult)
def post_reconcile(
    settings: Settings = Depends(get_settings_dep),
    sqlite: SQLiteStore = Depends(get_sqlite_store),
) -> ReconciliationResult:
    """Compare SQLite execution log to broker orders (same as ``carmel reconcile``)."""
    key = (settings.alpaca_api_key or "").strip()
    sec = (settings.alpaca_secret_key or "").strip()
    if not key or not sec:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Alpaca API credentials are not configured.",
        )
    try:
        broker = broker_from_settings(settings)
        rows = sqlite.get_executions(limit=500)
        orders = broker.list_recent_orders(limit=500)
        return reconcile_executions(rows, orders)
    except HTTPException:
        raise
    except ConfigurationError as exc:
        logger.exception("API reconcile failed: broker configuration")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Alpaca broker configuration or connectivity failed: {exc}",
        ) from exc
    except _ACTION_FAILURES as exc:
        logger.exception("API reconcile failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reconciliation failed",
        ) from exc
