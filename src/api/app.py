"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.auth import verify_api_token
from src.api.errors import APIError
from src.api.routes import (
    actions,
    ai,
    alerts_route,
    backtests,
    config_route,
    execution_quality_route,
    health,
    ml,
    performance,
    portfolio,
    regime,
    tax,
    trades,
)
from src.config import Settings, get_settings, hub_sqlite_path, parquet_dir
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.portfolio.tax_lots import LotLedger


def create_app(
    *,
    settings: Settings | None = None,
    sqlite_store: SQLiteStore | None = None,
    parquet_store: ParquetStore | None = None,
    lot_ledger: LotLedger | None = None,
) -> FastAPI:
    """Build the Carmel REST API (read endpoints + authenticated actions).

    For tests, pass explicit ``settings`` and store instances; production uses lifespan defaults.
    """
    s0 = settings if settings is not None else get_settings()
    version = s0.app.version

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings if settings is not None else get_settings()
        app.state.settings = s
        app.state.sqlite_store = (
            sqlite_store if sqlite_store is not None else SQLiteStore(hub_sqlite_path(s))
        )
        app.state.parquet_store = (
            parquet_store if parquet_store is not None else ParquetStore(parquet_dir(s))
        )
        app.state.lot_ledger = (
            lot_ledger if lot_ledger is not None else LotLedger(hub_sqlite_path(s))
        )
        app.state.started_perf = time.perf_counter()
        yield

    app = FastAPI(
        title="Carmel API",
        version=version,
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=APIError(error="Bad request", detail=exc.errors()).model_dump(
                exclude_none=True,
            ),
        )

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=APIError(error="Bad request", detail=str(exc)).model_dump(
                exclude_none=True,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        _request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        if exc.status_code == 404:
            detail = (
                str(exc.detail)
                if exc.detail is not None and str(exc.detail) != "Not Found"
                else None
            )
            return JSONResponse(
                status_code=404,
                content=APIError(error="Not found", detail=detail).model_dump(
                    exclude_none=True,
                ),
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=APIError(
                error="HTTP error",
                detail=str(exc.detail) if exc.detail is not None else None,
                code=str(exc.status_code),
            ).model_dump(exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).exception("Unhandled API error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=APIError(error="Internal server error").model_dump(exclude_none=True),
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_router = APIRouter(prefix="/api", dependencies=[Depends(verify_api_token)])
    api_router.include_router(health.router)
    api_router.include_router(config_route.router)
    api_router.include_router(portfolio.router)
    api_router.include_router(performance.router)
    api_router.include_router(trades.router)
    api_router.include_router(execution_quality_route.router)
    api_router.include_router(alerts_route.router)
    api_router.include_router(backtests.router)
    api_router.include_router(tax.router)
    api_router.include_router(regime.router)
    api_router.include_router(actions.router)
    api_router.include_router(ai.router)
    api_router.include_router(ml.router)
    app.include_router(api_router)

    return app
