"""FastAPI dependencies reading shared app state (lifespan singletons)."""

from __future__ import annotations

from fastapi import Request

from src.config import Settings
from src.data.storage.parquet_store import ParquetStore
from src.data.storage.sqlite_store import SQLiteStore
from src.portfolio.tax_lots import LotLedger


def get_settings(request: Request) -> Settings:
    """Settings instance attached in app lifespan."""
    return request.app.state.settings


def get_sqlite_store(request: Request) -> SQLiteStore:
    """SQLite metadata store from lifespan."""
    return request.app.state.sqlite_store


def get_parquet_store(request: Request) -> ParquetStore:
    """Parquet OHLCV store from lifespan."""
    return request.app.state.parquet_store


def get_lot_ledger(request: Request) -> LotLedger:
    """Tax lot ledger from lifespan."""
    return request.app.state.lot_ledger
