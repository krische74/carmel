"""Shared domain models (Pydantic) crossing module boundaries."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field


class OrderExecutionResult(BaseModel):
    """Outcome of attempting to route a signal through risk and the broker."""

    symbol: str
    submitted: bool
    order_id: str | None = None
    reason: str | None = None
    timestamp: dt.datetime = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC),
    )
    side: Literal["buy", "sell"] = "buy"
    qty: float | None = None
    fill_price: float | None = None
    strategy_name: str | None = None
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    limit_price: float | None = None
    stop_price: float | None = None
    order_status: str = "filled"
    filled_qty: float | None = None


class Signal(BaseModel):
    """A single strategy output for execution and explanation layers."""

    symbol: str
    direction: Literal["long", "flat", "cash"]
    weight: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1)
    timestamp: dt.datetime
    strategy_name: str | None = None
