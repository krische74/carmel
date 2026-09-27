"""Paginated list envelope for read APIs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PaginatedResponse(BaseModel):
    """Offset-based page of rows plus total count."""

    items: list[dict[str, Any]]
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
