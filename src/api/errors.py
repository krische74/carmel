"""Consistent JSON error bodies for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class APIError(BaseModel):
    """Standard error envelope returned by exception handlers."""

    error: str
    detail: str | list[dict[str, object]] | None = Field(default=None)
    code: str | None = Field(default=None)
