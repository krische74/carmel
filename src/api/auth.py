"""Bearer token verification for the REST API."""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


def verify_api_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Require ``Authorization: Bearer`` when ``Settings.api_token`` is set; otherwise open.

    When ``api_token`` is empty, all requests are allowed (local personal use).
    When set, missing or mismatched tokens yield ``401 Unauthorized``.
    """
    settings = request.app.state.settings
    expected = (settings.api_token or "").strip()
    if not expected:
        return
    if credentials is None:
        logger.warning("API request rejected: missing bearer token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    token = (credentials.credentials or "").strip()
    if token != expected:
        logger.warning("API request rejected: invalid bearer token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )
