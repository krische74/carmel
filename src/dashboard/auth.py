"""Streamlit session gate for the dashboard (optional password)."""

from __future__ import annotations

import hmac
import logging
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

_SESSION_KEY = "dashboard_authenticated"


def _effective_password(settings: Settings) -> str:
    """Env ``DASHBOARD_PASSWORD`` overrides YAML ``dashboard.auth_password``."""
    env_pw = (getattr(settings, "dashboard_password", "") or "").strip()
    yaml_pw = (settings.dashboard.auth_password or "").strip()
    return env_pw or yaml_pw


def require_auth(settings: Settings) -> bool:
    """Check auth and render login form if needed. Returns True if authenticated."""
    dash = settings.dashboard
    password = _effective_password(settings)
    if not dash.auth_enabled or not password:
        return True
    if st.session_state.get(_SESSION_KEY):
        return True

    st.subheader("Dashboard login")
    entered = st.text_input("Password", type="password", key="dashboard_password_input")
    if st.button("Login"):
        if hmac.compare_digest(entered, password):
            st.session_state[_SESSION_KEY] = True
            st.rerun()
            return True
        st.error("Incorrect password")
        logger.warning("Failed dashboard login attempt")
    return False
