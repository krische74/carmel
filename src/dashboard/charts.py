"""Shared Plotly layout and color constants for all dashboard charts."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

PLOTLY_LAYOUT: dict[str, Any] = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E0E0E0"),
    margin=dict(l=40, r=20, t=40, b=40),
    height=400,
)

COLOR_POSITIVE = "#4CAF50"
COLOR_NEGATIVE = "#EF5350"
COLOR_NEUTRAL = "#5B8DEF"
COLOR_SECONDARY = "#78909C"


def apply_hub_layout(fig: go.Figure, *, title: str = "", height: int = 400) -> go.Figure:
    """Apply standard Carmel layout to a Plotly figure."""
    layout_updates: dict[str, Any] = {**PLOTLY_LAYOUT, "height": height}
    if title:
        layout_updates["title"] = dict(text=title, x=0.02, font=dict(color="#E0E0E0"))
    fig.update_layout(**layout_updates)
    return fig
