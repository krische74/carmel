"""Plotly layout helpers."""

from __future__ import annotations

import plotly.graph_objects as go

from src.dashboard.charts import COLOR_NEUTRAL, PLOTLY_LAYOUT, apply_hub_layout


def test_apply_hub_layout_sets_template_font_height_title() -> None:
    fig = go.Figure()
    apply_hub_layout(fig, title="Equity", height=333)
    assert fig.layout.height == 333
    assert fig.layout.template is not None
    assert fig.layout.font is not None
    assert fig.layout.title is not None
    assert PLOTLY_LAYOUT["template"] == "plotly_dark"


def test_plotly_color_constants_are_hex_strings() -> None:
    assert COLOR_NEUTRAL.startswith("#")
