"""Reusable plotly chart builders shared by the Streamlit app and notebooks.

All helpers return :class:`plotly.graph_objects.Figure` instances so callers
can further customise or render them in any context.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

# Consistent palette used across the dashboard.
GREEK_COLORS = {
    "delta": "#2563eb",  # blue
    "gamma": "#16a34a",  # green
    "vega": "#ea580c",  # orange
    "theta": "#dc2626",  # red
    "rho": "#9333ea",  # purple
}


def surface_plot(
    x: Sequence[float],
    y: Sequence[float],
    z: np.ndarray,
    title: str,
    x_title: str = "Volatility",
    y_title: str = "Spot",
    z_title: str = "Value",
) -> go.Figure:
    """Build a 3-D surface figure.

    Parameters
    ----------
    x, y : sequence of float
        Axis grids (lengths match the columns/rows of ``z``).
    z : numpy.ndarray
        2-D values of shape ``(len(y), len(x))``.
    title : str
        Figure title.
    x_title, y_title, z_title : str, optional
        Axis labels.

    Returns
    -------
    plotly.graph_objects.Figure
        The surface figure.

    Examples
    --------
    >>> import numpy as np
    >>> fig = surface_plot([0.1, 0.2], [90, 100], np.zeros((2, 2)), "demo")
    >>> isinstance(fig, go.Figure)
    True
    """
    fig = go.Figure(data=[go.Surface(x=list(x), y=list(y), z=z, colorscale="Viridis")])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=x_title, yaxis_title=y_title, zaxis_title=z_title),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark",
    )
    return fig


def heatmap_plot(
    x: Sequence[float],
    y: Sequence[float],
    z: np.ndarray,
    title: str,
    x_title: str = "Δ Vol",
    y_title: str = "Δ Spot",
    colorscale: str = "RdYlGn",
) -> go.Figure:
    """Build a heatmap figure (e.g. for a P&L stress grid).

    Parameters
    ----------
    x, y : sequence of float
        Axis grids.
    z : numpy.ndarray
        2-D values of shape ``(len(y), len(x))``.
    title : str
        Figure title.
    x_title, y_title : str, optional
        Axis labels.
    colorscale : str, optional
        Plotly colorscale name. Default ``"RdYlGn"``.

    Returns
    -------
    plotly.graph_objects.Figure
        The heatmap figure.
    """
    fig = go.Figure(
        data=go.Heatmap(
            x=[f"{v:+.0%}" for v in x],
            y=[f"{v:+.0%}" for v in y],
            z=z,
            colorscale=colorscale,
            colorbar=dict(title="P&L"),
        )
    )
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title, template="plotly_dark"
    )
    return fig


def line_plot(
    x: Sequence[float],
    series: dict,
    title: str,
    x_title: str,
    y_title: str,
) -> go.Figure:
    """Build a multi-series line chart.

    Parameters
    ----------
    x : sequence of float
        Shared x-axis values.
    series : dict
        Mapping of ``label -> y-values``.
    title, x_title, y_title : str
        Figure title and axis labels.

    Returns
    -------
    plotly.graph_objects.Figure
        The line chart.
    """
    fig = go.Figure()
    for label, y in series.items():
        fig.add_trace(
            go.Scatter(x=list(x), y=list(y), mode="lines+markers", name=label)
        )
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title, template="plotly_dark"
    )
    return fig


def waterfall_plot(
    components: Sequence[str],
    values: Sequence[float],
    measures: Sequence[str],
    title: str,
) -> go.Figure:
    """Build a P&L waterfall chart.

    Parameters
    ----------
    components : sequence of str
        Bar labels (e.g. ``["Delta", "Gamma", ..., "Total"]``).
    values : sequence of float
        P&L value of each bar.
    measures : sequence of str
        Plotly waterfall measures (``"relative"`` or ``"total"``).
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        The waterfall chart.
    """
    fig = go.Figure(
        go.Waterfall(
            x=list(components),
            y=list(values),
            measure=list(measures),
            connector={"line": {"color": "rgb(120,120,120)"}},
        )
    )
    fig.update_layout(title=title, template="plotly_dark", showlegend=False)
    return fig
