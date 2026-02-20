"""
visualisation.py — LOB heatmap and book depth rendering (Phase 2).

Requires: pip install -e ".[viz]"

Expected plots:
  1. LOB Heatmap: price × time, colour = resting quantity (plotly)
  2. Book Depth Chart: cumulative depth on bid/ask sides at a point in time
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def lob_heatmap(
    df: pl.DataFrame,
    max_levels: int = 10,
    title: str = "LOB Heatmap",
):
    """
    Render a LOB heatmap: price level on Y, time on X, colour = quantity.

    Args:
        df: snapshot DataFrame with bid_prices, bid_quantities, ask_prices, ask_quantities
            and timestamp_exchange_us columns.
        max_levels: number of price levels to show per side.
        title: chart title.

    Returns: plotly Figure object.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Install viz extras: pip install -e '.[viz]'")

    # TODO (Phase 2): implement full heatmap with exploded bid/ask levels
    # Sketch: for each row, enumerate top-max_levels bid/ask levels,
    # build a matrix of (timestamp, price, quantity) for imshow.
    logger.warning("lob_heatmap: stub implementation — Phase 2 item.")

    fig = go.Figure()
    fig.update_layout(title=title)
    return fig


def book_depth_chart(
    snapshot_row: dict,
    title: str = "Order Book Depth",
):
    """
    Render cumulative depth at a single point in time.
    X = cumulative quantity, Y = price level.
    Bid (green) and ask (red) as mirrored bar chart.

    Args:
        snapshot_row: one row from the snapshot DataFrame as a dict.

    Returns: plotly Figure object.
    """
    try:
        import plotly.graph_objects as go
        import numpy as np
    except ImportError:
        raise ImportError("Install viz extras: pip install -e '.[viz]'")

    bid_prices = snapshot_row["bid_prices"]
    bid_qtys = snapshot_row["bid_quantities"]
    ask_prices = snapshot_row["ask_prices"]
    ask_qtys = snapshot_row["ask_quantities"]

    import numpy as np

    bid_cum = np.nancumsum(bid_qtys)
    ask_cum = np.nancumsum(ask_qtys)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=-bid_cum, y=bid_prices,
        orientation="h", name="Bids",
        marker_color="rgba(0,200,100,0.7)",
    ))
    fig.add_trace(go.Bar(
        x=ask_cum, y=ask_prices,
        orientation="h", name="Asks",
        marker_color="rgba(220,50,50,0.7)",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Cumulative Quantity",
        yaxis_title="Price",
        barmode="overlay",
    )
    return fig
