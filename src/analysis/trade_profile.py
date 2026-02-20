"""
trade_profile.py — DuckDB queries for LOB book statistics.

Phase 2 deliverable. Provides pre-built analytic queries:
  - Spread distribution (mean, median, p95, p99)
  - Book depth at top 1/5/10/20 levels
  - Trade arrival rate by hour (UTC)
  - Trade size distribution
  - Aggregate OFI rollups
  - Spread behaviour during high/low volume periods
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


def spread_distribution(
    conn: duckdb.DuckDBPyConnection,
    symbol: str = "BTCUSDT",
) -> pl.DataFrame:
    """
    Bid-ask spread distribution across all snapshots.
    Returns mean, median, p95, p99 in bps.
    """
    result = conn.execute(f"""
        SELECT
            AVG(ask_prices[1] - bid_prices[1]) / AVG((ask_prices[1] + bid_prices[1]) / 2) * 10000
                AS mean_spread_bps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ask_prices[1] - bid_prices[1])
                / AVG((ask_prices[1] + bid_prices[1]) / 2) * 10000
                AS median_spread_bps,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ask_prices[1] - bid_prices[1])
                / AVG((ask_prices[1] + bid_prices[1]) / 2) * 10000
                AS p95_spread_bps,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ask_prices[1] - bid_prices[1])
                / AVG((ask_prices[1] + bid_prices[1]) / 2) * 10000
                AS p99_spread_bps
        FROM lob_snapshots
        WHERE symbol = '{symbol}' AND is_clean = true
    """).pl()
    return result


def book_depth_summary(
    conn: duckdb.DuckDBPyConnection,
    symbol: str = "BTCUSDT",
) -> pl.DataFrame:
    """
    Average total quantity (bid + ask) at top 1, 5, 10, 20 levels.
    """
    # TODO (Phase 2): implement DuckDB list aggregation over nested arrays
    # Requires DuckDB list_aggregate or unnest — placeholder query
    logger.warning("book_depth_summary: full implementation pending Phase 2.")
    result = conn.execute(f"""
        SELECT
            date,
            COUNT(*) AS snapshot_count,
            AVG(bid_quantities[1]) AS avg_best_bid_qty,
            AVG(ask_quantities[1]) AS avg_best_ask_qty
        FROM lob_snapshots
        WHERE symbol = '{symbol}' AND is_clean = true
        GROUP BY date
        ORDER BY date
    """).pl()
    return result


def trade_arrival_rate(
    conn: duckdb.DuckDBPyConnection,
    symbol: str = "BTCUSDT",
) -> pl.DataFrame:
    """Trade count per UTC hour of day (averaged across all dates)."""
    result = conn.execute(f"""
        SELECT
            HOUR(epoch_us(timestamp_exchange_us * 1000)) AS hour_utc,
            COUNT(*) AS trade_count,
            SUM(quantity) AS total_volume
        FROM trades
        WHERE symbol = '{symbol}'
        GROUP BY hour_utc
        ORDER BY hour_utc
    """).pl()
    return result


def trade_size_distribution(
    conn: duckdb.DuckDBPyConnection,
    symbol: str = "BTCUSDT",
) -> pl.DataFrame:
    """Quantiles of trade quantity (log-scale — fat-tailed)."""
    result = conn.execute(f"""
        SELECT
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY quantity) AS p25,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY quantity) AS p50,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY quantity) AS p75,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY quantity) AS p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY quantity) AS p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY quantity) AS p99,
            MAX(quantity) AS max_qty,
            AVG(quantity) AS mean_qty
        FROM trades
        WHERE symbol = '{symbol}'
    """).pl()
    return result


def clean_data_fraction(
    conn: duckdb.DuckDBPyConnection,
    symbol: str = "BTCUSDT",
) -> pl.DataFrame:
    """
    Fraction of is_clean=True rows per date.
    Target: >99% clean. Red flag if <99%.
    """
    result = conn.execute(f"""
        SELECT
            date,
            COUNT(*) AS total_rows,
            SUM(CASE WHEN is_clean THEN 1 ELSE 0 END) AS clean_rows,
            SUM(CASE WHEN is_clean THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_clean
        FROM lob_snapshots
        WHERE symbol = '{symbol}'
        GROUP BY date
        ORDER BY date
    """).pl()
    return result
