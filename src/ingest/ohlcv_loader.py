"""
ohlcv_loader.py — stub for loading supplementary OHLCV data.

Phase 1: not needed. Will be used in Phase 3/4 to load
Binance kline data or LOBSTER OHLCV for regime context.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def load_binance_klines(
    symbol: str,
    interval: str,
    external_dir: Path,
) -> pl.LazyFrame:
    """
    Load Binance kline (OHLCV) CSV data from external_dir.

    Expected file path: {external_dir}/klines/{symbol}_{interval}.csv
    Expected columns: open_time, open, high, low, close, volume, close_time, ...

    Returns a LazyFrame with standardised column names.
    """
    # TODO (Phase 3): implement actual kline loader
    logger.warning("ohlcv_loader.load_binance_klines is a stub — not implemented.")

    # Return empty schema placeholder so downstream can type-check
    schema = {
        "timestamp_us": pl.Int64,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "symbol": pl.String,
        "interval": pl.String,
    }
    return pl.LazyFrame(schema=schema)
