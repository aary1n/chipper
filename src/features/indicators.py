"""
indicators.py — OHLCV-level technical indicators (stub).

Phase 3/4: supplement LOB features with regime context from OHLCV data.
All indicators return Polars expressions. No absolute values — always ratios or z-scores.
"""

from __future__ import annotations

import polars as pl


def ema(col: str, span: int) -> pl.Expr:
    """Exponential moving average. Returns raw EMA (not normalised)."""
    # TODO (Phase 3): implement with Polars ewm_mean
    return pl.col(col).ewm_mean(span=span)


def rsi(close_col: str = "close", period: int = 14) -> pl.Expr:
    """
    Relative Strength Index. Returns value in [0, 100].
    Stub — full implementation in Phase 3.
    """
    # TODO (Phase 3): implement RSI
    raise NotImplementedError("RSI not yet implemented — Phase 3 item")


def realised_vol_ohlcv(
    close_col: str = "close",
    window: int = 20,
) -> pl.Expr:
    """Rolling realised vol of log returns. Normalised: ratio to rolling median."""
    log_ret = pl.col(close_col).log().diff(1)
    rv = log_ret.rolling_std(window_size=window, min_periods=2)
    rv_median = rv.rolling_median(window_size=window * 5, min_periods=2)
    return rv / (rv_median + 1e-9)
