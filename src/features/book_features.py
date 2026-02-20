"""
book_features.py — LOB features as composable Polars lazy expressions.

All features are:
  - Pure functions returning pl.Expr (no side effects).
  - Normalised: ratios, z-scores, or inherently bounded.
  - Computed over the snapshot DataFrame columns (see DATA_CONTRACTS.md).

Usage:
    df = pl.scan_parquet("data/processed/lob/snapshots/**/*.parquet")
    df = df.with_columns([
        book_imbalance(k=1).alias("imbalance_1"),
        microprice().alias("microprice"),
        spread_bps().alias("spread_bps"),
    ])

Reference: Cont, Kukanov & Stoikov (2014) for OFI formulation.
"""

from __future__ import annotations

import polars as pl


# ── Level accessor helpers ─────────────────────────────────────────────────────

def _bid_price(i: int = 0) -> pl.Expr:
    return pl.col("bid_prices").list.get(i)

def _bid_qty(i: int = 0) -> pl.Expr:
    return pl.col("bid_quantities").list.get(i)

def _ask_price(i: int = 0) -> pl.Expr:
    return pl.col("ask_prices").list.get(i)

def _ask_qty(i: int = 0) -> pl.Expr:
    return pl.col("ask_quantities").list.get(i)


# ── Core features ──────────────────────────────────────────────────────────────

def mid_price() -> pl.Expr:
    """Simple arithmetic mid. (best_ask + best_bid) / 2."""
    return (_ask_price(0) + _bid_price(0)) / 2.0


def microprice() -> pl.Expr:
    """
    Weighted mid using best-level quantities.
    microprice = (ask_p0 × bid_q0 + bid_p0 × ask_q0) / (bid_q0 + ask_q0)

    Better fair-value estimate than simple mid when book is asymmetric.
    Also used as label basis (sign of microprice change).
    """
    bp0 = _bid_price(0)
    bq0 = _bid_qty(0)
    ap0 = _ask_price(0)
    aq0 = _ask_qty(0)
    return (ap0 * bq0 + bp0 * aq0) / (bq0 + aq0)


def spread_bps() -> pl.Expr:
    """
    Bid-ask spread in basis points relative to mid.
    spread_bps = (ask_p0 - bid_p0) / mid × 10000
    """
    mid = mid_price()
    spread = _ask_price(0) - _bid_price(0)
    return (spread / mid) * 10_000.0


def book_imbalance(k: int = 1) -> pl.Expr:
    """
    Order book imbalance at top-k levels.
    imbalance_k = (bid_qty_top_k - ask_qty_top_k) / (bid_qty_top_k + ask_qty_top_k)

    Inherently normalised to [-1, +1].
    k = 1, 3, 5, 10 recommended (Settings.imbalance_depths).
    """
    bid_qty = pl.col("bid_quantities").list.head(k).list.sum()
    ask_qty = pl.col("ask_quantities").list.head(k).list.sum()
    total = bid_qty + ask_qty
    return (bid_qty - ask_qty) / total


def ofi(window_size: int = 10) -> pl.Expr:
    """
    Order Flow Imbalance (OFI) — Cont, Kukanov & Stoikov (2014).

    OFI = Δbid_qty_best - Δask_qty_best (net change at best levels).

    For each row, computes change in best bid quantity and best ask quantity
    versus the previous row, then takes their difference.

    NOTE: This is a stub. Full OFI requires tracking signed changes
    (additions vs cancellations vs trades) which needs prev-row comparison.
    Implement as a rolling delta for now.

    Normalisation: z-score over rolling window (caller should apply .rolling_mean/.rolling_std).
    """
    # TODO (Phase 3): implement full Cont et al. OFI with signed deltas
    # Placeholder: net first-difference of best-level imbalance
    delta_bid_q = _bid_qty(0).diff(1)
    delta_ask_q = _ask_qty(0).diff(1)
    return delta_bid_q - delta_ask_q


def ofi_normalised(window_size: int = 100) -> pl.Expr:
    """
    OFI z-scored over a rolling window.
    Use window_size ≈ number of rows in N seconds at 10 Hz = N × 10.
    """
    raw = ofi()
    mean = raw.rolling_mean(window_size=window_size, min_periods=2)
    std = raw.rolling_std(window_size=window_size, min_periods=2)
    return (raw - mean) / (std + 1e-9)


def spread_change() -> pl.Expr:
    """Rate of change of spread (widening = uncertainty signal)."""
    return spread_bps().diff(1)


def depth_drop(k: int = 5) -> pl.Expr:
    """
    Sudden depth drop at top-k levels (liquidity vacuum detection).
    Returns ratio: current_depth / rolling_max_depth over last 100 rows.
    Values near 0 signal liquidity withdrawal.
    """
    total_depth = (
        pl.col("bid_quantities").list.head(k).list.sum()
        + pl.col("ask_quantities").list.head(k).list.sum()
    )
    rolling_max = total_depth.rolling_max(window_size=100, min_periods=2)
    return total_depth / (rolling_max + 1e-9)


def realised_vol(window_rows: int = 10) -> pl.Expr:
    """
    Realised volatility of microprice returns over a rolling window.
    window_rows ≈ 10 → 1s at 10 Hz; 50 → 5s.
    """
    returns = microprice().pct_change()
    return returns.rolling_std(window_size=window_rows, min_periods=2)


# ── Label construction ─────────────────────────────────────────────────────────

def future_microprice_return(horizon_rows: int) -> pl.Expr:
    """
    Forward microprice return over horizon_rows time steps.
    Positive → microprice went up. Used to build binary classification labels.
    horizon_rows ≈ horizon_seconds × 10 (at 10 Hz).
    """
    mp = microprice()
    future_mp = mp.shift(-horizon_rows)
    return (future_mp - mp) / mp


def label_direction(horizon_rows: int) -> pl.Expr:
    """
    Binary label: 1 if microprice goes up, 0 if goes down, null on tie.
    Excludes flat outcomes to reduce label noise.
    """
    ret = future_microprice_return(horizon_rows)
    return (
        pl.when(ret > 0).then(1)
        .when(ret < 0).then(0)
        .otherwise(None)
        .cast(pl.Int8)
    )


# ── Feature matrix builder ─────────────────────────────────────────────────────

def build_feature_exprs(
    imbalance_depths: list[int] | None = None,
    ofi_window: int = 100,
    label_horizons: list[int] | None = None,
) -> list[pl.Expr]:
    """
    Return a list of named Polars expressions for the full feature matrix.
    Apply with df.with_columns(build_feature_exprs()).
    """
    if imbalance_depths is None:
        imbalance_depths = [1, 3, 5, 10]
    if label_horizons is None:
        label_horizons = [5, 10, 50]  # rows; divide by 10 to get seconds at 10 Hz

    exprs: list[pl.Expr] = [
        mid_price().alias("mid_price"),
        microprice().alias("microprice"),
        spread_bps().alias("spread_bps"),
        spread_change().alias("spread_change"),
        ofi().alias("ofi_raw"),
        ofi_normalised(ofi_window).alias("ofi_z"),
        realised_vol(10).alias("realised_vol_1s"),
        realised_vol(50).alias("realised_vol_5s"),
        depth_drop(5).alias("depth_drop_5"),
    ]

    for k in imbalance_depths:
        exprs.append(book_imbalance(k).alias(f"imbalance_{k}"))

    for h in label_horizons:
        exprs.append(label_direction(h).alias(f"label_{h}rows"))

    return exprs
