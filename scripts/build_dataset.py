"""
scripts/build_dataset.py — Build feature matrix + labels from Parquet.

Usage:
    python scripts/build_dataset.py --symbol BTCUSDT
    make build SYMBOL=BTCUSDT

Reads: data/processed/lob/snapshots/
Writes: data/processed/features/

Phase 3 TODO: add trade flow features (signed volume) from trades table.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
import polars as pl

from config import settings
from features.book_features import build_feature_exprs


@click.command()
@click.option("--symbol", default=None, help="Trading symbol")
@click.option("--processed-dir", default=None, type=click.Path())
@click.option("--features-dir", default=None, type=click.Path())
@click.option("--ofi-window", default=100, type=int, help="OFI z-score rolling window (rows)")
@click.option(
    "--session-gap-s", default=5.0, type=float,
    help="Start a new session where snapshot inter-arrival exceeds this many "
         "seconds. No feature window or label ever crosses a session boundary.",
)
def main(
    symbol: str | None,
    processed_dir: str | None,
    features_dir: str | None,
    ofi_window: int,
    session_gap_s: float,
) -> None:
    """Build feature matrix + labels from normalised Parquet snapshots."""
    settings.configure_logging()
    log = logging.getLogger("build_dataset")

    sym = (symbol or settings.symbol).upper()
    src = Path(processed_dir) if processed_dir else settings.processed_dir
    dst = Path(features_dir) if features_dir else settings.features_dir

    snap_pattern = str(src / "snapshots" / f"symbol={sym}" / "date=*" / "*.parquet")
    log.info("Building features for %s from %s", sym, snap_pattern)

    df = pl.scan_parquet(snap_pattern, hive_partitioning=True).collect()
    if df.is_empty():
        log.warning("No data found — nothing to write.")
        return

    # ── Session segmentation ───────────────────────────────────────────────────
    # Rolling windows, diffs, and forward-shift labels are row-positional, so
    # they must never span a capture outage or resync hole. Sort by exchange
    # time and split into sessions wherever inter-arrival exceeds the gap
    # threshold; all features/labels are computed strictly within a session.
    df = df.sort("timestamp_exchange_us")
    gap_us = int(session_gap_s * 1_000_000)
    df = df.with_columns(
        (pl.col("timestamp_exchange_us").diff().fill_null(0) > gap_us)
        .cum_sum()
        .alias("session_id")
    )

    sessions = (
        df.group_by("session_id")
        .agg(
            pl.col("timestamp_exchange_us").min().alias("start_us"),
            pl.col("timestamp_exchange_us").max().alias("end_us"),
            pl.len().alias("rows"),
        )
        .sort("session_id")
    )
    log.info("Found %d session(s) at gap threshold %.1fs:", len(sessions), session_gap_s)
    for s in sessions.iter_rows(named=True):
        log.info(
            "  session %d: %s → %s (%d rows)",
            s["session_id"],
            pl.from_epoch(pl.Series([s["start_us"]]), time_unit="us")[0],
            pl.from_epoch(pl.Series([s["end_us"]]), time_unit="us")[0],
            s["rows"],
        )

    feature_exprs = build_feature_exprs(
        imbalance_depths=settings.imbalance_depths,
        ofi_window=ofi_window,
        label_horizons=[
            int(h * 10) for h in settings.label_horizons_s  # convert seconds → rows at 10 Hz
        ],
    )

    parts = [
        group.with_columns(feature_exprs)
        for _, group in df.group_by("session_id", maintain_order=True)
    ]
    df = pl.concat(parts)

    # Book arrays are not needed downstream of feature computation; dropping
    # them keeps the feature matrix ~10x smaller than the snapshots table.
    # Best bid/ask survive as scalars — replay's taker fills need them.
    df = df.with_columns(
        pl.col("bid_prices").list.get(0).alias("best_bid"),
        pl.col("ask_prices").list.get(0).alias("best_ask"),
    ).drop(["bid_prices", "bid_quantities", "ask_prices", "ask_quantities"])

    # Null-label accounting (session tails + ties) — reported, never filled.
    label_cols = [c for c in df.columns if c.startswith("label_")]
    for c in label_cols:
        n_null = df[c].null_count()
        log.info("  %s: %d/%d rows null (session-tail horizon + ties)", c, n_null, len(df))

    log.info("Computed %d rows. Writing feature partitions…", len(df))

    # Group by date and write each partition
    if "date" in df.columns:
        date_col = "date"
    else:
        # Derive date from timestamp
        df = df.with_columns(
            (pl.col("timestamp_exchange_us") // 1_000_000_000).cast(pl.Datetime("us"))
            .dt.strftime("%Y-%m-%d").alias("date")
        )
        date_col = "date"

    for date_val, group in df.group_by(date_col):
        date_str = date_val[0] if isinstance(date_val, (list, tuple)) else str(date_val)
        out_dir = dst / f"symbol={sym}" / f"date={date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "part-0.parquet"
        group.write_parquet(out_path, compression="snappy")
        log.info("Wrote %d rows → %s", len(group), out_path)

    log.info("Feature build complete.")


if __name__ == "__main__":
    main()
