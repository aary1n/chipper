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
def main(
    symbol: str | None,
    processed_dir: str | None,
    features_dir: str | None,
    ofi_window: int,
) -> None:
    """Build feature matrix + labels from normalised Parquet snapshots."""
    settings.configure_logging()
    log = logging.getLogger("build_dataset")

    sym = (symbol or settings.symbol).upper()
    src = Path(processed_dir) if processed_dir else settings.processed_dir
    dst = Path(features_dir) if features_dir else settings.features_dir

    snap_pattern = str(src / "snapshots" / f"symbol={sym}" / "date=*" / "*.parquet")
    log.info("Building features for %s from %s", sym, snap_pattern)

    lf = pl.scan_parquet(snap_pattern, hive_partitioning=True)

    feature_exprs = build_feature_exprs(
        imbalance_depths=settings.imbalance_depths,
        ofi_window=ofi_window,
        label_horizons=[
            int(h * 10) for h in settings.label_horizons_s  # convert seconds → rows at 10 Hz
        ],
    )

    out_lf = lf.with_columns(feature_exprs)

    # Write per date partition
    df = out_lf.collect()
    log.info("Collected %d rows. Writing feature partitions…", len(df))

    if df.is_empty():
        log.warning("No data found — nothing to write.")
        return

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
