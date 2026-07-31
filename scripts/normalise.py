"""
scripts/normalise.py — Convert raw NDJSON → canonical Parquet.

Usage:
    python scripts/normalise.py --symbol BTCUSDT
    make normalise SYMBOL=ETHUSDT

Reads all {SYMBOL}_*.ndjson files from data/raw/lob/
Writes hive-partitioned Parquet to data/processed/lob/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click

from config import settings
from ingest.lob_normaliser import normalise_directory


@click.command()
@click.option("--symbol", default=None, help="Trading symbol")
@click.option("--raw-dir", default=None, type=click.Path(), help="Override raw NDJSON dir")
@click.option("--processed-dir", default=None, type=click.Path(), help="Override Parquet output dir")
@click.option("--depth", default=None, type=int, help="Number of price levels (default: from config)")
@click.option(
    "--resume/--fresh", "resume", default=True,
    help="Resume from the per-file checkpoint if one exists (default). "
         "--fresh ignores it and reprocesses everything.",
)
def main(
    symbol: str | None,
    raw_dir: str | None,
    processed_dir: str | None,
    depth: int | None,
    resume: bool,
) -> None:
    """Normalise raw NDJSON to canonical hive-partitioned Parquet."""
    settings.configure_logging()
    log = logging.getLogger("normalise")

    sym = (symbol or settings.symbol).upper()
    src = Path(raw_dir) if raw_dir else settings.raw_dir
    dst = Path(processed_dir) if processed_dir else settings.processed_dir
    k = depth or settings.depth_levels

    log.info("Normalising %s: %s → %s (depth=%d, resume=%s)", sym, src, dst, k, resume)

    stats = normalise_directory(src, dst, symbol=sym, depth=k, resume=resume)
    n_snaps = sum(s.snapshot_rows for s in stats)
    n_trades = sum(s.trade_rows for s in stats)
    n_skipped = sum(s.malformed_lines + s.invalid_events for s in stats)
    n_anomalies = sum(s.gaps + s.out_of_order + s.gap_markers for s in stats)
    log.info(
        "Done: %d snapshot rows, %d trade rows, %d skipped lines/events, "
        "%d sequence anomalies. Per-file detail: %s",
        n_snaps, n_trades, n_skipped, n_anomalies,
        Path(dst) / "_normalise_report.json",
    )


if __name__ == "__main__":
    main()
