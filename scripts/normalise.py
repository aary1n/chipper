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
def main(
    symbol: str | None,
    raw_dir: str | None,
    processed_dir: str | None,
    depth: int | None,
) -> None:
    """Normalise raw NDJSON to canonical hive-partitioned Parquet."""
    settings.configure_logging()
    log = logging.getLogger("normalise")

    sym = (symbol or settings.symbol).upper()
    src = Path(raw_dir) if raw_dir else settings.raw_dir
    dst = Path(processed_dir) if processed_dir else settings.processed_dir
    k = depth or settings.depth_levels

    log.info("Normalising %s: %s → %s (depth=%d)", sym, src, dst, k)

    n_snaps, n_trades = normalise_directory(src, dst, symbol=sym, depth=k)
    log.info("Done: %d snapshot rows, %d trade rows.", n_snaps, n_trades)


if __name__ == "__main__":
    main()
