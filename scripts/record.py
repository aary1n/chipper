"""
scripts/record.py — Stream Binance LOB to NDJSON.

Usage:
    python scripts/record.py --symbol BTCUSDT
    make record SYMBOL=ETHUSDT

Ctrl+C to stop. Reconnects automatically on any error.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path when running as script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click

from config import settings
from ingest.lob_recorder import LOBRecorder


@click.command()
@click.option("--symbol", default=None, help="Trading symbol (default: from config)")
@click.option("--output-dir", default=None, type=click.Path(), help="Override raw output dir")
def main(symbol: str | None, output_dir: str | None) -> None:
    """Stream Binance LOB depth + trades to hourly NDJSON files."""
    settings.configure_logging()
    log = logging.getLogger("record")

    sym = (symbol or settings.symbol).upper()
    out = Path(output_dir) if output_dir else settings.raw_dir
    out.mkdir(parents=True, exist_ok=True)

    log.info("Starting LOB recorder: symbol=%s output=%s", sym, out)
    log.info("Ctrl+C to stop. Reconnects automatically on error.")

    recorder = LOBRecorder(
        symbol=sym,
        output_dir=out,
        reconnect_base=settings.reconnect_base_s,
        reconnect_max=settings.reconnect_max_s,
    )

    try:
        asyncio.run(recorder.run())
    except KeyboardInterrupt:
        log.info("Recorder stopped by user.")


if __name__ == "__main__":
    main()
