"""
scripts/replay.py — Replay Parquet through signal + taker-only simulation.

Usage:
    python scripts/replay.py --symbol BTCUSDT
    make replay SYMBOL=BTCUSDT

Loads best saved model, replays feature matrix chronologically,
simulates taker-only fills, prints P&L summary.

Phase 5 / 6: taker-only P&L after fees.
  - BUY: cross ask (pay ask_price), fee = ask_price × taker_fee_bps / 10000
  - SELL: cross bid (receive bid_price), fee = bid_price × taker_fee_bps / 10000
  - Net P&L per round-trip = exit_price - entry_price - fees
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
import numpy as np
import polars as pl

from config import settings


@click.command()
@click.option("--symbol", default=None)
@click.option("--features-dir", default=None, type=click.Path())
@click.option("--model-dir", default=None, type=click.Path(), help="Explicit model dir (else: latest)")
@click.option("--threshold", default=0.55, type=float, help="P(up) threshold to go long")
@click.option("--fee-bps", default=None, type=float, help="Taker fee in bps (default: from config)")
@click.option("--holding-rows", default=10, type=int, help="Rows to hold position before flat")
def main(
    symbol: str | None,
    features_dir: str | None,
    model_dir: str | None,
    threshold: float,
    fee_bps: float | None,
    holding_rows: int,
) -> None:
    """Replay historical feature data through trained signal + taker-only fill sim."""
    settings.configure_logging()
    log = logging.getLogger("replay")

    sym = (symbol or settings.symbol).upper()
    feat_dir = Path(features_dir) if features_dir else settings.features_dir
    fee = fee_bps if fee_bps is not None else settings.taker_fee_bps

    # Load model
    from models.registry import latest_model_dir, load_model

    if model_dir:
        m_dir = Path(model_dir)
    else:
        m_dir = latest_model_dir("logistic_baseline", settings.models_dir)

    if m_dir is None:
        log.error("No model found. Run 'make train' first.")
        return

    model, scaler, meta = load_model(m_dir)
    feature_cols: list[str] = meta.get("feature_cols", [])

    if not feature_cols:
        log.error("Model metadata missing feature_cols.")
        return

    log.info("Loaded model from %s | features=%d", m_dir, len(feature_cols))

    # Load feature data
    pattern = str(feat_dir / f"symbol={sym}" / "date=*" / "*.parquet")
    try:
        df = (
            pl.scan_parquet(pattern, hive_partitioning=True)
            .filter(pl.col("is_clean"))
            .select(feature_cols + ["timestamp_exchange_us", "bid_prices", "ask_prices"])
            .drop_nulls()
            .collect()
            .sort("timestamp_exchange_us")
        )
    except Exception as e:
        log.error("Failed to load features: %s", e)
        return

    if df.is_empty():
        log.warning("No clean feature data found.")
        return

    log.info("Loaded %d rows for replay.", len(df))

    # Generate signals
    X = df.select(feature_cols).to_numpy()
    X_scaled = scaler.transform(X)
    prob_up = model.predict_proba(X_scaled)[:, 1]

    # Taker-only simulation
    positions: list[dict] = []
    pnl_per_trade: list[float] = []

    i = 0
    n = len(df)
    while i < n - holding_rows:
        p = prob_up[i]
        if p >= threshold:
            # Go long: buy at ask, sell at ask+holding
            entry_ask = df["ask_prices"][i][0]
            exit_bid = df["bid_prices"][i + holding_rows][0]
            if entry_ask is None or exit_bid is None:
                i += 1
                continue
            entry_fee = entry_ask * fee / 10_000
            exit_fee = exit_bid * fee / 10_000
            net_pnl = (exit_bid - entry_ask) - entry_fee - exit_fee
            pnl_per_trade.append(net_pnl)
            i += holding_rows  # skip forward
        else:
            i += 1

    if not pnl_per_trade:
        log.warning("No trades generated. Try lowering --threshold.")
        return

    total_pnl = sum(pnl_per_trade)
    n_trades = len(pnl_per_trade)
    win_rate = sum(1 for p in pnl_per_trade if p > 0) / n_trades
    avg_pnl = total_pnl / n_trades

    log.info("=" * 50)
    log.info("Replay P&L Summary")
    log.info("  Symbol:     %s", sym)
    log.info("  Rows:       %d", n)
    log.info("  Trades:     %d", n_trades)
    log.info("  Total P&L:  %.6f", total_pnl)
    log.info("  Avg P&L:    %.6f per trade", avg_pnl)
    log.info("  Win rate:   %.1f%%", win_rate * 100)
    log.info("  Fee (bps):  %.1f", fee)
    log.info("=" * 50)


if __name__ == "__main__":
    main()
