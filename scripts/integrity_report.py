"""
scripts/integrity_report.py — Data-integrity audit over normalised Parquet.

Usage:
    python scripts/integrity_report.py --symbol BTCUSDT
    python scripts/integrity_report.py --processed-dir <dir> --out report.md

Reads `_normalise_report.json` (per-file skip-and-count stats from the
normaliser) plus the snapshots table, and reports:
  - totals: rows, skipped lines/events, sequence anomalies
  - per-file: rows, max inter-arrival gap (flagged above threshold)
  - data holes: every inter-arrival break in the snapshot timeline
  - per-day coverage vs the nominal 10 Hz cadence

Report only — never modifies data. No interpolation, no gap-bridging.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
import polars as pl

from config import settings

NOMINAL_ROWS_PER_HOUR = 36_000  # 10 Hz × 3600 s


def _fmt_us(ts_us: int) -> str:
    return datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _fmt_dur(us: int) -> str:
    s = us / 1_000_000
    if s < 120:
        return f"{s:.1f}s"
    return f"{s / 60:.1f}min"


@click.command()
@click.option("--symbol", default=None, help="Trading symbol")
@click.option("--processed-dir", default=None, type=click.Path())
@click.option("--hole-threshold-s", default=5.0, type=float,
              help="Report a data hole where snapshot inter-arrival exceeds this")
@click.option("--out", default=None, type=click.Path(),
              help="Markdown output path (default: <processed-dir>/_integrity_report.md)")
def main(
    symbol: str | None,
    processed_dir: str | None,
    hole_threshold_s: float,
    out: str | None,
) -> None:
    """Audit normalised data integrity; write a markdown report."""
    settings.configure_logging()
    log = logging.getLogger("integrity_report")

    sym = (symbol or settings.symbol).upper()
    src = Path(processed_dir) if processed_dir else settings.processed_dir
    out_path = Path(out) if out else src / "_integrity_report.md"
    thr_us = int(hole_threshold_s * 1_000_000)

    lines: list[str] = [f"# Integrity report — {sym}", ""]

    # ── 1. Normaliser skip-and-count stats ─────────────────────────────────────
    report_json = src / "_normalise_report.json"
    per_file: list[dict] = []
    if report_json.exists():
        norm = json.loads(report_json.read_text(encoding="utf-8"))
        per_file = norm.get("files", [])
        t = norm.get("totals", {})
        lines += [
            "## Normaliser totals (skip-and-count)",
            "",
            f"- files: {norm.get('n_files')}",
            f"- snapshot rows: {t.get('snapshot_rows'):,}",
            f"- trade rows: {t.get('trade_rows'):,}",
            f"- malformed lines skipped: {t.get('malformed_lines')}",
            f"- invalid events skipped: {t.get('invalid_events')}",
            f"- pre-sync depth events skipped: {t.get('pre_sync_skipped')}",
            f"- sequence gaps detected: {t.get('gaps')}",
            f"- out-of-order events: {t.get('out_of_order')}",
            f"- recorder gap markers: {t.get('gap_markers')}",
            f"- unknown event types: {t.get('unknown_type')}",
            "",
        ]
    else:
        lines += [f"⚠ `{report_json}` not found — run scripts/normalise.py first.", ""]
        log.warning("No _normalise_report.json at %s", report_json)

    # ── 2. Per-file max inter-arrival gap ──────────────────────────────────────
    if per_file:
        lines += [
            "## Per-file max inter-arrival gap (emitted snapshot rows)",
            "",
            "| file | rows | max gap | flags |",
            "|------|------|---------|-------|",
        ]
        for f in per_file:
            gap = f.get("max_interarrival_us") or 0
            flags = []
            if gap > thr_us:
                flags.append("GAP>THR")
            if f.get("malformed_lines"):
                flags.append(f"malformed={f['malformed_lines']}")
            if f.get("invalid_events"):
                flags.append(f"invalid={f['invalid_events']}")
            if f.get("gaps") or f.get("out_of_order"):
                flags.append(f"seq={f.get('gaps', 0)}+{f.get('out_of_order', 0)}")
            lines.append(
                f"| {f['file']} | {f.get('snapshot_rows', 0):,} | "
                f"{_fmt_dur(gap)} | {' '.join(flags)} |"
            )
        lines.append("")

    # ── 3. Timeline holes across the whole snapshots table ─────────────────────
    snap_pattern = str(src / "snapshots" / f"symbol={sym}" / "date=*" / "*.parquet")
    ts = (
        pl.scan_parquet(snap_pattern, hive_partitioning=True)
        .select("timestamp_exchange_us")
        .sort("timestamp_exchange_us")
        .with_columns(pl.col("timestamp_exchange_us").diff().alias("gap_us"))
        .filter(pl.col("gap_us") > thr_us)
        .collect()
    )
    full = (
        pl.scan_parquet(snap_pattern, hive_partitioning=True)
        .select(
            pl.col("timestamp_exchange_us").min().alias("t0"),
            pl.col("timestamp_exchange_us").max().alias("t1"),
            pl.len().alias("n"),
            pl.col("is_clean").not_().sum().alias("n_dirty"),
        )
        .collect()
    )
    t0, t1 = full["t0"][0], full["t1"][0]
    n_rows, n_dirty = full["n"][0], full["n_dirty"][0]

    lines += [
        "## Snapshot timeline",
        "",
        f"- range: {_fmt_us(t0)} → {_fmt_us(t1)} UTC",
        f"- rows: {n_rows:,} ({n_dirty} with is_clean=False)",
        f"- holes > {hole_threshold_s:.0f}s: {len(ts)}",
        "",
    ]
    if len(ts):
        lines += ["| hole start (UTC) | duration |", "|------------------|----------|"]
        for row in ts.iter_rows(named=True):
            start = row["timestamp_exchange_us"] - row["gap_us"]
            lines.append(f"| {_fmt_us(start)} | {_fmt_dur(row['gap_us'])} |")
        lines.append("")

    # ── 4. Per-day coverage ────────────────────────────────────────────────────
    daily = (
        pl.scan_parquet(snap_pattern, hive_partitioning=True)
        .group_by("date")
        .agg(
            pl.len().alias("rows"),
            (pl.col("timestamp_exchange_us").max()
             - pl.col("timestamp_exchange_us").min()).alias("span_us"),
        )
        .sort("date")
        .collect()
    )
    lines += [
        "## Per-day coverage",
        "",
        "| date | rows | span | rows/nominal-24h |",
        "|------|------|------|------------------|",
    ]
    for row in daily.iter_rows(named=True):
        pct = 100.0 * row["rows"] / (NOMINAL_ROWS_PER_HOUR * 24)
        lines.append(
            f"| {row['date']} | {row['rows']:,} | {_fmt_dur(row['span_us'])} | {pct:.1f}% |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written → %s", out_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
