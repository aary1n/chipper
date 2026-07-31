"""
lob_normaliser.py — raw NDJSON → canonical Parquet.

Reads hourly NDJSON files written by lob_recorder.py and:
  1. Reconstructs book state via BookReconstructor.
  2. Validates sequence continuity independently (defence in depth).
  3. Emits one snapshot row per depth event, with is_clean flag.
  4. Writes book snapshots + trades to hive-partitioned Parquet.

Schema: see DATA_CONTRACTS.md §Schemas.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum, auto
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ── Result enum ────────────────────────────────────────────────────────────────

class ApplyResult(Enum):
    OK = auto()
    NOT_SYNCED = auto()   # no snapshot loaded yet
    GAP = auto()          # next.U > prev.u + 1
    OUT_OF_ORDER = auto() # next.U < prev.u + 1


# ── Book Reconstructor ─────────────────────────────────────────────────────────

class BookReconstructor:
    """
    Maintains live order book state from a sequence of snapshot + delta events.

    Sequence invariant (DATA_CONTRACTS.md §Sequence Integrity):
        next_event.U == self._last_update_id + 1

    is_clean semantics:
        True  — book was built from continuous deltas since last REST snapshot.
        False — a gap or OOO was detected; state is invalid until apply_snapshot().
    """

    def __init__(self, symbol: str, depth: int = 20) -> None:
        self.symbol = symbol.upper()
        self.depth = depth
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_update_id: int | None = None
        self.is_clean: bool = False

    @property
    def is_synced(self) -> bool:
        return self._last_update_id is not None

    def apply_snapshot(self, snapshot: dict) -> None:
        """
        Load a REST snapshot dict (keys: lastUpdateId, bids, asks).
        Resets is_clean to True — fresh verified baseline.
        """
        self._bids = {
            float(p): float(q)
            for p, q in snapshot["bids"]
            if float(q) > 0
        }
        self._asks = {
            float(p): float(q)
            for p, q in snapshot["asks"]
            if float(q) > 0
        }
        self._last_update_id = int(snapshot["lastUpdateId"])
        self.is_clean = True
        logger.debug(
            "[%s] Snapshot applied: lastUpdateId=%d bids=%d asks=%d",
            self.symbol, self._last_update_id, len(self._bids), len(self._asks),
        )

    def apply_delta(self, event: dict) -> ApplyResult:
        """
        Apply a depth diff event. Returns ApplyResult.

        On GAP or OUT_OF_ORDER:
          - Sets is_clean=False
          - Invalidates state (is_synced becomes False)
          - Caller must resync via apply_snapshot() before continuing.
        """
        if not self.is_synced:
            return ApplyResult.NOT_SYNCED

        U: int = int(event["U"])
        u: int = int(event["u"])
        expected_U = self._last_update_id + 1  # type: ignore[operator]

        if U != expected_U:
            self.is_clean = False
            self._last_update_id = None  # invalidate
            if U > expected_U:
                logger.error(
                    "[%s] GAP: expected U=%d got U=%d (gap of %d)",
                    self.symbol, expected_U, U, U - expected_U,
                )
                return ApplyResult.GAP
            else:
                logger.error(
                    "[%s] OUT-OF-ORDER: expected U=%d got U=%d",
                    self.symbol, expected_U, U,
                )
                return ApplyResult.OUT_OF_ORDER

        # Apply bid updates
        for price_s, qty_s in event.get("b", []):
            price, qty = float(price_s), float(qty_s)
            if qty == 0.0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = qty

        # Apply ask updates
        for price_s, qty_s in event.get("a", []):
            price, qty = float(price_s), float(qty_s)
            if qty == 0.0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = qty

        self._last_update_id = u
        return ApplyResult.OK

    def top_k_snapshot(
        self,
        timestamp_exchange_us: int,
        timestamp_local_us: int,
    ) -> dict:
        """
        Extract top-K levels as fixed-length lists (padded with NaN).
        Returns a dict matching the snapshot schema in DATA_CONTRACTS.md.
        """
        top_bids = sorted(self._bids.items(), reverse=True)[: self.depth]
        top_asks = sorted(self._asks.items())[: self.depth]

        pad_bids = self.depth - len(top_bids)
        pad_asks = self.depth - len(top_asks)

        return {
            "timestamp_exchange_us": timestamp_exchange_us,
            "timestamp_local_us": timestamp_local_us,
            "symbol": self.symbol,
            "last_update_id": self._last_update_id,
            "bid_prices": [p for p, _ in top_bids] + [float("nan")] * pad_bids,
            "bid_quantities": [q for _, q in top_bids] + [float("nan")] * pad_bids,
            "ask_prices": [p for p, _ in top_asks] + [float("nan")] * pad_asks,
            "ask_quantities": [q for _, q in top_asks] + [float("nan")] * pad_asks,
            "is_clean": self.is_clean,
        }

    def reset(self) -> None:
        """Full state reset. is_synced becomes False."""
        self._bids = {}
        self._asks = {}
        self._last_update_id = None
        self.is_clean = False

    def to_state(self) -> dict:
        """Serializable snapshot of reconstruction state (checkpoint/resume)."""
        return {
            "bids": [[p, q] for p, q in self._bids.items()],
            "asks": [[p, q] for p, q in self._asks.items()],
            "last_update_id": self._last_update_id,
            "is_clean": self.is_clean,
        }

    def load_state(self, state: dict) -> None:
        """Restore state produced by to_state()."""
        self._bids = {float(p): float(q) for p, q in state["bids"]}
        self._asks = {float(p): float(q) for p, q in state["asks"]}
        lu = state["last_update_id"]
        self._last_update_id = int(lu) if lu is not None else None
        self.is_clean = bool(state["is_clean"])


# ── Snapshot Parquet schema ────────────────────────────────────────────────────

def _snapshot_arrow_schema(depth: int) -> pa.Schema:
    list_f64 = pa.list_(pa.float64())
    return pa.schema([
        pa.field("timestamp_exchange_us", pa.int64()),
        pa.field("timestamp_local_us", pa.int64()),
        pa.field("symbol", pa.string()),
        pa.field("last_update_id", pa.int64()),
        pa.field("bid_prices", list_f64),
        pa.field("bid_quantities", list_f64),
        pa.field("ask_prices", list_f64),
        pa.field("ask_quantities", list_f64),
        pa.field("is_clean", pa.bool_()),
    ])


def _trade_arrow_schema() -> pa.Schema:
    return pa.schema([
        pa.field("timestamp_exchange_us", pa.int64()),
        pa.field("timestamp_trade_us", pa.int64()),
        pa.field("timestamp_local_us", pa.int64()),
        pa.field("symbol", pa.string()),
        pa.field("side", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("quantity", pa.float64()),
        pa.field("trade_id", pa.int64()),
    ])


# ── Per-file normalisation stats ───────────────────────────────────────────────

@dataclass
class FileStats:
    """
    Skip-and-count accounting for one NDJSON file. Nothing is dropped
    silently: every line or event that does not become an output row is
    counted in exactly one of these fields.
    """

    file: str
    snapshot_rows: int = 0
    trade_rows: int = 0
    malformed_lines: int = 0    # unparseable JSON (e.g. truncation at ENOSPC)
    invalid_events: int = 0     # parseable JSON with missing/bad fields
    pre_sync_skipped: int = 0   # depth events before book sync (awaiting snapshot)
    gaps: int = 0               # sequence gaps detected (next.U > prev.u + 1)
    out_of_order: int = 0       # stale events detected (next.U < prev.u + 1)
    gap_markers: int = 0        # explicit gap_marker records from the recorder
    unknown_type: int = 0       # unrecognised event types
    first_exchange_us: int | None = None  # E of first emitted snapshot row
    last_exchange_us: int | None = None   # E of last emitted snapshot row
    max_interarrival_us: int = 0  # max E-gap between consecutive emitted rows


# ── File-level normalisation ───────────────────────────────────────────────────

def _parse_stem(stem: str) -> tuple[str, str, str]:
    """
    Parse NDJSON filename stem: BTCUSDT_20260220_17
    Returns (symbol, date_str, hour_str).  date_str format: YYYY-MM-DD.
    """
    parts = stem.split("_")
    symbol = parts[0] if parts else "UNKNOWN"
    date_str = ""
    hour_str = "00"
    if len(parts) >= 3:
        raw_date = parts[-2]           # e.g. "20260220"
        hour_str = parts[-1]           # e.g. "17"
        if len(raw_date) == 8:
            date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
    return symbol, date_str, hour_str


def normalise_file(
    ndjson_path: Path,
    processed_dir: Path,
    depth: int = 20,
    reconstructor: BookReconstructor | None = None,
) -> FileStats:
    """
    Read one NDJSON file, reconstruct book, write Parquet partitions.

    Pass a shared `reconstructor` to continue book state across consecutive
    hourly files: the recorder writes a `snapshot` event only at session
    start, so mid-session hourly files have no leading snapshot and would
    otherwise emit zero rows. If omitted, a fresh reconstructor is used
    (standalone single-file mode).

    Returns FileStats. Every skipped line or event is counted — nothing is
    dropped silently. The raw NDJSON is never modified.

    Partition path: {processed_dir}/{table}/symbol={sym}/date={date}/{stem}.parquet
    Part filename equals the NDJSON stem (e.g. BTCUSDT_20260220_17.parquet) so
    multiple hourly files for the same date never overwrite each other.
    """
    ndjson_path = Path(ndjson_path)
    processed_dir = Path(processed_dir)

    stem = ndjson_path.stem  # e.g. BTCUSDT_20260220_17
    sym_from_name, date_str, _ = _parse_stem(stem)

    rec = reconstructor if reconstructor is not None else BookReconstructor(
        symbol=sym_from_name, depth=depth
    )
    stats = FileStats(file=ndjson_path.name)
    snap_rows: list[dict] = []
    trade_rows: list[dict] = []
    prev_emit_us: int | None = None

    with open(ndjson_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                stats.malformed_lines += 1
                logger.warning(
                    "Skipping malformed line %d in %s: %s",
                    line_no, ndjson_path.name, exc,
                )
                continue

            etype = event.get("type")
            try:
                if etype == "snapshot":
                    rec.apply_snapshot(event)

                elif etype == "depth":
                    # Depth diff — apply and emit snapshot row
                    result = rec.apply_delta(event)
                    if result == ApplyResult.OK:
                        # E field is in ms; convert to µs
                        exc_us = int(event.get("E", 0)) * 1000
                        local_us = int(event.get("timestamp_local_us", 0))
                        row = rec.top_k_snapshot(exc_us, local_us)
                        snap_rows.append(row)
                        if stats.first_exchange_us is None:
                            stats.first_exchange_us = exc_us
                        stats.last_exchange_us = exc_us
                        if prev_emit_us is not None:
                            stats.max_interarrival_us = max(
                                stats.max_interarrival_us, exc_us - prev_emit_us
                            )
                        prev_emit_us = exc_us
                    elif result == ApplyResult.GAP:
                        stats.gaps += 1
                        logger.warning(
                            "[%s] GAP at line %d in %s — resync required.",
                            rec.symbol, line_no, ndjson_path.name,
                        )
                        rec.reset()
                    elif result == ApplyResult.OUT_OF_ORDER:
                        stats.out_of_order += 1
                        logger.warning(
                            "[%s] OUT_OF_ORDER at line %d in %s — resync required.",
                            rec.symbol, line_no, ndjson_path.name,
                        )
                        rec.reset()
                    else:  # NOT_SYNCED: awaiting snapshot after gap/session start
                        stats.pre_sync_skipped += 1

                elif etype == "trade":
                    # Trade — extract and pass through
                    side = "sell" if event.get("m") else "buy"
                    trade_rows.append({
                        "timestamp_exchange_us": int(event.get("E", 0)) * 1000,
                        "timestamp_trade_us": int(event.get("T", 0)) * 1000,
                        "timestamp_local_us": int(event.get("timestamp_local_us", 0)),
                        "symbol": rec.symbol,
                        "side": side,
                        "price": float(event.get("p", 0)),
                        "quantity": float(event.get("q", 0)),
                        "trade_id": int(event.get("t", 0)),
                    })

                elif etype == "gap_marker":
                    # Explicit gap from recorder — ensure state is reset
                    stats.gap_markers += 1
                    rec.reset()

                else:
                    stats.unknown_type += 1
                    logger.warning(
                        "Unknown event type %r at line %d in %s",
                        etype, line_no, ndjson_path.name,
                    )
            except (KeyError, ValueError, TypeError) as exc:
                stats.invalid_events += 1
                logger.warning(
                    "Skipping invalid %r event at line %d in %s: %s",
                    etype, line_no, ndjson_path.name, exc,
                )
                if etype == "depth":
                    # A depth event may have partially applied — book state is
                    # no longer trustworthy until the next snapshot.
                    rec.reset()

    stats.snapshot_rows = len(snap_rows)
    stats.trade_rows = len(trade_rows)

    if not date_str:
        logger.warning(
            "Could not determine date for %s — dropping %d snapshot / %d trade rows.",
            ndjson_path, len(snap_rows), len(trade_rows),
        )
        stats.snapshot_rows = 0
        stats.trade_rows = 0
        return stats

    sym = rec.symbol or "UNKNOWN"
    part_name = f"{stem}.parquet"  # e.g. BTCUSDT_20260220_17.parquet — unique per source file

    # Write snapshots
    if snap_rows:
        snap_df = pl.DataFrame(snap_rows)
        _write_partition(snap_df, "snapshots", sym, date_str, processed_dir, part_name)

    # Write trades
    if trade_rows:
        trade_df = pl.DataFrame(trade_rows)
        _write_partition(trade_df, "trades", sym, date_str, processed_dir, part_name)

    logger.info(
        "[%s %s] Normalised %s → %d snapshots, %d trades "
        "(malformed=%d invalid=%d pre_sync=%d gaps=%d ooo=%d markers=%d)",
        sym, date_str, ndjson_path.name, len(snap_rows), len(trade_rows),
        stats.malformed_lines, stats.invalid_events, stats.pre_sync_skipped,
        stats.gaps, stats.out_of_order, stats.gap_markers,
    )
    return stats


def _write_partition(
    df: pl.DataFrame,
    table: str,
    symbol: str,
    date: str,
    processed_dir: Path,
    part_name: str = "part-0.parquet",
) -> None:
    out_dir = processed_dir / table / f"symbol={symbol}" / f"date={date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / part_name
    df.write_parquet(out_path, compression="snappy")
    logger.debug("Wrote %s → %s (%d rows)", table, out_path, len(df))


# ── Batch normalisation ────────────────────────────────────────────────────────

_STAT_TOTAL_FIELDS = (
    "snapshot_rows", "trade_rows", "malformed_lines", "invalid_events",
    "pre_sync_skipped", "gaps", "out_of_order", "gap_markers", "unknown_type",
)

_CHECKPOINT_NAME = "_normalise_checkpoint.json"


def _write_checkpoint(
    processed_dir: Path,
    symbol: str,
    depth: int,
    stats_list: list[FileStats],
    rec: BookReconstructor,
) -> None:
    """Persist resume state after a completed file. Atomic replace so an
    interruption can never leave a torn checkpoint."""
    payload = {
        "symbol": symbol.upper(),
        "depth": depth,
        "last_completed_file": stats_list[-1].file,
        "reconstructor": rec.to_state(),
        "stats_so_far": [asdict(s) for s in stats_list],
    }
    path = processed_dir / _CHECKPOINT_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint(processed_dir: Path, symbol: str, depth: int) -> dict | None:
    """Load a resume checkpoint if present and compatible, else None."""
    path = processed_dir / _CHECKPOINT_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Corrupt checkpoint %s — ignoring, starting fresh.", path)
        return None
    if payload.get("symbol") != symbol.upper() or payload.get("depth") != depth:
        logger.warning(
            "Checkpoint %s is for symbol=%s depth=%s (want %s/%d) — ignoring.",
            path, payload.get("symbol"), payload.get("depth"), symbol.upper(), depth,
        )
        return None
    return payload


def normalise_directory(
    raw_dir: Path,
    processed_dir: Path,
    symbol: str,
    depth: int = 20,
    resume: bool = True,
) -> list[FileStats]:
    """
    Normalise all NDJSON files for a symbol in raw_dir, in chronological
    (filename) order, threading a single BookReconstructor across files so
    book state survives hourly-file boundaries. Sequence continuity is
    enforced across boundaries too: a discontinuity between files is
    detected as a gap exactly like an intra-file one.

    A checkpoint (`_normalise_checkpoint.json`) is written after every
    completed file; with resume=True (default) an interrupted run picks up
    exactly where it stopped, restoring the reconstructor's book state.
    The checkpoint is deleted on successful completion.

    Returns per-file FileStats. Also writes `_normalise_report.json`
    (per-file stats + totals) into processed_dir.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    pattern = f"{symbol.upper()}_*.ndjson"
    files = sorted(raw_dir.glob(pattern))

    if not files:
        logger.warning("No NDJSON files found for pattern: %s/%s", raw_dir, pattern)
        return []

    processed_dir.mkdir(parents=True, exist_ok=True)
    rec = BookReconstructor(symbol=symbol, depth=depth)
    all_stats: list[FileStats] = []

    ckpt = _load_checkpoint(processed_dir, symbol, depth) if resume else None
    if ckpt is not None:
        done = ckpt["last_completed_file"]
        remaining = [f for f in files if f.name > done]
        rec.load_state(ckpt["reconstructor"])
        all_stats = [FileStats(**s) for s in ckpt["stats_so_far"]]
        logger.info(
            "Resuming after %s — %d file(s) already done, %d remaining.",
            done, len(all_stats), len(remaining),
        )
        files = remaining

    for f in files:
        all_stats.append(normalise_file(f, processed_dir, depth=depth, reconstructor=rec))
        _write_checkpoint(processed_dir, symbol, depth, all_stats, rec)

    totals = {
        k: sum(getattr(s, k) for s in all_stats) for k in _STAT_TOTAL_FIELDS
    }
    report = {
        "symbol": symbol.upper(),
        "raw_dir": str(raw_dir),
        "depth": depth,
        "n_files": len(files),
        "totals": totals,
        "files": [asdict(s) for s in all_stats],
    }
    report_path = processed_dir / "_normalise_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Run completed — the mid-run checkpoint is no longer needed.
    (processed_dir / _CHECKPOINT_NAME).unlink(missing_ok=True)

    logger.info(
        "Normalised %d files → %d snapshot rows, %d trade rows "
        "(skipped: %d malformed, %d invalid; anomalies: %d gaps, %d ooo, "
        "%d gap_markers). Report: %s",
        len(files), totals["snapshot_rows"], totals["trade_rows"],
        totals["malformed_lines"], totals["invalid_events"],
        totals["gaps"], totals["out_of_order"], totals["gap_markers"],
        report_path,
    )
    return all_stats
