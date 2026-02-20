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
) -> tuple[int, int]:
    """
    Read one NDJSON file, reconstruct book, write Parquet partitions.

    Returns (snapshot_rows_written, trade_rows_written).
    Partition path: {processed_dir}/{table}/symbol={sym}/date={date}/{stem}.parquet

    Part filename equals the NDJSON stem (e.g. BTCUSDT_20260220_17.parquet) so
    multiple hourly files for the same date never overwrite each other.
    """
    ndjson_path = Path(ndjson_path)
    processed_dir = Path(processed_dir)

    stem = ndjson_path.stem  # e.g. BTCUSDT_20260220_17
    sym_from_name, date_str, _ = _parse_stem(stem)

    reconstructor = BookReconstructor(symbol=sym_from_name, depth=depth)
    snap_rows: list[dict] = []
    trade_rows: list[dict] = []

    with open(ndjson_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", ndjson_path)
                continue

            etype = event.get("type")

            if etype == "snapshot":
                reconstructor.apply_snapshot(event)

            elif etype == "depth":
                # Depth diff — apply and emit snapshot row
                result = reconstructor.apply_delta(event)
                if result == ApplyResult.OK:
                    # E field is in ms; convert to µs
                    exc_us = int(event.get("E", 0)) * 1000
                    local_us = int(event.get("timestamp_local_us", 0))
                    row = reconstructor.top_k_snapshot(exc_us, local_us)
                    snap_rows.append(row)
                elif result in (ApplyResult.GAP, ApplyResult.OUT_OF_ORDER):
                    logger.warning(
                        "[%s] %s — resync required. Rows so far: %d",
                        reconstructor.symbol, result.name, len(snap_rows),
                    )
                    reconstructor.reset()
                # NOT_SYNCED: no snapshot yet, skip

            elif etype == "trade":
                # Trade — extract and pass through
                side = "sell" if event.get("m") else "buy"
                trade_rows.append({
                    "timestamp_exchange_us": int(event.get("E", 0)) * 1000,
                    "timestamp_trade_us": int(event.get("T", 0)) * 1000,
                    "timestamp_local_us": int(event.get("timestamp_local_us", 0)),
                    "symbol": reconstructor.symbol,
                    "side": side,
                    "price": float(event.get("p", 0)),
                    "quantity": float(event.get("q", 0)),
                    "trade_id": int(event.get("t", 0)),
                })

            elif etype == "gap_marker":
                # Explicit gap from recorder — ensure state is reset
                reconstructor.reset()

    if not date_str:
        logger.warning("Could not determine date for %s — skipping write.", ndjson_path)
        return 0, 0

    sym = reconstructor.symbol or "UNKNOWN"
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
        "[%s %s] Normalised %s → %d snapshots, %d trades",
        sym, date_str, ndjson_path.name, len(snap_rows), len(trade_rows),
    )
    return len(snap_rows), len(trade_rows)


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

def normalise_directory(
    raw_dir: Path,
    processed_dir: Path,
    symbol: str,
    depth: int = 20,
) -> tuple[int, int]:
    """
    Normalise all NDJSON files for a symbol in raw_dir.
    Returns total (snapshot_rows, trade_rows).
    """
    raw_dir = Path(raw_dir)
    pattern = f"{symbol.upper()}_*.ndjson"
    files = sorted(raw_dir.glob(pattern))

    if not files:
        logger.warning("No NDJSON files found for pattern: %s/%s", raw_dir, pattern)
        return 0, 0

    total_snaps = total_trades = 0
    for f in files:
        s, t = normalise_file(f, processed_dir, depth=depth)
        total_snaps += s
        total_trades += t

    logger.info(
        "Normalised %d files → %d snapshot rows, %d trade rows",
        len(files), total_snaps, total_trades,
    )
    return total_snaps, total_trades
