"""
lob_recorder.py — WebSocket LOB recorder for Binance Spot.

Implements the full integrity protocol from DATA_CONTRACTS.md:
  - Initial sync: REST snapshot + buffer drain
  - Steady-state sequence continuity: next.U == prev.u + 1
  - Ping/pong keepalive (handled by websockets lib)
  - 24-hour pre-emptive session rotation
  - Reconnect with exponential backoff

Output: hourly NDJSON files in output_dir/{SYMBOL}_{YYYYMMDD_HH}.ndjson
Each line is one JSON event with an injected timestamp_local_us field.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import aiohttp
import websockets
import websockets.exceptions

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_WS_BASE = "wss://stream.binance.com:9443/ws"
_REST_BASE = "https://api.binance.com/api/v3"
_REST_SNAPSHOT_LIMIT = 1000  # max levels from REST endpoint
_PING_INTERVAL = 30          # seconds; websockets lib sends pings
_PING_TIMEOUT = 10           # seconds; if no pong → connection dead
_SESSION_LIMIT = 23 * 3600   # pre-empt Binance 24 h forced close
_RECONNECT_BASE = 1.0        # initial backoff seconds
_RECONNECT_MAX = 60.0        # cap on backoff


def _now_us() -> int:
    """Current wall-clock time in microseconds."""
    return time.time_ns() // 1000


class LOBRecorder:
    """
    Streams Binance depth + trade events for a single symbol to hourly NDJSON files.

    Event types written to NDJSON:
      {"type": "snapshot", "lastUpdateId": int, "bids": [...], "asks": [...], "timestamp_local_us": int}
      {"type": "depth",    "e": "depthUpdate", "E": int, "U": int, "u": int, "b": [...], "a": [...], "timestamp_local_us": int}
      {"type": "trade",    "e": "trade", "E": int, "T": int, "s": str, ..., "timestamp_local_us": int}
      {"type": "gap_marker", "expected_U": int, "got_U": int, "timestamp_local_us": int}
    """

    def __init__(
        self,
        symbol: str,
        output_dir: Path,
        reconnect_base: float = _RECONNECT_BASE,
        reconnect_max: float = _RECONNECT_MAX,
    ) -> None:
        self.symbol = symbol.upper()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reconnect_base = reconnect_base
        self._reconnect_max = reconnect_max
        self._backoff: float = reconnect_base
        self._fh: TextIO | None = None
        self._fh_hour: str | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop. Reconnects with exponential backoff on any error."""
        while True:
            try:
                await self._session()
                self._backoff = self._reconnect_base  # clean exit → reset backoff
            except asyncio.CancelledError:
                logger.info("Recorder cancelled — shutting down.")
                self._close_file()
                raise
            except Exception as exc:
                logger.error(
                    "Session error: %s — reconnecting in %.1f s", exc, self._backoff
                )
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self._reconnect_max)

    # ── Session ────────────────────────────────────────────────────────────────

    async def _session(self) -> None:
        """
        One WebSocket session:
          1. Connect, fetch REST snapshot (with local buffer drain).
          2. Sync to sequence boundary.
          3. Steady-state: enforce next.U == prev.u + 1.
          4. Pre-emptive close after SESSION_LIMIT.
        """
        streams = (
            f"{self.symbol.lower()}@depth@100ms"
            f"/{self.symbol.lower()}@trade"
        )
        url = f"{_WS_BASE}/{streams}"

        async with websockets.connect(
            url,
            ping_interval=_PING_INTERVAL,
            ping_timeout=_PING_TIMEOUT,
        ) as ws:
            logger.info("Connected: %s", url)
            session_start = asyncio.get_event_loop().time()

            # ── Phase 1: initial sync ──────────────────────────────────────────
            # Buffer depth events while fetching REST snapshot concurrently.
            depth_buffer: list[dict] = []

            snapshot = await self._fetch_snapshot()
            last_update_id: int = snapshot["lastUpdateId"]
            self._write({"type": "snapshot", **snapshot, "timestamp_local_us": _now_us()})
            logger.info("Snapshot received: lastUpdateId=%d", last_update_id)

            # Drain any messages that arrived since we opened the socket.
            # We'll use a short non-blocking peek to grab buffered messages.
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    evt = json.loads(raw)
                    evt["timestamp_local_us"] = _now_us()
                    if evt.get("e") == "depthUpdate":
                        depth_buffer.append(evt)
                except asyncio.TimeoutError:
                    break

            # Apply sync rule: drop u <= lastUpdateId, find first U<=lastUpdateId+1<=u
            synced = False
            prev_u: int = 0

            for evt in depth_buffer:
                u = evt["u"]
                U = evt["U"]
                if u <= last_update_id:
                    continue
                if U <= last_update_id + 1 <= u:
                    self._write({"type": "depth", **evt})
                    prev_u = u
                    synced = True
                    logger.info("Synced at U=%d u=%d", U, u)
                    break

            if not synced:
                # No buffered event hit the sync window; will sync from live stream
                logger.debug("Sync window not in buffer — waiting for live events.")

            # ── Phase 2: steady-state ──────────────────────────────────────────
            async for raw in ws:
                recv_us = _now_us()
                evt = json.loads(raw)
                evt["timestamp_local_us"] = recv_us

                # Trade events — pass through
                if evt.get("e") == "trade":
                    self._write({"type": "trade", **evt})
                    continue

                # Depth update
                if evt.get("e") != "depthUpdate":
                    continue  # unknown event type, skip

                U: int = evt["U"]
                u: int = evt["u"]

                if not synced:
                    # Still looking for sync boundary
                    if u <= last_update_id:
                        continue
                    if U <= last_update_id + 1 <= u:
                        self._write({"type": "depth", **evt})
                        prev_u = u
                        synced = True
                        logger.info("Synced (live) at U=%d u=%d", U, u)
                    else:
                        logger.warning(
                            "Cannot sync: lastUpdateId=%d U=%d u=%d — restarting",
                            last_update_id, U, u,
                        )
                        return  # triggers reconnect
                    continue

                # Sequence continuity check
                expected_U = prev_u + 1
                if U != expected_U:
                    action = "GAP" if U > expected_U else "OUT-OF-ORDER"
                    logger.error(
                        "Sequence %s: expected U=%d got U=%d — flagging + resyncing",
                        action, expected_U, U,
                    )
                    self._write({
                        "type": "gap_marker",
                        "expected_U": expected_U,
                        "got_U": U,
                        "timestamp_local_us": recv_us,
                    })
                    return  # triggers reconnect + resync

                self._write({"type": "depth", **evt})
                prev_u = u

                # Pre-emptive session rotation
                if asyncio.get_event_loop().time() - session_start >= _SESSION_LIMIT:
                    logger.info("Session limit reached — pre-emptive reconnect.")
                    return

    # ── REST snapshot ──────────────────────────────────────────────────────────

    async def _fetch_snapshot(self) -> dict:
        """Fetch REST depth snapshot. Returns raw Binance response dict."""
        url = f"{_REST_BASE}/depth?symbol={self.symbol}&limit={_REST_SNAPSHOT_LIMIT}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
        return data

    # ── File I/O ───────────────────────────────────────────────────────────────

    def _write(self, event: dict) -> None:
        """Write event as a JSON line to the current hourly file."""
        hour = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
        if hour != self._fh_hour:
            self._rotate(hour)
        assert self._fh is not None
        self._fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        self._fh.flush()

    def _rotate(self, hour: str) -> None:
        """Open a new hourly NDJSON file, closing the previous one."""
        self._close_file()
        path = self.output_dir / f"{self.symbol}_{hour}.ndjson"
        self._fh = open(path, "a", encoding="utf-8")
        self._fh_hour = hour
        logger.info("Rotated log → %s", path)

    def _close_file(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._fh_hour = None
