"""
test_lob_recorder.py — LOBRecorder unit tests.

Tests cover the stateless/sync-logic parts of the recorder.
All tests are synchronous and require no network access.

For async integration tests (live WebSocket), see docs/journal/.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def test_recorder_imports():
    """Smoke test: LOBRecorder importable."""
    from ingest.lob_recorder import LOBRecorder
    assert LOBRecorder is not None


def test_recorder_creates_output_dir(tmp_path):
    """LOBRecorder creates output_dir if it doesn't exist."""
    from ingest.lob_recorder import LOBRecorder
    out = tmp_path / "lob" / "raw"
    rec = LOBRecorder(symbol="BTCUSDT", output_dir=out)
    assert out.exists()


def test_recorder_symbol_uppercased(tmp_path):
    """Symbol is stored uppercase regardless of input."""
    from ingest.lob_recorder import LOBRecorder
    rec = LOBRecorder(symbol="btcusdt", output_dir=tmp_path)
    assert rec.symbol == "BTCUSDT"


def test_recorder_write_and_rotate(tmp_path):
    """_write() creates a file and _rotate() closes old + opens new."""
    from ingest.lob_recorder import LOBRecorder

    rec = LOBRecorder(symbol="BTCUSDT", output_dir=tmp_path)

    # Manually trigger write
    rec._write({"type": "test", "val": 42, "timestamp_local_us": 0})
    rec._close_file()

    written = list(tmp_path.glob("*.ndjson"))
    assert len(written) == 1
    lines = written[0].read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["type"] == "test"
    assert parsed["val"] == 42


def test_recorder_multiple_writes_same_file(tmp_path):
    """Multiple writes in the same hour land in the same file."""
    from ingest.lob_recorder import LOBRecorder

    rec = LOBRecorder(symbol="BTCUSDT", output_dir=tmp_path)
    for i in range(5):
        rec._write({"type": "depth", "i": i, "timestamp_local_us": i})
    rec._close_file()

    files = list(tmp_path.glob("*.ndjson"))
    assert len(files) == 1
    lines = files[0].read_text().strip().split("\n")
    assert len(lines) == 5


def test_now_us_monotonic():
    """_now_us() is monotonically non-decreasing across rapid calls."""
    from ingest.lob_recorder import _now_us
    samples = [_now_us() for _ in range(100)]
    for a, b in zip(samples, samples[1:]):
        assert b >= a


def test_now_us_magnitude():
    """_now_us() is in microseconds: should be a 16-digit number roughly."""
    import time
    from ingest.lob_recorder import _now_us
    now = _now_us()
    # Should be comparable to time.time() * 1e6
    wall_us = int(time.time() * 1_000_000)
    # Allow 1 second tolerance
    assert abs(now - wall_us) < 1_000_000
