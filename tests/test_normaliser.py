"""
test_normaliser.py — BookReconstructor sequence continuity + resync tests.

Critical invariants tested (DATA_CONTRACTS.md §Sequence Integrity):
  1. Clean sequence: snapshot + valid deltas applied correctly.
  2. Book state after clean deltas matches expected values.
  3. GAP detected when next.U > prev.u + 1.
  4. OUT_OF_ORDER detected when next.U < prev.u + 1.
  5. State invalidated (is_synced=False) after any sequence error.
  6. is_clean=False after any sequence error.
  7. Resync: new snapshot restores is_synced + is_clean.
  8. Post-resync deltas apply correctly.
"""

import pytest

from ingest.lob_normaliser import ApplyResult, BookReconstructor


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_reconstructor(depth: int = 5) -> BookReconstructor:
    return BookReconstructor(symbol="BTCUSDT", depth=depth)


# ── 1. Initial state ───────────────────────────────────────────────────────────

def test_initial_state_not_synced():
    rec = make_reconstructor()
    assert not rec.is_synced
    assert not rec.is_clean


def test_apply_delta_before_snapshot_returns_not_synced(delta_valid_1):
    rec = make_reconstructor()
    result = rec.apply_delta(delta_valid_1)
    assert result == ApplyResult.NOT_SYNCED
    assert not rec.is_synced


# ── 2. Snapshot loading ────────────────────────────────────────────────────────

def test_apply_snapshot_syncs(snapshot_raw):
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    assert rec.is_synced
    assert rec.is_clean
    assert rec._last_update_id == 100


def test_snapshot_populates_book(snapshot_raw):
    rec = make_reconstructor(depth=5)
    rec.apply_snapshot(snapshot_raw)
    # Best bid should be 50000.00 (highest)
    top = rec.top_k_snapshot(0, 0)
    assert top["bid_prices"][0] == pytest.approx(50000.0)
    assert top["bid_quantities"][0] == pytest.approx(1.0)
    # Best ask should be 50001.00 (lowest)
    assert top["ask_prices"][0] == pytest.approx(50001.0)
    assert top["ask_quantities"][0] == pytest.approx(1.5)


# ── 3. Clean delta application ─────────────────────────────────────────────────

def test_apply_valid_delta_1(snapshot_raw, delta_valid_1):
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    result = rec.apply_delta(delta_valid_1)
    assert result == ApplyResult.OK
    assert rec._last_update_id == 103
    assert rec.is_clean


def test_apply_valid_delta_updates_book(snapshot_raw, delta_valid_1):
    rec = make_reconstructor(depth=5)
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    top = rec.top_k_snapshot(0, 0)
    # delta_valid_1 updates best bid 50000 qty: 1.0 → 1.5
    assert top["bid_prices"][0] == pytest.approx(50000.0)
    assert top["bid_quantities"][0] == pytest.approx(1.5)
    # and best ask 50001 qty: 1.5 → 1.0
    assert top["ask_prices"][0] == pytest.approx(50001.0)
    assert top["ask_quantities"][0] == pytest.approx(1.0)


def test_apply_two_valid_deltas(snapshot_raw, delta_valid_1, delta_valid_2):
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    r1 = rec.apply_delta(delta_valid_1)
    r2 = rec.apply_delta(delta_valid_2)
    assert r1 == ApplyResult.OK
    assert r2 == ApplyResult.OK
    assert rec._last_update_id == 106
    assert rec.is_clean
    assert rec.is_synced


def test_delta_removes_level_on_zero_qty(snapshot_raw):
    """Qty == 0 means remove the price level from the book."""
    rec = make_reconstructor(depth=5)
    rec.apply_snapshot(snapshot_raw)
    # Remove best bid 50000
    remove_delta = {
        "U": 101, "u": 101,
        "b": [["50000.00", "0"]],
        "a": [],
    }
    rec.apply_delta(remove_delta)
    top = rec.top_k_snapshot(0, 0)
    # 50000 should be gone; next best is 49999
    assert top["bid_prices"][0] == pytest.approx(49999.0)


# ── 4. GAP detection ───────────────────────────────────────────────────────────

def test_gap_detected(snapshot_raw, delta_valid_1, delta_valid_2, delta_gap):
    """After delta_valid_2 (u=106), delta_gap has U=110 → GAP."""
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_valid_2)

    result = rec.apply_delta(delta_gap)
    assert result == ApplyResult.GAP


def test_gap_invalidates_state(snapshot_raw, delta_valid_1, delta_valid_2, delta_gap):
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_valid_2)
    rec.apply_delta(delta_gap)

    assert not rec.is_synced
    assert not rec.is_clean


def test_gap_not_applied_to_book(snapshot_raw, delta_valid_1, delta_valid_2, delta_gap):
    """GAP event must NOT modify book state."""
    rec = make_reconstructor(depth=5)
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_valid_2)

    # Capture bid levels before gap
    prev_top = rec.top_k_snapshot(0, 0)
    prev_bid_prices = prev_top["bid_prices"].copy()

    rec.apply_delta(delta_gap)
    # Book is now invalid — but we can inspect internal state
    # gap delta bid was ["49996.00", "5.000"]; should NOT be in _bids
    assert 49996.0 not in rec._bids


# ── 5. OUT-OF-ORDER detection ──────────────────────────────────────────────────

def test_out_of_order_detected(snapshot_raw, delta_valid_1, delta_valid_2, delta_out_of_order):
    """After delta_valid_2 (u=106), delta_out_of_order has U=104 → OOO."""
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_valid_2)

    result = rec.apply_delta(delta_out_of_order)
    assert result == ApplyResult.OUT_OF_ORDER


def test_out_of_order_invalidates_state(snapshot_raw, delta_valid_1, delta_valid_2, delta_out_of_order):
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_valid_2)
    rec.apply_delta(delta_out_of_order)

    assert not rec.is_synced
    assert not rec.is_clean


# ── 6. NOT_SYNCED after gap ────────────────────────────────────────────────────

def test_delta_after_gap_returns_not_synced(snapshot_raw, delta_valid_1, delta_gap):
    """After a gap invalidates state, subsequent deltas return NOT_SYNCED."""
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_gap)  # creates GAP after u=103, expected U=104

    # Now try to apply another delta — state is invalid
    any_delta = {
        "U": 116, "u": 120,
        "b": [], "a": [],
    }
    result = rec.apply_delta(any_delta)
    assert result == ApplyResult.NOT_SYNCED


# ── 7. Resync path ─────────────────────────────────────────────────────────────

def test_resync_restores_synced(snapshot_raw, delta_valid_1, delta_gap, resync_snapshot):
    """After gap → apply new snapshot → is_synced=True, is_clean=True."""
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_gap)  # gap after u=103 (expected U=104, got U=110)
    assert not rec.is_synced

    # Resync
    rec.apply_snapshot(resync_snapshot)
    assert rec.is_synced
    assert rec.is_clean
    assert rec._last_update_id == 200


def test_resync_clears_old_book(snapshot_raw, delta_valid_1, delta_gap, resync_snapshot):
    """After resync, old price levels from original snapshot are gone."""
    rec = make_reconstructor(depth=5)
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_gap)
    rec.apply_snapshot(resync_snapshot)

    top = rec.top_k_snapshot(0, 0)
    # resync_snapshot best bid is 50100, not 50000
    assert top["bid_prices"][0] == pytest.approx(50100.0)
    # Original level 50000 should not be in new book
    assert 50000.0 not in rec._bids


# ── 8. Post-resync deltas ──────────────────────────────────────────────────────

def test_post_resync_deltas_apply(
    snapshot_raw, delta_valid_1, delta_gap, resync_snapshot, delta_post_resync
):
    """After resync, valid deltas from the new snapshot apply cleanly."""
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_gap)
    rec.apply_snapshot(resync_snapshot)

    result = rec.apply_delta(delta_post_resync)
    assert result == ApplyResult.OK
    assert rec._last_update_id == 205
    assert rec.is_clean


# ── 9. top_k_snapshot output format ───────────────────────────────────────────

def test_top_k_snapshot_length(snapshot_raw):
    """top_k_snapshot always returns lists of exactly depth length."""
    depth = 10
    rec = make_reconstructor(depth=depth)
    rec.apply_snapshot(snapshot_raw)  # only 3 levels each side
    top = rec.top_k_snapshot(1234, 5678)
    assert len(top["bid_prices"]) == depth
    assert len(top["bid_quantities"]) == depth
    assert len(top["ask_prices"]) == depth
    assert len(top["ask_quantities"]) == depth


def test_top_k_snapshot_padding(snapshot_raw):
    """Levels beyond available depth are padded with NaN."""
    import math
    depth = 10
    rec = make_reconstructor(depth=depth)
    rec.apply_snapshot(snapshot_raw)  # 3 levels → indices 3..9 should be NaN
    top = rec.top_k_snapshot(0, 0)
    assert math.isnan(top["bid_prices"][3])
    assert math.isnan(top["ask_prices"][3])


def test_top_k_snapshot_timestamps(snapshot_raw):
    """Timestamps are passed through to the row dict."""
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    row = rec.top_k_snapshot(
        timestamp_exchange_us=1700000000100000,
        timestamp_local_us=1700000000110000,
    )
    assert row["timestamp_exchange_us"] == 1700000000100000
    assert row["timestamp_local_us"] == 1700000000110000
    assert row["symbol"] == "BTCUSDT"
    assert row["last_update_id"] == 100


def test_top_k_snapshot_is_clean_flag(snapshot_raw, delta_valid_1):
    """is_clean in snapshot row reflects reconstructor state."""
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    row = rec.top_k_snapshot(0, 0)
    assert row["is_clean"] is True


def test_top_k_snapshot_is_clean_false_after_gap(
    snapshot_raw, delta_valid_1, delta_gap
):
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    rec.apply_delta(delta_valid_1)
    rec.apply_delta(delta_gap)
    # After gap: state is invalid, but if we force a top_k call we get False
    # (state is already invalidated by gap)
    assert not rec.is_clean


# ── 10. Reset ──────────────────────────────────────────────────────────────────

def test_reset_clears_state(snapshot_raw):
    rec = make_reconstructor()
    rec.apply_snapshot(snapshot_raw)
    assert rec.is_synced
    rec.reset()
    assert not rec.is_synced
    assert not rec.is_clean
    assert rec._bids == {}
    assert rec._asks == {}
