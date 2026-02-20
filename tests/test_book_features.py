"""
test_book_features.py — LOB feature correctness against hand-computed values.

Uses a minimal synthetic snapshot DataFrame to verify formulas.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from features.book_features import (
    book_imbalance,
    label_direction,
    microprice,
    mid_price,
    spread_bps,
)


# ── Toy book fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def single_row_df() -> pl.DataFrame:
    """
    Single snapshot row:
      bid: [50000.0 @ 2.0, 49999.0 @ 3.0]
      ask: [50001.0 @ 1.0, 50002.0 @ 4.0]

    Hand-computed values:
      mid = (50001 + 50000) / 2 = 50000.5
      microprice = (50001 × 2 + 50000 × 1) / (2 + 1) = (100002 + 50000) / 3 = 50000.667
      spread_bps = (50001 - 50000) / 50000.5 × 10000 = 1 / 50000.5 × 10000 ≈ 0.1999 bps
      imbalance_1 = (2 - 1) / (2 + 1) = 1/3 ≈ 0.333
      imbalance_2 = (2+3 - 1+4) / (2+3 + 1+4) = (5-5)/(5+5) = 0.0
    """
    return pl.DataFrame({
        "timestamp_exchange_us": [1700000000000000],
        "timestamp_local_us": [1700000000010000],
        "symbol": ["BTCUSDT"],
        "last_update_id": [100],
        "bid_prices": [[50000.0, 49999.0, float("nan"), float("nan"), float("nan")]],
        "bid_quantities": [[2.0, 3.0, float("nan"), float("nan"), float("nan")]],
        "ask_prices": [[50001.0, 50002.0, float("nan"), float("nan"), float("nan")]],
        "ask_quantities": [[1.0, 4.0, float("nan"), float("nan"), float("nan")]],
        "is_clean": [True],
    })


@pytest.fixture
def two_row_df() -> pl.DataFrame:
    """Two rows for testing label_direction (needs forward shift)."""
    return pl.DataFrame({
        "timestamp_exchange_us": [1700000000000000, 1700000000100000],
        "timestamp_local_us": [1700000000010000, 1700000000110000],
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "last_update_id": [100, 103],
        "bid_prices": [
            [50000.0, 49999.0, float("nan"), float("nan"), float("nan")],
            [50010.0, 50009.0, float("nan"), float("nan"), float("nan")],  # price moved up
        ],
        "bid_quantities": [
            [2.0, 3.0, float("nan"), float("nan"), float("nan")],
            [2.0, 3.0, float("nan"), float("nan"), float("nan")],
        ],
        "ask_prices": [
            [50001.0, 50002.0, float("nan"), float("nan"), float("nan")],
            [50011.0, 50012.0, float("nan"), float("nan"), float("nan")],
        ],
        "ask_quantities": [
            [1.0, 4.0, float("nan"), float("nan"), float("nan")],
            [1.0, 4.0, float("nan"), float("nan"), float("nan")],
        ],
        "is_clean": [True, True],
    })


# ── Mid price ──────────────────────────────────────────────────────────────────

def test_mid_price(single_row_df):
    result = single_row_df.with_columns(mid_price().alias("mid")).select("mid")
    val = result["mid"][0]
    assert val == pytest.approx(50000.5, rel=1e-6)


# ── Microprice ────────────────────────────────────────────────────────────────

def test_microprice_formula(single_row_df):
    """
    microprice = (ask_p0 × bid_q0 + bid_p0 × ask_q0) / (bid_q0 + ask_q0)
              = (50001 × 2 + 50000 × 1) / (2 + 1)
              = (100002 + 50000) / 3
              = 150002 / 3
              ≈ 50000.667
    """
    result = single_row_df.with_columns(microprice().alias("mp")).select("mp")
    val = result["mp"][0]
    expected = (50001.0 * 2.0 + 50000.0 * 1.0) / (2.0 + 1.0)
    assert val == pytest.approx(expected, rel=1e-6)


def test_microprice_greater_than_mid_when_bid_qty_dominates():
    """When bid_qty > ask_qty at best level, microprice should be > mid."""
    df = pl.DataFrame({
        "bid_prices": [[100.0, float("nan")]],
        "bid_quantities": [[10.0, float("nan")]],  # heavy bid
        "ask_prices": [[101.0, float("nan")]],
        "ask_quantities": [[1.0, float("nan")]],   # thin ask
    })
    mp = df.with_columns(microprice().alias("mp")).select("mp")["mp"][0]
    mid = df.with_columns(mid_price().alias("mid")).select("mid")["mid"][0]
    # Heavy bid → microprice closer to ask → mp > mid
    assert mp > mid


# ── Spread ────────────────────────────────────────────────────────────────────

def test_spread_bps(single_row_df):
    """spread_bps = (ask_p0 - bid_p0) / mid × 10000 = 1 / 50000.5 × 10000."""
    result = single_row_df.with_columns(spread_bps().alias("spd")).select("spd")
    val = result["spd"][0]
    expected = 1.0 / 50000.5 * 10_000
    assert val == pytest.approx(expected, rel=1e-4)


def test_spread_bps_positive(single_row_df):
    """Spread must always be non-negative (ask >= bid)."""
    result = single_row_df.with_columns(spread_bps().alias("spd")).select("spd")
    assert result["spd"][0] >= 0.0


# ── Book imbalance ─────────────────────────────────────────────────────────────

def test_imbalance_k1(single_row_df):
    """
    imbalance_1 = (bid_q0 - ask_q0) / (bid_q0 + ask_q0) = (2 - 1) / (2 + 1) = 1/3.
    """
    result = single_row_df.with_columns(book_imbalance(1).alias("imb")).select("imb")
    val = result["imb"][0]
    assert val == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_imbalance_k2(single_row_df):
    """
    imbalance_2 = (bid_q0+q1 - ask_q0+q1) / sum
              = (2+3 - 1+4) / (2+3+1+4)
              = (5 - 5) / 10 = 0.
    NaN levels excluded by list.head(k).list.sum() treating NaN as 0 in polars.
    Actually polars sum skips NaN by default.
    """
    result = single_row_df.with_columns(book_imbalance(2).alias("imb2")).select("imb2")
    val = result["imb2"][0]
    assert val == pytest.approx(0.0, abs=1e-6)


def test_imbalance_range(single_row_df):
    """Book imbalance is bounded to [-1, +1]."""
    for k in [1, 2, 5]:
        result = single_row_df.with_columns(book_imbalance(k).alias("imb")).select("imb")
        val = result["imb"][0]
        if not math.isnan(val):
            assert -1.0 <= val <= 1.0


def test_imbalance_sign_matches_book(single_row_df):
    """When bids are heavier, imbalance should be positive."""
    # bid_q=2 > ask_q=1 at level 0
    result = single_row_df.with_columns(book_imbalance(1).alias("imb")).select("imb")
    assert result["imb"][0] > 0


# ── Label direction ────────────────────────────────────────────────────────────

def test_label_direction_up(two_row_df):
    """
    With 2 rows and horizon=1: row 0 label = sign(row 1 microprice - row 0 microprice).
    Both rows have same ask/bid qty so microprice ≈ ask_p0×2/(2+1)+bid_p0×1/(2+1).
    Row 1 has higher prices → label for row 0 should be 1 (up).
    """
    result = two_row_df.with_columns(label_direction(1).alias("label")).select("label")
    label_row0 = result["label"][0]
    assert label_row0 == 1


def test_label_direction_last_row_null(two_row_df):
    """Last row has no future → label is null."""
    result = two_row_df.with_columns(label_direction(1).alias("label")).select("label")
    label_last = result["label"][1]
    assert label_last is None
