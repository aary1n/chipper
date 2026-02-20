"""
conftest.py — shared pytest fixtures.

Synthetic LOB fixtures (no network calls):
  - snapshot_raw: REST-style snapshot (lastUpdateId=100, 3 bid/ask levels)
  - delta_valid_1: continuous from snapshot (U=101, u=103) — modifies best bid/ask qty
  - delta_valid_2: continuous from delta_1 (U=104, u=106) — adds new levels
  - delta_gap: sequence gap after delta_valid_2 (U=110, expected U=107) → GAP
  - delta_out_of_order: stale event after delta_valid_2 (U=104, expected U=107) → OUT_OF_ORDER

Sequence flow for clean path:
    snapshot(lastUpdateId=100) → delta_valid_1(U=101,u=103) → delta_valid_2(U=104,u=106)

Sequence flow for gap:
    ... delta_valid_2(u=106) → delta_gap(U=110) [GAP: expected U=107]

Sequence flow for OOO:
    ... delta_valid_2(u=106) → delta_out_of_order(U=104) [OOO: 104 < 107]
"""

import pytest


# ── Raw event fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def snapshot_raw() -> dict:
    """Binance REST depth snapshot format."""
    return {
        "lastUpdateId": 100,
        "bids": [
            ["50000.00", "1.000"],
            ["49999.00", "2.000"],
            ["49998.00", "0.500"],
        ],
        "asks": [
            ["50001.00", "1.500"],
            ["50002.00", "0.800"],
            ["50003.00", "0.300"],
        ],
    }


@pytest.fixture
def delta_valid_1() -> dict:
    """
    Valid depth update: continues from snapshot lastUpdateId=100.
    U=101, u=103 — satisfies U <= 100+1 <= u (101 <= 101 <= 103).
    Modifies best bid qty and best ask qty.
    """
    return {
        "e": "depthUpdate",
        "E": 1700000000100,   # ms
        "s": "BTCUSDT",
        "U": 101,
        "u": 103,
        "b": [["50000.00", "1.500"]],  # update best bid qty 1.0 → 1.5
        "a": [["50001.00", "1.000"]],  # update best ask qty 1.5 → 1.0
        "timestamp_local_us": 1700000000100000,
    }


@pytest.fixture
def delta_valid_2() -> dict:
    """
    Valid depth update: continues from delta_valid_1 (prev u=103).
    U=104, u=106 — satisfies next.U == prev.u + 1 (104 == 103+1).
    Adds new bid and ask levels.
    """
    return {
        "e": "depthUpdate",
        "E": 1700000000200,
        "s": "BTCUSDT",
        "U": 104,
        "u": 106,
        "b": [["49997.00", "3.000"]],  # new bid level
        "a": [["50004.00", "0.200"]],  # new ask level
        "timestamp_local_us": 1700000000200000,
    }


@pytest.fixture
def delta_gap() -> dict:
    """
    Gap event: after delta_valid_2 (u=106), expected U=107 but got U=110.
    Triggers ApplyResult.GAP → corruption flagged, state invalidated.
    """
    return {
        "e": "depthUpdate",
        "E": 1700000000300,
        "s": "BTCUSDT",
        "U": 110,   # EXPECTED: 107; GAP of 3
        "u": 115,
        "b": [["49996.00", "5.000"]],
        "a": [],
        "timestamp_local_us": 1700000000300000,
    }


@pytest.fixture
def delta_out_of_order() -> dict:
    """
    Out-of-order event: after delta_valid_2 (u=106), expected U=107 but got U=104.
    Triggers ApplyResult.OUT_OF_ORDER → state invalidated.
    """
    return {
        "e": "depthUpdate",
        "E": 1700000000250,
        "s": "BTCUSDT",
        "U": 104,   # EXPECTED: 107; stale / OOO
        "u": 106,
        "b": [],
        "a": [["50005.00", "0.100"]],
        "timestamp_local_us": 1700000000250000,
    }


@pytest.fixture
def resync_snapshot() -> dict:
    """A fresh REST snapshot used after a resync."""
    return {
        "lastUpdateId": 200,
        "bids": [["50100.00", "2.000"], ["50099.00", "1.000"]],
        "asks": [["50101.00", "1.800"], ["50102.00", "0.600"]],
    }


@pytest.fixture
def delta_post_resync() -> dict:
    """Valid delta continuing from resync_snapshot (lastUpdateId=200)."""
    return {
        "e": "depthUpdate",
        "E": 1700000001000,
        "s": "BTCUSDT",
        "U": 201,
        "u": 205,
        "b": [["50100.00", "2.500"]],
        "a": [["50101.00", "1.200"]],
        "timestamp_local_us": 1700000001000000,
    }
