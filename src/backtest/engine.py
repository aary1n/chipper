"""
backtest/engine.py — event-driven backtester skeleton (Phase 6 / v2).

DO NOT IMPLEMENT until Phase 5 paper trading shows out-of-sample edge.
This file is a scaffold only.

Design:
  - Events: BookUpdate, Trade, SignalGenerated, OrderPlaced, Fill
  - Process in timestamp order (priority queue by timestamp_us)
  - Taker-only fill model: cross spread, deduct fee
  - Configurable latency: signal → order delay (for sensitivity analysis)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

logger = logging.getLogger(__name__)


# ── Event types ────────────────────────────────────────────────────────────────

class Side(Enum):
    BUY = auto()
    SELL = auto()


@dataclass(order=True)
class Event:
    timestamp_us: int
    payload: object = field(compare=False)


@dataclass
class BookUpdate:
    timestamp_us: int
    symbol: str
    bid_price: float
    ask_price: float
    bid_qty: float
    ask_qty: float
    is_clean: bool


@dataclass
class SignalGenerated:
    timestamp_us: int
    symbol: str
    prob_up: float  # P(microprice moves up) from model
    features: dict


@dataclass
class Fill:
    timestamp_us: int
    symbol: str
    side: Side
    price: float
    quantity: float
    fee_bps: float


# ── Position tracker ───────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    cost_basis: float = 0.0
    realised_pnl: float = 0.0

    def apply_fill(self, fill: Fill) -> None:
        # TODO (Phase 6): implement position update + realised P&L
        raise NotImplementedError("Position tracking not yet implemented — Phase 6.")


# ── Taker fill simulator ───────────────────────────────────────────────────────

def simulate_taker_fill(
    side: Side,
    book_update: BookUpdate,
    quantity: float,
    fee_bps: float = 10.0,
) -> Fill:
    """
    Deterministic taker fill: cross the spread, pay fee.
    BUY: execute at ask_price. SELL: execute at bid_price.
    Fee deducted from proceeds.
    """
    price = book_update.ask_price if side == Side.BUY else book_update.bid_price
    return Fill(
        timestamp_us=book_update.timestamp_us,
        symbol=book_update.symbol,
        side=side,
        price=price,
        quantity=quantity,
        fee_bps=fee_bps,
    )


# ── Engine ─────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Skeleton event-driven backtester.

    Usage (stub — not yet runnable):
        engine = BacktestEngine(symbol="BTCUSDT", fee_bps=10.0, latency_us=5000)
        engine.run(events_iter)
    """

    def __init__(
        self,
        symbol: str,
        fee_bps: float = 10.0,
        latency_us: int = 0,
        signal_threshold: float = 0.55,
    ) -> None:
        self.symbol = symbol
        self.fee_bps = fee_bps
        self.latency_us = latency_us
        self.signal_threshold = signal_threshold
        self._position = Position(symbol=symbol)
        self._fills: list[Fill] = []

    def run(self, events_iter) -> list[Fill]:
        """
        Process events in timestamp order.
        TODO (Phase 6): implement full event loop.
        """
        raise NotImplementedError(
            "BacktestEngine.run() not yet implemented.\n"
            "Build this only after Phase 5 paper trading confirms out-of-sample edge."
        )

    def summary(self) -> dict:
        """Return backtest summary statistics."""
        # TODO (Phase 6): compute total P&L, Sharpe, drawdown, etc.
        raise NotImplementedError("BacktestEngine.summary() not yet implemented.")
