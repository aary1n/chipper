# Project: LOB Alpha — Limit Order Book Microstructure Research Engine

> A full-stack system for ingesting, storing, analysing, and extracting
> short-horizon predictive signals from limit order book data.

---

## Why This Project

You've built sub-12ms gesture-to-input pipelines with zero-copy shared memory
and lock-free IPC. Market making firms do the exact same thing, except the
"gesture" is an order book update and the "input" is a trade. This project
lets you prove that your systems skills transfer to the domain these firms
actually care about — and that you can find alpha in it, not just pipe data
around.

---

## The Core Idea

Limit order books contain far more information than OHLCV bars. Every bid,
every ask, every cancellation, every fill tells you something about the
intentions of other market participants. Most retail quants never touch this
data. You will.

Your goal: build a research engine that can answer the question —
**"Given the current state of the order book, what is the probability that
the mid-price moves up vs down in the next N seconds?"**

That's it. One clean prediction problem. Everything else is infrastructure
to answer it well.

---

## Non-Negotiable Principles

1. **Correctness above all else.** Every byte of data must be provably
   trustworthy. Sequence gaps, reconstruction errors, and timezone
   mismatches are integrity failures, not warnings. If you can't prove
   a data segment is clean, discard it.

2. **Honest evaluation.** Taker-only fill assumptions. Purged walk-forward
   CV with embargo. Deflated Sharpe. No shortcuts that make backtests
   look better than reality.

3. **Signal before infrastructure.** Ship working predictions against live
   data before building a full event-driven backtester. A correct signal
   with paper validation beats a beautiful engine with no edge.

---

## Data Sources (All Free or Cheap)

### Primary: Binance Spot (recommended to start)
- **WebSocket** — free real-time L2 order book depth (`@depth@100ms`)
  and trade (`@trade`) streams. No API key needed for public data.
- **REST** — depth snapshots for initial sync and resync after
  disconnections.
- **Note:** Binance L2 provides *aggregate* quantity per price level,
  not individual orders. True queue position is not observable. This is
  why we use taker-only evaluation (see Backtest section).

### Later: Equities LOB data
- **LOBSTER** (lobsterdata.com) — NASDAQ LOB reconstructed from ITCH data.
  L3 (individual order) resolution. Academic access may be available
  through Imperial. Check with library or CS/EEE departments.
- **Databento** — ITCH/PITCH feeds. More expensive, production-grade.

**Start with Binance.** Free, real-time, 24/7, and you can validate
signals against live markets immediately. Port to equities later — the
concepts transfer, and showing both on your CV demonstrates breadth.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LOB Alpha                            │
│                                                             │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────┐  │
│  │ Ingestion │───▶│ Warehouse │───▶│ Feature Engineering  │  │
│  │ (WS/REST) │    │ (Parquet/ │    │ (Polars exprs over   │  │
│  │           │    │  DuckDB)  │    │  book state + trades)│  │
│  └──────────┘    └───────────┘    └──────────┬───────────┘  │
│                                               │              │
│                                   ┌───────────▼───────────┐  │
│                                   │   Signal Research     │  │
│                                   │   (ML models, W&B,    │  │
│                                   │    walk-forward CV)   │  │
│                                   └───────────┬───────────┘  │
│                                               │              │
│                                   ┌───────────▼───────────┐  │
│                                   │   Paper Trading /     │  │
│                                   │   Live Validation     │  │
│                                   └───────────────────────┘  │
│                                                              │
│                         ── v2 ──                             │
│                                                              │
│                                   ┌───────────────────────┐  │
│                                   │   Event-Driven        │  │
│                                   │   Backtest Engine      │  │
│                                   └───────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Storage Schema

### Book Snapshots — Columnar Arrays (NOT Long Rows)

Each row is one point in time. Top-K levels stored as fixed-length arrays.

```
timestamp_exchange_us : i64          — Binance event time `E` (µs)
timestamp_local_us    : i64          — local receipt time (µs)
symbol                : str
last_update_id        : i64          — Binance `u` field
bid_prices            : list[f64]    — top-K, index 0 = best bid
bid_quantities        : list[f64]    — corresponding quantities
ask_prices            : list[f64]    — top-K, index 0 = best ask
ask_quantities        : list[f64]    — corresponding quantities
is_clean              : bool         — False if preceded by a resync gap
```

K = 20 recommended. Access best bid: `col("bid_prices").list.get(0)`.

**Why arrays:** Exploding 20 levels × 2 sides into 40 rows per snapshot
inflates data 40x and makes every downstream join and aggregation
miserable. Polars and DuckDB both handle nested list columns natively.

### Trades

```
timestamp_exchange_us : i64          — Binance event time `E`
timestamp_trade_us    : i64          — Binance trade time `T`
timestamp_local_us    : i64          — local receipt time
symbol                : str
side                  : enum(buy, sell)   — taker side
price                 : f64
quantity              : f64
trade_id              : i64
```

### Partitioning

Hive-partitioned Parquet by `symbol/date`. Snappy compression.

---

## Data Volume Estimates

At 10 Hz (100ms snapshots), top-20 levels, BTC-USDT only:

| Metric                      | Estimate            |
|-----------------------------|---------------------|
| Rows per day                | ~864,000            |
| Raw size per row            | ~720 bytes          |
| Uncompressed per day        | ~620 MiB            |
| Parquet + snappy per day    | ~120–150 MiB        |
| 7 days (Phase 1 target)     | ~0.8–1.0 GiB       |
| 30 days                     | ~3.5–4.5 GiB       |

Trade stream adds ~10–20% depending on activity. Size disk and DuckDB
query expectations accordingly.

---

## WebSocket Integrity Protocol

This is boring plumbing. It's also the foundation everything else rests on.
Get it wrong and your entire dataset is garbage.

### Connection Lifecycle

1. **Initial sync.** Open WebSocket for `@depth@100ms`. Buffer incoming
   diff events. Fetch REST depth snapshot (returns `lastUpdateId`). Drop
   buffered events where `u <= lastUpdateId`. Apply the first buffered
   event satisfying `U <= lastUpdateId + 1 <= u`. Process subsequent
   events normally.

2. **Steady-state sequence continuity.** Enforce `next.U == prev.u + 1`
   on every depth update. Any violation is a hard integrity failure:
   log it, flag the interval as corrupted (`is_clean = False`), and
   trigger a full resync.

3. **Ping/pong.** Binance requires a pong response to server pings. Send
   unsolicited pings on a timer (~30s) to detect dead connections early.
   No pong response within timeout = connection dead, reconnect.

4. **24-hour session limit.** Binance terminates WebSocket connections
   after 24 hours unconditionally. Handle this as a *scheduled event*,
   not an error. Pre-emptively open a second connection, sync its state,
   switch over, close the old one. Zero-downtime rotation.

5. **Reconnection.** On any disconnect (network blip, server kick, timeout):
   reconnect with exponential backoff, execute full initial sync, and
   flag the gap interval in the dataset.

### Ordering Guarantees

Process events by **sequence/update ID**, not arrival order. TCP delivery
order at the application layer can differ from exchange emission order,
especially under load or when multiplexing streams. The update ID is the
source of truth.

### High-Load Behaviour

During liquidation cascades and major moves, Binance's depth stream can
lag or deliver bursts. These are exactly the moments where data quality
matters most. Any sequencing anomaly under load is treated identically
to steady-state violations: hard integrity failure, discard, resync, log.

---

## Timestamp Discipline

Three timestamps per event. Always store all three. Be explicit in any
latency claim about which pair you're measuring.

| Timestamp              | Source            | Meaning                           |
|------------------------|-------------------|-----------------------------------|
| `timestamp_exchange_us`| Binance `E` field | When the exchange generated the event |
| `timestamp_trade_us`   | Binance `T` field | When the underlying trade occurred (trade stream only) |
| `timestamp_local_us`   | `time.monotonic_ns()` or equivalent | When your process received the message |

**Latency measurement:** "Pipeline latency" = `timestamp_local_us` of
signal output minus `timestamp_exchange_us` of the triggering book update.
Be precise about this in any documentation or interview discussion.

---

## Directory Structure

```
src/
  config.py              — pydantic-settings: paths, exchange config, API keys
  ingest/
    lob_recorder.py      — WebSocket client with full integrity protocol
    lob_normaliser.py    — raw JSON → canonical Parquet (array schema)
    ohlcv_loader.py      — stub for supplementary OHLCV data
  features/
    book_features.py     — LOB features as composable Polars expressions
    indicators.py        — OHLCV-level indicators (from existing repo)
  models/
    train.py             — training entrypoint with W&B wrapper
    registry.py          — model serialisation to versioned paths
  analysis/
    trade_profile.py     — DuckDB queries for book statistics
    visualisation.py     — LOB heatmap and book depth rendering
  backtest/              — v2: event-driven backtester (stub for now)
    engine.py
  warehouse.py           — DuckDB connection factory (context manager)
data/
  raw/lob/               — (gitignored) raw JSON from WebSocket
  processed/lob/         — (gitignored) normalised Parquet, hive-partitioned
  external/              — (gitignored) third-party OHLCV if needed
tests/
  test_lob_recorder.py   — sequence continuity, gap detection, resync
  test_book_features.py  — feature correctness against hand-computed values
  test_normaliser.py     — round-trip: raw → normalised → reconstructed book
  conftest.py            — fixtures: sample book states, temp DuckDB
docs/
  journal/               — dated research notes, hypotheses, dead ends
```

---

## TODO — Ordered by Phase

### Phase 1: Capture the Data (WEEK 1–2)

- [ ] **Build `lob_recorder.py`.**
      - Connect to Binance WebSocket for BTC-USDT.
      - Implement the full integrity protocol: initial REST sync,
        `U <= lastUpdateId + 1 <= u` validation, steady-state sequence
        continuity (`next.U == prev.u + 1`), ping/pong keepalive,
        24-hour scheduled reconnection with zero-downtime rotation.
      - Store three timestamps per event (`E`, `T` where applicable,
        local receipt).
      - Write raw events to newline-delimited JSON, rotating hourly.
      - Flag any integrity violation interval and log it.
      - On any sequencing anomaly: discard, resync, flag.

- [ ] **Build `lob_normaliser.py`.**
      - Read raw JSON, reconstruct book state at each snapshot.
      - Output Parquet with columnar array schema (top-20 levels as
        `list[f64]` columns, not exploded rows).
      - Hive-partition by `symbol/date`, snappy compression.
      - Trades as a separate table with same partitioning.

- [ ] **Reconstruction validation test.**
      - Replay raw depth diffs from JSON.
      - At known checkpoints, fetch REST snapshot independently.
      - Assert reconstructed book matches snapshot exactly.
      - This test is the single most important test in the entire repo.
        If it fails, stop and fix before doing anything else.

- [ ] **Record 5–7 days of data.**
      - Run on your machine or a cheap VPS (Hetzner, ~€4/month).
      - Verify no unhandled disconnections or silent gaps.
      - Spot-check: count flagged `is_clean = False` intervals.
        Aim for <1% data loss over the recording period.

### Phase 2: Understand the Book (WEEK 2–3)

- [ ] **Build LOB heatmap visualisation.**
      - Price on y-axis, time on x-axis, colour = resting quantity.
      - Use plotly for interactive exploration. This is your "vol surface
        moment" — LOB heatmaps are visually striking and deeply
        informative. Watching liquidity walls appear and dissolve around
        trades builds intuition no paper can give you.
      - Consider a live version that streams from your WebSocket data
        for demos.

- [ ] **Compute book statistics via DuckDB:**
      - Bid-ask spread distribution (mean, median, p95, p99)
      - Book depth at top 1/5/10/20 levels, both sides
      - Trade arrival rate by hour (UTC)
      - Trade size distribution (log-scale — it's fat-tailed)
      - Aggregate order flow imbalance over rolling windows
      - Spread behaviour during high-volume periods vs quiet periods

- [ ] **Read foundational papers:**
      - Cont, Stoikov & Talreja (2010) — stochastic model for order book
        dynamics. Gives you the vocabulary and mental model.
      - Cont, Kukanov & Stoikov (2014) — price impact of order book events.
        Introduces OFI, the most important single feature you'll build.
      - Cartea, Jaimungal & Penalva — chapters 1–3, 10. Your microstructure
        textbook. Available via Imperial library.

### Phase 3: Feature Engineering (WEEK 3–4)

All features as composable lazy Polars expressions. All features
**normalised** — no raw absolute values that break across regimes.

- [ ] **Order flow imbalance (OFI).**
      - Cont, Kukanov & Stoikov (2014) formulation.
      - Net change in bid vs ask volume at best levels.
      - The single most predictive LOB feature in the literature.
      - Normalise: z-score against rolling window.

- [ ] **Book imbalance at multiple depths.**
      - `(bid_qty_top_k - ask_qty_top_k) / (bid_qty_top_k + ask_qty_top_k)`
      - Compute at k = 1, 3, 5, 10.
      - Already a ratio, inherently normalised.

- [ ] **Trade imbalance / signed volume.**
      - Net aggressive buy minus sell volume over rolling windows
        (1s, 5s, 30s, 60s).
      - Normalise: divide by rolling total volume, or z-score.

- [ ] **Microprice.**
      - `(ask_price × bid_qty + bid_price × ask_qty) / (bid_qty + ask_qty)`
      - Better fair-value estimate than simple mid when book is asymmetric.
      - Use as label basis (see below) in addition to as a feature.

- [ ] **Spread and depth dynamics.**
      - Rate of change of spread (widening = uncertainty).
      - Sudden depth drops (liquidity vacuum detection).
      - All as ratios or z-scores, never absolute values.

- [ ] **Microstructure volatility.**
      - Realised vol of microprice returns at 1s and 5s scales.
      - Vol-of-vol for regime detection.
      - Normalised: ratio to rolling median vol.

- [ ] **Build the feature matrix.**
      - For every snapshot, compute all features from current book state
        + trailing trade flow. No lookahead.
      - **Labels:** sign of microprice change (not raw mid — reduces
        spread-bounce noise) over {500ms, 1s, 5s} horizons.
        Consider triple-barrier labeling (de Prado) for further noise
        reduction: label = which barrier hit first (up threshold, down
        threshold, time expiry).
      - Store as partitioned Parquet. Register in DuckDB.

### Phase 4: Signal Research (WEEK 5–7)

- [ ] **Measure signal half-life FIRST.**
      - Before any model: compute raw feature correlation with future
        returns at 100ms, 500ms, 1s, 2s, 5s, 10s, 30s, 60s horizons.
      - Plot correlation decay curves per feature.
      - This tells you (a) which features carry signal and (b) whether
        the signal survives your realistic pipeline latency. If the
        half-life of your best feature is 50ms and your pipeline takes
        200ms, stop and reconsider before building models.

- [ ] **Linear baseline.**
      - Logistic regression: features → P(microprice moves up in next 1s).
      - This reveals how much signal is linearly accessible. If linear
        AUC is ~0.50, the features need work. If 0.52–0.55, you're in
        the right range for microstructure alpha.

- [ ] **Gradient-boosted trees.**
      - LightGBM on same features.
      - **Purged walk-forward CV with embargo:** train on days 1–3,
        embargo day 4 (discard entirely), test on day 5. Slide forward.
        Never mix days across train/test boundary. The embargo period
        kills autocorrelation leakage.

- [ ] **Evaluation metrics.**
      - AUC-ROC (overall discrimination).
      - Precision at top/bottom decile (are extreme predictions useful?).
      - Calibration plots (does P=0.6 mean 60% of the time?).
      - **P&L under taker-only assumptions:** go long when P(up) >
        threshold by crossing the ask, go flat by hitting the bid.
        Deduct full taker fee (Binance: 0.1%, or 0.075% with BNB).
        This is the only evaluation that matters. If the signal doesn't
        cover the spread + fees, it's not tradeable.

- [ ] **SHAP analysis.**
      - Which features carry the signal? OFI? Book imbalance? Trade flow?
      - Feature importance ranking informs what to focus on improving.

- [ ] **Log everything to W&B.**
      - Every model, every horizon, every CV fold, every hyperparameter.
      - You'll run 50+ experiments. W&B is the only way you'll remember
        which one worked and why.

### Phase 5: Paper Trading / Live Validation (WEEK 7–8)

- [ ] **Live signal generation.**
      - Run your signal pipeline on live WebSocket data.
      - Log: timestamp, book state hash, features, prediction, confidence.
      - At end of day, join predictions to actual price outcomes.
      - Compare live accuracy to backtest accuracy. Discrepancy = bug or
        regime change.

- [ ] **Latency profiling.**
      - Measure actual pipeline latency: `timestamp_exchange_us` of
        triggering book update → `timestamp_local_us` of signal output.
      - Break down: WebSocket receipt, feature computation, model
        inference, total.
      - If total > 50ms, identify bottleneck and optimise.
      - If total > signal half-life, the strategy is dead-on-arrival at
        this latency. Either optimise or target longer horizons.

- [ ] **Paper P&L tracking.**
      - Simulate taker-only execution at the moment of signal generation.
      - Track: entry price (ask for long, bid for short), exit price,
        holding period, fees, net P&L.
      - Run for at least 2 weeks before considering real capital.

### Phase 6 (v2): Event-Driven Backtest Engine (LATER)

Only build this after Phases 1–5 produce a signal with demonstrated
out-of-sample edge under taker-only paper trading.

- [ ] **Event-driven backtester.**
      - Events: book update, trade, signal generated, order placed, fill.
      - Process in timestamp order.
      - 200–400 lines of clean Python. Not NautilusTrader-level.

- [ ] **Taker-only fill model.**
      - Deterministic fills: cross the spread, pay the fee. No queue
        position simulation (Binance L2 doesn't support it).
      - If you later move to LOBSTER/L3 data, you can add queue position
        modeling then.

- [ ] **Latency simulation.**
      - Configurable delay between signal timestamp and order timestamp.
      - Test sensitivity: does alpha survive at 10ms? 50ms? 100ms? 500ms?
      - Plot P&L vs simulated latency. This single chart tells the full
        story of how latency-sensitive your strategy is.

- [ ] **Transaction cost analysis.**
      - Maker/taker fees, spread crossing cost.
      - For larger order sizes: estimate market impact from your trade
        data (how much does a 1 BTC market buy move the price?).

---

## What Makes This Project Stand Out (for Interviews)

1. **You built the data pipeline from scratch.** Not Kaggle data. Not a
   textbook dataset. You connected to a live exchange and captured your
   own order book data with provable integrity guarantees.

2. **You understand microstructure.** OFI, book imbalance, microprice,
   taker-only evaluation, signal half-life — these are the concepts
   market makers live and breathe.

3. **You can talk about latency honestly.** "My signal has a half-life of
   ~2 seconds and my pipeline latency is 15ms" is the kind of sentence
   that makes interviewers pay attention. Doubly so when you can point
   to Grapple and say "I've built latency-critical systems before, in
   a completely different domain."

4. **You measured what matters.** Not just accuracy — P&L after costs,
   signal decay, latency sensitivity. This shows you think like a
   trader, not just a data scientist.

5. **The integrity engineering.** Sequence-validated book reconstruction,
   gap detection, zero-downtime connection rotation — this is the kind
   of infrastructure work that market making firms actually need and
   most candidates have never done.

6. **It's 100% yours.** No dependency on anyone else's data, edge, or code.

---

## Key Papers

- Cont, Stoikov & Talreja (2010) — "A stochastic model for order book
  dynamics" — foundational mental model
- Cont, Kukanov & Stoikov (2014) — "The price impact of order book events"
  — OFI, the single most important feature
- Avellaneda & Stoikov (2008) — "High-frequency trading in a limit order
  book" — classic market-making model, useful context
- Cartea, Jaimungal & Penalva — "Algorithmic and High-Frequency Trading"
  — chapters 1–3, 10 — your microstructure textbook (Imperial library)

---

## Time Budget (~10 hrs/week)

| Phase     | Weeks | Focus                                             |
|-----------|-------|---------------------------------------------------|
| Capture   | 1–2   | WebSocket recorder, normaliser, integrity tests, 5–7 days data |
| Explore   | 2–3   | LOB heatmap, book statistics, reading             |
| Features  | 3–4   | LOB feature library in Polars (normalised)        |
| Signals   | 5–7   | Half-life measurement, ML models, walk-forward CV |
| Validate  | 7–8   | Paper trading, latency profiling, live accuracy   |
| Backtest  | v2    | Event-driven sim (only after signal is validated) |

8 weeks to demonstrable paper trading results at 10 hrs/week.
Lands late April — well before summer interview cycles.

---

## Anti-Goals

- ❌ Don't build a market-making bot (yet) — pure signal research first
- ❌ Don't use Kaggle LOB datasets — capture your own, it's the story
- ❌ Don't optimise latency before you have a signal worth being fast for
- ❌ Don't trade real money until paper trading matches backtest for 2+ weeks
- ❌ Don't go multi-exchange until single-exchange works end-to-end
- ❌ Don't write a custom matching engine — taker-only is honest enough for v1
- ❌ Don't model queue position on L2 data — it's not observable, don't pretend
- ❌ Don't use absolute features that break across regimes — normalise everything