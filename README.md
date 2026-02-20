# LOB Alpha — chipper

A limit order book microstructure research engine for Binance Spot data.

**Core question:** Given the current state of the order book, what is the probability that the mid-price moves up vs down in the next N seconds?

---

## Architecture

```
Binance WS/REST
      │
      ▼
lob_recorder.py   →  data/raw/lob/{SYMBOL}_{YYYYMMDD_HH}.ndjson
      │
      ▼
lob_normaliser.py →  data/processed/lob/snapshots/symbol=.../date=.../
                  →  data/processed/lob/trades/symbol=.../date=.../
      │
      ▼
book_features.py  →  data/processed/features/symbol=.../date=.../
      │
      ▼
train.py          →  models/{name}/{timestamp}/
      │
      ▼
replay.py         →  paper P&L + latency report
```

---

## Quickstart

```bash
# 1. Clone and install
git clone <repo>
cd chipper
python -m venv .venv && source .venv/bin/activate
make install

# 2. Create data directories
make dirs

# 3. Stream live LOB data (runs until Ctrl+C)
make record

# 4. Normalise raw NDJSON to Parquet
make normalise

# 5. Build feature matrix
make build

# 6. Train baseline model
make train

# 7. Run taker-only simulation
make replay
```

---

## End-to-End Pipeline Commands

```bash
# Stream BTCUSDT LOB (default) or another symbol
make record SYMBOL=ETHUSDT

# Full processing pipeline
make pipeline           # normalise + build + train + replay

# Individual steps
make normalise          # NDJSON → Parquet (array schema, snappy, hive-partitioned)
make build              # Parquet → feature matrix + labels (microprice sign)
make train              # Walk-forward CV, logistic regression baseline
make replay             # Taker-only sim: cross spread, deduct 10bps fee

# Quality checks
make test               # pytest (synthetic fixtures, no network)
make lint && make fmt   # ruff

# Query data interactively
python - <<'EOF'
import duckdb
from warehouse import get_conn, register_lob_snapshots
from pathlib import Path

with get_conn() as conn:
    register_lob_snapshots(conn, Path("data/processed/lob"))
    result = conn.execute("""
        SELECT date, COUNT(*) as rows, AVG(ask_prices[1] - bid_prices[1]) as avg_spread
        FROM lob_snapshots
        WHERE symbol = 'BTCUSDT'
        GROUP BY date ORDER BY date
    """).fetchdf()
    print(result)
EOF
```

---

## Key Plots (planned)

1. **LOB Heatmap** (`scripts/replay.py --plot heatmap`)
   - Price level on Y-axis, time on X-axis, colour = resting quantity.
   - Reveals liquidity walls, spoofing patterns, pre-trade book changes.

2. **Signal Correlation Decay** (`make train --plot decay`)
   - Feature correlation with future returns at 100ms, 500ms, 1s, 5s, 10s, 30s, 60s.
   - Key diagnostic: if half-life < pipeline latency, strategy is dead-on-arrival.

3. **P&L vs Simulated Latency** (`make replay --latency-sweep`)
   - Cumulative taker P&L (after fees) as a function of configurable signal-to-order delay.
   - Tests strategy robustness at 10ms, 50ms, 100ms, 500ms latency.

---

## Data Schema (brief)

**Snapshots** (one row = one book state at one timestamp):
```
timestamp_exchange_us  i64     — Binance event time (µs)
timestamp_local_us     i64     — local receipt (µs)
symbol                 str
last_update_id         i64     — Binance sequence number
bid_prices             list[f64]  — top-20, index 0 = best bid
bid_quantities         list[f64]
ask_prices             list[f64]  — top-20, index 0 = best ask
ask_quantities         list[f64]
is_clean               bool    — False if preceded by a resync gap
```

Full schema + invariants: see [.claude/DATA_CONTRACTS.md](.claude/DATA_CONTRACTS.md).

---

## Project Docs

| Doc | Purpose |
|-----|---------|
| [.claude/INDEX.md](.claude/INDEX.md) | Task routing table |
| [.claude/SYSTEM_DESIGN.md](.claude/SYSTEM_DESIGN.md) | Architecture + pipeline detail |
| [.claude/DATA_CONTRACTS.md](.claude/DATA_CONTRACTS.md) | Schemas, sequence rules, invariants |
| [.claude/CONTRIBUTING.md](.claude/CONTRIBUTING.md) | Style, logging, testing |
| [.claude/ASSUMPTIONS.md](.claude/ASSUMPTIONS.md) | Design decisions + TODO by phase |

---

## Key Papers

- Cont, Stoikov & Talreja (2010) — stochastic model for order book dynamics
- Cont, Kukanov & Stoikov (2014) — price impact of order book events; defines OFI
- Avellaneda & Stoikov (2008) — HFT in a limit order book
- Cartea, Jaimungal & Penalva — *Algorithmic and High-Frequency Trading*, ch 1–3, 10
