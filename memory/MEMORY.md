# chipper — LOB Alpha Research Engine

## Project
- Goal: predict P(microprice moves up in next N seconds) from Binance L2 LOB data
- Primary exchange: Binance Spot, symbol BTCUSDT
- Phases: 1-Capture → 2-Explore → 3-Features → 4-Signals → 5-Validate → 6-Backtest(v2)

## Stack
- Python 3.14 (user's env), src layout (`src/` on PYTHONPATH)
- pyproject.toml build-backend: `setuptools.build_meta` (NOT `setuptools.backends.legacy:build`)
- polars, duckdb, pyarrow, websockets, aiohttp, pydantic-settings, scikit-learn, click
- pytest with `pythonpath = ["src"]` in pyproject.toml

## Key paths
- `.claude/INDEX.md` — always read first in any session
- `src/ingest/lob_normaliser.py` — BookReconstructor + ApplyResult (core integrity logic)
- `src/ingest/lob_recorder.py` — LOBRecorder async WS client
- `src/features/book_features.py` — Polars lazy feature exprs
- `tests/conftest.py` — synthetic fixtures (snapshot + delta_valid_1/2, delta_gap, delta_out_of_order)

## Critical invariants (never break)
- Sequence rule: `next.U == prev.u + 1` — violation → GAP or OUT_OF_ORDER → state reset
- 3 timestamps always stored: exchange_us, local_us (+ trade_us for trades)
- Parquet: list[f64] arrays of depth=20 per side, snappy, hive-partitioned by symbol/date

## Test results
- 41/41 tests passing as of scaffold (all in-memory, no network)

## Preferences observed
- Concise code, no over-engineering
- Stub Phase 6 backtest (BacktestEngine raises NotImplementedError — intentional)
