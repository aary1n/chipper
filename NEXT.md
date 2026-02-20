# Next Steps — chipper

Ordered by phase. Do not start a later phase until the earlier one is working end-to-end.

---

## Phase 1 — Capture (critical path)

- [ ] **Reconstruction validation test** — the most important test in the repo.
  Replay raw NDJSON diffs, fetch a REST snapshot at a known `lastUpdateId`,
  assert the reconstructed book matches the snapshot exactly.
  If this test fails, stop and fix before doing anything else.
  Add to [tests/test_normaliser.py](tests/test_normaliser.py).

- [ ] **Record 5–7 days of BTCUSDT data.**
  Run `make record`, verify no unhandled disconnections.
  After normalising, check the clean fraction:
  ```bash
  make normalise && python -c "
  import sys; sys.path.insert(0,'src')
  from pathlib import Path
  from warehouse import get_conn, register_lob_snapshots
  with get_conn() as c:
      register_lob_snapshots(c, Path('data/processed/lob'))
      print(c.execute(\"SELECT date, pct_clean FROM (SELECT date, SUM(CASE WHEN is_clean THEN 1 ELSE 0 END)*100.0/COUNT(*) AS pct_clean FROM lob_snapshots GROUP BY date) ORDER BY date\").df())
  "
  ```
  Target: >99% `is_clean=True` per day. Flag anything below as a bug.

- [ ] **REST rate-limit handling** in `lob_recorder.py`.
  Add retry-with-backoff around `_fetch_snapshot()` to handle HTTP 429.

---

## Phase 2 — Explore

- [ ] **LOB heatmap** — implement `analysis/visualisation.lob_heatmap()`.
  Price on Y-axis, time on X-axis, colour = resting quantity (plotly).
  Explode bid/ask list columns into (timestamp, price, qty) triples.
  Run: `pip install -e ".[viz]"` then call from a notebook or script.

- [ ] **Book statistics** — complete the DuckDB queries in
  [src/analysis/trade_profile.py](src/analysis/trade_profile.py):
  - Full `book_depth_summary()` with list aggregation over top 1/5/10/20 levels.
  - `trade_arrival_rate()` — verify timezone handling (all timestamps in UTC µs).

- [ ] **Read the papers:**
  - Cont, Stoikov & Talreja (2010) — mental model for book dynamics.
  - Cont, Kukanov & Stoikov (2014) — OFI definition; read before Phase 3.
  - Cartea, Jaimungal & Penalva — chapters 1–3, 10.

---

## Phase 3 — Features

- [ ] **Full OFI implementation** in [src/features/book_features.py](src/features/book_features.py).
  Replace the first-difference stub with the proper Cont et al. formulation:
  signed changes at the best bid/ask (additions, cancellations, trades).
  Requires join with trades table for fill-side attribution.

- [ ] **Trade flow features** (signed volume) in `build_dataset.py`.
  Join snapshots with trades table on timestamp; compute net buy−sell volume
  over rolling windows (1s, 5s, 30s, 60s). Normalise: divide by rolling total.

- [ ] **Feature matrix validation** — add `test_book_features.py` cases for:
  - OFI sign matches direction of best-level qty change.
  - `realised_vol` is always non-negative.
  - Labels have no lookahead (verify shift direction).

- [ ] **Triple-barrier labelling** (optional, noise reduction).
  Label = which barrier hit first: up threshold, down threshold, or time expiry.
  See López de Prado *Advances in Financial Machine Learning*, ch 3.

---

## Phase 4 — Signals

- [ ] **Signal half-life measurement** — before training any model.
  Compute raw Pearson correlation of each feature with future microprice returns
  at 100ms, 500ms, 1s, 2s, 5s, 10s, 30s, 60s horizons.
  Plot decay curves. If half-life < pipeline latency, reconsider the horizon.

- [ ] **LightGBM model** — after linear baseline is understood.
  `pip install -e ".[ml]"`, add `train_lgbm()` to [src/models/train.py](src/models/train.py).
  Same walk-forward CV protocol as logistic baseline.

- [ ] **SHAP analysis** — feature importance ranking.
  Which features carry signal? OFI? Book imbalance? Trade flow?

- [ ] **W&B logging** — set `CHIPPER_WANDB_ENABLED=true` in `.env`.
  Log every experiment, fold, metric. Essential once you run 50+ experiments.

- [ ] **Taker-only P&L gate** — a signal only counts if it covers spread + fees.
  Binance taker fee: 10 bps (7.5 bps with BNB). If net P&L per trade ≤ 0 after
  fees, the signal is not tradeable at this latency/horizon. Fix features first.

---

## Phase 5 — Validate

- [ ] **Live signal pipeline** — run signal on live WebSocket data.
  Log: timestamp, features, prediction, confidence.
  At end of day, join predictions to actual price outcomes.
  Compare live accuracy to walk-forward backtest. Discrepancy = bug or regime shift.

- [ ] **Latency profiling** — measure actual end-to-end pipeline latency:
  `timestamp_exchange_us` of triggering book update → `timestamp_local_us` of
  signal output. Break down: WS receipt / feature compute / model inference / total.
  If total > 50 ms, identify bottleneck. If total > signal half-life, strategy is dead.

- [ ] **Paper P&L tracking** — simulate taker-only fills at signal generation time.
  Track entry price, exit price, holding period, fees, net P&L.
  Run for at least 2 weeks before considering real capital.

---

## Phase 6 — Backtest (v2, only after Phase 5 confirms edge)

- [ ] Implement `BacktestEngine.run()` in [src/backtest/engine.py](src/backtest/engine.py).
- [ ] P&L vs simulated latency plot (10ms / 50ms / 100ms / 500ms sensitivity).
- [ ] Transaction cost analysis including estimated market impact.
