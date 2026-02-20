# How to Run — chipper

## First-time setup

```bash
# Activate the project venv (Python 3.11)
source .venv/Scripts/activate        # Windows bash
# .venv\Scripts\activate             # Windows cmd/PowerShell

# Install core + dev dependencies
make install

# Create data directories
make dirs
```

## Pipeline — step by step

```bash
# 1. Stream live LOB data (runs until Ctrl+C, reconnects automatically)
make record                      # BTCUSDT by default
make record SYMBOL=ETHUSDT       # override symbol

# 2. Normalise raw NDJSON → canonical Parquet (hive-partitioned, snappy)
make normalise

# 3. Build feature matrix + labels from Parquet
make build

# 4. Train walk-forward CV model, log metrics
make train

# 5. Taker-only P&L simulation over historical data
make replay
```

## Full pipeline in one command

```bash
make pipeline     # normalise + build + train + replay
```

## Quality checks

```bash
make test         # pytest with coverage (no network required)
make lint         # ruff linter
make fmt          # ruff formatter (in-place)
```

## Interactive DuckDB query

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from pathlib import Path
from warehouse import get_conn, register_lob_snapshots

with get_conn() as conn:
    register_lob_snapshots(conn, Path("data/processed/lob"))
    print(conn.execute("""
        SELECT date,
               COUNT(*) AS rows,
               AVG(ask_prices[1] - bid_prices[1]) AS avg_spread,
               SUM(CASE WHEN is_clean THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_clean
        FROM lob_snapshots
        WHERE symbol = 'BTCUSDT'
        GROUP BY date ORDER BY date
    """).df())
EOF
```

## Installing optional extras

```bash
pip install -e ".[viz]"          # plotly + matplotlib for LOB heatmaps
pip install -e ".[ml]"           # lightgbm + wandb + shap (Phase 4)
pip install -e ".[dev,viz,ml]"   # everything
```

## Environment variables

Create a `.env` file in the project root (gitignored):

```ini
CHIPPER_SYMBOL=BTCUSDT
CHIPPER_DEPTH_LEVELS=20
CHIPPER_WANDB_ENABLED=false
CHIPPER_LOG_LEVEL=INFO
```

All settings have defaults — `.env` is optional.
