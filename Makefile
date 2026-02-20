.PHONY: install test lint fmt typecheck record normalise build train replay clean help

SYMBOL ?= BTCUSDT
PYTHONPATH := src
export PYTHONPATH

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,viz,ml]"

# ── Quality ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ --cov=. --cov-report=term-missing

test-fast:
	pytest tests/ -x -q

lint:
	ruff check src/ tests/ scripts/

fmt:
	ruff format src/ tests/ scripts/

typecheck:
	mypy src/ --ignore-missing-imports

# ── Pipeline ───────────────────────────────────────────────────────────────────

record:
	@echo "Streaming $(SYMBOL) LOB to data/raw/lob/ — Ctrl+C to stop"
	python scripts/record.py --symbol $(SYMBOL)

normalise:
	@echo "Normalising raw NDJSON → Parquet for $(SYMBOL)"
	python scripts/normalise.py --symbol $(SYMBOL)

build:
	@echo "Building feature matrix from Parquet"
	python scripts/build_dataset.py --symbol $(SYMBOL)

train:
	@echo "Training model with walk-forward CV"
	python scripts/train.py --symbol $(SYMBOL)

train-smoke:
	@echo "Smoke-test training with minimal data (1 train day, no embargo)"
	python scripts/train.py --symbol $(SYMBOL) --min-train-days 1 --embargo-days 0

replay:
	@echo "Replaying Parquet through signal + taker sim"
	python scripts/replay.py --symbol $(SYMBOL)

# ── End-to-end pipeline ────────────────────────────────────────────────────────

pipeline: normalise build train replay
	@echo "Full pipeline complete."

# ── Utilities ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache

dirs:
	mkdir -p data/raw/lob data/processed/lob data/external docs/journal models logs

help:
	@echo ""
	@echo "  make install       Install dev dependencies"
	@echo "  make test          Run pytest with coverage"
	@echo "  make lint          Ruff linter"
	@echo "  make fmt           Ruff formatter"
	@echo ""
	@echo "  make record        Stream BTCUSDT LOB to NDJSON  (SYMBOL=ETHUSDT to override)"
	@echo "  make normalise     NDJSON → canonical Parquet"
	@echo "  make build         Parquet → feature matrix + labels"
	@echo "  make train         Walk-forward train + W&B logging"
	@echo "  make replay        Taker-only simulation over historical data"
	@echo "  make pipeline      normalise + build + train + replay"
	@echo ""
