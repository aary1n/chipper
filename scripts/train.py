"""
scripts/train.py — Walk-forward CV training entrypoint.

Usage:
    python scripts/train.py --symbol BTCUSDT
    make train SYMBOL=BTCUSDT

Phase 4: trains logistic regression baseline with purged walk-forward CV.
Logs metrics per fold. W&B logging optional (CHIPPER_WANDB_ENABLED=true).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import click
import numpy as np

from config import settings


# Default feature columns (updated once Phase 3 feature matrix is built)
DEFAULT_FEATURES = [
    "imbalance_1",
    "imbalance_3",
    "imbalance_5",
    "imbalance_10",
    "microprice",
    "spread_bps",
    "ofi_z",
    "realised_vol_1s",
    "realised_vol_5s",
    "depth_drop_5",
]


def _parse_segment(spec: str):
    """
    Parse a segment spec "NAME=START/END" with ISO-8601 UTC bounds, e.g.
    "A=2026-03-02T20:00/2026-03-06T09:00". Bounds are [start, end).
    """
    from datetime import datetime, timezone

    from models.train import Segment

    name, _, rng = spec.partition("=")
    start_s, _, end_s = rng.partition("/")
    if not (name and start_s and end_s):
        raise click.BadParameter(f"Segment spec must be NAME=START/END, got {spec!r}")

    def to_us(s: str) -> int:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)

    return Segment(name=name, start_us=to_us(start_s), end_us=to_us(end_s))


@click.command()
@click.option("--symbol", default=None)
@click.option("--features-dir", default=None, type=click.Path())
@click.option("--label-col", default="label_10rows", help="Label column name")
@click.option("--min-train-days", default=3, type=int)
@click.option("--embargo-days", default=1, type=int)
@click.option("--wandb/--no-wandb", default=None, help="Override W&B setting")
@click.option(
    "--segment", "segment_specs", multiple=True,
    help='Recording segment "NAME=ISO_START/ISO_END" (UTC, end-exclusive). '
         "Repeatable. Walk-forward runs per segment; no fold crosses a "
         "boundary. Omit to treat all data as one segment.",
)
def main(
    symbol: str | None,
    features_dir: str | None,
    label_col: str,
    min_train_days: int,
    embargo_days: int,
    wandb: bool | None,
    segment_specs: tuple[str, ...],
) -> None:
    """Train logistic regression baseline with purged walk-forward CV."""
    settings.configure_logging()
    log = logging.getLogger("train")

    sym = (symbol or settings.symbol).upper()
    feat_dir = Path(features_dir) if features_dir else settings.features_dir
    use_wandb = wandb if wandb is not None else settings.wandb_enabled
    segments = [_parse_segment(s) for s in segment_specs] or None

    log.info(
        "Walk-forward training: symbol=%s label=%s min_train=%d embargo=%d "
        "wandb=%s segments=%s",
        sym, label_col, min_train_days, embargo_days, use_wandb,
        [s.name for s in segments] if segments else "single (all data)",
    )

    from models.train import run_walk_forward

    results = run_walk_forward(
        features_dir=feat_dir,
        symbol=sym,
        feature_cols=DEFAULT_FEATURES,
        label_col=label_col,
        min_train_days=min_train_days,
        embargo_days=embargo_days,
        wandb_enabled=use_wandb,
        segments=segments,
    )

    if not results:
        log.warning("No folds completed. Check that feature data exists.")
        return

    aucs = [r.auc for r in results]
    log.info(
        "Walk-forward complete: %d folds | AUC mean=%.4f std=%.4f min=%.4f max=%.4f",
        len(results),
        np.nanmean(aucs),
        np.nanstd(aucs),
        np.nanmin(aucs),
        np.nanmax(aucs),
    )

    # Save best model
    from models.registry import save_model
    best = max(results, key=lambda r: r.auc if not np.isnan(r.auc) else -1)
    save_model(
        model=best.model,
        scaler=best.scaler,
        name="logistic_baseline",
        metadata={
            "symbol": sym,
            "label_col": label_col,
            "feature_cols": DEFAULT_FEATURES,
            "best_fold_auc": best.auc,
            "best_test_date": str(best.test_date),
            "best_segment": best.segment,
            "segments": [s for s in segment_specs] or ["all"],
            "n_folds": len(results),
        },
        models_dir=settings.models_dir,
    )


if __name__ == "__main__":
    main()
