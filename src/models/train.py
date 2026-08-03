"""
train.py — Walk-forward CV training entrypoint.

Phase 4 deliverable. Currently scaffolded with:
  - Walk-forward split generator (purged, with embargo).
  - Baseline logistic regression stub.
  - W&B logging hooks (disabled by default).

Walk-forward protocol:
  - Train on days [start, end-embargo-1].
  - Embargo: discard day [end-embargo, end-embargo+embargo_days-1].
  - Test on day [end-embargo+embargo_days].
  - Slide forward by 1 day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Generator

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Walk-forward split ─────────────────────────────────────────────────────────

@dataclass
class Segment:
    """
    A contiguous recording segment. CV folds never cross segment boundaries;
    walk-forward runs independently inside each segment. Boundary dates that
    contain data from two segments are filtered at row level by time range.
    """
    name: str
    start_us: int  # inclusive, exchange time µs
    end_us: int    # exclusive, exchange time µs

    def dates(self, all_dates: list[date]) -> list[date]:
        """Dates whose partitions can intersect [start_us, end_us)."""
        start_d = _us_to_date(self.start_us)
        end_d = _us_to_date(self.end_us - 1)
        return [d for d in all_dates if start_d <= d <= end_d]


def _us_to_date(ts_us: int) -> date:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc).date()


@dataclass
class WalkForwardSplit:
    """One fold of the walk-forward CV."""
    train_dates: list[date]
    embargo_dates: list[date]
    test_date: date


def walk_forward_splits(
    all_dates: list[date],
    min_train_days: int = 3,
    embargo_days: int = 1,
) -> Generator[WalkForwardSplit, None, None]:
    """
    Generate walk-forward splits from a sorted list of dates.

    Layout per fold:
        [train_start ... train_end] | [embargo...] | [test_date]
    """
    n = len(all_dates)
    required = min_train_days + embargo_days + 1

    if n < required:
        logger.warning(
            "Not enough dates (%d) for walk-forward (need %d).", n, required
        )
        return

    for i in range(min_train_days, n - embargo_days):
        train_dates = all_dates[:i]
        embargo = all_dates[i: i + embargo_days]
        test_date = all_dates[i + embargo_days]

        if i + embargo_days >= n:
            break

        yield WalkForwardSplit(
            train_dates=train_dates,
            embargo_dates=embargo,
            test_date=test_date,
        )


# ── Feature / label loading ────────────────────────────────────────────────────

def load_feature_matrix(
    features_dir: Path,
    symbol: str,
    dates: list[date],
    feature_cols: list[str],
    label_col: str,
    time_range_us: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load feature matrix and label vector for the given dates.
    Filters out is_clean=False rows. If time_range_us is given, rows outside
    [start_us, end_us) are excluded — this is how boundary dates shared by
    two recording segments are kept apart.
    Returns (X, y) as numpy arrays.
    """
    pattern = str(features_dir / f"symbol={symbol}" / "date=*" / "*.parquet")

    lf = (
        pl.scan_parquet(pattern, hive_partitioning=True)
        # hive `date` column is Date-typed; filter with date values, not strings
        .filter(pl.col("date").is_in(list(dates)))
        .filter(pl.col("is_clean"))
    )
    if time_range_us is not None:
        start_us, end_us = time_range_us
        lf = lf.filter(
            pl.col("timestamp_exchange_us").is_between(start_us, end_us, closed="left")
        )
    lf = lf.select(feature_cols + [label_col]).drop_nulls()
    df = lf.collect()

    if df.is_empty():
        return np.empty((0, len(feature_cols))), np.empty(0)

    X = df.select(feature_cols).to_numpy()
    y = df[label_col].to_numpy()
    return X, y


# ── Baseline model ─────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    fold: int
    test_date: date
    auc: float
    n_train: int
    n_test: int
    model: object = field(repr=False)
    scaler: object = field(repr=False)
    segment: str = ""


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fold: int,
    test_date: date,
) -> TrainResult:
    """Train logistic regression baseline, return AUC on test fold."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train_s, y_train)

    if len(np.unique(y_test)) < 2:
        logger.warning("Test set has single class on fold %d — AUC undefined.", fold)
        auc = float("nan")
    else:
        y_pred = model.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

    logger.info(
        "Fold %d | test=%s | train=%d | test=%d | AUC=%.4f",
        fold, test_date, len(y_train), len(y_test), auc,
    )
    return TrainResult(
        fold=fold,
        test_date=test_date,
        auc=auc,
        n_train=len(y_train),
        n_test=len(y_test),
        model=model,
        scaler=scaler,
    )


# ── Walk-forward runner ────────────────────────────────────────────────────────

def run_walk_forward(
    features_dir: Path,
    symbol: str,
    feature_cols: list[str],
    label_col: str = "label_10rows",
    min_train_days: int = 3,
    embargo_days: int = 1,
    wandb_enabled: bool = False,
    segments: list[Segment] | None = None,
) -> list[TrainResult]:
    """
    Full walk-forward cross-validation.

    If `segments` is given, walk-forward runs independently inside each
    segment: no fold's train/embargo/test window ever crosses a segment
    boundary, and rows on boundary dates are split by exchange timestamp.
    Returns list of TrainResult per fold.
    """
    # Discover available dates
    sym_dir = features_dir / f"symbol={symbol}"
    if not sym_dir.exists():
        logger.error("No feature data found at %s", sym_dir)
        return []

    all_dates = sorted(
        date.fromisoformat(p.name.replace("date=", ""))
        for p in sym_dir.iterdir()
        if p.is_dir() and p.name.startswith("date=")
    )

    if segments is None:
        segments = [Segment(name="all", start_us=0, end_us=2**62)]

    if wandb_enabled:
        try:
            import wandb
            wandb.init(project="chipper-lob", tags=["walk-forward", symbol])
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging.")
            wandb_enabled = False

    results: list[TrainResult] = []
    fold_i = 0
    for seg in segments:
        seg_dates = seg.dates(all_dates)
        seg_range = (seg.start_us, seg.end_us)
        logger.info(
            "Segment %r: %d candidate date(s) %s → %s",
            seg.name, len(seg_dates),
            seg_dates[0] if seg_dates else "-",
            seg_dates[-1] if seg_dates else "-",
        )
        n_seg_folds = 0
        for split in walk_forward_splits(seg_dates, min_train_days, embargo_days):
            X_train, y_train = load_feature_matrix(
                features_dir, symbol, split.train_dates, feature_cols, label_col,
                time_range_us=seg_range,
            )
            X_test, y_test = load_feature_matrix(
                features_dir, symbol, [split.test_date], feature_cols, label_col,
                time_range_us=seg_range,
            )

            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning("Empty data for fold %d — skipping.", fold_i)
                fold_i += 1
                continue

            result = train_logistic(
                X_train, y_train, X_test, y_test, fold_i, split.test_date
            )
            result.segment = seg.name
            results.append(result)
            n_seg_folds += 1

            if wandb_enabled:
                import wandb
                wandb.log({
                    "fold": fold_i,
                    "segment": seg.name,
                    "auc": result.auc,
                    "n_train": result.n_train,
                    "n_test": result.n_test,
                    "test_date": str(split.test_date),
                })
            fold_i += 1

        if n_seg_folds == 0:
            logger.warning(
                "Segment %r produced no folds (needs %d dates, has %d).",
                seg.name, min_train_days + embargo_days + 1, len(seg_dates),
            )

    if results:
        mean_auc = np.nanmean([r.auc for r in results])
        logger.info(
            "Walk-forward complete: %d folds, mean AUC=%.4f", len(results), mean_auc
        )

    if wandb_enabled:
        import wandb
        wandb.finish()

    return results
