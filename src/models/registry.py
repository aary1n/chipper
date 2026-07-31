"""
registry.py — versioned model serialisation.

Saves/loads models to: models/{name}/{timestamp}/
  - model.pkl      — serialised sklearn model (joblib)
  - scaler.pkl     — fitted StandardScaler
  - metadata.json  — symbol, label_col, features, AUC, training dates
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def save_model(
    model,
    scaler,
    name: str,
    metadata: dict,
    models_dir: Path = Path("models"),
) -> Path:
    """
    Save a trained model + scaler + metadata under a timestamped directory.
    Returns the directory path.
    """
    import joblib

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = models_dir / name / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"name": name, "timestamp": ts, **metadata}, f, indent=2, default=str)

    logger.info("Saved model → %s", out_dir)
    return out_dir


def load_model(model_dir: Path) -> tuple:
    """
    Load model + scaler from a versioned directory.
    Returns (model, scaler, metadata_dict).
    """
    import joblib

    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")

    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info("Loaded model from %s", model_dir)
    return model, scaler, metadata


def latest_model_dir(name: str, models_dir: Path = Path("models")) -> Path | None:
    """Return the most recent versioned directory for a model name."""
    base = Path(models_dir) / name
    if not base.exists():
        return None
    dirs = sorted(d for d in base.iterdir() if d.is_dir())
    return dirs[-1] if dirs else None
