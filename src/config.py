"""
Global configuration via pydantic-settings.
Env prefix: CHIPPER_   (e.g. CHIPPER_SYMBOL=ETHUSDT)
Supports .env file in project root.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CHIPPER_",
        env_file_encoding="utf-8",
    )

    # ── Exchange ───────────────────────────────────────────────────────────────
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    depth_levels: int = 20  # K = number of price levels stored per side

    # ── WebSocket / connection ─────────────────────────────────────────────────
    ws_base_url: str = "wss://stream.binance.com:9443/ws"
    rest_base_url: str = "https://api.binance.com/api/v3"
    ping_interval_s: float = 30.0
    ping_timeout_s: float = 10.0
    session_limit_s: float = 23 * 3600  # pre-empt 24 h Binance limit
    reconnect_base_s: float = 1.0
    reconnect_max_s: float = 60.0

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw/lob")
    processed_dir: Path = Path("data/processed/lob")
    features_dir: Path = Path("data/processed/features")
    external_dir: Path = Path("data/external")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    # ── DuckDB ─────────────────────────────────────────────────────────────────
    db_path: str = "data/chipper.duckdb"  # use ":memory:" for ad-hoc

    # ── Feature engineering ────────────────────────────────────────────────────
    imbalance_depths: list[int] = [1, 3, 5, 10]
    ofi_window_s: float = 1.0
    trade_flow_windows_s: list[float] = [1.0, 5.0, 30.0, 60.0]
    label_horizons_s: list[float] = [0.5, 1.0, 5.0]

    # ── Model / training ───────────────────────────────────────────────────────
    embargo_days: int = 1
    taker_fee_bps: float = 10.0  # 0.1% = 10 bps
    wandb_enabled: bool = False
    wandb_project: str = "chipper-lob"

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    @field_validator("symbol")
    @classmethod
    def upper_symbol(cls, v: str) -> str:
        return v.upper()

    def configure_logging(self) -> None:
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )


settings = Settings()
