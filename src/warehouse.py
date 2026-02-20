"""
DuckDB connection factory and Parquet registration helpers.

Usage:
    from warehouse import get_conn, register_lob_snapshots

    with get_conn() as conn:
        register_lob_snapshots(conn, Path("data/processed/lob"))
        df = conn.execute("SELECT * FROM lob_snapshots LIMIT 10").fetchdf()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

logger = logging.getLogger(__name__)


@contextmanager
def get_conn(db_path: str | Path = ":memory:") -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Context manager: open a DuckDB connection and close it on exit."""
    conn = duckdb.connect(str(db_path))
    try:
        _init_extensions(conn)
        yield conn
    finally:
        conn.close()


def _init_extensions(conn: duckdb.DuckDBPyConnection) -> None:
    """Load DuckDB extensions required for Parquet + JSON."""
    conn.execute("INSTALL parquet; LOAD parquet;")


# ── View registration ──────────────────────────────────────────────────────────

def register_lob_snapshots(
    conn: duckdb.DuckDBPyConnection,
    processed_dir: Path,
) -> None:
    """
    Register a DuckDB view over all hive-partitioned LOB snapshot Parquet files.
    Pattern: {processed_dir}/snapshots/symbol=*/date=*/*.parquet
    """
    pattern = str(processed_dir / "snapshots" / "symbol=*" / "date=*" / "*.parquet")
    conn.execute(f"""
        CREATE OR REPLACE VIEW lob_snapshots AS
        SELECT * FROM read_parquet('{pattern}', hive_partitioning = true)
    """)
    logger.debug("Registered view: lob_snapshots → %s", pattern)


def register_trades(
    conn: duckdb.DuckDBPyConnection,
    processed_dir: Path,
) -> None:
    """
    Register a DuckDB view over all hive-partitioned trade Parquet files.
    Pattern: {processed_dir}/trades/symbol=*/date=*/*.parquet
    """
    pattern = str(processed_dir / "trades" / "symbol=*" / "date=*" / "*.parquet")
    conn.execute(f"""
        CREATE OR REPLACE VIEW trades AS
        SELECT * FROM read_parquet('{pattern}', hive_partitioning = true)
    """)
    logger.debug("Registered view: trades → %s", pattern)


def register_features(
    conn: duckdb.DuckDBPyConnection,
    features_dir: Path,
) -> None:
    """
    Register a DuckDB view over the feature matrix Parquet files.
    Pattern: {features_dir}/symbol=*/date=*/*.parquet
    """
    pattern = str(features_dir / "symbol=*" / "date=*" / "*.parquet")
    conn.execute(f"""
        CREATE OR REPLACE VIEW features AS
        SELECT * FROM read_parquet('{pattern}', hive_partitioning = true)
    """)
    logger.debug("Registered view: features → %s", pattern)


# ── Write helpers ──────────────────────────────────────────────────────────────

def write_parquet_partition(
    df,  # polars.DataFrame
    table: str,
    symbol: str,
    date: str,
    processed_dir: Path,
) -> Path:
    """
    Write a Polars DataFrame as a Parquet partition.
    Path: {processed_dir}/{table}/symbol={symbol}/date={date}/part-0.parquet
    Compression: snappy.
    """
    import polars as pl

    partition_dir = processed_dir / table / f"symbol={symbol}" / f"date={date}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    out_path = partition_dir / "part-0.parquet"

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected polars.DataFrame, got {type(df)}")

    df.write_parquet(out_path, compression="snappy")
    logger.info("Wrote %d rows → %s", len(df), out_path)
    return out_path
