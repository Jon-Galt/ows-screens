"""
Shared storage-layer helpers for scoping SQLite tables by screen_id.

Each screen owns its own physical tables (e.g. "raw_data__short_screen"),
named via table_name(), so screens with different column shapes never share
a table and an ordinary to_sql(if_exists="replace") is always screen-safe.
The one exception is screen_membership: a small, fixed-shape table
(screen_id, ticker) shared across all screens to support cross-screen
overlap queries, which needs the scoped replace_screen_rows() helper
instead since it must hold every screen's rows at once.

A third write pattern, added in Phase 3d Part 2b: refresh.py's run-history
and snapshot tables (refresh_runs, refresh_screen_runs, refresh_snapshots)
are append-only by design — score history can't be reconstructed once
overwritten, unlike the other tables here. append_rows() is that pattern's
helper; unlike replace_screen_rows() it never deletes anything first.
"""

import logging
import re

import pandas as pd
from sqlalchemy import Engine, inspect, text

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_identifier(value: str, label: str) -> None:
    """Guard a value against unsafe SQL-identifier interpolation.

    SQLAlchemy does not parameterize identifiers (table/column names), only
    values, so any string interpolated into a raw SQL identifier position
    must be checked here first — every call site in this module that builds
    or receives a table name goes through this.

    Args:
        value: The candidate identifier.
        label: Name of the argument, used only in the error message.

    Raises:
        ValueError: If value does not match ^[a-z][a-z0-9_]*$.
    """
    if not _IDENTIFIER_PATTERN.match(value):
        raise ValueError(
            f"Unsafe {label} for SQL identifier use: {value!r}. "
            f"Must match {_IDENTIFIER_PATTERN.pattern}"
        )


def table_name(stage: str, screen_id: str) -> str:
    """Build the per-screen physical table name for a pipeline stage.

    Args:
        stage: Pipeline stage, e.g. "raw_data", "transformed_data",
            "scored_data".
        screen_id: The screen's identifier.

    Returns:
        The table name, e.g. "raw_data__short_screen".

    Raises:
        ValueError: If screen_id does not match ^[a-z][a-z0-9_]*$. This is
            interpolated directly into a SQL identifier and SQLAlchemy does
            not parameterize identifiers, so an unvalidated screen_id would
            be a SQL-identifier-injection risk.
    """
    _validate_identifier(screen_id, "screen_id")
    return f"{stage}__{screen_id}"


def replace_screen_rows(engine, df: pd.DataFrame, table: str, screen_id: str) -> None:
    """Replace one screen's rows within a shared, fixed-shape table.

    Deletes any existing rows for screen_id in `table`, then appends df.
    Only safe for tables whose column shape is identical across every
    screen (e.g. screen_membership). Per-screen-shaped tables (raw_data,
    transformed_data, scored_data) should use table_name() plus an ordinary
    to_sql(if_exists="replace") instead — this helper is not for them.

    Args:
        engine: SQLAlchemy engine.
        df: Rows to write for this screen. Must include a screen_id column
            with every value equal to screen_id.
        table: The shared table name (not a per-screen table_name() result).
        screen_id: The screen whose rows are being replaced.

    Raises:
        ValueError: If table does not match ^[a-z][a-z0-9_]*$ — it is
            interpolated directly into a SQL identifier, same reasoning as
            table_name(). screen_id itself is passed as a bound parameter,
            not interpolated, so it doesn't need this check.
    """
    _validate_identifier(table, "table")
    if inspect(engine).has_table(table):
        with engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {table} WHERE screen_id = :sid"), {"sid": screen_id})
    df.to_sql(table, engine, if_exists="append", index=False)


def append_rows(engine_or_conn, df: pd.DataFrame, table: str) -> None:
    """Append rows to an append-only table, creating it on first use.

    Never deletes or replaces existing rows — the write pattern for
    refresh.py's run-history and snapshot tables, which must never lose a
    prior run's data. Accepts either a SQLAlchemy Engine or an open
    Connection (e.g. one held inside `with engine.begin() as conn:`), so
    callers needing several of these writes plus other statements to commit
    or roll back together can pass the same Connection to all of them.

    Args:
        engine_or_conn: SQLAlchemy Engine or Connection.
        df: Rows to append.
        table: Destination table name.

    Raises:
        ValueError: If table does not match ^[a-z][a-z0-9_]*$.
    """
    _validate_identifier(table, "table")
    df.to_sql(table, engine_or_conn, if_exists="append", index=False)


def create_index_if_not_exists(engine_or_conn, index_name: str, table: str, columns: list) -> None:
    """Create an index if it doesn't already exist, idempotently.

    Accepts either a SQLAlchemy Engine or an open Connection, same as
    append_rows(). The target table must already exist (e.g. via a prior
    append_rows() call, or explicit DDL) — this only creates the index.

    Args:
        engine_or_conn: SQLAlchemy Engine or Connection.
        index_name: Name of the index to create.
        table: Table the index is built on.
        columns: Column names the index covers, in order.

    Raises:
        ValueError: If index_name, table, or any column name does not match
            ^[a-z][a-z0-9_]*$ — all three are interpolated directly into a
            SQL identifier position, same reasoning as table_name().
    """
    _validate_identifier(index_name, "index_name")
    _validate_identifier(table, "table")
    for col in columns:
        _validate_identifier(col, "column")
    col_list = ", ".join(columns)
    stmt = text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({col_list})")
    # A bare Engine has no open transaction, so it needs its own; a
    # Connection (e.g. one held inside `with engine.begin() as conn:`) is
    # already inside one and must not open a nested one of its own —
    # Connection.begin() exists too (for savepoints), so this must check
    # the concrete type rather than hasattr(..., "begin").
    if isinstance(engine_or_conn, Engine):
        with engine_or_conn.begin() as conn:
            conn.execute(stmt)
    else:
        engine_or_conn.execute(stmt)


def sync_screens_registry(engine, config: dict) -> None:
    """Rewrite the screens registry table from config["screens"].

    Fully idempotent: truncates and reinserts on every call, since the
    registry's content is entirely derived from config.yaml with no other
    state to preserve between calls. Safe to call from any pipeline
    entrypoint (ingest, transform, or score) regardless of run order.

    Args:
        engine: SQLAlchemy engine.
        config: Parsed config.yaml dict with a top-level "screens" key
            mapping screen_id -> {"display_name": ..., "type": ..., ...}.
    """
    rows = [
        {
            "screen_id": screen_id,
            "display_name": screen_cfg["display_name"],
            "screen_type": screen_cfg["type"],
            # A quant_composite screen doesn't necessarily have a factor
            # model yet (e.g. Rising Short Interest) — this is the same
            # signal score.py's dispatch guard uses to reject scoring for
            # such a screen, stored here so app.py can branch on it too
            # without re-deriving it a second, different way.
            "has_scoring": "factor_weights" in screen_cfg,
        }
        for screen_id, screen_cfg in config["screens"].items()
    ]
    registry_df = pd.DataFrame(
        rows, columns=["screen_id", "display_name", "screen_type", "has_scoring"]
    )
    registry_df.to_sql("screens", engine, if_exists="replace", index=False)
    logger.info("Synced screens registry: %d screen(s)", len(rows))
