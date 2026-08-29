"""
Shared storage-layer helpers for scoping SQLite tables by screen_id.

Each screen owns its own physical tables (e.g. "raw_data__short_screen"),
named via table_name(), so screens with different column shapes never share
a table and an ordinary to_sql(if_exists="replace") is always screen-safe.
The one exception is screen_membership: a small, fixed-shape table
(screen_id, ticker) shared across all screens to support cross-screen
overlap queries, which needs the scoped replace_screen_rows() helper
instead since it must hold every screen's rows at once.
"""

import logging
import re

import pandas as pd
from sqlalchemy import inspect, text

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
        }
        for screen_id, screen_cfg in config["screens"].items()
    ]
    registry_df = pd.DataFrame(rows, columns=["screen_id", "display_name", "screen_type"])
    registry_df.to_sql("screens", engine, if_exists="replace", index=False)
    logger.info("Synced screens registry: %d screen(s)", len(rows))
