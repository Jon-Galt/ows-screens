"""
Pure functions for refresh run history and per-run data snapshots (Phase 3d
Part 2b).

DataFrames and dicts in, DataFrames and dicts out. No SQLAlchemy, no
Streamlit, no file IO — same discipline as validate.py/transform.py/score.py/
overlap.py under Architecture Rule 1. No dependency on refresh.py or
validate.py's types either: refresh.py is documented as sitting above every
other module and never being imported by them, and importing ScreenResult
here would both violate that and create an actual import cycle
(refresh -> history -> refresh). build_run_row/build_screen_run_row
therefore take primitives only; refresh.py maps its own types to those
primitives at the call site.

`now`/timestamps are always passed in — this module never calls
datetime.now() itself, so every function here is deterministic and testable
with fixed inputs.
"""

import json
import math
import secrets
from datetime import datetime

import numpy as np
import pandas as pd


def new_run_id(now: datetime) -> str:
    """Build a lexically-sortable, collision-resistant run identifier.

    Lexical order matches chronological order (fixed-width UTC timestamp
    prefix), so `ORDER BY run_id DESC` works without parsing. The random
    hex suffix keeps two runs started in the same second distinct.

    Args:
        now: The run's start time (UTC).

    Returns:
        "YYYYMMDDTHHMMSSZ-xxxxxx", e.g. "20260901T140502Z-a1b2c3".
    """
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def _json_safe(value):
    """Normalize one cell value to something json.dumps can encode.

    Args:
        value: A single DataFrame cell value (numpy scalar, pandas scalar,
            or plain Python value).

    Returns:
        A JSON-safe Python value, with every kind of null/non-finite input
        (NaN, NaT, pd.NA, None, inf, -inf) mapped to None.
    """
    # Must run FIRST: pandas.NaT is a subclass of datetime.datetime, so a
    # Timestamp/datetime isinstance check placed before this would let
    # NaT.isoformat() ("NaT", a valid JSON string) through as a value
    # instead of null. Real timestamps are unaffected — pd.isna() on an
    # actual Timestamp is False.
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def encode_row(mapping: dict) -> str:
    """Encode one row (as a dict) to a stable, strictly-valid JSON string.

    Args:
        mapping: Column name -> cell value, e.g. one row of a DataFrame via
            `df.to_dict(orient="records")`.

    Returns:
        A JSON object string. Keys are sorted so two identical rows produce
        byte-identical output (enables run-over-run diffing later).

    Raises:
        ValueError: If a non-finite float somehow survives _json_safe's
            normalization (belt-and-braces — allow_nan=False makes
            json.dumps raise rather than silently emit the invalid bare
            token `NaN`/`Infinity`).
    """
    safe = {key: _json_safe(value) for key, value in mapping.items()}
    return json.dumps(safe, sort_keys=True, allow_nan=False)


def build_snapshot_frame(
    stored_df: pd.DataFrame, run_id: str, run_date: str, screen_id: str, stage: str
) -> pd.DataFrame:
    """Build the refresh_snapshots rows for one screen's just-written stage table.

    Args:
        stored_df: The screen's final-stage table, re-read from storage
            after writing (not the in-memory incoming DataFrame) so the
            snapshot is provably equal to what is actually stored.
        run_id: This run's identifier.
        run_date: This run's date, "YYYY-MM-DD" UTC.
        screen_id: The screen this snapshot is for.
        stage: Name of the stored table stored_df came from.

    Returns:
        One row per ticker: run_id, run_date, screen_id, ticker, stage, data
        (the full row, JSON-encoded via encode_row).
    """
    records = stored_df.to_dict(orient="records")
    tickers = stored_df["ticker"]
    return pd.DataFrame({
        "run_id": run_id,
        "run_date": run_date,
        "screen_id": screen_id,
        "ticker": tickers.values,
        "stage": stage,
        "data": [encode_row(record) for record in records],
    })


def snapshot_frame_to_stored_frame(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a stage table from its refresh_snapshots rows.

    The inverse of build_snapshot_frame's `data` encoding. Used by the
    round-trip regression lock and by any later reader of the snapshot
    dataset.

    Args:
        snapshot_df: Rows from refresh_snapshots for one (run_id, screen_id).

    Returns:
        A DataFrame with one row per snapshot row and columns in
        alphabetical order (json.dumps(..., sort_keys=True) on the write
        side means the reconstruction is not guaranteed to preserve the
        original table's column order — column order isn't semantic, so
        this isn't stored redundantly to recover it).
    """
    records = [json.loads(value) for value in snapshot_df["data"]]
    return pd.DataFrame.from_records(records)


def latest_snapshot_per_date(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve refresh_snapshots to one row per (screen_id, ticker, run_date).

    (screen_id, ticker, run_date) is NOT unique in refresh_snapshots — more
    than one run can happen on the same date (e.g. re-running after a
    corrected upload), and each run writes its own snapshot row. A consumer
    joining forward returns onto a date must resolve to a single row per
    date first, or it will double-count. This keeps, for each
    (screen_id, ticker, run_date), the row with the highest run_id — run_id's
    fixed-width UTC timestamp prefix makes lexical max equivalent to
    chronological latest, and run_id is unique, so there are no ties.

    Args:
        snapshot_df: Rows from refresh_snapshots (any number of runs/dates).

    Returns:
        One row per (screen_id, ticker, run_date), keeping the latest run.
        Empty DataFrame in, empty DataFrame out.
    """
    if len(snapshot_df) == 0:
        return snapshot_df
    sorted_df = snapshot_df.sort_values("run_id")
    return sorted_df.drop_duplicates(subset=["screen_id", "ticker", "run_date"], keep="last")


def build_run_row(
    run_id: str,
    run_date: str,
    started_at: datetime,
    finished_at: datetime,
    invocation: str,
    screen_ids: list,
    exit_code: int,
    git_sha: str,
) -> dict:
    """Build one refresh_runs row.

    Args:
        run_id: This run's identifier.
        run_date: This run's date, "YYYY-MM-DD" UTC.
        started_at: Run start time (UTC).
        finished_at: Run end time (UTC).
        invocation: The full invoked command line, as a single string.
        screen_ids: Screens requested, in run order.
        exit_code: This run's process exit code.
        git_sha: Full git SHA of the code that produced this run, or None
            if it couldn't be resolved.

    Returns:
        A dict matching refresh_runs' columns.
    """
    return {
        "run_id": run_id,
        "run_date": run_date,
        "started_at_utc": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at_utc": finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": invocation,
        "screens_requested": ",".join(screen_ids),
        "exit_code": exit_code,
        "git_sha": git_sha,
    }


def build_screen_run_row(
    run_id: str,
    screen_id: str,
    status: str,
    row_count: int,
    stage: str,
    snapshot_written: int,
    snapshot_row_count: int,
    findings: list,
    source_file_name: str,
    source_file_mtime_utc: str,
    source_file_sha256: str,
) -> dict:
    """Build one refresh_screen_runs row.

    Args:
        run_id: This run's identifier.
        screen_id: The screen this row is for.
        status: PASSED / FAILED / INCONSISTENT.
        row_count: Rows in the incoming data for this screen.
        stage: Derived final-stage table name, or None if nothing was
            written this run (a FAILED screen).
        snapshot_written: 1 if a snapshot was written for this screen this
            run, else 0.
        snapshot_row_count: Rows in the snapshot written, 0 if none.
        findings: This screen's validation/stage findings, as
            [{"check": ..., "message": ...}, ...]. Empty list when clean.
        source_file_name: Basename of the upload file, or None if prepare
            failed before a file was successfully read.
        source_file_mtime_utc: Upload file's mtime, ISO 8601 "Z", or None.
        source_file_sha256: Upload file's content hash, or None.

    Returns:
        A dict matching refresh_screen_runs' columns. findings_json is
        never None — "[]" when findings is empty.
    """
    return {
        "run_id": run_id,
        "screen_id": screen_id,
        "status": status,
        "row_count": row_count,
        "stage": stage,
        "snapshot_written": snapshot_written,
        "snapshot_row_count": snapshot_row_count,
        "findings_json": json.dumps(findings),
        "source_file_name": source_file_name,
        "source_file_mtime_utc": source_file_mtime_utc,
        "source_file_sha256": source_file_sha256,
    }
