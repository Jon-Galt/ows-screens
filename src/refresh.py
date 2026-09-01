"""
One-command refresh across all six screens, gated by pre-write validation.

Sits above ingest.py/curated_ingest.py/rsi_ingest.py/transform.py/score.py in
the layering — it imports all five, none of them import it back. For each
screen this module:

  1. Reproduces that screen's ingest module's own read -> validate_columns ->
     clean sequence (using that module's own public functions/constants, so
     the cleaning LOGIC is never duplicated here — only the sequence of
     already-public calls is), producing the incoming cleaned DataFrame
     without writing anything.
  2. Validates that DataFrame against the screen's currently stored table
     via src.validate.validate_screen().
  3. Only if validation passes, calls the real, unmodified ingest()/
     ingest_rsi()/ingest_curated() to actually read+clean+write a second
     time. This costs one extra file read on a passing run but means the
     data that gets validated and the data that gets written are produced
     by the exact same, already-tested code path — and none of the three
     ingest modules had to change.
  4. If applicable to this screen, runs transform() and/or score() in their
     own guarded step (see REFRESH_ONE's transform/score handling below).

A screen_id with no dispatch entry is a CODE bug (every registry screen must
have one by construction) and aborts the whole run immediately. A DATA
failure — a bad/missing upload, a failed validation check, a transform/score
error after a good write — is caught per-screen so the other screens still
run, per the Driver's "continue past a failing screen" policy.
"""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, inspect, text

# Allow `python src/refresh.py` to resolve `src.*` imports even though
# running a file directly doesn't put the project root on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import curated_ingest, history, ingest, rsi_ingest, score, transform
from src.config import CONFIG_PATH, ScreenTypeError, get_screen_type, load_config
from src.db import append_rows, create_index_if_not_exists, table_name
from src.loaders import UploadFileError, file_provenance, find_single_upload_file, read_upload, validate_columns
from src.validate import Finding, validate_screen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PASSED = "PASSED"
FAILED = "FAILED"
INCONSISTENT = "INCONSISTENT"

# (index_name, table, columns) for the three history/snapshot tables' indexes.
_HISTORY_INDEXES = [
    ("idx_refresh_snapshots_screen_ticker_date", "refresh_snapshots", ["screen_id", "ticker", "run_date"]),
    ("idx_refresh_snapshots_run", "refresh_snapshots", ["run_id"]),
    ("idx_refresh_screen_runs_run", "refresh_screen_runs", ["run_id"]),
]

_CREATE_HISTORY_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS refresh_runs (
        run_id TEXT, run_date TEXT, started_at_utc TEXT, finished_at_utc TEXT,
        argv TEXT, screens_requested TEXT, exit_code INTEGER, git_sha TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS refresh_screen_runs (
        run_id TEXT, screen_id TEXT, status TEXT, row_count INTEGER, stage TEXT,
        snapshot_written INTEGER, snapshot_row_count INTEGER, findings_json TEXT,
        source_file_name TEXT, source_file_mtime_utc TEXT, source_file_sha256 TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS refresh_snapshots (
        run_id TEXT, run_date TEXT, screen_id TEXT, ticker TEXT, stage TEXT, data TEXT
    )
    """,
]


@dataclass
class ScreenResult:
    """The outcome of refreshing (or dry-running, or gating out) one screen.

    Attributes:
        screen_id: The screen this result is for.
        status: One of PASSED, FAILED, or INCONSISTENT.
        row_count: Rows in the incoming data, if it was successfully
            prepared (0 if prepare itself failed).
        findings: Validation findings (on FAILED) or a stage-failure detail
            (on INCONSISTENT). Empty on a clean PASSED.
        dry_run: True if this result came from a --dry-run — report
            annotation only, does not affect status or the exit code.
        stage: Derived final-stage table name for this screen, populated
            whenever this run actually wrote to the DB (PASSED or
            INCONSISTENT), None on FAILED (nothing written).
        source_file_name: Basename of the upload file used, or None if
            prepare failed before a file was successfully read.
        source_file_mtime_utc: Upload file's mtime, ISO 8601 "Z", or None.
        source_file_sha256: Upload file's content hash, or None.
        run_id: Set by refresh() only on a real (non-dry-run) pass, after
            all screens have been processed — carried here purely so
            _print_report can display it without its own parameter.
        snapshots_total: Cumulative refresh_snapshots row count after this
            run's writes, set alongside run_id. None on a dry run.
        db_size_bytes: screener.db file size after this run's writes, set
            alongside run_id. None on a dry run.
    """

    screen_id: str
    status: str
    row_count: int = 0
    findings: list = field(default_factory=list)
    dry_run: bool = False
    stage: str = None
    source_file_name: str = None
    source_file_mtime_utc: str = None
    source_file_sha256: str = None
    run_id: str = None
    snapshots_total: int = None
    db_size_bytes: int = None


def _prepare_short_screen(upload_dir: str) -> tuple:
    """Reproduce ingest.ingest()'s read/validate/clean sequence for
    short_screen, without writing anything.

    Args:
        upload_dir: Directory holding short_screen's single export file.

    Returns:
        (cleaned DataFrame ingest.ingest() would write to
        raw_data__short_screen, path to the upload file it was read from).
    """
    cfg = ingest.SCREEN_INGEST_CONFIGS["short_screen"]
    filepath = find_single_upload_file(upload_dir, cfg["expected_extension"])
    raw = read_upload(filepath, cfg["sheet_name"])
    validate_columns(raw, cfg["required_columns"])
    return ingest.clean_dataframe(raw, cfg["column_map"], cfg["string_columns"]), filepath


def _prepare_rsi(upload_dir: str) -> tuple:
    """Reproduce rsi_ingest.ingest_rsi()'s read/trim/clean sequence for
    Rising Short Interest, without writing anything.

    Args:
        upload_dir: Directory holding the screen's single export file.

    Returns:
        (cleaned DataFrame ingest_rsi() would write to
        raw_data__rising_short_interest, path to the upload file it was
        read from).
    """
    filepath = find_single_upload_file(upload_dir, ".xlsx")
    raw = read_upload(filepath, sheet_name=0, header_row=rsi_ingest.RSI_HEADER_ROW)
    validate_columns(raw, rsi_ingest.RSI_REQUIRED_COLUMNS)
    trimmed = rsi_ingest.trim_rsi_export(raw)
    return rsi_ingest.clean_rsi_dataframe(trimmed), filepath


def _prepare_curated(upload_dir: str) -> tuple:
    """Reproduce curated_ingest.ingest_curated()'s read/clean sequence for
    any of the four curated screens, without writing anything.

    Args:
        upload_dir: Directory holding this curated screen's single export file.

    Returns:
        (cleaned DataFrame ingest_curated() would write to
        curated_data__<screen_id>, path to the upload file it was read from).
    """
    filepath = find_single_upload_file(upload_dir, ".csv")
    raw = read_upload(filepath, sheet_name=None)
    validate_columns(raw, curated_ingest.CURATED_REQUIRED_COLUMNS)
    return curated_ingest.clean_curated_dataframe(raw), filepath


# Which "prepare" replica and which real ingest function apply to each
# screen_id. Hardcoded, same precedent as SCREEN_INGEST_CONFIGS (ingest.py)
# and SCREEN_TRANSFORM_FUNCS (transform.py): "which Python callable handles
# this screen" is a code-level fact config.yaml cannot express. What config
# CAN express — the stored-table stage, whether transform/score apply — is
# derived below from config.yaml and the existing SCREEN_TRANSFORM_FUNCS
# registry instead of being duplicated into a third hardcoded dict.
_PREPARE_FUNCS = {
    "short_screen": _prepare_short_screen,
    "rising_short_interest": _prepare_rsi,
    "cyclicals": _prepare_curated,
    "competition": _prepare_curated,
    "structural": _prepare_curated,
    "management_comp": _prepare_curated,
}

_INGEST_FUNCS = {
    "short_screen": ingest.ingest,
    "rising_short_interest": rsi_ingest.ingest_rsi,
    "cyclicals": curated_ingest.ingest_curated,
    "competition": curated_ingest.ingest_curated,
    "structural": curated_ingest.ingest_curated,
    "management_comp": curated_ingest.ingest_curated,
}


def _stage_remediation_hint(module_name: str, func_name: str, screen_id: str) -> str:
    """Build a pasteable one-liner to manually rerun a failed downstream stage."""
    return f'python -c "from src.{module_name} import {func_name}; {func_name}(\'{screen_id}\')"'


def _final_stage_table(config: dict, screen_id: str, screen_type: str) -> str:
    """Derive which stored table holds this screen's final output.

    Derived from config.yaml and the existing SCREEN_TRANSFORM_FUNCS/
    factor_weights signals — not a fourth hardcoded screen_id dict — so it
    stays correct automatically if a screen's shape changes (e.g. Rising
    Short Interest gaining a factor model later moves it to scored_data
    with no code change here).

    Args:
        config: Full parsed config.yaml dict.
        screen_id: The screen to derive a stage for.
        screen_type: This screen's type ("quant_composite" or "curated").

    Returns:
        The final-stage table name for this screen.
    """
    if screen_type == "curated":
        return table_name("curated_data", screen_id)
    if "factor_weights" in score.get_screen_config(config, screen_id):
        return table_name("scored_data", screen_id)
    if screen_id in transform.SCREEN_TRANSFORM_FUNCS:
        return table_name("transformed_data", screen_id)
    return table_name("raw_data", screen_id)


def refresh_one(
    screen_id: str,
    upload_dir: str = None,
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
    dry_run: bool = False,
) -> ScreenResult:
    """Gate, and if it passes, refresh one screen end to end.

    Args:
        screen_id: Which screen to refresh.
        upload_dir: Directory containing this screen's single export file.
            Defaults to data/uploads/<screen_id>.
        db_path: Path to the SQLite database file.
        config_path: Path to config.yaml.
        dry_run: If True, run validation and report what would happen, but
            never call the real ingest/transform/score functions.

    Returns:
        This screen's ScreenResult.

    Raises:
        ScreenTypeError: If screen_id has no registered prepare/ingest
            dispatch entry — a code bug (every registry screen must have
            one), not a data failure, so this is not caught here.
    """
    if screen_id not in _PREPARE_FUNCS or screen_id not in _INGEST_FUNCS:
        raise ScreenTypeError(
            f"refresh() has no registered ingest dispatch for {screen_id!r}. "
            f"Known: {sorted(_PREPARE_FUNCS)}"
        )

    if upload_dir is None:
        upload_dir = os.path.join("data", "uploads", screen_id)

    config = load_config(config_path)
    screen_type = get_screen_type(config, screen_id)
    stage = _final_stage_table(config, screen_id, screen_type)

    try:
        incoming_df, filepath = _PREPARE_FUNCS[screen_id](upload_dir)
    except (UploadFileError, KeyError, ValueError) as exc:
        logger.warning("screen_id=%s: prepare failed: %s", screen_id, exc)
        return ScreenResult(screen_id, FAILED, findings=[Finding("prepare", str(exc))], dry_run=dry_run)

    # Provenance is only captured once prepare has actually succeeded — a
    # prepare failure (bad/missing/extra file, missing required columns)
    # leaves all three source_file_* fields None rather than trying to
    # recover a partial path from inside the exception above.
    provenance = file_provenance(filepath)

    engine = create_engine(f"sqlite:///{db_path}")
    stored_stage = "curated_data" if screen_type == "curated" else "raw_data"
    stored_table = table_name(stored_stage, screen_id)
    stored_df = pd.read_sql_table(stored_table, engine) if inspect(engine).has_table(stored_table) else None

    result = validate_screen(incoming_df, stored_df, config["refresh"])
    if not result.passed:
        logger.warning("screen_id=%s: validation failed: %s", screen_id, result.findings)
        return ScreenResult(
            screen_id, FAILED, row_count=len(incoming_df), findings=result.findings, dry_run=dry_run,
            source_file_name=provenance["name"], source_file_mtime_utc=provenance["mtime_utc"],
            source_file_sha256=provenance["sha256"],
        )

    if dry_run:
        # Nothing gets written on a dry run, so stage stays None — the
        # "populated whenever this run wrote to the DB" rule applies here
        # too, even though prepare/provenance did succeed.
        return ScreenResult(
            screen_id, PASSED, row_count=len(incoming_df), dry_run=True,
            source_file_name=provenance["name"], source_file_mtime_utc=provenance["mtime_utc"],
            source_file_sha256=provenance["sha256"],
        )

    _INGEST_FUNCS[screen_id](screen_id, upload_dir=upload_dir, db_path=db_path, config_path=config_path)

    status = PASSED
    findings = []

    # Broad by design, not silent: the ingest write above just succeeded, so
    # a transform/score failure here means good raw data landed but a
    # downstream derived table is now stale. Catching broadly (rather than
    # naming specific exception types) is deliberate — transform()/score()
    # can fail for reasons this orchestration layer has no way to enumerate
    # in advance, and per the Driver's policy this must be recorded and
    # reported, not allowed to crash the run for the other five screens.
    if screen_id in transform.SCREEN_TRANSFORM_FUNCS:
        try:
            transform.transform(screen_id, db_path=db_path, config_path=config_path)
        except Exception as exc:
            status = INCONSISTENT
            findings.append(Finding(
                "transform",
                f"raw_data written ({len(incoming_df)} rows), but transform failed: {exc}. "
                f"transformed_data/scored_data are STALE relative to raw_data. Fix and rerun: "
                f"{_stage_remediation_hint('transform', 'transform', screen_id)}",
            ))

    has_scoring = "factor_weights" in score.get_screen_config(config, screen_id)
    if status == PASSED and has_scoring:
        try:
            score.score(screen_id, db_path=db_path, config_path=config_path)
        except Exception as exc:
            status = INCONSISTENT
            findings.append(Finding(
                "score",
                f"raw_data/transformed_data written, but score failed: {exc}. scored_data is "
                f"STALE. Fix and rerun: {_stage_remediation_hint('score', 'score', screen_id)}",
            ))

    return ScreenResult(
        screen_id, status, row_count=len(incoming_df), findings=findings, stage=stage,
        source_file_name=provenance["name"], source_file_mtime_utc=provenance["mtime_utc"],
        source_file_sha256=provenance["sha256"],
    )


def _resolve_git_sha() -> str:
    """Full git SHA of the code that produced this run, or None if it
    can't be resolved (e.g. not a git checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Could not resolve git SHA: %s", exc)
        return None


def _create_history_tables(conn) -> None:
    """Create the three run-history/snapshot tables if they don't exist yet.

    Explicit DDL rather than relying on to_sql()'s implicit table creation
    on first append: a run where every screen FAILED never appends a
    snapshot row, so refresh_snapshots would otherwise never get created,
    and create_index_if_not_exists() would then run against a table that
    doesn't exist. Table/column names here are fixed literals in this
    module, not runtime-supplied, so they don't need _validate_identifier.
    """
    for ddl in _CREATE_HISTORY_TABLES_SQL:
        conn.execute(text(ddl))


def refresh(
    screen_ids: list = None,
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
    dry_run: bool = False,
    invocation: str = None,
) -> list:
    """Refresh one or more screens, isolating each from the others' failures.

    Args:
        screen_ids: Screens to refresh. Defaults to every screen in the
            registry.
        db_path: Path to the SQLite database file.
        config_path: Path to config.yaml.
        dry_run: If True, validate and report only; write nothing — not
            even a refresh_runs/refresh_screen_runs/refresh_snapshots row.
        invocation: The full invoked command line, as a single string, for
            refresh_runs.argv. Distinct from main()'s `argv` list parameter
            (used for argparse injection in tests) — this is a
            human-readable provenance string, not a parse target.

    Returns:
        One ScreenResult per screen, in the order given. On a real
        (non-dry-run) invocation, every result also carries the run's
        run_id once history has been persisted.
    """
    if screen_ids is None:
        config = load_config(config_path)
        screen_ids = sorted(config["screens"].keys())

    started_at = datetime.now(timezone.utc)

    results = []
    for screen_id in screen_ids:
        # The one broad per-screen boundary catch in this module: it is
        # what makes the Driver's "continue past a failing screen" policy
        # possible at all. refresh_one() already catches and reports the
        # specific data-failure modes it knows about (a bad upload, a
        # failed validation check, a downstream transform/score error);
        # this is the backstop for anything else so one screen's problem
        # can never take down the other five. A ScreenTypeError (a code
        # bug — no dispatch entry for a registry screen) is deliberately
        # NOT caught: it propagates out of this function entirely, before
        # any history/snapshot persistence below, so an aborted run leaves
        # zero trace in run history rather than a partial one.
        try:
            result = refresh_one(screen_id, db_path=db_path, config_path=config_path, dry_run=dry_run)
        except ScreenTypeError:
            raise
        except Exception as exc:
            logger.error("screen_id=%s: unexpected failure: %s", screen_id, exc)
            result = ScreenResult(screen_id, FAILED, findings=[Finding("unexpected", str(exc))])
        results.append(result)

    if dry_run:
        return results

    run_id = history.new_run_id(started_at)
    run_date = started_at.strftime("%Y-%m-%d")
    for result in results:
        result.run_id = run_id

    finished_at = datetime.now(timezone.utc)
    git_sha = _resolve_git_sha()

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        _create_history_tables(conn)

        run_row = history.build_run_row(
            run_id, run_date, started_at, finished_at,
            invocation, screen_ids, _exit_code(results), git_sha,
        )
        append_rows(conn, pd.DataFrame([run_row]), "refresh_runs")

        for result in results:
            snapshot_written, snapshot_row_count = 0, 0
            if result.status == PASSED:
                stored_df = pd.read_sql_table(result.stage, conn)
                snapshot_df = history.build_snapshot_frame(
                    stored_df, run_id, run_date, result.screen_id, result.stage
                )
                append_rows(conn, snapshot_df, "refresh_snapshots")
                snapshot_written, snapshot_row_count = 1, len(snapshot_df)

            findings_payload = [{"check": f.check, "message": f.message} for f in result.findings]
            screen_run_row = history.build_screen_run_row(
                run_id, result.screen_id, result.status, result.row_count, result.stage,
                snapshot_written, snapshot_row_count, findings_payload,
                result.source_file_name, result.source_file_mtime_utc, result.source_file_sha256,
            )
            append_rows(conn, pd.DataFrame([screen_run_row]), "refresh_screen_runs")

        for index_name, table, cols in _HISTORY_INDEXES:
            create_index_if_not_exists(conn, index_name, table, cols)

        snapshots_total = conn.execute(text("SELECT COUNT(*) FROM refresh_snapshots")).scalar()

    db_size_bytes = os.path.getsize(db_path)
    for result in results:
        result.snapshots_total = snapshots_total
        result.db_size_bytes = db_size_bytes

    return results


def _print_report(results: list) -> None:
    """Print the human-readable run report.

    A dry run's footer must not read like a real one — nothing was written,
    so "refreshed cleanly" would be a false claim of completed work. Every
    result in one invocation carries the same dry_run value (it's the
    invocation's own dry_run argument, not a per-screen outcome), so any()
    over the batch reliably says which footer applies. The run_id header
    line and the snapshot-count/db-size footer line are both derived from
    whether the results actually carry a run_id (only set by refresh() on a
    real pass, after persistence) rather than a second dry_run check that
    could drift from this one.
    """
    dry_run = any(r.dry_run for r in results)
    run_id = next((r.run_id for r in results if r.run_id), None)
    print("\n" + "=" * 70)
    print("REFRESH REPORT")
    if run_id:
        print(f"Run: {run_id}")
    print("=" * 70)
    for r in results:
        label = f"{r.status} (dry-run)" if r.dry_run else r.status
        print(f"\n{r.screen_id}: {label} — {r.row_count} row(s)")
        for f in r.findings:
            print(f"  [{f.check}] {f.message}")
    print("\n" + "-" * 70)
    needs_attention = [r for r in results if r.status != PASSED]
    if needs_attention:
        verb = "would need" if dry_run else "need"
        suffix = " Nothing written (dry run)." if dry_run else ""
        print(
            f"{len(needs_attention)}/{len(results)} screen(s) {verb} attention: "
            f"{[r.screen_id for r in needs_attention]}.{suffix}"
        )
    elif dry_run:
        print(f"All {len(results)} screen(s) would refresh cleanly. Nothing written (dry run).")
    else:
        print(f"All {len(results)} screen(s) refreshed cleanly.")
    if run_id:
        snapshots_total = next((r.snapshots_total for r in results if r.snapshots_total is not None), None)
        db_size_bytes = next((r.db_size_bytes for r in results if r.db_size_bytes is not None), None)
        print(f"refresh_snapshots: {snapshots_total} row(s) total. screener.db: {db_size_bytes:,} bytes.")
    print("=" * 70)


def _exit_code(results: list) -> int:
    """0 if every screen PASSED, else 1."""
    return 1 if any(r.status != PASSED for r in results) else 0


def _print_history(n: int, db_path: str) -> None:
    """Print the last n runs, newest first.

    Args:
        n: Number of runs to print.
        db_path: Path to the SQLite database file.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    if not inspect(engine).has_table("refresh_runs"):
        print("No refresh runs recorded.")
        return

    runs = pd.read_sql_query(
        "SELECT * FROM refresh_runs ORDER BY run_id DESC LIMIT :n", engine, params={"n": n}
    )
    screen_runs = pd.read_sql_table("refresh_screen_runs", engine)

    for _, run in runs.iterrows():
        git_sha_short = run["git_sha"][:8] if pd.notna(run["git_sha"]) else "unknown"
        print(f"\n{run['run_id']}  ({run['run_date']})  exit={run['exit_code']}  git={git_sha_short}")
        for _, sr in screen_runs[screen_runs["run_id"] == run["run_id"]].iterrows():
            print(
                f"  {sr['screen_id']}: {sr['status']} — {sr['row_count']} row(s), "
                f"snapshot={sr['snapshot_row_count']} row(s)"
            )


def main(argv=None) -> None:
    config = load_config(CONFIG_PATH)
    known_screens = sorted(config["screens"].keys())

    parser = argparse.ArgumentParser(
        description="Refresh one or all screens, gated by pre-write validation."
    )
    parser.add_argument(
        "--screen", choices=known_screens, help="Refresh only this screen. Default: all screens."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run every validation check and report; write nothing."
    )
    parser.add_argument(
        "--history", nargs="?", type=int, const=10, default=None, metavar="N",
        help="Print the last N runs (default 10) instead of refreshing. Cannot combine with --screen/--dry-run.",
    )
    args = parser.parse_args(argv)

    if args.history is not None and (args.screen or args.dry_run):
        parser.error("--history cannot be combined with --screen or --dry-run.")

    if args.history is not None:
        _print_history(args.history, "data/screener.db")
        return

    invocation = " ".join(argv) if argv is not None else " ".join(sys.argv)
    screen_ids = [args.screen] if args.screen else known_screens
    results = refresh(screen_ids, dry_run=args.dry_run, invocation=invocation)
    _print_report(results)
    sys.exit(_exit_code(results))


if __name__ == "__main__":
    main()
