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
import sys
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, inspect

# Allow `python src/refresh.py` to resolve `src.*` imports even though
# running a file directly doesn't put the project root on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src import curated_ingest, ingest, rsi_ingest, score, transform
from src.config import CONFIG_PATH, ScreenTypeError, get_screen_type, load_config
from src.db import table_name
from src.loaders import UploadFileError, find_single_upload_file, read_upload, validate_columns
from src.validate import Finding, validate_screen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PASSED = "PASSED"
FAILED = "FAILED"
INCONSISTENT = "INCONSISTENT"


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
    """

    screen_id: str
    status: str
    row_count: int = 0
    findings: list = field(default_factory=list)
    dry_run: bool = False


def _prepare_short_screen(upload_dir: str) -> pd.DataFrame:
    """Reproduce ingest.ingest()'s read/validate/clean sequence for
    short_screen, without writing anything.

    Args:
        upload_dir: Directory holding short_screen's single export file.

    Returns:
        The cleaned DataFrame ingest.ingest() would write to raw_data__short_screen.
    """
    cfg = ingest.SCREEN_INGEST_CONFIGS["short_screen"]
    filepath = find_single_upload_file(upload_dir, cfg["expected_extension"])
    raw = read_upload(filepath, cfg["sheet_name"])
    validate_columns(raw, cfg["required_columns"])
    return ingest.clean_dataframe(raw, cfg["column_map"], cfg["string_columns"])


def _prepare_rsi(upload_dir: str) -> pd.DataFrame:
    """Reproduce rsi_ingest.ingest_rsi()'s read/trim/clean sequence for
    Rising Short Interest, without writing anything.

    Args:
        upload_dir: Directory holding the screen's single export file.

    Returns:
        The cleaned DataFrame ingest_rsi() would write to
        raw_data__rising_short_interest.
    """
    filepath = find_single_upload_file(upload_dir, ".xlsx")
    raw = read_upload(filepath, sheet_name=0, header_row=rsi_ingest.RSI_HEADER_ROW)
    validate_columns(raw, rsi_ingest.RSI_REQUIRED_COLUMNS)
    trimmed = rsi_ingest.trim_rsi_export(raw)
    return rsi_ingest.clean_rsi_dataframe(trimmed)


def _prepare_curated(upload_dir: str) -> pd.DataFrame:
    """Reproduce curated_ingest.ingest_curated()'s read/clean sequence for
    any of the four curated screens, without writing anything.

    Args:
        upload_dir: Directory holding this curated screen's single export file.

    Returns:
        The cleaned DataFrame ingest_curated() would write to
        curated_data__<screen_id>.
    """
    filepath = find_single_upload_file(upload_dir, ".csv")
    raw = read_upload(filepath, sheet_name=None)
    validate_columns(raw, curated_ingest.CURATED_REQUIRED_COLUMNS)
    return curated_ingest.clean_curated_dataframe(raw)


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

    try:
        incoming_df = _PREPARE_FUNCS[screen_id](upload_dir)
    except (UploadFileError, KeyError, ValueError) as exc:
        logger.warning("screen_id=%s: prepare failed: %s", screen_id, exc)
        return ScreenResult(screen_id, FAILED, findings=[Finding("prepare", str(exc))], dry_run=dry_run)

    engine = create_engine(f"sqlite:///{db_path}")
    stored_stage = "curated_data" if screen_type == "curated" else "raw_data"
    stored_table = table_name(stored_stage, screen_id)
    stored_df = pd.read_sql_table(stored_table, engine) if inspect(engine).has_table(stored_table) else None

    result = validate_screen(incoming_df, stored_df, config["refresh"])
    if not result.passed:
        logger.warning("screen_id=%s: validation failed: %s", screen_id, result.findings)
        return ScreenResult(
            screen_id, FAILED, row_count=len(incoming_df), findings=result.findings, dry_run=dry_run
        )

    if dry_run:
        return ScreenResult(screen_id, PASSED, row_count=len(incoming_df), dry_run=True)

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

    return ScreenResult(screen_id, status, row_count=len(incoming_df), findings=findings)


def refresh(
    screen_ids: list = None,
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
    dry_run: bool = False,
) -> list:
    """Refresh one or more screens, isolating each from the others' failures.

    Args:
        screen_ids: Screens to refresh. Defaults to every screen in the
            registry.
        db_path: Path to the SQLite database file.
        config_path: Path to config.yaml.
        dry_run: If True, validate and report only; write nothing.

    Returns:
        One ScreenResult per screen, in the order given.
    """
    if screen_ids is None:
        config = load_config(config_path)
        screen_ids = sorted(config["screens"].keys())

    results = []
    for screen_id in screen_ids:
        # The one broad per-screen boundary catch in this module: it is
        # what makes the Driver's "continue past a failing screen" policy
        # possible at all. refresh_one() already catches and reports the
        # specific data-failure modes it knows about (a bad upload, a
        # failed validation check, a downstream transform/score error);
        # this is the backstop for anything else so one screen's problem
        # can never take down the other five.
        try:
            result = refresh_one(screen_id, db_path=db_path, config_path=config_path, dry_run=dry_run)
        except ScreenTypeError:
            raise
        except Exception as exc:
            logger.error("screen_id=%s: unexpected failure: %s", screen_id, exc)
            result = ScreenResult(screen_id, FAILED, findings=[Finding("unexpected", str(exc))])
        results.append(result)

    return results


def _print_report(results: list) -> None:
    """Print the human-readable run report.

    A dry run's footer must not read like a real one — nothing was written,
    so "refreshed cleanly" would be a false claim of completed work. Every
    result in one invocation carries the same dry_run value (it's the
    invocation's own dry_run argument, not a per-screen outcome), so any()
    over the batch reliably says which footer applies.
    """
    dry_run = any(r.dry_run for r in results)
    print("\n" + "=" * 70)
    print("REFRESH REPORT")
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
    print("=" * 70)


def _exit_code(results: list) -> int:
    """0 if every screen PASSED, else 1."""
    return 1 if any(r.status != PASSED for r in results) else 0


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
    args = parser.parse_args(argv)

    screen_ids = [args.screen] if args.screen else known_screens
    results = refresh(screen_ids, dry_run=args.dry_run)
    _print_report(results)
    sys.exit(_exit_code(results))


if __name__ == "__main__":
    main()
