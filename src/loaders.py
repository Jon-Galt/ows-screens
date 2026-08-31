"""
Generic file-reading helpers shared by every screen's ingest path.

These functions have no source-specific logic at all — no Bloomberg column
names, no Canary quirks — they just read a file by extension, check column
presence, summarize what was read, and find the single expected upload
file in a screen's folder. read_upload/validate_columns/log_summary used
to live in ingest.py, but that module is the Bloomberg/quant loader
(COLUMN_MAP, the "#N/A N/A" marker, etc.), and reaching into it from
curated_ingest.py for generic utilities would make ingest.py the
codebase's de facto shared IO module while being named and structured as
something source-specific. This module is the neutral home instead.
find_single_upload_file started out curated-specific (hardcoded to .csv)
but generalized once a second and third caller (short_screen, Rising
Short Interest) needed the same discipline with a different expected
extension.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def read_upload(filepath: str, sheet_name: str, header_row: int = 0) -> pd.DataFrame:
    """Read a single CSV or Excel file into a DataFrame.

    Args:
        filepath: Path to the CSV or Excel file.
        sheet_name: Sheet to read for Excel files (ignored for CSV).
        header_row: 0-indexed row containing column headers. Defaults to 0
            (the first row), which is correct for every export this
            codebase has seen except Rising Short Interest's, which has a
            two-row metadata preamble above its real header row. 0 is a
            safe default here (unlike sheet_name/column_map, which have no
            sensible default) because "no preamble" is genuinely the
            normal case, not a Short-Screen-shaped assumption.

    Returns:
        Raw DataFrame with original source column names.

    Raises:
        ValueError: If file extension is not .csv, .xlsx, or .xls.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath, dtype=str, header=header_row)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, dtype=str, sheet_name=sheet_name, header=header_row)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Expected .csv, .xlsx, or .xls")


def validate_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Check that all required columns are present.

    Args:
        df: DataFrame with original source column names.
        required_columns: Column names that must all be present.

    Raises:
        KeyError: If any required columns are missing, with the list of missing names.
    """
    present = set(df.columns)
    missing = [c for c in required_columns if c not in present]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} required column(s) in upload: {missing}"
        )


def extract_ticker(raw_id):
    """Split a Bloomberg identifier ("LYV US Equity") into its ticker.

    Splitting on the first space is correct at any ticker length, unlike
    a fixed-width LEFT(...,N) formula (the source Excel sheet's approach
    for Rising Short Interest, which corrupted any ticker that wasn't
    exactly four characters). Bloomberg-identifier splitting isn't
    specific to any one screen, so this lives here rather than in a
    per-screen ingest module.

    Args:
        raw_id: The raw Bloomberg identifier string.

    Returns:
        The ticker (text before the first space). Returns the input
        unchanged if it isn't a string or contains no space — callers are
        expected to have already trimmed non-data rows before this point.
    """
    if not isinstance(raw_id, str):
        return raw_id
    return raw_id.split(" ", 1)[0]


def log_summary(df: pd.DataFrame) -> None:
    """Log a summary of the ingested data."""
    logger.info("Ingested %d rows, %d columns", len(df), len(df.columns))
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.1]
    if len(high_null) > 0:
        logger.info("Columns with >10%% null rate:")
        for col, rate in high_null.items():
            logger.info("  %s: %.1f%%", col, rate * 100)


class UploadFileError(ValueError):
    """Raised when a screen's upload folder doesn't hold exactly one file
    of the expected type.

    Every screen so far has no way to tell itself apart from another
    screen of the same shape by file contents alone (curated screens share
    an identical schema across all four; Bloomberg exports carry no
    screen-identifying column either), so screen identity depends entirely
    on exactly one correct file sitting in exactly one correct folder.
    This covers every way that can go wrong: no file, more than one file,
    or a file that isn't the expected type.
    """


def find_single_upload_file(upload_dir: str, expected_extension: str) -> str:
    """Find the single expected upload file in a screen's upload folder.

    Args:
        upload_dir: Directory expected to hold exactly one export file.
        expected_extension: The required extension for that one file,
            e.g. ".csv" or ".xlsx".

    Returns:
        The full path to that one file.

    Raises:
        UploadFileError: If zero files are found, more than one file is
            found (named in the message — this still checks for ANY of
            .csv/.xlsx/.xls, regardless of expected_extension, so a stray
            file of the wrong kind is caught here rather than silently
            ignored), or the single file found doesn't have
            expected_extension.
    """
    candidates = sorted(
        f for f in os.listdir(upload_dir)
        if f.lower().endswith((".csv", ".xlsx", ".xls"))
    )
    if not candidates:
        raise UploadFileError(f"No export file found in {upload_dir}")
    if len(candidates) > 1:
        raise UploadFileError(
            f"Expected exactly one export file in {upload_dir}, found "
            f"{len(candidates)}: {candidates}. This screen has no way to "
            f"tell which export belongs to which screen from file "
            f"contents alone — remove all but the intended file."
        )
    (filename,) = candidates
    if not filename.lower().endswith(expected_extension):
        raise UploadFileError(
            f"{upload_dir} expects a {expected_extension} export; found "
            f"{filename!r} instead."
        )
    return os.path.join(upload_dir, filename)
