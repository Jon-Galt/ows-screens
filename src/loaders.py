"""
Generic file-reading helpers shared by every screen's ingest path.

These three functions have no source-specific logic at all — no Bloomberg
column names, no Canary quirks — they just read a file by extension, check
column presence, and summarize what was read. They used to live in
ingest.py, but that module is the Bloomberg/quant loader (COLUMN_MAP, the
"#N/A N/A" marker, etc.), and reaching into it from curated_ingest.py for
generic utilities would make ingest.py the codebase's de facto shared IO
module while being named and structured as something source-specific. This
module is the neutral home instead.
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def read_upload(filepath: str, sheet_name: str) -> pd.DataFrame:
    """Read a single CSV or Excel file into a DataFrame.

    Args:
        filepath: Path to the CSV or Excel file.
        sheet_name: Sheet to read for Excel files (ignored for CSV).

    Returns:
        Raw DataFrame with original source column names.

    Raises:
        ValueError: If file extension is not .csv, .xlsx, or .xls.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath, dtype=str)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, dtype=str, sheet_name=sheet_name)
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


def log_summary(df: pd.DataFrame) -> None:
    """Log a summary of the ingested data."""
    logger.info("Ingested %d rows, %d columns", len(df), len(df.columns))
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.1]
    if len(high_null) > 0:
        logger.info("Columns with >10%% null rate:")
        for col, rate in high_null.items():
            logger.info("  %s: %.1f%%", col, rate * 100)
