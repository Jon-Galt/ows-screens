"""
Ingest the Overvalued (FASTGraphs valuation) screen's CSV export into SQLite.

This is the eighth screen and the third quant_composite one, unscored like
Rising Short Interest and Negative Expert Transcripts (no factor_weights
block in config.yaml — see those screens' comments there). Its own module
because its source shape is unlike any existing loader's: a clean single-
header CSV (no preamble/count-row/footer to trim, unlike RSI), but every
numeric column arrives as an "x"-suffixed string ("37.90x") and its ticker
identifier is colon-suffixed with a region code ("AAPL:US") rather than
Bloomberg's space-suffixed one ("LYV US Equity") that loaders.extract_ticker
already handles.

Column dropped: Region (US on all rows in the verified export — a Driver
ruling, not a data gap). Column kept but derived nowhere else: GICS
Industry, alongside GICS Sector — a PM ruling, since it's informative and
free even though the Driver didn't name it explicitly.
"""

import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

# Allow direct use to resolve `src.*` imports even when the project root
# isn't already on sys.path (mirrors the other ingest modules' bootstrap).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import CONFIG_PATH, ScreenTypeError, get_screen_type, load_config  # noqa: E402
from src.db import replace_screen_rows, sync_screens_registry, table_name  # noqa: E402
from src.loaders import find_single_upload_file, log_summary, read_upload  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The export's own header row, exactly as FASTGraphs names it. Checked as a
# SET (see validate_overvalued_header_set), not just presence-of-required,
# because a vendor renaming or adding a column must fail loudly rather than
# silently passing a presence-only check.
OVERVALUED_EXPECTED_HEADERS = {
    "Ticker",
    "Company Name",
    "Region",
    "P/E Diluted",
    "Normal P/E 5Y",
    "Normal P/E 10Y",
    "Normal P/E 15Y",
    "GICS Sector",
    "GICS Industry",
}

# Source column -> stored column. "Ticker" and "Region" are handled
# separately (split-and-drop), not by this rename map.
OVERVALUED_COLUMN_MAP = {
    "Company Name": "name",
    "GICS Sector": "sector",
    "GICS Industry": "industry",
    "P/E Diluted": "pe_diluted",
    "Normal P/E 5Y": "normal_pe_5y",
    "Normal P/E 10Y": "normal_pe_10y",
    "Normal P/E 15Y": "normal_pe_15y",
}

_X_SUFFIXED_COLUMNS = ["pe_diluted", "normal_pe_5y", "normal_pe_10y", "normal_pe_15y"]


def validate_overvalued_header_set(df: pd.DataFrame) -> None:
    """Check the export's header row is EXACTLY the expected set — not a
    subset check, so a vendor-added or vendor-renamed column is caught too.

    Args:
        df: Raw DataFrame as read from the export, original source headers.

    Raises:
        ValueError: If the header set doesn't exactly match
            OVERVALUED_EXPECTED_HEADERS, naming both any missing and any
            unexpected columns.
    """
    present = set(df.columns)
    missing = sorted(OVERVALUED_EXPECTED_HEADERS - present)
    unexpected = sorted(present - OVERVALUED_EXPECTED_HEADERS)
    if missing or unexpected:
        raise ValueError(
            "Overvalued export header set doesn't match expectations. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )


def _parse_x_suffixed(value):
    """Parse a FASTGraphs "37.90x"-style string into a float.

    Args:
        value: The raw cell value — a string ending in "x", or NaN.

    Returns:
        The float with the trailing "x" stripped, or NaN if value is NaN.

    Raises:
        ValueError: If value is a non-null string that doesn't end in "x" —
            a silent format change here is exactly how a wrong number would
            reach a screen, so this fails loudly rather than coercing to
            NaN.
    """
    if pd.isna(value):
        return value
    text = str(value)
    if not text.endswith("x"):
        raise ValueError(f"Expected an 'x'-suffixed numeric string, got {value!r}")
    return float(text[:-1])


def _split_overvalued_ticker(raw_id):
    """Split a FASTGraphs identifier ("AAPL:US") at its first colon.

    Deliberately not loaders.extract_ticker, which splits Bloomberg's
    space-suffixed identifiers ("LYV US Equity") — a different source
    format. A dotted ticker (e.g. "BRK.B:US") keeps its dot: there is no
    in-repo precedent for a BRK-class mapping, so this stores what the
    source says rather than inventing one.

    Args:
        raw_id: The raw "TICKER:REGION" identifier string.

    Returns:
        The ticker (text before the first colon). Returns the input
        unchanged if it isn't a string or contains no colon.
    """
    if not isinstance(raw_id, str):
        return raw_id
    return raw_id.split(":", 1)[0]


def clean_overvalued_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, split the ticker, drop Region, and parse the four
    "x"-suffixed numeric columns for a raw Overvalued export.

    Args:
        df: Raw DataFrame, original source column names, already validated
            by validate_overvalued_header_set.

    Returns:
        Cleaned DataFrame: ticker, name, sector, industry (text) plus
        pe_diluted/normal_pe_5y/normal_pe_10y/normal_pe_15y (float).
        Region is dropped entirely.
    """
    df = df.rename(columns=OVERVALUED_COLUMN_MAP)
    df["ticker"] = df["Ticker"].apply(_split_overvalued_ticker)
    df = df.drop(columns=["Ticker", "Region"])

    for col in _X_SUFFIXED_COLUMNS:
        df[col] = df[col].apply(_parse_x_suffixed)

    return df


def ingest_overvalued(
    screen_id: str = "overvalued_screen",
    upload_dir: str = None,
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
) -> None:
    """Run the ingestion pipeline for Overvalued.

    Reads the single export file in the screen's upload folder, validates
    its header set, splits the ticker and parses the "x"-suffixed numerics,
    and writes to raw_data__<screen_id>. Also writes this screen's ticker
    universe to screen_membership, and syncs the screens registry.

    Args:
        screen_id: Which screen to ingest (default matches config.yaml).
        upload_dir: Directory containing the single export file. Defaults
            to data/uploads/<screen_id>.
        db_path: Path to the SQLite database file.
        config_path: Path to config.yaml.

    Raises:
        ScreenTypeError: If screen_id's config.yaml type isn't
            "quant_composite".
        UploadFileError: If upload_dir doesn't hold exactly one .csv file.
        ValueError: If the header set doesn't match, or an "x"-suffixed
            column holds a value that doesn't end in "x".
    """
    config = load_config(config_path)
    screen_type = get_screen_type(config, screen_id)
    if screen_type != "quant_composite":
        raise ScreenTypeError(
            f"ingest_overvalued() only supports quant_composite screens; "
            f"{screen_id!r} is type {screen_type!r}."
        )

    if upload_dir is None:
        upload_dir = os.path.join("data", "uploads", screen_id)

    filepath = find_single_upload_file(upload_dir, ".csv")

    logger.info("Reading %s", filepath)
    raw_df = read_upload(filepath, sheet_name=None)
    validate_overvalued_header_set(raw_df)

    cleaned = clean_overvalued_dataframe(raw_df)
    log_summary(cleaned)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    sync_screens_registry(engine, config)

    raw_table = table_name("raw_data", screen_id)
    cleaned.to_sql(raw_table, engine, if_exists="replace", index=False)
    logger.info("Wrote %d rows to %s table at %s", len(cleaned), raw_table, db_path)

    membership_df = pd.DataFrame({"screen_id": screen_id, "ticker": cleaned["ticker"]})
    replace_screen_rows(engine, membership_df, "screen_membership", screen_id)
    logger.info(
        "Wrote %d rows to screen_membership for screen_id=%s", len(membership_df), screen_id
    )


if __name__ == "__main__":
    ingest_overvalued()
