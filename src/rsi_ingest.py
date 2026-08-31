"""
Ingest the Rising Short Interest Bloomberg export into SQLite.

This is the second quant_composite screen, and the first from a Bloomberg
export whose shape differs from short_screen's: a two-row metadata
preamble, a self-describing count row ("None (82 securities)"), and a
trailing blank row plus a Bloomberg legal disclaimer. None of that exists
in short_screen's export, so this lives in its own module rather than
being folded into ingest.py's Bloomberg-specific but short-screen-shaped
cleaning path.

The export's Bloomberg identifiers ("LYV US Equity") are split on the
first space to get the ticker — a deliberate, Tom-approved divergence from
the source Excel sheet's LEFT(...,4) formula, which corrupts any ticker
that isn't exactly four characters (48 of 82 rows in the verified export).
Excel parity is the wrong target here; matching Excel would mean
reproducing a bug.
"""

import logging
import os
import re
import sys

import pandas as pd
from sqlalchemy import create_engine

# Allow direct use to resolve `src.*` imports even when the project root
# isn't already on sys.path (mirrors the other ingest modules' bootstrap).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import CONFIG_PATH, ScreenTypeError, get_screen_type, load_config
from src.db import replace_screen_rows, sync_screens_registry, table_name
from src.loaders import (
    extract_ticker,
    find_single_upload_file,
    log_summary,
    read_upload,
    validate_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The real header row is the third row of the sheet (0-indexed 2) — rows 1
# and 2 are Bloomberg metadata (EQY_FUND_CRNCY/REL_INDEX/FA_ADJUSTED, LCL).
RSI_HEADER_ROW = 2

# Bloomberg's raw column headers -> internal snake_case names. "Ticker" is
# handled separately (see extract_ticker) rather than a plain rename, since
# it needs splitting, not just renaming. Columns feeding a unit conversion
# in transform.py keep a "_raw" name so the final display name is owned
# entirely by the transform stage. Cntry Terrtry Of Inc and Tot Debt LF are
# carried through unused by any display metric, per scope.
RSI_COLUMN_MAP = {
    "Name": "name",
    "Market Cap": "market_cap_raw",
    "Cntry Terrtry Of Inc": "country_territory_of_inc",
    "Avg D Val Traded 20D:D-20": "adv_raw",
    "Shrt Int:D-1": "shrt_int_d1",
    "Shrt Int:M-3": "shrt_int_m3",
    "Shrt Int:M-6": "shrt_int_m6",
    "SI % Eqty Flt": "short_interest_pct_raw",
    "52Wk High Chg Pct": "week_52_high_chg_raw",
    "BEst Curr EV / BEst Sl BF12M": "ev_sales_raw",
    "Tot Debt LF": "tot_debt_lf",
    "Net Debt to EBITDA LF": "debt_ebitda_raw",
}

RSI_REQUIRED_COLUMNS = ["Ticker"] + list(RSI_COLUMN_MAP.keys())

# Columns that stay as strings (not coerced to numeric).
RSI_STRING_COLUMNS = {"ticker", "name", "country_territory_of_inc"}

_MAX_TICKER_LENGTH = 8  # defense in depth; the real data tops out at 5


def _extract_expected_count(value) -> int:
    """Parse the export's self-describing count row, e.g. "None (82 securities)".

    Args:
        value: The count row's first-column value.

    Returns:
        The expected number of data rows.

    Raises:
        ValueError: If value is missing or doesn't match the expected
            pattern — fails clearly rather than silently skipping the
            integrity check this row exists to provide.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"Expected a count-row string like 'None (82 securities)', got {value!r}"
        )
    match = re.search(r"\((\d+)\s+securities\)", value)
    if not match:
        raise ValueError(f"Could not parse expected row count from {value!r}")
    return int(match.group(1))


def trim_rsi_export(df: pd.DataFrame) -> pd.DataFrame:
    """Trim the count row and footer (blank row + Bloomberg disclaimer)
    from a raw RSI export, asserting the self-described row count matches
    what's actually present.

    Args:
        df: DataFrame read with header=RSI_HEADER_ROW — so df.iloc[0] is
            the count row, and everything after it is data plus footer.

    Returns:
        Exactly the real data rows, with the count row and footer removed.

    Raises:
        ValueError: If the count row is missing/unparseable (via
            _extract_expected_count), if fewer rows remain than the count
            row describes (a truncated export), or if a row beyond the
            expected count doesn't look like footer (blank or a long
            disclaimer-style string) — i.e. real data was about to be
            silently discarded.
    """
    expected = _extract_expected_count(df.iloc[0, 0])
    remainder = df.iloc[1:].reset_index(drop=True)

    if len(remainder) < expected:
        raise ValueError(
            f"Count row says {expected} securities, but only {len(remainder)} "
            f"rows remain after it — export may be truncated."
        )

    data = remainder.iloc[:expected].reset_index(drop=True)

    tail = remainder.iloc[expected:]
    for _, row in tail.iterrows():
        first_cell = row.iloc[0]
        if pd.notna(first_cell) and len(str(first_cell)) <= _MAX_TICKER_LENGTH:
            raise ValueError(
                f"Row after the expected {expected} data rows looks like real "
                f"data (first cell: {first_cell!r}), not the blank/disclaimer "
                f"footer — refusing to silently discard it."
            )

    return data


def clean_rsi_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, extract the ticker, and coerce types for a trimmed
    RSI export.

    Args:
        df: DataFrame already trimmed by trim_rsi_export (all str dtype).

    Returns:
        Cleaned DataFrame with snake_case columns, a clean ticker column,
        and numeric types where appropriate. No Bloomberg "#N/A N/A"
        handling and no comma-stripping — verified against the real
        export that neither is present in this file.
    """
    df = df.rename(columns=RSI_COLUMN_MAP)
    df["ticker"] = df["Ticker"].apply(extract_ticker)
    df = df.drop(columns=["Ticker"])

    if df["ticker"].str.len().max() > _MAX_TICKER_LENGTH:
        raise ValueError(
            "A ticker longer than expected survived trimming — likely the "
            "disclaimer row leaking through as data."
        )

    for col in df.columns:
        if col in RSI_STRING_COLUMNS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def ingest_rsi(
    screen_id: str = "rising_short_interest",
    upload_dir: str = None,
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
) -> None:
    """Run the ingestion pipeline for Rising Short Interest.

    Reads the single export file in the screen's upload folder, trims its
    preamble/count-row/footer, extracts clean tickers, coerces types, and
    writes to raw_data__<screen_id>. Also writes this screen's ticker
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
        UploadFileError: If upload_dir doesn't hold exactly one .xlsx file.
        ValueError: If the count-row assertion fails, or trimming would
            silently discard what looks like real data.
    """
    config = load_config(config_path)
    screen_type = get_screen_type(config, screen_id)
    if screen_type != "quant_composite":
        raise ScreenTypeError(
            f"ingest_rsi() only supports quant_composite screens; "
            f"{screen_id!r} is type {screen_type!r}."
        )

    if upload_dir is None:
        upload_dir = os.path.join("data", "uploads", screen_id)

    filepath = find_single_upload_file(upload_dir, ".xlsx")

    logger.info("Reading %s", filepath)
    # Sheet name is a generic Bloomberg default ("Sheet1") — read by
    # position rather than matching on an unreliable name.
    raw_df = read_upload(filepath, sheet_name=0, header_row=RSI_HEADER_ROW)
    validate_columns(raw_df, RSI_REQUIRED_COLUMNS)

    trimmed = trim_rsi_export(raw_df)
    cleaned = clean_rsi_dataframe(trimmed)
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
    ingest_rsi()
