"""
Ingest Canary curated-screen exports into SQLite, scoped by screen.

Shared by all four curated screens (Cyclicals, Competition, Structural,
Management Comp) since they share an identical 11-column schema — verified
against a real standalone export, not assumed. Handles Canary's
quote-wrapped numeric strings, the packed `scores` field, and the unit
conversions needed to match this codebase's existing storage conventions
(Architecture Rule 2: percentages stored as decimals).

Curated screens have no column identifying which screen an export belongs
to, and all four share the same schema, so a validate_columns-style check
is structurally incapable of catching a misfiled export. Screen identity
depends entirely on exactly one correct file sitting in exactly one
correct folder — see _find_single_upload_file, which makes any deviation
from that loud rather than silently concatenating or misreading files.
"""

import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

# Allow `python -c "from src.curated_ingest import ..."`-style direct use to
# resolve `src.*` imports even when the project root isn't already on
# sys.path (mirrors ingest.py/transform.py/score.py's own bootstrap).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import CONFIG_PATH, ScreenTypeError, get_screen_type, load_config
from src.db import replace_screen_rows, sync_screens_registry, table_name
from src.loaders import read_upload, validate_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class CuratedUploadError(ValueError):
    """Raised when a curated screen's upload folder doesn't hold exactly
    one usable .csv export.

    Curated screens can't be told apart by file contents (identical schema
    across all four), so the folder itself is the only source of screen
    identity. This exception covers every way that can go wrong: no file,
    more than one file, or a file that isn't .csv.
    """


# Canary's raw export column headers -> internal snake_case names.
# Verified against a real standalone export (canary-data-screen-export
# (19).csv, a Structural screen export, 135 rows) — the raw headers are
# already exactly these snake_case names, in this order. Only ticker_symbol
# is renamed, for consistency with screen_membership and short_screen,
# which both use "ticker".
CURATED_COLUMN_MAP = {
    "daily_traded_value": "daily_traded_value",
    "exchange_symbol": "exchange_symbol",
    "locations": "locations",
    "market_cap": "market_cap",
    "name": "name",
    "sector": "sector",
    "stock_performance": "stock_performance",
    "ticker_symbol": "ticker",
    "rationale": "rationale",
    "scores": "scores",
    "valuation_ev_revenue_ntm_percentile": "valuation_ev_revenue_ntm_percentile",
}

CURATED_REQUIRED_COLUMNS = list(CURATED_COLUMN_MAP.keys())

# Columns that stay as strings (not coerced to numeric).
CURATED_STRING_COLUMNS = {
    "ticker", "name", "sector", "exchange_symbol", "locations", "rationale", "scores",
}

# Columns that arrive as quote-wrapped numeric strings, e.g. '"76122.023693"'
# (verified against the real export — every one of these, every row).
CURATED_QUOTE_WRAPPED_NUMERIC_COLUMNS = {"daily_traded_value", "market_cap", "stock_performance"}

# Numeric columns that are NOT quote-wrapped.
CURATED_PLAIN_NUMERIC_COLUMNS = {"valuation_ev_revenue_ntm_percentile"}

_SCORE_LABELS = {
    "accounting and disclosure": "score_accounting_and_disclosure",
    "fraud": "score_fraud",
    "insider": "score_insider",
}


def strip_quoted_numeric(series: pd.Series) -> pd.Series:
    """Strip embedded double-quote characters, then coerce to numeric.

    Canary exports wrap some numeric fields as quote-wrapped strings, e.g.
    a cell whose value is the 3-character-quoted string 36748675276.212273
    — the field's literal string content includes the quote characters,
    not CSV delimiter quoting. Excel's reference formula strips these with
    VALUE(SUBSTITUTE(cell, CHAR(34), "")) — this mirrors that exactly.
    Missing this step leaves the quotes in place and pd.to_numeric
    silently produces NaN for every affected value.

    Args:
        series: Raw string column, possibly quote-wrapped.

    Returns:
        Numeric series; NaN for anything that still isn't a valid number
        after stripping.
    """
    stripped = series.astype(str).str.replace('"', "", regex=False)
    return pd.to_numeric(stripped, errors="coerce")


def parse_scores(raw) -> tuple:
    """Parse Canary's packed `scores` string into its three components.

    Expected format: "Accounting And Disclosure: NN | Fraud: NN | Insider: NN"
    Matches by label (case/whitespace-normalized), not position, so it's
    robust to key reordering, not merely tolerant of the documented shape.

    Args:
        raw: The packed scores string (or any value — non-strings are
            treated as fully malformed).

    Returns:
        (accounting_and_disclosure, fraud, insider) as floats. A missing
        label, a non-numeric value, or a fully malformed string all
        degrade to NaN for the affected value(s) rather than raising —
        this string arrives from an external export, and malformed input
        should degrade, not crash the pipeline.
    """
    values = {}
    if isinstance(raw, str):
        for part in raw.split("|"):
            if ":" not in part:
                continue
            label, _, value = part.partition(":")
            values[label.strip().lower()] = value.strip()

    def _get(label):
        return pd.to_numeric(values.get(label), errors="coerce")

    return (
        _get("accounting and disclosure"),
        _get("fraud"),
        _get("insider"),
    )


def clean_curated_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, coerce, unit-convert, and score-parse a curated screen's
    raw export.

    Args:
        df: DataFrame with original Canary column names (all str dtype).

    Returns:
        Cleaned DataFrame: snake_case columns, ticker/name/sector/etc. as
        strings, market_cap/daily_traded_value/stock_performance/
        valuation_ev_revenue_ntm_percentile as floats in this codebase's
        storage units, plus three parsed score_* columns alongside the
        retained raw `scores` string.
    """
    df = df.rename(columns=CURATED_COLUMN_MAP)

    for col in CURATED_QUOTE_WRAPPED_NUMERIC_COLUMNS:
        df[col] = strip_quoted_numeric(df[col])

    for col in CURATED_PLAIN_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Unit conversions:
    #   market_cap                              raw $M     -> store $M (no change)
    #   daily_traded_value                      raw dollars -> store $M
    #   stock_performance                       raw pct*100 -> store decimal (Rule 2)
    #   valuation_ev_revenue_ntm_percentile      raw 0-100  -> store 0-1
    df["daily_traded_value"] = df["daily_traded_value"] / 1_000_000
    df["stock_performance"] = df["stock_performance"] / 100
    df["valuation_ev_revenue_ntm_percentile"] = df["valuation_ev_revenue_ntm_percentile"] / 100

    parsed = df["scores"].apply(parse_scores)
    df["score_accounting_and_disclosure"] = parsed.apply(lambda t: t[0])
    df["score_fraud"] = parsed.apply(lambda t: t[1])
    df["score_insider"] = parsed.apply(lambda t: t[2])

    return df


def _find_single_upload_file(upload_dir: str) -> str:
    """Find the single expected upload file in a curated screen's folder.

    Curated screens have no column identifying which screen an export
    belongs to (all four share an identical schema), so screen identity
    depends entirely on exactly one correct file sitting in exactly one
    correct folder. This makes any deviation from that loud rather than
    silently concatenating or misreading files: more than one candidate
    file is an error naming what was found, and a single non-.csv
    candidate is an error too, rather than being read through a broken
    Excel sheet_name path (curated exports are csv-shaped; see the
    ScreenTypeError-style guards elsewhere in this phase for the same
    "fail clearly" principle).

    Args:
        upload_dir: Directory expected to hold exactly one export file.

    Returns:
        The full path to that one .csv file.

    Raises:
        CuratedUploadError: If zero files are found, more than one file is
            found (named in the message), or the single file found isn't
            a .csv.
    """
    candidates = sorted(
        f for f in os.listdir(upload_dir)
        if f.lower().endswith((".csv", ".xlsx", ".xls"))
    )
    if not candidates:
        raise CuratedUploadError(f"No export file found in {upload_dir}")
    if len(candidates) > 1:
        raise CuratedUploadError(
            f"Expected exactly one export file in {upload_dir}, found "
            f"{len(candidates)}: {candidates}. Curated screens have no way "
            f"to tell which export belongs to which screen from file "
            f"contents alone — remove all but the intended file."
        )
    (filename,) = candidates
    if not filename.lower().endswith(".csv"):
        raise CuratedUploadError(
            f"Curated ingest only supports .csv exports; found "
            f"{filename!r} in {upload_dir}. Canary curated exports are "
            f"csv-shaped today — if this is genuinely a new .xlsx export "
            f"format, curated_ingest.py needs updating (proper sheet_name "
            f"handling) before it can be read safely."
        )
    return os.path.join(upload_dir, filename)


def _log_curated_summary(screen_id: str, df: pd.DataFrame) -> None:
    """Log enough about a curated screen's cleaned data for a misfiled
    export to be visually obvious in the run output.

    Curated screens can't be told apart by column contents, so this is the
    primary signal a human has that the right export landed in the right
    folder — a Structural upload reporting 109 rows and Competition's
    tickers should look wrong at a glance.
    """
    tickers = sorted(df["ticker"].unique())
    logger.info(
        "screen_id=%s: %d rows, %d unique tickers", screen_id, len(df), len(tickers)
    )
    logger.info("  sector distribution: %s", df["sector"].value_counts().to_dict())
    logger.info("  ticker sample: %s", tickers[:10])


def ingest_curated(
    screen_id: str,
    upload_dir: str = None,
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
) -> None:
    """Run the ingestion pipeline for one curated screen.

    Reads the single export file in that screen's upload folder, validates
    required columns, cleans and unit-converts the data, writes to that
    screen's curated_data table (a wholesale snapshot replace — no merge,
    no edit history), and writes this screen's ticker universe to
    screen_membership. Also syncs the screens registry from config.yaml.

    Args:
        screen_id: Which curated screen to ingest.
        upload_dir: Directory containing this screen's single export file.
            Defaults to data/uploads/<screen_id>.
        db_path: Path to the SQLite database file.
        config_path: Path to config.yaml.

    Raises:
        ScreenTypeError: If screen_id's config.yaml type isn't "curated".
        CuratedUploadError: If upload_dir doesn't hold exactly one .csv file.
    """
    config = load_config(config_path)
    screen_type = get_screen_type(config, screen_id)
    if screen_type != "curated":
        raise ScreenTypeError(
            f"ingest_curated() only supports curated screens; {screen_id!r} "
            f"is type {screen_type!r}. Quant screens use ingest() instead."
        )

    if upload_dir is None:
        upload_dir = os.path.join("data", "uploads", screen_id)

    filepath = _find_single_upload_file(upload_dir)

    logger.info("Reading %s", filepath)
    raw_df = read_upload(filepath, sheet_name=None)
    validate_columns(raw_df, CURATED_REQUIRED_COLUMNS)
    cleaned = clean_curated_dataframe(raw_df)

    _log_curated_summary(screen_id, cleaned)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    sync_screens_registry(engine, config)

    dst_table = table_name("curated_data", screen_id)
    cleaned.to_sql(dst_table, engine, if_exists="replace", index=False)
    logger.info("Wrote %d rows to %s table at %s", len(cleaned), dst_table, db_path)

    membership_df = pd.DataFrame({"screen_id": screen_id, "ticker": cleaned["ticker"]})
    replace_screen_rows(engine, membership_df, "screen_membership", screen_id)
    logger.info(
        "Wrote %d rows to screen_membership for screen_id=%s", len(membership_df), screen_id
    )
