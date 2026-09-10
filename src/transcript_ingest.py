"""
Ingest Negative Expert Transcripts into SQLite.

Phase 6a's data foundation for the seventh screen, negative_expert_transcripts
— a quant_composite, unscored screen (no factor model), the same position as
Rising Short Interest. Unlike every other screen, this one is a header/detail
pair: a transcript can name more than one ticker, so the natural upload grain
is (transcript x ticker), not (screen x ticker). The detail table this module
writes, raw_data__negative_expert_transcripts, therefore ACCUMULATES ACROSS
UPLOADS by upsert on (doc_id, ticker) rather than being replaced wholesale on
each ingest — Architecture Rule 10's other deliberate exception, alongside
price_history.py's upsert/append-only pricing table. Dropping a prior
upload's transcripts on every refresh would destroy the very history this
screen exists to track.

DOCID BACKFILL DUPLICATION HAZARD (read before this screen's second real
upload): 14 of this vintage's 215 transcripts arrive with no DocID and are
given a synthesized "SYN-<hash>" id (see synthesize_doc_ids). If the SAME
period is ever re-exported with DocID now populated for those rows, the
re-exported rows carry different, real doc_ids and therefore do NOT match the
(doc_id, ticker) key of the already-stored SYN- rows — they are inserted IN
ADDITION to them, duplicating those transcripts and inflating mention_count
for their tickers on the next aggregate recompute. The Driver has agreed to
backfill DocID on every row in future exports, which is what makes this a
hazard worth guarding against rather than a hypothetical.

Phase 6b ships detection: find_duplicate_transcript_groups, called from
ingest_transcripts on the FULL accumulated raw table after every upsert,
flags any (transcript_date, theme, ticker) that resolves to more than one
doc_id and logs a WARNING per group — it reports only, never merges or
deletes. Because detection runs AFTER the insert rather than preventing it,
the OPERATIONAL RULE still stands: re-ingesting a period already stored
requires clearing this screen's raw_data table first; the upload files kept
in data/uploads/negative_expert_transcripts/_archive/ make the corpus
rebuildable, which is what makes that a safe instruction.

screen_membership is rewritten, on every ingest, from the distinct tickers in
the FULL accumulated raw table (read back after the upsert) rather than from
just the current upload's batch — so a ticker mentioned only in a prior
upload is never dropped from membership just because this upload doesn't
mention it again.
"""

import hashlib
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
from src.db import replace_screen_rows, sync_screens_registry, table_name, upsert_rows  # noqa: E402
from src.loaders import find_single_upload_file, log_summary, read_upload, validate_columns  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The only sheet read — "Ticker Counts" is fully derivable from this one and
# is never read (see PHASE6_SCOPE.md §2).
TRANSCRIPT_SHEET_NAME = "Transcript Summaries"

# Raw Excel header -> internal snake_case name. "Transcript Date" and
# "Tickers" are handled separately (date parsing / ticker explode), not a
# plain rename. "Sector" becomes source_sector, never "sector" — this
# source's 9-value non-GICS taxonomy must never reach overlap.py's
# cross-source sector fallback under the shared "sector" name.
TRANSCRIPT_COLUMN_MAP = {
    "Expert Position/Background": "expert_position",
    "Key Takeaways": "key_takeaways",
    "Negativity Rationale": "negativity_rationale",
    "Source Pack": "source_pack",
    "Sector": "source_sector",
    "Theme": "theme",
    "Company": "transcript_company",
    "DocID": "doc_id",
}

TRANSCRIPT_REQUIRED_COLUMNS = ["Transcript Date", "Tickers"] + list(TRANSCRIPT_COLUMN_MAP)

# Final detail-table column order (grain: transcript x ticker).
TRANSCRIPT_DETAIL_COLUMNS = [
    "doc_id", "doc_id_synthetic", "ticker", "transcript_date", "expert_position",
    "key_takeaways", "negativity_rationale", "theme", "transcript_company",
    "source_pack", "source_sector",
]


def clean_transcript_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and parse the transcript date for a raw Negative
    Expert Transcripts read. Grain stays one row per transcript (215 on the
    initial vintage) — ticker explosion happens later, in explode_tickers.

    Args:
        df: DataFrame as read from the "Transcript Summaries" sheet, all
            str dtype (loaders.read_upload's convention).

    Returns:
        Cleaned DataFrame with snake_case columns and transcript_date as an
        ISO "YYYY-MM-DD" string. "Tickers" and "doc_id" are left untouched
        for synthesize_doc_ids/explode_tickers to handle.

    Raises:
        ValueError: If any row's "Transcript Date" can't be parsed. A NaT
            here would silently corrupt synthesize_doc_ids' hash key and
            run_transcript_aggregation's corpus-max computation, so this
            fails loudly rather than following the Rule-3 "return NaN"
            default — that rule governs numeric metric calculations, not a
            row-identity date feeding a hash and a corpus-max. This is safe
            operationally on two independent grounds: it fires inside
            refresh.py's prepare step (clean_and_explode_transcripts is
            called from _prepare_negative_expert_transcripts before the
            real ingest ever runs), which refresh_one() already catches
            via `except (UploadFileError, KeyError, ValueError)` and turns
            into a FAILED result for this screen only — the other six
            screens still run and the run's exit code goes non-zero, per
            the Driver's continue-past-a-failing-screen policy; and
            refresh() also has a broad backstop around refresh_one() for
            anything unexpected. The same parse can't fire a second time,
            unguarded, at the real ingest call — prepare and ingest read
            the same file through this same function.
    """
    df = df.rename(columns=TRANSCRIPT_COLUMN_MAP)

    parsed_dates = pd.to_datetime(df["Transcript Date"], errors="coerce")
    bad = df[parsed_dates.isna()]
    if len(bad) > 0:
        examples = [
            f"(Transcript Date={row['Transcript Date']!r}, Theme={row['theme']!r})"
            for _, row in bad.head(5).iterrows()
        ]
        raise ValueError(
            f"{len(bad)} row(s) have an unparseable Transcript Date, e.g. {examples}"
        )
    df["transcript_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["Transcript Date"])

    for col in TRANSCRIPT_COLUMN_MAP.values():
        if col == "doc_id":
            continue
        # Vectorized .str ops pass NaN through unchanged rather than
        # stringifying it to the literal "nan" — no astype(str) needed
        # since read_upload already reads every column as str dtype.
        df[col] = df[col].str.strip()

    return df


def synthesize_doc_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Fill a missing DocID with a deterministic synthetic id, keyed on
    (transcript_date, theme) — computed BEFORE ticker explosion, so every
    ticker exploded from the same transcript row shares one doc_id. This is
    what makes 216 detail rows resolve to only 215 distinct doc_ids on the
    initial vintage (the one multi-ticker transcript shares its single
    doc_id across both exploded rows), rather than 216 independent ids.

    The Driver has agreed to populate DocID on every row in future exports,
    so this is a durable fallback for a bad export, not permanent
    architecture — see the module docstring's duplication hazard.

    Args:
        df: Output of clean_transcript_dataframe (one row per transcript).

    Returns:
        df with doc_id fully populated and a new doc_id_synthetic column:
        1 for a synthesized id, 0 for a real DocID from the source file.
    """
    df = df.copy()
    is_missing = df["doc_id"].isna()
    df["doc_id_synthetic"] = is_missing.astype(int)

    def _synthesize(row):
        key = f"{row['transcript_date']}|{row['theme']}"
        return "SYN-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    if is_missing.any():
        df.loc[is_missing, "doc_id"] = df.loc[is_missing].apply(_synthesize, axis=1)
    df["doc_id"] = df["doc_id"].str.strip()
    return df


def explode_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the comma-delimited Tickers column into one detail row per
    ticker (grain: transcript x ticker). Every column other than Tickers is
    inherited unchanged by every row exploded from the same transcript —
    including transcript_company, which is deliberately NOT re-derived per
    ticker (see module docstring: a multi-ticker transcript's Company names
    the transcript's subject, not each ticker individually — the HON/GE
    transcript's Company is "Honeywell International Inc" on both exploded
    rows, and that is correct, not a bug to fix).

    Args:
        df: Output of synthesize_doc_ids (one row per transcript, doc_id
            fully populated).

    Returns:
        One row per (transcript, ticker), columns exactly
        TRANSCRIPT_DETAIL_COLUMNS, in that order.
    """
    df = df.copy()
    df["ticker"] = df["Tickers"].str.split(",")
    df = df.explode("ticker")
    df["ticker"] = df["ticker"].str.strip()
    return df[TRANSCRIPT_DETAIL_COLUMNS].reset_index(drop=True)


def clean_and_explode_transcripts(df: pd.DataFrame) -> pd.DataFrame:
    """The full read-to-detail pipeline. Public so refresh.py's prepare
    replica and the real ingest call the identical sequence — same
    precedent as rsi_ingest's independently public trim_rsi_export/
    clean_rsi_dataframe — so the two paths cannot diverge.
    """
    return explode_tickers(synthesize_doc_ids(clean_transcript_dataframe(df)))


def find_duplicate_transcript_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Detect a DocID-backfill duplication (see module docstring): group
    the accumulated raw_data__negative_expert_transcripts frame on
    (transcript_date, theme, ticker) and return only the groups whose
    distinct doc_id count exceeds 1 — the signature of a SYN-<hash> row and
    a later real-DocID row landing under the same (date, theme, ticker) key
    without matching on (doc_id, ticker), so the upsert inserted the
    real-id row alongside the synthetic one instead of replacing it.

    ticker is deliberately part of the key alongside (transcript_date,
    theme), for two separate reasons:

    - Negative control: EC-1000000-230444 names two tickers and is
      genuinely one document split across two exploded rows sharing
      (date, theme) — but both rows share the SAME doc_id, so this shape
      is silent under either key (ticker included or not); it does not
      demonstrate why ticker matters, only that this function doesn't
      misfire on it.
    - The actual case ticker guards against: two DIFFERENT transcripts
      (different doc_id, different ticker) that coincidentally share
      (transcript_date, theme). Keyed with ticker, each is its own
      singleton group — nothing flags. Drop ticker from the key and they
      collapse into one group with two distinct doc_ids, wrongly reporting
      a coincidence as a duplicate.

    Disclosed limitation: two genuinely distinct transcripts naming the
    same ticker on the same date under an identical theme string would
    still flag as a false positive. None exist in the corpus today — every
    (transcript_date, theme, ticker) group currently resolves to exactly
    one doc_id. This function only reports; it never merges or deletes
    anything — a human decides what to do with a flagged group.

    Args:
        df: The FULL accumulated raw_data__negative_expert_transcripts
            frame (transcript x ticker grain), not just an incoming
            upload batch — the duplication this detects is between rows
            already stored and rows newly arriving, so a check applied to
            the incoming batch alone would never see it.

    Returns:
        One row per flagged (transcript_date, theme, ticker) group: those
        three key columns, doc_id_count, and doc_ids (a sorted list of the
        offending doc_id values). Empty (same columns, zero rows) if
        nothing is flagged.
    """
    rows = []
    for (transcript_date, theme, ticker), group in df.groupby(
        ["transcript_date", "theme", "ticker"]
    ):
        doc_ids = sorted(group["doc_id"].unique())
        if len(doc_ids) > 1:
            rows.append({
                "transcript_date": transcript_date,
                "theme": theme,
                "ticker": ticker,
                "doc_id_count": len(doc_ids),
                "doc_ids": doc_ids,
            })
    return pd.DataFrame(
        rows, columns=["transcript_date", "theme", "ticker", "doc_id_count", "doc_ids"]
    )


def ingest_transcripts(
    screen_id: str = "negative_expert_transcripts",
    upload_dir: str = None,
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
) -> None:
    """Ingest one upload of Negative Expert Transcripts.

    Reads the single export in the screen's upload folder, cleans and
    explodes it to (transcript x ticker) grain, and upserts it into
    raw_data__<screen_id> on (doc_id, ticker) — see module docstring for
    why this table accumulates rather than being replaced.

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
        KeyError: If a required column is missing from the sheet.
        ValueError: If a row's Transcript Date can't be parsed (see
            clean_transcript_dataframe).
    """
    config = load_config(config_path)
    screen_type = get_screen_type(config, screen_id)
    if screen_type != "quant_composite":
        raise ScreenTypeError(
            f"ingest_transcripts() only supports quant_composite screens; "
            f"{screen_id!r} is type {screen_type!r}."
        )

    if upload_dir is None:
        upload_dir = os.path.join("data", "uploads", screen_id)

    filepath = find_single_upload_file(upload_dir, ".xlsx")

    logger.info("Reading %s", filepath)
    raw_df = read_upload(filepath, sheet_name=TRANSCRIPT_SHEET_NAME)
    validate_columns(raw_df, TRANSCRIPT_REQUIRED_COLUMNS)

    cleaned = clean_and_explode_transcripts(raw_df)
    log_summary(cleaned)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    sync_screens_registry(engine, config)

    raw_table = table_name("raw_data", screen_id)
    upsert_rows(engine, cleaned, raw_table, key_columns=["doc_id", "ticker"])
    logger.info("Upserted %d row(s) into %s at %s", len(cleaned), raw_table, db_path)

    full_df = pd.read_sql_table(raw_table, engine)
    membership_df = pd.DataFrame({
        "screen_id": screen_id,
        "ticker": sorted(full_df["ticker"].dropna().unique()),
    })
    replace_screen_rows(engine, membership_df, "screen_membership", screen_id)
    logger.info(
        "Wrote %d rows to screen_membership for screen_id=%s (full accumulated corpus)",
        len(membership_df), screen_id,
    )

    duplicate_groups = find_duplicate_transcript_groups(full_df)
    if len(duplicate_groups) > 0:
        for _, grp in duplicate_groups.iterrows():
            logger.warning(
                "Possible duplicate transcript: ticker=%s transcript_date=%s theme=%r "
                "doc_ids=%s",
                grp["ticker"], grp["transcript_date"], grp["theme"], grp["doc_ids"],
            )
    else:
        logger.info(
            "No duplicate transcript groups detected in the accumulated corpus "
            "(%d rows).", len(full_df),
        )


if __name__ == "__main__":
    ingest_transcripts()
