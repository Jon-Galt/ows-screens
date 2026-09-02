"""
Phase 4a — standalone ingest of the historical position-outcomes workbook
("OWS Ideas Performance <date>.xlsx") into data/screener.db.

THIS IS NOT A SCREEN. It has no config.yaml screens-block entry, no
screens-registry row, no screen_membership row, and is never dispatched by
refresh.py's per-screen loop — refresh.py iterates registry screens and
gates each on validate.py's checks, a shape this two-sheet historical
import doesn't share. This module is only ever invoked directly:
`python src/historical_ingest.py [--dry-run]`.

Two sheets go to two tables (historical_active_shorts,
historical_whiteboard_shorts), written with if_exists="replace" — this is
a faithful import of an external system of record, fully reconstructable
from the source file, so Architecture Rule 10's append-only exception
(which exists because SCORE HISTORY can't be reconstructed once
overwritten) does not apply here. What genuinely can't be reconstructed
once a newer source file supersedes this one is the fact of what THIS
ingest saw — its row/defect counts and sign-check correlations — so
historical_ingest_runs (one provenance row per run) IS append-only, via
db.append_rows(), the same helper refresh.py's history tables use.

THE SIGN-CONVENTION GATE (see check_sign_convention) is the one thing in
this module allowed to abort the import outright: the performance columns
hold short P&L, not stock return, and a file that flipped that convention
would silently invert every conclusion in this phase while every row
count and null check still passed. Nothing else here hard-fails — data
defects and benchmark-labelling gaps are counted and reported, never
rejected, per the Driver's "none of these should abort the import" ruling
(see count_defects and check_benchmark_consistency).

BENCHMARK INSTRUMENT MISLABELLING (defect 7) is the second most important
thing here. Active's "SPX @ Initiation"/"SPX @ Close" columns actually
hold TWO instruments — SPY ETF prices on 62 rows (all pre-2018) and SPX
index levels on 376 rows — with no ticker to disambiguate, only price
magnitude. classify_benchmark_instrument's threshold+band rule is
DERIVED FROM AND ONLY VALID FOR THIS SPECIFIC EXPORT'S VINTAGE: it works
because every SPY-priced Active row happens to have closed by 2018-06-27,
leaving values in (270.39, 756.55) genuinely unclassifiable rather than
ambiguous. SPY itself already trades at 772 as of this file's date —
ABOVE the SPX-side floor of that same band — so this rule is expected to
start producing "unclassifiable" rows on a future file and that is
correct, flagging behavior, not a bug to fix by widening the band.
Whiteboard's own SPY @ WBA/WBR columns are NOT run through this
classifier at all: the source header already says "SPY" unambiguously
(confirmed instrument-pure — see clean_whiteboard_dataframe), so
Whiteboard's benchmark_instrument is a plain per-row constant, a label,
not an inference.

THE REAL GUARD, on both sheets, is check_benchmark_consistency: it
recomputes relative performance from each row's own price/benchmark
levels and flags any row where that diverges from the stored value beyond
config.yaml's historical.benchmark_consistency_tolerance. Every
performance value in this file is rounded to a whole percentage point
(verified across all six performance columns, both sheets — see
count_defects' performance_values_rounded_to_whole_percent), which bounds
the expected gap at ~0.005 arithmetically, not by measurement noise —
config.yaml's comment on this threshold explains why it must never be
widened to silence a real violation.
"""

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import CONFIG_PATH, load_config
from src.db import append_rows, create_index_if_not_exists
from src.loaders import file_provenance, find_single_upload_file, validate_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ACTIVE_SHEET = "Active Shorts Performance"
WHITEBOARD_SHEET = "Whiteboard Shorts Performance"

ACTIVE_COLUMN_MAP = {
    "OWS DB Ticker": "ticker",
    "BBG Ticker": "bbg_ticker",
    "Company Name": "company_name",
    "Setup": "setup",
    "Sector": "sector",
    "Sector Index": "sector_benchmark_ticker",
    "Status": "status",
    "Init Report Filename": "init_report_filename",
    "Init Report PDF Link": "init_report_pdf_link",
    "Init Report DOCX Link": "init_report_docx_link",
    "Initiation Date": "initiation_date",
    "Init Pricing Date": "init_pricing_date",
    "Market Cap @ Initiation": "market_cap_at_initiation",
    "Initiation Price": "initiation_price",
    "SPX @ Initiation": "benchmark_at_initiation",
    "Sector Index @ Initiation": "sector_benchmark_at_initiation",
    "Close Report Filename": "close_report_filename",
    "Close Report PDF Link": "close_report_pdf_link",
    "Close Report DOCX Link": "close_report_docx_link",
    "Close Date": "close_date",
    "Close Pricing Date": "close_pricing_date",
    "Close Price": "close_price",
    "SPX @ Close": "benchmark_at_close",
    "Sector Index @ Close": "sector_benchmark_at_close",
    "Duration": "duration_days",
    "Absolute Performance": "absolute_performance",
    "Relative SPY Performance": "relative_spy_performance",
    "Relative Sector Performance": "relative_sector_performance",
}

WHITEBOARD_COLUMN_MAP = {
    "OWS DB Ticker": "ticker",
    "BBG Ticker": "bbg_ticker",
    "Company Name": "company_name",
    "Setup": "setup",
    "Sector": "sector",
    "Sector ETF": "sector_benchmark_ticker",
    "Status": "status",
    "WBA Report Filename": "wba_report_filename",
    "WBA Report DOCX Link": "wba_report_docx_link",
    "WBA Report PDF Link": "wba_report_pdf_link",
    "WBA Date": "wba_date",
    "WBA Pricing Date": "wba_pricing_date",
    "Market Cap @ WBA ($B)": "market_cap_at_wba",
    "WBA Price": "wba_price",
    "SPY @ WBA": "benchmark_at_wba",
    "Sector ETF @ WBA": "sector_benchmark_at_wba",
    "Outcome": "outcome",
    "WBR Report Filename": "wbr_report_filename",
    "WBR Report DOCX Link": "wbr_report_docx_link",
    "WBR Report PDF Link": "wbr_report_pdf_link",
    "WBR Date": "wbr_date",
    "WBR Pricing Date": "wbr_pricing_date",
    "WBR Price": "wbr_price",
    "SPY @ WBR": "benchmark_at_wbr",
    "Sector ETF @ WBR": "sector_benchmark_at_wbr",
    "Duration": "duration_days",
    "Absolute Performance": "absolute_performance",
    "Relative SPY Performance": "relative_spy_performance",
    "Relative Sector Performance": "relative_sector_performance",
}

ACTIVE_DATE_COLUMNS = ["initiation_date", "init_pricing_date", "close_date", "close_pricing_date"]
ACTIVE_NUMERIC_COLUMNS = [
    "market_cap_at_initiation", "initiation_price", "benchmark_at_initiation",
    "sector_benchmark_at_initiation", "close_price", "benchmark_at_close",
    "sector_benchmark_at_close", "duration_days", "absolute_performance",
    "relative_spy_performance", "relative_sector_performance",
]
WHITEBOARD_DATE_COLUMNS = ["wba_date", "wba_pricing_date", "wbr_date", "wbr_pricing_date"]
WHITEBOARD_NUMERIC_COLUMNS = [
    "market_cap_at_wba", "wba_price", "benchmark_at_wba", "sector_benchmark_at_wba",
    "wbr_price", "benchmark_at_wbr", "sector_benchmark_at_wbr",
    "absolute_performance", "relative_spy_performance", "relative_sector_performance",
]

PERFORMANCE_COLUMNS = ["absolute_performance", "relative_spy_performance", "relative_sector_performance"]

# See the module docstring's "BENCHMARK INSTRUMENT MISLABELLING" section.
# These are the EXACT measured edges of the empty gap in Active's
# "SPX @ Initiation"/"SPX @ Close" values on the file this phase was built
# against (62 values <= 270.39, 376 values >= 756.55, nothing between).
# Active-sheet-specific and vintage-specific — do not reuse for Whiteboard
# (its own SPY column already exceeds this ceiling: SPY traded at 772.67
# as of this file) and do not widen these edges to keep classifying a
# future file whose SPY-priced Active rows are more recent than 2018 —
# widening would silently start guessing instead of flagging.
BENCHMARK_SPY_CEILING = 270.39
BENCHMARK_SPX_FLOOR = 756.55


class SignConventionError(ValueError):
    """Raised by ingest_historical when check_sign_convention fails on
    either sheet. Aborts the import before any write — see the module
    docstring."""


@dataclass(frozen=True)
class SignCheckResult:
    """Outcome of the sign-convention gate for one sheet.

    Attributes:
        corr: Pearson correlation between each row's own price move and its
            stored Absolute Performance. NaN if n < 2.
        n: Rows with both a computable price move and a non-null Absolute
            Performance.
        passed: True iff n >= 2 and corr <= -min_abs_corr. n < 2 always
            fails explicitly (not by accidentally relying on "NaN <= x is
            False") — an empty/all-NaN intersection must not pass by
            vacuous truth.
    """

    corr: float
    n: int
    passed: bool


@dataclass(frozen=True)
class ConsistencyResult:
    """Outcome of the benchmark-consistency check for one sheet.

    Attributes:
        n: Rows with all four levels (init/close price, init/close
            benchmark) and a non-null stored relative-performance value.
        violation_count: Rows where |implied - stored| > tolerance.
        max_abs_diff: Largest |implied - stored| observed, NaN if n == 0.
        tolerance: The tolerance this result was computed against.
        violation_tickers: Up to 20 offending tickers, worst first.
    """

    n: int
    violation_count: int
    max_abs_diff: float
    tolerance: float
    violation_tickers: list = field(default_factory=list)


def _stringify_preserving_null(series: pd.Series) -> pd.Series:
    """Render every non-null cell as its str(), preserving null as None.

    Used only for duration_raw: Whiteboard's source Duration column is
    dtype object (a mix of numbers and the literal string 'Error'), and
    this keeps that original value fully auditable alongside the coerced
    numeric duration_days column, per the Driver's "count and report every
    defect" ruling.
    """
    return series.map(lambda v: None if pd.isna(v) else str(v))


def classify_benchmark_instrument(
    level: pd.Series, spy_ceiling: float = BENCHMARK_SPY_CEILING, spx_floor: float = BENCHMARK_SPX_FLOOR
) -> pd.Series:
    """Classify a numeric benchmark level as SPY-scale or SPX-scale.

    Active-sheet-specific — see the module docstring and the
    BENCHMARK_SPY_CEILING/BENCHMARK_SPX_FLOOR constants' comment. Do not
    apply this to Whiteboard's benchmark columns: its source header
    already identifies the instrument unambiguously, and this threshold
    would misclassify the vast majority of Whiteboard's own values (SPY
    now trades well inside this rule's "unclassifiable" band).

    Args:
        level: A numeric benchmark-level column (e.g. benchmark_at_initiation).
        spy_ceiling: Values at or below this are "SPY".
        spx_floor: Values at or above this are "SPX".

    Returns:
        A Series of "SPY" / "SPX" / None (null input, or a value strictly
        between spy_ceiling and spx_floor — unclassifiable, not guessed).
    """

    def _classify(value):
        if pd.isna(value):
            return None
        if value <= spy_ceiling:
            return "SPY"
        if value >= spx_floor:
            return "SPX"
        return None

    return level.map(_classify)


def classify_sector_benchmark_instrument(ticker: pd.Series) -> pd.Series:
    """Classify a sector-benchmark ticker string as an ETF or an index.

    Exact string classification (Bloomberg tickers carry an unambiguous
    " Index" or " US Equity" suffix) — unlike classify_benchmark_instrument,
    this is not vintage-dependent and is applied identically to both
    sheets.

    Args:
        ticker: A sector-benchmark ticker column (e.g. sector_benchmark_ticker),
            values like "S5INDU Index" or "XLK US Equity".

    Returns:
        A Series of "INDEX" / "ETF" / None (null input, or a ticker string
        matching neither suffix — logged as "unclassified", never guessed).
    """

    def _classify(value):
        if pd.isna(value):
            return None
        if "US Equity" in value:
            return "ETF"
        if "Index" in value:
            return "INDEX"
        return None

    return ticker.map(_classify)


def clean_active_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the Active Shorts Performance sheet.

    Renames all 28 source columns to snake_case (ACTIVE_COLUMN_MAP),
    coerces dates/numerics with errors="coerce" (Architecture Rule 3 — a
    bad cell becomes NaN/NaT, never an exception), converts
    market_cap_at_initiation from the source's $B to this project's
    standing $M convention (exact, x1000), and derives benchmark_instrument
    /sector_benchmark_instrument labels. Rows with defects (missing Setup,
    Closed with no Close Date, Open with performance, etc.) are NOT
    dropped or altered here — they pass through unchanged; counting them
    is count_defects' job, not this function's.

    Args:
        raw: The Active Shorts Performance sheet as read by pd.read_excel.

    Returns:
        Cleaned DataFrame, same row count as raw.

    Raises:
        KeyError: Via loaders.validate_columns, if any of the 28 expected
            source columns is missing.
    """
    validate_columns(raw, list(ACTIVE_COLUMN_MAP.keys()))
    df = raw.rename(columns=ACTIVE_COLUMN_MAP).copy()

    for col in ACTIVE_DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ACTIVE_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["market_cap_at_initiation"] = df["market_cap_at_initiation"] * 1000
    df["ticker"] = df["ticker"].astype(str).str.strip()

    df["benchmark_instrument"] = classify_benchmark_instrument(df["benchmark_at_initiation"])
    df["sector_benchmark_instrument"] = classify_sector_benchmark_instrument(df["sector_benchmark_ticker"])

    return df


def clean_whiteboard_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the Whiteboard Shorts Performance sheet.

    Same discipline as clean_active_dataframe. Duration is dtype object in
    the source (one row carries the literal string 'Error') — duration_raw
    preserves that original value, duration_days is the errors="coerce"
    numeric version (that one row becomes NaN). WBR Date/WBR Pricing Date
    are also dtype object in the source (one row carries the literal
    string 'Not Found', on the SAME row as the Duration='Error' and
    Outcome=null defects — see count_defects) and are coerced via
    pd.to_datetime(errors="coerce") like every other date column.
    market_cap_at_wba is converted from the source's $B to $M (x1000).
    benchmark_instrument is set to the constant "SPY" for every row — see
    the module docstring for why this is a label, not an inference, and
    why check_benchmark_consistency (not this constant) is the actual
    guard against a future file mixing instruments here.

    Args:
        raw: The Whiteboard Shorts Performance sheet as read by pd.read_excel.

    Returns:
        Cleaned DataFrame, same row count as raw.

    Raises:
        KeyError: Via loaders.validate_columns, if any of the 29 expected
            source columns is missing.
    """
    validate_columns(raw, list(WHITEBOARD_COLUMN_MAP.keys()))
    duration_raw = _stringify_preserving_null(raw["Duration"])

    df = raw.rename(columns=WHITEBOARD_COLUMN_MAP).copy()
    df["duration_raw"] = duration_raw
    df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")

    for col in WHITEBOARD_DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in WHITEBOARD_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["market_cap_at_wba"] = df["market_cap_at_wba"] * 1000
    df["ticker"] = df["ticker"].astype(str).str.strip()

    df["benchmark_instrument"] = "SPY"
    df["sector_benchmark_instrument"] = classify_sector_benchmark_instrument(df["sector_benchmark_ticker"])

    return df


def check_sign_convention(
    df: pd.DataFrame, init_price_col: str, close_price_col: str, perf_col: str, min_abs_corr: float
) -> SignCheckResult:
    """The sign-convention gate: assert short P&L, not stock return.

    Computes each row's own price_move = (close - init) / init and
    correlates it against the stored perf_col. A healthy file has
    corr <= -min_abs_corr (short P&L moves opposite the stock). This
    function only REPORTS — ingest_historical raises SignConventionError
    and aborts the write when passed is False, per the module docstring.

    Args:
        df: A cleaned sheet (Active or Whiteboard).
        init_price_col: Column holding the position's entry price.
        close_price_col: Column holding the position's exit price.
        perf_col: Column holding the stored performance to check against
            (Absolute Performance, per the Driver's spec).
        min_abs_corr: Gate threshold (config.yaml's
            historical.sign_convention_min_abs_corr).

    Returns:
        SignCheckResult. n < 2 always yields passed=False explicitly (not
        merely NaN <= x evaluating False) — an empty/all-NaN intersection
        must not pass by vacuous truth.
    """
    price_move = (df[close_price_col] - df[init_price_col]) / df[init_price_col]
    mask = price_move.notna() & df[perf_col].notna()
    n = int(mask.sum())
    if n < 2:
        return SignCheckResult(corr=float("nan"), n=n, passed=False)
    corr = float(np.corrcoef(price_move[mask], df[perf_col][mask])[0, 1])
    return SignCheckResult(corr=corr, n=n, passed=corr <= -min_abs_corr)


def check_benchmark_consistency(
    df: pd.DataFrame,
    init_price_col: str,
    close_price_col: str,
    bench_init_col: str,
    bench_close_col: str,
    perf_col: str,
    tolerance: float,
    ticker_col: str = "ticker",
) -> ConsistencyResult:
    """The real guard behind the benchmark columns — see module docstring.

    Recomputes relative performance from each row's OWN levels
    (bench_move - price_move, matching this file's short-P&L sign
    convention) and compares it to the stored perf_col value. This
    catches a wrong instrument, a mid-column instrument switch, a swapped
    column, or a corrupted level — none of which classify_benchmark_
    instrument's magnitude/constant labels can see, since it doesn't rely
    on knowing the instrument's identity at all, only that the row is
    internally consistent.

    Does NOT raise and does NOT gate the write — per the Driver's ruling,
    only the sign-convention gate hard-fails. A violation is reported via
    count_defects and printed, never silently absorbed.

    Args:
        df: A cleaned sheet (Active or Whiteboard).
        init_price_col, close_price_col: The position's entry/exit price
            columns.
        bench_init_col, bench_close_col: The benchmark level columns.
        perf_col: The stored relative-performance column to check against.
        tolerance: Absolute tolerance (config.yaml's
            historical.benchmark_consistency_tolerance) — see that key's
            config.yaml comment for why 0.01 is not "current noise that
            could grow."
        ticker_col: Name of the ticker column, for violation_tickers.

    Returns:
        ConsistencyResult. n == 0 yields violation_count=0,
        max_abs_diff=NaN — an empty intersection reports as "nothing to
        check," not as a pass on invented evidence.
    """
    price_move = (df[close_price_col] - df[init_price_col]) / df[init_price_col]
    bench_move = (df[bench_close_col] - df[bench_init_col]) / df[bench_init_col]
    implied = bench_move - price_move
    mask = implied.notna() & df[perf_col].notna()
    n = int(mask.sum())
    if n == 0:
        return ConsistencyResult(n=0, violation_count=0, max_abs_diff=float("nan"), tolerance=tolerance)

    diff = (implied[mask] - df[perf_col][mask]).abs()
    max_abs_diff = float(diff.max())
    # A small epsilon guards the "exactly at tolerance is not a violation"
    # boundary against floating-point representation noise in diff's own
    # subtraction/division chain — without it, a diff mathematically equal
    # to tolerance can come out a few ULPs above it and misclassify.
    violation_mask = diff > (tolerance + 1e-9)
    violation_count = int(violation_mask.sum())

    violation_tickers = []
    if violation_count:
        ordered = diff[violation_mask].sort_values(ascending=False)
        violation_tickers = list(df.loc[ordered.index, ticker_col])[:20]

    return ConsistencyResult(
        n=n, violation_count=violation_count, max_abs_diff=max_abs_diff,
        tolerance=tolerance, violation_tickers=violation_tickers,
    )


def _all_whole_percent(df: pd.DataFrame, cols: list, atol: float = 1e-6) -> bool:
    """True iff every non-null value in every named column, x100, is
    within atol of a whole number. Used to record the file's observed
    rounding precision (see count_defects) — not a defect, a property."""
    for col in cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        pct = s * 100
        if not np.isclose(pct, pct.round(), atol=atol).all():
            return False
    return True


def count_defects(
    active_df: pd.DataFrame,
    whiteboard_df: pd.DataFrame,
    active_consistency: ConsistencyResult,
    whiteboard_consistency: ConsistencyResult,
) -> dict:
    """Count and structure every known data-quality property of this file.

    None of these abort the import — the Driver's standing ruling for
    this phase. Two known consolidations are reflected in the shape of
    what's counted, not hidden: CARG is one Whiteboard row carrying three
    defects at once (Duration='Error', Outcome null, WBR Date='Not
    Found'), and ANSS/MFE/GLYT are one group of three Active rows that are
    Status=Closed with BOTH Initiation Date and Close Date null — the
    entire "closed with no close date" defect population AND the entire
    null-Initiation-Date population are the same three rows. Also reports
    market_cap_zero — rows with a source market cap of exactly 0.0
    (assign_market_cap_bucket would otherwise silently sort these into
    "<$1B" alongside ordinary small caps) — as an observed property, same
    non-gating treatment as performance_values_rounded_to_whole_percent.

    Args:
        active_df: Cleaned Active sheet.
        whiteboard_df: Cleaned Whiteboard sheet.
        active_consistency: check_benchmark_consistency's result for Active.
        whiteboard_consistency: check_benchmark_consistency's result for
            Whiteboard.

    Returns:
        Nested dict — see the module's config.yaml/CLAUDE.md docs for the
        full shape. JSON-serializable after passing through _json_safe.
    """
    closed_no_close_date = sorted(
        active_df.loc[(active_df["status"] == "Closed") & active_df["close_date"].isna(), "ticker"]
    )
    open_with_performance = sorted(
        active_df.loc[(active_df["status"] == "Open") & active_df["absolute_performance"].notna(), "ticker"]
    )
    whiteboard_duration_error_tickers = sorted(
        whiteboard_df.loc[
            whiteboard_df["duration_raw"].astype(str).str.contains("Error", case=False, na=False), "ticker"
        ]
    )

    active_init_class = classify_benchmark_instrument(active_df["benchmark_at_initiation"])
    active_close_class = classify_benchmark_instrument(active_df["benchmark_at_close"])
    both_classified = active_init_class.notna() & active_close_class.notna()
    init_close_mismatch = int((active_init_class[both_classified] != active_close_class[both_classified]).sum())

    benchmark_instrument = {
        "active_spy": int((active_init_class == "SPY").sum()),
        "active_spx": int((active_init_class == "SPX").sum()),
        "active_unclassifiable": int(
            (active_df["benchmark_at_initiation"].notna() & active_init_class.isna()).sum()
        ),
        "active_null": int(active_df["benchmark_at_initiation"].isna().sum()),
        "active_init_close_mismatch": init_close_mismatch,
        "whiteboard": f'constant "SPY" ({len(whiteboard_df)} rows) — source header is unambiguous, see module docstring',
    }

    active_sector_class = active_df["sector_benchmark_instrument"]
    whiteboard_sector_class = whiteboard_df["sector_benchmark_instrument"]
    sector_benchmark_instrument = {
        "active_etf": int((active_sector_class == "ETF").sum()),
        "active_index": int((active_sector_class == "INDEX").sum()),
        "active_null": int(active_df["sector_benchmark_ticker"].isna().sum()),
        "active_unclassified": int(
            (active_df["sector_benchmark_ticker"].notna() & active_sector_class.isna()).sum()
        ),
        "whiteboard_etf": int((whiteboard_sector_class == "ETF").sum()),
        "whiteboard_null": int(whiteboard_df["sector_benchmark_ticker"].isna().sum()),
        "whiteboard_unclassified": int(
            (whiteboard_df["sector_benchmark_ticker"].notna() & whiteboard_sector_class.isna()).sum()
        ),
    }

    benchmark_consistency = {
        "active_n": active_consistency.n,
        "active_violations": active_consistency.violation_count,
        "active_max_abs_diff": active_consistency.max_abs_diff,
        "active_violation_tickers": active_consistency.violation_tickers,
        "whiteboard_n": whiteboard_consistency.n,
        "whiteboard_violations": whiteboard_consistency.violation_count,
        "whiteboard_max_abs_diff": whiteboard_consistency.max_abs_diff,
        "whiteboard_violation_tickers": whiteboard_consistency.violation_tickers,
    }

    return {
        "closed_no_close_date": closed_no_close_date,
        "open_with_performance": open_with_performance,
        "setup_missing_count": int(active_df["setup"].isna().sum()),
        "whiteboard_duration_error_tickers": whiteboard_duration_error_tickers,
        "outcome_missing_count": int(whiteboard_df["outcome"].isna().sum()),
        "whiteboard_sector_coverage": {
            "sector_benchmark_at_wba": f'{int(whiteboard_df["sector_benchmark_at_wba"].notna().sum())}/{len(whiteboard_df)}',
            "relative_sector_performance": f'{int(whiteboard_df["relative_sector_performance"].notna().sum())}/{len(whiteboard_df)}',
        },
        "benchmark_instrument": benchmark_instrument,
        "sector_benchmark_instrument": sector_benchmark_instrument,
        "benchmark_consistency": benchmark_consistency,
        "performance_values_rounded_to_whole_percent": {
            "active": _all_whole_percent(active_df, PERFORMANCE_COLUMNS),
            "whiteboard": _all_whole_percent(whiteboard_df, PERFORMANCE_COLUMNS),
        },
        "market_cap_zero": {
            "active": sorted(active_df.loc[active_df["market_cap_at_initiation"] == 0.0, "ticker"]),
            "whiteboard": sorted(whiteboard_df.loc[whiteboard_df["market_cap_at_wba"] == 0.0, "ticker"]),
        },
    }


def build_whiteboard_bridge(active_df: pd.DataFrame, whiteboard_df: pd.DataFrame) -> pd.DataFrame:
    """Join promoted Whiteboard ideas to their Active position on the exact
    key: (ticker, WBR Date == Initiation Date).

    Only whiteboard rows with outcome == "Initiation" are candidates.
    Overlapping column names between the two frames (status, setup,
    absolute_performance, etc.) are suffixed "_wb"/"_act" by the merge —
    the join key columns (ticker) are not.

    Args:
        active_df: Cleaned Active sheet.
        whiteboard_df: Cleaned Whiteboard sheet.

    Returns:
        One row per exact (ticker, date) match. Empty if none match.
    """
    promoted = whiteboard_df[whiteboard_df["outcome"] == "Initiation"]
    return promoted.merge(
        active_df,
        left_on=["ticker", "wbr_date"],
        right_on=["ticker", "initiation_date"],
        how="inner",
        suffixes=("_wb", "_act"),
    )


def chain_whiteboard_position(bridge_df: pd.DataFrame) -> pd.DataFrame:
    """Add chained WBA->position-close measures to a whiteboard bridge.

    chained_relative_spy_performance = (1+wb_leg)*(1+act_leg) - 1 (an
    approximation — relative returns don't compound the way total returns
    do, see PHASE4A_SCOPE.md section 7). chained_duration_days sums both
    legs' durations. Both are NaN (row not dropped) wherever either leg's
    value is null — callers filter to the both-legs-populated subset as
    needed (summarize_whiteboard_chained does this via dropna).

    Args:
        bridge_df: Output of build_whiteboard_bridge.

    Returns:
        bridge_df with two additional columns.
    """
    df = bridge_df.copy()
    both_perf = df["relative_spy_performance_wb"].notna() & df["relative_spy_performance_act"].notna()
    df["chained_relative_spy_performance"] = np.nan
    df.loc[both_perf, "chained_relative_spy_performance"] = (
        (1 + df.loc[both_perf, "relative_spy_performance_wb"]) * (1 + df.loc[both_perf, "relative_spy_performance_act"]) - 1
    )

    both_duration = df["duration_days_wb"].notna() & df["duration_days_act"].notna()
    df["chained_duration_days"] = np.nan
    df.loc[both_duration, "chained_duration_days"] = (
        df.loc[both_duration, "duration_days_wb"] + df.loc[both_duration, "duration_days_act"]
    )
    return df


def summarize_overall(df: pd.DataFrame, performance_cols: list) -> pd.DataFrame:
    """One row per performance column: n, mean, median, hit_rate.

    Args:
        df: Any population (already filtered to the scope of interest,
            e.g. Closed positions).
        performance_cols: Columns to summarize.

    Returns:
        DataFrame with columns measure/n/mean/median/hit_rate. n == 0
        yields NaN mean/median/hit_rate for that row, not an exception.
    """
    rows = []
    for col in performance_cols:
        s = df[col].dropna()
        n = len(s)
        rows.append({
            "measure": col,
            "n": n,
            "mean": s.mean() if n else float("nan"),
            "median": s.median() if n else float("nan"),
            "hit_rate": (s > 0).mean() if n else float("nan"),
        })
    return pd.DataFrame(rows, columns=["measure", "n", "mean", "median", "hit_rate"])


def summarize_by_cut(df: pd.DataFrame, cut_col: str, performance_cols: list, min_n: int = 1) -> tuple:
    """Group summarize_overall's computation by a cut column, and report
    what a plain groupby would silently drop.

    pandas' groupby drops null keys by default — on this file that's 26
    rows on the Setup cut and (restricted to Closed positions) 3 rows on
    the era cut. Returning the excluded population's own summarize_overall
    alongside the grouped table (rather than a bare count) is what lets a
    caller print "n=X; unassigned: Y" AND see what those unassigned rows'
    own performance looked like, not just how many there were.

    Args:
        df: Population to summarize (e.g. Active Closed positions).
        cut_col: Column to group by (e.g. "setup", "era_bucket").
        performance_cols: Columns to summarize within each group.
        min_n: Suppress a (group, measure) row from the grouped table if
            its n is below this — a report-only filter; it does not change
            what summarize_overall(df, ...) itself would compute, and the
            grouped-plus-unassigned invariant below only holds exactly at
            the default min_n=1 (a higher min_n excludes small non-null
            groups from BOTH tables, not just from the report).

    Returns:
        (grouped_df, unassigned_df). grouped_df has columns
        cut_value/measure/n/mean/median/hit_rate. unassigned_df is
        summarize_overall's output over the rows where cut_col is null.
        Invariant at min_n=1: for every measure,
        grouped_df[measure].n.sum() + unassigned_df[measure].n ==
        df[measure].notna().sum().
    """
    populated = df[df[cut_col].notna()]
    unassigned = df[df[cut_col].isna()]

    rows = []
    for cut_value, group_df in populated.groupby(cut_col):
        for col in performance_cols:
            s = group_df[col].dropna()
            n = len(s)
            if n < min_n:
                continue
            rows.append({
                "cut_value": cut_value,
                "measure": col,
                "n": n,
                "mean": s.mean(),
                "median": s.median(),
                "hit_rate": (s > 0).mean(),
            })
    grouped_df = pd.DataFrame(rows, columns=["cut_value", "measure", "n", "mean", "median", "hit_rate"])
    unassigned_df = summarize_overall(unassigned, performance_cols)
    return grouped_df, unassigned_df


def assign_hold_period_bucket(duration_days: pd.Series) -> pd.Series:
    """<90d / 90-180d / 180-365d / 1-2y (365-730d) / 2y+ (730d+). Null input
    -> None (not a bucket), never guessed."""

    def _bucket(days):
        if pd.isna(days):
            return None
        if days < 90:
            return "<90d"
        if days < 180:
            return "90-180d"
        if days < 365:
            return "180-365d"
        if days < 730:
            return "1-2y"
        return "2y+"

    return duration_days.map(_bucket)


def assign_market_cap_bucket(market_cap_m: pd.Series) -> pd.Series:
    """<$1B / $1-5B / $5-20B / $20B+, on a $M-denominated column (this
    project's standing convention). Null input -> None."""

    def _bucket(cap_m):
        if pd.isna(cap_m):
            return None
        if cap_m < 1_000:
            return "<$1B"
        if cap_m < 5_000:
            return "$1-5B"
        if cap_m < 20_000:
            return "$5-20B"
        return "$20B+"

    return market_cap_m.map(_bucket)


def assign_era_bucket(initiation_date: pd.Series) -> pd.Series:
    """1998-2007 / 2008-2012 / 2013-2019 / 2020-2026, by Initiation Date's
    year. NaT input -> None (the ANSS/MFE/GLYT defect rows land here)."""

    def _bucket(dt):
        if pd.isna(dt):
            return None
        year = dt.year
        if 1998 <= year <= 2007:
            return "1998-2007"
        if 2008 <= year <= 2012:
            return "2008-2012"
        if 2013 <= year <= 2019:
            return "2013-2019"
        if 2020 <= year <= 2026:
            return "2020-2026"
        return "other"

    return initiation_date.map(_bucket)


def summarize_era_initiation_counts(active_df: pd.DataFrame) -> tuple:
    """Count ALL Active rows (any status — Closed and Open alike) by era
    bucket, keyed on Initiation Date alone.

    Distinct from the era PERFORMANCE cut used elsewhere in the report
    (which is scoped to Closed positions, for a consistent denominator
    with the Setup/Sector/hold-period/market-cap cuts — a position that
    hasn't closed has no recorded performance to summarize). That scoping
    makes the most recent era's most recent positions invisible in the
    performance table; this function restores that visibility as a plain
    timing count, independent of outcome. Print both, labelled, rather
    than picking one — see PHASE4A_SCOPE.md.

    Args:
        active_df: Cleaned Active sheet (any status).

    Returns:
        (counts, unassigned) — counts is a pandas Series indexed by era
        bucket value (bucket -> row count, non-null buckets only);
        unassigned is the count of rows with a null Initiation Date
        (the ANSS/MFE/GLYT defect rows).
    """
    era = assign_era_bucket(active_df["initiation_date"])
    counts = era.value_counts(dropna=True)
    unassigned = int(era.isna().sum())
    return counts, unassigned


def summarize_whiteboard_naive(whiteboard_df: pd.DataFrame) -> pd.DataFrame:
    """Removed vs Initiation, WBA-to-outcome-date window.

    THIS IS A DOCUMENTED MEASUREMENT ARTIFACT, not a valid comparison —
    see PHASE4A_SCOPE.md section 7 and summarize_whiteboard_chained. A
    Removed idea's window runs WBA -> its actual removal; an Initiation
    idea's window stops at promotion, the day it becomes a position, so
    the two windows terminate on different events and are not comparable.
    This function exists so the artifact can be shown and immediately
    corrected in the same report — never print its output without
    summarize_whiteboard_chained alongside it.

    Args:
        whiteboard_df: Cleaned Whiteboard sheet.

    Returns:
        DataFrame with one row per outcome ("Removed", "Initiation"):
        outcome/n/median_relative_spy_performance/hit_rate/median_window_days.
    """
    rows = []
    for outcome in ["Removed", "Initiation"]:
        group = whiteboard_df[whiteboard_df["outcome"] == outcome]
        perf = group["relative_spy_performance"].dropna()
        window_days = (group["wbr_date"] - group["wba_date"]).dt.days.dropna()
        rows.append({
            "outcome": outcome,
            "n": len(perf),
            "median_relative_spy_performance": perf.median() if len(perf) else float("nan"),
            "hit_rate": (perf > 0).mean() if len(perf) else float("nan"),
            "median_window_days": window_days.median() if len(window_days) else float("nan"),
        })
    return pd.DataFrame(rows)


def summarize_whiteboard_chained(bridge_df: pd.DataFrame, whiteboard_df: pd.DataFrame) -> tuple:
    """The corrected whiteboard comparison — see summarize_whiteboard_naive.

    Promoted ideas are measured WBA -> position close (chaining both
    legs); Removed ideas are measured WBA -> removal, restricted to the
    same performance-bearing rows used for the median/hit-rate figures
    (n=71) — NOT summarize_whiteboard_naive's looser "any row with a
    valid WBA/WBR date" population (n=80), which is why this table's
    Removed row shows a different median day count (231) than the naive
    table's (259) for what looks like the same group. This is a
    descriptive reconciliation, not a test —
    n=12 on the promoted side, 8 of 24 promotions still open and
    excluded, chaining two relative returns multiplicatively is an
    approximation, the Active leg is benchmarked against SPX/S5-sector
    indices while the Whiteboard leg is benchmarked against SPY/sector
    ETFs (a small systematic bias, not corrected here — see the module
    docstring), and every input is rounded to a whole percentage point.
    Callers printing this MUST state all of these caveats, not just the
    headline numbers — see _print_report.

    Args:
        bridge_df: Output of build_whiteboard_bridge.
        whiteboard_df: Cleaned Whiteboard sheet (for the Removed group,
            which isn't part of the bridge).

    Returns:
        (promoted_summary, removed_summary), each a one-row DataFrame with
        columns group/n/median_relative_spy_performance/hit_rate/median_days.
    """
    chained = chain_whiteboard_position(bridge_df)
    promoted = chained.dropna(subset=["chained_relative_spy_performance", "chained_duration_days"])
    promoted_summary = pd.DataFrame([{
        "group": "Promoted, chained WBA->position close",
        "n": len(promoted),
        "median_relative_spy_performance": promoted["chained_relative_spy_performance"].median() if len(promoted) else float("nan"),
        "hit_rate": (promoted["chained_relative_spy_performance"] > 0).mean() if len(promoted) else float("nan"),
        "median_days": promoted["chained_duration_days"].median() if len(promoted) else float("nan"),
    }])

    removed = whiteboard_df[whiteboard_df["outcome"] == "Removed"]
    # Restricted to the SAME rows as the performance metric (relative_spy_
    # performance non-null, n=71) — deliberately NOT summarize_whiteboard_
    # naive's looser convention (any row with a valid WBA/WBR date, n=80,
    # median 259 days regardless of whether performance is recorded).
    # Apples-to-apples with the Promoted row above (which is inherently
    # restricted to rows carrying both legs' performance) requires this
    # narrower population; using the naive table's 80-row/259-day figure
    # here would compare two differently-scoped populations under one
    # table. Verified against the source file: this population's median
    # is 231 days, distinct from — and correctly distinct from — the
    # naive table's 259.
    removed_with_perf = removed[removed["relative_spy_performance"].notna()]
    removed_perf = removed_with_perf["relative_spy_performance"]
    removed_days = (removed_with_perf["wbr_date"] - removed_with_perf["wba_date"]).dt.days.dropna()
    removed_summary = pd.DataFrame([{
        "group": "Removed, WBA->removal",
        "n": len(removed_perf),
        "median_relative_spy_performance": removed_perf.median() if len(removed_perf) else float("nan"),
        "hit_rate": (removed_perf > 0).mean() if len(removed_perf) else float("nan"),
        "median_days": removed_days.median() if len(removed_days) else float("nan"),
    }])
    return promoted_summary, removed_summary


def _json_safe(obj):
    """Recursively normalize a nested dict/list of numpy/float values to
    something json.dumps can encode losslessly — same purpose as
    history.py's _json_safe, reimplemented locally rather than imported
    since that one is private to history.py and this module has no other
    reason to depend on it."""
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if math.isnan(value) else value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class IngestResult:
    """The outcome of one ingest_historical() call.

    Attributes:
        dry_run: True if this was a --dry-run (nothing written).
        active_df, whiteboard_df: The cleaned sheets, carried through so a
            caller (main()) can print the descriptive-analytics report
            without re-reading the database — populated on a dry run too.
        active_rows, whiteboard_rows: Row counts.
        active_sign, whiteboard_sign: Sign-convention gate results.
        active_consistency, whiteboard_consistency: Benchmark-consistency
            check results.
        defects: count_defects' output.
        source_file_name, source_file_mtime_utc, source_file_sha256:
            Provenance of the file this result came from.
        ingested_at_utc: Set only on a real (non-dry-run) write.
    """

    dry_run: bool
    active_df: pd.DataFrame
    whiteboard_df: pd.DataFrame
    active_rows: int
    whiteboard_rows: int
    active_sign: SignCheckResult
    whiteboard_sign: SignCheckResult
    active_consistency: ConsistencyResult
    whiteboard_consistency: ConsistencyResult
    defects: dict
    source_file_name: str
    source_file_mtime_utc: str
    source_file_sha256: str
    ingested_at_utc: str = None


def ingest_historical(
    upload_dir: str = os.path.join("data", "historical"),
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
    dry_run: bool = False,
) -> IngestResult:
    """Read, clean, gate, and (unless dry_run) write the historical workbook.

    Order: find the one .xlsx in upload_dir -> read both sheets -> clean
    both -> sign-convention gate on both (raises SignConventionError and
    writes NOTHING if either fails) -> benchmark-consistency check on both
    (never raises) -> count_defects -> if dry_run, return without writing;
    else write historical_active_shorts/historical_whiteboard_shorts
    (if_exists="replace") and append one historical_ingest_runs row
    (db.append_rows) plus three indexes.

    Args:
        upload_dir: Directory holding the one source .xlsx.
        db_path: Path to the SQLite database file.
        config_path: Path to config.yaml.
        dry_run: If True, gate and report but write nothing.

    Returns:
        IngestResult.

    Raises:
        SignConventionError: If either sheet fails the sign-convention
            gate. Raised before any write, dry_run or not.
    """
    config = load_config(config_path)
    thresholds = config["historical"]

    filepath = find_single_upload_file(upload_dir, ".xlsx")
    provenance = file_provenance(filepath)

    active_raw = pd.read_excel(filepath, sheet_name=ACTIVE_SHEET)
    whiteboard_raw = pd.read_excel(filepath, sheet_name=WHITEBOARD_SHEET)

    active_df = clean_active_dataframe(active_raw)
    whiteboard_df = clean_whiteboard_dataframe(whiteboard_raw)

    active_sign = check_sign_convention(
        active_df, "initiation_price", "close_price", "absolute_performance",
        thresholds["sign_convention_min_abs_corr"],
    )
    whiteboard_sign = check_sign_convention(
        whiteboard_df, "wba_price", "wbr_price", "absolute_performance",
        thresholds["sign_convention_min_abs_corr"],
    )
    if not active_sign.passed or not whiteboard_sign.passed:
        raise SignConventionError(
            f"Sign-convention gate failed — import aborted before any write. "
            f"active: corr={active_sign.corr} n={active_sign.n} passed={active_sign.passed}; "
            f"whiteboard: corr={whiteboard_sign.corr} n={whiteboard_sign.n} passed={whiteboard_sign.passed}; "
            f"both must be <= -{thresholds['sign_convention_min_abs_corr']}."
        )

    active_consistency = check_benchmark_consistency(
        active_df, "initiation_price", "close_price", "benchmark_at_initiation",
        "benchmark_at_close", "relative_spy_performance", thresholds["benchmark_consistency_tolerance"],
    )
    whiteboard_consistency = check_benchmark_consistency(
        whiteboard_df, "wba_price", "wbr_price", "benchmark_at_wba",
        "benchmark_at_wbr", "relative_spy_performance", thresholds["benchmark_consistency_tolerance"],
    )

    defects = count_defects(active_df, whiteboard_df, active_consistency, whiteboard_consistency)

    result = IngestResult(
        dry_run=dry_run,
        active_df=active_df,
        whiteboard_df=whiteboard_df,
        active_rows=len(active_df),
        whiteboard_rows=len(whiteboard_df),
        active_sign=active_sign,
        whiteboard_sign=whiteboard_sign,
        active_consistency=active_consistency,
        whiteboard_consistency=whiteboard_consistency,
        defects=defects,
        source_file_name=provenance["name"],
        source_file_mtime_utc=provenance["mtime_utc"],
        source_file_sha256=provenance["sha256"],
    )

    if dry_run:
        return result

    ingested_at = datetime.now(timezone.utc)
    result.ingested_at_utc = ingested_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    engine = create_engine(f"sqlite:///{db_path}")
    active_df.to_sql("historical_active_shorts", engine, if_exists="replace", index=False)
    whiteboard_df.to_sql("historical_whiteboard_shorts", engine, if_exists="replace", index=False)

    run_row = {
        "ingested_at_utc": result.ingested_at_utc,
        "source_file_name": provenance["name"],
        "source_file_mtime_utc": provenance["mtime_utc"],
        "source_file_sha256": provenance["sha256"],
        "active_rows": result.active_rows,
        "whiteboard_rows": result.whiteboard_rows,
        "active_sign_corr": active_sign.corr,
        "active_sign_n": active_sign.n,
        "whiteboard_sign_corr": whiteboard_sign.corr,
        "whiteboard_sign_n": whiteboard_sign.n,
        "active_benchmark_consistency_violations": active_consistency.violation_count,
        "whiteboard_benchmark_consistency_violations": whiteboard_consistency.violation_count,
        "defects_json": json.dumps(_json_safe(defects)),
    }
    append_rows(engine, pd.DataFrame([run_row]), "historical_ingest_runs")

    create_index_if_not_exists(
        engine, "idx_historical_ingest_runs_ingested_at", "historical_ingest_runs", ["ingested_at_utc"]
    )
    create_index_if_not_exists(engine, "idx_historical_active_shorts_ticker", "historical_active_shorts", ["ticker"])
    create_index_if_not_exists(
        engine, "idx_historical_whiteboard_shorts_ticker", "historical_whiteboard_shorts", ["ticker"]
    )

    return result


def _print_summary_table(title: str, summary_df: pd.DataFrame) -> None:
    print(f"\n{title}")
    for _, row in summary_df.iterrows():
        mean = row.get("mean")
        mean_str = f"mean={mean * 100:.1f}%  " if mean is not None and pd.notna(mean) else ""
        median = row["median"]
        hit = row["hit_rate"]
        label = row.get("measure", row.get("cut_value"))
        median_str = f"{median * 100:.1f}%" if pd.notna(median) else "n/a"
        hit_str = f"{hit * 100:.1f}%" if pd.notna(hit) else "n/a"
        print(f"  {label}: n={row['n']}  {mean_str}median={median_str}  hit={hit_str}")


def _print_cut(name: str, df: pd.DataFrame, cut_col: str) -> None:
    grouped_df, unassigned_df = summarize_by_cut(df, cut_col, PERFORMANCE_COLUMNS)
    print(f"\n--- By {name} ---")
    for cut_value, sub in grouped_df.groupby("cut_value"):
        _print_summary_table(f"{cut_value}", sub)
    unassigned_n = {row["measure"]: row["n"] for _, row in unassigned_df.iterrows()}
    print(f"  unassigned ({cut_col} missing): {unassigned_n}")


def _print_report(result: IngestResult) -> None:
    print("\n" + "=" * 70)
    print("HISTORICAL INGEST REPORT")
    print("=" * 70)
    print(f"Source: {result.source_file_name} (mtime {result.source_file_mtime_utc}, sha256 {result.source_file_sha256})")
    print(f"Active rows: {result.active_rows}  Whiteboard rows: {result.whiteboard_rows}")
    print(f"Sign-convention gate: active corr={result.active_sign.corr:.5f} n={result.active_sign.n} "
          f"passed={result.active_sign.passed} | whiteboard corr={result.whiteboard_sign.corr:.5f} "
          f"n={result.whiteboard_sign.n} passed={result.whiteboard_sign.passed}")

    d = result.defects
    print("\n--- Data defects (all counted, none abort the import) ---")
    print(f"1. Closed with no Close Date: {len(d['closed_no_close_date'])} {d['closed_no_close_date']}")
    print(f"2. Open with Absolute Performance: {len(d['open_with_performance'])} {d['open_with_performance']}")
    print(f"3. Setup missing (Active): {d['setup_missing_count']}")
    print(f"4. Whiteboard Duration='Error': {len(d['whiteboard_duration_error_tickers'])} {d['whiteboard_duration_error_tickers']}")
    print(f"5. Outcome missing (Whiteboard): {d['outcome_missing_count']}")
    print(f"6. Whiteboard sector coverage: {d['whiteboard_sector_coverage']}")
    print("7. Mixed/labelled benchmark instruments:")
    print(f"   benchmark_instrument: {d['benchmark_instrument']}")
    print(f"   sector_benchmark_instrument: {d['sector_benchmark_instrument']}")
    print(f"   benchmark_consistency (the real guard, see config.yaml historical block): {d['benchmark_consistency']}")
    print(f"Observed property (not a defect): performance values rounded to whole percentage "
          f"points: {d['performance_values_rounded_to_whole_percent']} "
          f"— sub-point differences anywhere in this report are not meaningful.")
    print(f"Observed property (not a defect): market cap exactly 0.0 in the source "
          f"(assign_market_cap_bucket would otherwise silently sort these into \"<$1B\"): "
          f"{d['market_cap_zero']}")

    active_df = result.active_df
    whiteboard_df = result.whiteboard_df
    closed = active_df[active_df["status"] == "Closed"].copy()
    closed["hold_period_bucket"] = assign_hold_period_bucket(closed["duration_days"])
    closed["market_cap_bucket"] = assign_market_cap_bucket(closed["market_cap_at_initiation"])
    closed["era_bucket"] = assign_era_bucket(closed["initiation_date"])

    _print_summary_table("Active, Closed only — overall", summarize_overall(closed, PERFORMANCE_COLUMNS))
    _print_cut("Setup", closed, "setup")
    _print_cut("Sector", closed, "sector")
    _print_cut("Hold-period bucket", closed, "hold_period_bucket")
    _print_cut("Market-cap bucket", closed, "market_cap_bucket")
    _print_cut("Era (performance, Closed only — n=441 denominator, same as every cut above)", closed, "era_bucket")

    era_counts, era_unassigned = summarize_era_initiation_counts(active_df)
    print("\n--- Era, initiation-count distribution (ALL Active rows, any status — n=453 "
          "denominator, a timing count independent of outcome) ---")
    for era_value, count in era_counts.sort_index().items():
        print(f"  {era_value}: {count}")
    print(f"  unassigned (initiation_date missing): {era_unassigned}")

    print("\n--- Whiteboard: naive comparison — MEASUREMENT ARTIFACT, see chained figures below ---")
    print(summarize_whiteboard_naive(whiteboard_df).to_string(index=False))

    bridge = build_whiteboard_bridge(active_df, whiteboard_df)
    promoted_summary, removed_summary = summarize_whiteboard_chained(bridge, whiteboard_df)
    print("\n--- Whiteboard: chained comparison (WBA -> true end of life) ---")
    print(pd.concat([promoted_summary, removed_summary], ignore_index=True).to_string(index=False))
    print(
        "  Note: the Removed row's median days (231) differs from the naive table's (259) "
        "for what looks like the same group — this table restricts to the same "
        "performance-bearing rows (n=71) used for the median/hit-rate figures; the naive "
        "table's day count instead uses every row with a valid WBA/WBR date (n=80), "
        "independent of whether performance was recorded. Not a discrepancy to fix."
    )
    print(
        "  Caveats: n=12 promoted (8 of 24 promotions still Open, excluded — censoring may "
        "flatter this group); chaining two relative returns multiplicatively is an "
        "approximation, not correct compounding; the Active leg is benchmarked against "
        "SPX/S5-sector indices while the Whiteboard leg is benchmarked against SPY/sector "
        "ETFs (a small systematic bias, ~dividend drag, not corrected); the removal date is "
        "discretionary; and every input is rounded to a whole percentage point. This is a "
        "descriptive reconciliation, not a test of selection skill."
    )
    print("=" * 70)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the historical position-outcomes workbook (Phase 4a). Not a screen."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Gate and report; write nothing to the database."
    )
    args = parser.parse_args(argv)

    try:
        result = ingest_historical(dry_run=args.dry_run)
    except SignConventionError as exc:
        logger.error(str(exc))
        sys.exit(1)

    _print_report(result)
    if args.dry_run:
        print("\nDRY RUN — nothing written.")


if __name__ == "__main__":
    main()
