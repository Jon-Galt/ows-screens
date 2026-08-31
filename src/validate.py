"""
Pure validation checks for the refresh gate, scoped by screen.

These functions take pandas DataFrames and plain thresholds in and return
findings out — no SQLAlchemy, no Streamlit, no file IO. src/refresh.py owns
reading the incoming upload and the currently stored table; this module only
judges the two DataFrames it's handed. Same discipline as transform.py/
score.py/overlap.py under Architecture Rule 1.

A screen's very first run has no stored table to compare against. Every
check that compares incoming data to a stored baseline treats a missing
baseline (stored_df is None) — and, for check_universe_delta specifically,
an empty-but-present baseline (stored_df has zero rows, which the per-screen
ingest functions can still produce since they stay directly callable with
no gate in front of them) — as "nothing to diff against," not as a failure.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Finding:
    """One validation check's failure detail.

    Attributes:
        check: Short name of the check that produced this finding, e.g.
            "row_count", "universe_delta", "null_rate_spike", "no_space_tickers".
        message: Human-readable detail for the run report.
    """

    check: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating one screen's incoming data.

    Attributes:
        passed: True if no check produced a Finding.
        findings: All findings from all checks, in the order they ran.
            Empty when passed is True.
    """

    passed: bool
    findings: list = field(default_factory=list)


def check_row_count(incoming_df: pd.DataFrame) -> Finding | None:
    """Flag an incoming DataFrame with zero rows.

    Args:
        incoming_df: The screen's freshly cleaned incoming data.

    Returns:
        A Finding if incoming_df is empty, else None.
    """
    if len(incoming_df) == 0:
        return Finding("row_count", "Incoming data has 0 rows.")
    return None


def check_universe_delta(
    incoming_df: pd.DataFrame,
    stored_df: pd.DataFrame | None,
    max_delta_pct: float,
    max_delta_abs: int,
) -> Finding | None:
    """Flag a universe size change beyond tolerance versus the stored table.

    Passes automatically when there's no baseline to compare against: no
    stored table yet (first run for this screen), or a stored table with
    zero rows (nothing meaningful to diff against, and computing a
    percentage against zero would raise ZeroDivisionError).

    The check passes if EITHER the percentage delta OR the absolute delta
    is within tolerance. A flat percentage alone is too tight for small
    screens (e.g. management_comp at 21 rows, where a 5-ticker change is
    23.8%) — the absolute floor covers ordinary turnover on small universes
    without weakening the percentage rule's ability to catch large-scale
    corruption on bigger screens.

    Args:
        incoming_df: The screen's freshly cleaned incoming data.
        stored_df: The screen's currently stored table, or None if this
            screen has never been ingested before.
        max_delta_pct: Maximum allowed fractional change in row count
            (e.g. 0.20 for 20%) before this alone would fail the check.
        max_delta_abs: Maximum allowed absolute change in row count before
            this alone would fail the check.

    Returns:
        A Finding if both the percentage and absolute deltas exceed their
        thresholds, else None.
    """
    if stored_df is None or len(stored_df) == 0:
        return None

    delta = abs(len(incoming_df) - len(stored_df))
    delta_pct = delta / len(stored_df)

    if delta_pct <= max_delta_pct or delta <= max_delta_abs:
        return None

    return Finding(
        "universe_delta",
        f"Universe size changed by {delta} rows ({delta_pct:.1%}), "
        f"from {len(stored_df)} to {len(incoming_df)} — exceeds tolerance "
        f"of {max_delta_pct:.0%} or {max_delta_abs} rows.",
    )


def check_null_rate_spike(
    incoming_df: pd.DataFrame,
    stored_df: pd.DataFrame | None,
    max_increase_pct: float,
) -> list:
    """Flag columns whose null rate rose beyond tolerance versus stored data.

    Passes automatically (no findings) when there's no stored baseline
    (first run for this screen). Only columns present in both frames are
    compared — a column added or dropped between runs is not this check's
    concern.

    Args:
        incoming_df: The screen's freshly cleaned incoming data.
        stored_df: The screen's currently stored table, or None if this
            screen has never been ingested before.
        max_increase_pct: Maximum allowed increase in a column's null rate
            (percentage points, e.g. 0.15 for 15 points) before it's flagged.

    Returns:
        A Finding per column whose null rate increased beyond tolerance.
        Empty list if none did, or if stored_df is None.
    """
    if stored_df is None:
        return []

    findings = []
    shared_columns = set(incoming_df.columns) & set(stored_df.columns)
    for col in sorted(shared_columns):
        incoming_rate = incoming_df[col].isnull().mean()
        stored_rate = stored_df[col].isnull().mean()
        increase = incoming_rate - stored_rate
        if increase > max_increase_pct:
            findings.append(
                Finding(
                    "null_rate_spike",
                    f"Column {col!r} null rate rose from {stored_rate:.1%} "
                    f"to {incoming_rate:.1%} (+{increase:.1%}) — exceeds "
                    f"tolerance of {max_increase_pct:.0%}.",
                )
            )
    return findings


def check_no_space_tickers(incoming_df: pd.DataFrame, ticker_col: str = "ticker") -> Finding | None:
    """Flag any ticker containing a space (Phase 3c.1's bug, re-checked every run).

    Args:
        incoming_df: The screen's freshly cleaned incoming data.
        ticker_col: Name of the ticker column.

    Returns:
        A Finding naming the offending tickers if any ticker contains a
        space, else None.
    """
    tickers = incoming_df[ticker_col].dropna().astype(str)
    with_space = sorted(tickers[tickers.str.contains(" ")].unique())
    if with_space:
        return Finding(
            "no_space_tickers",
            f"{len(with_space)} ticker(s) contain a space: {with_space}",
        )
    return None


def validate_screen(
    incoming_df: pd.DataFrame,
    stored_df: pd.DataFrame | None,
    thresholds: dict,
) -> ValidationResult:
    """Run every check for one screen and collect the results.

    Args:
        incoming_df: The screen's freshly cleaned incoming data.
        stored_df: The screen's currently stored table, or None if this
            screen has never been ingested before.
        thresholds: The config.yaml "refresh" block — must contain
            universe_size_max_delta_pct, universe_size_max_delta_abs, and
            null_rate_max_increase_pct.

    Returns:
        A ValidationResult with passed=True and no findings if every check
        cleared, else passed=False with every finding that fired.
    """
    findings = []

    row_count_finding = check_row_count(incoming_df)
    if row_count_finding is not None:
        findings.append(row_count_finding)

    universe_finding = check_universe_delta(
        incoming_df,
        stored_df,
        thresholds["universe_size_max_delta_pct"],
        thresholds["universe_size_max_delta_abs"],
    )
    if universe_finding is not None:
        findings.append(universe_finding)

    findings.extend(
        check_null_rate_spike(incoming_df, stored_df, thresholds["null_rate_max_increase_pct"])
    )

    ticker_finding = check_no_space_tickers(incoming_df)
    if ticker_finding is not None:
        findings.append(ticker_finding)

    return ValidationResult(passed=len(findings) == 0, findings=findings)
