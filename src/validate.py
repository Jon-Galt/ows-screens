"""
Pure validation checks for the refresh gate, scoped by screen.

These functions take pandas DataFrames and plain thresholds in and return
findings out — no SQLAlchemy, no Streamlit, no file IO. src/refresh.py owns
reading the incoming upload and the currently stored table; this module only
judges the two DataFrames it's handed. Same discipline as transform.py/
score.py/overlap.py under Architecture Rule 1.

A screen's very first run has no stored table to compare against. Every
check that compares incoming data to a stored baseline treats a missing
baseline (stored_df is None) as "nothing to diff against," not as a failure.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Finding:
    """One validation check's failure detail.

    Attributes:
        check: Short name of the check that produced this finding, e.g.
            "row_count", "composition_misfile", "null_rate_spike", "no_space_tickers".
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


def normalize_ticker_set(series: pd.Series) -> set:
    """Build a comparison-ready ticker set from a column.

    Used identically by check_composition_misfile (on incoming data) and by
    refresh.py's read_stored_ticker_sets (on each screen's stored baseline)
    so the two sides of every Jaccard comparison can never silently drift
    apart on dtype or whitespace — a mismatch there would make every score
    0.0 and the check inert while looking healthy.

    Args:
        series: A ticker column (incoming or stored).

    Returns:
        The set of non-null tickers, cast to str and stripped.
    """
    return set(series.dropna().astype(str).str.strip())


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two ticker sets. Callers here only ever pass
    two non-empty sets, so the union is never empty."""
    return len(a & b) / len(a | b)


def check_composition_misfile(
    incoming_df: pd.DataFrame,
    screen_id: str,
    baseline_tickers: dict,
    ticker_col: str = "ticker",
) -> Finding | None:
    """Flag an incoming export whose ticker composition matches another
    screen's stored baseline better than it matches its own.

    Curated exports are screen-anonymous by standing decision — identity
    comes from folder placement only — so the real risk this guards against
    is an export landing in the wrong screen's upload folder and silently
    overwriting it with another screen's names. Deliberately threshold-free:
    flags whenever ANY other screen's stored ticker set is a strictly better
    match (by Jaccard similarity) than this screen's own. A tie does not
    flag.

    Args:
        incoming_df: The screen's freshly cleaned incoming data.
        screen_id: The screen this incoming data is being ingested as.
        baseline_tickers: {screen_id: set[str]} — every registry screen's
            currently stored ticker set, frozen before any screen in this
            run has been written (see refresh.py's read_stored_ticker_sets).
        ticker_col: Name of the ticker column.

    Returns:
        A Finding naming the best-matching peer and both Jaccard scores if
        one strictly beats this screen's own score, else None. Also None if
        this screen has no baseline yet (first run) or incoming_df has no
        tickers (check_row_count's concern, not this check's).
    """
    own = baseline_tickers.get(screen_id) or set()
    if not own:
        return None

    incoming = normalize_ticker_set(incoming_df[ticker_col])
    if not incoming:
        return None

    own_score = _jaccard(incoming, own)

    best_peer_id, best_peer_score = None, -1.0
    for peer_id, peer_set in baseline_tickers.items():
        if peer_id == screen_id or not peer_set:
            continue
        peer_score = _jaccard(incoming, peer_set)
        if peer_score > best_peer_score:
            best_peer_id, best_peer_score = peer_id, peer_score

    if best_peer_id is None or best_peer_score <= own_score:
        return None

    return Finding(
        "composition_misfile",
        f"Incoming ticker set matches {best_peer_id!r}'s stored baseline "
        f"(Jaccard {best_peer_score:.3f}) better than its own ({screen_id!r}, "
        f"Jaccard {own_score:.3f}) — possible export placed in the wrong folder.",
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
    screen_id: str,
    baseline_tickers: dict,
) -> ValidationResult:
    """Run every check for one screen and collect the results.

    Args:
        incoming_df: The screen's freshly cleaned incoming data.
        stored_df: The screen's currently stored table, or None if this
            screen has never been ingested before.
        thresholds: The config.yaml "refresh" block — must contain
            null_rate_max_increase_pct.
        screen_id: The screen this incoming data is being ingested as.
        baseline_tickers: {screen_id: set[str]} — every registry screen's
            currently stored ticker set, frozen before this run's writes.
            See check_composition_misfile.

    Returns:
        A ValidationResult with passed=True and no findings if every check
        cleared, else passed=False with every finding that fired.
    """
    findings = []

    row_count_finding = check_row_count(incoming_df)
    if row_count_finding is not None:
        findings.append(row_count_finding)

    composition_finding = check_composition_misfile(incoming_df, screen_id, baseline_tickers)
    if composition_finding is not None:
        findings.append(composition_finding)

    findings.extend(
        check_null_rate_spike(incoming_df, stored_df, thresholds["null_rate_max_increase_pct"])
    )

    ticker_finding = check_no_space_tickers(incoming_df)
    if ticker_finding is not None:
        findings.append(ticker_finding)

    return ValidationResult(passed=len(findings) == 0, findings=findings)
