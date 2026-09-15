"""
Phase 5b-2 — pure cross-screen context computation behind the drill-down's
"also appears on" section.

Pandas only, no Streamlit/SQLAlchemy (Architecture Rule 1) — same standing as
src/selection.py, src/styling.py, and overlap.py's own functions. Depends on
overlap.UNIVERSE_SCREEN_ID only (no cycle: overlap.py does not import this
module back).

The population test behind this module's design (see PHASE5B2_PROMPT.md): a
ticker shared across multiple curated screens carries identical identity and
risk-score values on every one of them, but a different `rationale` (always)
and usually a different `stock_performance`. So "also appears on" never
repeats identity — it shows only what actually varies per screen: a curated
screen's rationale + stock performance, an unscored (RSI) screen's derived
metrics, or the universe screen's composite score.

classify_screen is the single taxonomy every loader that needs to resolve a
screen_id to "which per-screen loader/rendering path" shares (see app.py's
`_load_screen_df`, `load_all_screen_identity_data`, `load_screens_for_ticker`)
— one classification, not a fourth independent copy of the same dispatch.

Phase 6c: a second unscored screen (Negative Expert Transcripts) has its own
display shape, unrelated to RSI's. build_screen_contribution's optional
unscored_display_columns_resolver lets app.py supply "what this unscored
screen's own table shows" (already a single source of truth there,
resolve_unscored_display_columns) rather than this module hand-duplicating
a second per-screen column list.
"""

import pandas as pd

from src.overlap import UNIVERSE_SCREEN_ID


def other_screen_ids_for_ticker(
    ticker: str, current_screen_id: str, membership_df: pd.DataFrame
) -> list:
    """Every screen_id `ticker` belongs to (per membership_df, which includes
    the universe screen's own rows), except current_screen_id, sorted.

    Args:
        ticker: The ticker being viewed.
        current_screen_id: The screen whose drill-down is asking — excluded
            from the result even if `ticker` is a member of it.
        membership_df: The full screen_membership table (screen_id, ticker).

    Returns:
        A sorted list of screen_ids, possibly empty.
    """
    ids = membership_df.loc[membership_df["ticker"] == ticker, "screen_id"]
    return sorted(sid for sid in ids.unique() if sid != current_screen_id)


def classify_screen(
    screen_id: str, screens_df: pd.DataFrame, universe_screen_id: str = UNIVERSE_SCREEN_ID
) -> str:
    """The single taxonomy screen_id resolves to, shared by every loader that
    needs to know "which per-screen loader applies here".

    Args:
        screen_id: A screen_id, expected (but not required) to be a row in
            screens_df.
        screens_df: The screens registry (screen_id, display_name,
            screen_type, has_scoring).
        universe_screen_id: The screen treated as context, not a membership
            tick (see overlap.py's module docstring).

    Returns:
        "universe" if screen_id == universe_screen_id.
        "curated" if screen_type == "curated".
        "scored" if screen_type == "quant_composite" and has_scoring.
        "unscored" if screen_type == "quant_composite" and not has_scoring.
        "unknown" if screen_id isn't in screens_df, or its screen_type is
        none of the above — the taxonomy degrading to "no loader applies"
        rather than guessing, matching load_all_screen_identity_data's
        existing documented behavior for an unrecognized type.
    """
    if screen_id == universe_screen_id:
        return "universe"
    match = screens_df.loc[screens_df["screen_id"] == screen_id]
    if match.empty:
        return "unknown"
    row = match.iloc[0]
    if row["screen_type"] == "curated":
        return "curated"
    if row["screen_type"] == "quant_composite" and row["has_scoring"]:
        return "scored"
    if row["screen_type"] == "quant_composite" and not row["has_scoring"]:
        return "unscored"
    return "unknown"


# The RSI/unscored metric columns worth surfacing in an "also appears on"
# contribution — identity (ticker/name/market_cap) is deliberately excluded,
# per this module's docstring: identity is shown once by the caller, never
# repeated per screen. This is also the DEFAULT used by build_screen_
# contribution when no unscored_display_columns_resolver is given (Phase
# 6c), so every pre-6c caller/test keeps its exact current behavior.
_UNSCORED_METRIC_COLUMNS = [
    "adv", "short_interest_pct", "si_change_3m", "si_change_6m",
    "week_52_high_chg", "ev_sales", "debt_ebitda",
]

# Identity columns never shown as a metric — the caller (app.py) renders
# identity once, up front; see this module's docstring.
_IDENTITY_COLUMNS = {"ticker", "name", "market_cap"}

# Phase 8b — the required column set for a raw-detail frame to carry
# transcript takeaways into "also appears on". The gate is on THIS set,
# never on a screen_id literal (T40's recorded failure: a table-existence
# gate handed transcripts_for_ticker an RSI/short_screen frame and raised
# on the missing sort columns). ticker is needed to filter to one ticker,
# transcript_date/doc_id for the ordering contract, theme/key_takeaways for
# the entry itself.
_TRANSCRIPT_DETAIL_COLUMNS = {"ticker", "transcript_date", "doc_id", "key_takeaways", "theme"}


def is_transcript_shaped(df: pd.DataFrame | None) -> bool:
    """Whether df carries every column the transcript-takeaways render path
    needs (Phase 8b).

    Args:
        df: A candidate raw-detail frame, or None.

    Returns:
        False if df is None. Otherwise whether _TRANSCRIPT_DETAIL_COLUMNS
        is a subset of df's columns — true for the Negative Expert
        Transcripts detail table, false for any other screen's raw_data
        frame (e.g. rising_short_interest, short_screen).
    """
    return df is not None and _TRANSCRIPT_DETAIL_COLUMNS.issubset(df.columns)


def build_screen_contribution(
    screen_id: str,
    ticker: str,
    screens_df: pd.DataFrame,
    screen_data: dict,
    universe_screen_id: str = UNIVERSE_SCREEN_ID,
    unscored_display_columns_resolver=None,
    transcript_takeaways: dict | None = None,
) -> dict | None:
    """This one screen's contribution for `ticker`, or None if it has none.

    Args:
        screen_id: The screen to pull a contribution from.
        ticker: The ticker.
        screens_df: The screens registry.
        screen_data: screen_id -> that screen's identity-bearing DataFrame.
            For "universe", this may be a narrow synthetic frame containing
            only `ticker`/`overall_score` (see app.py's
            load_screens_for_ticker) rather than the full scored frame —
            this function doesn't know or care which, since it only ever
            does `df.loc[df["ticker"] == ticker]` against whatever frame it
            is handed.
        universe_screen_id: See classify_screen.
        unscored_display_columns_resolver: Optional
            (screen_id, df) -> list[str] callable (Phase 6c) — the screen's
            OWN display-column list (identity included), supplied by the
            caller (app.py's resolve_unscored_display_columns), so a second
            unscored screen's shape is never hand-duplicated here. Identity
            columns are stripped from whatever it returns before use — that
            exclusion stays owned by this module (see _IDENTITY_COLUMNS),
            never the caller's job. None (the default) falls back to
            _UNSCORED_METRIC_COLUMNS, i.e. today's exact behavior — no
            existing caller's output changes by omitting this argument.
        transcript_takeaways: Optional (Phase 8b) {screen_id: df} of that
            ONE ticker's already-filtered, already-ordered transcript rows
            (app.py resolves and gates this before calling — this module
            never sees the raw multi-ticker detail table or re-derives the
            ordering contract). None, or a missing/empty entry for
            screen_id, leaves the contribution exactly as it was before
            this phase — no existing caller's output changes by omitting
            this argument.

    Returns:
        None if screen_data has no table for screen_id, or ticker isn't a
        row in it (defensive — shouldn't happen for a ticker drawn from
        membership_df, but this function doesn't get to assume that).
        Otherwise a dict with "screen_id", "display_name", "kind", plus
        kind-specific fields:
          - "universe": "overall_score" (float, may be NaN).
          - "curated": "rationale", "stock_performance" (either may be NaN).
          - "unscored": "metrics" (dict of that screen's own derived-metric
            columns present in the row, minus identity, raw values —
            formatting is the caller's job). Plus, Phase 8b: "takeaways" —
            present ONLY when transcript_takeaways supplied a non-empty
            entry for this screen_id; a DataFrame of that one ticker's
            transcript rows, already filtered and ordered by the caller
            (see the transcript_takeaways arg above). Absent, not an empty
            frame, when there is nothing to show — a caller checks for the
            key rather than assuming it.
          - "scored" / "unknown": None (no rendering shape defined for a
            second scored screen or an unrecognized type; nothing today
            reaches this branch, but it degrades to None rather than
            guessing a shape, same posture as classify_screen's "unknown").
    """
    df = screen_data.get(screen_id)
    if df is None:
        return None
    match = df.loc[df["ticker"] == ticker]
    if match.empty:
        return None
    row = match.iloc[0]

    display_name_match = screens_df.loc[screens_df["screen_id"] == screen_id, "display_name"]
    display_name = display_name_match.iloc[0] if not display_name_match.empty else screen_id

    kind = classify_screen(screen_id, screens_df, universe_screen_id)

    if kind == "universe":
        return {
            "screen_id": screen_id,
            "display_name": display_name,
            "kind": kind,
            "overall_score": row.get("overall_score", float("nan")),
        }
    if kind == "curated":
        return {
            "screen_id": screen_id,
            "display_name": display_name,
            "kind": kind,
            "rationale": row.get("rationale"),
            "stock_performance": row.get("stock_performance", float("nan")),
        }
    if kind == "unscored":
        if unscored_display_columns_resolver is not None:
            candidate_cols = unscored_display_columns_resolver(screen_id, df)
        else:
            candidate_cols = _UNSCORED_METRIC_COLUMNS
        metric_cols = [c for c in candidate_cols if c not in _IDENTITY_COLUMNS]
        metrics = {
            col: row[col] for col in metric_cols if col in row.index
        }
        contribution = {
            "screen_id": screen_id,
            "display_name": display_name,
            "kind": kind,
            "metrics": metrics,
        }
        takeaways_df = (
            transcript_takeaways.get(screen_id) if transcript_takeaways else None
        )
        if takeaways_df is not None and not takeaways_df.empty:
            contribution["takeaways"] = takeaways_df
        return contribution
    return None


def build_also_appears_on(
    ticker: str,
    current_screen_id: str,
    membership_df: pd.DataFrame,
    screens_df: pd.DataFrame,
    screen_data: dict,
    universe_screen_id: str = UNIVERSE_SCREEN_ID,
    unscored_display_columns_resolver=None,
    transcript_takeaways: dict | None = None,
) -> list:
    """Every other screen's contribution for `ticker`, sorted by display_name.

    Args:
        ticker: The ticker being viewed.
        current_screen_id: The screen whose drill-down is asking — excluded
            from the result.
        membership_df: The full screen_membership table.
        screens_df: The screens registry.
        screen_data: screen_id -> that screen's identity-bearing DataFrame
            (see build_screen_contribution — may be narrow for "universe").
        universe_screen_id: See classify_screen.
        unscored_display_columns_resolver: See build_screen_contribution —
            threaded straight through, unchanged default.
        transcript_takeaways: See build_screen_contribution — threaded
            straight through, unchanged default.

    Returns:
        A list of build_screen_contribution dicts (None entries dropped),
        sorted by display_name. Empty if `ticker` is on no other screen.
    """
    other_ids = other_screen_ids_for_ticker(ticker, current_screen_id, membership_df)
    contributions = []
    for screen_id in other_ids:
        contribution = build_screen_contribution(
            screen_id, ticker, screens_df, screen_data, universe_screen_id,
            unscored_display_columns_resolver, transcript_takeaways,
        )
        if contribution is not None:
            contributions.append(contribution)
    contributions.sort(key=lambda c: c["display_name"])
    return contributions
