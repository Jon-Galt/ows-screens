"""
Excel-style all-column filtering (Phase 8c-6).

Pure — no Streamlit, no SQLAlchemy (Architecture Rule 1, same discipline as
src/selection.py, src/overlap.py and src/weighting.py). The UI half (popovers,
active-filter chips, session_state wiring, the pending-flag-then-st.rerun()
add/remove/clear discipline) lives in src/app.py's render_column_filter_bar;
this module only classifies a column's control kind and applies a set of
filter specs to a DataFrame.

Dtype drives the control, one rule, no per-column special cases: bool ->
values, other numeric -> range, everything else -> values. A bool column
(overvalued_flag/transcripts_flag/in_universe) is numeric per
pandas.api.types.is_numeric_dtype, so it is carved out explicitly rather than
falling into "range" and rendering a 0..1 slider for a binary flag (PM ruling,
Phase 8c-6 plan review) — mscore_flag stays int64 0/1 and therefore stays a
range control, a deliberate, visible inconsistency the PM asked to be flagged
in the build report rather than "fixed" with a cardinality-based rule (which
would make a control's shape depend on the data rather than its dtype).
"""

from typing import Literal

import pandas as pd

FilterKind = Literal["range", "values"]


def classify_filter_kind(series: pd.Series) -> FilterKind:
    """Which control kind a column's filter should use.

    Args:
        series: The column to classify (e.g. df[column]).

    Returns:
        "values" for a bool-dtype column or any non-numeric column;
        "range" for every other numeric column.
    """
    if pd.api.types.is_bool_dtype(series):
        return "values"
    if pd.api.types.is_numeric_dtype(series):
        return "range"
    return "values"


def get_filterable_columns(df: pd.DataFrame) -> list[str]:
    """Every column in df, in its own order — the full frame, never a
    display or export subset (Driver ruling R1, Phase 8c-6).

    Args:
        df: The screen's own frame.

    Returns:
        list(df.columns).
    """
    return list(df.columns)


def get_column_bounds(df: pd.DataFrame, column: str) -> tuple[float, float]:
    """A numeric column's (min, max), NaN-safe (Architecture Rule 3).

    Args:
        df: The frame to read from.
        column: The column to bound.

    Returns:
        (min, max) as floats, coercing non-numeric values to NaN first so a
        stray bad value can't raise.
    """
    series = pd.to_numeric(df[column], errors="coerce")
    return float(series.min()), float(series.max())


def get_column_values(df: pd.DataFrame, column: str) -> list:
    """A column's distinct non-null values, sorted.

    Args:
        df: The frame to read from.
        column: The column to enumerate.

    Returns:
        Sorted list of df[column]'s distinct non-null values. Cardinality-
        blind by design (Phase 8c-6 plan) — a 1,358-option ticker/name column
        returns just as plainly as a 10-option sector column; the UI layer
        decides how to present a long list, this function does not.
    """
    return sorted(df[column].dropna().unique().tolist())


def apply_column_filters(
    df: pd.DataFrame, filters: dict[str, tuple[FilterKind, object]]
) -> pd.DataFrame:
    """Apply every active column filter to df, AND-combined (Driver ruling
    R3, Phase 8c-6).

    Args:
        df: The frame to filter.
        filters: {column: ("range", (lo, hi))} or
            {column: ("values", [v1, v2, ...])}. A column absent from df is
            skipped (no restriction) rather than raising — the UI layer
            never constructs one, but a stale filter surviving a screen
            switch must not crash the next screen. An empty "values"
            selection also contributes no restriction, matching
            render_sidebar's existing "nothing selected == no filter"
            convention rather than inventing a second one.

    Returns:
        df restricted to rows passing every active filter. A "range" filter
        excludes NaN rows via pandas' own NaN-safe .between (the same
        NaN-drops-the-row behavior render_sidebar's market-cap/overall-score
        sliders already have); a "values" filter excludes NaN via .isin,
        which never matches NaN, for the same reason. Never raises on a
        combination matching zero rows — returns an empty frame.
    """
    mask = pd.Series(True, index=df.index)
    for column, (kind, value) in filters.items():
        if column not in df.columns:
            continue
        if kind == "range":
            lo, hi = value
            mask &= df[column].between(lo, hi)
        elif kind == "values":
            if not value:
                continue
            mask &= df[column].isin(value)
    return df[mask]
