"""
Phase 5b-1 — pure ticker-resolution logic behind the inline drill-down.

Pandas only, no Streamlit/SQLAlchemy (Architecture Rule 1) — same standing
as src/styling.py and overlap.py's style_overlap_table.

`resolve_selected_ticker` decides which stock the drill-down should show,
given the main table's row-selection state. It is intentionally stateless:
it does not know or care *why* `selected_rows` is empty or non-empty — that
judgment (was this rerun caused by a fresh table click, or by something
else that should not override a live selection?) is app.py's job, made
against Streamlit's session_state. See src/app.py's render_*_table
functions for the sync mechanism this feeds.

`find_ticker_row` is the inverse: given a resolved ticker, where does it
sit in today's display_df? Used to re-seed the table's own selection state
so the highlighted row never diverges from the drill-down (see app.py).
"""

import pandas as pd


def resolve_selected_ticker(
    display_df: pd.DataFrame,
    selected_rows: list,
    previous_ticker: str | None = None,
) -> str | None:
    """Resolve which ticker the drill-down should show.

    Precedence:
    1. The first entry of `selected_rows`, if in range and its ticker is
       non-null — a fresh row click (positional into `display_df` exactly
       as passed to st.dataframe, i.e. display order, not sort order or
       filtered-frame order).
    2. `previous_ticker`, if still present in `display_df["ticker"]` — a
       filter/sort change that didn't drop the previously-shown stock must
       not move the drill-down.
    3. The first non-null ticker in `display_df`'s own display order.
    4. `None`, if `display_df` is empty or has no non-null tickers.

    Args:
        display_df: The frame exactly as passed to st.dataframe — already
            sorted, already column-subset. Must contain a "ticker" column.
        selected_rows: Positional row indices from the selection state.
            May be empty. Only the first entry is consulted (this module
            only handles single-row selection).
        previous_ticker: The ticker previously shown in the drill-down, if
            any.

    Returns:
        The resolved ticker, or None if display_df has nothing to show.

    Edge cases: an out-of-range or negative index in `selected_rows` is not
    "in range" (0 <= idx < len(display_df)) and falls through to
    `previous_ticker`/first-ticker rather than raising or wrapping via
    Python's negative-index semantics. If the selected row's own ticker is
    NaN, it is treated the same as an out-of-range index (falls through) —
    this branch is defensive only; no live table has a null ticker.
    """
    if display_df is None or display_df.empty:
        return None
    tickers = display_df["ticker"]
    if tickers.dropna().empty:
        return None
    if selected_rows:
        idx = selected_rows[0]
        if 0 <= idx < len(display_df) and pd.notna(tickers.iloc[idx]):
            return tickers.iloc[idx]
    if previous_ticker is not None and (tickers == previous_ticker).any():
        return previous_ticker
    return tickers.dropna().iloc[0]


def find_ticker_row(display_df: pd.DataFrame, ticker: str | None) -> int | None:
    """Position of `ticker` in `display_df`'s display order, or None if absent.

    Positional (0-based into the frame as it would be passed to
    st.dataframe), not a pandas index lookup — display_df may carry a
    non-contiguous index after a sort_values/filter chain that never reset
    it, and using that index directly would reintroduce the same
    label-vs-position bug this module exists to avoid.

    Args:
        display_df: The frame exactly as passed to st.dataframe.
        ticker: The ticker to locate, or None (returns None immediately —
            resolve_selected_ticker returns None when display_df has
            nothing to show).

    Returns:
        The 0-based display-order position of `ticker`, or None if it's
        not present (or `ticker` is None, or `display_df` is empty).
    """
    if ticker is None or display_df is None or display_df.empty:
        return None
    positions = display_df["ticker"].reset_index(drop=True)
    matches = positions.index[positions == ticker]
    return int(matches[0]) if len(matches) else None
