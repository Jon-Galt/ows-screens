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


def resolve_nav_target(filtered: pd.DataFrame, ticker: str) -> tuple:
    """Whether a cross-screen navigation's target ticker survives the
    destination screen's active filters (Phase 5b-2).

    Gates what the caller is allowed to seed into a drill-down's ticker_key
    session-state entry, *before* resolve_selected_ticker ever runs. Without
    this gate, a navigated-to ticker excluded by the destination screen's
    sidebar filters would fail resolve_selected_ticker's precedence-2 branch
    (previous_ticker, if still present) and silently fall through to
    precedence 3 (the first ticker in display order) — a real, different,
    plausible company, with no error. This function exists so that failure
    mode is caught here instead, never inside resolve_selected_ticker itself
    (its precedence order is correct for the case it was written for and is
    not changed by this).

    Args:
        filtered: The destination screen's sidebar-filtered frame (same
            ticker set as the display_df eventually built from it).
        ticker: The ticker a cross-screen click navigated to.

    Returns:
        ("show", ticker) if ticker survives filtered's ticker set — safe to
        seed ticker_key, since resolve_selected_ticker's precedence-2 branch
        is then guaranteed to succeed.
        ("blocked", ticker) otherwise — the caller must show an explicit
        notice and must NOT seed ticker_key (leaving precedence 3 free to
        apply normally, to whatever it would have resolved to absent this
        navigation, rather than being blamed on the nav).
    """
    if (filtered["ticker"] == ticker).any():
        return ("show", ticker)
    return ("blocked", ticker)


def is_fresh_selection(pre_rows: list, last_rows) -> bool:
    """True iff `pre_rows` differs from the last-synced selection state
    (Phase 5b-2) — the line between a genuine new row click and a sticky
    rerun carrying forward a selection already accounted for.

    Both sync_drilldown_selection (app.py) and the overlap table's own
    click-detection need exactly this distinction — extracted as one shared
    predicate rather than left as two copies of the same check that could
    silently diverge as more selection-bearing tables are added.

    Args:
        pre_rows: The table's current raw selection-state row list.
        last_rows: The row list this table's selection was last synced
            against (None if never synced).

    Returns:
        pre_rows != last_rows.
    """
    return pre_rows != last_rows


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
