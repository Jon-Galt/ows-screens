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

`resolve_selected_cell` (Phase 5b-3) is the analogous resolver for a single
cell selection, behind the click-a-cell derivation panel. Unlike
`resolve_selected_ticker` it has no "previous value" precedence chain to
fall back through — see app.py's cell-derivation bookkeeping for why: a
cell selection, once made, is empirically sticky (confirmed against
streamlit 1.63.0 by direct probe — see app.py's module comment near
`render_main_table`), so the caller resolves it once, on the rerun where it
actually changes (per `is_fresh_selection`), and persists the result itself
rather than re-deriving it every rerun.
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


def is_fresh_selection(pre_selection: list, last_selection) -> bool:
    """True iff `pre_selection` differs from the last-synced selection state
    (Phase 5b-2; generalized to cell selections in Phase 5b-3) — the line
    between a genuine new click and a sticky rerun carrying forward a
    selection already accounted for.

    `sync_drilldown_selection` (rows), the overlap table's own row-click
    detection, and the cell-derivation panel's click detection (app.py) all
    need exactly this distinction — extracted as one shared predicate rather
    than left as multiple copies of the same check that could silently
    diverge as more selection-bearing tables/mechanisms are added. The
    parameter names were widened from row-specific ones (`pre_rows`/
    `last_rows`) because the body is unchanged generic list inequality and
    applies identically to a cells selection (a list of (row, column)
    entries) — the caller is responsible for keeping `last_selection`
    consistent with whatever shape `pre_selection` naturally has (for cells,
    that means storing the value read from session_state verbatim, never
    reconstructing it — see app.py's cell-derivation bookkeeping).

    Args:
        pre_selection: The table's current raw selection-state list (rows or
            cells).
        last_selection: The list this table's selection was last synced/
            processed against (None if never synced/processed).

    Returns:
        pre_selection != last_selection.
    """
    return pre_selection != last_selection


def should_process_cell_selection(pre_cells: list, last_cells) -> bool:
    """Whether app.py's process_cell_selection should resolve `pre_cells`
    and persist the result this rerun (Phase 5b-3).

    NOT simply `is_fresh_selection(pre_cells, last_cells)` — confirmed by
    live browser probe (against the real running app, not a static scratch
    script): when a sidebar filter reshapes filtered/display_df (a
    different row count/order on the SAME st.dataframe key), the frontend
    resets its own cells selection to empty on that rerun, REGARDLESS of
    whether the previously-clicked ticker survives the new filter. An empty
    `pre_cells` is therefore always this reshape side effect, never a
    deliberate user deselection (no gesture producing one was ever
    observed, unlike rows, which do support a real deselect-by-reclick) —
    so it must never be processed as a fresh, empty selection, which would
    silently overwrite a still-good persisted (ticker, column) with None.
    Whether the persisted ticker actually survived the new filter is a
    separate question, answered by find_ticker_row against the CURRENT
    display_df in render_cell_derivation_panel — not by this function.

    Args:
        pre_cells: The table's current raw cells selection-state list.
        last_cells: The cells list last processed (None if never processed).

    Returns:
        True only for a genuinely non-empty, genuinely new cells value.
    """
    return bool(pre_cells) and is_fresh_selection(pre_cells, last_cells)


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


def resolve_selected_cell(
    display_df: pd.DataFrame, selected_cells: list
) -> tuple[str, str] | None:
    """Resolve a single-cell selection into (ticker, column_name) (Phase 5b-3).

    Positional into `display_df` exactly as passed to `st.dataframe` — same
    discipline as `resolve_selected_ticker`/`find_ticker_row`, and confirmed
    (not assumed) to be the right frame to resolve against even under a live
    browser-side column sort: a clicked cell's row index is reported against
    the frame as originally passed to `st.dataframe`, unaffected by any
    further sort the user applies in the browser on top of it (verified
    directly against streamlit 1.63.0 in both directions — a visually-top
    row whose true position is last, and a visually-bottom row whose true
    position is first, both round-tripped correctly). This is the same
    invariant `st.dataframe` documents for row selections; it is not
    documented for cell selections, so it was measured rather than assumed.

    No previous-value precedence chain (unlike `resolve_selected_ticker`):
    a cell selection, once made, is empirically sticky across unrelated
    reruns (also measured, not assumed — see app.py's module comment near
    `render_main_table`), so the caller is responsible for calling this only
    on the rerun where `selected_cells` actually changes (via
    `is_fresh_selection`) and persisting the result itself. Calling it again
    on a later, unrelated rerun against a since-reordered/filtered
    `display_df` using the same stale `selected_cells` would resolve the
    wrong company at that position — the caller must not do that; see
    app.py's cell-derivation bookkeeping.

    Args:
        display_df: The frame exactly as passed to st.dataframe — already
            sorted, already column-subset.
        selected_cells: The raw cells selection-state list. Each entry is a
            (row_position, column_name) pair — a tuple when it arrives from
            a genuine streamlit interaction, but unpacked positionally here
            so a list-shaped entry (e.g. from a hand-built test fixture)
            resolves identically. Only the first entry is consulted (single-
            cell mode only). May be empty.

    Returns:
        (ticker, column_name) for the first selected cell, or None if
        `selected_cells` is empty, the row index is out of range, or the
        row's own ticker is NaN (defensive only — no live table has a null
        ticker).
    """
    if not selected_cells:
        return None
    row_idx, column_name = selected_cells[0]
    if display_df is None or display_df.empty:
        return None
    if not (0 <= row_idx < len(display_df)):
        return None
    ticker = display_df["ticker"].iloc[row_idx]
    if pd.isna(ticker):
        return None
    return (ticker, column_name)
