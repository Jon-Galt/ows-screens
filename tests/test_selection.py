"""Unit tests for src/selection.py's pure ticker-resolution logic (Phase 5b-1;
Phase 5b-3 adds resolve_selected_cell and cell-selection coverage for
is_fresh_selection). No test in this file reads data/screener.db — the live
1,358-row correspondence (real tickers, real positions) is verified once, in
the phase's acceptance run, per this project's established rule for
data/historical/'s sibling gitignored data (CLAUDE.md's Known Implementation
Decisions / Phase 4a discussion): a unit test may never depend on gitignored
project data, so every frame below is synthetic."""

import numpy as np
import pandas as pd

from src.selection import (
    find_ticker_row,
    is_fresh_selection,
    resolve_nav_target,
    resolve_selected_cell,
    resolve_selected_ticker,
    should_process_cell_selection,
)


def test_resolve_uses_display_order_not_source_order():
    """The discriminating test for this phase's trap: display_df's row
    order must win, not the order of some unsorted source frame it came
    from. If the implementation read a source frame's positions instead of
    display_df's own, this test fails: source-frame position 0 is AAA, but
    display_df puts CCC first, and CCC is what a click on row 0 must mean."""
    source_df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})
    display_df = source_df.sort_values("ticker", ascending=False).reset_index(drop=True)
    assert list(display_df["ticker"]) == ["CCC", "BBB", "AAA"]

    result = resolve_selected_ticker(display_df, selected_rows=[0])

    assert result == "CCC"


def test_empty_selection_keeps_previous_ticker_if_still_present():
    display_df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})

    result = resolve_selected_ticker(display_df, selected_rows=[], previous_ticker="BBB")

    assert result == "BBB"


def test_previous_ticker_dropped_by_filter_falls_back_to_first_in_display():
    display_df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})

    result = resolve_selected_ticker(display_df, selected_rows=[], previous_ticker="ZZZ")

    assert result == "AAA"


def test_out_of_range_index_falls_back_without_raising():
    display_df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})

    result = resolve_selected_ticker(display_df, selected_rows=[99], previous_ticker="BBB")

    assert result == "BBB"


def test_negative_index_is_not_treated_as_a_valid_trailing_selection():
    display_df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})

    result = resolve_selected_ticker(display_df, selected_rows=[-1], previous_ticker="BBB")

    assert result == "BBB"


def test_empty_display_df_returns_none():
    display_df = pd.DataFrame({"ticker": pd.Series(dtype="object")})

    result = resolve_selected_ticker(display_df, selected_rows=[], previous_ticker="AAA")

    assert result is None


def test_nan_ticker_in_non_selected_row_does_not_break_fallback():
    display_df = pd.DataFrame({"ticker": ["AAA", np.nan, "CCC"]})

    result = resolve_selected_ticker(display_df, selected_rows=[], previous_ticker="ZZZ")

    assert result == "AAA"


def test_selected_row_with_nan_ticker_falls_through_to_previous_ticker():
    display_df = pd.DataFrame({"ticker": ["AAA", np.nan, "CCC"]})

    result = resolve_selected_ticker(display_df, selected_rows=[1], previous_ticker="CCC")

    assert result == "CCC"


def test_find_ticker_row_uses_position_not_pandas_index_label():
    """display_df carries a non-contiguous pandas index (a realistic
    survivor of an upstream sort_values/filter chain that never called
    reset_index) — the returned position must be positional (0-based into
    display order), not the pandas index label."""
    display_df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]}, index=[7, 3, 9])

    result = find_ticker_row(display_df, "BBB")

    assert result == 1


def test_find_ticker_row_absent_ticker_returns_none():
    display_df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})

    result = find_ticker_row(display_df, "ZZZ")

    assert result is None


def test_find_ticker_row_none_ticker_returns_none():
    display_df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})

    result = find_ticker_row(display_df, None)

    assert result is None


# ---------------------------------------------------------------------------
# resolve_nav_target (Phase 5b-2)
#
# The regression lock on the click-through defect found in plan revision 1:
# a navigated-to ticker excluded by the destination screen's active filters
# must never be reported as "show" against a substitute ticker — it must be
# reported "blocked", full stop. "A" below is deliberately what
# resolve_selected_ticker's own precedence-3 fallback (first ticker in
# display order) would have silently shown instead, absent this gate.
# ---------------------------------------------------------------------------


def test_resolve_nav_target_blocked_when_filtered_out():
    filtered = pd.DataFrame({"ticker": ["A", "B", "C"]})

    result = resolve_nav_target(filtered, "IFF")

    assert result == ("blocked", "IFF")  # not ("show", "A")


def test_resolve_nav_target_shows_when_present():
    filtered = pd.DataFrame({"ticker": ["A", "IFF", "C"]})

    result = resolve_nav_target(filtered, "IFF")

    assert result == ("show", "IFF")


# ---------------------------------------------------------------------------
# is_fresh_selection (Phase 5b-2)
# ---------------------------------------------------------------------------


def test_is_fresh_selection_true_when_rows_differ():
    assert is_fresh_selection([2], [0]) is True


def test_is_fresh_selection_false_when_unchanged():
    assert is_fresh_selection([2], [2]) is False


def test_is_fresh_selection_false_when_both_empty():
    """A toggle-off (selection empties out) followed by an unrelated rerun
    must not read as a fresh selection on the second rerun."""
    assert is_fresh_selection([], []) is False


# ---------------------------------------------------------------------------
# resolve_selected_cell / cell-selection is_fresh_selection (Phase 5b-3)
#
# All frames below are synthetic. A 10-row frame where "raw" (construction)
# order and "sorted" (score-descending) order disagree at three positions —
# standing in for the live short_screen table's own VYX/NVDA-style
# divergence (verified once, out of band, in the phase's acceptance run;
# see CLAUDE.md) without any test here depending on data/screener.db.
#
#   position   sorted ticker   raw-order ticker
#      0            J                A
#      5            E                F
#      9            A                J
# ---------------------------------------------------------------------------

_RAW_TICKERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _raw_order_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": _RAW_TICKERS, "score": list(range(1, 11))}
    )


def _sorted_frame() -> pd.DataFrame:
    return _raw_order_frame().sort_values("score", ascending=False).reset_index(drop=True)


def test_resolve_selected_cell_positions_match_sort_order_not_raw_order():
    """§5.5: resolve_selected_cell resolves positionally against the frame
    passed in — display_df, i.e. the app's own sort_values order — not
    whatever order the underlying data happened to be constructed in."""
    sorted_df = _sorted_frame()
    assert list(sorted_df["ticker"]) == ["J", "I", "H", "G", "F", "E", "D", "C", "B", "A"]

    assert resolve_selected_cell(sorted_df, [(0, "abs_ps_factor")]) == ("J", "abs_ps_factor")
    assert resolve_selected_cell(sorted_df, [(5, "abs_ps_factor")]) == ("E", "abs_ps_factor")
    assert resolve_selected_cell(sorted_df, [(9, "abs_ps_factor")]) == ("A", "abs_ps_factor")


def test_resolve_selected_cell_through_raw_order_gives_the_wrong_answer():
    """The wrong-answer half (§5.5's model): resolving the SAME positions
    against the raw-construction-order frame gives different, real,
    named tickers — proving the sort is load-bearing rather than
    incidentally agreeing with the resolved values above."""
    raw_df = _raw_order_frame()
    assert resolve_selected_cell(raw_df, [(0, "abs_ps_factor")]) == ("A", "abs_ps_factor")
    assert resolve_selected_cell(raw_df, [(5, "abs_ps_factor")]) == ("F", "abs_ps_factor")
    assert resolve_selected_cell(raw_df, [(9, "abs_ps_factor")]) == ("J", "abs_ps_factor")


def test_resolve_selected_cell_empty_selection_returns_none():
    display_df = _sorted_frame()
    assert resolve_selected_cell(display_df, []) is None


def test_resolve_selected_cell_out_of_range_row_returns_none():
    display_df = _sorted_frame()
    assert resolve_selected_cell(display_df, [(99, "abs_ps_factor")]) is None


def test_resolve_selected_cell_empty_display_df_returns_none():
    display_df = pd.DataFrame({"ticker": pd.Series(dtype="object")})
    assert resolve_selected_cell(display_df, [(0, "abs_ps_factor")]) is None


def test_resolve_selected_cell_nan_ticker_returns_none():
    display_df = pd.DataFrame({"ticker": ["A", np.nan, "C"]})
    assert resolve_selected_cell(display_df, [(1, "abs_ps_factor")]) is None


def test_resolve_selected_cell_only_consults_first_entry():
    """Single-cell mode only — a second entry (which streamlit itself never
    actually sends, per _validate_selection_state's own truncation to one)
    must not change the result."""
    display_df = _sorted_frame()
    result = resolve_selected_cell(
        display_df, [(0, "abs_ps_factor"), (9, "def_rev_factor")]
    )
    assert result == ("J", "abs_ps_factor")


def test_resolve_selected_cell_unpacks_list_shaped_entries_identically():
    """A hand-built test fixture (or a raw, not-yet-deserialized frontend
    payload) may be list-shaped ([[0, "col"]]) rather than tuple-shaped
    ((0, "col")) — plain iterable unpacking handles both without any
    explicit normalization step."""
    display_df = _sorted_frame()
    assert resolve_selected_cell(display_df, [[0, "abs_ps_factor"]]) == ("J", "abs_ps_factor")


# ---------------------------------------------------------------------------
# The persistence-crossing test (Phase 5b-3 plan review round 2, replacing
# a round-1 test that encoded the exact defect the design forbids): once a
# cell is resolved, a later, unrelated rerun against a reordered frame must
# NOT re-resolve by the original click's stale row position. This is the
# regression lock for the mechanism itself, not just for resolve_selected_
# cell in isolation — it simulates app.py's own state machine
# (should_process_cell_selection gate, then find_ticker_row-by-identity,
# never a second call to resolve_selected_cell) directly, since app.py's
# Streamlit-coupled rendering code has no unit tests of its own in this
# repo (see this project's established pure-layer-only testing posture).
#
# pre_cells_2 below is `[]`, not a repeat of pre_cells_1 — this was corrected
# during the build after a live-app probe (clicking a cell, then actually
# changing a sidebar filter) showed the frontend resets `cells` to empty on
# any rerun where the underlying data reshapes, regardless of whether the
# clicked ticker survives the new filter. An earlier draft of this test
# modeled the wrong mechanism (assuming `cells` stays sticky across a data
# change, which the earlier static-frame probes had shown for an UNRELATED
# widget change only, never for a genuine data reshape) and would have
# passed for the wrong reason. See CLAUDE.md's Known Implementation
# Decisions #4.
# ---------------------------------------------------------------------------


def test_persisted_cell_survives_reorder_without_stale_reresolution():
    position, expected_ticker, wrong_ticker = 0, "J", "A"

    frame_a = _sorted_frame()
    pre_cells_1 = [(position, "abs_ps_factor")]
    last_cells = None  # never processed yet

    assert should_process_cell_selection(pre_cells_1, last_cells) is True
    resolved = resolve_selected_cell(frame_a, pre_cells_1)
    assert resolved == (expected_ticker, "abs_ps_factor")
    last_cells = pre_cells_1  # what process_cell_selection stores, verbatim

    # A filter/sort change: same universe, now in raw-construction order, so
    # `position` holds a different, named, real ticker. The frontend resets
    # its cells selection to empty on this kind of rerun (confirmed by live
    # probe, not the static-frame scratch scripts) — pre_cells_2 is `[]`,
    # NOT a repeat of pre_cells_1.
    frame_b = _raw_order_frame()
    pre_cells_2 = []

    assert should_process_cell_selection(pre_cells_2, last_cells) is False
    # The persisted value from step 1 is what a correct implementation
    # renders — resolve_selected_cell must NOT be called again here.
    persisted_ticker, _persisted_column = resolved
    assert persisted_ticker == expected_ticker
    assert persisted_ticker != frame_b["ticker"].iloc[position]
    assert frame_b["ticker"].iloc[position] == wrong_ticker

    # The re-validation path: find_ticker_row locates the persisted ticker
    # BY IDENTITY in the new frame, wherever it now sits — never by the
    # stale position.
    assert find_ticker_row(frame_b, persisted_ticker) == 9  # J is last in raw order


def test_a_repeat_click_after_a_reshape_resolves_against_the_new_frame():
    """Regression lock for the build-time defect found live: process_cell_
    selection must write its last_cells_key baseline UNCONDITIONALLY, not
    only inside the `should_process_cell_selection` guard.

    Sequence: click position 0 in frame_a (fresh, resolves to J, baseline
    becomes pre_cells_1) -> a reshape empties pre_cells (correctly not
    processed) -> the SAME position/column is clicked again in the NEW
    frame_b (the top row is the likeliest repeat click). If the baseline
    were only written inside the guard, it would still hold pre_cells_1
    from the first click, this third click's pre_cells (identical in shape
    to pre_cells_1) would compare EQUAL to it, `should_process_cell_
    selection` would report False, and the panel would either go dead (if
    the ticker had been filtered out) or keep showing J's derivation under
    a title that no longer matches what was just clicked (if the frame was
    only reordered) — a real, plausible, wrong company with no error.

    With the unconditional write, the reshape step already recorded `[]` as
    the baseline, so this third click is recognized as fresh and resolves
    against frame_b — the CURRENT frame — correctly landing on whichever
    ticker frame_b actually has at position 0 today (A, not J).
    """
    frame_a = _sorted_frame()
    pre_cells_1 = [(0, "abs_ps_factor")]
    last_cells = None

    assert should_process_cell_selection(pre_cells_1, last_cells) is True
    resolved = resolve_selected_cell(frame_a, pre_cells_1)
    assert resolved == ("J", "abs_ps_factor")
    last_cells = pre_cells_1  # process_cell_selection's unconditional write

    # Reshape: frame_b is raw-construction order. pre_cells resets to [].
    frame_b = _raw_order_frame()
    pre_cells_2 = []
    assert should_process_cell_selection(pre_cells_2, last_cells) is False
    last_cells = pre_cells_2  # the unconditional write — [] becomes the new baseline

    # Third click: same (row, column) as the very first click, but now
    # against frame_b, whose position 0 is a different, named ticker.
    pre_cells_3 = [(0, "abs_ps_factor")]
    assert should_process_cell_selection(pre_cells_3, last_cells) is True
    resolved_again = resolve_selected_cell(frame_b, pre_cells_3)
    assert resolved_again == ("A", "abs_ps_factor")
    assert resolved_again != resolved  # not J again — this click is fresh and re-resolved


def test_is_fresh_selection_true_on_first_click_no_prior_state():
    assert is_fresh_selection([(2, "abs_ps_factor")], None) is True


def test_is_fresh_selection_false_when_cell_selection_repeats():
    """Confirmed by direct probe (Phase 5b-3 plan review round 2): an
    unrelated rerun (the underlying data UNCHANGED) echoes back the SAME
    non-empty cells value the frontend last reported — sync_drilldown_
    selection's own `cells: []` push shapes only the return value of the
    run it happens in and has no durable effect on what the next rerun
    reads. last_cells_key must therefore store pre_cells verbatim (not
    force it back to []) for this to correctly read as NOT fresh."""
    assert is_fresh_selection([(2, "abs_ps_factor")], [(2, "abs_ps_factor")]) is False


def test_is_fresh_selection_true_on_genuinely_new_cell_click():
    assert is_fresh_selection([(4, "def_rev_factor")], [(2, "abs_ps_factor")]) is True


# ---------------------------------------------------------------------------
# should_process_cell_selection (Phase 5b-3, found during the build's own
# browser verification, not planned in any review round): a live probe
# against the real app showed that a rerun where filtered/display_df's
# CONTENT reshapes (any sidebar filter change) resets the frontend's cells
# selection to empty, regardless of whether the previously-clicked ticker
# survives the new filter. is_fresh_selection alone cannot distinguish that
# from a genuine (if hypothetical) user deselection, so app.py must gate on
# this function instead — the regression lock for the actual bug an earlier
# implementation shipped: a filter change that KEPT the clicked ticker still
# silently cleared the derivation panel, because pre_cells arriving empty
# was wrongly processed as a fresh, empty selection.
# ---------------------------------------------------------------------------


def test_should_process_cell_selection_false_when_pre_cells_empty_even_if_stale():
    """The regression lock: an empty pre_cells must NEVER be processed,
    even though it differs from a non-empty last_cells (is_fresh_selection
    alone would say True here) — an empty cells value is always a
    data-reshape side effect, never a deliberate deselection."""
    assert should_process_cell_selection([], [(0, "abs_ps_factor")]) is False


def test_should_process_cell_selection_true_on_first_genuine_click():
    assert should_process_cell_selection([(2, "abs_ps_factor")], None) is True


def test_should_process_cell_selection_false_when_repeated():
    assert should_process_cell_selection([(2, "abs_ps_factor")], [(2, "abs_ps_factor")]) is False


def test_should_process_cell_selection_true_on_genuinely_new_click():
    assert should_process_cell_selection([(4, "def_rev_factor")], [(2, "abs_ps_factor")]) is True


def test_should_process_cell_selection_false_when_both_empty():
    assert should_process_cell_selection([], None) is False
    assert should_process_cell_selection([], []) is False


def test_shape_mismatch_in_stored_value_is_the_failure_mode_storage_discipline_prevents():
    """This is a demonstration of a failure mode, not a test of production
    code — there is no separate "normalize" function to call; the actual
    fix is a one-line storage discipline in app.py (copy pre_cells into
    last_cells_key verbatim, never reconstruct/retype it — see
    process_cell_selection's comment at that assignment). If that
    discipline were ever violated (e.g. by rebuilding the stored value as
    a list literal instead of copying the tuple-shaped value streamlit
    itself returns), an otherwise-unchanged real selection would compare
    unequal here and misreport as fresh on every rerun, undoing the
    persistence guard entirely."""
    same_cell_as_tuple = [(2, "abs_ps_factor")]
    same_cell_as_reconstructed_list = [[2, "abs_ps_factor"]]
    assert is_fresh_selection(same_cell_as_tuple, same_cell_as_reconstructed_list) is True


# ---------------------------------------------------------------------------
# §5.6 crossings: interactions between two mechanisms, tested by crossing
# them in one sequence rather than as independent checks (5b-1's lesson).
# ---------------------------------------------------------------------------


def test_crossing_filter_change_keeps_ticker_panel_survives():
    """Crossing 1: resolve a cell, then simulate a filter/sort that keeps
    the ticker but reorders the frame — the panel must still resolve to
    the same ticker via find_ticker_row, not vanish or move."""
    frame_a = _sorted_frame()
    resolved = resolve_selected_cell(frame_a, [(5, "abs_ps_factor")])
    assert resolved == ("E", "abs_ps_factor")

    # Filter/sort change: same tickers, ascending instead of descending.
    frame_b = frame_a.sort_values("score", ascending=True).reset_index(drop=True)
    ticker, _column = resolved
    new_position = find_ticker_row(frame_b, ticker)
    assert new_position is not None
    assert frame_b["ticker"].iloc[new_position] == "E"


def test_crossing_filter_change_excludes_ticker_panel_clears():
    """Crossing 2: resolve a cell, then simulate a filter that excludes the
    ticker entirely — find_ticker_row must report None (the signal
    render_cell_derivation_panel uses to show its one-line caption and
    clear the stored key), never resolve a different company at the old
    position."""
    frame_a = _sorted_frame()
    resolved = resolve_selected_cell(frame_a, [(0, "abs_ps_factor")])
    assert resolved == ("J", "abs_ps_factor")

    ticker, _column = resolved
    frame_b = frame_a[frame_a["ticker"] != ticker].reset_index(drop=True)
    assert find_ticker_row(frame_b, ticker) is None


def test_crossing_cross_screen_navigation_leaves_no_stale_panel():
    """Crossing 4: resolve a cell against one screen's frame, then simulate
    navigating away to a different screen and back with a frame that no
    longer contains the original ticker (filters changed while away) —
    find_ticker_row must report None rather than resolving against
    whichever ticker now occupies the old position on the new screen's
    frame. apply_pending_nav's own screen_id-matching gate (untouched by
    this phase) is what prevents a mismatched-screen pending nav from ever
    reaching this table in the first place; this test covers what happens
    once the user is back on the original screen with reshaped data."""
    frame_before_nav = _sorted_frame()
    resolved = resolve_selected_cell(frame_before_nav, [(0, "abs_ps_factor")])
    assert resolved == ("J", "abs_ps_factor")
    ticker, _column = resolved

    # Back on the original screen, but its filters changed while the user
    # was away — a frame that shares no tickers with the one the cell was
    # originally clicked against.
    frame_after_nav = pd.DataFrame({"ticker": ["Z1", "Z2", "Z3"], "score": [1, 2, 3]})
    assert find_ticker_row(frame_after_nav, ticker) is None
