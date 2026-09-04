"""Unit tests for src/selection.py's pure ticker-resolution logic (Phase 5b-1)."""

import numpy as np
import pandas as pd

from src.selection import (
    find_ticker_row,
    is_fresh_selection,
    resolve_nav_target,
    resolve_selected_ticker,
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
