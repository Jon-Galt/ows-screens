"""
Unit tests for src/filtering.py (Phase 8c-6).

Covers:
1. Every column is offered (L1) — never a display/export subset.
2. Dtype picks the control kind (L2), including the bool-vs-int64-flag edge
   case the PM ruling flagged rather than "fixed" (mscore_flag stays range;
   overvalued_flag/transcripts_flag/in_universe-shaped bools go to values).
3. Multiple filters AND-combine (L3) and removing one leaves the other
   applied (L4).
4. A combination matching nothing returns an empty frame without raising
   (L7's pure-module half; the render-layer half is in tests/test_app.py).
5. Purity (no streamlit/sqlalchemy import), mirrors
   tests/test_weighting.py's AST-based lock (L9).

No test here reads data/screener.db, data/uploads/, or data/historical/ —
synthetic frames only (T25).
"""

import ast
import os

import pandas as pd

from src.filtering import (
    apply_column_filters,
    classify_filter_kind,
    get_column_bounds,
    get_column_values,
    get_filterable_columns,
)


# ---------------------------------------------------------------------------
# L1: every column in the frame is offered
# ---------------------------------------------------------------------------


class TestGetFilterableColumns:
    def test_returns_every_column_not_a_display_or_export_subset(self):
        """A column that would never appear in any hand-written display or
        export list must still come back — the anti-L1 case."""
        df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "some_new_factor_nobody_has_labeled_yet": [1.0, 2.0],
            }
        )
        assert get_filterable_columns(df) == [
            "ticker",
            "some_new_factor_nobody_has_labeled_yet",
        ]

    def test_preserves_column_order(self):
        df = pd.DataFrame({"c": [1], "a": [2], "b": [3]})
        assert get_filterable_columns(df) == ["c", "a", "b"]

    def test_does_not_truncate_a_frame_wider_than_any_display_or_export_list(self):
        """PM correction (8c-6 review, M4): a 3- or 4-column fixture can't
        discriminate against a build that caps the list at a fixed length —
        short_screen's live frame is 156 columns, wider than any constant in
        app.py, which is exactly why a cap wouldn't show up in the app
        either. FAILS IF get_filterable_columns is capped at any fixed
        length below 200."""
        columns = [f"col_{i}" for i in range(200)]
        df = pd.DataFrame({c: [1.0] for c in columns})
        assert get_filterable_columns(df) == columns
        assert len(get_filterable_columns(df)) == 200


# ---------------------------------------------------------------------------
# L2: dtype picks the control
# ---------------------------------------------------------------------------


class TestClassifyFilterKind:
    def test_float_column_is_range(self):
        assert classify_filter_kind(pd.Series([1.0, 2.0, float("nan")])) == "range"

    def test_int_column_is_range(self):
        """mscore_flag-shaped: int64 0/1, stays a range control (PM ruling
        — deliberately NOT rerouted to values by a cardinality heuristic)."""
        assert classify_filter_kind(pd.Series([0, 1, 0, 1], dtype="int64")) == "range"

    def test_text_column_is_values(self):
        assert classify_filter_kind(pd.Series(["Tech", "Health"])) == "values"

    def test_bool_column_is_values_not_range(self):
        """overvalued_flag/transcripts_flag/in_universe-shaped: bool is
        numeric per pandas.api.types.is_numeric_dtype, so this is the case
        the three-way dtype split exists for — a literal two-way numeric-vs-
        not split would give this a 0..1 range slider instead."""
        assert classify_filter_kind(pd.Series([True, False, True])) == "values"


# ---------------------------------------------------------------------------
# get_column_bounds / get_column_values
# ---------------------------------------------------------------------------


class TestGetColumnBounds:
    def test_returns_min_and_max_ignoring_nan(self):
        df = pd.DataFrame({"x": [1.0, 5.0, float("nan"), 3.0]})
        assert get_column_bounds(df, "x") == (1.0, 5.0)


class TestGetColumnValues:
    def test_returns_sorted_distinct_non_null_values(self):
        df = pd.DataFrame({"sector": ["Tech", "Health", "Tech", None]})
        assert get_column_values(df, "sector") == ["Health", "Tech"]

    def test_cardinality_blind_large_column_returns_all_distinct_values(self):
        """No truncation, no top-N — the UI layer decides presentation, not
        this function (Phase 8c-6 plan's 1,358-option analysis)."""
        df = pd.DataFrame({"ticker": [f"T{i}" for i in range(2000)]})
        assert len(get_column_values(df, "ticker")) == 2000


# ---------------------------------------------------------------------------
# L3/L4: AND-combination and independent removal
# ---------------------------------------------------------------------------


class TestApplyColumnFilters:
    def _df(self):
        return pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD"],
                "sector": ["Tech", "Tech", "Health", "Health"],
                "market_cap": [100.0, 500.0, 50.0, 900.0],
            }
        )

    def test_no_filters_returns_the_whole_frame(self):
        df = self._df()
        result = apply_column_filters(df, {})
        assert list(result["ticker"]) == ["AAA", "BBB", "CCC", "DDD"]

    def test_single_values_filter(self):
        df = self._df()
        result = apply_column_filters(df, {"sector": ("values", ["Tech"])})
        assert list(result["ticker"]) == ["AAA", "BBB"]

    def test_single_range_filter(self):
        df = self._df()
        result = apply_column_filters(df, {"market_cap": ("range", (200.0, 900.0))})
        assert list(result["ticker"]) == ["BBB", "DDD"]

    def test_two_filters_on_two_columns_both_apply_and_combined(self):
        """L3: FAILS if the second filter replaces the first, and FAILS if
        they OR instead of AND (AAA/BBB pass sector alone; BBB/DDD pass
        market_cap alone; only BBB passes both)."""
        df = self._df()
        result = apply_column_filters(
            df,
            {
                "sector": ("values", ["Tech"]),
                "market_cap": ("range", (200.0, 900.0)),
            },
        )
        assert list(result["ticker"]) == ["BBB"]

    def test_removing_one_filter_leaves_the_other_applied(self):
        """L4: the sector filter dropped from the dict entirely (as the UI
        does on a Remove click) — only market_cap still restricts."""
        df = self._df()
        result = apply_column_filters(df, {"market_cap": ("range", (200.0, 900.0))})
        assert list(result["ticker"]) == ["BBB", "DDD"]

    def test_empty_values_selection_is_no_restriction(self):
        """Matches render_sidebar's existing convention: nothing selected
        means the filter isn't applied, not "match nothing"."""
        df = self._df()
        result = apply_column_filters(df, {"sector": ("values", [])})
        assert list(result["ticker"]) == ["AAA", "BBB", "CCC", "DDD"]

    def test_combination_matching_nothing_returns_empty_frame_without_raising(self):
        """L7's pure-module half."""
        df = self._df()
        result = apply_column_filters(
            df,
            {
                "sector": ("values", ["Tech"]),
                "market_cap": ("range", (1000.0, 2000.0)),
            },
        )
        assert result.empty

    def test_nan_row_excluded_by_range_filter(self):
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
        result = apply_column_filters(df, {"x": ("range", (0.0, 10.0))})
        assert len(result) == 2

    def test_column_absent_from_frame_is_skipped_not_raised(self):
        """A stale filter surviving a screen switch must not crash the next
        screen's frame."""
        df = self._df()
        result = apply_column_filters(df, {"not_a_real_column": ("values", ["x"])})
        assert list(result["ticker"]) == ["AAA", "BBB", "CCC", "DDD"]


# ---------------------------------------------------------------------------
# L9: purity — parsed import graph, not a source-text substring scan (a
# substring scan would false-positive on this module's own docstring, which
# mentions streamlit/sqlalchemy by name to explain why they are absent).
# ---------------------------------------------------------------------------


def test_filtering_module_imports_neither_streamlit_nor_sqlalchemy():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "filtering.py")
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)

    forbidden = {"streamlit", "sqlalchemy"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)

    hits = {name for name in imported if any(name == f or name.startswith(f + ".") for f in forbidden)}
    assert not hits, f"src/filtering.py imports forbidden module(s): {hits}"
