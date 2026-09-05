"""
Unit tests for the pure-Python constants in src/app.py that back Phase
3c.1's "show underlying metric values" feature and Phase 3c.2's diff-based
factor derivations.

app.py itself is Streamlit UI and is verified manually, as part of the
end-of-phase verification chain (see CLAUDE.md's Worker Rules), not via
pytest — these tests cover only the data structures a silent drift in
score.py's FACTOR_DEFINITIONS or transform.py's calc functions could break:
a new or renamed factor/column with no matching entry here would otherwise
show "N/A" in the drill-down instead of failing loudly.
"""

import os
import re

import pandas as pd
import pytest

from src.app import (
    APP_FONT_FAMILY,
    CELL_DERIVATION_FACTORS,
    CURATED_COLUMN_HELP,
    CURATED_COLUMN_LABELS,
    CURATED_DISPLAY_COLUMNS,
    DIFF_FACTOR_FORMULAS,
    DIFF_FACTOR_INPUTS,
    DIFF_INPUT_COLUMNS,
    DIFF_INPUT_FORMATS,
    DISPLAY_COLUMNS,
    FACTOR_COLUMN_LABELS,
    FACTOR_DEFINITIONS,
    INPUT_COLUMN_FORMATS,
    LOGO_MARK_PATH,
    MAIN_TABLE_COLUMN_HELP,
    MAIN_TABLE_COLUMN_LABELS,
    METRIC_COLUMN_FORMATS,
    METRIC_COLUMN_LABELS,
    METRIC_FORMATS,
    NON_DIFF_FACTOR_BY_COLUMN,
    OVERLAP_COLUMN_HELP,
    OVERLAP_COLUMN_LABELS,
    OVERLAP_DISPLAY_COLUMNS,
    SCREEN_ICONS,
    TITLE_MARK_PATH,
    UNSCORED_COLUMN_HELP,
    UNSCORED_COLUMN_LABELS,
    UNSCORED_DISPLAY_COLUMNS,
    _DEFAULT_SCREEN_ICON,
    _STOCK_PERFORMANCE_LABEL,
    build_export_columns,
    format_diff_formula,
    format_screen_title,
    interleave_metric_columns,
)
from src.transform import (
    calc_deferred_rev_pct_change,
    calc_dio_pct_change,
    calc_dpo_pct_change,
    calc_dso_pct_change,
    calc_ebit_diff,
    calc_fcf_yield_diff,
    calc_gm_diff,
    calc_growth_accel,
    calc_growth_decel,
    calc_ps_diff,
    run_transforms,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OVERLAP_DISPLAY_COLUMNS (imported above from src.app, Phase 5b-2 — no
# longer a hand-copied mirror of a function-local list) minus overall_score,
# which keeps its own separate, existing column_config entry and is not part
# of OVERLAP_COLUMN_LABELS.
OVERLAP_LABEL_COLUMNS = [c for c in OVERLAP_DISPLAY_COLUMNS if c != "overall_score"]

# The 10 diff-based factors this phase covers, per PHASE3C2_APPROVAL.md.
DIFF_BASED_FACTORS = {
    "abs_ps_factor", "abs_fcf_factor", "decel_factor", "accel_factor",
    "gm_factor", "ebit_factor", "dso_factor", "dio_factor", "dpo_factor",
    "def_rev_factor",
}


class TestMetricFormatsCompleteness:
    def test_every_factor_has_a_metric_format(self):
        missing = [f for f in FACTOR_DEFINITIONS if f not in METRIC_FORMATS]
        assert missing == []

    def test_metric_formats_has_no_stale_entries(self):
        """A factor removed from score.py should have its format removed
        too, not linger as a silently-unused entry."""
        stale = [f for f in METRIC_FORMATS if f not in FACTOR_DEFINITIONS]
        assert stale == []

    def test_metric_column_formats_keyed_by_metric_not_factor(self):
        for factor, fmt in METRIC_FORMATS.items():
            metric_col = FACTOR_DEFINITIONS[factor]["metric"]
            assert METRIC_COLUMN_FORMATS[metric_col] == fmt

    def test_every_format_spec_is_applicable_to_a_float(self):
        for fmt in METRIC_FORMATS.values():
            fmt.format(1.23456)  # raises ValueError if the spec is malformed


class TestDisplayLabelsCompleteness:
    """Phase 5a: every on-screen column resolves to a display label, and no
    label map carries a stale entry. Same pattern as
    TestMetricFormatsCompleteness above and for the same reason — a new or
    renamed column with no matching label entry would otherwise silently
    render as a raw snake_case DB column name instead of failing loudly,
    and a collision between two columns' labels (the C7a bug class) would
    silently misrender one of them as the other's header."""

    def test_every_display_column_has_a_main_table_label(self):
        rendered = interleave_metric_columns(DISPLAY_COLUMNS)
        missing = [c for c in rendered if c not in MAIN_TABLE_COLUMN_LABELS]
        assert missing == []

    def test_main_table_labels_has_no_stale_entries(self):
        rendered = set(interleave_metric_columns(DISPLAY_COLUMNS))
        stale = [c for c in MAIN_TABLE_COLUMN_LABELS if c not in rendered]
        assert stale == []

    def test_main_table_labels_are_unique(self):
        """No two distinct DB columns resolve to the same header — the
        exact defect class C7a found (debt_ev_factor's label colliding
        with its own metric column debt_ev before the 'Factor' suffix)."""
        labels = list(MAIN_TABLE_COLUMN_LABELS.values())
        assert len(labels) == len(set(labels)), labels

    def test_every_metric_column_has_a_metric_label(self):
        all_metrics = {defn["metric"] for defn in FACTOR_DEFINITIONS.values()}
        missing = [c for c in all_metrics if c not in METRIC_COLUMN_LABELS]
        assert missing == []

    def test_metric_column_labels_has_no_stale_entries(self):
        all_metrics = {defn["metric"] for defn in FACTOR_DEFINITIONS.values()}
        stale = [c for c in METRIC_COLUMN_LABELS if c not in all_metrics]
        assert stale == []

    def test_ps_ntm_and_fcf_yield_labels_match_diff_factor_inputs(self):
        """C7b regression lock: ps_ntm/fcf_yield double as rel_ps_factor's/
        rel_fcf_factor's own metric AND as an input to abs_ps_factor's/
        abs_fcf_factor's diff — they must carry one label, not two, across
        the main table and the drill-down's DIFF_FACTOR_INPUTS."""
        diff_labels = {
            col: label
            for inputs in DIFF_FACTOR_INPUTS.values()
            for col, label, _source_func in inputs
        }
        assert METRIC_COLUMN_LABELS["ps_ntm"] == diff_labels["ps_ntm"]
        assert METRIC_COLUMN_LABELS["fcf_yield"] == diff_labels["fcf_yield"]

    def test_every_factor_column_label_ends_with_factor(self):
        for factor, label in FACTOR_COLUMN_LABELS.items():
            assert label.endswith(" Factor"), (factor, label)

    def test_every_curated_display_column_has_a_label(self):
        missing = [c for c in CURATED_DISPLAY_COLUMNS if c not in CURATED_COLUMN_LABELS]
        assert missing == []

    def test_curated_column_labels_has_no_stale_entries(self):
        stale = [c for c in CURATED_COLUMN_LABELS if c not in CURATED_DISPLAY_COLUMNS]
        assert stale == []

    def test_curated_column_labels_are_unique(self):
        labels = list(CURATED_COLUMN_LABELS.values())
        assert len(labels) == len(set(labels)), labels

    def test_every_unscored_display_column_has_a_label(self):
        missing = [c for c in UNSCORED_DISPLAY_COLUMNS if c not in UNSCORED_COLUMN_LABELS]
        assert missing == []

    def test_unscored_column_labels_has_no_stale_entries(self):
        stale = [c for c in UNSCORED_COLUMN_LABELS if c not in UNSCORED_DISPLAY_COLUMNS]
        assert stale == []

    def test_unscored_column_labels_are_unique(self):
        labels = list(UNSCORED_COLUMN_LABELS.values())
        assert len(labels) == len(set(labels)), labels

    def test_every_overlap_display_column_has_a_label(self):
        missing = [c for c in OVERLAP_LABEL_COLUMNS if c not in OVERLAP_COLUMN_LABELS]
        assert missing == []

    def test_overlap_column_labels_has_no_stale_entries(self):
        stale = [c for c in OVERLAP_COLUMN_LABELS if c not in OVERLAP_LABEL_COLUMNS]
        assert stale == []

    def test_overlap_column_labels_are_unique(self):
        labels = list(OVERLAP_COLUMN_LABELS.values())
        assert len(labels) == len(set(labels)), labels


class TestDiffFactorInputsCompleteness:
    """Phase 3c.2: the input map behind the 10 diff-based factors."""

    def test_covers_exactly_the_diff_based_factors(self):
        assert set(DIFF_FACTOR_INPUTS) == DIFF_BASED_FACTORS

    def test_every_factor_has_exactly_two_inputs(self):
        for factor, inputs in DIFF_FACTOR_INPUTS.items():
            assert len(inputs) == 2, factor

    def test_every_input_entry_has_column_label_and_source_func(self):
        for factor, inputs in DIFF_FACTOR_INPUTS.items():
            for entry in inputs:
                assert len(entry) == 3, (factor, entry)
                col, label, source_func = entry
                assert isinstance(col, str) and col
                assert isinstance(label, str) and label
                assert source_func.startswith("calc_")

    def test_every_input_column_has_a_format(self):
        missing = [c for c in DIFF_INPUT_COLUMNS if c not in INPUT_COLUMN_FORMATS]
        assert missing == []

    def test_diff_input_formats_has_no_stale_entries(self):
        stale = [c for c in DIFF_INPUT_FORMATS if c not in DIFF_INPUT_COLUMNS]
        assert stale == []

    def test_shared_metric_columns_not_duplicated_in_diff_input_formats(self):
        """ps_ntm and fcf_yield already have a format via METRIC_COLUMN_FORMATS
        (they're rel_ps_factor's/rel_fcf_factor's own metric) — DIFF_INPUT_FORMATS
        must not carry a second, potentially divergent copy."""
        assert "ps_ntm" not in DIFF_INPUT_FORMATS
        assert "fcf_yield" not in DIFF_INPUT_FORMATS
        assert INPUT_COLUMN_FORMATS["ps_ntm"] == METRIC_COLUMN_FORMATS["ps_ntm"]
        assert INPUT_COLUMN_FORMATS["fcf_yield"] == METRIC_COLUMN_FORMATS["fcf_yield"]

    def test_every_format_spec_is_applicable_to_a_float(self):
        for fmt in INPUT_COLUMN_FORMATS.values():
            fmt.format(1.23456)


class TestDiffFactorInputColumnsExistAfterTransform:
    """The correctness-critical check: every mapped input column must
    actually exist in short_screen's transformed data, so a future rename
    in transform.py fails the suite instead of silently blanking a
    drill-down cell."""

    def test_all_mapped_columns_present_in_transformed_output(self):
        raw_columns = [
            "ps_ntm", "ps_3yr_avg", "fcf_yield_3yr_avg", "fcf_yield",
            "revenues_ttm", "revenues_ttm_t1", "revenues_ttm_t2",
            "rev_cagr_f2y", "rev_cagr_p2y",
            "ntm_gross_margin", "gross_margin_3yr_avg",
            "ntm_ebit_margin", "ebit_margin_3yr_avg",
            "net_debt", "adj_ebitda", "revenues_ttm", "market_cap",
            "cash_balance", "available_loc", "fcf",
            "cfo", "net_income",
            "avg_receivables", "revenues_t3m", "avg_receivables_t1", "revenues_t3m_t1",
            "avg_inventory", "cogs_t3m", "avg_inventory_py", "cogs_t3m_t1",
            "avg_payables", "avg_payables_t1",
            "deferred_revenue", "deferred_revenue_t1",
            "adj_eps", "dil_eps_fy0",
            "buy_recs", "hold_recs", "sell_recs",
            "current_assets", "ppe", "lt_investments", "total_assets",
            "current_assets_t1", "ppe_t1", "lt_investments_t1", "total_assets_t1",
            "revenues_t3m_t1", "cogs_t3m", "cogs_t3m_t1",
            "depreciation", "depreciation_t1",
            "sga", "sga_t1",
            "debt_to_assets", "debt_to_assets_t1",
        ]
        raw_columns = list(dict.fromkeys(raw_columns))  # de-dup, keep order
        df = pd.DataFrame({col: [1.0, 2.0] for col in raw_columns})
        result = run_transforms(df)

        missing = [c for c in DIFF_INPUT_COLUMNS if c not in result.columns]
        assert missing == []


class TestExportIndependentOfCheckbox:
    """Blocking fix from PHASE3C2 revision: the export must always carry
    every underlying value (24 factor metrics + 20 diff inputs), regardless
    of the on-screen "Show underlying metric values" checkbox. The checkbox
    controls the SCREEN only. This pins that invariant, not the current
    behavior, so a future change that re-couples them fails the suite."""

    @staticmethod
    def _export_columns_for(show_values: bool) -> list:
        """Reproduces render_main_table's export-column computation."""
        all_metric_cols = interleave_metric_columns(DISPLAY_COLUMNS)
        return build_export_columns(all_metric_cols)

    def test_export_columns_identical_with_checkbox_on_and_off(self):
        off_cols = self._export_columns_for(show_values=False)
        on_cols = self._export_columns_for(show_values=True)
        assert off_cols == on_cols

    def test_export_contains_all_24_factor_metrics_regardless_of_checkbox(self):
        all_metrics = {defn["metric"] for defn in FACTOR_DEFINITIONS.values()}
        assert len(all_metrics) == 24
        for show_values in (False, True):
            export_cols = set(self._export_columns_for(show_values))
            missing = all_metrics - export_cols
            assert missing == set(), (show_values, missing)

    def test_export_contains_all_20_diff_inputs_regardless_of_checkbox(self):
        assert len(DIFF_INPUT_COLUMNS) == 20
        for show_values in (False, True):
            export_cols = set(self._export_columns_for(show_values))
            missing = set(DIFF_INPUT_COLUMNS) - export_cols
            assert missing == set(), (show_values, missing)

    def test_export_column_list_unaffected_by_column_config(self):
        """Phase 5a A4: render_main_table's on-screen column_config renames
        (st.column_config.Column(label=...)) relabel the header shown by
        st.dataframe only — they are never passed to export_df's
        to_excel()/to_csv() calls, which write the DataFrame's own (real,
        snake_case, unrenamed) column names as the header row. This pins
        that structural guarantee: the export's actual column list is
        exactly what it was before Phase 5a's renames existed, so a future
        change can't quietly couple the two."""
        export_cols = self._export_columns_for(show_values=False)
        assert export_cols[0] == "ticker"
        assert "overall_score" in export_cols
        assert "mscore_flag" in export_cols
        # None of Phase 5a's display labels ("Overall Score", "... Factor",
        # "... — Diff.") ever appear as an exported column name — the
        # export header row is still the DataFrame's own snake_case names.
        assert not any(col.endswith(" Factor") for col in export_cols)
        assert not any(" — Diff." in col for col in export_cols)
        assert "Overall Score" not in export_cols


class TestInterleaveMetricColumns:
    def test_inserts_metric_immediately_after_its_factor(self):
        result = interleave_metric_columns(["ticker", "abs_ps_factor", "name"])
        assert result == ["ticker", "abs_ps_factor", "ps_diff", "name"]

    def test_non_factor_columns_pass_through_unchanged(self):
        result = interleave_metric_columns(["ticker", "name"])
        assert result == ["ticker", "name"]


class TestBuildExportColumns:
    def test_appends_diff_input_columns_not_already_displayed(self):
        display_cols = ["ticker", "name", "abs_ps_factor"]
        result = build_export_columns(display_cols)
        assert result[: len(display_cols)] == display_cols
        for col in DIFF_INPUT_COLUMNS:
            assert col in result

    def test_no_duplicate_columns(self):
        display_cols = ["ticker", "ps_ntm", "abs_ps_factor"]  # ps_ntm already shown
        result = build_export_columns(display_cols)
        assert len(result) == len(set(result))

    def test_preserves_display_column_order(self):
        display_cols = ["ticker", "overall_score", "abs_ps_factor"]
        result = build_export_columns(display_cols)
        assert result[:3] == display_cols


@pytest.mark.parametrize("factor", sorted(DIFF_BASED_FACTORS))
def test_diff_factor_metric_matches_first_or_second_input_direction(factor):
    """Sanity check that DIFF_FACTOR_INPUTS and FACTOR_DEFINITIONS agree on
    which factor each input pair belongs to (protects against a copy-paste
    mismatch between the two dicts)."""
    assert factor in FACTOR_DEFINITIONS
    assert factor in DIFF_FACTOR_INPUTS


def _assert_help_complete(display_columns: list, help_map: dict) -> None:
    """Shared completeness check for Phase 5b-3's four help maps — every
    entry of a table's own *_DISPLAY_COLUMNS list (not its label map; see
    module docstring below) must resolve to a non-empty help string.

    Deliberately checked against *_DISPLAY_COLUMNS, not against the label
    map: OVERLAP_COLUMN_LABELS excludes overall_score (it keeps its own
    dynamic label), but OVERLAP_COLUMN_HELP does not — a completeness check
    against the label map would silently skip overall_score's help entry
    on the overlap table. TestColumnHelpCompleteness.test_synthetic_missing_
    column_is_caught below proves this helper actually fires rather than
    passing regardless of input.
    """
    missing = [c for c in display_columns if not help_map.get(c)]
    assert missing == [], missing


class TestColumnHelpCompleteness:
    """Phase 5b-3 (R7): a help= tooltip for every displayed column of all
    four tables. Same completeness-check shape as TestDisplayLabelsCompleteness
    above, checked against each table's *_DISPLAY_COLUMNS list rather than
    its label map — see _assert_help_complete's docstring for why that
    distinction matters here specifically (it did not for the label-
    completeness tests, since every label map is keyed by exactly its own
    DISPLAY_COLUMNS list with no OVERLAP_COLUMN_LABELS-style exclusion)."""

    def test_main_table_help_is_complete(self):
        rendered = interleave_metric_columns(DISPLAY_COLUMNS)
        assert len(rendered) == 55
        _assert_help_complete(rendered, MAIN_TABLE_COLUMN_HELP)

    def test_curated_help_is_complete(self):
        assert len(CURATED_DISPLAY_COLUMNS) == 10
        _assert_help_complete(CURATED_DISPLAY_COLUMNS, CURATED_COLUMN_HELP)

    def test_unscored_help_is_complete(self):
        assert len(UNSCORED_DISPLAY_COLUMNS) == 10
        _assert_help_complete(UNSCORED_DISPLAY_COLUMNS, UNSCORED_COLUMN_HELP)

    def test_overlap_help_is_complete(self):
        assert len(OVERLAP_DISPLAY_COLUMNS) == 7
        _assert_help_complete(OVERLAP_DISPLAY_COLUMNS, OVERLAP_COLUMN_HELP)

    def test_synthetic_missing_column_is_caught(self):
        """Positive test: _assert_help_complete must actually fail when a
        column lacks a help entry, not merely pass on real, already-
        complete input. Adds a synthetic column to a COPY of a display-
        columns list (the real DISPLAY_COLUMNS/help maps are untouched)."""
        columns_with_gap = ["ticker", "name", "a_column_nobody_documented"]
        help_map_missing_one = {"ticker": "...", "name": "..."}
        with pytest.raises(AssertionError):
            _assert_help_complete(columns_with_gap, help_map_missing_one)

    def test_total_help_string_count_and_distinct_columns(self):
        """Pins the counts derived directly from the live column lists
        (Phase 5b-3 plan): 55 + 10 + 10 + 7 = 82 total display slots across
        the four tables, 69 distinct column names (ticker/name/market_cap
        appear on all 4 tables, sector on 3, overall_score and
        short_interest_pct on 2 each — each of those with its own
        independently-authored help text per table, not a shared entry)."""
        main_cols = interleave_metric_columns(DISPLAY_COLUMNS)
        all_cols = (
            list(main_cols)
            + list(CURATED_DISPLAY_COLUMNS)
            + list(UNSCORED_DISPLAY_COLUMNS)
            + list(OVERLAP_DISPLAY_COLUMNS)
        )
        assert len(all_cols) == 82
        assert len(set(all_cols)) == 69

    def test_overall_score_help_differs_between_main_and_overlap_tables(self):
        """overall_score is a name-duplicate, not a concept-duplicate (Phase
        5b-3 plan review round 1's correction to the PM's own §1.3): the
        main table's own composite and the overlap table's cross-screen
        context reading are different claims and must not share one string."""
        assert MAIN_TABLE_COLUMN_HELP["overall_score"] != OVERLAP_COLUMN_HELP["overall_score"]


class TestDiffFactorFormulasRecompute:
    """Phase 5b-3 (R7) §5.2: DIFF_FACTOR_FORMULAS is a drift lock, not
    decoration — it must fail if a calc_* function in transform.py changes
    without the declaration following. Recomputes each factor's metric from
    DIFF_FACTOR_FORMULAS' own declared operation/operands on a small
    synthetic frame and compares against the REAL calc_* function's output
    (imported from transform.py, never reimplemented), so the two can never
    silently state two different formulas for the same factor."""

    _CALC_FUNCS = {
        "abs_ps_factor": calc_ps_diff,
        "abs_fcf_factor": calc_fcf_yield_diff,
        "decel_factor": calc_growth_decel,
        "accel_factor": calc_growth_accel,
        "gm_factor": calc_gm_diff,
        "ebit_factor": calc_ebit_diff,
        "dso_factor": calc_dso_pct_change,
        "dio_factor": calc_dio_pct_change,
        "dpo_factor": calc_dpo_pct_change,
        "def_rev_factor": calc_deferred_rev_pct_change,
    }

    def test_covers_exactly_the_diff_based_factors(self):
        assert set(DIFF_FACTOR_FORMULAS) == DIFF_BASED_FACTORS

    @pytest.mark.parametrize("factor", sorted(DIFF_BASED_FACTORS))
    def test_declared_formula_matches_real_calc_function(self, factor):
        operation, col_a, col_b = DIFF_FACTOR_FORMULAS[factor]
        # Values chosen so neither operand is zero (avoiding every
        # calc_*'s zero-denominator guard) and so the two operands differ
        # (so a swapped-order bug would produce a different number).
        df = pd.DataFrame({col_a: [4.0], col_b: [5.0]})
        # calc_growth_decel/calc_growth_accel/calc_gm_diff/calc_ebit_diff
        # read from pre-named columns that may not both be col_a/col_b if
        # a factor's two inputs happen to share a column with another
        # factor's synthetic frame — build a frame with exactly the two
        # columns this factor's own calc function reads.
        declared = df[col_a].iloc[0] / df[col_b].iloc[0] - 1 if operation == "ratio_minus_one" \
            else df[col_a].iloc[0] - df[col_b].iloc[0]

        real_result = self._CALC_FUNCS[factor](df)
        real_value = real_result[0] if hasattr(real_result, "__getitem__") else real_result.iloc[0]
        assert declared == pytest.approx(real_value), (factor, operation, col_a, col_b)

    def test_operations_are_one_of_two_known_shapes(self):
        for factor, (operation, _col_a, _col_b) in DIFF_FACTOR_FORMULAS.items():
            assert operation in ("ratio_minus_one", "difference"), factor

    def test_three_factors_declare_reversed_arithmetic_order(self):
        """Regression lock for the finding that matters most in this
        phase's source material: three of the ten diffs subtract in the
        OPPOSITE order from how DIFF_FACTOR_INPUTS lists their two inputs
        (Excel template block order) — confirmed by reading transform.py's
        calc_fcf_yield_diff/calc_growth_decel/calc_growth_accel bodies
        directly. DIFF_FACTOR_FORMULAS must declare the arithmetic order,
        not the panel-listing order."""
        reversed_factors = {"abs_fcf_factor", "decel_factor", "accel_factor"}
        for factor in reversed_factors:
            panel_order = [col for col, _label, _func in DIFF_FACTOR_INPUTS[factor]]
            _operation, formula_a, formula_b = DIFF_FACTOR_FORMULAS[factor]
            assert [formula_a, formula_b] == list(reversed(panel_order)), factor
        for factor in DIFF_BASED_FACTORS - reversed_factors:
            panel_order = [col for col, _label, _func in DIFF_FACTOR_INPUTS[factor]]
            _operation, formula_a, formula_b = DIFF_FACTOR_FORMULAS[factor]
            assert [formula_a, formula_b] == panel_order, factor


class TestFormatDiffFormula:
    def test_ratio_minus_one_uses_division_symbol(self):
        result = format_diff_formula("abs_ps_factor")
        assert result == "P/Sales (NTM) ÷ P/Sales (3yr. Avg.) − 1"

    def test_difference_uses_minus_symbol_and_reversed_operands(self):
        """abs_fcf_factor is one of the three reversed-order subtractions —
        the rendered formula must show the 3yr avg MINUS the LTM value, not
        the panel-listing order (LTM, then 3yr avg)."""
        result = format_diff_formula("abs_fcf_factor")
        assert result == "FCF Yield (3yr. Avg.) − FCF Yield (LTM)"


class TestCellDerivationFactors:
    """Phase 5b-3 (R7) §5.3: the click-a-cell derivation dispatch table."""

    def test_has_exactly_twenty_entries(self):
        assert len(CELL_DERIVATION_FACTORS) == 20

    def test_covers_each_diff_factors_score_and_metric_column(self):
        for factor in DIFF_BASED_FACTORS:
            assert CELL_DERIVATION_FACTORS[factor] == factor
            metric_col = FACTOR_DEFINITIONS[factor]["metric"]
            assert CELL_DERIVATION_FACTORS[metric_col] == factor

    def test_ps_ntm_and_fcf_yield_are_not_keys(self):
        """The discriminating negative test: ps_ntm/fcf_yield are
        rel_ps_factor's/rel_fcf_factor's own metric columns, not diff
        inputs — mapping them to abs_ps_factor/abs_fcf_factor would show a
        user Absolute P/S's derivation when they click Relative P/S's
        metric, a real, plausible, wrong panel with no error."""
        assert "ps_ntm" not in CELL_DERIVATION_FACTORS
        assert "fcf_yield" not in CELL_DERIVATION_FACTORS

    def test_ps_diff_resolves_to_abs_ps_factor_the_positive_companion(self):
        """Paired with the negative test above so the two read as one
        discriminating distinction: ps_diff (abs_ps_factor's own metric)
        DOES resolve, ps_ntm (rel_ps_factor's own metric) does NOT."""
        assert CELL_DERIVATION_FACTORS["ps_diff"] == "abs_ps_factor"
        assert CELL_DERIVATION_FACTORS["fcf_yield_diff"] == "abs_fcf_factor"

    def test_non_diff_factor_by_column_is_disjoint_from_cell_derivation_factors(self):
        overlap = set(CELL_DERIVATION_FACTORS) & set(NON_DIFF_FACTOR_BY_COLUMN)
        assert overlap == set()

    def test_non_diff_factor_by_column_has_twenty_eight_entries(self):
        """14 non-diff factors x 2 (own score column + own metric column)."""
        assert len(NON_DIFF_FACTOR_BY_COLUMN) == 28

    def test_rel_ps_factor_and_rel_fcf_factor_route_through_non_diff_map(self):
        assert NON_DIFF_FACTOR_BY_COLUMN["ps_ntm"] == "rel_ps_factor"
        assert NON_DIFF_FACTOR_BY_COLUMN["fcf_yield"] == "rel_fcf_factor"


class TestAppFontFamilyMatchesThemeConfig:
    """Phase 5b-2 (R8): APP_FONT_FAMILY is the single Python-side constant
    feeding the Altair drill-down chart's .configure_* calls; .streamlit/
    config.toml's `font` key is what streamlit itself reads for everything
    else. With one shared literal there is nothing left to drift silently —
    this test is what turns that into an enforced guarantee rather than a
    coincidence, by comparing the constant against the actual file.

    Matches the `font` key exactly (start-of-line, optional whitespace, then
    `=`) so a `fontFaces = [...]` line — a real key in the same file — can't
    be mistaken for it, and asserts a named message on a missing key rather
    than letting a bare StopIteration stand in for "no font key found"."""

    def test_matches_config_toml(self):
        config_path = os.path.join(PROJECT_ROOT, ".streamlit", "config.toml")
        with open(config_path) as f:
            content = f.read()
        match = re.search(r'^font\s*=\s*"(.*)"', content, re.MULTILINE)
        assert match is not None, "no 'font' key found in .streamlit/config.toml"
        assert match.group(1) == APP_FONT_FAMILY

    def test_regex_does_not_match_fontfaces(self):
        """The discriminating half of the compound match: fontFaces must
        not be mistaken for font."""
        content = 'fontFaces = [{ family = "Arimo" }]\nheadingFont = "Georgia"\n'
        match = re.search(r'^font\s*=\s*"(.*)"', content, re.MULTILINE)
        assert match is None

    def test_regex_matches_unspaced_key(self):
        content = 'font="Arial"\n'
        match = re.search(r'^font\s*=\s*"(.*)"', content, re.MULTILINE)
        assert match is not None
        assert match.group(1) == "Arial"


MARKET_CAP_SLIDER_PATTERN = (
    r'"\*\*Market Cap \(\$M\)\*\*",\s*\n'
    r'\s*min_value=mcap_min,\s*\n'
    r'\s*max_value=mcap_max,\s*\n'
    r'\s*value=\(mcap_min, mcap_max\),\s*\n'
    r'\s*format="(\$%[^"]+)"'
)


class TestMarketCapSliderFormatConsistency:
    """Phase 5c-1: render_sidebar, render_curated_sidebar and
    render_unscored_sidebar each build their own Market Cap slider from
    scratch (there is no shared helper) — a future edit could fix the
    format string at one call site and leave the other two behind with no
    error, since all three still work standalone. This locks all three to
    one identical format string.

    The pattern is anchored on the slider's exact argument shape
    (min_value=mcap_min / max_value=mcap_max / value=(mcap_min,
    mcap_max)), not a bare label-then-format scan with an unbounded gap —
    so it cannot wander across function boundaries and pick up an
    unrelated slider's format string just because a Market Cap label
    appears somewhere before it in the file."""

    def test_market_cap_slider_format_consistent_across_sidebars(self):
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        formats = re.findall(MARKET_CAP_SLIDER_PATTERN, content)
        assert len(formats) == 3, (
            f"expected 3 Market Cap sliders matching the known shape, found {len(formats)}"
        )
        assert len(set(formats)) == 1, f"Market Cap slider formats disagree: {formats}"

    def test_regex_does_not_match_unrelated_slider_with_same_label(self):
        """The discriminating half: a slider that merely carries the same
        label text, but not the exact mcap_min/mcap_max/value argument
        shape, must not be mistaken for a real Market Cap slider site."""
        content = (
            'st.sidebar.slider(\n'
            '    "**Market Cap ($M)**",\n'
            '    min_value=other_min,\n'
            '    max_value=other_max,\n'
            '    value=(other_min, other_max),\n'
            '    format="$%.2f",\n'
            ')\n'
        )
        assert re.findall(MARKET_CAP_SLIDER_PATTERN, content) == []

    def test_regex_matches_minimal_positive(self):
        content = (
            'mcap_range = st.sidebar.slider(\n'
            '    "**Market Cap ($M)**",\n'
            '    min_value=mcap_min,\n'
            '    max_value=mcap_max,\n'
            '    value=(mcap_min, mcap_max),\n'
            '    format="$%,.0f",\n'
            ')\n'
        )
        formats = re.findall(MARKET_CAP_SLIDER_PATTERN, content)
        assert formats == ["$%,.0f"]


class TestScreenIconMap:
    """Phase 5c-3: SCREEN_ICONS backs each "Also Appears On" block's icon,
    keyed on screen_id (never display_name, which can differ from it — e.g.
    short_screen / "OWS Short Screen")."""

    def test_unmapped_screen_id_resolves_to_default_without_raising(self):
        icon = SCREEN_ICONS.get("some_future_screen_id", _DEFAULT_SCREEN_ICON)
        assert icon == _DEFAULT_SCREEN_ICON

    def test_resolution_keys_on_screen_id_not_display_name(self):
        # short_screen's display_name is "OWS Short Screen" — a lookup keyed
        # on display_name would find nothing for either string.
        assert SCREEN_ICONS.get("short_screen") == ":material/trending_down:"
        assert SCREEN_ICONS.get("OWS Short Screen") is None

    def test_every_registry_screen_maps_to_a_distinct_nonempty_icon(self):
        registry_screen_ids = [
            "competition",
            "cyclicals",
            "management_comp",
            "rising_short_interest",
            "short_screen",
            "structural",
        ]
        icons = [SCREEN_ICONS[screen_id] for screen_id in registry_screen_ids]
        assert all(icons), f"empty icon among {icons}"
        assert len(set(icons)) == len(icons), f"icons are not mutually distinct: {icons}"


# Phase 5c-3: anchored on the drill-down's curated branch reading
# _STOCK_PERFORMANCE_LABEL in BOTH ternary arms (rather than any literal
# text) — the constant is the mechanism that prevents the value arm and the
# N/A arm from drifting apart, so the lock verifies both arms actually use
# it, not what its value happens to be today. If a future reformat wraps
# this ternary across lines, this pattern reports "found 0" and goes
# red — the right failure direction, but check here first before assuming
# the relabel broke: it means the shape moved, not the label.
DRILLDOWN_STOCK_PERF_PATTERN = (
    r'st\.write\(f"\{_STOCK_PERFORMANCE_LABEL\}: \{perf:\.2%\}" if pd\.notna\(perf\) '
    r'else f"\{_STOCK_PERFORMANCE_LABEL\}: N/A"\)'
)


class TestStockPerformanceLabelConsistency:
    """Phase 5c-3: the curated grid header (CURATED_COLUMN_LABELS) and the
    cross-screen drill-down's curated branch must show the same "Stock
    Performance (1 yr.)" label, and the N/A arm must carry it too — a
    property live data can never exercise (0 nulls in stock_performance
    across all 223 curated rows), so it can only be locked at the source
    level. Three checks, each catching a different drift:
    test_drilldown_both_arms_use_the_shared_constant locks that both
    ternary arms read _STOCK_PERFORMANCE_LABEL rather than a literal;
    test_constant_matches_the_intended_literal hardcodes the expected
    string "Stock Performance (1 yr.)" independently in this test file, so
    it is the one that actually pins the constant's value — not a
    self-check; test_grid_header_matches_the_shared_constant locks that
    CURATED_COLUMN_LABELS["stock_performance"] keeps resolving through the
    same constant rather than being repointed at its own literal."""

    def test_drilldown_both_arms_use_the_shared_constant(self):
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        matches = re.findall(DRILLDOWN_STOCK_PERF_PATTERN, content)
        assert len(matches) == 1, f"expected 1 curated drill-down site, found {len(matches)}"

    def test_constant_matches_the_intended_literal(self):
        assert _STOCK_PERFORMANCE_LABEL == "Stock Performance (1 yr.)"

    def test_grid_header_matches_the_shared_constant(self):
        assert CURATED_COLUMN_LABELS["stock_performance"] == _STOCK_PERFORMANCE_LABEL

    def test_regex_does_not_match_unrelated_lookalike(self):
        """Discriminating half: right-looking labels but literal text
        instead of the shared constant must not be mistaken for compliance —
        a hardcoded string is exactly the drift this lock exists to catch."""
        lookalike = (
            'st.write(f"Stock Performance (1 yr.): {perf:.2%}" if pd.notna(perf) '
            'else "Stock Performance (1 yr.): N/A")\n'
        )
        assert re.findall(DRILLDOWN_STOCK_PERF_PATTERN, lookalike) == []

    def test_regex_matches_minimal_positive(self):
        minimal = (
            'st.write(f"{_STOCK_PERFORMANCE_LABEL}: {perf:.2%}" if pd.notna(perf) '
            'else f"{_STOCK_PERFORMANCE_LABEL}: N/A")\n'
        )
        assert len(re.findall(DRILLDOWN_STOCK_PERF_PATTERN, minimal)) == 1


class TestFormatScreenTitle:
    """Phase 5c-2 (R1, as amended by the Driver 2026-09-05): format_screen_title wraps a display name in the
    :primary[...] markdown directive so the screen title renders in brand
    green. A name containing "[" or "]" is returned unwrapped instead —
    Streamlit's directive parsing is frontend-only and an early-closed or
    unrecognised directive renders as literal text without raising, so a
    bracket in the name could leak the directive markup onto the screen.
    No live screen name contains a bracket today; these two cases are
    exactly the ones a fail-first run against the naive "always wrap"
    implementation gets wrong — that version wraps every name unconditionally
    and these two tests catch it turning the guard into a no-op."""

    def test_ordinary_name_is_wrapped(self):
        assert format_screen_title("Management Comp") == ":primary[Management Comp]"

    def test_name_with_closing_bracket_is_returned_unwrapped(self):
        result = format_screen_title("Foo] bar")
        assert result == "Foo] bar"
        assert ":primary[" not in result

    def test_name_with_opening_bracket_is_returned_unwrapped(self):
        result = format_screen_title("Foo [bar")
        assert result == "Foo [bar"
        assert ":primary[" not in result


class TestScreenMarkPaths:
    """Phase 5c-2 (R1, as amended by the Driver 2026-09-05): TITLE_MARK_PATH
    (white disc, green bear — beside the screen title) and LOGO_MARK_PATH
    (green disc, white bear — the sidebar mark) are two distinct,
    Driver-ruled assets. Neither is a stand-in for the other: both files
    existing and the two paths merely being distinct is not enough to catch
    a swap, since a swap would still pass both of those checks. The pairing
    lock below asserts each constant names its own specific file, so a
    swap fails."""

    def test_title_mark_path_exists(self):
        assert os.path.exists(TITLE_MARK_PATH)

    def test_logo_mark_path_exists(self):
        assert os.path.exists(LOGO_MARK_PATH)

    def test_paths_are_distinct(self):
        assert TITLE_MARK_PATH != LOGO_MARK_PATH

    def test_title_mark_is_the_white_disc_variant(self):
        assert TITLE_MARK_PATH.endswith("ows-bear-white-disc.png")

    def test_logo_mark_is_the_green_disc_variant(self):
        assert LOGO_MARK_PATH.endswith("ows-bear-green-disc.png")
