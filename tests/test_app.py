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

import pandas as pd
import pytest

from src.app import (
    DIFF_FACTOR_INPUTS,
    DIFF_INPUT_COLUMNS,
    DIFF_INPUT_FORMATS,
    DISPLAY_COLUMNS,
    FACTOR_DEFINITIONS,
    INPUT_COLUMN_FORMATS,
    METRIC_COLUMN_FORMATS,
    METRIC_FORMATS,
    build_export_columns,
    interleave_metric_columns,
)
from src.transform import run_transforms

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
