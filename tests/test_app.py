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

import ast
import os
import re

import pandas as pd
import pytest
from streamlit.string_util import validate_icon_or_emoji

import streamlit as st

from src.app import (
    APP_FONT_FAMILY,
    CATEGORY_SUM_COLUMN_HELP,
    CATEGORY_SUM_COLUMN_LABELS,
    CELL_DERIVATION_FACTORS,
    CURATED_COLUMN_HELP,
    CURATED_COLUMN_LABELS,
    CURATED_DISPLAY_COLUMNS,
    DIFF_FACTOR_FORMULAS,
    DIFF_FACTOR_INPUTS,
    DIFF_INPUT_COLUMNS,
    DIFF_INPUT_FORMATS,
    DISPLAY_COLUMNS,
    EXPORT_BUTTON_ICON,
    FACTOR_CATEGORIES,
    FACTOR_COLUMN_LABELS,
    FACTOR_DEFINITIONS,
    INPUT_COLUMN_FORMATS,
    LOGO_MARK_PATH,
    MAIN_TABLE_COLUMN_HELP,
    MAIN_TABLE_COLUMN_LABELS,
    MAIN_TABLE_FLAG_COLUMNS,
    METRIC_COLUMN_FORMATS,
    METRIC_COLUMN_LABELS,
    METRIC_FORMATS,
    NON_DIFF_FACTOR_BY_COLUMN,
    OVERLAP_COLUMN_HELP,
    OVERLAP_COLUMN_LABELS,
    OVERLAP_DISPLAY_COLUMNS,
    OVERLAP_EXTRA_NA_REPS,
    OVERLAP_METRIC_FILL_VALUES,
    OVERLAP_METRIC_JOINS,
    OVERLAP_PSEUDO_DISPLAY_NAME,
    OVERLAP_PSEUDO_SCREEN_ID,
    OVERVALUED_SCREEN_ID,
    SCREEN_ICONS,
    TITLE_MARK_PATH,
    TRANSCRIPTS_SCREEN_ID,
    UNSCORED_COLUMN_HELP,
    UNSCORED_COLUMN_LABELS,
    UNSCORED_DISPLAY_COLUMNS_BY_SCREEN,
    UNSCORED_DISPLAY_COLUMNS_UNION,
    UNSCORED_METRIC_DISPLAY_NAMES,
    UNSCORED_METRIC_FORMATS,
    _DEFAULT_SCREEN_ICON,
    _STOCK_PERFORMANCE_LABEL,
    apply_overlap_metric_joins,
    build_export_columns,
    build_overlap_help_map,
    build_screen_selector_options,
    compute_screen_membership_flag,
    format_diff_formula,
    format_screen_title,
    format_unscored_metric_value,
    insert_config_weight_export_column,
    interleave_metric_columns,
    render_cross_screen_context,
    render_overlap_page,
    render_transcript_takeaways,
    render_unscored_drill_down,
    render_unscored_sidebar,
    resolve_expanded_display_columns,
    resolve_persisted_preset,
    resolve_transcript_takeaways_for_ticker,
    resolve_unscored_display_columns,
    should_reapply_preset,
    style_unscored_table,
    transcripts_for_ticker,
    unscored_export_basename,
)
from src.cross_screen_context import build_screen_contribution, classify_screen
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

# Phase 6c: a small synthetic overlap_df + screens_df for build_overlap_
# help_map's tests, entirely independent of the live, gitignored
# data/screener.db (T25). 5 tickers: 2 in_universe on 0 thematic screens,
# 1 in_universe on 1 thematic screen, 1 thematic-only (not in_universe),
# 1 fully in_universe+thematic — so every branch of the derivation
# (universe_total, zero_count, thematic_only, total_rows) has a
# non-degenerate value to check against.
_SYNTHETIC_OVERLAP_DF = pd.DataFrame({
    "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
    "screen_count": [0, 0, 1, 0, 2],
    "in_universe": [True, True, True, False, True],
})
_SYNTHETIC_SCREENS_DF = pd.DataFrame({
    "screen_id": ["short_screen", "structural", "competition"],
    "display_name": ["OWS Short Screen", "Structural", "Competition"],
    "screen_type": ["quant_composite", "curated", "curated"],
    "has_scoring": [True, False, False],
})


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
        missing = [c for c in UNSCORED_DISPLAY_COLUMNS_UNION if c not in UNSCORED_COLUMN_LABELS]
        assert missing == []

    def test_unscored_column_labels_has_no_stale_entries(self):
        stale = [c for c in UNSCORED_COLUMN_LABELS if c not in UNSCORED_DISPLAY_COLUMNS_UNION]
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
        # 64: Phase 8c-2's 40-column DISPLAY_COLUMNS (identity 5 +
        # overall_score 1 + 7 category sums + 24 factors + 3 flags) plus
        # the 24 interleaved factor metrics.
        rendered = interleave_metric_columns(DISPLAY_COLUMNS)
        assert len(rendered) == 64
        _assert_help_complete(rendered, MAIN_TABLE_COLUMN_HELP)

    def test_curated_help_is_complete(self):
        assert len(CURATED_DISPLAY_COLUMNS) == 10
        _assert_help_complete(CURATED_DISPLAY_COLUMNS, CURATED_COLUMN_HELP)

    def test_unscored_help_is_complete(self):
        # 20: RSI's 10 plus mention_count/latest_transcript_date/
        # days_since_latest, de-duplicated on the shared "ticker" column,
        # plus Overvalued's 7 non-identity columns (Phase 8a; ticker/name
        # already counted).
        assert len(UNSCORED_DISPLAY_COLUMNS_UNION) == 20
        _assert_help_complete(UNSCORED_DISPLAY_COLUMNS_UNION, UNSCORED_COLUMN_HELP)

    def test_overlap_help_is_complete(self):
        # Phase 6c: 7 -> 8 (mention_count added). Phase 8a: 8 -> 9
        # (pe_vs_normal_5y added). screen_count/screens_on/overall_score/
        # mention_count/pe_vs_normal_5y are no longer in the static
        # OVERLAP_COLUMN_HELP constant (their sentences are derived — see
        # build_overlap_help_map), so completeness is checked against ITS
        # output, not the static dict directly.
        assert len(OVERLAP_DISPLAY_COLUMNS) == 9
        help_map = build_overlap_help_map(_SYNTHETIC_OVERLAP_DF, _SYNTHETIC_SCREENS_DF)
        _assert_help_complete(OVERLAP_DISPLAY_COLUMNS, help_map)

    def test_overvalued_screen_columns_all_have_labels_and_help(self):
        """Phase 8a, Testing posture item 6: universal over THIS screen's
        own display columns (never a hand-typed literal list — 6b's
        lesson), so a column added to UNSCORED_DISPLAY_COLUMNS_BY_SCREEN
        without a label/help entry fails here rather than shipping silent."""
        cols = UNSCORED_DISPLAY_COLUMNS_BY_SCREEN["overvalued_screen"]
        assert len(cols) == 9
        _assert_help_complete(cols, UNSCORED_COLUMN_HELP)
        for col in cols:
            assert col in UNSCORED_COLUMN_LABELS, f"{col} has no display label"

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
        """Pins the counts derived directly from the live column lists.
        Phase 6c added mention_count to OVERLAP_DISPLAY_COLUMNS (7 -> 8
        slots), but that name was already counted via the transcripts
        screen's own list, so the distinct-name count was unchanged at 72.
        Phase 8a adds pe_vs_normal_5y (8 -> 9 slots) — this name is NEW to
        this concatenation (it's enumerated by literal screen-id key here,
        so overvalued_screen's own 9-column list is never counted; only
        its overlap-joined column is), so the distinct count moves too.
        Phase 8c-2 moves the main table from 55 to 64 rendered slots (40-
        column DISPLAY_COLUMNS + 24 interleaved metrics vs. the prior 31 +
        24) and adds 9 new, non-colliding names (7 category sums + 2
        membership flags), moving the distinct count by the same +9.
        64 (main) + 10 (curated) + 10 (RSI) + 4 (transcripts) + 9 (overlap)
        = 97 total display slots, 82 distinct column names."""
        main_cols = interleave_metric_columns(DISPLAY_COLUMNS)
        all_cols = (
            list(main_cols)
            + list(CURATED_DISPLAY_COLUMNS)
            + list(UNSCORED_DISPLAY_COLUMNS_BY_SCREEN["rising_short_interest"])
            + list(UNSCORED_DISPLAY_COLUMNS_BY_SCREEN[TRANSCRIPTS_SCREEN_ID])
            + list(OVERLAP_DISPLAY_COLUMNS)
        )
        assert len(all_cols) == 97
        assert len(set(all_cols)) == 82

    def test_overall_score_help_differs_between_main_and_overlap_tables(self):
        """overall_score is a name-duplicate, not a concept-duplicate (Phase
        5b-3 plan review round 1's correction to the PM's own §1.3): the
        main table's own composite and the overlap table's cross-screen
        context reading are different claims and must not share one string.
        Phase 6c: overall_score's overlap help text is now derived (see
        build_overlap_help_map), so this reads the function's output rather
        than the retired _OVERLAP_OVERALL_SCORE_HELP constant."""
        help_map = build_overlap_help_map(_SYNTHETIC_OVERLAP_DF, _SYNTHETIC_SCREENS_DF)
        assert MAIN_TABLE_COLUMN_HELP["overall_score"] != help_map["overall_score"]

    def test_mention_count_help_differs_between_transcripts_and_overlap_tables(self):
        """Same shape as the overall_score lock above (Phase 6c, ruling 2f):
        a name-duplicate, not a concept-duplicate. This table's 0 means "not
        on the Negative Expert Transcripts screen at all," a claim the
        screen's own table never has to make about itself."""
        help_map = build_overlap_help_map(_SYNTHETIC_OVERLAP_DF, _SYNTHETIC_SCREENS_DF)
        assert UNSCORED_COLUMN_HELP["mention_count"] != help_map["mention_count"]


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


OVERALL_SCORE_SLIDER_PATTERN = (
    r'"\*\*Overall Score\*\*",\s*\n'
    r'\s*min_value=score_min,\s*\n'
    r'\s*max_value=score_max,\s*\n'
    r'\s*value=\(score_min, score_max\),\s*\n'
    r'\s*step=0\.1,'
)

# The pre-Phase-5d shape this replaces — a hardcoded 0.0/7.0 ceiling that was
# never the true one (config.yaml's weights already summed to 6.999999, and
# reweighting can push the true max well past 7.0). Property #4 (Phase 5d
# Worker prompt §7): must fire when this literal is reintroduced as a bound.
STALE_HARDCODED_SCORE_BOUND_PATTERN = (
    r'"\*\*Overall Score\*\*",\s*\n'
    r'\s*min_value=0\.0,\s*\n'
    r'\s*max_value=7\.0,'
)


class TestOverallScoreSliderBoundIsDerived:
    """Phase 5d: the Overall Score slider's bounds must be derived from the
    live data (score_min/score_max), matching the Market Cap slider's own
    pattern immediately above it — never a hardcoded literal, since
    reweighting can move the true max well past the old 7.0 ceiling
    (acceptance preset C pushes it to 7.390019).

    Fail-first / positive-control pair (Worker Rules — this is a compound
    condition on text this phase moved): run against the pre-edit source at
    d5f6646 (`git show d5f6646:src/app.py`), STALE_HARDCODED_SCORE_BOUND_
    PATTERN matches (the literal 0.0/7.0 bound is present) and
    OVERALL_SCORE_SLIDER_PATTERN does not (score_min/score_max didn't exist
    as slider bounds yet) — the fail-first run. Run against the post-edit
    source below — OVERALL_SCORE_SLIDER_PATTERN matches exactly once and
    STALE_HARDCODED_SCORE_BOUND_PATTERN matches zero times — the positive
    control. Both reported in the build report; this class only encodes the
    post-edit permanent lock (embedding a git-history read in a permanent
    test would make it fragile and slow for no lasting benefit)."""

    def test_overall_score_slider_bound_is_derived(self):
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        matches = re.findall(OVERALL_SCORE_SLIDER_PATTERN, content)
        assert len(matches) == 1, (
            f"expected exactly 1 Overall Score slider matching the derived-bound "
            f"shape, found {len(matches)}"
        )

    def test_stale_hardcoded_bound_is_gone(self):
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        assert re.findall(STALE_HARDCODED_SCORE_BOUND_PATTERN, content) == []

    def test_regex_matches_minimal_positive(self):
        content = (
            'score_range = st.sidebar.slider(\n'
            '    "**Overall Score**",\n'
            '    min_value=score_min,\n'
            '    max_value=score_max,\n'
            '    value=(score_min, score_max),\n'
            '    step=0.1,\n'
            ')\n'
        )
        assert len(re.findall(OVERALL_SCORE_SLIDER_PATTERN, content)) == 1
        assert re.findall(STALE_HARDCODED_SCORE_BOUND_PATTERN, content) == []

    def test_regex_matches_pre_phase_5d_shape(self):
        """Discriminating half: confirms STALE_HARDCODED_SCORE_BOUND_PATTERN
        actually recognizes the shape this phase removed (the fail-first
        evidence, reproduced as a permanent check rather than a git read)."""
        content = (
            'score_range = st.sidebar.slider(\n'
            '    "**Overall Score**",\n'
            '    min_value=0.0,\n'
            '    max_value=7.0,\n'
            '    value=(score_min, score_max),\n'
            '    step=0.1,\n'
            ')\n'
        )
        assert len(re.findall(STALE_HARDCODED_SCORE_BOUND_PATTERN, content)) == 1
        assert re.findall(OVERALL_SCORE_SLIDER_PATTERN, content) == []

    def test_degenerate_score_bounds_are_guarded(self):
        """Found live during Phase 5d acceptance testing: an all-zero
        weight state (legal under D9 — nothing stops an analyst zeroing
        every category) makes every stock's overall_score exactly 0.0, so
        score_min == score_max, and st.slider raises
        StreamlitInvalidMinMaxError when min_value == max_value. Confirms
        render_sidebar guards this before reaching the slider call, using
        the same house pattern as render_overlap_sidebar's ceiling==1
        case."""
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        assert "if score_min == score_max:" in content


class TestInsertConfigWeightExportColumn:
    """Phase 5d (D3), and the defect the PM's review caught in plan review:
    reverting the insertion into render_main_table's export_cols left every
    prior test green (nothing depended on it) — the exact way this nearly
    shipped with the export silently missing overall_score_config_weights.
    This locks the pure helper directly, independent of render_main_table.

    Mutation that must turn this red: reverting the insertion in
    src/app.py's insert_config_weight_export_column back to a no-op (i.e.
    `return export_cols`) — not editing this test. Run that mutation
    yourself before trusting this lock; do not just read the assertions."""

    def test_inserted_immediately_after_overall_score(self):
        export_cols = ["ticker", "name", "overall_score", "mscore_flag", "abs_ps_factor"]
        available_columns = export_cols + ["overall_score_config_weights"]
        result = insert_config_weight_export_column(export_cols, available_columns)
        assert result == [
            "ticker", "name", "overall_score", "overall_score_config_weights",
            "mscore_flag", "abs_ps_factor",
        ]

    def test_absent_column_passes_through_unchanged(self):
        export_cols = ["ticker", "name", "overall_score", "mscore_flag"]
        available_columns = export_cols  # overall_score_config_weights not present
        result = insert_config_weight_export_column(export_cols, available_columns)
        assert result == export_cols

    def test_no_overall_score_column_passes_through_unchanged(self):
        """Defensive: a caller that somehow lacks overall_score in export_cols
        (but still has overall_score_config_weights available) must not crash
        looking up an index that doesn't exist."""
        export_cols = ["ticker", "name", "mscore_flag"]
        available_columns = export_cols + ["overall_score_config_weights"]
        result = insert_config_weight_export_column(export_cols, available_columns)
        assert result == export_cols


class TestShouldReapplyPreset:
    """Phase 5d, the misattribution fix's pure predicate. Two independent
    triggers must both fire a reapply (name changed; weights dropped by
    Streamlit's widget pruning after a screen switch), and — the failure
    this must catch, per the PM's spec — it must NOT fire on an ordinary
    rerun where the same preset is selected and the widgets are present:
    that is the analyst mid-edit, and a spurious reapply there would
    silently discard their own typed value. Mutation-verified against the
    source (not just this test) below."""

    def test_fires_on_first_render(self):
        assert should_reapply_preset(None, "Default (config weights)", weights_absent=True) is True

    def test_fires_on_genuine_preset_change(self):
        assert should_reapply_preset("Default (config weights)", "B - Valuation only", weights_absent=False) is True

    def test_fires_when_weights_dropped_but_name_unchanged(self):
        """The exact scenario found live: same preset name, but the widgets
        were dropped by a screen switch and re-entry — must still reapply."""
        assert should_reapply_preset("B - Valuation only", "B - Valuation only", weights_absent=True) is True

    def test_does_not_fire_on_ordinary_mid_edit_rerun(self):
        """The failure it must catch: firing here would discard an
        in-progress edit — same preset name, widgets still present."""
        assert should_reapply_preset("B - Valuation only", "B - Valuation only", weights_absent=False) is False


class TestResolvePersistedPreset:
    """Phase 5d, the two live-reproduced failure modes from the PM's review:
    (a) data/weight_presets.yaml becomes unloadable mid-session (preset_
    options collapses to [Default] alone) while a non-Default preset was
    selected; (b) the selected preset is deleted — this session or another,
    since the file is shared — before this session's next rerun. Both were
    reproduced live against the real app (corrupt the file / empty it while
    "B - Valuation only" was active, trigger a rerun) and neither crashed
    even before this extraction — this locks the guard so a future edit
    can't quietly drop the fallback and reintroduce a KeyError on
    presets[selected_preset]."""

    def test_valid_persisted_name_passes_through_unchanged(self):
        result = resolve_persisted_preset(
            "B - Valuation only", ["Default (config weights)", "B - Valuation only"]
        )
        assert result == "B - Valuation only"

    def test_stale_persisted_name_falls_back_to_default(self):
        """Reproduces the live scenario: the file became unloadable or the
        preset was deleted, so preset_options no longer contains the name
        _weight_last_applied_preset still holds."""
        result = resolve_persisted_preset("B - Valuation only", ["Default (config weights)"])
        assert result == "Default (config weights)"

    def test_default_itself_is_always_valid(self):
        result = resolve_persisted_preset("Default (config weights)", ["Default (config weights)"])
        assert result == "Default (config weights)"


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
        """Phase 6b: derives the id list from the real config.yaml (tracked,
        not gitignored — T25 is satisfied) rather than a hand-copied
        mirror. The prior hardcoded six-id list is exactly why Phase 6a
        could add a seventh registry screen with no icon and this suite
        stayed green; this now covers all seven and KeyErrors for an
        eighth screen with no SCREEN_ICONS entry."""
        from src.config import CONFIG_PATH, load_config

        registry_screen_ids = list(load_config(CONFIG_PATH)["screens"].keys())
        icons = [SCREEN_ICONS[screen_id] for screen_id in registry_screen_ids]
        assert all(icons), f"empty icon among {icons}"
        assert len(set(icons)) == len(icons), f"icons are not mutually distinct: {icons}"


class TestScreenIconsAreValidMaterialIcons:
    """Phase 6b (queue item 6): insurance, not a fix — every SCREEN_ICONS
    value plus _DEFAULT_SCREEN_ICON already resolves through streamlit's
    own validator today. Copies 5e's TestExportButtonIconIsValidMaterialIcon
    pattern (reaching app.py's real constants by import) so a future typo
    fails here instead of at render.

    Proven to fire: temporarily setting one SCREEN_ICONS value to
    ":material/not_a_real_icon:" and re-running this test raises
    StreamlitAPIException from validate_icon_or_emoji before the revert.
    """

    def test_all_screen_icons_are_valid(self):
        for screen_id, icon in SCREEN_ICONS.items():
            assert validate_icon_or_emoji(icon) == icon, screen_id
        assert validate_icon_or_emoji(_DEFAULT_SCREEN_ICON) == _DEFAULT_SCREEN_ICON


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
    """Phase 5c-2 (R1, as amended by the Driver 2026-09-05), revised in
    5c-2b: TITLE_MARK_PATH (the bear alone, no disc — beside the screen
    title) and LOGO_MARK_PATH (green disc, white bear — the sidebar mark)
    are two distinct, Driver-ruled assets. Neither is a stand-in for the
    other: both files existing and the two paths merely being distinct is
    not enough to catch a swap, since a swap would still pass both of those
    checks. The pairing lock below asserts each constant names its own
    specific file, so a swap fails."""

    def test_title_mark_path_exists(self):
        assert os.path.exists(TITLE_MARK_PATH)

    def test_logo_mark_path_exists(self):
        assert os.path.exists(LOGO_MARK_PATH)

    def test_paths_are_distinct(self):
        assert TITLE_MARK_PATH != LOGO_MARK_PATH

    def test_title_mark_is_the_cropped_glyph_variant(self):
        assert TITLE_MARK_PATH.endswith("ows-bear-glyph.png")

    def test_logo_mark_is_the_green_disc_variant(self):
        assert LOGO_MARK_PATH.endswith("ows-bear-green-disc.png")


class TestBuildScreenSelectorOptions:
    """Phase 5c-4: build_screen_selector_options is the pure (no Streamlit
    calls) function that splices the overlap pseudo-screen into the
    Screen selector's option list and display-name map, WITHOUT ever
    touching screens_df itself — the pseudo id must never reach
    classify_screen, compute_overlap, or the registry. short_screen is
    deliberately NOT first in this fixture, so a passing default_index
    assertion proves the lookup is by value, not by position."""

    SCREENS_DF = pd.DataFrame(
        {
            "screen_id": ["cyclicals", "short_screen", "structural"],
            "display_name": ["Cyclicals", "OWS Short Screen", "Structural"],
            "screen_type": ["curated", "quant_composite", "curated"],
            "has_scoring": [False, True, False],
        }
    )

    def test_does_not_mutate_screens_df(self):
        before = self.SCREENS_DF.copy()
        build_screen_selector_options(self.SCREENS_DF)
        pd.testing.assert_frame_equal(self.SCREENS_DF, before)

    def test_pseudo_id_appended_last_after_real_ids_in_registry_order(self):
        screen_ids, _, _ = build_screen_selector_options(self.SCREENS_DF)
        assert screen_ids == ["cyclicals", "short_screen", "structural", OVERLAP_PSEUDO_SCREEN_ID]

    def test_pseudo_id_not_among_the_real_ids(self):
        screen_ids, _, _ = build_screen_selector_options(self.SCREENS_DF)
        assert OVERLAP_PSEUDO_SCREEN_ID not in screen_ids[:-1]

    def test_pseudo_display_name_is_set(self):
        _, display_names, _ = build_screen_selector_options(self.SCREENS_DF)
        assert display_names[OVERLAP_PSEUDO_SCREEN_ID] == OVERLAP_PSEUDO_DISPLAY_NAME

    def test_default_index_resolves_to_short_screen_by_value(self):
        screen_ids, _, default_index = build_screen_selector_options(self.SCREENS_DF)
        assert screen_ids[default_index] == "short_screen"


class TestOverlapPseudoScreenClassification:
    """Phase 5c-4: the overlap pseudo-screen is never a screens_df row, so
    classify_screen must fall through to its existing 'unknown' default for
    it — same as any other id absent from the registry. Locks the
    scope-correction finding that no sixth classify_screen kind is needed:
    a future registry change that starts routing this id into a real
    loader would flip this red."""

    SCREENS_DF = pd.DataFrame(
        {
            "screen_id": ["short_screen", "structural"],
            "display_name": ["OWS Short Screen", "Structural"],
            "screen_type": ["quant_composite", "curated"],
            "has_scoring": [True, False],
        }
    )

    def test_pseudo_screen_classifies_as_unknown(self):
        assert classify_screen(OVERLAP_PSEUDO_SCREEN_ID, self.SCREENS_DF) == "unknown"


# Phase 5e: the pre-existing shape all four export-button sites used to
# share (a proportional-width column ratio, which clamps button width to
# 1/10 of the main block and truncates the label on narrow viewports).
# Must be gone post-fix.
OLD_EXPORT_COLUMNS_PATTERN = r"st\.columns\(\s*\[\s*1\s*,\s*1\s*,\s*8\s*\]\s*\)"

# The fixed-gap horizontal container the four sites now share, anchored on
# the literal gap=16 so it can't be confused with the Phase 5c-2 title-row
# container, which uses the same st.container(horizontal=True, ...)
# primitive at gap=12 for a different site.
NEW_EXPORT_CONTAINER_PATTERN = r"st\.container\(horizontal=True, gap=16\)"

# The icon wiring must go through the named constant everywhere it's used
# in an export block, never a re-typed inline literal. Lock 2 below only
# checks EXPORT_BUTTON_ICON's own value by import — it would not catch a
# call site that stopped using the constant and re-typed the literal
# inline instead (even a correct literal). This source scan is what does.
EXPORT_ICON_CONSTANT_PATTERN = r"icon=EXPORT_BUTTON_ICON"
INLINE_MATERIAL_ICON_LITERAL_PATTERN = r'icon=":material/'


class TestExportButtonLayoutPrimitive:
    """Phase 5e: locks that all four export-button sites (scored, curated,
    unscored, overlap render paths) use the fixed-gap horizontal container
    instead of the proportional st.columns([1, 1, 8]) ratio that clamped
    button width to the viewport and truncated the label. Also locks that
    the icon wiring goes through EXPORT_BUTTON_ICON at all eight
    st.download_button call sites (four sites x two buttons), never an
    inline ":material/..." literal, so a reverted or typo'd inline icon is
    caught here rather than only by the (icon-validity-only) Lock 2 below.

    Fail-first (Worker Rules — source-anchored test): run against the
    pre-edit source (HEAD 357239e, `git show 357239e:src/app.py`), this
    module fails at IMPORT (EXPORT_BUTTON_ICON does not exist yet), before
    any of the four assertions below ever execute. That proves the fix
    hadn't landed; it proves nothing about whether these assertions can
    discriminate real breakage, since they never ran.

    That discrimination is instead proven by mutation, against the actual
    post-edit source (restored via md5 after each): reverting one site to
    st.columns([1, 1, 8]) turns test_old_columns_ratio_is_gone AND
    test_four_sites_use_the_fixed_gap_container red; replacing one
    icon=EXPORT_BUTTON_ICON with an inline ":material/download:" literal
    turns test_eight_download_buttons_use_the_named_icon_constant AND
    test_no_inline_material_icon_literal_remains red; drifting one site's
    gap from 16 to 12 turns test_four_sites_use_the_fixed_gap_container red.
    Restoring the source returns all to green. This mutation matrix, not
    the collection-time import failure, is the evidence these locks fire
    on the real file; reported in the build report."""

    def test_old_columns_ratio_is_gone(self):
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        assert re.findall(OLD_EXPORT_COLUMNS_PATTERN, content) == []

    def test_four_sites_use_the_fixed_gap_container(self):
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        matches = re.findall(NEW_EXPORT_CONTAINER_PATTERN, content)
        assert len(matches) == 4, (
            f"expected 4 export-button sites using the gap=16 container, found {len(matches)}"
        )

    def test_eight_download_buttons_use_the_named_icon_constant(self):
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        matches = re.findall(EXPORT_ICON_CONSTANT_PATTERN, content)
        assert len(matches) == 8, (
            f"expected 8 st.download_button calls using EXPORT_BUTTON_ICON, found {len(matches)}"
        )

    def test_no_inline_material_icon_literal_remains(self):
        app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
        with open(app_path) as f:
            content = f.read()
        assert re.findall(INLINE_MATERIAL_ICON_LITERAL_PATTERN, content) == []

    def test_regex_matches_minimal_positive(self):
        """Discriminating half: confirms the patterns actually recognize
        the shapes they're meant to, on synthetic text rather than only
        the real file."""
        old_shape = "col1, col2, col3 = st.columns([1, 1, 8])\n"
        new_shape = (
            'with st.container(horizontal=True, gap=16):\n'
            '    st.download_button(\n'
            '        label="Excel",\n'
            '        icon=EXPORT_BUTTON_ICON,\n'
            '    )\n'
            '    st.download_button(\n'
            '        label="CSV",\n'
            '        icon=EXPORT_BUTTON_ICON,\n'
            '    )\n'
        )
        inline_literal = 'st.download_button(icon=":material/download:")\n'

        assert len(re.findall(OLD_EXPORT_COLUMNS_PATTERN, old_shape)) == 1
        assert re.findall(OLD_EXPORT_COLUMNS_PATTERN, new_shape) == []

        assert len(re.findall(NEW_EXPORT_CONTAINER_PATTERN, new_shape)) == 1
        assert re.findall(NEW_EXPORT_CONTAINER_PATTERN, old_shape) == []

        assert len(re.findall(EXPORT_ICON_CONSTANT_PATTERN, new_shape)) == 2
        assert re.findall(EXPORT_ICON_CONSTANT_PATTERN, inline_literal) == []

        assert len(re.findall(INLINE_MATERIAL_ICON_LITERAL_PATTERN, inline_literal)) == 1
        assert re.findall(INLINE_MATERIAL_ICON_LITERAL_PATTERN, new_shape) == []


class TestExportButtonIconIsValidMaterialIcon:
    """Phase 5e, Lock 2: the icon EXPORT_BUTTON_ICON actually resolves to
    is a valid Material icon shortcode, checked through streamlit's own
    validator rather than by re-typing the literal or checking mere
    package-level validity in the abstract. Reaches src/app.py's real
    constant (via the import above) so a reverted change or a typo in
    EXPORT_BUTTON_ICON itself fails here, unlike a test that only asserts
    facts about the streamlit package.

    Proven to fire: temporarily setting EXPORT_BUTTON_ICON to
    ":material/downlaod:" (the same typo streamlit's own validator raises
    on -- see CLAUDE.md's new Known Implementation Decision) turned this
    red; restoring ":material/download:" turned it green. Both runs
    reported in the build report."""

    def test_export_button_icon_is_a_valid_material_icon(self):
        assert validate_icon_or_emoji(EXPORT_BUTTON_ICON) == EXPORT_BUTTON_ICON


# ---------------------------------------------------------------------------
# render_unscored_sidebar / render_unscored_drill_down guards (Phase 6a)
# ---------------------------------------------------------------------------
#
# app.py is normally verified manually (see module docstring above) rather
# than via pytest, since most of it is Streamlit UI polish only a browser
# can settle. These two guards are different: the property under test is
# "does this Python function raise," a deterministic control-flow fact, not
# a rendering/frontend-only concern — and streamlit 1.63.0's bare mode
# (confirmed empirically: st.sidebar.slider/metric/selectbox/dataframe all
# execute outside a ScriptRunContext, emitting only a bare-mode warning,
# never raising, and st.selectbox returns its first option with no
# interaction) reproduces that fact deterministically without a browser.

class TestRenderUnscoredSidebarMarketCapGuard:
    def test_market_cap_absent_does_not_raise_and_keeps_every_row(self):
        """Negative Expert Transcripts shape: no market_cap column."""
        df = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})
        filtered = render_unscored_sidebar(df)
        assert list(filtered["ticker"]) == ["AAA", "BBB", "CCC"]

    def test_market_cap_constant_does_not_raise(self):
        """The pre-existing Phase 5d min==max st.slider crash, guarded at
        the same call site as the Phase 6a fix."""
        df = pd.DataFrame({"ticker": ["AAA", "BBB"], "market_cap": [500.0, 500.0]})
        filtered = render_unscored_sidebar(df)
        assert list(filtered["ticker"]) == ["AAA", "BBB"]

    def test_market_cap_present_slider_still_filters(self, monkeypatch):
        """RSI shape: market_cap present. Proves the guard narrowed
        nothing — a user-selected (simulated via monkeypatch, since bare
        mode has no real widget interaction) narrower range still excludes
        rows outside it."""
        df = pd.DataFrame({
            "ticker": ["AAA", "BBB", "CCC"],
            "market_cap": [100.0, 500.0, 900.0],
        })
        monkeypatch.setattr(st.sidebar, "slider", lambda *a, **k: (400.0, 600.0))
        filtered = render_unscored_sidebar(df)
        assert list(filtered["ticker"]) == ["BBB"]


class TestRenderUnscoredDrillDownColumnGuard:
    def test_market_cap_and_name_absent_does_not_raise(self, monkeypatch):
        """Negative Expert Transcripts shape: neither market_cap nor name.

        Phase 6b: current_screen_id == TRANSCRIPTS_SCREEN_ID now also fires
        the Transcripts panel, which calls load_raw_detail_data — stubbed
        to None here (T25: no unit test may reach the real, gitignored
        data/screener.db) so this stays the market_cap/name guard test it
        always was, not a new DB dependency.
        """
        import src.app as app_module
        monkeypatch.setattr(app_module, "load_raw_detail_data", lambda screen_id: None)
        df = pd.DataFrame({"ticker": ["AAA"], "mention_count": [3]})
        render_unscored_drill_down(df, ticker_key="test_ticker_key_1",
                                    current_screen_id="negative_expert_transcripts",
                                    membership_df=None, screens_df=pd.DataFrame())

    def test_market_cap_and_name_present_unchanged(self):
        """RSI shape: both present — the guard must not have removed
        either field for a screen that actually has them."""
        df = pd.DataFrame({
            "ticker": ["AAA"], "name": ["Some Co"], "market_cap": [1234.0],
            "adv": [5.0], "short_interest_pct": [0.1], "si_change_3m": [0.0],
            "si_change_6m": [0.0], "week_52_high_chg": [0.0], "ev_sales": [1.0],
            "debt_ebitda": [1.0],
        })
        render_unscored_drill_down(df, ticker_key="test_ticker_key_2",
                                    current_screen_id="rising_short_interest",
                                    membership_df=None, screens_df=pd.DataFrame())

    def test_string_date_column_does_not_raise(self):
        """Property 3 (Correction 3's DB-free shape): latest_transcript_date
        is a TEXT 'YYYY-MM-DD' column. Uses a screen_id NOT in
        UNSCORED_DISPLAY_COLUMNS_BY_SCREEN so the fallback resolver returns
        df's own columns (ticker, latest_transcript_date) and the
        TRANSCRIPTS_SCREEN_ID gate never fires — no DB access at all. Red
        against the pre-edit f"{val:.4f}" fallback: formatting the string
        '2026-08-28' with a numeric spec raises ValueError. Green once the
        fallback degrades to str() instead.
        """
        df = pd.DataFrame({"ticker": ["AAA"], "latest_transcript_date": ["2026-08-28"]})
        render_unscored_drill_down(df, ticker_key="test_ticker_key_3",
                                    current_screen_id="some_future_unmapped_screen",
                                    membership_df=None, screens_df=pd.DataFrame())

    def test_no_format_entry_for_latest_transcript_date(self):
        """Property 4: a numeric format spec against this TEXT column
        raises only at render time (Styler.format / str.format), where no
        unit test would see it — so this is a direct, cheap guard against
        a future 'helpful' addition."""
        assert "latest_transcript_date" not in UNSCORED_METRIC_FORMATS

    def test_no_metrics_table_when_no_metric_columns(self, monkeypatch):
        """Property 5: an unmapped screen whose frame is ticker-only must
        render no 'Metrics' subheader/table at all, not an empty (0, 0)
        frame. Red against the pre-6b unconditional st.subheader("Metrics")
        + always-rendered table."""
        calls = []
        monkeypatch.setattr(st, "subheader", lambda *a, **k: calls.append(a))
        df = pd.DataFrame({"ticker": ["AAA"]})
        render_unscored_drill_down(df, ticker_key="test_ticker_key_4",
                                    current_screen_id="some_future_unmapped_screen",
                                    membership_df=None, screens_df=pd.DataFrame())
        assert not any(c and c[0] == "Metrics" for c in calls)

    def test_transcript_panel_degrades_when_loader_returns_none(self, monkeypatch):
        """Property 8: the Transcripts panel must degrade (a caption), not
        raise, when load_raw_detail_data can't find the table."""
        import src.app as app_module
        monkeypatch.setattr(app_module, "load_raw_detail_data", lambda screen_id: None)
        df = pd.DataFrame({
            "ticker": ["ZZZ"], "mention_count": [1],
            "latest_transcript_date": ["2026-01-01"], "days_since_latest": [0],
        })
        render_unscored_drill_down(df, ticker_key="test_ticker_key_5",
                                    current_screen_id=TRANSCRIPTS_SCREEN_ID,
                                    membership_df=None, screens_df=pd.DataFrame())

    def test_transcript_panel_gated_to_transcripts_screen_only(self, monkeypatch):
        """DB-free lock on the current_screen_id == TRANSCRIPTS_SCREEN_ID
        gate itself (build-report Correction 2). Stubs load_raw_detail_data
        to return an RSI-SHAPED frame — no transcript_date/theme/
        key_takeaways columns — so an ungated panel would KeyError trying
        to read t['transcript_date']; asserting only "does not raise" would
        be too weak here (an empty-state caption would also not raise), so
        this records st.subheader calls directly and asserts "Transcripts"
        never appears for a non-transcripts screen, the same technique
        test_no_metrics_table_when_no_metric_columns already uses.

        Red under a `current_screen_id == TRANSCRIPTS_SCREEN_ID` ->
        `True` mutation (the gate always fires): on this RSI-shaped stub,
        that mutation either raises a KeyError reading t['transcript_date']
        (if any row survives the loader) or, as built here (empty detail
        frame), renders the panel's "Transcripts" subheader anyway — this
        assertion catches the latter unconditionally, the former via
        pytest's own exception propagation.
        """
        import src.app as app_module
        rsi_shaped_detail = pd.DataFrame({
            "ticker": ["AAA"], "market_cap": [500.0], "adv": [5.0],
            "short_interest_pct": [0.1],
        })
        monkeypatch.setattr(
            app_module, "load_raw_detail_data", lambda screen_id: rsi_shaped_detail
        )
        calls = []
        monkeypatch.setattr(st, "subheader", lambda *a, **k: calls.append(a))
        df = pd.DataFrame({
            "ticker": ["AAA"], "name": ["Some Co"], "market_cap": [1234.0],
            "adv": [5.0], "short_interest_pct": [0.1], "si_change_3m": [0.0],
            "si_change_6m": [0.0], "week_52_high_chg": [0.0], "ev_sales": [1.0],
            "debt_ebitda": [1.0],
        })
        render_unscored_drill_down(df, ticker_key="test_ticker_key_6",
                                    current_screen_id="rising_short_interest",
                                    membership_df=None, screens_df=pd.DataFrame())
        assert not any(c and c[0] == "Transcripts" for c in calls)


# ---------------------------------------------------------------------------
# Phase 6b: resolve_unscored_display_columns, unscored_export_basename,
# transcripts_for_ticker
# ---------------------------------------------------------------------------


class TestResolveUnscoredDisplayColumns:
    """Property 1: each mapped screen gets its OWN list, and an unmapped
    screen gets its own columns — never another screen's."""

    def test_rsi_gets_its_own_list(self):
        cols = UNSCORED_DISPLAY_COLUMNS_BY_SCREEN["rising_short_interest"]
        df = pd.DataFrame({c: [1] for c in cols})
        assert resolve_unscored_display_columns("rising_short_interest", df) == cols

    def test_transcripts_gets_its_own_list(self):
        cols = UNSCORED_DISPLAY_COLUMNS_BY_SCREEN[TRANSCRIPTS_SCREEN_ID]
        df = pd.DataFrame({c: [1] for c in cols})
        assert resolve_unscored_display_columns(TRANSCRIPTS_SCREEN_ID, df) == cols

    def test_unmapped_screen_uses_its_own_columns_not_rsis(self):
        """Red against the pre-6b single global list: intersecting RSI's
        ten columns with this df would collapse to ["ticker"] alone (since
        "widget_count" isn't one of RSI's columns) — a silently wrong
        result, which is exactly the "moved one screen along" defect this
        function exists to prevent.

        ticker is built NOT first in the source dict (build-report
        Correction 3), so this also discriminates the fallback's
        ticker-first reordering — a fixture with ticker already first
        cannot tell "returns df's own columns" apart from "returns df's own
        columns, unreordered," and the reordering half of the contract
        would go untested. Red if the ticker-first reordering is deleted
        (result would be ["widget_count", "ticker"] instead)."""
        df = pd.DataFrame({"widget_count": [3], "ticker": ["AAA"]})
        result = resolve_unscored_display_columns("some_future_screen", df)
        assert result == ["ticker", "widget_count"]
        assert "market_cap" not in result


class TestUnscoredExportBasename:
    """Property 6 (part 1): the pure filename-stem function."""

    def test_rsi_filename_unchanged(self):
        assert unscored_export_basename("rising_short_interest") == "ows_rising_short_interest"

    def test_transcripts_filename(self):
        assert (
            unscored_export_basename(TRANSCRIPTS_SCREEN_ID)
            == "ows_negative_expert_transcripts"
        )


class TestRenderUnscoredTableExportFilenameSource:
    """Property 6 (part 2, source-anchored per CLAUDE.md's rule): the
    literal "ows_rising_short_interest" must no longer appear anywhere in
    render_unscored_table's source, and the filename must be built through
    unscored_export_basename. Written against the POST-edit source — see
    the build report for the required fail-first (against the pre-edit
    source) and positive-control (against this actual post-edit source)
    runs."""

    def test_no_hardcoded_rsi_filename_in_source(self):
        import inspect

        import src.app as app_module

        source = inspect.getsource(app_module.render_unscored_table)
        assert "ows_rising_short_interest" not in source
        assert "unscored_export_basename(" in source


class TestStyleUnscoredTable:
    """Phase 8a: style_unscored_table, extracted from render_unscored_table
    so the "nanx" defect (a bulk .format() call has no na_rep at all) is
    testable via .to_html() directly, mirroring style_overlap_table's own
    testability reason."""

    @staticmethod
    def _sample_df():
        return pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "pe_vs_normal_5y": [1.31, float("nan")],
        })

    def test_non_null_ratio_keeps_x_suffix_and_two_decimals(self):
        html = style_unscored_table(self._sample_df()).to_html()
        assert "1.31x" in html

    def test_null_ratio_renders_em_dash_not_literal_nanx(self):
        """The regression this class exists to prevent: a plain
        `.style.format({"pe_vs_normal_5y": "{:.2f}x"})` with no na_rep
        renders the literal string "nanx" for a null cell (measured). The
        placeholder is an em dash, not "N/A" — the same blank the overlap
        view uses for this column (OVERLAP_EXTRA_NA_REPS), so a reader
        moving between the screen's own table and the overlap view sees
        one vocabulary for "no data", not two."""
        html = style_unscored_table(self._sample_df()).to_html()
        assert "nanx" not in html
        assert "—" in html

    def test_null_ratio_does_not_disturb_a_column_with_no_na_rep_entry(self):
        """A real column absent from UNSCORED_EXTRA_NA_REPS (market_cap,
        RSI's own) must format exactly as before — proves the extraction
        didn't change any other column's behavior."""
        df = self._sample_df()
        df["market_cap"] = [1234.0, 5678.0]
        html = style_unscored_table(df).to_html()
        assert "$1,234" in html


class TestOverlapMetricFillValues:
    """Phase 8a, R1: a joined RATIO column must fill NaN, never 0, for a
    ticker absent from the joining screen — 0 would render as the most
    extreme possible undervaluation on the Overvalued screen's own
    column."""

    def test_pe_vs_normal_5y_fills_nan_not_zero(self):
        assert OVERLAP_METRIC_JOINS["overvalued_screen"] == "pe_vs_normal_5y"
        fill = OVERLAP_METRIC_FILL_VALUES["pe_vs_normal_5y"]
        assert isinstance(fill, float) and fill != fill  # NaN

    def test_mention_count_has_no_entry_so_default_zero_still_applies(self):
        assert "mention_count" not in OVERLAP_METRIC_FILL_VALUES

    def test_overlap_label_matches_unscored_display_name_exactly(self):
        """OVERLAP_COLUMN_LABELS and UNSCORED_METRIC_DISPLAY_NAMES must
        never drift for a column shown in both places — the same rule
        mention_count's own comment states."""
        assert (
            OVERLAP_COLUMN_LABELS["pe_vs_normal_5y"]
            == UNSCORED_METRIC_DISPLAY_NAMES["pe_vs_normal_5y"]
        )

    def test_pe_vs_normal_5y_has_an_overlap_na_rep(self):
        assert OVERLAP_EXTRA_NA_REPS["pe_vs_normal_5y"] == "—"


class TestTranscriptsForTicker:
    """Property 7: newest-first ordering with a doc_id tie-break."""

    def test_newest_first_with_doc_id_tiebreak(self):
        """Built to distinguish the tie-break from input order (the live
        WSO shape): both rows share transcript_date 2026-09-03, and input
        order lists doc_id "...246345" BEFORE "...246257" — the reverse of
        what doc_id-ascending must produce. A fixture whose input order
        already matched the expected output couldn't catch a broken/absent
        tie-break."""
        df = pd.DataFrame({
            "ticker": ["WSO", "WSO", "WSO"],
            "transcript_date": ["2026-09-03", "2026-09-03", "2026-08-01"],
            "doc_id": ["EC-1000000-246345", "EC-1000000-246257", "EC-1000000-100000"],
            "theme": ["a", "b", "c"],
        })
        result = transcripts_for_ticker(df, "WSO")
        assert list(result["doc_id"]) == [
            "EC-1000000-246257", "EC-1000000-246345", "EC-1000000-100000",
        ]
        assert list(result.index) == [0, 1, 2]

    def test_ticker_with_no_transcripts_returns_empty_without_raising(self):
        """Property 8 (pure-function half): unreachable on today's data
        (the aggregate's 173 tickers are exactly the detail table's 173),
        but the branch must degrade rather than raise."""
        df = pd.DataFrame({
            "ticker": ["AAA"], "transcript_date": ["2026-01-01"],
            "doc_id": ["X"], "theme": ["t"],
        })
        result = transcripts_for_ticker(df, "ZZZ")
        assert result.empty


# ---------------------------------------------------------------------------
# Phase 8b: transcript takeaways in "also appears on"
# ---------------------------------------------------------------------------


class TestResolveTranscriptTakeawaysForTicker:
    """T40's gate, wired to the real transcripts_for_ticker."""

    def test_rsi_shaped_frame_returns_none_not_raise(self):
        """T40's recorded failure verbatim: an RSI-shaped frame handed to
        the pre-8b table-existence gate would reach transcripts_for_ticker
        and KeyError on the missing sort columns. Red under a mutation that
        drops the is_transcript_shaped check (calls transcripts_for_ticker
        unconditionally): this fixture has no transcript_date/doc_id, so
        that call raises KeyError instead of returning None."""
        rsi_shaped = pd.DataFrame({
            "ticker": ["AAA"], "market_cap": [500.0], "adv": [5.0],
            "short_interest_pct": [0.1],
        })
        assert resolve_transcript_takeaways_for_ticker(rsi_shaped, "AAA") is None

    def test_none_detail_df_returns_none(self):
        """Property 8/T40: data/screener.db absent (or the raw table
        missing) means load_raw_detail_data returns None upstream — this
        must degrade the same way, not raise."""
        assert resolve_transcript_takeaways_for_ticker(None, "AAA") is None

    def test_ticker_with_zero_transcripts_returns_none(self):
        df = pd.DataFrame({
            "ticker": ["AAA"], "transcript_date": ["2026-01-01"],
            "doc_id": ["X"], "theme": ["t"], "key_takeaways": ["k"],
        })
        assert resolve_transcript_takeaways_for_ticker(df, "ZZZ") is None

    def test_multi_row_ticker_ordered_newest_first_with_doc_id_tiebreak(self):
        """Reuses transcripts_for_ticker unchanged — same tie-break fixture
        shape as TestTranscriptsForTicker.
        test_newest_first_with_doc_id_tiebreak, exercised THROUGH the new
        gated resolver this time."""
        df = pd.DataFrame({
            "ticker": ["WSO", "WSO", "WSO"],
            "transcript_date": ["2026-09-03", "2026-09-03", "2026-08-01"],
            "doc_id": ["EC-1000000-246345", "EC-1000000-246257", "EC-1000000-100000"],
            "theme": ["a", "b", "c"],
            "key_takeaways": ["ka", "kb", "kc"],
        })
        result = resolve_transcript_takeaways_for_ticker(df, "WSO")
        assert list(result["doc_id"]) == [
            "EC-1000000-246257", "EC-1000000-246345", "EC-1000000-100000",
        ]


class TestAlsoAppearsOnTranscriptTakeaways:
    """Phase 8b end-to-end through render_cross_screen_context. Ordering
    (item 4's fix): st.markdown and st.write calls are recorded into ONE
    shared ordered sequence, since both the metric lines and the takeaway
    bodies go through st.write — a plain count/scan of st.write alone
    cannot tell "takeaways below metrics" from "takeaways above metrics"."""

    def _render(self, monkeypatch, transcripts_detail_df, ticker="AAA"):
        import src.app as app_module

        events = []
        monkeypatch.setattr(st, "markdown", lambda *a, **k: events.append(("markdown", a[0] if a else None)))
        monkeypatch.setattr(st, "write", lambda *a, **k: events.append(("write", a[0] if a else None)))
        monkeypatch.setattr(st, "caption", lambda *a, **k: events.append(("caption", a[0] if a else None)))
        monkeypatch.setattr(st, "subheader", lambda *a, **k: events.append(("subheader", a[0] if a else None)))

        rsi_df = pd.DataFrame({
            "ticker": [ticker], "name": ["A Co"], "market_cap": [123.0],
            "adv": [5.0], "short_interest_pct": [0.1], "si_change_3m": [0.0],
            "si_change_6m": [0.0], "week_52_high_chg": [0.0], "ev_sales": [1.0],
            "debt_ebitda": [1.0],
        })
        transcripts_aggregate_df = pd.DataFrame({
            "ticker": [ticker], "mention_count": [3],
            "latest_transcript_date": ["2026-08-06"], "days_since_latest": [5],
        })
        monkeypatch.setattr(
            app_module, "load_screens_for_ticker",
            lambda *a, **k: {
                "rising_short_interest": rsi_df,
                TRANSCRIPTS_SCREEN_ID: transcripts_aggregate_df,
            },
        )
        monkeypatch.setattr(
            app_module, "load_raw_detail_data", lambda screen_id: transcripts_detail_df
        )
        membership_df = pd.DataFrame({
            "screen_id": ["short_screen", "rising_short_interest", TRANSCRIPTS_SCREEN_ID],
            "ticker": [ticker, ticker, ticker],
        })
        screens_df = pd.DataFrame({
            "screen_id": ["short_screen", "rising_short_interest", TRANSCRIPTS_SCREEN_ID],
            "display_name": [
                "OWS Short Screen", "Rising Short Interest", "Negative Expert Transcripts",
            ],
            "screen_type": ["quant_composite", "quant_composite", "quant_composite"],
            "has_scoring": [True, False, False],
        })
        render_cross_screen_context(ticker, "short_screen", membership_df, screens_df, None)
        return events

    def test_n_transcripts_render_n_takeaway_bodies(self, monkeypatch):
        """A ticker with N transcripts renders N takeaway entries — must
        fail against an iloc[0]-style implementation that shows only the
        first (verified by hand during the build: temporarily slicing the
        takeaways to one row made this assertion go red)."""
        detail_df = pd.DataFrame({
            "ticker": ["AAA", "AAA", "AAA"],
            "transcript_date": ["2026-08-06", "2026-07-01", "2026-06-01"],
            "doc_id": ["C", "B", "A"],
            "theme": ["x", "y", "z"],
            "key_takeaways": ["takeaway-one", "takeaway-two", "takeaway-three"],
        })
        events = self._render(monkeypatch, detail_df)
        takeaway_writes = [
            v for kind, v in events
            if kind == "write" and isinstance(v, str) and v.startswith("takeaway-")
        ]
        assert takeaway_writes == ["takeaway-one", "takeaway-two", "takeaway-three"]

    def test_takeaways_render_after_metric_lines_not_before(self, monkeypatch):
        """Item 4's fix: relative position in ONE shared sequence, not a
        count. Compares the takeaway body against the TRANSCRIPTS
        contribution's OWN 6c metric line (Transcript Mentions) — not
        RSI's, since contributions sort by display_name and "Negative
        Expert Transcripts" < "Rising Short Interest" alphabetically, so a
        cross-contribution comparison would pass or fail on sort order
        alone rather than on Rule 3's within-contribution ordering. Red
        against an implementation that emits the takeaways block before
        this contribution's own metrics loop."""
        detail_df = pd.DataFrame({
            "ticker": ["AAA"], "transcript_date": ["2026-08-06"],
            "doc_id": ["A"], "theme": ["x"], "key_takeaways": ["the-takeaway"],
        })
        events = self._render(monkeypatch, detail_df)
        kinds_and_values = [v for kind, v in events if kind == "write" and isinstance(v, str)]
        metric_index = next(
            i for i, v in enumerate(kinds_and_values) if "Transcript Mentions" in v
        )
        takeaway_index = next(
            i for i, v in enumerate(kinds_and_values) if v == "the-takeaway"
        )
        assert takeaway_index > metric_index

    def test_no_heading_or_caption_for_takeaways_block(self, monkeypatch):
        """Ruling: no "Transcripts" heading and no count caption inside
        also-appears-on — only the {date} · {theme} markdown header per
        entry, exactly as a curated screen's rationale has none."""
        detail_df = pd.DataFrame({
            "ticker": ["AAA"], "transcript_date": ["2026-08-06"],
            "doc_id": ["A"], "theme": ["x"], "key_takeaways": ["the-takeaway"],
        })
        events = self._render(monkeypatch, detail_df)
        assert not any(v == "Transcripts" for kind, v in events if kind == "subheader")
        assert not any(
            isinstance(v, str) and "newest first" in v for kind, v in events if kind == "caption"
        )

    @staticmethod
    def _takeaway_entry_headers(events):
        """The per-transcript '**date** · theme' markdown lines only —
        distinct from the per-contribution 'icon **Display Name**' header
        markdown (which has no '·' and always renders once a screen
        contributes at all)."""
        return [
            v for kind, v in events
            if kind == "markdown" and isinstance(v, str) and "·" in v
        ]

    def test_rsi_shaped_detail_df_renders_no_takeaways_block(self, monkeypatch):
        """T40 end-to-end: the loader returning an RSI-shaped frame (as if
        misrouted) must render zero takeaway bodies, no raise. The
        contribution's own header markdown still fires (the ticker is
        still a member of the transcripts screen) — only the takeaway
        entries themselves must be absent."""
        events = self._render(monkeypatch, transcripts_detail_df=pd.DataFrame({
            "ticker": ["AAA"], "market_cap": [500.0], "adv": [5.0],
        }))
        assert self._takeaway_entry_headers(events) == []

    def test_none_detail_df_renders_no_takeaways_block(self, monkeypatch):
        """data/screener.db absent (T40/T25): must not raise, must not
        render a takeaways block."""
        events = self._render(monkeypatch, transcripts_detail_df=None)
        assert self._takeaway_entry_headers(events) == []

    def test_zero_transcripts_for_ticker_renders_no_takeaways_block(self, monkeypatch):
        """Synthetic-only (PM-confirmed): on live data every screen this
        ticker also appears on has at least one takeaway, since the
        aggregate/detail/membership ticker sets are identical — this
        branch cannot fire against real data today, so it's locked here
        with a detail table that carries a different ticker's row only."""
        detail_df = pd.DataFrame({
            "ticker": ["ZZZ"], "transcript_date": ["2026-08-06"],
            "doc_id": ["A"], "theme": ["x"], "key_takeaways": ["the-takeaway"],
        })
        events = self._render(monkeypatch, detail_df, ticker="AAA")
        assert self._takeaway_entry_headers(events) == []


class TestRenderTranscriptTakeawaysHelper:
    """The single render site (T42) — deliberately just the loop, no
    heading/caption of its own."""

    def test_renders_date_theme_header_and_body_per_row(self, monkeypatch):
        events = []
        monkeypatch.setattr(st, "markdown", lambda *a, **k: events.append(a[0]))
        monkeypatch.setattr(st, "write", lambda *a, **k: events.append(a[0]))
        df = pd.DataFrame({
            "transcript_date": ["2026-08-06", "2026-07-01"],
            "theme": ["Theme A", "Theme B"],
            "key_takeaways": ["Body A", "Body B"],
        })
        render_transcript_takeaways(df)
        assert events == [
            "**2026-08-06** · Theme A", "Body A",
            "**2026-07-01** · Theme B", "Body B",
        ]

    def test_no_subheader_or_caption_emitted(self, monkeypatch):
        calls = []
        monkeypatch.setattr(st, "subheader", lambda *a, **k: calls.append(a))
        monkeypatch.setattr(st, "caption", lambda *a, **k: calls.append(a))
        df = pd.DataFrame({
            "transcript_date": ["2026-08-06"], "theme": ["Theme A"],
            "key_takeaways": ["Body A"],
        })
        render_transcript_takeaways(df)
        assert calls == []


# ---------------------------------------------------------------------------
# Phase 6c
# ---------------------------------------------------------------------------


class TestFormatUnscoredMetricValue:
    """format_unscored_metric_value — the guard extracted from Phase 6b's
    render_unscored_drill_down so render_cross_screen_context's unscored
    branch (which never had it) can share the same fix (property iii)."""

    def test_nan_renders_as_na(self):
        assert format_unscored_metric_value("mention_count", float("nan")) == "N/A"

    def test_formatted_column_uses_its_spec(self):
        assert format_unscored_metric_value("mention_count", 5) == "5"

    def test_unformatted_numeric_falls_back_to_four_decimals(self):
        assert format_unscored_metric_value("some_future_metric", 1.5) == "1.5000"

    def test_non_numeric_value_degrades_to_str_not_raise(self):
        """Red against the pre-6b/6c fallback: f"{'2026-08-06':.4f}" raises
        ValueError. latest_transcript_date has no UNSCORED_METRIC_FORMATS
        entry (see that dict's own comment), so a TEXT value reaches this
        branch."""
        assert format_unscored_metric_value("latest_transcript_date", "2026-08-06") == "2026-08-06"


class TestRenderCrossScreenContextUnscoredValueGuard:
    """Property (iii), the live crash site: render_cross_screen_context's
    "unscored" branch had NO guard before Phase 6c (unlike render_unscored_
    drill_down, which 6b already fixed) — reachable only once a second
    unscored screen's non-numeric column (latest_transcript_date) can flow
    into it, which Item 1's resolver threading is what makes possible.

    T40: render_cross_screen_context calls load_screens_for_ticker, which
    reads DB_PATH via _load_screen_df -> load_unscored_quant_data — stubbed
    here so the result depends on the synthetic frame below, not on
    data/screener.db being present (or its real contents) on whatever
    machine runs this test. "Does not raise" alone is too weak (6b's own
    precedent, TestRenderUnscoredDrillDownColumnGuard.
    test_transcript_panel_gated_to_transcripts_screen_only): an empty-
    contributions path also does not raise, so st.write's actual calls are
    recorded and checked for the formatted date string.
    """

    def test_non_numeric_metric_renders_as_string_not_raise(self, monkeypatch):
        import src.app as app_module

        # Phase 8b: render_cross_screen_context now also calls
        # load_raw_detail_data(TRANSCRIPTS_SCREEN_ID) — stubbed to None so
        # this test's result depends only on the synthetic frames below,
        # never on data/screener.db (T25/T40).
        monkeypatch.setattr(app_module, "load_raw_detail_data", lambda screen_id: None)

        transcripts_df = pd.DataFrame({
            "ticker": ["AAA"],
            "mention_count": [3],
            "latest_transcript_date": ["2026-08-06"],
            "days_since_latest": [5],
        })
        monkeypatch.setattr(
            app_module, "load_screens_for_ticker",
            lambda *a, **k: {TRANSCRIPTS_SCREEN_ID: transcripts_df},
        )
        membership_df = pd.DataFrame({
            "screen_id": ["short_screen", TRANSCRIPTS_SCREEN_ID],
            "ticker": ["AAA", "AAA"],
        })
        screens_df = pd.DataFrame({
            "screen_id": ["short_screen", TRANSCRIPTS_SCREEN_ID],
            "display_name": ["OWS Short Screen", "Negative Expert Transcripts"],
            "screen_type": ["quant_composite", "quant_composite"],
            "has_scoring": [True, False],
        })

        writes = []
        monkeypatch.setattr(st, "write", lambda *a, **k: writes.append(a))

        render_cross_screen_context("AAA", "short_screen", membership_df, screens_df, None)

        rendered = " | ".join(str(a[0]) for a in writes if a)
        assert "2026-08-06" in rendered
        assert "Latest Transcript" in rendered

    def test_rsi_contribution_unchanged_with_resolver_threaded(self, monkeypatch):
        """Property (ii): RSI's contribution, rendered THROUGH
        render_cross_screen_context (which now threads
        resolve_unscored_display_columns), is byte-identical to its
        pre-6c 7-metric shape — the resolver must not have changed RSI's
        column set or order."""
        import src.app as app_module

        # Phase 8b: see test_non_numeric_metric_renders_as_string_not_raise.
        monkeypatch.setattr(app_module, "load_raw_detail_data", lambda screen_id: None)

        rsi_df = pd.DataFrame({
            "ticker": ["AAA"], "name": ["A Co"], "market_cap": [123.0],
            "adv": [5.0], "short_interest_pct": [0.1], "si_change_3m": [0.0],
            "si_change_6m": [0.0], "week_52_high_chg": [0.0], "ev_sales": [1.0],
            "debt_ebitda": [1.0],
        })
        monkeypatch.setattr(
            app_module, "load_screens_for_ticker",
            lambda *a, **k: {"rising_short_interest": rsi_df},
        )
        membership_df = pd.DataFrame({
            "screen_id": ["short_screen", "rising_short_interest"],
            "ticker": ["AAA", "AAA"],
        })
        screens_df = pd.DataFrame({
            "screen_id": ["short_screen", "rising_short_interest"],
            "display_name": ["OWS Short Screen", "Rising Short Interest"],
            "screen_type": ["quant_composite", "quant_composite"],
            "has_scoring": [True, False],
        })

        writes = []
        monkeypatch.setattr(st, "write", lambda *a, **k: writes.append(a))

        render_cross_screen_context("AAA", "short_screen", membership_df, screens_df, None)

        rendered = " | ".join(str(a[0]) for a in writes if a)
        for label in (
            "Avg Daily Value Traded ($M)", "Short Interest %", "SI Change (3M)",
            "SI Change (6M)", "Change from 52W High", "EV / Sales", "Net Debt / EBITDA",
        ):
            assert label in rendered, f"missing: {label}"


class TestBuildScreenContributionResolver:
    """cross_screen_context.build_screen_contribution's optional
    unscored_display_columns_resolver (Item 1, properties i/ii/v). Purely
    synthetic — no DB, no Streamlit."""

    def test_default_none_matches_legacy_rsi_columns(self):
        """No resolver given: falls back to _UNSCORED_METRIC_COLUMNS,
        today's exact behavior — an existing caller/test that never passes
        this argument keeps its current meaning unchanged (1c)."""
        df = pd.DataFrame({
            "ticker": ["AAA"], "adv": [5.0], "short_interest_pct": [0.1],
            "si_change_3m": [0.0], "si_change_6m": [0.0], "week_52_high_chg": [0.0],
            "ev_sales": [1.0], "debt_ebitda": [1.0],
        })
        screens_df = pd.DataFrame({
            "screen_id": ["rising_short_interest"],
            "display_name": ["Rising Short Interest"],
            "screen_type": ["quant_composite"],
            "has_scoring": [False],
        })
        result = build_screen_contribution(
            "rising_short_interest", "AAA", screens_df, {"rising_short_interest": df}
        )
        assert set(result["metrics"]) == {
            "adv", "short_interest_pct", "si_change_3m", "si_change_6m",
            "week_52_high_chg", "ev_sales", "debt_ebitda",
        }

    def test_transcripts_contribution_keys_derived_from_real_display_list(self):
        """Property (i): resolved via the real resolve_unscored_display_
        columns + UNSCORED_DISPLAY_COLUMNS_BY_SCREEN[TRANSCRIPTS_SCREEN_ID]
        — never a literal — so a future edit to that list is mirrored here
        automatically instead of silently going stale."""
        df = pd.DataFrame({
            "ticker": ["AAA"], "mention_count": [3],
            "latest_transcript_date": ["2026-08-06"], "days_since_latest": [5],
        })
        screens_df = pd.DataFrame({
            "screen_id": [TRANSCRIPTS_SCREEN_ID],
            "display_name": ["Negative Expert Transcripts"],
            "screen_type": ["quant_composite"],
            "has_scoring": [False],
        })
        result = build_screen_contribution(
            TRANSCRIPTS_SCREEN_ID, "AAA", screens_df, {TRANSCRIPTS_SCREEN_ID: df},
            unscored_display_columns_resolver=resolve_unscored_display_columns,
        )
        expected = set(UNSCORED_DISPLAY_COLUMNS_BY_SCREEN[TRANSCRIPTS_SCREEN_ID]) - {
            "ticker", "name", "market_cap",
        }
        assert set(result["metrics"]) == expected
        assert expected == {"mention_count", "latest_transcript_date", "days_since_latest"}

    def test_rsi_contribution_unchanged_with_resolver(self):
        """Property (ii), pure-function half: resolve_unscored_display_
        columns threaded in produces the same 7 keys, same values, as the
        no-resolver default for RSI."""
        df = pd.DataFrame({
            "ticker": ["AAA"], "name": ["A Co"], "market_cap": [123.0],
            "adv": [5.0], "short_interest_pct": [0.1], "si_change_3m": [0.0],
            "si_change_6m": [0.0], "week_52_high_chg": [0.0], "ev_sales": [1.0],
            "debt_ebitda": [1.0],
        })
        screens_df = pd.DataFrame({
            "screen_id": ["rising_short_interest"],
            "display_name": ["Rising Short Interest"],
            "screen_type": ["quant_composite"],
            "has_scoring": [False],
        })
        without_resolver = build_screen_contribution(
            "rising_short_interest", "AAA", screens_df, {"rising_short_interest": df}
        )
        with_resolver = build_screen_contribution(
            "rising_short_interest", "AAA", screens_df, {"rising_short_interest": df},
            unscored_display_columns_resolver=resolve_unscored_display_columns,
        )
        assert without_resolver == with_resolver

    def test_unmapped_screen_gets_its_own_columns_never_rsis(self):
        """Property (v): a screen_id absent from UNSCORED_DISPLAY_COLUMNS_
        BY_SCREEN must render its OWN columns (minus identity), never RSI's
        7 — the mistake resolve_unscored_display_columns's fallback exists
        to prevent (T39's shape), now exercised through the resolver."""
        df = pd.DataFrame({"ticker": ["AAA"], "some_future_metric": [42.0]})
        screens_df = pd.DataFrame({
            "screen_id": ["some_future_unmapped_screen"],
            "display_name": ["Some Future Screen"],
            "screen_type": ["quant_composite"],
            "has_scoring": [False],
        })
        result = build_screen_contribution(
            "some_future_unmapped_screen", "AAA", screens_df,
            {"some_future_unmapped_screen": df},
            unscored_display_columns_resolver=resolve_unscored_display_columns,
        )
        assert set(result["metrics"]) == {"some_future_metric"}


class TestApplyOverlapMetricJoins:
    """app.apply_overlap_metric_joins — pure, no DB, no Streamlit (the
    function T40 would otherwise worry about; get_overlap_df is the only
    caller that reads real data, and it's not exercised by any unit test)."""

    def _overlap_df(self):
        return pd.DataFrame({
            "ticker": ["AAA", "BBB", "CCC"],
            "screen_count": [1, 0, 2],
        })

    def test_present_screen_joins_and_fills_zero_for_absent_tickers(self):
        transcripts_df = pd.DataFrame({"ticker": ["AAA", "CCC"], "mention_count": [5, 2]})
        result = apply_overlap_metric_joins(
            self._overlap_df(), {TRANSCRIPTS_SCREEN_ID: transcripts_df}
        )
        assert list(result["mention_count"]) == [5, 0, 2]

    def test_absent_screen_leaves_column_out_entirely(self):
        """Ruling 2c: transcripts screen not loadable (e.g. a fresh clone)
        -> mention_count is simply absent, not a column of all-zero/NaN."""
        result = apply_overlap_metric_joins(self._overlap_df(), {})
        assert "mention_count" not in result.columns

    def test_joined_column_is_integer_not_float(self):
        """Correction 3: the merge's NaN-fill must not leave mention_count
        promoted to float64 — the exported CSV must show whole numbers
        (5, 0), never (5.0, 0.0), matching screen_count's own dtype."""
        transcripts_df = pd.DataFrame({"ticker": ["AAA"], "mention_count": [5]})
        result = apply_overlap_metric_joins(
            self._overlap_df(), {TRANSCRIPTS_SCREEN_ID: transcripts_df}
        )
        assert result["mention_count"].dtype == transcripts_df["mention_count"].dtype
        csv_text = result.to_csv(index=False)
        assert ",0\n" in csv_text or csv_text.rstrip().endswith(",0")
        assert ",0.0" not in csv_text

    def test_pe_vs_normal_5y_fills_nan_through_the_real_call_site_while_mention_count_still_fills_zero(
        self,
    ):
        """Phase 8a review round 2, Correction 1: TestOverlapMetricFillValues
        only asserted what's IN OVERLAP_METRIC_FILL_VALUES — nothing proved
        apply_overlap_metric_joins (the real call site, app.py:3308) ever
        reads it. Deleting `fill_value=OVERLAP_METRIC_FILL_VALUES.get(...)`
        from that call reinstates the 0.00x defect with the full suite
        green, since every OTHER existing test only exercises the constant
        or join_metric_column directly, never this function with BOTH
        columns joined at once. This test goes through
        apply_overlap_metric_joins itself, synthetic screen_data only
        (T25), and checks both columns in one call so a future "fix" that
        flips the default to NaN globally (silently breaking mention_count)
        would also be caught by the third assertion."""
        overvalued_df = pd.DataFrame({"ticker": ["AAA", "CCC"], "pe_vs_normal_5y": [1.5, 2.0]})
        transcripts_df = pd.DataFrame({"ticker": ["AAA", "CCC"], "mention_count": [5, 2]})
        result = apply_overlap_metric_joins(
            self._overlap_df(),
            {"overvalued_screen": overvalued_df, TRANSCRIPTS_SCREEN_ID: transcripts_df},
        )
        # Present tickers carry their real ratio.
        assert result.loc[result["ticker"] == "AAA", "pe_vs_normal_5y"].iloc[0] == 1.5
        assert result.loc[result["ticker"] == "CCC", "pe_vs_normal_5y"].iloc[0] == 2.0
        # Absent ticker: NaN, not 0 — the defect this test exists to catch.
        bbb_ratio = result.loc[result["ticker"] == "BBB", "pe_vs_normal_5y"].iloc[0]
        assert bbb_ratio != bbb_ratio  # NaN
        # mention_count's own default is untouched by pe_vs_normal_5y's entry.
        assert result.loc[result["ticker"] == "BBB", "mention_count"].iloc[0] == 0



# Phase 6c review round 1, Correction 2.2: a registry where the STRUCTURAL
# thematic-screen count (4) and screen_count_ceiling()'s OBSERVED maximum
# (2, from _SYNTHETIC_OVERLAP_DF's max screen_count) DIFFER — the fixture
# _SYNTHETIC_SCREENS_DF (3 screens, thematic count 2) could not catch a
# derivation that silently swapped in the observed max, because both
# readings happened to equal 2 there. This one can.
_SCREENS_DF_CEILING_DIFFERS_FROM_OBSERVED_MAX = pd.DataFrame({
    "screen_id": ["short_screen", "structural", "competition", "cyclicals", "management_comp"],
    "display_name": [
        "OWS Short Screen", "Structural", "Competition", "Cyclicals", "Management Comp",
    ],
    "screen_type": ["quant_composite", "curated", "curated", "curated", "curated"],
    "has_scoring": [True, False, False, False, False],
})


class TestBuildOverlapHelpMapDerivation:
    """build_overlap_help_map (Item 3) — every figure computed from the
    synthetic frames the test itself provides, never the live 1,042/1,358
    etc. (T25). Every assertion below is the EXACT rendered substring (not
    loose numeral membership) — see Phase 6c review round 1, Correction
    2.3: "2" in help_map["screens_on"] passes on "2 of 4" and equally on
    the argument-order defect "4 of 2", so a bare numeral cannot catch the
    one bug most likely in a two-count f-string.

    _SYNTHETIC_OVERLAP_DF (5 tickers):
        AAA in_universe, screen_count 0   -> counts toward zero_count
        BBB in_universe, screen_count 0   -> counts toward zero_count
        CCC in_universe, screen_count 1
        DDD NOT in_universe, screen_count 0 -> the one thematic-only row
        EEE in_universe, screen_count 2
    So: universe_total=4, zero_count=2 (AAA, BBB); thematic_only=1 (DDD);
    total_rows=5.
    _SYNTHETIC_SCREENS_DF: 3 registered screens (short_screen + 2
    thematic), so n_thematic_screens=2, n_total_screens=3."""

    def test_screens_on_help_states_exact_zero_count_sentence(self):
        help_map = build_overlap_help_map(_SYNTHETIC_OVERLAP_DF, _SYNTHETIC_SCREENS_DF)
        assert "2 of 4 in-universe tickers are on none." in help_map["screens_on"]

    def test_overall_score_help_states_exact_thematic_only_sentence(self):
        help_map = build_overlap_help_map(_SYNTHETIC_OVERLAP_DF, _SYNTHETIC_SCREENS_DF)
        assert "1 of 5 are thematic-only." in help_map["overall_score"]

    def test_screen_count_help_states_thematic_ceiling_not_observed_max(self):
        """Ruling: the STRUCTURAL ceiling (registry thematic-screen count),
        never screen_count_ceiling()'s OBSERVED maximum. Uses
        _SCREENS_DF_CEILING_DIFFERS_FROM_OBSERVED_MAX (4 thematic screens
        registered) against _SYNTHETIC_OVERLAP_DF (observed max
        screen_count is 2) so the two readings actually diverge — asserting
        the exact sentence catches a derivation that swapped in
        int(overlap_df["screen_count"].max()) instead of the registry
        count, which a fixture where both readings agree cannot."""
        help_map = build_overlap_help_map(
            _SYNTHETIC_OVERLAP_DF, _SCREENS_DF_CEILING_DIFFERS_FROM_OBSERVED_MAX
        )
        assert "the ceiling is 4, not 5." in help_map["screen_count"]

    def test_static_identity_help_unchanged(self):
        help_map = build_overlap_help_map(_SYNTHETIC_OVERLAP_DF, _SYNTHETIC_SCREENS_DF)
        assert help_map["ticker"] == OVERLAP_COLUMN_HELP["ticker"]
        assert help_map["market_cap"] == OVERLAP_COLUMN_HELP["market_cap"]

    def test_all_zero_screen_count_edge_case(self):
        df = pd.DataFrame({
            "ticker": ["AAA", "BBB"], "screen_count": [0, 0], "in_universe": [True, True],
        })
        screens_df = pd.DataFrame({
            "screen_id": ["short_screen"], "display_name": ["OWS Short Screen"],
            "screen_type": ["quant_composite"], "has_scoring": [True],
        })
        help_map = build_overlap_help_map(df, screens_df)
        assert "2 of 2 in-universe tickers are on none." in help_map["screens_on"]
        assert "the ceiling is 0, not 1." in help_map["screen_count"]


class TestRenderOverlapPageMissingMentionCount:
    """Property (iv): render_overlap_page must not KeyError when `filtered`
    lacks mention_count (the transcripts screen's table not loadable — see
    apply_overlap_metric_joins). filtered is hand-built here (no DB read
    needed for the shape itself), but render_overlap_page's OWN first
    action is `load_screen_membership()`, a DB_PATH-reading loader — T40
    requires that be stubbed regardless of whether its return value
    changes THIS test's outcome, so a future edit that starts depending on
    it doesn't silently start reading the real database in this test."""

    def _filtered_without_mention_count(self):
        return pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "name": ["A Co", "B Co"],
            "sector": ["Tech", "Health"],
            "market_cap": [100.0, 200.0],
            "screen_count": [1, 0],
            "screens_on": ["Structural", ""],
            "overall_score": [3.0, float("nan")],
            "in_universe": [True, False],
        })

    def _screens_df(self):
        return pd.DataFrame({
            "screen_id": ["short_screen", "structural"],
            "display_name": ["OWS Short Screen", "Structural"],
            "screen_type": ["quant_composite", "curated"],
            "has_scoring": [True, False],
        })

    def test_missing_mention_count_does_not_raise(self, monkeypatch):
        import src.app as app_module
        monkeypatch.setattr(app_module, "load_screen_membership", lambda: None)
        filtered = self._filtered_without_mention_count()
        help_map = build_overlap_help_map(filtered, self._screens_df())
        render_overlap_page(filtered, self._screens_df(), help_map)

    def test_present_mention_count_still_renders(self, monkeypatch):
        """Phase 6c review round 1, Correction 2.4: asserting only that the
        call does not raise cannot distinguish mention_count actually
        reaching the rendered table from OVERLAP_DISPLAY_COLUMNS (or
        present_cols) silently dropping it — both render without error.
        Captures the real st.dataframe call and asserts mention_count's
        column_config entry carries the label a viewer would actually see."""
        import src.app as app_module
        monkeypatch.setattr(app_module, "load_screen_membership", lambda: None)
        filtered = self._filtered_without_mention_count()
        filtered["mention_count"] = [5, 0]
        help_map = build_overlap_help_map(filtered, self._screens_df())

        calls = []
        monkeypatch.setattr(st, "dataframe", lambda *a, **k: calls.append(k))

        render_overlap_page(filtered, self._screens_df(), help_map)

        assert len(calls) == 1
        column_config = calls[0]["column_config"]
        assert "mention_count" in column_config
        assert column_config["mention_count"]["label"] == "Transcript Mentions"


# ---------------------------------------------------------------------------
# Phase 7a: use_container_width retirement, locked with ast rather than a
# text/regex scan (T35: a substring/regex match cannot distinguish a real
# keyword argument from a comment, string literal or docstring). Both
# assertions are universal over every ast.Call the walk finds in src/app.py,
# not over a fixed line list, so a call added at a later phase is covered
# automatically.
# ---------------------------------------------------------------------------


def _parse_app_module():
    app_path = os.path.join(PROJECT_ROOT, "src", "app.py")
    with open(app_path) as f:
        source = f.read()
    return ast.parse(source, filename=app_path)


def _call_final_attr(node):
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def test_no_call_anywhere_passes_use_container_width():
    tree = _parse_app_module()
    offenders = [
        (node.lineno, _call_final_attr(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "use_container_width"
    ]
    assert offenders == [], f"use_container_width still passed at: {offenders}"


def test_every_dataframe_and_altair_chart_call_has_width_stretch():
    tree = _parse_app_module()
    targets = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_final_attr(node) in ("dataframe", "altair_chart")
    ]
    assert targets, "no st.dataframe/st.altair_chart calls found in src/app.py — test is miswired"

    offenders = []
    for node in targets:
        width_kw = next((kw for kw in node.keywords if kw.arg == "width"), None)
        if width_kw is None:
            offenders.append((node.lineno, "missing width"))
        elif not (isinstance(width_kw.value, ast.Constant) and width_kw.value.value == "stretch"):
            offenders.append((node.lineno, f"width={ast.dump(width_kw.value)}"))
    assert offenders == [], f"non-compliant width kwarg at: {offenders}"


# ---------------------------------------------------------------------------
# Phase 8c-2: category-sum columns, membership flags, expansion band
# ---------------------------------------------------------------------------


class TestDisplayColumnsCategorySumPlacement:
    def test_each_categorys_factors_sit_immediately_after_its_own_sum_column(self):
        """FAILS IF the interleaving reverts to 'all sums, then all factors'
        (or any other placement) instead of each sum immediately preceding
        its own category's factors."""
        for category, factors in FACTOR_CATEGORIES.items():
            sum_col_name = category_sum_col_for(category)
            sum_idx = DISPLAY_COLUMNS.index(sum_col_name)
            actual = DISPLAY_COLUMNS[sum_idx + 1 : sum_idx + 1 + len(factors)]
            assert actual == factors, (category, actual, factors)

    def test_display_columns_has_no_duplicates(self):
        assert len(DISPLAY_COLUMNS) == len(set(DISPLAY_COLUMNS))

    def test_display_columns_ends_with_the_three_flags_in_order(self):
        assert DISPLAY_COLUMNS[-3:] == MAIN_TABLE_FLAG_COLUMNS

    def test_category_sum_help_is_generated_from_the_real_taxonomy(self):
        """FAILS IF the generated help spine stops naming a category's own
        factors (drifting from FACTOR_CATEGORIES) or drops the
        reconciliation sentence."""
        for category, factors in FACTOR_CATEGORIES.items():
            help_text = CATEGORY_SUM_COLUMN_HELP[category_sum_col_for(category)]
            assert str(len(factors)) in help_text
            assert "add back to Overall Score" in help_text


def category_sum_col_for(category: str) -> str:
    return next(name for name, label in CATEGORY_SUM_COLUMN_LABELS.items() if label == category)


class TestResolveExpandedDisplayColumns:
    def test_nothing_expanded_is_the_collapsed_default_view(self):
        """FAILS IF the default (nothing selected) view still shows the 24
        factor columns instead of collapsing to identity + overall_score +
        7 sums + 3 flags (16 columns)."""
        result = resolve_expanded_display_columns(DISPLAY_COLUMNS, FACTOR_CATEGORIES, [])
        assert len(result) == 16
        assert not any(c.endswith("_factor") for c in result)
        assert result[-3:] == MAIN_TABLE_FLAG_COLUMNS

    def test_expanding_one_category_reveals_only_its_own_factors(self):
        """FAILS IF expanding Growth also leaks another category's factors,
        or drops a factor it should show."""
        result = resolve_expanded_display_columns(DISPLAY_COLUMNS, FACTOR_CATEGORIES, ["Growth"])
        assert "decel_factor" in result and "accel_factor" in result
        other_factors = {
            f for cat, factors in FACTOR_CATEGORIES.items() if cat != "Growth" for f in factors
        }
        assert not (other_factors & set(result))
        growth_sum_idx = result.index("growth_sum")
        assert result[growth_sum_idx + 1 : growth_sum_idx + 3] == ["decel_factor", "accel_factor"]

    def test_expanding_a_synthetic_taxonomy_is_generic(self):
        """FAILS IF the un-hiding logic hardcodes the real taxonomy instead
        of walking whatever `categories` it's handed."""
        display_columns = ["id", "a_sum", "f1", "f2", "b_sum", "f3"]
        categories = {"A": ["f1", "f2"], "B": ["f3"]}
        assert resolve_expanded_display_columns(display_columns, categories, []) == [
            "id", "a_sum", "b_sum",
        ]
        assert resolve_expanded_display_columns(display_columns, categories, ["B"]) == [
            "id", "a_sum", "b_sum", "f3",
        ]

    @staticmethod
    def _synthetic_filtered():
        return pd.DataFrame({
            "ticker": ["AAAA", "BBBB", "CCCC", "DDDD"],
            "overall_score": [3.1, 1.2, 4.5, 2.3],
            "valuation_sum": [0.7, 0.2, 0.9, 0.4],
            "abs_ps_factor": [0.5, 0.1, 0.8, 0.3],
            "rel_ps_factor": [0.6, 0.2, 0.7, 0.4],
            "growth_sum": [0.3, 0.1, 0.5, 0.2],
            "decel_factor": [0.4, 0.2, 0.6, 0.3],
            "mscore_flag": [False, True, False, False],
            "overvalued_flag": [True, False, False, True],
            "transcripts_flag": [False, False, True, False],
        })

    @staticmethod
    def _display_df(filtered, expanded_categories):
        """Reproduces render_main_table's own column-then-sort composition
        (resolve_expanded_display_columns -> subset -> sort by
        overall_score), so this test exercises the same sequence of
        operations without needing Streamlit."""
        columns = resolve_expanded_display_columns(DISPLAY_COLUMNS, FACTOR_CATEGORIES, expanded_categories)
        available_cols = [c for c in columns if c in filtered.columns]
        return filtered[available_cols].sort_values("overall_score", ascending=False)

    def test_expansion_changes_columns_never_rows(self):
        """FAILS IF a future edit lets expansion state leak into row
        filtering, row count, or row order."""
        filtered = self._synthetic_filtered()
        collapsed = self._display_df(filtered, [])
        expanded = self._display_df(filtered, ["Growth"])

        assert collapsed["ticker"].tolist() == expanded["ticker"].tolist()
        assert len(collapsed) == len(expanded) == 4
        assert "decel_factor" not in collapsed.columns
        assert "decel_factor" in expanded.columns
        assert "abs_ps_factor" not in expanded.columns  # Valuation never expanded


class TestComputeScreenMembershipFlag:
    def test_flags_true_only_for_members(self):
        tickers = pd.Series(["AAAA", "BBBB", "CCCC"])
        membership_df = pd.DataFrame({
            "screen_id": ["overvalued_screen", "overvalued_screen", "negative_expert_transcripts"],
            "ticker": ["AAAA", "CCCC", "BBBB"],
        })
        flag = compute_screen_membership_flag(tickers, membership_df, OVERVALUED_SCREEN_ID)
        assert flag.tolist() == [True, False, True]

    def test_none_membership_df_returns_all_false(self):
        """FAILS IF a missing screen_membership table raises instead of
        degrading to all-False, matching load_screen_membership's own
        None-on-missing contract."""
        tickers = pd.Series(["AAAA", "BBBB"])
        flag = compute_screen_membership_flag(tickers, None, OVERVALUED_SCREEN_ID)
        assert flag.tolist() == [False, False]


def _get_function_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in src/app.py")


class TestRenderMainTableUsesGenericExpansion:
    """T47 — a lock on the USE, not just the constant. Proves render_main_
    table's own source actually calls resolve_expanded_display_columns with
    the taxonomy-derived DISPLAY_COLUMNS/FACTOR_CATEGORIES, so a future
    revert to a hardcoded column list (while resolve_expanded_display_
    columns itself stays correct and tested) fails here — exactly the gap a
    lock on the standalone function, or on a separately-defined constant,
    would miss."""

    def test_render_main_table_calls_resolve_expanded_display_columns_with_the_real_taxonomy(self):
        tree = _parse_app_module()
        fn = _get_function_node(tree, "render_main_table")
        calls = [
            node for node in ast.walk(fn)
            if isinstance(node, ast.Call) and _call_final_attr(node) == "resolve_expanded_display_columns"
        ]
        assert len(calls) == 1, f"expected exactly one call, found {len(calls)}"
        args = calls[0].args
        assert len(args) == 3
        assert isinstance(args[0], ast.Name) and args[0].id == "DISPLAY_COLUMNS"
        assert isinstance(args[1], ast.Name) and args[1].id == "FACTOR_CATEGORIES"


class TestExportCarriesCategorySums:
    def test_export_contains_all_seven_category_sums(self):
        """Extends TestExportIndependentOfCheckbox's own invariant: the
        export must carry all 7 category sums regardless of the on-screen
        checkbox. FAILS IF expansion/checkbox state leaks into the export
        column list, or a sum column is dropped from export."""
        for show_values in (False, True):
            columns = interleave_metric_columns(DISPLAY_COLUMNS) if show_values else DISPLAY_COLUMNS
            export_cols = set(build_export_columns(columns))
            missing = set(CATEGORY_SUM_COLUMN_LABELS) - export_cols
            assert missing == set(), (show_values, missing)


class TestOverlapViewNeverCarriesReweightedSums:
    """Standing Driver ruling (2026-09-08): a reweight never follows
    short_screen's Overall Score into the Cross-Screen Overlap view — and
    the 7 category sums are reweight-derived by construction, so they're
    bound by the same ruling.

    The real lock (T47): render_overlap_page's rendered/exported frames
    (display_df, export_df) are built SOLELY from OVERLAP_DISPLAY_COLUMNS/
    present_cols and never have a new column assigned into them. A
    disjointness check against the OVERLAP_DISPLAY_COLUMNS constant alone
    is necessary but not sufficient — it never touches the render path, so
    it stays green even if a later edit assigns
    display_df["valuation_sum"] = ... straight into the rendered frame
    (PM-verified: this exact mutation passed the constant-only check with
    the whole suite green). Walking render_overlap_page's own AST for a
    Subscript-assignment onto display_df/export_df is what actually
    catches that."""

    def test_overlap_display_columns_excludes_sums_and_new_flags(self):
        """Cheap, necessary but not sufficient on its own — kept as a
        second check alongside the AST lock below, not as the lock."""
        forbidden = set(CATEGORY_SUM_COLUMN_LABELS) | {"overvalued_flag", "transcripts_flag"}
        assert forbidden.isdisjoint(OVERLAP_DISPLAY_COLUMNS)

    def test_render_overlap_page_never_assigns_a_new_column_into_its_rendered_frames(self):
        """FAILS IF render_overlap_page (or anything it calls inline) sets
        display_df[<col>] = ... or export_df[<col>] = ... — the exact shape
        of routing a reweighted sum (or any other column not sourced from
        OVERLAP_DISPLAY_COLUMNS/present_cols) into the overlap view."""
        tree = _parse_app_module()
        fn = _get_function_node(tree, "render_overlap_page")
        offenders = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in ("display_df", "export_df")
                ):
                    offenders.append((node.lineno, target.value.id))
        assert offenders == [], f"column assigned into rendered/export frame at: {offenders}"
