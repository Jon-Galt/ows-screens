"""
OWS Screens — Streamlit Web UI.

A sidebar screen selector reads the screens registry and branches into one
of three rendering paths per screen: scored quant_composite screens (e.g.
short_screen — factor chart, M-Score, sector/industry filters), curated
screens (narrative rationale + three risk scores, no factor model), and
unscored quant_composite screens (e.g. Rising Short Interest — has a
transform stage but no factor model yet, so no chart/M-Score either). All
three get a filterable/sortable main table, a stock drill-down, and
Excel/CSV export.
"""

import io
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect

# Allow `streamlit run src/app.py` to resolve `src.*` imports even though
# running a file directly doesn't put the project root on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.db import table_name
from src.overlap import (
    UNIVERSE_SCREEN_ID,
    build_presence_matrix,
    compute_overlap,
    screen_count_ceiling,
    style_overlap_table,
)
from src.score import FACTOR_DEFINITIONS
from src.styling import build_color_scale_domain, style_scored_table

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OWS Screens",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "screener.db")

CURATED_DISPLAY_COLUMNS = [
    "ticker", "name", "sector", "market_cap", "daily_traded_value",
    "stock_performance", "valuation_ev_revenue_ntm_percentile",
    "score_accounting_and_disclosure", "score_fraud", "score_insider",
]

CURATED_SCORE_DISPLAY_NAMES = {
    "score_accounting_and_disclosure": "Accounting & Disclosure",
    "score_fraud": "Fraud",
    "score_insider": "Insider",
}

# Rising Short Interest: quant_composite in type, but unscored — no factor
# model, so no factor chart, no M-Score, and (unlike curated screens) no
# rationale field at all. Just the identity + the 8 derived metrics, flat.
UNSCORED_DISPLAY_COLUMNS = [
    "ticker", "name", "market_cap", "adv", "short_interest_pct",
    "si_change_3m", "si_change_6m", "week_52_high_chg", "ev_sales", "debt_ebitda",
]

UNSCORED_METRIC_DISPLAY_NAMES = {
    "market_cap": "Market Cap ($M)",
    "adv": "Avg Daily Value Traded ($M)",
    "short_interest_pct": "Short Interest %",
    "si_change_3m": "SI Change (3M)",
    "si_change_6m": "SI Change (6M)",
    "week_52_high_chg": "Change from 52W High",
    "ev_sales": "EV / Sales",
    "debt_ebitda": "Net Debt / EBITDA",
}

# Format-spec strings shared between the table (via Styler.format, which
# accepts these directly) and the drill-down (via str.format), so the two
# views render each metric identically rather than drifting apart.
UNSCORED_METRIC_FORMATS = {
    "market_cap": "${:,.0f}",
    "adv": "${:,.1f}",
    "short_interest_pct": "{:.2%}",
    "si_change_3m": "{:.1%}",
    "si_change_6m": "{:.1%}",
    "week_52_high_chg": "{:.1%}",
    "ev_sales": "{:.2f}",
    "debt_ebitda": "{:.2f}",
}

DISPLAY_COLUMNS = [
    "ticker", "name", "sector", "industry", "market_cap",
    "overall_score", "mscore_flag",
    "abs_ps_factor", "rel_ps_factor", "abs_fcf_factor", "rel_fcf_factor",
    "decel_factor", "accel_factor",
    "gm_factor", "ebit_factor",
    "debt_ebitda_factor", "debt_sales_factor", "debt_ev_factor",
    "refi_risk_factor", "liquidity_risk_factor",
    "fcf_conv_factor", "accrual_factor", "dso_factor", "dio_factor",
    "dpo_factor", "def_rev_factor", "dilution_factor",
    "ebit_adj_factor", "eps_adj_factor",
    "short_int_factor", "ratings_factor",
]

FACTOR_CATEGORIES = {
    "Valuation": ["abs_ps_factor", "rel_ps_factor", "abs_fcf_factor", "rel_fcf_factor"],
    "Growth": ["decel_factor", "accel_factor"],
    "Profitability": ["gm_factor", "ebit_factor"],
    "Balance Sheet": [
        "debt_ebitda_factor", "debt_sales_factor", "debt_ev_factor",
        "refi_risk_factor", "liquidity_risk_factor",
    ],
    "Cash Flow": [
        "fcf_conv_factor", "accrual_factor", "dso_factor", "dio_factor",
        "dpo_factor", "def_rev_factor", "dilution_factor",
    ],
    "Non-GAAP": ["ebit_adj_factor", "eps_adj_factor"],
    "Sentiment": ["short_int_factor", "ratings_factor"],
}

FACTOR_DISPLAY_NAMES = {
    "abs_ps_factor": "Abs. P/S",
    "rel_ps_factor": "Rel. P/S",
    "abs_fcf_factor": "Abs. FCF%",
    "rel_fcf_factor": "Rel. FCF%",
    "decel_factor": "Deceleration",
    "accel_factor": "Acceleration",
    "gm_factor": "Gross Margin",
    "ebit_factor": "EBIT Margin",
    "debt_ebitda_factor": "Debt/EBITDA",
    "debt_sales_factor": "Debt/Sales",
    "debt_ev_factor": "Debt/EV",
    "refi_risk_factor": "Refi Risk",
    "liquidity_risk_factor": "Liquidity Risk",
    "fcf_conv_factor": "FCF Conversion",
    "accrual_factor": "Accrual",
    "dso_factor": "DSO",
    "dio_factor": "DIO",
    "dpo_factor": "DPO",
    "def_rev_factor": "Def. Revenue",
    "dilution_factor": "Dilution",
    "ebit_adj_factor": "EBIT Adj.",
    "eps_adj_factor": "EPS Adj.",
    "short_int_factor": "Short Interest",
    "ratings_factor": "Ratings",
}

# Format-spec string for each factor's underlying metric column (i.e. the
# raw value a factor's percentile rank is computed from — FACTOR_DEFINITIONS
# gives the factor -> metric mapping). Pinned to the actual Excel number
# format on the "Screen" sheet of notebooks/OWS Short Screen (April
# 2026).xlsx rather than guessed, since the 24 metrics span wildly
# different scales (ratios, percentage-point diffs, dollar figures, years,
# multiples) that aren't inferable from the column name alone. Keyed by
# factor name (not metric column name) so it sits next to
# FACTOR_DISPLAY_NAMES and is trivially looked up alongside it; every
# factor maps to exactly one metric column, so there's no collision risk
# in keying it this way.
METRIC_FORMATS = {
    "abs_ps_factor": "{:.1%}",       # ps_diff — Excel "Diff." = 0.0%
    "rel_ps_factor": "{:.1f}",       # ps_ntm — Excel "P/Sales (NTM)" = 0.0
    "abs_fcf_factor": "{:.1%}",      # fcf_yield_diff — Excel "Diff." = 0.0%
    "rel_fcf_factor": "{:.1%}",      # fcf_yield — Excel "FCF Yield (LTM)" = 0.0%
    "decel_factor": "{:.1%}",        # growth_decel — Excel "Diff" = 0.0%
    "accel_factor": "{:.1%}",        # growth_accel — Excel "Diff." = 0.0%
    "gm_factor": "{:.1%}",           # gm_diff — Excel "Diff." = 0.0%
    "ebit_factor": "{:.1%}",         # ebit_diff — Excel "Diff." = 0.0%
    "debt_ebitda_factor": "{:.2f}",  # leverage_ratio_calc — Excel "Leverage Ratio" = 0.00
    "debt_sales_factor": "{:.2f}",   # debt_sales — Excel "Debt/TTM Sales" = 0.00
    "debt_ev_factor": "{:.1f}",      # debt_ev — Excel "Debt/EV" = 0.0
    "refi_risk_factor": "{:.1f}",    # weighted_avg_maturity — Excel "Weighted Avg.Maturity (Y)" = 0.0
    "liquidity_risk_factor": "{:.1f}",  # remaining_liquidity_years — Excel "Remaining Liquidity (Y)" = General
    "fcf_conv_factor": "{:.0%}",     # fcf_conversion — Excel "FCF Conv." = 0%
    "accrual_factor": "{:.1%}",      # accrual_ratio — Excel "CFO/Net Income (TTM)" = 0.0%
    "dso_factor": "{:.1%}",          # dso_pct_change — Excel "Diff." = 0.0%
    "dio_factor": "{:.1%}",          # dio_pct_change — Excel "Diff." = 0.0%
    "dpo_factor": "{:.1%}",          # dpo_pct_change — Excel "Diff." = 0.0%
    "def_rev_factor": "{:.1%}",      # deferred_rev_pct_change — Excel "Diff." = 0.0%
    "dilution_factor": "{:.1%}",     # dilution_p3y — Excel "Dilution (P3Y)" = 0.0%
    "ebit_adj_factor": "{:.1f}",     # non_gaap_gaap_ebit — Excel "Adj. EBIT/GAAP EBIT" = 0.0
    "eps_adj_factor": "{:.1f}",      # eps_adj_ratio — Excel "Adj. EPS/GAAP EPS" = accounting 0.0
    "short_int_factor": "{:.1%}",    # short_interest_pct — Excel "Short Int. (%)" = 0.0%
    "ratings_factor": "{:.1%}",      # hold_sell_pct — Excel "Hold/Sell %" = 0.0%
}

# METRIC_FORMATS re-keyed by metric column name (rather than factor name),
# for merging directly into a Styler.format() dict alongside factor-score
# formats, which are keyed by their own column names.
METRIC_COLUMN_FORMATS = {
    FACTOR_DEFINITIONS[factor]["metric"]: fmt for factor, fmt in METRIC_FORMATS.items()
}

# ---------------------------------------------------------------------------
# Phase 3c.2: pre-diff inputs behind the 10 diff-based factors
# ---------------------------------------------------------------------------
#
# Each factor's two inputs, in the order the Excel template's block layout
# presents them (notebooks/OWS Short Screen (April 2026).xlsx, "Screen"
# sheet), with the transform.py calc function each pair was read from —
# not guessed from column names. This mapping is display-only; it does not
# change what's computed or stored.
#
# DSOs/DIOs/DPOs are the case this guards against: the raw upload also
# carries a "3yr. Avg." column for each (dsos_3yr_avg, dios_3yr_avg,
# dpos_3yr_avg — see ingest.py), and it sits in the same template block,
# but none of calc_dso_pct_change/calc_dio_pct_change/calc_dpo_pct_change
# reference it — those diffs are current-quarter vs. same-quarter-prior-
# year (the "PY"/"T-1" column), not vs. the 3-year average. Confirmed by
# reading the calc function bodies, not by matching plausible names.
DIFF_FACTOR_INPUTS = {
    "abs_ps_factor": [
        ("ps_ntm", "P/Sales (NTM)", "calc_ps_diff"),
        ("ps_3yr_avg", "P/Sales (3yr. Avg.)", "calc_ps_diff"),
    ],
    "abs_fcf_factor": [
        ("fcf_yield", "FCF Yield (LTM)", "calc_fcf_yield_diff"),
        ("fcf_yield_3yr_avg", "FCF Yield (3yr. Avg.)", "calc_fcf_yield_diff"),
    ],
    "decel_factor": [
        ("yoy_growth_ttm_t1", "Y/Y Growth (TTM-1)", "calc_growth_decel"),
        ("yoy_growth_ttm", "Y/Y Growth (TTM)", "calc_growth_decel"),
    ],
    "accel_factor": [
        ("rev_cagr_p2y", "Rev CAGR (P2Y)", "calc_growth_accel"),
        ("rev_cagr_f2y", "Rev CAGR (F2Y)", "calc_growth_accel"),
    ],
    "gm_factor": [
        ("ntm_gross_margin", "GM% (NTM)", "calc_gm_diff"),
        ("gross_margin_3yr_avg", "GM% (3yr. Avg.)", "calc_gm_diff"),
    ],
    "ebit_factor": [
        ("ntm_ebit_margin", "EBIT% (NTM)", "calc_ebit_diff"),
        ("ebit_margin_3yr_avg", "EBIT% (3yr. Avg.)", "calc_ebit_diff"),
    ],
    "dso_factor": [
        ("dsos", "DSOs", "calc_dso_pct_change"),
        ("dsos_py", "DSOs (PY)", "calc_dso_pct_change"),
    ],
    "dio_factor": [
        ("dios", "DIOs", "calc_dio_pct_change"),
        ("dios_t1", "DIOs (T-1)", "calc_dio_pct_change"),
    ],
    "dpo_factor": [
        ("dpos", "DPOs", "calc_dpo_pct_change"),
        ("dpos_t1", "DPOs (T-1)", "calc_dpo_pct_change"),
    ],
    "def_rev_factor": [
        ("days_deferred_rev", "Days Def. Rev.", "calc_deferred_rev_pct_change"),
        ("days_deferred_rev_t1", "Days Def. Rev. (T-1)", "calc_deferred_rev_pct_change"),
    ],
}

# Format-spec strings for the input columns above, pinned to the same Excel
# template's number_format the same way METRIC_FORMATS was. Deliberately
# excludes ps_ntm and fcf_yield — those two double as rel_ps_factor's and
# rel_fcf_factor's own metric (see SHARED_INPUT_NOTES below) and already
# have a format in METRIC_COLUMN_FORMATS; duplicating them here would be a
# second copy that could drift out of sync with the first.
DIFF_INPUT_FORMATS = {
    "ps_3yr_avg": "{:.1f}",             # Excel "P/Sales (3yr. Avg.)" = 0.0
    "fcf_yield_3yr_avg": "{:.1%}",      # Excel "FCF Yield (3yr. Avg.)" = 0.0%
    "yoy_growth_ttm": "{:.1%}",         # Excel "Y/Y Growth (TTM)" = 0.0%
    "yoy_growth_ttm_t1": "{:.1%}",      # Excel "Y/Y Growth (TTM-1)" = 0.0%
    "rev_cagr_p2y": "{:.1%}",           # Excel "Rev CAGR (P2Y)" = 0.0%
    "rev_cagr_f2y": "{:.1%}",           # Excel "Rev CAGR (F2Y)" = 0.0%
    "ntm_gross_margin": "{:.1%}",       # Excel "GM% (NTM)" = 0.0%
    "gross_margin_3yr_avg": "{:.1%}",   # Excel "GM% (3yr. Avg.)" = 0.0%
    "ntm_ebit_margin": "{:.1%}",        # Excel "EBIT% (NTM)" = 0.0%
    "ebit_margin_3yr_avg": "{:.1%}",    # Excel "EBIT% (3yr. Avg.)" = 0.0%
    "dsos": "{:.1f}",                   # Excel "DSOs" = 0.0
    "dsos_py": "{:.1f}",                # Excel "DSOs (PY)" = 0.0
    "dios": "{:.1f}",                   # Excel "DIOs" = 0.0
    "dios_t1": "{:.1f}",                # Excel "DIOs (T-1)" = 0.0
    "dpos": "{:.1f}",                   # Excel "DPOs" = 0.0
    "dpos_t1": "{:.1f}",                # Excel "DPOs (T-1)" = 0.0
    "days_deferred_rev": "{:.1f}",      # Excel "Days Def. Rev." = 0.0
    "days_deferred_rev_t1": "{:.1f}",   # Excel "Days Def. Rev. (T-1)" = 0.0
}

# Single format lookup for every diff-based factor's input column, keyed by
# column name, so the drill-down derivation and the export both format each
# column identically — ps_ntm/fcf_yield resolve through METRIC_COLUMN_FORMATS
# (already correct), everything else through DIFF_INPUT_FORMATS.
INPUT_COLUMN_FORMATS = {**METRIC_COLUMN_FORMATS, **DIFF_INPUT_FORMATS}

# Every input column referenced anywhere in DIFF_FACTOR_INPUTS, deduplicated.
# Used to extend the Excel/CSV export with the full derivation even though
# these columns stay out of the on-screen main table (Phase 3c.2 scope
# decision — see PHASE3C2_APPROVAL.md).
DIFF_INPUT_COLUMNS = sorted({
    col for cols in DIFF_FACTOR_INPUTS.values() for col, _label, _func in cols
})

# A column that doubles as another factor's own metric (ps_ntm is also
# rel_ps_factor's metric; fcf_yield is also rel_fcf_factor's metric) gets an
# inline note in the derivation expander so the recurrence reads as "this is
# the same figure, on purpose" rather than as a second, different number.
# Phrased as a relationship, not a position, so it stays true regardless of
# where either factor's block ends up on the page.
SHARED_INPUT_NOTES = {
    "ps_ntm": "also used by Rel. P/S",
    "fcf_yield": "also used by Rel. FCF%",
}

# ---------------------------------------------------------------------------
# Phase 5a: on-screen column header labels (main table + curated + RSI +
# overlap). Display-only — st.column_config.Column(label=...) relabels a
# table's header without touching the DataFrame's own column names, so
# every existing sort_values/format/export keyed by DB column name is
# unaffected. Exports are untouched by construction: they go through
# DataFrame.to_excel/to_csv directly, never through st.dataframe's
# column_config.
#
# FACTOR_DISPLAY_NAMES itself is left unmodified — it is used today only by
# the Stock Drill-Down chart/table, which is out of this phase's scope, so
# its abbreviated strings ("Abs. P/S", "Refi Risk", etc.) are not touched
# as a side effect of the main table's relabeling. FACTOR_EXPANDED_LABELS
# below is a separate, main-table-only expansion.
# ---------------------------------------------------------------------------

IDENTITY_COLUMN_LABELS = {
    "ticker": "Ticker",
    "name": "Name",
    "sector": "Sector",
    "industry": "Industry",
    "market_cap": "Market Cap ($M)",
}

# FACTOR_DISPLAY_NAMES' text with only the abbreviations Tom named expanded
# to full words (Abs.->Absolute, Rel.->Relative, Refi->Refinancing,
# "Def. Revenue"->"Deferred Revenue", Adj.->Adjusted, Conv.->Conversion).
# Standard finance shorthand he did not ask to expand (DSO, DIO, DPO, EBIT,
# FCF, P/S, EV, GM, M-Score, SI, TTM, NTM, LTM) is left as-is. Hand-written,
# not derived by string substitution, so it's reviewable as a literal table.
FACTOR_EXPANDED_LABELS = {
    "abs_ps_factor": "Absolute P/S",
    "rel_ps_factor": "Relative P/S",
    "abs_fcf_factor": "Absolute FCF%",
    "rel_fcf_factor": "Relative FCF%",
    "decel_factor": "Deceleration",
    "accel_factor": "Acceleration",
    "gm_factor": "Gross Margin",
    "ebit_factor": "EBIT Margin",
    "debt_ebitda_factor": "Debt/EBITDA",
    "debt_sales_factor": "Debt/Sales",
    "debt_ev_factor": "Debt/EV",
    "refi_risk_factor": "Refinancing Risk",
    "liquidity_risk_factor": "Liquidity Risk",
    "fcf_conv_factor": "FCF Conversion",
    "accrual_factor": "Accrual",
    "dso_factor": "DSO",
    "dio_factor": "DIO",
    "dpo_factor": "DPO",
    "def_rev_factor": "Deferred Revenue",
    "dilution_factor": "Dilution",
    "ebit_adj_factor": "EBIT Adjusted",
    "eps_adj_factor": "EPS Adjusted",
    "short_int_factor": "Short Interest",
    "ratings_factor": "Ratings",
}

# The Excel template's own header row appends "Factor" to a factor score
# column's name (e.g. "Debt/EV Factor") — adopted here for the same reason:
# without it, a factor score column and its own metric column (shown when
# "Show underlying metric values" is on) would read identically, e.g.
# debt_ev_factor and debt_ev both as "Debt/EV".
FACTOR_COLUMN_LABELS = {
    factor: f"{label} Factor" for factor, label in FACTOR_EXPANDED_LABELS.items()
}


def _label_from_diff_inputs(metric_col: str) -> str:
    """The column label DIFF_FACTOR_INPUTS already uses for metric_col as
    one of its two diff inputs — reused verbatim (not retyped) for ps_ntm
    and fcf_yield below, since both double as another factor's own metric
    (see SHARED_INPUT_NOTES) and must read identically in the main table
    and the drill-down, not as two different names for one column."""
    for inputs in DIFF_FACTOR_INPUTS.values():
        for col, label, _source_func in inputs:
            if col == metric_col:
                return label
    raise KeyError(metric_col)


# The 14 non-diff metric columns' own Excel names (METRIC_FORMATS'
# comments), with the same six expansions applied where they appear
# (non_gaap_gaap_ebit, eps_adj_ratio). ps_ntm and fcf_yield are excluded
# here — they're populated from DIFF_FACTOR_INPUTS below instead, via
# _label_from_diff_inputs, so there is exactly one place that could drift.
# weighted_avg_maturity's Excel-comment source string is missing a space
# ("Weighted Avg.Maturity (Y)") — treated as a comment typo, not a header
# spec, and transcribed here with the space restored.
_NON_DIFF_METRIC_LABELS = {
    "leverage_ratio_calc": "Leverage Ratio",
    "debt_sales": "Debt/TTM Sales",
    "debt_ev": "Debt/EV",
    "weighted_avg_maturity": "Weighted Avg. Maturity (Y)",
    "remaining_liquidity_years": "Remaining Liquidity (Y)",
    "fcf_conversion": "FCF Conversion",
    "accrual_ratio": "CFO/Net Income (TTM)",
    "dilution_p3y": "Dilution (P3Y)",
    "non_gaap_gaap_ebit": "Adjusted EBIT/GAAP EBIT",
    "eps_adj_ratio": "Adjusted EPS/GAAP EPS",
    "short_interest_pct": "Short Int. (%)",
    "hold_sell_pct": "Hold/Sell %",
}

# Every interleaved metric column's main-table label: the 10 diff-based
# metrics as "<factor's expanded label> — Diff." (previously all identically
# "Diff." — see PHASE5A build history), the other 14 via their own Excel
# name above (ps_ntm/fcf_yield via DIFF_FACTOR_INPUTS specifically).
METRIC_COLUMN_LABELS = {
    FACTOR_DEFINITIONS[factor]["metric"]: f"{FACTOR_EXPANDED_LABELS[factor]} — Diff."
    for factor in DIFF_FACTOR_INPUTS
}
METRIC_COLUMN_LABELS.update(_NON_DIFF_METRIC_LABELS)
METRIC_COLUMN_LABELS["ps_ntm"] = _label_from_diff_inputs("ps_ntm")
METRIC_COLUMN_LABELS["fcf_yield"] = _label_from_diff_inputs("fcf_yield")

# The label map render_main_table's column_config draws from: identity
# columns, Overall Score / M-Score Flag, every factor score column, and
# every interleaved metric column.
MAIN_TABLE_COLUMN_LABELS = {
    **IDENTITY_COLUMN_LABELS,
    "overall_score": "Overall Score",
    "mscore_flag": "M-Score Flag",
    **FACTOR_COLUMN_LABELS,
    **METRIC_COLUMN_LABELS,
}

# render_curated_table's column_config map.
CURATED_COLUMN_LABELS = {
    "ticker": "Ticker",
    "name": "Name",
    "sector": "Sector",
    "market_cap": "Market Cap ($M)",
    "daily_traded_value": "Daily Traded Value ($M)",
    "stock_performance": "Stock Performance",
    "valuation_ev_revenue_ntm_percentile": "EV/Revenue (NTM) Percentile",
    **CURATED_SCORE_DISPLAY_NAMES,
}

# render_unscored_table's column_config map — reuses the existing
# UNSCORED_METRIC_DISPLAY_NAMES (already full words) and adds the two
# identity columns it doesn't cover.
UNSCORED_COLUMN_LABELS = {
    "ticker": "Ticker",
    "name": "Name",
    **UNSCORED_METRIC_DISPLAY_NAMES,
}

# render_overlap_view's column_config map for every display_cols entry
# except overall_score, which keeps its own existing
# f"{universe_display_name} Composite Score" label (Phase 3d Part 1,
# preserved as-is).
OVERLAP_COLUMN_LABELS = {
    "ticker": "Ticker",
    "name": "Name",
    "sector": "Sector",
    "market_cap": "Market Cap ($M)",
    "screen_count": "Screen Count",
    "screens_on": "Screens On",
}


def interleave_metric_columns(columns: list) -> list:
    """Insert each factor's underlying metric column immediately after it.

    Shared by the "Show underlying metric values" checkbox (render_main_table)
    and the export column list, which always includes every metric
    regardless of that checkbox — see build_export_columns.

    Args:
        columns: A column list (e.g. DISPLAY_COLUMNS) that may contain
            factor-score column names.

    Returns:
        columns with each factor's FACTOR_DEFINITIONS metric column
        inserted right after it.
    """
    result = []
    for col in columns:
        result.append(col)
        if col in FACTOR_DEFINITIONS:
            result.append(FACTOR_DEFINITIONS[col]["metric"])
    return result


def build_export_columns(display_columns: list) -> list:
    """Column list for the Excel/CSV export: the on-screen display columns
    plus every diff-based factor's input column, even though those inputs
    stay out of the on-screen main table.

    Tom's mental model of this data is the flat export file, where every
    input sits in its own column — drill-down-only is the right on-screen
    answer, but the export should still give him that flat view.

    Args:
        display_columns: The columns currently shown on screen (varies with
            the "Show underlying metric values" checkbox).

    Returns:
        display_columns plus any DIFF_INPUT_COLUMNS not already present,
        in that order, with no duplicates.
    """
    seen = set(display_columns)
    extra = [c for c in DIFF_INPUT_COLUMNS if c not in seen]
    return display_columns + extra


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def list_screens() -> pd.DataFrame | None:
    """Load the screens registry (screen_id, display_name, screen_type,
    has_scoring).

    Returns None if the database, the screens table, or any registered
    screens don't exist yet.
    """
    if not os.path.exists(DB_PATH):
        return None
    engine = create_engine(f"sqlite:///{DB_PATH}")
    if "screens" not in set(inspect(engine).get_table_names()):
        return None
    df = pd.read_sql_table("screens", engine)
    if df.empty:
        return None
    return df


@st.cache_data
def load_quant_data(screen_id: str) -> pd.DataFrame | None:
    """Load a quant_composite screen's scored_data from SQLite.

    Returns None if the database or that screen's scored_data table
    doesn't exist yet, or if that table is empty.
    """
    if not os.path.exists(DB_PATH):
        return None
    engine = create_engine(f"sqlite:///{DB_PATH}")
    scored_table = table_name("scored_data", screen_id)
    if scored_table not in set(inspect(engine).get_table_names()):
        return None
    df = pd.read_sql_table(scored_table, engine)
    if df.empty:
        return None
    return df


@st.cache_data
def load_curated_data(screen_id: str) -> pd.DataFrame | None:
    """Load a curated screen's curated_data from SQLite.

    Returns None if the database or that screen's curated_data table
    doesn't exist yet, or if that table is empty.
    """
    if not os.path.exists(DB_PATH):
        return None
    engine = create_engine(f"sqlite:///{DB_PATH}")
    curated_table = table_name("curated_data", screen_id)
    if curated_table not in set(inspect(engine).get_table_names()):
        return None
    df = pd.read_sql_table(curated_table, engine)
    if df.empty:
        return None
    return df


@st.cache_data
def load_unscored_quant_data(screen_id: str) -> pd.DataFrame | None:
    """Load an unscored quant_composite screen's transformed_data from
    SQLite (e.g. Rising Short Interest — quant_composite in type, but with
    no factor model, so it has no scored_data table at all).

    Returns None if the database or that screen's transformed_data table
    doesn't exist yet, or if that table is empty.
    """
    if not os.path.exists(DB_PATH):
        return None
    engine = create_engine(f"sqlite:///{DB_PATH}")
    transformed_table = table_name("transformed_data", screen_id)
    if transformed_table not in set(inspect(engine).get_table_names()):
        return None
    df = pd.read_sql_table(transformed_table, engine)
    if df.empty:
        return None
    return df


@st.cache_data
def load_screen_membership() -> pd.DataFrame | None:
    """Load the full screen_membership table (screen_id, ticker) across
    every screen, for the cross-screen overlap view.

    Returns None if the database or the screen_membership table doesn't
    exist yet, or if it's empty.
    """
    if not os.path.exists(DB_PATH):
        return None
    engine = create_engine(f"sqlite:///{DB_PATH}")
    if "screen_membership" not in set(inspect(engine).get_table_names()):
        return None
    df = pd.read_sql_table("screen_membership", engine)
    if df.empty:
        return None
    return df


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered DataFrame."""
    st.sidebar.header("Filters")

    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    # Sector filter
    all_sectors = sorted(df["sector"].dropna().unique())
    selected_sectors = st.sidebar.multiselect("Sector", options=all_sectors)

    # Industry filter — dependent on sector selection
    if selected_sectors:
        available_industries = sorted(
            df[df["sector"].isin(selected_sectors)]["industry"].dropna().unique()
        )
    else:
        available_industries = sorted(df["industry"].dropna().unique())
    selected_industries = st.sidebar.multiselect("Industry", options=available_industries)

    # Market cap slider
    mcap_min = float(df["market_cap"].min())
    mcap_max = float(df["market_cap"].max())
    mcap_range = st.sidebar.slider(
        "Market Cap ($M)",
        min_value=mcap_min,
        max_value=mcap_max,
        value=(mcap_min, mcap_max),
        format="$%.0f",
    )

    # Overall score slider
    score_min = float(df["overall_score"].min())
    score_max = float(df["overall_score"].max())
    score_range = st.sidebar.slider(
        "Overall Score",
        min_value=0.0,
        max_value=7.0,
        value=(score_min, score_max),
        step=0.1,
    )

    # Apply filters
    filtered = df.copy()
    if selected_sectors:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]
    if selected_industries:
        filtered = filtered[filtered["industry"].isin(selected_industries)]
    filtered = filtered[
        (filtered["market_cap"] >= mcap_range[0])
        & (filtered["market_cap"] <= mcap_range[1])
    ]
    filtered = filtered[
        (filtered["overall_score"] >= score_range[0])
        & (filtered["overall_score"] <= score_range[1])
    ]

    st.sidebar.divider()
    st.sidebar.metric("Stocks shown", len(filtered))

    return filtered


# ---------------------------------------------------------------------------
# Main table
# ---------------------------------------------------------------------------


def render_main_table(filtered: pd.DataFrame, domain_df: pd.DataFrame) -> None:
    """Render the main scored table with export buttons.

    A checkbox (default off, so the existing view is unchanged unless a
    user opts in) shows each factor's underlying metric — the raw value
    its percentile score was computed from — immediately after it.

    Args:
        filtered: The sidebar-filtered DataFrame to display.
        domain_df: The full UNFILTERED scored short_screen DataFrame, used
            only to compute each colour-scaled column's (lo, mid, hi)
            domain once (Driver ruling, Phase 5a: a stock's colour must not
            change when sidebar filters change).
    """
    show_values = st.checkbox(
        "Show underlying metric values",
        value=False,
        help="Insert each factor's underlying value (what its percentile "
        "score was computed from) immediately after it.",
    )

    # Prepare display DataFrame
    columns = interleave_metric_columns(DISPLAY_COLUMNS) if show_values else DISPLAY_COLUMNS

    available_cols = [c for c in columns if c in filtered.columns]
    display_df = filtered[available_cols].sort_values("overall_score", ascending=False)

    # Export always includes every underlying value — all 24 factor metrics
    # and all 20 diff-based inputs — regardless of the on-screen checkbox.
    # The checkbox controls the SCREEN only; it must not gate what's
    # exported, or the export's contents would silently depend on whether
    # a user happened to have it ticked when they clicked download.
    all_metric_cols = interleave_metric_columns(DISPLAY_COLUMNS)
    export_cols = [c for c in build_export_columns(all_metric_cols) if c in filtered.columns]
    export_df = filtered[export_cols].sort_values("overall_score", ascending=False)

    # Export buttons
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        xlsx_buffer = io.BytesIO()
        export_df.to_excel(xlsx_buffer, index=False, engine="openpyxl")
        st.download_button(
            label="Export to Excel",
            data=xlsx_buffer.getvalue(),
            file_name="ows_short_screen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="Export to CSV",
            data=csv_data,
            file_name="ows_short_screen.csv",
            mime="text/csv",
        )

    # Style and display
    format_dict = {
        "market_cap": "${:,.0f}",
        "overall_score": "{:.3f}",
        **{f: "{:.3f}" for f in available_cols if f.endswith("_factor")},
    }
    if show_values:
        format_dict.update(
            {c: METRIC_COLUMN_FORMATS[c] for c in available_cols if c in METRIC_COLUMN_FORMATS}
        )

    factor_columns = [c for c in available_cols if c.endswith("_factor")]
    scale_columns = [c for c in (["overall_score"] + factor_columns) if c in domain_df.columns]
    domain = build_color_scale_domain(domain_df, scale_columns)

    styled = style_scored_table(display_df, domain, factor_columns)
    styled = styled.format(format_dict)

    column_config = {
        col: st.column_config.Column(label=MAIN_TABLE_COLUMN_LABELS[col])
        for col in available_cols
        if col in MAIN_TABLE_COLUMN_LABELS
    }

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config=column_config,
    )


# ---------------------------------------------------------------------------
# Drill-down
# ---------------------------------------------------------------------------


def render_diff_derivation(row: pd.Series, factor: str) -> None:
    """Expander showing a diff-based factor's full derivation: its two
    inputs (in the Excel template's block order), the diff, and the score.

    Only called for the 10 factors in DIFF_FACTOR_INPUTS — this is the
    Phase 3c.2 drill-down-only treatment; non-diff factors keep the plain
    score+metric row from render_drill_down's factor table.

    Args:
        row: One stock's row from a scored short_screen DataFrame.
        factor: A key of DIFF_FACTOR_INPUTS.
    """
    label = FACTOR_DISPLAY_NAMES.get(factor, factor)
    with st.expander(f"Show derivation — {label}"):
        for col, input_label, _source_func in DIFF_FACTOR_INPUTS[factor]:
            if col in row.index and pd.notna(row[col]):
                fmt = INPUT_COLUMN_FORMATS.get(col, "{}")
                value_str = fmt.format(row[col])
            else:
                value_str = "N/A"
            note = SHARED_INPUT_NOTES.get(col)
            line = f"- **{input_label}**: {value_str}"
            if note:
                line += f" _({note})_"
            st.markdown(line)

        metric_col = FACTOR_DEFINITIONS[factor]["metric"]
        if metric_col in row.index and pd.notna(row[metric_col]):
            diff_str = METRIC_FORMATS.get(factor, "{}").format(row[metric_col])
        else:
            diff_str = "N/A"
        st.markdown(f"- **Diff.**: {diff_str}")

        score_val = row[factor]
        score_str = f"{score_val:.3f}" if pd.notna(score_val) else "N/A"
        st.markdown(f"- **Score**: {score_str}")


def render_drill_down(filtered: pd.DataFrame) -> None:
    """Render the individual stock drill-down view."""
    tickers = sorted(filtered["ticker"].dropna().unique())
    if not tickers:
        st.info("No stocks match the current filters.")
        return

    selected_ticker = st.selectbox("Select a stock", options=tickers)
    row = filtered[filtered["ticker"] == selected_ticker].iloc[0]

    # Identity card
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", row["ticker"])
    col2.metric("Overall Score", f"{row['overall_score']:.3f}")
    col3.metric("M-Score", f"{row['mscore']:.2f}" if pd.notna(row["mscore"]) else "N/A")
    mscore_flag_str = "Yes" if row.get("mscore_flag", False) else "No"
    col4.metric("Manipulation Flag", mscore_flag_str)

    st.markdown(
        f"**{row['name']}** · {row['sector']} · {row['industry']} · "
        f"Market Cap: ${row['market_cap']:,.0f}M"
    )

    st.divider()

    # Build factor data for chart and table
    chart_rows = []
    for category, factors in FACTOR_CATEGORIES.items():
        for factor in factors:
            if factor in row.index and pd.notna(row[factor]):
                chart_rows.append({
                    "Category": category,
                    "Factor": FACTOR_DISPLAY_NAMES.get(factor, factor),
                    "Score": float(row[factor]),
                })

    if not chart_rows:
        st.warning("No factor score data available for this stock.")
        return

    chart_df = pd.DataFrame(chart_rows)

    # Preserve factor order (top-to-bottom in chart = first category first)
    factor_order = list(chart_df["Factor"])
    category_order = list(FACTOR_CATEGORIES.keys())

    # Altair horizontal bar chart
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Factor Score"),
            y=alt.Y("Factor:N", sort=factor_order, title=None),
            color=alt.Color(
                "Category:N",
                sort=category_order,
                legend=alt.Legend(title="Category"),
            ),
            tooltip=["Category", "Factor", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=max(len(chart_rows) * 25, 300))
    )
    st.altair_chart(chart, use_container_width=True)

    # Factor table by category — Score alongside the underlying metric
    # Value it was computed from, so a score is actually research-usable
    # (e.g. a 0.92 next to the 11.5x behind it) rather than shown alone.
    st.subheader("Factor Scores by Category")
    for category, factors in FACTOR_CATEGORIES.items():
        table_rows = []
        for factor in factors:
            if factor in row.index:
                val = row[factor]
                metric_col = FACTOR_DEFINITIONS.get(factor, {}).get("metric")
                if metric_col and metric_col in row.index and pd.notna(row[metric_col]):
                    value_str = METRIC_FORMATS.get(factor, "{}").format(row[metric_col])
                else:
                    value_str = "N/A"
                table_rows.append({
                    "Factor": FACTOR_DISPLAY_NAMES.get(factor, factor),
                    "Score": f"{val:.3f}" if pd.notna(val) else "N/A",
                    "Value": value_str,
                })
        if table_rows:
            st.markdown(f"**{category}**")
            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True,
                hide_index=True,
                height=min(len(table_rows) * 40 + 40, 300),
            )
            for factor in factors:
                if factor in DIFF_FACTOR_INPUTS and factor in row.index:
                    render_diff_derivation(row, factor)


# ---------------------------------------------------------------------------
# Curated screens — separate rendering path
# ---------------------------------------------------------------------------
#
# Curated screens have no factor scores, no overall_score, no M-Score —
# just identity/market fields, three risk scores, and a narrative rationale.
# These functions are entirely separate from render_sidebar/render_main_table/
# render_drill_down above, which stay unchanged for short_screen.


def render_curated_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters for a curated screen and return the filtered
    DataFrame. Sector and market cap only — there is no score to filter on."""
    st.sidebar.header("Filters")

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    all_sectors = sorted(df["sector"].dropna().unique())
    selected_sectors = st.sidebar.multiselect("Sector", options=all_sectors)

    mcap_min = float(df["market_cap"].min())
    mcap_max = float(df["market_cap"].max())
    mcap_range = st.sidebar.slider(
        "Market Cap ($M)",
        min_value=mcap_min,
        max_value=mcap_max,
        value=(mcap_min, mcap_max),
        format="$%.0f",
    )

    filtered = df.copy()
    if selected_sectors:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]
    filtered = filtered[
        (filtered["market_cap"] >= mcap_range[0])
        & (filtered["market_cap"] <= mcap_range[1])
    ]

    st.sidebar.divider()
    st.sidebar.metric("Stocks shown", len(filtered))

    return filtered


def render_curated_table(filtered: pd.DataFrame) -> None:
    """Render the main curated table with export buttons. No M-Score
    highlighting and no factor columns — there aren't any."""
    available_cols = [c for c in CURATED_DISPLAY_COLUMNS if c in filtered.columns]
    display_df = filtered[available_cols].sort_values("ticker")

    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        xlsx_buffer = io.BytesIO()
        display_df.to_excel(xlsx_buffer, index=False, engine="openpyxl")
        st.download_button(
            label="Export to Excel",
            data=xlsx_buffer.getvalue(),
            file_name="ows_curated_screen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="Export to CSV",
            data=csv_data,
            file_name="ows_curated_screen.csv",
            mime="text/csv",
        )

    styled = display_df.style.format(
        {
            "market_cap": "${:,.0f}",
            "daily_traded_value": "${:,.1f}",
            "stock_performance": "{:.2%}",
            "valuation_ev_revenue_ntm_percentile": "{:.1%}",
            "score_accounting_and_disclosure": "{:.0f}",
            "score_fraud": "{:.0f}",
            "score_insider": "{:.0f}",
        }
    )

    column_config = {
        col: st.column_config.Column(label=CURATED_COLUMN_LABELS[col])
        for col in available_cols
        if col in CURATED_COLUMN_LABELS
    }

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config=column_config,
    )


def render_curated_drill_down(filtered: pd.DataFrame) -> None:
    """Render the individual stock drill-down view for a curated screen:
    identity, the three risk scores, and the full narrative rationale."""
    tickers = sorted(filtered["ticker"].dropna().unique())
    if not tickers:
        st.info("No stocks match the current filters.")
        return

    selected_ticker = st.selectbox("Select a stock", options=tickers)
    row = filtered[filtered["ticker"] == selected_ticker].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", row["ticker"])
    col2.metric("Accounting & Disclosure",
                f"{row['score_accounting_and_disclosure']:.0f}"
                if pd.notna(row["score_accounting_and_disclosure"]) else "N/A")
    col3.metric("Fraud", f"{row['score_fraud']:.0f}" if pd.notna(row["score_fraud"]) else "N/A")
    col4.metric("Insider", f"{row['score_insider']:.0f}" if pd.notna(row["score_insider"]) else "N/A")

    st.markdown(
        f"**{row['name']}** · {row['sector']} · "
        f"Market Cap: ${row['market_cap']:,.0f}M"
    )

    st.divider()

    st.subheader("Rationale")
    st.write(row["rationale"] if pd.notna(row["rationale"]) else "No rationale available.")


# ---------------------------------------------------------------------------
# Unscored quant screens (e.g. Rising Short Interest) — a third display
# case: quant_composite in type, but with no factor model, so neither the
# curated view (no rationale/risk scores here) nor short_screen's factor
# chart/M-Score view applies. Just identity + the derived metrics, flat.
# ---------------------------------------------------------------------------


def render_unscored_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters for an unscored quant screen and return the
    filtered DataFrame. Market cap only — there's no sector/industry
    column and no composite score to filter on."""
    st.sidebar.header("Filters")

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    mcap_min = float(df["market_cap"].min())
    mcap_max = float(df["market_cap"].max())
    mcap_range = st.sidebar.slider(
        "Market Cap ($M)",
        min_value=mcap_min,
        max_value=mcap_max,
        value=(mcap_min, mcap_max),
        format="$%.0f",
    )

    filtered = df[
        (df["market_cap"] >= mcap_range[0]) & (df["market_cap"] <= mcap_range[1])
    ].copy()

    st.sidebar.divider()
    st.sidebar.metric("Stocks shown", len(filtered))

    return filtered


def render_unscored_table(filtered: pd.DataFrame) -> None:
    """Render the main table for an unscored quant screen, with export
    buttons. No M-Score highlighting, no factor columns — there aren't any."""
    available_cols = [c for c in UNSCORED_DISPLAY_COLUMNS if c in filtered.columns]
    display_df = filtered[available_cols].sort_values("ticker")

    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        xlsx_buffer = io.BytesIO()
        display_df.to_excel(xlsx_buffer, index=False, engine="openpyxl")
        st.download_button(
            label="Export to Excel",
            data=xlsx_buffer.getvalue(),
            file_name="ows_rising_short_interest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="Export to CSV",
            data=csv_data,
            file_name="ows_rising_short_interest.csv",
            mime="text/csv",
        )

    styled = display_df.style.format(UNSCORED_METRIC_FORMATS)

    column_config = {
        col: st.column_config.Column(label=UNSCORED_COLUMN_LABELS[col])
        for col in available_cols
        if col in UNSCORED_COLUMN_LABELS
    }

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config=column_config,
    )


def render_unscored_drill_down(filtered: pd.DataFrame) -> None:
    """Render the individual stock drill-down view for an unscored quant
    screen: identity plus a flat list of the 8 derived metrics. No factor
    chart (no factor model) and no rationale (not curated data)."""
    tickers = sorted(filtered["ticker"].dropna().unique())
    if not tickers:
        st.info("No stocks match the current filters.")
        return

    selected_ticker = st.selectbox("Select a stock", options=tickers)
    row = filtered[filtered["ticker"] == selected_ticker].iloc[0]

    col1, col2 = st.columns(2)
    col1.metric("Ticker", row["ticker"])
    col2.metric("Market Cap", f"${row['market_cap']:,.0f}M")

    st.markdown(f"**{row['name']}**")

    st.divider()

    st.subheader("Metrics")
    metric_rows = []
    for col in UNSCORED_DISPLAY_COLUMNS:
        if col in ("ticker", "name", "market_cap") or col not in row.index:
            continue
        val = row[col]
        if pd.notna(val) and col in UNSCORED_METRIC_FORMATS:
            value_str = UNSCORED_METRIC_FORMATS[col].format(val)
        elif pd.notna(val):
            value_str = f"{val:.4f}"
        else:
            value_str = "N/A"
        metric_rows.append({
            "Metric": UNSCORED_METRIC_DISPLAY_NAMES.get(col, col),
            "Value": value_str,
        })
    st.dataframe(
        pd.DataFrame(metric_rows),
        use_container_width=True,
        hide_index=True,
        height=min(len(metric_rows) * 40 + 40, 300),
    )


# ---------------------------------------------------------------------------
# Cross-screen overlap view (Phase 3d Part 1) — a separate top-level view,
# not a fourth per-screen render path. Selected above the existing screen
# selector (see main()'s guard clause), since this is cross-screen by
# nature and doesn't fit the "one screen's data" shape the three existing
# render paths assume.
# ---------------------------------------------------------------------------


def load_all_screen_identity_data(screens_df: pd.DataFrame) -> dict:
    """Load every screen's own identity-bearing table, keyed by screen_id.

    Reuses the three existing cached loaders (load_quant_data,
    load_unscored_quant_data, load_curated_data) rather than reading SQL
    directly — dispatch mirrors main()'s existing 3-way branch, but reads
    screen_type/has_scoring from screens_df instead of a hardcoded
    screen-type list, so it stays correct if a screen's type combination
    changes in config.yaml.

    Args:
        screens_df: The screens registry (screen_id, display_name,
            screen_type, has_scoring).

    Returns:
        screen_id -> that screen's loaded DataFrame. A screen whose table
        doesn't exist yet is simply omitted — compute_overlap and
        build_presence_matrix both tolerate a missing screen_data entry.
    """
    screen_data = {}
    for _, row in screens_df.iterrows():
        screen_id = row["screen_id"]
        if row["screen_type"] == "quant_composite" and row["has_scoring"]:
            df = load_quant_data(screen_id)
        elif row["screen_type"] == "quant_composite" and not row["has_scoring"]:
            df = load_unscored_quant_data(screen_id)
        elif row["screen_type"] == "curated":
            df = load_curated_data(screen_id)
        else:
            df = None
        if df is not None:
            screen_data[screen_id] = df
    return screen_data


def render_overlap_view(screens_df: pd.DataFrame) -> None:
    """Render the cross-screen overlap view: how many of the thematic/RSI
    screens each ticker sits on, which ones, and short_screen's composite
    score as context — not as a membership tick (see src/overlap.py's
    module docstring for why short_screen is treated differently).
    """
    display_names = dict(zip(screens_df["screen_id"], screens_df["display_name"]))
    universe_display_name = display_names.get(UNIVERSE_SCREEN_ID, UNIVERSE_SCREEN_ID)

    membership_df = load_screen_membership()
    if membership_df is None:
        st.error("No screen_membership data found. Run an ingest pipeline first.")
        return

    screen_data = load_all_screen_identity_data(screens_df)
    overlap_df = compute_overlap(membership_df, screens_df, screen_data)

    st.sidebar.header("Filters")

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.divider()

    include_zero = st.sidebar.checkbox(
        "Include short_screen-only names (on 0 thematic screens)",
        value=False,
        help="The only control that reveals tickers on zero thematic/RSI "
        "screens. The slider below governs only the 1-or-more band.",
    )

    # Ceiling is computed from the UNFILTERED overlap_df, once, before any
    # filter below is applied — otherwise an unrelated sector selection
    # would silently move the slider's own bound out from under the user.
    ceiling = screen_count_ceiling(overlap_df)
    if ceiling == 1:
        st.sidebar.caption(
            "Every thematic/RSI-screen ticker appears on exactly 1 screen — "
            "no minimum-count slider to show."
        )
        min_screen_count = 1
    else:
        min_screen_count = st.sidebar.slider(
            "Minimum screen count", min_value=1, max_value=ceiling, value=1
        )

    count_mask = overlap_df["screen_count"] >= min_screen_count
    if include_zero:
        count_mask = count_mask | (overlap_df["screen_count"] == 0)
    filtered = overlap_df[count_mask]

    all_sectors = sorted(overlap_df["sector"].dropna().unique())
    selected_sectors = st.sidebar.multiselect("Sector", options=all_sectors)
    if selected_sectors:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]

    st.sidebar.divider()
    st.sidebar.metric("Tickers shown", len(filtered))

    filtered = filtered.sort_values(
        ["screen_count", "overall_score"], ascending=[False, False]
    )

    tab_overlap, tab_matrix = st.tabs(["Overlap Table", "Per-Screen Presence Matrix"])

    with tab_overlap:
        display_cols = [
            "ticker", "name", "sector", "market_cap",
            "screen_count", "screens_on", "overall_score",
        ]
        display_df = filtered[display_cols]

        # Export is always a fixed superset of the on-screen columns — same
        # house pattern as render_main_table's export (24 factor metrics +
        # 20 diff inputs regardless of a display-only checkbox). in_universe
        # is included unconditionally so the 10 not-in-universe rows are
        # distinguishable in the exported file via a native boolean column,
        # not via a blank cell or a label baked into the numeric score
        # column (which would break Excel's ability to sort it).
        export_cols = display_cols + ["in_universe"]
        export_df = filtered[export_cols]

        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            xlsx_buffer = io.BytesIO()
            export_df.to_excel(xlsx_buffer, index=False, engine="openpyxl")
            st.download_button(
                label="Export to Excel",
                data=xlsx_buffer.getvalue(),
                file_name="ows_overlap.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col2:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="Export to CSV",
                data=csv_data,
                file_name="ows_overlap.csv",
                mime="text/csv",
            )

        styled = style_overlap_table(display_df)

        column_config = {
            col: st.column_config.Column(label=OVERLAP_COLUMN_LABELS[col])
            for col in display_cols
            if col in OVERLAP_COLUMN_LABELS
        }
        column_config["overall_score"] = st.column_config.Column(
            label=f"{universe_display_name} Composite Score"
        )

        st.dataframe(
            styled,
            use_container_width=True,
            height=600,
            hide_index=True,
            column_config=column_config,
        )

    with tab_matrix:
        matrix_df = build_presence_matrix(membership_df, screens_df, overlap_df)
        matrix_filtered = matrix_df[matrix_df["ticker"].isin(filtered["ticker"])]

        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            xlsx_buffer = io.BytesIO()
            matrix_filtered.to_excel(xlsx_buffer, index=False, engine="openpyxl")
            st.download_button(
                label="Export to Excel",
                data=xlsx_buffer.getvalue(),
                file_name="ows_overlap_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col2:
            csv_data = matrix_filtered.to_csv(index=False)
            st.download_button(
                label="Export to CSV",
                data=csv_data,
                file_name="ows_overlap_matrix.csv",
                mime="text/csv",
            )

        st.dataframe(matrix_filtered, use_container_width=True, height=600, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    screens_df = list_screens()
    if screens_df is None:
        st.title("OWS Short Screen")
        st.error(
            "No data found. Run the pipeline first:\n\n"
            "`python src/ingest.py && python src/transform.py && python src/score.py`"
        )
        return

    view = st.sidebar.radio("View", ["Screen View", "Cross-Screen Overlap"], index=0)
    if view == "Cross-Screen Overlap":
        render_overlap_view(screens_df)
        return

    screen_ids = list(screens_df["screen_id"])
    display_names = dict(zip(screens_df["screen_id"], screens_df["display_name"]))
    screen_types = dict(zip(screens_df["screen_id"], screens_df["screen_type"]))
    has_scoring_by_id = dict(zip(screens_df["screen_id"], screens_df["has_scoring"]))
    default_index = screen_ids.index("short_screen") if "short_screen" in screen_ids else 0

    selected_screen_id = st.sidebar.selectbox(
        "Screen",
        options=screen_ids,
        index=default_index,
        format_func=lambda sid: display_names.get(sid, sid),
    )
    st.sidebar.divider()

    st.title(display_names[selected_screen_id])

    screen_type = screen_types[selected_screen_id]

    if screen_type == "quant_composite" and has_scoring_by_id[selected_screen_id]:
        df = load_quant_data(selected_screen_id)
        if df is None:
            st.error(
                f"No scored data found for {display_names[selected_screen_id]}. "
                "Run the pipeline first."
            )
            return
        filtered = render_sidebar(df)
        tab_screener, tab_drilldown = st.tabs(["Screener", "Stock Drill-Down"])
        with tab_screener:
            render_main_table(filtered, df)
        with tab_drilldown:
            render_drill_down(filtered)
    elif screen_type == "quant_composite" and not has_scoring_by_id[selected_screen_id]:
        df = load_unscored_quant_data(selected_screen_id)
        if df is None:
            st.error(
                f"No transformed data found for {display_names[selected_screen_id]}. "
                "Run ingest + transform first."
            )
            return
        filtered = render_unscored_sidebar(df)
        tab_screener, tab_drilldown = st.tabs(["Screener", "Stock Drill-Down"])
        with tab_screener:
            render_unscored_table(filtered)
        with tab_drilldown:
            render_unscored_drill_down(filtered)
    elif screen_type == "curated":
        df = load_curated_data(selected_screen_id)
        if df is None:
            st.error(
                f"No curated data found for {display_names[selected_screen_id]}. "
                "Run the curated ingest pipeline first."
            )
            return
        filtered = render_curated_sidebar(df)
        tab_screener, tab_drilldown = st.tabs(["Screener", "Stock Drill-Down"])
        with tab_screener:
            render_curated_table(filtered)
        with tab_drilldown:
            render_curated_drill_down(filtered)
    else:
        st.error(f"Unknown screen type {screen_type!r} for screen {selected_screen_id!r}.")


if __name__ == "__main__":
    main()
