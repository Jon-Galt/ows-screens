"""
OWS Screens — Streamlit Web UI.

A sidebar screen selector reads the screens registry and branches into one
of three rendering paths per screen: scored quant_composite screens (e.g.
short_screen — factor chart, M-Score, sector/industry filters), curated
screens (narrative rationale + three risk scores, no factor model), and
unscored quant_composite screens (e.g. Rising Short Interest — has a
transform stage but no factor model yet, so no chart/M-Score either). All
three get a filterable/sortable main table, a stock drill-down (Phase 5b-2
adds a cross-screen "Also Appears On" section to all three), and Excel/CSV
export. The cross-screen overlap table (Phase 3d Part 1) is rendered once
at the bottom of every screen's page, in a collapsed expander — not a
separate top-level view.
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

from src.cross_screen_context import (
    build_also_appears_on,
    classify_screen,
    other_screen_ids_for_ticker,
)
from src.config import load_config
from src.db import table_name
from src.overlap import (
    UNIVERSE_SCREEN_ID,
    apply_zero_thematic_label,
    compute_overlap,
    resolve_overlap_click_target,
    screen_count_ceiling,
    style_overlap_table,
    zero_thematic_summary,
)
from src.score import FACTOR_DEFINITIONS, get_screen_config
from src.selection import (
    find_ticker_row,
    is_fresh_selection,
    resolve_nav_target,
    resolve_selected_cell,
    resolve_selected_ticker,
    should_process_cell_selection,
)
from src.styling import bold_ticker_column, build_color_scale_domain, style_scored_table

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

# Phase 5b-2 (R8): the single source of truth for the app's font family
# outside .streamlit/config.toml (which streamlit reads directly and this
# process never parses). Read by the Altair drill-down chart's own
# .configure_* calls, since Vega-Lite draws its own axis/legend/title text
# and doesn't inherit the theme. tests/test_app.py locks this literal
# against the actual config.toml file's `font` line, so the two can't
# silently drift apart.
APP_FONT_FAMILY = "Arial, Helvetica, sans-serif"

# Phase 5c-2 (R1, as amended by the Driver 2026-09-05): two distinct mark
# variants, not interchangeable. TITLE_MARK_PATH (white disc, green bear)
# sits beside the screen title on the white page; LOGO_MARK_PATH (green
# disc, white bear) is rendered at the top of the sidebar, on the sidebar's
# secondaryBackgroundColor ground. Both derive from ows-logo-on-green.pdf.
TITLE_MARK_PATH = os.path.join(_PROJECT_ROOT, "assets", "ows-bear-white-disc.png")
LOGO_MARK_PATH = os.path.join(_PROJECT_ROOT, "assets", "ows-bear-green-disc.png")

# Phase 5c-2: the title mark is sized to match st.title's own rendered h1
# font-size, measured in the browser against the installed streamlit theme
# (44px at the theme's default 2.75rem h1 size / 16px base). Re-measure if
# the theme's font sizing ever changes — this is not derived from config.
TITLE_MARK_SIZE_PX = 44

# The sidebar mark's width, ~4x an st.logo "large" mark (32px is st.logo's
# hard cap in this streamlit version — there is no argument that gets it
# further, which is why this is a plain st.sidebar.image call instead).
SIDEBAR_MARK_WIDTH_PX = 128

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

# Phase 5c-3: shared between the curated grid header (below) and the
# cross-screen drill-down's curated branch (render_cross_screen_context) so
# the two user-visible sites cannot drift independently.
_STOCK_PERFORMANCE_LABEL = "Stock Performance (1 yr.)"

# render_curated_table's column_config map.
CURATED_COLUMN_LABELS = {
    "ticker": "Ticker",
    "name": "Name",
    "sector": "Sector",
    "market_cap": "Market Cap ($M)",
    "daily_traded_value": "Daily Traded Value ($M)",
    "stock_performance": _STOCK_PERFORMANCE_LABEL,
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

# The overlap section's on-screen columns (Phase 3d Part 1; relocated into a
# bottom expander in Phase 5b-2 — see render_overlap_section). Hoisted to a
# module-level constant (rather than a function-local list) so
# tests/test_app.py's label-completeness tests import the real list instead
# of maintaining a hand-copied mirror that could silently drift from it.
OVERLAP_DISPLAY_COLUMNS = [
    "ticker", "name", "sector", "market_cap",
    "screen_count", "screens_on", "overall_score",
]

# render_overlap_section's column_config map for every OVERLAP_DISPLAY_COLUMNS
# entry except overall_score, which keeps its own existing
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

# ---------------------------------------------------------------------------
# Phase 5b-3 (R7): column-header help + click-a-cell derivation.
#
# Factor weights and NaN defaults are read from config.yaml at module import
# time, never hardcoded (Architecture Rules 7 & 9) — this app.py has not
# needed a config.yaml read before now (every other display-layer constant
# above is derived from FACTOR_DEFINITIONS/config-independent data), so this
# is a new, deliberate, read-only dependency. No try/except: a missing file
# or a missing short_screen block must fail app startup loudly rather than
# silently default a weight that would then appear, wrong, in a tooltip that
# looks authoritative (Worker Rule 1).
_SHORT_SCREEN_CONFIG = get_screen_config(load_config(), "short_screen")
_FACTOR_WEIGHTS = _SHORT_SCREEN_CONFIG["factor_weights"]
_SCORING_CFG = _SHORT_SCREEN_CONFIG["scoring"]
_NAN_DEFAULT_STANDARD = _SCORING_CFG["nan_default_standard"]
_NAN_DEFAULT_ZERO_FACTORS = set(_SCORING_CFG["nan_default_zero_factors"])

# Reverse of FACTOR_CATEGORIES, built once — which category a factor belongs
# to, for the header-help spine sentence's "Weight {w} within {category}".
_CATEGORY_BY_FACTOR = {
    factor: category for category, factors in FACTOR_CATEGORIES.items() for factor in factors
}

# Each of the 10 diff-based factors' arithmetic, in the ORDER THE ARITHMETIC
# RUNS — not the order DIFF_FACTOR_INPUTS lists them in (that order follows
# the Excel template's block layout, which for three factors is the
# opposite of what the calc function actually subtracts — confirmed by
# reading calc_fcf_yield_diff/calc_growth_decel/calc_growth_accel's bodies
# in transform.py directly, not inferred from DIFF_FACTOR_INPUTS' order).
# operation is "ratio_minus_one" (col_a / col_b - 1) or "difference"
# (col_a - col_b).
DIFF_FACTOR_FORMULAS = {
    "abs_ps_factor": ("ratio_minus_one", "ps_ntm", "ps_3yr_avg"),
    "abs_fcf_factor": ("difference", "fcf_yield_3yr_avg", "fcf_yield"),
    "decel_factor": ("difference", "yoy_growth_ttm", "yoy_growth_ttm_t1"),
    "accel_factor": ("difference", "rev_cagr_f2y", "rev_cagr_p2y"),
    "gm_factor": ("difference", "ntm_gross_margin", "gross_margin_3yr_avg"),
    "ebit_factor": ("difference", "ntm_ebit_margin", "ebit_margin_3yr_avg"),
    "dso_factor": ("ratio_minus_one", "dsos", "dsos_py"),
    "dio_factor": ("ratio_minus_one", "dios", "dios_t1"),
    "dpo_factor": ("ratio_minus_one", "dpos", "dpos_t1"),
    "def_rev_factor": ("ratio_minus_one", "days_deferred_rev", "days_deferred_rev_t1"),
}


def format_diff_formula(factor: str) -> str:
    """Render a diff-based factor's actual arithmetic as a short label,
    e.g. "P/Sales (NTM) ÷ P/Sales (3yr. Avg.) − 1" or
    "FCF Yield (3yr. Avg.) − FCF Yield (LTM)".

    Reuses DIFF_FACTOR_INPUTS' own operand labels (via _label_from_diff_
    inputs) rather than retyping them, so the derivation panel's "Diff."
    line and the generated metric-column tooltip (METRIC_COLUMN_HELP) can
    never state two different formulas for the same factor.

    Args:
        factor: A key of DIFF_FACTOR_FORMULAS.

    Returns:
        The formula as a single-line string.
    """
    operation, col_a, col_b = DIFF_FACTOR_FORMULAS[factor]
    label_a = _label_from_diff_inputs(col_a)
    label_b = _label_from_diff_inputs(col_b)
    if operation == "ratio_minus_one":
        return f"{label_a} ÷ {label_b} − 1"
    return f"{label_a} − {label_b}"


# The click-a-cell derivation panel's dispatch table: which factor a clicked
# column's derivation belongs to. Built from DIFF_FACTOR_INPUTS/
# FACTOR_DEFINITIONS rather than retyped, so it can't drift from either.
# 20 entries: each of the 10 diff factor SCORE columns maps to itself, and
# each of their 10 own METRIC columns maps to the same factor. Deliberately
# excludes ps_ntm and fcf_yield, which are rel_ps_factor's/rel_fcf_factor's
# own metric columns (rel_ps_factor/rel_fcf_factor are not diff-based) —
# mapping them here would show a user Absolute P/S's derivation when they
# click Relative P/S's metric, a real, plausible, wrong panel with no error.
# They are excluded by construction (rel_ps_factor/rel_fcf_factor are not
# keys of DIFF_FACTOR_INPUTS), not by special-casing — see
# tests/test_app.py's negative test locking this.
CELL_DERIVATION_FACTORS = {factor: factor for factor in DIFF_FACTOR_INPUTS}
CELL_DERIVATION_FACTORS.update(
    {FACTOR_DEFINITIONS[factor]["metric"]: factor for factor in DIFF_FACTOR_INPUTS}
)

# The 14 non-diff factors' own score/metric columns — a click here gets the
# generated "no step-by-step derivation available" one-liner (§3.4 case 2)
# rather than silence, built from the same FACTOR_DEFINITIONS/label data as
# the header-help feature (no 14 new authored strings). Built the same way
# as CELL_DERIVATION_FACTORS, so the two dispatch tables are provably
# disjoint by construction rather than by hand-checking two lists.
_NON_DIFF_FACTORS = [f for f in FACTOR_DEFINITIONS if f not in DIFF_FACTOR_INPUTS]
NON_DIFF_FACTOR_BY_COLUMN = {factor: factor for factor in _NON_DIFF_FACTORS}
NON_DIFF_FACTOR_BY_COLUMN.update(
    {FACTOR_DEFINITIONS[factor]["metric"]: factor for factor in _NON_DIFF_FACTORS}
)

# ---------------------------------------------------------------------------
# Phase 5b-3: column-header help strings — a help= tooltip for every
# displayed column of all four tables. Four separate maps, parallel to the
# four label maps above, per the Phase 5b-2 finding that a column name does
# not mean the same thing on every screen (e.g. curated vs short_screen
# "name"): a single shared column-keyed dict would silently collapse
# overall_score's two genuinely different meanings (this screen's own
# composite vs. short_screen's composite shown as cross-screen context) and
# short_interest_pct's two genuinely different sources (RSI's own export
# column vs. short_screen's factor metric). Only truly identical text
# (ticker/name/sector/market_cap) is factored into a shared constant below.
# ---------------------------------------------------------------------------

_TICKER_HELP = "Bloomberg ticker, normalised. The join key across every screen and every historical table."
_NAME_HELP = (
    "Company name as the source export spells it. The curated exports and the short_screen "
    "export disagree on capitalisation for the same company."
)
_SECTOR_HELP = "Sector as supplied by the source export."
_MARKET_CAP_HELP = "Market capitalisation in $M. Stored in $M everywhere in this project."
_INDUSTRY_HELP = "Industry as supplied by the source export."

_OVERALL_SCORE_HELP_MAIN = (
    "The composite: the sum of 24 weighted factor scores. Each of the 7 categories' weights "
    "sum to 1.0, so each contributes at most 1.0 and the maximum possible score is 7.0. The "
    "M-Score is deliberately excluded."
)
_MSCORE_FLAG_HELP = (
    "Beneish M-Score above the manipulation threshold (−2.22). A standalone flag — never "
    "part of the composite score."
)

# The 24 authored per-factor clauses (§4.2). Five of these were originally
# flagged [UNVERIFIED] because their metric column has no calc function in
# transform.py to verify against (raw Bloomberg passthrough) — resolved by
# attributing the definition to its source instead of asserting it (Phase
# 5b-3 plan review round 1, ruling (a)), which needs no verification and
# carries no marker.
FACTOR_HELP_CLAUSES = {
    "abs_ps_factor": (
        "Trading at a higher P/S multiple than its own 3-year average — priced above its own "
        "history."
    ),
    "rel_ps_factor": (
        "The stock's own P/Sales (NTM), ranked against the rest of the universe rather than "
        "against its own history. P/Sales (NTM) comes from the Bloomberg export."
    ),
    "abs_fcf_factor": (
        "FCF yield has fallen below the stock's own 3-year average — more expensive on a "
        "cash-flow basis than it used to be."
    ),
    "rel_fcf_factor": (
        "The stock's own FCF Yield (LTM), ranked against the rest of the universe rather than "
        "against its own history. FCF Yield (LTM) comes from the Bloomberg export."
    ),
    "decel_factor": "Revenue growth is slower this TTM than last.",
    "accel_factor": (
        "Consensus expects forward revenue growth to exceed the trailing two years. Elevated "
        "expectations create asymmetric downside when they are missed."
    ),
    "gm_factor": (
        "Consensus expects gross margin to expand versus the 3-year average — priced for "
        "perfection."
    ),
    "ebit_factor": (
        "Consensus expects EBIT margin to expand versus the 3-year average — priced for "
        "perfection."
    ),
    "debt_ebitda_factor": "Net debt against adjusted EBITDA.",
    "debt_sales_factor": "Debt against TTM sales.",
    "debt_ev_factor": "Debt as a share of enterprise value.",
    "refi_risk_factor": (
        "Weighted average debt maturity, as supplied by the Bloomberg export. Shorter "
        "maturities mean refinancing has to happen sooner."
    ),
    "liquidity_risk_factor": "Years of remaining liquidity at the current cash burn.",
    "fcf_conv_factor": (
        "Free cash flow as a share of reported earnings — weaker conversion means lower cash "
        "quality."
    ),
    "accrual_factor": (
        "The gap between cash from operations and net income. A large divergence in either "
        "direction says reported earnings are driven by non-cash items."
    ),
    "dso_factor": (
        "Days sales outstanding against the same quarter a year ago. Rising DSOs mean "
        "receivables are outpacing revenue."
    ),
    "dio_factor": (
        "Days inventory outstanding against the prior quarter. Rising DIOs mean inventory is "
        "building against COGS."
    ),
    "dpo_factor": (
        "Days payable outstanding against the prior quarter. Falling DPOs mean the company is "
        "paying suppliers faster, which consumes cash."
    ),
    "def_rev_factor": (
        "Days of deferred revenue against the prior quarter. Falling deferred revenue means "
        "fewer customer prepayments."
    ),
    "dilution_factor": "Dilution over the past three years, as supplied by the Bloomberg export.",
    "ebit_adj_factor": (
        "Non-GAAP EBIT as a multiple of GAAP EBIT, as supplied by the Bloomberg export — a "
        "higher ratio means more aggressive adjustments."
    ),
    "eps_adj_factor": (
        "Adjusted EPS as a multiple of GAAP EPS — a higher ratio means more aggressive "
        "adjustments."
    ),
    "short_int_factor": (
        "Short interest as reported in the Bloomberg export. It measures how crowded the trade "
        "already is as much as it measures opportunity."
    ),
    "ratings_factor": "The share of sell-side ratings that are Hold or Sell.",
}


def _factor_help_spine(factor: str) -> str:
    """The mechanical half of a factor's header-help text (§4.1) — never
    typed by hand, so it can't drift from FACTOR_DEFINITIONS/config.yaml.

    For a diff-based factor, the "metric" clause is the actual formula
    (via format_diff_formula) rather than METRIC_COLUMN_LABELS' on-screen
    column header (which reads "<label> — Diff." — fine as a compact table
    header, but not as a sentence naming what's being ranked).

    Args:
        factor: A key of FACTOR_DEFINITIONS.

    Returns:
        "Percentile rank of {metric}...; {direction}. Weight {w} within
        {category}. Missing data scores {default}." — see FACTOR_HELP.
    """
    defn = FACTOR_DEFINITIONS[factor]
    if factor in DIFF_FACTOR_INPUTS:
        metric_label = f"({format_diff_formula(factor)})"
    else:
        metric_label = METRIC_COLUMN_LABELS[defn["metric"]]
    direction_clause = (
        "higher values rank higher (more bearish)"
        if defn["direction"] == "straight"
        else "lower values rank higher (more bearish)"
    )
    # Weight formatted to 3 d.p. for consistency with the NaN-default
    # formatting below. The seven Cash Flow factors' weights are exactly
    # 1/7 (config.yaml stores 0.142857) and so display as 0.143 each,
    # visibly "summing" to 1.001 — that is display rounding of an exact
    # 1/7, not a config error; do not "fix" config.yaml's weights in
    # response to it.
    weight = _FACTOR_WEIGHTS[factor]
    category = _CATEGORY_BY_FACTOR[factor]
    nan_default = 0.0 if factor in _NAN_DEFAULT_ZERO_FACTORS else _NAN_DEFAULT_STANDARD
    return (
        f"Percentile rank of {metric_label} across the screen's universe; {direction_clause}. "
        f"Weight {weight:.3f} within {category}. Missing data scores {nan_default:.3f}."
    )


# Every factor's full header-help text: the authored clause (§4.2) plus the
# generated spine (§4.1).
FACTOR_HELP = {
    factor: f"{FACTOR_HELP_CLAUSES[factor]} {_factor_help_spine(factor)}"
    for factor in FACTOR_DEFINITIONS
}


def _metric_column_help(factor: str) -> str:
    """Generated header-help text for a factor's own underlying metric
    column (§4.4) — never authored by hand.

    Args:
        factor: A key of FACTOR_DEFINITIONS.

    Returns:
        "The raw value {Factor label} ranks — {metric label}." with the
        actual formula appended for the 10 diff-based factors, from the
        same DIFF_FACTOR_FORMULAS declaration format_diff_formula reads —
        so this tooltip and the derivation panel's "Diff." line can never
        state two different formulas for the same factor.
    """
    metric_col = FACTOR_DEFINITIONS[factor]["metric"]
    factor_label = FACTOR_COLUMN_LABELS[factor]
    if factor in DIFF_FACTOR_INPUTS:
        return f"The raw value {factor_label} ranks — the diff: {format_diff_formula(factor)}."
    return f"The raw value {factor_label} ranks — {METRIC_COLUMN_LABELS[metric_col]}."


METRIC_COLUMN_HELP = {
    FACTOR_DEFINITIONS[factor]["metric"]: _metric_column_help(factor)
    for factor in FACTOR_DEFINITIONS
}

# Non-factor column help, curated/unscored/overlap-specific columns (§4.3).
_DAILY_TRADED_VALUE_HELP = "Average daily traded value in $M, from the Canary export."
_STOCK_PERFORMANCE_HELP = (
    "One-year stock performance as supplied by the Canary export, where the source workbook "
    "heads the column \"Stock Perf (1 yr.)\". This is the one identity-ish field that "
    "genuinely differs between two curated screens for the same ticker."
)
_VALUATION_EV_REVENUE_HELP = "EV/Revenue (NTM) percentile, from the Canary export."
_SCORE_ACCOUNTING_HELP = (
    "Canary's Accounting & Disclosure risk score, parsed from the packed scores field in the "
    "export. Not computed here."
)
_SCORE_FRAUD_HELP = (
    "Canary's Fraud risk score, parsed from the packed scores field in the export. Not computed "
    "here."
)
_SCORE_INSIDER_HELP = (
    "Canary's Insider risk score, parsed from the packed scores field in the export. Not "
    "computed here."
)
_ADV_HELP = "Average daily value traded, $M."
_SHORT_INTEREST_PCT_RSI_HELP = "Short interest percentage from the Bloomberg short-interest export."
_SI_CHANGE_3M_HELP = "Change in short interest against the 3-month lookback."
_SI_CHANGE_6M_HELP = "Change in short interest against the 6-month lookback."
_WEEK_52_HIGH_CHG_HELP = "Change from the 52-week high."
_EV_SALES_HELP = "Enterprise value to sales."
_DEBT_EBITDA_RSI_HELP = "Net debt to EBITDA."
_SCREEN_COUNT_HELP = (
    "How many thematic screens carry this ticker. short_screen is context, not a membership "
    "tick, so it never counts here — the ceiling is 5, not 6."
)
_SCREENS_ON_HELP = (
    "Which thematic screens carry this ticker. 1,171 of the 1,358 in-universe tickers are on "
    "none."
)
_OVERLAP_OVERALL_SCORE_HELP = (
    "This ticker's short_screen composite, shown as context. Blank means the ticker is not in "
    "short_screen's universe at all — 17 of the 1,375 are thematic-only."
)

# The four tables' complete help maps — parallel to MAIN_TABLE_COLUMN_LABELS/
# CURATED_COLUMN_LABELS/UNSCORED_COLUMN_LABELS/OVERLAP_COLUMN_LABELS above.
MAIN_TABLE_COLUMN_HELP = {
    "ticker": _TICKER_HELP,
    "name": _NAME_HELP,
    "sector": _SECTOR_HELP,
    "industry": _INDUSTRY_HELP,
    "market_cap": _MARKET_CAP_HELP,
    "overall_score": _OVERALL_SCORE_HELP_MAIN,
    "mscore_flag": _MSCORE_FLAG_HELP,
    **FACTOR_HELP,
    **METRIC_COLUMN_HELP,
}

CURATED_COLUMN_HELP = {
    "ticker": _TICKER_HELP,
    "name": _NAME_HELP,
    "sector": _SECTOR_HELP,
    "market_cap": _MARKET_CAP_HELP,
    "daily_traded_value": _DAILY_TRADED_VALUE_HELP,
    "stock_performance": _STOCK_PERFORMANCE_HELP,
    "valuation_ev_revenue_ntm_percentile": _VALUATION_EV_REVENUE_HELP,
    "score_accounting_and_disclosure": _SCORE_ACCOUNTING_HELP,
    "score_fraud": _SCORE_FRAUD_HELP,
    "score_insider": _SCORE_INSIDER_HELP,
}

UNSCORED_COLUMN_HELP = {
    "ticker": _TICKER_HELP,
    "name": _NAME_HELP,
    "market_cap": _MARKET_CAP_HELP,
    "adv": _ADV_HELP,
    "short_interest_pct": _SHORT_INTEREST_PCT_RSI_HELP,
    "si_change_3m": _SI_CHANGE_3M_HELP,
    "si_change_6m": _SI_CHANGE_6M_HELP,
    "week_52_high_chg": _WEEK_52_HIGH_CHG_HELP,
    "ev_sales": _EV_SALES_HELP,
    "debt_ebitda": _DEBT_EBITDA_RSI_HELP,
}

# Includes overall_score directly (unlike OVERLAP_COLUMN_LABELS, which
# excludes it because its LABEL is built dynamically in render_overlap_
# section — its help text has no such dynamic component and is static).
OVERLAP_COLUMN_HELP = {
    "ticker": _TICKER_HELP,
    "name": _NAME_HELP,
    "sector": _SECTOR_HELP,
    "market_cap": _MARKET_CAP_HELP,
    "screen_count": _SCREEN_COUNT_HELP,
    "screens_on": _SCREENS_ON_HELP,
    "overall_score": _OVERLAP_OVERALL_SCORE_HELP,
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

    # Sector filter
    all_sectors = sorted(df["sector"].dropna().unique())
    selected_sectors = st.sidebar.multiselect("**Sector**", options=all_sectors)

    # Industry filter — dependent on sector selection
    if selected_sectors:
        available_industries = sorted(
            df[df["sector"].isin(selected_sectors)]["industry"].dropna().unique()
        )
    else:
        available_industries = sorted(df["industry"].dropna().unique())
    selected_industries = st.sidebar.multiselect("**Industry**", options=available_industries)

    # Market cap slider
    mcap_min = float(df["market_cap"].min())
    mcap_max = float(df["market_cap"].max())
    mcap_range = st.sidebar.slider(
        "**Market Cap ($M)**",
        min_value=mcap_min,
        max_value=mcap_max,
        value=(mcap_min, mcap_max),
        format="$%,.0f",
    )

    # Overall score slider
    score_min = float(df["overall_score"].min())
    score_max = float(df["overall_score"].max())
    score_range = st.sidebar.slider(
        "**Overall Score**",
        min_value=0.0,
        max_value=7.0,
        value=(score_min, score_max),
        step=0.1,
    )

    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

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
# Phase 5b-1: inline drill-down selection wiring
# ---------------------------------------------------------------------------
#
# The main table's row selection and the drill-down's selectbox are one
# piece of state (ticker_key), not two. Streamlit's own row-selection state
# is sticky across reruns (a click stays selected until a different click
# changes it) and, critically, is stored as a RAW POSITIONAL INDEX — if the
# underlying frame reshapes (a sidebar filter) without a fresh click, that
# stale index gets silently reinterpreted against the new frame at the same
# position, highlighting a different stock with no error. Confirmed against
# the installed streamlit 1.63.0 with a throwaway probe (see
# PHASE5B1_PROMPT.md's plan-revision history) before this was written.
#
# sync_drilldown_selection runs inside each render_*_table function, after
# display_df is built but BEFORE st.dataframe(key=table_key, ...) is
# instantiated — the only ordering under which writing
# st.session_state[table_key] is legal (a widget's session_state key can be
# set any time before that widget is instantiated in the same run, never
# after). It:
#   1. Treats the table's row-selection payload as meaningful only on the
#      rerun where it actually just changed (compared against last_rows_key,
#      a plain bookkeeping entry this function owns) — otherwise a stale
#      sticky selection would override a manual selectbox change on every
#      subsequent rerun.
#   2. Resolves the authoritative ticker via the pure resolve_selected_ticker.
#   3. Re-seeds st.session_state[table_key] so the table's own highlight is
#      repainted onto that ticker's CURRENT position — this is what keeps
#      the highlight from diverging from the drill-down after a filter
#      reshapes/reorders the frame, and (since a click on an already-
#      selected row toggles it OFF, observed directly against 1.63.0) is
#      also what makes re-clicking a previously-clicked-then-navigated-away
#      row work instead of silently deselecting.
#
# One accepted consequence of holding both invariants this way: clicking
# the currently-highlighted row deselects it, resolve_selected_ticker then
# falls through to previous_ticker, and this function immediately re-paints
# that same row — so a click on the highlighted row is visually inert. The
# drill-down always shows a stock and the highlight always matches it; there
# is no state that means "nothing selected." This is intentional, not a bug.


def sync_drilldown_selection(
    display_df: pd.DataFrame, table_key: str, ticker_key: str, last_rows_key: str
) -> None:
    """Resolve the drill-down ticker from the table's selection state and
    re-seed the table's own highlight to match it. Must be called after
    display_df is built and before st.dataframe(key=table_key, ...) — see
    module-level comment above for why.

    Args:
        display_df: The frame about to be passed to st.dataframe.
        table_key: The main table's st.dataframe key (its selection state
            lives at st.session_state[table_key]["selection"]["rows"]).
        ticker_key: The drill-down selectbox's key — also the authoritative
            "currently shown" ticker, read/written here before that
            selectbox is instantiated.
        last_rows_key: A plain (non-widget) session_state entry this
            function owns, used to detect a fresh row click vs. a sticky,
            unrelated rerun.
    """
    pre_rows = st.session_state.get(table_key, {}).get("selection", {}).get("rows", [])
    if is_fresh_selection(pre_rows, st.session_state.get(last_rows_key)):
        selected_rows = pre_rows
    else:
        selected_rows = []

    resolved = resolve_selected_ticker(
        display_df, selected_rows, previous_ticker=st.session_state.get(ticker_key)
    )
    if resolved is not None:
        st.session_state[ticker_key] = resolved

    target_idx = find_ticker_row(display_df, resolved)
    st.session_state[table_key] = {
        "selection": {
            "rows": [target_idx] if target_idx is not None else [],
            "columns": [],
            "cells": [],
        }
    }
    st.session_state[last_rows_key] = [target_idx] if target_idx is not None else []


# ---------------------------------------------------------------------------
# Phase 5b-2: cross-screen click-through navigation.
#
# A click on the overlap table's row (render_overlap_section) sets
# st.session_state["_pending_nav"] = (target_screen_id, ticker) and reruns.
# At the very top of main(), before the Screen selectbox (key="screen_
# selector") is instantiated, that pending marker is consumed and turned
# into two writes: forcing screen_selector to target_screen_id (the only
# legal time to set a widget's session_state key is before that widget is
# created in the same run), and stashing (target_screen_id, ticker) as
# "_nav_target" for apply_pending_nav below to consume once the target
# screen's own sidebar filters have actually been applied.
#
# _nav_target carries its own screen_id (not just the ticker) and is popped
# unconditionally by apply_pending_nav, regardless of whether it matches the
# currently-rendering screen. This removes a dependence on an ordering
# guarantee rather than merely documenting one: main() has early returns
# (the screens_df-is-None guard, and a df-is-None guard per branch) between
# where screen_selector is forced and where apply_pending_nav actually runs.
# If one of those early returns fires, apply_pending_nav for the *intended*
# screen never gets a chance to pop "_nav_target" this rerun, and it survives
# untouched for a later rerun once that screen's data actually loads. If,
# on the other hand, apply_pending_nav DOES run but for a mismatched screen
# (shouldn't happen given screen_selector is forced in the same write, but
# not assumed), the stale marker is discarded rather than held to misfire
# on some later, unrelated rerun.
def apply_pending_nav(filtered: pd.DataFrame, ticker_key: str, screen_id: str) -> None:
    """Consume a pending cross-screen navigation targeting `screen_id`, if
    one exists, seeding the drill-down's ticker_key only when the target
    ticker actually survives this screen's active filters.

    Must run after `filtered` (this screen's sidebar-filtered frame) is
    built and before sync_drilldown_selection reads ticker_key as
    previous_ticker — same ordering discipline sync_drilldown_selection's
    own module comment documents.

    This is the fix for the click-through's one failure mode: without this
    gate, a navigated-to ticker excluded by the destination's own filters
    would fail resolve_selected_ticker's precedence-2 branch and silently
    fall through to precedence 3 (the first ticker in display order) — a
    real, different, plausible company, with no error. resolve_nav_target
    is checked here, before ticker_key is ever touched, so that never
    happens as a consequence of navigation.

    Args:
        filtered: The destination screen's sidebar-filtered frame.
        ticker_key: The drill-down selectbox's key for this screen.
        screen_id: The screen currently being rendered (selected_screen_id).
    """
    pending = st.session_state.pop("_nav_target", None)
    if pending is None:
        return
    target_screen_id, ticker = pending
    if target_screen_id != screen_id:
        return
    outcome, _ = resolve_nav_target(filtered, ticker)
    if outcome == "show":
        st.session_state[ticker_key] = ticker
    else:
        st.warning(
            f"{ticker} is outside the current filters on this screen. "
            "Clear filters to view it."
        )


def select_drilldown_row(filtered: pd.DataFrame, ticker_key: str) -> pd.Series | None:
    """Shared opening of every drill-down function: the ticker selectbox
    and its row lookup. Returns None (caller shows the "no stocks match"
    message) if there are no tickers to choose from.

    The selectbox's own dropdown order is alphabetical regardless of the
    main table's display sort — unrelated to which row is highlighted in
    the table, and not specified otherwise.

    Args:
        filtered: The sidebar-filtered DataFrame (not display_df — this is
            every stock a user could choose from, independent of table
            sort order).
        ticker_key: The selectbox's key, shared with sync_drilldown_selection
            so a row click and a manual selectbox change are one state.

    Returns:
        The selected stock's row (a pd.Series), or None if filtered has no
        non-null tickers.
    """
    tickers = sorted(filtered["ticker"].dropna().unique())
    if not tickers:
        return None
    st.subheader("Select a stock")
    selected_ticker = st.selectbox(
        "Select a stock", options=tickers, key=ticker_key, label_visibility="collapsed"
    )
    return filtered[filtered["ticker"] == selected_ticker].iloc[0]


# ---------------------------------------------------------------------------
# Main table
# ---------------------------------------------------------------------------


def render_main_table(
    filtered: pd.DataFrame,
    domain_df: pd.DataFrame,
    table_key: str,
    ticker_key: str,
    last_rows_key: str,
) -> pd.DataFrame:
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
        table_key: This table's st.dataframe key (Phase 5b-1 row selection;
            Phase 5b-3 cell selection, derived keys — see module comment
            above render_cell_derivation_panel).
        ticker_key: The paired drill-down selectbox's key.
        last_rows_key: Selection-sync bookkeeping — see
            sync_drilldown_selection's docstring.

    Returns:
        display_df — the exact frame passed to st.dataframe, for the caller
        to pass on to the drill-down (not used directly here beyond that;
        the selection sync already happened inside this function).
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
    styled = bold_ticker_column(styled)

    column_config = {
        col: st.column_config.Column(
            label=MAIN_TABLE_COLUMN_LABELS[col], help=MAIN_TABLE_COLUMN_HELP.get(col)
        )
        for col in available_cols
        if col in MAIN_TABLE_COLUMN_LABELS
    }

    # Phase 5b-3: capture the cell selection BEFORE sync_drilldown_selection
    # overwrites the whole st.session_state[table_key] dict (cells included)
    # — same ordering discipline as pre_rows everywhere else in this file.
    pre_cells = st.session_state.get(table_key, {}).get("selection", {}).get("cells", [])

    sync_drilldown_selection(display_df, table_key, ticker_key, last_rows_key)

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config=column_config,
        key=table_key,
        on_select="rerun",
        selection_mode=["single-row", "single-cell"],
    )

    process_cell_selection(pre_cells, display_df, table_key)
    render_cell_derivation_panel(display_df, filtered, table_key)

    return display_df


# ---------------------------------------------------------------------------
# Drill-down
# ---------------------------------------------------------------------------


def _render_diff_derivation_body(row: pd.Series, factor: str) -> None:
    """The four markdown lines shared by both diff-derivation entry points:
    the two inputs, the diff (now naming its actual formula — Phase 5b-3,
    replacing the previous plain "Diff." label, which didn't distinguish
    the five ratio-minus-one factors from the five subtractions, three of
    which subtract in the opposite order from how their two inputs are
    listed below), and the score. No st.expander/container of its own —
    the caller supplies that, so the drill-down's expander title
    (unchanged, Phase 3c.2) and the click-a-cell panel's ticker-naming
    title (Phase 5b-3) can differ without duplicating this body.

    Args:
        row: One stock's row from a scored short_screen DataFrame.
        factor: A key of DIFF_FACTOR_INPUTS.
    """
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
    st.markdown(f"- **Diff.** ({format_diff_formula(factor)}): {diff_str}")

    score_val = row[factor]
    score_str = f"{score_val:.3f}" if pd.notna(score_val) else "N/A"
    st.markdown(f"- **Score**: {score_str}")


def render_diff_derivation(row: pd.Series, factor: str) -> None:
    """Expander showing a diff-based factor's full derivation: its two
    inputs (in the Excel template's block order), the diff, and the score.

    Only called for the 10 factors in DIFF_FACTOR_INPUTS — this is the
    Phase 3c.2 drill-down-only treatment; non-diff factors keep the plain
    score+metric row from render_drill_down's factor table. Phase 5b-3
    extracted the body into _render_diff_derivation_body (shared with the
    click-a-cell derivation panel — see render_cell_derivation_panel) but
    left this function's own signature and output (including this exact
    expander title) unchanged: the drill-down already names the stock
    above this expander, so repeating it here across ten expanders would
    be redundant and noisy — unlike the cell panel, which can show a
    different company from whatever the drill-down currently displays.

    Args:
        row: One stock's row from a scored short_screen DataFrame.
        factor: A key of DIFF_FACTOR_INPUTS.
    """
    label = FACTOR_DISPLAY_NAMES.get(factor, factor)
    with st.expander(f"Show derivation — {label}"):
        _render_diff_derivation_body(row, factor)


# ---------------------------------------------------------------------------
# Phase 5b-3 (R7): click-a-cell derivation, main scored table only.
#
# A cell click and a row click are independent, undocumented pieces of
# frontend state within streamlit 1.63.0's combined single-row+single-cell
# selection mode — confirmed by direct probe, not assumed: neither click
# disturbs the other. A programmatic `cells: []` push (which
# sync_drilldown_selection makes unconditionally, every rerun, for every
# selection-bearing table — untouched by this phase) shapes only the return
# value of the run it happens in; it has no durable effect on what the next
# rerun reads back, unlike the equivalent row push, which does durably
# repaint the highlight (also confirmed by direct probe). Consequence: once
# a cell is clicked, st.session_state[table_key]["selection"]["cells"]
# keeps reporting that same click on every later rerun WHERE THE UNDERLYING
# DATA IS UNCHANGED — there is no user gesture that clears it back to
# empty, mirroring the already-shipped row-selection quirk documented in
# sync_drilldown_selection's own module comment (clicking the highlighted
# row deselects it, then is immediately re-painted — "intentional, not a
# bug"). BUT a rerun where filtered/display_df's own content actually
# reshapes (any sidebar filter change) DOES reset the frontend's cells
# selection to empty on that rerun — confirmed by live browser probe
# against the real app (not the earlier scratch-script probes, which only
# ever changed an UNRELATED widget against a static frame and so never
# exercised this). Critically, this empty reset happens REGARDLESS of
# whether the previously-clicked ticker still survives the new filter, so
# it cannot be read as "the user deselected" — process_cell_selection
# therefore ignores an empty pre_cells unconditionally (see its own
# docstring) rather than treating it as a fresh, empty selection.
#
# This means resolve_selected_cell must be called ONCE, on the rerun where
# the selection actually changes (is_fresh_selection), and its result
# persisted — never re-called on a later, unrelated rerun against a since-
# reordered/filtered display_df using that same stale cells value, which
# would silently resolve whatever ticker now happens to sit at that old row
# position (the 5b-1 positional trap, reappearing through this door).
# process_cell_selection below does the resolve-once step;
# render_cell_derivation_panel re-validates the persisted ticker BY IDENTITY
# (find_ticker_row) on every rerun, never by re-resolving position.
#
# A clicked cell's row index IS confirmed sort-invariant, like rows
# (verified directly against streamlit 1.63.0 in both directions — a
# visually-top row whose true position is last, and a visually-bottom row
# whose true position is first, both round-tripped correctly) — this is
# documented for rows in the installed streamlit source and was NOT
# documented for cells, so it was measured rather than assumed. See
# src/selection.py's resolve_selected_cell docstring.
#
# Known, accepted consequence: because a row click and a cell click never
# touch each other, a user can move the drill-down (below, via a row click)
# to a different company while this panel keeps showing whichever
# company's cell was last clicked — a real "two companies visible on one
# page" state. The panel's own "Derivation — {ticker}, ..." heading (unlike
# render_diff_derivation's title, deliberately left unchanged — see that
# function's docstring) makes this truthful rather than misleading, which
# is why it's acceptable to ship rather than something to design around.
# ---------------------------------------------------------------------------


def process_cell_selection(pre_cells: list, display_df: pd.DataFrame, table_key: str) -> None:
    """Resolve a fresh cell click and persist (ticker, column) — the only
    place resolve_selected_cell is ever called (see module comment above).

    Must run after sync_drilldown_selection (which owns
    st.session_state[table_key] as a whole) and after st.dataframe(...) is
    instantiated with key=table_key, mirroring where render_cell_derivation_
    panel is also called from render_main_table.

    Args:
        pre_cells: The table's raw cells selection-state list, captured
            BEFORE sync_drilldown_selection overwrote st.session_state[
            table_key] — see render_main_table.
        display_df: The exact frame passed to st.dataframe this run.
        table_key: This table's st.dataframe key — used to derive this
            function's two bookkeeping session_state keys so render_main_
            table's own call site (and main()'s call to it) needs no new
            parameters.

    Uses should_process_cell_selection, not is_fresh_selection directly —
    see that function's docstring for why an empty pre_cells must never be
    treated as a fresh (deselecting) click. render_cell_derivation_panel's
    own find_ticker_row check is what decides whether a persisted ticker
    survived a filter change (§5.6 crossing 1) or must be cleared with a
    caption (crossing 2) — this function only ever resolves and persists on
    a genuinely non-empty, genuinely new cells value (a real click).

    The last_cells_key write below is UNCONDITIONAL — outside the `if`, not
    inside it. It must always track "what the frontend last reported",
    never just "what we last resolved", or it goes stale across a reshape:
    a reshape correctly skips resolving (pre_cells is empty) but must still
    update the baseline to that empty value, so a click that lands back on
    the SAME (row, column) in the new frame — the top row is the likeliest
    repeat — is still recognized as fresh and resolved against the CURRENT
    display_df, rather than silently comparing equal to a now-stale
    pre-reshape baseline and being skipped (which would either leave a dead
    click with nothing rendered, or a stale, wrong-company panel still
    showing under the old ticker's name). Do not move this write back
    inside the guard.
    """
    last_cells_key = f"{table_key}_last_cells"
    cell_derivation_key = f"{table_key}_cell_derivation"
    last_cells = st.session_state.get(last_cells_key)
    if should_process_cell_selection(pre_cells, last_cells):
        st.session_state[cell_derivation_key] = resolve_selected_cell(display_df, pre_cells)
    # Store pre_cells VERBATIM, never reconstructed/retyped — a rebuilt
    # (e.g. list-literal) copy could compare unequal to a later, unchanged,
    # tuple-shaped read and misreport as fresh, undoing the guard above. See
    # tests/test_selection.py::test_shape_mismatch_in_stored_value_is_the_failure_mode_storage_discipline_prevents.
    st.session_state[last_cells_key] = pre_cells


def render_cell_derivation_panel(
    display_df: pd.DataFrame, filtered: pd.DataFrame, table_key: str
) -> None:
    """Render the persisted cell-derivation panel, if any (Phase 5b-3).

    Re-validates the persisted ticker BY IDENTITY (find_ticker_row) against
    the CURRENT display_df on every call — never re-resolves by the
    original click's row position (see module comment above). Renders
    nothing if no cell has been clicked yet (no default-shown panel before
    the first click).

    The row rendered comes from `filtered`, NOT `display_df`: display_df is
    column-subset to DISPLAY_COLUMNS (or its interleaved form) for the
    on-screen table and never carries the 20 raw diff-input columns
    (ps_ntm, ps_3yr_avg, etc. — Phase 3c.2's deliberate scope decision, see
    DIFF_INPUT_COLUMNS' module comment) — only the render_drill_down path's
    own `filtered`-sourced row ever did. Reading the row from display_df
    here would make _render_diff_derivation_body's "N/A" branch fire for
    every diff input on every cell click, indistinguishable from a genuine
    NaN — display_df is used only to confirm the ticker is still on
    screen, exactly like render_drill_down's own select_drilldown_row.

    Args:
        display_df: The exact frame passed to st.dataframe this run — used
            only to check the persisted ticker is still present.
        filtered: The sidebar-filtered frame with every column (the same
            frame render_main_table received), used to look up the row's
            actual content by ticker identity.
        table_key: This table's st.dataframe key — see process_cell_
            selection for how the bookkeeping keys are derived from it.
    """
    cell_derivation_key = f"{table_key}_cell_derivation"
    resolved = st.session_state.get(cell_derivation_key)
    if resolved is None:
        return
    ticker, column = resolved

    if find_ticker_row(display_df, ticker) is None:
        st.caption(f"Derivation cleared: {ticker} is no longer in the filtered table.")
        st.session_state[cell_derivation_key] = None
        return
    row = filtered.loc[filtered["ticker"] == ticker].iloc[0]

    if column in CELL_DERIVATION_FACTORS:
        factor = CELL_DERIVATION_FACTORS[column]
        label = FACTOR_DISPLAY_NAMES.get(factor, factor)
        with st.expander(f"Derivation — {ticker}, {label}", expanded=True):
            _render_diff_derivation_body(row, factor)
        return

    non_diff_factor = NON_DIFF_FACTOR_BY_COLUMN.get(column)
    if non_diff_factor is not None:
        label = FACTOR_DISPLAY_NAMES.get(non_diff_factor, non_diff_factor)
        metric_col = FACTOR_DEFINITIONS[non_diff_factor]["metric"]
        metric_label = METRIC_COLUMN_LABELS.get(metric_col, metric_col)
        with st.expander(f"Derivation — {ticker}, {label}", expanded=True):
            st.markdown(
                f"No step-by-step derivation is available for *{label}*. Its score is the "
                f"percentile rank of *{metric_label}* across the screen's universe."
            )
        return

    if column in ("overall_score", "mscore_flag"):
        label = MAIN_TABLE_COLUMN_LABELS[column]
        with st.expander(f"Derivation — {ticker}, {label}", expanded=True):
            st.markdown(MAIN_TABLE_COLUMN_HELP[column])
        return

    if column in IDENTITY_COLUMN_LABELS:
        label = IDENTITY_COLUMN_LABELS[column]
        with st.expander(f"Derivation — {ticker}, {label}", expanded=True):
            st.markdown(f"*{label}* is an identity field — nothing to derive here.")
        return

    # Defensive only: every column ever passed to st.dataframe by
    # render_main_table is one of identity/overall_score/mscore_flag/a
    # factor score/a factor metric, all covered above. Not silent if this
    # is ever wrong, but not expected to fire.
    with st.expander(f"Derivation — {ticker}, {column}", expanded=True):
        st.markdown("No derivation is available for this column.")


def render_drill_down(
    filtered: pd.DataFrame,
    ticker_key: str,
    current_screen_id: str,
    membership_df: pd.DataFrame | None,
    screens_df: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    """Render the individual stock drill-down view.

    Args (Phase 5b-2): current_screen_id/membership_df/screens_df — see
    render_cross_screen_context's docstring; df — the UNFILTERED scored
    short_screen frame (same one render_sidebar/render_main_table receive),
    used only for zero_thematic_summary's ticker-set denominator, which must
    not move when sidebar filters change (same reasoning as styling.py's
    build_color_scale_domain).
    """
    row = select_drilldown_row(filtered, ticker_key)
    if row is None:
        st.info("No stocks match the current filters.")
        return

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
        render_cross_screen_context(
            row["ticker"], current_screen_id, membership_df, screens_df, df
        )
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
        .configure_axis(labelFont=APP_FONT_FAMILY, titleFont=APP_FONT_FAMILY)
        .configure_legend(labelFont=APP_FONT_FAMILY, titleFont=APP_FONT_FAMILY)
        .configure_title(font=APP_FONT_FAMILY)
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

    render_cross_screen_context(row["ticker"], current_screen_id, membership_df, screens_df, df)


# ---------------------------------------------------------------------------
# Phase 5b-2 (R5): cross-screen context, shared by all three drill-down paths.
# ---------------------------------------------------------------------------

# Phase 5c-3: per-screen icon for each "Also Appears On" block, keyed on
# screen_id (never display_name, which can differ from it — e.g.
# short_screen / "OWS Short Screen"). Looked up with .get(), never [], so an
# unmapped future screen_id renders with _DEFAULT_SCREEN_ICON rather than
# raising — the same genericity property tests/test_overlap.py's
# TestGenericityRegressionLock protects for the overlap table.
SCREEN_ICONS = {
    "competition": ":material/swords:",
    "cyclicals": ":material/autorenew:",
    "management_comp": ":material/payments:",
    "rising_short_interest": ":material/trending_up:",
    "short_screen": ":material/trending_down:",
    "structural": ":material/foundation:",
}
_DEFAULT_SCREEN_ICON = ":material/label:"


def render_cross_screen_context(
    ticker: str,
    current_screen_id: str,
    membership_df: pd.DataFrame | None,
    screens_df: pd.DataFrame,
    universe_df: pd.DataFrame | None,
) -> None:
    """Render the drill-down's "Also Appears On" section.

    Identity (name/sector/market_cap) is never repeated per screen here —
    only what the population test behind this phase found actually varies:
    a curated screen's rationale + stock performance, an unscored (RSI)
    screen's derived metrics, or the universe screen's composite score (see
    src/cross_screen_context.py's module docstring).

    The zero-contributions case is worded differently depending on which
    screen is asking, since "on zero other screens" means something
    different from each vantage point:
      - Viewed from the universe screen (short_screen): every one of its
        own rows is, by definition, in-universe, so "zero other screens"
        means zero thematic/RSI screens — the common case (see
        zero_thematic_summary), worth stating as the norm, with the live
        proportion, not as a gap.
      - Viewed from a thematic/RSI screen: "zero other screens" can only
        mean the ticker is outside the universe AND on no other thematic
        screen — a plain statement, no percentage claim, since that stat
        describes the universe's own membership, which this ticker isn't
        even part of.

    Args:
        ticker: The ticker currently shown in the drill-down.
        current_screen_id: The screen this drill-down belongs to.
        membership_df: The full screen_membership table, or None if it
            failed to load (a fresh/partial database) — the section is
            skipped with a soft caption rather than crashing.
        screens_df: The screens registry.
        universe_df: The UNFILTERED short_screen scored frame (used only
            when current_screen_id is the universe screen, for
            zero_thematic_summary's denominator) — may be None on other
            screens' pages, where it's simply not needed.
    """
    st.divider()
    st.subheader("Also Appears On")

    if membership_df is None:
        st.caption("Cross-screen context isn't available yet — run an ingest pipeline first.")
        return

    screen_data = load_screens_for_ticker(ticker, current_screen_id, membership_df, screens_df)
    contributions = build_also_appears_on(
        ticker, current_screen_id, membership_df, screens_df, screen_data
    )

    if not contributions:
        if current_screen_id == UNIVERSE_SCREEN_ID and universe_df is not None:
            zero_count, universe_total = zero_thematic_summary(
                set(universe_df["ticker"]), membership_df
            )
            pct = zero_count / universe_total if universe_total else 0.0
            display_names = dict(zip(screens_df["screen_id"], screens_df["display_name"]))
            universe_display_name = display_names.get(UNIVERSE_SCREEN_ID, UNIVERSE_SCREEN_ID)
            st.caption(
                f"{ticker} does not appear on any other screen — that's the norm, not a "
                f"gap: {zero_count:,} of {universe_total:,} stocks in {universe_display_name}'s "
                f"universe ({pct:.1%}) are on no thematic screen at all."
            )
        else:
            st.caption(f"{ticker} does not appear on any other screen.")
        return

    for contribution in contributions:
        icon = SCREEN_ICONS.get(contribution["screen_id"], _DEFAULT_SCREEN_ICON)
        st.markdown(f"{icon} **{contribution['display_name']}**")
        kind = contribution["kind"]
        if kind == "universe":
            score = contribution["overall_score"]
            st.write(f"Composite score: {score:.3f}" if pd.notna(score) else "Composite score: N/A")
        elif kind == "curated":
            perf = contribution["stock_performance"]
            st.write(f"{_STOCK_PERFORMANCE_LABEL}: {perf:.2%}" if pd.notna(perf) else f"{_STOCK_PERFORMANCE_LABEL}: N/A")
            rationale = contribution["rationale"]
            st.write(rationale if pd.notna(rationale) else "No rationale available.")
        elif kind == "unscored":
            for col, val in contribution["metrics"].items():
                label = UNSCORED_METRIC_DISPLAY_NAMES.get(col, col)
                if pd.notna(val) and col in UNSCORED_METRIC_FORMATS:
                    value_str = UNSCORED_METRIC_FORMATS[col].format(val)
                elif pd.notna(val):
                    value_str = f"{val:.4f}"
                else:
                    value_str = "N/A"
                st.write(f"{label}: {value_str}")


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

    all_sectors = sorted(df["sector"].dropna().unique())
    selected_sectors = st.sidebar.multiselect("**Sector**", options=all_sectors)

    mcap_min = float(df["market_cap"].min())
    mcap_max = float(df["market_cap"].max())
    mcap_range = st.sidebar.slider(
        "**Market Cap ($M)**",
        min_value=mcap_min,
        max_value=mcap_max,
        value=(mcap_min, mcap_max),
        format="$%,.0f",
    )

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

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


def render_curated_table(
    filtered: pd.DataFrame, table_key: str, ticker_key: str, last_rows_key: str
) -> pd.DataFrame:
    """Render the main curated table with export buttons. No M-Score
    highlighting and no factor columns — there aren't any.

    Args (Phase 5b-1): table_key/ticker_key/last_rows_key — see
    render_main_table's docstring; identical role here.

    Returns:
        display_df — the exact frame passed to st.dataframe.
    """
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

    styled = bold_ticker_column(styled)

    column_config = {
        col: st.column_config.Column(
            label=CURATED_COLUMN_LABELS[col], help=CURATED_COLUMN_HELP.get(col)
        )
        for col in available_cols
        if col in CURATED_COLUMN_LABELS
    }

    sync_drilldown_selection(display_df, table_key, ticker_key, last_rows_key)

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config=column_config,
        key=table_key,
        on_select="rerun",
        selection_mode="single-row",
    )

    return display_df


def render_curated_drill_down(
    filtered: pd.DataFrame,
    ticker_key: str,
    current_screen_id: str,
    membership_df: pd.DataFrame | None,
    screens_df: pd.DataFrame,
) -> None:
    """Render the individual stock drill-down view for a curated screen:
    identity, the three risk scores, and the full narrative rationale.

    Args (Phase 5b-2): current_screen_id/membership_df/screens_df — see
    render_cross_screen_context's docstring. universe_df is omitted here
    (passed as None) — a curated screen's own page is never the vantage
    point that needs zero_thematic_summary's percentage sentence (see that
    function's docstring: only current_screen_id == UNIVERSE_SCREEN_ID uses
    it).
    """
    row = select_drilldown_row(filtered, ticker_key)
    if row is None:
        st.info("No stocks match the current filters.")
        return

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

    render_cross_screen_context(row["ticker"], current_screen_id, membership_df, screens_df, None)


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

    mcap_min = float(df["market_cap"].min())
    mcap_max = float(df["market_cap"].max())
    mcap_range = st.sidebar.slider(
        "**Market Cap ($M)**",
        min_value=mcap_min,
        max_value=mcap_max,
        value=(mcap_min, mcap_max),
        format="$%,.0f",
    )

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    filtered = df[
        (df["market_cap"] >= mcap_range[0]) & (df["market_cap"] <= mcap_range[1])
    ].copy()

    st.sidebar.divider()
    st.sidebar.metric("Stocks shown", len(filtered))

    return filtered


def render_unscored_table(
    filtered: pd.DataFrame, table_key: str, ticker_key: str, last_rows_key: str
) -> pd.DataFrame:
    """Render the main table for an unscored quant screen, with export
    buttons. No M-Score highlighting, no factor columns — there aren't any.

    Args (Phase 5b-1): table_key/ticker_key/last_rows_key — see
    render_main_table's docstring; identical role here.

    Returns:
        display_df — the exact frame passed to st.dataframe.
    """
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
    styled = bold_ticker_column(styled)

    column_config = {
        col: st.column_config.Column(
            label=UNSCORED_COLUMN_LABELS[col], help=UNSCORED_COLUMN_HELP.get(col)
        )
        for col in available_cols
        if col in UNSCORED_COLUMN_LABELS
    }

    sync_drilldown_selection(display_df, table_key, ticker_key, last_rows_key)

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config=column_config,
        key=table_key,
        on_select="rerun",
        selection_mode="single-row",
    )

    return display_df


def render_unscored_drill_down(
    filtered: pd.DataFrame,
    ticker_key: str,
    current_screen_id: str,
    membership_df: pd.DataFrame | None,
    screens_df: pd.DataFrame,
) -> None:
    """Render the individual stock drill-down view for an unscored quant
    screen: identity plus a flat list of the 8 derived metrics. No factor
    chart (no factor model) and no rationale (not curated data).

    Args (Phase 5b-2): current_screen_id/membership_df/screens_df — see
    render_cross_screen_context's docstring; universe_df omitted (None) for
    the same reason as render_curated_drill_down.
    """
    row = select_drilldown_row(filtered, ticker_key)
    if row is None:
        st.info("No stocks match the current filters.")
        return

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

    render_cross_screen_context(row["ticker"], current_screen_id, membership_df, screens_df, None)


# ---------------------------------------------------------------------------
# Cross-screen overlap (Phase 3d Part 1) — relocated in Phase 5b-2 from a
# separate top-level view into a collapsed expander rendered at the bottom
# of every screen's page (see main()), since it's global by nature (every
# screen shares one overlap table) rather than a fourth per-screen render
# path.
# ---------------------------------------------------------------------------


def _load_screen_df(screen_id: str, kind: str) -> pd.DataFrame | None:
    """screen_id's identity-bearing table via the correct existing
    per-screen cached loader for the given classify_screen() kind
    (Phase 5b-2). The one place both load_all_screen_identity_data (every
    registered screen) and load_screens_for_ticker (only a specific
    ticker's other screens) resolve screen_id -> loader, so the two loaders
    and classify_screen's own taxonomy cannot disagree with each other.

    Args:
        screen_id: The screen to load.
        kind: A classify_screen(...) return value.

    Returns:
        The loaded frame, or None for "unknown" (or an unrecognized kind) —
        degrading to "no loader applies" rather than guessing, per
        classify_screen's own contract.
    """
    if kind in ("universe", "scored"):
        return load_quant_data(screen_id)
    if kind == "curated":
        return load_curated_data(screen_id)
    if kind == "unscored":
        return load_unscored_quant_data(screen_id)
    return None


def load_all_screen_identity_data(screens_df: pd.DataFrame) -> dict:
    """Load every screen's own identity-bearing table, keyed by screen_id.

    Dispatches via classify_screen + _load_screen_df (Phase 5b-2) — the
    same taxonomy and loader-selection load_screens_for_ticker below uses,
    so the two loaders can't disagree, and so a screen's type combination
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
    for screen_id in screens_df["screen_id"]:
        kind = classify_screen(screen_id, screens_df, UNIVERSE_SCREEN_ID)
        df = _load_screen_df(screen_id, kind)
        if df is not None:
            screen_data[screen_id] = df
    return screen_data


def load_screens_for_ticker(
    ticker: str, current_screen_id: str, membership_df: pd.DataFrame, screens_df: pd.DataFrame
) -> dict:
    """Load ONLY the screens `ticker` is actually on besides
    current_screen_id (Phase 5b-2), via classify_screen + _load_screen_df —
    not load_all_screen_identity_data's eager load of every registered
    screen, which would pay a needless copy (up to short_screen's full
    1,358-row frame) on every drill-down render regardless of how many of
    the other five screens that ticker even touches. Most tickers sit on
    0-2 other screens (ceiling 5).

    The "universe" kind is special-cased to use get_universe_scores' narrow
    {ticker: overall_score} lookup instead of the full scored frame, built
    into a 1-row synthetic DataFrame so build_screen_contribution's generic
    df.loc[df["ticker"] == ticker] pattern needs no special-casing on the
    pure-module side.

    Args:
        ticker: The ticker being viewed.
        current_screen_id: The screen whose drill-down is asking.
        membership_df: The full screen_membership table.
        screens_df: The screens registry.

    Returns:
        screen_id -> that screen's identity-bearing DataFrame (full, except
        a narrow 1-row synthetic frame for "universe").
    """
    other_ids = other_screen_ids_for_ticker(ticker, current_screen_id, membership_df)
    screen_data = {}
    for screen_id in other_ids:
        kind = classify_screen(screen_id, screens_df, UNIVERSE_SCREEN_ID)
        if kind == "universe":
            scores = get_universe_scores()
            if scores is not None and ticker in scores:
                screen_data[screen_id] = pd.DataFrame(
                    {"ticker": [ticker], "overall_score": [scores[ticker]]}
                )
            continue
        df = _load_screen_df(screen_id, kind)
        if df is not None:
            screen_data[screen_id] = df
    return screen_data


@st.cache_data
def get_universe_scores() -> dict | None:
    """{ticker: overall_score} for every short_screen ticker (Phase 5b-2).

    Calls load_quant_data(UNIVERSE_SCREEN_ID) (the full 1,358-row frame)
    exactly once — on whichever rerun first needs this — because this
    function is itself cached: every later call is a cache hit returning
    the small dict directly, never re-invoking load_quant_data at all (the
    same outer-cache-hit-skips-inner-call property get_overlap_df below
    relies on). Every subsequent rerun of any in-universe ticker's
    cross-screen context then pays a small-dict copy, not a 1,358x40-column
    frame copy.

    Returns:
        {ticker: overall_score}, or None if short_screen's scored data
        doesn't exist yet.
    """
    df = load_quant_data(UNIVERSE_SCREEN_ID)
    if df is None:
        return None
    return dict(zip(df["ticker"], df["overall_score"]))


@st.cache_data
def get_screen_data_and_membership():
    """(membership_df, screen_data) for compute_overlap (Phase 5b-2) — the
    one eager load of every registered screen's full identity table, used
    ONLY by get_overlap_df below. Cached with no arguments (a singleton),
    so a cache hit skips this body — including the eager load — entirely;
    the global st.cache_data.clear() the app's own Refresh Data buttons
    already call invalidates it like everything else.

    Returns:
        (membership_df, screen_data dict), or None if either the screens
        registry or screen_membership doesn't exist yet.
    """
    screens_df = list_screens()
    if screens_df is None:
        return None
    membership_df = load_screen_membership()
    if membership_df is None:
        return None
    screen_data = load_all_screen_identity_data(screens_df)
    return membership_df, screen_data


@st.cache_data
def get_overlap_df() -> pd.DataFrame | None:
    """compute_overlap's result, cached with no arguments (Phase 5b-2) — a
    real ~0.6s computation over 1,375 tickers, now paid once per session
    (until Refresh Data clears the cache) rather than once per rerun of the
    overlap expander, which st.expander does NOT make lazy (its body
    executes on every rerun even while collapsed).

    Returns:
        compute_overlap's DataFrame, or None if the underlying data isn't
        loaded yet.
    """
    screens_df = list_screens()
    if screens_df is None:
        return None
    bundle = get_screen_data_and_membership()
    if bundle is None:
        return None
    membership_df, screen_data = bundle
    return compute_overlap(membership_df, screens_df, screen_data)


def render_overlap_section(screens_df: pd.DataFrame) -> None:
    """Render the cross-screen overlap section: how many of the thematic/RSI
    screens each ticker sits on, which ones, and short_screen's composite
    score as context — not as a membership tick (see src/overlap.py's
    module docstring for why short_screen is treated differently).

    Phase 5b-2: relocated from a top-level view into a collapsed expander,
    called once from main() regardless of which screen is selected, so it's
    reachable from every screen rather than only its own dedicated page.
    Its own filters live inside the expander body (not the sidebar, which
    stays about the currently-selected screen) and its own Refresh Data
    button is deliberately NOT duplicated here — every screen's sidebar
    already has one, and it calls the same global st.cache_data.clear()
    that invalidates get_overlap_df too.
    """
    display_names = dict(zip(screens_df["screen_id"], screens_df["display_name"]))
    universe_display_name = display_names.get(UNIVERSE_SCREEN_ID, UNIVERSE_SCREEN_ID)

    with st.expander("Cross-Screen Overlap", expanded=False):
        overlap_df = get_overlap_df()
        if overlap_df is None:
            st.error("No screen_membership data found. Run an ingest pipeline first.")
            return
        membership_df = load_screen_membership()

        include_zero = st.checkbox(
            "Include short_screen-only names (on 0 thematic screens)",
            value=False,
            help="The only control that reveals tickers on zero thematic/RSI "
            "screens. The slider below governs only the 1-or-more band.",
        )

        # Ceiling is computed from the UNFILTERED overlap_df, once, before
        # any filter below is applied — otherwise an unrelated sector
        # selection would silently move the slider's own bound out from
        # under the user.
        ceiling = screen_count_ceiling(overlap_df)
        if ceiling == 1:
            st.caption(
                "Every thematic/RSI-screen ticker appears on exactly 1 screen — "
                "no minimum-count slider to show."
            )
            min_screen_count = 1
        else:
            min_screen_count = st.slider(
                "Minimum screen count", min_value=1, max_value=ceiling, value=1
            )

        count_mask = overlap_df["screen_count"] >= min_screen_count
        if include_zero:
            count_mask = count_mask | (overlap_df["screen_count"] == 0)
        filtered = overlap_df[count_mask]

        all_sectors = sorted(overlap_df["sector"].dropna().unique())
        selected_sectors = st.multiselect("Sector", options=all_sectors)
        if selected_sectors:
            filtered = filtered[filtered["sector"].isin(selected_sectors)]

        st.metric("Tickers shown", len(filtered))

        filtered = filtered.sort_values(
            ["screen_count", "overall_score"], ascending=[False, False]
        )

        display_df = filtered[OVERLAP_DISPLAY_COLUMNS]
        display_df = apply_zero_thematic_label(display_df)

        # Export is always a fixed superset of the on-screen columns — same
        # house pattern as render_main_table's export (24 factor metrics +
        # 20 diff inputs regardless of a display-only checkbox). in_universe
        # is included unconditionally so the 17 not-in-universe rows are
        # distinguishable in the exported file via a native boolean column,
        # not via a blank cell or a label baked into the numeric score
        # column (which would break Excel's ability to sort it). The export
        # keeps screens_on's real empty string (never apply_zero_thematic_
        # label's on-screen placeholder) — a spreadsheet consumer can filter
        # on that directly.
        export_cols = OVERLAP_DISPLAY_COLUMNS + ["in_universe"]
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
        styled = bold_ticker_column(styled)

        column_config = {
            col: st.column_config.Column(
                label=OVERLAP_COLUMN_LABELS[col], help=OVERLAP_COLUMN_HELP.get(col)
            )
            for col in OVERLAP_DISPLAY_COLUMNS
            if col in OVERLAP_COLUMN_LABELS
        }
        column_config["overall_score"] = st.column_config.Column(
            label=f"{universe_display_name} Composite Score",
            help=OVERLAP_COLUMN_HELP.get("overall_score"),
        )

        # Phase 5b-2 click-through: a fresh click navigates to the clicked
        # ticker's drill-down on the appropriate screen (see
        # resolve_overlap_click_target). The fresh-click check is read
        # BEFORE sync_drilldown_selection overwrites last_rows_key — same
        # ordering discipline as everywhere else this pattern appears.
        pre_rows = st.session_state.get("overlap_table", {}).get("selection", {}).get("rows", [])
        fresh_click = bool(
            pre_rows
            and is_fresh_selection(pre_rows, st.session_state.get("overlap_table_last_rows"))
        )

        sync_drilldown_selection(
            display_df, "overlap_table", "overlap_selected_ticker", "overlap_table_last_rows"
        )

        st.dataframe(
            styled,
            use_container_width=True,
            height=600,
            hide_index=True,
            column_config=column_config,
            key="overlap_table",
            on_select="rerun",
            selection_mode="single-row",
        )

        if fresh_click and membership_df is not None:
            idx = pre_rows[0]
            if 0 <= idx < len(display_df):
                ticker = display_df["ticker"].iloc[idx]
                # in_universe is not in OVERLAP_DISPLAY_COLUMNS (it's an
                # export-only column) — resolved by a TICKER LOOKUP against
                # filtered (the pre-column-subset frame, which does carry
                # it), never by indexing filtered at the selection's
                # positional index. filtered and display_df share the same
                # ticker set but not the same column set, and filtered's
                # own row order need not match display_df's — indexing it
                # positionally here would be exactly the 5b-1 trap this
                # phase has otherwise avoided throughout.
                in_universe_match = filtered.loc[filtered["ticker"] == ticker, "in_universe"]
                in_universe = bool(in_universe_match.iloc[0]) if not in_universe_match.empty else False
                target_screen = resolve_overlap_click_target(
                    ticker, in_universe, membership_df, screens_df
                )
                if target_screen is not None:
                    st.session_state["_pending_nav"] = (target_screen, ticker)
                    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def format_screen_title(display_name: str) -> str:
    """Wrap a screen's display name in the :primary[...] markdown directive.

    A display name containing "[" or "]" is returned unwrapped, rendered in
    the default text colour instead of brand green. Streamlit's directive
    parsing is frontend-only: an unrecognised or early-closed directive
    renders as literal text and does not raise, so a bracket in the name
    could leak the directive markup onto the screen. No live screen name
    contains a bracket today, but correct text beats brand colour.
    """
    if "[" in display_name or "]" in display_name:
        return display_name
    return f":primary[{display_name}]"


def main():
    screens_df = list_screens()
    if screens_df is None:
        st.title("OWS Short Screen")
        st.error(
            "No data found. Run the pipeline first:\n\n"
            "`python src/ingest.py && python src/transform.py && python src/score.py`"
        )
        return

    # Phase 5c-2: st.logo caps at 32px ("large") in this streamlit version,
    # which cannot reach the requested ~4x sizing, so the mark is rendered
    # as a plain sidebar image instead. Disclosed consequence: st.logo's
    # icon_image used to draw a mark in the app's upper-left corner when the
    # sidebar is collapsed; a sidebar image cannot do that, so collapsing
    # the sidebar now leaves no mark on screen. Accepted by the Driver —
    # the app also sets initial_sidebar_state="expanded".
    if os.path.exists(LOGO_MARK_PATH):
        st.sidebar.image(LOGO_MARK_PATH, width=SIDEBAR_MARK_WIDTH_PX)

    # Phase 5b-2: consume a pending cross-screen navigation (see
    # render_overlap_section's click-through), BEFORE the Screen selectbox
    # below is instantiated — the only legal time to force its session_state
    # key. See apply_pending_nav's module comment for the full mechanism,
    # including why _nav_target carries its own screen_id rather than
    # relying on this ordering alone.
    pending_nav = st.session_state.pop("_pending_nav", None)
    if pending_nav is not None:
        target_screen_id, nav_ticker = pending_nav
        st.session_state["screen_selector"] = target_screen_id
        st.session_state["_nav_target"] = (target_screen_id, nav_ticker)

    screen_ids = list(screens_df["screen_id"])
    display_names = dict(zip(screens_df["screen_id"], screens_df["display_name"]))
    screen_types = dict(zip(screens_df["screen_id"], screens_df["screen_type"]))
    has_scoring_by_id = dict(zip(screens_df["screen_id"], screens_df["has_scoring"]))
    default_index = screen_ids.index("short_screen") if "short_screen" in screen_ids else 0

    selected_screen_id = st.sidebar.selectbox(
        "**Screen**",
        options=screen_ids,
        index=default_index,
        format_func=lambda sid: display_names.get(sid, sid),
        key="screen_selector",
    )
    st.sidebar.divider()

    # Phase 5c-2: a horizontal container, not a column ratio — st.columns
    # splits proportionally to viewport width, so a ratio tuned to look
    # right at one window width gives a gap that grows with every wider
    # window (verified: it does). A flexbox row sizes each child to its own
    # natural width and holds a FIXED pixel gap between them regardless of
    # viewport width.
    with st.container(horizontal=True, vertical_alignment="center", gap=8):
        st.image(TITLE_MARK_PATH, width=TITLE_MARK_SIZE_PX)
        st.title(format_screen_title(display_names[selected_screen_id]))

    screen_type = screen_types[selected_screen_id]

    # Phase 5b-1: the main table's row selection and the drill-down's
    # selectbox are one piece of state, namespaced per screen_id so
    # switching screens can't cross-contaminate selection.
    table_key = f"{selected_screen_id}_main_table"
    ticker_key = f"{selected_screen_id}_drilldown_ticker"
    last_rows_key = f"{selected_screen_id}_last_selected_rows"

    # Phase 5b-2: the drill-down's "also appears on" section needs the full
    # membership table (cheap, already @st.cache_data) regardless of screen
    # type.
    membership_df = load_screen_membership()

    if screen_type == "quant_composite" and has_scoring_by_id[selected_screen_id]:
        df = load_quant_data(selected_screen_id)
        if df is None:
            st.error(
                f"No scored data found for {display_names[selected_screen_id]}. "
                "Run the pipeline first."
            )
            return
        filtered = render_sidebar(df)
        apply_pending_nav(filtered, ticker_key, selected_screen_id)
        render_main_table(filtered, df, table_key, ticker_key, last_rows_key)
        st.divider()
        render_drill_down(filtered, ticker_key, selected_screen_id, membership_df, screens_df, df)
    elif screen_type == "quant_composite" and not has_scoring_by_id[selected_screen_id]:
        df = load_unscored_quant_data(selected_screen_id)
        if df is None:
            st.error(
                f"No transformed data found for {display_names[selected_screen_id]}. "
                "Run ingest + transform first."
            )
            return
        filtered = render_unscored_sidebar(df)
        apply_pending_nav(filtered, ticker_key, selected_screen_id)
        render_unscored_table(filtered, table_key, ticker_key, last_rows_key)
        st.divider()
        render_unscored_drill_down(
            filtered, ticker_key, selected_screen_id, membership_df, screens_df
        )
    elif screen_type == "curated":
        df = load_curated_data(selected_screen_id)
        if df is None:
            st.error(
                f"No curated data found for {display_names[selected_screen_id]}. "
                "Run the curated ingest pipeline first."
            )
            return
        filtered = render_curated_sidebar(df)
        apply_pending_nav(filtered, ticker_key, selected_screen_id)
        render_curated_table(filtered, table_key, ticker_key, last_rows_key)
        st.divider()
        render_curated_drill_down(
            filtered, ticker_key, selected_screen_id, membership_df, screens_df
        )
    else:
        st.error(f"Unknown screen type {screen_type!r} for screen {selected_screen_id!r}.")
        return

    st.divider()
    render_overlap_section(screens_df)


if __name__ == "__main__":
    main()
