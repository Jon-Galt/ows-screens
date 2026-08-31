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
from src.score import FACTOR_DEFINITIONS

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


def highlight_mscore_rows(row: pd.Series) -> list[str]:
    """Apply light red background to rows where mscore_flag is True."""
    if row.get("mscore_flag", False):
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)


def render_main_table(filtered: pd.DataFrame) -> None:
    """Render the main scored table with export buttons.

    A checkbox (default off, so the existing view is unchanged unless a
    user opts in) shows each factor's underlying metric — the raw value
    its percentile score was computed from — immediately after it.
    """
    show_values = st.checkbox(
        "Show underlying metric values",
        value=False,
        help="Insert each factor's underlying value (what its percentile "
        "score was computed from) immediately after it.",
    )

    # Prepare display DataFrame
    columns = DISPLAY_COLUMNS
    if show_values:
        columns = []
        for col in DISPLAY_COLUMNS:
            columns.append(col)
            if col in FACTOR_DEFINITIONS:
                columns.append(FACTOR_DEFINITIONS[col]["metric"])

    available_cols = [c for c in columns if c in filtered.columns]
    display_df = filtered[available_cols].sort_values("overall_score", ascending=False)

    # Export buttons
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        xlsx_buffer = io.BytesIO()
        display_df.to_excel(xlsx_buffer, index=False, engine="openpyxl")
        st.download_button(
            label="Export to Excel",
            data=xlsx_buffer.getvalue(),
            file_name="ows_short_screen.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col2:
        csv_data = display_df.to_csv(index=False)
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

    styled = display_df.style.apply(highlight_mscore_rows, axis=1)
    styled = styled.format(format_dict)

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Drill-down
# ---------------------------------------------------------------------------


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

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
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

    st.dataframe(
        styled,
        use_container_width=True,
        height=600,
        hide_index=True,
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
            render_main_table(filtered)
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
