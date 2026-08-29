"""
OWS Short Screen — Streamlit Web UI.

Reads the active screen's scored_data table and provides:
- Filterable, sortable main table with M-Score flag highlighting
- Sidebar filters: sector, industry, market cap, overall score
- Individual stock drill-down with factor scores grouped by category
- Excel and CSV export of the filtered table
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OWS Short Screen",
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def list_screens() -> pd.DataFrame | None:
    """Load the screens registry (screen_id, display_name, screen_type).

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
    """Render the main scored table with export buttons."""
    # Prepare display DataFrame
    available_cols = [c for c in DISPLAY_COLUMNS if c in filtered.columns]
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
    styled = display_df.style.apply(highlight_mscore_rows, axis=1)
    styled = styled.format(
        {
            "market_cap": "${:,.0f}",
            "overall_score": "{:.3f}",
            **{f: "{:.3f}" for f in available_cols if f.endswith("_factor")},
        }
    )

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

    # Factor table by category
    st.subheader("Factor Scores by Category")
    for category, factors in FACTOR_CATEGORIES.items():
        table_rows = []
        for factor in factors:
            if factor in row.index:
                val = row[factor]
                table_rows.append({
                    "Factor": FACTOR_DISPLAY_NAMES.get(factor, factor),
                    "Score": f"{val:.3f}" if pd.notna(val) else "N/A",
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

    if screen_type == "quant_composite":
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
