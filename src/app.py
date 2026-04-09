"""
OWS Short Screen — Streamlit Web UI.

Reads from the scored_data SQLite table and provides:
- Filterable, sortable main table with M-Score flag highlighting
- Sidebar filters: sector, industry, market cap, overall score
- Individual stock drill-down with factor scores grouped by category
- Excel and CSV export of the filtered table
"""

import io
import os

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect

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
def load_data() -> pd.DataFrame | None:
    """Load scored_data from SQLite. Returns None if unavailable."""
    if not os.path.exists(DB_PATH):
        return None
    engine = create_engine(f"sqlite:///{DB_PATH}")
    if "scored_data" not in inspect(engine).get_table_names():
        return None
    df = pd.read_sql_table("scored_data", engine)
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
# Main
# ---------------------------------------------------------------------------


def main():
    st.title("OWS Short Screen")

    df = load_data()

    if df is None:
        st.error(
            "No data found. Run the pipeline first:\n\n"
            "`python src/ingest.py && python src/transform.py && python src/score.py`"
        )
        return

    filtered = render_sidebar(df)

    tab_screener, tab_drilldown = st.tabs(["Screener", "Stock Drill-Down"])

    with tab_screener:
        render_main_table(filtered)

    with tab_drilldown:
        render_drill_down(filtered)


if __name__ == "__main__":
    main()
