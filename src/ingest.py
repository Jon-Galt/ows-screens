"""
Ingest raw CSV/Excel exports into SQLite, scoped by screen.

Reads files from data/uploads/<screen_id>/ using that screen's ingest config
(sheet name, column map, required columns), coerces types, handles Bloomberg's
"#N/A N/A" string, and writes to that screen's raw_data__<screen_id> table.
"""

import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

# Allow `python src/ingest.py` to resolve `src.*` imports even though running
# a file directly doesn't put the project root on sys.path (only `src/` is).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import CONFIG_PATH, ScreenTypeError, get_screen_type, load_config
from src.db import replace_screen_rows, sync_screens_registry, table_name
from src.loaders import (
    extract_ticker,
    find_single_upload_file,
    log_summary,
    read_upload,
    validate_columns,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Bloomberg column name -> Python snake_case mapping.
# Derived from the Data sheet of the reference Excel file (81 columns).
COLUMN_MAP = {
    "Ticker": "ticker",
    "Name": "name",
    "Sector": "sector",
    "Industry": "industry",
    "Market Cap ($M)": "market_cap",
    "Enterprise Value ($M)": "enterprise_value",
    "Short Interest %": "short_interest_pct",
    "P/S - NTM": "ps_ntm",
    "P/S 3 Yr. Avg.": "ps_3yr_avg",
    "FCF Yield": "fcf_yield",
    "NTM Gross Margin": "ntm_gross_margin",
    "Gross Margin (3Yr. Avg.)": "gross_margin_3yr_avg",
    "NTM EBIT Margin": "ntm_ebit_margin",
    "EBIT Margin (3yr. Avg.)": "ebit_margin_3yr_avg",
    "Revenue CAGR (P2Y)": "rev_cagr_p2y",
    "Revenue CAGR (F2Y)": "rev_cagr_f2y",
    "ROIC (%)": "roic",
    "ROIC (3yr. Avg.)": "roic_3yr_avg",
    "Buy Recs": "buy_recs",
    "Hold Recs": "hold_recs",
    "Sell Recs": "sell_recs",
    "Leverage Ratio": "leverage_ratio",
    "52 Week High (%)": "week_52_high_pct",
    "52 Week Low (%)": "week_52_low_pct",
    "FCF ($M)": "fcf",
    "Adj. EBITDA ($M)": "adj_ebitda",
    "30 Day Avg. Volume ($M)": "avg_volume_30d",
    "Adj. EPS (FY-2)": "adj_eps_fy2",
    "Adj. EPS (FY-1)": "adj_eps_fy1",
    "Adj. EPS": "adj_eps",
    "Dil. EPS  (FY-2)": "dil_eps_fy2",
    "Dil. EPS (FY-1)": "dil_eps_fy1",
    "Dil. EPS (FY0)": "dil_eps_fy0",
    "Non-GAAP/GAAP EBIT": "non_gaap_gaap_ebit",
    "Dil. Wtd. Avg. Shares (FY-3 to FY0)": "dilution_p3y",
    "Net Debt ($M)": "net_debt",
    "Weighted Avg. Maturity": "weighted_avg_maturity",
    "TTM Cash Burn": "ttm_cash_burn",
    "Cash Balance": "cash_balance",
    "Available LOC": "available_loc",
    "Revenues TTM": "revenues_ttm",
    "Revenues TTM (T-1)": "revenues_ttm_t1",
    "Revenues TTM (T-2)": "revenues_ttm_t2",
    "Revenues T3M": "revenues_t3m",
    "Revenues T3M (T-1)": "revenues_t3m_t1",
    "COGS TTM": "cogs_ttm",
    "COGS TTM (T-1)": "cogs_ttm_t1",
    "COGS T3M": "cogs_t3m",
    "COGS T3M (T-1)": "cogs_t3m_t1",
    "SG&A": "sga",
    "SG&A (T-1)": "sga_t1",
    "Depr.": "depreciation",
    "Depr. (T-1)": "depreciation_t1",
    "Net Income": "net_income",
    "Net Income (T-1)": "net_income_t1",
    "CFO": "cfo",
    "CFO (T-1)": "cfo_t1",
    "Avg. Rec.": "avg_receivables",
    "Avg. Rec. (T-1)": "avg_receivables_t1",
    "Avg. Inventory": "avg_inventory",
    "Avg. Inventory (PY)": "avg_inventory_py",
    "Current Assets": "current_assets",
    "Current Assets (T-1)": "current_assets_t1",
    "PP&E": "ppe",
    "PP&E (T-1)": "ppe_t1",
    "LT Inv.": "lt_investments",
    "LT Inv. (T-1)": "lt_investments_t1",
    "Avg. Payables": "avg_payables",
    "Avg. Payables (T-1)": "avg_payables_t1",
    "Deferred Rev.": "deferred_revenue",
    "Deferred Rev (T-1)": "deferred_revenue_t1",
    "Debt to Assets": "debt_to_assets",
    "Debt to Assets (T-1)": "debt_to_assets_t1",
    "Total Assets": "total_assets",
    "Total Assets (T-1)": "total_assets_t1",
    "DSOs (3yr. Avg.)": "dsos_3yr_avg",
    "DIOS (3yr. Avg.)": "dios_3yr_avg",
    "DPOS (3yr. Avg.)": "dpos_3yr_avg",
    "FCF Yield (3yr. Avg.)": "fcf_yield_3yr_avg",
    "1W Perf.": "perf_1w",
    "1M Perf.": "perf_1m",
}

# Columns required for the pipeline to function. Missing any of these is fatal.
REQUIRED_COLUMNS = list(COLUMN_MAP.keys())

# Columns that should remain as strings (not coerced to numeric).
STRING_COLUMNS = {"ticker", "name", "sector", "industry"}

# The Bloomberg missing-data marker.
BLOOMBERG_NA = "#N/A N/A"

# Defense in depth for extract_ticker's output, mirroring rsi_ingest.py's
# same check — the real data tops out well below this.
_MAX_TICKER_LENGTH = 8

# Per-screen ingest configuration: sheet name, column map, required columns,
# and string columns for each screen's upload format. Only "short_screen" is
# defined today, but read_upload/validate_columns/clean_dataframe all take
# these as explicit parameters (no Short-Screen-shaped defaults) so a future
# screen with a different export shape plugs in here without touching the
# reading path itself.
SCREEN_INGEST_CONFIGS = {
    "short_screen": {
        "sheet_name": "Data",
        "column_map": COLUMN_MAP,
        "required_columns": REQUIRED_COLUMNS,
        "string_columns": STRING_COLUMNS,
        "expected_extension": ".xlsx",
    },
}


def clean_dataframe(df: pd.DataFrame, column_map: dict, string_columns: set) -> pd.DataFrame:
    """Rename columns, extract the ticker, handle Bloomberg NA strings, and
    coerce types.

    Args:
        df: DataFrame with original source column names (all str dtype).
        column_map: Source column name -> snake_case Python field name.
        string_columns: Snake_case column names to leave as strings (not
            coerced to numeric).

    Returns:
        Cleaned DataFrame with snake_case column names, a bare ticker (not
        the full Bloomberg identifier) if a ticker column is present, NaN
        for missing data, and numeric types where appropriate.
    """
    # Rename columns to snake_case
    df = df.rename(columns=column_map)

    # Extract the ticker from Bloomberg's full identifier ("AAPL US Equity")
    # rather than storing it whole — screen_membership and every other
    # screen's ticker column are already bare symbols, and a whole-identifier
    # ticker here silently breaks any ticker-based join against them (e.g.
    # short_screen vs. curated screens both carrying AAPL under different
    # keys).
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].apply(extract_ticker)
        if df["ticker"].str.len().max() > _MAX_TICKER_LENGTH:
            raise ValueError(
                "A ticker longer than expected survived extraction — check "
                "for a malformed identifier in the source export."
            )

    # Replace Bloomberg NA marker with NaN for all columns EXCEPT available_loc
    for col in df.columns:
        if col == "available_loc":
            # For available_loc, Bloomberg N/A means no credit line → 0
            df[col] = df[col].replace(BLOOMBERG_NA, "0")
        else:
            df[col] = df[col].replace(BLOOMBERG_NA, pd.NA)

    # Coerce numeric columns
    for col in df.columns:
        if col in string_columns:
            continue
        # Strip commas and whitespace that Bloomberg sometimes embeds
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def ingest(
    screen_id: str = "short_screen",
    upload_dir: str = None,
    db_path: str = "data/screener.db",
    config_path: str = CONFIG_PATH,
) -> None:
    """Run the full ingestion pipeline for one screen.

    Reads all CSV/Excel files in upload_dir using that screen's ingest
    config, validates required columns, cleans data, and writes to that
    screen's raw_data table. Also writes this screen's ticker universe to
    screen_membership, and syncs the screens registry from config.yaml.

    Args:
        screen_id: Which screen's ingest config to use, and which per-screen
            table to write.
        upload_dir: Directory containing this screen's export files.
            Defaults to data/uploads/<screen_id>.
        db_path: Path to the SQLite database file.
        config_path: Path to config.yaml (used for the screens registry).
    """
    config = load_config(config_path)
    screen_type = get_screen_type(config, screen_id)
    if screen_type != "quant_composite":
        raise ScreenTypeError(
            f"ingest() only supports quant_composite screens; {screen_id!r} "
            f"is type {screen_type!r}. Curated screens use "
            f"curated_ingest.ingest_curated() instead."
        )

    if upload_dir is None:
        upload_dir = os.path.join("data", "uploads", screen_id)

    try:
        ingest_cfg = SCREEN_INGEST_CONFIGS[screen_id]
    except KeyError:
        raise ScreenTypeError(
            f"ingest() has no registered ingest config for {screen_id!r}. "
            f"Known: {list(SCREEN_INGEST_CONFIGS)}"
        ) from None

    filepath = find_single_upload_file(upload_dir, ingest_cfg["expected_extension"])

    logger.info("Reading %s", filepath)
    df = read_upload(filepath, ingest_cfg["sheet_name"])
    validate_columns(df, ingest_cfg["required_columns"])
    cleaned = clean_dataframe(df, ingest_cfg["column_map"], ingest_cfg["string_columns"])
    log_summary(cleaned)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    sync_screens_registry(engine, config)

    raw_table = table_name("raw_data", screen_id)
    cleaned.to_sql(raw_table, engine, if_exists="replace", index=False)
    logger.info("Wrote %d rows to %s table at %s", len(cleaned), raw_table, db_path)

    membership_df = pd.DataFrame({"screen_id": screen_id, "ticker": cleaned["ticker"]})
    replace_screen_rows(engine, membership_df, "screen_membership", screen_id)
    logger.info(
        "Wrote %d rows to screen_membership for screen_id=%s", len(membership_df), screen_id
    )


if __name__ == "__main__":
    ingest()
