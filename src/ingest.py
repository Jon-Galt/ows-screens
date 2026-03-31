"""
Ingest raw Bloomberg CSV/Excel exports into SQLite.

Reads files from data/uploads/, maps Bloomberg column names to snake_case,
coerces types, handles "#N/A N/A" strings, and writes to the raw_data table.
"""

import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

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


def read_upload(filepath: str) -> pd.DataFrame:
    """Read a single CSV or Excel file into a DataFrame.

    Args:
        filepath: Path to the CSV or Excel file.

    Returns:
        Raw DataFrame with original Bloomberg column names.

    Raises:
        ValueError: If file extension is not .csv, .xlsx, or .xls.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath, dtype=str)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath, dtype=str, sheet_name="Data")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Expected .csv, .xlsx, or .xls")


def validate_columns(df: pd.DataFrame) -> None:
    """Check that all required Bloomberg columns are present.

    Args:
        df: DataFrame with original Bloomberg column names.

    Raises:
        KeyError: If any required columns are missing, with the list of missing names.
    """
    present = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} required column(s) in upload: {missing}"
        )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, handle Bloomberg NA strings, and coerce types.

    Args:
        df: DataFrame with original Bloomberg column names (all str dtype).

    Returns:
        Cleaned DataFrame with snake_case column names, NaN for missing data,
        and numeric types where appropriate.
    """
    # Rename columns to snake_case
    df = df.rename(columns=COLUMN_MAP)

    # Replace Bloomberg NA marker with NaN for all columns EXCEPT available_loc
    for col in df.columns:
        if col == "available_loc":
            # For available_loc, Bloomberg N/A means no credit line → 0
            df[col] = df[col].replace(BLOOMBERG_NA, "0")
        else:
            df[col] = df[col].replace(BLOOMBERG_NA, pd.NA)

    # Coerce numeric columns
    for col in df.columns:
        if col in STRING_COLUMNS:
            continue
        # Strip commas and whitespace that Bloomberg sometimes embeds
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def log_summary(df: pd.DataFrame) -> None:
    """Log a summary of the ingested data."""
    logger.info("Ingested %d rows, %d columns", len(df), len(df.columns))
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.1]
    if len(high_null) > 0:
        logger.info("Columns with >10%% null rate:")
        for col, rate in high_null.items():
            logger.info("  %s: %.1f%%", col, rate * 100)


def ingest(upload_dir: str = "data/uploads", db_path: str = "data/screener.db") -> None:
    """Run the full ingestion pipeline.

    Reads all CSV/Excel files in upload_dir, validates required columns,
    cleans data, and writes to the raw_data SQLite table.

    Args:
        upload_dir: Directory containing Bloomberg export files.
        db_path: Path to the SQLite database file.
    """
    files = [
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir)
        if f.lower().endswith((".csv", ".xlsx", ".xls"))
    ]
    if not files:
        logger.error("No CSV or Excel files found in %s", upload_dir)
        sys.exit(1)

    frames = []
    for filepath in files:
        logger.info("Reading %s", filepath)
        df = read_upload(filepath)
        validate_columns(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    cleaned = clean_dataframe(combined)
    log_summary(cleaned)

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    cleaned.to_sql("raw_data", engine, if_exists="replace", index=False)
    logger.info("Wrote %d rows to raw_data table at %s", len(cleaned), db_path)


if __name__ == "__main__":
    ingest()
