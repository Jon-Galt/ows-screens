"""
Score transformed data with percentile ranking and weighted composite.

Reads from the transformed_data SQLite table, percentile-ranks each factor
using scipy.stats.percentileofscore (kind='rank') to match Excel PERCENTRANK.INC,
applies factor weights from config.yaml, and writes to the scored_data table.

No web or database imports are used in calculation functions — they operate
on pandas DataFrames/Series only.
"""

import logging
import os

import numpy as np
import pandas as pd
import yaml
from scipy.stats import percentileofscore
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Path to config relative to project root
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load scoring configuration from config.yaml.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Factor scoring definitions
# ---------------------------------------------------------------------------
# Each entry maps factor_name -> (raw_metric_column, direction, nan_default_key).
#
# direction:
#   "straight" = higher raw value -> higher percentile -> more bearish for short thesis
#   "inverted" = 1 - percentile (lower raw value -> higher factor score)
#
# The nan_default is resolved at runtime from config.yaml:
#   - Factors listed in scoring.nan_default_zero_factors use 0.0
#   - All others use scoring.nan_default_standard (0.5)

FACTOR_DEFINITIONS = {
    # --- Valuation ---
    "abs_ps_factor": {
        "metric": "ps_diff",
        "direction": "straight",
        # Higher P/S premium vs. history = more bearish
    },
    "rel_ps_factor": {
        "metric": "ps_ntm",
        "direction": "straight",
        # Higher absolute P/S = more bearish
    },
    "abs_fcf_factor": {
        "metric": "fcf_yield_diff",
        "direction": "straight",
        # Positive diff = FCF yield declined vs. history = more bearish
    },
    "rel_fcf_factor": {
        "metric": "fcf_yield",
        "direction": "inverted",
        # Lower FCF yield = more expensive = more bearish (1 - pctile)
    },

    # --- Growth ---
    "decel_factor": {
        "metric": "growth_decel",
        "direction": "inverted",
        # More negative decel = growth falling faster = more bearish (1 - pctile)
    },
    "accel_factor": {
        "metric": "growth_accel",
        "direction": "straight",
        # Elevated expectations create asymmetric downside risk: companies where
        # consensus expects significant revenue acceleration vs. history are short
        # targets because high forward growth assumptions are vulnerable to
        # disappointment, creating asymmetric downside when they miss.
    },

    # --- Profitability ---
    "gm_factor": {
        "metric": "gm_diff",
        "direction": "straight",
        # Higher expected margin expansion = more priced for perfection
    },
    "ebit_factor": {
        "metric": "ebit_diff",
        "direction": "straight",
        # Higher expected EBIT expansion = more priced for perfection
    },

    # --- Balance Sheet ---
    "debt_ebitda_factor": {
        "metric": "leverage_ratio_calc",
        "direction": "straight",
        # Higher leverage = more bearish
    },
    "debt_sales_factor": {
        "metric": "debt_sales",
        "direction": "straight",
        # Higher debt/sales = more bearish
    },
    "debt_ev_factor": {
        "metric": "debt_ev",
        "direction": "straight",
        # Higher debt/EV = more bearish
    },
    "refi_risk_factor": {
        "metric": "weighted_avg_maturity",
        "direction": "inverted",
        # Shorter maturity = higher refi risk = more bearish (1 - pctile)
    },
    "liquidity_risk_factor": {
        "metric": "remaining_liquidity_years",
        "direction": "inverted",
        # Fewer years of runway = more bearish (1 - pctile)
    },

    # --- Cash Flow ---
    "fcf_conv_factor": {
        "metric": "fcf_conversion",
        "direction": "inverted",
        # Lower FCF conversion = worse cash quality = more bearish (1 - pctile)
    },
    "accrual_factor": {
        "metric": "accrual_ratio",
        "direction": "straight",
        # Divergence between CFO and Net Income signals heavy accrual influence
        # on reported earnings: a large gap (in either direction) between cash
        # flow and income indicates the financials are driven by non-cash items,
        # which the screen treats as a quality red flag. Higher ratio = more bearish.
    },
    "dso_factor": {
        "metric": "dso_pct_change",
        "direction": "straight",
        # Rising DSOs = receivables outpacing revenue = more bearish
    },
    "dio_factor": {
        "metric": "dio_pct_change",
        "direction": "straight",
        # Rising DIOs = inventory building = more bearish
    },
    "dpo_factor": {
        "metric": "dpo_pct_change",
        "direction": "inverted",
        # Declining payables = paying suppliers faster = more bearish (1 - pctile)
    },
    "def_rev_factor": {
        "metric": "deferred_rev_pct_change",
        "direction": "inverted",
        # Declining deferred revenue = fewer advance payments = more bearish (1 - pctile)
    },
    "dilution_factor": {
        "metric": "dilution_p3y",
        "direction": "straight",
        # More dilution = more bearish
    },

    # --- Non-GAAP ---
    "ebit_adj_factor": {
        "metric": "non_gaap_gaap_ebit",
        "direction": "straight",
        # Higher Non-GAAP/GAAP EBIT ratio = more aggressive adjustments
    },
    "eps_adj_factor": {
        "metric": "eps_adj_ratio",
        "direction": "straight",
        # Higher Adj EPS/GAAP EPS = more aggressive adjustments
    },

    # --- Sentiment ---
    "short_int_factor": {
        "metric": "short_interest_pct",
        "direction": "straight",
        # Higher short interest = more bearish sentiment
    },
    "ratings_factor": {
        "metric": "hold_sell_pct",
        "direction": "straight",
        # Higher % hold/sell = more negative analyst view
    },
}


def percentile_rank(arr: np.ndarray, val: float) -> float:
    """Compute PERCENTRANK.INC-compatible percentile for a single value.

    Excel's PERCENTRANK.INC uses the position of the first occurrence in the
    sorted array: (count of values strictly less than x) / (n - 1). This
    produces values between 0.0 and 1.0 inclusive.

    For tied values, Excel uses the MIN rank (first occurrence position), NOT
    the average rank. We use scipy.stats.percentileofscore(kind='strict')
    which returns (count < x) / n * 100, then convert:
        count_less = percentileofscore(kind='strict') / 100 * n
        PERCENTRANK.INC = count_less / (n - 1)

    Args:
        arr: Array of all non-NaN values in the universe for this factor.
        val: The value to rank.

    Returns:
        Percentile between 0.0 and 1.0 inclusive.
    """
    n = len(arr)
    if n <= 1:
        return 1.0 if n == 1 else np.nan
    # percentileofscore(kind='strict') returns (count < val) / n * 100
    count_less = percentileofscore(arr, val, kind="strict") / 100 * n
    return count_less / (n - 1)


def rank_factor(series: pd.Series, direction: str, nan_default: float) -> pd.Series:
    """Percentile-rank an entire factor column.

    Args:
        series: Raw metric values for all stocks.
        direction: "straight" or "inverted" (1 - percentile).
        nan_default: Value to assign when the raw metric is NaN.

    Returns:
        Series of factor scores between 0 and 1 (or nan_default for missing).
    """
    valid_mask = series.notna()
    valid_values = series[valid_mask].values

    if len(valid_values) == 0:
        return pd.Series(nan_default, index=series.index)

    result = pd.Series(nan_default, index=series.index, dtype=float)

    for idx in series.index[valid_mask]:
        pctile = percentile_rank(valid_values, series[idx])
        if direction == "inverted":
            pctile = 1 - pctile
        result[idx] = pctile

    return result


def compute_factor_scores(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute all 24 factor scores via percentile ranking.

    Args:
        df: DataFrame from transformed_data table.
        config: Parsed config.yaml dictionary.

    Returns:
        DataFrame with factor score columns added.
    """
    scoring_cfg = config["scoring"]
    nan_default_std = scoring_cfg["nan_default_standard"]
    zero_factors = set(scoring_cfg["nan_default_zero_factors"])

    for factor_name, defn in FACTOR_DEFINITIONS.items():
        metric_col = defn["metric"]
        direction = defn["direction"]
        nan_default = 0.0 if factor_name in zero_factors else nan_default_std

        raw = pd.to_numeric(df[metric_col], errors="coerce")
        df[factor_name] = rank_factor(raw, direction, nan_default)

    return df


def compute_overall_score(df: pd.DataFrame, config: dict) -> pd.Series:
    """Compute weighted composite score from all 24 factor scores.

    The overall score is the sum of (factor_weight * factor_score) across
    all 24 factors. Each category's weights sum to 1.0, so each category
    contributes a maximum of 1.0 to the overall score (max total = 7.0).

    Args:
        df: DataFrame with all factor score columns.
        config: Parsed config.yaml dictionary.

    Returns:
        Series of overall composite scores.
    """
    weights = config["factor_weights"]
    score = pd.Series(0.0, index=df.index, dtype=float)

    for factor_name in FACTOR_DEFINITIONS:
        weight = weights[factor_name]
        score += weight * df[factor_name]

    return score


def compute_mscore_flag(df: pd.DataFrame, config: dict) -> pd.Series:
    """Flag stocks with M-Score above the manipulation threshold.

    M-Score is NOT included in the composite overall score. It is a
    standalone indicator displayed separately.

    Args:
        df: DataFrame with mscore column.
        config: Parsed config.yaml dictionary.

    Returns:
        Boolean Series. True = potential earnings manipulation.
    """
    threshold = config["scoring"]["mscore_manipulation_threshold"]
    mscore = pd.to_numeric(df["mscore"], errors="coerce")
    return mscore > threshold


def run_scoring(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply all scoring logic to a transformed DataFrame.

    Args:
        df: DataFrame from transformed_data table.
        config: Parsed config.yaml dictionary.

    Returns:
        DataFrame with factor scores, overall score, and M-Score flag added.
    """
    df = compute_factor_scores(df, config)
    df["overall_score"] = compute_overall_score(df, config)
    df["mscore_flag"] = compute_mscore_flag(df, config)
    return df


def score(db_path: str = "data/screener.db", config_path: str = CONFIG_PATH) -> None:
    """Run the full scoring pipeline.

    Reads transformed_data from SQLite, computes percentile rankings and
    composite scores, and writes to the scored_data table.
    """
    config = load_config(config_path)
    engine = create_engine(f"sqlite:///{db_path}")

    logger.info("Reading transformed_data from %s", db_path)
    df = pd.read_sql_table("transformed_data", engine)
    logger.info("Loaded %d rows", len(df))

    df = run_scoring(df, config)

    df.to_sql("scored_data", engine, if_exists="replace", index=False)
    logger.info("Wrote %d rows to scored_data table", len(df))


if __name__ == "__main__":
    score()
