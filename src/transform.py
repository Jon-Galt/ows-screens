"""
Transform raw Bloomberg data into derived metrics.

Reads from the raw_data SQLite table, calculates all derived metrics as
standalone functions, and writes to the transformed_data table.

No web or database imports are used in calculation functions — they operate
on pandas DataFrames/Series only.
"""

import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valuation metrics
# ---------------------------------------------------------------------------

def calc_ps_diff(df: pd.DataFrame) -> pd.Series:
    """P/S NTM relative to 3-year average: (P/S NTM / P/S 3yr avg) - 1.

    Inputs: ps_ntm, ps_3yr_avg (both ratios, not percentages).
    Output: Decimal ratio. Positive means current P/S exceeds historical avg.
    Edge cases: Returns NaN if either input is NaN or ps_3yr_avg is zero.
    """
    ps_ntm = pd.to_numeric(df["ps_ntm"], errors="coerce")
    ps_3yr = pd.to_numeric(df["ps_3yr_avg"], errors="coerce")
    return np.where(ps_3yr == 0, np.nan, ps_ntm / ps_3yr - 1)


def calc_fcf_yield_diff(df: pd.DataFrame) -> pd.Series:
    """FCF yield deterioration: FCF Yield 3yr avg - FCF Yield LTM.

    Inputs: fcf_yield_3yr_avg, fcf_yield (both decimals, e.g. 0.05 = 5%).
    Output: Decimal. Positive means current yield is lower than historical
            (i.e., valuation has become more expensive on a cash flow basis).
    Edge cases: Returns NaN if either input is NaN.
    """
    fcf_3yr = pd.to_numeric(df["fcf_yield_3yr_avg"], errors="coerce")
    fcf_ltm = pd.to_numeric(df["fcf_yield"], errors="coerce")
    return fcf_3yr - fcf_ltm


# ---------------------------------------------------------------------------
# Growth metrics
# ---------------------------------------------------------------------------

def calc_yoy_growth_ttm(df: pd.DataFrame) -> pd.Series:
    """Year-over-year revenue growth (TTM): Revenues TTM / Revenues TTM (T-1) - 1.

    Inputs: revenues_ttm, revenues_ttm_t1 (both in $M).
    Output: Decimal growth rate. 0.10 = 10% growth.
    Edge cases: Returns NaN if revenues_ttm_t1 is zero or NaN.
    """
    rev = pd.to_numeric(df["revenues_ttm"], errors="coerce")
    rev_t1 = pd.to_numeric(df["revenues_ttm_t1"], errors="coerce")
    return np.where(rev_t1 == 0, np.nan, rev / rev_t1 - 1)


def calc_yoy_growth_ttm_t1(df: pd.DataFrame) -> pd.Series:
    """Year-over-year revenue growth one year ago: Revenues TTM (T-1) / Revenues TTM (T-2) - 1.

    Inputs: revenues_ttm_t1, revenues_ttm_t2 (both in $M).
    Output: Decimal growth rate.
    Edge cases: Returns NaN if revenues_ttm_t2 is zero or NaN.
    """
    rev_t1 = pd.to_numeric(df["revenues_ttm_t1"], errors="coerce")
    rev_t2 = pd.to_numeric(df["revenues_ttm_t2"], errors="coerce")
    return np.where(rev_t2 == 0, np.nan, rev_t1 / rev_t2 - 1)


def calc_growth_decel(df: pd.DataFrame) -> pd.Series:
    """Revenue growth deceleration: Y/Y Growth TTM - Y/Y Growth TTM-1.

    Inputs: Pre-computed yoy_growth_ttm and yoy_growth_ttm_t1 columns.
    Output: Decimal. Negative means growth is decelerating.
    Edge cases: Returns NaN if either input is NaN.
    """
    return pd.to_numeric(df["yoy_growth_ttm"], errors="coerce") - pd.to_numeric(
        df["yoy_growth_ttm_t1"], errors="coerce"
    )


def calc_growth_accel(df: pd.DataFrame) -> pd.Series:
    """Expected revenue growth acceleration: Rev CAGR F2Y - Rev CAGR P2Y.

    Inputs: rev_cagr_f2y, rev_cagr_p2y (both decimals from Bloomberg).
    Output: Decimal. Positive means consensus expects acceleration vs. history.
    Edge cases: Returns NaN if either input is NaN.
    """
    f2y = pd.to_numeric(df["rev_cagr_f2y"], errors="coerce")
    p2y = pd.to_numeric(df["rev_cagr_p2y"], errors="coerce")
    return f2y - p2y


# ---------------------------------------------------------------------------
# Profitability metrics
# ---------------------------------------------------------------------------

def calc_gm_diff(df: pd.DataFrame) -> pd.Series:
    """Gross margin expansion: NTM Gross Margin - Gross Margin 3yr avg.

    Inputs: ntm_gross_margin, gross_margin_3yr_avg (both decimals).
    Output: Decimal. Positive means margins expected to expand.
    Edge cases: Returns NaN if either input is NaN.
    """
    ntm = pd.to_numeric(df["ntm_gross_margin"], errors="coerce")
    avg = pd.to_numeric(df["gross_margin_3yr_avg"], errors="coerce")
    return ntm - avg


def calc_ebit_diff(df: pd.DataFrame) -> pd.Series:
    """EBIT margin expansion: NTM EBIT Margin - EBIT Margin 3yr avg.

    Inputs: ntm_ebit_margin, ebit_margin_3yr_avg (both decimals).
    Output: Decimal. Positive means margins expected to expand.
    Edge cases: Returns NaN if either input is NaN.
    """
    ntm = pd.to_numeric(df["ntm_ebit_margin"], errors="coerce")
    avg = pd.to_numeric(df["ebit_margin_3yr_avg"], errors="coerce")
    return ntm - avg


# ---------------------------------------------------------------------------
# Balance Sheet metrics
# ---------------------------------------------------------------------------

def calc_leverage_ratio(df: pd.DataFrame) -> pd.Series:
    """Net Debt / Adj. EBITDA. Only meaningful when both are positive.

    Inputs: net_debt, adj_ebitda (both in $M).
    Output: Ratio. Higher = more leveraged.
    Edge cases: Returns NaN if net_debt <= 0, adj_ebitda <= 0, or ratio < 0.
              Matches Excel: IF(ratio < 0, "N/A", ratio).
    """
    debt = pd.to_numeric(df["net_debt"], errors="coerce")
    ebitda = pd.to_numeric(df["adj_ebitda"], errors="coerce")
    ratio = debt / ebitda.replace(0, np.nan)
    # Excel returns N/A when ratio is negative (net cash or negative EBITDA)
    return np.where(ratio < 0, np.nan, ratio)


def calc_debt_sales(df: pd.DataFrame) -> pd.Series:
    """Net Debt / TTM Revenues. Only meaningful when both are positive.

    Inputs: net_debt, revenues_ttm (both in $M).
    Output: Ratio. Higher = more debt relative to revenue.
    Edge cases: Returns NaN if net_debt <= 0, revenues_ttm <= 0, or ratio < 0.
    """
    debt = pd.to_numeric(df["net_debt"], errors="coerce")
    rev = pd.to_numeric(df["revenues_ttm"], errors="coerce")
    ratio = debt / rev.replace(0, np.nan)
    return np.where(ratio < 0, np.nan, ratio)


def calc_enterprise_value(df: pd.DataFrame) -> pd.Series:
    """Enterprise Value = Market Cap + Net Debt.

    Inputs: market_cap, net_debt (both in $M).
    Output: EV in $M.
    Edge cases: Returns NaN if market_cap is NaN.
    """
    mcap = pd.to_numeric(df["market_cap"], errors="coerce")
    debt = pd.to_numeric(df["net_debt"], errors="coerce")
    return mcap + debt


def calc_debt_ev(df: pd.DataFrame) -> pd.Series:
    """Net Debt / Enterprise Value (computed as Market Cap + Net Debt).

    Inputs: net_debt, enterprise_value_calc (pre-computed in run_transforms).
    Output: Ratio. Can be negative if net cash exceeds market cap impact.
    Edge cases: Returns NaN if enterprise_value_calc is zero or NaN.
    Note: Uses computed EV (Market Cap + Net Debt), not the raw Bloomberg
          Enterprise Value column, matching the Screen sheet formula.
    """
    debt = pd.to_numeric(df["net_debt"], errors="coerce")
    ev = pd.to_numeric(df["enterprise_value_calc"], errors="coerce")
    return np.where(ev == 0, np.nan, debt / ev)


def calc_liquidity(df: pd.DataFrame) -> pd.Series:
    """Total liquidity: Cash Balance + Available LOC.

    Inputs: cash_balance, available_loc (both in $M).
              available_loc is already 0 for Bloomberg N/A (handled in ingest).
    Output: Liquidity in $M.
    Edge cases: Returns NaN only if cash_balance is NaN.
    """
    cash = pd.to_numeric(df["cash_balance"], errors="coerce")
    loc = pd.to_numeric(df["available_loc"], errors="coerce").fillna(0)
    return cash + loc


def calc_remaining_liquidity_years(df: pd.DataFrame) -> pd.Series:
    """Years of remaining liquidity for cash-burning companies.

    Formula: Liquidity / (-FCF), only when FCF < 0.
    Inputs: Pre-computed liquidity column, fcf (in $M).
    Output: Years of runway. Only defined for cash burners (FCF < 0).
    Edge cases: Returns NaN if FCF >= 0 (profitable companies not at risk).
    """
    liquidity = pd.to_numeric(df["liquidity"], errors="coerce")
    fcf = pd.to_numeric(df["fcf"], errors="coerce")
    return np.where(fcf < 0, liquidity / (-fcf), np.nan)


# ---------------------------------------------------------------------------
# Cash Flow metrics
# ---------------------------------------------------------------------------

def calc_fcf_conversion(df: pd.DataFrame) -> pd.Series:
    """FCF conversion: FCF / Adj. EBITDA.

    Inputs: fcf, adj_ebitda (both in $M).
    Output: Ratio. Higher = better cash conversion.
    Edge cases: Returns NaN if adj_ebitda is zero or NaN.
    """
    fcf = pd.to_numeric(df["fcf"], errors="coerce")
    ebitda = pd.to_numeric(df["adj_ebitda"], errors="coerce")
    return np.where(ebitda == 0, np.nan, fcf / ebitda)


def calc_accrual_ratio(df: pd.DataFrame) -> pd.Series:
    """Accrual ratio: (CFO - Net Income) / CFO.

    Inputs: cfo, net_income (both in $M).
    Output: Ratio. Higher values indicate larger divergence between cash flow
            and reported earnings, signaling heavy accrual influence.
    Edge cases: Returns NaN if CFO is zero or NaN.
    """
    cfo = pd.to_numeric(df["cfo"], errors="coerce")
    ni = pd.to_numeric(df["net_income"], errors="coerce")
    return np.where(cfo == 0, np.nan, (cfo - ni) / cfo)


def calc_dsos(df: pd.DataFrame) -> pd.Series:
    """Days Sales Outstanding (current quarter): (Avg Receivables / Revenues T3M) * 90.

    Inputs: avg_receivables, revenues_t3m (both in $M).
    Output: Days.
    Edge cases: Returns NaN if revenues_t3m is zero or NaN.
    """
    rec = pd.to_numeric(df["avg_receivables"], errors="coerce")
    rev = pd.to_numeric(df["revenues_t3m"], errors="coerce")
    return np.where(rev == 0, np.nan, (rec / rev) * 90)


def calc_dsos_py(df: pd.DataFrame) -> pd.Series:
    """Days Sales Outstanding (prior year quarter): (Avg Receivables T-1 / Revenues T3M T-1) * 90.

    Inputs: avg_receivables_t1, revenues_t3m_t1 (both in $M).
    Output: Days.
    Edge cases: Returns NaN if revenues_t3m_t1 is zero or NaN.
    """
    rec = pd.to_numeric(df["avg_receivables_t1"], errors="coerce")
    rev = pd.to_numeric(df["revenues_t3m_t1"], errors="coerce")
    return np.where(rev == 0, np.nan, (rec / rev) * 90)


def calc_dso_pct_change(df: pd.DataFrame) -> pd.Series:
    """Percentage change in DSOs: DSOs / DSOs PY - 1.

    Inputs: Pre-computed dsos, dsos_py columns.
    Output: Decimal. Positive means receivables growing faster than revenue.
    Edge cases: Returns NaN if dsos_py is zero or NaN.
    """
    dsos = pd.to_numeric(df["dsos"], errors="coerce")
    dsos_py = pd.to_numeric(df["dsos_py"], errors="coerce")
    return np.where(dsos_py == 0, np.nan, dsos / dsos_py - 1)


def calc_dios(df: pd.DataFrame) -> pd.Series:
    """Days Inventory Outstanding (current quarter): (Avg Inventory / COGS T3M) * 90.

    Inputs: avg_inventory, cogs_t3m (both in $M).
    Output: Days.
    Edge cases: Returns NaN if cogs_t3m is zero or NaN.
    """
    inv = pd.to_numeric(df["avg_inventory"], errors="coerce")
    cogs = pd.to_numeric(df["cogs_t3m"], errors="coerce")
    return np.where(cogs == 0, np.nan, (inv / cogs) * 90)


def calc_dios_t1(df: pd.DataFrame) -> pd.Series:
    """Days Inventory Outstanding (prior year): (Avg Inventory PY / COGS T3M T-1) * 90.

    Inputs: avg_inventory_py, cogs_t3m_t1 (both in $M).
    Output: Days.
    Edge cases: Returns NaN if cogs_t3m_t1 is zero or NaN.
    """
    inv = pd.to_numeric(df["avg_inventory_py"], errors="coerce")
    cogs = pd.to_numeric(df["cogs_t3m_t1"], errors="coerce")
    return np.where(cogs == 0, np.nan, (inv / cogs) * 90)


def calc_dio_pct_change(df: pd.DataFrame) -> pd.Series:
    """Percentage change in DIOs: DIOs / DIOs T-1 - 1.

    Inputs: Pre-computed dios, dios_t1 columns.
    Output: Decimal. Positive means inventory building relative to COGS.
    Edge cases: Returns NaN if dios_t1 is zero or NaN.
    """
    dios = pd.to_numeric(df["dios"], errors="coerce")
    dios_t1 = pd.to_numeric(df["dios_t1"], errors="coerce")
    return np.where(dios_t1 == 0, np.nan, dios / dios_t1 - 1)


def calc_dpos(df: pd.DataFrame) -> pd.Series:
    """Days Payable Outstanding (current quarter): (Avg Payables / COGS T3M) * 90.

    Inputs: avg_payables, cogs_t3m (both in $M).
    Output: Days.
    Edge cases: Returns NaN if cogs_t3m is zero or NaN.
    """
    pay = pd.to_numeric(df["avg_payables"], errors="coerce")
    cogs = pd.to_numeric(df["cogs_t3m"], errors="coerce")
    return np.where(cogs == 0, np.nan, (pay / cogs) * 90)


def calc_dpos_t1(df: pd.DataFrame) -> pd.Series:
    """Days Payable Outstanding (prior year): (Avg Payables T-1 / COGS T3M T-1) * 90.

    Inputs: avg_payables_t1, cogs_t3m_t1 (both in $M).
    Output: Days.
    Edge cases: Returns NaN if cogs_t3m_t1 is zero or NaN.
    """
    pay = pd.to_numeric(df["avg_payables_t1"], errors="coerce")
    cogs = pd.to_numeric(df["cogs_t3m_t1"], errors="coerce")
    return np.where(cogs == 0, np.nan, (pay / cogs) * 90)


def calc_dpo_pct_change(df: pd.DataFrame) -> pd.Series:
    """Percentage change in DPOs: DPOs / DPOs T-1 - 1.

    Inputs: Pre-computed dpos, dpos_t1 columns.
    Output: Decimal. Positive means payables growing (paying suppliers slower).
    Edge cases: Returns NaN if dpos_t1 is zero or NaN.
    """
    dpos = pd.to_numeric(df["dpos"], errors="coerce")
    dpos_t1 = pd.to_numeric(df["dpos_t1"], errors="coerce")
    return np.where(dpos_t1 == 0, np.nan, dpos / dpos_t1 - 1)


def calc_days_deferred_rev(df: pd.DataFrame) -> pd.Series:
    """Days Deferred Revenue (current): (Deferred Rev / Revenues T3M) * 90.

    Inputs: deferred_revenue, revenues_t3m (both in $M).
    Output: Days.
    Edge cases: Returns NaN if revenues_t3m is zero or NaN.
    """
    drev = pd.to_numeric(df["deferred_revenue"], errors="coerce")
    rev = pd.to_numeric(df["revenues_t3m"], errors="coerce")
    return np.where(rev == 0, np.nan, (drev / rev) * 90)


def calc_days_deferred_rev_t1(df: pd.DataFrame) -> pd.Series:
    """Days Deferred Revenue (prior year): (Deferred Rev T-1 / Revenues T3M T-1) * 90.

    Inputs: deferred_revenue_t1, revenues_t3m_t1 (both in $M).
    Output: Days.
    Edge cases: Returns NaN if revenues_t3m_t1 is zero or NaN.
    """
    drev = pd.to_numeric(df["deferred_revenue_t1"], errors="coerce")
    rev = pd.to_numeric(df["revenues_t3m_t1"], errors="coerce")
    return np.where(rev == 0, np.nan, (drev / rev) * 90)


def calc_deferred_rev_pct_change(df: pd.DataFrame) -> pd.Series:
    """Percentage change in Days Deferred Revenue: current / prior - 1.

    Inputs: Pre-computed days_deferred_rev, days_deferred_rev_t1 columns.
    Output: Decimal. Positive means deferred revenue growing relative to sales.
    Edge cases: Returns NaN if days_deferred_rev_t1 is zero or NaN.
    """
    curr = pd.to_numeric(df["days_deferred_rev"], errors="coerce")
    prev = pd.to_numeric(df["days_deferred_rev_t1"], errors="coerce")
    return np.where(prev == 0, np.nan, curr / prev - 1)


# ---------------------------------------------------------------------------
# Non-GAAP metrics
# ---------------------------------------------------------------------------

def calc_eps_adj_ratio(df: pd.DataFrame) -> pd.Series:
    """Non-GAAP EPS adjustment ratio: Adj. EPS / Dil. EPS (FY0).

    Inputs: adj_eps, dil_eps_fy0.
    Output: Ratio. Values > 1 mean Non-GAAP EPS exceeds GAAP EPS.
    Edge cases: Returns NaN if dil_eps_fy0 is zero or NaN.
    """
    adj = pd.to_numeric(df["adj_eps"], errors="coerce")
    gaap = pd.to_numeric(df["dil_eps_fy0"], errors="coerce")
    return np.where(gaap == 0, np.nan, adj / gaap)


# ---------------------------------------------------------------------------
# Sentiment metrics
# ---------------------------------------------------------------------------

def calc_hold_sell_pct(df: pd.DataFrame) -> pd.Series:
    """Proportion of negative analyst ratings: (Hold + Sell) / Total.

    Inputs: buy_recs, hold_recs, sell_recs (integer counts).
    Output: Decimal between 0 and 1.
    Edge cases: Returns NaN if total recommendations is zero.
    """
    buy = pd.to_numeric(df["buy_recs"], errors="coerce")
    hold = pd.to_numeric(df["hold_recs"], errors="coerce")
    sell = pd.to_numeric(df["sell_recs"], errors="coerce")
    total = buy + hold + sell
    return np.where(total == 0, np.nan, (hold + sell) / total)


# ---------------------------------------------------------------------------
# M-Score (Beneish) components
# ---------------------------------------------------------------------------

def calc_dsri(df: pd.DataFrame) -> pd.Series:
    """Days Sales in Receivables Index: DSOs / DSOs PY.

    Inputs: Pre-computed dsos, dsos_py columns.
    Output: Ratio. > 1 means receivables growing relative to revenue.
    Edge cases: Returns NaN if dsos_py is zero or NaN.
    """
    dsos = pd.to_numeric(df["dsos"], errors="coerce")
    dsos_py = pd.to_numeric(df["dsos_py"], errors="coerce")
    return np.where(dsos_py == 0, np.nan, dsos / dsos_py)


def calc_gmi(df: pd.DataFrame) -> pd.Series:
    """Gross Margin Index: GM% T3M prior year / GM% T3M current.

    Formula: ((Rev_T3M_T1 - COGS_T3M_T1) / Rev_T3M_T1) / ((Rev_T3M - COGS_T3M) / Rev_T3M)
    Inputs: revenues_t3m, revenues_t3m_t1, cogs_t3m, cogs_t3m_t1.
    Output: Ratio. > 1 means margins are declining.
    Edge cases: Returns NaN if either quarter's revenue is zero.
    """
    rev = pd.to_numeric(df["revenues_t3m"], errors="coerce")
    rev_t1 = pd.to_numeric(df["revenues_t3m_t1"], errors="coerce")
    cogs = pd.to_numeric(df["cogs_t3m"], errors="coerce")
    cogs_t1 = pd.to_numeric(df["cogs_t3m_t1"], errors="coerce")
    gm_curr = np.where(rev == 0, np.nan, (rev - cogs) / rev)
    gm_prior = np.where(rev_t1 == 0, np.nan, (rev_t1 - cogs_t1) / rev_t1)
    return np.where(
        (np.isnan(gm_curr) | (gm_curr == 0)),
        np.nan,
        gm_prior / gm_curr,
    )


def calc_aqi(df: pd.DataFrame) -> pd.Series:
    """Asset Quality Index.

    Formula: (1 - (CA + PP&E + LT Inv) / TA) / (1 - (CA_T1 + PP&E_T1 + LT Inv_T1) / TA_T1)
    Inputs: current_assets, ppe, lt_investments, total_assets (current and T-1).
    Output: Ratio. > 1 means proportion of soft/intangible assets is growing.
    Edge cases: Returns NaN if total assets is zero or denominator is zero.
    """
    ca = pd.to_numeric(df["current_assets"], errors="coerce")
    ppe = pd.to_numeric(df["ppe"], errors="coerce")
    lti = pd.to_numeric(df["lt_investments"], errors="coerce")
    ta = pd.to_numeric(df["total_assets"], errors="coerce")

    ca_t1 = pd.to_numeric(df["current_assets_t1"], errors="coerce")
    ppe_t1 = pd.to_numeric(df["ppe_t1"], errors="coerce")
    lti_t1 = pd.to_numeric(df["lt_investments_t1"], errors="coerce")
    ta_t1 = pd.to_numeric(df["total_assets_t1"], errors="coerce")

    aq_curr = np.where(ta == 0, np.nan, 1 - (ca + ppe + lti) / ta)
    aq_prior = np.where(ta_t1 == 0, np.nan, 1 - (ca_t1 + ppe_t1 + lti_t1) / ta_t1)
    return np.where(
        (np.isnan(aq_prior) | (aq_prior == 0)),
        np.nan,
        aq_curr / aq_prior,
    )


def calc_sgi(df: pd.DataFrame) -> pd.Series:
    """Sales Growth Index: Revenues T3M / Revenues T3M T-1.

    Inputs: revenues_t3m, revenues_t3m_t1 (both in $M).
    Output: Ratio. > 1 means quarterly revenue is growing.
    Edge cases: Returns NaN if revenues_t3m_t1 is zero or NaN.
    """
    rev = pd.to_numeric(df["revenues_t3m"], errors="coerce")
    rev_t1 = pd.to_numeric(df["revenues_t3m_t1"], errors="coerce")
    return np.where(rev_t1 == 0, np.nan, rev / rev_t1)


def calc_depi(df: pd.DataFrame) -> pd.Series:
    """Depreciation Index.

    Formula: (Depr_T1 / (PP&E_T1 + Depr_T1)) / (Depr / (Depr + PP&E))
    Inputs: depreciation, depreciation_t1, ppe, ppe_t1.
    Output: Ratio. > 1 means depreciation rate is declining (red flag).
    Edge cases: Returns NaN if either denominator is zero.
    """
    depr = pd.to_numeric(df["depreciation"], errors="coerce")
    depr_t1 = pd.to_numeric(df["depreciation_t1"], errors="coerce")
    ppe = pd.to_numeric(df["ppe"], errors="coerce")
    ppe_t1 = pd.to_numeric(df["ppe_t1"], errors="coerce")

    denom_curr = depr + ppe
    denom_prior = ppe_t1 + depr_t1
    rate_curr = np.where(denom_curr == 0, np.nan, depr / denom_curr)
    rate_prior = np.where(denom_prior == 0, np.nan, depr_t1 / denom_prior)
    return np.where(
        (np.isnan(rate_curr) | (rate_curr == 0)),
        np.nan,
        rate_prior / rate_curr,
    )


def calc_sgai(df: pd.DataFrame) -> pd.Series:
    """SG&A Index: (SG&A / Revenue TTM) / (SG&A T-1 / Revenue TTM T-1).

    Inputs: sga, sga_t1, revenues_ttm, revenues_ttm_t1.
    Output: Ratio. > 1 means SG&A growing faster than revenue.
    Edge cases: Returns NaN if either revenue is zero.
    """
    sga = pd.to_numeric(df["sga"], errors="coerce")
    sga_t1 = pd.to_numeric(df["sga_t1"], errors="coerce")
    rev = pd.to_numeric(df["revenues_ttm"], errors="coerce")
    rev_t1 = pd.to_numeric(df["revenues_ttm_t1"], errors="coerce")

    ratio_curr = np.where(rev == 0, np.nan, sga / rev)
    ratio_prior = np.where(rev_t1 == 0, np.nan, sga_t1 / rev_t1)
    return np.where(
        (np.isnan(ratio_prior) | (ratio_prior == 0)),
        np.nan,
        ratio_curr / ratio_prior,
    )


def calc_lvgi(df: pd.DataFrame) -> pd.Series:
    """Leverage Index: Debt to Assets / Debt to Assets T-1.

    Inputs: debt_to_assets, debt_to_assets_t1 (both ratios).
    Output: Ratio. > 1 means leverage increasing.
    Edge cases: Returns NaN if debt_to_assets_t1 is zero or NaN.
    """
    dta = pd.to_numeric(df["debt_to_assets"], errors="coerce")
    dta_t1 = pd.to_numeric(df["debt_to_assets_t1"], errors="coerce")
    return np.where(dta_t1 == 0, np.nan, dta / dta_t1)


def calc_tata(df: pd.DataFrame) -> pd.Series:
    """Total Accruals to Total Assets: (Net Income - CFO) / Total Assets.

    Inputs: net_income, cfo, total_assets (all in $M).
    Output: Ratio. Higher = more accrual-based earnings.
    Edge cases: Returns NaN if total_assets is zero or NaN.
    """
    ni = pd.to_numeric(df["net_income"], errors="coerce")
    cfo = pd.to_numeric(df["cfo"], errors="coerce")
    ta = pd.to_numeric(df["total_assets"], errors="coerce")
    return np.where(ta == 0, np.nan, (ni - cfo) / ta)


def calc_mscore(df: pd.DataFrame) -> pd.Series:
    """Beneish M-Score from 8 component variables.

    Formula: -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
             + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

    Inputs: Pre-computed dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata columns.
    Output: M-Score value. > -2.22 indicates potential earnings manipulation.
    Edge cases: Returns NaN if any component is NaN.
    """
    return (
        -4.84
        + 0.920 * pd.to_numeric(df["dsri"], errors="coerce")
        + 0.528 * pd.to_numeric(df["gmi"], errors="coerce")
        + 0.404 * pd.to_numeric(df["aqi"], errors="coerce")
        + 0.892 * pd.to_numeric(df["sgi"], errors="coerce")
        + 0.115 * pd.to_numeric(df["depi"], errors="coerce")
        - 0.172 * pd.to_numeric(df["sgai"], errors="coerce")
        + 4.679 * pd.to_numeric(df["tata"], errors="coerce")
        - 0.327 * pd.to_numeric(df["lvgi"], errors="coerce")
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all transforms to a raw_data DataFrame.

    Args:
        df: DataFrame from the raw_data SQLite table.

    Returns:
        DataFrame with all original columns plus derived metric columns.
    """
    # Valuation
    df["ps_diff"] = calc_ps_diff(df)
    df["fcf_yield_diff"] = calc_fcf_yield_diff(df)

    # Growth
    df["yoy_growth_ttm"] = calc_yoy_growth_ttm(df)
    df["yoy_growth_ttm_t1"] = calc_yoy_growth_ttm_t1(df)
    df["growth_decel"] = calc_growth_decel(df)
    df["growth_accel"] = calc_growth_accel(df)

    # Profitability
    df["gm_diff"] = calc_gm_diff(df)
    df["ebit_diff"] = calc_ebit_diff(df)

    # Balance Sheet
    df["leverage_ratio_calc"] = calc_leverage_ratio(df)
    df["debt_sales"] = calc_debt_sales(df)
    df["enterprise_value_calc"] = calc_enterprise_value(df)
    df["debt_ev"] = calc_debt_ev(df)
    df["liquidity"] = calc_liquidity(df)
    df["remaining_liquidity_years"] = calc_remaining_liquidity_years(df)

    # Cash Flow — intermediate metrics first, then pct changes
    df["fcf_conversion"] = calc_fcf_conversion(df)
    df["accrual_ratio"] = calc_accrual_ratio(df)
    df["dsos"] = calc_dsos(df)
    df["dsos_py"] = calc_dsos_py(df)
    df["dso_pct_change"] = calc_dso_pct_change(df)
    df["dios"] = calc_dios(df)
    df["dios_t1"] = calc_dios_t1(df)
    df["dio_pct_change"] = calc_dio_pct_change(df)
    df["dpos"] = calc_dpos(df)
    df["dpos_t1"] = calc_dpos_t1(df)
    df["dpo_pct_change"] = calc_dpo_pct_change(df)
    df["days_deferred_rev"] = calc_days_deferred_rev(df)
    df["days_deferred_rev_t1"] = calc_days_deferred_rev_t1(df)
    df["deferred_rev_pct_change"] = calc_deferred_rev_pct_change(df)

    # Non-GAAP
    df["eps_adj_ratio"] = calc_eps_adj_ratio(df)

    # Sentiment
    df["hold_sell_pct"] = calc_hold_sell_pct(df)

    # M-Score components
    df["dsri"] = calc_dsri(df)
    df["gmi"] = calc_gmi(df)
    df["aqi"] = calc_aqi(df)
    df["sgi"] = calc_sgi(df)
    df["depi"] = calc_depi(df)
    df["sgai"] = calc_sgai(df)
    df["lvgi"] = calc_lvgi(df)
    df["tata"] = calc_tata(df)
    df["mscore"] = calc_mscore(df)

    return df


def transform(db_path: str = "data/screener.db") -> None:
    """Run the full transform pipeline.

    Reads raw_data from SQLite, computes all derived metrics,
    and writes to the transformed_data table.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    logger.info("Reading raw_data from %s", db_path)
    df = pd.read_sql_table("raw_data", engine)
    logger.info("Loaded %d rows", len(df))

    df = run_transforms(df)

    df.to_sql("transformed_data", engine, if_exists="replace", index=False)
    logger.info("Wrote %d rows to transformed_data table", len(df))


if __name__ == "__main__":
    transform()
