"""
Unit tests for all transform functions in src/transform.py.

Each test uses a small synthetic DataFrame with known inputs and expected outputs.
Coverage: happy path, zero-denominator / division-by-zero, all-NaN input,
and negative values where meaningful.
"""

import numpy as np
import pandas as pd
import pytest

from src.transform import (
    calc_accrual_ratio,
    calc_aqi,
    calc_days_deferred_rev,
    calc_days_deferred_rev_t1,
    calc_debt_ev,
    calc_debt_sales,
    calc_deferred_rev_pct_change,
    calc_depi,
    calc_dio_pct_change,
    calc_dios,
    calc_dios_t1,
    calc_dpo_pct_change,
    calc_dpos,
    calc_dpos_t1,
    calc_dso_pct_change,
    calc_dsos,
    calc_dsos_py,
    calc_dsri,
    calc_ebit_diff,
    calc_enterprise_value,
    calc_eps_adj_ratio,
    calc_fcf_conversion,
    calc_fcf_yield_diff,
    calc_gm_diff,
    calc_gmi,
    calc_growth_accel,
    calc_growth_decel,
    calc_hold_sell_pct,
    calc_leverage_ratio,
    calc_liquidity,
    calc_lvgi,
    calc_mscore,
    calc_ps_diff,
    calc_remaining_liquidity_years,
    calc_sgai,
    calc_sgi,
    calc_tata,
    calc_yoy_growth_ttm,
    calc_yoy_growth_ttm_t1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_df():
    """A 5-row DataFrame with reasonable numeric values for testing."""
    return pd.DataFrame({
        "ps_ntm": [5.0, 10.0, 2.0, np.nan, 8.0],
        "ps_3yr_avg": [4.0, 5.0, 2.0, 3.0, 0.0],
        "fcf_yield": [0.05, 0.02, 0.10, np.nan, 0.03],
        "fcf_yield_3yr_avg": [0.04, 0.06, 0.08, np.nan, 0.05],
        "revenues_ttm": [1000, 500, 0, 200, np.nan],
        "revenues_ttm_t1": [900, 600, 100, 0, 300],
        "revenues_ttm_t2": [800, 700, 200, 100, 0],
        "rev_cagr_f2y": [0.10, 0.05, np.nan, 0.20, 0.15],
        "rev_cagr_p2y": [0.08, 0.10, 0.05, np.nan, 0.12],
        "ntm_gross_margin": [0.40, 0.30, 0.50, np.nan, 0.35],
        "gross_margin_3yr_avg": [0.35, 0.32, 0.45, 0.40, np.nan],
        "ntm_ebit_margin": [0.15, 0.10, 0.20, np.nan, 0.12],
        "ebit_margin_3yr_avg": [0.12, 0.15, 0.18, 0.10, np.nan],
        "net_debt": [500, -200, 300, 0, 100],
        "adj_ebitda": [200, 100, 0, -50, 150],
        "market_cap": [5000, 3000, 1000, 2000, 4000],
        "fcf": [100, -50, 200, 0, -30],
        "cash_balance": [300, 500, 100, 200, np.nan],
        "available_loc": [100, np.nan, 0, 50, 200],
        "cfo": [150, 0, 80, -20, 100],
        "net_income": [100, 50, 80, -10, 60],
        "avg_receivables": [50, 30, 0, 20, np.nan],
        "revenues_t3m": [250, 0, 100, 50, 300],
        "avg_receivables_t1": [40, 25, 10, np.nan, 20],
        "revenues_t3m_t1": [220, 130, 0, 60, 280],
        "avg_inventory": [80, 0, 40, np.nan, 60],
        "cogs_t3m": [100, 50, 0, 30, 120],
        "avg_inventory_py": [70, 10, 35, 25, np.nan],
        "cogs_t3m_t1": [90, 0, 40, 25, 100],
        "avg_payables": [60, 30, 20, np.nan, 50],
        "avg_payables_t1": [55, 25, 0, 15, 45],
        "deferred_revenue": [30, 0, 15, np.nan, 25],
        "deferred_revenue_t1": [25, 10, 0, 20, 20],
        "adj_eps": [3.0, 1.5, np.nan, 2.0, 4.0],
        "dil_eps_fy0": [2.0, 0.0, 1.0, np.nan, 3.0],
        "buy_recs": [10, 5, 0, np.nan, 8],
        "hold_recs": [5, 3, 0, 2, 6],
        "sell_recs": [1, 2, 0, np.nan, 1],
        "dilution_p3y": [0.05, -0.02, np.nan, 0.10, 0.0],
        "non_gaap_gaap_ebit": [1.5, 2.0, np.nan, 1.0, 3.0],
        "short_interest_pct": [0.05, 0.02, np.nan, 0.10, 0.03],
        "weighted_avg_maturity": [5.0, 3.0, np.nan, 7.0, 2.0],
        "current_assets": [500, 300, 200, np.nan, 400],
        "current_assets_t1": [450, 280, 180, 250, np.nan],
        "ppe": [800, 600, 400, np.nan, 700],
        "ppe_t1": [750, 550, 380, 500, np.nan],
        "lt_investments": [100, 50, 30, np.nan, 80],
        "lt_investments_t1": [90, 45, 25, 60, np.nan],
        "total_assets": [2000, 1500, 1000, np.nan, 1800],
        "total_assets_t1": [1800, 1400, 900, 1100, np.nan],
        "depreciation": [50, 30, 20, np.nan, 40],
        "depreciation_t1": [45, 28, 18, 25, np.nan],
        "sga": [100, 80, 60, np.nan, 90],
        "sga_t1": [90, 70, 55, 50, np.nan],
        "debt_to_assets": [0.25, 0.15, 0.30, np.nan, 0.20],
        "debt_to_assets_t1": [0.22, 0.0, 0.28, 0.18, np.nan],
    })


@pytest.fixture
def all_nan_df():
    """A DataFrame where all numeric columns are NaN."""
    cols = [
        "ps_ntm", "ps_3yr_avg", "fcf_yield", "fcf_yield_3yr_avg",
        "revenues_ttm", "revenues_ttm_t1", "revenues_ttm_t2",
        "rev_cagr_f2y", "rev_cagr_p2y",
        "ntm_gross_margin", "gross_margin_3yr_avg",
        "ntm_ebit_margin", "ebit_margin_3yr_avg",
        "net_debt", "adj_ebitda", "market_cap", "fcf",
        "cash_balance", "available_loc",
        "cfo", "net_income",
        "avg_receivables", "revenues_t3m",
        "avg_receivables_t1", "revenues_t3m_t1",
        "avg_inventory", "cogs_t3m", "avg_inventory_py", "cogs_t3m_t1",
        "avg_payables", "avg_payables_t1",
        "deferred_revenue", "deferred_revenue_t1",
        "adj_eps", "dil_eps_fy0",
        "buy_recs", "hold_recs", "sell_recs",
        "current_assets", "current_assets_t1",
        "ppe", "ppe_t1", "lt_investments", "lt_investments_t1",
        "total_assets", "total_assets_t1",
        "depreciation", "depreciation_t1",
        "sga", "sga_t1",
        "debt_to_assets", "debt_to_assets_t1",
    ]
    return pd.DataFrame({c: [np.nan, np.nan, np.nan] for c in cols})


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

class TestPsDiff:
    def test_happy_path(self, basic_df):
        result = calc_ps_diff(basic_df)
        # Row 0: 5/4 - 1 = 0.25
        assert result[0] == pytest.approx(0.25)
        # Row 2: 2/2 - 1 = 0
        assert result[2] == pytest.approx(0.0)

    def test_zero_denominator(self, basic_df):
        result = calc_ps_diff(basic_df)
        # Row 4: ps_3yr_avg = 0
        assert np.isnan(result[4])

    def test_nan_input(self, basic_df):
        result = calc_ps_diff(basic_df)
        # Row 3: ps_ntm is NaN
        assert np.isnan(result[3])

    def test_all_nan(self, all_nan_df):
        result = calc_ps_diff(all_nan_df)
        assert all(np.isnan(result))


class TestFcfYieldDiff:
    def test_happy_path(self, basic_df):
        result = calc_fcf_yield_diff(basic_df)
        # Row 0: 0.04 - 0.05 = -0.01
        assert result[0] == pytest.approx(-0.01)
        # Row 1: 0.06 - 0.02 = 0.04
        assert result[1] == pytest.approx(0.04)

    def test_nan_input(self, basic_df):
        result = calc_fcf_yield_diff(basic_df)
        assert np.isnan(result[3])

    def test_all_nan(self, all_nan_df):
        result = calc_fcf_yield_diff(all_nan_df)
        assert all(np.isnan(result))


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------

class TestYoyGrowthTtm:
    def test_happy_path(self, basic_df):
        result = calc_yoy_growth_ttm(basic_df)
        # Row 0: 1000/900 - 1 = 0.1111...
        assert result[0] == pytest.approx(1000 / 900 - 1)

    def test_zero_denominator(self, basic_df):
        result = calc_yoy_growth_ttm(basic_df)
        # Row 3: revenues_ttm_t1 = 0
        assert np.isnan(result[3])

    def test_zero_numerator(self, basic_df):
        result = calc_yoy_growth_ttm(basic_df)
        # Row 2: revenues_ttm = 0, revenues_ttm_t1 = 100 -> -1.0
        assert result[2] == pytest.approx(-1.0)

    def test_all_nan(self, all_nan_df):
        result = calc_yoy_growth_ttm(all_nan_df)
        assert all(np.isnan(result))


class TestYoyGrowthTtmT1:
    def test_happy_path(self, basic_df):
        result = calc_yoy_growth_ttm_t1(basic_df)
        # Row 0: 900/800 - 1 = 0.125
        assert result[0] == pytest.approx(0.125)

    def test_zero_denominator(self, basic_df):
        result = calc_yoy_growth_ttm_t1(basic_df)
        # Row 4: revenues_ttm_t2 = 0
        assert np.isnan(result[4])

    def test_all_nan(self, all_nan_df):
        result = calc_yoy_growth_ttm_t1(all_nan_df)
        assert all(np.isnan(result))


class TestGrowthDecel:
    def test_happy_path(self):
        df = pd.DataFrame({
            "yoy_growth_ttm": [0.10, 0.05, np.nan],
            "yoy_growth_ttm_t1": [0.15, 0.05, 0.10],
        })
        result = calc_growth_decel(df)
        # Row 0: 0.10 - 0.15 = -0.05 (decelerating)
        assert result[0] == pytest.approx(-0.05)
        # Row 1: 0.05 - 0.05 = 0 (flat)
        assert result[1] == pytest.approx(0.0)
        # Row 2: NaN
        assert np.isnan(result[2])


class TestGrowthAccel:
    def test_happy_path(self, basic_df):
        result = calc_growth_accel(basic_df)
        # Row 0: 0.10 - 0.08 = 0.02
        assert result[0] == pytest.approx(0.02)

    def test_nan_input(self, basic_df):
        result = calc_growth_accel(basic_df)
        # Row 2: rev_cagr_f2y is NaN
        assert np.isnan(result[2])
        # Row 3: rev_cagr_p2y is NaN
        assert np.isnan(result[3])


# ---------------------------------------------------------------------------
# Profitability
# ---------------------------------------------------------------------------

class TestGmDiff:
    def test_happy_path(self, basic_df):
        result = calc_gm_diff(basic_df)
        # Row 0: 0.40 - 0.35 = 0.05
        assert result[0] == pytest.approx(0.05)

    def test_nan_input(self, basic_df):
        result = calc_gm_diff(basic_df)
        assert np.isnan(result[3])  # ntm is NaN
        assert np.isnan(result[4])  # avg is NaN


class TestEbitDiff:
    def test_happy_path(self, basic_df):
        result = calc_ebit_diff(basic_df)
        # Row 0: 0.15 - 0.12 = 0.03
        assert result[0] == pytest.approx(0.03)

    def test_nan_input(self, basic_df):
        result = calc_ebit_diff(basic_df)
        assert np.isnan(result[3])


# ---------------------------------------------------------------------------
# Balance Sheet
# ---------------------------------------------------------------------------

class TestLeverageRatio:
    def test_happy_path(self, basic_df):
        result = calc_leverage_ratio(basic_df)
        # Row 0: 500/200 = 2.5
        assert result[0] == pytest.approx(2.5)

    def test_negative_ratio(self, basic_df):
        result = calc_leverage_ratio(basic_df)
        # Row 1: -200/100 = -2.0 -> NaN (net cash, ratio is negative)
        assert np.isnan(result[1])

    def test_zero_net_debt_negative_ebitda(self, basic_df):
        result = calc_leverage_ratio(basic_df)
        # Row 3: net_debt=0, adj_ebitda=-50 -> 0/-50 = 0.0 (not negative, Excel keeps it)
        # Excel: IF(0/-50 < 0, "N/A", 0/-50) -> returns 0 since 0 is not < 0
        assert result[3] == pytest.approx(0.0, abs=1e-10)

    def test_zero_ebitda(self, basic_df):
        result = calc_leverage_ratio(basic_df)
        # Row 2: adj_ebitda = 0 -> NaN
        assert np.isnan(result[2])


class TestDebtSales:
    def test_happy_path(self, basic_df):
        result = calc_debt_sales(basic_df)
        # Row 0: 500/1000 = 0.5
        assert result[0] == pytest.approx(0.5)

    def test_negative_ratio(self, basic_df):
        result = calc_debt_sales(basic_df)
        # Row 1: -200/500 = -0.4 -> NaN (net cash)
        assert np.isnan(result[1])

    def test_zero_revenue(self, basic_df):
        result = calc_debt_sales(basic_df)
        # Row 2: revenues_ttm = 0 -> NaN
        assert np.isnan(result[2])


class TestEnterpriseValue:
    def test_happy_path(self, basic_df):
        result = calc_enterprise_value(basic_df)
        # Row 0: 5000 + 500 = 5500
        assert result[0] == pytest.approx(5500)
        # Row 1: 3000 + (-200) = 2800
        assert result[1] == pytest.approx(2800)


class TestDebtEv:
    def test_happy_path(self):
        df = pd.DataFrame({
            "net_debt": [500, -200],
            "enterprise_value": [5500, 2800],
        })
        result = calc_debt_ev(df)
        assert result[0] == pytest.approx(500 / 5500)
        assert result[1] == pytest.approx(-200 / 2800)

    def test_zero_ev(self):
        df = pd.DataFrame({"net_debt": [100], "enterprise_value": [0]})
        result = calc_debt_ev(df)
        assert np.isnan(result[0])


class TestLiquidity:
    def test_happy_path(self, basic_df):
        result = calc_liquidity(basic_df)
        # Row 0: 300 + 100 = 400
        assert result[0] == pytest.approx(400)
        # Row 1: 500 + 0 (NaN -> 0) = 500
        assert result[1] == pytest.approx(500)

    def test_nan_cash(self, basic_df):
        result = calc_liquidity(basic_df)
        # Row 4: cash_balance is NaN
        assert np.isnan(result[4])


class TestRemainingLiquidityYears:
    def test_cash_burner(self):
        df = pd.DataFrame({"liquidity": [400, 500], "fcf": [-50, 100]})
        result = calc_remaining_liquidity_years(df)
        # Row 0: 400/50 = 8 years
        assert result[0] == pytest.approx(8.0)
        # Row 1: FCF > 0, not a cash burner -> NaN
        assert np.isnan(result[1])

    def test_zero_fcf(self):
        df = pd.DataFrame({"liquidity": [400], "fcf": [0]})
        result = calc_remaining_liquidity_years(df)
        assert np.isnan(result[0])


# ---------------------------------------------------------------------------
# Cash Flow
# ---------------------------------------------------------------------------

class TestFcfConversion:
    def test_happy_path(self, basic_df):
        result = calc_fcf_conversion(basic_df)
        # Row 0: 100/200 = 0.5
        assert result[0] == pytest.approx(0.5)

    def test_zero_ebitda(self, basic_df):
        result = calc_fcf_conversion(basic_df)
        # Row 2: adj_ebitda = 0
        assert np.isnan(result[2])

    def test_negative_fcf(self, basic_df):
        result = calc_fcf_conversion(basic_df)
        # Row 1: -50/100 = -0.5
        assert result[1] == pytest.approx(-0.5)


class TestAccrualRatio:
    def test_happy_path(self, basic_df):
        result = calc_accrual_ratio(basic_df)
        # Row 0: (150 - 100)/150 = 0.3333
        assert result[0] == pytest.approx(1 / 3)

    def test_zero_cfo(self, basic_df):
        result = calc_accrual_ratio(basic_df)
        # Row 1: cfo = 0
        assert np.isnan(result[1])

    def test_negative_cfo(self, basic_df):
        result = calc_accrual_ratio(basic_df)
        # Row 3: (-20 - (-10))/-20 = -10/-20 = 0.5
        assert result[3] == pytest.approx(0.5)


class TestDsos:
    def test_happy_path(self, basic_df):
        result = calc_dsos(basic_df)
        # Row 0: (50/250)*90 = 18.0
        assert result[0] == pytest.approx(18.0)

    def test_zero_revenue(self, basic_df):
        result = calc_dsos(basic_df)
        # Row 1: revenues_t3m = 0
        assert np.isnan(result[1])


class TestDsosPy:
    def test_happy_path(self, basic_df):
        result = calc_dsos_py(basic_df)
        # Row 0: (40/220)*90 = 16.3636...
        assert result[0] == pytest.approx(40 / 220 * 90)

    def test_zero_revenue(self, basic_df):
        result = calc_dsos_py(basic_df)
        # Row 2: revenues_t3m_t1 = 0
        assert np.isnan(result[2])


class TestDsoPctChange:
    def test_happy_path(self):
        df = pd.DataFrame({"dsos": [20.0, 15.0], "dsos_py": [18.0, 0.0]})
        result = calc_dso_pct_change(df)
        assert result[0] == pytest.approx(20 / 18 - 1)
        assert np.isnan(result[1])


class TestDios:
    def test_happy_path(self, basic_df):
        result = calc_dios(basic_df)
        # Row 0: (80/100)*90 = 72
        assert result[0] == pytest.approx(72.0)

    def test_zero_cogs(self, basic_df):
        result = calc_dios(basic_df)
        # Row 2: cogs_t3m = 0
        assert np.isnan(result[2])


class TestDiosT1:
    def test_happy_path(self, basic_df):
        result = calc_dios_t1(basic_df)
        # Row 0: (70/90)*90 = 70
        assert result[0] == pytest.approx(70.0)

    def test_zero_cogs(self, basic_df):
        result = calc_dios_t1(basic_df)
        # Row 1: cogs_t3m_t1 = 0
        assert np.isnan(result[1])


class TestDioPctChange:
    def test_happy_path(self):
        df = pd.DataFrame({"dios": [72.0, 50.0], "dios_t1": [70.0, 0.0]})
        result = calc_dio_pct_change(df)
        assert result[0] == pytest.approx(72 / 70 - 1)
        assert np.isnan(result[1])


class TestDpos:
    def test_happy_path(self, basic_df):
        result = calc_dpos(basic_df)
        # Row 0: (60/100)*90 = 54
        assert result[0] == pytest.approx(54.0)


class TestDposT1:
    def test_happy_path(self, basic_df):
        result = calc_dpos_t1(basic_df)
        # Row 0: (55/90)*90 = 55
        assert result[0] == pytest.approx(55.0)

    def test_zero_cogs(self, basic_df):
        result = calc_dpos_t1(basic_df)
        assert np.isnan(result[1])  # cogs_t3m_t1 = 0


class TestDpoPctChange:
    def test_happy_path(self):
        df = pd.DataFrame({"dpos": [54.0, 40.0], "dpos_t1": [55.0, 0.0]})
        result = calc_dpo_pct_change(df)
        assert result[0] == pytest.approx(54 / 55 - 1)
        assert np.isnan(result[1])


class TestDaysDeferredRev:
    def test_happy_path(self, basic_df):
        result = calc_days_deferred_rev(basic_df)
        # Row 0: (30/250)*90 = 10.8
        assert result[0] == pytest.approx(10.8)


class TestDaysDeferredRevT1:
    def test_happy_path(self, basic_df):
        result = calc_days_deferred_rev_t1(basic_df)
        # Row 0: (25/220)*90 = 10.2272...
        assert result[0] == pytest.approx(25 / 220 * 90)


class TestDeferredRevPctChange:
    def test_happy_path(self):
        df = pd.DataFrame({
            "days_deferred_rev": [10.8, 5.0],
            "days_deferred_rev_t1": [10.0, 0.0],
        })
        result = calc_deferred_rev_pct_change(df)
        assert result[0] == pytest.approx(10.8 / 10.0 - 1)
        assert np.isnan(result[1])


# ---------------------------------------------------------------------------
# Non-GAAP
# ---------------------------------------------------------------------------

class TestEpsAdjRatio:
    def test_happy_path(self, basic_df):
        result = calc_eps_adj_ratio(basic_df)
        # Row 0: 3.0/2.0 = 1.5
        assert result[0] == pytest.approx(1.5)

    def test_zero_denominator(self, basic_df):
        result = calc_eps_adj_ratio(basic_df)
        # Row 1: dil_eps_fy0 = 0
        assert np.isnan(result[1])

    def test_nan_input(self, basic_df):
        result = calc_eps_adj_ratio(basic_df)
        assert np.isnan(result[2])  # adj_eps NaN
        assert np.isnan(result[3])  # dil_eps_fy0 NaN


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

class TestHoldSellPct:
    def test_happy_path(self, basic_df):
        result = calc_hold_sell_pct(basic_df)
        # Row 0: (5+1)/(10+5+1) = 6/16 = 0.375
        assert result[0] == pytest.approx(0.375)

    def test_zero_total(self, basic_df):
        result = calc_hold_sell_pct(basic_df)
        # Row 2: all recs = 0
        assert np.isnan(result[2])

    def test_nan_input(self, basic_df):
        result = calc_hold_sell_pct(basic_df)
        # Row 3: buy_recs and sell_recs are NaN -> total is NaN
        assert np.isnan(result[3])


# ---------------------------------------------------------------------------
# M-Score components
# ---------------------------------------------------------------------------

class TestDsri:
    def test_happy_path(self):
        df = pd.DataFrame({"dsos": [20.0, 15.0], "dsos_py": [18.0, 0.0]})
        result = calc_dsri(df)
        assert result[0] == pytest.approx(20 / 18)
        assert np.isnan(result[1])


class TestGmi:
    def test_happy_path(self, basic_df):
        result = calc_gmi(basic_df)
        # Row 0: GM_prior = (220 - 90)/220 = 0.59090909
        #         GM_curr  = (250 - 100)/250 = 0.60
        #         GMI = 0.59090909 / 0.60 = 0.98484848
        gm_prior = (220 - 90) / 220
        gm_curr = (250 - 100) / 250
        assert result[0] == pytest.approx(gm_prior / gm_curr)

    def test_zero_revenue(self, basic_df):
        result = calc_gmi(basic_df)
        # Row 1: revenues_t3m = 0 for dsos calc, but here cogs matters
        # Actually revenues_t3m for row 1 is 0, so gm_curr denominator is 0
        assert np.isnan(result[1])


class TestAqi:
    def test_happy_path(self, basic_df):
        result = calc_aqi(basic_df)
        # Row 0: curr = 1 - (500+800+100)/2000 = 1 - 0.7 = 0.3
        #         prior = 1 - (450+750+90)/1800 = 1 - 0.71666 = 0.28333
        #         AQI = 0.3 / 0.28333 = 1.0588...
        curr = 1 - (500 + 800 + 100) / 2000
        prior = 1 - (450 + 750 + 90) / 1800
        assert result[0] == pytest.approx(curr / prior)

    def test_nan_total_assets(self, basic_df):
        result = calc_aqi(basic_df)
        assert np.isnan(result[3])  # total_assets NaN


class TestSgi:
    def test_happy_path(self, basic_df):
        result = calc_sgi(basic_df)
        # Row 0: 250/220 = 1.13636...
        assert result[0] == pytest.approx(250 / 220)

    def test_zero_denominator(self, basic_df):
        result = calc_sgi(basic_df)
        # Row 2: revenues_t3m_t1 = 0
        assert np.isnan(result[2])


class TestDepi:
    def test_happy_path(self, basic_df):
        result = calc_depi(basic_df)
        # Row 0: rate_prior = 45/(750+45) = 45/795 = 0.05660377
        #         rate_curr  = 50/(50+800) = 50/850 = 0.05882353
        #         DEPI = 0.05660377 / 0.05882353 = 0.96226...
        rate_prior = 45 / (750 + 45)
        rate_curr = 50 / (50 + 800)
        assert result[0] == pytest.approx(rate_prior / rate_curr)


class TestSgai:
    def test_happy_path(self, basic_df):
        result = calc_sgai(basic_df)
        # Row 0: (100/1000) / (90/900) = 0.10 / 0.10 = 1.0
        assert result[0] == pytest.approx(1.0)

    def test_zero_revenue(self, basic_df):
        result = calc_sgai(basic_df)
        # Row 2: revenues_ttm = 0
        assert np.isnan(result[2])


class TestLvgi:
    def test_happy_path(self, basic_df):
        result = calc_lvgi(basic_df)
        # Row 0: 0.25/0.22 = 1.13636...
        assert result[0] == pytest.approx(0.25 / 0.22)

    def test_zero_prior(self, basic_df):
        result = calc_lvgi(basic_df)
        # Row 1: debt_to_assets_t1 = 0
        assert np.isnan(result[1])


class TestTata:
    def test_happy_path(self, basic_df):
        result = calc_tata(basic_df)
        # Row 0: (100 - 150)/2000 = -0.025
        assert result[0] == pytest.approx(-0.025)

    def test_zero_total_assets(self):
        df = pd.DataFrame({
            "net_income": [100], "cfo": [80], "total_assets": [0]
        })
        result = calc_tata(df)
        assert np.isnan(result[0])


class TestMscore:
    def test_happy_path(self):
        """Test M-Score with known component values."""
        df = pd.DataFrame({
            "dsri": [1.0],
            "gmi": [1.0],
            "aqi": [1.0],
            "sgi": [1.0],
            "depi": [1.0],
            "sgai": [1.0],
            "lvgi": [1.0],
            "tata": [0.0],
        })
        result = calc_mscore(df)
        expected = (
            -4.84
            + 0.920 * 1.0
            + 0.528 * 1.0
            + 0.404 * 1.0
            + 0.892 * 1.0
            + 0.115 * 1.0
            - 0.172 * 1.0
            + 4.679 * 0.0
            - 0.327 * 1.0
        )
        assert result[0] == pytest.approx(expected)

    def test_nan_propagation(self):
        df = pd.DataFrame({
            "dsri": [np.nan],
            "gmi": [1.0],
            "aqi": [1.0],
            "sgi": [1.0],
            "depi": [1.0],
            "sgai": [1.0],
            "lvgi": [1.0],
            "tata": [0.0],
        })
        result = calc_mscore(df)
        assert np.isnan(result[0])
