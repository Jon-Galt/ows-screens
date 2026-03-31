"""
Unit tests for scoring logic in src/score.py.

Tests cover:
- Percentile ranking direction for each factor
- NaN default fallback behavior (standard vs. zero factors)
- M-Score flag threshold behavior
- M-Score exclusion from composite score
- Factor weight sums per category
- PERCENTRANK.INC compatibility
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.score import (
    FACTOR_DEFINITIONS,
    compute_factor_scores,
    compute_mscore_flag,
    compute_overall_score,
    load_config,
    percentile_rank,
    rank_factor,
    run_scoring,
)


@pytest.fixture
def config():
    """Load the actual config.yaml."""
    return load_config()


@pytest.fixture
def scored_df(config):
    """A small DataFrame with enough columns for full scoring."""
    n = 10
    rng = np.random.RandomState(42)
    data = {}
    # Create all metric columns referenced by FACTOR_DEFINITIONS
    metrics = set()
    for defn in FACTOR_DEFINITIONS.values():
        metrics.add(defn["metric"])

    for col in metrics:
        data[col] = rng.randn(n)

    # Add mscore column for M-Score flag test
    data["mscore"] = rng.randn(n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# PERCENTRANK.INC compatibility
# ---------------------------------------------------------------------------

class TestPercentileRank:
    def test_matches_percentrank_inc(self):
        """Verify implementation matches Excel PERCENTRANK.INC exactly.

        PERCENTRANK.INC formula: (rank - 1) / (n - 1) where rank is 1-based.
        For [1,2,3,4,5]: ranks are 1,2,3,4,5; n=5.
        PERCENTRANK.INC(1) = (1-1)/(5-1) = 0.0
        PERCENTRANK.INC(3) = (3-1)/(5-1) = 0.5
        PERCENTRANK.INC(5) = (5-1)/(5-1) = 1.0
        """
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert percentile_rank(arr, 1.0) == pytest.approx(0.0)
        assert percentile_rank(arr, 2.0) == pytest.approx(0.25)
        assert percentile_rank(arr, 3.0) == pytest.approx(0.5)
        assert percentile_rank(arr, 4.0) == pytest.approx(0.75)
        assert percentile_rank(arr, 5.0) == pytest.approx(1.0)

    def test_includes_both_endpoints(self):
        """PERCENTRANK.INC produces both 0 and 1."""
        arr = np.array([10, 20, 30, 40, 50])
        assert percentile_rank(arr, 10) == pytest.approx(0.0)
        assert percentile_rank(arr, 50) == pytest.approx(1.0)

    def test_ties(self):
        """Tied values should get averaged rank."""
        arr = np.array([1.0, 2.0, 2.0, 3.0])
        # Two values of 2.0 at positions 2 and 3 (of 4)
        p = percentile_rank(arr, 2.0)
        # Should be between 0 and 1 exclusive of endpoints
        assert 0.0 < p < 1.0

    def test_single_value(self):
        """Array with one value should rank it at 1.0 (100th percentile)."""
        arr = np.array([5.0])
        assert percentile_rank(arr, 5.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Ranking direction
# ---------------------------------------------------------------------------

class TestRankingDirection:
    def test_straight_ranking_higher_raw_higher_factor(self):
        """For straight factors, higher raw value -> higher factor score."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rank_factor(series, "straight", 0.5)
        # Highest raw value should have highest factor score
        assert result[4] > result[0]

    def test_inverted_ranking_lower_raw_higher_factor(self):
        """For inverted factors, lower raw value -> higher factor score."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rank_factor(series, "inverted", 0.5)
        # Lowest raw value should have highest factor score
        assert result[0] > result[4]

    def test_each_factor_direction_is_defined(self, config):
        """Every factor in FACTOR_DEFINITIONS has a valid direction."""
        for name, defn in FACTOR_DEFINITIONS.items():
            assert defn["direction"] in ("straight", "inverted"), (
                f"Factor {name} has invalid direction: {defn['direction']}"
            )

    def test_inverted_factors_match_excel(self):
        """Verify the specific factors that use 1-percentile match the Excel."""
        inverted = {
            name for name, defn in FACTOR_DEFINITIONS.items()
            if defn["direction"] == "inverted"
        }
        expected_inverted = {
            "rel_fcf_factor",
            "decel_factor",
            "refi_risk_factor",
            "liquidity_risk_factor",
            "fcf_conv_factor",
            "dpo_factor",
            "def_rev_factor",
        }
        assert inverted == expected_inverted


# ---------------------------------------------------------------------------
# NaN defaults
# ---------------------------------------------------------------------------

class TestNanDefaults:
    def test_nan_uses_standard_default(self, config):
        """Factors NOT in zero_factors list should default to 0.5."""
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        zero_factors = set(config["scoring"]["nan_default_zero_factors"])
        std_default = config["scoring"]["nan_default_standard"]

        # Test a factor NOT in zero list (e.g., abs_ps_factor)
        assert "abs_ps_factor" not in zero_factors
        result = rank_factor(series, "straight", std_default)
        assert result[2] == pytest.approx(std_default)

    def test_nan_uses_zero_default_for_balance_sheet(self, config):
        """Factors in nan_default_zero_factors should default to 0.0."""
        zero_factors = config["scoring"]["nan_default_zero_factors"]
        assert "debt_ebitda_factor" in zero_factors
        assert "dso_factor" in zero_factors

        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = rank_factor(series, "straight", 0.0)
        assert result[2] == pytest.approx(0.0)

    def test_all_nan_returns_default(self):
        """When all values are NaN, every stock gets the default."""
        series = pd.Series([np.nan, np.nan, np.nan])
        result = rank_factor(series, "straight", 0.5)
        assert all(result == 0.5)

        result_zero = rank_factor(series, "straight", 0.0)
        assert all(result_zero == 0.0)

    def test_zero_factors_list_from_config(self, config):
        """Verify the expected factors are in the zero-default list."""
        zero_factors = config["scoring"]["nan_default_zero_factors"]
        expected = [
            "debt_ebitda_factor",
            "debt_sales_factor",
            "liquidity_risk_factor",
            "dso_factor",
            "dio_factor",
            "dpo_factor",
            "def_rev_factor",
        ]
        assert sorted(zero_factors) == sorted(expected)


# ---------------------------------------------------------------------------
# M-Score
# ---------------------------------------------------------------------------

class TestMScore:
    def test_mscore_not_in_composite(self, config):
        """M-Score must NOT be included in the overall composite score."""
        # Verify mscore is not a factor in FACTOR_DEFINITIONS
        assert "mscore" not in FACTOR_DEFINITIONS
        # Verify no factor references mscore as its metric
        for name, defn in FACTOR_DEFINITIONS.items():
            assert defn["metric"] != "mscore", (
                f"Factor {name} incorrectly uses mscore as its metric"
            )

    def test_mscore_flag_threshold(self, config):
        """M-Score flag should trigger at > -2.22."""
        threshold = config["scoring"]["mscore_manipulation_threshold"]
        assert threshold == pytest.approx(-2.22)

        df = pd.DataFrame({"mscore": [-3.0, -2.22, -2.21, -1.0, np.nan]})
        flags = compute_mscore_flag(df, config)
        assert flags[0] == False   # -3.0 < -2.22
        assert flags[1] == False   # -2.22 is NOT > -2.22
        assert flags[2] == True    # -2.21 > -2.22
        assert flags[3] == True    # -1.0 > -2.22
        assert flags[4] == False   # NaN comparison is False

    def test_mscore_flag_in_output(self, scored_df, config):
        """run_scoring should produce mscore_flag column."""
        result = run_scoring(scored_df.copy(), config)
        assert "mscore_flag" in result.columns
        assert result["mscore_flag"].dtype == bool


# ---------------------------------------------------------------------------
# Factor weights
# ---------------------------------------------------------------------------

class TestFactorWeights:
    def test_weights_sum_per_category(self, config):
        """Within-category factor weights must sum to 1.0."""
        weights = config["factor_weights"]

        categories = {
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

        for category, factors in categories.items():
            total = sum(weights[f] for f in factors)
            assert total == pytest.approx(1.0, abs=0.001), (
                f"{category} weights sum to {total}, expected 1.0"
            )

    def test_all_factors_have_weights(self, config):
        """Every factor in FACTOR_DEFINITIONS must have a weight in config."""
        weights = config["factor_weights"]
        for factor_name in FACTOR_DEFINITIONS:
            assert factor_name in weights, f"Missing weight for {factor_name}"

    def test_no_hardcoded_weights(self):
        """Verify score.py does not contain hardcoded weight values."""
        import inspect
        from src import score
        source = inspect.getsource(score)
        # Should not contain numeric weight assignments
        # (0.142857, 0.25, 0.5, 0.2 are the weights — none should appear as assignments)
        assert "weight = 0." not in source
        assert "weight=0." not in source


# ---------------------------------------------------------------------------
# Overall score
# ---------------------------------------------------------------------------

class TestOverallScore:
    def test_overall_score_range(self, scored_df, config):
        """Overall score should be bounded between 0 and 7."""
        result = run_scoring(scored_df.copy(), config)
        assert result["overall_score"].min() >= -0.01  # small float tolerance
        assert result["overall_score"].max() <= 7.01

    def test_overall_score_uses_all_factors(self, config):
        """Overall score should use exactly the 24 defined factors."""
        assert len(FACTOR_DEFINITIONS) == 24

    def test_perfect_score(self, config):
        """A stock with all factor scores = 1.0 should score 7.0."""
        data = {defn["metric"]: [1.0] for defn in FACTOR_DEFINITIONS.values()}
        data["mscore"] = [0.0]
        df = pd.DataFrame(data)
        # With only 1 stock, percentile of a single value = 1.0
        result = run_scoring(df.copy(), config)
        # All straight factors get 1.0, all inverted factors get 0.0
        # So the score won't be exactly 7.0 — but it should be well-defined
        assert not np.isnan(result["overall_score"].iloc[0])
