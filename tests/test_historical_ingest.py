"""
Unit tests for src/historical_ingest.py (Phase 4a).

Small synthetic DataFrames throughout, per the Worker convention — the
real source workbook (data/historical/) is gitignored, so nothing here
may depend on it. The 24/24 join and every acceptance-criteria figure
(row counts, sign-check correlations, by-Setup/by-era medians, the
chained whiteboard comparison) are verified separately against the real
file as part of the mandatory end-to-end run, not asserted here.
"""

import json
import math
import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect

from src.historical_ingest import (
    ConsistencyResult,
    SignConventionError,
    assign_era_bucket,
    assign_hold_period_bucket,
    assign_market_cap_bucket,
    build_whiteboard_bridge,
    chain_whiteboard_position,
    check_benchmark_consistency,
    check_sign_convention,
    classify_benchmark_instrument,
    classify_sector_benchmark_instrument,
    clean_active_dataframe,
    clean_whiteboard_dataframe,
    count_defects,
    ingest_historical,
    summarize_by_cut,
    summarize_era_initiation_counts,
    summarize_overall,
    summarize_whiteboard_chained,
    summarize_whiteboard_naive,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal raw-shaped DataFrames matching the source workbook's
# exact column names, small enough to hand-verify every assertion.
# ---------------------------------------------------------------------------

def _active_raw_row(**overrides):
    row = {
        "OWS DB Ticker": "AAA",
        "BBG Ticker": "AAA US Equity",
        "Company Name": "Alpha Corp",
        "Setup": "Structural",
        "Sector": "Industrials",
        "Sector Index": "S5INDU Index",
        "Status": "Closed",
        "Init Report Filename": "init.pdf",
        "Init Report PDF Link": "http://x/init.pdf",
        "Init Report DOCX Link": "http://x/init.docx",
        "Initiation Date": pd.Timestamp("2020-01-01"),
        "Init Pricing Date": pd.Timestamp("2020-01-01"),
        "Market Cap @ Initiation": 2.0,
        "Initiation Price": 100.0,
        "SPX @ Initiation": 3200.0,
        "Sector Index @ Initiation": 650.0,
        "Close Report Filename": "close.pdf",
        "Close Report PDF Link": "http://x/close.pdf",
        "Close Report DOCX Link": "http://x/close.docx",
        "Close Date": pd.Timestamp("2020-07-01"),
        "Close Pricing Date": pd.Timestamp("2020-07-01"),
        "Close Price": 80.0,
        "SPX @ Close": 3300.0,
        "Sector Index @ Close": 660.0,
        "Duration": 182,
        "Absolute Performance": 0.20,
        "Relative SPY Performance": 0.23,
        "Relative Sector Performance": 0.24,
    }
    row.update(overrides)
    return row


def _whiteboard_raw_row(**overrides):
    row = {
        "OWS DB Ticker": "BBB",
        "BBG Ticker": "BBB US Equity",
        "Company Name": "Beta Corp",
        "Setup": "Cyclical",
        "Sector": "Financials",
        "Sector ETF": "XLF US Equity",
        "Status": "Closed",
        "WBA Report Filename": "wba.pdf",
        "WBA Report DOCX Link": "http://x/wba.docx",
        "WBA Report PDF Link": "http://x/wba.pdf",
        "WBA Date": pd.Timestamp("2024-01-01"),
        "WBA Pricing Date": pd.Timestamp("2024-01-01"),
        "Market Cap @ WBA ($B)": 5.0,
        "WBA Price": 50.0,
        "SPY @ WBA": 470.0,
        "Sector ETF @ WBA": 38.0,
        "Outcome": "Removed",
        "WBR Report Filename": "wbr.pdf",
        "WBR Report DOCX Link": "http://x/wbr.docx",
        "WBR Report PDF Link": "http://x/wbr.pdf",
        "WBR Date": pd.Timestamp("2024-06-01"),
        "WBR Pricing Date": pd.Timestamp("2024-06-01"),
        "WBR Price": 40.0,
        "SPY @ WBR": 500.0,
        "Sector ETF @ WBR": 40.0,
        "Duration": 152,
        "Absolute Performance": 0.20,
        "Relative SPY Performance": 0.14,
        "Relative Sector Performance": 0.12,
    }
    row.update(overrides)
    return row


@pytest.fixture
def active_raw():
    return pd.DataFrame([_active_raw_row()])


@pytest.fixture
def whiteboard_raw():
    return pd.DataFrame([_whiteboard_raw_row()])


# ---------------------------------------------------------------------------
# clean_active_dataframe / clean_whiteboard_dataframe
# ---------------------------------------------------------------------------

class TestCleanActiveDataframe:
    def test_happy_path_renames_and_converts_units(self, active_raw):
        df = clean_active_dataframe(active_raw)
        assert df.loc[0, "ticker"] == "AAA"
        assert df.loc[0, "benchmark_at_initiation"] == 3200.0
        # $B -> $M, exact x1000
        assert df.loc[0, "market_cap_at_initiation"] == 2000.0

    def test_nan_and_hash_marker_survive_as_nan_not_raise(self, active_raw):
        raw = pd.DataFrame([_active_raw_row(**{"Initiation Price": None, "Setup": None})])
        df = clean_active_dataframe(raw)
        assert pd.isna(df.loc[0, "initiation_price"])
        assert pd.isna(df.loc[0, "setup"])

    def test_ansS_mfe_glyt_shaped_row_both_dates_null_and_status_closed(self):
        """Regression lock for the confirmed consolidation: a row that is
        Status=Closed with BOTH Initiation Date and Close Date null (and no
        prices) must clean without raising, and must still carry its
        recorded performance."""
        raw = pd.DataFrame([_active_raw_row(**{
            "Initiation Date": None, "Init Pricing Date": None,
            "Close Date": None, "Close Pricing Date": None,
            "Initiation Price": None, "Close Price": None,
            "Absolute Performance": -0.51,
        })])
        df = clean_active_dataframe(raw)
        assert pd.isna(df.loc[0, "initiation_date"])
        assert pd.isna(df.loc[0, "close_date"])
        assert df.loc[0, "status"] == "Closed"
        assert df.loc[0, "absolute_performance"] == -0.51

    def test_missing_required_column_raises(self):
        raw = pd.DataFrame([_active_raw_row()]).drop(columns=["Setup"])
        with pytest.raises(KeyError):
            clean_active_dataframe(raw)


class TestCleanWhiteboardDataframe:
    def test_happy_path_renames_and_converts_units(self, whiteboard_raw):
        df = clean_whiteboard_dataframe(whiteboard_raw)
        assert df.loc[0, "ticker"] == "BBB"
        assert df.loc[0, "benchmark_at_wba"] == 470.0
        assert df.loc[0, "market_cap_at_wba"] == 5000.0
        assert df.loc[0, "benchmark_instrument"] == "SPY"

    def test_carg_shaped_row_three_defects_coerce_without_raising(self):
        """Regression lock for the confirmed consolidation: one row
        carrying Duration='Error', Outcome=null, and WBR Date='Not Found'
        simultaneously must clean without raising, each field going to
        NaN/NaT independently."""
        raw = pd.DataFrame([_whiteboard_raw_row(**{
            "Duration": "Error", "Outcome": None, "WBR Date": "Not Found",
        })])
        df = clean_whiteboard_dataframe(raw)
        assert pd.isna(df.loc[0, "duration_days"])
        assert df.loc[0, "duration_raw"] == "Error"
        assert pd.isna(df.loc[0, "outcome"])
        assert pd.isna(df.loc[0, "wbr_date"])

    def test_duration_raw_preserves_numeric_value_as_string(self, whiteboard_raw):
        df = clean_whiteboard_dataframe(whiteboard_raw)
        assert df.loc[0, "duration_raw"] == "152"
        assert df.loc[0, "duration_days"] == 152.0

    def test_missing_required_column_raises(self):
        raw = pd.DataFrame([_whiteboard_raw_row()]).drop(columns=["Outcome"])
        with pytest.raises(KeyError):
            clean_whiteboard_dataframe(raw)


# ---------------------------------------------------------------------------
# classify_benchmark_instrument (band-guarded threshold)
# ---------------------------------------------------------------------------

class TestClassifyBenchmarkInstrument:
    def test_below_ceiling_is_spy(self):
        result = classify_benchmark_instrument(pd.Series([100.0]))
        assert result.iloc[0] == "SPY"

    def test_above_floor_is_spx(self):
        result = classify_benchmark_instrument(pd.Series([3000.0]))
        assert result.iloc[0] == "SPX"

    def test_inside_band_is_unclassifiable(self):
        result = classify_benchmark_instrument(pd.Series([500.0]))
        assert result.iloc[0] is None

    def test_null_input_is_none(self):
        result = classify_benchmark_instrument(pd.Series([float("nan")]))
        assert result.iloc[0] is None

    def test_boundary_at_ceiling_classifies_spy(self):
        result = classify_benchmark_instrument(pd.Series([270.39]), spy_ceiling=270.39, spx_floor=756.55)
        assert result.iloc[0] == "SPY"

    def test_boundary_at_floor_classifies_spx(self):
        result = classify_benchmark_instrument(pd.Series([756.55]), spy_ceiling=270.39, spx_floor=756.55)
        assert result.iloc[0] == "SPX"

    def test_whiteboard_style_values_mostly_land_in_unclassifiable_band(self):
        """Documents WHY Whiteboard's own benchmark_instrument must not be
        derived via this function — a real SPY value from Whiteboard's
        vintage (e.g. 772) would misclassify as SPX under the naive
        threshold, or land in the unclassifiable band with the guard. See
        the module docstring."""
        result = classify_benchmark_instrument(pd.Series([772.0]))
        assert result.iloc[0] == "SPX"  # naive misclassification if this were ever applied to Whiteboard


class TestClassifySectorBenchmarkInstrument:
    def test_index_ticker(self):
        result = classify_sector_benchmark_instrument(pd.Series(["S5INDU Index"]))
        assert result.iloc[0] == "INDEX"

    def test_etf_ticker(self):
        result = classify_sector_benchmark_instrument(pd.Series(["XLK US Equity"]))
        assert result.iloc[0] == "ETF"

    def test_null_ticker_is_none(self):
        result = classify_sector_benchmark_instrument(pd.Series([None]))
        assert result.iloc[0] is None

    def test_unrecognized_ticker_is_none_not_guessed(self):
        result = classify_sector_benchmark_instrument(pd.Series(["ZZZZ Comdty"]))
        assert result.iloc[0] is None


# ---------------------------------------------------------------------------
# check_sign_convention
# ---------------------------------------------------------------------------

def _sign_df(price_moves_and_perfs):
    inits, closes, perfs = [], [], []
    for move, perf in price_moves_and_perfs:
        inits.append(100.0)
        closes.append(100.0 * (1 + move))
        perfs.append(perf)
    return pd.DataFrame({"init": inits, "close": closes, "perf": perfs})


class TestCheckSignConvention:
    def test_happy_path_anti_correlated_passes(self):
        df = _sign_df([(0.10, -0.10), (0.20, -0.20), (-0.10, 0.10), (-0.30, 0.30)])
        result = check_sign_convention(df, "init", "close", "perf", min_abs_corr=0.95)
        assert result.passed is True
        assert result.corr < -0.95

    def test_rejects_sign_flipped_fixture(self):
        """The fixture that would silently invert every conclusion if this
        gate didn't exist: performance moves WITH the stock, not against
        it."""
        df = _sign_df([(0.10, 0.10), (0.20, 0.20), (-0.10, -0.10), (-0.30, -0.30)])
        result = check_sign_convention(df, "init", "close", "perf", min_abs_corr=0.95)
        assert result.passed is False

    def test_rejects_empty_intersection_not_vacuously_true(self):
        df = pd.DataFrame({"init": [100.0, 100.0], "close": [110.0, 90.0], "perf": [float("nan"), float("nan")]})
        result = check_sign_convention(df, "init", "close", "perf", min_abs_corr=0.95)
        assert result.n == 0
        assert result.passed is False

    def test_boundary_exactly_at_threshold_passes(self):
        df = _sign_df([(0.10, -0.10), (0.20, -0.20), (-0.10, 0.10), (-0.30, 0.30)])
        result = check_sign_convention(df, "init", "close", "perf", min_abs_corr=0.95)
        # Re-run with a threshold set to exactly this corr's magnitude (<=).
        boundary_result = check_sign_convention(df, "init", "close", "perf", min_abs_corr=abs(result.corr))
        assert boundary_result.passed is True

    def test_boundary_just_inside_threshold_fails(self):
        df = _sign_df([(0.10, -0.10), (0.20, -0.20), (-0.10, 0.10), (-0.30, 0.30)])
        result = check_sign_convention(df, "init", "close", "perf", min_abs_corr=0.95)
        just_over = check_sign_convention(df, "init", "close", "perf", min_abs_corr=abs(result.corr) + 1e-6)
        assert just_over.passed is False


# ---------------------------------------------------------------------------
# check_benchmark_consistency
# ---------------------------------------------------------------------------

def _consistency_df(rows):
    return pd.DataFrame(rows, columns=["ticker", "init", "close", "bench_init", "bench_close", "perf"])


class TestCheckBenchmarkConsistency:
    def test_happy_path_within_tolerance(self):
        # price_move = -0.10, bench_move = 0.05, implied relative = 0.15
        df = _consistency_df([["AAA", 100.0, 90.0, 1000.0, 1050.0, 0.15]])
        result = check_benchmark_consistency(df, "init", "close", "bench_init", "bench_close", "perf", tolerance=0.01)
        assert result.violation_count == 0
        assert result.n == 1

    def test_flags_engineered_violation(self):
        """The counterpart to the sign-gate's sign-flipped test: a
        deliberately wrong bench_close value must be caught, proving this
        is a real guard and not a decorative check."""
        df = _consistency_df([
            ["AAA", 100.0, 90.0, 1000.0, 1050.0, 0.15],   # consistent
            ["BBB", 100.0, 90.0, 1000.0, 2000.0, 0.15],   # bench_close swapped/corrupted
        ])
        result = check_benchmark_consistency(df, "init", "close", "bench_init", "bench_close", "perf", tolerance=0.01)
        assert result.violation_count == 1
        assert result.violation_tickers == ["BBB"]

    def test_boundary_exactly_at_tolerance_not_a_violation(self):
        # implied = 0.15 exactly; stored = 0.15 - tolerance -> diff == tolerance -> not a violation (<=)
        tolerance = 0.01
        df = _consistency_df([["AAA", 100.0, 90.0, 1000.0, 1050.0, 0.15 - tolerance]])
        result = check_benchmark_consistency(df, "init", "close", "bench_init", "bench_close", "perf", tolerance=tolerance)
        assert result.violation_count == 0

    def test_boundary_just_over_tolerance_is_a_violation(self):
        tolerance = 0.01
        df = _consistency_df([["AAA", 100.0, 90.0, 1000.0, 1050.0, 0.15 - tolerance - 1e-6]])
        result = check_benchmark_consistency(df, "init", "close", "bench_init", "bench_close", "perf", tolerance=tolerance)
        assert result.violation_count == 1

    def test_empty_intersection_reports_zero_not_a_pass_on_invented_evidence(self):
        df = _consistency_df([["AAA", 100.0, 90.0, 1000.0, 1050.0, float("nan")]])
        result = check_benchmark_consistency(df, "init", "close", "bench_init", "bench_close", "perf", tolerance=0.01)
        assert result.n == 0
        assert result.violation_count == 0
        assert math.isnan(result.max_abs_diff)


# ---------------------------------------------------------------------------
# count_defects
# ---------------------------------------------------------------------------

class TestCountDefects:
    def test_one_of_each_defect(self):
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(),
            _active_raw_row(**{"OWS DB Ticker": "CLOSED_NO_DATE", "Close Date": None, "Close Price": None}),
            _active_raw_row(**{"OWS DB Ticker": "OPEN_WITH_PERF", "Status": "Open", "Absolute Performance": 0.1}),
            _active_raw_row(**{"OWS DB Ticker": "NO_SETUP", "Setup": None}),
        ]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([
            _whiteboard_raw_row(),
            _whiteboard_raw_row(**{"OWS DB Ticker": "DUR_ERROR", "Duration": "Error"}),
            _whiteboard_raw_row(**{"OWS DB Ticker": "NO_OUTCOME", "Outcome": None}),
        ]))
        active_consistency = check_benchmark_consistency(
            active, "initiation_price", "close_price", "benchmark_at_initiation",
            "benchmark_at_close", "relative_spy_performance", tolerance=0.01,
        )
        whiteboard_consistency = check_benchmark_consistency(
            whiteboard, "wba_price", "wbr_price", "benchmark_at_wba",
            "benchmark_at_wbr", "relative_spy_performance", tolerance=0.01,
        )
        defects = count_defects(active, whiteboard, active_consistency, whiteboard_consistency)

        assert defects["closed_no_close_date"] == ["CLOSED_NO_DATE"]
        assert defects["open_with_performance"] == ["OPEN_WITH_PERF"]
        assert defects["setup_missing_count"] == 1
        assert defects["whiteboard_duration_error_tickers"] == ["DUR_ERROR"]
        assert defects["outcome_missing_count"] == 1

    def test_no_defects_all_counts_zero(self):
        """Boundary: a fixture with NONE of the defects proves the counters
        aren't hardcoded to the real file's numbers."""
        active = clean_active_dataframe(pd.DataFrame([_active_raw_row()]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([_whiteboard_raw_row()]))
        active_consistency = check_benchmark_consistency(
            active, "initiation_price", "close_price", "benchmark_at_initiation",
            "benchmark_at_close", "relative_spy_performance", tolerance=0.01,
        )
        whiteboard_consistency = check_benchmark_consistency(
            whiteboard, "wba_price", "wbr_price", "benchmark_at_wba",
            "benchmark_at_wbr", "relative_spy_performance", tolerance=0.01,
        )
        defects = count_defects(active, whiteboard, active_consistency, whiteboard_consistency)

        assert defects["closed_no_close_date"] == []
        assert defects["open_with_performance"] == []
        assert defects["setup_missing_count"] == 0
        assert defects["whiteboard_duration_error_tickers"] == []
        assert defects["outcome_missing_count"] == 0

    def test_mixed_benchmark_instruments_reported(self):
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(**{"OWS DB Ticker": "SPY_ROW", "SPX @ Initiation": 100.0, "SPX @ Close": 110.0}),
            _active_raw_row(**{"OWS DB Ticker": "SPX_ROW", "SPX @ Initiation": 3000.0, "SPX @ Close": 3100.0}),
            _active_raw_row(**{
                "OWS DB Ticker": "ETF_SECTOR", "Sector Index": "XLK US Equity",
            }),
        ]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([_whiteboard_raw_row()]))
        active_consistency = check_benchmark_consistency(
            active, "initiation_price", "close_price", "benchmark_at_initiation",
            "benchmark_at_close", "relative_spy_performance", tolerance=0.01,
        )
        whiteboard_consistency = check_benchmark_consistency(
            whiteboard, "wba_price", "wbr_price", "benchmark_at_wba",
            "benchmark_at_wbr", "relative_spy_performance", tolerance=0.01,
        )
        defects = count_defects(active, whiteboard, active_consistency, whiteboard_consistency)

        assert defects["benchmark_instrument"]["active_spy"] == 1
        assert defects["benchmark_instrument"]["active_spx"] == 2
        assert defects["sector_benchmark_instrument"]["active_etf"] == 1
        assert defects["sector_benchmark_instrument"]["active_index"] == 2

    def test_performance_rounding_property_true_when_whole_percent(self):
        active = clean_active_dataframe(pd.DataFrame([_active_raw_row()]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([_whiteboard_raw_row()]))
        consistency = check_benchmark_consistency(
            active, "initiation_price", "close_price", "benchmark_at_initiation",
            "benchmark_at_close", "relative_spy_performance", tolerance=0.01,
        )
        wb_consistency = check_benchmark_consistency(
            whiteboard, "wba_price", "wbr_price", "benchmark_at_wba",
            "benchmark_at_wbr", "relative_spy_performance", tolerance=0.01,
        )
        defects = count_defects(active, whiteboard, consistency, wb_consistency)
        assert defects["performance_values_rounded_to_whole_percent"]["active"] is True

    def test_performance_rounding_property_false_when_not_whole_percent(self):
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(**{"Absolute Performance": 0.20125}),
        ]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([_whiteboard_raw_row()]))
        consistency = check_benchmark_consistency(
            active, "initiation_price", "close_price", "benchmark_at_initiation",
            "benchmark_at_close", "relative_spy_performance", tolerance=0.01,
        )
        wb_consistency = check_benchmark_consistency(
            whiteboard, "wba_price", "wbr_price", "benchmark_at_wba",
            "benchmark_at_wbr", "relative_spy_performance", tolerance=0.01,
        )
        defects = count_defects(active, whiteboard, consistency, wb_consistency)
        assert defects["performance_values_rounded_to_whole_percent"]["active"] is False

    def test_market_cap_zero_reported_not_silently_bucketed(self):
        """Regression lock: a source market cap of exactly 0.0 (e.g. the
        real file's two NOBN rows) must surface in count_defects, not just
        disappear into assign_market_cap_bucket's '<$1B' bucket."""
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(**{"OWS DB Ticker": "ZERO_CAP", "Market Cap @ Initiation": 0.0}),
            _active_raw_row(**{"OWS DB Ticker": "NORMAL_CAP", "Market Cap @ Initiation": 2.0}),
        ]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([_whiteboard_raw_row()]))
        consistency = check_benchmark_consistency(
            active, "initiation_price", "close_price", "benchmark_at_initiation",
            "benchmark_at_close", "relative_spy_performance", tolerance=0.01,
        )
        wb_consistency = check_benchmark_consistency(
            whiteboard, "wba_price", "wbr_price", "benchmark_at_wba",
            "benchmark_at_wbr", "relative_spy_performance", tolerance=0.01,
        )
        defects = count_defects(active, whiteboard, consistency, wb_consistency)
        assert defects["market_cap_zero"]["active"] == ["ZERO_CAP"]
        assert defects["market_cap_zero"]["whiteboard"] == []

    def test_defects_json_round_trip(self):
        """Mutation-style lock: the returned dict must survive a JSON
        encode/decode via the same _json_safe path ingest_historical uses,
        including nested violation-ticker lists and NaN-bearing floats."""
        from src.historical_ingest import _json_safe

        active = clean_active_dataframe(pd.DataFrame([_active_raw_row()]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([_whiteboard_raw_row()]))
        consistency = ConsistencyResult(n=0, violation_count=0, max_abs_diff=float("nan"), tolerance=0.01, violation_tickers=[])
        defects = count_defects(active, whiteboard, consistency, consistency)

        encoded = json.dumps(_json_safe(defects))
        decoded = json.loads(encoded)
        assert decoded["setup_missing_count"] == defects["setup_missing_count"]
        assert decoded["benchmark_consistency"]["active_max_abs_diff"] is None  # NaN -> null


# ---------------------------------------------------------------------------
# build_whiteboard_bridge / chain_whiteboard_position — synthetic join test
# ---------------------------------------------------------------------------

class TestBuildWhiteboardBridge:
    def test_exact_match_only(self):
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(**{"OWS DB Ticker": "MATCH", "Initiation Date": pd.Timestamp("2024-06-01")}),
            _active_raw_row(**{"OWS DB Ticker": "MATCH", "Initiation Date": pd.Timestamp("2024-09-01")}),  # same ticker, different date
            _active_raw_row(**{"OWS DB Ticker": "OTHER", "Initiation Date": pd.Timestamp("2024-06-01")}),  # different ticker, same date
        ]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([
            _whiteboard_raw_row(**{"OWS DB Ticker": "MATCH", "Outcome": "Initiation", "WBR Date": pd.Timestamp("2024-06-01")}),
        ]))
        bridge = build_whiteboard_bridge(active, whiteboard)
        assert len(bridge) == 1
        assert bridge.iloc[0]["ticker"] == "MATCH"
        assert bridge.iloc[0]["wbr_date"] == bridge.iloc[0]["initiation_date"]

    def test_non_initiation_outcome_never_matches(self):
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(**{"OWS DB Ticker": "MATCH", "Initiation Date": pd.Timestamp("2024-06-01")}),
        ]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([
            _whiteboard_raw_row(**{"OWS DB Ticker": "MATCH", "Outcome": "Removed", "WBR Date": pd.Timestamp("2024-06-01")}),
        ]))
        bridge = build_whiteboard_bridge(active, whiteboard)
        assert len(bridge) == 0


class TestChainWhiteboardPosition:
    def test_both_legs_present_computes_chain(self):
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(**{
                "OWS DB Ticker": "MATCH", "Initiation Date": pd.Timestamp("2024-06-01"),
                "Relative SPY Performance": 0.10, "Duration": 100,
            }),
        ]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([
            _whiteboard_raw_row(**{
                "OWS DB Ticker": "MATCH", "Outcome": "Initiation", "WBR Date": pd.Timestamp("2024-06-01"),
                "Relative SPY Performance": 0.20, "Duration": 50,
            }),
        ]))
        bridge = build_whiteboard_bridge(active, whiteboard)
        chained = chain_whiteboard_position(bridge)
        expected = (1.20 * 1.10) - 1
        assert chained.iloc[0]["chained_relative_spy_performance"] == pytest.approx(expected)
        assert chained.iloc[0]["chained_duration_days"] == 150

    def test_one_leg_missing_chained_is_nan_row_not_dropped(self):
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(**{
                "OWS DB Ticker": "MATCH", "Initiation Date": pd.Timestamp("2024-06-01"),
                "Relative SPY Performance": None,
            }),
        ]))
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([
            _whiteboard_raw_row(**{
                "OWS DB Ticker": "MATCH", "Outcome": "Initiation", "WBR Date": pd.Timestamp("2024-06-01"),
                "Relative SPY Performance": 0.20,
            }),
        ]))
        bridge = build_whiteboard_bridge(active, whiteboard)
        chained = chain_whiteboard_position(bridge)
        assert len(chained) == 1
        assert pd.isna(chained.iloc[0]["chained_relative_spy_performance"])


# ---------------------------------------------------------------------------
# summarize_overall / summarize_by_cut
# ---------------------------------------------------------------------------

class TestSummarizeOverall:
    def test_happy_path_hand_computed(self):
        df = pd.DataFrame({"perf": [0.10, 0.20, -0.10, 0.30]})
        result = summarize_overall(df, ["perf"])
        row = result.iloc[0]
        assert row["n"] == 4
        assert row["mean"] == pytest.approx(0.125)
        assert row["median"] == pytest.approx(0.15)
        assert row["hit_rate"] == pytest.approx(0.75)

    def test_all_null_group_n_zero_not_a_crash(self):
        df = pd.DataFrame({"perf": [float("nan"), float("nan")]})
        result = summarize_overall(df, ["perf"])
        row = result.iloc[0]
        assert row["n"] == 0
        assert math.isnan(row["mean"])


class TestSummarizeByCut:
    def test_group_and_unassigned_reconcile_setup_shaped(self):
        df = pd.DataFrame({
            "setup": ["A", "A", "B", None, None],
            "perf": [0.10, 0.20, 0.30, 0.40, None],
        })
        grouped, unassigned = summarize_by_cut(df, "setup", ["perf"])
        grouped_n = grouped[grouped["measure"] == "perf"]["n"].sum()
        unassigned_n = unassigned[unassigned["measure"] == "perf"]["n"].iloc[0]
        assert grouped_n + unassigned_n == df["perf"].notna().sum()

    def test_group_and_unassigned_reconcile_era_shaped(self):
        df = pd.DataFrame({
            "era": ["1998-2007", "1998-2007", "2020-2026", None],
            "perf": [0.10, 0.20, 0.30, 0.40],
        })
        grouped, unassigned = summarize_by_cut(df, "era", ["perf"])
        grouped_n = grouped[grouped["measure"] == "perf"]["n"].sum()
        unassigned_n = unassigned[unassigned["measure"] == "perf"]["n"].iloc[0]
        assert grouped_n + unassigned_n == df["perf"].notna().sum()
        assert unassigned_n == 1

    def test_min_n_suppresses_small_groups_from_grouped_table_only(self):
        df = pd.DataFrame({"setup": ["A", "B", "B", "B"], "perf": [0.1, 0.2, 0.3, 0.4]})
        grouped, _ = summarize_by_cut(df, "setup", ["perf"], min_n=2)
        assert set(grouped["cut_value"]) == {"B"}


# ---------------------------------------------------------------------------
# assign_hold_period_bucket / assign_market_cap_bucket / assign_era_bucket
# ---------------------------------------------------------------------------

class TestAssignHoldPeriodBucket:
    @pytest.mark.parametrize("days,expected", [
        (89, "<90d"), (90, "90-180d"), (179, "90-180d"), (180, "180-365d"),
        (364, "180-365d"), (365, "1-2y"), (729, "1-2y"), (730, "2y+"),
    ])
    def test_boundaries(self, days, expected):
        result = assign_hold_period_bucket(pd.Series([days]))
        assert result.iloc[0] == expected

    def test_null_is_none(self):
        result = assign_hold_period_bucket(pd.Series([float("nan")]))
        assert result.iloc[0] is None


class TestAssignMarketCapBucket:
    @pytest.mark.parametrize("cap_m,expected", [
        (999, "<$1B"), (1_000, "$1-5B"), (4_999, "$1-5B"), (5_000, "$5-20B"),
        (19_999, "$5-20B"), (20_000, "$20B+"),
    ])
    def test_boundaries(self, cap_m, expected):
        result = assign_market_cap_bucket(pd.Series([cap_m]))
        assert result.iloc[0] == expected

    def test_null_is_none(self):
        result = assign_market_cap_bucket(pd.Series([float("nan")]))
        assert result.iloc[0] is None


class TestAssignEraBucket:
    @pytest.mark.parametrize("year,expected", [
        (1998, "1998-2007"), (2007, "1998-2007"), (2008, "2008-2012"),
        (2012, "2008-2012"), (2013, "2013-2019"), (2019, "2013-2019"),
        (2020, "2020-2026"), (2026, "2020-2026"),
    ])
    def test_boundaries(self, year, expected):
        result = assign_era_bucket(pd.Series([pd.Timestamp(f"{year}-06-01")]))
        assert result.iloc[0] == expected

    def test_nat_is_none(self):
        result = assign_era_bucket(pd.Series([pd.NaT]))
        assert result.iloc[0] is None


class TestSummarizeEraInitiationCounts:
    def test_counts_all_rows_regardless_of_status(self):
        """Regression lock for the fix distinguishing this from the era
        PERFORMANCE cut: an Open row with no performance still counts here
        — this is a timing count, not an outcome summary."""
        active = clean_active_dataframe(pd.DataFrame([
            _active_raw_row(**{"OWS DB Ticker": "A", "Status": "Closed", "Initiation Date": pd.Timestamp("2021-01-01")}),
            _active_raw_row(**{"OWS DB Ticker": "B", "Status": "Open", "Initiation Date": pd.Timestamp("2021-06-01"),
                                "Close Date": None, "Close Price": None, "Absolute Performance": None,
                                "Relative SPY Performance": None, "Relative Sector Performance": None}),
            _active_raw_row(**{"OWS DB Ticker": "C", "Status": "Closed", "Initiation Date": None}),
        ]))
        counts, unassigned = summarize_era_initiation_counts(active)
        assert counts["2020-2026"] == 2  # both A and B, despite B being Open with no performance
        assert unassigned == 1  # C


# ---------------------------------------------------------------------------
# summarize_whiteboard_naive / summarize_whiteboard_chained
# ---------------------------------------------------------------------------

class TestSummarizeWhiteboardNaive:
    def test_uses_all_valid_dates_independent_of_performance_nullness(self):
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([
            _whiteboard_raw_row(**{
                "OWS DB Ticker": "HAS_PERF", "Outcome": "Removed",
                "WBA Date": pd.Timestamp("2024-01-01"), "WBR Date": pd.Timestamp("2024-06-01"),
                "Relative SPY Performance": 0.10,
            }),
            _whiteboard_raw_row(**{
                "OWS DB Ticker": "NO_PERF_HAS_DATE", "Outcome": "Removed",
                "WBA Date": pd.Timestamp("2024-01-01"), "WBR Date": pd.Timestamp("2025-01-01"),
                "Relative SPY Performance": None,
            }),
        ]))
        result = summarize_whiteboard_naive(whiteboard)
        removed_row = result[result["outcome"] == "Removed"].iloc[0]
        assert removed_row["n"] == 1  # performance-bearing rows only
        # window days uses BOTH rows (152 and 366), independent of performance
        assert removed_row["median_window_days"] != 152


class TestSummarizeWhiteboardChained:
    def test_removed_days_restricted_to_performance_bearing_rows(self):
        """Regression lock for the 259-vs-231 distinction discovered while
        verifying this module against the real file: the chained table's
        Removed row must use the SAME row population for its day count as
        for its performance metric, unlike the naive table."""
        whiteboard = clean_whiteboard_dataframe(pd.DataFrame([
            _whiteboard_raw_row(**{
                "OWS DB Ticker": "HAS_PERF", "Outcome": "Removed",
                "WBA Date": pd.Timestamp("2024-01-01"), "WBR Date": pd.Timestamp("2024-06-01"),  # ~152 days
                "Relative SPY Performance": 0.10,
            }),
            _whiteboard_raw_row(**{
                "OWS DB Ticker": "NO_PERF_HAS_DATE", "Outcome": "Removed",
                "WBA Date": pd.Timestamp("2024-01-01"), "WBR Date": pd.Timestamp("2025-01-01"),  # ~366 days
                "Relative SPY Performance": None,
            }),
        ]))
        empty_bridge = pd.DataFrame(columns=[
            "ticker", "relative_spy_performance_wb", "relative_spy_performance_act",
            "duration_days_wb", "duration_days_act",
        ])
        _, removed_summary = summarize_whiteboard_chained(empty_bridge, whiteboard)
        row = removed_summary.iloc[0]
        assert row["n"] == 1
        assert row["median_days"] == pytest.approx(152, abs=1)


# ---------------------------------------------------------------------------
# ingest_historical — end to end, dry-run, and the sign gate wired into
# the write path
# ---------------------------------------------------------------------------

@pytest.fixture
def historical_dir(tmp_path):
    """A workbook with >=2 rows per sheet and a genuine (anti-correlated)
    sign convention on both — the sign gate needs n>=2 to compute a
    correlation at all, which a single-row fixture can never satisfy."""
    upload_dir = tmp_path / "historical"
    upload_dir.mkdir()
    filepath = upload_dir / "OWS Ideas Performance test.xlsx"
    active_rows = [
        _active_raw_row(**{"OWS DB Ticker": f"T{i}", "Close Price": 100.0 - delta, "Absolute Performance": delta / 100.0})
        for i, delta in enumerate([10, 20, -10, -20, 30])
    ]
    whiteboard_rows = [
        _whiteboard_raw_row(**{"OWS DB Ticker": f"W{i}", "WBR Price": 50.0 - delta, "Absolute Performance": delta / 50.0})
        for i, delta in enumerate([5, 10, -5, -10, 15])
    ]
    with pd.ExcelWriter(filepath) as writer:
        pd.DataFrame(active_rows).to_excel(writer, sheet_name="Active Shorts Performance", index=False)
        pd.DataFrame(whiteboard_rows).to_excel(writer, sheet_name="Whiteboard Shorts Performance", index=False)
    return str(upload_dir)


@pytest.fixture
def historical_dir_sign_flipped(tmp_path):
    """A fixture whose Active sheet's performance moves WITH the stock
    (sign-flipped), engineered with enough rows for a computable
    correlation."""
    upload_dir = tmp_path / "historical"
    upload_dir.mkdir()
    filepath = upload_dir / "OWS Ideas Performance flipped.xlsx"
    active_rows = [
        _active_raw_row(**{"OWS DB Ticker": f"T{i}", "Close Price": 100.0 + delta, "Absolute Performance": delta / 100.0})
        for i, delta in enumerate([10, 20, -10, -20, 30])
    ]
    with pd.ExcelWriter(filepath) as writer:
        pd.DataFrame(active_rows).to_excel(writer, sheet_name="Active Shorts Performance", index=False)
        pd.DataFrame([_whiteboard_raw_row()]).to_excel(writer, sheet_name="Whiteboard Shorts Performance", index=False)
    return str(upload_dir)


class TestIngestHistorical:
    def test_dry_run_writes_nothing(self, historical_dir, tmp_path):
        db_path = str(tmp_path / "test.db")
        result = ingest_historical(upload_dir=historical_dir, db_path=db_path, dry_run=True)
        assert result.dry_run is True
        assert not os.path.exists(db_path)

    def test_real_run_writes_both_tables_and_provenance_row(self, historical_dir, tmp_path):
        db_path = str(tmp_path / "test.db")
        result = ingest_historical(upload_dir=historical_dir, db_path=db_path, dry_run=False)
        assert result.dry_run is False
        engine = create_engine(f"sqlite:///{db_path}")
        insp = inspect(engine)
        assert insp.has_table("historical_active_shorts")
        assert insp.has_table("historical_whiteboard_shorts")
        assert insp.has_table("historical_ingest_runs")
        runs = pd.read_sql_table("historical_ingest_runs", engine)
        assert len(runs) == 1

    def test_historical_ingest_runs_is_append_only(self, historical_dir, tmp_path):
        db_path = str(tmp_path / "test.db")
        ingest_historical(upload_dir=historical_dir, db_path=db_path, dry_run=False)
        ingest_historical(upload_dir=historical_dir, db_path=db_path, dry_run=False)
        engine = create_engine(f"sqlite:///{db_path}")
        runs = pd.read_sql_table("historical_ingest_runs", engine)
        active = pd.read_sql_table("historical_active_shorts", engine)
        assert len(runs) == 2  # append-only
        assert len(active) == 5  # replace, not doubled (5 rows in the fixture, ingested twice)

    def test_sign_gate_blocks_write_entirely(self, historical_dir_sign_flipped, tmp_path):
        db_path = str(tmp_path / "test.db")
        with pytest.raises(SignConventionError):
            ingest_historical(upload_dir=historical_dir_sign_flipped, db_path=db_path, dry_run=False)
        assert not os.path.exists(db_path)

    def test_sign_gate_blocks_dry_run_too(self, historical_dir_sign_flipped, tmp_path):
        db_path = str(tmp_path / "test.db")
        with pytest.raises(SignConventionError):
            ingest_historical(upload_dir=historical_dir_sign_flipped, db_path=db_path, dry_run=True)

