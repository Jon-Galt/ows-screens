"""
Unit tests for src/whiteboard_horizons.py (Phase 4b).

Small synthetic DataFrames throughout — no dependency on data/historical/ or
data/screener.db (both gitignored/proprietary). The 9-row backward-roll
confirmation table and every acceptance-criteria figure (149-series universe,
127/104 elapsed ceilings, promoted-arm n=21, the 101/109 replication
populations) are verified separately against the real database as part of the
mandatory end-to-end run, not asserted here — same convention as
test_historical_ingest.py.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.whiteboard_horizons import (
    EventWindowReplicationError,
    check_anchor_reconciliation,
    check_event_window_replication,
    check_price_coverage,
    compute_horizon_returns,
    flag_spurious_stored_relative,
    resolve_price_panel,
    roll_backward,
    run_whiteboard_horizons,
    summarize_by_arm_and_horizon,
    write_gap_csv,
    _to_price_map,
)


def _prices(bbg_ticker, start, n_days, close_fn, source="yfinance"):
    """n_days consecutive calendar-day rows (weekends included, matching a
    resolve-by-roll scenario) starting at `start`, close given by close_fn(i)."""
    rows = []
    for i in range(n_days):
        d = start + pd.Timedelta(days=i)
        rows.append({
            "bbg_ticker": bbg_ticker, "date": d.date(), "close": close_fn(i),
            "source": source, "vendor_symbol": bbg_ticker.split(" ")[0],
        })
    return pd.DataFrame(rows)


def _trading_days_prices(bbg_ticker, start_date, n_days, close_fn, source="yfinance"):
    """Weekday-only rows, closer to a real trading calendar — used where
    weekend-roll behavior matters."""
    rows = []
    d = start_date
    i = 0
    count = 0
    while count < n_days:
        if d.weekday() < 5:
            rows.append({
                "bbg_ticker": bbg_ticker, "date": d, "close": close_fn(i),
                "source": source, "vendor_symbol": bbg_ticker.split(" ")[0],
            })
            count += 1
            i += 1
        d = d + pd.Timedelta(days=1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# roll_backward
# ---------------------------------------------------------------------------

def test_roll_backward_exact_match():
    price_map = _to_price_map(pd.DataFrame([
        {"bbg_ticker": "AAA", "date": date(2024, 1, 5), "close": 10.0, "source": "yfinance"},
    ]))
    d, close, source, roll_days = roll_backward(price_map["AAA"], date(2024, 1, 5), cap_days=4)
    assert (d, close, source, roll_days) == (date(2024, 1, 5), 10.0, "yfinance", 0)


def test_roll_backward_rolls_to_prior_friday_over_a_weekend():
    # Sunday 2024-01-07, only Friday 2024-01-05 has a price.
    price_map = _to_price_map(pd.DataFrame([
        {"bbg_ticker": "AAA", "date": date(2024, 1, 5), "close": 47.54, "source": "yfinance"},
    ]))
    d, close, source, roll_days = roll_backward(price_map["AAA"], date(2024, 1, 7), cap_days=4)
    assert d == date(2024, 1, 5)
    assert close == 47.54
    assert roll_days == 2


def test_roll_backward_beyond_cap_returns_none():
    price_map = _to_price_map(pd.DataFrame([
        {"bbg_ticker": "AAA", "date": date(2024, 1, 1), "close": 10.0, "source": "yfinance"},
    ]))
    d, close, source, roll_days = roll_backward(price_map["AAA"], date(2024, 1, 7), cap_days=4)
    assert (d, close, source, roll_days) == (None, None, None, None)


def test_roll_backward_at_exactly_cap_days_still_resolves():
    price_map = _to_price_map(pd.DataFrame([
        {"bbg_ticker": "AAA", "date": date(2024, 1, 1), "close": 10.0, "source": "yfinance"},
    ]))
    d, close, source, roll_days = roll_backward(price_map["AAA"], date(2024, 1, 5), cap_days=4)
    assert (d, roll_days) == (date(2024, 1, 1), 4)


def test_roll_backward_never_looks_forward():
    price_map = _to_price_map(pd.DataFrame([
        {"bbg_ticker": "AAA", "date": date(2024, 1, 10), "close": 10.0, "source": "yfinance"},
    ]))
    d, close, source, roll_days = roll_backward(price_map["AAA"], date(2024, 1, 5), cap_days=10)
    assert (d, close) == (None, None)


# ---------------------------------------------------------------------------
# resolve_price_panel / not_yet_matured vs horizon_gap
# ---------------------------------------------------------------------------

def _base_whiteboard_row(**overrides):
    row = {
        "ticker": "AAA", "bbg_ticker": "AAA US Equity", "sector_benchmark_ticker": "XLK US Equity",
        "wba_date": pd.Timestamp("2024-01-05"), "wba_pricing_date": pd.Timestamp("2024-01-05"),
        "wbr_date": pd.Timestamp("2024-06-05"), "wbr_pricing_date": pd.Timestamp("2024-06-05"),
        "outcome": "Removed", "wba_price": 100.0, "wbr_price": 90.0,
        "absolute_performance": 0.10, "relative_spy_performance": 0.08,
        "benchmark_at_wba": 400.0, "benchmark_at_wbr": 432.0,
    }
    row.update(overrides)
    return row


def _rich_price_history(n_days=400, start=pd.Timestamp("2023-12-01")):
    return pd.concat([
        _trading_days_prices("AAA US Equity", start, n_days, lambda i: 100 - i * 0.05),
        _trading_days_prices("SPY US Equity", start, n_days, lambda i: 400 + i * 0.1),
        _trading_days_prices("XLK US Equity", start, n_days, lambda i: 150 + i * 0.02),
    ], ignore_index=True)


def test_resolve_price_panel_not_yet_matured_when_horizon_in_future():
    wb = pd.DataFrame([_base_whiteboard_row(wba_date=pd.Timestamp("2026-08-01"), wba_pricing_date=pd.Timestamp("2026-08-01"))])
    ph = _rich_price_history(start=pd.Timestamp("2023-12-01"), n_days=700)
    panel = resolve_price_panel(wb, ph, {}, roll_cap_days=4, lookback_days=5, run_date=date(2026, 9, 2))
    row_6mo = panel[panel["horizon_months"] == 6].iloc[0]
    row_12mo = panel[panel["horizon_months"] == 12].iloc[0]
    assert row_6mo["stock_status"] == "not_yet_matured"
    assert row_6mo["within_elapsed_ceiling"] is np.False_ or row_6mo["within_elapsed_ceiling"] == False  # noqa: E712
    assert row_12mo["stock_status"] == "not_yet_matured"


def test_resolve_price_panel_not_yet_matured_is_not_horizon_gap_even_with_no_price_data():
    """Regression test: a ticker with ZERO price_history rows must still be
    classified not_yet_matured (not a gap) when the horizon hasn't happened
    yet — maturity is a calendar fact, independent of data availability."""
    wb = pd.DataFrame([_base_whiteboard_row(
        ticker="ZZZ", bbg_ticker="ZZZ US Equity",
        wba_date=pd.Timestamp("2026-08-01"), wba_pricing_date=pd.Timestamp("2026-08-01"),
    )])
    ph = pd.DataFrame(columns=["bbg_ticker", "date", "close", "source", "vendor_symbol"])
    panel = resolve_price_panel(wb, ph, {}, roll_cap_days=4, lookback_days=5, run_date=date(2026, 9, 2))
    assert (panel["stock_status"] == "not_yet_matured").all()
    gaps = check_price_coverage(panel, ph, roll_cap_days=4, lookback_days=5)
    assert gaps.empty


def test_resolve_price_panel_horizon_gap_when_matured_but_no_data():
    wb = pd.DataFrame([_base_whiteboard_row(
        ticker="ZZZ", bbg_ticker="ZZZ US Equity",
        wba_date=pd.Timestamp("2024-01-05"), wba_pricing_date=pd.Timestamp("2024-01-05"),
    )])
    ph = pd.DataFrame(columns=["bbg_ticker", "date", "close", "source", "vendor_symbol"])
    panel = resolve_price_panel(wb, ph, {}, roll_cap_days=4, lookback_days=5, run_date=date(2026, 9, 2))
    assert (panel["stock_status"] == "no_series").all()
    gaps = check_price_coverage(panel, ph, roll_cap_days=4, lookback_days=5)
    assert set(gaps["reason"]) == {"no_price_history_for_ticker"}


# ---------------------------------------------------------------------------
# compute_horizon_returns
# ---------------------------------------------------------------------------

def test_compute_horizon_returns_hand_computed_example():
    panel = pd.DataFrame([{
        "bbg_ticker": "AAA US Equity",
        "stock_anchor_price": 100.0, "stock_horizon_price": 90.0, "stock_status": "measurable",
        "spy_anchor_price": 400.0, "spy_horizon_price": 440.0, "spy_status": "measurable",
        "sector_anchor_price": 150.0, "sector_horizon_price": 165.0, "sector_status": "measurable",
    }])
    out = compute_horizon_returns(panel)
    # price_move = (90-100)/100 = -0.10 -> absolute_short_pnl = +0.10
    assert out.loc[0, "absolute_short_pnl"] == pytest.approx(0.10)
    # spy_move = 0.10; relative_spy = 0.10 - (-0.10) = 0.20
    assert out.loc[0, "relative_spy_short_pnl"] == pytest.approx(0.20)
    # sector_move = 0.10; relative_sector = 0.10 - (-0.10) = 0.20
    assert out.loc[0, "relative_sector_short_pnl"] == pytest.approx(0.20)


def test_compute_horizon_returns_nan_when_leg_unmeasurable():
    panel = pd.DataFrame([{
        "bbg_ticker": "AAA US Equity",
        "stock_anchor_price": 100.0, "stock_horizon_price": 90.0, "stock_status": "measurable",
        "spy_anchor_price": None, "spy_horizon_price": None, "spy_status": "anchor_gap",
        "sector_anchor_price": 150.0, "sector_horizon_price": 165.0, "sector_status": "measurable",
    }])
    out = compute_horizon_returns(panel)
    assert out.loc[0, "absolute_measurable"]
    assert not out.loc[0, "relative_spy_measurable"]
    assert pd.isna(out.loc[0, "relative_spy_short_pnl"])
    assert out.loc[0, "relative_sector_measurable"]


def test_compute_horizon_returns_currency_mismatch_flag():
    panel = pd.DataFrame([
        {"bbg_ticker": "AAA US Equity", "stock_anchor_price": 1.0, "stock_horizon_price": 1.0, "stock_status": "measurable",
         "spy_anchor_price": 1.0, "spy_horizon_price": 1.0, "spy_status": "measurable",
         "sector_anchor_price": 1.0, "sector_horizon_price": 1.0, "sector_status": "measurable"},
        {"bbg_ticker": "SN/ LN Equity", "stock_anchor_price": 1.0, "stock_horizon_price": 1.0, "stock_status": "measurable",
         "spy_anchor_price": 1.0, "spy_horizon_price": 1.0, "spy_status": "measurable",
         "sector_anchor_price": 1.0, "sector_horizon_price": 1.0, "sector_status": "measurable"},
    ])
    out = compute_horizon_returns(panel)
    assert out.loc[0, "currency_mismatch"] == False  # noqa: E712
    assert out.loc[1, "currency_mismatch"] == True  # noqa: E712


# ---------------------------------------------------------------------------
# check_price_coverage / write_gap_csv — POSITIVE TEST (b)
# ---------------------------------------------------------------------------

def test_check_price_coverage_flags_missing_horizon_bar_and_writes_gap_csv(tmp_path):
    """Acceptance criterion 9b: remove a covered name's horizon-date bar (and
    the lookback window around it), prove check_price_coverage flags it AND
    that it appears in the written GAPS_<rundate>.csv."""
    wb = pd.DataFrame([_base_whiteboard_row(
        ticker="COV", bbg_ticker="COV US Equity",
        wba_date=pd.Timestamp("2024-01-05"), wba_pricing_date=pd.Timestamp("2024-01-05"),
    )])
    ph = _rich_price_history(start=pd.Timestamp("2023-12-01"), n_days=400)
    ph = pd.concat([ph, _trading_days_prices("COV US Equity", pd.Timestamp("2023-12-01"), 400, lambda i: 50 - i * 0.01)], ignore_index=True)

    horizon_target = date(2024, 7, 5)  # 2024-01-05 + 6 months
    ph = ph[~(
        (ph["bbg_ticker"] == "COV US Equity")
        & (pd.to_datetime(ph["date"]).dt.date >= horizon_target - pd.Timedelta(days=5))
        & (pd.to_datetime(ph["date"]).dt.date <= horizon_target)
    )]

    panel = resolve_price_panel(wb, ph, {"COV US Equity": "COV"}, roll_cap_days=4, lookback_days=5, run_date=date(2026, 9, 2))
    six_month = panel[panel["horizon_months"] == 6].iloc[0]
    assert six_month["stock_status"] == "horizon_gap"

    gaps = check_price_coverage(panel, ph, roll_cap_days=4, lookback_days=5)
    cov_gaps = gaps[(gaps["ticker"] == "COV") & (gaps["horizon_months"] == 6) & (gaps["leg"] == "stock")]
    assert len(cov_gaps) == 1
    assert cov_gaps.iloc[0]["reason"] == "horizon_unresolved_beyond_lookback_cap"

    path = write_gap_csv(gaps, "2026-09-02", out_dir=str(tmp_path))
    written = pd.read_csv(path)
    assert ((written["ticker"] == "COV") & (written["horizon_months"] == 6)).any()


def test_write_gap_csv_writes_empty_file_when_no_gaps(tmp_path):
    empty = pd.DataFrame(columns=["bbg_ticker", "vendor_symbol", "leg", "ticker", "wba_date",
                                   "horizon_months", "required_start", "required_end",
                                   "last_available_date", "reason"])
    path = write_gap_csv(empty, "2026-09-02", out_dir=str(tmp_path))
    written = pd.read_csv(path)
    assert len(written) == 0


# ---------------------------------------------------------------------------
# check_anchor_reconciliation — POSITIVE TEST (a)
# ---------------------------------------------------------------------------

def test_check_anchor_reconciliation_flags_wrong_company_symbol():
    """Acceptance criterion 9a: map one ticker to a wrong-company (recycled
    symbol) price and prove check_anchor_reconciliation flags it."""
    wb = pd.DataFrame([
        _base_whiteboard_row(ticker="AAA", bbg_ticker="AAA US Equity", wba_price=100.0),
        _base_whiteboard_row(ticker="BBB", bbg_ticker="BBB US Equity", wba_price=50.0),
    ])
    panel = pd.DataFrame([
        {"ticker": "AAA", "wba_date": date(2024, 1, 5), "horizon_months": 6, "stock_anchor_price": 100.05},
        {"ticker": "AAA", "wba_date": date(2024, 1, 5), "horizon_months": 12, "stock_anchor_price": 100.05},
        # BBB's vendor anchor price is wildly inconsistent with its stored
        # wba_price of 50.0 — simulating a recycled ticker now pointing at a
        # different, much larger company.
        {"ticker": "BBB", "wba_date": date(2024, 1, 5), "horizon_months": 6, "stock_anchor_price": 4500.0},
        {"ticker": "BBB", "wba_date": date(2024, 1, 5), "horizon_months": 12, "stock_anchor_price": 4500.0},
    ])
    result = check_anchor_reconciliation(wb, panel, tolerance=0.02)
    assert result.n_attempted == 2
    assert result.n_evaluated == 2
    assert result.violation_count == 1
    assert "BBB" in result.violation_tickers


def test_check_anchor_reconciliation_reports_unevaluable_not_asserts_105():
    wb = pd.DataFrame([
        _base_whiteboard_row(ticker="AAA", bbg_ticker="AAA US Equity", wba_price=100.0),
        _base_whiteboard_row(ticker="BBB", bbg_ticker="BBB US Equity", wba_price=50.0),
    ])
    panel = pd.DataFrame([
        {"ticker": "AAA", "wba_date": date(2024, 1, 5), "horizon_months": 6, "stock_anchor_price": 100.0},
        # BBB never resolved a vendor anchor price (a coverage gap on a row
        # that DOES have a stored price) — must show up as unevaluable, not
        # silently shrink a "clean" evaluated base.
        {"ticker": "BBB", "wba_date": date(2024, 1, 5), "horizon_months": 6, "stock_anchor_price": None},
    ])
    result = check_anchor_reconciliation(wb, panel, tolerance=0.02)
    assert result.n_attempted == 2
    assert result.n_evaluated == 1
    assert result.n_unevaluable == 1


# ---------------------------------------------------------------------------
# check_event_window_replication
# ---------------------------------------------------------------------------

def _replication_whiteboard(n_rows, corrupt_last=False):
    rows = []
    for i in range(n_rows):
        rows.append(_base_whiteboard_row(
            ticker=f"T{i}", bbg_ticker=f"T{i} US Equity",
            wba_date=pd.Timestamp("2024-01-05") + pd.Timedelta(days=i),
            wba_pricing_date=pd.Timestamp("2024-01-05") + pd.Timedelta(days=i),
            wbr_date=pd.Timestamp("2024-06-05") + pd.Timedelta(days=i),
            wbr_pricing_date=pd.Timestamp("2024-06-05") + pd.Timedelta(days=i),
            wba_price=None, wbr_price=None,
            absolute_performance=0.10, relative_spy_performance=0.08,
        ))
    return pd.DataFrame(rows)


def _replication_price_history(n_rows, source="yfinance"):
    frames = []
    for i in range(n_rows):
        ticker = f"T{i} US Equity"
        anchor = pd.Timestamp("2024-01-05") + pd.Timedelta(days=i)
        # 220 calendar days of daily closes bracketing anchor->wbr (~150 days)
        frames.append(_trading_days_prices(ticker, anchor - pd.Timedelta(days=5), 220, lambda j: 100 - j * 0.1, source=source))
    frames.append(_trading_days_prices("SPY US Equity", pd.Timestamp("2024-01-01"), 250, lambda j: 400 + j * 0.05, source=source))
    return pd.concat(frames, ignore_index=True)


def test_check_event_window_replication_happy_path_passes():
    wb = _replication_whiteboard(40)
    ph = _replication_price_history(40)
    results = check_event_window_replication(
        wb, ph, roll_cap_days=4, lookback_days=5, abs_tolerance=0.5,
        min_corr=-1.0, min_evaluated_frac=0.5, min_evaluated_abs=5,
    )
    # min_corr=-1.0 as a permissive floor here since this test's purpose is
    # exercising the evaluated-population plumbing, not the real 101/109
    # correlation (that's verified on the real run) — n_evaluated should be
    # close to n_attempted given rich synthetic price coverage.
    assert results["absolute"].n_attempted == 40
    assert results["absolute"].n_evaluated >= 35
    assert results["relative"].n_attempted == 40


def test_check_event_window_replication_insufficient_evaluated_population_fails_even_with_perfect_correlation():
    """Regression test for the defect the PM caught: n_attempted=20, only 5
    evaluable, but the 5 survivors correlate at 1.0. Must FAIL on population
    grounds (failure_reason="insufficient_evaluated_population"), not pass
    because the correlation floor alone was satisfied."""
    wb = _replication_whiteboard(20)
    # Only give 5 of the 20 tickers any price history at all -> only those 5
    # can possibly evaluate, and because absolute_performance is constant
    # (0.10) across all rows in _replication_whiteboard while the 5 priced
    # tickers share an identical price path, their computed values are
    # perfectly correlated (degenerate corr=1.0 or nan on constant input —
    # forced non-degenerate by staggering closes slightly below).
    frames = []
    for i in range(5):
        ticker = f"T{i} US Equity"
        anchor = pd.Timestamp("2024-01-05") + pd.Timedelta(days=i)
        frames.append(_trading_days_prices(ticker, anchor - pd.Timedelta(days=5), 220, lambda j, i=i: 100 - j * (0.1 + i * 0.001)))
    frames.append(_trading_days_prices("SPY US Equity", pd.Timestamp("2024-01-01"), 250, lambda j: 400 + j * 0.05))
    ph = pd.concat(frames, ignore_index=True)

    results = check_event_window_replication(
        wb, ph, roll_cap_days=4, lookback_days=5, abs_tolerance=0.5,
        min_corr=0.0, min_evaluated_frac=0.80, min_evaluated_abs=30,
    )
    absolute = results["absolute"]
    assert absolute.n_attempted == 20
    assert absolute.n_evaluated == 5
    assert absolute.min_evaluated_required == pytest.approx(max(30, 0.80 * 20))
    assert absolute.passed is False
    assert absolute.failure_reason == "insufficient_evaluated_population"


def test_check_event_window_replication_corr_below_floor_distinct_failure_reason():
    wb = _replication_whiteboard(30)
    wb = wb.copy()
    # Flip the sign of every stored absolute_performance relative to what the
    # vendor prices (a steady decline) will compute -> strong negative corr.
    wb["absolute_performance"] = -0.10
    ph = _replication_price_history(30)
    results = check_event_window_replication(
        wb, ph, roll_cap_days=4, lookback_days=5, abs_tolerance=0.5,
        min_corr=0.95, min_evaluated_frac=0.5, min_evaluated_abs=5,
    )
    assert results["absolute"].passed is False
    assert results["absolute"].failure_reason == "corr_below_floor"


def test_check_event_window_replication_relative_only_rows_included_not_filtered():
    """Regression test: a row with no stored wba_price/wbr_price at all (like
    the 9 real relative-only rows) must still be a full member of the
    RELATIVE population — not treated as damaged or excluded."""
    row = _base_whiteboard_row(
        ticker="RELONLY", bbg_ticker="RELONLY US Equity",
        wba_price=None, wbr_price=None,
        absolute_performance=None, relative_spy_performance=0.08,
    )
    wb = pd.DataFrame([row])
    ph = pd.concat([
        _trading_days_prices("RELONLY US Equity", pd.Timestamp("2024-01-01"), 220, lambda j: 100 - j * 0.1),
        _trading_days_prices("SPY US Equity", pd.Timestamp("2024-01-01"), 220, lambda j: 400 + j * 0.05),
    ], ignore_index=True)
    results = check_event_window_replication(
        wb, ph, roll_cap_days=4, lookback_days=5, abs_tolerance=1.0,
        min_corr=-1.0, min_evaluated_frac=0.0, min_evaluated_abs=0,
    )
    assert results["absolute"].n_attempted == 0  # no absolute_performance -> not in absolute population
    assert results["relative"].n_attempted == 1
    assert results["relative"].n_evaluated == 1


def test_check_event_window_replication_vendor_breakdown_partitions_by_source():
    wb = _replication_whiteboard(10)
    ph = _replication_price_history(10, source="stooq")
    results = check_event_window_replication(
        wb, ph, roll_cap_days=4, lookback_days=5, abs_tolerance=1.0,
        min_corr=-1.0, min_evaluated_frac=0.0, min_evaluated_abs=0,
    )
    breakdown = results["absolute"].vendor_breakdown
    assert breakdown["stooq"]["n"] >= 1
    assert breakdown["yfinance"] == "n/a — no yfinance-sourced series this run"


def test_check_event_window_replication_n_evaluated_below_two_fails_explicitly():
    wb = _replication_whiteboard(1)
    ph = _replication_price_history(1)
    results = check_event_window_replication(
        wb, ph, roll_cap_days=4, lookback_days=5, abs_tolerance=1.0,
        min_corr=-1.0, min_evaluated_frac=0.0, min_evaluated_abs=0,
    )
    assert results["absolute"].failure_reason == "insufficient_n"
    assert results["absolute"].passed is False


# ---------------------------------------------------------------------------
# flag_spurious_stored_relative
# ---------------------------------------------------------------------------

def _wb_row_for_spurious_check(ticker, outcome, wba_price, wbr_price,
                                relative_spy_performance, benchmark_at_wba, benchmark_at_wbr):
    return {
        "ticker": ticker, "outcome": outcome, "wba_price": wba_price, "wbr_price": wbr_price,
        "relative_spy_performance": relative_spy_performance,
        "benchmark_at_wba": benchmark_at_wba, "benchmark_at_wbr": benchmark_at_wbr,
        "wba_date": pd.Timestamp("2024-01-05"),
    }


def test_flag_spurious_stored_relative_fires_on_null_price_pattern():
    """The real defect: BOTH stock prices null, and stored
    relative_spy_performance equals the SPY move alone (price_move silently
    treated as zero)."""
    wb = pd.DataFrame([_wb_row_for_spurious_check(
        "ROKU", "Removed", wba_price=None, wbr_price=None,
        relative_spy_performance=0.15, benchmark_at_wba=400.0, benchmark_at_wbr=460.0,  # spy_move = 0.15
    )])
    flagged = flag_spurious_stored_relative(wb, tolerance=0.01)
    assert len(flagged) == 1
    assert flagged.iloc[0]["ticker"] == "ROKU"


def test_flag_spurious_stored_relative_does_not_fire_on_normal_row():
    """A normal row: both prices present, stored value reflects a real
    stock-specific move distinct from the SPY move."""
    wb = pd.DataFrame([_wb_row_for_spurious_check(
        "AAA", "Removed", wba_price=100.0, wbr_price=80.0,
        relative_spy_performance=0.35, benchmark_at_wba=400.0, benchmark_at_wbr=460.0,  # spy_move=0.15, not 0.35
    )])
    flagged = flag_spurious_stored_relative(wb, tolerance=0.01)
    assert flagged.empty


def test_flag_spurious_stored_relative_load_bearing_case_does_not_fire():
    """THE test that matters (PM's instruction): a stock that genuinely went
    nowhere, with BOTH prices present, coincidentally matches the SPY move —
    this real-data case (ALAB in the live database) must NOT be flagged. If
    the null-price condition were dropped, this row would incorrectly fire."""
    wb = pd.DataFrame([_wb_row_for_spurious_check(
        "ALAB", "Note", wba_price=91.02, wbr_price=90.80,  # stock barely moved
        relative_spy_performance=0.02, benchmark_at_wba=400.0, benchmark_at_wbr=406.8,  # spy_move ~= 0.017, close to stored 0.02
    )])
    flagged = flag_spurious_stored_relative(wb, tolerance=0.01)
    assert flagged.empty


def test_flag_spurious_stored_relative_fires_when_only_one_price_leg_null():
    """The Open-arm case: wba_price missing (no anchor recorded yet at the
    time), wbr_price present (current mark) — still a null-as-zero artifact,
    not just the both-null case."""
    wb = pd.DataFrame([_wb_row_for_spurious_check(
        "HUBB", "Open", wba_price=None, wbr_price=450.82,
        relative_spy_performance=0.15, benchmark_at_wba=400.0, benchmark_at_wbr=460.0,
    )])
    flagged = flag_spurious_stored_relative(wb, tolerance=0.01)
    assert len(flagged) == 1


def test_flag_spurious_stored_relative_does_not_modify_source_dataframe():
    wb = pd.DataFrame([_wb_row_for_spurious_check(
        "ROKU", "Removed", wba_price=None, wbr_price=None,
        relative_spy_performance=0.15, benchmark_at_wba=400.0, benchmark_at_wbr=460.0,
    )])
    original = wb.copy(deep=True)
    flag_spurious_stored_relative(wb, tolerance=0.01)
    pd.testing.assert_frame_equal(wb, original)


# ---------------------------------------------------------------------------
# summarize_by_arm_and_horizon
# ---------------------------------------------------------------------------

def test_summarize_by_arm_and_horizon_reports_all_arms_even_when_fully_unmeasurable():
    panel = pd.DataFrame([
        {"outcome": "Removed", "horizon_months": 6, "within_elapsed_ceiling": True,
         "absolute_measurable": True, "absolute_short_pnl": 0.10,
         "relative_spy_measurable": True, "relative_spy_short_pnl": 0.08},
        {"outcome": "Note", "horizon_months": 6, "within_elapsed_ceiling": True,
         "absolute_measurable": False, "absolute_short_pnl": np.nan,
         "relative_spy_measurable": False, "relative_spy_short_pnl": np.nan},
        {"outcome": None, "horizon_months": 6, "within_elapsed_ceiling": True,
         "absolute_measurable": True, "absolute_short_pnl": -0.05,
         "relative_spy_measurable": True, "relative_spy_short_pnl": -0.02},
    ])
    summary = summarize_by_arm_and_horizon(panel)
    outcomes = set(summary["outcome"])
    assert {"Removed", "Note", "null"}.issubset(outcomes)
    note_abs = summary[(summary["outcome"] == "Note") & (summary["measure"] == "absolute")].iloc[0]
    assert note_abs["n_ceiling"] == 1
    assert note_abs["n_measurable"] == 0
    assert note_abs["n_gap"] == 1
    assert pd.isna(note_abs["median"])


def test_summarize_by_arm_and_horizon_not_yet_matured_excluded_from_ceiling_and_counted_separately():
    panel = pd.DataFrame([
        {"outcome": "Open", "horizon_months": 12, "within_elapsed_ceiling": True,
         "absolute_measurable": True, "absolute_short_pnl": 0.05,
         "relative_spy_measurable": True, "relative_spy_short_pnl": 0.03},
        {"outcome": "Open", "horizon_months": 12, "within_elapsed_ceiling": False,
         "absolute_measurable": False, "absolute_short_pnl": np.nan,
         "relative_spy_measurable": False, "relative_spy_short_pnl": np.nan},
    ])
    summary = summarize_by_arm_and_horizon(panel)
    row = summary[(summary["outcome"] == "Open") & (summary["measure"] == "absolute")].iloc[0]
    assert row["n_ceiling"] == 1  # the not-yet-matured row excluded from the ceiling population
    assert row["n_not_yet_matured"] == 1
    assert row["n_measurable"] == 1


# ---------------------------------------------------------------------------
# run_whiteboard_horizons ordering — the item-3 regression test
# ---------------------------------------------------------------------------

def _seed_whiteboard_and_prices(engine, n_rows=30, corrupt_sign=False):
    wb = _replication_whiteboard(n_rows)
    wb.to_sql("historical_whiteboard_shorts", engine, index=False)
    if corrupt_sign:
        with engine.begin() as conn:
            conn.execute(text("UPDATE historical_whiteboard_shorts SET absolute_performance = -0.10"))
    ph = _replication_price_history(n_rows)
    from src.price_history import upsert_price_history
    upsert_price_history(engine, ph)


def _config(min_corr=0.95, min_evaluated_frac=0.5, min_evaluated_abs=5):
    return {
        "prices": {
            "anchor_roll_cap_days": 4, "horizon_lookback_days": 5,
            "anchor_reconciliation_rel_tolerance": 0.02,
            "event_window_min_corr": min_corr,
            "event_window_min_evaluated_frac": min_evaluated_frac,
            "event_window_min_evaluated_abs": min_evaluated_abs,
            "event_window_abs_tolerance": 0.5,
        }
    }


def test_run_whiteboard_horizons_happy_path_writes_both_tables_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = create_engine("sqlite:///:memory:")
    _seed_whiteboard_and_prices(engine, n_rows=30)
    result = run_whiteboard_horizons(engine, _config(min_corr=-1.0, min_evaluated_frac=0.5, min_evaluated_abs=5), run_date=date(2026, 9, 2))
    assert result["rows_written"] > 0
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM whiteboard_horizon_runs")).scalar() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM whiteboard_horizon_returns")).scalar() == result["rows_written"]


def test_run_whiteboard_horizons_ordering_on_gate_failure(tmp_path, monkeypatch):
    """Acceptance criterion / item-3 regression: on a replication-gate
    failure, the gap CSV must exist, the diagnostic result must be
    returned/raised with detail, EventWindowReplicationError must be raised,
    and NEITHER database table may have gained a row."""
    monkeypatch.chdir(tmp_path)
    engine = create_engine("sqlite:///:memory:")
    _seed_whiteboard_and_prices(engine, n_rows=30, corrupt_sign=True)

    with pytest.raises(EventWindowReplicationError):
        run_whiteboard_horizons(engine, _config(min_corr=0.95, min_evaluated_frac=0.5, min_evaluated_abs=5), run_date=date(2026, 9, 2))

    gap_files = list((tmp_path / "data" / "historical" / "prices").glob("GAPS_*.csv"))
    assert len(gap_files) == 1

    with engine.connect() as conn:
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        table_names = {t[0] for t in tables}
    assert "whiteboard_horizon_returns" not in table_names
    assert "whiteboard_horizon_runs" not in table_names
