"""
Phase 4b — fixed-horizon Whiteboard outcome measurement.

Replaces Phase 4a's event-terminated Whiteboard windows (WBA -> whatever ended
the idea) with a COMMON ANCHOR AND A FIXED CLOCK: every Whiteboard idea
measured from its own WBA date over the same 6 and 12 months, whatever
happened to it afterwards. That is the only construction under which the four
outcome arms (Removed / Initiation / Note / Open) are comparable — see
PHASE4B_SCOPE.md section 1.

THIS IS NOT A SCREEN. Same standing as historical_ingest.py / price_history.py.

Architecture Rule 1 discipline: every function up to and including
run_whiteboard_horizons' pure helpers takes DataFrames/dicts and returns
DataFrames/dicts, no SQLAlchemy. Only run_whiteboard_horizons itself (the
orchestrator) touches the database and the filesystem.

Label definition (fixed in writing by the PM, not re-derived here):
  - Anchor date = wba_pricing_date (== wba_date, 152/152, verified).
  - Anchor price = vendor close on the anchor date, backward-rolled up to
    anchor_roll_cap_days on a non-trading day or short data gap. No forward
    look. Beyond the cap: unmeasurable, flagged, never dropped.
  - Horizon date = the ORIGINAL anchor date (wba_pricing_date, NOT the rolled
    trading date used to price it) + 6 or 12 calendar months. If that date has
    no price, look back up to horizon_lookback_days for the latest close
    on-or-before it. Beyond that: unmeasurable, flagged.
  - A horizon date in the future (relative to the run date) is
    "not_yet_matured" — distinct from a genuine data gap. Conflating the two
    would send a BDH pull request for data that doesn't exist yet.
  - Price basis: split-adjusted, dividend-unadjusted close, identical across
    stock/SPY/sector legs (see price_history.py).
  - Returns: absolute_short_pnl = -price_move; relative_spy_short_pnl =
    spy_move - price_move; relative_sector_short_pnl = sector_move -
    price_move. Short P&L, not stock return.
  - Currency: no FX translation (Driver ruling). currency_mismatch is a
    descriptive flag on the 7 non-US names (+ PRY.IM), never an exclusion.
  - No survivorship exclusion, no imputation: a missing price is flagged to
    the gap report, never guessed.
"""

import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import inspect, text

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.db import append_rows
from src.price_history import SPY_BBG_TICKER, default_vendor_symbol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZONS_MONTHS = (6, 12)
LEGS = ("stock", "spy", "sector")


class EventWindowReplicationError(ValueError):
    """Raised by run_whiteboard_horizons when check_event_window_replication
    fails on either leg. Aborts the DATABASE write — the gap CSV and the
    printed diagnostic report are produced first and survive this raise; see
    run_whiteboard_horizons' docstring for the ordering rationale."""


def _to_price_map(price_history_df: pd.DataFrame) -> dict:
    """Build {bbg_ticker: {date: (close, source)}} for O(1) roll/lookback
    lookups. date keys are python date objects."""
    price_map = {}
    if price_history_df.empty:
        return price_map
    df = price_history_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for bbg_ticker, group in df.groupby("bbg_ticker"):
        price_map[bbg_ticker] = {
            row.date: (row.close, row.source) for row in group.itertuples(index=False)
        }
    return price_map


def roll_backward(ticker_prices: dict, target_date: date, cap_days: int):
    """Find target_date's price, or the latest available price up to
    cap_days of calendar days before it (no forward look).

    Args:
        ticker_prices: {date: (close, source)} for one ticker, from
            _to_price_map.
        target_date: The date to price.
        cap_days: Maximum calendar days to roll backward.

    Returns:
        (resolved_date, close, source, roll_days) if found within the cap,
        else (None, None, None, None).
    """
    if not ticker_prices:
        return None, None, None, None
    for offset in range(0, cap_days + 1):
        candidate = target_date - timedelta(days=offset)
        if candidate in ticker_prices:
            close, source = ticker_prices[candidate]
            return candidate, close, source, offset
    return None, None, None, None


def _resolve_leg(ticker_prices: dict, anchor_original: date, horizon_months: int,
                  roll_cap_days: int, lookback_days: int, run_date: date) -> dict:
    """Resolve one leg's (anchor, horizon) prices for one (row, horizon_months).

    Returns a dict of the leg's resolved fields. status is one of:
      "measurable"      — both anchor and horizon prices resolved
      "anchor_gap"       — anchor unresolved beyond roll_cap_days
      "horizon_gap"       — anchor resolved, horizon unresolved beyond lookback_days
      "not_yet_matured"  — the horizon date is still in the future
      "no_series"         — ticker_prices is empty (no price_history rows at all)
    """
    horizon_target = anchor_original + relativedelta(months=horizon_months)

    # Maturity is a calendar fact, independent of whether we happen to have
    # any price data for this ticker — check it FIRST. Checking data
    # availability first would misreport a ticker with zero price_history
    # rows as an "anchor_gap"/"no_series" coverage gap even when the horizon
    # hasn't happened yet and no vendor could possibly have that data — that
    # is exactly the "gap vs too-early" conflation the module docstring
    # warns against.
    if horizon_target > run_date:
        return {
            "anchor_date_used": None, "anchor_price": None, "anchor_roll_days": None,
            "horizon_date_used": None, "horizon_price": None, "horizon_lookback_days_used": None,
            "horizon_target": horizon_target, "status": "not_yet_matured",
        }

    if not ticker_prices:
        return {
            "anchor_date_used": None, "anchor_price": None, "anchor_roll_days": None,
            "horizon_date_used": None, "horizon_price": None, "horizon_lookback_days_used": None,
            "horizon_target": horizon_target, "status": "no_series",
        }

    anchor_date_used, anchor_price, _anchor_source, anchor_roll_days = roll_backward(
        ticker_prices, anchor_original, roll_cap_days
    )
    if anchor_date_used is None:
        return {
            "anchor_date_used": None, "anchor_price": None, "anchor_roll_days": None,
            "horizon_date_used": None, "horizon_price": None, "horizon_lookback_days_used": None,
            "horizon_target": horizon_target, "status": "anchor_gap",
        }

    horizon_date_used, horizon_price, _horizon_source, horizon_lookback_used = roll_backward(
        ticker_prices, horizon_target, lookback_days
    )
    if horizon_date_used is None:
        return {
            "anchor_date_used": anchor_date_used, "anchor_price": anchor_price,
            "anchor_roll_days": anchor_roll_days,
            "horizon_date_used": None, "horizon_price": None, "horizon_lookback_days_used": None,
            "horizon_target": horizon_target, "status": "horizon_gap",
        }

    return {
        "anchor_date_used": anchor_date_used, "anchor_price": anchor_price,
        "anchor_roll_days": anchor_roll_days,
        "horizon_date_used": horizon_date_used, "horizon_price": horizon_price,
        "horizon_lookback_days_used": horizon_lookback_used,
        "horizon_target": horizon_target, "status": "measurable",
    }


def resolve_price_panel(
    whiteboard_df: pd.DataFrame, price_history_df: pd.DataFrame, symbol_map: dict,
    roll_cap_days: int, lookback_days: int, run_date: date,
) -> pd.DataFrame:
    """Resolve anchor/horizon prices for every (ticker, wba_date, horizon)
    triple, for all three legs (stock, SPY, sector). Single source of gap
    truth — every downstream consumer (returns, coverage, summaries) reads
    this output rather than re-deriving resolution.

    Args:
        whiteboard_df: historical_whiteboard_shorts (ticker, bbg_ticker,
            wba_date, wba_pricing_date, outcome, sector_benchmark_ticker
            columns required; wba_pricing_date must equal wba_date, per
            PHASE4B_SCOPE.md's verified fact — this is not re-checked here).
        price_history_df: price_history (bbg_ticker, date, close, source).
        symbol_map: {bbg_ticker: vendor_symbol}, for gap-report traceability.
        roll_cap_days, lookback_days: config prices.anchor_roll_cap_days /
            horizon_lookback_days.
        run_date: "today" for the not_yet_matured determination.

    Returns:
        Long-format DataFrame, one row per (ticker, wba_date, horizon_months).
    """
    price_map = _to_price_map(price_history_df)
    rows = []

    for wb_row in whiteboard_df.itertuples(index=False):
        anchor_original = pd.Timestamp(wb_row.wba_pricing_date).date()
        sector_ticker = wb_row.sector_benchmark_ticker if pd.notna(wb_row.sector_benchmark_ticker) else None

        for horizon_months in HORIZONS_MONTHS:
            stock_leg = _resolve_leg(
                price_map.get(wb_row.bbg_ticker, {}), anchor_original, horizon_months,
                roll_cap_days, lookback_days, run_date,
            )
            spy_leg = _resolve_leg(
                price_map.get(SPY_BBG_TICKER, {}), anchor_original, horizon_months,
                roll_cap_days, lookback_days, run_date,
            )
            if sector_ticker is not None:
                sector_leg = _resolve_leg(
                    price_map.get(sector_ticker, {}), anchor_original, horizon_months,
                    roll_cap_days, lookback_days, run_date,
                )
            else:
                sector_leg = {
                    "anchor_date_used": None, "anchor_price": None, "anchor_roll_days": None,
                    "horizon_date_used": None, "horizon_price": None,
                    "horizon_lookback_days_used": None,
                    "horizon_target": anchor_original + relativedelta(months=horizon_months),
                    "status": "no_sector_benchmark",
                }

            within_elapsed_ceiling = stock_leg["status"] != "not_yet_matured"

            rows.append({
                "ticker": wb_row.ticker,
                "bbg_ticker": wb_row.bbg_ticker,
                "wba_date": anchor_original,
                "outcome": wb_row.outcome,
                "horizon_months": horizon_months,
                "within_elapsed_ceiling": within_elapsed_ceiling,
                "stock_vendor_symbol": symbol_map.get(wb_row.bbg_ticker),
                "stock_anchor_date_used": stock_leg["anchor_date_used"],
                "stock_anchor_price": stock_leg["anchor_price"],
                "stock_anchor_roll_days": stock_leg["anchor_roll_days"],
                "stock_horizon_date_used": stock_leg["horizon_date_used"],
                "stock_horizon_price": stock_leg["horizon_price"],
                "stock_horizon_lookback_days_used": stock_leg["horizon_lookback_days_used"],
                "stock_horizon_target": stock_leg["horizon_target"],
                "stock_status": stock_leg["status"],
                "spy_vendor_symbol": symbol_map.get(SPY_BBG_TICKER),
                "spy_anchor_date_used": spy_leg["anchor_date_used"],
                "spy_anchor_price": spy_leg["anchor_price"],
                "spy_anchor_roll_days": spy_leg["anchor_roll_days"],
                "spy_horizon_date_used": spy_leg["horizon_date_used"],
                "spy_horizon_price": spy_leg["horizon_price"],
                "spy_horizon_lookback_days_used": spy_leg["horizon_lookback_days_used"],
                "spy_horizon_target": spy_leg["horizon_target"],
                "spy_status": spy_leg["status"],
                "sector_ticker": sector_ticker,
                "sector_vendor_symbol": symbol_map.get(sector_ticker) if sector_ticker else None,
                "sector_anchor_date_used": sector_leg["anchor_date_used"],
                "sector_anchor_price": sector_leg["anchor_price"],
                "sector_anchor_roll_days": sector_leg["anchor_roll_days"],
                "sector_horizon_date_used": sector_leg["horizon_date_used"],
                "sector_horizon_price": sector_leg["horizon_price"],
                "sector_horizon_lookback_days_used": sector_leg["horizon_lookback_days_used"],
                "sector_horizon_target": sector_leg["horizon_target"],
                "sector_status": sector_leg["status"],
            })

    return pd.DataFrame(rows)


def compute_horizon_returns(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Add return columns and the currency_mismatch flag to resolve_price_panel's
    output. NaN wherever the relevant leg(s) aren't measurable — Architecture
    Rule 3 (never raise, never guess).

    Returns formulas match historical_ingest.py's check_benchmark_consistency
    convention exactly: absolute_short_pnl = -price_move; relative_*_short_pnl
    = bench_move - price_move.
    """
    df = panel_df.copy()

    df["absolute_measurable"] = df["stock_status"] == "measurable"
    df["relative_spy_measurable"] = df["absolute_measurable"] & (df["spy_status"] == "measurable")
    df["relative_sector_measurable"] = df["absolute_measurable"] & (df["sector_status"] == "measurable")

    stock_move = (df["stock_horizon_price"] - df["stock_anchor_price"]) / df["stock_anchor_price"]
    spy_move = (df["spy_horizon_price"] - df["spy_anchor_price"]) / df["spy_anchor_price"]
    sector_move = (df["sector_horizon_price"] - df["sector_anchor_price"]) / df["sector_anchor_price"]

    df["absolute_short_pnl"] = np.where(df["absolute_measurable"], -stock_move, np.nan)
    df["relative_spy_short_pnl"] = np.where(df["relative_spy_measurable"], spy_move - stock_move, np.nan)
    df["relative_sector_short_pnl"] = np.where(df["relative_sector_measurable"], sector_move - stock_move, np.nan)

    df["currency_mismatch"] = df["bbg_ticker"].map(lambda t: default_vendor_symbol(t) is None)

    return df


def check_price_coverage(
    panel_df: pd.DataFrame, price_history_df: pd.DataFrame, roll_cap_days: int, lookback_days: int
) -> pd.DataFrame:
    """Build the gap report: one row per (row, horizon, leg) that is
    anchor_gap or horizon_gap. not_yet_matured and no_sector_benchmark are
    NOT gaps — they are excluded here on purpose (see the module docstring).

    Args:
        panel_df: resolve_price_panel's output.
        price_history_df: price_history, used to report each gap's
            last_available_date.
        roll_cap_days, lookback_days: same config values used to build
            panel_df — needed here only to state each gap's searched window
            (required_start/required_end), not to re-resolve anything.

    Returns:
        DataFrame: bbg_ticker, vendor_symbol, leg, ticker, wba_date,
        horizon_months, required_start, required_end, last_available_date, reason.
    """
    price_map = _to_price_map(price_history_df)
    gap_rows = []

    leg_specs = [
        ("stock", "stock_vendor_symbol", "stock_status", "stock_horizon_target"),
        ("spy", "spy_vendor_symbol", "spy_status", "spy_horizon_target"),
        ("sector", "sector_vendor_symbol", "sector_status", "sector_horizon_target"),
    ]

    for row in panel_df.itertuples(index=False):
        for leg, symbol_col, status_col, target_col in leg_specs:
            status = getattr(row, status_col)
            if status not in ("anchor_gap", "horizon_gap", "no_series"):
                continue
            if leg == "stock":
                bbg_ticker = row.bbg_ticker
            elif leg == "spy":
                bbg_ticker = SPY_BBG_TICKER
            else:
                bbg_ticker = row.sector_ticker
            vendor_symbol = getattr(row, symbol_col)
            wba_date = row.wba_date

            if status in ("anchor_gap", "no_series"):
                required_start = wba_date - timedelta(days=roll_cap_days)
                required_end = wba_date
                reason = "no_price_history_for_ticker" if status == "no_series" else "anchor_unresolved_beyond_roll_cap"
            else:
                horizon_target = getattr(row, target_col)
                required_start = horizon_target - timedelta(days=lookback_days)
                required_end = horizon_target
                reason = "horizon_unresolved_beyond_lookback_cap"

            ticker_prices = price_map.get(bbg_ticker, {})
            prior_dates = [d for d in ticker_prices if d <= required_end]
            last_available_date = max(prior_dates) if prior_dates else None

            gap_rows.append({
                "bbg_ticker": bbg_ticker,
                "vendor_symbol": vendor_symbol,
                "leg": leg,
                "ticker": row.ticker,
                "wba_date": wba_date,
                "horizon_months": row.horizon_months,
                "required_start": required_start,
                "required_end": required_end,
                "last_available_date": last_available_date,
                "reason": reason,
            })

    return pd.DataFrame(
        gap_rows,
        columns=["bbg_ticker", "vendor_symbol", "leg", "ticker", "wba_date", "horizon_months",
                 "required_start", "required_end", "last_available_date", "reason"],
    )


def write_gap_csv(gaps_df: pd.DataFrame, rundate: str,
                   out_dir: str = os.path.join("data", "historical", "prices")) -> str:
    """Write the gap report to data/historical/prices/GAPS_<rundate>.csv.
    Always writes (even an empty gaps_df, as a zero-row file with headers) so
    a run's absence of gaps is itself on record.

    Args:
        gaps_df: check_price_coverage's output.
        rundate: Date string for the filename, e.g. "2026-09-02".
        out_dir: Destination directory, created if absent.

    Returns:
        The path written.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"GAPS_{rundate}.csv")
    gaps_df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return path


@dataclass(frozen=True)
class AnchorReconciliationResult:
    """check_anchor_reconciliation's output.

    Attributes:
        n_attempted: 105 — Whiteboard rows with a stored wba_price. Fixed,
            independent of vendor coverage.
        n_evaluated: Of those, how many had a resolvable vendor anchor price.
        n_unevaluable: n_attempted - n_evaluated (a vendor coverage gap on a
            row that DOES have a stored price — reported, not silently
            absorbed into a shrunk "clean" base).
        violation_count: Rows where |vendor - stored| / stored > tolerance.
        max_abs_diff: Largest fractional diff observed among evaluated rows.
        tolerance: The tolerance this result was computed against.
        violation_tickers: Up to 20 offending tickers, worst first.
    """

    n_attempted: int
    n_evaluated: int
    n_unevaluable: int
    violation_count: int
    max_abs_diff: float
    tolerance: float
    violation_tickers: list = field(default_factory=list)


def check_anchor_reconciliation(
    whiteboard_df: pd.DataFrame, panel_df: pd.DataFrame, tolerance: float
) -> AnchorReconciliationResult:
    """Compare the vendor's anchor-date close against the stored wba_price,
    for the 105 rows that have one. A mismatch beyond tolerance means a
    recycled symbol, a wrong exchange, or a split-adjustment disagreement —
    flagged, never aborting (see the module docstring / PHASE4B_SCOPE.md
    section 6.2).

    Args:
        whiteboard_df: historical_whiteboard_shorts.
        panel_df: resolve_price_panel's output (only horizon_months==6 rows
            are used — the anchor price doesn't depend on horizon_months).
        tolerance: config prices.anchor_reconciliation_rel_tolerance.

    Returns:
        AnchorReconciliationResult.
    """
    stored = whiteboard_df[whiteboard_df["wba_price"].notna()][["ticker", "wba_date", "wba_price"]].copy()
    # panel_df's wba_date is a python date (built row-by-row in
    # resolve_price_panel); whiteboard_df's is a pandas Timestamp (read via
    # pd.read_sql(..., parse_dates=[...])) — normalize both to date objects
    # before merging, or the join silently matches nothing.
    stored["wba_date"] = pd.to_datetime(stored["wba_date"]).dt.date
    n_attempted = len(stored)

    six_month = panel_df[panel_df["horizon_months"] == 6][["ticker", "wba_date", "stock_anchor_price"]]
    merged = stored.merge(six_month, on=["ticker", "wba_date"], how="left")

    evaluated = merged[merged["stock_anchor_price"].notna()]
    n_evaluated = len(evaluated)
    n_unevaluable = n_attempted - n_evaluated

    if n_evaluated == 0:
        return AnchorReconciliationResult(
            n_attempted=n_attempted, n_evaluated=0, n_unevaluable=n_unevaluable,
            violation_count=0, max_abs_diff=float("nan"), tolerance=tolerance,
        )

    diff = (evaluated["stock_anchor_price"] - evaluated["wba_price"]).abs() / evaluated["wba_price"]
    max_abs_diff = float(diff.max())
    violation_mask = diff > (tolerance + 1e-9)
    violation_count = int(violation_mask.sum())

    violation_tickers = []
    if violation_count:
        ordered = diff[violation_mask].sort_values(ascending=False)
        violation_tickers = list(evaluated.loc[ordered.index, "ticker"])[:20]

    return AnchorReconciliationResult(
        n_attempted=n_attempted, n_evaluated=n_evaluated, n_unevaluable=n_unevaluable,
        violation_count=violation_count, max_abs_diff=max_abs_diff, tolerance=tolerance,
        violation_tickers=violation_tickers,
    )


@dataclass(frozen=True)
class EventReplicationResult:
    """check_event_window_replication's output, one instance per leg
    ("absolute" or "relative").

    Attributes:
        leg: "absolute" or "relative".
        n_attempted: Rows with wba_date + wbr_date + the relevant stored
            performance figure — 101 for absolute, 109 for relative. Does NOT
            require a stored wba_price/wbr_price (the 9 relative-only rows
            have neither and are still full population members — see the
            module docstring).
        n_evaluated: Of those, how many had a resolvable vendor price at both
            the anchor and the wbr close date (and, for relative, SPY too).
        n_unevaluable: n_attempted - n_evaluated.
        min_evaluated_required: The computed threshold
            max(min_evaluated_abs, min_evaluated_frac * n_attempted).
        evaluated_frac_actual: n_evaluated / n_attempted.
        corr: Pearson correlation between computed and stored performance.
        max_abs_diff: Largest |computed - stored| among evaluated rows.
        violation_count: Rows where |computed - stored| > abs_tolerance.
        violation_tickers: Up to 20 offending tickers, worst first.
        passed: True iff n_evaluated >= 2 AND n_evaluated meets
            min_evaluated_required AND corr >= min_corr.
        failure_reason: None if passed, else one of "insufficient_n",
            "insufficient_evaluated_population", "corr_below_floor" — a
            coverage collapse and a genuine sign/data problem are different
            failure modes and must not be reported under one boolean.
        vendor_breakdown: {source: {"n": int, "mean_abs_diff": float,
            "max_abs_diff": float}} — partitioned by the vendor that supplied
            each row's close-date price (falling back to the anchor-date
            source if the close date has none), so a Stooq-specific bias is
            visible even when pooled with a much larger yfinance-sourced
            population would average it away. Always includes every source
            seen among evaluated rows; a source with zero evaluated rows this
            run is reported as "n/a — no <source>-sourced series this run".
    """

    leg: str
    n_attempted: int
    n_evaluated: int
    n_unevaluable: int
    min_evaluated_required: float
    evaluated_frac_actual: float
    corr: float
    max_abs_diff: float
    violation_count: int
    violation_tickers: list
    passed: bool
    failure_reason: str
    vendor_breakdown: dict = field(default_factory=dict)


def _replication_population(whiteboard_df: pd.DataFrame, leg: str) -> pd.DataFrame:
    """The population for one replication leg. See EventReplicationResult's
    docstring: no stored-price requirement, only the stored performance
    figure being validated."""
    perf_col = "absolute_performance" if leg == "absolute" else "relative_spy_performance"
    required = ["wba_date", "wbr_date", perf_col]
    return whiteboard_df.dropna(subset=required)[["ticker", "bbg_ticker", "wba_date", "wbr_date", perf_col]].copy()


def check_event_window_replication(
    whiteboard_df: pd.DataFrame, price_history_df: pd.DataFrame, roll_cap_days: int,
    lookback_days: int, abs_tolerance: float, min_corr: float,
    min_evaluated_frac: float, min_evaluated_abs: float,
) -> dict:
    """The anti-tautology sign/data gate — REPLACES a naive
    corr(short_pnl, price_move) check (which is a mathematical identity,
    always exactly -1, and validates nothing — see PHASE4B build history).

    Recomputes the STORED EVENT WINDOW from vendor prices — anchor at
    wba_pricing_date, close at wbr_pricing_date, using the SAME roll rules as
    the main fixed-horizon measure (roll_cap_days / lookback_days reused
    directly, not separate constants, so this validates the shipped
    resolution procedure and not a parallel one) — and correlates the result
    against the workbook's own stored figures, over 101 (absolute) / 109
    (relative) real outcomes.

    Hard-fails (via run_whiteboard_horizons raising
    EventWindowReplicationError) if EITHER leg fails. A per-row violation
    beyond abs_tolerance is reported but does not by itself fail the gate —
    only the aggregate corr does, together with the minimum-evaluated-
    population floor below.

    Args:
        whiteboard_df: historical_whiteboard_shorts.
        price_history_df: price_history.
        roll_cap_days, lookback_days: same as the main measure's config
            values — reused deliberately, see above.
        abs_tolerance: config prices.event_window_abs_tolerance.
        min_corr: config prices.event_window_min_corr.
        min_evaluated_frac, min_evaluated_abs: config
            prices.event_window_min_evaluated_frac /
            event_window_min_evaluated_abs. HARD-FAILS if
            n_evaluated < max(min_evaluated_abs, min_evaluated_frac *
            n_attempted) — a handful of survivors correlating perfectly is
            not evidence the vendor pull mostly succeeded.

    Returns:
        {"absolute": EventReplicationResult, "relative": EventReplicationResult}
    """
    price_map = _to_price_map(price_history_df)
    results = {}

    for leg in ("absolute", "relative"):
        population = _replication_population(whiteboard_df, leg)
        n_attempted = len(population)
        perf_col = "absolute_performance" if leg == "absolute" else "relative_spy_performance"

        computed = []
        sources = []
        for row in population.itertuples(index=False):
            anchor_original = pd.Timestamp(row.wba_date).date()
            close_target = pd.Timestamp(row.wbr_date).date()

            stock_prices = price_map.get(row.bbg_ticker, {})
            _, anchor_price, anchor_source, _ = roll_backward(stock_prices, anchor_original, roll_cap_days)
            _, close_price, close_source, _ = roll_backward(stock_prices, close_target, lookback_days)

            if anchor_price is None or close_price is None:
                computed.append(np.nan)
                sources.append(None)
                continue

            stock_move = (close_price - anchor_price) / anchor_price

            if leg == "absolute":
                computed.append(-stock_move)
            else:
                spy_prices = price_map.get(SPY_BBG_TICKER, {})
                _, spy_anchor, _, _ = roll_backward(spy_prices, anchor_original, roll_cap_days)
                _, spy_close, _, _ = roll_backward(spy_prices, close_target, lookback_days)
                if spy_anchor is None or spy_close is None:
                    computed.append(np.nan)
                    sources.append(None)
                    continue
                spy_move = (spy_close - spy_anchor) / spy_anchor
                computed.append(spy_move - stock_move)

            sources.append(close_source if close_source is not None else anchor_source)

        population = population.copy()
        population["computed"] = computed
        population["vendor_source"] = sources
        population["stored"] = population[perf_col]

        evaluated = population.dropna(subset=["computed", "stored"])
        n_evaluated = len(evaluated)
        n_unevaluable = n_attempted - n_evaluated
        min_evaluated_required = max(min_evaluated_abs, min_evaluated_frac * n_attempted)
        evaluated_frac_actual = (n_evaluated / n_attempted) if n_attempted else 0.0

        vendor_breakdown = {}
        for source, group in evaluated.groupby("vendor_source"):
            source_diff = (group["computed"] - group["stored"]).abs()
            vendor_breakdown[source] = {
                "n": len(group),
                "mean_abs_diff": float(source_diff.mean()),
                "max_abs_diff": float(source_diff.max()),
            }
        for source in ("yfinance", "stooq", "bloomberg_manual"):
            if source not in vendor_breakdown:
                vendor_breakdown[source] = "n/a — no {}-sourced series this run".format(source)

        if n_evaluated < 2:
            results[leg] = EventReplicationResult(
                leg=leg, n_attempted=n_attempted, n_evaluated=n_evaluated,
                n_unevaluable=n_unevaluable, min_evaluated_required=min_evaluated_required,
                evaluated_frac_actual=evaluated_frac_actual, corr=float("nan"),
                max_abs_diff=float("nan"), violation_count=0, violation_tickers=[],
                passed=False, failure_reason="insufficient_n", vendor_breakdown=vendor_breakdown,
            )
            continue

        corr = float(np.corrcoef(evaluated["computed"], evaluated["stored"])[0, 1])
        diff = (evaluated["computed"] - evaluated["stored"]).abs()
        max_abs_diff = float(diff.max())
        violation_mask = diff > (abs_tolerance + 1e-9)
        violation_count = int(violation_mask.sum())
        violation_tickers = []
        if violation_count:
            ordered = diff[violation_mask].sort_values(ascending=False)
            violation_tickers = list(evaluated.loc[ordered.index, "ticker"])[:20]

        if n_evaluated < min_evaluated_required:
            failure_reason = "insufficient_evaluated_population"
            passed = False
        elif corr < min_corr:
            failure_reason = "corr_below_floor"
            passed = False
        else:
            failure_reason = None
            passed = True

        results[leg] = EventReplicationResult(
            leg=leg, n_attempted=n_attempted, n_evaluated=n_evaluated,
            n_unevaluable=n_unevaluable, min_evaluated_required=min_evaluated_required,
            evaluated_frac_actual=evaluated_frac_actual, corr=corr, max_abs_diff=max_abs_diff,
            violation_count=violation_count, violation_tickers=violation_tickers,
            passed=passed, failure_reason=failure_reason, vendor_breakdown=vendor_breakdown,
        )

    return results


def flag_spurious_stored_relative(whiteboard_df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Detect stored relative_spy_performance values that are a null-price
    formula artifact in the source workbook, not a real measurement.

    Discovered via check_event_window_replication's own violation report on
    its first real run (see PHASE4B build history): 9 of the 9 flagged
    relative-leg rows all had BOTH wba_price and wbr_price null, and in every
    one, stored relative_spy_performance equals the SPY move ALONE — i.e.
    bench_move - price_move with price_move silently treated as zero when a
    price is missing, rather than the cell being left blank. That is not this
    stock's relative performance; it's the benchmark's move with a zero
    substituted for the (unmeasurable) stock leg. Measured across the full
    109-row population, this null-as-zero pattern reproduces exactly wherever
    the null-price condition holds (29 of 152 whiteboard rows) and does NOT
    reproduce spuriously elsewhere — see the load-bearing test below.

    Where a row is flagged spurious, THE VENDOR-COMPUTED VALUE (this module's
    own relative_spy_short_pnl, from check_event_window_replication/
    compute_horizon_returns) is the correct one; the stored column is not a
    measurement for that row. This does not modify historical_whiteboard_shorts
    — 4a's tables remain a faithful import of the source file; detect and
    report, never silently repair (see CLAUDE.md Known Issues).

    Args:
        whiteboard_df: historical_whiteboard_shorts.
        tolerance: Absolute tolerance for "stored equals the SPY move alone" —
            reuse config prices.event_window_abs_tolerance, the same
            whole-percent-rounding arithmetic basis (measured max diff on the
            real null-price population: 0.00456, comfortably inside 0.01).

    Returns:
        DataFrame: ticker, wba_date, outcome, stored_relative_spy_performance,
        spy_move, diff — one row per flagged row. Empty if none flagged.
    """
    df = whiteboard_df.copy()
    spy_move = (df["benchmark_at_wbr"] - df["benchmark_at_wba"]) / df["benchmark_at_wba"]
    diff = (df["relative_spy_performance"] - spy_move).abs()
    has_null_price = df["wba_price"].isna() | df["wbr_price"].isna()

    # The null-price condition is LOAD-BEARING, not decorative: a row whose
    # stock genuinely moved in lockstep with SPY, with BOTH prices present
    # (one such row exists in the real data — a stock that simply went
    # nowhere over the window), can coincidentally match the SPY move too.
    # Without this condition, that row would be misflagged as spurious when
    # its stored value is a perfectly real (if unremarkable) measurement.
    mask = (
        df["relative_spy_performance"].notna() & spy_move.notna()
        & (diff <= tolerance) & has_null_price
    )

    flagged = df.loc[mask, ["ticker", "wba_date", "outcome"]].copy()
    flagged["stored_relative_spy_performance"] = df.loc[mask, "relative_spy_performance"]
    flagged["spy_move"] = spy_move[mask]
    flagged["diff"] = diff[mask]
    return flagged.reset_index(drop=True)


def summarize_by_arm_and_horizon(panel_and_returns_df: pd.DataFrame) -> pd.DataFrame:
    """n / median / hit_rate per outcome arm x horizon x measure, with the
    unmeasurable count (split gap vs not-yet-matured) printed alongside every
    arm. All arms reported, including the single null-outcome row — pooling
    or excluding any arm is a reporting-time choice made later, never baked
    in here (Driver ruling, PHASE4B_SCOPE.md section 3).

    Args:
        panel_and_returns_df: compute_horizon_returns' output.

    Returns:
        DataFrame: outcome/horizon_months/measure/n_ceiling/n_measurable/
        n_gap/n_not_yet_matured/median/hit_rate. n_ceiling is the elapsed-time
        ceiling population for that (outcome, horizon) — matches
        PHASE4B_SCOPE.md's 127/104 totals when summed across horizon_months.
    """
    df = panel_and_returns_df.copy()
    df["outcome_label"] = df["outcome"].fillna("null")

    rows = []
    measures = [
        ("absolute", "absolute_short_pnl", "absolute_measurable"),
        ("relative_spy", "relative_spy_short_pnl", "relative_spy_measurable"),
    ]
    for (outcome_label, horizon_months), group in df.groupby(["outcome_label", "horizon_months"]):
        ceiling_group = group[group["within_elapsed_ceiling"]]
        not_yet_matured = int((~group["within_elapsed_ceiling"]).sum())
        n_ceiling = len(ceiling_group)

        for measure_name, value_col, measurable_col in measures:
            measurable = ceiling_group[ceiling_group[measurable_col]]
            n_measurable = len(measurable)
            n_gap = n_ceiling - n_measurable
            values = measurable[value_col].dropna()
            rows.append({
                "outcome": outcome_label,
                "horizon_months": horizon_months,
                "measure": measure_name,
                "n_ceiling": n_ceiling,
                "n_measurable": n_measurable,
                "n_gap": n_gap,
                "n_not_yet_matured": not_yet_matured,
                "median": values.median() if len(values) else float("nan"),
                "hit_rate": (values > 0).mean() if len(values) else float("nan"),
            })

    return pd.DataFrame(
        rows,
        columns=["outcome", "horizon_months", "measure", "n_ceiling", "n_measurable",
                 "n_gap", "n_not_yet_matured", "median", "hit_rate"],
    )


def _json_safe(obj):
    """Same purpose as historical_ingest.py's _json_safe, reimplemented
    locally rather than imported (private to that module)."""
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, float):
        return None if (obj != obj) else obj  # NaN check without importing math here
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if value != value else value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _event_result_to_dict(result: EventReplicationResult) -> dict:
    return {
        "leg": result.leg, "n_attempted": result.n_attempted, "n_evaluated": result.n_evaluated,
        "n_unevaluable": result.n_unevaluable, "min_evaluated_required": result.min_evaluated_required,
        "evaluated_frac_actual": result.evaluated_frac_actual, "corr": result.corr,
        "max_abs_diff": result.max_abs_diff, "violation_count": result.violation_count,
        "violation_tickers": result.violation_tickers, "passed": result.passed,
        "failure_reason": result.failure_reason, "vendor_breakdown": result.vendor_breakdown,
    }


def run_whiteboard_horizons(engine, config: dict, run_date: date = None) -> dict:
    """Orchestrate the fixed-horizon measurement, in the order that keeps
    diagnostics available even on a hard-fail:

      1. resolve_price_panel
      2. check_price_coverage -> gaps_df
      3. write_gap_csv (FILE write, unconditional — survives a later raise)
      4. check_anchor_reconciliation
      5. build the full diagnostic report (returned to the caller regardless
         of outcome, so a failing run's printout is not lost)
      6. check_event_window_replication — RAISES EventWindowReplicationError
         here on failure, before any DATABASE write
      7. (only if 6 passed) write whiteboard_horizon_returns (if_exists="replace")
      8. (only if 6 passed) append one whiteboard_horizon_runs row

    A failed run at step 6 leaves the gap CSV and the returned diagnostic
    report intact and writes nothing to data/screener.db — the gate protects
    the database, not the diagnostics needed to fix what caused it to fire.

    Args:
        engine: SQLAlchemy engine.
        config: Full parsed config.yaml dict.
        run_date: "today" for not_yet_matured / gap-CSV filename purposes.
            Defaults to datetime.now(timezone.utc).date().

    Returns:
        dict with keys: panel (returns-augmented DataFrame), gaps_df,
        gap_csv_path, anchor_reconciliation (AnchorReconciliationResult),
        event_window (dict of two EventReplicationResult), summary
        (summarize_by_arm_and_horizon's output), rows_written (0 if the gate
        failed and nothing was written).

    Raises:
        EventWindowReplicationError: If either replication leg fails. Nothing
        is written to the database; the gap CSV and report are still built.
    """
    if run_date is None:
        run_date = datetime.now(timezone.utc).date()
    prices_cfg = config["prices"]

    whiteboard_df = pd.read_sql("select * from historical_whiteboard_shorts", engine, parse_dates=["wba_date", "wba_pricing_date", "wbr_date", "wbr_pricing_date"])
    price_history_df = pd.read_sql("select bbg_ticker, date, close, source, vendor_symbol, ingested_at from price_history", engine)
    symbol_map = dict(zip(price_history_df["bbg_ticker"], price_history_df["vendor_symbol"])) if not price_history_df.empty else {}

    roll_cap_days = prices_cfg["anchor_roll_cap_days"]
    lookback_days = prices_cfg["horizon_lookback_days"]

    panel = resolve_price_panel(
        whiteboard_df, price_history_df, symbol_map, roll_cap_days, lookback_days, run_date,
    )
    panel = compute_horizon_returns(panel)

    gaps_df = check_price_coverage(panel, price_history_df, roll_cap_days, lookback_days)
    rundate_str = run_date.strftime("%Y-%m-%d")
    gap_csv_path = write_gap_csv(gaps_df, rundate_str)

    anchor_reconciliation = check_anchor_reconciliation(
        whiteboard_df, panel, prices_cfg["anchor_reconciliation_rel_tolerance"]
    )
    summary = summarize_by_arm_and_horizon(panel)
    spurious_stored_relative = flag_spurious_stored_relative(
        whiteboard_df, prices_cfg["event_window_abs_tolerance"]
    )

    price_history_max_ingested_at = price_history_df["ingested_at"].max() if not price_history_df.empty else None
    price_history_row_count = len(price_history_df)
    vendor_counts = (
        price_history_df.groupby("source").size().to_dict() if not price_history_df.empty else {}
    )

    result = {
        "panel": panel, "gaps_df": gaps_df, "gap_csv_path": gap_csv_path,
        "anchor_reconciliation": anchor_reconciliation, "summary": summary,
        "spurious_stored_relative": spurious_stored_relative,
        "rows_written": 0,
    }

    event_window = check_event_window_replication(
        whiteboard_df, price_history_df, prices_cfg["anchor_roll_cap_days"],
        prices_cfg["horizon_lookback_days"], prices_cfg["event_window_abs_tolerance"],
        prices_cfg["event_window_min_corr"], prices_cfg["event_window_min_evaluated_frac"],
        prices_cfg["event_window_min_evaluated_abs"],
    )
    result["event_window"] = event_window

    failing = {leg: r for leg, r in event_window.items() if not r.passed}
    if failing:
        detail = "; ".join(
            f"{leg}: n_evaluated={r.n_evaluated}/{r.n_attempted} "
            f"(required>={r.min_evaluated_required:.1f}) corr={r.corr} "
            f"reason={r.failure_reason}" for leg, r in failing.items()
        )
        raise EventWindowReplicationError(
            f"Event-window replication gate failed — write aborted before any "
            f"database write. Gap CSV ({gap_csv_path}) and diagnostics were "
            f"still produced. {detail}"
        )

    computed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    panel.to_sql("whiteboard_horizon_returns", engine, if_exists="replace", index=False)
    result["rows_written"] = len(panel)

    run_row = {
        "computed_at_utc": computed_at,
        "price_history_max_ingested_at": str(price_history_max_ingested_at) if price_history_max_ingested_at is not None else None,
        "price_history_row_count": price_history_row_count,
        "vendor_counts_json": json.dumps(_json_safe(vendor_counts)),
        "rows_written": len(panel),
        "gap_csv_path": gap_csv_path,
        "coverage_gap_count": len(gaps_df),
        "anchor_reconciliation_json": json.dumps(_json_safe(vars(anchor_reconciliation))),
        "event_window_absolute_json": json.dumps(_json_safe(_event_result_to_dict(event_window["absolute"]))),
        "event_window_relative_json": json.dumps(_json_safe(_event_result_to_dict(event_window["relative"]))),
        "spurious_stored_relative_json": json.dumps(_json_safe({
            "n_flagged": len(spurious_stored_relative),
            "by_outcome": spurious_stored_relative["outcome"].fillna("null").value_counts().to_dict(),
            "tickers": spurious_stored_relative["ticker"].tolist(),
        })),
        "roll_lookback_distribution_json": json.dumps(_json_safe({
            col: panel[col].value_counts(dropna=True).to_dict()
            for col in ("stock_anchor_roll_days", "stock_horizon_lookback_days_used",
                        "spy_anchor_roll_days", "spy_horizon_lookback_days_used",
                        "sector_anchor_roll_days", "sector_horizon_lookback_days_used")
        })),
    }
    _ensure_columns(engine, "whiteboard_horizon_runs", run_row)
    append_rows(engine, pd.DataFrame([run_row]), "whiteboard_horizon_runs")

    return result


def _ensure_columns(engine, table: str, row: dict) -> None:
    """Additively migrate `table` to include any column in `row` it's
    missing, as TEXT, before an append_rows() insert.

    whiteboard_horizon_runs is append-only (Architecture Rule 10): existing
    rows are never touched. But its own schema can still grow across builds
    (this function exists because it did — spurious_stored_relative_json was
    added after the table already existed from an earlier run this phase).
    ALTER TABLE ... ADD COLUMN is purely additive — it changes no existing
    row and satisfies append-only exactly as well as a table created fresh
    with the new column would have. No-op if the table doesn't exist yet
    (append_rows/to_sql creates it fresh with every current column already
    present).
    """
    identifier = re.compile(r"^[a-z][a-z0-9_]*$")
    if not inspect(engine).has_table(table):
        return
    existing = {col["name"] for col in inspect(engine).get_columns(table)}
    missing = [col for col in row if col not in existing]
    if not missing:
        return
    with engine.begin() as conn:
        for col in missing:
            if not identifier.match(col):
                raise ValueError(f"Unsafe column name for SQL identifier use: {col!r}")
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} TEXT"))
