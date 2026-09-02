"""
Unit tests for src/price_history.py (Phase 4b).

Small synthetic DataFrames throughout — no real network calls (yfinance/
requests are mocked at their call boundary), no dependency on
data/historical/ (gitignored). The real vendor pull is verified once, live,
as part of the mandatory end-to-end run, not asserted here.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.price_history import (
    SymbolMappingError,
    UniverseResult,
    assemble_universe,
    default_vendor_symbol,
    fetch_price_series,
    ingest_manual_fill,
    resolve_vendor_symbol,
    run_price_pull,
    upsert_price_history,
)


@pytest.fixture
def engine():
    return create_engine("sqlite:///:memory:")


# ---------------------------------------------------------------------------
# resolve_vendor_symbol / default_vendor_symbol
# ---------------------------------------------------------------------------

def test_default_vendor_symbol_strips_us_equity_suffix():
    assert default_vendor_symbol("AAPL US Equity") == "AAPL"


def test_default_vendor_symbol_none_for_non_matching_shape():
    assert default_vendor_symbol("EDEN FP Equity") is None
    assert default_vendor_symbol("PRY.IM") is None


def test_resolve_vendor_symbol_default_rule():
    assert resolve_vendor_symbol("AAPL US Equity", {}) == "AAPL"


def test_resolve_vendor_symbol_override_wins_over_default():
    overrides = {"AAPL US Equity": "AAPL-OVERRIDE"}
    assert resolve_vendor_symbol("AAPL US Equity", overrides) == "AAPL-OVERRIDE"


def test_resolve_vendor_symbol_malformed_ticker_without_override_raises():
    with pytest.raises(SymbolMappingError):
        resolve_vendor_symbol("PRY.IM", {})


def test_resolve_vendor_symbol_malformed_ticker_with_override_resolves():
    assert resolve_vendor_symbol("PRY.IM", {"PRY.IM": "PRY.MI"}) == "PRY.MI"


# ---------------------------------------------------------------------------
# assemble_universe
# ---------------------------------------------------------------------------

def test_assemble_universe_dedupes_and_reports_overlap_without_raising():
    whiteboard_df = pd.DataFrame({
        "bbg_ticker": ["AAA US Equity", "BBB US Equity", "XLK US Equity"],  # XLK overlaps with sector below
        "sector_benchmark_ticker": ["XLK US Equity", "XLE US Equity", "XLK US Equity"],
    })
    result = assemble_universe(whiteboard_df, overrides={})
    assert isinstance(result, UniverseResult)
    # 2 stocks + XLK(stock) + XLK(sector, dup) + XLE(sector) + SPY = dedup to 5
    assert len(result.universe) == 5
    assert "XLK US Equity" in result.overlaps
    assert result.universe["bbg_ticker"].is_unique


def test_assemble_universe_no_overlap_case_matches_verified_live_shape():
    whiteboard_df = pd.DataFrame({
        "bbg_ticker": [f"T{i} US Equity" for i in range(5)],
        "sector_benchmark_ticker": ["XLK US Equity"] * 5,
    })
    result = assemble_universe(whiteboard_df, overrides={})
    assert len(result.universe) == 5 + 1 + 1  # 5 stocks + 1 sector + SPY
    assert result.overlaps == []


def test_assemble_universe_raises_on_unmappable_ticker():
    whiteboard_df = pd.DataFrame({
        "bbg_ticker": ["PRY.IM"],
        "sector_benchmark_ticker": ["XLK US Equity"],
    })
    with pytest.raises(SymbolMappingError):
        assemble_universe(whiteboard_df, overrides={})


# ---------------------------------------------------------------------------
# fetch_price_series
# ---------------------------------------------------------------------------

def _yf_frame(dates, closes):
    return pd.DataFrame({"Close": closes, "Adj Close": [c * 0.9 for c in closes]}, index=pd.to_datetime(dates))


def test_fetch_price_series_calls_yfinance_with_auto_adjust_false():
    frame = _yf_frame(["2024-01-02", "2024-01-03"], [100.0, 101.0])
    with patch("yfinance.download", return_value=frame) as mock_download:
        result = fetch_price_series("AAPL", "2024-01-01", "2024-01-05")
    _, kwargs = mock_download.call_args
    assert kwargs["auto_adjust"] is False
    assert result.source == "yfinance"
    assert list(result.prices["close"]) == [100.0, 101.0]  # the unadjusted Close, not Adj Close


def test_fetch_price_series_falls_back_to_stooq_when_yfinance_empty():
    empty = pd.DataFrame(columns=["Close"])
    stooq_csv = "Date,Open,High,Low,Close,Volume\n2024-01-02,10,11,9,10.5,1000\n"
    mock_response = MagicMock(status_code=200, text=stooq_csv)
    with patch("yfinance.download", return_value=empty), \
         patch("requests.get", return_value=mock_response):
        result = fetch_price_series("XYZ", "2024-01-01", "2024-01-05")
    assert result.source == "stooq"
    assert list(result.prices["close"]) == [10.5]


def test_fetch_price_series_both_vendors_fail_returns_empty_not_raise():
    empty = pd.DataFrame(columns=["Close"])
    mock_response = MagicMock(status_code=404, text="")
    with patch("yfinance.download", return_value=empty), \
         patch("requests.get", return_value=mock_response):
        result = fetch_price_series("NOPE", "2024-01-01", "2024-01-05")
    assert result.source is None
    assert result.prices.empty


def test_fetch_price_series_yfinance_exception_falls_through_to_stooq():
    stooq_csv = "Date,Open,High,Low,Close,Volume\n2024-01-02,10,11,9,10.5,1000\n"
    mock_response = MagicMock(status_code=200, text=stooq_csv)
    with patch("yfinance.download", side_effect=RuntimeError("scraper broke")), \
         patch("requests.get", return_value=mock_response):
        result = fetch_price_series("XYZ", "2024-01-01", "2024-01-05")
    assert result.source == "stooq"


def test_fetch_price_series_stooq_bot_challenge_page_treated_as_no_data():
    """Regression test for the live finding: Stooq's endpoint currently
    returns an HTML bot-challenge page, not CSV. This must degrade to "no
    data", never raise and never be parsed as if it were prices."""
    empty = pd.DataFrame(columns=["Close"])
    challenge_html = '<!DOCTYPE html><html><head></head><body><noscript>JS required</noscript></body></html>'
    mock_response = MagicMock(status_code=200, text=challenge_html)
    with patch("yfinance.download", return_value=empty), \
         patch("requests.get", return_value=mock_response):
        result = fetch_price_series("XYZ", "2024-01-01", "2024-01-05")
    assert result.source is None
    assert result.prices.empty


# ---------------------------------------------------------------------------
# upsert_price_history — DDL, precedence, and the manual-fill round trip
# ---------------------------------------------------------------------------

def _row(bbg_ticker="AAA US Equity", d="2024-01-02", close=100.0, source="yfinance"):
    return pd.DataFrame([{
        "bbg_ticker": bbg_ticker, "date": d, "close": close, "source": source,
        "vendor_symbol": "AAA",
    }])


def test_upsert_price_history_fresh_insert(engine):
    upsert_price_history(engine, _row(close=100.0, source="yfinance"))
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT close, source FROM price_history")).fetchall()
    assert rows == [(100.0, "yfinance")]


def test_upsert_price_history_api_over_api_updates(engine):
    upsert_price_history(engine, _row(close=100.0, source="yfinance"))
    upsert_price_history(engine, _row(close=105.0, source="yfinance"))
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT close FROM price_history")).fetchall()
    assert rows == [(105.0,)]


def test_upsert_price_history_null_source_rejected_by_schema(engine):
    bad = _row()
    bad["source"] = None
    with pytest.raises(Exception):
        upsert_price_history(engine, bad)


def test_upsert_price_history_invalid_source_rejected_by_check_constraint(engine):
    bad = _row(source="totally_made_up_vendor")
    with pytest.raises(Exception):
        upsert_price_history(engine, bad)


def test_manual_fill_survives_api_repull_round_trip(engine):
    """Acceptance criterion 9: ingest a bloomberg_manual row, re-run the API
    pull for the same key with a different close, and prove the manual row
    is UNCHANGED. Then prove a second manual row for the same key DOES
    apply — the precedence rule holds in both directions, not just the
    protected one."""
    upsert_price_history(engine, _row(close=42.00, source="bloomberg_manual"))

    # A conflicting API-sourced row for the same (bbg_ticker, date) must NOT overwrite it.
    upsert_price_history(engine, _row(close=999.0, source="yfinance"))
    with engine.connect() as conn:
        row = conn.execute(text("SELECT close, source FROM price_history")).fetchone()
    assert row == (42.00, "bloomberg_manual")

    # A second, correcting manual row for the same key MUST apply.
    upsert_price_history(engine, _row(close=43.50, source="bloomberg_manual"))
    with engine.connect() as conn:
        row = conn.execute(text("SELECT close, source FROM price_history")).fetchone()
    assert row == (43.50, "bloomberg_manual")


def test_ingest_manual_fill_reads_one_csv_and_tags_bloomberg_manual(engine, tmp_path):
    (tmp_path / "manual_fill.csv").write_text("bbg_ticker,date,close\nAAA US Equity,2024-01-02,50.0\n")
    n = ingest_manual_fill(engine, upload_dir=str(tmp_path))
    assert n == 1
    with engine.connect() as conn:
        row = conn.execute(text("SELECT close, source, vendor_symbol FROM price_history")).fetchone()
    assert row == (50.0, "bloomberg_manual", "AAA US Equity")


# ---------------------------------------------------------------------------
# run_price_pull — vendor_counts_json, provenance
# ---------------------------------------------------------------------------

def test_run_price_pull_records_per_vendor_counts_and_one_provenance_row(engine):
    whiteboard = pd.DataFrame({
        "bbg_ticker": ["AAA US Equity", "BBB US Equity"],
        "sector_benchmark_ticker": ["XLK US Equity", "XLK US Equity"],
    })
    whiteboard.to_sql("historical_whiteboard_shorts", engine, index=False)

    yf_frame = _yf_frame(["2024-01-02"], [10.0])
    stooq_csv = "Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,20.0,1\n"

    def fake_download(symbol, **kwargs):
        if symbol == "BBB":
            return pd.DataFrame(columns=["Close"])  # forces Stooq fallback
        return yf_frame

    with patch("yfinance.download", side_effect=fake_download), \
         patch("requests.get", return_value=MagicMock(status_code=200, text=stooq_csv)):
        config = {"prices": {"symbol_overrides": {}}}
        summary = run_price_pull(engine, config, "2023-08-06", "2026-09-02")

    assert summary["universe_size"] == 4  # AAA, BBB, XLK, SPY
    assert summary["vendor_counts"]["yfinance"]["series"] >= 1
    assert summary["vendor_counts"]["stooq"]["series"] >= 1

    with engine.connect() as conn:
        run_count = conn.execute(text("SELECT COUNT(*) FROM price_history_runs")).scalar()
    assert run_count == 1
    with engine.connect() as conn:
        row = conn.execute(text("SELECT vendor_counts_json FROM price_history_runs")).fetchone()
    vendor_counts = json.loads(row[0])
    assert vendor_counts["yfinance"]["series"] >= 1
    assert vendor_counts["stooq"]["series"] >= 1
