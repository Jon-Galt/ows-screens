"""
Unit tests for the curated-screen loader in src/curated_ingest.py.

Coverage: quote-stripping, each unit conversion, scores parsing (including
malformed input the real data doesn't have), the upload-folder guard that
makes a misfiled or duplicate export loud instead of silent, the summary
logging that makes a misfile visually obvious, and one end-to-end curated
ingest against a small fixture.
"""

import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from src.curated_ingest import (
    CuratedUploadError,
    _find_single_upload_file,
    _log_curated_summary,
    clean_curated_dataframe,
    ingest_curated,
    parse_scores,
    strip_quoted_numeric,
)
from src.db import table_name


# ---------------------------------------------------------------------------
# strip_quoted_numeric
# ---------------------------------------------------------------------------

class TestStripQuotedNumeric:
    def test_quote_wrapped_value(self):
        result = strip_quoted_numeric(pd.Series(['"76122.023693"']))
        assert result[0] == pytest.approx(76122.023693)

    def test_already_clean_value(self):
        """A value with no embedded quotes still parses correctly."""
        result = strip_quoted_numeric(pd.Series(["76122.023693"]))
        assert result[0] == pytest.approx(76122.023693)

    def test_garbage_value(self):
        result = strip_quoted_numeric(pd.Series(['"not a number"']))
        assert np.isnan(result[0])

    def test_nan_input(self):
        result = strip_quoted_numeric(pd.Series([np.nan]))
        assert np.isnan(result[0])


# ---------------------------------------------------------------------------
# parse_scores
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_happy_path(self):
        result = parse_scores("Accounting And Disclosure: 18 | Fraud: 28 | Insider: 43")
        assert result == (18.0, 28.0, 43.0)

    def test_reordered_and_whitespace_varied_keys(self):
        """Matches by label, not position — robust to reordering."""
        result = parse_scores("insider:  40   |Fraud:35|  ACCOUNTING AND DISCLOSURE :59")
        assert result == (59.0, 35.0, 40.0)

    def test_missing_key(self):
        result = parse_scores("Fraud: 28 | Insider: 43")
        accounting, fraud, insider = result
        assert np.isnan(accounting)
        assert fraud == 28.0
        assert insider == 43.0

    def test_non_numeric_value(self):
        result = parse_scores("Accounting And Disclosure: high | Fraud: 28 | Insider: 43")
        accounting, fraud, insider = result
        assert np.isnan(accounting)
        assert fraud == 28.0

    def test_garbage_string(self):
        result = parse_scores("not a scores string at all")
        assert all(np.isnan(v) for v in result)

    def test_empty_string(self):
        result = parse_scores("")
        assert all(np.isnan(v) for v in result)

    def test_none_input(self):
        result = parse_scores(None)
        assert all(np.isnan(v) for v in result)

    def test_nan_input(self):
        result = parse_scores(np.nan)
        assert all(np.isnan(v) for v in result)


# ---------------------------------------------------------------------------
# clean_curated_dataframe
# ---------------------------------------------------------------------------

class TestCleanCuratedDataframe:
    @pytest.fixture
    def raw_df(self):
        """Two rows shaped like the real Canary export: quote-wrapped
        numerics, a plain percentile, a well-formed scores string."""
        return pd.DataFrame({
            "daily_traded_value": ['"431380930.272727"', '"1000000"'],
            "exchange_symbol": ["NYSE", "NasdaqGS"],
            "locations": ["US", "US"],
            "market_cap": ['"76122.023693"', '"1062488.0"'],
            "name": ["Norfolk Southern Corporation", "Micron Technology, Inc."],
            "sector": ["Industrials", "Information Technology"],
            "stock_performance": ['"21.08"', '"667.81"'],
            "ticker_symbol": ["NSC", "MU"],
            "rationale": ["Some rationale.", "Another rationale."],
            "scores": [
                "Accounting And Disclosure: 18 | Fraud: 28 | Insider: 43",
                "Accounting And Disclosure: 64 | Fraud: 55 | Insider: 71",
            ],
            "valuation_ev_revenue_ntm_percentile": ["86.8", "72.4"],
        })

    def test_renames_ticker_symbol(self, raw_df):
        result = clean_curated_dataframe(raw_df)
        assert "ticker" in result.columns
        assert "ticker_symbol" not in result.columns
        assert list(result["ticker"]) == ["NSC", "MU"]

    def test_market_cap_unchanged(self, raw_df):
        """market_cap: already $M, quote-stripped but not unit-converted."""
        result = clean_curated_dataframe(raw_df)
        assert result["market_cap"][0] == pytest.approx(76122.023693)
        # Micron spot-check from the acceptance criteria.
        assert result["market_cap"][1] == pytest.approx(1062488.0)

    def test_daily_traded_value_divided_by_1e6(self, raw_df):
        result = clean_curated_dataframe(raw_df)
        assert result["daily_traded_value"][0] == pytest.approx(431.380930272727)

    def test_stock_performance_divided_by_100(self, raw_df):
        """Micron spot-check: raw "667.81" -> stored 6.6781."""
        result = clean_curated_dataframe(raw_df)
        assert result["stock_performance"][1] == pytest.approx(6.6781)

    def test_valuation_percentile_divided_by_100(self, raw_df):
        result = clean_curated_dataframe(raw_df)
        assert result["valuation_ev_revenue_ntm_percentile"][0] == pytest.approx(0.868)

    def test_score_columns_parsed(self, raw_df):
        result = clean_curated_dataframe(raw_df)
        assert result["score_accounting_and_disclosure"][0] == 18.0
        assert result["score_fraud"][0] == 28.0
        assert result["score_insider"][0] == 43.0
        # Raw string retained for provenance.
        assert result["scores"][0] == "Accounting And Disclosure: 18 | Fraud: 28 | Insider: 43"

    def test_no_nans_in_the_three_previously_quote_wrapped_columns(self, raw_df):
        """A missed quote-strip is exactly what produces NaN here."""
        result = clean_curated_dataframe(raw_df)
        for col in ("market_cap", "daily_traded_value", "stock_performance"):
            assert result[col].isna().sum() == 0


# ---------------------------------------------------------------------------
# _find_single_upload_file
# ---------------------------------------------------------------------------

class TestFindSingleUploadFile:
    def test_no_files(self, tmp_path):
        with pytest.raises(CuratedUploadError, match="No export file found"):
            _find_single_upload_file(str(tmp_path))

    def test_multiple_files_named_in_error(self, tmp_path):
        (tmp_path / "a.csv").write_text("x")
        (tmp_path / "b.csv").write_text("x")
        with pytest.raises(CuratedUploadError) as exc_info:
            _find_single_upload_file(str(tmp_path))
        assert "a.csv" in str(exc_info.value)
        assert "b.csv" in str(exc_info.value)

    def test_stray_xlsx_alongside_csv_is_caught_not_ignored(self, tmp_path):
        """A leftover .xlsx sitting in the folder must not be silently
        skipped — it has to surface as part of the 'found N files' error."""
        (tmp_path / "real_export.csv").write_text("x")
        (tmp_path / "leftover.xlsx").write_text("x")
        with pytest.raises(CuratedUploadError) as exc_info:
            _find_single_upload_file(str(tmp_path))
        assert "real_export.csv" in str(exc_info.value)
        assert "leftover.xlsx" in str(exc_info.value)

    def test_single_xlsx_file_rejected_clearly(self, tmp_path):
        """Exactly one file, but not a .csv — a clear error, not a silent
        dict from pd.read_excel(sheet_name=None)."""
        (tmp_path / "export.xlsx").write_text("x")
        with pytest.raises(CuratedUploadError, match="only supports .csv"):
            _find_single_upload_file(str(tmp_path))

    def test_single_csv_file_returns_its_path(self, tmp_path):
        target = tmp_path / "export.csv"
        target.write_text("x")
        result = _find_single_upload_file(str(tmp_path))
        assert result == str(target)


# ---------------------------------------------------------------------------
# _log_curated_summary
# ---------------------------------------------------------------------------

class TestLogCuratedSummary:
    def test_logs_row_count_ticker_count_and_sector_distribution(self, caplog):
        df = pd.DataFrame({
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Industrials", "Industrials", "Health Care"],
        })
        with caplog.at_level(logging.INFO):
            _log_curated_summary("structural", df)

        combined = "\n".join(r.message for r in caplog.records)
        assert "3 rows" in combined
        assert "3 unique tickers" in combined
        assert "'Industrials': 2" in combined
        assert "AAA" in combined and "BBB" in combined and "CCC" in combined


# ---------------------------------------------------------------------------
# ingest_curated — dispatch guard and end-to-end fixture
# ---------------------------------------------------------------------------

def _write_fake_curated_config(config_path, screen_id) -> None:
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "screens": {
                    screen_id: {
                        "display_name": screen_id,
                        "type": "curated",
                        "universe": {"name": screen_id, "as_of": "2026-08"},
                    }
                }
            },
            f,
        )


def _write_curated_fixture_csv(path, tickers) -> None:
    rows = []
    for i, ticker in enumerate(tickers):
        rows.append({
            "daily_traded_value": f'"{1_000_000 + i}"',
            "exchange_symbol": "NYSE",
            "locations": "US",
            "market_cap": f'"{5000.0 + i}"',
            "name": f"Company {ticker}",
            "sector": "Industrials",
            "stock_performance": f'"{10.0 + i}"',
            "ticker_symbol": ticker,
            "rationale": f"Rationale for {ticker}.",
            "scores": f"Accounting And Disclosure: {10 + i} | Fraud: {20 + i} | Insider: {30 + i}",
            "valuation_ev_revenue_ntm_percentile": str(60.0 + i),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


class TestIngestCuratedEndToEnd:
    def test_single_file_happy_path(self, tmp_path):
        screen_id = "fake_curated_screen"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_curated_config(config_path, screen_id)

        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        _write_curated_fixture_csv(upload_dir / "export.csv", ["AAA", "BBB"])

        db_path = str(tmp_path / "test.db")
        ingest_curated(screen_id=screen_id, upload_dir=str(upload_dir), db_path=db_path,
                        config_path=config_path)

        engine_url = f"sqlite:///{db_path}"
        from sqlalchemy import create_engine
        engine = create_engine(engine_url)

        curated = pd.read_sql_table(table_name("curated_data", screen_id), engine)
        assert len(curated) == 2
        assert set(curated["ticker"]) == {"AAA", "BBB"}
        assert curated["market_cap"].isna().sum() == 0

        membership = pd.read_sql_table("screen_membership", engine)
        screen_rows = membership[membership["screen_id"] == screen_id]
        assert set(screen_rows["ticker"]) == {"AAA", "BBB"}

    def test_multi_file_folder_raises_before_touching_database(self, tmp_path):
        screen_id = "fake_curated_screen"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_curated_config(config_path, screen_id)

        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        _write_curated_fixture_csv(upload_dir / "export_1.csv", ["AAA"])
        _write_curated_fixture_csv(upload_dir / "export_2.csv", ["BBB"])

        db_path = str(tmp_path / "test.db")
        with pytest.raises(CuratedUploadError):
            ingest_curated(screen_id=screen_id, upload_dir=str(upload_dir), db_path=db_path,
                            config_path=config_path)

        import os
        assert not os.path.exists(db_path)
