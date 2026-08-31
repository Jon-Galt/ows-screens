"""
Unit tests for src/ingest.py's clean_dataframe().

Phase 3c.1 added ticker extraction (the full Bloomberg identifier ->
bare ticker) to this function, matching what rsi_ingest.py already did
for Rising Short Interest. Before this, short_screen's tickers were a
direct rename with no split, which silently broke any ticker-based join
against the other five screens (all of which already store bare tickers).
"""

import numpy as np
import pandas as pd

from src.ingest import clean_dataframe


class TestCleanDataframeTickerExtraction:
    def test_full_identifier_extracted_to_bare_ticker(self):
        df = pd.DataFrame({"Ticker": ["AAPL US Equity", "RUSHA US Equity"]})
        result = clean_dataframe(df, column_map={"Ticker": "ticker"}, string_columns={"ticker"})
        assert list(result["ticker"]) == ["AAPL", "RUSHA"]

    def test_no_ticker_column_is_a_no_op(self):
        """Some callers of clean_dataframe (e.g. a screen with no ticker
        column) should not error just because there's nothing to extract."""
        df = pd.DataFrame({"Name": ["Some Co"]})
        result = clean_dataframe(df, column_map={"Name": "name"}, string_columns={"name"})
        assert list(result["name"]) == ["Some Co"]

    def test_missing_ticker_value_passes_through(self):
        df = pd.DataFrame({"Ticker": ["AAPL US Equity", np.nan]})
        result = clean_dataframe(df, column_map={"Ticker": "ticker"}, string_columns={"ticker"})
        assert result["ticker"][0] == "AAPL"
        assert pd.isna(result["ticker"][1])
