"""
Unit tests for the Rising Short Interest loader in src/rsi_ingest.py.

Coverage: count-row parsing (match, mismatch, missing/unparseable), the
footer-trim's defensive "does the discarded tail look like real data"
check, each unit conversion, both SI-change ratios with a synthetic
zero-denominator case (CLAUDE.md bug pattern 5), and one end-to-end
ingest+transform against a small fixture shaped like the real export.
extract_ticker itself is tested in test_loaders.py — it's a shared helper
now, not RSI-specific.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import CONFIG_PATH, load_config
from src.db import table_name
from src.loaders import extract_ticker
from src.rsi_ingest import (
    RSI_COLUMN_MAP,
    _extract_expected_count,
    clean_rsi_dataframe,
    ingest_rsi,
    trim_rsi_export,
)
from src.transform import (
    calc_rsi_debt_ebitda,
    calc_rsi_ev_sales,
    calc_rsi_market_cap,
    calc_rsi_si_change_3m,
    calc_rsi_si_change_6m,
    run_rsi_transforms,
)


# ---------------------------------------------------------------------------
# _extract_expected_count
# ---------------------------------------------------------------------------

class TestExtractExpectedCount:
    def test_happy_path(self):
        assert _extract_expected_count("None (82 securities)") == 82

    def test_missing_value_raises(self):
        with pytest.raises(ValueError):
            _extract_expected_count(np.nan)

    def test_unparseable_value_raises(self):
        with pytest.raises(ValueError):
            _extract_expected_count("some other text entirely")


# ---------------------------------------------------------------------------
# trim_rsi_export
# ---------------------------------------------------------------------------

class TestTrimRsiExport:
    def _shaped_df(self, n_data_rows, footer_rows):
        """Build a DataFrame shaped like a raw RSI read: count row, N data
        rows, then arbitrary footer rows."""
        rows = [[f"None ({n_data_rows} securities)"] + [None] * 3]
        for i in range(n_data_rows):
            rows.append([f"TICK{i} US Equity", f"Company {i}", "100", "US"])
        rows.extend(footer_rows)
        return pd.DataFrame(rows, columns=["Ticker", "Name", "Market Cap", "Cntry Terrtry Of Inc"])

    def test_count_matches_trims_footer(self):
        df = self._shaped_df(3, footer_rows=[[None, None, None, None],
                                              ["A" * 1300, None, None, None]])
        result = trim_rsi_export(df)
        assert len(result) == 3
        assert list(result["Ticker"]) == ["TICK0 US Equity", "TICK1 US Equity", "TICK2 US Equity"]

    def test_count_mismatch_truncated_export_raises(self):
        """Count row says more rows than actually remain."""
        rows = [["None (5 securities)"] + [None] * 3]
        rows.append(["TICK0 US Equity", "Company 0", "100", "US"])
        df = pd.DataFrame(rows, columns=["Ticker", "Name", "Market Cap", "Cntry Terrtry Of Inc"])
        with pytest.raises(ValueError, match="only 1 rows remain"):
            trim_rsi_export(df)

    def test_missing_count_row_raises(self):
        df = self._shaped_df(2, footer_rows=[])
        df.iloc[0, 0] = None
        with pytest.raises(ValueError):
            trim_rsi_export(df)

    def test_tail_that_looks_like_real_data_raises(self):
        """A short 'footer' row (<= 8 chars in its first cell) could
        plausibly be a real ticker, not blank/disclaimer footer — must not
        be silently discarded. (The real disclaimer is ~1,300 chars and a
        blank row is NaN, so neither ever trips this; this is a
        deliberately implausible tail to prove the guard actually fires.)"""
        df = self._shaped_df(2, footer_rows=[["TICK2", "Company 2", "100", "US"]])
        with pytest.raises(ValueError, match="looks like real data"):
            trim_rsi_export(df)


# ---------------------------------------------------------------------------
# clean_rsi_dataframe — unit conversions
# ---------------------------------------------------------------------------
#
# extract_ticker itself is now tested in tests/test_loaders.py, since it
# generalized in Phase 3c.1 into a shared helper. It's still imported here
# for clean_rsi_dataframe's own tests below, which exercise it indirectly.

class TestCleanRsiDataframe:
    @pytest.fixture
    def raw_df(self):
        columns = ["Ticker"] + list(RSI_COLUMN_MAP.keys())
        return pd.DataFrame([
            {
                "Ticker": "LYV US Equity",
                "Name": "LIVE NATION ENTERTAINMENT IN",
                "Market Cap": "43401400673.40001",
                "Cntry Terrtry Of Inc": "US",
                "Avg D Val Traded 20D:D-20": "387276128",
                "Shrt Int:D-1": "24575848",
                "Shrt Int:M-3": "21544706",
                "Shrt Int:M-6": "16603769",
                "SI % Eqty Flt": "15.344791914443464",
                "52Wk High Chg Pct": "-2.671140584956011",
                "BEst Curr EV / BEst Sl BF12M": "1.6266517694691742",
                "Tot Debt LF": "11282712000",
                "Net Debt to EBITDA LF": "1.2272376033493684",
            },
        ], columns=columns)

    def test_ticker_extracted(self, raw_df):
        result = clean_rsi_dataframe(raw_df)
        assert result["ticker"][0] == "LYV"

    def test_null_ev_sales_preserved_not_filled(self, raw_df):
        raw_df.loc[0, "BEst Curr EV / BEst Sl BF12M"] = None
        result = clean_rsi_dataframe(raw_df)
        assert pd.isna(result["ev_sales_raw"][0])

    def test_string_columns_not_coerced(self, raw_df):
        result = clean_rsi_dataframe(raw_df)
        assert result["country_territory_of_inc"][0] == "US"
        assert result["name"][0] == "LIVE NATION ENTERTAINMENT IN"


class TestRsiTransformCalcs:
    @pytest.fixture
    def transformed_row(self):
        df = pd.DataFrame([{
            "market_cap_raw": 43401400673.40001,
            "shrt_int_d1": 24575848,
            "shrt_int_m3": 21544706,
            "shrt_int_m6": 16603769,
            "ev_sales_raw": None,
            "debt_ebitda_raw": 1.2272376033493684,
        }])
        return df

    def test_market_cap_conversion(self, transformed_row):
        result = calc_rsi_market_cap(transformed_row)
        assert result[0] == pytest.approx(43401.40067340001)

    def test_si_change_3m_happy_path(self, transformed_row):
        result = calc_rsi_si_change_3m(transformed_row)
        assert result[0] == pytest.approx(0.14069080357838248)

    def test_si_change_6m_happy_path(self, transformed_row):
        result = calc_rsi_si_change_6m(transformed_row)
        assert result[0] == pytest.approx(0.4801367087195685)

    def test_si_change_3m_zero_denominator_returns_nan(self):
        """CLAUDE.md bug pattern 5 — not exercised by real data, tested anyway."""
        df = pd.DataFrame([{"shrt_int_d1": 100, "shrt_int_m3": 0}])
        result = calc_rsi_si_change_3m(df)
        assert np.isnan(result[0])

    def test_si_change_6m_zero_denominator_returns_nan(self):
        df = pd.DataFrame([{"shrt_int_d1": 100, "shrt_int_m6": 0}])
        result = calc_rsi_si_change_6m(df)
        assert np.isnan(result[0])

    def test_ev_sales_null_preserved(self, transformed_row):
        result = calc_rsi_ev_sales(transformed_row)
        assert pd.isna(result[0])

    def test_debt_ebitda_direct(self, transformed_row):
        result = calc_rsi_debt_ebitda(transformed_row)
        assert result[0] == pytest.approx(1.2272376033493684)


# ---------------------------------------------------------------------------
# ingest_rsi — end-to-end fixture
# ---------------------------------------------------------------------------

def _write_fake_rsi_config(config_path, screen_id) -> None:
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "screens": {
                    screen_id: {
                        "display_name": screen_id,
                        "type": "quant_composite",
                        "universe": {"name": screen_id, "as_of": "2026-08"},
                    }
                }
            },
            f,
        )


def _write_rsi_fixture_xlsx(path, tickers) -> None:
    """Build a tiny .xlsx shaped like the real export: 2 metadata rows,
    a header row, a count row, N data rows, a blank row, a disclaimer."""
    header = ["Ticker"] + list(RSI_COLUMN_MAP.keys())
    rows = [
        ["EQY_FUND_CRNCY", "REL_INDEX", "FA_ADJUSTED"] + [None] * (len(header) - 3),
        ["LCL"] + [None] * (len(header) - 1),
        header,
        [f"None ({len(tickers)} securities)"] + [None] * (len(header) - 1),
    ]
    for i, ticker in enumerate(tickers):
        rows.append([
            f"{ticker} US Equity", f"Company {ticker}", str(1_000_000_000 + i),
            "US", "50000000", "1000000", "900000", "800000",
            "10.0", "-1.0", "2.0", "500000000", "1.5",
        ])
    rows.append([None] * len(header))
    rows.append(["Disclaimer text " * 100] + [None] * (len(header) - 1))

    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, header=False, sheet_name="Sheet1")


class TestIngestRsiEndToEnd:
    def test_full_ingest_and_transform(self, tmp_path):
        screen_id = "fake_rsi_screen"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_rsi_config(config_path, screen_id)

        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        _write_rsi_fixture_xlsx(upload_dir / "export.xlsx", ["AAA", "BBB", "CCC"])

        db_path = str(tmp_path / "test.db")
        ingest_rsi(screen_id=screen_id, upload_dir=str(upload_dir), db_path=db_path,
                   config_path=config_path)

        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{db_path}")

        raw = pd.read_sql_table(table_name("raw_data", screen_id), engine)
        assert len(raw) == 3
        assert set(raw["ticker"]) == {"AAA", "BBB", "CCC"}
        # No disclaimer text leaked into the loaded data.
        assert raw["ticker"].str.len().max() <= 8

        membership = pd.read_sql_table("screen_membership", engine)
        screen_rows = membership[membership["screen_id"] == screen_id]
        assert set(screen_rows["ticker"]) == {"AAA", "BBB", "CCC"}

        transformed = run_rsi_transforms(raw.copy())
        for col in ("market_cap", "adv", "short_interest_pct", "si_change_3m",
                    "si_change_6m", "week_52_high_chg", "ev_sales", "debt_ebitda"):
            assert col in transformed.columns
