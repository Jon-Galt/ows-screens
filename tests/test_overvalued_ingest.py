"""
Unit tests for the Overvalued (FASTGraphs valuation) loader in
src/overvalued_ingest.py.

Coverage: the "x"-suffix parser (positive control, raise on a non-"x"
value), the colon-split ticker (dotted BRK.B-shaped ticker keeps its dot;
a fixed-width/space-split approach would fail), the header-set validator
(exact-set check, not presence-only — a fixture with a renamed column must
raise, and the real header set must NOT raise), Region dropped from the
stored table, and one end-to-end ingest against a small fixture shaped
like the real export. All synthetic frames (T25) — no gitignored DB.
"""

import pandas as pd
import pytest
import yaml

from src.db import table_name
from src.overvalued_ingest import (
    OVERVALUED_EXPECTED_HEADERS,
    _parse_x_suffixed,
    _split_overvalued_ticker,
    clean_overvalued_dataframe,
    ingest_overvalued,
    validate_overvalued_header_set,
)
from src.transform import calc_overvalued_pe_vs_normal_5y


def _real_headers_df(n_rows=2) -> pd.DataFrame:
    return pd.DataFrame({
        "Ticker": ["AAPL:US", "BRK.B:US"][:n_rows],
        "Company Name": ["APPLE INC", "BERKSHIRE HATHAWAY"][:n_rows],
        "Region": ["US", "US"][:n_rows],
        "P/E Diluted": ["37.90x", "10.00x"][:n_rows],
        "Normal P/E 5Y": ["30.45x", "9.11x"][:n_rows],
        "Normal P/E 10Y": ["23.62x", "9.00x"][:n_rows],
        "Normal P/E 15Y": ["20.56x", "8.50x"][:n_rows],
        "GICS Sector": ["Information Technology", "Financials"][:n_rows],
        "GICS Industry": ["Technology Hardware Storage & Peripherals", "Insurance"][:n_rows],
    })


# ---------------------------------------------------------------------------
# _parse_x_suffixed
# ---------------------------------------------------------------------------


class TestParseXSuffixed:
    def test_parses_x_suffixed_string_to_float(self):
        assert _parse_x_suffixed("37.90x") == 37.90

    def test_nan_passes_through(self):
        assert pd.isna(_parse_x_suffixed(float("nan")))

    def test_non_x_suffixed_value_raises(self):
        """A silent format change (vendor drops the 'x') must be caught
        loudly at ingest, not coerced to NaN."""
        with pytest.raises(ValueError):
            _parse_x_suffixed("37.90")


# ---------------------------------------------------------------------------
# _split_overvalued_ticker
# ---------------------------------------------------------------------------


class TestSplitOvervaluedTicker:
    def test_splits_at_first_colon(self):
        assert _split_overvalued_ticker("AAPL:US") == "AAPL"

    def test_dotted_ticker_keeps_its_dot(self):
        """A fixed-width or space-split approach would corrupt this —
        there is no in-repo precedent for a BRK-class mapping, so the dot
        must survive exactly as the source wrote it."""
        assert _split_overvalued_ticker("BRK.B:US") == "BRK.B"

    def test_non_string_passes_through_unchanged(self):
        assert pd.isna(_split_overvalued_ticker(float("nan")))


# ---------------------------------------------------------------------------
# validate_overvalued_header_set
# ---------------------------------------------------------------------------


class TestValidateOvervaluedHeaderSet:
    def test_real_header_set_does_not_raise(self):
        """Positive control — the exact real header set must pass."""
        validate_overvalued_header_set(_real_headers_df())

    def test_renamed_column_raises(self):
        """FASTGraphs' own plausible drift: dropping the slash from
        'P/E Diluted'. A presence-only check (loaders.validate_columns)
        would not catch a column that's merely renamed and still present
        under a different name — this must be a SET check."""
        df = _real_headers_df().rename(columns={"Normal P/E 5Y": "Normal PE 5Y"})
        with pytest.raises(ValueError):
            validate_overvalued_header_set(df)

    def test_missing_column_raises(self):
        df = _real_headers_df().drop(columns=["GICS Industry"])
        with pytest.raises(ValueError):
            validate_overvalued_header_set(df)

    def test_extra_column_raises(self):
        df = _real_headers_df()
        df["Some New Column"] = "x"
        with pytest.raises(ValueError):
            validate_overvalued_header_set(df)

    def test_expected_headers_constant_matches_real_export_shape(self):
        assert OVERVALUED_EXPECTED_HEADERS == set(_real_headers_df().columns)


# ---------------------------------------------------------------------------
# clean_overvalued_dataframe
# ---------------------------------------------------------------------------


class TestCleanOvervaluedDataframe:
    def test_region_is_dropped(self):
        cleaned = clean_overvalued_dataframe(_real_headers_df())
        assert "Region" not in cleaned.columns
        assert "region" not in cleaned.columns

    def test_ticker_split_and_dot_preserved(self):
        cleaned = clean_overvalued_dataframe(_real_headers_df())
        assert list(cleaned["ticker"]) == ["AAPL", "BRK.B"]

    def test_numeric_columns_parsed_to_float(self):
        cleaned = clean_overvalued_dataframe(_real_headers_df())
        for col in ("pe_diluted", "normal_pe_5y", "normal_pe_10y", "normal_pe_15y"):
            assert cleaned[col].dtype.kind == "f"
        assert cleaned["pe_diluted"].iloc[0] == 37.90

    def test_text_columns_stay_text(self):
        cleaned = clean_overvalued_dataframe(_real_headers_df())
        assert cleaned["sector"].iloc[0] == "Information Technology"
        assert cleaned["industry"].iloc[0] == "Technology Hardware Storage & Peripherals"


# ---------------------------------------------------------------------------
# calc_overvalued_pe_vs_normal_5y (transform.py)
# ---------------------------------------------------------------------------


class TestCalcOvervaluedPeVsNormal5y:
    def test_ratio_computed_where_both_inputs_present(self):
        df = pd.DataFrame({"pe_diluted": [37.90], "normal_pe_5y": [30.45]})
        result = calc_overvalued_pe_vs_normal_5y(df)
        assert result.iloc[0] == pytest.approx(37.90 / 30.45)

    def test_null_pe_diluted_gives_null_ratio_not_a_crash(self):
        df = pd.DataFrame({"pe_diluted": [float("nan")], "normal_pe_5y": [218.31]})
        result = calc_overvalued_pe_vs_normal_5y(df)
        assert pd.isna(result.iloc[0])

    def test_zero_denominator_gives_null_not_a_raise_or_warning(self):
        """Architecture Rule 3: this function must never raise. A masked
        divide (not np.where) so the zero-denominator branch is never
        evaluated — no division actually happens on that row."""
        df = pd.DataFrame({"pe_diluted": [10.0], "normal_pe_5y": [0.0]})
        result = calc_overvalued_pe_vs_normal_5y(df)
        assert pd.isna(result.iloc[0])

    def test_unparseable_input_reaching_transform_returns_nan_not_a_raise(self):
        """The raise-vs-NaN boundary: _parse_x_suffixed (ingest) raises on
        bad input; this calc function (transform) must not, even fed a
        non-numeric string directly."""
        df = pd.DataFrame({"pe_diluted": ["not a number"], "normal_pe_5y": [30.45]})
        result = calc_overvalued_pe_vs_normal_5y(df)
        assert pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# End-to-end ingest
# ---------------------------------------------------------------------------


def _write_fake_overvalued_config(config_path, screen_id) -> None:
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "screens": {
                    screen_id: {
                        "display_name": screen_id,
                        "type": "quant_composite",
                        "universe": {"name": screen_id, "as_of": "2026-09"},
                    }
                }
            },
            f,
        )


class TestIngestOvervaluedEndToEnd:
    def test_full_ingest_and_transform(self, tmp_path):
        screen_id = "fake_overvalued_screen"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_overvalued_config(config_path, screen_id)

        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        _real_headers_df().to_csv(upload_dir / "export.csv", index=False)

        db_path = str(tmp_path / "test.db")
        ingest_overvalued(
            screen_id=screen_id, upload_dir=str(upload_dir), db_path=db_path,
            config_path=config_path,
        )

        from sqlalchemy import create_engine
        engine = create_engine(f"sqlite:///{db_path}")

        raw = pd.read_sql_table(table_name("raw_data", screen_id), engine)
        assert len(raw) == 2
        assert set(raw["ticker"]) == {"AAPL", "BRK.B"}
        assert "region" not in raw.columns and "Region" not in raw.columns

        membership = pd.read_sql_table("screen_membership", engine)
        screen_rows = membership[membership["screen_id"] == screen_id]
        assert set(screen_rows["ticker"]) == {"AAPL", "BRK.B"}

        from src.transform import run_overvalued_transforms

        transformed = run_overvalued_transforms(raw.copy())
        assert "pe_vs_normal_5y" in transformed.columns
        assert transformed["pe_vs_normal_5y"].iloc[0] == pytest.approx(37.90 / 30.45)
