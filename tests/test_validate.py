"""
Unit tests for the refresh gate's pure validation checks in src/validate.py.

Small synthetic DataFrames with known shapes throughout, per the Worker
convention — no dependency on real screen data.
"""

import pandas as pd

from src.validate import (
    check_no_space_tickers,
    check_null_rate_spike,
    check_row_count,
    check_universe_delta,
    validate_screen,
)


class TestValidateModuleImports:
    def test_no_forbidden_imports(self):
        """validate.py must import neither SQLAlchemy nor Streamlit — it
        operates on DataFrames handed to it by refresh.py, which owns all
        database and file IO. Reading the source text (rather than
        introspecting sys.modules after import) means this test fails even
        if some other already-imported module happens to pull SQLAlchemy
        in transitively."""
        with open("src/validate.py") as f:
            source = f.read()
        forbidden = ("import sqlalchemy", "from sqlalchemy", "import streamlit", "from streamlit")
        found = [line for line in forbidden if line in source]
        assert not found, f"src/validate.py must not import: {found}"


class TestCheckRowCount:
    def test_nonzero_rows_passes(self):
        df = pd.DataFrame({"ticker": ["AAA", "BBB"]})
        assert check_row_count(df) is None

    def test_zero_rows_flagged(self):
        df = pd.DataFrame({"ticker": []})
        finding = check_row_count(df)
        assert finding is not None
        assert finding.check == "row_count"


class TestCheckUniverseDelta:
    def test_no_stored_table_always_passes(self):
        incoming = pd.DataFrame({"ticker": ["AAA"]})
        assert check_universe_delta(incoming, None, max_delta_pct=0.20, max_delta_abs=5) is None

    def test_stored_table_with_zero_rows_always_passes(self):
        """A previously written empty table (e.g. from a hand-run ingest
        against a truncated export — the per-screen functions stay directly
        callable with no gate in front of them) is treated the same as no
        baseline at all, and must not raise ZeroDivisionError."""
        incoming = pd.DataFrame({"ticker": ["AAA", "BBB"]})
        stored = pd.DataFrame({"ticker": []})
        assert check_universe_delta(incoming, stored, max_delta_pct=0.20, max_delta_abs=5) is None

    def test_percentage_within_tolerance_passes(self):
        incoming = pd.DataFrame({"ticker": range(110)})
        stored = pd.DataFrame({"ticker": range(100)})
        # 10 rows / 100 = 10%, within the 20% tolerance; also within the
        # abs floor, so either condition alone would pass this.
        assert check_universe_delta(incoming, stored, max_delta_pct=0.20, max_delta_abs=5) is None

    def test_percentage_fails_but_absolute_floor_rescues(self):
        """management_comp-shaped case: 21 stored rows, a 5-ticker change
        is 23.8% (fails the 20% rule alone) but is within the abs floor."""
        incoming = pd.DataFrame({"ticker": range(16)})
        stored = pd.DataFrame({"ticker": range(21)})
        assert check_universe_delta(incoming, stored, max_delta_pct=0.20, max_delta_abs=5) is None

    def test_both_percentage_and_absolute_fail(self):
        incoming = pd.DataFrame({"ticker": range(5)})
        stored = pd.DataFrame({"ticker": range(21)})
        finding = check_universe_delta(incoming, stored, max_delta_pct=0.20, max_delta_abs=5)
        assert finding is not None
        assert finding.check == "universe_delta"


class TestCheckNullRateSpike:
    def test_no_stored_table_always_passes(self):
        incoming = pd.DataFrame({"metric": [1.0, None, None, None]})
        assert check_null_rate_spike(incoming, None, max_increase_pct=0.15) == []

    def test_null_rate_within_tolerance_passes(self):
        stored = pd.DataFrame({"metric": [1.0] * 10})  # 0% null
        incoming = pd.DataFrame({"metric": [1.0] * 9 + [None]})  # 10% null, +10pp <= 15pp
        assert check_null_rate_spike(incoming, stored, max_increase_pct=0.15) == []

    def test_null_rate_spike_flagged(self):
        stored = pd.DataFrame({"metric": [1.0] * 10})  # 0% null
        incoming = pd.DataFrame({"metric": [None] * 10})  # 100% null, +100pp
        findings = check_null_rate_spike(incoming, stored, max_increase_pct=0.15)
        assert len(findings) == 1
        assert findings[0].check == "null_rate_spike"
        assert "metric" in findings[0].message

    def test_column_only_on_one_side_is_ignored(self):
        stored = pd.DataFrame({"a": [1.0] * 5})
        incoming = pd.DataFrame({"b": [None] * 5})
        assert check_null_rate_spike(incoming, stored, max_increase_pct=0.15) == []


class TestCheckNoSpaceTickers:
    def test_clean_tickers_pass(self):
        df = pd.DataFrame({"ticker": ["AAPL", "TSLA"]})
        assert check_no_space_tickers(df) is None

    def test_ticker_with_space_flagged(self):
        df = pd.DataFrame({"ticker": ["AAPL", "TS LA"]})
        finding = check_no_space_tickers(df)
        assert finding is not None
        assert finding.check == "no_space_tickers"
        assert "TS LA" in finding.message


class TestValidateScreen:
    THRESHOLDS = {
        "universe_size_max_delta_pct": 0.20,
        "universe_size_max_delta_abs": 5,
        "null_rate_max_increase_pct": 0.15,
    }

    def test_clean_data_passes_with_no_findings(self):
        incoming = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})
        result = validate_screen(incoming, None, self.THRESHOLDS)
        assert result.passed is True
        assert result.findings == []

    def test_multiple_failing_checks_all_collected(self):
        incoming = pd.DataFrame({"ticker": ["A A", "BBB"]})
        stored = pd.DataFrame({"ticker": [f"T{i}" for i in range(100)]})
        result = validate_screen(incoming, stored, self.THRESHOLDS)
        assert result.passed is False
        check_names = {f.check for f in result.findings}
        assert "universe_delta" in check_names
        assert "no_space_tickers" in check_names
