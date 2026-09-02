"""
Unit tests for the refresh gate's pure validation checks in src/validate.py.

Small synthetic DataFrames with known shapes throughout, per the Worker
convention — no dependency on real screen data.
"""

import pandas as pd

from src.validate import (
    check_composition_misfile,
    check_no_space_tickers,
    check_null_rate_spike,
    check_row_count,
    normalize_ticker_set,
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


class TestNormalizeTickerSet:
    def test_drops_nulls_casts_and_strips(self):
        series = pd.Series(["AAA", " BBB ", None, "CCC"])
        assert normalize_ticker_set(series) == {"AAA", "BBB", "CCC"}


class TestCheckCompositionMisfile:
    def test_correct_placement_passes(self):
        incoming = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})
        baseline = {
            "structural": {"AAA", "BBB", "CCC"},
            "competition": {"XXX", "YYY", "ZZZ"},
        }
        assert check_composition_misfile(incoming, "structural", baseline) is None

    def test_swap_is_flagged(self):
        """structural's own baseline barely overlaps the incoming export,
        but competition's stored baseline matches it closely — the
        misfile-into-the-wrong-folder scenario this check exists for."""
        incoming = pd.DataFrame({"ticker": ["XXX", "YYY", "ZZZ"]})
        baseline = {
            "structural": {"AAA", "BBB", "CCC"},
            "competition": {"XXX", "YYY", "ZZZ"},
        }
        finding = check_composition_misfile(incoming, "structural", baseline)
        assert finding is not None
        assert finding.check == "composition_misfile"
        assert "competition" in finding.message

    def test_missing_own_baseline_passes(self):
        """First run for this screen — nothing to compare against yet."""
        incoming = pd.DataFrame({"ticker": ["AAA", "BBB"]})
        baseline = {"competition": {"XXX", "YYY", "ZZZ"}}
        assert check_composition_misfile(incoming, "structural", baseline) is None

    def test_empty_peer_entry_is_skipped(self):
        incoming = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})
        baseline = {
            "structural": {"AAA", "BBB"},  # own score < 1.0
            "competition": set(),  # would otherwise be a perfect 0/0 comparison
        }
        assert check_composition_misfile(incoming, "structural", baseline) is None

    def test_exact_tie_does_not_flag(self):
        incoming = pd.DataFrame({"ticker": ["AAA", "BBB"]})
        baseline = {
            "structural": {"AAA", "BBB", "CCC"},
            "competition": {"AAA", "BBB", "DDD"},
        }
        # Both peers share exactly 2 of 4 union tickers with incoming -> tied Jaccard.
        assert check_composition_misfile(incoming, "structural", baseline) is None

    def test_empty_incoming_passes(self):
        incoming = pd.DataFrame({"ticker": []})
        baseline = {"structural": {"AAA"}, "competition": {"BBB"}}
        assert check_composition_misfile(incoming, "structural", baseline) is None


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
    THRESHOLDS = {"null_rate_max_increase_pct": 0.15}

    def test_clean_data_passes_with_no_findings(self):
        incoming = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"]})
        result = validate_screen(incoming, None, self.THRESHOLDS, "structural", {})
        assert result.passed is True
        assert result.findings == []

    def test_multiple_failing_checks_all_collected(self):
        incoming = pd.DataFrame({"ticker": ["A A", "BBB"]})
        baseline = {
            "structural": {"ZZZ"},  # own score 0.0 — no overlap at all
            "competition": {"A A", "BBB", "CCC"},  # a much better match
        }
        result = validate_screen(incoming, None, self.THRESHOLDS, "structural", baseline)
        assert result.passed is False
        check_names = {f.check for f in result.findings}
        assert "composition_misfile" in check_names
        assert "no_space_tickers" in check_names
