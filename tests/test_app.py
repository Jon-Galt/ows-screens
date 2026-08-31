"""
Unit tests for the pure-Python constants in src/app.py that back Phase
3c.1's "show underlying metric values" feature.

app.py itself is Streamlit UI and is verified manually (per CLAUDE.md), not
via pytest — these tests cover only the data structures a silent drift in
score.py's FACTOR_DEFINITIONS could break: a new or renamed factor with no
matching entry here would otherwise show "N/A" in the drill-down instead
of failing loudly.
"""

from src.app import FACTOR_DEFINITIONS, METRIC_COLUMN_FORMATS, METRIC_FORMATS


class TestMetricFormatsCompleteness:
    def test_every_factor_has_a_metric_format(self):
        missing = [f for f in FACTOR_DEFINITIONS if f not in METRIC_FORMATS]
        assert missing == []

    def test_metric_formats_has_no_stale_entries(self):
        """A factor removed from score.py should have its format removed
        too, not linger as a silently-unused entry."""
        stale = [f for f in METRIC_FORMATS if f not in FACTOR_DEFINITIONS]
        assert stale == []

    def test_metric_column_formats_keyed_by_metric_not_factor(self):
        for factor, fmt in METRIC_FORMATS.items():
            metric_col = FACTOR_DEFINITIONS[factor]["metric"]
            assert METRIC_COLUMN_FORMATS[metric_col] == fmt

    def test_every_format_spec_is_applicable_to_a_float(self):
        for fmt in METRIC_FORMATS.values():
            fmt.format(1.23456)  # raises ValueError if the spec is malformed
