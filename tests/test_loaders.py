"""
Unit tests for the generic file-reading helpers in src/loaders.py.

find_single_upload_file is genuinely new/generalized logic as of Phase 3c
(it started curated-specific, hardcoded to .csv, and is now parameterized
by expected_extension for three different callers), so it gets its own
tests here rather than inheriting the "no backfill for untested
pre-existing code" reasoning applied to read_upload/validate_columns/
log_summary when they were relocated in 3c's earlier round.
"""

import numpy as np
import pytest

from src.loaders import UploadFileError, extract_ticker, find_single_upload_file


class TestFindSingleUploadFile:
    def test_happy_path_csv(self, tmp_path):
        target = tmp_path / "export.csv"
        target.write_text("x")
        result = find_single_upload_file(str(tmp_path), ".csv")
        assert result == str(target)

    def test_happy_path_xlsx(self, tmp_path):
        target = tmp_path / "export.xlsx"
        target.write_text("x")
        result = find_single_upload_file(str(tmp_path), ".xlsx")
        assert result == str(target)

    def test_no_files(self, tmp_path):
        with pytest.raises(UploadFileError, match="No export file found"):
            find_single_upload_file(str(tmp_path), ".csv")

    def test_multiple_files_named_in_error(self, tmp_path):
        (tmp_path / "a.csv").write_text("x")
        (tmp_path / "b.csv").write_text("x")
        with pytest.raises(UploadFileError) as exc_info:
            find_single_upload_file(str(tmp_path), ".csv")
        assert "a.csv" in str(exc_info.value)
        assert "b.csv" in str(exc_info.value)

    def test_stray_wrong_extension_alongside_expected_is_caught(self, tmp_path):
        """A stray file of the wrong kind must not be silently ignored —
        it has to surface via the 'found N files' branch."""
        (tmp_path / "real_export.xlsx").write_text("x")
        (tmp_path / "leftover.csv").write_text("x")
        with pytest.raises(UploadFileError) as exc_info:
            find_single_upload_file(str(tmp_path), ".xlsx")
        assert "real_export.xlsx" in str(exc_info.value)
        assert "leftover.csv" in str(exc_info.value)

    def test_single_file_wrong_extension_csv_expected(self, tmp_path):
        (tmp_path / "export.xlsx").write_text("x")
        with pytest.raises(UploadFileError, match=r"expects a \.csv export"):
            find_single_upload_file(str(tmp_path), ".csv")

    def test_single_file_wrong_extension_xlsx_expected(self, tmp_path):
        """Proves the function is genuinely parameterized, not still
        secretly csv-only under a new name."""
        (tmp_path / "export.csv").write_text("x")
        with pytest.raises(UploadFileError, match=r"expects a \.xlsx export"):
            find_single_upload_file(str(tmp_path), ".xlsx")


class TestExtractTicker:
    """Moved from test_rsi_ingest.py in Phase 3c.1 when extract_ticker
    generalized from an RSI-only helper to a shared one used by both
    Rising Short Interest and short_screen."""

    @pytest.mark.parametrize(
        "raw_id,expected",
        [
            ("BJ US Equity", "BJ"),        # 2 chars
            ("LYV US Equity", "LYV"),      # 3 chars
            ("HEI US Equity", "HEI"),      # 3 chars
            ("RUSHA US Equity", "RUSHA"),  # 5 chars
            ("BC US Equity", "BC"),
            ("KN US Equity", "KN"),
            ("VC US Equity", "VC"),
            ("TH US Equity", "TH"),
        ],
    )
    def test_named_corrections_and_lengths(self, raw_id, expected):
        assert extract_ticker(raw_id) == expected

    def test_four_char_ticker(self):
        assert extract_ticker("HEIA US Equity") == "HEIA"

    def test_non_string_passthrough(self):
        assert np.isnan(extract_ticker(np.nan))
