"""
Unit tests for the pure run-history/snapshot functions in src/history.py.

Small synthetic DataFrames with known shapes, per the Worker convention.
`now` is always a fixed, passed-in datetime — history.py never calls
datetime.now() itself, so every test here is deterministic.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.history import (
    build_run_row,
    build_screen_run_row,
    build_snapshot_frame,
    encode_row,
    latest_snapshot_per_date,
    new_run_id,
    snapshot_frame_to_stored_frame,
)


# ---------------------------------------------------------------------------
# new_run_id
# ---------------------------------------------------------------------------

class TestNewRunId:
    def test_format(self):
        now = datetime(2026, 9, 1, 14, 5, 2, tzinfo=timezone.utc)
        run_id = new_run_id(now)
        assert run_id.startswith("20260901T140502Z-")
        prefix, _, suffix = run_id.partition("-")
        assert len(suffix) == 6
        int(suffix, 16)  # must be valid hex

    def test_two_calls_same_second_are_distinct(self):
        now = datetime(2026, 9, 1, 14, 5, 2, tzinfo=timezone.utc)
        a = new_run_id(now)
        b = new_run_id(now)
        assert a != b

    def test_lexical_order_matches_chronological_order(self):
        earlier = new_run_id(datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc))
        later = new_run_id(datetime(2026, 9, 1, 11, 0, 0, tzinfo=timezone.utc))
        assert sorted([later, earlier]) == [earlier, later]


# ---------------------------------------------------------------------------
# encode_row / _json_safe normalization
# ---------------------------------------------------------------------------

class TestEncodeRow:
    def test_nan_becomes_null(self):
        result = json.loads(encode_row({"metric": float("nan")}))
        assert result["metric"] is None

    def test_nat_becomes_null(self):
        """Regression lock: pd.NaT subclasses datetime, so a
        Timestamp/datetime isinstance check placed before the null check
        would let NaT.isoformat() ("NaT", a valid JSON string) through as
        a value instead of null. Asserted with `is None` specifically —
        "NaT" is truthy and would pass a weaker check."""
        result = json.loads(encode_row({"as_of": pd.NaT}))
        assert result["as_of"] is None

    def test_none_becomes_null(self):
        result = json.loads(encode_row({"metric": None}))
        assert result["metric"] is None

    def test_pd_na_becomes_null(self):
        result = json.loads(encode_row({"metric": pd.NA}))
        assert result["metric"] is None

    def test_inf_becomes_null(self):
        result = json.loads(encode_row({"metric": float("inf")}))
        assert result["metric"] is None

    def test_negative_inf_becomes_null(self):
        result = json.loads(encode_row({"metric": float("-inf")}))
        assert result["metric"] is None

    def test_numpy_int_becomes_python_int(self):
        encoded = encode_row({"metric": np.int64(5)})
        assert encoded == '{"metric": 5}'

    def test_numpy_float_becomes_python_float(self):
        encoded = encode_row({"metric": np.float64(1.5)})
        assert encoded == '{"metric": 1.5}'

    def test_numpy_bool_becomes_python_bool(self):
        encoded = encode_row({"flag": np.bool_(True)})
        assert encoded == '{"flag": true}'

    def test_real_timestamp_is_isoformatted_not_nulled(self):
        ts = pd.Timestamp("2026-09-01T00:00:00")
        result = json.loads(encode_row({"as_of": ts}))
        assert result["as_of"] == ts.isoformat()

    def test_all_null_row(self):
        encoded = encode_row({"a": None, "b": float("nan"), "c": pd.NaT})
        assert json.loads(encoded) == {"a": None, "b": None, "c": None}

    def test_sort_keys_gives_stable_output(self):
        a = encode_row({"z": 1, "a": 2})
        b = encode_row({"a": 2, "z": 1})
        assert a == b

    def test_output_is_valid_strict_json(self):
        """Independent read-side proof, distinct from the write-side
        allow_nan=False guard — catches a future edit that quietly
        reintroduces a bare NaN/Infinity token."""
        def _reject_constant(name):
            raise ValueError(f"non-finite constant in JSON: {name}")

        encoded = encode_row({"a": 1, "b": None})
        json.loads(encoded, parse_constant=_reject_constant)  # must not raise


# ---------------------------------------------------------------------------
# build_snapshot_frame / snapshot_frame_to_stored_frame round trip
# ---------------------------------------------------------------------------

def _rows_as_normalized_dicts(df: pd.DataFrame) -> list:
    """Both nulls collapse to Python None; dict `==` then does the
    comparison, so a same-value-different-dtype cell (5 vs 5.0) passes and
    a changed value fails."""
    normalized = df.astype(object).where(df.notna(), None)
    return normalized.to_dict(orient="records")


def _assert_round_trip(orig: pd.DataFrame, recon: pd.DataFrame) -> None:
    assert set(recon.columns) == set(orig.columns)  # 1 — before any reindex/align
    assert len(recon) == len(orig)                   # 2
    cols = sorted(orig.columns)                       # 3 — only now align by name
    assert _rows_as_normalized_dicts(orig[cols]) == _rows_as_normalized_dicts(recon[cols])


@pytest.fixture
def stored_df():
    return pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC"],
        "score": [1.5, float("nan"), 3.25],
        "count": [1, 2, 3],
        "flag": [True, False, True],
        "name": ["Alpha", None, "Charlie"],
    })


class TestBuildSnapshotFrame:
    def test_shape_and_columns(self, stored_df):
        snap = build_snapshot_frame(stored_df, "run1", "2026-09-01", "short_screen", "scored_data__short_screen")
        assert list(snap.columns) == ["run_id", "run_date", "screen_id", "ticker", "stage", "data"]
        assert len(snap) == 3
        assert list(snap["ticker"]) == ["AAA", "BBB", "CCC"]
        assert (snap["run_id"] == "run1").all()
        assert (snap["stage"] == "scored_data__short_screen").all()

    def test_data_column_is_strict_json_with_nulls(self, stored_df):
        snap = build_snapshot_frame(stored_df, "run1", "2026-09-01", "short_screen", "scored_data__short_screen")
        row = json.loads(snap.iloc[1]["data"])
        assert row["score"] is None  # NaN -> null
        assert row["name"] is None   # None -> null
        assert row["ticker"] == "BBB"


class TestRoundTrip:
    def test_reconstructs_exactly(self, stored_df):
        snap = build_snapshot_frame(stored_df, "run1", "2026-09-01", "short_screen", "scored_data__short_screen")
        recon = snapshot_frame_to_stored_frame(snap)
        _assert_round_trip(stored_df, recon)

    def test_empty_dataframe(self):
        empty = pd.DataFrame({"ticker": pd.Series(dtype=str), "score": pd.Series(dtype=float)})
        snap = build_snapshot_frame(empty, "run1", "2026-09-01", "short_screen", "scored_data__short_screen")
        assert len(snap) == 0

    # --- Mutation tests: the lock must FAIL on all four, report N/4 ---

    def test_lock_fails_on_added_column(self, stored_df):
        snap = build_snapshot_frame(stored_df, "run1", "2026-09-01", "short_screen", "stage")
        recon = snapshot_frame_to_stored_frame(snap)
        recon["extra"] = "unexpected"
        with pytest.raises(AssertionError):
            _assert_round_trip(stored_df, recon)

    def test_lock_fails_on_dropped_column(self, stored_df):
        snap = build_snapshot_frame(stored_df, "run1", "2026-09-01", "short_screen", "stage")
        recon = snapshot_frame_to_stored_frame(snap).drop(columns=["score"])
        with pytest.raises(AssertionError):
            _assert_round_trip(stored_df, recon)

    def test_lock_fails_on_mutated_value(self, stored_df):
        snap = build_snapshot_frame(stored_df, "run1", "2026-09-01", "short_screen", "stage")
        recon = snapshot_frame_to_stored_frame(snap)
        recon.loc[0, "count"] = 999
        with pytest.raises(AssertionError):
            _assert_round_trip(stored_df, recon)

    def test_lock_fails_on_dropped_row(self, stored_df):
        snap = build_snapshot_frame(stored_df, "run1", "2026-09-01", "short_screen", "stage")
        recon = snapshot_frame_to_stored_frame(snap).iloc[:-1]
        with pytest.raises(AssertionError):
            _assert_round_trip(stored_df, recon)


# ---------------------------------------------------------------------------
# latest_snapshot_per_date
# ---------------------------------------------------------------------------

def _snap_row(run_id, screen_id, ticker, run_date):
    return {"run_id": run_id, "run_date": run_date, "screen_id": screen_id, "ticker": ticker,
            "stage": "stage", "data": "{}"}


class TestLatestSnapshotPerDate:
    def test_two_runs_same_date_resolves_to_later_run(self):
        df = pd.DataFrame([
            _snap_row("20260901T100000Z-aaaaaa", "short_screen", "AAA", "2026-09-01"),
            _snap_row("20260901T110000Z-bbbbbb", "short_screen", "AAA", "2026-09-01"),
        ])
        result = latest_snapshot_per_date(df)
        assert len(result) == 1
        assert result.iloc[0]["run_id"] == "20260901T110000Z-bbbbbb"

    def test_different_dates_both_survive(self):
        df = pd.DataFrame([
            _snap_row("20260901T100000Z-aaaaaa", "short_screen", "AAA", "2026-09-01"),
            _snap_row("20260902T100000Z-cccccc", "short_screen", "AAA", "2026-09-02"),
        ])
        result = latest_snapshot_per_date(df)
        assert len(result) == 2
        assert set(result["run_date"]) == {"2026-09-01", "2026-09-02"}

    def test_single_run_unchanged(self):
        df = pd.DataFrame([_snap_row("20260901T100000Z-aaaaaa", "short_screen", "AAA", "2026-09-01")])
        result = latest_snapshot_per_date(df)
        assert len(result) == 1
        assert result.iloc[0]["run_id"] == "20260901T100000Z-aaaaaa"

    def test_empty_frame_returns_empty(self):
        df = pd.DataFrame(columns=["run_id", "run_date", "screen_id", "ticker", "stage", "data"])
        result = latest_snapshot_per_date(df)
        assert len(result) == 0

    def test_distinct_tickers_and_screens_all_independent(self):
        df = pd.DataFrame([
            _snap_row("20260901T100000Z-aaaaaa", "short_screen", "AAA", "2026-09-01"),
            _snap_row("20260901T100000Z-aaaaaa", "short_screen", "BBB", "2026-09-01"),
            _snap_row("20260901T100000Z-aaaaaa", "cyclicals", "AAA", "2026-09-01"),
        ])
        result = latest_snapshot_per_date(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# build_run_row / build_screen_run_row
# ---------------------------------------------------------------------------

class TestBuildRunRow:
    def test_fields(self):
        started = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
        finished = datetime(2026, 9, 1, 10, 5, 0, tzinfo=timezone.utc)
        row = build_run_row(
            "run1", "2026-09-01", started, finished,
            "src/refresh.py --dry-run", ["short_screen", "cyclicals"], 0, "abc123",
        )
        assert row["run_id"] == "run1"
        assert row["started_at_utc"] == "2026-09-01T10:00:00Z"
        assert row["finished_at_utc"] == "2026-09-01T10:05:00Z"
        assert row["screens_requested"] == "short_screen,cyclicals"
        assert row["exit_code"] == 0
        assert row["git_sha"] == "abc123"

    def test_no_reference_to_refresh_or_validate_types(self):
        """Every argument here is a primitive — proves no ScreenResult
        (defined in refresh.py) or Finding (defined in validate.py) needs
        to be constructed to call this, which is what keeps history.py
        free of the refresh.py import cycle."""
        started = datetime(2026, 9, 1, tzinfo=timezone.utc)
        row = build_run_row("r", "2026-09-01", started, started, "cmd", [], 1, None)
        assert row["git_sha"] is None


class TestBuildScreenRunRow:
    def test_fields(self):
        row = build_screen_run_row(
            "run1", "short_screen", "PASSED", 1299, "scored_data__short_screen",
            1, 1299, [], "export.xlsx", "2026-08-31T12:00:00Z", "deadbeef",
        )
        assert row["findings_json"] == "[]"
        assert row["snapshot_written"] == 1
        assert row["source_file_name"] == "export.xlsx"

    def test_findings_json_never_none(self):
        row = build_screen_run_row(
            "run1", "cyclicals", "FAILED", 0, None, 0, 0, [], None, None, None
        )
        assert row["findings_json"] == "[]"

    def test_findings_serialized(self):
        findings = [{"check": "row_count", "message": "Incoming data has 0 rows."}]
        row = build_screen_run_row(
            "run1", "cyclicals", "FAILED", 0, None, 0, 0, findings, None, None, None
        )
        assert json.loads(row["findings_json"]) == findings
