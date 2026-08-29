"""
Unit tests for the multi-screen storage helpers in src/db.py, and an
end-to-end regression lock proving those helpers actually deliver
per-screen isolation across the real ingest -> transform -> score pipeline.

Coverage: table_name()'s identifier convention and validation,
sync_screens_registry()'s idempotency, replace_screen_rows()'s per-screen
isolation within a shared, fixed-shape table, and — the most important
guarantee of this phase — that running one screen's full pipeline cannot
alter another screen's per-screen physical tables.
"""

import yaml
import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.config import CONFIG_PATH, load_config
from src.db import replace_screen_rows, sync_screens_registry, table_name
from src.ingest import SCREEN_INGEST_CONFIGS, ingest
from src.score import score
from src.transform import transform


@pytest.fixture
def engine(tmp_path):
    """A fresh, empty SQLite database for each test."""
    return create_engine(f"sqlite:///{tmp_path}/test.db")


# ---------------------------------------------------------------------------
# table_name
# ---------------------------------------------------------------------------

class TestTableName:
    def test_happy_path(self):
        assert table_name("raw_data", "short_screen") == "raw_data__short_screen"
        assert table_name("scored_data", "rising_short_interest") == (
            "scored_data__rising_short_interest"
        )

    @pytest.mark.parametrize(
        "bad_screen_id",
        ["short screen", "short;screen", "Short_Screen", "1screen", "", "short-screen"],
    )
    def test_rejects_unsafe_screen_id(self, bad_screen_id):
        with pytest.raises(ValueError):
            table_name("raw_data", bad_screen_id)


# ---------------------------------------------------------------------------
# sync_screens_registry
# ---------------------------------------------------------------------------

class TestSyncScreensRegistry:
    def test_matches_config(self, engine):
        config = {
            "screens": {
                "short_screen": {"display_name": "OWS Short Screen", "type": "quant_composite"},
            }
        }
        sync_screens_registry(engine, config)
        result = pd.read_sql_table("screens", engine)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["screen_id"] == "short_screen"
        assert row["display_name"] == "OWS Short Screen"
        assert row["screen_type"] == "quant_composite"

    def test_idempotent_under_repeated_calls(self, engine):
        config = {"screens": {"a": {"display_name": "A", "type": "quant_composite"}}}
        sync_screens_registry(engine, config)
        sync_screens_registry(engine, config)
        result = pd.read_sql_table("screens", engine)
        assert len(result) == 1

    def test_rewrites_when_config_changes(self, engine):
        """A screen removed from config.yaml disappears from the registry."""
        sync_screens_registry(
            engine, {"screens": {"a": {"display_name": "A", "type": "quant_composite"}}}
        )
        sync_screens_registry(
            engine, {"screens": {"b": {"display_name": "B", "type": "curated"}}}
        )
        result = pd.read_sql_table("screens", engine)
        assert list(result["screen_id"]) == ["b"]


# ---------------------------------------------------------------------------
# replace_screen_rows
# ---------------------------------------------------------------------------

class TestReplaceScreenRows:
    def test_first_run_creates_table(self, engine):
        """No prior table should not raise; it should just create and insert."""
        df = pd.DataFrame({"screen_id": ["a"], "ticker": ["AAA"]})
        replace_screen_rows(engine, df, "screen_membership", "a")
        result = pd.read_sql_table("screen_membership", engine)
        assert list(result["ticker"]) == ["AAA"]

    def test_rerunning_one_screen_leaves_another_untouched(self, engine):
        """The core multi-screen isolation guarantee for a shared table."""
        df_a = pd.DataFrame({"screen_id": ["a", "a"], "ticker": ["AAA", "BBB"]})
        df_b = pd.DataFrame({"screen_id": ["b", "b"], "ticker": ["CCC", "DDD"]})
        replace_screen_rows(engine, df_a, "screen_membership", "a")
        replace_screen_rows(engine, df_b, "screen_membership", "b")

        # Screen a's universe changes on a rerun (e.g. a new upload).
        df_a_rerun = pd.DataFrame({"screen_id": ["a"], "ticker": ["EEE"]})
        replace_screen_rows(engine, df_a_rerun, "screen_membership", "a")

        result = pd.read_sql_table("screen_membership", engine)
        a_tickers = set(result[result["screen_id"] == "a"]["ticker"])
        b_tickers = set(result[result["screen_id"] == "b"]["ticker"])
        assert a_tickers == {"EEE"}
        assert b_tickers == {"CCC", "DDD"}

    @pytest.mark.parametrize(
        "bad_table", ["screen membership", "screen;membership", "Screen_Membership", "1table"]
    )
    def test_rejects_unsafe_table_name(self, engine, bad_table):
        """The table argument gets the same identifier guard as table_name()."""
        df = pd.DataFrame({"screen_id": ["a"], "ticker": ["AAA"]})
        with pytest.raises(ValueError):
            replace_screen_rows(engine, df, bad_table, "a")


# ---------------------------------------------------------------------------
# Full-pipeline isolation regression lock
# ---------------------------------------------------------------------------
#
# The property under test: running one screen's ENTIRE ingest -> transform ->
# score pipeline must not alter another screen's per-screen physical tables.
# This is trivially true today because each screen writes to its own
# table_name()-derived table via an ordinary to_sql(if_exists="replace") —
# but that is exactly why it needs a test now, while it costs little to
# write. It is the only thing that will catch a future change that
# reintroduces a shared table, passes the wrong stage string, or
# mis-derives a table name — and in 3b, once several screens write
# concurrently, this stops being trivially true.

FAKE_SCREEN_A = "fake_screen_a"
FAKE_SCREEN_B = "fake_screen_b"


def _fixture_row(ticker: str) -> dict:
    """One minimal row using the short_screen's raw Bloomberg column shape
    (source column names, all-numeric values) so the real transform/scoring
    functions run against it without KeyErrors. The exact metric values
    don't matter here — only that the real pipeline runs end-to-end and
    writes to the tables under test."""
    ingest_cfg = SCREEN_INGEST_CONFIGS["short_screen"]
    column_map = ingest_cfg["column_map"]
    string_columns = ingest_cfg["string_columns"]
    row = {}
    for bloomberg_col in ingest_cfg["required_columns"]:
        snake_col = column_map[bloomberg_col]
        if snake_col == "ticker":
            row[bloomberg_col] = ticker
        elif snake_col in string_columns:
            row[bloomberg_col] = "Test"
        else:
            row[bloomberg_col] = "1"
    return row


def _write_fixture_csv(path, tickers) -> None:
    rows = [_fixture_row(t) for t in tickers]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_fake_screens_config(config_path, screen_ids) -> None:
    """A temp config.yaml defining the given screen_ids, reusing the real
    short_screen's factor_weights/scoring block (FACTOR_DEFINITIONS is still
    a single global set as of this phase, so any quant_composite screen
    must be scored against that same factor/weight shape today)."""
    short_screen_cfg = load_config(CONFIG_PATH)["screens"]["short_screen"]
    screens = {
        sid: {
            "display_name": sid,
            "type": "quant_composite",
            "universe": {"name": sid, "as_of": "2026-01"},
            "factor_weights": short_screen_cfg["factor_weights"],
            "scoring": short_screen_cfg["scoring"],
        }
        for sid in screen_ids
    }
    with open(config_path, "w") as f:
        yaml.safe_dump({"screens": screens}, f)


def _run_full_pipeline(screen_id, upload_dir, db_path, config_path) -> None:
    ingest(screen_id=screen_id, upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)
    transform(screen_id=screen_id, db_path=db_path, config_path=config_path)
    score(screen_id=screen_id, db_path=db_path, config_path=config_path)


class TestPipelineIsolation:
    def test_rerunning_one_screens_pipeline_leaves_anothers_tables_untouched(
        self, tmp_path, monkeypatch
    ):
        # Give both synthetic screens an ingest config (real screens beyond
        # short_screen don't exist until 3b — reusing short_screen's shape
        # is enough to prove the storage-isolation mechanism itself).
        monkeypatch.setitem(SCREEN_INGEST_CONFIGS, FAKE_SCREEN_A, SCREEN_INGEST_CONFIGS["short_screen"])
        monkeypatch.setitem(SCREEN_INGEST_CONFIGS, FAKE_SCREEN_B, SCREEN_INGEST_CONFIGS["short_screen"])

        db_path = str(tmp_path / "test.db")
        config_path = str(tmp_path / "config.yaml")
        _write_fake_screens_config(config_path, [FAKE_SCREEN_A, FAKE_SCREEN_B])

        upload_a = tmp_path / "uploads" / FAKE_SCREEN_A
        upload_a.mkdir(parents=True)
        _write_fixture_csv(upload_a / "data.csv", ["AAA", "BBB"])

        upload_b = tmp_path / "uploads" / FAKE_SCREEN_B
        upload_b.mkdir(parents=True)
        _write_fixture_csv(upload_b / "data.csv", ["CCC", "DDD"])

        _run_full_pipeline(FAKE_SCREEN_A, upload_a, db_path, config_path)
        _run_full_pipeline(FAKE_SCREEN_B, upload_b, db_path, config_path)

        engine = create_engine(f"sqlite:///{db_path}")
        stages = ["raw_data", "transformed_data", "scored_data"]
        before = {stage: pd.read_sql_table(table_name(stage, FAKE_SCREEN_B), engine) for stage in stages}

        # Rerun screen A's full pipeline with a different ticker set.
        upload_a_rerun = tmp_path / "uploads" / "fake_screen_a_rerun"
        upload_a_rerun.mkdir()
        _write_fixture_csv(upload_a_rerun / "data.csv", ["EEE"])
        _run_full_pipeline(FAKE_SCREEN_A, upload_a_rerun, db_path, config_path)

        # Screen A's tables reflect the rerun...
        a_raw = pd.read_sql_table(table_name("raw_data", FAKE_SCREEN_A), engine)
        assert list(a_raw["ticker"]) == ["EEE"]

        # ...but screen B's three per-screen tables are byte-for-byte untouched.
        for stage in stages:
            after = pd.read_sql_table(table_name(stage, FAKE_SCREEN_B), engine)
            pd.testing.assert_frame_equal(before[stage], after)

        # screen_membership holds exactly each screen's ingested ticker set,
        # and survives a rerun of the OTHER screen's ingest.
        membership = pd.read_sql_table("screen_membership", engine)
        a_tickers = set(membership[membership["screen_id"] == FAKE_SCREEN_A]["ticker"])
        b_tickers = set(membership[membership["screen_id"] == FAKE_SCREEN_B]["ticker"])
        assert a_tickers == {"EEE"}
        assert b_tickers == {"CCC", "DDD"}
