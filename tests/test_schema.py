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
from sqlalchemy import create_engine, text

from src.config import CONFIG_PATH, ScreenTypeError, load_config
from src.curated_ingest import ingest_curated
from src.db import append_rows, create_index_if_not_exists, replace_screen_rows, sync_screens_registry, table_name
from src.ingest import SCREEN_INGEST_CONFIGS, ingest
from src.loaders import UploadFileError
from src.score import score
from src.transform import SCREEN_TRANSFORM_FUNCS, transform


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
# append_rows / create_index_if_not_exists (Phase 3d Part 2b)
# ---------------------------------------------------------------------------

class TestAppendRows:
    def test_appends_rather_than_replaces(self, engine):
        first = pd.DataFrame({"run_id": ["r1"], "row_count": [5]})
        second = pd.DataFrame({"run_id": ["r2"], "row_count": [7]})
        append_rows(engine, first, "refresh_runs")
        append_rows(engine, second, "refresh_runs")
        result = pd.read_sql_table("refresh_runs", engine)
        assert list(result["run_id"]) == ["r1", "r2"]

    def test_creates_table_on_first_use(self, engine):
        df = pd.DataFrame({"run_id": ["r1"]})
        append_rows(engine, df, "refresh_runs")
        assert len(pd.read_sql_table("refresh_runs", engine)) == 1

    @pytest.mark.parametrize("bad_table", ["refresh runs", "Refresh_Runs", "1table"])
    def test_rejects_unsafe_table_name(self, engine, bad_table):
        df = pd.DataFrame({"run_id": ["r1"]})
        with pytest.raises(ValueError):
            append_rows(engine, df, bad_table)


class TestCreateIndexIfNotExists:
    def test_idempotent_across_two_calls(self, engine):
        df = pd.DataFrame({"run_id": ["r1"], "screen_id": ["short_screen"]})
        append_rows(engine, df, "refresh_screen_runs")
        create_index_if_not_exists(engine, "idx_test_run", "refresh_screen_runs", ["run_id"])
        create_index_if_not_exists(engine, "idx_test_run", "refresh_screen_runs", ["run_id"])  # no error

        with engine.connect() as conn:
            names = [
                row[0] for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_test_run'")
                )
            ]
        assert names == ["idx_test_run"]

    @pytest.mark.parametrize("bad_index_name", ["idx test", "Idx_Test", "1idx"])
    def test_rejects_unsafe_index_name(self, engine, bad_index_name):
        with pytest.raises(ValueError):
            create_index_if_not_exists(engine, bad_index_name, "refresh_runs", ["run_id"])

    @pytest.mark.parametrize("bad_table", ["refresh runs", "Refresh_Runs"])
    def test_rejects_unsafe_table_name(self, engine, bad_table):
        with pytest.raises(ValueError):
            create_index_if_not_exists(engine, "idx_test", bad_table, ["run_id"])

    def test_rejects_unsafe_column_name(self, engine):
        """The security assertion — columns are interpolated into the
        index DDL too, and refresh.py's integration tests wouldn't reach
        this path since every real caller passes fixed column-name literals."""
        df = pd.DataFrame({"run_id": ["r1"]})
        append_rows(engine, df, "refresh_runs")
        with pytest.raises(ValueError):
            create_index_if_not_exists(engine, "idx_test", "refresh_runs", ["run_id; DROP TABLE refresh_runs"])


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


def _write_fixture_xlsx(path, tickers) -> None:
    """Written as .xlsx with a "Data" sheet, matching
    SCREEN_INGEST_CONFIGS["short_screen"]'s expected_extension and
    sheet_name — the single-file check now enforces both."""
    rows = [_fixture_row(t) for t in tickers]
    pd.DataFrame(rows).to_excel(path, index=False, sheet_name="Data")


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
        # Give both synthetic screens an ingest config and a transform
        # function (real screens beyond short_screen don't exist until 3b/
        # 3c — reusing short_screen's shape is enough to prove the
        # storage-isolation mechanism itself).
        monkeypatch.setitem(SCREEN_INGEST_CONFIGS, FAKE_SCREEN_A, SCREEN_INGEST_CONFIGS["short_screen"])
        monkeypatch.setitem(SCREEN_INGEST_CONFIGS, FAKE_SCREEN_B, SCREEN_INGEST_CONFIGS["short_screen"])
        monkeypatch.setitem(SCREEN_TRANSFORM_FUNCS, FAKE_SCREEN_A, SCREEN_TRANSFORM_FUNCS["short_screen"])
        monkeypatch.setitem(SCREEN_TRANSFORM_FUNCS, FAKE_SCREEN_B, SCREEN_TRANSFORM_FUNCS["short_screen"])

        db_path = str(tmp_path / "test.db")
        config_path = str(tmp_path / "config.yaml")
        _write_fake_screens_config(config_path, [FAKE_SCREEN_A, FAKE_SCREEN_B])

        upload_a = tmp_path / "uploads" / FAKE_SCREEN_A
        upload_a.mkdir(parents=True)
        _write_fixture_xlsx(upload_a / "data.xlsx", ["AAA", "BBB"])

        upload_b = tmp_path / "uploads" / FAKE_SCREEN_B
        upload_b.mkdir(parents=True)
        _write_fixture_xlsx(upload_b / "data.xlsx", ["CCC", "DDD"])

        _run_full_pipeline(FAKE_SCREEN_A, upload_a, db_path, config_path)
        _run_full_pipeline(FAKE_SCREEN_B, upload_b, db_path, config_path)

        engine = create_engine(f"sqlite:///{db_path}")
        stages = ["raw_data", "transformed_data", "scored_data"]
        before = {stage: pd.read_sql_table(table_name(stage, FAKE_SCREEN_B), engine) for stage in stages}

        # Rerun screen A's full pipeline with a different ticker set.
        upload_a_rerun = tmp_path / "uploads" / "fake_screen_a_rerun"
        upload_a_rerun.mkdir()
        _write_fixture_xlsx(upload_a_rerun / "data.xlsx", ["EEE"])
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


# ---------------------------------------------------------------------------
# Type-aware dispatch
# ---------------------------------------------------------------------------
#
# Without these guards, calling score() on a curated screen would reach
# compute_factor_scores and raise an opaque KeyError: 'scoring' deep inside
# scoring logic — "does something undefined" rather than failing clearly at
# the top. Each guard checks config.yaml's screen type before touching any
# table, using real screen_ids from config.yaml (short_screen is
# quant_composite; cyclicals is curated) rather than synthetic ones, since
# both now genuinely exist there.

class TestTypeAwareDispatch:
    def test_ingest_rejects_curated_screen(self):
        with pytest.raises(ScreenTypeError, match="quant_composite"):
            ingest(screen_id="cyclicals")

    def test_ingest_curated_rejects_quant_composite_screen(self):
        with pytest.raises(ScreenTypeError, match="curated"):
            ingest_curated(screen_id="short_screen")

    def test_transform_rejects_curated_screen(self):
        with pytest.raises(ScreenTypeError, match="quant_composite"):
            transform(screen_id="cyclicals")

    def test_score_rejects_curated_screen(self):
        with pytest.raises(ScreenTypeError, match="quant_composite"):
            score(screen_id="cyclicals")

    def test_score_rejects_unscored_quant_composite_screen(self):
        """Rising Short Interest is quant_composite in type but has no
        factor_weights — a different failure mode from the curated
        rejection above, so asserting on message content proves the
        RIGHT guard fired, not the type-check one by coincidence."""
        with pytest.raises(ScreenTypeError, match="factor_weights"):
            score(screen_id="rising_short_interest")

    def test_transform_succeeds_for_rising_short_interest(self, tmp_path):
        """Regression lock for the gap found while planning 3c:
        transform() must dispatch to run_rsi_transforms for this
        screen_id, not crash inside short_screen's calc functions on a
        missing column like ps_ntm."""
        screen_id = "rising_short_interest"
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(
                {
                    "screens": {
                        screen_id: {
                            "display_name": "Rising Short Interest",
                            "type": "quant_composite",
                            "universe": {"name": screen_id, "as_of": "2026-01"},
                        }
                    }
                },
                f,
            )
        db_path = str(tmp_path / "test.db")
        engine = create_engine(f"sqlite:///{db_path}")
        raw_df = pd.DataFrame([{
            "ticker": "AAA",
            "name": "Company AAA",
            "market_cap_raw": 1_000_000.0,
            "country_territory_of_inc": "US",
            "adv_raw": 500_000.0,
            "shrt_int_d1": 100.0,
            "shrt_int_m3": 90.0,
            "shrt_int_m6": 80.0,
            "short_interest_pct_raw": 10.0,
            "week_52_high_chg_raw": -5.0,
            "ev_sales_raw": 2.0,
            "tot_debt_lf": 500.0,
            "debt_ebitda_raw": 1.5,
        }])
        raw_df.to_sql(table_name("raw_data", screen_id), engine, if_exists="replace", index=False)

        transform(screen_id=screen_id, db_path=db_path, config_path=config_path)

        result = pd.read_sql_table(table_name("transformed_data", screen_id), engine)
        for col in (
            "market_cap", "adv", "short_interest_pct", "si_change_3m",
            "si_change_6m", "week_52_high_chg", "ev_sales", "debt_ebitda",
        ):
            assert col in result.columns

    def test_ingest_rejects_multiple_files_in_short_screen_folder(self, tmp_path):
        """The single-file discipline built for curated ingest now reaches
        short_screen's path too, not just curated_ingest.py's tests."""
        upload_dir = tmp_path / "uploads" / "short_screen"
        upload_dir.mkdir(parents=True)
        (upload_dir / "a.xlsx").write_text("x")
        (upload_dir / "b.xlsx").write_text("x")

        with pytest.raises(UploadFileError):
            ingest(
                screen_id="short_screen",
                upload_dir=str(upload_dir),
                db_path=str(tmp_path / "test.db"),
            )
