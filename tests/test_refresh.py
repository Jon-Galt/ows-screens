"""
Tests for the one-command refresh orchestrator in src/refresh.py.

Synthetic fixtures via tmp_path, isolated config.yaml, isolated db_path —
same convention as test_curated_ingest.py / test_rsi_ingest.py /
test_schema.py. Local fixture-writer helpers are kept in this file rather
than imported from another test module's private helpers, matching how
those three files each keep their own.

The three TestPrepareMatchesIngestWrite tests use the real config.yaml
(CONFIG_PATH, read-only) and the real screen_ids, because refresh.py's
dispatch tables are keyed on those exact ids (short_screen's ingest config
in particular is hardcoded to the literal key "short_screen" — see
ingest.py's SCREEN_INGEST_CONFIGS). All other tests that exercise multiple
screens through refresh()/refresh_one() also use the real config.yaml for
the same reason; db_path is always a tmp_path file, so nothing touches the
real database.
"""

import hashlib
import json

import yaml
import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect, text

from src.config import CONFIG_PATH, ScreenTypeError
from src.db import table_name
from src.ingest import SCREEN_INGEST_CONFIGS
from src.rsi_ingest import RSI_COLUMN_MAP
import src.refresh as refresh


# ---------------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------------

def _short_screen_fixture_row(ticker: str) -> dict:
    """One minimal row using short_screen's raw Bloomberg column shape,
    built from SCREEN_INGEST_CONFIGS itself rather than hand-typing the 81
    column headers — this stays correct if that config ever changes.

    Limit: because the fixture's columns are derived from the same config
    the code reads, this cannot catch a missing-required-column bug.
    That's fine here — these tests compare prepare-output against
    ingest-write, not required-column coverage (which lives in
    tests/test_loaders.py's validate_columns tests)."""
    cfg = SCREEN_INGEST_CONFIGS["short_screen"]
    row = {}
    for bloomberg_col in cfg["required_columns"]:
        snake_col = cfg["column_map"][bloomberg_col]
        if snake_col == "ticker":
            row[bloomberg_col] = f"{ticker} US Equity"
        elif snake_col in cfg["string_columns"]:
            row[bloomberg_col] = "Test"
        else:
            row[bloomberg_col] = "1"
    return row


def _write_short_screen_fixture_xlsx(path, tickers) -> None:
    rows = [_short_screen_fixture_row(t) for t in tickers]
    pd.DataFrame(rows).to_excel(path, index=False, sheet_name="Data")


def _write_short_screen_config(config_path) -> None:
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "screens": {
                    "short_screen": {
                        "display_name": "short_screen",
                        "type": "quant_composite",
                        "universe": {"name": "short_screen", "as_of": "2026-08"},
                    }
                },
                "refresh": {
                    "null_rate_max_increase_pct": 0.15,
                },
            },
            f,
        )


def _write_rsi_fixture_xlsx(path, tickers) -> None:
    """Built exactly as test_rsi_ingest.py's own fixture: 2 metadata rows,
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


def _write_curated_fixture_csv(path, tickers) -> None:
    rows = []
    for i, ticker in enumerate(tickers):
        rows.append({
            "daily_traded_value": f'"{1_000_000 + i}"',
            "exchange_symbol": "NYSE",
            "locations": "US",
            "market_cap": f'"{5000.0 + i}"',
            "name": f"Company {ticker}",
            "sector": "Industrials",
            "stock_performance": f'"{10.0 + i}"',
            "ticker_symbol": ticker,
            "rationale": f"Rationale for {ticker}.",
            "scores": f"Accounting And Disclosure: {10 + i} | Fraud: {20 + i} | Insider: {30 + i}",
            "valuation_ev_revenue_ntm_percentile": str(60.0 + i),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_refresh_config(config_path, screens: dict) -> None:
    """A temp config.yaml with the given screen_id -> config block mapping,
    plus the standard refresh thresholds block."""
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "screens": screens,
                "refresh": {
                    "null_rate_max_increase_pct": 0.15,
                },
            },
            f,
        )


def _curated_screen_block(screen_id) -> dict:
    return {
        "display_name": screen_id,
        "type": "curated",
        "universe": {"name": screen_id, "as_of": "2026-08"},
    }


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------

class TestRegistryCoverage:
    def test_every_registry_screen_has_dispatch_entries(self):
        from src.config import load_config
        config = load_config(CONFIG_PATH)
        for screen_id in config["screens"]:
            assert screen_id in refresh._PREPARE_FUNCS, f"{screen_id} missing a prepare fn"
            assert screen_id in refresh._INGEST_FUNCS, f"{screen_id} missing an ingest fn"

    def test_unregistered_screen_id_raises_loudly(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"fake_seventh_screen": _curated_screen_block("fake_seventh_screen")})
        with pytest.raises(ScreenTypeError):
            refresh.refresh_one(
                "fake_seventh_screen",
                upload_dir=str(tmp_path / "uploads"),
                db_path=str(tmp_path / "test.db"),
                config_path=config_path,
            )


# ---------------------------------------------------------------------------
# Prepare output matches what the real ingest function actually writes
# ---------------------------------------------------------------------------

class TestPrepareMatchesIngestWrite:
    def test_short_screen(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        _write_short_screen_config(config_path)
        upload_dir = tmp_path / "uploads" / "short_screen"
        upload_dir.mkdir(parents=True)
        _write_short_screen_fixture_xlsx(upload_dir / "export.xlsx", ["AAA", "BBB", "CCC", "DDD", "EEE"])

        prepared, prepared_filepath = refresh._prepare_short_screen(str(upload_dir))
        assert prepared_filepath == str(upload_dir / "export.xlsx")

        db_path = str(tmp_path / "test.db")
        from src.ingest import ingest as run_ingest
        run_ingest(screen_id="short_screen", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)
        engine = create_engine(f"sqlite:///{db_path}")
        written = pd.read_sql_table(table_name("raw_data", "short_screen"), engine)

        assert set(prepared.columns) == set(written.columns)
        pd.testing.assert_frame_equal(
            prepared, written.reindex(columns=prepared.columns), check_dtype=False
        )

    def test_rising_short_interest(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {
            "rising_short_interest": {
                "display_name": "rising_short_interest",
                "type": "quant_composite",
                "universe": {"name": "rising_short_interest", "as_of": "2026-08"},
            }
        })
        upload_dir = tmp_path / "uploads" / "rising_short_interest"
        upload_dir.mkdir(parents=True)
        _write_rsi_fixture_xlsx(upload_dir / "export.xlsx", ["AAA", "BBB", "CCC"])

        prepared, prepared_filepath = refresh._prepare_rsi(str(upload_dir))
        assert prepared_filepath == str(upload_dir / "export.xlsx")

        db_path = str(tmp_path / "test.db")
        from src.rsi_ingest import ingest_rsi
        ingest_rsi(screen_id="rising_short_interest", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)
        engine = create_engine(f"sqlite:///{db_path}")
        written = pd.read_sql_table(table_name("raw_data", "rising_short_interest"), engine)

        assert set(prepared.columns) == set(written.columns)
        pd.testing.assert_frame_equal(
            prepared, written.reindex(columns=prepared.columns), check_dtype=False
        )

    def test_curated_screen(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"structural": _curated_screen_block("structural")})
        upload_dir = tmp_path / "uploads" / "structural"
        upload_dir.mkdir(parents=True)
        _write_curated_fixture_csv(upload_dir / "export.csv", ["AAA", "BBB", "CCC"])

        prepared, prepared_filepath = refresh._prepare_curated(str(upload_dir))
        assert prepared_filepath == str(upload_dir / "export.csv")

        db_path = str(tmp_path / "test.db")
        from src.curated_ingest import ingest_curated
        ingest_curated(screen_id="structural", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)
        engine = create_engine(f"sqlite:///{db_path}")
        written = pd.read_sql_table(table_name("curated_data", "structural"), engine)

        assert set(prepared.columns) == set(written.columns)
        pd.testing.assert_frame_equal(
            prepared, written.reindex(columns=prepared.columns), check_dtype=False
        )


# ---------------------------------------------------------------------------
# Gating behavior
# ---------------------------------------------------------------------------

class TestGatingPreservesStoredData:
    def test_validation_failure_leaves_stored_table_untouched(self, tmp_path):
        """A composition-misfile trigger: cyclicals gets a real stored
        baseline first, management_comp's first run establishes its own
        (against no baseline, so it can't misfile against itself), then its
        second run's export is swapped for one that matches cyclicals'
        stored tickers far better than management_comp's own — the export-
        landed-in-the-wrong-folder scenario this check exists for."""
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {
            "management_comp": _curated_screen_block("management_comp"),
            "cyclicals": _curated_screen_block("cyclicals"),
        })
        db_path = str(tmp_path / "test.db")

        cyclicals_dir = tmp_path / "uploads" / "cyclicals"
        cyclicals_dir.mkdir(parents=True)
        cyclicals_tickers = [f"C{i}" for i in range(5)]
        _write_curated_fixture_csv(cyclicals_dir / "export.csv", cyclicals_tickers)
        cyclicals_result = refresh.refresh_one(
            "cyclicals", upload_dir=str(cyclicals_dir), db_path=db_path, config_path=config_path
        )
        assert cyclicals_result.status == refresh.PASSED

        upload_dir = tmp_path / "uploads" / "management_comp"
        upload_dir.mkdir(parents=True)
        good_tickers = [f"M{i}" for i in range(21)]
        _write_curated_fixture_csv(upload_dir / "export.csv", good_tickers)
        first = refresh.refresh_one("management_comp", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)
        assert first.status == refresh.PASSED

        engine = create_engine(f"sqlite:///{db_path}")
        stored_before = pd.read_sql_table(table_name("curated_data", "management_comp"), engine)

        # Swap the export for one identical to cyclicals' stored tickers —
        # zero overlap with management_comp's own baseline, perfect overlap
        # with cyclicals'.
        (upload_dir / "export.csv").unlink()
        _write_curated_fixture_csv(upload_dir / "export.csv", cyclicals_tickers)
        second = refresh.refresh_one("management_comp", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)

        assert second.status == refresh.FAILED
        assert any(f.check == "composition_misfile" for f in second.findings)
        assert any("cyclicals" in f.message for f in second.findings if f.check == "composition_misfile")

        stored_after = pd.read_sql_table(table_name("curated_data", "management_comp"), engine)
        pd.testing.assert_frame_equal(stored_before, stored_after)

    def test_emptied_upload_folder_fails_without_writing(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"cyclicals": _curated_screen_block("cyclicals")})
        db_path = str(tmp_path / "test.db")
        upload_dir = tmp_path / "uploads" / "cyclicals"
        upload_dir.mkdir(parents=True)  # empty — no export file

        result = refresh.refresh_one("cyclicals", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)

        assert result.status == refresh.FAILED
        assert any(f.check == "prepare" for f in result.findings)
        engine = create_engine(f"sqlite:///{db_path}")
        assert not inspect(engine).has_table(table_name("curated_data", "cyclicals"))


# ---------------------------------------------------------------------------
# Baseline freeze — order independence (Phase 3d Part 2c)
# ---------------------------------------------------------------------------

class TestBaselineOrderIndependence:
    def test_composition_check_is_order_independent(self, tmp_path, monkeypatch):
        """Poisoning setup, using two real curated screen_ids (cyclicals and
        structural — arbitrary picks, just two with real prepare/ingest
        dispatch entries): both screens' OLD stored baselines are disjoint
        from each other and from this run's shared incoming ticker family
        Q, so BOTH screens tie (0.0 vs 0.0) against a peer's real pre-run
        state and pass cleanly. But both screens' NEW incoming is the SAME
        family Q. If read_stored_ticker_sets were read fresh inside
        refresh()'s per-screen loop instead of once before it, whichever
        screen processes SECOND would see the FIRST screen's peer baseline
        as its just-written new Q-family data (a perfect match) rather than
        its real disjoint pre-run state — flipping that second screen's tie
        into a strict loss and flagging it. Which screen is "second" (and
        therefore the one that would flip) depends entirely on order, so
        this asserts BOTH screens pass cleanly under BOTH orders — not just
        that the two orders agree, which a same-way-wrong implementation
        could also satisfy."""
        monkeypatch.chdir(tmp_path)
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {
            "cyclicals": _curated_screen_block("cyclicals"),
            "structural": _curated_screen_block("structural"),
        })

        old_tickers = {"cyclicals": ["Z1", "Z2", "Z3", "Z4"], "structural": ["W1", "W2", "W3", "W4"]}
        new_tickers = {"cyclicals": ["Q1", "Q2", "Q3", "Q4"], "structural": ["Q1", "Q2", "Q3", "Q4"]}

        def _seed_and_get_results(order: list, db_name: str):
            db_path = str(tmp_path / db_name)
            for screen_id, tickers in old_tickers.items():
                upload_dir = tmp_path / "data" / "uploads" / screen_id
                upload_dir.mkdir(parents=True)
                _write_curated_fixture_csv(upload_dir / "export.csv", tickers)
                seed_result = refresh.refresh_one(
                    screen_id, upload_dir=str(upload_dir), db_path=db_path, config_path=config_path
                )
                assert seed_result.status == refresh.PASSED

            for screen_id, tickers in new_tickers.items():
                upload_dir = tmp_path / "data" / "uploads" / screen_id
                (upload_dir / "export.csv").unlink()
                _write_curated_fixture_csv(upload_dir / "export.csv", tickers)

            results = refresh.refresh(order, db_path=db_path, config_path=config_path)
            for screen_id in order:
                (tmp_path / "data" / "uploads" / screen_id / "export.csv").unlink()
                (tmp_path / "data" / "uploads" / screen_id).rmdir()
            return {r.screen_id: r for r in results}

        results_ab = _seed_and_get_results(["cyclicals", "structural"], "order_ab.db")
        results_ba = _seed_and_get_results(["structural", "cyclicals"], "order_ba.db")

        for order_name, results in [("[cyclicals, structural]", results_ab), ("[structural, cyclicals]", results_ba)]:
            for screen_id in ("cyclicals", "structural"):
                checks = [f.check for f in results[screen_id].findings]
                assert "composition_misfile" not in checks, (
                    f"order {order_name}: expected no misfile finding for {screen_id}, got {checks}"
                )
                assert results[screen_id].status == refresh.PASSED

        assert [f.check for f in results_ab["cyclicals"].findings] == [f.check for f in results_ba["cyclicals"].findings]
        assert [f.check for f in results_ab["structural"].findings] == [f.check for f in results_ba["structural"].findings]


class TestContinuePastFailure:
    def test_one_screens_failure_does_not_stop_the_others(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {
            "management_comp": _curated_screen_block("management_comp"),
            "cyclicals": _curated_screen_block("cyclicals"),
        })
        db_path = str(tmp_path / "test.db")

        good_dir = tmp_path / "uploads" / "management_comp"
        good_dir.mkdir(parents=True)
        _write_curated_fixture_csv(good_dir / "export.csv", [f"T{i}" for i in range(5)])

        empty_dir = tmp_path / "uploads" / "cyclicals"
        empty_dir.mkdir(parents=True)  # empty — triggers a prepare failure

        results = [
            refresh.refresh_one(
                screen_id, upload_dir=str(tmp_path / "uploads" / screen_id),
                db_path=db_path, config_path=config_path,
            )
            for screen_id in ["management_comp", "cyclicals"]
        ]

        by_id = {r.screen_id: r for r in results}
        assert by_id["management_comp"].status == refresh.PASSED
        assert by_id["management_comp"].row_count == 5
        assert by_id["cyclicals"].status == refresh.FAILED
        assert refresh._exit_code(results) == 1


class TestDryRun:
    def test_dry_run_writes_nothing_and_exits_clean(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        upload_dir = tmp_path / "uploads" / "management_comp"
        upload_dir.mkdir(parents=True)
        _write_curated_fixture_csv(upload_dir / "export.csv", [f"T{i}" for i in range(5)])

        result = refresh.refresh_one(
            "management_comp", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path, dry_run=True
        )

        assert result.status == refresh.PASSED
        assert result.dry_run is True
        assert refresh._exit_code([result]) == 0  # a clean dry run must not exit non-zero

        # Checking for a stored table (has_table) touches the sqlite file,
        # so its mere existence doesn't prove anything; assert it holds no
        # tables at all — nothing was ever written.
        engine = create_engine(f"sqlite:///{db_path}")
        assert inspect(engine).get_table_names() == []

    def test_clean_dry_run_footer_does_not_claim_a_real_refresh(self, tmp_path, capsys):
        """The footer text is a deliverable Tom reads, and a dry run is not
        an actual refresh — its footer must say so, not reuse the real
        run's wording verbatim."""
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        upload_dir = tmp_path / "uploads" / "management_comp"
        upload_dir.mkdir(parents=True)
        _write_curated_fixture_csv(upload_dir / "export.csv", [f"T{i}" for i in range(5)])

        result = refresh.refresh_one(
            "management_comp", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path, dry_run=True
        )
        refresh._print_report([result])
        out = capsys.readouterr().out

        assert "would refresh cleanly" in out
        assert "Nothing written (dry run)" in out
        assert "refreshed cleanly." not in out  # the real-run wording must not leak into a dry run

    def test_gated_dry_run_footer_also_marked_as_dry_run(self, tmp_path, capsys):
        """A dry run that gates a screen out must also say so in its
        needs-attention footer, not just the clean-run footer."""
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"cyclicals": _curated_screen_block("cyclicals")})
        db_path = str(tmp_path / "test.db")
        upload_dir = tmp_path / "uploads" / "cyclicals"
        upload_dir.mkdir(parents=True)  # empty — triggers a prepare failure

        result = refresh.refresh_one(
            "cyclicals", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path, dry_run=True
        )
        assert result.status == refresh.FAILED
        assert result.dry_run is True  # the failure itself happened during a dry run

        refresh._print_report([result])
        out = capsys.readouterr().out

        assert "would need attention" in out
        assert "Nothing written (dry run)" in out


# ---------------------------------------------------------------------------
# Downstream (transform/score) failure after a successful, validated write
# ---------------------------------------------------------------------------

class TestInconsistentDownstream:
    def test_transform_failure_after_good_ingest_is_reported_inconsistent(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_short_screen_config(config_path)
        db_path = str(tmp_path / "test.db")
        upload_dir = tmp_path / "uploads" / "short_screen"
        upload_dir.mkdir(parents=True)
        _write_short_screen_fixture_xlsx(upload_dir / "export.xlsx", ["AAA", "BBB", "CCC"])

        def _boom(*args, **kwargs):
            raise ValueError("synthetic transform failure")

        monkeypatch.setattr(refresh.transform, "transform", _boom)

        result = refresh.refresh_one("short_screen", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)

        assert result.status == refresh.INCONSISTENT
        assert refresh._exit_code([result]) == 1
        assert any(f.check == "transform" for f in result.findings)
        assert "synthetic transform failure" in result.findings[0].message
        assert "python -c" in result.findings[0].message

        engine = create_engine(f"sqlite:///{db_path}")
        raw = pd.read_sql_table(table_name("raw_data", "short_screen"), engine)
        assert len(raw) == 3  # the ingest write itself succeeded and is intact


# ---------------------------------------------------------------------------
# Run history + per-run data snapshots (Phase 3d Part 2b)
# ---------------------------------------------------------------------------

def _history_tables(engine):
    """The three history/snapshot tables, or empty DataFrames if a table
    doesn't exist yet (e.g. immediately after a dry run)."""
    names = inspect(engine).get_table_names()
    return {
        t: (pd.read_sql_table(t, engine) if t in names else pd.DataFrame())
        for t in ("refresh_runs", "refresh_screen_runs", "refresh_snapshots")
    }


def _chdir_with_default_upload(tmp_path, monkeypatch, screen_id) -> None:
    """refresh() (the batch orchestrator) takes no upload_dir parameter —
    it always calls refresh_one() with the default data/uploads/<screen_id>
    (relative to CWD). Every test below that calls refresh() rather than
    refresh_one() directly must therefore chdir into tmp_path and place its
    fixture at that default relative location, or it would silently read
    whatever real folder happens to be at data/uploads/<screen_id> in the
    actual project directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "uploads" / screen_id).mkdir(parents=True)


class TestHistoryDryRunWritesNothing:
    def test_dry_run_leaves_db_byte_identical(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = tmp_path / "test.db"
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        _write_curated_fixture_csv(
            tmp_path / "data" / "uploads" / "management_comp" / "export.csv", [f"T{i}" for i in range(5)]
        )

        # A prior real run establishes a non-empty db so the dry run has
        # something to leave untouched, not just "still empty."
        refresh.refresh(["management_comp"], db_path=str(db_path), config_path=config_path)
        before = hashlib.md5(db_path.read_bytes()).hexdigest()

        refresh.refresh(["management_comp"], db_path=str(db_path), config_path=config_path, dry_run=True)
        after = hashlib.md5(db_path.read_bytes()).hexdigest()

        assert before == after

    def test_dry_run_writes_no_history_rows_from_a_clean_db(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        _write_curated_fixture_csv(
            tmp_path / "data" / "uploads" / "management_comp" / "export.csv", [f"T{i}" for i in range(5)]
        )

        refresh.refresh(["management_comp"], db_path=db_path, config_path=config_path, dry_run=True)

        engine = create_engine(f"sqlite:///{db_path}")
        tables = _history_tables(engine)
        assert all(len(df) == 0 for df in tables.values())


class TestHistoryPersistenceByStatus:
    def test_passed_writes_run_screen_run_and_snapshot(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        _write_curated_fixture_csv(
            tmp_path / "data" / "uploads" / "management_comp" / "export.csv", [f"T{i}" for i in range(5)]
        )

        results = refresh.refresh(["management_comp"], db_path=db_path, config_path=config_path)
        assert results[0].status == refresh.PASSED
        assert results[0].run_id is not None

        engine = create_engine(f"sqlite:///{db_path}")
        tables = _history_tables(engine)
        assert len(tables["refresh_runs"]) == 1
        assert len(tables["refresh_screen_runs"]) == 1
        sr = tables["refresh_screen_runs"].iloc[0]
        assert sr["status"] == "PASSED"
        assert sr["snapshot_written"] == 1
        assert sr["snapshot_row_count"] == 5
        assert sr["stage"] == table_name("curated_data", "management_comp")
        assert sr["source_file_name"] == "export.csv"
        assert len(tables["refresh_snapshots"]) == 5
        assert set(tables["refresh_snapshots"]["ticker"]) == {f"T{i}" for i in range(5)}

        # Every snapshot's data value is strict JSON — an independent
        # read-side proof distinct from the write-side allow_nan=False guard.
        def _reject_constant(name):
            raise ValueError(f"non-finite constant: {name}")
        for value in tables["refresh_snapshots"]["data"]:
            json.loads(value, parse_constant=_reject_constant)

    def test_failed_writes_run_and_screen_run_but_no_snapshot(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"cyclicals": _curated_screen_block("cyclicals")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "cyclicals")  # left empty — triggers a prepare failure

        results = refresh.refresh(["cyclicals"], db_path=db_path, config_path=config_path)
        assert results[0].status == refresh.FAILED

        engine = create_engine(f"sqlite:///{db_path}")
        tables = _history_tables(engine)
        assert len(tables["refresh_runs"]) == 1
        assert len(tables["refresh_screen_runs"]) == 1
        sr = tables["refresh_screen_runs"].iloc[0]
        assert sr["status"] == "FAILED"
        assert sr["snapshot_written"] == 0
        assert pd.isna(sr["stage"])
        assert json.loads(sr["findings_json"])  # non-empty — the prepare finding
        assert len(tables["refresh_snapshots"]) == 0

    def test_inconsistent_writes_run_and_screen_run_with_snapshot_written_zero(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_short_screen_config(config_path)
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "short_screen")
        _write_short_screen_fixture_xlsx(
            tmp_path / "data" / "uploads" / "short_screen" / "export.xlsx", ["AAA", "BBB", "CCC"]
        )

        def _boom(*args, **kwargs):
            raise ValueError("synthetic transform failure")
        monkeypatch.setattr(refresh.transform, "transform", _boom)

        results = refresh.refresh(["short_screen"], db_path=db_path, config_path=config_path)
        assert results[0].status == refresh.INCONSISTENT

        engine = create_engine(f"sqlite:///{db_path}")
        tables = _history_tables(engine)
        sr = tables["refresh_screen_runs"].iloc[0]
        assert sr["status"] == "INCONSISTENT"
        assert sr["snapshot_written"] == 0
        assert sr["stage"] == table_name("transformed_data", "short_screen")  # records what WOULD have been snapshotted
        assert len(tables["refresh_snapshots"]) == 0


class TestHistoryTwoConsecutiveRuns:
    def test_both_runs_persist_with_distinct_run_ids(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        _write_curated_fixture_csv(
            tmp_path / "data" / "uploads" / "management_comp" / "export.csv", [f"T{i}" for i in range(5)]
        )

        first = refresh.refresh(["management_comp"], db_path=db_path, config_path=config_path)
        second = refresh.refresh(["management_comp"], db_path=db_path, config_path=config_path)

        run_id_1, run_id_2 = first[0].run_id, second[0].run_id
        assert run_id_1 != run_id_2

        engine = create_engine(f"sqlite:///{db_path}")
        tables = _history_tables(engine)
        assert set(tables["refresh_runs"]["run_id"]) == {run_id_1, run_id_2}
        assert set(tables["refresh_screen_runs"]["run_id"]) == {run_id_1, run_id_2}
        assert set(tables["refresh_snapshots"]["run_id"]) == {run_id_1, run_id_2}
        assert len(tables["refresh_snapshots"]) == 10  # 5 rows x 2 runs, appended not replaced


class TestHistoryIndexes:
    def test_three_indexes_created(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        _write_curated_fixture_csv(
            tmp_path / "data" / "uploads" / "management_comp" / "export.csv", [f"T{i}" for i in range(5)]
        )

        refresh.refresh(["management_comp"], db_path=db_path, config_path=config_path)

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            names = {
                row[0] for row in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='index'")
                )
            }
        assert {
            "idx_refresh_snapshots_screen_ticker_date",
            "idx_refresh_snapshots_run",
            "idx_refresh_screen_runs_run",
        } <= names


class TestScreenTypeErrorAbortsWithNoHistoryTrace:
    def test_no_dispatch_entry_leaves_zero_history_rows(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"fake_seventh_screen": _curated_screen_block("fake_seventh_screen")})
        db_path = str(tmp_path / "test.db")

        with pytest.raises(ScreenTypeError):
            refresh.refresh(["fake_seventh_screen"], db_path=db_path, config_path=config_path)

        engine = create_engine(f"sqlite:///{db_path}")
        assert inspect(engine).get_table_names() == []  # nothing created at all


class TestHistoryReport:
    def test_run_id_and_footer_shown_on_real_run(self, tmp_path, capsys, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        _write_curated_fixture_csv(
            tmp_path / "data" / "uploads" / "management_comp" / "export.csv", [f"T{i}" for i in range(5)]
        )

        results = refresh.refresh(["management_comp"], db_path=db_path, config_path=config_path)
        refresh._print_report(results)
        out = capsys.readouterr().out

        assert f"Run: {results[0].run_id}" in out
        assert "refresh_snapshots:" in out

    def test_run_id_and_footer_absent_on_dry_run(self, tmp_path, capsys, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        _write_curated_fixture_csv(
            tmp_path / "data" / "uploads" / "management_comp" / "export.csv", [f"T{i}" for i in range(5)]
        )

        results = refresh.refresh(["management_comp"], db_path=db_path, config_path=config_path, dry_run=True)
        refresh._print_report(results)
        out = capsys.readouterr().out

        assert "Run:" not in out
        assert "refresh_snapshots:" not in out


class TestHistoryCli:
    def test_history_before_any_run_exists(self, tmp_path, capsys):
        """Exercises _print_history directly against an explicit db_path
        rather than through main() (which hardcodes "data/screener.db"
        relative to CWD, same as refresh()'s own default) — a fresh db
        file that was never opened doesn't have refresh_runs yet."""
        db_path = str(tmp_path / "never_refreshed.db")
        refresh._print_history(10, db_path)
        out = capsys.readouterr().out
        assert "No refresh runs recorded." in out

    def test_history_rejects_combination_with_screen(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            refresh.main(["--history", "--screen", "cyclicals"])
        assert exc_info.value.code == 2

    def test_history_rejects_combination_with_dry_run(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            refresh.main(["--history", "--dry-run"])
        assert exc_info.value.code == 2

    def test_history_prints_existing_runs(self, tmp_path, capsys, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        _write_curated_fixture_csv(
            tmp_path / "data" / "uploads" / "management_comp" / "export.csv", [f"T{i}" for i in range(5)]
        )
        results = refresh.refresh(["management_comp"], db_path=db_path, config_path=config_path)

        refresh._print_history(10, db_path)
        out = capsys.readouterr().out

        assert results[0].run_id in out
        assert "management_comp" in out


# ---------------------------------------------------------------------------
# Per-screen --force (Phase 3d Part 2c)
# ---------------------------------------------------------------------------

def _quant_screen_block(screen_id) -> dict:
    return {"display_name": screen_id, "type": "quant_composite", "universe": {"name": screen_id, "as_of": "2026-08"}}


class TestForceOverride:
    def _seeded_misfile_scenario(self, tmp_path, monkeypatch):
        """cyclicals gets a real stored baseline first; short_screen's first
        run establishes its own baseline (against nothing, so it passes);
        then short_screen's export is swapped for one matching cyclicals'
        tickers far better than its own — the same composition-misfile
        trigger TestGatingPreservesStoredData uses, reused here so --force
        has a real finding to override. Fixtures are placed at the CWD-
        relative data/uploads/<screen_id> refresh() always reads (same
        convention as _chdir_with_default_upload), since some callers below
        exercise refresh() rather than refresh_one() directly. Returns
        (config_path, db_path, upload_dir) with the poisoned export already
        in place, ready for the caller to refresh_one("short_screen",
        force=...) or refresh(["short_screen"], force_screen_ids=...)."""
        monkeypatch.chdir(tmp_path)
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {
            "short_screen": _quant_screen_block("short_screen"),
            "cyclicals": _curated_screen_block("cyclicals"),
        })
        db_path = str(tmp_path / "test.db")

        cyclicals_dir = tmp_path / "data" / "uploads" / "cyclicals"
        cyclicals_dir.mkdir(parents=True)
        cyclicals_tickers = [f"C{i}" for i in range(5)]
        _write_curated_fixture_csv(cyclicals_dir / "export.csv", cyclicals_tickers)
        assert refresh.refresh_one(
            "cyclicals", upload_dir=str(cyclicals_dir), db_path=db_path, config_path=config_path
        ).status == refresh.PASSED

        upload_dir = tmp_path / "data" / "uploads" / "short_screen"
        upload_dir.mkdir(parents=True)
        _write_short_screen_fixture_xlsx(upload_dir / "export.xlsx", [f"S{i}" for i in range(5)])
        assert refresh.refresh_one(
            "short_screen", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path
        ).status == refresh.PASSED

        (upload_dir / "export.xlsx").unlink()
        _write_short_screen_fixture_xlsx(upload_dir / "export.xlsx", cyclicals_tickers)

        return config_path, db_path, upload_dir

    def test_force_writes_despite_findings_and_records_forced(self, tmp_path, monkeypatch):
        config_path, db_path, upload_dir = self._seeded_misfile_scenario(tmp_path, monkeypatch)

        result = refresh.refresh(
            ["short_screen"], db_path=db_path, config_path=config_path, force_screen_ids=["short_screen"]
        )[0]

        assert result.status == refresh.PASSED
        assert result.forced == 1
        assert any(f.check == "composition_misfile" for f in result.findings)

        engine = create_engine(f"sqlite:///{db_path}")
        sr = pd.read_sql_table("refresh_screen_runs", engine)
        sr = sr[sr["run_id"] == result.run_id].iloc[0]
        assert sr["forced"] == 1
        assert json.loads(sr["findings_json"])  # non-empty — the overridden finding preserved

    def test_force_with_no_findings_is_a_noop(self, tmp_path, monkeypatch):
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        _chdir_with_default_upload(tmp_path, monkeypatch, "management_comp")
        upload_dir = tmp_path / "data" / "uploads" / "management_comp"
        _write_curated_fixture_csv(upload_dir / "export.csv", [f"T{i}" for i in range(5)])

        result = refresh.refresh(
            ["management_comp"], db_path=db_path, config_path=config_path, force_screen_ids=["management_comp"]
        )[0]

        assert result.status == refresh.PASSED
        assert result.forced == 0
        assert result.findings == []

        refresh._print_report([result])

    def test_force_unknown_screen_id_errors(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            refresh.main(["--force", "not_a_real_screen"])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "not_a_real_screen" in err

    def test_force_rejected_with_history(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            refresh.main(["--history", "--force", "cyclicals"])
        assert exc_info.value.code == 2

    def test_forced_screen_still_produces_snapshot(self, tmp_path, monkeypatch):
        config_path, db_path, upload_dir = self._seeded_misfile_scenario(tmp_path, monkeypatch)

        result = refresh.refresh(
            ["short_screen"], db_path=db_path, config_path=config_path, force_screen_ids=["short_screen"]
        )[0]

        assert result.status == refresh.PASSED
        engine = create_engine(f"sqlite:///{db_path}")
        snapshots = pd.read_sql_table("refresh_snapshots", engine)
        assert len(snapshots[(snapshots["run_id"] == result.run_id) & (snapshots["screen_id"] == "short_screen")]) == 5

    def test_dry_run_force_reports_would_be_forced(self, tmp_path, capsys, monkeypatch):
        config_path, db_path, upload_dir = self._seeded_misfile_scenario(tmp_path, monkeypatch)

        result = refresh.refresh_one(
            "short_screen", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path,
            dry_run=True, force=True,
        )

        assert result.status == refresh.PASSED
        assert result.dry_run is True
        assert result.forced == 1

        refresh._print_report([result])
        out = capsys.readouterr().out
        assert "would be forced" in out
        assert "Nothing written (dry run)" in out

    def test_forced_screen_can_still_end_inconsistent_downstream(self, tmp_path, monkeypatch):
        config_path, db_path, upload_dir = self._seeded_misfile_scenario(tmp_path, monkeypatch)

        def _boom(*args, **kwargs):
            raise ValueError("synthetic transform failure")
        monkeypatch.setattr(refresh.transform, "transform", _boom)

        result = refresh.refresh_one(
            "short_screen", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path, force=True
        )

        assert result.status == refresh.INCONSISTENT
        assert result.forced == 1  # forced records the validation-gate override, unaffected by the later failure
        checks = [f.check for f in result.findings]
        assert "composition_misfile" in checks
        assert "transform" in checks
