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

import yaml
import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect

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
                    "universe_size_max_delta_pct": 0.20,
                    "universe_size_max_delta_abs": 5,
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
                    "universe_size_max_delta_pct": 0.20,
                    "universe_size_max_delta_abs": 5,
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

        prepared = refresh._prepare_short_screen(str(upload_dir))

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

        prepared = refresh._prepare_rsi(str(upload_dir))

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

        prepared = refresh._prepare_curated(str(upload_dir))

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
        config_path = str(tmp_path / "config.yaml")
        _write_refresh_config(config_path, {"management_comp": _curated_screen_block("management_comp")})
        db_path = str(tmp_path / "test.db")
        upload_dir = tmp_path / "uploads" / "management_comp"
        upload_dir.mkdir(parents=True)

        good_tickers = [f"T{i}" for i in range(21)]
        _write_curated_fixture_csv(upload_dir / "export.csv", good_tickers)
        first = refresh.refresh_one("management_comp", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)
        assert first.status == refresh.PASSED

        engine = create_engine(f"sqlite:///{db_path}")
        stored_before = pd.read_sql_table(table_name("curated_data", "management_comp"), engine)

        # Replace the export with one whose universe collapses far past
        # both the percentage and absolute tolerance (21 -> 2 rows).
        (upload_dir / "export.csv").unlink()
        _write_curated_fixture_csv(upload_dir / "export.csv", good_tickers[:2])
        second = refresh.refresh_one("management_comp", upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)

        assert second.status == refresh.FAILED
        assert any(f.check == "universe_delta" for f in second.findings)

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
