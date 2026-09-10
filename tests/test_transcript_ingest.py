"""
Unit tests for the Negative Expert Transcripts loader in src/transcript_ingest.py.

Coverage: multi-ticker explode, deterministic DocID synthesis, the
unparseable-date guard's message, the "Ticker Counts" sheet never being
read, source_sector never leaking under the name "sector", and an
end-to-end ingest exercising upsert-driven accumulation and deduped
membership. Generic upsert_rows() semantics (idempotence, in-place update,
cross-batch accumulation) are tested in tests/test_schema.py's
TestUpsertRows, alongside every other src/db.py helper.
"""

import logging

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine

from src.config import ScreenTypeError
from src.db import table_name
from src.transcript_ingest import (
    TRANSCRIPT_COLUMN_MAP,
    clean_and_explode_transcripts,
    clean_transcript_dataframe,
    explode_tickers,
    find_duplicate_transcript_groups,
    ingest_transcripts,
    synthesize_doc_ids,
)

RAW_COLUMNS = ["Transcript Date", "Tickers"] + list(TRANSCRIPT_COLUMN_MAP)


def _raw_row(**overrides):
    row = {
        "Transcript Date": "July 6, 2026",
        "Tickers": "AAA",
        "Expert Position/Background": "Former Exec",
        "Key Takeaways": "Some takeaway",
        "Negativity Rationale": "Clear bearish implication",
        "Source Pack": "Special Jul–Aug",
        "Sector": "Tech",
        "Theme": "Some Theme",
        "Company": "Some Company Inc",
        "DocID": "EC-1",
    }
    row.update(overrides)
    return row


def _raw_df(rows):
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


# ---------------------------------------------------------------------------
# explode_tickers
# ---------------------------------------------------------------------------

class TestExplodeTickers:
    def test_multi_ticker_row_splits_into_separate_rows(self):
        df = synthesize_doc_ids(clean_transcript_dataframe(
            _raw_df([_raw_row(Tickers="HON, GE", DocID="EC-172")])
        ))
        result = explode_tickers(df)
        assert len(result) == 2
        assert list(result["ticker"]) == ["HON", "GE"]
        # No literal "HON, GE" ticker and no leaked whitespace.
        assert "HON, GE" not in set(result["ticker"])
        assert all(t == t.strip() for t in result["ticker"])

    def test_single_ticker_row_unaffected(self):
        df = synthesize_doc_ids(clean_transcript_dataframe(_raw_df([_raw_row(Tickers="AAA")])))
        result = explode_tickers(df)
        assert len(result) == 1
        assert result["ticker"].iloc[0] == "AAA"

    def test_exploded_rows_share_one_doc_id_and_company(self):
        """The HON/GE case: both exploded rows carry the SAME doc_id and
        the transcript's own (not per-ticker) transcript_company — company
        name is never re-derived per ticker."""
        df = synthesize_doc_ids(clean_transcript_dataframe(
            _raw_df([_raw_row(Tickers="HON, GE", DocID="EC-172", Company="Honeywell International Inc")])
        ))
        result = explode_tickers(df)
        assert result["doc_id"].nunique() == 1
        assert set(result["transcript_company"]) == {"Honeywell International Inc"}


# ---------------------------------------------------------------------------
# clean_transcript_dataframe — the unparseable-date guard
# ---------------------------------------------------------------------------

class TestCleanTranscriptDataframe:
    def test_source_sector_column_present_not_sector(self):
        result = clean_transcript_dataframe(_raw_df([_raw_row(Sector="Tech")]))
        assert "source_sector" in result.columns
        assert "sector" not in result.columns
        assert result["source_sector"].iloc[0] == "Tech"

    def test_unparseable_date_raises_naming_row_and_theme(self):
        with pytest.raises(ValueError) as exc_info:
            clean_transcript_dataframe(_raw_df([_raw_row(**{"Transcript Date": "not a date"}, Theme="My Theme")]))
        message = str(exc_info.value)
        assert "1" in message
        assert "not a date" in message
        assert "My Theme" in message


# ---------------------------------------------------------------------------
# synthesize_doc_ids
# ---------------------------------------------------------------------------

class TestSynthesizeDocIds:
    def test_real_doc_id_preserved_and_not_flagged(self):
        df = clean_transcript_dataframe(_raw_df([_raw_row(DocID="EC-999")]))
        result = synthesize_doc_ids(df)
        assert result["doc_id"].iloc[0] == "EC-999"
        assert result["doc_id_synthetic"].iloc[0] == 0

    def test_missing_doc_id_synthesized_and_flagged(self):
        df = clean_transcript_dataframe(_raw_df([_raw_row(DocID=None)]))
        result = synthesize_doc_ids(df)
        assert result["doc_id"].iloc[0].startswith("SYN-")
        assert result["doc_id_synthetic"].iloc[0] == 1

    def test_deterministic_across_separate_calls(self):
        df = clean_transcript_dataframe(
            _raw_df([_raw_row(DocID=None, **{"Transcript Date": "August 1, 2026"}, Theme="Theme A")])
        )
        first = synthesize_doc_ids(df.copy())["doc_id"].iloc[0]
        second = synthesize_doc_ids(df.copy())["doc_id"].iloc[0]
        assert first == second

    def test_distinct_inputs_produce_distinct_ids(self):
        df = clean_transcript_dataframe(_raw_df([
            _raw_row(DocID=None, **{"Transcript Date": "August 1, 2026"}, Theme="Theme A"),
            _raw_row(DocID=None, **{"Transcript Date": "August 2, 2026"}, Theme="Theme B"),
        ]))
        result = synthesize_doc_ids(df)
        assert result["doc_id"].iloc[0] != result["doc_id"].iloc[1]

    def test_no_collision_between_synthetic_and_real(self):
        """A real DocID that happened to be null elsewhere must not collide
        with a synthesized id sharing the same (date, theme)."""
        df = clean_transcript_dataframe(_raw_df([
            _raw_row(DocID=None, **{"Transcript Date": "August 1, 2026"}, Theme="Theme A"),
            _raw_row(DocID="EC-real", **{"Transcript Date": "August 1, 2026"}, Theme="Theme A"),
        ]))
        result = synthesize_doc_ids(df)
        assert result["doc_id"].nunique() == 2


# ---------------------------------------------------------------------------
# ingest_transcripts — end to end
# ---------------------------------------------------------------------------

def _write_fake_config(config_path, screen_id, screen_type="quant_composite") -> None:
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "screens": {
                    screen_id: {
                        "display_name": screen_id,
                        "type": screen_type,
                        "universe": {"name": screen_id, "as_of": "2026-09"},
                    }
                }
            },
            f,
        )


def _write_transcript_fixture_xlsx(path, rows, ticker_counts_df=None) -> None:
    """Build a small workbook shaped like the real export: a "Transcript
    Summaries" sheet plus an optional, deliberately-disagreeing "Ticker
    Counts" sheet (never read by this module)."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _raw_df(rows).to_excel(writer, sheet_name="Transcript Summaries", index=False)
        if ticker_counts_df is not None:
            ticker_counts_df.to_excel(writer, sheet_name="Ticker Counts", index=False)


class TestIngestTranscriptsEndToEnd:
    def test_rejects_curated_screen_type(self, tmp_path):
        screen_id = "fake_transcripts"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_config(config_path, screen_id, screen_type="curated")
        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        _write_transcript_fixture_xlsx(upload_dir / "export.xlsx", [_raw_row()])

        with pytest.raises(ScreenTypeError):
            ingest_transcripts(screen_id=screen_id, upload_dir=str(upload_dir),
                                db_path=str(tmp_path / "test.db"), config_path=config_path)

    def test_ticker_counts_sheet_not_read(self, tmp_path):
        """Same Transcript Summaries content, wildly different (and
        malformed) Ticker Counts sheets, must produce identical output."""
        screen_id = "fake_transcripts"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_config(config_path, screen_id)
        rows = [_raw_row(Tickers="AAA", DocID="EC-1"), _raw_row(Tickers="BBB", DocID="EC-2")]

        results = []
        for ticker_counts in [
            None,
            pd.DataFrame({"Ticker": ["ZZZ"], "Mentions": [999], "Company Name": ["Nonsense"],
                          "Market Cap": ["not even numeric"]}),
        ]:
            upload_dir = tmp_path / f"uploads_{len(results)}" / screen_id
            upload_dir.mkdir(parents=True)
            _write_transcript_fixture_xlsx(upload_dir / "export.xlsx", rows, ticker_counts_df=ticker_counts)
            db_path = str(tmp_path / f"test_{len(results)}.db")
            ingest_transcripts(screen_id=screen_id, upload_dir=str(upload_dir),
                                db_path=db_path, config_path=config_path)
            engine = create_engine(f"sqlite:///{db_path}")
            results.append(pd.read_sql_table(table_name("raw_data", screen_id), engine))

        pd.testing.assert_frame_equal(results[0], results[1])

    def test_membership_deduped_to_one_row_per_ticker(self, tmp_path):
        """A ticker mentioned in 5 detail rows produces exactly 1
        screen_membership row for it."""
        screen_id = "fake_transcripts"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_config(config_path, screen_id)
        rows = [
            _raw_row(Tickers="AAA", DocID=f"EC-{i}", **{"Transcript Date": f"August {i+1}, 2026"})
            for i in range(5)
        ]
        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        _write_transcript_fixture_xlsx(upload_dir / "export.xlsx", rows)

        db_path = str(tmp_path / "test.db")
        ingest_transcripts(screen_id=screen_id, upload_dir=str(upload_dir),
                            db_path=db_path, config_path=config_path)

        engine = create_engine(f"sqlite:///{db_path}")
        detail = pd.read_sql_table(table_name("raw_data", screen_id), engine)
        assert len(detail) == 5

        membership = pd.read_sql_table("screen_membership", engine)
        screen_rows = membership[membership["screen_id"] == screen_id]
        assert list(screen_rows["ticker"]) == ["AAA"]

    def test_membership_survives_a_disjoint_second_upload(self, tmp_path):
        """A ticker from the first upload stays in membership even though
        the second upload doesn't mention it again — membership is derived
        from the full accumulated corpus, not just the latest batch."""
        screen_id = "fake_transcripts"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_config(config_path, screen_id)
        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        db_path = str(tmp_path / "test.db")

        _write_transcript_fixture_xlsx(upload_dir / "export.xlsx", [_raw_row(Tickers="AAA", DocID="EC-1")])
        ingest_transcripts(screen_id=screen_id, upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)

        _write_transcript_fixture_xlsx(upload_dir / "export.xlsx", [_raw_row(Tickers="BBB", DocID="EC-2")])
        ingest_transcripts(screen_id=screen_id, upload_dir=str(upload_dir), db_path=db_path, config_path=config_path)

        engine = create_engine(f"sqlite:///{db_path}")
        membership = pd.read_sql_table("screen_membership", engine)
        screen_rows = membership[membership["screen_id"] == screen_id]
        assert set(screen_rows["ticker"]) == {"AAA", "BBB"}


# ---------------------------------------------------------------------------
# clean_and_explode_transcripts — the single public entry point used by
# both the real ingest and refresh.py's prepare replica.
# ---------------------------------------------------------------------------

class TestCleanAndExplodeTranscripts:
    def test_full_pipeline_shape(self):
        raw = _raw_df([_raw_row(Tickers="HON, GE", DocID="EC-172")])
        result = clean_and_explode_transcripts(raw)
        assert len(result) == 2
        assert list(result.columns) == [
            "doc_id", "doc_id_synthetic", "ticker", "transcript_date", "expert_position",
            "key_takeaways", "negativity_rationale", "theme", "transcript_company",
            "source_pack", "source_sector",
        ]


# ---------------------------------------------------------------------------
# Phase 6b: find_duplicate_transcript_groups (T37 duplicate detection)
# ---------------------------------------------------------------------------

def _detail_row(**overrides):
    row = {
        "doc_id": "EC-1", "doc_id_synthetic": 0, "ticker": "AAA",
        "transcript_date": "2026-08-01", "theme": "Some Theme",
    }
    row.update(overrides)
    return row


class TestFindDuplicateTranscriptGroups:
    def test_forces_a_flag_on_the_syn_backfill_shape(self):
        """Positive test (property 11a): a SYN- row and a later real-DocID
        row sharing (transcript_date, theme, ticker) is exactly the
        DocID-backfill hazard this function exists to catch."""
        df = pd.DataFrame([
            _detail_row(doc_id="SYN-abc123"),
            _detail_row(doc_id="EC-1000000-999"),
        ])
        result = find_duplicate_transcript_groups(df)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAA"
        assert result.iloc[0]["doc_id_count"] == 2
        assert result.iloc[0]["doc_ids"] == ["EC-1000000-999", "SYN-abc123"]

    def test_ticker_is_load_bearing_one_transcript_two_tickers_not_flagged(self):
        """Negative test (property 11b): the live EC-1000000-230444 shape
        — one doc_id shared across two exploded ticker rows, same
        (transcript_date, theme). Must NOT flag: keyed with ticker, this
        transcript becomes two separate one-row groups, each trivially a
        single doc_id."""
        df = pd.DataFrame([
            _detail_row(doc_id="EC-1000000-230444", ticker="AAA"),
            _detail_row(doc_id="EC-1000000-230444", ticker="BBB"),
        ])
        result = find_duplicate_transcript_groups(df)
        assert result.empty

    def test_ticker_is_load_bearing_same_file_sanity_check(self):
        """Same-file sanity check that ticker is the discriminating column,
        not decoration: two DIFFERENT transcripts (different doc_id, for
        different tickers) that happen to share (transcript_date, theme)
        by coincidence. Keyed WITH ticker (the real function), each
        ticker's own group has exactly one doc_id, so neither is flagged —
        correctly, since two unrelated documents sharing a date/theme
        isn't the DocID-backfill hazard. Dropping ticker from the key
        would collapse them into one group with 2 distinct doc_ids and
        wrongly flag a coincidence as a duplicate — this is the concrete
        failure a (transcript_date, theme)-only key would introduce."""
        df = pd.DataFrame([
            _detail_row(doc_id="EC-100", ticker="AAA"),
            _detail_row(doc_id="EC-200", ticker="BBB"),
        ])
        result = find_duplicate_transcript_groups(df)
        assert result.empty

        grouped_without_ticker = df.groupby(["transcript_date", "theme"])["doc_id"].nunique()
        assert (grouped_without_ticker > 1).any()

    def test_same_ticker_same_date_different_theme_not_flagged(self):
        """Property 11c: mirrors the six live same-day pairs' actual shape
        — two distinct transcripts, same ticker and date, different theme
        and doc_id. Must NOT flag."""
        df = pd.DataFrame([
            _detail_row(doc_id="EC-1", theme="Theme A"),
            _detail_row(doc_id="EC-2", theme="Theme B"),
        ])
        result = find_duplicate_transcript_groups(df)
        assert result.empty

    def test_clean_input_returns_empty_without_raising(self):
        """Property 11d: every group a singleton (the live corpus's actual
        shape) — empty result, no exception, and the empty frame still
        carries its columns."""
        df = pd.DataFrame([
            _detail_row(doc_id="EC-1", ticker="AAA"),
            _detail_row(doc_id="EC-2", ticker="BBB", transcript_date="2026-08-02"),
        ])
        result = find_duplicate_transcript_groups(df)
        assert result.empty
        assert list(result.columns) == [
            "transcript_date", "theme", "ticker", "doc_id_count", "doc_ids",
        ]


class TestIngestTranscriptsDuplicateLogging:
    """Property 12: the ingest path calls the detector on the ACCUMULATED
    frame (not just the incoming batch) and logs accordingly. tmp_path
    fixture DB only — no test here reads data/screener.db (T25)."""

    def test_logs_warning_when_a_docid_backfill_creates_a_duplicate(self, tmp_path, caplog):
        screen_id = "fake_transcripts"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_config(config_path, screen_id)
        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        db_path = str(tmp_path / "test.db")

        # First upload: DocID missing -> synthesized SYN- id.
        _write_transcript_fixture_xlsx(
            upload_dir / "export.xlsx",
            [_raw_row(Tickers="AAA", DocID=None, **{"Transcript Date": "August 1, 2026"})],
        )
        ingest_transcripts(screen_id=screen_id, upload_dir=str(upload_dir),
                            db_path=db_path, config_path=config_path)

        # Second upload: same (transcript_date, theme, ticker), but DocID
        # is now populated with a real, different id -> does not match the
        # stored SYN- row's (doc_id, ticker) upsert key, so it's inserted
        # IN ADDITION, exactly the hazard find_duplicate_transcript_groups
        # is meant to catch.
        with caplog.at_level(logging.WARNING, logger="src.transcript_ingest"):
            _write_transcript_fixture_xlsx(
                upload_dir / "export.xlsx",
                [_raw_row(Tickers="AAA", DocID="EC-1000000-999",
                          **{"Transcript Date": "August 1, 2026"})],
            )
            ingest_transcripts(screen_id=screen_id, upload_dir=str(upload_dir),
                                db_path=db_path, config_path=config_path)

        engine = create_engine(f"sqlite:///{db_path}")
        detail = pd.read_sql_table(table_name("raw_data", screen_id), engine)
        assert len(detail) == 2  # both rows present -> the duplication actually occurred

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("AAA" in r.getMessage() for r in warnings)

    def test_logs_info_when_no_duplicates_found(self, tmp_path, caplog):
        screen_id = "fake_transcripts"
        config_path = str(tmp_path / "config.yaml")
        _write_fake_config(config_path, screen_id)
        upload_dir = tmp_path / "uploads" / screen_id
        upload_dir.mkdir(parents=True)
        db_path = str(tmp_path / "test.db")

        with caplog.at_level(logging.INFO, logger="src.transcript_ingest"):
            _write_transcript_fixture_xlsx(
                upload_dir / "export.xlsx", [_raw_row(Tickers="AAA", DocID="EC-1")],
            )
            ingest_transcripts(screen_id=screen_id, upload_dir=str(upload_dir),
                                db_path=db_path, config_path=config_path)

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert any("No duplicate transcript groups detected" in r.getMessage() for r in infos)
