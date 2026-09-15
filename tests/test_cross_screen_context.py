"""Unit tests for src/cross_screen_context.py (Phase 5b-2)."""

import math

import pandas as pd

from src.cross_screen_context import (
    build_also_appears_on,
    build_screen_contribution,
    classify_screen,
    is_transcript_shaped,
    other_screen_ids_for_ticker,
)

SCREENS_DF = pd.DataFrame(
    {
        "screen_id": ["short_screen", "structural", "rising_short_interest", "mystery"],
        "display_name": ["OWS Short Screen", "Structural", "Rising Short Interest", "Mystery"],
        "screen_type": ["quant_composite", "curated", "quant_composite", "curated"],
        "has_scoring": [True, False, False, False],
    }
)


class TestOtherScreenIdsForTicker:
    def test_excludes_current_screen(self):
        membership_df = pd.DataFrame(
            {
                "screen_id": ["short_screen", "structural", "rising_short_interest"],
                "ticker": ["AAPL", "AAPL", "AAPL"],
            }
        )
        result = other_screen_ids_for_ticker("AAPL", "structural", membership_df)
        assert result == ["rising_short_interest", "short_screen"]

    def test_ticker_on_no_other_screen_returns_empty(self):
        membership_df = pd.DataFrame({"screen_id": ["structural"], "ticker": ["AAPL"]})
        assert other_screen_ids_for_ticker("AAPL", "structural", membership_df) == []

    def test_ticker_absent_entirely_returns_empty(self):
        membership_df = pd.DataFrame({"screen_id": ["structural"], "ticker": ["AAPL"]})
        assert other_screen_ids_for_ticker("ZZZZ", "structural", membership_df) == []


class TestClassifyScreen:
    def test_universe(self):
        assert classify_screen("short_screen", SCREENS_DF) == "universe"

    def test_curated(self):
        assert classify_screen("structural", SCREENS_DF) == "curated"

    def test_scored_not_confused_with_unscored(self):
        """Compound-condition regression lock: a quant_composite screen WITH
        scoring must classify as 'scored', not 'unscored' — has_scoring is
        the discriminating half of the condition, not screen_type alone."""
        scored_screens = pd.DataFrame(
            {
                "screen_id": ["second_scored"],
                "display_name": ["Second Scored"],
                "screen_type": ["quant_composite"],
                "has_scoring": [True],
            }
        )
        assert classify_screen("second_scored", scored_screens) == "scored"

    def test_unscored_not_confused_with_scored(self):
        assert classify_screen("rising_short_interest", SCREENS_DF) == "unscored"

    def test_unknown_screen_id_not_in_registry(self):
        assert classify_screen("does_not_exist", SCREENS_DF) == "unknown"

    def test_unknown_unrecognized_screen_type(self):
        weird_screens = pd.DataFrame(
            {
                "screen_id": ["odd"],
                "display_name": ["Odd"],
                "screen_type": ["something_else"],
                "has_scoring": [False],
            }
        )
        assert classify_screen("odd", weird_screens) == "unknown"


class TestBuildScreenContribution:
    def test_universe_kind(self):
        screen_data = {"short_screen": pd.DataFrame({"ticker": ["AAPL"], "overall_score": [3.5]})}
        result = build_screen_contribution("short_screen", "AAPL", SCREENS_DF, screen_data)
        assert result == {
            "screen_id": "short_screen",
            "display_name": "OWS Short Screen",
            "kind": "universe",
            "overall_score": 3.5,
        }

    def test_universe_kind_nan_score_not_crashed(self):
        screen_data = {
            "short_screen": pd.DataFrame({"ticker": ["AAPL"], "overall_score": [float("nan")]})
        }
        result = build_screen_contribution("short_screen", "AAPL", SCREENS_DF, screen_data)
        assert math.isnan(result["overall_score"])

    def test_curated_kind(self):
        screen_data = {
            "structural": pd.DataFrame(
                {
                    "ticker": ["AAPL"],
                    "rationale": ["Some rationale"],
                    "stock_performance": [0.05],
                }
            )
        }
        result = build_screen_contribution("structural", "AAPL", SCREENS_DF, screen_data)
        assert result == {
            "screen_id": "structural",
            "display_name": "Structural",
            "kind": "curated",
            "rationale": "Some rationale",
            "stock_performance": 0.05,
        }

    def test_curated_kind_null_rationale_not_crashed(self):
        screen_data = {
            "structural": pd.DataFrame(
                {"ticker": ["AAPL"], "rationale": [None], "stock_performance": [float("nan")]}
            )
        }
        result = build_screen_contribution("structural", "AAPL", SCREENS_DF, screen_data)
        assert result["rationale"] is None
        assert math.isnan(result["stock_performance"])

    def test_unscored_kind(self):
        screen_data = {
            "rising_short_interest": pd.DataFrame(
                {"ticker": ["AAPL"], "short_interest_pct": [0.12], "adv": [50.0]}
            )
        }
        result = build_screen_contribution(
            "rising_short_interest", "AAPL", SCREENS_DF, screen_data
        )
        assert result["kind"] == "unscored"
        assert result["metrics"] == {"adv": 50.0, "short_interest_pct": 0.12}

    def test_none_when_screen_missing_from_screen_data(self):
        result = build_screen_contribution("structural", "AAPL", SCREENS_DF, {})
        assert result is None

    def test_none_when_ticker_not_in_that_screens_table(self):
        screen_data = {"structural": pd.DataFrame({"ticker": ["MSFT"], "rationale": ["x"]})}
        result = build_screen_contribution("structural", "AAPL", SCREENS_DF, screen_data)
        assert result is None

    def test_none_for_unknown_kind(self):
        screen_data = {"mystery": pd.DataFrame({"ticker": ["AAPL"], "rationale": ["x"]})}
        # "mystery" is screen_type "curated" in SCREENS_DF, so make an
        # unrelated registry where it's unrecognized instead.
        weird_screens = pd.DataFrame(
            {
                "screen_id": ["mystery"],
                "display_name": ["Mystery"],
                "screen_type": ["something_else"],
                "has_scoring": [False],
            }
        )
        result = build_screen_contribution("mystery", "AAPL", weird_screens, screen_data)
        assert result is None


class TestBuildAlsoAppearsOn:
    def test_end_to_end_sorted_by_display_name(self):
        membership_df = pd.DataFrame(
            {
                "screen_id": ["short_screen", "structural", "rising_short_interest"],
                "ticker": ["AAPL", "AAPL", "AAPL"],
            }
        )
        screen_data = {
            "short_screen": pd.DataFrame({"ticker": ["AAPL"], "overall_score": [3.5]}),
            "structural": pd.DataFrame(
                {"ticker": ["AAPL"], "rationale": ["r"], "stock_performance": [0.1]}
            ),
            "rising_short_interest": pd.DataFrame(
                {"ticker": ["AAPL"], "short_interest_pct": [0.2]}
            ),
        }
        result = build_also_appears_on(
            "AAPL", "some_other_screen", membership_df, SCREENS_DF, screen_data
        )
        # Sorted by display_name: "OWS Short Screen" < "Rising Short
        # Interest" < "Structural"
        assert [c["display_name"] for c in result] == [
            "OWS Short Screen",
            "Rising Short Interest",
            "Structural",
        ]

    def test_excludes_current_screen(self):
        membership_df = pd.DataFrame(
            {"screen_id": ["structural"], "ticker": ["AAPL"]}
        )
        screen_data = {
            "structural": pd.DataFrame(
                {"ticker": ["AAPL"], "rationale": ["r"], "stock_performance": [0.1]}
            )
        }
        result = build_also_appears_on("AAPL", "structural", membership_df, SCREENS_DF, screen_data)
        assert result == []

    def test_empty_when_ticker_on_no_other_screen(self):
        membership_df = pd.DataFrame({"screen_id": ["short_screen"], "ticker": ["AAPL"]})
        screen_data = {"short_screen": pd.DataFrame({"ticker": ["AAPL"], "overall_score": [3.5]})}
        result = build_also_appears_on(
            "AAPL", "short_screen", membership_df, SCREENS_DF, screen_data
        )
        assert result == []


class TestIsTranscriptShaped:
    """Phase 8b's T40 gate — on the column set, never a screen_id literal."""

    def test_none_is_not_shaped(self):
        assert is_transcript_shaped(None) is False

    def test_transcript_shaped_frame_passes(self):
        df = pd.DataFrame({
            "ticker": ["AAA"], "transcript_date": ["2026-01-01"],
            "doc_id": ["X"], "key_takeaways": ["t"], "theme": ["m"],
        })
        assert is_transcript_shaped(df) is True

    def test_rsi_shaped_frame_fails(self):
        """T40's recorded failure verbatim: an RSI/short_screen-shaped
        frame must never be treated as transcript-shaped."""
        rsi_df = pd.DataFrame({
            "ticker": ["AAA"], "market_cap": [500.0], "adv": [5.0],
            "short_interest_pct": [0.1],
        })
        assert is_transcript_shaped(rsi_df) is False

    def test_missing_one_required_column_fails(self):
        """Partial overlap (all but doc_id) must still fail — the gate is
        the FULL required set, not "looks plausible"."""
        df = pd.DataFrame({
            "ticker": ["AAA"], "transcript_date": ["2026-01-01"],
            "key_takeaways": ["t"], "theme": ["m"],
        })
        assert is_transcript_shaped(df) is False


_TRANSCRIPTS_SCREENS_DF = pd.DataFrame({
    "screen_id": ["negative_expert_transcripts"],
    "display_name": ["Negative Expert Transcripts"],
    "screen_type": ["quant_composite"],
    "has_scoring": [False],
})


class TestBuildScreenContributionTranscriptTakeaways:
    """Phase 8b: the pre-resolved {screen_id: df} transcript_takeaways
    dict, threaded through build_screen_contribution's "unscored" branch.
    Purely synthetic (T25) — the gate/filter/sort themselves are app.py's
    job (resolve_transcript_takeaways_for_ticker), never this module's."""

    def _screen_data(self):
        return {
            "negative_expert_transcripts": pd.DataFrame({
                "ticker": ["AAPL"], "mention_count": [3],
                "latest_transcript_date": ["2026-08-06"], "days_since_latest": [5],
            })
        }

    def test_multi_row_takeaways_pass_through_unmodified(self):
        """The iloc[0] trap this phase exists to avoid: 3 rows in must be
        3 rows out, not silently collapsed to the first. FAIL-FIRST: this
        assertion goes red if build_screen_contribution's "unscored" branch
        slices its takeaways_df to .iloc[[0]] before attaching it (verified
        by hand during the build, not left in the source)."""
        takeaways_df = pd.DataFrame({
            "transcript_date": ["2026-08-06", "2026-07-01", "2026-06-01"],
            "doc_id": ["C", "B", "A"],
            "theme": ["x", "y", "z"],
            "key_takeaways": ["t1", "t2", "t3"],
        })
        result = build_screen_contribution(
            "negative_expert_transcripts", "AAPL", _TRANSCRIPTS_SCREENS_DF, self._screen_data(),
            transcript_takeaways={"negative_expert_transcripts": takeaways_df},
        )
        assert len(result["takeaways"]) == 3
        assert list(result["takeaways"]["doc_id"]) == ["C", "B", "A"]

    def test_no_takeaways_key_when_argument_omitted(self):
        """No existing caller's output changes by omitting the new
        argument — the same property the 6c resolver precedent set."""
        result = build_screen_contribution(
            "negative_expert_transcripts", "AAPL", _TRANSCRIPTS_SCREENS_DF, self._screen_data(),
        )
        assert "takeaways" not in result

    def test_no_takeaways_key_when_dict_has_no_entry_for_this_screen(self):
        result = build_screen_contribution(
            "negative_expert_transcripts", "AAPL", _TRANSCRIPTS_SCREENS_DF, self._screen_data(),
            transcript_takeaways={"some_other_screen": pd.DataFrame({"doc_id": ["A"]})},
        )
        assert "takeaways" not in result

    def test_no_takeaways_key_when_entry_is_empty_frame(self):
        """Synthetic-only (PM-confirmed): on the live DB every screen the
        transcripts screen contributes to has at least one takeaway for
        that ticker (the aggregate/detail/membership ticker sets are
        identical), so this branch cannot fire against real data today —
        covered here so the degrade path is still locked."""
        result = build_screen_contribution(
            "negative_expert_transcripts", "AAPL", _TRANSCRIPTS_SCREENS_DF, self._screen_data(),
            transcript_takeaways={"negative_expert_transcripts": pd.DataFrame()},
        )
        assert "takeaways" not in result

    def test_non_unscored_kind_ignores_transcript_takeaways(self):
        """Only the "unscored" branch attaches takeaways — a curated
        contribution must not pick one up even if handed one."""
        screen_data = {
            "structural": pd.DataFrame(
                {"ticker": ["AAPL"], "rationale": ["r"], "stock_performance": [0.1]}
            )
        }
        result = build_screen_contribution(
            "structural", "AAPL", SCREENS_DF, screen_data,
            transcript_takeaways={"structural": pd.DataFrame({"doc_id": ["A"]})},
        )
        assert "takeaways" not in result
