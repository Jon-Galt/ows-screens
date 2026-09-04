"""Unit tests for src/cross_screen_context.py (Phase 5b-2)."""

import math

import pandas as pd

from src.cross_screen_context import (
    build_also_appears_on,
    build_screen_contribution,
    classify_screen,
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
