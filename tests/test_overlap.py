"""
Unit tests for src/overlap.py — the Phase 3d Part 1 cross-screen overlap
calculations.

Synthetic fixture: short_screen (the universe) plus 6 thematic/RSI
screens — the 5 real ones (cyclicals, competition, structural,
management_comp, rising_short_interest) plus one synthetic 6th
(fake_screen_g), the last one added specifically to prove the overlap
functions are generic over N screens with no code change (see
TestGenericityRegressionLock). Tickers are placed to exercise every
count/identity edge case called out in the Phase 3d Part 1 plan.
"""

import pandas as pd
import pytest

from src.overlap import (
    UNIVERSE_SCREEN_ID,
    build_presence_matrix,
    compute_overlap,
    screen_count_ceiling,
    style_overlap_table,
)

SCREENS_DF = pd.DataFrame([
    {"screen_id": "short_screen", "display_name": "OWS Short Screen",
     "screen_type": "quant_composite", "has_scoring": True},
    {"screen_id": "competition", "display_name": "Competition",
     "screen_type": "curated", "has_scoring": False},
    {"screen_id": "cyclicals", "display_name": "Cyclicals",
     "screen_type": "curated", "has_scoring": False},
    {"screen_id": "management_comp", "display_name": "Management Comp",
     "screen_type": "curated", "has_scoring": False},
    {"screen_id": "structural", "display_name": "Structural",
     "screen_type": "curated", "has_scoring": False},
    {"screen_id": "rising_short_interest", "display_name": "Rising Short Interest",
     "screen_type": "quant_composite", "has_scoring": False},
    {"screen_id": "fake_screen_g", "display_name": "Fake Screen G",
     "screen_type": "curated", "has_scoring": False},
])

# 6 thematic/RSI screens (everything except short_screen).
THEMATIC_IDS = [sid for sid in SCREENS_DF["screen_id"] if sid != UNIVERSE_SCREEN_ID]


def _screens_df_without(screen_id: str) -> pd.DataFrame:
    return SCREENS_DF[SCREENS_DF["screen_id"] != screen_id].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fixture: tickers covering every count/identity case
#
#   AAAA - on 4 thematic screens (competition, management_comp, structural,
#          fake_screen_g), present in short_screen -> in_universe=True.
#   BBBB - on exactly 1 thematic screen (cyclicals), present in short_screen.
#   CCCC - on 0 thematic screens; short_screen-only.
#   DDDD - on 2 thematic screens (rising_short_interest, structural), NOT
#          in short_screen -> in_universe=False, overall_score NaN. RSI
#          sorts before structural alphabetically among thematic ids, so
#          this ticker proves per-FIELD fallback: sector must resolve from
#          structural (RSI has no sector column) even though RSI is tried
#          first in source order.
#   EEEE - on competition (market_cap NaN) AND cyclicals (market_cap valid).
#          competition sorts before cyclicals, so this proves the resolver
#          actually falls through past a null value to the next source,
#          rather than just picking whichever source has the ticker.
#   FFFF - on structural only, with market_cap NaN and no other source at
#          all. Defensive case: no real table in this codebase leaves
#          market_cap null (confirmed 0 nulls across all six live tables),
#          but the resolver must not crash if one ever did.
# ---------------------------------------------------------------------------


def _base_membership_and_data():
    membership_rows = [
        {"screen_id": "short_screen", "ticker": "AAAA"},
        {"screen_id": "competition", "ticker": "AAAA"},
        {"screen_id": "management_comp", "ticker": "AAAA"},
        {"screen_id": "structural", "ticker": "AAAA"},
        {"screen_id": "fake_screen_g", "ticker": "AAAA"},

        {"screen_id": "short_screen", "ticker": "BBBB"},
        {"screen_id": "cyclicals", "ticker": "BBBB"},

        {"screen_id": "short_screen", "ticker": "CCCC"},

        {"screen_id": "rising_short_interest", "ticker": "DDDD"},
        {"screen_id": "structural", "ticker": "DDDD"},

        {"screen_id": "competition", "ticker": "EEEE"},
        {"screen_id": "cyclicals", "ticker": "EEEE"},

        {"screen_id": "structural", "ticker": "FFFF"},
    ]
    membership_df = pd.DataFrame(membership_rows)

    screen_data = {
        "short_screen": pd.DataFrame([
            {"ticker": "AAAA", "name": "Alpha Co", "sector": "Industrials",
             "market_cap": 5000.0, "overall_score": 3.805},
            {"ticker": "BBBB", "name": "Bravo Co", "sector": "Tech",
             "market_cap": 2000.0, "overall_score": 4.5},
            {"ticker": "CCCC", "name": "Charlie Co", "sector": "Energy",
             "market_cap": 1000.0, "overall_score": 2.1},
        ]),
        "competition": pd.DataFrame([
            {"ticker": "AAAA", "name": "Alpha Co", "sector": "Industrials", "market_cap": 5000.0},
            {"ticker": "EEEE", "name": "Echo Co", "sector": "Healthcare", "market_cap": float("nan")},
        ]),
        "cyclicals": pd.DataFrame([
            {"ticker": "BBBB", "name": "Bravo Co", "sector": "Tech", "market_cap": 2000.0},
            {"ticker": "EEEE", "name": "Echo Co", "sector": "Healthcare", "market_cap": 750.0},
        ]),
        "management_comp": pd.DataFrame([
            {"ticker": "AAAA", "name": "Alpha Co", "sector": "Industrials", "market_cap": 5000.0},
        ]),
        "structural": pd.DataFrame([
            {"ticker": "AAAA", "name": "Alpha Co", "sector": "Industrials", "market_cap": 5000.0},
            {"ticker": "DDDD", "name": "Delta Co", "sector": "Materials", "market_cap": 800.0},
            {"ticker": "FFFF", "name": "Foxtrot Co", "sector": "Utilities", "market_cap": float("nan")},
        ]),
        # RSI shape: no sector column at all.
        "rising_short_interest": pd.DataFrame([
            {"ticker": "DDDD", "name": "Delta Co", "market_cap": 810.0},
        ]),
        "fake_screen_g": pd.DataFrame([
            {"ticker": "AAAA", "name": "Alpha Co", "sector": "Industrials", "market_cap": 5000.0},
        ]),
    }
    return membership_df, screen_data


# ---------------------------------------------------------------------------
# compute_overlap
# ---------------------------------------------------------------------------

class TestComputeOverlap:
    def test_four_screen_ticker_in_universe(self):
        membership_df, screen_data = _base_membership_and_data()
        result = compute_overlap(membership_df, SCREENS_DF, screen_data)
        row = result.set_index("ticker").loc["AAAA"]
        assert row["screen_count"] == 4
        assert row["screens_on"] == "Competition, Fake Screen G, Management Comp, Structural"
        assert bool(row["in_universe"]) is True
        assert row["overall_score"] == pytest.approx(3.805)

    def test_one_screen_ticker(self):
        membership_df, screen_data = _base_membership_and_data()
        result = compute_overlap(membership_df, SCREENS_DF, screen_data)
        row = result.set_index("ticker").loc["BBBB"]
        assert row["screen_count"] == 1
        assert row["screens_on"] == "Cyclicals"
        assert bool(row["in_universe"]) is True

    def test_short_screen_only_ticker_has_zero_count_and_is_still_present(self):
        membership_df, screen_data = _base_membership_and_data()
        result = compute_overlap(membership_df, SCREENS_DF, screen_data)
        assert "CCCC" in set(result["ticker"])
        row = result.set_index("ticker").loc["CCCC"]
        assert row["screen_count"] == 0
        assert row["screens_on"] == ""
        assert bool(row["in_universe"]) is True

    def test_not_in_universe_ticker_has_nan_score_not_dropped(self):
        membership_df, screen_data = _base_membership_and_data()
        result = compute_overlap(membership_df, SCREENS_DF, screen_data)
        row = result.set_index("ticker").loc["DDDD"]
        assert row["screen_count"] == 2
        assert bool(row["in_universe"]) is False
        assert pd.isna(row["overall_score"])

    def test_field_level_fallback_resolves_sector_from_curated_not_rsi(self):
        """DDDD sits on rising_short_interest (no sector column) and
        structural (has sector). RSI sorts before structural alphabetically
        among thematic screen_ids, so this proves resolution happens per
        FIELD, not per row/source: sector must come from structural even
        though RSI is tried first in the deterministic source order."""
        membership_df, screen_data = _base_membership_and_data()
        result = compute_overlap(membership_df, SCREENS_DF, screen_data)
        row = result.set_index("ticker").loc["DDDD"]
        assert row["sector"] == "Materials"

    def test_field_level_fallback_skips_null_and_uses_next_source(self):
        """EEEE is on competition (market_cap NaN) and cyclicals (market_cap
        750.0). competition sorts before cyclicals, so this proves the
        resolver actually falls through past a null value rather than just
        taking whichever source has the ticker first."""
        membership_df, screen_data = _base_membership_and_data()
        result = compute_overlap(membership_df, SCREENS_DF, screen_data)
        row = result.set_index("ticker").loc["EEEE"]
        assert row["market_cap"] == pytest.approx(750.0)

    def test_field_level_fallback_returns_none_when_all_sources_null(self):
        """Defensive case, not a live one: no real table in this codebase
        leaves market_cap null today (confirmed 0 nulls across all six
        live tables), but the resolver must degrade to None, not crash,
        if a future source ever did."""
        membership_df, screen_data = _base_membership_and_data()
        result = compute_overlap(membership_df, SCREENS_DF, screen_data)
        row = result.set_index("ticker").loc["FFFF"]
        assert row["market_cap"] is None or pd.isna(row["market_cap"])

    def test_rsi_only_ticker_has_null_sector_not_a_crash(self):
        membership_df, screen_data = _base_membership_and_data()
        # DDDD's only source with a sector value is structural; verify a
        # ticker with genuinely no sector anywhere degrades to null.
        screen_data["structural"] = screen_data["structural"][
            screen_data["structural"]["ticker"] != "DDDD"
        ]
        result = compute_overlap(membership_df, SCREENS_DF, screen_data)
        row = result.set_index("ticker").loc["DDDD"]
        assert row["sector"] is None or pd.isna(row["sector"])


# ---------------------------------------------------------------------------
# build_presence_matrix
# ---------------------------------------------------------------------------

class TestBuildPresenceMatrix:
    def test_presence_values_and_columns(self):
        membership_df, screen_data = _base_membership_and_data()
        overlap_df = compute_overlap(membership_df, SCREENS_DF, screen_data)
        matrix = build_presence_matrix(membership_df, SCREENS_DF, overlap_df)

        expected_thematic_display_names = {
            "Competition", "Cyclicals", "Management Comp", "Structural",
            "Rising Short Interest", "Fake Screen G",
        }
        assert expected_thematic_display_names.issubset(set(matrix.columns))
        assert "OWS Short Screen" not in matrix.columns  # universe excluded

        row = matrix.set_index("ticker").loc["AAAA"]
        assert row["Competition"] == 1
        assert row["Cyclicals"] == 0
        assert bool(row["in_universe"]) is True

    def test_not_in_universe_ticker_marked_false_in_matrix(self):
        membership_df, screen_data = _base_membership_and_data()
        overlap_df = compute_overlap(membership_df, SCREENS_DF, screen_data)
        matrix = build_presence_matrix(membership_df, SCREENS_DF, overlap_df)
        row = matrix.set_index("ticker").loc["DDDD"]
        assert bool(row["in_universe"]) is False


# ---------------------------------------------------------------------------
# Genericity regression lock
# ---------------------------------------------------------------------------

class TestGenericityRegressionLock:
    def test_adding_a_seventh_screen_requires_no_code_change(self):
        """Same compute_overlap()/build_presence_matrix() functions,
        unchanged, run against two fixtures differing only in whether the
        synthetic fake_screen_g screen is present. Proves the increment in
        AAAA's screen_count, and fake_screen_g's appearance as a matrix
        column, come from data alone."""
        membership_df, screen_data = _base_membership_and_data()

        # Baseline: fake_screen_g removed from both registry and data/membership.
        baseline_screens_df = _screens_df_without("fake_screen_g")
        baseline_membership = membership_df[membership_df["screen_id"] != "fake_screen_g"]
        baseline_data = {k: v for k, v in screen_data.items() if k != "fake_screen_g"}

        baseline_overlap = compute_overlap(baseline_membership, baseline_screens_df, baseline_data)
        extended_overlap = compute_overlap(membership_df, SCREENS_DF, screen_data)

        baseline_count = baseline_overlap.set_index("ticker").loc["AAAA", "screen_count"]
        extended_count = extended_overlap.set_index("ticker").loc["AAAA", "screen_count"]
        assert extended_count == baseline_count + 1

        baseline_matrix = build_presence_matrix(baseline_membership, baseline_screens_df, baseline_overlap)
        extended_matrix = build_presence_matrix(membership_df, SCREENS_DF, extended_overlap)

        assert "Fake Screen G" not in baseline_matrix.columns
        assert "Fake Screen G" in extended_matrix.columns
        assert extended_matrix.set_index("ticker").loc["AAAA", "Fake Screen G"] == 1


# ---------------------------------------------------------------------------
# screen_count_ceiling
# ---------------------------------------------------------------------------

class TestScreenCountCeiling:
    def test_returns_max_of_given_frame(self):
        """The function computes from whatever frame it's given — the
        guarantee that the slider's bound doesn't drift under an unrelated
        filter comes from render_overlap_view always calling this with the
        UNFILTERED overlap_df, not from this function refusing filtered
        input. This test documents that contract, not a caller behavior."""
        membership_df, screen_data = _base_membership_and_data()
        overlap_df = compute_overlap(membership_df, SCREENS_DF, screen_data)
        assert screen_count_ceiling(overlap_df) == 4

        filtered_subset = overlap_df[overlap_df["screen_count"] <= 2]
        assert screen_count_ceiling(filtered_subset) == 2

    def test_degenerate_all_zero_returns_one(self):
        degenerate = pd.DataFrame({"screen_count": [0, 0, 0]})
        assert screen_count_ceiling(degenerate) == 1

    def test_empty_frame_returns_one(self):
        empty = pd.DataFrame({"screen_count": []})
        assert screen_count_ceiling(empty) == 1


# ---------------------------------------------------------------------------
# style_overlap_table
#
# NOTE: this is the first test in the suite to exercise pandas' .style
# accessor at all. .style requires jinja2>=3.1.2 (pandas raises
# AttributeError below that version); requirements.txt pins jinja2
# explicitly for exactly this reason (altair pulls it in transitively with
# no floor). This test doubles as the regression lock for that pin — if it
# is ever "simplified" away, this test starts failing on a machine with an
# older jinja2 already installed, even though every other test still
# passes.
# ---------------------------------------------------------------------------

class TestStyleOverlapTable:
    @staticmethod
    def _sample_display_df():
        return pd.DataFrame([
            {"ticker": "AAAA", "name": "Alpha Co", "sector": "Industrials",
             "market_cap": 5000.0, "screen_count": 4, "screens_on": "Competition, Structural",
             "overall_score": 3.805},
            {"ticker": "DDDD", "name": "Delta Co", "sector": float("nan"),
             "market_cap": 800.0, "screen_count": 2, "screens_on": "Structural",
             "overall_score": float("nan")},
        ])

    def test_null_sector_renders_em_dash_not_universe_label(self):
        html = style_overlap_table(self._sample_display_df()).to_html()
        # DDDD's row: sector cell shows the em-dash placeholder...
        assert "—" in html
        # ...and does NOT render the overall_score column's label — this is
        # the exact defect an unscoped na_rep would introduce.
        sector_cell_count = html.count("Not in short_screen universe")
        assert sector_cell_count == 1  # appears exactly once, in overall_score's cell only

    def test_null_overall_score_renders_universe_label(self):
        html = style_overlap_table(self._sample_display_df()).to_html()
        assert "Not in short_screen universe" in html

    def test_market_cap_is_dollar_formatted(self):
        html = style_overlap_table(self._sample_display_df()).to_html()
        assert "$5,000" in html
        assert "$800" in html
