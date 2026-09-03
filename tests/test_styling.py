"""
Unit tests for src/styling.py — Phase 5a's conditional-formatting colour
scale for the scored short_screen main table.
"""

import pandas as pd

from src.styling import (
    FACTOR_ANCHORS,
    MSCORE_FLAG_COLOR,
    MSCORE_NO_FLAG_COLOR,
    OVERALL_SCORE_ANCHORS,
    build_color_scale_domain,
    factor_color,
    interpolate_color,
    mscore_flag_color,
    overall_score_color,
    style_scored_table,
)

FACTOR_LO, FACTOR_MID, FACTOR_HI = FACTOR_ANCHORS
SCORE_LO, SCORE_MID, SCORE_HI = OVERALL_SCORE_ANCHORS


# ---------------------------------------------------------------------------
# build_color_scale_domain
# ---------------------------------------------------------------------------


class TestBuildColorScaleDomain:
    def test_domain_uses_min_median_max(self):
        df = pd.DataFrame({"factor_a": [0.0, 0.5, 0.5, 1.0]})
        domain = build_color_scale_domain(df, ["factor_a"])
        lo, mid, hi = domain["factor_a"]
        assert lo == 0.0
        assert mid == 0.5
        assert hi == 1.0

    def test_median_diverges_from_arithmetic_midpoint(self):
        """The T2/ratings_factor shape: a skewed column where the median is
        not the midpoint of min and max."""
        df = pd.DataFrame({"ratings_factor": [0.0] * 5 + [0.5] * 13 + [0.971726]})
        domain = build_color_scale_domain(df, ["ratings_factor"])
        lo, mid, hi = domain["ratings_factor"]
        assert lo == 0.0
        assert mid == 0.5  # NOT (0.0 + 0.971726) / 2 == 0.485863
        assert hi == 0.971726

    def test_nan_excluded_from_domain(self):
        df = pd.DataFrame({"factor_a": [0.0, float("nan"), 1.0]})
        domain = build_color_scale_domain(df, ["factor_a"])
        lo, mid, hi = domain["factor_a"]
        assert lo == 0.0
        assert hi == 1.0


# ---------------------------------------------------------------------------
# interpolate_color — happy path, anchors, boundary
# ---------------------------------------------------------------------------


class TestInterpolateColorHappyPath:
    def test_min_value_is_lo_hex(self):
        assert interpolate_color(0.0, 0.0, 0.5, 1.0, "#63BE7B", "#FCFCFF", "#F8696B") == "#63BE7B"

    def test_median_value_is_mid_hex(self):
        assert interpolate_color(0.5, 0.0, 0.5, 1.0, "#63BE7B", "#FCFCFF", "#F8696B") == "#FCFCFF"

    def test_max_value_is_hi_hex(self):
        assert interpolate_color(1.0, 0.0, 0.5, 1.0, "#63BE7B", "#FCFCFF", "#F8696B") == "#F8696B"

    def test_quarter_point_interpolates_lower_half(self):
        # value=0.25 is halfway between lo (0.0) and mid (0.5) -> halfway
        # between lo_hex and mid_hex.
        result = interpolate_color(0.25, 0.0, 0.5, 1.0, "#000000", "#FFFFFF", "#FF0000")
        assert result == "#808080"  # halfway from black to white

    def test_three_quarter_point_interpolates_upper_half(self):
        # value=0.75 is halfway between mid (0.5) and hi (1.0).
        result = interpolate_color(0.75, 0.0, 0.5, 1.0, "#000000", "#FFFFFF", "#000000")
        assert result == "#808080"  # halfway from white back to black


class TestInterpolateColorNaN:
    def test_nan_returns_none(self):
        assert interpolate_color(float("nan"), 0.0, 0.5, 1.0, "#000000", "#FFFFFF", "#FF0000") is None


# ---------------------------------------------------------------------------
# T2 regression: own-column-max anchoring, not a hardcoded 0..1 ceiling
# ---------------------------------------------------------------------------


class TestOwnColumnMaxAnchoring:
    def test_ratings_factor_max_is_full_red_not_97_percent(self):
        """ratings_factor's real max is 0.971726, not 1.0. A cell at that
        value must resolve to the exact hi_hex anchor (full red) — a
        hardcoded 0..1 domain would instead land at ~97% of the way there,
        a visibly lighter red."""
        result = interpolate_color(
            0.971726, 0.0, 0.5, 0.971726, *FACTOR_ANCHORS
        )
        assert result == FACTOR_HI  # "#FF7376", exact, not a lighter shade


# ---------------------------------------------------------------------------
# C6 — degenerate-domain handling
# ---------------------------------------------------------------------------


class TestDegenerateDomainMidEqualsLo:
    """def_rev_factor / liquidity_risk_factor's real shape: lo == mid ==
    0.0, hi == 1.0, with the overwhelming majority of rows at that shared
    lo/mid value."""

    LO, MID, HI = 0.0, 0.0, 1.0

    def test_rows_at_shared_lo_mid_get_mid_hex_not_a_crash(self):
        result = interpolate_color(0.0, self.LO, self.MID, self.HI, *FACTOR_ANCHORS)
        assert result == FACTOR_MID  # "#FCFCFF" — no ZeroDivisionError

    def test_upper_half_still_interpolates_correctly(self):
        # value at the midpoint of (mid, hi) = 0.5 -> halfway between
        # mid_hex and hi_hex, computed against a literal expected hex, not
        # against interpolate_color's own output.
        result = interpolate_color(0.5, self.LO, self.MID, self.HI, "#000000", "#FFFFFF", "#0000FF")
        assert result == "#8080FF"  # halfway from white (255,255,255) to blue (0,0,255)

    def test_upper_half_max_is_hi_hex(self):
        result = interpolate_color(1.0, self.LO, self.MID, self.HI, *FACTOR_ANCHORS)
        assert result == FACTOR_HI


class TestDegenerateDomainMidEqualsHi:
    """Mirror case: mid == hi. No value can exceed mid, so every row falls
    into the value <= mid branch; the row at hi == mid must still land
    exactly on mid_hex, via the branch's own arithmetic (frac == 1.0), not
    a dedicated guard."""

    LO, MID, HI = 0.0, 1.0, 1.0

    def test_max_value_gets_mid_hex(self):
        result = interpolate_color(1.0, self.LO, self.MID, self.HI, *FACTOR_ANCHORS)
        assert result == FACTOR_MID  # "#FCFCFF"

    def test_lower_half_still_interpolates_correctly(self):
        # value at the midpoint of (lo, mid) = 0.5 -> halfway between
        # lo_hex and mid_hex, against a literal expected hex.
        result = interpolate_color(0.5, self.LO, self.MID, self.HI, "#000000", "#FFFFFF", "#0000FF")
        assert result == "#808080"  # halfway from black to white

    def test_min_value_gets_lo_hex(self):
        result = interpolate_color(0.0, self.LO, self.MID, self.HI, *FACTOR_ANCHORS)
        assert result == FACTOR_LO


class TestDegenerateDomainConstantColumn:
    """lo == hi (== mid, necessarily): every non-null value is mid_hex."""

    def test_constant_value_gets_mid_hex(self):
        result = interpolate_color(3.0, 3.0, 3.0, 3.0, *OVERALL_SCORE_ANCHORS)
        assert result == SCORE_MID

    def test_nan_in_constant_column_still_none(self):
        result = interpolate_color(float("nan"), 3.0, 3.0, 3.0, *OVERALL_SCORE_ANCHORS)
        assert result is None


# ---------------------------------------------------------------------------
# overall_score_color / factor_color — the anchor sets actually used
# ---------------------------------------------------------------------------


class TestAnchorSets:
    def test_overall_score_uses_its_own_anchors(self):
        assert overall_score_color(1.665496, (1.665496, 3.207357, 4.887982)) == SCORE_LO
        assert overall_score_color(3.207357, (1.665496, 3.207357, 4.887982)) == SCORE_MID
        assert overall_score_color(4.887982, (1.665496, 3.207357, 4.887982)) == SCORE_HI

    def test_factor_uses_its_own_lighter_anchors(self):
        assert factor_color(0.0, (0.0, 0.5, 1.0)) == FACTOR_LO
        assert factor_color(1.0, (0.0, 0.5, 1.0)) == FACTOR_HI
        assert FACTOR_LO != SCORE_LO  # factor green is lighter than overall_score green
        assert FACTOR_HI != SCORE_HI  # factor red is lighter than overall_score red


# ---------------------------------------------------------------------------
# mscore_flag_color — solid fill, not a scale
# ---------------------------------------------------------------------------


class TestMscoreFlagColor:
    def test_flagged_is_red(self):
        assert mscore_flag_color(True) == MSCORE_FLAG_COLOR

    def test_not_flagged_is_green(self):
        assert mscore_flag_color(False) == MSCORE_NO_FLAG_COLOR

    def test_int_zero_one_from_sqlite_handled(self):
        """mscore_flag round-trips through SQLite as int64 (0/1), not bool
        — the Styler.map callable must handle that, not just Python bools."""
        assert mscore_flag_color(1) == MSCORE_FLAG_COLOR
        assert mscore_flag_color(0) == MSCORE_NO_FLAG_COLOR


# ---------------------------------------------------------------------------
# style_scored_table — T3 (metric columns never colored), T5 (NaN -> no
# background), integration through .to_html()
# ---------------------------------------------------------------------------


class TestStyleScoredTable:
    @staticmethod
    def _sample_df():
        return pd.DataFrame([
            {"ticker": "AAAA", "overall_score": 1.665496, "mscore_flag": 0,
             "abs_ps_factor": 0.0, "ps_diff": 0.123},
            {"ticker": "BBBB", "overall_score": 4.887982, "mscore_flag": 1,
             "abs_ps_factor": 1.0, "ps_diff": 0.456},
            {"ticker": "CCCC", "overall_score": float("nan"), "mscore_flag": 0,
             "abs_ps_factor": float("nan"), "ps_diff": float("nan")},
        ])

    @staticmethod
    def _domain():
        return {
            "overall_score": (1.665496, 3.276739, 4.887982),
            "abs_ps_factor": (0.0, 0.5, 1.0),
        }

    def test_factor_and_score_columns_colored(self):
        html = style_scored_table(
            self._sample_df(), self._domain(), factor_columns=["abs_ps_factor"]
        ).to_html()
        assert "#63BE7B" in html  # overall_score min
        assert "#F8696B" in html  # overall_score max
        assert "#75E194" in html  # abs_ps_factor min
        assert "#FF7376" in html  # abs_ps_factor max

    def test_mscore_flag_colored_solid(self):
        html = style_scored_table(
            self._sample_df(), self._domain(), factor_columns=["abs_ps_factor"]
        ).to_html()
        assert MSCORE_FLAG_COLOR in html
        assert MSCORE_NO_FLAG_COLOR in html

    def test_nan_cell_has_no_background(self):
        # CCCC's row has NaN overall_score/abs_ps_factor — those specific
        # cells must not carry any of the scale's background-color values.
        styler = style_scored_table(
            self._sample_df(), self._domain(), factor_columns=["abs_ps_factor"]
        )
        # Locate CCCC's row index and confirm its overall_score/abs_ps_factor
        # cells render with no background-color style at all.
        ctx = styler._compute().ctx
        df = self._sample_df().reset_index(drop=True)
        ccc_row = df.index[df["ticker"] == "CCCC"][0]
        overall_col = df.columns.get_loc("overall_score")
        factor_col = df.columns.get_loc("abs_ps_factor")
        assert ctx.get((ccc_row, overall_col), []) == []
        assert ctx.get((ccc_row, factor_col), []) == []

    def test_metric_column_not_colored(self):
        """T3: the interleaved metric column (ps_diff) must carry no
        background-color at all, even though it sits right next to a
        colored factor column."""
        styler = style_scored_table(
            self._sample_df(), self._domain(), factor_columns=["abs_ps_factor"]
        )
        ctx = styler._compute().ctx
        df = self._sample_df().reset_index(drop=True)
        metric_col = df.columns.get_loc("ps_diff")
        for row in range(len(df)):
            assert ctx.get((row, metric_col), []) == []
