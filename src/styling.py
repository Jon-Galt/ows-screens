"""
Phase 5a — conditional-formatting colour scale for the scored short_screen
main table.

Pure pandas, no Streamlit/SQLAlchemy (Architecture Rule 1) — kept here, not
in app.py, for the same reason overlap.py's style_overlap_table is separate:
it's testable directly via .to_html().

Reproduces the Excel template's per-column three-anchor colorScale (min /
50th percentile / max), extracted from notebooks/OWS Short Screen (April
2026).xlsx with openpyxl — not approximated:

  - Overall Score:        #63BE7B (min) -> #FCFCFF (50th pctile) -> #F8696B (max)
  - The 24 Factor columns: #75E194 (min) -> #FCFCFF (50th pctile) -> #FF7376 (max)
  - M-Score flag: not a scale — solid #75E194 (not flagged) / #FFC7CE (flagged).

Green is at the minimum, red at the maximum: factor scores are direction-
adjusted (Architecture Rule 6), so a high score means more bearish, and red
correctly reads as "stronger short candidate." Do not invert this.

Driver ruling (2026-09-02): the domain for every scaled column is the FULL
UNFILTERED screen, computed once and held fixed — a stock's colour must not
change when sidebar filters change. Callers must pass the unfiltered frame
to build_color_scale_domain, not whatever the sidebar currently filters to.

The mid anchor is each column's own MEDIAN (50th percentile), not the
arithmetic midpoint of its min and max. On 23 of the 24 factor columns these
coincide (median 0.5, range 0..1), which hides the distinction; on
def_rev_factor and liquidity_risk_factor the two Architecture-Rule-7 "**"
balance-sheet/liquidity defaults collapse the median onto the minimum
(lo == mid == 0.0, the majority of rows), which is exactly the case
interpolate_color's degenerate-domain handling below exists for.

lo/hi are each column's own observed min/max, never a hardcoded 0..1 domain
(ratings_factor's real max is 0.971726, not 1.0 — hardcoding would silently
under-color its top row).
"""

import pandas as pd


def build_color_scale_domain(df: pd.DataFrame, columns: list) -> dict:
    """Per-column (lo, mid, hi) anchors for the given columns, computed once
    from the full unfiltered frame.

    Args:
        df: The unfiltered scored short_screen DataFrame.
        columns: Column names to compute a domain for (overall_score plus
            the 24 factor columns).

    Returns:
        column -> (lo, mid, hi) tuple of floats. lo/hi are that column's
        own min/max; mid is its median (pandas' median, i.e. the 50th
        percentile point) — never an arithmetic midpoint of lo and hi, and
        never a hardcoded 0..1 range. NaN values are excluded from all
        three (pandas' min/median/max already skip NaN by default).
    """
    domain = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        domain[col] = (float(series.min()), float(series.median()), float(series.max()))
    return domain


def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(round(c) for c in rgb))


def _lerp_hex(hex_a: str, hex_b: str, frac: float) -> str:
    """Linear interpolation between two hex colors, one channel at a time.

    frac is clamped to [0, 1] so floating-point drift at either endpoint
    (e.g. frac = 1.0000000002) can't push a channel outside its byte range.
    """
    frac = max(0.0, min(1.0, frac))
    rgb_a = _hex_to_rgb(hex_a)
    rgb_b = _hex_to_rgb(hex_b)
    return _rgb_to_hex(tuple(a + (b - a) * frac for a, b in zip(rgb_a, rgb_b)))


def interpolate_color(
    value, lo: float, mid: float, hi: float, lo_hex: str, mid_hex: str, hi_hex: str
):
    """A value's colour under a three-anchor (lo/mid/hi) linear colour scale.

    Args:
        value: The cell's raw value (may be NaN).
        lo: The column's own minimum (the min-anchor).
        mid: The column's own median (the 50th-percentile anchor) — NOT the
            arithmetic midpoint of lo and hi. These coincide for a
            symmetric column and diverge for a skewed one (e.g.
            ratings_factor, or a column dominated by an Architecture-Rule-7
            default value).
        hi: The column's own maximum (the max-anchor).
        lo_hex, mid_hex, hi_hex: Hex colors for the three anchors.

    Returns:
        A hex color string, or None for a NaN value (renders with no
        background, matching Excel's blanks — never a default color).

    Degenerate domains:
        - lo == hi (a constant column): every non-null value takes mid_hex.
        - mid == lo (the lower half has zero width — e.g. def_rev_factor,
          liquidity_risk_factor, whose Architecture Rule 7 default value is
          also their median): every value at that shared lo/mid takes
          mid_hex directly, rather than dividing by zero. The upper half
          (mid -> hi) is unaffected and interpolates normally.
        - mid == hi needs no special case: hi is the column's own observed
          maximum, so no value can exceed mid when mid == hi, meaning every
          value falls into the value <= mid branch, and the row at
          value == hi == mid evaluates frac = (hi - lo) / (mid - lo) = 1.0,
          landing on mid_hex by construction rather than by a guard.
    """
    if pd.isna(value):
        return None
    value = float(value)

    if lo == hi:
        return mid_hex

    if value <= mid:
        if mid == lo:
            return mid_hex
        frac = (value - lo) / (mid - lo)
        return _lerp_hex(lo_hex, mid_hex, frac)
    else:
        frac = (value - mid) / (hi - mid)
        return _lerp_hex(mid_hex, hi_hex, frac)


OVERALL_SCORE_ANCHORS = ("#63BE7B", "#FCFCFF", "#F8696B")
FACTOR_ANCHORS = ("#75E194", "#FCFCFF", "#FF7376")
MSCORE_FLAG_COLOR = "#FFC7CE"
MSCORE_NO_FLAG_COLOR = "#75E194"


def overall_score_color(value, domain: tuple):
    """value's colour under the Overall Score scale (green/white/red)."""
    lo, mid, hi = domain
    return interpolate_color(value, lo, mid, hi, *OVERALL_SCORE_ANCHORS)


def factor_color(value, domain: tuple):
    """value's colour under a Factor column's scale (green/white/red,
    lighter than Overall Score's per the Excel spec)."""
    lo, mid, hi = domain
    return interpolate_color(value, lo, mid, hi, *FACTOR_ANCHORS)


def mscore_flag_color(flagged) -> str:
    """M-Score flag's solid fill: red if flagged, green if not.

    Not a scale (Excel spec) — always filled, never NaN, since mscore_flag
    is a boolean/0-1 column with no missing values.

    Args:
        flagged: Truthy/falsy (bool, or 0/1 as SQLite/pandas stores it).

    Returns:
        MSCORE_FLAG_COLOR if flagged, else MSCORE_NO_FLAG_COLOR.
    """
    return MSCORE_FLAG_COLOR if flagged else MSCORE_NO_FLAG_COLOR


def _css_background(hex_color) -> str:
    """Translate a colour (or None) into the CSS a Styler.map callable must
    return. None -> "" (no background), never a default color — this is
    the boundary where interpolate_color's pure "None means no fill"
    contract becomes an actual CSS declaration."""
    if hex_color is None:
        return ""
    return f"background-color: {hex_color}"


def style_scored_table(display_df: pd.DataFrame, domain: dict, factor_columns: list):
    """Apply the short_screen main table's conditional formatting.

    Args:
        display_df: The (already filtered, already sorted) DataFrame being
            rendered. May include interleaved metric columns (the "Show
            underlying metric values" checkbox) — those are never colored.
        domain: A build_color_scale_domain(...) result computed from the
            UNFILTERED frame (Driver ruling — colour must not move when
            sidebar filters change), covering "overall_score" plus every
            column in factor_columns.
        factor_columns: The factor score columns present in display_df
            (a subset of DISPLAY_COLUMNS ending in "_factor").

    Returns:
        A pandas Styler with Styler.map applied per column, confined via
        subset so interleaved metric columns are never touched.
    """
    styler = display_df.style

    if "overall_score" in display_df.columns:
        lo, mid, hi = domain["overall_score"]
        styler = styler.map(
            lambda v, lo=lo, mid=mid, hi=hi: _css_background(
                overall_score_color(v, (lo, mid, hi))
            ),
            subset=["overall_score"],
        )

    for col in factor_columns:
        if col not in display_df.columns or col not in domain:
            continue
        lo, mid, hi = domain[col]
        styler = styler.map(
            lambda v, lo=lo, mid=mid, hi=hi: _css_background(
                factor_color(v, (lo, mid, hi))
            ),
            subset=[col],
        )

    if "mscore_flag" in display_df.columns:
        styler = styler.map(
            lambda v: _css_background(mscore_flag_color(v)),
            subset=["mscore_flag"],
        )

    return styler
