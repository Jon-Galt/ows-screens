"""
Cross-screen overlap calculations (Phase 3d Part 1).

Turns screen_membership + the screens registry + each screen's own
identity-bearing table into an overlap table and a presence matrix. Like
transform.py/score.py, every function here is pandas-in/pandas-out with no
SQLAlchemy or Streamlit imports (Architecture Rule 1) — the SQLite reads
happen in app.py via the existing per-screen-type cached loaders, not here.

short_screen is treated as context, not as a membership tick: it is the
broad ~1,300-name universe the five thematic/RSI screens are drawn from,
so counting it as "just another screen" would make every one of its names
read as "on 2+ screens" purely for being in the main universe. Its
overall_score is carried along as context instead (see compute_overlap).
"""

import pandas as pd

UNIVERSE_SCREEN_ID = "short_screen"


def _thematic_screen_ids(screens_df: pd.DataFrame, universe_screen_id: str) -> list:
    """Every screen_id in the registry except the universe screen.

    No hardcoded screen list — this is what makes compute_overlap and
    build_presence_matrix generic over however many thematic/RSI screens
    exist in config.yaml.
    """
    return [sid for sid in screens_df["screen_id"] if sid != universe_screen_id]


def _resolve_field(ticker: str, field: str, source_order: list, screen_data: dict):
    """First non-null value of `field` for `ticker`, walking source_order.

    Resolution is per-field, not per-row: a ticker present on both an
    RSI-shaped screen (no `sector` column at all) and a curated screen
    resolves `sector` from the curated screen specifically, because RSI's
    row for that field is null (or the column is absent) and gets skipped
    — even if `name`/`market_cap` happen to resolve from RSI first because
    RSI sorts earlier in source_order. Returns None if no source has a
    non-null value (or the ticker isn't in any source at all) rather than
    raising.
    """
    for screen_id in source_order:
        df = screen_data.get(screen_id)
        if df is None or field not in df.columns:
            continue
        match = df.loc[df["ticker"] == ticker, field]
        if not match.empty and pd.notna(match.iloc[0]):
            return match.iloc[0]
    return None


def compute_overlap(
    membership_df: pd.DataFrame,
    screens_df: pd.DataFrame,
    screen_data: dict,
    universe_screen_id: str = UNIVERSE_SCREEN_ID,
) -> pd.DataFrame:
    """Build the cross-screen overlap table.

    Args:
        membership_df: The full screen_membership table (screen_id,
            ticker) across every screen, including universe_screen_id.
        screens_df: The screens registry (screen_id, display_name,
            screen_type, has_scoring).
        screen_data: screen_id -> that screen's own loaded identity-bearing
            table (scored_data/curated_data/transformed_data, whichever
            applies), each with at least a `ticker` column.
        universe_screen_id: The screen treated as context rather than a
            membership tick (default "short_screen").

    Returns:
        One row per unique ticker across the UNION of every screen
        (including tickers on zero thematic/RSI screens — filtering those
        out is a UI-level concern, not this function's), with columns:
        ticker, screen_count (thematic/RSI screens only), screens_on
        (comma-joined display names, sorted, thematic/RSI only), name,
        sector, market_cap (field-level fallback across sources — see
        _resolve_field), in_universe (bool), overall_score (float, NaN if
        not in_universe; never backfilled from a thematic screen since
        only the universe screen produces it).
    """
    thematic_ids = _thematic_screen_ids(screens_df, universe_screen_id)
    display_names = dict(zip(screens_df["screen_id"], screens_df["display_name"]))
    source_order = [universe_screen_id] + sorted(thematic_ids)

    all_tickers = sorted(membership_df["ticker"].unique())

    thematic_membership = membership_df[membership_df["screen_id"].isin(thematic_ids)]
    screens_by_ticker = thematic_membership.groupby("ticker")["screen_id"].apply(list)

    uni_df = screen_data.get(universe_screen_id)
    uni_tickers = set(uni_df["ticker"]) if uni_df is not None else set()

    rows = []
    for ticker in all_tickers:
        screen_ids_on = screens_by_ticker.get(ticker, [])
        screens_on = ", ".join(sorted(display_names[sid] for sid in screen_ids_on))

        in_universe = ticker in uni_tickers
        overall_score = float("nan")
        if in_universe:
            match = uni_df.loc[uni_df["ticker"] == ticker, "overall_score"]
            if not match.empty:
                overall_score = match.iloc[0]

        rows.append({
            "ticker": ticker,
            "screen_count": len(screen_ids_on),
            "screens_on": screens_on,
            "name": _resolve_field(ticker, "name", source_order, screen_data),
            "sector": _resolve_field(ticker, "sector", source_order, screen_data),
            "market_cap": _resolve_field(ticker, "market_cap", source_order, screen_data),
            "in_universe": in_universe,
            "overall_score": overall_score,
        })

    return pd.DataFrame(
        rows,
        columns=[
            "ticker", "screen_count", "screens_on", "name", "sector",
            "market_cap", "in_universe", "overall_score",
        ],
    )


def build_presence_matrix(
    membership_df: pd.DataFrame,
    screens_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    universe_screen_id: str = UNIVERSE_SCREEN_ID,
) -> pd.DataFrame:
    """Build the per-screen 0/1 presence matrix (the Excel Summary tab's
    genuinely useful part, made generic over N screens).

    Args:
        membership_df: The full screen_membership table.
        screens_df: The screens registry.
        overlap_df: A compute_overlap() result, used only to join in
            in_universe so the exported matrix is self-contained rather
            than interpretable only next to the main overlap table.
        universe_screen_id: Excluded as a matrix column — same reasoning
            as compute_overlap's screen_count: short_screen is context,
            not a membership tick, and including it as a column would
            make ~261 rows read as "present" for being in the broad
            universe rather than for matching any thematic thesis.

    Returns:
        One row per ticker that appears on at least one thematic/RSI
        screen, columns: ticker, one 0/1 int column per thematic/RSI
        screen's display_name, in_universe (bool, joined from overlap_df;
        False if the ticker isn't in overlap_df for some reason, which
        should not happen in practice since overlap_df already covers the
        full membership union).
    """
    thematic_ids = _thematic_screen_ids(screens_df, universe_screen_id)
    display_names = dict(zip(screens_df["screen_id"], screens_df["display_name"]))

    thematic_membership = membership_df[membership_df["screen_id"].isin(thematic_ids)].copy()
    thematic_membership["display_name"] = thematic_membership["screen_id"].map(display_names)
    thematic_membership["present"] = 1

    matrix = thematic_membership.pivot_table(
        index="ticker", columns="display_name", values="present",
        fill_value=0, aggfunc="max",
    ).astype(int).reset_index()

    # A thematic screen with zero tickers wouldn't produce a pivot column
    # at all — add it explicitly so the matrix is still generic over N
    # screens even in that edge case, rather than silently dropping a
    # screen from the output.
    for sid in thematic_ids:
        dn = display_names[sid]
        if dn not in matrix.columns:
            matrix[dn] = 0

    matrix = matrix.merge(overlap_df[["ticker", "in_universe"]], on="ticker", how="left")
    matrix["in_universe"] = matrix["in_universe"].fillna(False)
    return matrix


def join_metric_column(
    overlap_df: pd.DataFrame,
    metric_df: pd.DataFrame | None,
    column: str,
    fill_value: float = 0,
) -> pd.DataFrame:
    """Left-join a single per-ticker metric column onto the overlap frame
    (Phase 6c) — generic over which screen's aggregate supplies it; no
    screen_id is baked in here, so a second such join needs no change to
    this function (the screen_id -> column mapping lives in app.py, beside
    its other screen-specific display constants).

    Args:
        overlap_df: A compute_overlap() result (or any frame with a
            `ticker` column) — never mutated in place.
        metric_df: A per-ticker aggregate carrying `ticker` and `column`
            (e.g. the transcripts screen's transformed aggregate), or None
            if that screen's table isn't loadable (a fresh/partial
            database). Not required to be one-row-per-ticker for this
            function's own correctness, but every current caller's
            aggregate is, which is what keeps the join from fanning out.
        column: The metric column to pull from metric_df.
        fill_value: Value for a ticker in overlap_df with no row in
            metric_df. Defaults to 0, matching this frame's existing
            screen_count convention (a ticker off the screen entirely is a
            measured zero, not missing data) — but that convention is
            right for a COUNT and wrong for anything where zero is itself
            a meaningful value (e.g. a ratio): a caller joining such a
            column must pass NaN explicitly.

    Returns:
        A copy of overlap_df with `column` added, or overlap_df UNCHANGED
        (same columns as today, `column` absent) if metric_df is None or
        lacks `column` — the caller's KeyError guard, not this function's,
        is what makes that safe to render (see app.py's render_overlap_
        page). When added, `column` is cast to metric_df's own dtype
        (typically int64) after filling, so a whole-number count doesn't
        silently become a float via the merge's NaN promotion.
    """
    if metric_df is None or column not in metric_df.columns:
        return overlap_df
    result = overlap_df.merge(metric_df[["ticker", column]], on="ticker", how="left")
    result[column] = result[column].fillna(fill_value).astype(metric_df[column].dtype)
    return result


def screen_count_ceiling(overlap_df: pd.DataFrame) -> int:
    """Max screen_count in the given frame, for the overlap view's
    minimum-screen-count slider upper bound.

    Callers must pass the UNFILTERED overlap_df so the slider's own bound
    never moves out from under the user because of an unrelated filter
    (e.g. a sector selection).

    Returns:
        int(overlap_df["screen_count"].max()), or 1 if that would be
        undefined or degenerate: an empty frame (max() is NaN — would
        raise on int() conversion) or a frame where every screen_count is
        0 (no ticker on more than one thematic screen). 1 is always a
        valid, non-degenerate return — the caller compares it to 1 to
        decide whether to skip the slider (st.slider raises when
        min_value == max_value).
    """
    if overlap_df.empty:
        return 1
    max_count = overlap_df["screen_count"].max()
    if pd.isna(max_count):
        return 1
    return max(int(max_count), 1)


def resolve_overlap_click_target(
    ticker: str,
    in_universe: bool,
    membership_df: pd.DataFrame,
    screens_df: pd.DataFrame,
    universe_screen_id: str = UNIVERSE_SCREEN_ID,
) -> str | None:
    """Which screen a click on this ticker's overlap row should land on
    (Phase 5b-2 click-through).

    An in-universe ticker lands on the universe screen — it has a row for
    every in-universe ticker and gives the fullest view (factor chart,
    M-Score, etc). A not-in-universe ticker has no row there (see
    CLAUDE.md's Known Issues on the 17 thematic-only names), so landing it
    on the universe screen would be wrong; it lands instead on the
    alphabetically-first screen_id (not display name — arbitrary but
    deterministic) among the thematic/RSI screens it actually belongs to.

    Args:
        ticker: The clicked ticker.
        in_universe: Whether it's in the universe screen's own table.
        membership_df: The full screen_membership table.
        screens_df: The screens registry (unused directly here, but kept
            for symmetry with this module's other functions and in case a
            future revision needs display-name tie-breaking).
        universe_screen_id: The universe screen_id.

    Returns:
        A screen_id, or None only if `ticker` has zero membership rows at
        all — shouldn't happen for a ticker drawn from compute_overlap's
        own membership_df, but the caller must not assume it.
    """
    if in_universe:
        return universe_screen_id
    ids = sorted(membership_df.loc[membership_df["ticker"] == ticker, "screen_id"].unique())
    ids = [sid for sid in ids if sid != universe_screen_id]
    return ids[0] if ids else None


def apply_zero_thematic_label(
    display_df: pd.DataFrame, label: str = "No thematic screens"
) -> pd.DataFrame:
    """Replace the zero-thematic-screens placeholder in screens_on with an
    explicit label (Phase 5b-2).

    screens_on is the empty string "" for a zero-thematic-screen ticker, not
    NaN (see compute_overlap), so style_overlap_table's existing na_rep
    mechanism can't reach it — and that function is deliberately left alone
    (see its own docstring on why its three na_rep calls are each scoped via
    `subset` to exactly one column). This is a separate, third placeholder,
    distinct from the sector em-dash and from "Not in short_screen universe"
    — three different facts about three different columns.

    Args:
        display_df: The overlap table's display frame (must include
            screens_on).
        label: The replacement text for an empty screens_on value.

    Returns:
        A copy of display_df with screens_on's empty strings replaced.
        Confined to the screens_on column via direct column reassignment
        (never a frame-wide .replace), so an empty string that happens to
        appear in any other column is untouched.
    """
    display_df = display_df.copy()
    display_df["screens_on"] = display_df["screens_on"].replace("", label)
    return display_df


def zero_thematic_summary(
    uni_tickers: set,
    membership_df: pd.DataFrame,
    universe_screen_id: str = UNIVERSE_SCREEN_ID,
) -> tuple:
    """(zero_count, universe_total) for the drill-down's zero-case sentence
    (Phase 5b-2) — how many of the universe's own tickers sit on zero
    thematic/RSI screens, out of how many are in the universe at all.

    Deliberately independent of compute_overlap: this stat only needs the
    universe ticker set and membership_df, not every screen's full identity
    table, so the drill-down's zero-case sentence doesn't have to pull in
    the overlap view's own (larger, cross-screen-identity-resolving)
    machinery just to report two counts.

    Args:
        uni_tickers: The universe screen's own ticker set. Callers must pass
            the UNFILTERED set (e.g. from the unfiltered scored frame, not a
            sidebar-filtered one) — same reasoning as styling.py's
            build_color_scale_domain: a sidebar filter must not move this
            stat.
        membership_df: The full screen_membership table.
        universe_screen_id: The universe screen_id, excluded when
            determining thematic membership.

    Returns:
        (zero_count, universe_total) as plain ints.
    """
    universe_total = len(uni_tickers)
    thematic_tickers = set(
        membership_df.loc[membership_df["screen_id"] != universe_screen_id, "ticker"]
    )
    zero_count = len(uni_tickers - thematic_tickers)
    return zero_count, universe_total


def style_overlap_table(
    display_df: pd.DataFrame,
    extra_formats: dict | None = None,
    extra_na_reps: dict | None = None,
):
    """Apply the overlap view's on-screen formatting.

    Three independently-scoped format calls, chained: dollar-format
    market_cap; an em-dash placeholder for a null sector (deliberately
    distinct from the overall_score placeholder below — "no data" is a
    different fact from "no data, and here is why"); an explicit
    "Not in short_screen universe" sentence for a null overall_score. Then
    one additional scoped call per extra_na_reps entry (Phase 8a).

    Each na_rep is confined via `subset` so it cannot bleed into another
    column's cells — an unscoped na_rep applies frame-wide and would
    mislabel a null sector as "Not in short_screen universe" too, which is
    wrong (RSI-only tickers have no sector column at all, independent of
    whether they're in short_screen's universe).

    Phase 6c review round 1, Correction 4.1: this module knows no column
    name or format spec it did not itself produce — market_cap/sector/
    overall_score are compute_overlap's own output columns, but
    mention_count is injected by app.py's apply_overlap_metric_joins, so
    its format spec is supplied BY app.py via extra_formats rather than
    hardcoded here (which would have hand-copied UNSCORED_METRIC_FORMATS's
    entry a second time, with nothing enforcing the two ever agreeing).

    Phase 8a: a joined column whose own fill is NaN (see
    join_metric_column's fill_value) needs BOTH a number format and a
    na_rep on the SAME cell — e.g. pe_vs_normal_5y renders "1.31x" or "—".
    A column in extra_na_reps is therefore EXCLUDED from the bulk
    `formats` dict and given exactly one scoped `.format(spec, subset=...,
    na_rep=...)` call instead, the same shape overall_score's own call
    already uses. Measured: a LATER na_rep-only scoped call for a column
    already in the bulk dict does not merge with that column's format —
    it REPLACES it — so "1.31x" would silently become "1.310000" (the
    format lost, not just the na_rep gained). Combining both in one call
    is the only correct order.

    Args:
        display_df: The overlap table's display columns (must include
            market_cap, sector, overall_score).
        extra_formats: Additional {column: format_spec} entries for
            columns that may or may not be present in display_df (e.g.
            {"mention_count": "{:,.0f}"}) — applied only for the ones
            actually present, so a caller can pass the full mapping
            unconditionally without checking display_df's columns first;
            an absent column is silently skipped rather than raising (the
            transcripts screen's table not being loadable is exactly this
            case — see join_metric_column / apply_overlap_metric_joins).
        extra_na_reps: Additional {column: na_rep_string} entries for
            columns whose fill_value is NaN rather than 0 (e.g.
            {"pe_vs_normal_5y": "—"}). A column named here is pulled out
            of the bulk `formats` dict (if present there) and formatted in
            its own scoped call combining extra_formats' spec for it with
            this na_rep. Silently skipped if absent from display_df, same
            convention as extra_formats.

    Returns:
        A pandas Styler. No Streamlit import — this is pure pandas, kept
        here (not in app.py) so it's testable via .to_html() directly.
    """
    extra_na_reps = extra_na_reps or {}
    formats = {"market_cap": "${:,.0f}"}
    for col, spec in (extra_formats or {}).items():
        if col in display_df.columns and col not in extra_na_reps:
            formats[col] = spec

    styler = (
        display_df.style
        .format(formats)
        .format(subset=["sector"], na_rep="—")
        .format("{:.3f}", subset=["overall_score"], na_rep="Not in short_screen universe")
    )
    for col, na_rep in extra_na_reps.items():
        if col not in display_df.columns:
            continue
        spec = (extra_formats or {}).get(col)
        if spec is None:
            styler = styler.format(subset=[col], na_rep=na_rep)
        else:
            styler = styler.format(spec, subset=[col], na_rep=na_rep)
    return styler
