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


def style_overlap_table(display_df: pd.DataFrame):
    """Apply the overlap view's on-screen formatting.

    Three independently-scoped format calls, chained: dollar-format
    market_cap; an em-dash placeholder for a null sector (deliberately
    distinct from the overall_score placeholder below — "no data" is a
    different fact from "no data, and here is why"); an explicit
    "Not in short_screen universe" sentence for a null overall_score.

    Each na_rep is confined via `subset` so it cannot bleed into another
    column's cells — an unscoped na_rep applies frame-wide and would
    mislabel a null sector as "Not in short_screen universe" too, which is
    wrong (RSI-only tickers have no sector column at all, independent of
    whether they're in short_screen's universe).

    Args:
        display_df: The overlap table's display columns (must include
            market_cap, sector, overall_score).

    Returns:
        A pandas Styler. No Streamlit import — this is pure pandas, kept
        here (not in app.py) so it's testable via .to_html() directly.
    """
    return (
        display_df.style
        .format({"market_cap": "${:,.0f}"})
        .format(subset=["sector"], na_rep="—")
        .format("{:.3f}", subset=["overall_score"], na_rep="Not in short_screen universe")
    )
