"""
Phase 4b — external daily-close price loader for the fixed-horizon Whiteboard
measurement (src/whiteboard_horizons.py).

THIS IS NOT A SCREEN. Same standing as historical_ingest.py (Phase 4a): no
config.yaml screens-block entry, no screens-registry row, no screen_membership
row, never dispatched by refresh.py. Layering: sits beside historical_ingest.py
and imports nothing from refresh.py or validate.py.

price_history IS UPSERT/APPEND-ONLY, NEVER REPLACED — see upsert_price_history.
A delisted name's history stops being retrievable once a newer vendor pull no
longer covers it, so this table is not reconstructable from its source the way
historical_active_shorts/historical_whiteboard_shorts are from the workbook.
A row with source='bloomberg_manual' must survive every subsequent API pull;
getting that backwards would silently destroy exactly the rows that cost the
most to obtain (see the upsert's ON CONFLICT ... WHERE clause).

Price basis: split-adjusted, DIVIDEND-UNADJUSTED close (yfinance's
auto_adjust=False "Close" column, not "Adj Close" — yfinance's own default
changed to auto_adjust=True across releases, which would silently back-adjust
for dividends and disagree with Bloomberg PX_LAST / the stored wba_price on
every dividend-paying name). This must be set explicitly on every call, never
left to the installed version's default.

Vendor: yfinance primary. Stooq was scoped as a documented fallback, but as of
this build Stooq's public CSV endpoint (stooq.com/q/d/l/) returns a JavaScript
bot-challenge page rather than CSV data — it is no longer reachable by a plain
HTTP GET. That challenge is not bypassed here (bot-detection circumvention is
out of scope regardless of purpose); fetch_price_series still attempts the
Stooq path and validates the response shape, but in practice today a
yfinance failure falls through to an uncovered gap (reported by
check_price_coverage, filled manually via ingest_manual_fill) rather than a
working automatic fallback. See PHASE4B_BUILD_REPORT.md for the standing
finding.

Symbol mapping: default rule "<TICKER> US Equity" -> <TICKER>; anything that
doesn't fit (the 7 non-US Whiteboard listings, plus the malformed PRY.IM) must
resolve through config.yaml's prices.symbol_overrides — never inferred, never
silently defaulted. See resolve_vendor_symbol.
"""

import io
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version

import pandas as pd
import requests
from sqlalchemy import text

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.db import append_rows
from src.loaders import find_single_upload_file, validate_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SUFFIX = " US Equity"
SPY_BBG_TICKER = "SPY US Equity"
VENDOR_SOURCES = ("yfinance", "stooq", "bloomberg_manual")

_STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
_STOOQ_EXPECTED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}

MANUAL_FILL_COLUMN_MAP = {"bbg_ticker": "bbg_ticker", "date": "date", "close": "close"}


class SymbolMappingError(ValueError):
    """Raised by resolve_vendor_symbol when a bbg_ticker matches neither an
    explicit config.yaml override nor the default "<TICKER> US Equity" rule.
    Never silently defaulted — see the module docstring."""


@dataclass(frozen=True)
class UniverseResult:
    """assemble_universe's output.

    Attributes:
        universe: DataFrame[bbg_ticker, vendor_symbol, category] — one row per
            distinct series to pull, deduplicated across categories.
        overlaps: bbg_tickers that appeared in more than one of
            {stock, sector_benchmark, spy_benchmark} before dedup — reported,
            not raised on. Empty in the verified live data (no overlap), but
            this must not silently swallow one on a future file.
    """

    universe: pd.DataFrame
    overlaps: list = field(default_factory=list)


@dataclass
class FetchResult:
    """fetch_price_series's output.

    Attributes:
        prices: DataFrame[date, close], empty if neither vendor produced data.
        source: One of "yfinance" / "stooq", or None if both failed — never
            raises on "no data", see the module docstring.
    """

    prices: pd.DataFrame
    source: str = None


def default_vendor_symbol(bbg_ticker: str) -> str:
    """Apply the default "<TICKER> US Equity" -> <TICKER> mapping rule.

    Args:
        bbg_ticker: A Bloomberg identifier string.

    Returns:
        The plain ticker if bbg_ticker ends with " US Equity", else None
        (caller falls through to an explicit override or raises).
    """
    if bbg_ticker.endswith(DEFAULT_SUFFIX):
        return bbg_ticker[: -len(DEFAULT_SUFFIX)]
    return None


def resolve_vendor_symbol(bbg_ticker: str, overrides: dict) -> str:
    """Resolve one bbg_ticker to a yfinance/Stooq-compatible symbol.

    Explicit overrides (config.yaml prices.symbol_overrides) always win over
    the default rule, so a future non-US listing that happens to coincide
    with the default pattern can still be corrected explicitly.

    Args:
        bbg_ticker: A Bloomberg identifier string, e.g. "AAPL US Equity" or
            "EDEN FP Equity".
        overrides: config["prices"]["symbol_overrides"] dict.

    Returns:
        The vendor symbol string.

    Raises:
        SymbolMappingError: If bbg_ticker matches neither overrides nor the
            default rule.
    """
    if bbg_ticker in overrides:
        return overrides[bbg_ticker]
    default = default_vendor_symbol(bbg_ticker)
    if default is not None:
        return default
    raise SymbolMappingError(
        f"No vendor symbol mapping for bbg_ticker {bbg_ticker!r} — matches "
        f"neither prices.symbol_overrides nor the default '<TICKER> US Equity' "
        f"rule. Add an explicit override in config.yaml; never inferred."
    )


def assemble_universe(whiteboard_df: pd.DataFrame, overrides: dict) -> UniverseResult:
    """Build the full set of price series to pull.

    139 distinct bbg_ticker (stocks) + 9 distinct sector_benchmark_ticker
    (sector ETFs) + SPY US Equity (the single benchmark), deduplicated.
    Overlap between categories is DEDUPED AND REPORTED, never raised on — an
    overlap is harmless to the pull itself (the series is fetched once either
    way), so aborting the whole universe assembly over it has no upside.

    Args:
        whiteboard_df: historical_whiteboard_shorts, read as a DataFrame
            (bbg_ticker, sector_benchmark_ticker columns required).
        overrides: config["prices"]["symbol_overrides"] dict.

    Returns:
        UniverseResult.

    Raises:
        SymbolMappingError: Via resolve_vendor_symbol, if any bbg_ticker in
            the universe has no resolvable vendor symbol.
    """
    stocks = set(whiteboard_df["bbg_ticker"].dropna())
    sectors = set(whiteboard_df["sector_benchmark_ticker"].dropna())
    spy = {SPY_BBG_TICKER}

    tagged = []
    for bbg_ticker in sorted(stocks):
        tagged.append((bbg_ticker, "stock"))
    for bbg_ticker in sorted(sectors):
        tagged.append((bbg_ticker, "sector_benchmark"))
    for bbg_ticker in sorted(spy):
        tagged.append((bbg_ticker, "spy_benchmark"))

    seen = {}
    overlaps = []
    rows = []
    for bbg_ticker, category in tagged:
        if bbg_ticker in seen:
            overlaps.append(bbg_ticker)
            continue
        seen[bbg_ticker] = category
        rows.append({
            "bbg_ticker": bbg_ticker,
            "vendor_symbol": resolve_vendor_symbol(bbg_ticker, overrides),
            "category": category,
        })

    if overlaps:
        logger.warning(
            "assemble_universe: %d bbg_ticker(s) appeared in more than one "
            "category, deduped to first occurrence: %s", len(overlaps), overlaps,
        )

    universe = pd.DataFrame(rows, columns=["bbg_ticker", "vendor_symbol", "category"])
    return UniverseResult(universe=universe, overlaps=overlaps)


def fetch_price_series(vendor_symbol: str, start: str, end: str) -> FetchResult:
    """Fetch one daily-close series, yfinance primary, Stooq fallback.

    Never raises on "no data" or a vendor failure — that is
    check_price_coverage's job to report, not this function's to abort on.
    yfinance is called with auto_adjust=False and only its "Close" column is
    used (split-adjusted, dividend-unadjusted) — see the module docstring.

    The `except Exception` around the yfinance call is deliberately broad:
    yfinance is an unofficial, undocumented scraper against a website (see
    PHASE4B_SCOPE.md section 5) whose internal failure surface is not a fixed,
    enumerable set of exception types and changes across releases. The known
    failure mode being handled is exactly that unpredictability, not a lazy
    catch-all — every occurrence is logged with the vendor_symbol and falls
    through to the Stooq attempt rather than silently continuing.

    Args:
        vendor_symbol: A yfinance/Stooq-compatible symbol from
            resolve_vendor_symbol.
        start: ISO date string, inclusive.
        end: ISO date string, exclusive (matches yfinance's convention).

    Returns:
        FetchResult. prices is empty (source=None) if both vendors failed.
    """
    import yfinance as yf

    try:
        raw = yf.download(
            vendor_symbol, start=start, end=end, auto_adjust=False,
            progress=False, threads=False, multi_level_index=False,
        )
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s: %s", vendor_symbol, exc)
        raw = None

    if raw is not None and not raw.empty and "Close" in raw.columns:
        # Built off the index directly (not a reset_index().rename() keyed on
        # the literal string "Date") — robust to whatever the index is named,
        # since that's an incidental property of the vendor response, not a
        # guaranteed contract.
        close = raw["Close"]
        prices = pd.DataFrame({
            "date": pd.to_datetime(close.index).date,
            "close": close.to_numpy(),
        })
        return FetchResult(prices=prices, source="yfinance")

    logger.warning("yfinance returned no usable data for %s, trying Stooq", vendor_symbol)
    stooq_prices = _fetch_stooq(vendor_symbol)
    if stooq_prices is not None and not stooq_prices.empty:
        return FetchResult(prices=stooq_prices, source="stooq")

    logger.warning("No vendor produced data for %s (yfinance and Stooq both failed)", vendor_symbol)
    return FetchResult(prices=pd.DataFrame(columns=["date", "close"]), source=None)


def _fetch_stooq(vendor_symbol: str) -> pd.DataFrame:
    """Attempt Stooq's CSV endpoint. Returns None on any failure — including
    a non-CSV response (e.g. Stooq's JS bot-challenge page, observed as of
    this build; see the module docstring). This function does NOT attempt to
    solve or bypass that challenge; a non-CSV response is treated identically
    to "vendor has no data."""
    stooq_symbol = vendor_symbol.lower()
    if not stooq_symbol.endswith((".us", ".uk", ".de", ".fr", ".it", ".nl", ".au", ".to")):
        # Stooq's own suffix convention differs from yfinance's for US names
        # (".us" is required there but absent from yfinance's bare symbol).
        # Only the default-rule (US) case is remapped here automatically;
        # every non-US override in config.yaml supplies its own vendor_symbol
        # already in the vendor's own convention where that vendor is
        # intended to be reachable.
        stooq_symbol = f"{stooq_symbol}.us"
    url = _STOOQ_URL.format(symbol=stooq_symbol)
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    except requests.exceptions.RequestException as exc:
        logger.warning("Stooq request failed for %s: %s", vendor_symbol, exc)
        return None

    if response.status_code != 200:
        logger.warning("Stooq returned HTTP %d for %s", response.status_code, vendor_symbol)
        return None

    try:
        raw = pd.read_csv(io.StringIO(response.text))
    except (pd.errors.ParserError, ValueError) as exc:
        logger.warning("Stooq response for %s did not parse as CSV: %s", vendor_symbol, exc)
        return None

    if not _STOOQ_EXPECTED_COLUMNS.issubset(set(raw.columns)):
        logger.warning(
            "Stooq response for %s missing expected OHLC columns (got %s) — "
            "likely a non-data response (e.g. a bot-challenge page), not usable.",
            vendor_symbol, list(raw.columns),
        )
        return None

    prices = raw[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    return prices


def _ensure_price_history_table(conn) -> None:
    """Create price_history if it doesn't exist, with its PK and source
    vocabulary enforced at the schema level.

    source is NOT NULL: a NULL source would make the upsert's
    `price_history.source != 'bloomberg_manual'` comparison evaluate to NULL
    (neither true nor false in SQL three-valued logic) rather than TRUE,
    silently making that row permanently unupdatable by any future pull. The
    CHECK constraint closes the source vocabulary to exactly the three known
    values, so an unrecognized source string fails loudly at insert time
    rather than polluting provenance.
    """
    conn.execute(text(
        """
        CREATE TABLE IF NOT EXISTS price_history (
            bbg_ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            source TEXT NOT NULL CHECK(source IN ('yfinance', 'stooq', 'bloomberg_manual')),
            vendor_symbol TEXT,
            ingested_at TEXT NOT NULL,
            PRIMARY KEY (bbg_ticker, date)
        )
        """
    ))


def upsert_price_history(engine, df: pd.DataFrame) -> int:
    """Upsert rows into price_history. NEVER a replace — see the module
    docstring.

    A row with source='bloomberg_manual' already stored is never overwritten
    by an incoming row with a different source. All other combinations
    (api-over-api, api-over-nothing, manual-over-anything) apply freely. This
    is the entire protection mechanism; see the WHERE clause below.

    Args:
        engine: SQLAlchemy engine.
        df: DataFrame with columns bbg_ticker, date, close, source,
            vendor_symbol. date may be a date/Timestamp/ISO string; ingested_at
            is stamped here, uniformly, for the whole batch.

    Returns:
        Number of rows submitted (not necessarily the number that changed
        stored data, since a blocked manual-protection case is a no-op).
    """
    if df.empty:
        return 0

    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = df.copy()
    payload["date"] = pd.to_datetime(payload["date"]).dt.strftime("%Y-%m-%d")
    payload["ingested_at"] = ingested_at
    records = payload[["bbg_ticker", "date", "close", "source", "vendor_symbol", "ingested_at"]].to_dict("records")

    with engine.begin() as conn:
        _ensure_price_history_table(conn)
        conn.execute(
            text(
                """
                INSERT INTO price_history (bbg_ticker, date, close, source, vendor_symbol, ingested_at)
                VALUES (:bbg_ticker, :date, :close, :source, :vendor_symbol, :ingested_at)
                ON CONFLICT(bbg_ticker, date) DO UPDATE SET
                    close = excluded.close,
                    source = excluded.source,
                    vendor_symbol = excluded.vendor_symbol,
                    ingested_at = excluded.ingested_at
                WHERE price_history.source != 'bloomberg_manual'
                   OR excluded.source = 'bloomberg_manual'
                """
            ),
            records,
        )
    return len(records)


def ingest_manual_fill(engine, upload_dir: str = os.path.join("data", "historical", "prices")) -> int:
    """Ingest the one live manual-fill CSV in upload_dir, tagged
    source='bloomberg_manual'.

    Same one-live-file-per-folder discipline as every other upload path
    (find_single_upload_file) — superseded files are expected to be moved to
    an _archive/ subfolder by whoever drops the new one, matching the
    project's standing convention.

    Args:
        engine: SQLAlchemy engine.
        upload_dir: Directory holding exactly one manual-fill CSV, columns
            bbg_ticker, date, close.

    Returns:
        Number of rows upserted.
    """
    filepath = find_single_upload_file(upload_dir, ".csv")
    raw = pd.read_csv(filepath)
    validate_columns(raw, list(MANUAL_FILL_COLUMN_MAP.keys()))
    df = raw.rename(columns=MANUAL_FILL_COLUMN_MAP).copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["source"] = "bloomberg_manual"
    df["vendor_symbol"] = df["bbg_ticker"]
    return upsert_price_history(engine, df)


def run_price_pull(engine, config: dict, start: str, end: str) -> dict:
    """Orchestrate one full price pull: assemble the universe, fetch every
    series, upsert, and append one price_history_runs provenance row.

    Args:
        engine: SQLAlchemy engine.
        config: Full parsed config.yaml dict.
        start: ISO date string, inclusive (2023-08-06, the earliest WBA date,
            per PHASE4B_SCOPE.md).
        end: ISO date string, exclusive.

    Returns:
        Summary dict: universe_size, overlaps, series_fetched (per source),
        rows_upserted, series_failed (vendor_symbol list), yfinance_version.
    """
    whiteboard_df = pd.read_sql("select bbg_ticker, sector_benchmark_ticker from historical_whiteboard_shorts", engine)
    overrides = config["prices"]["symbol_overrides"]
    universe_result = assemble_universe(whiteboard_df, overrides)
    universe = universe_result.universe

    vendor_counts = {source: {"series": 0, "rows": 0} for source in ("yfinance", "stooq")}
    series_failed = []
    total_rows_upserted = 0

    for _, row in universe.iterrows():
        fetch_result = fetch_price_series(row["vendor_symbol"], start, end)
        if fetch_result.source is None:
            series_failed.append(row["vendor_symbol"])
            continue
        batch = fetch_result.prices.copy()
        batch["bbg_ticker"] = row["bbg_ticker"]
        batch["source"] = fetch_result.source
        batch["vendor_symbol"] = row["vendor_symbol"]
        rows_written = upsert_price_history(engine, batch)
        vendor_counts[fetch_result.source]["series"] += 1
        vendor_counts[fetch_result.source]["rows"] += rows_written
        total_rows_upserted += rows_written

    try:
        yfinance_version = pkg_version("yfinance")
    except PackageNotFoundError:
        yfinance_version = None

    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_row = {
        "ingested_at_utc": ingested_at,
        "universe_size": len(universe),
        "overlaps_json": json.dumps(universe_result.overlaps),
        "vendor_counts_json": json.dumps(vendor_counts),
        "series_failed_json": json.dumps(series_failed),
        "rows_upserted": total_rows_upserted,
        "yfinance_version": yfinance_version,
    }
    append_rows(engine, pd.DataFrame([run_row]), "price_history_runs")

    return {
        "universe_size": len(universe),
        "overlaps": universe_result.overlaps,
        "vendor_counts": vendor_counts,
        "series_failed": series_failed,
        "rows_upserted": total_rows_upserted,
        "yfinance_version": yfinance_version,
    }
