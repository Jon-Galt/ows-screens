# OWS Short Screen

A Python-based quantitative stock screening tool for identifying short candidates across a broad equity universe. Rebuilt from a Bloomberg/Excel-based workflow into a maintainable, extensible codebase with a web-based UI.

---

## Overview

This tool ingests fundamental and market data for ~1,300+ stocks, calculates derived metrics across six factor categories, percentile ranks each metric across the full universe, and produces a weighted composite score to surface the highest-priority short candidates for further research.

It is a direct rebuild of a prior Excel-based screener, with identical logic and factor weights, plus an architecture designed to accommodate new data sources, additional factors, and automated data refresh over time.

### Factor Categories

| Category | Description |
|---|---|
| **Valuation** | Absolute and relative Price/Sales and FCF Yield vs. historical averages |
| **Growth** | Revenue growth deceleration and forward vs. historical CAGR differential |
| **Profitability** | NTM gross margin and EBIT margin vs. 3-year historical averages |
| **Balance Sheet** | Leverage, debt coverage, refinancing risk, and liquidity runway |
| **Cash Flow** | FCF conversion, accruals quality, working capital trends, dilution |
| **Non-GAAP** | EBIT and EPS adjustment ratios (Non-GAAP vs. GAAP) |
| **Sentiment** | Short interest and analyst rating distribution |
| **M-Score** | Beneish earnings manipulation model (displayed separately, not in composite) |

---

## Repository Structure

```
/data/
  uploads/
    <screen_id>/      ← Drop each screen's raw CSV/Excel export(s) here (e.g. short_screen/, cyclicals/)
  screener.db         ← SQLite database (auto-generated, not committed to git)

/src/
  ingest.py           ← Quant/Bloomberg loader: reads a screen's uploaded files, loads into raw_data__<screen_id>
  curated_ingest.py   ← Canary curated-screen loader (shared by all 4 curated screens), loads into curated_data__<screen_id>
  transform.py        ← Calculates all derived metrics for a quant_composite screen
  score.py            ← Percentile ranking and weighted composite score for a quant_composite screen
  config.yaml loader  ← src/config.py: load_config()/CONFIG_PATH, plus get_screen_type()/ScreenTypeError (shared type dispatch)
  loaders.py          ← Generic file-reading helpers shared by ingest.py and curated_ingest.py
  db.py               ← Multi-screen storage helpers: table_name(), sync_screens_registry(), replace_screen_rows()
  app.py              ← Streamlit web UI with a screen selector; separate rendering paths for quant vs. curated screens

/tests/
  test_transform.py     ← Unit tests for all transform functions
  test_score.py         ← Unit tests for ranking and scoring logic
  test_schema.py        ← Unit tests for the multi-screen storage helpers, type dispatch guards, and a pipeline-isolation regression lock
  test_curated_ingest.py ← Unit tests for the curated loader (quote-stripping, unit conversions, scores parsing, upload-folder guard)

/notebooks/
  OWS Short Screen (March 2026).xlsx  ← Original Excel file (kept for validation)
  validation.ipynb      ← Side-by-side comparison of Excel vs. Python outputs (quant pipeline only)

config.yaml           ← Per-screen config keyed by screen_id: display_name, type, universe, and (quant_composite only) factor_weights/scoring
requirements.txt      ← Python dependencies
.gitignore
README.md
```

---

## Data Architecture

Storage is scoped per screen: each screen owns its own physical tables, named by convention as `<stage>__<screen_id>` (e.g. `raw_data__short_screen`, `curated_data__cyclicals`), so screens with different column shapes never share a table. A `screens` registry table and a shared `screen_membership(screen_id, ticker)` table (for the cross-screen overlap view planned in Phase 3e) sit alongside them.

Quant screens flow through four sequential layers:

```
Raw CSV/Excel upload (data/uploads/<screen_id>/)
        ↓
   [ ingest.py ]  →  raw_data__<screen_id>  (SQLite)
        ↓
[ transform.py ]  →  transformed_data__<screen_id>  (SQLite)
        ↓
   [ score.py  ]  →  scored_data__<screen_id>  (SQLite)
        ↓
   [ app.py    ]  →  Streamlit UI + Excel/CSV export
```

Curated screens have no transform or scoring stage — there's nothing to rank or compose — so they flow through two:

```
Raw CSV upload (data/uploads/<screen_id>/, exactly one file)
        ↓
[ curated_ingest.py ]  →  curated_data__<screen_id>  (SQLite)
        ↓
      [ app.py       ]  →  Streamlit UI + Excel/CSV export
```

Each screen's pipeline runs independently of every other screen's tables, and calling the wrong stage against the wrong screen type (e.g. `score.py` against a curated screen) fails clearly with `ScreenTypeError` rather than doing something undefined. `short_screen` (quant) and `cyclicals`/`competition`/`structural`/`management_comp` (curated) are populated today; see the Development Phases section below for the roadmap onto this architecture.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/your-org/ows-short-screen.git
cd ows-short-screen
pip install -r requirements.txt
```

### Running the Screener

**Short Screen (quant):**

1. Drop your Bloomberg CSV/Excel export into `/data/uploads/short_screen/`
2. Run the pipeline:

```bash
python src/ingest.py
python src/transform.py
python src/score.py
```

**A curated screen (Cyclicals, Competition, Structural, Management Comp):**

1. Drop the screen's single Canary CSV export into `/data/uploads/<screen_id>/` (e.g. `/data/uploads/cyclicals/`) — exactly one file; the loader has no way to tell screens apart by content, so more than one file is rejected as an error
2. Run the curated ingest:

```bash
python -c "from src.curated_ingest import ingest_curated; ingest_curated('cyclicals')"
```

**Then, for either:**

```bash
streamlit run src/app.py
```

and pick the screen from the sidebar selector.

### Running Tests

```bash
pytest tests/
```

---

## Configuration

Each screen has its own config block in `config.yaml`, keyed by `screen_id` under a top-level `screens:` map. `quant_composite` screens (like `short_screen`) carry `factor_weights` and a `scoring` block; `curated` screens carry neither — there's nothing to weight or score. Edit this file to adjust a screen's composite score weighting without touching any Python code.

```yaml
screens:
  short_screen:
    display_name: "OWS Short Screen"
    type: quant_composite

    universe:
      name: "OWS Short Screen"
      as_of: "2026-03"

    factor_weights:
      # Valuation (sum: 1.0)
      abs_ps_factor: 0.25
      rel_ps_factor: 0.25
      abs_fcf_factor: 0.25
      rel_fcf_factor: 0.25
      # ... (Growth, Profitability, Balance Sheet, Cash Flow, Non-GAAP, Sentiment)

    scoring:
      mscore_manipulation_threshold: -2.22
      nan_default_standard: 0.5   # Default percentile for most missing factors
      nan_default_balance_sheet: 0.0  # Default for balance sheet / liquidity factors

  cyclicals:
    display_name: "Cyclicals"
    type: curated

    universe:
      name: "Cyclicals"
      as_of: "2026-08"
    # No factor_weights, no scoring — curated screens aren't ranked or composited.

  # competition, structural, management_comp follow the same curated shape.
```

---

## Development Phases

This project is being built incrementally. Each phase has a defined scope and acceptance criteria before moving to the next.

---

### Phase 1 — Replication (Complete)

**Goal:** Faithfully replicate all Excel logic in Python and validate parity with the original file.

**Scope:**
- `src/ingest.py` — load CSV/Excel exports into SQLite, handle `"#N/A N/A"` strings and data type coercion
- `src/transform.py` — all 30+ derived metric calculations, matching original Excel formulas exactly
- `src/score.py` — percentile ranking (matching Excel's `PERCENTRANK.INC`) and weighted composite score
- `config.yaml` — factor weights and thresholds
- `tests/test_transform.py` — unit tests for every transform function with edge cases
- `tests/test_score.py` — unit tests for ranking direction and default fallback logic
- `notebooks/validation.ipynb` — row-by-row comparison of Python output vs. Excel for the March 2026 file

**Acceptance criteria:**
- All 24 factor scores match Excel output within ±0.001 for 95%+ of stocks
- All unit tests pass
- No unhandled exceptions on the reference dataset

---

### Phase 2 — Web UI (Complete)

**Goal:** Build an interactive Streamlit interface to replace direct Excel browsing.

**Scope:**
- `src/app.py` — Streamlit application with:
  - Filterable, sortable data table showing all scored stocks
  - Sector and industry filter dropdowns
  - Market cap range slider
  - Overall score range filter
  - M-Score flag indicator (highlight stocks > -2.22)
  - Individual stock drill-down showing all factor scores
  - Export to Excel and CSV

**Acceptance criteria:**
- All filters work correctly and update the table in real time
- Export produces a correctly formatted Excel file
- App loads the full 1,300+ stock universe without performance issues

---

### Phase 3a — Multi-Screen Architecture (Complete)

**Goal:** Generalize the single-screen pipeline into a multi-screen foundation, then migrate the existing Short Screen onto it with zero behavior change, before any new screen's data is loaded.

**Scope:**
- A `screens` registry (`screen_id`, `display_name`, `type`: `quant_composite` or `curated`)
- Per-screen physical tables via `table_name(stage, screen_id)` (e.g. `raw_data__short_screen`) instead of one global table per pipeline stage
- A shared `screen_membership(screen_id, ticker)` table, built for the cross-screen overlap view planned in Phase 3e
- `config.yaml` restructured to a per-screen block keyed by `screen_id`
- Short Screen migrated onto the new schema, verified against a pre-migration snapshot

**Acceptance criteria:**
- Short Screen's output is numerically identical to the pre-migration pipeline (verified against a snapshot and a `validation.ipynb` parity re-run)
- All existing tests pass, plus a new end-to-end regression lock proving one screen's pipeline run cannot alter another screen's tables

---

### Phase 3b — Onboard the 4 Curated Screens (Complete)

**Goal:** Onboard Cyclicals, Competition, Structural, and Management Comp — the first real exercise of the `curated` screen type — and eliminate the manual Excel workbook consolidation those four screens previously required.

**Scope:**
- `src/curated_ingest.py` — one loader shared by all four screens, since they share an identical 11-column Canary export schema: strips Canary's quote-wrapped numeric strings, applies the codebase's unit conventions, parses the packed `scores` field into three numeric columns (retaining the raw string for provenance), and writes to `curated_data__<screen_id>`
- Exactly one `.csv` export per screen's upload folder, enforced in code — curated screens have no column identifying which screen an export belongs to, so a misfile or a stray second file is a loud, named error rather than a silent concat or misread
- Type-aware dispatch: curated screens cannot run `transform.py` or `score.py` — there is no ranking, no composite score, no M-Score for them — and invoking either fails clearly with `ScreenTypeError` instead of an opaque error deep in scoring logic
- `src/app.py` — a screen selector, plus a separate curated rendering path (table, narrative rationale, three risk scores); the quant view (factor breakdown, M-Score, filters) is unchanged for Short Screen
- `screen_membership` populated for each curated screen (the overlap *view* itself is Phase 3e — this phase builds the data, not the UI)

**Acceptance criteria:**
- All four curated screens load with row counts matching their source exports exactly
- Zero NaNs in the numeric columns that arrive quote-wrapped — that's exactly what a missed quote-strip produces
- Short Screen's output and existing tests are completely unaffected

---

### Phase 3c — Rising Short Interest

**Goal:** Onboard the second `quant_composite` screen — a Bloomberg short-interest export — and give it its own percentile/composite scoring layer.

**Scope (to be fully defined when this phase is scoped):** A second Bloomberg-shaped ingest config; its own factor and weight definitions, since `FACTOR_DEFINITIONS` in `score.py` is currently a single global set sized for Short Screen; revisiting whether `ingest.py`'s multi-file-concat behavior needs the same "a misfile is a loud error" treatment Phase 3b gave curated screens, now that a second quant screen means a second upload folder someone could mix up.

---

### Phase 3d — Automation

**Goal:** Reduce manual effort in the refresh cycle, across all screens.

**Scope (to be fully defined when this phase is scoped):** Scheduled refresh; a validation report generated at each run (missing columns, NaN-rate spikes, universe size changes); a run-history log spanning every screen's scored or curated data, not just Short Screen's.

---

### Phase 3e — Cross-Screen Overlap View

**Goal:** Make "which screens flag this ticker" a first-class feature, using the `screen_membership` table Phase 3a built and Phase 3b started populating for exactly this purpose.

**Scope (to be fully defined when this phase is scoped):** A UI view (and/or query surface) over `screen_membership` showing cross-screen overlap — replacing the old consolidated workbook's `Summary` sheet, whose `COUNTIF` formulas broke every time a screen was added or resized.

---

### Phase 3f — Canary API Integration

**Goal:** Live API sourcing for Canary data that *is* API-accessible.

**Scope (to be fully defined when this phase is scoped):** This is separate from the curated screens onboarded in Phase 3b — their narrative rationale and risk scores arrive via the Canary CSV export and are not available through the API. Likely scope here is risk scores as a new factor or enrichment for existing screens, not a replacement for the curated screens' export-based refresh.

---

### Phase 4 — Expanded Analytics

**Goal:** Add analytical depth beyond the original Excel scope, once the multi-screen foundation and automation are in place.

**Scope (to be fully defined when this phase is scoped):**
- Historical score tracking — chart how a stock's composite score has changed over time
- Sector-relative scoring — percentile rank within sector in addition to full-universe ranking
- Backtesting — assess whether high composite scores have historically predicted underperformance

---

## Key Design Decisions

**Why SQLite?** It requires no server, lives as a single file in the repo (excluded from git), and is fully readable by pandas. It can be swapped for Postgres later with minimal code changes if multi-user access becomes necessary.

**Why Streamlit?** It is Python-only, requires no frontend knowledge, and supports interactive tables, filters, and file downloads out of the box. It is the fastest path to a usable web UI for a small team.

**Why config.yaml for weights?** Factor weights are the primary thing that changes between research iterations. Keeping them out of Python code means they can be adjusted, version-controlled, and reviewed independently from the calculation logic.

**Why separate ingest / transform / score?** Each step has a different failure mode and a different reason to be rerun independently. If a new field is added to the data export, only `ingest.py` and `transform.py` need to change. If a factor weight is adjusted, only `score.py` needs to rerun.

---

## Notes on Excel Parity

The original Excel screener used `PERCENTRANK.INC`, which includes both endpoints in the percentile range and produces values between 0 and 1 inclusive. The Python implementation uses `scipy.stats.percentileofscore(..., kind='rank') / 100` to match this behavior exactly.

Some factors use `1 - percentile` because a lower raw value is worse for the short thesis (e.g., lower FCF yield is worse, shorter debt maturity is worse). These are documented explicitly in the factor scoring table in `src/score.py`.

The Beneish M-Score is calculated and displayed but is **not included in the composite overall score**, consistent with the original Excel design. Stocks with M-Score > -2.22 are flagged as potential earnings manipulators.

---

## Data Source

Two data sources feed this tool today, both via manual export:

- **Short Screen** (quant): Bloomberg, via manual CSV/Excel export. Required fields and column naming conventions are documented in `src/ingest.py`.
- **Cyclicals, Competition, Structural, Management Comp** (curated): Canary, via manual CSV export — one export per screen, dropped into that screen's own upload folder. The schema and cleaning rules are documented in `src/curated_ingest.py`. Canary's narrative rationale and risk scores are not available through its API, so this export-based refresh isn't going away even after Phase 3f adds API sourcing for the data that is API-accessible.

Phase 3d will add scheduled/automated refresh across both sources.

---

## Contributing

This codebase is maintained with Claude Code. When proposing changes:
- Each new metric or factor should be added as a standalone function in `transform.py` or `score.py`
- All new functions require a corresponding unit test in `/tests/`
- Weight changes belong in `config.yaml`, not in Python code
- The `notebooks/validation.ipynb` should be re-run after any change to transform or scoring logic
