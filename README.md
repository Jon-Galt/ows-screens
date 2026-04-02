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
  uploads/            ← Drop raw CSV/Excel exports here
  screener.db         ← SQLite database (auto-generated, not committed to git)

/src/
  ingest.py           ← Reads uploaded files and loads into SQLite
  transform.py        ← Calculates all derived metrics
  score.py            ← Percentile ranking and weighted composite score
  app.py              ← Streamlit web UI

/tests/
  test_transform.py   ← Unit tests for all transform functions
  test_score.py       ← Unit tests for ranking and scoring logic

/notebooks/
  OWS Short Screen (March 2026).xlsx  ← Original Excel file (kept for validation)
  validation.ipynb      ← Side-by-side comparison of Excel vs. Python outputs

config.yaml           ← Factor weights, universe settings, thresholds
requirements.txt      ← Python dependencies
.gitignore
README.md
```

---

## Data Architecture

Data flows through four sequential layers, each writing to a named SQLite table:

```
Raw CSV/Excel upload
        ↓
   [ ingest.py ]  →  raw_data  (SQLite)
        ↓
[ transform.py ]  →  transformed_data  (SQLite)
        ↓
   [ score.py  ]  →  scored_data  (SQLite)
        ↓
   [ app.py    ]  →  Streamlit UI + Excel/CSV export
```

This separation means each layer can be run independently, tested in isolation, and updated without touching other parts of the pipeline.

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

1. Drop your Bloomberg CSV/Excel export into `/data/uploads/`
2. Run the pipeline:

```bash
python src/ingest.py
python src/transform.py
python src/score.py
```

3. Launch the UI:

```bash
streamlit run src/app.py
```

### Running Tests

```bash
pytest tests/
```

---

## Configuration

Factor weights and universe settings are controlled in `config.yaml`. Edit this file to adjust the composite score weighting without touching any Python code.

```yaml
universe:
  name: "OWS Short Screen"
  as_of: "2026-03"

factor_weights:
  # Valuation (sum: 1.0)
  abs_ps_factor: 0.25
  rel_ps_factor: 0.25
  abs_fcf_factor: 0.25
  rel_fcf_factor: 0.25

  # Growth (sum: 1.0)
  decel_factor: 0.5
  accel_factor: 0.5

  # Profitability (sum: 1.0)
  gm_factor: 0.5
  ebit_factor: 0.5

  # Balance Sheet (sum: 1.0)
  debt_ebitda_factor: 0.2
  debt_sales_factor: 0.2
  debt_ev_factor: 0.2
  refi_risk_factor: 0.2
  liquidity_risk_factor: 0.2

  # Cash Flow (sum: 1.0, each weight: 1/7)
  fcf_conv_factor: 0.142857
  accrual_factor: 0.142857
  dso_factor: 0.142857
  dio_factor: 0.142857
  dpo_factor: 0.142857
  def_rev_factor: 0.142857
  dilution_factor: 0.142857

  # Non-GAAP (sum: 1.0)
  ebit_adj_factor: 0.5
  eps_adj_factor: 0.5

  # Sentiment (sum: 1.0)
  short_int_factor: 0.5
  ratings_factor: 0.5

scoring:
  mscore_manipulation_threshold: -2.22
  nan_default_standard: 0.5   # Default percentile for most missing factors
  nan_default_balance_sheet: 0.0  # Default for balance sheet / liquidity factors
```

---

## Development Phases

This project is being built incrementally. Each phase has a defined scope and acceptance criteria before moving to the next.

---

### Phase 1 — Replication (Current)

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

### Phase 2 — Web UI

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

### Phase 3 — Automation

**Goal:** Reduce manual effort in the monthly refresh cycle.

**Scope:**
- Standardized CSV export template and validation — confirm required columns are present and formatted correctly before ingesting
- GitHub Actions workflow to run the pipeline on a schedule or on file upload
- Data validation report generated at each run — flags missing values, outliers, and column count mismatches vs. prior month
- Run history log — store each monthly `scored_data` snapshot with a date stamp for trend tracking

**Acceptance criteria:**
- Pipeline runs end-to-end without manual intervention after file drop
- Validation report catches at least: missing required columns, >10% NaN rate in key fields, universe size change >5%

---

### Phase 4 — Expanded Functionality

**Goal:** Add analytical depth and new data sources beyond the original Excel scope.

**Potential scope (to be prioritized):**
- Historical score tracking — chart how a stock's composite score has changed over time
- Factor-level trend analysis — identify stocks with rapidly improving or deteriorating scores
- Sector-relative scoring — percentile rank within sector in addition to full-universe ranking
- New data source integration — additional fundamental or alternative data via API or CSV
- Watchlist and annotation — flag specific stocks with notes for team reference
- Backtesting module — assess whether high composite scores have historically predicted underperformance

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

Raw data is currently sourced from Bloomberg via manual CSV/Excel export. The required fields and column naming conventions are documented in `src/ingest.py`. Future phases may add direct API ingestion to automate this step.

---

## Contributing

This codebase is maintained with Claude Code. When proposing changes:
- Each new metric or factor should be added as a standalone function in `transform.py` or `score.py`
- All new functions require a corresponding unit test in `/tests/`
- Weight changes belong in `config.yaml`, not in Python code
- The `notebooks/validation.ipynb` should be re-run after any change to transform or scoring logic
