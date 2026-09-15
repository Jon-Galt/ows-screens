# OWS Short Screen

A Python quantitative stock screening tool for identifying short candidates across a broad equity
universe (~1,300 stocks), with a Streamlit UI. Rebuilt from a Bloomberg/Excel workflow into a
maintainable, extensible codebase.

**This file is a short orientation page, not the documentation.** It was retired to this form on
2026-09-13 (Driver ruling) because its architecture and phase sections had become a second,
unmaintained copy of material that lives elsewhere and is kept current. Read the real thing:

| For | Read |
|---|---|
| Architecture rules, file layout, commands, current status | `CLAUDE.md` |
| Known issues and known implementation decisions | `docs/KNOWN_ISSUES.md` |
| The live traps, T1–T42 — read before scoping or reviewing | `docs/TRAPS.md` |
| Recurring bug patterns in `transform.py` / `score.py` | `docs/BUG_PATTERNS.md` |
| What each phase set out to do, and closed-phase narrative | `PHASE_HISTORY.md` |
| Live options, open decisions, and the build queue | `PM_HANDOFF.md` |

---

## What it does

Ingests fundamental and market data, calculates derived metrics, percentile-ranks each metric across
the full universe, and produces a weighted composite score to surface the highest-priority short
candidates for further research. Identical logic and factor weights to the Excel screener it
replaces.

There are several screens today, of two kinds — `quant_composite` (scored or unscored) and `curated`.
**`config.yaml`'s `screens` block is the authority on which screens exist**; `CLAUDE.md`'s Current
Status says what each one is.

### Factor categories

The short screen's factor model. **`config.yaml`'s `screens.short_screen.factor_categories` block is
the single source of truth for the taxonomy** (it is nested under the screen, not top-level) — this
table is description, not specification.

| Category | Description |
|---|---|
| **Valuation** | Absolute and relative Price/Sales and FCF Yield vs. historical averages |
| **Growth** | Revenue growth deceleration and forward vs. historical CAGR differential |
| **Profitability** | NTM gross margin and EBIT margin vs. 3-year historical averages |
| **Balance Sheet** | Leverage, debt coverage, refinancing risk, and liquidity runway |
| **Cash Flow** | FCF conversion, accruals quality, working capital trends, dilution |
| **Non-GAAP** | EBIT and EPS adjustment ratios (Non-GAAP vs. GAAP) |
| **Sentiment** | Short interest and analyst rating distribution |
| **M-Score** | Beneish earnings manipulation model — displayed separately, **never in the composite** |

---

## How the data flows

Storage is scoped per screen: each screen owns its own physical tables, named `<stage>__<screen_id>`
(e.g. `raw_data__short_screen`, `curated_data__cyclicals`), so screens with different column shapes
never share a table. A `screens` registry and a shared `screen_membership(screen_id, ticker)` table
sit alongside them and drive the cross-screen overlap view.

Quant screens flow through four sequential stages:

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

Curated screens have nothing to rank or compose, so they flow through two:

```
Raw CSV upload (data/uploads/<screen_id>/, exactly one file)
        ↓
[ curated_ingest.py ]  →  curated_data__<screen_id>  (SQLite)
        ↓
      [ app.py       ]  →  Streamlit UI + Excel/CSV export
```

Each screen's pipeline runs independently of every other screen's tables, and calling the wrong stage
against the wrong screen type (e.g. `score.py` against a curated screen) fails with `ScreenTypeError`
rather than doing something undefined.

---

## Quick start

**Prerequisites: Python 3.10+** (the codebase uses `X | None` syntax) and pip. The project's `.venv`
is 3.11.

```bash
pip install -r requirements.txt
```

Drop each screen's export into its own folder under `data/uploads/<screen_id>/` — exactly one file
per folder; the loaders cannot tell screens apart by content, so more than one is rejected.

```bash
python src/refresh.py          # one-command refresh across every registered screen
streamlit run src/app.py       # launch the UI, then pick a screen in the sidebar
pytest tests/                  # run the suite
```

A curated screen can also be ingested on its own:

```bash
python -c "from src.curated_ingest import ingest_curated; ingest_curated('cyclicals')"
```

**`CLAUDE.md`'s Commands section is the full and current list**, including `refresh.py`'s
`--screen` / `--dry-run` / `--force` / `--history` flags and the lint command. Factor weights are
edited in `config.yaml`, never in Python.

---

## Where the data comes from

Both sources are manual exports; automation here means one local command, not a scheduler.

- **Quant screens** — Bloomberg, via CSV/Excel export. Required fields and column naming are
  documented in `src/ingest.py` (and `src/rsi_ingest.py`, `src/transcript_ingest.py` for the screens
  with their own loaders).
- **Curated screens** — Canary, via CSV export, one per screen. Schema and cleaning rules are in
  `src/curated_ingest.py`. Canary's narrative rationale and risk scores are not available through its
  API, so this export-based refresh does not go away even if API sourcing is added later.

---

## Why it is built this way

**SQLite** needs no server, lives as a single gitignored file in the repo, and is readable by pandas
directly. It can be swapped for Postgres later if multi-user access ever matters.

**Streamlit** is Python-only, needs no frontend work, and gives interactive tables, filters and
downloads out of the box — the fastest path to a usable UI for a small team.

**Weights in `config.yaml`** because factor weights are the thing that changes most between research
iterations. Keeping them out of Python means they can be adjusted, version-controlled and reviewed
independently of the calculation logic.

**Separate ingest / transform / score** because each stage has a different failure mode and a
different reason to be rerun. A new field in the export touches `ingest.py` and `transform.py`; a
changed weight reruns `score.py` alone.

---

## Contributing

This codebase is maintained with Claude Code. **Read `CLAUDE.md` first** — it carries the mandatory
Architecture Rules, the Worker rules, and the team workflow. In short: new metrics are standalone
functions in `transform.py` or `score.py`, every new function needs a unit test in `tests/`, and no
unit test may depend on gitignored data.
