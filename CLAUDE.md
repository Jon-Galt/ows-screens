# CLAUDE.md

## Project
OWS Short Screen — a Python-based quantitative stock screening tool for identifying short candidates across a broad equity universe (~1,300 stocks). Rebuilt from a Bloomberg/Excel workflow into a maintainable, extensible codebase. See README.md for full architecture overview.

## Current Status

- **Phase 1** complete: data ingestion, metric calculations, percentile ranking, composite scoring, unit tests
- **Phase 2** complete: Streamlit web UI (`src/app.py`)
- **Phase 3a** complete: multi-screen architecture (foundation)
- **Phase 3b** complete: onboarded the 4 curated screens (Cyclicals, Competition, Structural, Management Comp) from standalone Canary CSV exports
- **Phase 3c** complete: onboarded Rising Short Interest, the second `quant_composite` screen (ingest + transform only — it has no factor model, so no scoring yet)
- See `README.md §Development Phases` and `PHASE3_PLAN.md` for the full roadmap (3a–3e, then Phase 4)

## Commands
- Run all tests: `pytest tests/ -v`
- Run specific test file: `pytest tests/test_transform.py -v`
- Run full pipeline (short_screen): `python src/ingest.py && python src/transform.py && python src/score.py`
- Ingest a curated screen: `python -c "from src.curated_ingest import ingest_curated; ingest_curated('cyclicals')"` (no automated CLI yet — that's Phase 3d)
- Launch UI: `streamlit run src/app.py`
- Lint: `ruff check src/ tests/`

## File Layout
- `src/ingest.py` — Bloomberg/quant loader (short_screen, RSI): reads the one export file in `data/uploads/<screen_id>/` per `SCREEN_INGEST_CONFIGS`, writes `raw_data__<screen_id>`. Rejects curated screen_ids.
- `src/rsi_ingest.py` — Rising Short Interest's own loader: trims the export's preamble/count-row/footer, fixes the ticker-extraction bug, writes `raw_data__rising_short_interest`.
- `src/curated_ingest.py` — shared loader for the 4 curated screens: unwraps Canary's quoted numerics, unit-converts, parses the packed `scores` field, writes `curated_data__<screen_id>`.
- `src/transform.py` — per-screen derived-metric functions dispatched via `SCREEN_TRANSFORM_FUNCS`, writes `transformed_data__<screen_id>`. Rejects curated screen_ids.
- `src/score.py` — percentile-ranks and composite-scores a screen's transformed data via `get_screen_config`, writes `scored_data__<screen_id>`. Rejects curated screen_ids and quant_composite screens with no `factor_weights`.
- `src/config.py` — `load_config()`/`CONFIG_PATH` for `config.yaml`; `get_screen_type()`/`ScreenTypeError`, the shared type-dispatch guard used by ingest/transform/score.
- `src/loaders.py` — generic upload-file IO shared by all ingest paths: `read_upload`, `validate_columns`, `log_summary`, `find_single_upload_file`/`UploadFileError`.
- `src/db.py` — multi-screen storage helpers: `table_name(stage, screen_id)`, `sync_screens_registry()`, `replace_screen_rows()` (the shared `screen_membership` table only — per-screen tables use plain `to_sql(if_exists="replace")`).
- `src/app.py` — Streamlit UI: sidebar screen selector plus three rendering paths (scored quant_composite, unscored quant_composite, curated).
- `tests/test_transform.py`, `tests/test_score.py` — unit tests for short_screen's calc/ranking/scoring functions.
- `tests/test_schema.py` — storage helpers, per-screen-type dispatch guards, and the cross-screen pipeline-isolation regression lock.
- `tests/test_curated_ingest.py`, `tests/test_rsi_ingest.py`, `tests/test_loaders.py` — unit + end-to-end tests for the curated loader, the RSI loader/transforms, and the shared upload-file discipline.
- `config.yaml` — per-screen config under `screens`: `display_name`, `type`, `universe`, and — `quant_composite` screens only, and only if scored — `factor_weights`/`scoring`.
- `data/uploads/<screen_id>/` — drop each screen's export here (gitignored); every ingest path enforces exactly one file per folder.
- `data/screener.db` — SQLite database (gitignored): `screens` registry, shared `screen_membership(screen_id, ticker)`, and each screen's own `raw_data__*`/`transformed_data__*`/`scored_data__*` (quant) or `curated_data__*` (curated) tables.
- `notebooks/OWS Short Screen (March 2026).xlsx` — original Excel file kept for validation.
- `notebooks/validation.ipynb` — Python vs. Excel comparison for short_screen only (curated/RSI have no percentile output to validate against Excel).

## Architecture Rules (mandatory)

1. **Calculation functions have zero web and database dependencies.** Functions in `transform.py` and `score.py` operate on pandas DataFrames only — they do not import SQLAlchemy, Streamlit, or anything from the web layer. Database reads and writes happen in the main execution block, not inside calculation functions.

2. **Percentage fields are stored and used as decimals throughout.** `0.05` means 5%. This convention applies in the raw data, SQLite storage, calculations, and the UI display layer (which formats for display only). Never multiply a percentage field by 100 in a calculation.

3. **Every calculation function handles missing data explicitly.** Use `pd.to_numeric(..., errors='coerce')` for inputs. Return `NaN` rather than raising exceptions. Never let a single bad row crash the pipeline.

4. **The `"#N/A N/A"` string is Bloomberg's missing data marker.** It must be converted to `NaN` on ingest for all fields — except `available_loc`, where it should be treated as `0` (no available credit line). This conversion happens in `ingest.py` before any data reaches the database.

5. **Percentile ranking must match Excel's `PERCENTRANK.INC` exactly.** Use `scipy.stats.percentileofscore(arr, val, kind='rank') / 100`. This produces values between 0 and 1 inclusive. Do not use pandas rank or any other method.

6. **Ranking direction is explicit per factor.** Some factors score higher when the raw metric is higher (more bearish). Others use `1 - percentile` because a lower raw value is worse for the short thesis. The correct direction for each factor is defined in `score.py` and documented in `config.yaml`. Never infer direction — always check the spec.

7. **NaN fallback defaults are factor-dependent.** Most factors default to `0.5` when a stock has insufficient data (marked `*`). Balance sheet and liquidity factors default to `0.0` (marked `**`) — absence of data means no balance sheet concern. These defaults are defined in `config.yaml` and must not be hardcoded in Python.

8. **The M-Score is never included in the composite overall score.** It is calculated separately and displayed as a standalone indicator. The manipulation threshold is `> -2.22` and is defined in `config.yaml`.

9. **Factor weights live in `config.yaml`, not in Python code.** `score.py` reads weights at runtime. Changing a weight requires editing only `config.yaml`.

10. **The pipeline runs sequentially and each step is independently re-runnable.** `ingest.py` → `transform.py` → `score.py`. Each reads from the prior step's SQLite table and writes to the next. Rerunning any step overwrites its output table cleanly.

## Team Workflow

**Roles**: Driver (Steve) · Product Manager (Claude chat) · Inspector (Claude Code) · Worker (Claude Code) · Reviewer (Claude Code)

**Process**: Driver + PM decide priorities/direction → PM defines the scoped phase → Inspector checks the repo and reports facts → PM writes the Worker prompt → Worker plans and implements within scope → PM reviews against scope and acceptance criteria → Reviewer validates material phases → merge → fresh session for next phase

**Principles**: narrow phases · boring modular architecture · no broad refactors · explicit validation before moving on

**Plan review rule**: PM either approves the Worker's plan as-is, or sends a revision prompt back. Never both — no "approved with changes." If revisions are needed, the Worker resubmits a new plan.

**Role responsibilities**:
- **Driver**: sets goals, priorities, risk tolerance, final decisions
- **Product Manager**: scope, acceptance criteria, architecture judgment, sequencing, and the final Worker prompts (written as clean, copy/paste-ready codeblocks). No codebase claims without Inspector grounding — the PM has no direct repo access, so "I think it works like X" is never acceptable; verify via an Inspector prompt or ask the Driver
- **Inspector**: read-only — inspects the repo, traces logic across files, and reports exact files/functions/classes/risks/ambiguities to the PM. Does not own scope, make undocumented assumptions, design architecture, write code, or declare a phase done — that always requires passing `pytest` output plus explicit PM sign-off against acceptance criteria, not code-reading alone
- **Worker**: builds the scoped change and adds/updates tests. If the prompt says "propose a plan," propose a plan — don't write code yet. Do not refactor, rename, or touch files/signatures outside the current phase's scope; flag surprises instead of silently changing approach
- **Reviewer**: fresh review for bugs, regressions, edge cases, scope creep, and Architecture Rule compliance — run `pytest` first (full output in the summary)

### Worker Rules — error handling & testing
- Every `try/except` must handle a specific known failure mode, or re-raise after adding context. No bare `except:`/`except Exception:` that silently continues.
- Write tests before or alongside implementation, named `tests/test_<module>.py`. One clear assertion per test beats many weak ones — if you could delete the implementation and the test would still pass, the test is broken. Use small (5–10 row) synthetic DataFrames with known inputs/outputs, and cover edge cases: `NaN` inputs, zero denominators, negative values, the `"#N/A N/A"` string, and all-missing-data rows.
- **Verification efficiency**: run the full verification chain (snapshot comparison, validation notebook, Streamlit check) ONCE, at the end of the phase, and report it once. Do not re-run a check after every intermediate change. Re-run a specific check mid-phase only when you've changed something that could plausibly break that specific thing, and say why.

### Before You Report Done (mandatory checklist)
- [ ] Run `pytest`. ALL tests must pass. Include the full output in your summary.
- [ ] New logic has tests: happy path, one edge case, one boundary condition.
- [ ] New functions have docstrings covering inputs, outputs, and NaN/edge-case behavior.
- [ ] Imports are correct and minimal — no unused imports, no circular imports.
- [ ] Confirmed compliance with Architecture Rules 1–10 above (no web/db imports in calc functions, decimal convention, explicit NaN handling, config-sourced weights/defaults, M-Score excluded from the composite, `PERCENTRANK.INC`-compatible ranking).
- [ ] No broad refactors outside the scope of the current task.

Recurring bug patterns worth re-reading before touching `transform.py`/`score.py`: see `docs/BUG_PATTERNS.md`.

## Known Issues (do not fix unless explicitly scoped into current phase)

### Functional
- Universe is defined implicitly by whatever rows appear in the uploaded file. Explicit universe management (add/remove tickers, maintain a master list) is deferred.

### Cleanup
- none

### Test hardening
- none

## Known Implementation Decisions

- **Percentile ranking uses `kind='strict'`**, not `kind='rank'`. Although
  `PERCENTRANK.INC` documentation suggests average-rank behavior, validation
  against the March 2026 reference file confirmed that `kind='strict'` (minimum
  rank for ties) matches Excel output exactly for all 24 factors. This matters
  for any column with many tied values (ratings, maturity, Non-GAAP ratios).
  Do not change this without re-running validation.ipynb.
