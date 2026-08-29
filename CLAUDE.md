# CLAUDE.md

## Project
OWS Short Screen — a Python-based quantitative stock screening tool for identifying short candidates across a broad equity universe (~1,300 stocks). Rebuilt from a Bloomberg/Excel workflow into a maintainable, extensible codebase. See README.md for full architecture overview.

## Current Status

- **Phase 1** complete: data ingestion, metric calculations, percentile ranking, composite scoring, unit tests
- **Phase 2** complete: Streamlit web UI (`src/app.py`)
- **Phase 3a** complete: multi-screen architecture (foundation) — the pipeline is now scoped by `screen_id`, with the existing Short Screen migrated onto it as the only populated screen.
- **Phase 3b** complete: onboarded the 4 curated screens (Cyclicals, Competition, Structural, Management Comp) from standalone Canary CSV exports — no more manual workbook consolidation. Type-aware dispatch keeps curated screens out of the quant transform/score stages. See `PHASE3_PLAN.md` and `README.md §Development Phases` for the full roadmap (3a–3f, then Phase 4).
- See `README.md §Development Phases` for full phase definitions and acceptance criteria

## Commands
- Run all tests: `pytest tests/ -v`
- Run specific test file: `pytest tests/test_transform.py -v`
- Run full pipeline (short_screen): `python src/ingest.py && python src/transform.py && python src/score.py`
- Ingest a curated screen: `python -c "from src.curated_ingest import ingest_curated; ingest_curated('cyclicals')"` (no automated CLI yet — that's Phase 3d)
- Launch UI: `streamlit run src/app.py`
- Lint: `ruff check src/ tests/`

## File Layout
- `src/ingest.py` — the quant/Bloomberg loader: reads CSV/Excel uploads from `/data/uploads/<screen_id>/`, using that screen's ingest config (sheet name, Bloomberg column map, required columns) from `SCREEN_INGEST_CONFIGS`, coerces types, loads into that screen's `raw_data__<screen_id>` SQLite table. Rejects curated `screen_id`s with a clear `ScreenTypeError`
- `src/curated_ingest.py` — the Canary curated-screen loader, shared by all four curated screens (identical 11-column schema): strips Canary's quote-wrapped numeric strings, unit-converts, parses the packed `scores` field into three numeric columns (plus retains the raw string), and writes to `curated_data__<screen_id>`. `_find_single_upload_file()` requires exactly one `.csv` in the upload folder — curated screens have no column identifying which screen an export belongs to, so a misfile or a stray second file is a loud, named error, not a silent concat or misread. `_log_curated_summary()` logs row/ticker counts and a sector/ticker sample so a misfile is visible in the run output too
- `src/transform.py` — reads a screen's `raw_data__<screen_id>`, computes all derived metrics as individual named functions, writes to `transformed_data__<screen_id>`. Rejects curated `screen_id`s with `ScreenTypeError` — there is no transform stage for curated screens
- `src/score.py` — reads a screen's `transformed_data__<screen_id>`, applies percentile ranking per factor (with correct ranking direction) using that screen's config sub-dict (`get_screen_config`), computes weighted composite score and M-Score, writes to `scored_data__<screen_id>`. Rejects curated `screen_id`s with `ScreenTypeError` — no ranking, no composite, no M-Score for curated screens
- `src/config.py` — `load_config()` / `CONFIG_PATH`: loads the full multi-screen `config.yaml`. `get_screen_type()` / `ScreenTypeError`: the shared type-dispatch guard used by `ingest.py`, `curated_ingest.py`, `transform.py`, and `score.py` — lives here (not in `score.py`'s `get_screen_config`) so `transform.py`/`ingest.py` never have to import from `score.py`
- `src/loaders.py` — generic file-reading helpers with no source-specific logic (`read_upload`, `validate_columns`, `log_summary`), shared by `ingest.py` and `curated_ingest.py`. Moved out of `ingest.py` in 3b once a second consumer needed them, so `ingest.py` doesn't become the codebase's de facto shared IO module while being named and structured as the Bloomberg-specific loader
- `src/db.py` — multi-screen storage helpers: `table_name(stage, screen_id)` derives each screen's physical table name (e.g. `raw_data__short_screen`, `curated_data__cyclicals`); `sync_screens_registry()` idempotently rewrites the `screens` registry table from `config.yaml`; `replace_screen_rows()` does a screen-scoped delete+append, used only for the shared, fixed-shape `screen_membership` table (never for the per-screen `raw_data__*`/`transformed_data__*`/`scored_data__*`/`curated_data__*` tables, which use an ordinary `to_sql(if_exists="replace")` on their own table)
- `src/app.py` — Streamlit web UI with a sidebar screen selector (reads the `screens` registry). `quant_composite` screens (short_screen) get the original filterable table, sector/industry filters, market cap and score sliders, M-Score flags, and factor-chart drill-down — all unchanged. `curated` screens get a separate rendering path: sector/market-cap filters only, a table of identity/market fields plus the three risk scores, and a drill-down showing the full narrative rationale — no factor chart, no M-Score, since there are none
- `tests/test_transform.py` — unit tests for every transform function
- `tests/test_score.py` — unit tests for ranking logic, direction, defaults, composite scoring, and `get_screen_config`
- `tests/test_schema.py` — unit tests for the multi-screen storage helpers, the type-aware dispatch guards (`ingest`/`ingest_curated`/`transform`/`score` each rejecting the wrong screen type), and an end-to-end regression lock proving one screen's pipeline run cannot alter another screen's tables
- `tests/test_curated_ingest.py` — unit tests for the curated loader: quote-stripping, each unit conversion, `scores` parsing (including malformed input), the single-file upload-folder guard, the summary logging, and one end-to-end curated ingest against a small fixture
- `config.yaml` — per-screen config keyed by `screen_id` under a top-level `screens` block: `display_name`, `type` (`quant_composite` or `curated`), `universe`, and — `quant_composite` screens only — `factor_weights` and the `scoring` block (NaN fallback defaults, M-Score threshold). Curated screens have neither
- `data/uploads/<screen_id>/` — drop each screen's CSV/Excel exports here (not committed to git). Curated screens: exactly one `.csv` file per folder — see `curated_ingest.py`'s upload-folder guard
- `data/screener.db` — SQLite database, auto-generated (not committed to git). Holds a `screens` registry table, a shared `screen_membership(screen_id, ticker)` table, and each screen's own `raw_data__<screen_id>` / `transformed_data__<screen_id>` / `scored_data__<screen_id>` (quant) or `curated_data__<screen_id>` (curated) tables
- `notebooks/OWS Short Screen (March 2026).xlsx` — original Excel file kept for validation
- `notebooks/validation.ipynb` — side-by-side comparison of Python vs. Excel outputs for the Short Screen (quant pipeline only — curated screens have no percentile/composite output to validate against Excel)

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

**Process**: Driver + PM decide priorities/direction → PM defines the scoped phase → Inspector checks the actual repo and reports relevant files, patterns, dependencies, and risks → PM writes the Worker prompt → Worker plans and implements within scope → PM reviews result against scope and acceptance criteria → Reviewer validates material phases → merge → fresh session for next phase

**Prompt rule**: The Product Manager writes the final Worker prompts as clean, copy/paste-ready codeblocks. The Inspector does not own scope, does not make undocumented assumptions, and does not replace the PM.

**Principles**: narrow phases · boring modular architecture · no broad refactors · explicit validation before moving on

**Role split**:
- **Driver**: sets goals, priorities, risk tolerance, and final decisions
- **Product Manager**: scope, acceptance criteria, architecture judgment, sequencing, tradeoff decisions, and final Worker prompts
- **Inspector**: repo-grounded inspection only — reads code, identifies files/functions/patterns/risks, and reports facts to the PM
- **Worker**: builds the scoped change and adds/updates tests
- **Reviewer**: fresh review for bugs, regressions, edge cases, and scope creep

**Rule of thumb**: use PM for judgment and scoping; use Inspector for factual grounding; use Worker to build; use Reviewer to review

**Plan review rule**: PM either approves the Worker's plan as-is, or provides a revision prompt to send back. Never both. No "approved with changes." If revisions are needed, the Worker resubmits a new plan.

### PM Rules

**No assumptions.** The PM must not assume facts about the codebase, data behavior, column naming, config defaults, or runtime behavior. If the PM doesn't know, ask the Driver or send an Inspector prompt. "I think it works like X" is never acceptable — verify or ask.

**No codebase claims without Inspector grounding.** The PM has no direct access to the repo. Any PM statement about what the code does, what a function returns, or how modules interact must come from a recent Inspector report — not from memory or prior phases.

## Inspector Rules

**Inspector is not the Product Manager, not the Worker, and not the Reviewer.**

Inspector must not:
- define final scope
- make architecture decisions unless explicitly asked by the PM
- invent undocumented column names, schema details, or behavioral assumptions
- write the final Worker prompt unless the PM explicitly asks for draft material only
- write code, edit files, or act as Worker
- declare a phase done

### Inspector allowed actions

Inspector may:
- inspect the repo and read code
- trace logic across files
- identify exact files and functions involved
- describe existing patterns, constraints, and dependencies
- identify risks, ambiguities, and likely touchpoints
- answer targeted factual questions from the PM

### Inspector required output format

When asked to inspect a phase, Inspector should return:
1. Current status and whether the requested slice appears to be the right next step
2. Exact files likely to change
3. Exact functions, modules, and classes likely involved
4. Factual risks, ambiguities, and assumptions needing PM judgment
5. Anything discovered in the repo that materially narrows or changes scope

### Completion rule

Inspector should NEVER declare a phase complete based on reading code alone. Completion requires:
- all tests passing (`pytest` output included)
- acceptance criteria from `README.md` explicitly checked off
- PM sign-off

## Worker Rules

### Scope discipline
- Read the Worker prompt carefully. Do exactly what it says.
- If the prompt says "propose a plan," propose a plan — do not start coding.
- If the prompt says "implement," implement — do not re-scope.
- If you discover something unexpected, flag it and ask — do not silently change the approach.
- Do not refactor, rename, or "improve" code outside the scope of the current phase.
- Do not touch function signatures, file structure, or imports in modules you were not asked to change.

### On error handling
- Every `try/except` must either handle a specific known failure mode, or re-raise after recording context.
- No bare `except:` or `except Exception:` that silently continues.

### On testing
- Each test should be able to fail meaningfully — if you can delete the implementation and the test still passes, the test is broken.
- Prefer one clear assertion per test over many weak assertions.
- Use small synthetic DataFrames (5–10 rows) with known inputs and expected outputs.

### Before You Report Done (mandatory checklist)
- [ ] Run `pytest`. ALL tests must pass. Include the full output in your summary.
- [ ] If you added new logic, you wrote tests for it. Aim for: happy path, one edge case, one boundary condition.
- [ ] All new functions have docstrings explaining inputs, outputs, and NaN/edge-case behavior.
- [ ] Imports are correct and minimal — no unused imports, no circular imports.
- [ ] `transform.py` and `score.py` have no imports from `streamlit`, `fastapi`, or any web package.
- [ ] Percentage fields are used as decimals (0.05 = 5%) throughout — no implicit multiply/divide by 100.
- [ ] `"#N/A N/A"` strings are converted to `NaN` (or `0` for `available_loc`) in `ingest.py` before any downstream use.
- [ ] Percentile ranking uses `scipy.stats.percentileofscore(..., kind='rank') / 100` — no substitutions.
- [ ] NaN fallback defaults come from `config.yaml`, not hardcoded values.
- [ ] M-Score is computed but not included in `overall_score`.
- [ ] No broad refactors outside the scope of the current task.

### Writing Tests
- Write tests FIRST when possible — define expected behavior before implementing.
- Test file naming: `tests/test_<module>.py`
- Use fixtures for database sessions and synthetic DataFrames.
- Always include edge cases: `NaN` inputs, zero denominators, negative values, the `"#N/A N/A"` string in raw data, stocks with all missing data (should use fallback default, not crash).

### Recurring bug patterns (take these seriously)

1. **Ranking direction inversion.** Several factors use `1 - percentile` because lower raw values are worse for the short thesis. Applying straight percentile rank to these factors will score high-FCF-yield companies as short candidates. Always verify direction against the factor table in `score.py`.

2. **Decimal vs. display unit confusion.** All percentage fields (margins, yields, growth rates, short interest) are stored as decimals. Treat them as decimals in all calculations. If a result looks 100x too large or small, check for an accidental multiply/divide by 100.

3. **Bloomberg `"#N/A N/A"` strings surviving into calculations.** If `ingest.py` fails to catch one of these strings and it reaches `transform.py`, `pd.to_numeric` will coerce it to `NaN` — but only if `errors='coerce'` is used. Always use `errors='coerce'` on any column that could contain this string.

4. **NaN propagation in chained calculations.** If an intermediate metric (e.g. `dsos_t3m`) is `NaN`, any downstream metric that depends on it (e.g. `dso_pct_change`) will also be `NaN`. This is correct behavior — do not mask it with fillna before the final fallback step in `score.py`.

5. **Division by zero in ratio calculations.** Revenue, EBITDA, COGS, and share counts can all be zero. Always check denominators explicitly with `np.where` or `pd.Series.where` before dividing, and return `NaN` rather than raising.

6. **PERCENTRANK.INC range behavior.** Excel's `PERCENTRANK.INC` includes both 0 and 1 as possible outputs. `scipy.stats.percentileofscore(..., kind='rank') / 100` matches this. `kind='weak'` and `kind='strict'` do not — never substitute them.

## Reviewer Rules
- Run `pip install -r requirements.txt` before starting your review, then run `pytest` and confirm all tests pass. Include full output in your summary.
- Check test coverage for new and changed code — are edge cases covered?
- Verify error handling: what happens when inputs are malformed, a Bloomberg string survives ingest, or a zero denominator appears?
- Confirm adherence to the architecture rules above — especially the no-web-imports rule for `transform.py` and `score.py`, and the decimal convention.
- Confirm no scope creep beyond the stated phase objective.
- Verify percentile ranking uses `PERCENTRANK.INC`-compatible method throughout.
- Verify NaN fallback defaults are read from `config.yaml`, not hardcoded.

## Known Issues (do not fix unless explicitly scoped into current phase)

These are documented, deferred issues. Do not attempt to fix them as part of unrelated work.

### Functional
- Data ingestion is currently manual (CSV/Excel drop into `/data/uploads/`). Automated ingestion via Bloomberg API is deferred to Phase 3.
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
