# CLAUDE.md

## Project
OWS Short Screen — a Python-based quantitative stock screening tool for identifying short candidates across a broad equity universe (~1,300 stocks). Rebuilt from a Bloomberg/Excel workflow into a maintainable, extensible codebase. See README.md for full architecture overview.

## Current Status

- **Phase 1** in progress: data ingestion, metric calculations, percentile ranking, composite scoring, unit tests
- See `README.md §Development Phases` for full phase definitions and acceptance criteria

## Commands
- Run all tests: `pytest tests/ -v`
- Run specific test file: `pytest tests/test_transform.py -v`
- Run full pipeline: `python src/ingest.py && python src/transform.py && python src/score.py`
- Launch UI: `streamlit run src/app.py`
- Lint: `ruff check src/ tests/`

## File Layout
- `src/ingest.py` — reads CSV/Excel uploads from `/data/uploads/`, maps raw Bloomberg column names to snake_case Python fields, coerces types, loads into SQLite `raw_data` table
- `src/transform.py` — reads `raw_data`, computes all derived metrics as individual named functions, writes to `transformed_data` table
- `src/score.py` — reads `transformed_data`, applies percentile ranking per factor (with correct ranking direction), computes weighted composite score and M-Score, writes to `scored_data` table
- `src/app.py` — Streamlit web UI: filterable/sortable table, sector/industry filters, market cap and score range sliders, M-Score flags, stock drill-down, Excel/CSV export
- `tests/test_transform.py` — unit tests for every transform function
- `tests/test_score.py` — unit tests for ranking logic, direction, defaults, and composite scoring
- `config.yaml` — factor weights, NaN fallback defaults, M-Score threshold, universe metadata
- `data/uploads/` — drop Bloomberg CSV/Excel exports here (not committed to git)
- `data/screener.db` — SQLite database, auto-generated (not committed to git)
- `notebooks/excel_reference.xlsx` — original Excel file kept for validation
- `notebooks/validation.ipynb` — side-by-side comparison of Python vs. Excel outputs

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
