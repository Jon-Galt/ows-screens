# CLAUDE.md

## Project
OWS Short Screen — a Python-based quantitative stock screening tool for identifying short candidates across a broad equity universe (~1,300 stocks). Rebuilt from a Bloomberg/Excel workflow into a maintainable, extensible codebase. See README.md for full architecture overview.

## Current Status

- **Phase 1** complete: data ingestion, metric calculations, percentile ranking, composite scoring, unit tests
- **Phase 2** complete: Streamlit web UI (`src/app.py`)
- **Phase 3a** complete: multi-screen architecture (foundation)
- **Phase 3b** complete: onboarded the 4 curated screens (Cyclicals, Competition, Structural, Management Comp) from standalone Canary CSV exports
- **Phase 3c** complete: onboarded Rising Short Interest, the second `quant_composite` screen (ingest + transform only — it has no factor model, so no scoring yet)
- **Phase 3d Part 1** complete: cross-screen overlap view (`src/overlap.py`)
- **Phase 3d Part 2a** complete: one-command refresh across all six screens, gated by pre-write validation (`src/refresh.py`, `src/validate.py`)
- **Phase 3d Part 2b** complete: run history and per-run data snapshots on top of the refresh orchestrator (`src/history.py`; `refresh_runs`/`refresh_screen_runs`/`refresh_snapshots` in `data/screener.db`, append-only) — the first brick of a future backtest dataset
- **Phase 3d Part 2c** complete: replaced the universe-size validation check (assumed stable ticker counts, which heavy month-to-month turnover violates) with a composition/misfile check comparing incoming tickers against every screen's stored baseline; froze that baseline once per run to keep the check order-independent; added per-screen `--force` to override findings and proceed
- **Phase 4a** complete: standalone ingest of the historical position-outcomes workbook (`src/historical_ingest.py`) — NOT a screen, not part of `refresh.py`'s dispatch. Descriptive analytics (by Setup/Sector/hold-period/market-cap/era) on ~25 years of realized short P&L, plus the Whiteboard-to-position chained-outcome reconciliation. See that module's docstring and `PHASE4A_SCOPE.md`.
- See `README.md §Development Phases` and `PHASE3_PLAN.md` for the full roadmap (3a–3e, then Phase 4)

## Commands
- Fresh env: `pip install -r requirements.txt`. `jinja2>=3.1.2` is pinned deliberately — pandas' `.style` accessor requires it and every table in `app.py` uses it, but altair pulls jinja2 in unpinned, so an older jinja2 already installed survives the install and breaks every table at runtime with the whole suite still green.
- Run all tests: `pytest tests/ -v`
- Run specific test file: `pytest tests/test_transform.py -v`
- Run full pipeline (short_screen): `python src/ingest.py && python src/transform.py && python src/score.py`
- Refresh all six screens, gated by pre-write validation: `python src/refresh.py`. One screen only: `python src/refresh.py --screen cyclicals`. Validate without writing: `python src/refresh.py --dry-run`. Override a screen's validation findings and proceed with the write anyway, repeatable: `python src/refresh.py --force cyclicals --force structural`. Print the last N runs (default 10), newest first: `python src/refresh.py --history [N]` — mutually exclusive with `--screen`/`--dry-run`/`--force`.
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
- `src/db.py` — multi-screen storage helpers: `table_name(stage, screen_id)`, `sync_screens_registry()`, `replace_screen_rows()` (the shared `screen_membership` table only — per-screen tables use plain `to_sql(if_exists="replace")`), `append_rows()`, `create_index_if_not_exists()` (Phase 3d Part 2b's append-only write pattern — see Architecture Rule 10).
- `src/app.py` — Streamlit UI: a top-level View toggle (per-screen vs. cross-screen overlap), then, for per-screen, a sidebar screen selector plus three rendering paths (scored quant_composite, unscored quant_composite, curated).
- `src/overlap.py` — Phase 3d Part 1 cross-screen overlap calculations: `compute_overlap()`, `build_presence_matrix()`, `screen_count_ceiling()`, `style_overlap_table()`. Treats `short_screen` as context (its composite score), not a membership tick.
- `src/refresh.py` — Phase 3d Part 2a one-command refresh orchestrator: dispatches each registry screen_id to its prepare/ingest functions, gates the write on `src/validate.py`'s checks, then runs transform/score where applicable. `python src/refresh.py [--screen <id>] [--dry-run] [--force <id> ...] [--history [N]]`. Sits above ingest/curated_ingest/rsi_ingest/transform/score/history; never imported by them. Phase 3d Part 2b added: derives each screen's final-stage table (curated_data / scored_data / transformed_data / raw_data, in that preference order) and, on a real (non-dry-run) invocation, persists one `refresh_runs` row, one `refresh_screen_runs` row per screen, and (for each `PASSED` screen) a `refresh_snapshots` row per ticker — all in a single transaction after every screen has been processed, so a code-bug abort (`ScreenTypeError`) leaves zero trace rather than a partial one. Phase 3d Part 2c added `read_stored_ticker_sets()`, called once before the per-screen loop to freeze every registry screen's stored ticker set for `check_composition_misfile` (order-independence — see `tests/test_refresh.py`'s `TestBaselineOrderIndependence`), and `--force <screen_id>` (repeatable): findings are still computed and recorded, but the write proceeds and `refresh_screen_runs.forced` is set to 1.
- `src/validate.py` — pure pre-write validation checks (row count, composition/misfile via `check_composition_misfile()` + `normalize_ticker_set()`, null-rate spike, no-space-tickers) used by `refresh.py`. DataFrames in, findings out — no SQLAlchemy, no Streamlit.
- `src/history.py` — Phase 3d Part 2b pure functions for refresh run history and snapshot encoding: `new_run_id()`, `encode_row()`, `build_snapshot_frame()`, `snapshot_frame_to_stored_frame()`, `latest_snapshot_per_date()`, `build_run_row()`, `build_screen_run_row()`. DataFrames and dicts in, DataFrames and dicts out — no SQLAlchemy, no Streamlit, no file IO, and no dependency on `refresh.py`'s or `validate.py`'s types (would create an import cycle and invert the documented layering).
- `src/historical_ingest.py` — Phase 4a standalone ingest of the historical position-outcomes workbook (`data/historical/OWS Ideas Performance <date>.xlsx`, two sheets: Active Shorts Performance, Whiteboard Shorts Performance). **Not a screen** — no `config.yaml` screens-block entry, no registry row, no `screen_membership` row, never dispatched by `refresh.py`. Writes `historical_active_shorts`/`historical_whiteboard_shorts` (`if_exists="replace"` — a faithful, fully-reconstructable import of an external system of record, not score history, so Architecture Rule 10's append-only exception doesn't apply) and an append-only `historical_ingest_runs` provenance table (via `db.append_rows()` — what CAN'T be reconstructed once a newer file supersedes this one is the fact of what an earlier ingest saw). Market cap is converted from the source's $B to this project's standing $M convention (exact, ×1000) on both sheets. The sign-convention gate (`check_sign_convention`) aborts the import (`SignConventionError`, nothing written) if either sheet's price-move-vs-performance correlation isn't ≈−1 (short P&L, not stock return) — threshold in `config.yaml`'s `historical` block. Everything else is counted, never aborts: `count_defects()` reports six data-quality defects (see the module docstring; two are single rows carrying multiple defects at once — CARG on Whiteboard, ANSS/MFE/GLYT on Active) plus a seventh, **mixed benchmark instruments**: Active's `SPX @ Initiation`/`SPX @ Close` columns actually hold two instruments (62 SPY-priced rows, all pre-2018; 376 SPX-indexed rows) with no ticker to disambiguate, so `classify_benchmark_instrument()` uses a price-magnitude threshold+band rule (exact edges 270.39/756.55) that is **vintage-specific to this file and Active-only** — SPY already trades at 772 as of this file, above the band's SPX-side floor, so a future file's SPY-priced Active rows will start landing "unclassifiable" by design, not error. Whiteboard's own SPY/sector-ETF columns are NOT run through that classifier (its header already says "SPY" unambiguously); its `benchmark_instrument` is a plain constant label. The actual correctness guard on **both** sheets is `check_benchmark_consistency()`, which recomputes relative performance from each row's own price/benchmark levels and flags a mismatch beyond `config.yaml`'s `benchmark_consistency_tolerance` (0.01) — chosen because every performance value in this file is rounded to a whole percentage point (an observed, reported property, not a defect), which bounds the expected noise arithmetically and means the tolerance must never be widened to silence a real violation. `build_whiteboard_bridge()`/`chain_whiteboard_position()` implement the exact `(ticker, WBR Date == Initiation Date)` join between the two sheets; `summarize_whiteboard_naive()` documents a measurement artifact (never publish without `summarize_whiteboard_chained()` alongside it).
- `tests/test_transform.py`, `tests/test_score.py` — unit tests for short_screen's calc/ranking/scoring functions.
- `tests/test_overlap.py` — unit tests for the cross-screen overlap calculations, including the synthetic-seventh-screen genericity regression lock.
- `tests/test_schema.py` — storage helpers, per-screen-type dispatch guards, and the cross-screen pipeline-isolation regression lock.
- `tests/test_curated_ingest.py`, `tests/test_rsi_ingest.py`, `tests/test_loaders.py` — unit + end-to-end tests for the curated loader, the RSI loader/transforms, and the shared upload-file discipline.
- `tests/test_refresh.py`, `tests/test_validate.py` — dispatch-coverage, prepare-matches-ingest-write equivalence, gating, dry-run, continue-past-failure, inconsistency-reporting, and run-history/snapshot-persistence tests for the refresh orchestrator; unit tests for the validation checks.
- `tests/test_history.py` — unit tests for the pure run-history/snapshot functions, including the mutation-tested snapshot round-trip regression lock.
- `tests/test_historical_ingest.py` — Phase 4a unit tests: cleaning/coercion, the sign-convention gate (happy path, sign-flipped rejection, empty-intersection rejection, boundary), the benchmark-consistency check (happy path, an engineered violation, boundary), the band-guarded benchmark classifier, the synthetic join-key regression test (the real file's 24/24 match is verified separately, once, against real data — `data/historical/` is gitignored, so no unit test may depend on it), `summarize_by_cut`'s grouped-plus-unassigned reconciliation, and the append-only `historical_ingest_runs` table.
- `config.yaml` — per-screen config under `screens`: `display_name`, `type`, `universe`, and — `quant_composite` screens only, and only if scored — `factor_weights`/`scoring`. Also a top-level `refresh` block: thresholds for `src/validate.py`'s checks (currently just `null_rate_max_increase_pct` — the composition/misfile check added in Phase 3d Part 2c is deliberately threshold-free). Also a top-level `historical` block: thresholds for `src/historical_ingest.py` (`sign_convention_min_abs_corr`, `benchmark_consistency_tolerance`) — not a screen, see that module's docstring.
- `data/uploads/<screen_id>/` — drop each screen's export here (gitignored); every ingest path enforces exactly one file per folder.
- `data/historical/` — drop the historical position-outcomes workbook here (gitignored, proprietary — same reason as `data/uploads/`); `src/historical_ingest.py` enforces exactly one `.xlsx` file, same discipline as every screen's upload folder.
- `data/screener.db` — SQLite database (gitignored): `screens` registry, shared `screen_membership(screen_id, ticker)`, each screen's own `raw_data__*`/`transformed_data__*`/`scored_data__*` (quant) or `curated_data__*` (curated) tables, and (Phase 3d Part 2b, append-only) `refresh_runs`, `refresh_screen_runs` (Phase 3d Part 2c added a `forced` 0/1 column), `refresh_snapshots` — each `refresh_snapshots` row is uniquely identified by `(run_id, screen_id, ticker)`, not `(screen_id, ticker, run_date)` (more than one run per day is normal); resolve to one row per date via `history.latest_snapshot_per_date()`. Phase 4a added `historical_active_shorts`/`historical_whiteboard_shorts` (replaced wholesale each ingest — NOT part of the screen tables above) and append-only `historical_ingest_runs` (one row per `historical_ingest.py` run; `defects_json` holds the full structured defect report, same pattern as `refresh_screen_runs.findings_json`).
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

10. **The pipeline runs sequentially and each step is independently re-runnable.** `ingest.py` → `transform.py` → `score.py`. Each reads from the prior step's SQLite table and writes to the next. Rerunning any step overwrites its output table cleanly. **Exception (Phase 3d Part 2b, deliberate and documented):** `refresh.py`'s run-history and snapshot tables (`refresh_runs`, `refresh_screen_runs`, `refresh_snapshots`) are append-only — score history can't be reconstructed once overwritten, unlike every other table above. Do not "fix" these back to `replace`.

## Team Workflow

**Roles**: Driver (Tom) · Product Manager (Claude chat) · Inspector (Claude Code) · Worker (Claude Code) · Reviewer (Claude Code)

**Process**: Driver + PM decide priorities/direction → PM defines the scoped phase → Inspector checks the repo and reports facts → PM writes the Worker prompt → Worker plans and implements within scope → PM reviews against scope and acceptance criteria → Reviewer validates material phases → merge → fresh session for next phase

**Principles**: narrow phases · boring modular architecture · no broad refactors · explicit validation before moving on

**Plan review rule**: PM either approves the Worker's plan as-is, or sends a revision prompt back. Never both — no "approved with changes." If revisions are needed, the Worker resubmits a new plan.

**Role responsibilities**:
- **Driver**: sets goals, priorities, risk tolerance, final decisions
- **Product Manager**: scope, acceptance criteria, architecture judgment, sequencing, and the final Worker prompts (written as clean, copy/paste-ready codeblocks). No codebase claims without verification — verify directly against the repo and database, and use an Inspector prompt only when a change must be traced across more files than a single PM session can hold
- **Inspector**: read-only — inspects the repo, traces logic across files, and reports exact files/functions/classes/risks/ambiguities to the PM. Does not own scope, make undocumented assumptions, design architecture, write code, or declare a phase done — that always requires passing `pytest` output plus explicit PM sign-off against acceptance criteria, not code-reading alone. Reports follow Output format below.
- **Worker**: builds the scoped change and adds/updates tests. If the prompt says "propose a plan," propose a plan — don't write code yet. Do not refactor, rename, or touch files/signatures outside the current phase's scope; flag surprises instead of silently changing approach. Plans and reports follow Output format below.
- **Reviewer**: fresh review for bugs, regressions, edge cases, scope creep, and Architecture Rule compliance — run `pytest` first (full output in the summary). Reports follow Output format below.

### Worker Rules — error handling & testing
- Every `try/except` must handle a specific known failure mode, or re-raise after adding context. No bare `except:`/`except Exception:` that silently continues.
- Write tests before or alongside implementation, named `tests/test_<module>.py`. One clear assertion per test beats many weak ones — if you could delete the implementation and the test would still pass, the test is broken. Use small (5–10 row) synthetic DataFrames with known inputs/outputs, and cover edge cases: `NaN` inputs, zero denominators, negative values, the `"#N/A N/A"` string, and all-missing-data rows.
- **Verification efficiency**: run the full verification chain (the phase's real end-to-end command, snapshot comparison, validation notebook, Streamlit check) ONCE, at the end of the phase, and report it once. Do not re-run a check after every intermediate change. Re-run a specific check mid-phase only when you've changed something that could plausibly break that specific thing, and say why.

### Output format (mandatory, applies to Worker/Inspector/Reviewer alike)
Deliver every plan and every report as ONE fenced block holding the whole thing, so it copies in one action (Tom pastes between windows by hand). Use a four-backtick outer fence so inner triple-backtick code blocks survive. Output it once — no prose repeat. Nothing outside the block but an optional one-line preamble. **This applies to the closing summary of an execution turn too, not just to plans and formal reports.** Narrating what you are doing as you work through an approved task is fine and expected; the summary that ends the turn goes in the block.

### Before You Report Done (mandatory checklist)
- [ ] Output is ONE four-backtick fenced block per Output format above — plan, build report, and the closing summary of an execution turn alike.
- [ ] Run `pytest`. ALL tests must pass. Include the full output in your summary.
- [ ] Ran the phase's real command end-to-end on real data, once, and pasted the actual output — not just the unit suite. Green tests prove the code does what its tests say; they do not prove the feature is right. Check the output against a number derived independently beforehand. If the run writes to `data/screener.db`, back it up first, state exactly what it wrote, and leave any restore decision to the Driver.
- [ ] New logic has tests: happy path, one edge case, one boundary condition.
- [ ] New functions have docstrings covering inputs, outputs, and NaN/edge-case behavior.
- [ ] Imports are correct and minimal — no unused imports, no circular imports.
- [ ] Confirmed compliance with Architecture Rules 1–10 above (no web/db imports in calc functions, decimal convention, explicit NaN handling, config-sourced weights/defaults, M-Score excluded from the composite, `PERCENTRANK.INC`-compatible ranking).
- [ ] No broad refactors outside the scope of the current task.

Recurring bug patterns worth re-reading before touching `transform.py`/`score.py`: see `docs/BUG_PATTERNS.md`.

## Known Issues (do not fix unless explicitly scoped into current phase)

### Functional
- Universe is defined implicitly by whatever rows appear in the uploaded file. Explicit universe management (add/remove tickers, maintain a master list) is deferred.
- The composition/misfile check's headroom is thin specifically for competition's export: correctly filed, it scores Jaccard 0.260 against its own stored baseline vs. 0.221 against structural's — a +0.039 margin, the thinnest of the six screens — because competition and structural genuinely share a meaningful number of names. The same +0.039 margin is what catches competition's export if it's misfiled into structural's folder instead. (structural's own export is not fragile in the same way: 0.399 against its own baseline vs. 0.187 against competition's, a comfortable +0.212.) Not a bug — all 30 misfile permutations across the six screens do flag; competition's pairing with structural just has the least headroom of the six correct-placement scores.
- **Small screens with heavy turnover can false-positive the composition/misfile check.** `management_comp` is down to 10 rows; if it turns over completely on a future refresh, its own Jaccard against its stored baseline is 0.0, and any peer screen sharing even a single ticker with the new export beats that trivially, blocking a perfectly legitimate write. `--force` exists mainly as the escape hatch for this case.

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
