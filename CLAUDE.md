# CLAUDE.md

## Project
OWS Short Screen — a Python-based quantitative stock screening tool for identifying short candidates across a broad equity universe (~1,300 stocks). Rebuilt from a Bloomberg/Excel workflow into a maintainable, extensible codebase. See README.md for full architecture overview.

## Current Status

One line per phase. **The full record for any phase is its `PHASE<N>_*.md` docs**; the traps that
still bite are in `PM_HANDOFF.md`, and closed-phase narrative is in `PHASE_HISTORY.md`.

- **Phase 1** — ingestion, metric calculations, percentile ranking, composite scoring, unit tests.
- **Phase 2** — Streamlit web UI (`src/app.py`).
- **Phase 3a** — multi-screen architecture (foundation).
- **Phase 3b** — the 4 curated screens (Cyclicals, Competition, Structural, Management Comp), from Canary Excel exports.
- **Phase 3c / 3c.1 / 3c.2** — Rising Short Interest (the second `quant_composite` screen, **no factor model, so no scoring**); ticker normalization; pre-diff inputs + export.
- **Phase 3d Part 1** — cross-screen overlap view (`src/overlap.py`).
- **Phase 3d Part 2a/2b/2c** — one-command refresh gated by pre-write validation (`src/refresh.py`, `src/validate.py`); run history and per-run snapshots (`src/history.py`, append-only — the first brick of a future backtest dataset); composition/misfile validation replacing the deleted universe-size check, plus per-screen `--force`.
- **Phase 4a** — historical position-outcomes ingest (`src/historical_ingest.py`). **NOT a screen**, never dispatched by `refresh.py`.
- **Phase 4b** — fixed-horizon Whiteboard outcome measurement (`src/whiteboard_horizons.py` + `src/price_history.py`). **NOT a screen.** The only construction under which the four outcome arms are comparable.
- **Phase 5a** — display polish: `st.column_config` labels, the Excel template's colour scales (`src/styling.py`), theme config.
- **Phase 5b-1** — inline drill-down driven by a row click (`src/selection.py`).
- **Phase 5b-2** — cross-screen "Also Appears On" context (`src/cross_screen_context.py`), the overlap table relocated into a per-screen expander, click-through navigation, and the R8 brand theme.
- **Phase 5b-3** — column-header help on every displayed column of all four tables, plus click-a-cell derivation for the 10 diff factors (`selection_mode=["single-row","single-cell"]`).
- **Phase 5c-1** — sidebar polish: Refresh Data moved below the filters, bold sidebar labels, Market Cap thousands separators (`$%,.0f`), and "Select a stock" promoted to a subheader.
- **Phase 5c-2** — brand and layout: the screen title in brand green with the white-disc mark beside it at the title's own font height, the green-disc mark at the top of the sidebar (replacing `st.logo`, whose 32px cap could not meet the requested size), and the grid header band to light green (`#E8F1EA`).
- **Phase 3e** — PARKED, not cancelled. No Canary API key. `PHASE3E_SCOPE.md`/`PHASE3E_PROMPT.md` are complete and current.
- Roadmap: `PHASE3_PLAN.md`. Live options and open decisions: `PM_HANDOFF.md`.

## Commands
- Fresh env: `pip install -r requirements.txt`. `jinja2>=3.1.2` is pinned deliberately — pandas' `.style` accessor requires it and every table in `app.py` uses it, but altair pulls jinja2 in unpinned, so an older jinja2 already installed survives the install and breaks every table at runtime with the whole suite still green.
- Run all tests: `pytest tests/ -v`
- Run specific test file: `pytest tests/test_transform.py -v`
- Run full pipeline (short_screen): `python src/ingest.py && python src/transform.py && python src/score.py`
- Refresh all six screens, gated by pre-write validation: `python src/refresh.py`. One screen only: `python src/refresh.py --screen cyclicals`. Validate without writing: `python src/refresh.py --dry-run`. Override a screen's validation findings and proceed with the write anyway, repeatable: `python src/refresh.py --force cyclicals --force structural`. Print the last N runs (default 10), newest first: `python src/refresh.py --history [N]` — mutually exclusive with `--screen`/`--dry-run`/`--force`.
- Launch UI: `streamlit run src/app.py`
- Lint: `ruff check src/ tests/`

## File Layout

One or two sentences per module. **The reasoning lives in each module's own docstring** and in that
phase's `PHASE<N>_*.md`; do not re-expand this section into essays — it is read at the start of every
Worker and PM session.

**Pipeline (short_screen, RSI)** — sequential, each step independently re-runnable:
- `src/ingest.py` — Bloomberg/quant loader: reads the one export in `data/uploads/<screen_id>/` per `SCREEN_INGEST_CONFIGS`, writes `raw_data__<screen_id>`. Rejects curated screen_ids.
- `src/rsi_ingest.py` — Rising Short Interest's own loader: trims the export's preamble/count-row/footer, fixes the ticker-extraction bug.
- `src/curated_ingest.py` — shared loader for the 4 curated screens: unwraps Canary's quoted numerics, unit-converts, parses the packed `scores` field.
- `src/transform.py` — per-screen derived metrics via `SCREEN_TRANSFORM_FUNCS`. Rejects curated screen_ids.
- `src/score.py` — percentile-ranks and composite-scores via `get_screen_config`. `FACTOR_DEFINITIONS` holds each factor's metric + ranking direction. Rejects curated screens and unscored quant screens.

**Shared infrastructure:**
- `src/config.py` — `load_config()`/`CONFIG_PATH`; `get_screen_type()`/`ScreenTypeError`, the type-dispatch guard used by ingest/transform/score.
- `src/loaders.py` — generic upload-file IO: `read_upload`, `validate_columns`, `log_summary`, `find_single_upload_file`/`UploadFileError`.
- `src/db.py` — `table_name(stage, screen_id)`, `sync_screens_registry()`, `replace_screen_rows()`, `append_rows()`, `create_index_if_not_exists()`.
- `src/refresh.py` — the one-command orchestrator. Dispatches each registry screen, gates writes on `validate.py`, then transforms/scores. Persists run history + snapshots in a single transaction after all screens are processed. `read_stored_ticker_sets()` is called ONCE before the per-screen loop so the misfile check is order-independent. Sits above every other module; never imported by them.
- `src/validate.py` — pure pre-write checks (row count, composition/misfile, null-rate spike, no-space-tickers). DataFrames in, findings out.
- `src/history.py` — pure run-history/snapshot functions. **No dependency on `refresh.py`'s or `validate.py`'s types** — that would invert the documented layering.

**Not screens** — no registry row, no `screen_membership` row, never dispatched by `refresh.py`:
- `src/historical_ingest.py` — Phase 4a ingest of `data/historical/OWS Ideas Performance <date>.xlsx` (Active Shorts + Whiteboard Shorts). Aborts on `check_sign_convention`; counts seven data-quality defects without aborting. `check_benchmark_consistency` is the real correctness guard on both sheets. `summarize_whiteboard_naive()` documents a measurement artifact — **never publish it without `summarize_whiteboard_chained()` alongside.**
- `src/price_history.py` — Phase 4b external daily-close loader. **Upsert/append-only, never `replace`**, and a `bloomberg_manual` row always survives a later API pull. yfinance primary; Stooq is documented but non-functional (Known Issues).
- `src/whiteboard_horizons.py` — Phase 4b fixed-horizon measurement from the WBA anchor. Only its orchestrator touches the DB/filesystem. No survivorship exclusion, no imputation — a missing price is flagged, never guessed.

**Display layer** — pandas only, no Streamlit and no SQLAlchemy (Architecture Rule 1):
- `src/overlap.py` — `compute_overlap()`, `build_presence_matrix()`, `screen_count_ceiling()`, `style_overlap_table()`, plus Phase 5b-2's `resolve_overlap_click_target()`, `apply_zero_thematic_label()`, `zero_thematic_summary()`. Treats `short_screen` as context, not a membership tick.
- `src/styling.py` — the Excel template's per-column three-anchor colour scale, anchored on each column's own **min / 50th percentile / max**, never a hardcoded 0..1. Plus `bold_ticker_column()`.
- `src/selection.py` — `resolve_selected_ticker()`, `find_ticker_row()`, `resolve_nav_target()`, `is_fresh_selection()`, and Phase 5b-3's `resolve_selected_cell()`/`should_process_cell_selection()`. **Resolves positionally against the frame exactly as passed to `st.dataframe`** — the defect this module exists to prevent.
- `src/cross_screen_context.py` — `classify_screen()` (the single screen taxonomy both `app.py` loaders dispatch through), `build_screen_contribution()`, `build_also_appears_on()`. Identity is never repeated per screen.
- `src/app.py` — the Streamlit UI: a sidebar screen selector plus three per-screen render paths (scored / unscored / curated), each ending in a drill-down. The overlap table renders once at the bottom of every screen in a cached, collapsed expander. Cross-screen navigation is a pending-nav-then-rerun pattern gated by `resolve_nav_target()`. Phase 5b-3 added column-header help and the click-a-cell derivation panel.

**Tests** — one file per module, same name. Notable ones:
- `tests/test_overlap.py` — includes the synthetic-seventh-screen genericity regression lock (`TestGenericityRegressionLock`), which exercises `compute_overlap` **and** `build_presence_matrix`.
- `tests/test_schema.py` — storage helpers, per-screen-type dispatch guards, cross-screen pipeline isolation.
- `tests/test_refresh.py` / `tests/test_validate.py` — dispatch coverage, prepare-matches-ingest equivalence, gating, dry-run, continue-past-failure, run-history persistence, and `TestBaselineOrderIndependence`.
- `tests/test_history.py` — includes the mutation-tested snapshot round-trip lock.
- `tests/test_curated_ingest.py`, `tests/test_rsi_ingest.py`, `tests/test_loaders.py` — the curated/RSI loaders and shared upload-file discipline.
- `tests/test_historical_ingest.py`, `tests/test_price_history.py`, `tests/test_whiteboard_horizons.py` — 4a/4b, including `flag_spurious_stored_relative`'s null-price-leg suite and the Stooq bot-challenge response-shape lock.
- `tests/test_styling.py`, `tests/test_selection.py`, `tests/test_cross_screen_context.py`, `tests/test_app.py` — the display layer, including `resolve_nav_target`'s blocked-case lock and `classify_screen`'s compound-condition lock.
- **No unit test may depend on gitignored data.** `data/screener.db`, `data/uploads/` and `data/historical/` are all gitignored; every test uses synthetic frames or a `tmp_path` fixture DB. Real-data correspondences are verified once, in a phase's acceptance run, and reported there.

**Config and data:**
- `config.yaml` — per-screen `screens` block (`display_name`, `type`, `universe`, and for scored quant screens `factor_weights`/`scoring`), plus top-level `refresh` and `historical` blocks.
- `data/uploads/<screen_id>/` — one export file per folder (gitignored). `data/historical/` — same discipline for the outcomes workbook.
- `data/screener.db` — SQLite (gitignored): `screens`, `screen_membership`, each screen's stage tables, the append-only `refresh_runs`/`refresh_screen_runs`/`refresh_snapshots`, and 4a/4b's tables. **A `refresh_snapshots` row is unique on `(run_id, screen_id, ticker)`, NOT `(screen_id, ticker, run_date)`** — resolve to one row per date via `history.latest_snapshot_per_date()`.
- `notebooks/OWS Short Screen (April 2026).xlsx` + `notebooks/validation.ipynb` — the notebook still references a March 2026 vintage no longer on disk and cannot be run as-is. See Known Issues.
- `.streamlit/config.toml` — the R8 brand theme: green (`#1E552D`) as an **accent only** on a light palette. `font` is Arial; `app.py`'s `APP_FONT_FAMILY` is locked to this file's `font` line by `tests/test_app.py`.
- `assets/` — `ows-bear-green-disc.png` (green disc, white bear: the mark at the top of the
  sidebar, on the sidebar's `secondaryBackgroundColor` ground) and `ows-bear-white-disc.png`
  (white disc, green bear: the mark beside the screen title, on the white page). Both are derived
  from `ows-logo-on-green.pdf`, the Illustrator vector source. `ows-mark.png` and
  `ows-lockup-white-on-green.jpg` are superseded.

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

**Plan review rule**: PM either approves the Worker's plan as-is, or sends a revision prompt back. Never both — no "approved with changes." If revisions are needed, the Worker resubmits a new plan. **Two scoped exceptions, both added 2026-09-05:** a *build report* may be approved with required corrections before commit (4b, 5b-2 and 5b-3 all shipped that way); and a plan revision that contains **no design change** — only mechanical corrections the Worker needs no further sign-off to apply — may be issued as **"REVISE: apply and build, no resubmission."** Anything touching design keeps the binary.

**Role responsibilities**:
- **Driver**: sets goals, priorities, risk tolerance, final decisions
- **Product Manager**: scope, acceptance criteria, architecture judgment, sequencing, and the final Worker prompts. **Every Worker or revision prompt is delivered IN THE CHAT as ONE four-backtick fenced block, ready to copy in a single action — not only written to a file.** Also save it to `PHASE<N>_PROMPT.md`: the repo copy is the record, the pasted block is the deliverable. No codebase claims without verification — verify directly against the repo and database, and use an Inspector prompt only when a change must be traced across more files than a single PM session can hold
- **Inspector**: read-only — inspects the repo, traces logic across files, and reports exact files/functions/classes/risks/ambiguities to the PM. Does not own scope, make undocumented assumptions, design architecture, write code, or declare a phase done — that always requires passing `pytest` output plus explicit PM sign-off against acceptance criteria, not code-reading alone. Reports follow Output format below.
- **Worker**: builds the scoped change and adds/updates tests. If the prompt says "propose a plan," propose a plan — don't write code yet. Do not refactor, rename, or touch files/signatures outside the current phase's scope; flag surprises instead of silently changing approach. Plans and reports follow Output format below.
- **Reviewer**: fresh review for bugs, regressions, edge cases, scope creep, and Architecture Rule compliance — run `pytest` first (full output in the summary). Reports follow Output format below.


### Prompt and review economics (added 2026-09-05)

Measured in `PROCESS_EFFICIENCY.md`: revisions per phase trended **up** across the 5b series (1 → 2
→ 3) while prompt size grew 10.8 KB → 21.7 KB → 37.9 KB. Front-loading the prompt worked through 4b
and then inverted — more PM pre-specification meant more PM surface area to be wrong on, and roughly
**fourteen of the ~thirty defects this process has caught were errors in the PM's own prompt.**
Three rules follow.

- **Specify the property and the failure; let the Worker write the test.** The PM cannot run
  `pytest`, so every test written into a prompt is unverified code shipped as an instruction. State
  what must be locked and what wrong behaviour must make it fail, then let the Worker — who can run
  it in seconds — design it and report the fail-first evidence. Two of 5b-3's three revision rounds
  were PM test-design errors: a unit test reading the gitignored live DB, and a sort test that could
  not fail in either world.
  **The PM still pre-specifies what it can verify or author** — acceptance numbers derived against
  live data, display copy, scope boundaries, and design rulings. Those have been reliable.
- **Batch API unknowns into ONE probe, before the first plan.** 5b-3 spent three separate browser
  probes across three rounds on three foreseeable questions about one widget. When a phase adopts an
  unfamiliar API, the prompt requires a single batched probe covering: **(a)** how it interacts with
  the state we already rely on, **(b)** its lifecycle — how it is set and cleared, and whether a
  programmatic write sticks, **(c)** whether it survives user-side transforms invisible to Python
  (sort, filter, reshape), **(d)** its exact return type. One Worker session, one report.
- **A claim only a browser can settle must be settled in a browser.** A Python-side probe that
  passes whether or not the screen renders is not evidence — `Styler.set_table_styles` output is
  marshalled to the frontend even though the canvas-drawn grid has no `<th>` to match it. Equally,
  a pure unit test cannot lock a frontend behaviour: measure it, then record it in Known
  Implementation Decisions rather than writing a test that cannot fail.

**Keep the docs trimmed.** This file is read at the start of every Worker *and* every PM session, so
its size is paid repeatedly. It grew 39 KB → 49 KB during the 5b-3 build alone. Trim **between**
phases, never during one, and never silently — a trim that drops a standing decision is expensive,
so it lands as its own reviewable diff.
### Worker Rules — error handling & testing
- Every `try/except` must handle a specific known failure mode, or re-raise after adding context. No bare `except:`/`except Exception:` that silently continues.
- **A test that matches against source text must be written against the POST-edit source.** 5c-1's
  first draft anchored a regex on `"Market Cap ($M)"` while the same phase was rewriting that label
  to `"**Market Cap ($M)**"` — it would have matched zero times and gone red on its own change. Any
  source-anchored assertion is a compound condition on text this phase may be moving; check it
  against the text as it will read after your edit, not as it reads now.
- Write tests before or alongside implementation, named `tests/test_<module>.py`. One clear assertion per test beats many weak ones — if you could delete the implementation and the test would still pass, the test is broken. Use small (5–10 row) synthetic DataFrames with known inputs/outputs, and cover edge cases: `NaN` inputs, zero denominators, negative values, the `"#N/A N/A"` string, and all-missing-data rows.
- **Verification efficiency**: run the full verification chain (the phase's real end-to-end command, snapshot comparison, validation notebook, Streamlit check) ONCE, at the end of the phase, and report it once. Do not re-run a check after every intermediate change. Re-run a specific check mid-phase only when you've changed something that could plausibly break that specific thing, and say why.

### Output format (mandatory, applies to PM/Worker/Inspector/Reviewer alike)
Deliver every plan and every report as ONE fenced block holding the whole thing, so it copies in one action (Tom pastes between windows by hand). Use a four-backtick outer fence so inner triple-backtick code blocks survive. Output it once — no prose repeat. Nothing outside the block but an optional one-line preamble. **This applies to the closing summary of an execution turn too, not just to plans and formal reports.** Narrating what you are doing as you work through an approved task is fine and expected; the summary that ends the turn goes in the block. **For the PM the same rule governs Worker and revision prompts: the block goes in the chat, so Tom can copy it in one action without opening a file.**

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
- **29 rows in `historical_whiteboard_shorts` carry a spurious stored `relative_spy_performance`** — a null-price formula artifact in the source workbook, not a real measurement, discovered by Phase 4b's `check_event_window_replication` on its first real run and confirmed structurally by `src/whiteboard_horizons.py`'s `flag_spurious_stored_relative`. Wherever `wba_price` and/or `wbr_price` is null, the workbook's own formula silently treats the missing price leg as zero rather than leaving the cell blank, so the stored `relative_spy_performance` equals the SPY benchmark move ALONE (`bench_move - 0` instead of `bench_move - price_move`) — not this stock's relative performance. Split by outcome: 20 Open / 7 Removed / 2 Initiation. The null-price condition is load-bearing, not a coincidence: one real row (ALAB) has both prices present and its stored value happens to be close to the SPY move too (the stock genuinely went nowhere) and is correctly NOT flagged — `flag_spurious_stored_relative`'s test suite locks this distinction. **Consequence: Phase 4a's published naive Whiteboard comparison (`summarize_whiteboard_naive`) rests on 9 fabricated values out of its 95 Removed+Initiation rows — 7 of the 71 Removed and 2 of the 24 Initiation.** Where a row is flagged spurious, Phase 4b's own vendor-computed relative return (from `whiteboard_horizon_returns`) is the correct figure; the stored column is not a measurement for that row. Detected and reported only — `historical_whiteboard_shorts` is NOT modified (4a's tables stay a faithful import of the source file, per that module's docstring).
- **This is invisible to 4a's `check_benchmark_consistency`**, which is why it went unseen in Phase 4a: that check masks on `implied.notna()`, and `implied` (`bench_move - price_move`) is NaN whenever a price leg is null — it structurally cannot see the rows whose prices are missing, which are exactly the rows where the source formula misbehaves. Do not "fix" `check_benchmark_consistency` to catch this in a future phase without deliberately scoping that change; it is a documented blind spot, not scoped for repair here.
- **Stooq (the documented fallback vendor for `src/price_history.py`'s price pull) is currently non-functional.** Its public CSV endpoint (`stooq.com/q/d/l/`) now returns a JavaScript bot-challenge page instead of CSV data. The fallback path is coded and unit-tested (including a regression test for this exact response shape), but was not made to work around the challenge — bot-detection circumvention is out of scope regardless of purpose. Coverage currently rests entirely on yfinance plus manual Bloomberg fills (`ingest_manual_fill`).
- **`notebooks/validation.ipynb` references `OWS Short Screen (March 2026).xlsx`, which is not present in the repo** (only the April 2026 workbook is kept — see File Layout), so the notebook cannot currently be run as-is. Do not re-point it at the April workbook and do not re-run it as a documentation fix: the Known Implementation Decision below on `kind='strict'` percentile ranking says that choice was validated against the March 2026 file specifically, and re-pointing the notebook would be a scoped validation decision (does `kind='strict'` still hold against the April file?), not a documentation tidy.
- **The main table's row highlight may not repaint after a filter change while a browser-side column sort is active.** `sync_drilldown_selection` re-seeds the table's selection with a position computed in server-side `display_df` order; Streamlit gives Python no signal that the user has sorted a column in the browser. Observed in one Linux/Chromium reproduction as the selection checkbox clearing after a filter change under an active sort; not reproducible on macOS. **The drill-down panel is unaffected in every observed case and always names the correct stock**, and any row click immediately restores the highlight. Cosmetic; not a data-correctness defect.

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

- **Four `st.dataframe` selection-state behaviors were measured against the
  installed streamlit 1.63.0 by direct probe during Phase 5b-3** (a combined
  `selection_mode=["single-row","single-cell"]` widget), because none of the
  four is documented upstream (one directly contradicts the one thing that
  *is* documented — row sort-invariance — by extending it silently to cells).
  The first three were found with a static scratch script (a fixed 5-6-row
  frame, only an unrelated widget changing); the fourth was found only by
  running the real app end-to-end and reshaping real data, which the scratch
  script never did — a reminder that a scratch probe validates the mechanism
  it actually exercises, not the whole surface a real screen touches.
  Re-measure against the installed version before relying on any of these if
  streamlit is ever upgraded; do not assume they still hold.
  1. **A row click and a cell click are independent.** Neither disturbs the
     other's selection state — a user can have a row selected and a
     different row's cell selected at the same time, and clicking either
     leaves the other exactly as it was.
  2. **A programmatic `cells: []` push (written to
     `st.session_state[table_key]` before the widget is instantiated) shapes
     only the return value of the *run it happens in* — it has no durable
     effect on what the *next* rerun reads back, PROVIDED the underlying
     data is unchanged.** The frontend's last real click keeps being
     reported, indefinitely, until a genuinely different cell is clicked or
     the data reshapes (see #4). The equivalent row push (`rows:
     [target_idx]`) *does* durably repaint the highlight, already relied on
     since Phase 5b-1 — this is a one-way asymmetry between the two
     selection kinds, not a general rule about pushes. This is the finding
     most likely to cost a future session a full afternoon to rediscover:
     `app.py`'s `process_cell_selection()`/`render_cell_derivation_panel()`
     (Phase 5b-3) are built around it (resolve a cell selection once, on the
     rerun where it actually changes; persist the result; re-validate by
     ticker identity on every later rerun; never re-resolve by the original
     row position).
  3. **A clicked cell's row index is reported against the frame as
     originally passed to `st.dataframe`, unaffected by a further browser-
     side column sort** — the same invariant `st.dataframe`'s own docstring
     states for row selections, extended here to cells, where it is not
     documented. Confirmed in both directions (a visually-top row whose true
     position is last, and a visually-bottom row whose true position is
     first, both round-tripped correctly), with a same-row `rows` control
     landing on the documented value. `src/selection.py`'s
     `resolve_selected_cell()` therefore needs no sort-tracking logic of its
     own — resolving positionally against `display_df` is already correct.
  4. **A rerun where `filtered`/`display_df`'s own CONTENT reshapes (any
     sidebar filter change — a different row count/order on the SAME
     `st.dataframe` key) resets the frontend's cells selection to empty on
     that rerun — unlike #2, which holds only when the data is unchanged.**
     This reset happens regardless of whether the previously-clicked ticker
     survives the new filter, so an empty `cells` value can never be read as
     "the user deselected" (no such gesture was ever observed for cells,
     unlike rows). Found only by clicking a cell in the real running app and
     then changing a real sidebar filter — the scratch-script probes behind
     #1–#3 never reshaped the data they passed to `st.dataframe`, so they
     could not have found this. `src/selection.py`'s
     `should_process_cell_selection()` is built around it: an empty
     `pre_cells` is never processed, so a still-good persisted `(ticker,
     column)` survives a filter change that keeps the ticker, and only
     `render_cell_derivation_panel()`'s own `find_ticker_row()` check (never
     a re-resolve) decides whether a filter that excludes the ticker should
     clear the panel.
