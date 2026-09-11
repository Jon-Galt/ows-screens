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
- **Phase 5c-2 / 5c-2b** — brand and layout: the screen title in brand green with the bear glyph beside it, sized by its measured ink to the title's own height; the green-disc mark at the top of the sidebar (replacing `st.logo`, whose 32px cap could not meet the requested size); and the grid header band to light green (`#E8F1EA`).
- **Phase 5c-3** — per-screen material icons on "Also Appears On", and `stock_performance` relabelled "Stock Performance (1 yr.)" at all three display sites via `_STOCK_PERFORMANCE_LABEL`. Shipped `13fccbc`, before 5c-2.
- **Phase 5c-4** — the cross-screen overlap view becomes its own Screen-selector entry and the per-screen expander is retired. A **UI-only pseudo-screen** (`OVERLAP_PSEUDO_SCREEN_ID`, spliced in by `build_screen_selector_options`) — **never a `screens` or `screen_membership` row**, so `classify_screen`/`compute_overlap`/`build_also_appears_on` never see it. Its filters moved to the sidebar.
- **Phase 5d** — dynamic factor weighting: a two-level (category x factor) weight panel mirroring the April 2026 workbook, named presets in a gitignored `data/weight_presets.yaml`, and Overall Score slider bounds **derived** from the recomputed column, replacing the hardcoded `0.0-7.0` (which was never the real ceiling — config's weights sum to 6.999999). **Display-only:** a reweighted score never reaches `data/screener.db`, and the Cross-Screen Overlap view stays on config weights. `config.yaml`'s new `factor_categories` block is the single source of truth for the taxonomy; `app.py`'s `FACTOR_CATEGORIES` derives from it.
- **Phase 6a** — the seventh screen, `negative_expert_transcripts` (`quant_composite`, unscored): a header/detail pair, `src/transcript_ingest.py`, accumulating by upsert (Rule 10 exception above). Data foundation only — no new display column; the screen's own page renders sparsely (ticker and little else) until Phase 6b.
- **Phase 6b** — the screen's own page: per-screen unscored display columns/export filenames (retiring T39), a "Transcripts" drill-down panel, the `SCREEN_ICONS` entry plus a registry-coverage lock derived from `config.yaml` (closing the Phase 6a icon gap), and `find_duplicate_transcript_groups` (T37 duplicate detection, report-only).
- **Phase 3e** — PARKED, not cancelled. No Canary API key. `PHASE3E_SCOPE.md`/`PHASE3E_PROMPT.md` are complete and current.
- Roadmap: `PHASE3_PLAN.md`. Live options and open decisions: `PM_HANDOFF.md`.

## Commands
- Fresh env: `pip install -r requirements.txt`. `jinja2>=3.1.2` is pinned deliberately — pandas' `.style` accessor requires it and every table in `app.py` uses it, but altair pulls jinja2 in unpinned, so an older jinja2 already installed survives the install and breaks every table at runtime with the whole suite still green.
- Run all tests: `pytest tests/ -v`
- Run specific test file: `pytest tests/test_transform.py -v`
- Run full pipeline (short_screen): `python src/ingest.py && python src/transform.py && python src/score.py`
- Refresh all six screens, gated by pre-write validation: `python src/refresh.py`. One screen only: `python src/refresh.py --screen cyclicals`. Validate without writing: `python src/refresh.py --dry-run`. Override a screen's validation findings and proceed with the write anyway, repeatable: `python src/refresh.py --force cyclicals --force structural`. Print the last N runs (default 10), newest first: `python src/refresh.py --history [N]` — mutually exclusive with `--screen`/`--dry-run`/`--force`.
- Launch UI: `streamlit run src/app.py`
- Lint: `ruff check src/ tests/`. Note (Phase 6a): new modules suppress the `sys.path`-bootstrap `E402` at the site with `# noqa: E402` (e.g. `src/transcript_ingest.py`); pre-existing modules (e.g. `src/rsi_ingest.py`) carry the identical violation inside the 44-error baseline instead — "44 errors" means three new E402s were suppressed at the site, not that new code has none. Unifying the two conventions is unscoped.

## File Layout

One or two sentences per module. **The reasoning lives in each module's own docstring** and in that
phase's `PHASE<N>_*.md`; do not re-expand this section into essays — it is read at the start of every
Worker and PM session.

**Pipeline (short_screen, RSI)** — sequential, each step independently re-runnable:
- `src/ingest.py` — Bloomberg/quant loader: reads the one export in `data/uploads/<screen_id>/` per `SCREEN_INGEST_CONFIGS`, writes `raw_data__<screen_id>`. Rejects curated screen_ids.
- `src/rsi_ingest.py` — Rising Short Interest's own loader: trims the export's preamble/count-row/footer, fixes the ticker-extraction bug.
- `src/transcript_ingest.py` — Phase 6a's Negative Expert Transcripts loader: explodes multi-ticker transcripts, synthesizes a missing `DocID`, and upserts the detail table so it accumulates across uploads (see the module docstring's DocID-backfill duplication hazard, also in Known Issues). Phase 6b adds `find_duplicate_transcript_groups`, called from `ingest_transcripts` on the full accumulated table after every upsert to detect (not prevent) that hazard.
- `src/curated_ingest.py` — shared loader for the 4 curated screens: unwraps Canary's quoted numerics, unit-converts, parses the packed `scores` field.
- `src/transform.py` — per-screen derived metrics via `SCREEN_TRANSFORM_FUNCS`. Rejects curated screen_ids.
- `src/score.py` — percentile-ranks and composite-scores via `get_screen_config`. `FACTOR_DEFINITIONS` holds each factor's metric + ranking direction. Rejects curated screens and unscored quant screens.

**Shared infrastructure:**
- `src/config.py` — `load_config()`/`CONFIG_PATH`; `get_screen_type()`/`ScreenTypeError`, the type-dispatch guard used by ingest/transform/score.
- `src/loaders.py` — generic upload-file IO: `read_upload`, `validate_columns`, `log_summary`, `find_single_upload_file`/`UploadFileError`.
- `src/db.py` — `table_name(stage, screen_id)`, `sync_screens_registry()`, `replace_screen_rows()`, `append_rows()`, `create_index_if_not_exists()`, and (Phase 6a) `upsert_rows()`.
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
- `src/weighting.py` — Phase 5d's pure weighting layer: `load_factor_categories()`, `validate_taxonomy()`, `compute_effective_weights()` (category weight x within-category weight), and the preset file's load/save/delete. **Imports neither Streamlit, SQLAlchemy nor `src.score`** — locked by an AST import-graph test, because a substring scan false-positives on its own docstrings. `app.py` composes its output with `score.compute_overall_score`; the composite is never reimplemented here.
- `src/app.py` — the Streamlit UI: a sidebar screen selector plus three per-screen render paths (scored / unscored / curated), each ending in a drill-down, and (Phase 5c-4) a fourth path for the overlap pseudo-screen, which returns **before** `screen_type` is read and is the only branch that does not call `apply_pending_nav` — it discards a stale `_nav_target` instead. Cross-screen navigation is a pending-nav-then-rerun pattern gated by `resolve_nav_target()`. Phase 5b-3 added column-header help and the click-a-cell derivation panel. Phase 5d added the weight panel (`render_weight_panel`) at a **single seam** immediately after `load_quant_data` — reweighting `df` there means the slider bounds, filter, sort, colour domain, drill-down and export all follow with no other change — plus the pure predicates `should_reapply_preset`, `resolve_persisted_preset` and `insert_config_weight_export_column`.

**Tests** — one file per module, same name. Notable ones:
- `tests/test_overlap.py` — includes the synthetic-seventh-screen genericity regression lock (`TestGenericityRegressionLock`), which exercises `compute_overlap` **and** `build_presence_matrix`.
- `tests/test_schema.py` — storage helpers, per-screen-type dispatch guards, cross-screen pipeline isolation.
- `tests/test_refresh.py` / `tests/test_validate.py` — dispatch coverage, prepare-matches-ingest equivalence, gating, dry-run, continue-past-failure, run-history persistence, and `TestBaselineOrderIndependence`.
- `tests/test_history.py` — includes the mutation-tested snapshot round-trip lock.
- `tests/test_curated_ingest.py`, `tests/test_rsi_ingest.py`, `tests/test_loaders.py` — the curated/RSI loaders and shared upload-file discipline.
- `tests/test_historical_ingest.py`, `tests/test_price_history.py`, `tests/test_whiteboard_horizons.py` — 4a/4b, including `flag_spurious_stored_relative`'s null-price-leg suite and the Stooq bot-challenge response-shape lock.
- `tests/test_styling.py`, `tests/test_selection.py`, `tests/test_cross_screen_context.py`, `tests/test_app.py` — the display layer, including `resolve_nav_target`'s blocked-case lock and `classify_screen`'s compound-condition lock.
- `tests/test_weighting.py` — the taxonomy / effective-weight / preset-file layer, including the AST-based purity lock.
- **No unit test may depend on gitignored data.** `data/screener.db`, `data/uploads/` and `data/historical/` are all gitignored; every test uses synthetic frames or a `tmp_path` fixture DB. Real-data correspondences are verified once, in a phase's acceptance run, and reported there.

**Config and data:**
- `config.yaml` — per-screen `screens` block (`display_name`, `type`, `universe`, and for scored quant screens `factor_weights`/`scoring`), plus top-level `refresh` and `historical` blocks.
- `data/uploads/<screen_id>/` — one export file per folder (gitignored). `data/historical/` — same discipline for the outcomes workbook.
- `data/screener.db` — SQLite (gitignored): `screens`, `screen_membership`, each screen's stage tables, the append-only `refresh_runs`/`refresh_screen_runs`/`refresh_snapshots`, and 4a/4b's tables. **A `refresh_snapshots` row is unique on `(run_id, screen_id, ticker)`, NOT `(screen_id, ticker, run_date)`** — resolve to one row per date via `history.latest_snapshot_per_date()`.
- `notebooks/OWS Short Screen (April 2026).xlsx` + `notebooks/validation.ipynb` — the notebook still references a March 2026 vintage no longer on disk and cannot be run as-is. See Known Issues.
- `.streamlit/config.toml` — the R8 brand theme: green (`#1E552D`) as an **accent only** on a light palette. `font` is Arial; `app.py`'s `APP_FONT_FAMILY` is locked to this file's `font` line by `tests/test_app.py`.
- `assets/` — two marks, **not interchangeable**, locked by `TestScreenMarkPaths`:
  `ows-bear-glyph.png` (bear alone, no disc) beside the screen title, `ows-bear-green-disc.png` at
  the top of the sidebar. `ows-bear-white-disc.png` is a third variant for a coloured ground,
  currently unreferenced. All three were cut from `ows-logo-on-green.pdf`, the Illustrator vector
  source; `ows-mark.png` and `ows-lockup-white-on-green.jpg` are superseded.

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

10. **The pipeline runs sequentially and each step is independently re-runnable.** `ingest.py` → `transform.py` → `score.py`. Each reads from the prior step's SQLite table and writes to the next. Rerunning any step overwrites its output table cleanly. **Exception (Phase 3d Part 2b, deliberate and documented):** `refresh.py`'s run-history and snapshot tables (`refresh_runs`, `refresh_screen_runs`, `refresh_snapshots`) are append-only — score history can't be reconstructed once overwritten, unlike every other table above. Do not "fix" these back to `replace`. **Exception (Phase 6a, deliberate and documented):** `raw_data__negative_expert_transcripts` accumulates by upsert on `(doc_id, ticker)` (`src/db.py`'s `upsert_rows()`) rather than being replaced on each ingest — see `src/transcript_ingest.py`'s module docstring. Its aggregate, `transformed_data__negative_expert_transcripts`, stays a pure `replace` recompute.

## Team Workflow

**Roles**: Driver (Tom) — **Ursus Major** · Product Manager (Claude chat) — **Rain Man** · Inspector (Claude Code) · Worker (Claude Code) — **Lunch Pail** · Reviewer (Claude Code). Callsigns and sign-offs: see Callsigns below.

**Process**: Driver + PM decide priorities/direction → PM defines the scoped phase → Inspector checks the repo and reports facts → PM writes the Worker prompt → Worker plans and implements within scope → PM reviews against scope and acceptance criteria → Reviewer validates material phases → merge → fresh session for next phase

**Principles**: narrow phases · boring modular architecture · no broad refactors · explicit validation before moving on

**Plan review rule**: PM either approves the Worker's plan as-is, or sends a revision prompt back. Never both — no "approved with changes." If revisions are needed, the Worker resubmits a new plan. **Two scoped exceptions, both added 2026-09-05:** a *build report* may be approved with required corrections before commit (4b, 5b-2 and 5b-3 all shipped that way); and a plan revision that contains **no design change** — only mechanical corrections the Worker needs no further sign-off to apply — may be issued as **"REVISE: apply and build, no resubmission."** Anything touching design keeps the binary.

### Callsigns (Ursus Major ruling, 2026-09-11) — the sign-off IS the direction marker

| Role | Callsign |
|---|---|
| Driver | **Ursus Major** — Tom. Primarily just "Tom"; "Ursus Major Tom" in acknowledgments. Currently the only one; the scheme allows others later. |
| PM / Reviewer | **Rain Man `<sub-stage>`** — e.g. Rain Man 6c-1 |
| Worker | **Lunch Pail `<sub-stage>`** — e.g. Lunch Pail 6c-1 |

**Numbering.** The sub-stage is the shared unit: Rain Man 6c-1 and Lunch Pail 6c-1 are the PM and
the Worker of sub-stage 6c-1. Normally one-to-one, since a fresh Worker session per sub-phase is
already standing practice. **Collision clause:** a second Worker session on the same sub-stage is
`6c-1b`, and a PM handoff mid-stage likewise — a repeat must be visible, never silently ambiguous.
A docs round preceding a stage takes that stage with `-0`. Spelling is **Lunch Pail**; "Lunch Pale"
means the same thing and is not off-protocol. The Inspector has no callsign — none was assigned,
and inventing one is a question for Ursus Major, not for a PM or a Worker.

**Sign-offs — last line INSIDE the block, never outside it:**
- **Rain Man -> Lunch Pail** (every Worker prompt, revision prompt, build-correction prompt):
  `Rain Man 6c-1 to Lunch Pail over`
- **Lunch Pail -> Rain Man** (every plan, build report, review, and the closing summary of an
  execution turn — including a turn that only applied corrections or only committed):
  `Lunch Pail 6c-1 over`
- **Rain Man -> Ursus Major**: NO sign-off and NO block. Discussion, verdicts, findings,
  recommendations, questions, state summaries — ordinary prose to the final reader.

**So an unsigned message is by definition for Ursus Major, and a signed one is by definition for
pasting onward.** A signed block sitting in Tom's chat with no Worker to paste it to is itself the
error signal. This does not replace the Output format rule below — it is that rule's visible token,
and the two must agree.

**Acknowledgments.** A Lunch Pail's first turn opens INSIDE its block with
`Lunch Pail <sub-stage> reporting for duty.` Both roles acknowledge an Ursus Major command by title
and identity — `Copy, Ursus Major Tom.` — plus a **brief recap of the instruction**. The recap is
the point: it is where a misread surfaces while it is still cheap to fix.

**Role responsibilities**:
- **Driver**: sets goals, priorities, risk tolerance, final decisions
- **Product Manager**: scope, acceptance criteria, architecture judgment, sequencing, and the final Worker prompts. **Every Worker or revision prompt is delivered IN THE CHAT as ONE four-backtick fenced block, ready to copy in a single action — not only written to a file** — while discussion, verdicts and questions addressed to the Driver are ordinary prose, not a block. Also save it to `PHASE<N>_PROMPT.md`: the repo copy is the record, the pasted block is the deliverable. No codebase claims without verification — verify directly against the repo and database, and use an Inspector prompt only when a change must be traced across more files than a single PM session can hold
- **Inspector**: read-only — inspects the repo, traces logic across files, and reports exact files/functions/classes/risks/ambiguities to the PM. Does not own scope, make undocumented assumptions, design architecture, write code, or declare a phase done — that always requires passing `pytest` output plus explicit PM sign-off against acceptance criteria, not code-reading alone. Reports follow Output format below.
- **Worker**: builds the scoped change and adds/updates tests. If the prompt says "propose a plan," propose a plan — don't write code yet. Do not refactor, rename, or touch files/signatures outside the current phase's scope; flag surprises instead of silently changing approach. Plans and reports follow Output format below.
- **Reviewer**: fresh review for bugs, regressions, edge cases, scope creep, and Architecture Rule compliance — run `pytest` first (full output in the summary). Reports follow Output format below.


### Prompt and test evidence (added 2026-09-05; split 2026-09-08)

These are Worker-facing rules. The economics argument behind them — front-loading, prompt size vs.
rounds, and which defects were whose — is PM material and lives in `PM_HANDOFF.md` (Usage, and "The
PM-mistake pattern") plus `PROCESS_EFFICIENCY.md`. Do not read it to build.

- **The PM specifies the property and the failure; the Worker writes the test.** The PM cannot run
  `pytest`, so a test written into a prompt is unverified code shipped as an instruction. The prompt
  states what must be locked and what wrong behaviour must make it fail; **you** design the test and
  **report the fail-first evidence**. The PM does still pre-specify what it can author or verify —
  acceptance numbers derived against live data, display copy, layout details, scope boundaries,
  design rulings — and those are binding.
- **Batch API unknowns into ONE probe, before the first plan.** When a phase adopts an unfamiliar
  API, the Worker runs a single probe covering: **(a)** how it interacts with the state we already
  rely on, **(b)** its lifecycle — how it is set and cleared, and whether a programmatic write
  sticks, **(c)** whether it survives user-side transforms invisible to Python (sort, filter,
  reshape), **(d)** its exact return type. One session, one report.
- **A claim only a browser can settle must be settled in a browser.** A Python-side probe that
  passes whether or not the screen renders is not evidence — `Styler.set_table_styles` output is
  marshalled to the frontend even though the canvas-drawn grid has no `<th>` to match it. Equally,
  a pure unit test cannot lock a frontend behaviour: measure it, then record it in Known
  Implementation Decisions rather than writing a test that cannot fail.

**Keep the docs trimmed.** This file is read at the start of every Worker *and* every PM session, so
its size is paid repeatedly. It grew 39 KB → 49 KB during the 5b-3 build alone. Trim **between**
phases, never during one, and never silently — a trim that drops a standing decision is expensive,
so it lands as its own reviewable diff. A new Known Issue or Known Implementation Decision belongs
in `docs/KNOWN_ISSUES.md`, not here — appending one back into this file is the specific way the
2026-09-11 split would rot. This file came down to roughly 33,400 bytes on 2026-09-11
by that move, so the honest measure of growth from now on is `CLAUDE.md` **plus**
`docs/KNOWN_ISSUES.md`, not this file alone.
### Worker Rules — error handling & testing
- Every `try/except` must handle a specific known failure mode, or re-raise after adding context. No bare `except:`/`except Exception:` that silently continues.
- **A source-anchored test must be written against the POST-edit source, and needs BOTH a
  fail-first run against the pre-edit source AND a positive control against the actual post-edit
  source.** Fail-first alone proves only that the test can go red — never that it goes green on the
  change it is meant to lock. Two phases have hit this: 5c-1's first draft anchored a regex on
  `"Market Cap ($M)"` while that same phase was rewriting the label to `"**Market Cap ($M)**"`, and
  5c-3's redesigned lock passed its fail-first run yet matched **zero** sites in the implementation
  it guards. Any source-anchored assertion is a compound condition on text this phase may be
  moving; run it against the text as it will read after your edit, and report both runs.
- Write tests before or alongside implementation, named `tests/test_<module>.py`. One clear assertion per test beats many weak ones — if you could delete the implementation and the test would still pass, the test is broken. Use small (5–10 row) synthetic DataFrames with known inputs/outputs, and cover edge cases: `NaN` inputs, zero denominators, negative values, the `"#N/A N/A"` string, and all-missing-data rows.
- **Verification efficiency**: run the full verification chain (the phase's real end-to-end command, snapshot comparison, validation notebook, Streamlit check) ONCE, at the end of the phase, and report it once. Do not re-run a check after every intermediate change. Re-run a specific check mid-phase only when you've changed something that could plausibly break that specific thing, and say why.

### Output format (mandatory) — the test is WHO READS IT NEXT

The block exists for one reason: **Tom pastes text between windows by hand.** So the question is
never "is this a plan, a report, or a summary?" — it is **"will Tom paste this onward into another
window?"** If yes, it is ONE block. If he is the final reader, it is ordinary prose.

**ONE BLOCK — always:**
- **Worker / Inspector / Reviewer → PM:** every plan, every build report, every review, and **the
  closing summary of an execution turn** — including a turn that only applied corrections or only
  committed. If the PM's prompt ends "report back: 1… 2… 3…", those numbered results go **inside**
  the block, not after it. Sign it "Lunch Pail <sub-stage> over" as the last line inside the block.
- **PM → Worker:** every Worker prompt and every revision prompt, in the chat as ONE block, and
  saved to `PHASE<N>_*.md` as well. The repo copy is the record; the pasted block is the
  deliverable. A digest, or "it's in the file," is not a delivery. Sign it "Rain Man <sub-stage> to
  Lunch Pail over" as the last line inside the block.

**NOT a block:**
- **PM → Tom (Driver):** discussion, review verdicts and the reasoning behind them, findings,
  recommendations, questions, state summaries, anything asking him to decide. He is the final
  reader; he pastes none of it. Ordinary prose. (Rewritten 2026-09-10 on the Driver's instruction,
  because the previous wording told the PM to block these and it did.) Unsigned, too — the absence
  of a callsign sign-off is what marks it as addressed to him.
- Narration while working through an approved task — fine and expected. Only the turn's closing
  summary is the block.

**Mechanics:** four-backtick outer fence, so inner triple-backtick code blocks survive. Output it
once — no prose repeat of the same content. Nothing outside the block but an optional one-line
preamble.

### Before You Report Done (mandatory checklist)
- [ ] Output is ONE four-backtick fenced block per Output format above — plan, build report, review, AND the closing summary of an execution turn, including a turn that only applied corrections or only committed. Numbered "report back" results go INSIDE the block. Signed "Lunch Pail <sub-stage> over" as its last line, inside the block.
- [ ] Run `pytest`. ALL tests must pass. Include the full output in your summary.
- [ ] Ran the phase's real command end-to-end on real data, once, and pasted the actual output — not just the unit suite. Green tests prove the code does what its tests say; they do not prove the feature is right. Check the output against a number derived independently beforehand. If the run writes to `data/screener.db`, back it up first, state exactly what it wrote, and leave any restore decision to the Driver.
- [ ] New logic has tests: happy path, one edge case, one boundary condition.
- [ ] New functions have docstrings covering inputs, outputs, and NaN/edge-case behavior.
- [ ] Imports are correct and minimal — no unused imports, no circular imports.
- [ ] Confirmed compliance with Architecture Rules 1–10 above (no web/db imports in calc functions, decimal convention, explicit NaN handling, config-sourced weights/defaults, M-Score excluded from the composite, `PERCENTRANK.INC`-compatible ranking).
- [ ] No broad refactors outside the scope of the current task.

Recurring bug patterns worth re-reading before touching `transform.py`/`score.py`: see `docs/BUG_PATTERNS.md`.
Known Issues and Known Implementation Decisions: see `docs/KNOWN_ISSUES.md`.

## Known Issues (do not fix unless explicitly scoped into current phase)

**MOVED to `docs/KNOWN_ISSUES.md` on 2026-09-11** — tracked, so nothing is lost from a clone; moved
out because it is consulted on demand, not read at session start. Read it before touching anything it
covers, and add new Known Issues THERE, not here.

## Known Implementation Decisions

**MOVED to `docs/KNOWN_ISSUES.md` on 2026-09-11** — same file, kept in its original order below the
Known Issues section, so the `kind='strict'` cross-reference still resolves.
