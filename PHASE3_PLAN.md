# OWS Screens — Next Phase Plan

*Drafted 2026-08-28. Grounded in a fresh clone of `ows-screens` (all 101 tests passing) and a direct inspection of `OWS Screens.xlsx` on Tom's machine. This is the roadmap of record and is kept current as phases complete — it is not a frozen snapshot of the original proposal.*

**Note on drift from the original draft:** the executed sequence diverged from Section 4 as first written below — Phase 3b onboarded the 4 curated screens only, and Rising Short Interest became its own Phase 3c, rather than both shipping together as one "onboard the 5 new screens" phase. Git history holds the original draft if the reasoning behind that split is ever needed.

## 1. Where the project actually stands

| Phase | Status |
|---|---|
| Phase 1 — Replication | **Done.** All 30+ metrics, percentile ranking, and the M-Score are implemented and validated. |
| Phase 2 — Web UI | **Done.** `src/app.py` has sector/industry filters, mkt cap and score sliders, M-Score highlighting, drill-down, and CSV/Excel export. |
| Phase 3a — Multi-screen architecture (foundation) | **Done.** |
| Phase 3b — Onboard the 4 curated screens | **Done.** |
| Phase 3c — Onboard Rising Short Interest | **Done** (ingest + transform only — no factor model yet, so no scoring). |
| Phase 3d — Automation + cross-screen overlap view | Scoped below, not built |
| Phase 3e — Canary API integration | Ideas only |
| Phase 4 — Expanded analytics | Ideas only |

Tom's four selected goals — folding in 5 new Excel-based screens (plus building for future extensibility), automating data refresh, integrating new data sources (e.g. Canary), and expanding analytics — layer on top of this. Tom asked that the **multi-screen architecture be scoped first**, since it changes the shape of everything else.

## 2. What the 5 new screens actually are

`OWS Screens.xlsx` (on Tom's machine, not yet in the repo) contains 5 screens, each with a display sheet and a backing "Data" sheet, plus a cross-screen `Summary` sheet:

- **Cyclicals**, **Competition**, **Structural**, **Management Comp** — these four are *not* percentile-composite screens like the main Short Screen. Each is a curated list of names sharing a common metadata schema (ticker, name, market cap, sector, ADV, 1yr stock performance, EV/Sales percentile) plus a **narrative `rationale` column** written per name, and a **`scores` field** formatted as `Accounting And Disclosure: NN | Fraud: NN | Insider: NN`. **Confirmed by Tom:** both the `scores` field and the narrative rationale are Canary output, delivered *inside the Excel file Tom exports from Canary* — and the narrative rationale in particular is **not available via the Canary API**. So the refresh mechanism for these screens is uploading that Canary Excel export (scores + rationale already baked in), not an API call. This decouples the curated screens from the "Canary API integration" goal: the API work (Phase 3d) is a separate, later concern for other data, not the source for these four screens' rationale/scores.
- **Rising Short Interest** — this one *is* quantitative: it's built directly off a raw Bloomberg export (`Short Int. Data`, same shape as the main screener's raw ingest) with formula-driven ratios (SI change vs. 3-month and 6-month lookbacks, etc.), currently without a percentile/composite score — closer to a simple ranked/filtered screen today. **Confirmed by Tom:** percentile/composite scoring for this screen is a wanted future addition, so it should be onboarded as a `quant_composite`-type screen with room for a scoring layer, even if the first cut ships as ranked/filtered.
- **Summary** — cross-tabulates which of the 5 screens each ticker appears in, via `COUNTIF` formulas hardcoded to each screen's row range. This "which names show up across multiple independent theses" view is genuinely useful and worth making a first-class, general feature — but as built today it breaks (or needs manual formula edits) every time a screen is added or resized.

Net effect: the current architecture (one universe, one set of factor weights, one SQLite pipeline shaped around percentile-composite scoring) doesn't have a slot for curated/narrative screens, multiple independent universes, or a general overlap view. That gap is exactly what "multi-screen architecture" needs to close.

## 3. Proposed architecture direction

1. **Introduce a `screens` concept.** Each screen is a config entry — name, type (`quant_composite` vs. `curated`), its own ticker universe, its own metadata/factor schema — instead of the pipeline assuming one fixed ~1,300-name universe and one `config.yaml` factor block.
2. **Support two screen types**, not one:
   - *Quant composite* (today's Short Screen pattern: ingest → transform → percentile rank → weighted composite). Rising Short Interest is the natural second instance of this type. It ships ranked/filtered first, then gains its own percentile/composite scoring layer (a wanted addition) — which is exactly why it belongs in the `quant_composite` type rather than being treated as curated.
   - *Curated*: structured metadata + narrative rationale + optional external risk-score fields, populated by research input rather than computed — but still flowing through the same ingest/storage/UI/export layers as quant screens.
3. **Extend the SQLite schema**: add a `screens` table, and scope `raw_data` / `transformed_data` / `scored_data` (or parallel curated-screen tables) by `screen_id` rather than one global universe.
4. **Rebuild the overlap view as a real feature**, not a hardcoded formula: a `(screen_id, ticker)` membership table makes "which names appear in N screens" a simple query that scales to any number of screens automatically.
5. **Generalize `config.yaml`** from one `factor_weights` block to a per-screen config block.
6. **Generalize the UI**: a screen selector, per-type drill-down (rationale + risk-score display for curated screens vs. factor breakdown for quant screens), and a first-class cross-screen overlap tab.
7. **Introduce a pluggable ingest layer**: Bloomberg CSV/XLSX (existing main screen), the Canary Excel export (the 4 curated screens — read the per-screen "Data" sheets with their own column maps, rationale + scores already embedded), the Bloomberg short-interest export (Rising Short Interest), and — separately — a future live Canary API adapter for data that *is* API-accessible (not the rationale, which isn't). Note the existing loader hardcodes `sheet_name="Data"` and a Bloomberg-specific column map, so the abstraction is a real, required change, not cosmetic.

## 4. Sequencing

- **Phase 3a — Multi-screen architecture (foundation). Done.** Schema + config generalization, ingest abstraction, migrated the *existing* Short Screen onto the new schema with zero behavior change, generalized the UI to support N screens.
- **Phase 3b — Onboard the 4 curated screens. Done.** Ingest adapter for Cyclicals/Competition/Structural/Management Comp, loaded into the new schema, surfaced in the UI.
- **Phase 3c — Onboard Rising Short Interest. Done.** Ingest + transform for the second `quant_composite` screen. No scoring layer yet — deferred as a research decision, not built speculatively.
- **Phase 3d — Automation + cross-screen overlap view.** Scheduled refresh, a per-run validation report (missing columns, NaN-rate spikes, universe size changes), and a run-history log spanning every screen — folded together with the overlap view: `screen_membership` already has the data from 3a-3c, so the overlap view is largely a query plus a UI tab, not a phase of its own.
- **Phase 3e — Canary API integration.** A separate goal from the curated screens (whose rationale/scores arrive via the Canary Excel export, not the API). Scope here is live API sourcing for data that *is* API-accessible — e.g. risk scores as a potential new factor/enrichment for the main Short Screen or other screens.
- **Phase 4 — Expanded analytics. PREREQUISITE, added 2026-08-31:** the database has
  **no time dimension in any table** (verified) and no forward-return data — only trailing
  columns (`perf_1w`, `perf_1m`, `week_52_high_pct`, `week_52_low_pct`, curated
  `stock_performance`). There is therefore no dependent variable, and because every stage
  writes with `if_exists="replace"`, **each refresh overwrites the previous cycle's scores.**
  Every refresh run to date has produced and discarded exactly the observation a backtest
  needs. Fundamentals history can in principle be reconstructed later (expensively, via
  point-in-time sourcing); score history cannot be reconstructed at all without solving
  that same problem first. **This is the only item on the roadmap with a clock on it.**

  Consequence for **3d Part 2b**: its per-run snapshots were approved for data-quality
  reasons, but with a schema that also carries the run date and leaves a slot for forward
  returns to be joined later, those same snapshots become the first brick of the backtest
  dataset. Design the snapshot shape with that second purpose in mind before the Worker
  starts — it is the cheapest high-value action available and it cannot be retrofitted
  onto data that was never recorded.

  Ordering for any backtest work: **measurement before model.** Build the harness against
  the existing 24 factors first and establish an honest baseline, then ask which existing
  factors carry signal (a 24-hypothesis question, correctable), and only then add
  datasets — one at a time, each with a stated economic mechanism and a hold-out period.
  Do not assume the inherited weights will backtest poorly: they encode years of analyst
  judgment, informed priors routinely beat naive optimization out of sample, and
  improvement measured against a methodologically weak baseline is indistinguishable from
  fitting the baseline's flaws.

- **Phase 4 — Expanded analytics.** Historical tracking, sector-relative scoring, backtesting, watchlist/annotation — made easier by the screens/run-history model already in place from 3a-3d.

This keeps each phase narrow (matching the "narrow phases, no broad refactors" principle already in `CLAUDE.md`).

## 5. Open questions for Tom

1. ~~Is the `scores` field in the 4 curated screens coming from Canary?~~ **Answered:** yes — scores *and* narrative rationale come from the Canary Excel export; the rationale is not API-accessible, so these screens refresh by file upload, not API.
2. ~~Should Rising Short Interest get percentile/composite scoring?~~ **Answered:** not yet — it onboarded in 3c as a `quant_composite` screen with no factor model; a scoring layer is a future research decision, not scheduled to a phase yet.
3. Should curated-screen rationale text be editable in the app itself (for updating a thesis later), or is it treated as a point-in-time snapshot re-generated each research cycle (i.e. replaced wholesale by the next Canary Excel export)? **Still open.**
4. Any additional screens beyond these 5 already planned, that should shape how general the schema needs to be from day one? **Still open.**

5. **Direct data sourcing beyond Canary — unscoped idea, not a phase.** Tom's longer-term
   direction is to move as much data sourcing as possible from manual exports to direct
   API access. SEC EDGAR (XBRL company facts) is the obvious candidate for the *filed
   historicals* half of the Short Screen's inputs — 3-year averages, balance-sheet items,
   cash-flow and accruals inputs, dilution, the working-capital series. It cannot replace
   Bloomberg wholesale: of the 81 mapped ingest columns, the forward-looking ones (NTM
   P/S, NTM margins, forward revenue CAGR, analyst ratings, Non-GAAP/adjusted EPS
   estimates) are consensus estimates that EDGAR does not carry, and short interest is
   exchange/FINRA data, not a filing. So the realistic shape is a mixed-source Short
   Screen, not a substitution. **Raised while brainstorming 2026-08-31; no phase, no
   scope, no decision.** The blocking question if it ever gets scoped is point-in-time
   correctness (see below), because getting it wrong silently invalidates Phase 4's
   backtesting rather than failing loudly.

## 6. Draft Phase 3a Worker prompt

See `PHASE3A_WORKER_PROMPT.md` for a copy/paste-ready prompt scoped to just the foundation phase, in the format your `CLAUDE.md` specifies for Worker handoff. Review and adjust before sending it to Claude Code — per your own PM rule, this should be approved as-is or revised, not run with partial changes.
