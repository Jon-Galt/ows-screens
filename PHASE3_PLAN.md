# OWS Screens — Next Phase Plan

*Drafted 2026-08-28. Grounded in a fresh clone of `ows-screens` (all 101 tests passing) and a direct inspection of `OWS Screens.xlsx` on Tom's machine.*

## 1. Where the project actually stands

| Phase | README status | Actual status |
|---|---|---|
| Phase 1 — Replication | — | **Done.** All 30+ metrics, percentile ranking, and the M-Score are implemented and validated; 101/101 tests pass. |
| Phase 2 — Web UI | — | **Done**, though `CLAUDE.md` still says "Phase 1 in progress" — that line is stale and should be corrected. `src/app.py` has sector/industry filters, mkt cap and score sliders, M-Score highlighting, drill-down, and CSV/Excel export. |
| Phase 3 — Automation | Scoped, not built | Not started |
| Phase 4 — Expanded functionality | Ideas only | Not started |

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

## 4. Recommended sequencing

- **Phase 3a — Multi-screen architecture (foundation).** Schema + config generalization, ingest abstraction, migrate the *existing* Short Screen onto the new schema with zero behavior change (must still pass all 101 current tests, or their equivalents), generalize the UI to support N screens plus the overlap view. No new screens' data loaded yet — this phase is purely about proving the generalized architecture against the screen that already works.
- **Phase 3b — Onboard the 5 new screens.** Ingest adapters for the 4 curated screens and Rising Short Interest, loaded into the new schema, surfaced in the UI.
- **Phase 3c — Automation.** Scheduled refresh, validation reporting, run-history — now spans all screens instead of being single-purpose.
- **Phase 3d — Canary API integration.** A separate goal from the curated screens (whose rationale/scores arrive via the Canary Excel export, not the API). Scope here is live API sourcing for data that *is* API-accessible — e.g. risk scores as a potential new factor/enrichment for the main Short Screen or other screens.
- **Phase 4 — Expanded analytics.** Historical tracking, sector-relative scoring, backtesting, watchlist/annotation — now much easier with a real screens/run-history model already in place from 3a/3c.

This keeps each phase narrow (matching the "narrow phases, no broad refactors" principle already in `CLAUDE.md`) while making sure the foundation phase is validated against real, already-passing behavior before anything new is layered on.

## 5. Open questions for Tom

1. ~~Is the `scores` field in the 4 curated screens coming from Canary?~~ **Answered:** yes — scores *and* narrative rationale come from the Canary Excel export; the rationale is not API-accessible, so these screens refresh by file upload, not API.
2. ~~Should Rising Short Interest get percentile/composite scoring?~~ **Answered:** yes, eventually — onboard it as a `quant_composite` screen (ranked/filtered first, scoring layer added later).
3. Should curated-screen rationale text be editable in the app itself (for updating a thesis later), or is it treated as a point-in-time snapshot re-generated each research cycle (i.e. replaced wholesale by the next Canary Excel export)?
4. Any additional screens beyond these 5 already planned, that should shape how general the schema needs to be from day one?

## 6. Draft Phase 3a Worker prompt

See `PHASE3A_WORKER_PROMPT.md` for a copy/paste-ready prompt scoped to just the foundation phase, in the format your `CLAUDE.md` specifies for Worker handoff. Review and adjust before sending it to Claude Code — per your own PM rule, this should be approved as-is or revised, not run with partial changes.
