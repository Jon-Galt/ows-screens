# Phase 3b — Onboard the 4 Curated Screens

*Drafted 2026-08-28, after Phase 3a implementation review. Grounded in a direct inspection of the four `* Data` sheets in `OWS Screens.xlsx`.*

## Prerequisite

Phase 3a's two blocking review items (the missing per-screen pipeline regression lock, and moving config loading out of `score.py`) land and merge first. 3b builds directly on the `curated` screen type and per-screen table design that 3a established, so it should not start against an unmerged 3a.

## Decisions taken (Tom, this session)

- **Rationale is a snapshot**, replaced wholesale by each Canary Excel upload — but the schema is designed so a separate annotations layer can be added later without migrating existing tables.
- **3b covers the 4 curated screens only.** Rising Short Interest moves to its own phase (different source, different shape, and an eventual scoring layer).
- **The cross-screen overlap view is deferred until after automation.** 3b still populates `screen_membership` for each curated screen, so the data is ready and waiting when that phase arrives.

## What the data actually looks like

All four backing sheets — `Cyclical Data`, `Competition Data`, `Structural Data`, `Management Comp Data` — share an **identical 11-column schema**, verified directly:

```
daily_traded_value, exchange_symbol, locations, market_cap, name, sector,
stock_performance, ticker_symbol, rationale, scores,
valuation_ev_revenue_ntm_percentile
```

Row counts: Cyclicals 33, Competition 109, Structural 135, Management Comp 21 — **298 rows, 222 unique tickers**, so 76 rows are cross-screen appearances. No duplicate tickers within any sheet, no whitespace-damaged tickers, and **zero null or malformed `scores` fields** across all 298 rows, with exactly three score keys throughout (`Accounting And Disclosure`, `Fraud`, `Insider`).

One schema, four screens, clean data — so this is one loader, not four.

### Data quirks the loader must handle

1. **Three numeric columns arrive as quote-wrapped strings**: `daily_traded_value`, `market_cap`, and `stock_performance` come through as `'"36748675276.212273"'` — literal double-quote characters inside the string. The Excel display sheets handle this with `VALUE(SUBSTITUTE(...,"""",""))`. The loader must strip the embedded quotes before numeric coercion, or every one of these becomes NaN.
2. **`valuation_ev_revenue_ntm_percentile` is a real float on a 0–100 scale** (observed range 50 to 99.9), not quote-wrapped.
3. **`scores` is a packed string** — `Accounting And Disclosure: 59 | Fraud: 35 | Insider: 40` — that needs parsing into three numeric columns.
4. **Read the `* Data` sheets, never the display sheets.** The display sheets (`Cyclicals`, `Competition`, etc.) are sorted *views* whose formulas reference arbitrary source rows — e.g. display row 2 pulls from data row 10. The `Data` sheets are the source of truth.

### Unit conventions to settle (Architecture Rule 2)

Rule 2 requires percentages stored as decimals. Derived from the Excel display formulas:

| Field | Raw form | Proposed stored form |
|---|---|---|
| `market_cap` | $M (display ÷1000 → $B) | $M — **matches the existing Short Screen's `market_cap`**, so no conflict |
| `daily_traded_value` | dollars (display ÷1e6 → $M) | $M, for consistency with the above |
| `stock_performance` | percent ×100 (`"667.81"` = 667.81%) | ÷100 → decimal, per Rule 2 |
| `valuation_ev_revenue_ntm_percentile` | 0–100 | ÷100 → 0–1, matching how every existing factor score is stored |

The last two are the ones worth an explicit decision rather than a silent choice — flagging them for the Worker to confirm, not assume.

## The workbook is a manual consolidation — and 3b removes the need for it

**Confirmed by Tom:** Canary exports each screen *separately*. `OWS Screens.xlsx` is a manual consolidation — the `* Data` tabs are hidden (verified: all five `Data` tabs have `sheet_state = hidden`) and hold raw Canary export content pasted in, while the visible tabs are formula-driven views that pull from them.

Two consequences, both good:

1. **The 11-column schema inspected above IS the raw Canary export format**, not an artifact of consolidation. The quote-wrapped numerics and the packed `scores` string are how Canary emits data, so the loader spec below is grounded in the real input rather than a derived one.
2. **Phase 3a's `data/uploads/<screen_id>/` convention is correct exactly as built.** An earlier draft of this plan proposed generalizing the ingest config to let several screens share one source file — that generalization is now unnecessary and has been dropped. Each curated screen reads its own export from its own folder, which is what 3a already does.

**Eliminating the manual consolidation step is an explicit goal of this phase.** Today Tom pastes four separate Canary exports into hidden tabs of a combined workbook on every refresh. After 3b, the refresh is four files dropped into four folders and the pipeline does the rest. The consolidated workbook stops being an input and becomes useful only as a validation reference and test fixture.

The one detail still to confirm at implementation time is what a *standalone* Canary export file looks like on disk — its file format (`.xlsx` vs `.csv`) and, if Excel, its sheet name. The column schema is already settled by the hidden tabs; only the file envelope is unknown. The Worker should confirm this against one real export as its first step rather than assuming.

## Proposed scope

1. **A curated ingest path.** One loader shared by all four screens: reads its named sheet, strips embedded quotes, coerces numerics, applies the unit conventions above, parses `scores` into three numeric columns, and writes to `curated_data__<screen_id>` (reusing 3a's per-screen table design unchanged). Keep the raw `scores` string alongside the parsed columns for provenance.
2. **Four new `config.yaml` screen blocks**, `type: curated`, each reading from its own `data/uploads/<screen_id>/` folder per the existing 3a convention. No `factor_weights`, no `scoring` block — curated screens have neither.
3. **Type-aware pipeline dispatch.** Curated screens do not run `transform.py` or `score.py` — there is no percentile ranking, no composite, no M-Score. Invoking those against a curated `screen_id` must fail with a clear, explicit error rather than doing something undefined.
4. **`screen_membership` populated** for each curated screen, so the deferred overlap view has its data ready.
5. **UI: a screen selector**, plus a curated display mode — the table, the narrative rationale, and the three risk scores — while the existing quant view (factor breakdown, M-Score) stays exactly as it is for the Short Screen.
6. **Annotations-ready schema**: a stable `(screen_id, ticker)` natural key so a future annotations table can join to these rows without migrating them.
7. **Tests**: the quote-stripping and unit conversions, `scores` parsing (including a malformed input, even though the current file has none), curated-vs-quant dispatch, and one end-to-end curated ingest against a small fixture.

### Explicitly out of scope

Rising Short Interest; the cross-screen overlap view; any scoring or ranking for curated screens; automation; the Canary API; annotation editing UI.

## Acceptance criteria

- All four curated screens load, with row counts matching the source exactly (33 / 109 / 135 / 21).
- Spot-checked values match the workbook's own display sheets after unit conversion — e.g. Micron's market cap renders as ~$1,062,488M, its 1-year performance as 6.6781.
- No NaNs in `market_cap`, `daily_traded_value`, or `stock_performance` (which is what a missed quote-strip would produce), and all 298 rows yield three parsed scores.
- `screen_membership` totals 298 rows across the four screens, covering 222 unique tickers.
- Running `transform.py` or `score.py` against a curated screen fails with a clear error.
- The Short Screen is completely unaffected — same output, all existing tests still passing.
- **The manual consolidation step is gone**: four raw Canary exports dropped into their four upload folders produce the same four screens, with no combined workbook required anywhere in the path. Validate by loading from the standalone exports and confirming the result matches what the consolidated workbook's hidden tabs contain.

## Revised phase sequence

| Phase | Content |
|---|---|
| 3a | Multi-screen architecture *(implemented; 2 review fixes pending merge)* |
| **3b** | **The 4 curated screens** ← next |
| 3c | Rising Short Interest (+ its percentile/composite scoring layer) |
| 3d | Automation — scheduled refresh, validation reporting, run history |
| 3e | Cross-screen overlap view |
| 3f | Canary API integration (API-accessible data only — not the rationale) |
| 4 | Expanded analytics — historical tracking, sector-relative scoring, backtesting |

## Open questions

1. ~~Is the Canary export one workbook or separate files?~~ **Answered:** separate per screen; the combined workbook is Tom's manual consolidation, and removing that step is now a goal of this phase.
2. **Remaining, for implementation time:** the file format and sheet name of a *standalone* Canary export. The column schema is settled; only the file envelope is unknown. First Worker step should be to confirm it against one real export rather than assume.
3. Does the Rising Short Interest export follow the same separate-file pattern? It matters for Phase 3c, not 3b — its hidden `Short Int. Data` tab has a different, Bloomberg-shaped schema with a two-row header preamble, so it needs its own loader regardless.
