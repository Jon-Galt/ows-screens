# Known Issues & Known Implementation Decisions

Moved out of `CLAUDE.md` on 2026-09-11 to keep the session-start read cheap — this file is
tracked, so nothing is lost from a fresh clone, but it is consulted on demand rather than read
at the start of every Worker/PM session. Add new Known Issues and Known Implementation Decisions
HERE, not back in `CLAUDE.md`.

## Known Issues (do not fix unless explicitly scoped into current phase)

### Functional
- Universe is defined implicitly by whatever rows appear in the uploaded file. Explicit universe management (add/remove tickers, maintain a master list) is deferred.
- The composition/misfile check's headroom is thin specifically for competition's export: correctly filed, it scores Jaccard 0.260 against its own stored baseline vs. 0.221 against structural's — a +0.039 margin, the thinnest of the six screens — because competition and structural genuinely share a meaningful number of names. The same +0.039 margin is what catches competition's export if it's misfiled into structural's folder instead. (structural's own export is not fragile in the same way: 0.399 against its own baseline vs. 0.187 against competition's, a comfortable +0.212.) Not a bug — all 30 misfile permutations across the six screens do flag; competition's pairing with structural just has the least headroom of the six correct-placement scores.
- **Small screens with heavy turnover can false-positive the composition/misfile check.** `management_comp` is down to 10 rows; if it turns over completely on a future refresh, its own Jaccard against its stored baseline is 0.0, and any peer screen sharing even a single ticker with the new export beats that trivially, blocking a perfectly legitimate write. `--force` exists mainly as the escape hatch for this case.
- **The composition/misfile check can also false-positive on a small incremental upload to an accumulating screen.** `negative_expert_transcripts`' `raw_data` accumulates (Rule 10 exception), so `check_composition_misfile` Jaccards each upload against the WHOLE corpus, not just the prior upload. Measured: *Month Aug–Sep* (83 tickers) vs. stored *Special Jul–Aug* (101) passes at own 0.064 vs. best peer `short_screen` 0.053 — a +0.011 margin, thinnest in the project (thinner than +0.039 above) — a mild variation flips it (own 0.003 vs. structural 0.019). Packs share only 11 of 173 tickers, so low recurrence is normal. Not fixed — see `PHASE6_SCOPE.md` §7. **`src/validate.py` is not to be "fixed" for this margin — that is the Driver's ruling, not an open task.**
- **Phase 6a's acceptance run left two rows in the append-only run-history tables for `negative_expert_transcripts` — deliberate, not an accidental double run (T7's shape).** `refresh_runs` holds run_ids `20260910T210341Z-0b7469` and `20260910T210401Z-a5fbff` (both `--screen negative_expert_transcripts`, run_date 2026-09-10); `refresh_screen_runs` has two matching PASSED/216-row entries; `refresh_snapshots` has 346 rows across both run_ids on that one date — the only screen in the table with more than one run. The second run was the upsert-idempotence proof the phase's acceptance criteria required. Verified: `history.latest_snapshot_per_date()` already resolves this correctly, keeping the later run_id (`...-a5fbff`) and returning 173 rows — no fix needed.
- **A `negative_expert_transcripts` DocID backfill can duplicate transcripts.** 14 of 215 initial-vintage transcripts lack a source `DocID` and are stored under a synthesized `SYN-<hash>` id (mechanism: `src/transcript_ingest.py`'s `synthesize_doc_ids` docstring). A future re-export with `DocID` populated would miss the `(doc_id, ticker)` key of the stored `SYN-` rows and insert IN ADDITION, duplicating transcripts and inflating `mention_count`. **The Driver has agreed to backfill `DocID` on every row in future exports**, which is what makes this hazard worth guarding against rather than hypothetical. **Phase 6b ships detection**: `find_duplicate_transcript_groups` groups the accumulated `raw_data` on `(transcript_date, theme, ticker)` and flags any group resolving to more than one `doc_id`, logging a `WARNING` per group — it reports only, never merges or deletes. Because detection runs AFTER the insert rather than preventing it, the **operational rule still stands: re-ingesting a stored period requires clearing `raw_data` first** — `data/uploads/negative_expert_transcripts/_archive/` keeps the corpus rebuildable, which makes that safe.
- **29 rows in `historical_whiteboard_shorts` carry a spurious stored `relative_spy_performance`** — a null-price formula artifact in the source workbook, not a real measurement, discovered by Phase 4b's `check_event_window_replication` on its first real run and confirmed structurally by `src/whiteboard_horizons.py`'s `flag_spurious_stored_relative`. Wherever `wba_price` and/or `wbr_price` is null, the workbook's own formula silently treats the missing price leg as zero rather than leaving the cell blank, so the stored `relative_spy_performance` equals the SPY benchmark move ALONE (`bench_move - 0` instead of `bench_move - price_move`) — not this stock's relative performance. Split by outcome: 20 Open / 7 Removed / 2 Initiation. The null-price condition is load-bearing, not a coincidence: one real row (ALAB) has both prices present and its stored value happens to be close to the SPY move too (the stock genuinely went nowhere) and is correctly NOT flagged — `flag_spurious_stored_relative`'s test suite locks this distinction. **Consequence: Phase 4a's published naive Whiteboard comparison (`summarize_whiteboard_naive`) rests on 9 fabricated values out of its 95 Removed+Initiation rows — 7 of the 71 Removed and 2 of the 24 Initiation.** Where a row is flagged spurious, Phase 4b's own vendor-computed relative return (from `whiteboard_horizon_returns`) is the correct figure; the stored column is not a measurement for that row. Detected and reported only — `historical_whiteboard_shorts` is NOT modified (4a's tables stay a faithful import of the source file, per that module's docstring).
- **This is invisible to 4a's `check_benchmark_consistency`**, which is why it went unseen in Phase 4a: that check masks on `implied.notna()`, and `implied` (`bench_move - price_move`) is NaN whenever a price leg is null — it structurally cannot see the rows whose prices are missing, which are exactly the rows where the source formula misbehaves. Do not "fix" `check_benchmark_consistency` to catch this in a future phase without deliberately scoping that change; it is a documented blind spot, not scoped for repair here.
- **Stooq (the documented fallback vendor for `src/price_history.py`'s price pull) is currently non-functional.** Its public CSV endpoint (`stooq.com/q/d/l/`) now returns a JavaScript bot-challenge page instead of CSV data. The fallback path is coded and unit-tested (including a regression test for this exact response shape), but was not made to work around the challenge — bot-detection circumvention is out of scope regardless of purpose. Coverage currently rests entirely on yfinance plus manual Bloomberg fills (`ingest_manual_fill`).
- **`notebooks/validation.ipynb` references `OWS Short Screen (March 2026).xlsx`, which is not present in the repo** (only the April 2026 workbook is kept — see File Layout), so the notebook cannot currently be run as-is. Do not re-point it at the April workbook and do not re-run it as a documentation fix: the Known Implementation Decision below on `kind='strict'` percentile ranking says that choice was validated against the March 2026 file specifically, and re-pointing the notebook would be a scoped validation decision (does `kind='strict'` still hold against the April file?), not a documentation tidy.
- **The main table's row highlight may not repaint after a filter change while a browser-side column sort is active.** `sync_drilldown_selection` re-seeds the table's selection with a position computed in server-side `display_df` order; Streamlit gives Python no signal that the user has sorted a column in the browser. Observed in one Linux/Chromium reproduction as the selection checkbox clearing after a filter change under an active sort; not reproducible on macOS. **The drill-down panel is unaffected in every observed case and always names the correct stock**, and any row click immediately restores the highlight. Cosmetic; not a data-correctness defect.

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
     unlike rows). `src/selection.py`'s
     `should_process_cell_selection()` is built around it: an empty
     `pre_cells` is never processed, so a still-good persisted `(ticker,
     column)` survives a filter change that keeps the ticker, and only
     `render_cell_derivation_panel()`'s own `find_ticker_row()` check (never
     a re-resolve) decides whether a filter that excludes the ticker should
     clear the panel.

- **Streamlit drops a widget's `session_state` entry on the first script run in which that
  widget is not instantiated — no grace period — while a plain, non-widget key survives
  indefinitely.** Read from the installed 1.63.0 source during Phase 5d
  (`runtime/state/session_state.py`): `on_script_finished` calls
  `_remove_stale_widgets(active_widget_ids)` at the end of **every** run unconditionally, and the
  pruning is filtered by `is_element_id`, so a key never bound to a widget's `key=` is never
  touched. **Consequence, and the reason this is recorded rather than unit-tested:** any panel
  sitting behind an early return in `main()` loses all of its widget state the moment the user
  visits another screen and comes back. Phase 5d's weight panel is the live instance, and the
  failure it produced was **misattribution, not lost work**: the panel silently reverted to
  config weights while the analyst believed a preset was still in force. Three pieces hold it
  together:
  `_weight_last_applied_preset` is a deliberate non-widget key that survives;
  `should_reapply_preset`'s `weights_absent` trigger re-pushes the weights on re-entry; and
  `resolve_persisted_preset` membership-checks the surviving name against the current options, so
  a preset deleted or made unloadable between visits falls back to Default instead of reaching
  the selectbox or a `presets[name]` lookup. **`weights_absent` checks a SINGLE canary key**, on
  the correct-but-fragile assumption that all 31 weight widgets are instantiated and dropped
  together — they are unconditional siblings in one expander body today. **If any weight widget
  ever becomes conditional, that canary stops being sufficient.** Re-measure if streamlit is
  upgraded.

- **A streamlit colour directive closes at the FIRST `]` and fails silently.** Measured in a
  browser during Phase 5c-2 against the installed streamlit 1.63.0: `:primary[Foo] bar]` renders
  "Foo" in `theme.primaryColor` and `" bar]"` as literal text, **raising nothing** — so a bracket
  in a display name ships as a visible defect, not a crash. 1.63 takes both the palette form
  (`:primary[...]`, resolving to `theme.primaryColor`, already `#1E552D`) and a custom-hex form
  (`:color[x]{foreground="#1E552D"}`). `app.py`'s `format_screen_title()` returns any display name
  containing `[` or `]` **unwrapped** for exactly this reason. Directive parsing is
  **frontend-only** — there is no Python-side regex to read — so this is browser-only and cannot
  be locked by a unit test. Re-measure if streamlit is upgraded.

- **`download_button`'s `icon=` argument validates and raises**, the opposite of the markdown-body
  colour-directive behaviour documented immediately above, which renders an unknown name as literal
  text and raises nothing. `icon=` routes through `streamlit.string_util.validate_icon_or_emoji` →
  `validate_material_icon` → `is_material_icon`, which checks membership in
  `streamlit.material_icon_names.ALL_MATERIAL_ICONS` (4,271 names at 1.63.0) and raises
  `StreamlitAPIException` on a miss. This is why the export buttons' icon constant (Phase 5e,
  `EXPORT_BUTTON_ICON`) can be locked by a plain unit test — a typo fails in the suite instead of at
  render.
