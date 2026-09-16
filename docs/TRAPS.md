**The live trap list, T1–T48.** Moved **VERBATIM** out of `PM_HANDOFF.md` on 2026-09-13 (sub-stage
7b-0). **Nothing was dropped and no T-number changed**, so a T-number cited in `PM_HANDOFF.md`,
`PHASE_HISTORY.md`, `CLAUDE.md`, `PROCESS_EFFICIENCY.md`, `NEXT_PM_PROMPT.md` or any `PHASE*.md`
still resolves: **T1–T10 forward to `PHASE_HISTORY.md`** (they were already a forwarding stub before
this move), **T11–T48 are below.**

**Read this ON DEMAND, not at session start** — before scoping or reviewing a phase that touches
what a trap covers. Same rule and the same reason as `docs/KNOWN_ISSUES.md`: it is reference
material consulted against a specific question, and the session-start read is what its size was
being charged to. **Moving it did not shrink the corpus** — the 2026-09-11 `CLAUDE.md` split is the
precedent and it grew the pair from 45,406 to 49,524 bytes. What a move buys is bytes off the
session-start read, and that is the only saving either split should ever be claimed to deliver.

**Adding a trap.** A trap is discovered in review, and review is followed by the Worker's commit
round — so the new trap's text rides with the build approval and the Worker appends it here in that
same commit. Append with the next free T-number; **never renumber** (a T-number is cited across six
other documents), and never delete a retired one — T39 is kept as a single line for exactly that
reason.

# THE LIVE TRAPS

Everything below still bites. Full derivations in `PHASE_HISTORY.md` under the named phase.

## Data — the 4a/4b traps (T1-T10) — MOVED

**T1-T10 moved VERBATIM to `PHASE_HISTORY.md` under "Carry-forward: the 4a/4b data traps (T1-T10)"
on 2026-09-11**, to pay for Phase 6b's additions. Nothing was dropped and the numbering is unchanged,
so a T-number cited anywhere still resolves.

They are **analysis-facing, not build-facing**: what the three datasets can and cannot answer · why
neither 4a nor 4b can validate the screens · the 29 fabricated `relative_spy_performance` rows and
4a's structural blindness to them · the naive-vs-chained Whiteboard artifact · `price_history`'s
upsert-only rule · the two run-log traps · whole-point rounding · the mixed-benchmark band guard ·
Stooq. **READ THEM BEFORE SCOPING ANY ANALYSIS PHASE** — above all the Whiteboard horizon-returns
analysis, which is the next thing after the build queue and is the phase they were written for. A
display or screen phase does not need them. T3, T4 and T10 also have their full derivations in
`CLAUDE.md`'s Known Issues, which is tracked.

## Display — the 5a colour scale

**T11.** The mid anchor is each column's **MEDIAN**, not the midpoint of min and max, and `lo`/`hi`
are each column's own observed min/max, never a hardcoded 0..1. `ratings_factor` is the only column where they diverge (derivation in `PHASE_HISTORY.md`).
`def_rev_factor` and `liquidity_risk_factor` have **median == minimum** (the majority of both
columns sits at 0.0 via Architecture Rule 7's `**` default) — naive interpolation divides by zero
there; those rows take the **mid** hex. **Green is at the MINIMUM and red at
the MAXIMUM** because factor scores are direction-adjusted and high = more bearish. Intended, not
inverted.

**T12. This is also why R8 ruled brand green as ACCENT ONLY.** The mid anchor is near-white
(`#FCFCFF`), so on a dark ground "exactly median" becomes the brightest thing on screen. **Going dark
= re-deriving the scale, not a config edit.**

## Interaction — the selection machinery

**T13. `st.dataframe`'s selection indices are positional into the frame EXACTLY as passed** — the
sorted, column-subset `display_df`, never into `filtered`. All render paths sort before display.
Mapping a selection back through `filtered` returns a real, plausible, **wrong** company with no
error: short_screen row 1 is VYX in display order and NVDA in `filtered` order (101st CENX/CSX; last
MELI/CADL; structural 1st/54th/last AAPL/LNW/ZETA vs NSC/RBRK/PAG; RSI ACIW/INGM/XPEL vs KDP/BOX/NPK).
**The rule: resolve positionally ONLY against the frame passed to `st.dataframe`, and key by TICKER
for anything else.** `find_ticker_row` calls `reset_index(drop=True)` because `sort_values` does not.

**T14. `display_df` is column-subset and does not carry every column.** It holds `DISPLAY_COLUMNS`
(or its interleaved form) only — never the 20 raw diff-input columns, and not `in_universe`. Reading
a row's *content* from `display_df` renders `N/A` indistinguishably from a genuine NaN. **Read
content from `filtered` by ticker; use `display_df` only to confirm a ticker is on screen.** This
shipped as a real defect in 5b-3 and was caught only by running the app.

**T15. There are FOUR selection-bearing tables**: the three per-screen tables keyed
`{screen_id}_main_table` plus the overlap table keyed `overlap_table` / `overlap_selected_ticker` /
`overlap_table_last_rows`. `is_fresh_selection(pre, last)` in `src/selection.py` is the **one**
predicate distinguishing a real click from a sticky rerun; a third copy was explicitly ruled out.
5b-3 added `should_process_cell_selection`, which is `bool(pre_cells) and is_fresh_selection(...)`.

**T16. Cross-screen nav:** `_pending_nav` → forced `screen_selector` + `_nav_target` (carrying its
OWN `(screen_id, ticker)` so it survives `main()`'s four early returns without misfiring) →
`apply_pending_nav`, called after each sidebar and BEFORE each table. **Any new render path must call
`apply_pending_nav` or navigation silently stops working for that path.**

**T17. `resolve_nav_target` exists because `resolve_selected_ticker`'s precedence 3 returns the top
table row** when `previous_ticker` is not in `display_df`. Without the gate, a click-through to a
ticker the destination screen's filters exclude resolved to a different real company, silently.
**Do not change precedence 3** — it is correct for the filter-change case it was written for. The
gate is what must stay.

**T18. Four measured `streamlit==1.63.0` selection behaviours, none documented upstream**, now in
`CLAUDE.md` Known Implementation Decisions #1–#4: row and cell selections are independent and
neither click disturbs the other · a programmatic `cells: []` push shapes only the return value of
the run it happens in and does **not** durably clear what the next rerun reads (the equivalent row
push *does* durably repaint) · a clicked cell's row index is reported against the frame as passed,
unaffected by a browser-side sort, same as rows · **a data reshape resets `cells` to empty
unconditionally, regardless of whether the clicked ticker survives.** The fourth was reachable only
by reshaping real data in the real app; three scratch probes missed it.

**T19. Column headers are canvas-drawn** (`data-grid-canvas` / glide-data-grid), so no CSS reaches
them — **but streamlit DOES marshal `Styler.set_table_styles` output to the frontend, so a Python
probe asserting the `th` CSS was generated PASSES while the screen shows nothing.** Any
header-styling probe must be **visual**. The complete set of dataframe theme options in 1.63 is
`dataframeBorderColor` and `dataframeHeaderBackgroundColor` — there is **no**
`dataframeHeaderTextColor` and no header font-weight lever. `theme.baseFontWeight` is the whole app's
root weight; `headingFontWeights` is h1–h6 markdown headings. **5c-2 moved the band to `#E8F1EA`:**
column-title text measures a ~130/255 luminance gap against it (readable), but the header's
**select-all checkbox glyph measures only ~22/255** — verified **inert** in
`selection_mode=["single-row","single-cell"]` (clicking it does nothing), so it is cosmetic and the
band stays.

**T20. `st.expander` is NOT lazy** — its body runs on every rerun while collapsed. This is why the
overlap table was put behind cached `get_overlap_df()` / `get_screen_data_and_membership()`. **5c-4
retired that expander** (overlap is its own screen now, so the compute is genuinely lazy at last)
**but the streamlit fact is unchanged and still bites any OTHER expander** — `app.py` still has
several in the drill-down and derivation panels. The caching stays.** (Timings in `PHASE_HISTORY.md`.)

**T21. `APP_FONT_FAMILY` (`app.py:79`) is the single font constant**, locked against
`.streamlit/config.toml`'s actual `font` line by `tests/test_app.py`. **Vega-Lite does not inherit
the theme**, so anything new that draws text needs it explicitly.

## Cross-screen facts

**T22. The corrected distribution.** 1,375 distinct tickers · 1,358 in short_screen's universe ·
**1,171 on ZERO thematic screens** · 187 in-universe on ≥1 · 17 thematic-only OUTSIDE the universe
(and those same 17 are exactly the NaN `overall_score` rows in the overlap frame).
1,171 + 187 = 1,358; + 17 = 1,375. Ceiling **5**. Distribution 0:1,171 · 1:137 · 2:56 · 3:9 ·
4:1 (SON) · 5:1 (IFF). **Re-derived 2026-09-05 and confirmed.**

**T23. The curated screens carry IDENTICAL identity and risk scores for a shared ticker.** Tested
across all 63 tickers on 2+ curated screens, 82 pairwise comparisons: `name`, `sector`,
`market_cap`, all three risk scores, the packed `scores` string, `daily_traded_value` and
`valuation_ev_revenue_ntm_percentile` disagree in **0 of 82**; `stock_performance` in **62 of 82**;
`rationale` in **82 of 82**. So per-screen content is rationale plus `stock_performance`, and
identity belongs **once**. **Identity DOES disagree across SOURCES**: curated vs short_screen match
on `name` **0 of 96** shared structural names ("Microsoft Corporation" vs "MICROSOFT CORP"),
`market_cap` differs ~1% (export vintage, not a defect); RSI vs short_screen agrees exactly on all 62.

## Environment

**T24. The Python floor is 3.10+** (stated in `README.md`'s Quick start — cited without a line
number since 7b-1 moved it); `.venv` is 3.11 and **Tom's Mac defaults to
3.9**. `tomllib` is 3.11-only — a 5b-2 plan proposed it and was sent back, because it would have
worked on Tom's machine and stayed green while breaking on a 3.10 checkout.

**T25. No unit test may depend on gitignored data.** `data/screener.db`, `data/uploads/` and
`data/historical/` are all gitignored; `CLAUDE.md`'s File Layout (Tests) states the rule. Of 26 `create_engine` calls
across `tests/`, **zero** read the live database — synthetic frames and `tmp_path` fixtures only.
Real-data correspondences are verified once, in the acceptance run, and reported there.

**T26. `notebooks/validation.ipynb` references an `OWS Short Screen (March 2026).xlsx`** that is not
in `notebooks/` (only the April 2026 workbook is), so it cannot be run as-is. **Do NOT re-point it at
the April file as a documentation tidy** — `CLAUDE.md`'s `kind='strict'` decision was validated
against the March file specifically, so re-pointing it is a scoped validation decision.

**T27. RETIRED by 7b-1 (`README.md` retired to an orientation page, 2026-09-13) — kept as one entry
so nobody re-derives it.** It named three `README.md` sites calling the cross-screen overlap view
"planned in Phase 3e" when it shipped in 3d Part 1, and forbade tidying them. **Two of the three
were accurate as history and one was wrong as current state, and the trap did not distinguish
them** — which is exactly why the Driver had to rule on what the file was FOR. He ruled 2026-09-13:
the phase narrative moved VERBATIM to `PHASE_HISTORY.md` ("Carry-forward: README's Development
Phases narrative") and the duplicated current-state sections were deleted. **The lesson survives:
a claim can be wrong as current state and correct as history, and a document that mixes both makes
"is this a bug?" unanswerable.** No `README.md` line-number citation exists anywhere now — do not
reintroduce one; the file is short enough to search.

**T28. Google Drive is out of scope** (Driver instruction 2026-09-02) — the local `data/historical/`
workbook is the sole system of record.

**T29. `:material/<name>:` renders in `st.markdown` BODY TEXT on 1.63.0 — measured, not documented.**
Browser-probed before 5c-3's plan. Four findings: it draws a real glyph (a `<span role="img">`
carrying the `Material Symbols Rounded` family), not literal text · streamlit applies
`vertical-align: bottom` itself, so it sits on the baseline beside bold Arial · an **invalid** name
renders as literal text and does **NOT** raise, so a missing mapping ships as a visible defect rather
than a crash · the glyph **inherits the surrounding text colour** (confirmed inside `:red[...]`),
which is why these icons can follow brand green in 5c-2 for free. The icon font ships **weight 400
only**, so an icon inside `**bold**` gets a browser-synthesised faux bold — **5c-3's standing ruling
is `f"{icon} **{name}**"`, icon OUTSIDE the bold.** `ALL_MATERIAL_ICONS` holds **4,271** names on
1.63.0 — the version this project's `.venv` runs, and the one every measurement from 5c-3 through 8b
was made against — and **4,277** on 1.64.0, which is what a fresh unpinned install resolves to today
(PM-measured on the bridge VM at 8c-1). **4,271 was never wrong: the count is version-dependent, and
`requirements.txt` says `streamlit>=1.28`, a FLOOR.** Check membership in the environment you are
about to render in, and re-measure there rather than copying either number. See T48.
**The `icon=` PARAMETER behaves the OPPOSITE way** (measured 5e): it routes through
`validate_icon_or_emoji` and **raises** on an unknown name, so a typo there is loud and CAN be
locked by a plain unit test. Full statement is the Known Implementation Decision in `CLAUDE.md` —
read it there, do not copy it back here.

**T31. `st.logo` caps at 32px** — `"small"` 20 / `"medium"` 24 / `"large"` 32, read from the
installed docstring. **No asset resolution changes this.** 5c-2 needed ~4x, so the call was
**removed** and replaced by `st.sidebar.image()` at the top of the sidebar. **Disclosed cost:
`icon_image` was what drew a mark in the app's upper-left when the sidebar is COLLAPSED, and
nothing does that now.** Accepted because `initial_sidebar_state="expanded"`.

**T32. A streamlit colour directive closes at the FIRST `]` and fails SILENTLY.** Measured in a
browser during 5c-2: `:primary[Foo] bar]` renders "Foo" green and `" bar]"` as literal text — no
raise. 1.63 supports the palette form (`:primary[...]`, resolving to `theme.primaryColor`, already
`#1E552D`) **and** a custom-hex form, `:color[x]{foreground="#1E552D"}`. `format_screen_title()`
in `app.py` returns any display name containing `[` or `]` **unwrapped** for exactly this reason.
**Directive parsing is frontend-only — there is no Python-side regex to read, so this is
browser-only.** **Now a Known Implementation Decision in `CLAUDE.md` (`c33fe65`)** — a top-level
bullet, deliberately NOT a fifth item in the `st.dataframe` list, which stays #1-#4.

**T34. `pytest` on Tom's Mac must be run through `.venv/bin` — AND `which pytest` REPORTS THE WRONG
BINARY.** Found 2026-09-11 by Lunch Pail 6c-0: `which pytest` on Tom's Mac resolves to
`/opt/homebrew/bin/pytest`, the ambient one WITHOUT `yfinance`, even in a session that correctly
invoked `.venv/bin/pytest`. **So "report `which pytest`" is a defective instruction** — it returns the
wrong path whether or not the Worker did the right thing, and a PM reading it would flag a correct
run as wrong. Ask instead for **the path actually invoked**. The rest of this trap: Worker-observed 2026-09-07
(not independently reproducible from the bridge VM, so recorded as an observation): the ambient
`python3`/`pytest` outside `.venv` has no `yfinance`, and 6 `price_history` tests fail there with
`ModuleNotFoundError`. An environment-selection failure that looks exactly like a code regression.
Pairs with T24. A build report claiming a test count should say which interpreter produced it.

**T35. The overlap pseudo-screen is a UI-only id and must stay that way.** 5c-4 added
`OVERLAP_PSEUDO_SCREEN_ID = "__overlap__"` to the Screen selector via `build_screen_selector_options`
(pure, no Streamlit). It is spliced into the option list and the display-name map **only** — never
into `screens_df`, `screens`, or `screen_membership` — because `screens_df` drives `compute_overlap`,
`build_also_appears_on`, `load_all_screen_identity_data` and `classify_screen`, and a pseudo row
there would put a phantom screen into every ticker's "Also Appears On". `screen_types` and
`has_scoring_by_id` are deliberately left un-extended so a mis-ordered branch fails with a loud
`KeyError` instead of rendering something wrong. Three sub-facts:
- **The overlap branch is a FIFTH early return in `main()` and the only one that does not call
  `apply_pending_nav`.** It pops and discards `_nav_target` instead, immediately after the
  `overlap_df is None` guard and **before** `render_overlap_sidebar` — whose Refresh Data button
  calls `st.rerun()`, which would skip a pop placed after it.
- **`classify_screen` needs NO sixth kind** and `cross_screen_context.py` was not touched. It is
  reached from exactly **two** call sites, `load_all_screen_identity_data` and
  `load_screens_for_ticker`, both iterating registry- or membership-derived ids. `PHASE5C_TRIAGE.md`
  §5 says otherwise and is wrong. The PM's first count said three, because a `classify_screen\(`
  text search matched `_load_screen_df`'s DOCSTRING — the Worker caught it. **Count call sites from
  the call graph, not from a text match.**
- **The lock is mutation-verified.** Appending a pseudo ROW to `screens_df` in place turns three of
  the six new tests red; the PM ran that mutation against the source, not just against the tests.

**T33. Three sizing traps, each of which cost 5c-2 a round.**
- **`st.image` silently clamps to its parent container.** A 44px mark inside a too-narrow column
  rendered at 34.5px with no error and no warning.
- **`st.columns` ratios make a gap PROPORTIONAL to viewport width.** `[1, 14]` measured a 13.9px
  mark-to-title gap at an 809px content block and would have been ~111px at 2200px — the defect
  the Driver had already rejected once, reintroduced by a criterion measured at one width.
  **`st.container(horizontal=True, vertical_alignment=..., gap=<int px>)` exists in 1.63 and is the
  fixed-gap primitive**; verified byte-identical at 969px and 1489px content blocks.
- **An asset's invisible padding decouples the code number from the rendered size.**
  `ows-bear-white-disc.png`'s green ink is only ~69% of its box, so `width=44` drew a **30px**
  bear. **Size against measured ink, never the box.**
  `assets/ows-bear-glyph.png` (416x352) is the cropped, padding-free variant now used beside the
  title at `width=52`, rendering 44px tall.

**T30. The curated `stock_performance` N/A arm is unreachable from live data.** It is null in **0 of
223** curated rows (competition 0/75 · cyclicals 0/31 · management_comp 0/10 · structural 0/107);
`rationale` likewise 0 null. No acceptance run can exercise that branch, which is why 5c-3 locked it
at source level instead. **An `N/A` appearing in a curated block is a flag, not a pass.**

## Phase 6 — the transcripts screen (full text in `CLAUDE.md` Known Issues; do not re-summarize)

**T36. `negative_expert_transcripts` has TWO rows in every run-history table, on one `run_date`, and
that is DELIBERATE — T7's shape again.** `refresh_runs` `20260910T210341Z-0b7469` and
`...T210401Z-a5fbff`; `refresh_screen_runs` two PASSED/216; `refresh_snapshots` **346 rows across
both run_ids on 2026-09-10** — the only screen with more than one run. The second was the
upsert-idempotence proof the acceptance criteria required. **Verified: `latest_snapshot_per_date()`
resolves it correctly (keeps `...a5fbff`, returns 173).** Never read it as an uncounted re-run, and
never delete rows from an append-only table to "clean" it. **Queue item 4 meets this first.**

**T37. A `DocID` backfill on an already-ingested period DUPLICATES transcripts. Detected since 6b,
still not PREVENTED.** 14 of 215 rows have no source `DocID` and sit under a synthesized
`SYN-<hash>`; the same period re-exported with `DocID` populated arrives under real ids, misses the
`(doc_id, ticker)` upsert key, and is inserted IN ADDITION — inflating `mention_count`. The Driver
has agreed to backfill `DocID` going forward, which makes this **more** likely. **6b shipped
`find_duplicate_transcript_groups`** (`src/transcript_ingest.py`): it groups the accumulated
`raw_data` on `(transcript_date, theme, ticker)`, flags any group resolving to more than one
`doc_id`, and logs a WARNING per group from `ingest_transcripts` — **report only, after the insert.**
So the **operational rule still stands: clear this screen's `raw_data` before re-ingesting a stored
period** (`_archive/` keeps every source file, so the corpus is rebuildable). PM-measured 2026-09-10,
and sharper than "the July-Aug pack": **all 14 `SYN-` rows sit in the `"Month Aug-Sep"` source pack,
over 13 distinct tickers (`EA` twice)** — both packs arrive inside the one upload file
`Negative Expert Transcripts (July-Aug 2026).xlsx`, so re-exporting that file duplicates exactly
those 14 rows. On the live corpus the detector flags **0** groups (216 groups over 216 rows).
**Why `ticker` is in the key** — a PM claim that was WRONG until the Worker disproved it by running
it: `EC-1000000-230444` (one `doc_id`, two exploded ticker rows) is the **negative control**, silent
under either key because `nunique` stays 1. `ticker` guards a different case — two DIFFERENT
transcripts, different `doc_id`s, different tickers, coincidentally sharing `(transcript_date,
theme)`. Keyed on `(transcript_date, theme)` alone the live corpus also flags 0, so that key is not a
live false positive today; it is a correctness hazard, not an observed one.

**T38. This screen is the thinnest margin in the project on the composition/misfile check, from its
SECOND upload.** Its `raw_data` accumulates, so `check_composition_misfile` Jaccards each pack
against the whole corpus. Measured on its own two packs: own **0.064** vs `short_screen` **0.053** —
a **+0.011** margin, against competition-vs-structural's +0.039, previously the thinnest of six. A
mild variation flips it to a flag (own 0.003 vs structural 0.019), and the two packs share only
**11 of 173** tickers, so low recurrence is the normal regime. `--force negative_expert_transcripts`
is the hatch. Derivation in `PHASE6_SCOPE.md` §7. **`validate.py` is unscoped — the ruling is the
Driver's.**

**T39. RETIRED by 6b (`edd77ff`) — kept as one line so nobody re-derives it.** The unscored render
path is now generic: `UNSCORED_DISPLAY_COLUMNS_BY_SCREEN` + `resolve_unscored_display_columns`
(unmapped screen -> its OWN columns, ticker first, never RSI's), and
`unscored_export_basename(screen_id)` -> `f"ows_{screen_id}"`, which reproduces RSI's filename
byte-identically. The empty `(0, 0)` "Metrics" frame is gone behind an `if metric_rows:` guard.
**`cross_screen_context._UNSCORED_METRIC_COLUMNS` was the last RSI-shaped piece and 6c retired it**
(`85249d8`) — it is now only the DEFAULT when no resolver is passed; `app.py` always passes
`resolve_unscored_display_columns`, so each unscored screen contributes its own columns. **Still
RSI-shaped and never scoped: `ows_curated_screen.*`, one export filename shared by four curated
screens** — same class, pre-existing.

**T40. A regression can be locked ONLY by the incidental presence of the gitignored database — the
inverse of T25, measured at 6b's review.** `app.py`'s `DB_PATH` resolves to the repo's real
`data/screener.db`. Mutating away 6b's transcripts-panel gate (`if current_screen_id ==
TRANSCRIPTS_SCREEN_ID:` -> `if True:`) left the suite **fully green on a clean checkout** and turned
a 6a test **red only when `data/screener.db` was present** — because the ungated panel then loaded
RSI's real `raw_data` and raised on a missing column. So on a fresh clone or in CI that regression
ships silently. **The rule: any test exercising a render path that loads from `DB_PATH` must stub the
loader, or it is environment-dependent in a way no one will notice.** 6b closed this specific case
with a DB-free gate test; the pattern applies to every future loader added to a render path. The
review technique that found it is the general lesson: **run the mutation matrix on a checkout WITHOUT
`data/`, not only on Tom's machine.**

## Phase 6c — the cross-screen seam (shipped `85249d8`, 2026-09-13)

**T41. Three overlap-view constants stopped being safe to read directly, and none of them fails
loudly if you forget.**
- **`OVERLAP_DISPLAY_COLUMNS` is no longer guaranteed present in the frame.** `mention_count` is
  joined on by `app.py`'s `apply_overlap_metric_joins` from the transcripts aggregate; if that table
  isn't loadable the column is simply absent, and `filtered[OVERLAP_DISPLAY_COLUMNS]` raises
  KeyError and kills the whole page. `render_overlap_page` filters to `present_cols` first. **Any
  new reader of that constant must do the same.**
- **`OVERLAP_COLUMN_HELP` is deliberately INCOMPLETE** — four static identity entries only.
  `screen_count` / `screens_on` / `overall_score` / `mention_count` are derived at render time by
  `build_overlap_help_map(overlap_df, screens_df)`, which must be passed the **UNFILTERED** frame,
  same rule as `screen_count_ceiling` and `zero_thematic_summary`. Reading the constant directly
  gets you four silently missing tooltips, which is why `render_overlap_page`'s `overlap_help_map`
  is a **required** parameter with no default (review round 1 made it required; a `None` default
  would have degraded silently, against T35's loud-failure precedent).
- **`style_overlap_table` knows no column name it did not itself produce.** `market_cap`/`sector`/
  `overall_score` are `compute_overlap`'s own; anything joined on arrives via `extra_formats`, built
  in `app.py` as `OVERLAP_EXTRA_FORMATS` from `OVERLAP_METRIC_JOINS` × `UNSCORED_METRIC_FORMATS`.
  The first build hardcoded `"mention_count"` and `"{:,.0f}"` there with a docstring claiming they
  matched `UNSCORED_METRIC_FORMATS` and nothing enforcing it — caught in review, now enforced by
  construction.

**T42. A fix applied at ONE of two sibling render sites. This is the 6c defect worth generalising.**
6b hit `ValueError` formatting `latest_transcript_date` (TEXT) with a numeric spec and wrapped the
fallback in `try/except` — in `render_unscored_drill_down` only. **`render_cross_screen_context`'s
unscored branch carried the identical five-line block with NO guard**, unreachable until 6c let a
non-numeric column flow into it, at which point it would have crashed every transcripts drill-down
on another screen's page. 6c extracted **`format_unscored_metric_value(col, val)`** and both sites
call it. **A third unscored render site must call that helper, never re-inline the block.** The
general lesson: when you fix a formatting/guard bug, grep for the block you just fixed — a sibling
copy one function away is the normal case in `app.py`, not the exception.

**6c's own figures, PM-derived against the live DB at `85249d8` and unchanged by the build:**
overlap frame **1,400** rows · `mention_count` non-zero on **173**, zero on **1,227**, dtype
**int64**, distribution `{1:146, 2:18, 3:4, 4:3, 5:2}`, max **5** (META, NKE) · the transcripts
contribution renders on **148** distinct tickers across **179** (ticker, host-screen) slots — 148 on
short_screen + 31 thematic (structural 14, competition 12, RSI 3, cyclicals 1, management_comp 1)
over **19** distinct tickers. **173 / 148 / 179 are three different questions** and the 6c handoff
inherited "all 173 tickers' other-screen pages", which is wrong — 25 of the 173 are on no other
screen, so the block never renders for them.

## Phase 8b — the transcripts block in "Also Appears On" (shipped `a2bdca5`, 2026-09-15)

**T43. The gate for a detail-backed render block is the COLUMN SET, never table existence. This is
T40's lesson extended one step.** Three `raw_data__*` tables exist and two of them carry **none** of
`transcript_date` / `doc_id` / `key_takeaways` / `theme` — `raw_data__rising_short_interest` (62
rows) and `raw_data__short_screen` (1,358). So a gate written as "does this screen have a `raw_data`
table?" hands the takeaways sort an RSI frame and raises, which is **T40's recorded failure reached
again by a design that looks entirely reasonable**. `is_transcript_shaped()` in
`src/cross_screen_context.py` is the check, and it was verified against the **live** RSI frame, not
a stub. **Which table to load is still a deliberate `TRANSCRIPTS_SCREEN_ID` literal, and that is not
an oversight** — shape-checking every screen's `raw_data` on every drill-down render would load
short_screen's 1,358-row table to learn what is already known; `app.py`'s own comment says so.
**The rule: gate on the columns the block actually reads, and keep the choice of table a literal.**

## Phase 8c-0 — the docs round (shipped `fa39982`, 2026-09-15)

**T44. The pytest FAILURE COUNT is itself a diagnostic, and three counts mean three different
things: 0 = the correct environment · 6 = `yfinance` missing · 8 = THE WRONG INTERPRETER.** Derived
at 8c-0 and re-measured independently at 8c-1 on the PM's bridge VM at `d79ffd6` (Python 3.10.12,
`python -m pytest`, cwd at the repo root). With `yfinance` installed — it resolves to **1.7.0** — the
suite is **820 passed** and `tests/test_price_history.py` alone is **21 passed / 0 failed**. With
`yfinance` uninstalled and nothing else changed it is exactly **6 failed / 814 passed**, still 820,
and that same file alone is **6 failed / 15 passed**, so all six sit in it. **The 8c-0 Worker
reported EIGHT failures and called them "network/vendor-dependent"; both halves were wrong.** That
file patches `requests.get` at every vendor call site (lines 125/135/145/158/253) and touches no
network at all, and the eight came from bare `pytest` resolving to `/opt/homebrew/bin/pytest`, which
runs CommandLineTools **Python 3.9.6**, below the repo's 3.10 floor (T24). **The 0 and the 6 are
PM-measured both ways at two different HEADs; the 8 is Worker-observed on Tom's Mac and NOT
bridge-reproducible** (the bridge VM carries 3.10 only), and is recorded on the same footing as
T34's second half.

**The mechanism, because a right number with a wrong mechanism is the defect that survives review.**
The six are NOT a collection error and NOT a module-level import failure: `src/price_history.py:240`
imports yfinance **inside** `fetch_price_series`, so the suite still collects cleanly, and what
fails is `patch("yfinance.download", ...)` resolving its dotted target on context-enter —
`ModuleNotFoundError: No module named 'yfinance'` raised from inside `unittest.mock`'s patcher, in
the test body rather than at import. That is also why `tests/test_whiteboard_horizons.py` stays
GREEN while naming yfinance **11 times across 10 lines**: every one of the 11 is the string literal
`"yfinance"` in a `source=` field or an assertion message, never an import and never a patch target.
**So "which tests mention yfinance" does not predict which tests fail without it** — the patch
targets do, and only `tests/test_price_history.py` has any.

**What this obliges a prompt to do.** Ask for `sys.executable` and require `python -m pytest`; never
bare `pytest`, and never `which pytest`, which reports the wrong binary whether or not the Worker
did the right thing (T34). Then read the count as an instrument reading rather than as a verdict:
**a suite run under the wrong interpreter did not observe anything**, so a failure count is evidence
only once the interpreter that produced it is known. **And the diagnostic is the FAILURE count, not
the total** — the total moves every phase (787 at `fa39982`, 820 at `d79ffd6`) while 0 / 6 / 8 is
the signature that does not. Pairs with T24, T34 and T48.

## Phase 8a — the Overvalued screen (shipped `d79ffd6`, 2026-09-16)

**T45. `has_scoring` is DERIVED from the mere PRESENCE of `factor_weights`, while the 24 factors it
implies are a module literal that nothing checks against the screen.** Two independent sites derive
the same flag the same way and neither looks at the screen's columns: `src/db.py:228` writes
`"has_scoring": "factor_weights" in screen_cfg` into the `screens` registry, and `src/refresh.py:468`
recomputes `"factor_weights" in score.get_screen_config(config, screen_id)` to gate the scoring call.
`src/score.py`'s `FACTOR_DEFINITIONS` is a **24-entry module-level literal** and both scoring
functions iterate it unconditionally — `compute_factor_scores` does `df[metric_col]` for all 24,
`compute_overall_score` does `weights[factor_name]` for all 24. **So adding a `factor_weights` block
to a screen that lacks the 24 metric columns KeyErrors at refresh time, and so does supplying a
`factor_weights` block that is short even one key.** PM-reproduced at 8c-1 against the real
`config.yaml`: a two-column frame through `compute_factor_scores` raises `KeyError: 'ps_diff'` (the
first of the 24 it reaches), and a complete frame with one weight deleted raises `KeyError` on that
weight's own name inside `compute_overall_score`. **There is no partial scoring and no graceful
degradation — the flag is a promise about 24 columns that nothing verifies.** This is why the
Overvalued screen ships UNSCORED and its `P/E Diluted / Normal P/E 5Y` is a derived DISPLAY column:
an unscored `quant_composite` screen is the supported shape (`rising_short_interest` is the
precedent), and scoring a new screen means designing a factor model **for that screen**, never
borrowing short_screen's. **Never add `factor_weights` to onboard a screen.**

**T46. A later `Styler.format` call for the same `subset` REPLACES that column's formatter rather
than merging with it — so an `na_rep`-only follow-up silently drops the number format while the
null cell still looks right.** PM-measured at 8a and re-measured at 8c-1 on pandas 2.3.3:
`df.style.format({"pe_vs_normal_5y": "{:.2f}x"})` followed by
`.format(na_rep="nanx", subset=["pe_vs_normal_5y"])` renders the live value **`1.310000`** and the
null **`nanx`**, while the single call
`df.style.format("{:.2f}x", na_rep="nanx", subset=["pe_vs_normal_5y"])` renders **`1.31x`** and
**`nanx`**. **The null cell is correct in BOTH, and that is the whole trap** — the follow-up call
looks like it worked, and the cell it broke is the one nobody was checking. Both of `app.py`'s
stylers now pass the spec and the `na_rep` in ONE scoped call per column. **The rule:
`Styler.format` is last-write-wins per subset, so every option a column needs travels in the same
call; a second `.format` for a column that already has one is an overwrite, not an extension.** The
check that catches it is rendering the Styler to HTML and reading the **live** cell, never the null
one — a verification that only looks at what the second call was added for cannot see what it
destroyed.

**T47. A lock on a constant is not a lock on its use.** 8a's first build added
`TestOverlapMetricFillValues`, which asserts that `OVERLAP_METRIC_FILL_VALUES["pe_vs_normal_5y"]` is
NaN and that `mention_count` has no entry. Nothing in it proved that `apply_overlap_metric_joins` —
the real call site, `src/app.py:3308` — ever READS the constant, so deleting the single argument
`fill_value=OVERLAP_METRIC_FILL_VALUES.get(column, 0)` from that call reinstates the whole 0.00x
defect, `overlap.join_metric_column`'s own default being `0`. **PM-reproduced at 8c-1 as a
three-run pair on a `git archive HEAD` extraction outside the repo** (no `data/`, so T40 cannot
flatter the result): baseline **820 passed**; with the argument deleted, **1 failed / 819 passed**
and the single red is the use-site test 8a's review round 2 required; with that one test deselected
and the mutation still in place, **819 passed, 1 deselected** — the suite green with the defect
live, which is 8a's first build exactly. Under that same mutation `TestOverlapMetricFillValues`
itself reports **4 passed**. **So a constant-lock's green is evidence about the constant and about
nothing else.** The test that closes it goes through the real function with BOTH joined columns in
one call, so a "fix" that flipped the default to NaN globally — silently breaking `mention_count`'s
measured zero — fails the same test. **The review technique: mutate the USE, not the definition. If
deleting the argument, the lookup or the call leaves the suite green, the constant is decoration.**
Pairs with T41, this file's other lesson about reading these same overlap constants directly.

## Phase 8c-1 — the docs round's own defect (2026-09-16)

**T48. A version-stamped measurement is not corrected by re-measuring on a different version — and
this round was issued carrying exactly that error into three tracked files.** 8c-1's prompt
"corrected" `ALL_MATERIAL_ICONS` from 4,271 to **4,277**, on the strength of an inherited 8a note
reading "4,277 on streamlit 1.64.0, not the 4,271 five handoffs carried." Both PMs had measured on
the **bridge VM**, where a fresh unpinned `pip install streamlit` resolves to the latest release.
Tom's `.venv` — the environment that actually runs the app, and the one that produced this very
round's 820-pass baseline — is **streamlit 1.63.0, where the count is 4,271**. Re-confirmed at 8c-1
from the installed tree: `.venv/lib/python3.11/site-packages/streamlit-1.63.0.dist-info` and an AST
count of that install's `ALL_MATERIAL_ICONS` literal returning **4,271**, against the bridge's
**1.64.0 / 4,277**. **So the inherited figure was right, the "correction" was environment drift, and
the five handoffs that carried 4,271 forward had carried it correctly.**

**The Worker flagged it rather than placing it, and that is the rule working.** The prompt said
flag, do not fix — an acceptance figure that moves during a build stops being an acceptance figure —
and the flag arrived in the PLAN, before a byte was written into a tracked file.

**The general rule: before calling an inherited number wrong, establish that you measured it in the
same environment it was measured in.** A figure that differs across environments is not a defect to
fix, it is a fact to STAMP — which is why `docs/KNOWN_ISSUES.md`'s own "(4,271 names at 1.63.0)" had
been right all along while T29's bare "the installed" had not, and why `requirements.txt`'s
`streamlit>=1.28` FLOOR is the mechanism that makes the two diverge (the same exposure 7a recorded
for `use_container_width`: Tom's `.venv` keeps working while a fresh install moves). **This does NOT
soften the copied-not-corroborated rule — it sharpens it.** Deriving a number where it is about to
be used means deriving it in the environment it is about to be used in; a derivation in the wrong
environment is a copy with extra steps. Pairs with T44, which is the same lesson one layer down: an
instrument reading is evidence only once the instrument is identified.

