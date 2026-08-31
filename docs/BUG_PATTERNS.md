# Recurring Bug Patterns

Moved out of `CLAUDE.md` to keep that file loading into every session, but
kept verbatim — pattern 5 alone was cited in the Phase 3c plan, so this is
proven useful, not dead weight. Worth re-reading before touching
`transform.py` or `score.py`.

1. **Ranking direction inversion.** Several factors use `1 - percentile` because lower raw values are worse for the short thesis. Applying straight percentile rank to these factors will score high-FCF-yield companies as short candidates. Always verify direction against the factor table in `score.py`.

2. **Decimal vs. display unit confusion.** All percentage fields (margins, yields, growth rates, short interest) are stored as decimals. Treat them as decimals in all calculations. If a result looks 100x too large or small, check for an accidental multiply/divide by 100.

3. **Bloomberg `"#N/A N/A"` strings surviving into calculations.** If `ingest.py` fails to catch one of these strings and it reaches `transform.py`, `pd.to_numeric` will coerce it to `NaN` — but only if `errors='coerce'` is used. Always use `errors='coerce'` on any column that could contain this string.

4. **NaN propagation in chained calculations.** If an intermediate metric (e.g. `dsos_t3m`) is `NaN`, any downstream metric that depends on it (e.g. `dso_pct_change`) will also be `NaN`. This is correct behavior — do not mask it with fillna before the final fallback step in `score.py`.

5. **Division by zero in ratio calculations.** Revenue, EBITDA, COGS, and share counts can all be zero. Always check denominators explicitly with `np.where` or `pd.Series.where` before dividing, and return `NaN` rather than raising.

6. **PERCENTRANK.INC range behavior.** Excel's `PERCENTRANK.INC` includes both 0 and 1 as possible outputs. `scipy.stats.percentileofscore(..., kind='rank') / 100` matches this. `kind='weak'` and `kind='strict'` do not — never substitute them.
