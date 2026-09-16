"""
Unit tests for src/weighting.py (Phase 5d).

Covers the properties specified in the Phase 5d Worker prompt (§7):
1. Taxonomy coverage
2. Defaults round-trip
3. Uniform-scale invariance (weights level + score level)
4. (Slider bound derivation is covered in tests/test_app.py — source-anchored.)
5. Purity (no streamlit/sqlalchemy/src.score import)
6. Preset round-trip and loud failure
7. (Overlap-untouched is covered in tests/test_overlap.py + the live acceptance run.)

No test here reads data/screener.db, data/uploads/, or data/historical/ — only
config.yaml (not gitignored) and tmp_path fixtures.
"""

import ast
import os

import pandas as pd
import pytest

from src.config import load_config
from src.score import FACTOR_DEFINITIONS, compute_overall_score, get_screen_config
from src.weighting import (
    category_sum_column_name,
    category_sum_columns,
    compute_category_sums,
    compute_effective_weights,
    delete_preset,
    load_factor_categories,
    load_presets,
    save_preset,
    validate_taxonomy,
)

SCREEN_ID = "short_screen"


@pytest.fixture
def screen_config():
    """The actual short_screen config.yaml sub-dict."""
    return get_screen_config(load_config(), SCREEN_ID)


@pytest.fixture
def real_categories(screen_config):
    return load_factor_categories(screen_config)


# ---------------------------------------------------------------------------
# load_factor_categories
# ---------------------------------------------------------------------------


def test_load_factor_categories_preserves_order(real_categories, screen_config):
    expected_order = screen_config["factor_categories"]["order"]
    assert list(real_categories.keys()) == expected_order


def test_load_factor_categories_order_mismatch_raises():
    block = {
        "factor_categories": {
            "order": ["A", "B"],
            "A": ["f1"],
            # "B" is missing entirely, and "C" is defined but not in order.
            "C": ["f2"],
        }
    }
    with pytest.raises(ValueError):
        load_factor_categories(block)


# ---------------------------------------------------------------------------
# Property 1: taxonomy coverage
# ---------------------------------------------------------------------------


def test_real_taxonomy_covers_every_known_factor_exactly_once(real_categories):
    """The actual config.yaml taxonomy, validated against the actual
    FACTOR_DEFINITIONS — the real regression lock."""
    validate_taxonomy(real_categories, FACTOR_DEFINITIONS.keys())


def test_validate_taxonomy_fires_on_a_dropped_factor():
    known = ["f1", "f2", "f3"]
    categories = {"A": ["f1", "f2"]}  # f3 dropped
    with pytest.raises(ValueError, match="missing"):
        validate_taxonomy(categories, known)


def test_validate_taxonomy_fires_on_a_duplicated_factor():
    known = ["f1", "f2"]
    categories = {"A": ["f1"], "B": ["f1", "f2"]}  # f1 listed twice
    with pytest.raises(ValueError, match="more than one category"):
        validate_taxonomy(categories, known)


def test_validate_taxonomy_fires_on_a_misspelled_factor():
    known = ["f1", "f2"]
    categories = {"A": ["f1"], "B": ["f2_typo"]}  # f2 misspelled, f2 itself missing
    with pytest.raises(ValueError):
        validate_taxonomy(categories, known)


# ---------------------------------------------------------------------------
# Property 2: defaults round-trip
# ---------------------------------------------------------------------------


def test_defaults_round_trip_reproduces_flat_weights_exactly(real_categories, screen_config):
    factor_weights = screen_config["factor_weights"]
    category_weights = {c: 1.0 for c in real_categories}
    effective = compute_effective_weights(category_weights, factor_weights, real_categories)
    assert effective == pytest.approx(factor_weights, abs=1e-12)


# ---------------------------------------------------------------------------
# Property 3: uniform-scale invariance
# ---------------------------------------------------------------------------


def test_uniform_category_scaling_scales_every_effective_weight(real_categories, screen_config):
    factor_weights = screen_config["factor_weights"]
    k = 3.7
    baseline = compute_effective_weights(
        {c: 1.0 for c in real_categories}, factor_weights, real_categories
    )
    scaled = compute_effective_weights(
        {c: k for c in real_categories}, factor_weights, real_categories
    )
    for factor in factor_weights:
        assert scaled[factor] == pytest.approx(k * baseline[factor], abs=1e-9)


def test_uniform_category_scaling_scales_every_score_and_preserves_rank(real_categories, screen_config):
    """Score-level proof (not just the weights level): would fail against a
    build that renormalises weights (e.g. divides by their sum) inside the
    panel before calling compute_overall_score, since that would break the
    scaling identity even though compute_effective_weights itself is fine."""
    factor_weights = screen_config["factor_weights"]
    rows = 8
    df = pd.DataFrame({name: [(i + 1) / (rows + 1) for i in range(rows)] for name in FACTOR_DEFINITIONS})
    df.insert(0, "ticker", [f"T{i}" for i in range(rows)])

    k = 2.5
    baseline_weights = compute_effective_weights(
        {c: 1.0 for c in real_categories}, factor_weights, real_categories
    )
    scaled_weights = compute_effective_weights(
        {c: k for c in real_categories}, factor_weights, real_categories
    )

    baseline_scores = compute_overall_score(df, {"factor_weights": baseline_weights})
    scaled_scores = compute_overall_score(df, {"factor_weights": scaled_weights})

    assert scaled_scores.to_numpy() == pytest.approx(k * baseline_scores.to_numpy(), abs=1e-9)
    assert list(baseline_scores.rank()) == list(scaled_scores.rank())


# ---------------------------------------------------------------------------
# Property 8c-2: category-sum columns
# ---------------------------------------------------------------------------


def test_category_sum_column_name_slugs_multiword_and_hyphenated_categories():
    assert category_sum_column_name("Valuation") == "valuation_sum"
    assert category_sum_column_name("Balance Sheet") == "balance_sheet_sum"
    assert category_sum_column_name("Cash Flow") == "cash_flow_sum"
    assert category_sum_column_name("Non-GAAP") == "non_gaap_sum"


def test_category_sum_columns_is_generic_not_a_literal_list():
    """FAILS IF category_sum_columns hand-types a 7-item list instead of
    walking whatever taxonomy it's handed."""
    categories = {"A": ["f1", "f2"], "B": ["f3"]}
    assert category_sum_columns(categories) == ["a_sum", "b_sum"]
    categories["C"] = ["f4"]
    assert category_sum_columns(categories) == ["a_sum", "b_sum", "c_sum"]


def test_compute_category_sums_reconciles_to_overall_score_at_config_weights(
    real_categories, screen_config
):
    """The headline property: the category sums, added together, equal
    Overall Score. FAILS IF a build sums raw unweighted factor values."""
    factor_weights = screen_config["factor_weights"]
    rows = 8
    df = pd.DataFrame(
        {name: [(i + 1) / (rows + 1) for i in range(rows)] for name in FACTOR_DEFINITIONS}
    )
    df.insert(0, "ticker", [f"T{i}" for i in range(rows)])

    effective = compute_effective_weights(
        {c: 1.0 for c in real_categories}, factor_weights, real_categories
    )
    sums = compute_category_sums(df, effective, real_categories)
    sigma7 = sums.sum(axis=1)
    overall = compute_overall_score(df, {"factor_weights": effective})
    assert sigma7.to_numpy() == pytest.approx(overall.to_numpy(), abs=1e-12)


def test_compute_category_sums_reconciles_under_non_default_weight_state(
    real_categories, screen_config
):
    """Same reconciliation, but under a MOVED category-weight state (Cash
    Flow 2.0, Sentiment 0.0, others default 1.0) — mirrors the PM's own
    probe. FAILS IF the sums are computed at config weights while
    overall_score uses a different (live/reweighted) set — the case a
    config-weights-only build passes and this one does not."""
    factor_weights = screen_config["factor_weights"]
    rows = 8
    df = pd.DataFrame(
        {name: [(i + 1) / (rows + 1) for i in range(rows)] for name in FACTOR_DEFINITIONS}
    )
    df.insert(0, "ticker", [f"T{i}" for i in range(rows)])

    category_weights = {c: 1.0 for c in real_categories}
    category_weights["Cash Flow"] = 2.0
    category_weights["Sentiment"] = 0.0
    effective = compute_effective_weights(category_weights, factor_weights, real_categories)
    sums = compute_category_sums(df, effective, real_categories)
    sigma7 = sums.sum(axis=1)
    overall = compute_overall_score(df, {"factor_weights": effective})
    assert sigma7.to_numpy() == pytest.approx(overall.to_numpy(), abs=1e-12)


def test_compute_category_sums_is_generic_over_taxonomy():
    """FAILS IF the sum computation hardcodes the real 24-factor/7-category
    taxonomy instead of walking whatever `categories` it's handed."""
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "f3": [5.0, 6.0], "f4": [7.0, 8.0]})
    categories = {"A": ["f1", "f2"], "B": ["f3"]}
    weights = {"f1": 1.0, "f2": 2.0, "f3": 0.5, "f4": 10.0}

    sums = compute_category_sums(df, weights, categories)
    assert list(sums.columns) == ["a_sum", "b_sum"]
    assert sums["a_sum"].tolist() == pytest.approx([1 * 1.0 + 3 * 2.0, 2 * 1.0 + 4 * 2.0])
    assert sums["b_sum"].tolist() == pytest.approx([5 * 0.5, 6 * 0.5])

    # Adding a third category (with a factor not referenced by the first
    # two) must appear with no other code change.
    categories["C"] = ["f4"]
    sums2 = compute_category_sums(df, weights, categories)
    assert list(sums2.columns) == ["a_sum", "b_sum", "c_sum"]
    assert sums2["c_sum"].tolist() == pytest.approx([7 * 10.0, 8 * 10.0])


def test_compute_category_sums_propagates_nan():
    """A missing factor value must show up as a missing sum, not silently
    as a zero contribution. FAILS IF a future change fills NaN with 0
    before summing."""
    df = pd.DataFrame({"f1": [1.0, float("nan")], "f2": [3.0, 4.0]})
    categories = {"A": ["f1", "f2"]}
    weights = {"f1": 1.0, "f2": 1.0}
    sums = compute_category_sums(df, weights, categories)
    assert sums["a_sum"].iloc[0] == pytest.approx(4.0)
    assert pd.isna(sums["a_sum"].iloc[1])


# ---------------------------------------------------------------------------
# Property 5: purity — parsed import graph, not a source-text substring scan
# (a substring scan would false-positive on this module's own docstrings,
# which mention streamlit/sqlalchemy/src.score by name to explain why they
# are absent).
# ---------------------------------------------------------------------------


def test_weighting_module_imports_neither_streamlit_sqlalchemy_nor_src_score():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "weighting.py")
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)

    forbidden = {"streamlit", "sqlalchemy", "src.score"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module)

    hits = {name for name in imported if any(name == f or name.startswith(f + ".") for f in forbidden)}
    assert not hits, f"src/weighting.py imports forbidden module(s): {hits}"


# ---------------------------------------------------------------------------
# Property 6: preset round-trip and loud failure
# ---------------------------------------------------------------------------


def test_preset_round_trip(tmp_path):
    path = tmp_path / "weight_presets.yaml"
    category_weights = {"Valuation": 2.0, "Growth": 1.0}
    factor_weights = {"abs_ps_factor": 0.3, "decel_factor": 0.7}

    save_preset("My Preset", category_weights, factor_weights, path=str(path))
    loaded = load_presets(path=str(path))

    assert loaded["My Preset"]["category_weights"] == category_weights
    assert loaded["My Preset"]["factor_weights"] == factor_weights


def test_preset_delete_round_trip(tmp_path):
    path = tmp_path / "weight_presets.yaml"
    save_preset("Temp", {"Valuation": 1.0}, {"abs_ps_factor": 0.5}, path=str(path))
    assert "Temp" in load_presets(path=str(path))

    delete_preset("Temp", path=str(path))
    assert "Temp" not in load_presets(path=str(path))


def test_load_presets_missing_file_returns_empty_dict(tmp_path):
    path = tmp_path / "does_not_exist.yaml"
    assert load_presets(path=str(path)) == {}


def test_load_presets_malformed_yaml_raises_with_file_named(tmp_path):
    path = tmp_path / "weight_presets.yaml"
    path.write_text("not: valid: yaml: [structure")
    with pytest.raises(ValueError, match=str(path)):
        load_presets(path=str(path))


def test_load_presets_unknown_factor_raises_loudly(tmp_path):
    path = tmp_path / "weight_presets.yaml"
    save_preset(
        "Bad", {"Valuation": 1.0}, {"abs_ps_factor": 0.5, "nonexistent_factor": 0.5},
        path=str(path),
    )
    with pytest.raises(ValueError, match="nonexistent_factor"):
        load_presets(
            path=str(path),
            known_categories=["Valuation"],
            known_factors=["abs_ps_factor"],
        )


def test_load_presets_missing_required_factor_raises_loudly(tmp_path):
    path = tmp_path / "weight_presets.yaml"
    save_preset("Partial", {"Valuation": 1.0}, {"abs_ps_factor": 0.5}, path=str(path))
    with pytest.raises(ValueError, match="missing"):
        load_presets(
            path=str(path),
            known_categories=["Valuation"],
            known_factors=["abs_ps_factor", "decel_factor"],
        )
