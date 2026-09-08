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
