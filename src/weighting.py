"""
Dynamic factor weighting for the OWS Short Screen composite (Phase 5d).

Pure — no Streamlit, no SQLAlchemy, no src.score import (D5 of the Phase 5d
Worker prompt). This module builds the effective flat 24-key weight map from
a two-level (category, within-category) weight structure, and manages named
weight presets on disk. src/app.py composes this with
src.score.compute_overall_score/FACTOR_DEFINITIONS; keeping that composition
in app.py (not here) means this module never needs to know what a "factor"
actually is beyond a string id, and can be unit-tested with a synthetic
taxonomy that has nothing to do with the real 24 factors.

Display-only: nothing here writes to data/screener.db. Presets persist in
their own gitignored YAML file, never in config.yaml (which stays the only
thing score.py's write path reads) and never in the database (which has a
restore point that would silently delete an analyst's presets).
"""

import os
import tempfile
from typing import Iterable

import yaml

DEFAULT_PRESETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "weight_presets.yaml"
)


def load_factor_categories(screen_config: dict) -> dict[str, list[str]]:
    """Extract a screen's category -> [factor_ids] taxonomy from its config
    sub-dict, in the config's authoritative `order`.

    Args:
        screen_config: A screen's config.yaml sub-dict (from
            src.score.get_screen_config), containing a "factor_categories"
            key with an "order" list plus one list per named category.

    Returns:
        {category: [factor_ids]}, in `order`.

    Raises:
        KeyError: If "factor_categories" or "order" is missing.
        ValueError: If `order` and the block's other keys don't name exactly
            the same set of categories (one listed but not defined, or
            defined but missing from `order`).
    """
    block = screen_config["factor_categories"]
    order = block["order"]
    defined = set(block.keys()) - {"order"}
    ordered = set(order)
    if defined != ordered:
        only_in_order = ordered - defined
        only_defined = defined - ordered
        raise ValueError(
            "factor_categories' `order` and its category keys disagree: "
            f"in order but not defined: {sorted(only_in_order)}; "
            f"defined but missing from order: {sorted(only_defined)}"
        )
    return {category: list(block[category]) for category in order}


def validate_taxonomy(categories: dict[str, list[str]], known_factors: Iterable[str]) -> None:
    """Validate that `categories` partitions `known_factors` exactly:
    every known factor appears in exactly one category, and no category
    names a factor outside known_factors.

    Args:
        categories: {category: [factor_ids]}, as returned by
            load_factor_categories.
        known_factors: The complete set of valid factor ids (callers pass
            src.score.FACTOR_DEFINITIONS.keys() — this module never imports
            src.score itself, to keep it decoupled from the scoring engine).

    Raises:
        ValueError: Naming the specific factors that are missing, duplicated,
            or unknown.
    """
    known = set(known_factors)
    seen: dict[str, str] = {}
    duplicates = []
    unknown = []
    for category, factors in categories.items():
        for factor in factors:
            if factor not in known:
                unknown.append((category, factor))
            if factor in seen:
                duplicates.append((factor, seen[factor], category))
            else:
                seen[factor] = category

    missing = sorted(known - set(seen.keys()))

    if missing or duplicates or unknown:
        problems = []
        if missing:
            problems.append(f"missing from any category: {missing}")
        if duplicates:
            problems.append(
                "listed in more than one category: "
                + ", ".join(f"{f!r} (in {a!r} and {b!r})" for f, a, b in duplicates)
            )
        if unknown:
            problems.append(
                "not a known factor: "
                + ", ".join(f"{f!r} (in category {c!r})" for c, f in unknown)
            )
        raise ValueError("Factor category taxonomy is invalid: " + "; ".join(problems))


def compute_effective_weights(
    category_weights: dict[str, float],
    factor_weights: dict[str, float],
    categories: dict[str, list[str]],
) -> dict[str, float]:
    """The effective flat 24-key weight map: category_weight x
    within-category (per-factor) weight.

    Args:
        category_weights: {category: weight}. A category absent from this
            dict defaults to 1.0 (the config-weights default — every
            category's weight is 1.0 unless the analyst has moved it).
        factor_weights: {factor: weight} — the within-category weight for
            each factor. Defaults to config.yaml's factor_weights; the
            analyst can override individual factors too.
        categories: {category: [factor_ids]}, as returned by
            load_factor_categories.

    Returns:
        {factor: category_weight * factor_weights[factor]}, one entry per
        factor named in `categories`.
    """
    effective = {}
    for category, factors in categories.items():
        cat_weight = category_weights.get(category, 1.0)
        for factor in factors:
            effective[factor] = cat_weight * factor_weights[factor]
    return effective


def _read_presets_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or "presets" not in data:
        raise ValueError(f"{path} is malformed: expected a top-level 'presets' key")
    return data


def load_presets(
    path: str = DEFAULT_PRESETS_PATH,
    known_categories: Iterable[str] | None = None,
    known_factors: Iterable[str] | None = None,
) -> dict[str, dict]:
    """Load every saved preset from `path`.

    Args:
        path: Path to the preset YAML file.
        known_categories: If given, every preset's category_weights must
            name exactly this set of categories — no more, no less. Passed
            as plain strings (this module never imports src.score), so a
            caller not interested in this check can omit it entirely.
        known_factors: Same check for factor_weights.

    Returns:
        {preset_name: {"category_weights": {...}, "factor_weights": {...}}}.
        {} if the file doesn't exist yet — that's the normal first-run state,
        not an error.

    Raises:
        ValueError: If the file exists but isn't valid YAML, isn't shaped
            like a presets file (missing "category_weights"/"factor_weights"
            on some entry), or (when known_categories/known_factors are
            given) a preset names a category/factor outside that set, or is
            missing one from it — named with the file path and the specific
            problem, never silently defaulted.
    """
    data = _read_presets_file(path)
    presets = data.get("presets") or {}
    for name, preset in presets.items():
        if "category_weights" not in preset or "factor_weights" not in preset:
            raise ValueError(
                f"{path}: preset {name!r} is missing 'category_weights' or "
                f"'factor_weights'"
            )
        if known_categories is not None:
            _check_key_set(path, name, "category_weights", preset["category_weights"], known_categories)
        if known_factors is not None:
            _check_key_set(path, name, "factor_weights", preset["factor_weights"], known_factors)
    return presets


def _check_key_set(path: str, preset_name: str, field: str, actual: dict, known: Iterable[str]) -> None:
    known_set = set(known)
    actual_set = set(actual)
    unknown = actual_set - known_set
    missing = known_set - actual_set
    if unknown or missing:
        problems = []
        if unknown:
            problems.append(f"unknown: {sorted(unknown)}")
        if missing:
            problems.append(f"missing: {sorted(missing)}")
        raise ValueError(
            f"{path}: preset {preset_name!r}'s {field!r} is invalid ({'; '.join(problems)})"
        )


def save_preset(
    name: str,
    category_weights: dict[str, float],
    factor_weights: dict[str, float],
    path: str = DEFAULT_PRESETS_PATH,
) -> None:
    """Save (or overwrite) one named preset, storing the full weight set.

    Read-modify-write, atomic: written to a temp file in the same directory
    then moved into place with os.replace, so a crash mid-write can't leave
    a corrupt/partial presets file behind.

    Args:
        name: Preset name (as typed into the "Save as" field).
        category_weights: The full {category: weight} map at save time.
        factor_weights: The full {factor: weight} map at save time.
        path: Path to the preset YAML file.
    """
    data = _read_presets_file(path) if os.path.exists(path) else {"presets": {}}
    data.setdefault("presets", {})
    data["presets"][name] = {
        "category_weights": dict(category_weights),
        "factor_weights": dict(factor_weights),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def delete_preset(name: str, path: str = DEFAULT_PRESETS_PATH) -> None:
    """Delete one named preset. A no-op if it doesn't exist or the file
    doesn't exist yet — deleting an already-gone preset is not an error.

    Args:
        name: Preset name to remove.
        path: Path to the preset YAML file.
    """
    if not os.path.exists(path):
        return
    data = _read_presets_file(path)
    presets = data.get("presets") or {}
    if name not in presets:
        return
    del presets[name]
    data["presets"] = presets
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
