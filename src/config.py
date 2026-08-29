"""
Load the multi-screen configuration from config.yaml.

Config loading is infrastructure every pipeline stage needs (ingest,
transform, and score all sync the screens registry from it), not a concern
specific to any one stage — so it lives here rather than in score.py, which
should only be a scoring-logic dependency, not a config-loading one.
"""

import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load the full multi-screen configuration from config.yaml.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dictionary with a top-level "screens" key mapping
        screen_id -> that screen's display_name/type/universe/
        factor_weights/scoring block.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ScreenTypeError(ValueError):
    """Raised when an operation is invoked against a screen of the wrong type.

    E.g. running the quant transform/score pipeline against a curated
    screen, or the curated loader against a quant_composite screen. Kept
    as its own exception (rather than a bare ValueError) so callers and
    tests can target this specific failure mode precisely.
    """


def get_screen_type(config: dict, screen_id: str) -> str:
    """Look up one screen's type ("quant_composite" or "curated").

    Lives here, not in score.py's get_screen_config, because transform.py
    and ingest.py must not import from score.py (score.py is a scoring-
    logic dependency for others, not a config-loading one — see the module
    docstring above). Every module that needs to dispatch on screen type
    uses this one function, so they all produce the same error shape.

    Args:
        config: Full parsed config.yaml dict, as returned by load_config().
        screen_id: The screen to look up.

    Returns:
        That screen's "type" value.

    Raises:
        KeyError: If screen_id is not defined under config["screens"], with
            the list of known screen ids for context.
    """
    try:
        return config["screens"][screen_id]["type"]
    except KeyError:
        raise KeyError(
            f"screen_id {screen_id!r} not found in config.yaml screens block. "
            f"Known screens: {list(config.get('screens', {}).keys())}"
        ) from None
