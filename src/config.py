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
