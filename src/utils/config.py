"""
Small helper to load config.yaml into a plain dict.

Kept separate so every module imports config the same way instead of
each one calling yaml.safe_load() with its own path logic.
"""
import os
import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config.yaml",
)


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
