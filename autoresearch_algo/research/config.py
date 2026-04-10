"""
Configuration loader for ONVOX AutoResearch pipeline.
=========================================
Single source of truth — reads config.yaml and resolves paths.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def find_project_root() -> Path:
    """Find the project root by locating config.yaml, walking up from this file."""
    current = Path(__file__).resolve().parent  # research/
    for _ in range(5):
        if (current / "config.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not find config.yaml. "
        "Set AUTORESEARCH_BASE_DIR environment variable or run from the project root."
    )


def load_config(config_path: Optional[str] = None, reload: bool = False) -> Dict[str, Any]:
    """
    Load and return the project configuration.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to config.yaml. If None, auto-detects from project root.
    reload : bool
        Force reload even if already cached.

    Returns
    -------
    dict
        The full configuration dictionary with resolved paths.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not reload:
        return _CONFIG_CACHE

    if config_path is None:
        root = find_project_root()
        config_path = str(root / "config.yaml")
    else:
        root = Path(config_path).resolve().parent

    logger.info("Loading config from: %s", config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve base_dir: env var > config value > config file location
    env_base = os.environ.get("AUTORESEARCH_BASE_DIR")
    if env_base:
        cfg["base_dir"] = str(Path(env_base).resolve())
    elif cfg.get("base_dir") is None:
        cfg["base_dir"] = str(root)
    else:
        cfg["base_dir"] = str(Path(cfg["base_dir"]).resolve())

    logger.info("Base directory: %s", cfg["base_dir"])

    _CONFIG_CACHE = cfg
    return cfg


def get_base_dir(cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Return the resolved base directory as a Path."""
    if cfg is None:
        cfg = load_config()
    return Path(cfg["base_dir"])


def get_participant_config(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Dict]:
    """Return only participants that have glucose CSV files configured."""
    if cfg is None:
        cfg = load_config()
    participants = {}
    for name, pcfg in cfg.get("participants", {}).items():
        if pcfg.get("glucose_csv"):
            participants[name] = pcfg
    return participants
