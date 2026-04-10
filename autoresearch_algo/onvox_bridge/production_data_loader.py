"""
Production Data Loader — Load synced Supabase NPZ files for autoresearch evaluation.
=====================================================================================

Reads per-user NPZ files from data/synced/features/ and returns data in the same
shape as load_all_audio() but with pre-extracted features instead of audio paths.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SYNC_DIR = PROJECT_ROOT / "data" / "synced"
FEATURES_DIR = SYNC_DIR / "features"
MANIFEST_PATH = SYNC_DIR / "sync_manifest.json"


def load_production_data(
    sync_dir: Optional[Path] = None,
    min_samples: int = 5,
) -> Dict[str, Dict]:
    """Load all synced user data from NPZ files.

    Returns dict compatible with autoresearch evaluation functions:
        {user_id: {"features": np.ndarray, "glucose": np.ndarray, "timestamps": list}}
    """
    features_dir = (sync_dir or FEATURES_DIR)
    if not features_dir.exists():
        logger.warning("Sync directory not found: %s. Run supabase_syncer first.", features_dir)
        return {}

    npz_files = sorted(features_dir.glob("*_features.npz"))
    if not npz_files:
        logger.warning("No NPZ files found in %s", features_dir)
        return {}

    data = {}
    for npz_path in npz_files:
        user_id = npz_path.stem.replace("_features", "")
        try:
            npz = np.load(npz_path, allow_pickle=True)
            X = npz["features"]
            y = npz["glucose"]
            ts = npz["timestamps"]

            if len(y) < min_samples:
                logger.info("  %s: %d samples (< %d min), skipping", user_id[:8], len(y), min_samples)
                continue

            data[user_id] = {
                "features": X.astype(np.float64),
                "glucose": y.astype(np.float64),
                "timestamps": list(ts),
            }
            logger.info("  %s: %d samples, %d-dim features", user_id[:8], len(y), X.shape[1])

        except Exception as e:
            logger.warning("  %s: Failed to load: %s", user_id[:8], e)
            continue

    logger.info("Loaded %d production users", len(data))
    return data


def load_manifest() -> Optional[dict]:
    """Load the sync manifest for metadata (sync time, checksums, etc.)."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
