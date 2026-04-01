"""
Feature Caching
================
Cache extracted features to disk using joblib for fast re-runs.
HuBERT extraction takes minutes per file — caching avoids re-extraction.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class FeatureCache:
    """
    Disk-based feature cache keyed by (audio_file_hash, extractor_config_hash).

    Parameters
    ----------
    cache_dir : str or Path
        Directory to store cached features.
    enabled : bool
        If False, all operations are no-ops (useful for debugging).
    """

    def __init__(self, cache_dir: str = ".cache/features", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled and HAS_JOBLIB

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Feature cache: %s", self.cache_dir)
        elif enabled and not HAS_JOBLIB:
            logger.warning("joblib not installed — feature caching disabled. pip install joblib")

    def _make_key(self, audio_path: str, extractor_id: str) -> str:
        """Generate a cache key from file path + extractor config."""
        raw = f"{audio_path}|{extractor_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, audio_path: str, extractor_id: str) -> Optional[np.ndarray]:
        """
        Retrieve cached features, or None if not cached.

        Parameters
        ----------
        audio_path : str
            Full path to the audio file.
        extractor_id : str
            String identifying the extractor configuration (e.g., "mfcc_n20_sr16000").
        """
        if not self.enabled:
            return None

        key = self._make_key(audio_path, extractor_id)
        cache_file = self.cache_dir / f"{key}.npz"

        if cache_file.exists():
            try:
                data = np.load(cache_file)
                return data["features"]
            except Exception:
                cache_file.unlink(missing_ok=True)

        return None

    def put(self, audio_path: str, extractor_id: str, features: np.ndarray):
        """
        Store features in cache.

        Parameters
        ----------
        audio_path : str
            Full path to the audio file.
        extractor_id : str
            Extractor configuration identifier.
        features : np.ndarray
            Feature vector to cache.
        """
        if not self.enabled:
            return

        key = self._make_key(audio_path, extractor_id)
        cache_file = self.cache_dir / f"{key}.npz"

        try:
            np.savez_compressed(cache_file, features=features)
        except Exception as e:
            logger.warning("Failed to cache features for %s: %s", audio_path, e)

    def clear(self):
        """Remove all cached features."""
        if not self.enabled:
            return

        count = 0
        for f in self.cache_dir.glob("*.npz"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cached feature files", count)

    @property
    def size(self) -> int:
        """Number of cached feature files."""
        if not self.enabled:
            return 0
        return len(list(self.cache_dir.glob("*.npz")))
