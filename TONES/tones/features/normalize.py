"""
Within-Speaker Feature Normalization
=======================================
The t-SNE visualization proves that current features encode speaker identity
(~95% of variance) rather than glucose-related physiological state. Within-speaker
normalization removes this dominant noise source, forcing models to see ONLY
within-speaker variation — exactly where the glucose signal lives.

Strategies:
  1. Static z-normalization: (x - mu_speaker) / sigma_speaker
  2. Running z-normalization: z-score relative to sliding window of past K samples
  3. Rank normalization: percentile rank within speaker (robust to outliers)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def zscore_per_speaker(
    features_by_participant: Dict[str, np.ndarray],
    epsilon: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Static within-speaker z-normalization.

    For each participant, subtract their mean feature vector and divide
    by their std. This collapses the t-SNE clusters to the origin.

    Parameters
    ----------
    features_by_participant : dict
        {participant_name: feature_matrix (n_samples, n_features)}
    epsilon : float
        Small constant to prevent division by zero.

    Returns
    -------
    dict
        {participant_name: normalized_feature_matrix}
    """
    normalized = {}
    for name, X in features_by_participant.items():
        if len(X) < 2:
            logger.warning("%s: Too few samples (%d) for z-normalization, skipping", name, len(X))
            normalized[name] = X
            continue

        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + epsilon
        normalized[name] = (X - mu) / sigma

        logger.debug(
            "%s: z-normalized %d samples x %d features",
            name, X.shape[0], X.shape[1],
        )

    return normalized


def running_zscore_per_speaker(
    features_by_participant: Dict[str, np.ndarray],
    timestamps_by_participant: Dict[str, np.ndarray],
    window_size: int = 20,
    min_history: int = 5,
    epsilon: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Running within-speaker z-normalization with a sliding window.

    For each sample, compute z-score relative to the previous K samples
    (in chronological order). This handles slow voice drift (colds, fatigue)
    while preserving glucose-sensitive short-term fluctuations.

    Parameters
    ----------
    features_by_participant : dict
        {participant_name: feature_matrix (n_samples, n_features)}
    timestamps_by_participant : dict
        {participant_name: timestamps array (for ordering)}
    window_size : int
        Number of past samples to use for running statistics.
    min_history : int
        Minimum history samples before using running stats (fallback to global).
    epsilon : float
        Small constant to prevent division by zero.

    Returns
    -------
    dict
        {participant_name: normalized_feature_matrix}
    """
    normalized = {}

    for name, X in features_by_participant.items():
        ts = timestamps_by_participant.get(name)
        if ts is None or len(X) < min_history:
            # Fallback to static normalization
            mu = X.mean(axis=0)
            sigma = X.std(axis=0) + epsilon
            normalized[name] = (X - mu) / sigma
            continue

        # Sort by timestamp
        order = np.argsort(ts)
        X_sorted = X[order]

        X_norm = np.zeros_like(X_sorted, dtype=np.float64)

        for i in range(len(X_sorted)):
            if i < min_history:
                # Not enough history: use all available up to this point
                window = X_sorted[:i + 1]
            else:
                start = max(0, i - window_size)
                window = X_sorted[start:i]  # exclude current sample

            mu = window.mean(axis=0)
            sigma = window.std(axis=0) + epsilon
            X_norm[i] = (X_sorted[i] - mu) / sigma

        # Restore original order
        inv_order = np.argsort(order)
        normalized[name] = X_norm[inv_order]

    return normalized


def rank_normalize_per_speaker(
    features_by_participant: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Percentile rank normalization within each speaker.

    Each feature value is replaced by its percentile rank within the
    speaker's data. This is robust to outliers and non-Gaussian distributions.

    Parameters
    ----------
    features_by_participant : dict
        {participant_name: feature_matrix (n_samples, n_features)}

    Returns
    -------
    dict
        {participant_name: rank_normalized_matrix} with values in [0, 1]
    """
    from scipy.stats import rankdata

    normalized = {}
    for name, X in features_by_participant.items():
        if len(X) < 3:
            normalized[name] = X
            continue

        X_rank = np.zeros_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            ranks = rankdata(X[:, j], method="average")
            X_rank[:, j] = (ranks - 1) / (len(ranks) - 1)  # Scale to [0, 1]

        normalized[name] = X_rank

    return normalized


def normalize_features(
    data_by_participant: Dict[str, Dict],
    method: str = "zscore",
    window_size: int = 20,
    min_history: int = 5,
) -> Dict[str, Dict]:
    """
    Apply within-speaker normalization to the full data dictionary.

    Parameters
    ----------
    data_by_participant : dict
        {participant_name: {"features": np.ndarray, "glucose": np.ndarray,
                            "timestamps": list, ...}}
    method : str
        One of "zscore", "running_zscore", "rank", "none".
    window_size : int
        Window size for running z-normalization.
    min_history : int
        Minimum history for running z-normalization.

    Returns
    -------
    dict
        Same structure with features replaced by normalized versions.
        Original features are preserved under "features_raw".
    """
    if method == "none":
        logger.info("Feature normalization: none (disabled)")
        return data_by_participant

    # Extract feature matrices
    features_dict = {
        name: data["features"]
        for name, data in data_by_participant.items()
    }
    timestamps_dict = {
        name: np.array(data.get("timestamps", []))
        for name, data in data_by_participant.items()
    }

    logger.info("Applying within-speaker normalization: %s", method)

    if method == "zscore":
        normalized = zscore_per_speaker(features_dict)
    elif method == "running_zscore":
        normalized = running_zscore_per_speaker(
            features_dict, timestamps_dict,
            window_size=window_size, min_history=min_history,
        )
    elif method == "rank":
        normalized = rank_normalize_per_speaker(features_dict)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Update data dict, preserving raw features
    result = {}
    for name, data in data_by_participant.items():
        result[name] = dict(data)  # shallow copy
        result[name]["features_raw"] = data["features"]
        result[name]["features"] = normalized[name]

    return result
