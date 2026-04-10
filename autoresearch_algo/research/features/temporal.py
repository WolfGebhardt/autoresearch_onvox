"""
Temporal Context Features
===========================
Features that capture the temporal dynamics of voice changes.

Currently, each sample is treated independently. But glucose changes voice
over TIME — the sequence and deltas between recordings carry information.

Features:
  1. Delta features: feature(t) - feature(t-1) for consecutive recordings
  2. Circadian encoding: sin/cos of hour-of-day (glucose has circadian patterns)
  3. Time since last recording
  4. Rolling statistics: mean/std of features over last N recordings
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_circadian_features(timestamps: List[str]) -> np.ndarray:
    """
    Encode time-of-day as cyclical sin/cos features.

    Glucose has strong circadian patterns; encoding time-of-day lets
    the model exploit this without discontinuity at midnight.

    Parameters
    ----------
    timestamps : list of str
        ISO-format timestamp strings.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 4): [hour_sin, hour_cos, minute_sin, minute_cos]
    """
    circadian = []
    for ts_str in timestamps:
        try:
            if isinstance(ts_str, str):
                ts = datetime.fromisoformat(ts_str)
            else:
                ts = ts_str

            hour = ts.hour + ts.minute / 60.0
            # Day-of-week encoding (weekday patterns in eating/activity)
            dow = ts.weekday()

            circadian.append([
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * dow / 7),
                np.cos(2 * np.pi * dow / 7),
            ])
        except (ValueError, AttributeError):
            circadian.append([0.0, 0.0, 0.0, 0.0])

    return np.array(circadian, dtype=np.float32)


def compute_delta_features(
    features: np.ndarray,
    timestamps: List[str],
    max_gap_hours: float = 4.0,
) -> np.ndarray:
    """
    Compute delta features: feature(t) - feature(t-1) for consecutive recordings.

    Captures voice CHANGES that might track glucose changes.
    Only computes deltas for recordings within max_gap_hours.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features), in chronological order.
    timestamps : list of str
        ISO-format timestamps (for gap detection).
    max_gap_hours : float
        Maximum time gap for delta computation. If gap is larger,
        delta is set to zero (recordings too far apart to compare).

    Returns
    -------
    np.ndarray
        Delta feature matrix (n_samples, n_features). First sample gets zeros.
    """
    n_samples, n_features = features.shape
    deltas = np.zeros_like(features)

    # Parse timestamps for gap detection
    parsed_ts = []
    for ts_str in timestamps:
        try:
            if isinstance(ts_str, str):
                parsed_ts.append(datetime.fromisoformat(ts_str))
            else:
                parsed_ts.append(ts_str)
        except (ValueError, AttributeError):
            parsed_ts.append(None)

    # Sort by timestamp
    valid_pairs = [(i, parsed_ts[i]) for i in range(n_samples) if parsed_ts[i] is not None]
    valid_pairs.sort(key=lambda x: x[1])

    for pos in range(1, len(valid_pairs)):
        curr_idx, curr_ts = valid_pairs[pos]
        prev_idx, prev_ts = valid_pairs[pos - 1]

        gap_hours = (curr_ts - prev_ts).total_seconds() / 3600.0

        if gap_hours <= max_gap_hours:
            deltas[curr_idx] = features[curr_idx] - features[prev_idx]
        # else: deltas remain zero (recordings too far apart)

    return deltas


def compute_time_since_last(timestamps: List[str]) -> np.ndarray:
    """
    Compute time since previous recording for each sample.

    Parameters
    ----------
    timestamps : list of str
        ISO-format timestamps.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 1) with hours since last recording.
        First recording gets 0.
    """
    parsed = []
    for ts_str in timestamps:
        try:
            if isinstance(ts_str, str):
                parsed.append(datetime.fromisoformat(ts_str))
            else:
                parsed.append(ts_str)
        except (ValueError, AttributeError):
            parsed.append(None)

    # Sort indices by timestamp
    indexed = [(i, t) for i, t in enumerate(parsed) if t is not None]
    indexed.sort(key=lambda x: x[1])

    time_since = np.zeros((len(timestamps), 1), dtype=np.float32)

    for pos in range(1, len(indexed)):
        curr_idx, curr_ts = indexed[pos]
        prev_idx, prev_ts = indexed[pos - 1]
        gap_hours = (curr_ts - prev_ts).total_seconds() / 3600.0
        time_since[curr_idx, 0] = min(gap_hours, 48.0)  # Cap at 48 hours

    return time_since


def compute_rolling_stats(
    features: np.ndarray,
    timestamps: List[str],
    window_size: int = 5,
) -> np.ndarray:
    """
    Compute rolling mean and std of features over the last N recordings.

    Provides temporal smoothing and context about recent feature trends.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features).
    timestamps : list of str
        ISO-format timestamps for ordering.
    window_size : int
        Number of past recordings to include.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 2 * n_features) — [rolling_mean, rolling_std]
    """
    n_samples, n_features = features.shape

    # Parse and sort timestamps
    parsed = []
    for ts_str in timestamps:
        try:
            if isinstance(ts_str, str):
                parsed.append(datetime.fromisoformat(ts_str))
            else:
                parsed.append(ts_str)
        except (ValueError, AttributeError):
            parsed.append(datetime.min)

    order = np.argsort(parsed)

    rolling = np.zeros((n_samples, 2 * n_features), dtype=np.float32)

    for pos, idx in enumerate(order):
        # Gather window of past samples
        start = max(0, pos - window_size)
        window_indices = order[start:pos + 1]
        window_features = features[window_indices]

        rolling[idx, :n_features] = window_features.mean(axis=0)
        if len(window_features) > 1:
            rolling[idx, n_features:] = window_features.std(axis=0)

    return rolling


def build_temporal_features(
    data_by_participant: Dict[str, Dict],
    include_circadian: bool = True,
    include_deltas: bool = True,
    include_time_since: bool = True,
    include_rolling: bool = False,
    rolling_window: int = 5,
    max_gap_hours: float = 4.0,
) -> Dict[str, Dict]:
    """
    Build temporal context features and append them to existing features.

    Parameters
    ----------
    data_by_participant : dict
        {participant_name: {"features": np.ndarray, "timestamps": list, ...}}
    include_circadian : bool
        Add time-of-day sin/cos encoding.
    include_deltas : bool
        Add delta features (feature(t) - feature(t-1)).
    include_time_since : bool
        Add time-since-last-recording feature.
    include_rolling : bool
        Add rolling mean/std features.
    rolling_window : int
        Window size for rolling features.
    max_gap_hours : float
        Max gap for delta computation.

    Returns
    -------
    dict
        Same structure with temporal features appended to feature matrices.
    """
    result = {}
    total_temporal_dims = 0

    for name, data in data_by_participant.items():
        X = data["features"]
        timestamps = data.get("timestamps", [])

        if len(timestamps) == 0 or len(X) == 0:
            result[name] = data
            continue

        extra_features = []

        if include_circadian:
            circ = compute_circadian_features(timestamps)
            extra_features.append(circ)

        if include_deltas:
            deltas = compute_delta_features(X, timestamps, max_gap_hours)
            extra_features.append(deltas)

        if include_time_since:
            time_since = compute_time_since_last(timestamps)
            extra_features.append(time_since)

        if include_rolling:
            rolling = compute_rolling_stats(X, timestamps, rolling_window)
            extra_features.append(rolling)

        if extra_features:
            X_temporal = np.hstack([X] + extra_features)
            total_temporal_dims = X_temporal.shape[1] - X.shape[1]
        else:
            X_temporal = X

        result[name] = dict(data)
        result[name]["features"] = X_temporal
        result[name]["features_base"] = X  # Keep base features for ensemble

    if total_temporal_dims > 0:
        logger.info(
            "Added %d temporal feature dimensions to each participant",
            total_temporal_dims,
        )

    return result
