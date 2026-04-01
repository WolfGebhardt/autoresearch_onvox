"""
Temporal (Chronological) Validation
======================================
Honest evaluation strategies that respect the time-series nature of
voice-glucose data. Replaces LOO-CV which causes temporal leakage.

Key insight: adjacent recordings from the same day share confounds
(meal timing, stress, environment) that leak across LOO folds.

Strategies:
  1. Chronological split: Train on first 70%, test on last 30%
  2. Expanding-window walk-forward: Train [0..k], test [k+1], expand
  3. Time-gap split: Ensure a minimum time gap between train and test sets
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


def _build_pipeline(model) -> Pipeline:
    """Wrap a model in a standardization pipeline."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", model),
    ])


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r, p_value = stats.pearsonr(y_true, y_pred)
    else:
        r, p_value = 0.0, 1.0

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "r": float(r),
        "p_value": float(p_value),
        "n_samples": len(y_true),
        "n_train": 0,  # filled by caller
        "n_test": 0,   # filled by caller
    }


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    train_fraction: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data chronologically: first train_fraction for training, rest for testing.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target values.
    timestamps : np.ndarray
        Timestamps for ordering (string ISO format or datetime-like).
    train_fraction : float
        Fraction of data for training (default 0.7).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # Sort by timestamp
    order = np.argsort(timestamps)
    X_sorted = X[order]
    y_sorted = y[order]

    split_idx = int(len(X_sorted) * train_fraction)
    split_idx = max(split_idx, 5)  # Ensure at least 5 training samples
    split_idx = min(split_idx, len(X_sorted) - 3)  # Ensure at least 3 test samples

    return (
        X_sorted[:split_idx],
        X_sorted[split_idx:],
        y_sorted[:split_idx],
        y_sorted[split_idx:],
    )


def train_personalized_temporal(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    model,
    train_fraction: float = 0.7,
) -> Dict[str, float]:
    """
    Train and evaluate a personalized model using chronological split.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Glucose values (mg/dL).
    timestamps : np.ndarray
        Timestamps for each sample (for chronological ordering).
    model : sklearn estimator
        Unfitted model to train.
    train_fraction : float
        Fraction of data for training.

    Returns
    -------
    dict
        Metrics including mae, rmse, r, r2, predictions, etc.
    """
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    X_train, X_test, y_train, y_test = chronological_split(
        X, y, timestamps, train_fraction
    )

    pipeline = _build_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = _compute_metrics(y_test, y_pred)
    metrics["n_train"] = len(y_train)
    metrics["n_test"] = len(y_test)
    metrics["predictions"] = y_pred
    metrics["actual_test"] = y_test
    metrics["cv_strategy"] = "chronological_split"

    return metrics


def train_personalized_walkforward(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    model_factory,
    min_train_samples: int = 10,
    step_size: int = 1,
) -> Dict[str, float]:
    """
    Expanding-window walk-forward validation.

    Train on [0..k], predict k+1, expand window. This simulates
    real-world deployment where the model improves with each new sample.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Glucose values (mg/dL).
    timestamps : np.ndarray
        Timestamps for chronological ordering.
    model_factory : callable
        Function that returns a new unfitted model instance.
    min_train_samples : int
        Minimum training samples before starting predictions.
    step_size : int
        Number of samples to predict before retraining.

    Returns
    -------
    dict
        Aggregated metrics with all out-of-sample predictions.
    """
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Sort by timestamp
    order = np.argsort(timestamps)
    X_sorted = X[order]
    y_sorted = y[order]

    all_preds = []
    all_actual = []

    for start_test in range(min_train_samples, len(X_sorted), step_size):
        end_test = min(start_test + step_size, len(X_sorted))

        X_train = X_sorted[:start_test]
        y_train = y_sorted[:start_test]
        X_test = X_sorted[start_test:end_test]
        y_test = y_sorted[start_test:end_test]

        if len(X_test) == 0:
            break

        try:
            model = model_factory()
            pipeline = _build_pipeline(model)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            all_preds.extend(preds)
            all_actual.extend(y_test)
        except Exception as e:
            logger.warning("Walk-forward step %d failed: %s", start_test, e)
            continue

    if len(all_preds) == 0:
        return {"mae": float("inf"), "r": 0.0, "n_samples": 0}

    y_true = np.array(all_actual)
    y_pred = np.array(all_preds)

    metrics = _compute_metrics(y_true, y_pred)
    metrics["n_train"] = min_train_samples  # initial training size
    metrics["n_test"] = len(y_pred)
    metrics["predictions"] = y_pred
    metrics["actual_test"] = y_true
    metrics["cv_strategy"] = "walk_forward"

    return metrics


def evaluate_all_temporal(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    model_names: Optional[List[str]] = None,
    model_params: Optional[Dict] = None,
    min_samples: int = 20,
    train_fraction: float = 0.7,
) -> Dict[str, Dict]:
    """
    Evaluate multiple models using both chronological split and walk-forward.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Glucose values.
    timestamps : np.ndarray
        Sample timestamps.
    model_names : list of str, optional
        Which models to evaluate.
    model_params : dict, optional
        Per-model hyperparameters.
    min_samples : int
        Minimum samples required.
    train_fraction : float
        Train/test split ratio for chronological split.

    Returns
    -------
    dict
        {model_name: {"chrono": metrics, "walkforward": metrics}}
    """
    from tones.models.train import get_model

    if len(X) < min_samples:
        logger.warning("Too few samples (%d < %d) for temporal validation", len(X), min_samples)
        return {}

    if model_names is None:
        model_names = ["SVR", "BayesianRidge", "RandomForest", "GradientBoosting", "KNN"]

    model_params = model_params or {}

    results = {}
    for name in model_names:
        try:
            params = model_params.get(name.lower(), model_params.get(name, {}))

            # Chronological split
            model = get_model(name, params)
            chrono_metrics = train_personalized_temporal(
                X, y, timestamps, model, train_fraction
            )
            chrono_metrics["model_name"] = name

            # Walk-forward
            def factory(n=name, p=params):
                return get_model(n, p)

            wf_metrics = train_personalized_walkforward(
                X, y, timestamps, factory, min_train_samples=max(10, int(len(X) * 0.3))
            )
            wf_metrics["model_name"] = name

            results[name] = {
                "chrono": chrono_metrics,
                "walkforward": wf_metrics,
            }

            logger.info(
                "  %s: Chrono MAE=%.2f r=%.3f | WF MAE=%.2f r=%.3f",
                name,
                chrono_metrics["mae"], chrono_metrics["r"],
                wf_metrics["mae"], wf_metrics["r"],
            )
        except Exception as e:
            logger.warning("  %s temporal eval failed: %s", name, e)

    return results
