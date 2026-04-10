"""
Model Training & Cross-Validation
===================================
Unified model training for personalized and population glucose estimation.

Includes:
  - Standard regression (LOO, K-Fold, LOPO)
  - Deviation-from-personal-mean regression (Phase 2A)
  - Glucose rate-of-change classification (Phase 2B)
  - Clinically-relevant regime classification (Phase 2C)

Consolidates the duplicated model evaluation code from:
- comprehensive_analysis_v7.py
- full_production_analysis.py
- combined_hubert_mfcc_model.py
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import (
    BayesianRidge,
    Ridge,
    ElasticNet,
    LogisticRegression,
    Lasso,
    HuberRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import (
    cross_val_predict,
    LeaveOneOut,
    KFold,
    StratifiedKFold,
    LeaveOneGroupOut,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model Registry
# =============================================================================

def get_model(name: str, params: Optional[Dict] = None):
    """
    Create a scikit-learn regressor by name.

    Parameters
    ----------
    name : str
        One of: SVR, BayesianRidge, Ridge, ElasticNet, RandomForest,
        GradientBoosting, KNN.
    params : dict, optional
        Keyword arguments to pass to the constructor.
    """
    params = params or {}
    registry = {
        "SVR": lambda: SVR(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 10),
            gamma=params.get("gamma", "scale"),
        ),
        "BayesianRidge": lambda: BayesianRidge(**params),
        "Ridge": lambda: Ridge(alpha=params.get("alpha", 1.0)),
        "ElasticNet": lambda: ElasticNet(
            alpha=params.get("alpha", 0.1),
            l1_ratio=params.get("l1_ratio", 0.5),
        ),
        "Lasso": lambda: Lasso(alpha=params.get("alpha", 0.05)),
        "Huber": lambda: HuberRegressor(
            epsilon=params.get("epsilon", 1.35),
            alpha=params.get("alpha", 0.0001),
        ),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=params.get("random_state", 42),
        ),
        "GradientBoosting": lambda: GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            random_state=params.get("random_state", 42),
        ),
        "KNN": lambda: KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "distance"),
        ),
        "ExtraTrees": lambda: ExtraTreesRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            random_state=params.get("random_state", 42),
        ),
    }

    if name not in registry:
        raise ValueError(f"Unknown model: {name}. Available: {list(registry.keys())}")

    return registry[name]()


def _build_pipeline(model) -> Pipeline:
    """Wrap a model in a standardization pipeline."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", model),
    ])


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics."""
    from research.evaluation.metrics import clarke_error_grid, clarke_zone_percentages

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r, p_value = stats.pearsonr(y_true, y_pred)
    else:
        r, p_value = 0.0, 1.0

    # Clinical metrics
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    mard = float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)
    bias = float(np.mean(y_pred - y_true))

    zones = clarke_error_grid(y_true, y_pred)
    zones_pct = clarke_zone_percentages(zones)
    clarke_ab_pct = float(zones_pct.get("A", 0.0) + zones_pct.get("B", 0.0))

    low_mask = y_true < 70
    normal_mask = (y_true >= 70) & (y_true <= 180)
    high_mask = y_true > 180
    mae_low = float(np.mean(np.abs(y_pred[low_mask] - y_true[low_mask]))) if np.any(low_mask) else float("nan")
    mae_normal = float(np.mean(np.abs(y_pred[normal_mask] - y_true[normal_mask]))) if np.any(normal_mask) else float("nan")
    mae_high = float(np.mean(np.abs(y_pred[high_mask] - y_true[high_mask]))) if np.any(high_mask) else float("nan")

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "r": float(r),
        "p_value": float(p_value),
        "n_samples": len(y_true),
        "mard": mard,
        "bias": bias,
        "mae_low": mae_low,
        "mae_normal": mae_normal,
        "mae_high": mae_high,
        "clarke_a_pct": float(zones_pct.get("A", 0.0)),
        "clarke_b_pct": float(zones_pct.get("B", 0.0)),
        "clarke_ab_pct": clarke_ab_pct,
        "clarke_c_pct": float(zones_pct.get("C", 0.0)),
        "clarke_d_pct": float(zones_pct.get("D", 0.0)),
        "clarke_e_pct": float(zones_pct.get("E", 0.0)),
    }


# =============================================================================
# Personalized Model Training
# =============================================================================

def train_personalized(
    X: np.ndarray,
    y: np.ndarray,
    model_names: Optional[List[str]] = None,
    model_params: Optional[Dict[str, Dict]] = None,
    min_samples: int = 20,
    cv_kfold_threshold: int = 50,
    cv_kfold_splits: int = 10,
) -> Dict[str, Dict]:
    """
    Train and evaluate personalized models for a single participant using CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Glucose values (mg/dL).
    model_names : list of str, optional
        Which models to evaluate. Defaults to all available.
    model_params : dict, optional
        Per-model hyperparameters from config.yaml.
    min_samples : int
        Minimum samples required to train.
    cv_kfold_threshold : int
        Use K-fold instead of LOO when n_samples exceeds this.
    cv_kfold_splits : int
        Number of K-fold splits.

    Returns
    -------
    dict
        {model_name: {mae, rmse, r, r2, predictions, ...}} sorted by MAE.
    """
    if len(X) < min_samples:
        logger.warning("Too few samples (%d < %d) for personalized model", len(X), min_samples)
        return {}

    # Clean data
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    if model_names is None:
        model_names = ["SVR", "BayesianRidge", "RandomForest", "GradientBoosting", "KNN"]

    model_params = model_params or {}

    # Choose CV strategy
    if len(X) <= cv_kfold_threshold:
        cv = LeaveOneOut()
        cv_name = "LOO"
    else:
        cv = KFold(n_splits=cv_kfold_splits, shuffle=True, random_state=42)
        cv_name = f"{cv_kfold_splits}-fold"

    logger.info("Personalized CV: %s on %d samples", cv_name, len(X))

    results = {}
    for name in model_names:
        try:
            params = model_params.get(name.lower(), model_params.get(name, {}))
            model = get_model(name, params)
            pipeline = _build_pipeline(model)

            preds = cross_val_predict(pipeline, X, y, cv=cv)
            metrics = compute_metrics(y, preds)
            metrics["predictions"] = preds
            metrics["model_name"] = name

            results[name] = metrics
            logger.info("  %s: MAE=%.2f, r=%.3f", name, metrics["mae"], metrics["r"])

        except Exception as e:
            logger.warning("  %s failed: %s", name, e)

    # Sort by MAE
    results = dict(sorted(results.items(), key=lambda x: x[1]["mae"]))
    return results


def get_best_personalized(results: Dict[str, Dict]) -> Optional[Dict]:
    """Return the best model result (lowest MAE)."""
    if not results:
        return None
    best_name = min(results, key=lambda k: results[k]["mae"])
    return {**results[best_name], "model_name": best_name}


# =============================================================================
# Population Model Training
# =============================================================================

def train_population(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Optional[List[str]] = None,
    model_params: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """
    Train and evaluate population models using Leave-One-Person-Out CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Glucose values (mg/dL).
    groups : np.ndarray
        Participant labels for each sample.
    model_names : list of str, optional
        Which models to evaluate.
    model_params : dict, optional
        Per-model hyperparameters.

    Returns
    -------
    dict
        {model_name: {mae, rmse, r, r2, predictions, per_person, ...}}
    """
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    if model_names is None:
        model_names = ["BayesianRidge", "SVR", "RandomForest"]

    model_params = model_params or {}
    logo = LeaveOneGroupOut()

    unique_groups = np.unique(groups)
    logger.info(
        "Population LOPO: %d samples, %d participants",
        len(X), len(unique_groups),
    )

    results = {}
    for name in model_names:
        try:
            params = model_params.get(name.lower(), model_params.get(name, {}))
            model = get_model(name, params)
            pipeline = _build_pipeline(model)

            preds = cross_val_predict(pipeline, X, y, cv=logo, groups=groups)
            metrics = compute_metrics(y, preds)
            metrics["predictions"] = preds

            # Per-person breakdown
            per_person = {}
            for g in unique_groups:
                mask = groups == g
                if mask.sum() > 2:
                    per_person[g] = compute_metrics(y[mask], preds[mask])
            metrics["per_person"] = per_person

            results[name] = metrics
            logger.info("  %s: MAE=%.2f, r=%.3f", name, metrics["mae"], metrics["r"])

        except Exception as e:
            logger.warning("  %s failed: %s", name, e)

    results = dict(sorted(results.items(), key=lambda x: x[1]["mae"]))
    return results


# =============================================================================
# Baseline Comparison
# =============================================================================

def mean_predictor_baseline(y: np.ndarray) -> Dict[str, float]:
    """
    Compute the mean-predictor baseline (predicting the global mean).

    This is the minimum bar that any model must beat. If a model achieves
    similar MAE with near-zero correlation, it's just predicting the mean.
    """
    mean_pred = np.full_like(y, np.mean(y))
    return compute_metrics(y, mean_pred)


# =============================================================================
# Phase 2A: Deviation-from-Personal-Mean Regression
# =============================================================================

def normalize_target_deviation(
    y: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Transform glucose to deviation-from-personal-mean.

    target = (glucose - mean) / std

    Parameters
    ----------
    y : np.ndarray
        Raw glucose values (mg/dL).

    Returns
    -------
    tuple of (y_normalized, mean, std)
        Normalized target, original mean, original std for reconstruction.
    """
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma < 1e-6:
        sigma = 1.0
    y_norm = (y - mu) / sigma
    return y_norm, mu, sigma


def denormalize_predictions(
    y_pred_norm: np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Reconstruct mg/dL predictions from deviation predictions."""
    return y_pred_norm * sigma + mu


def train_personalized_deviation(
    X: np.ndarray,
    y: np.ndarray,
    model_names: Optional[List[str]] = None,
    model_params: Optional[Dict[str, Dict]] = None,
    min_samples: int = 20,
    cv_kfold_threshold: int = 50,
    cv_kfold_splits: int = 10,
) -> Dict[str, Dict]:
    """
    Train personalized models predicting deviation-from-personal-mean.

    This reframes the problem: instead of absolute glucose estimation,
    the model detects anomalies relative to a personal baseline.

    Parameters
    ----------
    X, y, model_names, model_params, min_samples, cv_kfold_threshold, cv_kfold_splits
        Same as train_personalized().

    Returns
    -------
    dict
        {model_name: {mae (in mg/dL), rmse, r, predictions (in mg/dL), ...}}
    """
    if len(X) < min_samples:
        logger.warning("Too few samples (%d < %d)", len(X), min_samples)
        return {}

    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Normalize target
    y_norm, mu, sigma = normalize_target_deviation(y)

    if model_names is None:
        model_names = ["SVR", "BayesianRidge", "RandomForest", "GradientBoosting", "KNN"]

    model_params = model_params or {}

    # CV strategy
    if len(X) <= cv_kfold_threshold:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=cv_kfold_splits, shuffle=True, random_state=42)

    results = {}
    for name in model_names:
        try:
            params = model_params.get(name.lower(), model_params.get(name, {}))
            model = get_model(name, params)
            pipeline = _build_pipeline(model)

            # Predict in normalized space
            preds_norm = cross_val_predict(pipeline, X, y_norm, cv=cv)

            # Denormalize back to mg/dL for evaluation
            preds_mgdl = denormalize_predictions(preds_norm, mu, sigma)

            metrics = compute_metrics(y, preds_mgdl)
            metrics["predictions"] = preds_mgdl
            metrics["model_name"] = name
            metrics["target_type"] = "deviation"
            metrics["target_mu"] = mu
            metrics["target_sigma"] = sigma

            results[name] = metrics
            logger.info("  %s (deviation): MAE=%.2f, r=%.3f", name, metrics["mae"], metrics["r"])

        except Exception as e:
            logger.warning("  %s deviation failed: %s", name, e)

    results = dict(sorted(results.items(), key=lambda x: x[1]["mae"]))
    return results


# =============================================================================
# Phase 2B: Glucose Rate-of-Change Classification
# =============================================================================

def get_classifier(name: str, params: Optional[Dict] = None):
    """Create a scikit-learn classifier by name."""
    params = params or {}
    registry = {
        "LogisticRegression": lambda: LogisticRegression(
            C=params.get("C", 1.0), max_iter=1000, random_state=42,
        ),
        "SVC": lambda: SVC(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 10),
            gamma=params.get("gamma", "scale"),
        ),
        "RandomForestClassifier": lambda: RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=42,
        ),
        "GradientBoostingClassifier": lambda: GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            random_state=42,
        ),
        "KNNClassifier": lambda: KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "distance"),
        ),
    }

    if name not in registry:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(registry.keys())}")

    return registry[name]()


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "n_samples": len(y_true),
    }


def train_rate_of_change_classifier(
    X: np.ndarray,
    rate_labels: np.ndarray,
    classifier_names: Optional[List[str]] = None,
    min_samples: int = 20,
) -> Dict[str, Dict]:
    """
    Train classifiers for glucose rate-of-change prediction.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    rate_labels : np.ndarray
        Labels: "rising", "stable", "falling" (or "unknown").
    classifier_names : list of str, optional
        Classifiers to evaluate.
    min_samples : int
        Minimum samples.

    Returns
    -------
    dict
        {classifier_name: {accuracy, f1_macro, f1_weighted, predictions, ...}}
    """
    # Filter out "unknown" labels
    mask = rate_labels != "unknown"
    X_filt = X[mask]
    y_filt = rate_labels[mask]

    if len(X_filt) < min_samples:
        logger.warning("Too few labeled samples (%d < %d) for rate classification", len(X_filt), min_samples)
        return {}

    # Check class balance
    unique, counts = np.unique(y_filt, return_counts=True)
    logger.info("Rate-of-change class distribution: %s", dict(zip(unique, counts)))

    if len(unique) < 2:
        logger.warning("Only one class present — cannot train classifier")
        return {}

    X_filt = np.nan_to_num(X_filt, nan=0, posinf=0, neginf=0)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_filt)

    if classifier_names is None:
        classifier_names = [
            "LogisticRegression", "SVC", "RandomForestClassifier",
            "GradientBoostingClassifier", "KNNClassifier",
        ]

    # Use stratified K-fold or LOO
    min_class_count = min(counts)
    if min_class_count >= 5:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = LeaveOneOut()

    results = {}
    for name in classifier_names:
        try:
            clf = get_classifier(name)
            pipeline = _build_pipeline(clf)

            preds_encoded = cross_val_predict(pipeline, X_filt, y_encoded, cv=cv)
            preds = le.inverse_transform(preds_encoded)

            metrics = compute_classification_metrics(y_filt, preds)
            metrics["predictions"] = preds
            metrics["classifier_name"] = name
            metrics["class_names"] = list(le.classes_)

            results[name] = metrics
            logger.info(
                "  %s: Accuracy=%.1f%%, F1=%.3f",
                name, metrics["accuracy"] * 100, metrics["f1_macro"],
            )

        except Exception as e:
            logger.warning("  %s rate classification failed: %s", name, e)

    results = dict(sorted(results.items(), key=lambda x: -x[1]["f1_macro"]))
    return results


# =============================================================================
# Phase 2C: Clinically-Relevant Regime Classification
# =============================================================================

def train_regime_classifier(
    X: np.ndarray,
    regime_labels: np.ndarray,
    classifier_names: Optional[List[str]] = None,
    min_samples: int = 20,
) -> Dict[str, Dict]:
    """
    Train classifiers for glucose regime (hypo/normal/hyper) prediction.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    regime_labels : np.ndarray
        Labels: "hypo_risk", "normal", "hyper_risk".
    classifier_names : list of str, optional
        Classifiers to evaluate.
    min_samples : int
        Minimum samples.

    Returns
    -------
    dict
        {classifier_name: {accuracy, f1_macro, f1_weighted, predictions, ...}}
    """
    if len(X) < min_samples:
        logger.warning("Too few samples (%d < %d) for regime classification", len(X), min_samples)
        return {}

    unique, counts = np.unique(regime_labels, return_counts=True)
    logger.info("Regime class distribution: %s", dict(zip(unique, counts)))

    if len(unique) < 2:
        logger.warning("Only one regime class present — cannot train classifier")
        return {}

    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    le = LabelEncoder()
    y_encoded = le.fit_transform(regime_labels)

    if classifier_names is None:
        classifier_names = [
            "LogisticRegression", "SVC", "RandomForestClassifier",
            "GradientBoostingClassifier", "KNNClassifier",
        ]

    min_class_count = min(counts)
    if min_class_count >= 5:
        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = LeaveOneOut()

    results = {}
    for name in classifier_names:
        try:
            clf = get_classifier(name)
            pipeline = _build_pipeline(clf)

            preds_encoded = cross_val_predict(pipeline, X, y_encoded, cv=cv)
            preds = le.inverse_transform(preds_encoded)

            metrics = compute_classification_metrics(regime_labels, preds)
            metrics["predictions"] = preds
            metrics["classifier_name"] = name
            metrics["class_names"] = list(le.classes_)

            results[name] = metrics
            logger.info(
                "  %s: Accuracy=%.1f%%, F1=%.3f",
                name, metrics["accuracy"] * 100, metrics["f1_macro"],
            )

        except Exception as e:
            logger.warning("  %s regime classification failed: %s", name, e)

    results = dict(sorted(results.items(), key=lambda x: -x[1]["f1_macro"]))
    return results


# =============================================================================
# Phase 0A: Temporal (Chronological) Validation — Re-export
# =============================================================================

def train_personalized_temporal(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    model,
    train_fraction: float = 0.7,
) -> Dict[str, Any]:
    """
    Train and evaluate a personalized model using chronological split.

    Re-exports research.evaluation.temporal_cv.train_personalized_temporal.
    Use this for honest time-series evaluation (no temporal leakage).
    """
    from research.evaluation.temporal_cv import train_personalized_temporal as _temporal
    return _temporal(X, y, timestamps, model, train_fraction)
