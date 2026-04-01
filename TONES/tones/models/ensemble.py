"""
Diverse Ensemble for Glucose Estimation
==========================================
Combines models that capture DIFFERENT aspects of the voice-glucose signal.

Key insight: diversity comes from different FEATURE SETS, not just different
algorithms on the same features. Each sub-model specializes in a different
signal pathway:

  - Ridge on MFCC features: captures spectral envelope changes
  - SVR_RBF on voice quality features: captures jitter/shimmer/tremor
  - GBR on combined features: captures non-linear interactions
  - BayesianRidge on temporal delta features: captures dynamics

The ensemble averages predictions (optionally weighted by per-model confidence).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_predict, LeaveOneOut, KFold

logger = logging.getLogger(__name__)


class DiverseEnsemble:
    """
    Ensemble that combines models trained on different feature subsets.

    Parameters
    ----------
    models : list of (name, model, feature_key) tuples
        Each entry specifies:
          - name: identifier for the sub-model
          - model: unfitted sklearn estimator
          - feature_key: key into the feature dict (e.g., "mfcc", "voice_quality")
    weighting : str
        One of "equal", "inverse_mae" (weight by 1/MAE from CV).
    """

    def __init__(
        self,
        models: List[Tuple[str, object, str]],
        weighting: str = "inverse_mae",
    ):
        self.models = models
        self.weighting = weighting
        self._fitted_pipelines: Dict[str, Pipeline] = {}
        self._weights: Dict[str, float] = {}

    def fit_and_predict_cv(
        self,
        feature_sets: Dict[str, np.ndarray],
        y: np.ndarray,
        cv=None,
    ) -> Tuple[np.ndarray, Dict[str, Dict]]:
        """
        Fit each sub-model via CV and produce ensemble predictions.

        Parameters
        ----------
        feature_sets : dict
            {feature_key: np.ndarray (n_samples, n_features_k)}
        y : np.ndarray
            Target values (mg/dL).
        cv : sklearn CV object, optional
            Cross-validation strategy. Defaults to LOO or 10-fold.

        Returns
        -------
        tuple of (ensemble_predictions, per_model_results)
        """
        from tones.models.train import compute_metrics

        n = len(y)

        if cv is None:
            if n <= 50:
                cv = LeaveOneOut()
            else:
                cv = KFold(n_splits=10, shuffle=True, random_state=42)

        all_predictions = {}
        per_model_results = {}

        for name, model, feat_key in self.models:
            if feat_key not in feature_sets:
                logger.warning("  Ensemble: feature set '%s' not found, skipping %s", feat_key, name)
                continue

            X = feature_sets[feat_key]
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

            if X.shape[0] != n:
                logger.warning(
                    "  Ensemble: feature set '%s' has %d samples, expected %d. Skipping.",
                    feat_key, X.shape[0], n,
                )
                continue

            try:
                pipeline = Pipeline([
                    ("scaler", RobustScaler()),
                    ("model", model),
                ])

                preds = cross_val_predict(pipeline, X, y, cv=cv)
                metrics = compute_metrics(y, preds)

                all_predictions[name] = preds
                per_model_results[name] = {
                    "mae": metrics["mae"],
                    "r": metrics["r"],
                    "feature_set": feat_key,
                    "n_features": X.shape[1],
                    "predictions": preds,
                }

                logger.info(
                    "  Ensemble member %s (%s, %d feats): MAE=%.2f, r=%.3f",
                    name, feat_key, X.shape[1], metrics["mae"], metrics["r"],
                )

            except Exception as e:
                logger.warning("  Ensemble member %s failed: %s", name, e)

        if not all_predictions:
            return np.full(n, np.mean(y)), {}

        # Compute weights
        if self.weighting == "inverse_mae":
            for name, res in per_model_results.items():
                mae = res["mae"]
                self._weights[name] = 1.0 / (mae + 1e-6)
        else:
            for name in per_model_results:
                self._weights[name] = 1.0

        # Normalize weights
        total_weight = sum(self._weights.values())
        for name in self._weights:
            self._weights[name] /= total_weight

        # Weighted average
        ensemble_preds = np.zeros(n)
        for name, preds in all_predictions.items():
            w = self._weights.get(name, 0.0)
            ensemble_preds += w * preds

        return ensemble_preds, per_model_results


def build_default_ensemble(cfg: Dict = None) -> DiverseEnsemble:
    """
    Build the default diverse ensemble configuration.

    Feature sets expected:
      - "mfcc": standard MFCC features
      - "voice_quality": jitter/shimmer/tremor/formants
      - "combined": all features concatenated
      - "temporal": features with temporal context appended
    """
    from tones.models.train import get_model

    models = [
        ("Ridge_MFCC", get_model("Ridge", {"alpha": 1.0}), "mfcc"),
        ("SVR_VoiceQuality", get_model("SVR", {"kernel": "rbf", "C": 10}), "voice_quality"),
        ("GBR_Combined", get_model("GradientBoosting", {"n_estimators": 100, "max_depth": 5}), "combined"),
        ("BayesRidge_Temporal", get_model("BayesianRidge", {}), "temporal"),
    ]

    return DiverseEnsemble(models=models, weighting="inverse_mae")


def train_ensemble_personalized(
    data_by_participant: Dict[str, Dict],
    feature_sets_by_participant: Dict[str, Dict[str, np.ndarray]],
    min_samples: int = 20,
) -> Dict[str, Dict]:
    """
    Train diverse ensemble for each participant.

    Parameters
    ----------
    data_by_participant : dict
        {name: {"glucose": np.ndarray, ...}}
    feature_sets_by_participant : dict
        {name: {"mfcc": np.ndarray, "voice_quality": np.ndarray, ...}}
    min_samples : int
        Minimum samples per participant.

    Returns
    -------
    dict
        {name: {"ensemble_mae", "ensemble_r", "per_model": {...}, ...}}
    """
    from tones.models.train import compute_metrics, mean_predictor_baseline

    results = {}

    for name, data in data_by_participant.items():
        y = data["glucose"]

        if len(y) < min_samples:
            continue

        feature_sets = feature_sets_by_participant.get(name, {})
        if not feature_sets:
            continue

        logger.info("\n  %s (%d samples):", name, len(y))

        ensemble = build_default_ensemble()
        ensemble_preds, per_model = ensemble.fit_and_predict_cv(feature_sets, y)

        if len(per_model) == 0:
            continue

        ensemble_metrics = compute_metrics(y, ensemble_preds)
        baseline = mean_predictor_baseline(y)

        results[name] = {
            "ensemble_mae": ensemble_metrics["mae"],
            "ensemble_rmse": ensemble_metrics["rmse"],
            "ensemble_r": ensemble_metrics["r"],
            "ensemble_predictions": ensemble_preds,
            "baseline_mae": baseline["mae"],
            "improvement": baseline["mae"] - ensemble_metrics["mae"],
            "weights": dict(ensemble._weights),
            "per_model": {
                k: {"mae": v["mae"], "r": v["r"], "feature_set": v["feature_set"]}
                for k, v in per_model.items()
            },
            "n_samples": len(y),
        }

        logger.info(
            "    ENSEMBLE: MAE=%.2f, r=%.3f (baseline=%.2f, improvement=%.2f)",
            ensemble_metrics["mae"], ensemble_metrics["r"],
            baseline["mae"], baseline["mae"] - ensemble_metrics["mae"],
        )

    return results
