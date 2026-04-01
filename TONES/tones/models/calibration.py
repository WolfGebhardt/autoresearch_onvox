"""
Improved Few-Shot Calibration
================================
Extends the original FewShotCalibrator (which only corrected mean bias)
with slope+intercept correction, feature-based calibration, and
recency-weighted adaptation.

Use case: When deploying a population model to a new user, the first K
samples from that user are used to calibrate the model to their personal
baseline. This is much cheaper than training a full personalized model.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


class FewShotCalibrator:
    """
    Few-shot calibration for personalizing population model predictions.

    Strategies:
      1. Bias correction: calibrated = predicted + mean_error
      2. Slope+intercept: calibrated = a * predicted + b
      3. Feature-based: small Ridge on top of population predictions
      4. Recency-weighted: weight calibration samples by recency

    Parameters
    ----------
    min_samples : int
        Minimum calibration samples before calibration starts.
    max_samples : int
        Maximum calibration samples to retain (sliding window).
    strategy : str
        One of "bias", "linear", "feature_based".
    recency_half_life : int
        Half-life for recency weighting (in samples). None disables.
    """

    def __init__(
        self,
        min_samples: int = 3,
        max_samples: int = 30,
        strategy: str = "linear",
        recency_half_life: Optional[int] = 10,
    ):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.strategy = strategy
        self.recency_half_life = recency_half_life

        self.calibration_samples: List[Dict] = []
        self.is_calibrated = False
        self._bias = 0.0
        self._slope = 1.0
        self._intercept = 0.0
        self._feature_model = None

    def add_sample(
        self,
        features: np.ndarray,
        predicted: float,
        actual: float,
    ):
        """
        Add a calibration sample (a prediction-actual pair).

        Parameters
        ----------
        features : np.ndarray
            Feature vector for this sample.
        predicted : float
            Model's prediction for this sample.
        actual : float
            Ground truth glucose value.
        """
        self.calibration_samples.append({
            "features": features,
            "predicted": predicted,
            "actual": actual,
            "error": actual - predicted,
        })

        # Maintain sliding window
        if len(self.calibration_samples) > self.max_samples:
            self.calibration_samples = self.calibration_samples[-self.max_samples:]

        # Update calibration when we have enough samples
        if len(self.calibration_samples) >= self.min_samples:
            self._update_calibration()

    def _compute_weights(self) -> np.ndarray:
        """Compute recency weights for calibration samples."""
        n = len(self.calibration_samples)
        if self.recency_half_life is None or n == 0:
            return np.ones(n)

        # Exponential decay: most recent sample has weight 1.0
        indices = np.arange(n)
        weights = np.exp(-np.log(2) * (n - 1 - indices) / self.recency_half_life)
        return weights / weights.sum() * n  # Normalize to sum=n

    def _update_calibration(self):
        """Recompute calibration parameters."""
        weights = self._compute_weights()

        errors = np.array([s["error"] for s in self.calibration_samples])
        predictions = np.array([s["predicted"] for s in self.calibration_samples])
        actuals = np.array([s["actual"] for s in self.calibration_samples])

        if self.strategy == "bias":
            # Simple weighted bias correction
            self._bias = np.average(errors, weights=weights)

        elif self.strategy == "linear":
            # Weighted linear regression: actual = a * predicted + b
            if len(predictions) >= 3 and np.std(predictions) > 1e-6:
                # Weighted least squares
                W = np.diag(weights)
                X = np.column_stack([predictions, np.ones(len(predictions))])
                try:
                    beta = np.linalg.lstsq(W @ X, W @ actuals, rcond=None)[0]
                    self._slope = float(beta[0])
                    self._intercept = float(beta[1])
                except np.linalg.LinAlgError:
                    self._bias = np.average(errors, weights=weights)
            else:
                # Not enough variance: fallback to bias correction
                self._bias = np.average(errors, weights=weights)

        elif self.strategy == "feature_based":
            # Small Ridge model on [predicted, features] -> actual
            features = np.array([s["features"] for s in self.calibration_samples])
            X_calib = np.column_stack([predictions.reshape(-1, 1), features])

            self._feature_model = Ridge(alpha=10.0)
            self._feature_model.fit(X_calib, actuals, sample_weight=weights)

        self.is_calibrated = True

    def calibrate(
        self,
        prediction: float,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Calibrate a model prediction.

        Parameters
        ----------
        prediction : float
            Raw model prediction.
        features : np.ndarray, optional
            Feature vector (needed for feature_based strategy).

        Returns
        -------
        tuple of (calibrated_prediction, confidence)
            Confidence is based on calibration sample count and consistency.
        """
        if not self.is_calibrated:
            return prediction, 0.0

        n = len(self.calibration_samples)
        confidence = min(1.0, n / self.max_samples)

        if self.strategy == "bias":
            calibrated = prediction + self._bias

        elif self.strategy == "linear":
            calibrated = self._slope * prediction + self._intercept

        elif self.strategy == "feature_based":
            if self._feature_model is not None and features is not None:
                X_input = np.concatenate([[prediction], features]).reshape(1, -1)
                calibrated = float(self._feature_model.predict(X_input)[0])
            else:
                calibrated = prediction + self._bias

        else:
            calibrated = prediction

        return calibrated, confidence

    def get_diagnostics(self) -> Dict:
        """Return calibration diagnostics."""
        if not self.calibration_samples:
            return {"n_samples": 0, "is_calibrated": False}

        errors = np.array([s["error"] for s in self.calibration_samples])
        return {
            "n_samples": len(self.calibration_samples),
            "is_calibrated": self.is_calibrated,
            "strategy": self.strategy,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "bias": self._bias,
            "slope": self._slope,
            "intercept": self._intercept,
        }


def evaluate_few_shot_calibration(
    data_by_participant: Dict[str, Dict],
    population_model,
    scaler,
    n_calibration: int = 10,
    strategy: str = "linear",
) -> Dict[str, Dict]:
    """
    Evaluate few-shot calibration for each participant.

    Simulates the deployment scenario:
    1. Use a pre-trained population model
    2. For each participant, use first n_calibration samples to calibrate
    3. Evaluate on remaining samples

    Parameters
    ----------
    data_by_participant : dict
        {name: {"features": np.ndarray, "glucose": np.ndarray, "timestamps": list}}
    population_model : fitted sklearn model
        Pre-trained population model.
    scaler : fitted sklearn scaler
        Pre-trained feature scaler.
    n_calibration : int
        Number of samples to use for calibration.
    strategy : str
        Calibration strategy.

    Returns
    -------
    dict
        {participant_name: {"uncalibrated_mae", "calibrated_mae", "improvement", ...}}
    """
    from tones.models.train import compute_metrics

    results = {}

    for name, data in data_by_participant.items():
        X = data["features"]
        y = data["glucose"]
        timestamps = data.get("timestamps", list(range(len(y))))

        if len(X) < n_calibration + 5:
            continue

        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Sort chronologically
        order = np.argsort(timestamps)
        X_sorted = X[order]
        y_sorted = y[order]

        # Split into calibration and evaluation
        X_cal = X_sorted[:n_calibration]
        y_cal = y_sorted[:n_calibration]
        X_eval = X_sorted[n_calibration:]
        y_eval = y_sorted[n_calibration:]

        X_cal_scaled = scaler.transform(X_cal)
        X_eval_scaled = scaler.transform(X_eval)

        # Uncalibrated predictions
        uncal_preds = population_model.predict(X_eval_scaled)
        uncal_metrics = compute_metrics(y_eval, uncal_preds)

        # Calibrate
        calibrator = FewShotCalibrator(
            min_samples=3, max_samples=30, strategy=strategy,
        )

        # Add calibration samples
        for i in range(n_calibration):
            pred = population_model.predict(X_cal_scaled[i:i + 1])[0]
            calibrator.add_sample(X_cal[i], pred, y_cal[i])

        # Calibrated predictions
        cal_preds = []
        for i in range(len(X_eval)):
            pred = population_model.predict(X_eval_scaled[i:i + 1])[0]
            cal_pred, conf = calibrator.calibrate(pred, X_eval[i])
            cal_preds.append(cal_pred)
        cal_preds = np.array(cal_preds)

        cal_metrics = compute_metrics(y_eval, cal_preds)

        results[name] = {
            "uncalibrated_mae": uncal_metrics["mae"],
            "calibrated_mae": cal_metrics["mae"],
            "improvement": uncal_metrics["mae"] - cal_metrics["mae"],
            "uncalibrated_r": uncal_metrics["r"],
            "calibrated_r": cal_metrics["r"],
            "n_calibration": n_calibration,
            "n_evaluation": len(y_eval),
            "strategy": strategy,
            "diagnostics": calibrator.get_diagnostics(),
        }

        logger.info(
            "  %s: Uncal MAE=%.2f → Cal MAE=%.2f (improvement=%.2f)",
            name, uncal_metrics["mae"], cal_metrics["mae"],
            uncal_metrics["mae"] - cal_metrics["mae"],
        )

    return results
