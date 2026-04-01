"""
Hierarchical Bayesian Personalization Model
=============================================
The centerpiece innovation: a principled approach that sits between
"fully personalized" (small N, high variance) and "population"
(no personalization, high bias).

Architecture:
  - Population-level priors learned from ALL participants
  - Per-participant coefficients shrunk toward population
  - Participants with little data → regularized to population (cold start)
  - Participants with lots of data → free to deviate (personalization)
  - Built-in uncertainty quantification for each prediction

Implementation uses PyMC for MCMC inference, with a scikit-learn-compatible
fallback using empirical Bayes (mixed-effects linear model) when PyMC is
not available.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

# Try importing PyMC
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    logger.info("PyMC not installed — using empirical Bayes fallback. pip install pymc arviz")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r, p_val = stats.pearsonr(y_true, y_pred)
    else:
        r, p_val = 0.0, 1.0
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2), "r": float(r), "p_value": float(p_val)}


class HierarchicalBayesianModel:
    """
    Hierarchical Bayesian regression for personalized glucose estimation.

    Uses PyMC when available, otherwise falls back to an empirical Bayes
    approach using per-participant Ridge regression with a shared regularizer.

    Parameters
    ----------
    n_features : int
        Number of input features (after any dimensionality reduction).
    n_samples_mcmc : int
        Number of MCMC samples (PyMC only).
    n_chains : int
        Number of MCMC chains (PyMC only).
    target_accept : float
        NUTS target acceptance rate (PyMC only).
    """

    def __init__(
        self,
        n_features: int = 20,
        n_samples_mcmc: int = 1000,
        n_chains: int = 2,
        target_accept: float = 0.9,
    ):
        self.n_features = n_features
        self.n_samples_mcmc = n_samples_mcmc
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.trace = None
        self.scaler = StandardScaler()
        self.participant_map = {}
        self._fallback_models = {}
        self._population_model = None

    def fit_pymc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        participant_ids: np.ndarray,
    ):
        """
        Fit the hierarchical model using PyMC MCMC.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_total_samples, n_features).
        y : np.ndarray
            Glucose values (mg/dL).
        participant_ids : np.ndarray
            Integer participant indices.
        """
        if not HAS_PYMC:
            raise ImportError("PyMC required for MCMC. Install: pip install pymc arviz")

        X_scaled = self.scaler.fit_transform(X)
        n_participants = len(np.unique(participant_ids))
        n_features = X_scaled.shape[1]

        with pm.Model() as model:
            # Population-level priors
            mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, shape=n_features)
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, shape=n_features)

            # Per-participant coefficients (shrunk toward population)
            beta = pm.Normal(
                "beta", mu=mu_beta, sigma=sigma_beta,
                shape=(n_participants, n_features),
            )

            # Per-participant intercept
            mu_alpha = pm.Normal("mu_alpha", mu=np.mean(y), sigma=20)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=20)
            alpha = pm.Normal(
                "alpha", mu=mu_alpha, sigma=sigma_alpha,
                shape=n_participants,
            )

            # Observation noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=10)

            # Linear prediction
            mu_pred = alpha[participant_ids] + pm.math.sum(
                X_scaled * beta[participant_ids], axis=1
            )

            # Likelihood
            pm.Normal("y_obs", mu=mu_pred, sigma=sigma_obs, observed=y)

            # Sample
            self.trace = pm.sample(
                self.n_samples_mcmc,
                chains=self.n_chains,
                target_accept=self.target_accept,
                return_inferencedata=True,
                progressbar=True,
            )

        logger.info("Hierarchical Bayesian model fitted with PyMC (%d samples)", self.n_samples_mcmc)

    def predict_pymc(
        self,
        X: np.ndarray,
        participant_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using posterior samples.

        Returns
        -------
        tuple of (mean_predictions, std_predictions)
        """
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit_pymc first.")

        X_scaled = self.scaler.transform(X)
        post = self.trace.posterior

        alpha_samples = post["alpha"].values.reshape(-1, post["alpha"].shape[-1])
        beta_samples = post["beta"].values.reshape(-1, *post["beta"].shape[-2:])

        n_posterior = alpha_samples.shape[0]
        predictions = np.zeros((n_posterior, len(X_scaled)))

        for s in range(n_posterior):
            predictions[s] = alpha_samples[s, participant_id] + X_scaled @ beta_samples[s, participant_id]

        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        return mean_pred, std_pred

    def fit_empirical_bayes(
        self,
        data_by_participant: Dict[str, Dict],
        alpha_population: float = 10.0,
        alpha_personal: float = 1.0,
    ):
        """
        Empirical Bayes fallback: train a population model as the prior,
        then fine-tune per participant with stronger regularization.

        This mimics the hierarchical structure without MCMC:
        - Population Ridge provides the prior (shared coefficients)
        - Per-participant Ridge with warm start provides the posterior
        - Shrinkage strength depends inversely on sample size

        Parameters
        ----------
        data_by_participant : dict
            {name: {"features": np.ndarray, "glucose": np.ndarray}}
        alpha_population : float
            Regularization for population model.
        alpha_personal : float
            Base regularization for personal models (scaled by 1/n_samples).
        """
        # Combine all data for population model
        all_X = []
        all_y = []
        for name, data in data_by_participant.items():
            all_X.append(data["features"])
            all_y.append(data["glucose"])

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)

        X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)

        # Fit scaler on all data
        X_scaled = self.scaler.fit_transform(X_all)

        # Population model (the prior)
        self._population_model = BayesianRidge()
        self._population_model.fit(X_scaled, y_all)
        logger.info("Population prior fitted on %d samples", len(y_all))

        # Per-participant models (the posteriors)
        # Regularization inversely proportional to sample size
        self._fallback_models = {}
        offset = 0
        for name, data in data_by_participant.items():
            n = len(data["glucose"])
            X_p = self.scaler.transform(
                np.nan_to_num(data["features"], nan=0, posinf=0, neginf=0)
            )
            y_p = data["glucose"]

            # Shrinkage: small datasets → strong regularization toward population
            shrinkage_alpha = alpha_personal * max(1.0, 100.0 / n)

            personal_model = Ridge(alpha=shrinkage_alpha)

            # Initialize with population coefficients (warm start analogy)
            # by adding population predictions as a feature
            pop_pred = self._population_model.predict(X_p).reshape(-1, 1)
            X_augmented = np.hstack([X_p, pop_pred])

            personal_model.fit(X_augmented, y_p)
            self._fallback_models[name] = personal_model

            logger.debug(
                "%s: personal model fitted (n=%d, alpha=%.1f)",
                name, n, shrinkage_alpha,
            )

        self.participant_map = {name: i for i, name in enumerate(data_by_participant.keys())}

    def predict_empirical_bayes(
        self,
        X: np.ndarray,
        participant_name: str,
    ) -> np.ndarray:
        """Predict using empirical Bayes model."""
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)
        pop_pred = self._population_model.predict(X_scaled).reshape(-1, 1)
        X_augmented = np.hstack([X_scaled, pop_pred])

        if participant_name in self._fallback_models:
            return self._fallback_models[participant_name].predict(X_augmented)
        else:
            # New participant: use population model only
            return pop_pred.ravel()


def train_hierarchical_bayesian(
    data_by_participant: Dict[str, Dict],
    use_pymc: bool = True,
    min_samples_per_person: int = 10,
) -> Dict[str, Dict]:
    """
    Train and evaluate the hierarchical Bayesian model using LOO-per-participant.

    For each participant, train on ALL other participants + the current
    participant's training data, then predict the held-out samples.

    Parameters
    ----------
    data_by_participant : dict
        {name: {"features": np.ndarray, "glucose": np.ndarray, "timestamps": list}}
    use_pymc : bool
        Use full PyMC MCMC if available.
    min_samples_per_person : int
        Skip participants with fewer samples.

    Returns
    -------
    dict
        {participant_name: {"mae", "rmse", "r", "predictions", "actual", ...}}
    """
    # For each participant, use leave-one-out on that participant
    # while using all other participants as additional training data
    results = {}

    for held_out_name, held_out_data in data_by_participant.items():
        if len(held_out_data["glucose"]) < min_samples_per_person:
            continue

        X_ho = held_out_data["features"]
        y_ho = held_out_data["glucose"]

        # Build training set: all other participants + part of held-out
        # Using empirical Bayes approach
        model = HierarchicalBayesianModel()

        # For evaluation, use chronological split on the held-out participant
        from tones.evaluation.temporal_cv import chronological_split

        timestamps_ho = np.array(held_out_data.get("timestamps", list(range(len(y_ho)))))
        X_train_ho, X_test_ho, y_train_ho, y_test_ho = chronological_split(
            X_ho, y_ho, timestamps_ho, train_fraction=0.7,
        )

        # Build training data for empirical Bayes
        train_data = {}
        for name, data in data_by_participant.items():
            if name == held_out_name:
                # Only include training portion
                train_data[name] = {
                    "features": X_train_ho,
                    "glucose": y_train_ho,
                }
            else:
                train_data[name] = {
                    "features": data["features"],
                    "glucose": data["glucose"],
                }

        try:
            model.fit_empirical_bayes(train_data)
            predictions = model.predict_empirical_bayes(X_test_ho, held_out_name)

            metrics = _compute_metrics(y_test_ho, predictions)
            metrics["predictions"] = predictions
            metrics["actual"] = y_test_ho
            metrics["n_train"] = len(y_train_ho)
            metrics["n_test"] = len(y_test_ho)
            metrics["model_type"] = "hierarchical_bayesian"

            results[held_out_name] = metrics
            logger.info(
                "  %s (hierarchical): MAE=%.2f, r=%.3f (train=%d, test=%d)",
                held_out_name, metrics["mae"], metrics["r"],
                len(y_train_ho), len(y_test_ho),
            )
        except Exception as e:
            logger.warning("  %s hierarchical model failed: %s", held_out_name, e)

    return results
