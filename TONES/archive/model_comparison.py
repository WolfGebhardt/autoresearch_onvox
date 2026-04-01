"""
Model Comparison for Voice-Based Glucose Estimation
====================================================
Compares different modeling approaches:
1. Per-person models (personalized)
2. Population model with person embeddings
3. Transfer learning approach
4. Different ML algorithms
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    LeaveOneOut, KFold, cross_val_predict,
    GridSearchCV, LeaveOneGroupOut
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# For mixed-effects models (optional)
try:
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ModelComparison:
    """Compare different modeling approaches for glucose prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = self._get_models()

    def _get_models(self) -> Dict:
        """Define models to compare."""
        return {
            # Linear models (good for small datasets)
            'ridge': Ridge(alpha=1.0),
            'ridge_cv': Ridge(alpha=0.1),  # Lower regularization
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
            'lasso': Lasso(alpha=0.1, random_state=self.random_state),
            'bayesian_ridge': BayesianRidge(),

            # Tree-based (can capture non-linear relationships)
            'rf_small': RandomForestRegressor(n_estimators=30, max_depth=3,
                                               random_state=self.random_state),
            'rf_medium': RandomForestRegressor(n_estimators=50, max_depth=5,
                                                random_state=self.random_state),
            'gbm': GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                              learning_rate=0.1, random_state=self.random_state),
            'extra_trees': ExtraTreesRegressor(n_estimators=30, max_depth=4,
                                                random_state=self.random_state),

            # Instance-based
            'knn_3': KNeighborsRegressor(n_neighbors=3, weights='distance'),
            'knn_5': KNeighborsRegressor(n_neighbors=5, weights='distance'),

            # SVM (good with small datasets)
            'svr_rbf': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'svr_linear': SVR(kernel='linear', C=1.0),

            # Neural network (may overfit on small data)
            'mlp_small': MLPRegressor(hidden_layer_sizes=(32,), max_iter=500,
                                       random_state=self.random_state, early_stopping=True),
        }

    def evaluate_models(self, X: np.ndarray, y: np.ndarray,
                        cv_method: str = 'loo') -> pd.DataFrame:
        """
        Evaluate all models using specified cross-validation.

        Args:
            X: Feature matrix
            y: Target values (glucose)
            cv_method: 'loo' for leave-one-out, 'kfold' for 5-fold

        Returns:
            DataFrame with model performance metrics
        """
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X)

        # Cross-validation setup
        if cv_method == 'loo':
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=self.random_state)

        results = []

        for name, model in self.models.items():
            try:
                # Predict using cross-validation
                y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

                # Calculate metrics
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)

                # Correlation
                pearson_r, pearson_p = pearsonr(y, y_pred)

                # Clarke Error Grid zones A+B
                ceg_ab = self._clarke_ab_percentage(y, y_pred)

                results.append({
                    'model': name,
                    'mae_mgdl': mae,
                    'rmse_mgdl': rmse,
                    'r2': r2,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'clarke_ab_pct': ceg_ab,
                    'n_samples': len(y)
                })

            except Exception as e:
                print(f"Error with {name}: {e}")

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('mae_mgdl')

        return results_df

    def _clarke_ab_percentage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate percentage of predictions in Clarke Error Grid zones A+B."""
        n = len(y_true)
        ab_count = 0

        for ref, pred in zip(y_true, y_pred):
            # Zone A
            if (ref <= 70 and pred <= 70) or \
               (ref >= 70 and abs(pred - ref) <= 0.2 * ref):
                ab_count += 1
            # Zone B (simplified)
            elif abs(pred - ref) <= 0.4 * ref:
                ab_count += 1

        return ab_count / n * 100


class PersonalizedModelTrainer:
    """Train personalized models for each participant."""

    def __init__(self, base_model: str = 'ridge', random_state: int = 42):
        self.base_model = base_model
        self.random_state = random_state
        self.models = {}
        self.scalers = {}

    def fit_person(self, person_id: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit a personalized model for one person.
        Returns cross-validation results.
        """
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        model = self._get_model()

        # Leave-one-out CV for evaluation
        loo = LeaveOneOut()
        y_pred = cross_val_predict(model, X_scaled, y, cv=loo)

        # Fit final model
        model.fit(X_scaled, y)

        self.models[person_id] = model
        self.scalers[person_id] = scaler

        return {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'pearson_r': pearsonr(y, y_pred)[0],
            'y_true': y,
            'y_pred': y_pred
        }

    def predict(self, person_id: str, X: np.ndarray) -> np.ndarray:
        """Predict glucose for a person."""
        if person_id not in self.models:
            raise ValueError(f"No model for person: {person_id}")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scalers[person_id].transform(X)
        return self.models[person_id].predict(X_scaled)

    def _get_model(self):
        models = {
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'gbm': GradientBoostingRegressor(n_estimators=50, max_depth=3),
            'svr': SVR(kernel='rbf', C=1.0),
            'rf': RandomForestRegressor(n_estimators=50, max_depth=5)
        }
        return models.get(self.base_model, Ridge(alpha=1.0))


class TransferLearningApproach:
    """
    Transfer learning: pre-train on all participants, fine-tune per person.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.population_model = None
        self.population_scaler = None
        self.personalized_models = {}

    def fit_population(self, X: np.ndarray, y: np.ndarray,
                       person_ids: np.ndarray) -> Dict:
        """
        Fit population-level model (learns general voice-glucose patterns).
        """
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.population_scaler = RobustScaler()
        X_scaled = self.population_scaler.fit_transform(X)

        # Leave-one-person-out cross-validation
        logo = LeaveOneGroupOut()
        y_pred = cross_val_predict(
            GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=self.random_state),
            X_scaled, y, cv=logo, groups=person_ids
        )

        # Fit final population model
        self.population_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, random_state=self.random_state
        )
        self.population_model.fit(X_scaled, y)

        return {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'pearson_r': pearsonr(y, y_pred)[0]
        }

    def fine_tune_person(self, person_id: str, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fine-tune for a specific person using population model predictions as feature.
        """
        if self.population_model is None:
            raise ValueError("Must fit population model first")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.population_scaler.transform(X)

        # Get population model predictions
        pop_pred = self.population_model.predict(X_scaled).reshape(-1, 1)

        # Augment features with population prediction
        X_augmented = np.hstack([X_scaled, pop_pred])

        # Fit correction model
        correction_model = Ridge(alpha=0.5)

        loo = LeaveOneOut()
        y_pred = cross_val_predict(correction_model, X_augmented, y, cv=loo)

        correction_model.fit(X_augmented, y)
        self.personalized_models[person_id] = correction_model

        return {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'pearson_r': pearsonr(y, y_pred)[0]
        }


class FeatureSelector:
    """Select most relevant features for glucose prediction."""

    @staticmethod
    def correlation_selection(X: np.ndarray, y: np.ndarray,
                              feature_names: List[str],
                              top_k: int = 20) -> Tuple[np.ndarray, List[str]]:
        """
        Select features most correlated with glucose.
        """
        correlations = []

        for i, name in enumerate(feature_names):
            # Handle NaN
            mask = ~np.isnan(X[:, i])
            if np.sum(mask) > 5:
                r, p = pearsonr(X[mask, i], y[mask])
                correlations.append((i, name, abs(r), p))
            else:
                correlations.append((i, name, 0, 1))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[2], reverse=True)

        # Select top k
        selected_idx = [c[0] for c in correlations[:top_k]]
        selected_names = [c[1] for c in correlations[:top_k]]

        print("\nTop correlated features with glucose:")
        for i, name, r, p in correlations[:10]:
            print(f"  {name}: r={r:.3f}, p={p:.4f}")

        return X[:, selected_idx], selected_names

    @staticmethod
    def variance_threshold_selection(X: np.ndarray, feature_names: List[str],
                                     threshold: float = 0.01) -> Tuple[np.ndarray, List[str]]:
        """
        Remove low-variance features.
        """
        from sklearn.feature_selection import VarianceThreshold

        X_clean = np.nan_to_num(X, nan=0.0)
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X_clean)

        mask = selector.get_support()
        selected_names = [n for n, m in zip(feature_names, mask) if m]

        print(f"\nFeatures after variance threshold: {len(selected_names)}/{len(feature_names)}")

        return X_selected, selected_names


def run_model_comparison_pipeline(df: pd.DataFrame,
                                   feature_prefix: str = 'librosa_') -> Dict:
    """
    Run comprehensive model comparison on a dataset.

    Args:
        df: DataFrame with features and glucose values
        feature_prefix: Prefix of feature columns to use

    Returns:
        Dictionary with comparison results
    """
    # Get features
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]

    if len(feature_cols) == 0:
        print(f"No features with prefix '{feature_prefix}'")
        return {}

    X = df[feature_cols].values
    y = df['glucose_mgdl'].values

    print(f"\nDataset: {len(y)} samples, {len(feature_cols)} features")
    print(f"Glucose range: {y.min():.1f} - {y.max():.1f} mg/dL")
    print(f"Glucose mean: {y.mean():.1f} mg/dL, std: {y.std():.1f} mg/dL")

    # Feature selection
    selector = FeatureSelector()
    X_selected, selected_features = selector.correlation_selection(
        X, y, feature_cols, top_k=min(30, len(feature_cols))
    )

    # Model comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Leave-One-Out CV)")
    print("=" * 60)

    comparator = ModelComparison()
    results = comparator.evaluate_models(X_selected, y, cv_method='loo')

    print("\nResults sorted by MAE:")
    print(results[['model', 'mae_mgdl', 'rmse_mgdl', 'pearson_r', 'clarke_ab_pct']].to_string(index=False))

    return {
        'comparison_results': results,
        'selected_features': selected_features,
        'best_model': results.iloc[0]['model'],
        'best_mae': results.iloc[0]['mae_mgdl']
    }


# Example usage
if __name__ == "__main__":
    print("Model Comparison Module")
    print("Use run_model_comparison_pipeline(df) with your dataset")
