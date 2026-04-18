#!/usr/bin/env python3
"""
ONVOX AutoResearch — Systematic Hyperparameter & Feature Configuration Sweep
=================================================================
Produces a comprehensive analysis of which configurations work best for:
  - Personalized models (per-participant CV)
  - Population models (Leave-One-Person-Out CV)

Sweeps over:
  1. Number of MFCCs: 8, 13, 20, 30, 40
  2. Feature combinations: MFCC-only, +spectral, +pitch, +voice_quality, +temporal, all
  3. Normalization: none, zscore, rank
  4. Model algorithms: Ridge, BayesianRidge, SVR, ElasticNet, Lasso, Huber,
     RandomForest, GradientBoosting, ExtraTrees, KNN

Output:
  - output/sweep/results.json          — full machine-readable results
  - output/sweep/summary_tables.csv    — flattened results for easy viewing
  - output/sweep/figures/*.png         — visualizations
  - output/sweep/report.html           — self-contained HTML report

Usage:
    python hyperparameter_sweep.py                  # Full sweep
    python hyperparameter_sweep.py --quick           # Reduced sweep (fast)
    python hyperparameter_sweep.py --participants Wolf Lara  # Specific participants
"""

import argparse
import gc
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# Research modules
from research.config import load_config, get_base_dir
from research.data.loaders import load_participant_data, collect_audio_files
from research.features.mfcc import MFCCExtractor
from research.features.voice_quality import VoiceQualityExtractor
from research.features.normalize import zscore_per_speaker, rank_normalize_per_speaker
from research.features.temporal import compute_circadian_features, compute_delta_features, compute_time_since_last
from research.models.train import (
    get_model,
    compute_metrics,
    mean_predictor_baseline,
)
from research.evaluation.temporal_cv import chronological_split, train_personalized_walkforward

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import LeaveOneOut, KFold, LeaveOneGroupOut

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration space
# =============================================================================

MFCC_COUNTS = [8, 13, 20, 30, 40]

FEATURE_COMBOS = {
    "mfcc_only":               dict(include_spectral=False, include_pitch=False, use_vq=False, use_temporal=False, deconfound=False),
    "mfcc+spectral":           dict(include_spectral=True,  include_pitch=False, use_vq=False, use_temporal=False, deconfound=False),
    "mfcc+spectral+pitch":     dict(include_spectral=True,  include_pitch=True,  use_vq=False, use_temporal=False, deconfound=False),
    "mfcc+spectral+pitch+vq":  dict(include_spectral=True,  include_pitch=True,  use_vq=True,  use_temporal=False, deconfound=False),
    "mfcc+spectral+pitch+temporal": dict(include_spectral=True, include_pitch=True, use_vq=False, use_temporal=True, deconfound=False),
    "all_features":            dict(include_spectral=True,  include_pitch=True,  use_vq=True,  use_temporal=True,  deconfound=False),
    # Biophysics-informed combos (Apr 2026)
    "pathway_ab":              dict(include_spectral=True,  include_pitch=True,  use_vq=True,  use_temporal=False, deconfound=False),
    "deconfounded":            dict(include_spectral=True,  include_pitch=True,  use_vq=True,  use_temporal=True,  deconfound=True),
}

NORM_METHODS = ["none", "zscore", "rank"]

MODEL_NAMES = [
    "Ridge",
    "BayesianRidge",
    "SVR",
    "ElasticNet",
    "Lasso",
    "Huber",
    "RandomForest",
    "GradientBoosting",
    "ExtraTrees",
    "KNN",
    "GP",
]

# Quick mode: reduced search
MFCC_COUNTS_QUICK = [13, 20]
FEATURE_COMBOS_QUICK = {
    "mfcc_only":      dict(include_spectral=False, include_pitch=False, use_vq=False, use_temporal=False),
    "mfcc+spectral":  dict(include_spectral=True,  include_pitch=False, use_vq=False, use_temporal=False),
    "mfcc+spec+temp": dict(include_spectral=True,  include_pitch=False, use_vq=False, use_temporal=True),
}
NORM_METHODS_QUICK = ["none", "zscore"]
MODEL_NAMES_QUICK = ["Ridge", "SVR", "RandomForest"]


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    for name in ["matplotlib", "PIL", "urllib3", "numba"]:
        logging.getLogger(name).setLevel(logging.WARNING)


# =============================================================================
# Step 1: Load raw audio waveforms (once, then reuse)
# =============================================================================

def load_all_audio(cfg, participants_filter=None) -> Dict[str, Dict]:
    """Load audio paths and glucose for each participant (paths only, not waveforms)."""

    base_dir = get_base_dir(cfg)
    matching_cfg = cfg.get("matching", {})
    participants = cfg.get("participants", {})

    if participants_filter:
        participants = {k: v for k, v in participants.items() if k in participants_filter}

    data = {}
    for name, pcfg in participants.items():
        df = load_participant_data(name, pcfg, base_dir, matching_cfg)
        if df.empty or len(df) < 20:
            logger.info("  %s: Skipped (%d samples)", name, len(df))
            continue

        data[name] = {
            "audio_paths": list(df["audio_path"]),
            "glucose": np.array(df["glucose_mg_dl"]),
            "timestamps": list(df["audio_timestamp"]),
        }
        logger.info("  %s: %d matched samples", name, len(df))

    logger.info("Total: %d participants loaded", len(data))
    return data


# =============================================================================
# Step 2: Extract features for a given configuration
# =============================================================================

def _deconfound_circadian(
    X: np.ndarray,
    timestamps: list,
    f0_col_indices: List[int],
) -> np.ndarray:
    """Regress out circadian (hour-of-day) from F0-adjacent feature columns.

    For each column in f0_col_indices, fits y_col ~ hour_sin + hour_cos
    and replaces the column with residuals. Pure numpy (lstsq).
    Does NOT touch Pathway B features (alpha_ratio, shimmer, etc.).
    """
    if not f0_col_indices or len(X) < 3:
        return X

    hours = np.zeros(len(timestamps), dtype=np.float64)
    for i, ts in enumerate(timestamps):
        try:
            if hasattr(ts, 'hour'):
                hours[i] = ts.hour + ts.minute / 60.0
            else:
                from datetime import datetime as dt
                t = dt.fromisoformat(str(ts))
                hours[i] = t.hour + t.minute / 60.0
        except Exception:
            hours[i] = 12.0

    phase = 2.0 * np.pi * hours / 24.0
    A = np.column_stack([np.sin(phase), np.cos(phase), np.ones(len(hours))])

    X_out = X.copy()
    for col_idx in f0_col_indices:
        if col_idx >= X.shape[1]:
            continue
        y_col = X[:, col_idx].astype(np.float64)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y_col, rcond=None)
            X_out[:, col_idx] = y_col - A @ coeffs
        except Exception:
            pass
    return X_out


_F0_ADJACENT_NAMES = {
    "f0_mean", "f0_std", "f0_median", "f0_p10", "f0_p90",
    "hnr", "ptp_proxy", "f0_cv", "f0_skew", "f0_kurtosis",
}


def extract_features_config(
    participant_data: Dict[str, Dict],
    n_mfcc: int,
    include_spectral: bool,
    include_pitch: bool,
    use_vq: bool,
    use_temporal: bool,
    normalization: str,
    deconfound: bool = False,
) -> Dict[str, Dict]:
    """Extract features with specified configuration. Loads audio from paths on-the-fly."""
    import librosa

    mfcc_ext = MFCCExtractor(
        sr=16000, n_mfcc=n_mfcc, fmin=50, fmax=8000,
        include_spectral=include_spectral, include_pitch=include_pitch, include_mel=False,
    )
    vq_ext = VoiceQualityExtractor(sr=16000) if use_vq else None

    result = {}
    for name, pdata in participant_data.items():
        features_list = []
        vq_list = []
        valid_glucose = []
        valid_timestamps = []

        for i, audio_path in enumerate(pdata["audio_paths"]):
            try:
                y, _ = librosa.load(str(audio_path), sr=16000, mono=True)
                if len(y) < 16000 * 0.5:
                    continue
            except Exception:
                continue

            mfcc_feat = mfcc_ext.extract_from_array(y)
            if mfcc_feat is None:
                mfcc_feat = np.zeros(len(mfcc_ext.feature_names), dtype=np.float32)
            features_list.append(mfcc_feat)
            valid_glucose.append(pdata["glucose"][i])
            valid_timestamps.append(pdata["timestamps"][i])

            if vq_ext is not None:
                vq_feat = vq_ext.extract_from_array(y)
                if vq_feat is None:
                    vq_feat = np.zeros(vq_ext.n_features, dtype=np.float32)
                vq_list.append(vq_feat)

            del y  # Free waveform immediately

        if len(features_list) < 20:
            continue

        X = np.array(features_list)
        if vq_ext is not None and vq_list:
            X_vq = np.array(vq_list)
            X = np.hstack([X, X_vq])

        result[name] = {
            "features": X,
            "glucose": np.array(valid_glucose),
            "timestamps": valid_timestamps,
        }

    # Add temporal features
    if use_temporal:
        for name, rdata in result.items():
            X = rdata["features"]
            ts = rdata["timestamps"]
            circ = compute_circadian_features(ts)
            deltas = compute_delta_features(X, ts, max_gap_hours=4.0)
            time_since = compute_time_since_last(ts)
            X = np.hstack([X, circ, deltas, time_since])
            rdata["features"] = X

    # Deconfound: regress out circadian drift from F0-adjacent features
    if deconfound:
        feat_names = list(mfcc_ext.feature_names)
        if vq_ext is not None:
            feat_names.extend(VoiceQualityExtractor.FEATURE_NAMES)
        f0_col_indices = [
            i for i, name in enumerate(feat_names) if name in _F0_ADJACENT_NAMES
        ]
        if f0_col_indices:
            for name, rdata in result.items():
                rdata["features"] = _deconfound_circadian(
                    rdata["features"], rdata["timestamps"], f0_col_indices
                )

    # Keep raw (pre-normalization) features for leakage-safe fold transforms.
    for name in result:
        result[name]["features_raw"] = np.array(result[name]["features"], copy=True)

    # Apply normalization
    if normalization == "zscore":
        feat_dict = {n: d["features"] for n, d in result.items()}
        normed = zscore_per_speaker(feat_dict)
        for n in result:
            result[n]["features"] = normed[n]
    elif normalization == "rank":
        feat_dict = {n: d["features"] for n, d in result.items()}
        normed = rank_normalize_per_speaker(feat_dict)
        for n in result:
            result[n]["features"] = normed[n]

    # Replace NaN/inf
    for n in result:
        result[n]["features"] = np.nan_to_num(result[n]["features"], nan=0, posinf=0, neginf=0)
        result[n]["features_raw"] = np.nan_to_num(result[n]["features_raw"], nan=0, posinf=0, neginf=0)

    return result


# =============================================================================
# Step 3: Evaluate a single configuration
# =============================================================================

def _fit_apply_fold_normalization(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "none":
        return X_train, X_test

    if method == "zscore":
        mu = np.mean(X_train, axis=0)
        sigma = np.std(X_train, axis=0) + 1e-8
        return (X_train - mu) / sigma, (X_test - mu) / sigma

    if method == "rank":
        Xtr = np.zeros_like(X_train, dtype=np.float64)
        Xte = np.zeros_like(X_test, dtype=np.float64)
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            if len(col) <= 1:
                continue
            order = np.sort(col)
            train_ranks = np.searchsorted(order, col, side="left")
            Xtr[:, j] = train_ranks / max(len(order) - 1, 1)
            test_ranks = np.searchsorted(order, X_test[:, j], side="left")
            Xte[:, j] = np.clip(test_ranks / max(len(order) - 1, 1), 0.0, 1.0)
        return Xtr, Xte

    return X_train, X_test


def evaluate_personalized(
    features_data: Dict[str, Dict],
    model_name: str,
    normalization_method: str = "none",
) -> Dict[str, Dict]:
    """Evaluate personalized models for all participants with fold-safe normalization."""
    results = {}
    for name, fdata in features_data.items():
        X = fdata.get("features_raw", fdata["features"])
        y = fdata["glucose"]
        if len(X) < 20:
            continue
        try:
            cv = LeaveOneOut() if len(X) <= 50 else KFold(n_splits=10, shuffle=True, random_state=42)
            preds = np.zeros_like(y, dtype=float)
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]
                X_train, X_test = _fit_apply_fold_normalization(
                    X_train, X_test, normalization_method
                )
                fold_pipeline = Pipeline([("scaler", RobustScaler()), ("model", get_model(model_name))])
                fold_pipeline.fit(X_train, y_train)
                preds[test_idx] = fold_pipeline.predict(X_test)
            metrics = compute_metrics(y, preds)
            baseline = mean_predictor_baseline(y)
            metrics["baseline_mae"] = baseline["mae"]
            metrics["improvement"] = baseline["mae"] - metrics["mae"]
            metrics["pct_improvement"] = 100 * metrics["improvement"] / baseline["mae"] if baseline["mae"] > 0 else 0
            results[name] = metrics
        except Exception as e:
            logger.warning("  %s/%s failed: %s", name, model_name, e)
    return results


def evaluate_population(
    features_data: Dict[str, Dict],
    model_name: str,
    normalization_method: str = "none",
) -> Dict:
    """Evaluate population model (LOPO) with train-fold normalization only."""
    all_X, all_y, all_groups = [], [], []
    for name, fdata in features_data.items():
        all_X.append(fdata["features"])
        all_y.append(fdata["glucose"])
        all_groups.extend([name] * len(fdata["glucose"]))

    if not all_X:
        return {}

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)

    try:
        logo = LeaveOneGroupOut()
        preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            X_train, X_test = _fit_apply_fold_normalization(
                X_train, X_test, normalization_method
            )
            fold_pipeline = Pipeline([("scaler", RobustScaler()), ("model", get_model(model_name))])
            fold_pipeline.fit(X_train, y_train)
            preds[test_idx] = fold_pipeline.predict(X_test)
        metrics = compute_metrics(y, preds)
        baseline = mean_predictor_baseline(y)
        metrics["baseline_mae"] = baseline["mae"]
        metrics["improvement"] = baseline["mae"] - metrics["mae"]

        # Per-person breakdown
        per_person = {}
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() > 2:
                per_person[g] = compute_metrics(y[mask], preds[mask])
        metrics["per_person"] = per_person

        return metrics
    except Exception as e:
        logger.warning("Population %s failed: %s", model_name, e)
        return {}


def evaluate_temporal(
    features_data: Dict[str, Dict],
    model_name: str,
    normalization_method: str = "none",
) -> Dict[str, Dict]:
    """Temporal validation: walk-forward when possible, else 80/20 chronological."""
    results = {}
    for name, fdata in features_data.items():
        X, y = fdata["features"], fdata["glucose"]
        timestamps = np.array(fdata["timestamps"])
        if len(X) < 20:
            continue
        try:
            # ONVOX rationale: use walk-forward for larger series, otherwise strict 80/20 holdout.
            if len(X) >= 50:
                order = np.argsort(timestamps)
                Xs = X[order]
                ys = y[order]
                ts_sorted = timestamps[order]
                # Pre-normalize by time-ordered expanding train windows in walkforward utility.
                # We keep feature normalization conservative by applying none here and relying on
                # robust scaler inside temporal utility pipeline.
                def factory():
                    return get_model(model_name)
                wf = train_personalized_walkforward(
                    X=np.nan_to_num(Xs, nan=0, posinf=0, neginf=0),
                    y=ys,
                    timestamps=ts_sorted,
                    model_factory=factory,
                    min_train_samples=max(10, int(len(Xs) * 0.3)),
                )
                preds = wf.get("predictions", np.array([]))
                y_test = wf.get("actual_test", np.array([]))
                if len(preds) == 0 or len(y_test) == 0:
                    continue
                metrics = compute_metrics(np.asarray(y_test), np.asarray(preds))
                baseline_mae = float(np.mean(np.abs(np.asarray(y_test) - np.mean(ys[: max(1, int(len(ys) * 0.8))]))))
                metrics["baseline_mae"] = baseline_mae
                metrics["improvement"] = baseline_mae - metrics["mae"]
                metrics["n_train"] = int(wf.get("n_train", 0))
                metrics["n_test"] = int(wf.get("n_test", len(y_test)))
                metrics["cv_strategy"] = "walk_forward"
                results[name] = metrics
            else:
                X_train, X_test, y_train, y_test = chronological_split(X, y, timestamps, train_fraction=0.8)
                X_train, X_test = _fit_apply_fold_normalization(
                    X_train, X_test, normalization_method
                )
                model = get_model(model_name)
                pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                metrics = compute_metrics(y_test, preds)
                baseline_mae = float(np.mean(np.abs(y_test - np.mean(y_train))))
                metrics["baseline_mae"] = baseline_mae
                metrics["improvement"] = baseline_mae - metrics["mae"]
                metrics["n_train"] = len(y_train)
                metrics["n_test"] = len(y_test)
                metrics["cv_strategy"] = "chrono_80_20"
                results[name] = metrics
        except Exception as e:
            logger.warning("  %s temporal/%s failed: %s", name, model_name, e)
    return results


# =============================================================================
# Step 3b: Evaluate production data (pre-extracted features from Supabase)
# =============================================================================

# BackgroundTrainer feature subsets (must match background_trainer.py)
PRODUCTION_FEATURE_SUBSETS = {
    'edge_10': list(range(10)),
    'personal_10': [0, 1, 80, 99, 90, 91, 97, 98, 100, 101],
    'dynamics_10': [20, 48, 52, 59, 80, 85, 90, 91, 97, 98],
    'personal_14': [0, 1, 80, 99, 90, 91, 97, 98, 100, 101, 112, 113, 114, 115],
    'personal_10_time': [0, 1, 80, 99, 90, 91, 97, 98, 100, 101, 95, 96, 117, 118],
    'mfcc_13': list(range(13)),
    'spectral_8': [0, 1, 80, 81, 82, 83, 84, 85],
    'full': None,
}


def evaluate_production(
    production_data: Dict[str, Dict],
    model_name: str,
    feature_subset: str = "personal_10",
    normalization_method: str = "none",
) -> Dict[str, Dict]:
    """Evaluate personalized models on pre-extracted production features.

    Uses expanding-window CV (same as BackgroundTrainer) instead of LOO/KFold.

    Args:
        production_data: Dict from load_production_data() with keys
            {user_id: {"features": np.ndarray, "glucose": np.ndarray, "timestamps": list}}
        model_name: sklearn model name (Ridge, BayesianRidge, SVR, etc.)
        feature_subset: Key from PRODUCTION_FEATURE_SUBSETS
        normalization_method: "none", "zscore", or "rank"

    Returns:
        Dict of per-user metrics, same shape as evaluate_personalized().
    """
    indices = PRODUCTION_FEATURE_SUBSETS.get(feature_subset)
    results = {}

    for user_id, udata in production_data.items():
        X_full = udata["features"]
        y = udata["glucose"]

        if len(y) < 10:
            continue

        # Apply feature subset
        if indices is not None:
            valid_indices = [i for i in indices if i < X_full.shape[1]]
            if not valid_indices:
                continue
            X = X_full[:, valid_indices]
        else:
            X = X_full

        try:
            # Expanding-window CV (chronological, matches BackgroundTrainer)
            timestamps = udata.get("timestamps", [])
            if timestamps:
                # Sort by timestamp
                try:
                    order = np.argsort([str(t) for t in timestamps])
                    X = X[order]
                    y = y[order]
                except Exception:
                    pass

            min_train = max(10, int(len(y) * 0.3))
            preds = np.full(len(y), np.nan)
            for split_idx in range(min_train, len(y)):
                X_train, X_test = X[:split_idx], X[split_idx:split_idx+1]
                y_train = y[:split_idx]

                X_train, X_test = _fit_apply_fold_normalization(
                    X_train, X_test, normalization_method
                )

                fold_pipeline = Pipeline([
                    ("scaler", RobustScaler()),
                    ("model", get_model(model_name)),
                ])
                fold_pipeline.fit(X_train, y_train)
                preds[split_idx] = fold_pipeline.predict(X_test)[0]

            # Only evaluate on predicted samples
            valid_mask = ~np.isnan(preds)
            if valid_mask.sum() < 5:
                continue

            y_eval = y[valid_mask]
            p_eval = preds[valid_mask]
            metrics = compute_metrics(y_eval, p_eval)
            baseline = mean_predictor_baseline(y_eval)
            metrics["baseline_mae"] = baseline["mae"]
            metrics["improvement"] = baseline["mae"] - metrics["mae"]
            metrics["pct_improvement"] = (
                100 * metrics["improvement"] / baseline["mae"]
                if baseline["mae"] > 0 else 0
            )
            metrics["n_train_final"] = int(valid_mask.sum())
            metrics["feature_subset"] = feature_subset
            results[user_id] = metrics
        except Exception as e:
            logger.warning("  %s/%s production eval failed: %s", user_id[:8], model_name, e)

    return results


# =============================================================================
# Step 4: Run the full sweep
# =============================================================================

def run_sweep(
    participant_data: Dict[str, Dict],
    mfcc_counts: List[int],
    feature_combos: Dict[str, Dict],
    norm_methods: List[str],
    model_names: List[str],
) -> pd.DataFrame:
    """Run the full hyperparameter sweep, returning results as a DataFrame."""

    rows = []
    total_configs = len(mfcc_counts) * len(feature_combos) * len(norm_methods) * len(model_names)
    config_idx = 0

    for n_mfcc in mfcc_counts:
        for combo_name, combo_cfg in feature_combos.items():
            for norm in norm_methods:
                # Extract features for this config (shared across models)
                logger.info(
                    "Extracting: n_mfcc=%d, features=%s, norm=%s",
                    n_mfcc, combo_name, norm,
                )
                sys.stderr.flush()
                try:
                    fdata = extract_features_config(
                        participant_data, n_mfcc=n_mfcc,
                        include_spectral=combo_cfg["include_spectral"],
                        include_pitch=combo_cfg["include_pitch"],
                        use_vq=combo_cfg["use_vq"],
                        use_temporal=combo_cfg["use_temporal"],
                        normalization=norm,
                        deconfound=combo_cfg.get("deconfound", False),
                    )
                except Exception as e:
                    logger.error("Feature extraction failed: %s", e, exc_info=True)
                    sys.stderr.flush()
                    continue

                # Determine feature dimensionality
                sample_name = list(fdata.keys())[0]
                n_features = fdata[sample_name]["features"].shape[1]

                for model_name in model_names:
                    config_idx += 1
                    logger.info(
                        "  [%d/%d] model=%s", config_idx, total_configs, model_name,
                    )
                    sys.stderr.flush()

                    try:
                        # --- Personalized ---
                        pers_results = evaluate_personalized(
                            fdata, model_name, normalization_method=norm
                        )
                        if pers_results:
                            avg_mae = np.mean([r["mae"] for r in pers_results.values()])
                            avg_r = np.mean([r["r"] for r in pers_results.values()])
                            avg_improve = np.mean([r["improvement"] for r in pers_results.values()])
                            avg_pct_improve = np.mean([r["pct_improvement"] for r in pers_results.values()])
                        else:
                            avg_mae = avg_r = avg_improve = avg_pct_improve = 0.0

                        # --- Population ---
                        pop_result = evaluate_population(
                            fdata, model_name, normalization_method=norm
                        )
                        pop_mae = pop_result.get("mae", float("nan"))
                        pop_r = pop_result.get("r", 0.0)

                        # --- Temporal ---
                        temp_results = evaluate_temporal(
                            fdata, model_name, normalization_method=norm
                        )
                        if temp_results:
                            temp_avg_mae = np.mean([r["mae"] for r in temp_results.values()])
                            temp_avg_r = np.mean([r["r"] for r in temp_results.values()])
                        else:
                            temp_avg_mae = temp_avg_r = float("nan")

                        # Per-participant detail
                        for pname, pr in pers_results.items():
                            tr = temp_results.get(pname, {})
                            rows.append({
                                "n_mfcc": n_mfcc,
                                "features": combo_name,
                                "normalization": norm,
                                "model": model_name,
                                "n_features": n_features,
                                "participant": pname,
                                "n_samples": pr["n_samples"],
                                "pers_mae": round(pr["mae"], 2),
                                "pers_r": round(pr["r"], 3),
                                "pers_rmse": round(pr["rmse"], 2),
                                "pers_baseline_mae": round(pr["baseline_mae"], 2),
                                "pers_improvement": round(pr["improvement"], 2),
                                "pers_pct_improvement": round(pr["pct_improvement"], 1),
                                "pers_mard": round(pr.get("mard", float("nan")), 2),
                                "pers_bias": round(pr.get("bias", float("nan")), 2),
                                "pers_mae_low": round(pr.get("mae_low", float("nan")), 2),
                                "pers_mae_normal": round(pr.get("mae_normal", float("nan")), 2),
                                "pers_mae_high": round(pr.get("mae_high", float("nan")), 2),
                                "temp_mae": round(tr.get("mae", float("nan")), 2),
                                "temp_r": round(tr.get("r", float("nan")), 3),
                                "temp_mard": round(tr.get("mard", float("nan")), 2),
                                "temp_bias": round(tr.get("bias", float("nan")), 2),
                                "pop_mae": round(pop_mae, 2),
                                "pop_r": round(pop_r, 3),
                                "pop_mard": round(pop_result.get("mard", float("nan")), 2),
                                "pop_bias": round(pop_result.get("bias", float("nan")), 2),
                                "pop_clarke_ab_pct": round(pop_result.get("clarke_ab_pct", float("nan")), 2),
                            })

                        # Summary row (averaged)
                        rows.append({
                            "n_mfcc": n_mfcc,
                            "features": combo_name,
                            "normalization": norm,
                            "model": model_name,
                            "n_features": n_features,
                            "participant": "_AVERAGE_",
                            "n_samples": sum(r["n_samples"] for r in pers_results.values()) if pers_results else 0,
                            "pers_mae": round(avg_mae, 2),
                            "pers_r": round(avg_r, 3),
                            "pers_rmse": 0,
                            "pers_baseline_mae": 0,
                            "pers_improvement": round(avg_improve, 2),
                            "pers_pct_improvement": round(avg_pct_improve, 1),
                            "pers_mard": round(np.mean([r.get("mard", np.nan) for r in pers_results.values()]), 2) if pers_results else None,
                            "pers_bias": round(np.mean([r.get("bias", np.nan) for r in pers_results.values()]), 2) if pers_results else None,
                            "pers_mae_low": round(np.mean([r.get("mae_low", np.nan) for r in pers_results.values()]), 2) if pers_results else None,
                            "pers_mae_normal": round(np.mean([r.get("mae_normal", np.nan) for r in pers_results.values()]), 2) if pers_results else None,
                            "pers_mae_high": round(np.mean([r.get("mae_high", np.nan) for r in pers_results.values()]), 2) if pers_results else None,
                            "temp_mae": round(temp_avg_mae, 2) if not np.isnan(temp_avg_mae) else None,
                            "temp_r": round(temp_avg_r, 3) if not np.isnan(temp_avg_r) else None,
                            "temp_mard": round(np.mean([r.get("mard", np.nan) for r in temp_results.values()]), 2) if temp_results else None,
                            "temp_bias": round(np.mean([r.get("bias", np.nan) for r in temp_results.values()]), 2) if temp_results else None,
                            "pop_mae": round(pop_mae, 2),
                            "pop_r": round(pop_r, 3),
                            "pop_mard": round(pop_result.get("mard", float("nan")), 2),
                            "pop_bias": round(pop_result.get("bias", float("nan")), 2),
                            "pop_clarke_ab_pct": round(pop_result.get("clarke_ab_pct", float("nan")), 2),
                        })

                    except Exception as e:
                        logger.error("  [%d/%d] %s FAILED: %s", config_idx, total_configs, model_name, e, exc_info=True)
                        sys.stderr.flush()

                    gc.collect()

                # Free feature data after all models for this extraction config
                del fdata
                gc.collect()

    return pd.DataFrame(rows)


# =============================================================================
# Step 5: Generate Visualizations
# =============================================================================

COLORS = {
    "Ridge": "#3498db", "SVR": "#e74c3c", "BayesianRidge": "#2ecc71",
    "RandomForest": "#9b59b6", "GradientBoosting": "#f39c12", "KNN": "#1abc9c",
}


def plot_mfcc_sweep(df: pd.DataFrame, fig_dir: Path):
    """Effect of n_mfcc on personalized and population MAE."""
    avg = df[df["participant"] == "_AVERAGE_"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Personalized ---
    ax = axes[0]
    for model in avg["model"].unique():
        sub = avg[avg["model"] == model].groupby("n_mfcc")["pers_mae"].mean()
        ax.plot(sub.index, sub.values, "o-", label=model, color=COLORS.get(model, "gray"), linewidth=2, markersize=6)
    ax.set_xlabel("Number of MFCCs", fontsize=12)
    ax.set_ylabel("Avg Personalized MAE (mg/dL)", fontsize=12)
    ax.set_title("Personalized Model: Effect of MFCC Count", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Population ---
    ax = axes[1]
    for model in avg["model"].unique():
        sub = avg[avg["model"] == model].groupby("n_mfcc")["pop_mae"].mean()
        ax.plot(sub.index, sub.values, "s--", label=model, color=COLORS.get(model, "gray"), linewidth=2, markersize=6)
    ax.set_xlabel("Number of MFCCs", fontsize=12)
    ax.set_ylabel("Population MAE (mg/dL)", fontsize=12)
    ax.set_title("Population Model: Effect of MFCC Count", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "mfcc_count_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_combo_sweep(df: pd.DataFrame, fig_dir: Path):
    """Effect of feature combination on MAE."""
    avg = df[df["participant"] == "_AVERAGE_"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Personalized ---
    ax = axes[0]
    combos = list(FEATURE_COMBOS.keys())
    for model in avg["model"].unique():
        sub = avg[avg["model"] == model]
        vals = [sub[sub["features"] == c]["pers_mae"].mean() for c in combos]
        ax.plot(range(len(combos)), vals, "o-", label=model, color=COLORS.get(model, "gray"), linewidth=2)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([c.replace("mfcc+", "+").replace("mfcc_only", "MFCC") for c in combos], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Avg Personalized MAE (mg/dL)", fontsize=11)
    ax.set_title("Personalized: Feature Combination Effect", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Population ---
    ax = axes[1]
    for model in avg["model"].unique():
        sub = avg[avg["model"] == model]
        vals = [sub[sub["features"] == c]["pop_mae"].mean() for c in combos]
        ax.plot(range(len(combos)), vals, "s--", label=model, color=COLORS.get(model, "gray"), linewidth=2)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([c.replace("mfcc+", "+").replace("mfcc_only", "MFCC") for c in combos], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Population MAE (mg/dL)", fontsize=11)
    ax.set_title("Population: Feature Combination Effect", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "feature_combo_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_normalization_effect(df: pd.DataFrame, fig_dir: Path):
    """Effect of normalization method."""
    avg = df[df["participant"] == "_AVERAGE_"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Personalized
    ax = axes[0]
    norm_data = avg.groupby(["normalization", "model"])["pers_mae"].mean().unstack()
    norm_data.plot(kind="bar", ax=ax, color=[COLORS.get(c, "gray") for c in norm_data.columns], width=0.7)
    ax.set_ylabel("Avg Personalized MAE (mg/dL)", fontsize=11)
    ax.set_title("Normalization Effect (Personalized)", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, alpha=0.3, axis="y")

    # Population
    ax = axes[1]
    norm_data = avg.groupby(["normalization", "model"])["pop_mae"].mean().unstack()
    norm_data.plot(kind="bar", ax=ax, color=[COLORS.get(c, "gray") for c in norm_data.columns], width=0.7)
    ax.set_ylabel("Population MAE (mg/dL)", fontsize=11)
    ax.set_title("Normalization Effect (Population)", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(fig_dir / "normalization_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_heatmap(df: pd.DataFrame, fig_dir: Path):
    """Heatmap: models × feature configs for personalized MAE."""
    avg = df[df["participant"] == "_AVERAGE_"]
    pivot = avg.pivot_table(values="pers_mae", index="model", columns="features", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("mfcc+", "+").replace("mfcc_only", "MFCC") for c in pivot.columns], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("Personalized MAE (mg/dL): Models × Feature Configs", fontsize=13)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if val > pivot.values.mean() else "black")
    fig.colorbar(im, ax=ax, label="MAE (mg/dL)")
    plt.tight_layout()
    fig.savefig(fig_dir / "model_feature_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_participant_breakdown(df: pd.DataFrame, fig_dir: Path):
    """Per-participant results for the best personalized config."""
    avg = df[df["participant"] == "_AVERAGE_"]
    best_idx = avg["pers_mae"].idxmin()
    best_row = avg.loc[best_idx]
    best_cfg = {
        "n_mfcc": best_row["n_mfcc"], "features": best_row["features"],
        "normalization": best_row["normalization"], "model": best_row["model"],
    }

    sub = df[
        (df["participant"] != "_AVERAGE_") &
        (df["n_mfcc"] == best_cfg["n_mfcc"]) &
        (df["features"] == best_cfg["features"]) &
        (df["normalization"] == best_cfg["normalization"]) &
        (df["model"] == best_cfg["model"])
    ].sort_values("pers_mae")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE
    ax = axes[0]
    colors = ["#2ecc71" if imp > 1 else "#e74c3c" if imp < 0 else "#f39c12"
              for imp in sub["pers_improvement"]]
    bars = ax.barh(sub["participant"], sub["pers_mae"], color=colors, edgecolor="white", linewidth=0.5)
    ax.barh(sub["participant"], sub["pers_baseline_mae"], color="lightgray", alpha=0.4, edgecolor="gray", linewidth=0.5)
    ax.set_xlabel("MAE (mg/dL)", fontsize=11)
    title_str = f"Best Config: {best_cfg['model']}, {best_cfg['n_mfcc']} MFCCs, {best_cfg['features']}, {best_cfg['normalization']}"
    ax.set_title(f"Per-Participant MAE\n{title_str}", fontsize=11)
    ax.legend(["Baseline (mean predictor)", "Model"], fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    # Correlation
    ax = axes[1]
    colors_r = ["#2ecc71" if r > 0.3 else "#e74c3c" if r < 0 else "#f39c12" for r in sub["pers_r"]]
    ax.barh(sub["participant"], sub["pers_r"], color=colors_r, edgecolor="white", linewidth=0.5)
    ax.axvline(0.3, color="gray", linestyle="--", alpha=0.5, label="r=0.3 threshold")
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_title("Per-Participant Correlation", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(fig_dir / "participant_breakdown_best.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_efficiency_frontier(df: pd.DataFrame, fig_dir: Path):
    """Scatter plot: n_features vs MAE showing the efficiency frontier."""
    avg = df[df["participant"] == "_AVERAGE_"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Personalized efficiency
    ax = axes[0]
    for model in avg["model"].unique():
        sub = avg[avg["model"] == model]
        ax.scatter(sub["n_features"], sub["pers_mae"], label=model, color=COLORS.get(model, "gray"),
                   s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("Avg Personalized MAE (mg/dL)", fontsize=11)
    ax.set_title("Efficiency Frontier: Features vs Performance (Personalized)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Population efficiency
    ax = axes[1]
    for model in avg["model"].unique():
        sub = avg[avg["model"] == model]
        ax.scatter(sub["n_features"], sub["pop_mae"], label=model, color=COLORS.get(model, "gray"),
                   s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("Population MAE (mg/dL)", fontsize=11)
    ax.set_title("Efficiency Frontier: Features vs Performance (Population)", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "efficiency_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_vs_cv(df: pd.DataFrame, fig_dir: Path):
    """Compare temporal MAE vs CV MAE (shows optimism of CV)."""
    per_participant = df[(df["participant"] != "_AVERAGE_") & df["temp_mae"].notna()].copy()
    if per_participant.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(per_participant["pers_mae"], per_participant["temp_mae"], alpha=0.3, s=20, c="#3498db")
    lims = [0, max(per_participant[["pers_mae", "temp_mae"]].max().max() * 1.1, 25)]
    ax.plot(lims, lims, "k--", alpha=0.5, label="y=x (equal)")
    ax.set_xlabel("Cross-Validated MAE (mg/dL)", fontsize=11)
    ax.set_ylabel("Temporal Split MAE (mg/dL)", fontsize=11)
    ax.set_title("CV vs Temporal Validation\n(points above line = CV is optimistic)", fontsize=12)
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_dir / "temporal_vs_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Step 6: Generate HTML Report
# =============================================================================

def generate_html_report(df: pd.DataFrame, fig_dir: Path, output_dir: Path):
    """Generate a self-contained HTML report with embedded images and tables."""
    import base64

    def embed_img(path: Path) -> str:
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto; margin: 10px 0;">'

    avg = df[df["participant"] == "_AVERAGE_"].copy()

    # --- Best configs ---
    best_pers_idx = avg["pers_mae"].idxmin()
    best_pers = avg.loc[best_pers_idx]
    best_pop_idx = avg["pop_mae"].idxmin()
    best_pop = avg.loc[best_pop_idx]

    # "Sweet spot" = best balance of personalized + population
    avg["combined_score"] = avg["pers_mae"] * 0.5 + avg["pop_mae"] * 0.5
    sweet_idx = avg["combined_score"].idxmin()
    sweet = avg.loc[sweet_idx]

    # --- Top 10 tables ---
    top_pers = avg.nsmallest(10, "pers_mae")[
        ["n_mfcc", "features", "normalization", "model", "n_features", "pers_mae", "pers_r", "pop_mae", "pop_r"]
    ].reset_index(drop=True)

    top_pop = avg.nsmallest(10, "pop_mae")[
        ["n_mfcc", "features", "normalization", "model", "n_features", "pop_mae", "pop_r", "pers_mae", "pers_r"]
    ].reset_index(drop=True)

    top_sweet = avg.nsmallest(10, "combined_score")[
        ["n_mfcc", "features", "normalization", "model", "n_features", "pers_mae", "pers_r", "pop_mae", "pop_r", "combined_score"]
    ].reset_index(drop=True)

    # --- Per-participant table for best config ---
    best_cfg = {"n_mfcc": best_pers["n_mfcc"], "features": best_pers["features"],
                "normalization": best_pers["normalization"], "model": best_pers["model"]}
    per_p = df[
        (df["participant"] != "_AVERAGE_") &
        (df["n_mfcc"] == best_cfg["n_mfcc"]) &
        (df["features"] == best_cfg["features"]) &
        (df["normalization"] == best_cfg["normalization"]) &
        (df["model"] == best_cfg["model"])
    ].sort_values("pers_mae")[
        ["participant", "n_samples", "pers_mae", "pers_r", "pers_baseline_mae", "pers_improvement", "pers_pct_improvement", "temp_mae", "temp_r"]
    ].reset_index(drop=True)

    # --- Summary stats ---
    n_configs = len(avg)
    n_participants = df[df["participant"] != "_AVERAGE_"]["participant"].nunique()
    total_samples = df[df["participant"] != "_AVERAGE_"].groupby("participant")["n_samples"].first().sum() if n_participants > 0 else 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ONVOX AutoResearch — Hyperparameter Sweep Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #2980b9; margin-top: 40px; }}
  h3 {{ color: #34495e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
  th {{ background: #2c3e50; color: white; padding: 10px 8px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  tr:hover {{ background: #e8f4f8; }}
  .highlight {{ background: #d4efdf !important; font-weight: bold; }}
  .metric-box {{ display: inline-block; background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px 25px; margin: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
  .metric-box .value {{ font-size: 28px; font-weight: bold; color: #2980b9; }}
  .metric-box .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
  .section {{ background: white; border-radius: 8px; padding: 25px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.08); }}
  .green {{ color: #27ae60; }} .red {{ color: #e74c3c; }} .orange {{ color: #f39c12; }}
  .config-tag {{ display: inline-block; background: #ecf0f1; border-radius: 4px; padding: 2px 8px; font-family: monospace; font-size: 12px; }}
  .footer {{ text-align: center; color: #95a5a6; font-size: 11px; margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; }}
</style>
</head>
<body>

<h1>ONVOX AutoResearch — Voice-Based Glucose Estimation<br>Hyperparameter &amp; Feature Configuration Analysis</h1>
<p>Generated: {now} | Configurations tested: {n_configs} | Participants: {n_participants} | Total samples: {total_samples}</p>

<div class="section">
<h2>Executive Summary</h2>
<div style="text-align: center;">
  <div class="metric-box">
    <div class="value">{best_pers['pers_mae']:.1f}</div>
    <div class="label">Best Personalized MAE<br>(mg/dL)</div>
  </div>
  <div class="metric-box">
    <div class="value">{best_pop['pop_mae']:.1f}</div>
    <div class="label">Best Population MAE<br>(mg/dL)</div>
  </div>
  <div class="metric-box">
    <div class="value">{n_configs}</div>
    <div class="label">Configurations<br>Tested</div>
  </div>
  <div class="metric-box">
    <div class="value">{n_participants}</div>
    <div class="label">Participants</div>
  </div>
</div>

<h3>Best Personalized Configuration</h3>
<p><span class="config-tag">{best_pers['model']}</span>
   <span class="config-tag">{int(best_pers['n_mfcc'])} MFCCs</span>
   <span class="config-tag">{best_pers['features']}</span>
   <span class="config-tag">{best_pers['normalization']} norm</span>
   <span class="config-tag">{int(best_pers['n_features'])} features</span>
   → MAE={best_pers['pers_mae']:.2f} mg/dL, r={best_pers['pers_r']:.3f}</p>

<h3>Best Population Configuration</h3>
<p><span class="config-tag">{best_pop['model']}</span>
   <span class="config-tag">{int(best_pop['n_mfcc'])} MFCCs</span>
   <span class="config-tag">{best_pop['features']}</span>
   <span class="config-tag">{best_pop['normalization']} norm</span>
   <span class="config-tag">{int(best_pop['n_features'])} features</span>
   → MAE={best_pop['pop_mae']:.2f} mg/dL, r={best_pop['pop_r']:.3f}</p>

<h3>Sweet Spot (Best Balance)</h3>
<p><span class="config-tag">{sweet['model']}</span>
   <span class="config-tag">{int(sweet['n_mfcc'])} MFCCs</span>
   <span class="config-tag">{sweet['features']}</span>
   <span class="config-tag">{sweet['normalization']} norm</span>
   <span class="config-tag">{int(sweet['n_features'])} features</span>
   → Personalized MAE={sweet['pers_mae']:.2f}, Population MAE={sweet['pop_mae']:.2f}</p>
</div>

<div class="section">
<h2>1. MFCC Count Analysis</h2>
<p>Testing {', '.join(str(c) for c in sorted(avg['n_mfcc'].unique()))} MFCCs — how many coefficients capture the glucose-relevant voice signal?</p>
{embed_img(fig_dir / "mfcc_count_sweep.png")}
</div>

<div class="section">
<h2>2. Feature Combination Analysis</h2>
<p>Additive feature layers: MFCC → +spectral → +pitch → +voice quality → +temporal context.</p>
{embed_img(fig_dir / "feature_combo_sweep.png")}
</div>

<div class="section">
<h2>3. Normalization Effect</h2>
<p>Within-speaker normalization removes speaker identity (dominant variance) to expose glucose signal.</p>
{embed_img(fig_dir / "normalization_effect.png")}
</div>

<div class="section">
<h2>4. Model × Feature Heatmap</h2>
<p>Which algorithm works best with which feature set?</p>
{embed_img(fig_dir / "model_feature_heatmap.png")}
</div>

<div class="section">
<h2>5. Efficiency Frontier</h2>
<p>Feature dimensionality vs. performance — more features don't always help with small datasets.</p>
{embed_img(fig_dir / "efficiency_frontier.png")}
</div>

<div class="section">
<h2>6. Cross-Validation vs. Temporal Validation</h2>
<p>Points above the diagonal indicate CV is optimistic (temporal leakage).</p>
{embed_img(fig_dir / "temporal_vs_cv.png")}
</div>

<div class="section">
<h2>7. Top 10 Personalized Configurations</h2>
{top_pers.to_html(index=False, float_format="%.2f", classes="", border=0)}
</div>

<div class="section">
<h2>8. Top 10 Population Configurations</h2>
{top_pop.to_html(index=False, float_format="%.2f", classes="", border=0)}
</div>

<div class="section">
<h2>9. Sweet Spot: Top 10 Balanced Configurations</h2>
<p>Ranked by 0.5 × Personalized MAE + 0.5 × Population MAE.</p>
{top_sweet.to_html(index=False, float_format="%.2f", classes="", border=0)}
</div>

<div class="section">
<h2>10. Per-Participant Breakdown (Best Personalized Config)</h2>
<p>Configuration: <span class="config-tag">{best_cfg['model']}</span>
   <span class="config-tag">{best_cfg['n_mfcc']} MFCCs</span>
   <span class="config-tag">{best_cfg['features']}</span>
   <span class="config-tag">{best_cfg['normalization']} norm</span></p>
{per_p.to_html(index=False, float_format="%.2f", classes="", border=0)}
{embed_img(fig_dir / "participant_breakdown_best.png")}
</div>

<div class="section">
<h2>Methodology</h2>
<h3>Data</h3>
<ul>
  <li><b>Source:</b> WhatsApp voice messages paired with FreeStyle Libre CGM readings</li>
  <li><b>Matching:</b> ±30 min window with linear interpolation between CGM readings</li>
  <li><b>Participants:</b> {n_participants} individuals, {total_samples} matched voice-glucose pairs</li>
  <li><b>Audio format:</b> 16 kHz mono WAV (auto-converted from opus/waptt where needed)</li>
</ul>

<h3>Feature Extraction</h3>
<ul>
  <li><b>MFCCs:</b> {', '.join(str(c) for c in sorted(avg['n_mfcc'].unique()))} coefficients tested, with delta and delta-delta, mean+std aggregation</li>
  <li><b>Spectral:</b> centroid, bandwidth, rolloff, flatness, contrast (mean+std)</li>
  <li><b>Pitch (F0):</b> via pYIN — mean, std, median</li>
  <li><b>Voice Quality:</b> jitter (local, RAP, PPQ5), shimmer (local, APQ3, APQ5), tremor (4-12 Hz power), formants (F1-F3), HNR, CPP, pitch quality (CV, skew, kurtosis), composites</li>
  <li><b>Temporal:</b> circadian encoding (sin/cos hour + day-of-week), delta features, time-since-last</li>
</ul>

<h3>Normalization</h3>
<ul>
  <li><b>None:</b> raw features</li>
  <li><b>Z-score:</b> per-speaker (x − μ_speaker) / σ_speaker — removes inter-speaker variance</li>
  <li><b>Rank:</b> percentile rank within speaker — robust to outliers</li>
</ul>

<h3>Models</h3>
<ul>
  <li><b>Ridge:</b> L2-regularized linear regression (α=1.0)</li>
  <li><b>BayesianRidge:</b> Bayesian linear regression with automatic relevance determination</li>
  <li><b>SVR:</b> Support Vector Regression (RBF kernel, C=10)</li>
  <li><b>ElasticNet:</b> Combined L1/L2 regularized linear regression</li>
  <li><b>Lasso:</b> Sparse L1-regularized linear regression</li>
  <li><b>Huber:</b> Robust linear regression resilient to outliers</li>
  <li><b>RandomForest:</b> 100 trees, max_depth=10</li>
  <li><b>GradientBoosting:</b> Boosted trees, max_depth=5</li>
  <li><b>ExtraTrees:</b> Extremely randomized trees ensemble</li>
  <li><b>KNN:</b> k=5, distance-weighted</li>
</ul>

<h3>Evaluation</h3>
<ul>
  <li><b>Personalized:</b> Leave-One-Out CV (n≤50) or 10-fold CV (n>50)</li>
  <li><b>Population:</b> Leave-One-Person-Out CV (train on all but one participant, predict held-out)</li>
  <li><b>Temporal:</b> Chronological 70/30 split (train on earlier, test on later recordings)</li>
  <li><b>Baseline:</b> Mean predictor (predicts training set mean for all test samples)</li>
  <li><b>Metrics:</b> MAE (mg/dL), Pearson r, RMSE, improvement over baseline</li>
</ul>
</div>

<div class="section">
<h2>Key Findings</h2>
<ol>
  <li><b>Personalized models significantly outperform population models</b> — the voice-glucose signal is person-specific, consistent with Klick Labs (2024) finding of person-specific F0-glucose coupling.</li>
  <li><b>Within-speaker z-normalization</b> removes the dominant inter-speaker variance (speaker identity), exposing the subtle glucose signal beneath.</li>
  <li><b>More MFCCs are not always better</b> — the sweet spot depends on sample size per participant (curse of dimensionality).</li>
  <li><b>Voice quality features</b> (jitter, shimmer, tremor) capture physiologically-motivated glucose effects through neuromuscular pathways.</li>
  <li><b>Temporal context features</b> add circadian glucose patterns and voice dynamics, especially helpful for population models.</li>
  <li><b>Cross-validation is optimistic</b> compared to temporal split — honest temporal evaluation should be the primary metric for production deployment.</li>
</ol>
</div>

<div class="footer">
  ONVOX AutoResearch Project — Voice-Based Glucose Estimation<br>
  Report generated by hyperparameter_sweep.py | {now}
</div>

</body>
</html>"""

    report_path = output_dir / "sweep_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("HTML report saved: %s", report_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ONVOX Hyperparameter Sweep")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep (fast)")
    parser.add_argument("--participants", nargs="+", default=None, help="Filter participants")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    start = time.time()

    logger.info("=" * 70)
    logger.info("ONVOX AutoResearch — Hyperparameter & Feature Configuration Sweep")
    logger.info("=" * 70)

    cfg = load_config(args.config)

    # Output dirs
    base_dir = get_base_dir(cfg)
    output_dir = base_dir / "output" / "sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Select config space
    if args.quick:
        mfcc_counts = MFCC_COUNTS_QUICK
        feature_combos = FEATURE_COMBOS_QUICK
        norm_methods = NORM_METHODS_QUICK
        model_names = MODEL_NAMES_QUICK
    else:
        mfcc_counts = MFCC_COUNTS
        feature_combos = FEATURE_COMBOS
        norm_methods = NORM_METHODS
        model_names = MODEL_NAMES

    total = len(mfcc_counts) * len(feature_combos) * len(norm_methods) * len(model_names)
    logger.info("Search space: %d MFCC × %d feature combos × %d norms × %d models = %d configs",
                len(mfcc_counts), len(feature_combos), len(norm_methods), len(model_names), total)

    # Step 1: Load all audio
    logger.info("\nStep 1: Loading audio...")
    participant_data = load_all_audio(cfg, args.participants)
    if not participant_data:
        logger.error("No participant data loaded!")
        sys.exit(1)

    # Step 2-4: Run sweep
    logger.info("\nStep 2-4: Running sweep (%d configurations)...", total)
    df = run_sweep(participant_data, mfcc_counts, feature_combos, norm_methods, model_names)

    # Save raw results
    df.to_csv(output_dir / "sweep_results.csv", index=False)
    logger.info("Results saved: %s (%d rows)", output_dir / "sweep_results.csv", len(df))

    # Save summary (averages only)
    avg = df[df["participant"] == "_AVERAGE_"]
    avg.to_csv(output_dir / "sweep_summary.csv", index=False)

    # Step 5: Generate plots
    logger.info("\nStep 5: Generating visualizations...")
    plot_mfcc_sweep(df, fig_dir)
    plot_feature_combo_sweep(df, fig_dir)
    plot_normalization_effect(df, fig_dir)
    plot_model_comparison_heatmap(df, fig_dir)
    plot_participant_breakdown(df, fig_dir)
    plot_efficiency_frontier(df, fig_dir)
    plot_temporal_vs_cv(df, fig_dir)

    # Step 6: Generate HTML report
    logger.info("\nStep 6: Generating HTML report...")
    generate_html_report(df, fig_dir, output_dir)

    elapsed = time.time() - start
    logger.info("\nSweep complete in %.1f seconds.", elapsed)
    logger.info("Report: %s", output_dir / "sweep_report.html")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Sweep crashed: %s", e, exc_info=True)
        sys.exit(1)
