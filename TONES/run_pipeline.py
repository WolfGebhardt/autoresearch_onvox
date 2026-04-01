#!/usr/bin/env python3
"""
TONES Pipeline — Novel Strategy for Maximum-Performance Glucose Estimation
============================================================================
Run the full voice-based glucose estimation pipeline with novel strategies:

  Phase 1: Load and match all participant data (with opus/waptt auto-conversion)
  Phase 2: Extract features — MFCC + voice quality + temporal context
  Phase 3: Within-speaker normalization (z-score, running, rank)
  Phase 4: Standard personalized models (LOO/K-Fold CV)
  Phase 5: Novel strategies:
           a) Deviation-from-personal-mean regression
           b) Rate-of-change classification
           c) Regime classification (hypo/normal/hyper)
           d) Hierarchical Bayesian personalization
           e) Few-shot calibration evaluation
           f) Diverse ensemble (different feature sets + algorithms)
  Phase 6: Temporal validation (chronological split + walk-forward)
  Phase 7: Population model (LOPO)
  Phase 8: Output — visualizations, JSON report

Usage:
    python run_pipeline.py                        # Full pipeline with all strategies
    python run_pipeline.py --data-only            # Just build canonical dataset
    python run_pipeline.py --participants Wolf Lara  # Specific participants
    python run_pipeline.py --no-cache             # Skip feature cache
    python run_pipeline.py --skip-novel           # Only run standard pipeline
    python run_pipeline.py --config path/to/config.yaml
"""

import argparse
import gc
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TONES modules
from tones.config import load_config, get_base_dir, get_participant_config
from tones.data.loaders import (
    load_participant_data,
    load_all_participants,
    collect_audio_files,
    parse_timestamp_from_filename,
    load_glucose_csv,
)
from tones.features.mfcc import MFCCExtractor, create_extractor_from_config
from tones.features.voice_quality import VoiceQualityExtractor, create_voice_quality_extractor
from tones.features.cache import FeatureCache
from tones.features.normalize import normalize_features
from tones.features.temporal import build_temporal_features
from tones.models.train import (
    train_personalized,
    train_personalized_deviation,
    train_rate_of_change_classifier,
    train_regime_classifier,
    train_population,
    get_best_personalized,
    mean_predictor_baseline,
    compute_metrics,
)
from tones.models.bayesian import train_hierarchical_bayesian
from tones.models.ensemble import train_ensemble_personalized, build_default_ensemble
from tones.models.calibration import evaluate_few_shot_calibration
from tones.evaluation.temporal_cv import evaluate_all_temporal
from tones.evaluation.metrics import (
    clarke_error_grid,
    clarke_zone_percentages,
    plot_clarke_error_grid,
    plot_scatter_per_participant,
    plot_model_comparison,
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pymc").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1: Build Canonical Dataset
# =============================================================================

def phase_build_dataset(cfg: Dict, participants_filter: Optional[List[str]] = None) -> pd.DataFrame:
    """Load and match all audio-glucose pairs (with auto-conversion of opus/waptt)."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Building Canonical Dataset")
    logger.info("=" * 70)

    base_dir = get_base_dir(cfg)
    matching_cfg = cfg.get("matching", {})
    participants = cfg.get("participants", {})

    if participants_filter:
        participants = {k: v for k, v in participants.items() if k in participants_filter}
        logger.info("Filtered to %d participants: %s", len(participants), participants_filter)

    all_dfs = []
    summary = []

    for name, pcfg in participants.items():
        if not pcfg.get("glucose_csv"):
            continue

        df = load_participant_data(name, pcfg, base_dir, matching_cfg)

        audio_files = collect_audio_files(
            pcfg.get("audio_dirs", []),
            pcfg.get("audio_ext", [".wav"]),
            base_dir,
        )

        n_audio = len(audio_files)
        n_matched = len(df)
        match_rate = 100 * n_matched / n_audio if n_audio > 0 else 0

        summary.append({
            "participant": name,
            "audio_files": n_audio,
            "matched": n_matched,
            "match_rate": round(match_rate, 1),
        })

        if not df.empty:
            all_dfs.append(df)

    # Print summary table
    logger.info("")
    logger.info("%-15s %8s %8s %8s", "Participant", "Audio", "Matched", "Rate")
    logger.info("-" * 45)
    total_audio = total_matched = 0
    for row in summary:
        logger.info(
            "%-15s %8d %8d %7.1f%%",
            row["participant"], row["audio_files"], row["matched"], row["match_rate"],
        )
        total_audio += row["audio_files"]
        total_matched += row["matched"]
    logger.info("-" * 45)
    total_rate = 100 * total_matched / total_audio if total_audio > 0 else 0
    logger.info("%-15s %8d %8d %7.1f%%", "TOTAL", total_audio, total_matched, total_rate)

    if not all_dfs:
        logger.error("No data loaded!")
        return pd.DataFrame()

    canonical = pd.concat(all_dfs, ignore_index=True)
    logger.info("\nCanonical dataset: %d samples from %d participants", len(canonical), len(all_dfs))
    return canonical


# =============================================================================
# Phase 2: Feature Extraction (MFCC + Voice Quality)
# =============================================================================

def phase_extract_features(
    canonical_df: pd.DataFrame,
    cfg: Dict,
    use_cache: bool = True,
) -> Dict[str, Dict]:
    """Extract MFCC and voice quality features for all samples."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: Feature Extraction (MFCC + Voice Quality)")
    logger.info("=" * 70)

    feat_cfg = cfg.get("features", {})
    use_vq = feat_cfg.get("use_voice_quality", True)

    extractor = create_extractor_from_config(cfg)
    vq_extractor = create_voice_quality_extractor(cfg) if use_vq else None

    # Setup caches
    cache_dir = feat_cfg.get("cache_dir", ".cache/features")
    base_cache_dir = str(get_base_dir(cfg) / cache_dir)
    cache_enabled = use_cache and feat_cfg.get("use_cache", True)

    mfcc_cache = FeatureCache(cache_dir=base_cache_dir, enabled=cache_enabled)
    vq_cache = FeatureCache(
        cache_dir=str(Path(base_cache_dir) / "voice_quality"),
        enabled=cache_enabled,
    ) if use_vq else None

    mfcc_id = f"mfcc_n{extractor.n_mfcc}_sr{extractor.sr}"
    vq_id = "voice_quality_v1"

    data_by_participant: Dict[str, Dict] = defaultdict(lambda: {
        "features": [], "voice_quality": [], "glucose": [],
        "timestamps": [], "audio_paths": [],
        "glucose_rates": [], "glucose_rate_labels": [], "glucose_regimes": [],
    })

    total = len(canonical_df)
    mfcc_cached = mfcc_extracted = vq_extracted = failed = 0

    for idx, (i, row) in enumerate(canonical_df.iterrows()):
        audio_path = row["audio_path"]
        subject = row["subject"]

        try:
            # --- MFCC features ---
            mfcc_features = mfcc_cache.get(audio_path, mfcc_id)
            if mfcc_features is not None:
                mfcc_cached += 1
            else:
                mfcc_features = extractor.extract_from_file(audio_path)
                if mfcc_features is not None:
                    mfcc_cache.put(audio_path, mfcc_id, mfcc_features)
                    mfcc_extracted += 1
                else:
                    failed += 1
                    continue

            # --- Voice quality features ---
            vq_features = None
            if vq_extractor is not None:
                vq_features = vq_cache.get(audio_path, vq_id)
                if vq_features is None:
                    vq_features = vq_extractor.extract_from_file(audio_path)
                    if vq_features is not None:
                        vq_cache.put(audio_path, vq_id, vq_features)
                        vq_extracted += 1

            data_by_participant[subject]["features"].append(mfcc_features)
            if vq_extractor is not None:
                # Use zero-vector fallback when extraction fails (e.g. pYIN on short/noisy audio)
                vq = vq_features if vq_features is not None else np.zeros(vq_extractor.n_features, dtype=np.float32)
                data_by_participant[subject]["voice_quality"].append(vq)
            data_by_participant[subject]["glucose"].append(row["glucose_mg_dl"])
            data_by_participant[subject]["timestamps"].append(row["audio_timestamp"])
            data_by_participant[subject]["audio_paths"].append(audio_path)
            data_by_participant[subject]["glucose_rates"].append(row.get("glucose_rate"))
            data_by_participant[subject]["glucose_rate_labels"].append(row.get("glucose_rate_label", "unknown"))
            data_by_participant[subject]["glucose_regimes"].append(row.get("glucose_regime", "normal"))
        except Exception as e:
            logger.warning("  Failed on sample %d (%s): %s", idx, audio_path, e)
            failed += 1
            continue

        if (idx + 1) % 50 == 0:
            logger.info("  Processed %d/%d samples...", idx + 1, total)
            sys.stdout.flush()
            gc.collect()

    logger.info(
        "MFCC: %d cached, %d extracted, %d failed out of %d total",
        mfcc_cached, mfcc_extracted, failed, total,
    )
    if use_vq:
        logger.info("Voice quality: %d extracted", vq_extracted)

    # Convert lists to arrays and concatenate voice_quality into main features when enabled
    for name, data in data_by_participant.items():
        data["features_mfcc"] = np.array(data["features"])  # MFCC only (for ensemble diversity)
        data["features"] = np.array(data["features"])
        data["glucose"] = np.array(data["glucose"])
        data["glucose_rates"] = np.array(data["glucose_rates"], dtype=object)
        data["glucose_rate_labels"] = np.array(data["glucose_rate_labels"])
        data["glucose_regimes"] = np.array(data["glucose_regimes"])

        if data["voice_quality"]:
            data["voice_quality"] = np.array(data["voice_quality"])
            # Concatenate MFCC + voice_quality for combined features (used by personalized models)
            data["features"] = np.hstack([data["features"], data["voice_quality"]])
        else:
            data["voice_quality"] = None

        logger.info("  %s: %d samples, %d features", name, len(data["glucose"]), data["features"].shape[1])

    return dict(data_by_participant)


# =============================================================================
# Phase 3: Within-Speaker Normalization
# =============================================================================

def phase_normalize(data_by_participant: Dict[str, Dict], cfg: Dict) -> Dict[str, Dict]:
    """Apply within-speaker feature normalization."""
    feat_cfg = cfg.get("features", {})
    method = feat_cfg.get("normalization", "zscore")

    if method == "none":
        logger.info("\nPHASE 3: Normalization — SKIPPED (disabled)")
        return data_by_participant

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: Within-Speaker Normalization (%s)", method)
    logger.info("=" * 70)

    return normalize_features(
        data_by_participant,
        method=method,
        window_size=feat_cfg.get("normalization_window", 20),
        min_history=feat_cfg.get("normalization_min_history", 5),
    )


# =============================================================================
# Phase 4: Temporal Context Features
# =============================================================================

def phase_temporal_features(data_by_participant: Dict[str, Dict], cfg: Dict) -> Dict[str, Dict]:
    """Add temporal context features (circadian, delta, time-since)."""
    feat_cfg = cfg.get("features", {})
    if not feat_cfg.get("use_temporal", True):
        logger.info("\nPHASE 4: Temporal Features — SKIPPED (disabled)")
        return data_by_participant

    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: Temporal Context Features")
    logger.info("=" * 70)

    return build_temporal_features(
        data_by_participant,
        include_circadian=True,
        include_deltas=True,
        include_time_since=True,
        include_rolling=feat_cfg.get("use_rolling_stats", False),
        max_gap_hours=feat_cfg.get("temporal_max_gap_hours", 4.0),
    )


# =============================================================================
# Phase 5: Standard Personalized Models
# =============================================================================

def phase_personalized(data_by_participant: Dict[str, Dict], cfg: Dict) -> Dict[str, Dict]:
    """Train standard personalized models for each participant."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 5: Personalized Models (per-participant CV)")
    logger.info("=" * 70)

    model_cfg = cfg.get("models", {})
    pers_cfg = model_cfg.get("personalized", {})
    min_samples = pers_cfg.get("min_samples", 20)

    all_results = {}

    for name, data in sorted(data_by_participant.items()):
        X = data["features"]
        y = data["glucose"]

        if len(X) < min_samples:
            logger.info("  %s: Skipped (%d < %d samples)", name, len(X), min_samples)
            continue

        logger.info("\n  %s (%d samples, glucose %.0f-%.0f mg/dL):", name, len(y), y.min(), y.max())

        # Baseline
        baseline = mean_predictor_baseline(y)
        logger.info("    Mean predictor baseline: MAE=%.2f", baseline["mae"])

        results = train_personalized(
            X, y,
            min_samples=min_samples,
            cv_kfold_threshold=pers_cfg.get("cv_kfold_splits", 50),
            model_params=model_cfg,
        )

        best = get_best_personalized(results)
        if best:
            improvement = baseline["mae"] - best["mae"]
            all_results[name] = {
                "n_samples": len(y),
                "glucose_mean": float(np.mean(y)),
                "glucose_std": float(np.std(y)),
                "glucose_min": float(np.min(y)),
                "glucose_max": float(np.max(y)),
                "best_model": best["model_name"],
                "mae": best["mae"],
                "rmse": best["rmse"],
                "r": best["r"],
                "baseline_mae": baseline["mae"],
                "improvement": improvement,
                "actual": y,
                "predictions": best["predictions"],
                "all_models": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                               for k, v in results.items()},
            }
            logger.info(
                "    BEST: %s — MAE=%.2f, r=%.3f (baseline=%.2f, improvement=%.2f)",
                best["model_name"], best["mae"], best["r"],
                baseline["mae"], improvement,
            )

    return all_results


# =============================================================================
# Phase 6: Novel Strategies
# =============================================================================

def phase_novel_strategies(
    data_by_participant: Dict[str, Dict],
    cfg: Dict,
) -> Dict[str, Dict]:
    """Run all novel modeling strategies from the plan."""
    strategies_cfg = cfg.get("models", {}).get("strategies", {})
    model_cfg = cfg.get("models", {})
    min_samples = model_cfg.get("personalized", {}).get("min_samples", 20)

    novel_results = {}

    # --- Phase 2A: Deviation-from-personal-mean ---
    if strategies_cfg.get("use_deviation_target", True):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 6a: Deviation-from-Personal-Mean Regression")
        logger.info("=" * 70)

        deviation_results = {}
        for name, data in sorted(data_by_participant.items()):
            X, y = data["features"], data["glucose"]
            if len(X) < min_samples:
                continue

            logger.info("\n  %s (%d samples):", name, len(y))
            results = train_personalized_deviation(
                X, y, min_samples=min_samples, model_params=model_cfg,
            )
            if results:
                best_name = min(results, key=lambda k: results[k]["mae"])
                deviation_results[name] = results[best_name]

        novel_results["deviation"] = deviation_results

    # --- Phase 2B: Rate-of-change classification ---
    if strategies_cfg.get("use_rate_classification", True):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 6b: Glucose Rate-of-Change Classification")
        logger.info("=" * 70)

        rate_results = {}
        for name, data in sorted(data_by_participant.items()):
            X = data["features"]
            rate_labels = data.get("glucose_rate_labels")

            if rate_labels is None or len(X) < min_samples:
                continue

            # Check if we have enough non-unknown labels
            valid_mask = rate_labels != "unknown"
            if valid_mask.sum() < min_samples:
                logger.info("  %s: Too few rate labels (%d)", name, valid_mask.sum())
                continue

            logger.info("\n  %s (%d labeled samples):", name, valid_mask.sum())
            results = train_rate_of_change_classifier(
                X, rate_labels, min_samples=min_samples,
            )
            if results:
                best_name = max(results, key=lambda k: results[k]["f1_macro"])
                rate_results[name] = results[best_name]

        novel_results["rate_classification"] = rate_results

    # --- Phase 2C: Regime classification ---
    if strategies_cfg.get("use_regime_classification", True):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 6c: Glucose Regime Classification (hypo/normal/hyper)")
        logger.info("=" * 70)

        regime_results = {}
        for name, data in sorted(data_by_participant.items()):
            X = data["features"]
            regime_labels = data.get("glucose_regimes")

            if regime_labels is None or len(X) < min_samples:
                continue

            logger.info("\n  %s (%d samples):", name, len(X))
            results = train_regime_classifier(
                X, regime_labels, min_samples=min_samples,
            )
            if results:
                best_name = max(results, key=lambda k: results[k]["f1_macro"])
                regime_results[name] = results[best_name]

        novel_results["regime_classification"] = regime_results

    # --- Phase 3A: Hierarchical Bayesian ---
    if strategies_cfg.get("use_hierarchical_bayesian", True):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 6d: Hierarchical Bayesian Personalization")
        logger.info("=" * 70)

        bayesian_results = train_hierarchical_bayesian(
            data_by_participant,
            use_pymc=(strategies_cfg.get("bayesian_backend", "empirical") == "pymc"),
        )
        novel_results["hierarchical_bayesian"] = bayesian_results

    # --- Phase 3C: Diverse Ensemble ---
    if strategies_cfg.get("use_ensemble", True):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 6e: Diverse Ensemble (multi-feature-set)")
        logger.info("=" * 70)

        # Build feature sets for ensemble (diverse feature sets for robust prediction)
        feature_sets_by_participant = {}
        for name, data in data_by_participant.items():
            fs = {
                "mfcc": data.get("features_mfcc", data.get("features_raw", data["features"])),
                "combined": data["features"],  # MFCC + VQ + temporal
            }
            if data.get("voice_quality") is not None:
                fs["voice_quality"] = data["voice_quality"]
            fs["temporal"] = data["features"]

            feature_sets_by_participant[name] = fs

        ensemble_results = train_ensemble_personalized(
            data_by_participant,
            feature_sets_by_participant,
            min_samples=min_samples,
        )
        novel_results["ensemble"] = ensemble_results

    # --- Phase 3B: Few-Shot Calibration ---
    if strategies_cfg.get("use_few_shot_calibration", True):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 6e2: Few-Shot Calibration (slope+intercept, recency-weighted)")
        logger.info("=" * 70)

        from sklearn.linear_model import BayesianRidge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        all_X = np.vstack([data["features"] for data in data_by_participant.values()])
        all_y = np.concatenate([data["glucose"] for data in data_by_participant.values()])
        all_X = np.nan_to_num(all_X, nan=0, posinf=0, neginf=0)

        scaler = RobustScaler()
        population_model = Pipeline([
            ("scaler", scaler),
            ("model", BayesianRidge()),
        ])
        population_model.fit(all_X, all_y)
        logger.info("  Population model fitted on %d samples", len(all_y))

        calib_results = evaluate_few_shot_calibration(
            data_by_participant,
            population_model.named_steps["model"],
            population_model.named_steps["scaler"],
            n_calibration=strategies_cfg.get("calibration_samples", 10),
            strategy=strategies_cfg.get("calibration_strategy", "linear"),
        )
        novel_results["few_shot_calibration"] = calib_results

    # --- Phase 0A: Temporal Validation ---
    if strategies_cfg.get("use_temporal_validation", True):
        logger.info("")
        logger.info("=" * 70)
        logger.info("PHASE 6f: Temporal (Chronological) Validation")
        logger.info("=" * 70)

        temporal_results = {}
        train_frac = strategies_cfg.get("temporal_train_fraction", 0.7)

        for name, data in sorted(data_by_participant.items()):
            X, y = data["features"], data["glucose"]
            timestamps = np.array(data.get("timestamps", []))

            if len(X) < min_samples or len(timestamps) == 0:
                continue

            logger.info("\n  %s (%d samples):", name, len(y))
            results = evaluate_all_temporal(
                X, y, timestamps,
                min_samples=min_samples,
                train_fraction=train_frac,
                model_params=model_cfg,
            )
            if results:
                temporal_results[name] = results

        novel_results["temporal_validation"] = temporal_results

    return novel_results


# =============================================================================
# Phase 7: Population Model
# =============================================================================

def phase_population(data_by_participant: Dict[str, Dict], cfg: Dict) -> Dict[str, Dict]:
    """Train population model using LOPO CV."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 7: Population Model (Leave-One-Person-Out)")
    logger.info("=" * 70)

    all_X, all_y, all_groups = [], [], []
    for name, data in data_by_participant.items():
        all_X.append(data["features"])
        all_y.append(data["glucose"])
        all_groups.extend([name] * len(data["glucose"]))

    if not all_X:
        logger.error("No data for population model!")
        return {}

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)

    baseline = mean_predictor_baseline(y)
    logger.info("  Mean predictor baseline: MAE=%.2f, r=%.3f", baseline["mae"], baseline["r"])

    model_cfg = cfg.get("models", {})
    results = train_population(X, y, groups, model_params=model_cfg)

    if results:
        best_name = min(results, key=lambda k: results[k]["mae"])
        best = results[best_name]
        logger.info("\n  Per-person breakdown (%s):", best_name)
        for person, metrics in sorted(best.get("per_person", {}).items()):
            logger.info("    %s: MAE=%.2f, r=%.3f", person, metrics["mae"], metrics["r"])

    return results


# =============================================================================
# Phase 8: Output & Visualization
# =============================================================================

def phase_output(
    personalized_results: Dict,
    population_results: Dict,
    novel_results: Dict,
    cfg: Dict,
):
    """Generate visualizations and save comprehensive results."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 8: Output & Visualization")
    logger.info("=" * 70)

    base_dir = get_base_dir(cfg)
    output_dir = base_dir / cfg.get("output", {}).get("dir", "output")
    output_dir.mkdir(exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Scatter plots per participant
    if personalized_results:
        plot_scatter_per_participant(
            personalized_results,
            save_path=str(figures_dir / "scatter_per_participant.png"),
        )
        plt.close("all")

        plot_model_comparison(
            personalized_results,
            save_path=str(figures_dir / "model_comparison.png"),
        )
        plt.close("all")

        # Clarke Error Grid (best participant)
        best_name = min(personalized_results, key=lambda k: personalized_results[k]["mae"])
        best = personalized_results[best_name]
        plot_clarke_error_grid(
            best["actual"], best["predictions"],
            title=f"Clarke Error Grid — {best_name}",
            save_path=str(figures_dir / "clarke_error_grid.png"),
        )
        plt.close("all")

    # 2. Build comprehensive results JSON
    summary = {
        "generated": datetime.now().isoformat(),
        "pipeline_version": "2.0-novel-strategy",
        "n_participants": len(personalized_results),
        "total_samples": sum(r["n_samples"] for r in personalized_results.values()) if personalized_results else 0,
        "personalized": {},
        "population": {},
        "novel_strategies": {},
    }

    # Standard personalized results
    for name, res in personalized_results.items():
        summary["personalized"][name] = {
            "n_samples": res["n_samples"],
            "glucose_mean": res["glucose_mean"],
            "glucose_std": res["glucose_std"],
            "glucose_range": [res["glucose_min"], res["glucose_max"]],
            "best_model": res["best_model"],
            "mae": round(res["mae"], 2),
            "rmse": round(res["rmse"], 2),
            "r": round(res["r"], 3),
            "baseline_mae": round(res.get("baseline_mae", 0), 2),
            "improvement": round(res.get("improvement", 0), 2),
        }

    # Population results
    for model_name, res in population_results.items():
        summary["population"][model_name] = {
            "mae": round(res["mae"], 2),
            "rmse": round(res["rmse"], 2),
            "r": round(res["r"], 3),
        }

    # Novel strategy results
    for strategy_name, strategy_results in novel_results.items():
        if not strategy_results:
            continue

        strategy_summary = {}
        for name, res in strategy_results.items():
            if isinstance(res, dict):
                # Extract serializable metrics
                clean = {}
                for k, v in res.items():
                    if isinstance(v, (int, float, str, bool)):
                        clean[k] = round(v, 4) if isinstance(v, float) else v
                    elif isinstance(v, dict):
                        # Nested dict (e.g., per-model results)
                        clean[k] = {
                            kk: round(vv, 4) if isinstance(vv, float) else vv
                            for kk, vv in v.items()
                            if isinstance(vv, (int, float, str, bool))
                        }
                strategy_summary[name] = clean

        summary["novel_strategies"][strategy_name] = strategy_summary

    results_path = output_dir / "results_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results saved: %s", results_path)
    logger.info("Figures saved: %s", figures_dir)

    # Print final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE — RESULTS SUMMARY")
    logger.info("=" * 70)

    logger.info("")
    logger.info("Standard Personalized Models:")
    logger.info("  %-15s %6s %8s %8s %8s %8s %8s", "Participant", "N", "MAE", "r", "Baseline", "Improve", "Model")
    logger.info("  " + "-" * 72)
    for name, res in sorted(personalized_results.items(), key=lambda x: x[1]["mae"]):
        logger.info(
            "  %-15s %6d %8.2f %8.3f %8.2f %+8.2f %8s",
            name, res["n_samples"], res["mae"], res["r"],
            res.get("baseline_mae", 0), res.get("improvement", 0), res["best_model"],
        )

    # Novel strategy summaries
    for strategy_name, strategy_results in novel_results.items():
        if not strategy_results:
            continue

        logger.info("")
        logger.info("Novel: %s", strategy_name.replace("_", " ").title())

        for name, res in sorted(strategy_results.items()):
            if "mae" in res:
                logger.info("  %-15s MAE=%.2f, r=%.3f", name, res["mae"], res.get("r", 0))
            elif "accuracy" in res:
                logger.info("  %-15s Acc=%.1f%%, F1=%.3f", name, res["accuracy"] * 100, res.get("f1_macro", 0))
            elif "ensemble_mae" in res:
                logger.info(
                    "  %-15s MAE=%.2f, r=%.3f (improvement=%.2f)",
                    name, res["ensemble_mae"], res.get("ensemble_r", 0), res.get("improvement", 0),
                )
            elif "calibrated_mae" in res:
                logger.info(
                    "  %-15s Uncal MAE=%.2f → Cal MAE=%.2f (improvement=%.2f)",
                    name, res["uncalibrated_mae"], res["calibrated_mae"], res.get("improvement", 0),
                )

    if population_results:
        logger.info("")
        logger.info("Population Models (LOPO):")
        for model_name, res in sorted(population_results.items(), key=lambda x: x[1]["mae"]):
            logger.info("  %s: MAE=%.2f, r=%.3f", model_name, res["mae"], res["r"])


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TONES Voice-Glucose Pipeline (Novel Strategy)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--data-only", action="store_true", help="Only build canonical dataset")
    parser.add_argument("--participants", nargs="+", default=None, help="Filter to specific participants")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature caching")
    parser.add_argument("--skip-novel", action="store_true", help="Skip novel strategies (standard pipeline only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    start_time = time.time()
    logger.info("TONES Voice-Based Glucose Estimation — Novel Strategy Pipeline v2.0")
    logger.info("=" * 70)

    # Load config
    cfg = load_config(args.config)
    logger.info("Config loaded from: %s", cfg["base_dir"])

    # Phase 1: Build dataset (with opus/waptt auto-conversion)
    canonical_df = phase_build_dataset(cfg, args.participants)
    if canonical_df.empty:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # Save canonical CSV
    output_dir = get_base_dir(cfg) / cfg.get("output", {}).get("dir", "output")
    output_dir.mkdir(exist_ok=True)
    canonical_path = output_dir / "canonical_dataset.csv"
    canonical_df.to_csv(canonical_path, index=False)
    logger.info("Canonical dataset saved: %s (%d rows)", canonical_path, len(canonical_df))

    if args.data_only:
        logger.info("--data-only: Stopping after dataset build.")
        return

    # Phase 2: Feature extraction (MFCC + voice quality)
    data_by_participant = phase_extract_features(
        canonical_df, cfg, use_cache=not args.no_cache
    )

    # Phase 3: Within-speaker normalization
    data_by_participant = phase_normalize(data_by_participant, cfg)

    # Phase 4: Temporal context features
    data_by_participant = phase_temporal_features(data_by_participant, cfg)

    # Phase 5: Standard personalized models
    personalized_results = phase_personalized(data_by_participant, cfg)

    # Phase 6: Novel strategies
    novel_results = {}
    if not args.skip_novel:
        novel_results = phase_novel_strategies(data_by_participant, cfg)

    # Phase 7: Population model
    population_results = phase_population(data_by_participant, cfg)

    # Phase 8: Output
    phase_output(personalized_results, population_results, novel_results, cfg)

    elapsed = time.time() - start_time
    logger.info("\nTotal time: %.1f seconds", elapsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Pipeline crashed: %s", e, exc_info=True)
        sys.exit(1)
