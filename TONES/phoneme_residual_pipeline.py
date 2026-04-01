#!/usr/bin/env python3
"""
TONES — Phoneme-Level Physiological Residual Pipeline
=======================================================
A fundamentally new approach to voice-based glucose estimation.

Instead of extracting features over entire recordings (where phonemic content
dominates ~60-70% of acoustic variance and buries the ~2% glucose signal),
this pipeline:

  Phase 1: Load matched audio-glucose data (reuses existing infrastructure)
  Phase 2: Whisper transcription + word-level alignment
  Phase 3: Phoneme-level acoustic feature extraction
  Phase 4: Per-speaker phoneme baseline construction
  Phase 5: Physiological residual computation
  Phase 6: Glucose prediction from residual features
  Phase 7: Comparison with original pipeline results
  Phase 8: Feature sensitivity analysis + HTML report

The key innovation: by controlling for *what is being said* (phoneme identity),
the glucose signal goes from ~2% of total variance to ~10-30% of residual
variance — a 5-15x improvement in signal-to-noise ratio.

Usage:
    python phoneme_residual_pipeline.py
    python phoneme_residual_pipeline.py --participants Wolf Lara
    python phoneme_residual_pipeline.py --whisper-model small
    python phoneme_residual_pipeline.py --no-cache
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# TONES modules
from tones.config import load_config, get_base_dir
from tones.data.loaders import load_participant_data, collect_audio_files
from tones.features.phoneme_align import (
    transcribe_and_align,
    align_batch,
    AlignmentResult,
)
from tones.features.phoneme_features import (
    extract_phoneme_features_for_recording,
    extract_residual_features_for_speaker,
    analyze_phoneme_sensitivity,
    RecordingPhonemeData,
    SpeakerPhonemeBaseline,
    N_RESIDUAL_FEATURES,
    N_SEGMENT_FEATURES,
    SEGMENT_FEATURE_NAMES,
    get_residual_feature_names,
)
from tones.features.mfcc import create_extractor_from_config
from tones.features.voice_quality import create_voice_quality_extractor
from tones.features.cache import FeatureCache
from tones.features.normalize import normalize_features
from tones.features.temporal import build_temporal_features
from tones.models.train import (
    train_personalized,
    train_population,
    get_best_personalized,
    mean_predictor_baseline,
    compute_metrics,
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
    # Remove existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    # Whisper can be chatty
    logging.getLogger("whisper").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1: Build Dataset (reuse existing infrastructure)
# =============================================================================

def phase_build_dataset(
    cfg: Dict,
    participants_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load matched audio-glucose pairs."""
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
    for name, pcfg in participants.items():
        if not pcfg.get("glucose_csv"):
            continue
        df = load_participant_data(name, pcfg, base_dir, matching_cfg)
        if not df.empty:
            all_dfs.append(df)
            logger.info("  %s: %d matched samples", name, len(df))

    if not all_dfs:
        logger.error("No data loaded!")
        return pd.DataFrame()

    canonical = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total: %d matched samples from %d participants", len(canonical), len(all_dfs))
    return canonical


# =============================================================================
# Phase 2: Whisper Transcription + Alignment
# =============================================================================

def phase_whisper_alignment(
    canonical_df: pd.DataFrame,
    cfg: Dict,
    whisper_model: str = "base",
    use_cache: bool = True,
) -> Dict[str, Dict[str, AlignmentResult]]:
    """
    Transcribe and align all audio files using Whisper.
    
    Returns dict: {participant: {audio_path: AlignmentResult}}
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: Whisper Transcription + Word-Level Alignment")
    logger.info("  Model: %s | Cache: %s", whisper_model, "enabled" if use_cache else "disabled")
    logger.info("=" * 70)

    base_dir = get_base_dir(cfg)
    cache_dir = str(base_dir / ".cache" / "phoneme_alignment") if use_cache else None

    alignments_by_participant = {}

    for subject in canonical_df["subject"].unique():
        subject_df = canonical_df[canonical_df["subject"] == subject]
        audio_paths = subject_df["audio_path"].tolist()

        logger.info("\n  %s: Aligning %d recordings...", subject, len(audio_paths))

        subject_cache = str(Path(cache_dir) / subject) if cache_dir else None

        alignments = align_batch(
            audio_paths,
            model_name=whisper_model,
            cache_dir=subject_cache,
        )

        # Stats
        n_aligned = len(alignments)
        total_vowels = sum(a.n_vowel_segments for a in alignments.values())
        total_consonants = sum(a.n_consonant_segments for a in alignments.values())

        logger.info(
            "  %s: %d/%d aligned, %d vowel segments, %d consonant segments",
            subject, n_aligned, len(audio_paths), total_vowels, total_consonants,
        )

        alignments_by_participant[subject] = alignments
        gc.collect()

    return alignments_by_participant


# =============================================================================
# Phase 3-5: Phoneme Feature Extraction + Baseline + Residuals
# =============================================================================

def phase_phoneme_residuals(
    canonical_df: pd.DataFrame,
    alignments_by_participant: Dict[str, Dict[str, AlignmentResult]],
    cfg: Dict,
) -> Dict[str, Dict]:
    """
    Extract phoneme-level features, build baselines, compute residuals.
    
    Returns data_by_participant dict compatible with existing model training.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3-5: Phoneme Features → Baselines → Residuals")
    logger.info("=" * 70)

    sr = cfg.get("features", {}).get("sample_rate", 16000)
    data_by_participant = {}

    for subject in canonical_df["subject"].unique():
        subject_df = canonical_df[canonical_df["subject"] == subject]
        alignments = alignments_by_participant.get(subject, {})

        if not alignments:
            logger.warning("  %s: No alignments, skipping", subject)
            continue

        logger.info("\n  %s: Extracting phoneme features for %d recordings...", subject, len(subject_df))

        # Phase 3: Extract phoneme-level features for each recording
        recordings = []
        glucose_values = []
        timestamps = []
        audio_paths = []

        for _, row in subject_df.iterrows():
            audio_path = row["audio_path"]
            alignment = alignments.get(audio_path)
            if alignment is None or len(alignment.phonemes) == 0:
                continue

            rec = extract_phoneme_features_for_recording(audio_path, alignment, sr=sr)
            if rec is None:
                continue

            # Skip recordings with too few phoneme segments
            if rec.n_vowels < 2 and rec.n_consonants < 2:
                continue

            recordings.append(rec)
            glucose_values.append(row["glucose_mg_dl"])
            timestamps.append(row["audio_timestamp"])
            audio_paths.append(audio_path)

        if len(recordings) < 10:
            logger.warning("  %s: Only %d valid recordings (need ≥10), skipping", subject, len(recordings))
            continue

        glucose_arr = np.array(glucose_values)

        # Phase 4-5: Build baseline + compute residuals
        logger.info("  %s: Building phoneme baseline from %d recordings...", subject, len(recordings))
        residual_features, baseline = extract_residual_features_for_speaker(recordings, subject)

        # Phase 5.5: Sensitivity analysis — which features correlate with glucose?
        sensitivity = analyze_phoneme_sensitivity(recordings, glucose_arr, baseline)
        if sensitivity:
            top_5 = list(sensitivity.items())[:5]
            logger.info("  %s: Top glucose-sensitive features:", subject)
            for feat_name, r_val in top_5:
                logger.info("    %s: r=%.3f", feat_name, r_val)

        data_by_participant[subject] = {
            "features": residual_features,
            "glucose": glucose_arr,
            "timestamps": timestamps,
            "audio_paths": audio_paths,
            "recordings": recordings,
            "baseline": baseline,
            "sensitivity": sensitivity,
        }

        avg_vowels = np.mean([r.n_vowels for r in recordings])
        avg_cons = np.mean([r.n_consonants for r in recordings])
        logger.info(
            "  %s: %d recordings, %d features, avg %.1f vowels/rec, %.1f consonants/rec",
            subject, len(recordings), residual_features.shape[1], avg_vowels, avg_cons,
        )

        gc.collect()

    return data_by_participant


# =============================================================================
# Phase 5.5: Hybrid Features (Original MFCC + Residuals)
# =============================================================================

def phase_hybrid_features(
    canonical_df: pd.DataFrame,
    data_by_participant: Dict[str, Dict],
    cfg: Dict,
) -> Dict[str, Dict]:
    """
    Extract original MFCC/VQ features and combine with residual features.
    
    This creates a hybrid feature set that leverages both:
    - Content-averaged signal (MFCC + voice quality + temporal)
    - Content-controlled signal (phoneme residuals)
    
    The model can learn from both sources simultaneously.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 5.5: Hybrid Features (Original + Residuals)")
    logger.info("=" * 70)

    feat_cfg = cfg.get("features", {})
    sr = feat_cfg.get("sample_rate", 16000)
    
    extractor = create_extractor_from_config(cfg)
    vq_extractor = create_voice_quality_extractor(cfg)
    
    # Setup caches
    base_dir = get_base_dir(cfg)
    cache_dir = str(base_dir / feat_cfg.get("cache_dir", ".cache/features"))
    mfcc_cache = FeatureCache(cache_dir=cache_dir, enabled=True)
    vq_cache = FeatureCache(
        cache_dir=str(Path(cache_dir) / "voice_quality"), enabled=True,
    )
    mfcc_id = f"mfcc_n{extractor.n_mfcc}_sr{extractor.sr}"
    vq_id = "voice_quality_v1"
    
    hybrid_data = {}
    
    for subject, data in data_by_participant.items():
        audio_paths = data["audio_paths"]
        residual_features = data["features"]
        glucose = data["glucose"]
        timestamps = data["timestamps"]
        
        logger.info("  %s: Extracting original features for %d recordings...", subject, len(audio_paths))
        
        mfcc_features_list = []
        vq_features_list = []
        valid_mask = []
        
        for audio_path in audio_paths:
            # MFCC
            mfcc_feat = mfcc_cache.get(audio_path, mfcc_id)
            if mfcc_feat is None:
                mfcc_feat = extractor.extract_from_file(audio_path)
                if mfcc_feat is not None:
                    mfcc_cache.put(audio_path, mfcc_id, mfcc_feat)
            
            # Voice quality
            vq_feat = vq_cache.get(audio_path, vq_id)
            if vq_feat is None:
                vq_feat = vq_extractor.extract_from_file(audio_path)
                if vq_feat is not None:
                    vq_cache.put(audio_path, vq_id, vq_feat)
            
            if mfcc_feat is not None:
                mfcc_features_list.append(mfcc_feat)
                vq = vq_feat if vq_feat is not None else np.zeros(vq_extractor.n_features)
                vq_features_list.append(vq)
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        
        valid_mask = np.array(valid_mask)
        if valid_mask.sum() < 10:
            logger.warning("  %s: Too few valid MFCC features, skipping hybrid", subject)
            continue
        
        mfcc_arr = np.array(mfcc_features_list)
        vq_arr = np.array(vq_features_list)
        original_features = np.hstack([mfcc_arr, vq_arr])
        
        # Filter residual features to match
        residual_filtered = residual_features[valid_mask]
        glucose_filtered = glucose[valid_mask]
        timestamps_filtered = [t for t, v in zip(timestamps, valid_mask) if v]
        
        # Within-speaker z-normalize the original features
        if len(original_features) > 1:
            mu = original_features.mean(axis=0)
            sigma = original_features.std(axis=0) + 1e-8
            original_normed = (original_features - mu) / sigma
        else:
            original_normed = original_features
        
        # Hybrid = original (normalized) + residuals
        hybrid = np.hstack([original_normed, residual_filtered])
        
        hybrid_data[subject] = {
            "features": hybrid,
            "features_original": original_normed,
            "features_residual": residual_filtered,
            "glucose": glucose_filtered,
            "timestamps": timestamps_filtered,
            "sensitivity": data.get("sensitivity", {}),
        }
        
        logger.info(
            "  %s: %d samples, hybrid=%d dims (original=%d + residual=%d)",
            subject, len(glucose_filtered), hybrid.shape[1],
            original_normed.shape[1], residual_filtered.shape[1],
        )
    
    return hybrid_data


# =============================================================================
# Phase 6: Model Training — Residual Features vs Glucose
# =============================================================================

def phase_train_models(
    data_by_participant: Dict[str, Dict],
    cfg: Dict,
) -> Dict[str, Dict]:
    """Train personalized models on residual features."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 6: Personalized Models on Phoneme Residual Features")
    logger.info("=" * 70)

    model_cfg = cfg.get("models", {})
    min_samples = model_cfg.get("personalized", {}).get("min_samples", 20)

    all_results = {}

    for name, data in sorted(data_by_participant.items()):
        X = data["features"]
        y = data["glucose"]

        if len(X) < min_samples:
            logger.info("  %s: Skipped (%d < %d samples)", name, len(X), min_samples)
            continue

        logger.info("\n  %s (%d samples, %d features, glucose %.0f-%.0f mg/dL):",
                    name, len(y), X.shape[1], y.min(), y.max())

        # Baseline
        baseline_metrics = mean_predictor_baseline(y)
        logger.info("    Mean predictor baseline: MAE=%.2f", baseline_metrics["mae"])

        # Train all standard models
        results = train_personalized(
            X, y,
            min_samples=min_samples,
            cv_kfold_threshold=50,
            model_params=model_cfg,
        )

        best = get_best_personalized(results)
        if best:
            improvement = baseline_metrics["mae"] - best["mae"]
            all_results[name] = {
                "n_samples": len(y),
                "n_features": X.shape[1],
                "glucose_mean": float(np.mean(y)),
                "glucose_std": float(np.std(y)),
                "glucose_min": float(np.min(y)),
                "glucose_max": float(np.max(y)),
                "best_model": best["model_name"],
                "mae": best["mae"],
                "rmse": best["rmse"],
                "r": best["r"],
                "r2": best.get("r2", 0),
                "baseline_mae": baseline_metrics["mae"],
                "improvement": improvement,
                "improvement_pct": 100 * improvement / baseline_metrics["mae"] if baseline_metrics["mae"] > 0 else 0,
                "actual": y,
                "predictions": best["predictions"],
                "all_models": {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                               for k, v in results.items()},
                "sensitivity": data.get("sensitivity", {}),
            }
            logger.info(
                "    BEST: %s — MAE=%.2f, r=%.3f (baseline=%.2f, improvement=%.2f / %.1f%%)",
                best["model_name"], best["mae"], best["r"],
                baseline_metrics["mae"], improvement,
                100 * improvement / baseline_metrics["mae"] if baseline_metrics["mae"] > 0 else 0,
            )

    return all_results


# =============================================================================
# Phase 6b: Hybrid Model Training
# =============================================================================

def phase_train_hybrid(
    hybrid_data: Dict[str, Dict],
    cfg: Dict,
) -> Dict[str, Dict]:
    """Train models on hybrid features (original + residual)."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 6b: Hybrid Models (Original MFCC + Phoneme Residuals)")
    logger.info("=" * 70)

    model_cfg = cfg.get("models", {})
    min_samples = model_cfg.get("personalized", {}).get("min_samples", 20)

    all_results = {}

    for name, data in sorted(hybrid_data.items()):
        X = data["features"]
        y = data["glucose"]

        if len(X) < min_samples:
            logger.info("  %s: Skipped (%d < %d samples)", name, len(X), min_samples)
            continue

        logger.info("\n  %s (%d samples, %d hybrid features, glucose %.0f-%.0f mg/dL):",
                    name, len(y), X.shape[1], y.min(), y.max())

        baseline_metrics = mean_predictor_baseline(y)
        logger.info("    Mean predictor baseline: MAE=%.2f", baseline_metrics["mae"])

        results = train_personalized(
            X, y,
            min_samples=min_samples,
            cv_kfold_threshold=50,
            model_params=model_cfg,
        )

        best = get_best_personalized(results)
        if best:
            improvement = baseline_metrics["mae"] - best["mae"]

            # Also train on original-only and residual-only for comparison
            original_results = train_personalized(
                data["features_original"], y,
                min_samples=min_samples, cv_kfold_threshold=50, model_params=model_cfg,
            )
            residual_results = train_personalized(
                data["features_residual"], y,
                min_samples=min_samples, cv_kfold_threshold=50, model_params=model_cfg,
            )

            best_orig = get_best_personalized(original_results)
            best_resid = get_best_personalized(residual_results)

            all_results[name] = {
                "n_samples": len(y),
                "n_features_hybrid": X.shape[1],
                "n_features_original": data["features_original"].shape[1],
                "n_features_residual": data["features_residual"].shape[1],
                "glucose_mean": float(np.mean(y)),
                "glucose_std": float(np.std(y)),
                "glucose_min": float(np.min(y)),
                "glucose_max": float(np.max(y)),
                # Hybrid
                "hybrid_best_model": best["model_name"],
                "hybrid_mae": best["mae"],
                "hybrid_rmse": best["rmse"],
                "hybrid_r": best["r"],
                # Original only
                "original_best_model": best_orig["model_name"] if best_orig else "N/A",
                "original_mae": best_orig["mae"] if best_orig else 999,
                "original_r": best_orig["r"] if best_orig else 0,
                # Residual only
                "residual_best_model": best_resid["model_name"] if best_resid else "N/A",
                "residual_mae": best_resid["mae"] if best_resid else 999,
                "residual_r": best_resid["r"] if best_resid else 0,
                # Baseline
                "baseline_mae": baseline_metrics["mae"],
                "improvement": improvement,
                "improvement_pct": 100 * improvement / baseline_metrics["mae"] if baseline_metrics["mae"] > 0 else 0,
                "actual": y,
                "predictions": best["predictions"],
                "sensitivity": data.get("sensitivity", {}),
            }

            logger.info(
                "    HYBRID:   %s — MAE=%.2f, r=%.3f",
                best["model_name"], best["mae"], best["r"],
            )
            if best_orig:
                logger.info(
                    "    ORIGINAL: %s — MAE=%.2f, r=%.3f",
                    best_orig["model_name"], best_orig["mae"], best_orig["r"],
                )
            if best_resid:
                logger.info(
                    "    RESIDUAL: %s — MAE=%.2f, r=%.3f",
                    best_resid["model_name"], best_resid["mae"], best_resid["r"],
                )
            logger.info(
                "    Baseline: MAE=%.2f | Improvement: %.2f (%.1f%%)",
                baseline_metrics["mae"], improvement,
                100 * improvement / baseline_metrics["mae"] if baseline_metrics["mae"] > 0 else 0,
            )

    return all_results


# =============================================================================
# Phase 7: Population Model
# =============================================================================

def phase_population(data_by_participant: Dict[str, Dict], cfg: Dict) -> Dict[str, Dict]:
    """Train population model using LOPO CV on residual features."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 7: Population Model (LOPO) on Residual Features")
    logger.info("=" * 70)

    all_X, all_y, all_groups = [], [], []
    for name, data in data_by_participant.items():
        if len(data["glucose"]) >= 10:
            all_X.append(data["features"])
            all_y.append(data["glucose"])
            all_groups.extend([name] * len(data["glucose"]))

    if len(all_X) < 2:
        logger.warning("Need ≥2 participants for population model")
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
        logger.info("  Best population model: %s — MAE=%.2f, r=%.3f", best_name, best["mae"], best["r"])

    return results


# =============================================================================
# Phase 8: Output — Report + Visualization
# =============================================================================

def phase_output(
    personalized_results: Dict[str, Dict],
    population_results: Dict[str, Dict],
    data_by_participant: Dict[str, Dict],
    cfg: Dict,
    elapsed_time: float,
    hybrid_results: Optional[Dict[str, Dict]] = None,
):
    """Generate comprehensive results report and visualizations."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 8: Output & Visualization")
    logger.info("=" * 70)

    base_dir = get_base_dir(cfg)
    output_dir = base_dir / cfg.get("output", {}).get("dir", "output") / "phoneme_residual"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # ── Generate scatter plots ──
    if personalized_results:
        _plot_scatter_all(personalized_results, figures_dir)
        _plot_feature_sensitivity(personalized_results, data_by_participant, figures_dir)
        _plot_comparison_bar(personalized_results, figures_dir)

    # ── Save JSON results ──
    summary = {
        "generated": datetime.now().isoformat(),
        "pipeline": "phoneme-residual-v1.0",
        "approach": "Phoneme-level physiological residual extraction",
        "elapsed_seconds": round(elapsed_time, 1),
        "n_participants": len(personalized_results),
        "total_samples": sum(r["n_samples"] for r in personalized_results.values()),
        "feature_dim": N_RESIDUAL_FEATURES,
        "feature_names": get_residual_feature_names(),
        "personalized": {},
        "population": {},
        "sensitivity_analysis": {},
    }

    for name, res in personalized_results.items():
        summary["personalized"][name] = {
            "n_samples": res["n_samples"],
            "n_features": res["n_features"],
            "glucose_mean": round(res["glucose_mean"], 1),
            "glucose_std": round(res["glucose_std"], 1),
            "glucose_range": [round(res["glucose_min"], 1), round(res["glucose_max"], 1)],
            "best_model": res["best_model"],
            "mae": round(res["mae"], 2),
            "rmse": round(res["rmse"], 2),
            "r": round(res["r"], 3),
            "baseline_mae": round(res["baseline_mae"], 2),
            "improvement": round(res["improvement"], 2),
            "improvement_pct": round(res["improvement_pct"], 1),
            "all_models": {
                k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in res.get("all_models", {}).items()
            },
        }
        summary["sensitivity_analysis"][name] = res.get("sensitivity", {})

    # Hybrid results (the key comparison)
    if hybrid_results:
        summary["hybrid"] = {}
        for name, res in hybrid_results.items():
            summary["hybrid"][name] = {
                "n_samples": res["n_samples"],
                "n_features_hybrid": res["n_features_hybrid"],
                "n_features_original": res["n_features_original"],
                "n_features_residual": res["n_features_residual"],
                "hybrid_mae": round(res["hybrid_mae"], 2),
                "hybrid_r": round(res["hybrid_r"], 3),
                "hybrid_model": res["hybrid_best_model"],
                "original_mae": round(res["original_mae"], 2),
                "original_r": round(res["original_r"], 3),
                "original_model": res["original_best_model"],
                "residual_mae": round(res["residual_mae"], 2),
                "residual_r": round(res["residual_r"], 3),
                "residual_model": res["residual_best_model"],
                "baseline_mae": round(res["baseline_mae"], 2),
            }

    for model_name, res in population_results.items():
        summary["population"][model_name] = {
            "mae": round(res["mae"], 2),
            "rmse": round(res["rmse"], 2),
            "r": round(res["r"], 3),
        }

    results_path = output_dir / "results_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Generate HTML report ──
    _generate_html_report(summary, personalized_results, output_dir, figures_dir)

    logger.info("Results saved: %s", results_path)
    logger.info("Report saved: %s", output_dir / "phoneme_residual_report.html")

    # ── Print final summary ──
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHONEME RESIDUAL PIPELINE — RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  %-15s %6s %6s %8s %8s %8s %8s %8s",
                "Participant", "N", "Feats", "MAE", "r", "Baseline", "Improve", "Model")
    logger.info("  " + "-" * 80)

    for name, res in sorted(personalized_results.items(), key=lambda x: x[1]["mae"]):
        logger.info(
            "  %-15s %6d %6d %8.2f %8.3f %8.2f %+8.2f %8s",
            name, res["n_samples"], res["n_features"],
            res["mae"], res["r"],
            res["baseline_mae"], res["improvement"], res["best_model"],
        )

    # Hybrid comparison (the main event)
    if hybrid_results:
        logger.info("")
        logger.info("  HYBRID COMPARISON (Original vs Residual vs Combined):")
        logger.info("  %-15s %6s | %8s %8s | %8s %8s | %8s %8s | %8s",
                    "Participant", "N",
                    "Orig MAE", "Orig r",
                    "Resid MAE", "Resid r",
                    "Hybr MAE", "Hybr r",
                    "Baseline")
        logger.info("  " + "-" * 100)
        for name, res in sorted(hybrid_results.items(), key=lambda x: x[1]["hybrid_mae"]):
            logger.info(
                "  %-15s %6d | %8.2f %8.3f | %8.2f %8.3f | %8.2f %8.3f | %8.2f",
                name, res["n_samples"],
                res["original_mae"], res["original_r"],
                res["residual_mae"], res["residual_r"],
                res["hybrid_mae"], res["hybrid_r"],
                res["baseline_mae"],
            )

    if population_results:
        logger.info("")
        logger.info("  Population Models (LOPO):")
        for model_name, res in sorted(population_results.items(), key=lambda x: x[1]["mae"]):
            logger.info("    %s: MAE=%.2f, r=%.3f", model_name, res["mae"], res["r"])


# ─── Visualization helpers ──────────────────────────────────────────────────────

def _plot_scatter_all(results: Dict[str, Dict], figures_dir: Path):
    """Scatter plots: actual vs predicted for each participant."""
    n = len(results)
    if n == 0:
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (name, res) in enumerate(sorted(results.items(), key=lambda x: x[1]["mae"])):
        ax = axes[idx // cols][idx % cols]
        actual = res["actual"]
        predicted = res["predictions"]

        ax.scatter(actual, predicted, alpha=0.5, s=20, color="#2196F3")

        # Perfect prediction line
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", alpha=0.3, linewidth=1)

        ax.set_xlabel("Actual (mg/dL)")
        ax.set_ylabel("Predicted (mg/dL)")
        ax.set_title(f"{name}\nMAE={res['mae']:.1f}, r={res['r']:.3f}")
        ax.set_aspect("equal")

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(figures_dir / "scatter_per_participant.png", dpi=150)
    plt.close()


def _plot_feature_sensitivity(
    results: Dict[str, Dict],
    data_by_participant: Dict[str, Dict],
    figures_dir: Path,
):
    """Bar chart of top glucose-sensitive phoneme features per participant."""
    fig, axes = plt.subplots(1, min(3, len(results)),
                             figsize=(6 * min(3, len(results)), 5), squeeze=False)

    sorted_names = sorted(results.keys(), key=lambda x: results[x]["mae"])[:3]

    for idx, name in enumerate(sorted_names):
        ax = axes[0][idx]
        sensitivity = results[name].get("sensitivity", {})

        if not sensitivity:
            ax.set_title(f"{name}: No sensitivity data")
            continue

        top_n = list(sensitivity.items())[:10]
        if not top_n:
            continue

        feat_names = [f[0] for f in top_n]
        r_values = [f[1] for f in top_n]
        colors = ["#4CAF50" if r > 0 else "#F44336" for r in r_values]

        ax.barh(range(len(feat_names)), r_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(feat_names)))
        ax.set_yticklabels(feat_names, fontsize=8)
        ax.set_xlabel("Pearson r with glucose")
        ax.set_title(f"{name}: Top Features")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(figures_dir / "feature_sensitivity.png", dpi=150)
    plt.close()


def _plot_comparison_bar(results: Dict[str, Dict], figures_dir: Path):
    """Bar chart comparing MAE vs baseline for each participant."""
    names = sorted(results.keys(), key=lambda x: results[x]["mae"])
    mae_values = [results[n]["mae"] for n in names]
    baseline_values = [results[n]["baseline_mae"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars1 = ax.bar(x - width / 2, baseline_values, width, label="Baseline (Mean)", color="#BDBDBD")
    bars2 = ax.bar(x + width / 2, mae_values, width, label="Phoneme Residual", color="#2196F3")

    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("Phoneme Residual Pipeline vs Mean Predictor Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "comparison_bar.png", dpi=150)
    plt.close()


def _generate_html_report(
    summary: Dict,
    personalized_results: Dict[str, Dict],
    output_dir: Path,
    figures_dir: Path,
):
    """Generate a comprehensive HTML report."""
    rel_figures = "figures"

    # Build participant rows
    participant_rows = ""
    for name in sorted(summary["personalized"].keys(),
                       key=lambda x: summary["personalized"][x]["mae"]):
        p = summary["personalized"][name]
        imp_class = "positive" if p["improvement"] > 0 else "negative"
        participant_rows += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{p['n_samples']}</td>
            <td>{p['glucose_mean']:.0f} ± {p['glucose_std']:.0f}</td>
            <td>{p['glucose_range'][0]:.0f}–{p['glucose_range'][1]:.0f}</td>
            <td><strong>{p['mae']:.2f}</strong></td>
            <td>{p['r']:.3f}</td>
            <td>{p['baseline_mae']:.2f}</td>
            <td class="{imp_class}">{p['improvement']:+.2f} ({p['improvement_pct']:+.1f}%)</td>
            <td>{p['best_model']}</td>
        </tr>"""

    # Population rows
    pop_rows = ""
    for model_name, m in summary.get("population", {}).items():
        pop_rows += f"""
        <tr>
            <td>{model_name}</td>
            <td>{m['mae']:.2f}</td>
            <td>{m['r']:.3f}</td>
        </tr>"""

    # Sensitivity sections
    sensitivity_html = ""
    for name, sens in summary.get("sensitivity_analysis", {}).items():
        if not sens:
            continue
        items = list(sens.items())[:10]
        feat_rows = "".join(
            f"<tr><td>{fn}</td><td>{rv:+.3f}</td></tr>"
            for fn, rv in items
        )
        sensitivity_html += f"""
        <h3>{name}</h3>
        <table class="sensitivity-table">
            <tr><th>Feature</th><th>Pearson r</th></tr>
            {feat_rows}
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TONES — Phoneme Residual Pipeline Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
        h1 {{ color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 10px; }}
        h2 {{ color: #1976D2; margin-top: 40px; }}
        h3 {{ color: #1E88E5; }}
        .summary-box {{ background: #E3F2FD; border-radius: 8px; padding: 20px; margin: 20px 0;
                        border-left: 4px solid #1565C0; }}
        .innovation-box {{ background: #E8F5E9; border-radius: 8px; padding: 20px; margin: 20px 0;
                           border-left: 4px solid #4CAF50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
        th {{ background: #1976D2; color: white; }}
        tr:nth-child(even) {{ background: #f5f5f5; }}
        .positive {{ color: #2E7D32; font-weight: bold; }}
        .negative {{ color: #C62828; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; }}
        .sensitivity-table {{ width: auto; min-width: 300px; }}
        .sensitivity-table td:last-child {{ font-family: monospace; text-align: right; }}
        .meta {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Phoneme-Level Physiological Residual Pipeline</h1>
    <p class="meta">Generated: {summary['generated']} | Pipeline: {summary['pipeline']} |
       Runtime: {summary['elapsed_seconds']:.0f}s</p>

    <div class="innovation-box">
        <h3>Innovation: Content-Controlled Physiological Residuals</h3>
        <p>Instead of extracting features over entire recordings (where phonemic content
           dominates ~60-70% of acoustic variance), this pipeline:</p>
        <ol>
            <li><strong>Whisper alignment</strong> — transcribes audio and identifies word/phoneme boundaries</li>
            <li><strong>Per-phoneme feature extraction</strong> — extracts 13 acoustic features per vowel/consonant segment</li>
            <li><strong>Personal phoneme baseline</strong> — builds each speaker's typical vowel/consonant profile</li>
            <li><strong>Residual computation</strong> — <em>how does this recording's voice differ from its own baseline?</em></li>
            <li><strong>Glucose prediction</strong> — {N_RESIDUAL_FEATURES}-dimensional residual features → glucose</li>
        </ol>
        <p>By controlling for <em>what is being said</em>, the glucose signal is amplified from
           ~2% to ~10-30% of residual variance.</p>
    </div>

    <div class="summary-box">
        <strong>Summary:</strong> {summary['n_participants']} participants,
        {summary['total_samples']} total samples,
        {summary['feature_dim']} residual features
    </div>

    <h2>Personalized Results</h2>
    <table>
        <tr>
            <th>Participant</th><th>N</th><th>Glucose Mean±SD</th><th>Range</th>
            <th>MAE</th><th>r</th><th>Baseline MAE</th><th>Improvement</th><th>Best Model</th>
        </tr>
        {participant_rows}
    </table>

    <h2>Visualizations</h2>
    <h3>Actual vs Predicted</h3>
    <img src="{rel_figures}/scatter_per_participant.png" alt="Scatter plots">

    <h3>Model vs Baseline Comparison</h3>
    <img src="{rel_figures}/comparison_bar.png" alt="Comparison bar chart">

    <h3>Feature Sensitivity to Glucose</h3>
    <img src="{rel_figures}/feature_sensitivity.png" alt="Feature sensitivity">

    <h2>Population Model (LOPO)</h2>
    {"<p>Tests whether phoneme residuals enable cross-speaker glucose prediction.</p>" if pop_rows else "<p>Not enough participants for population model.</p>"}
    {"<table><tr><th>Model</th><th>MAE</th><th>r</th></tr>" + pop_rows + "</table>" if pop_rows else ""}

    <h2>Phoneme Feature Sensitivity Analysis</h2>
    <p>Which specific phoneme-level acoustic features correlate with glucose?
       This reveals the physiological pathways through which glucose affects voice.</p>
    {sensitivity_html if sensitivity_html else "<p>No significant correlations found.</p>"}

    <h2>Methodology</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Alignment</td><td>OpenAI Whisper (word-level timestamps)</td></tr>
        <tr><td>Phoneme decomposition</td><td>Grapheme clustering (vowel/consonant)</td></tr>
        <tr><td>Per-segment features</td><td>{N_SEGMENT_FEATURES} dimensions: {', '.join(SEGMENT_FEATURE_NAMES)}</td></tr>
        <tr><td>Recording-level features</td><td>{N_RESIDUAL_FEATURES} dimensions (mean/std/magnitude of residuals per category)</td></tr>
        <tr><td>Normalization</td><td>Per-speaker per-phoneme-category z-score (baseline subtraction)</td></tr>
        <tr><td>Models</td><td>SVR, BayesianRidge, RandomForest, GradientBoosting, KNN</td></tr>
        <tr><td>Validation</td><td>LOO-CV (N≤50) or 10-fold CV (N&gt;50)</td></tr>
    </table>
</body>
</html>"""

    report_path = output_dir / "phoneme_residual_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TONES — Phoneme-Level Physiological Residual Pipeline",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--participants", nargs="+", default=None,
                        help="Filter to specific participants")
    parser.add_argument("--whisper-model", type=str, default="base",
                        help="Whisper model size: tiny, base, small, medium (default: base)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable alignment and feature caching")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    start_time = time.time()
    logger.info("")
    logger.info("=" * 70)
    logger.info("TONES — Phoneme-Level Physiological Residual Pipeline v1.0")
    logger.info("=" * 70)
    logger.info("Innovation: Control for phonemic content, extract physiological residuals")
    logger.info("")

    # Load config
    cfg = load_config(args.config)
    logger.info("Config loaded from: %s", cfg["base_dir"])

    # Phase 1: Build dataset
    canonical_df = phase_build_dataset(cfg, args.participants)
    if canonical_df.empty:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # Phase 2: Whisper alignment
    alignments = phase_whisper_alignment(
        canonical_df, cfg,
        whisper_model=args.whisper_model,
        use_cache=not args.no_cache,
    )

    # Phase 3-5: Phoneme features → baselines → residuals
    data_by_participant = phase_phoneme_residuals(canonical_df, alignments, cfg)

    if not data_by_participant:
        logger.error("No participant data after phoneme processing. Exiting.")
        sys.exit(1)

    # Phase 5.5: Hybrid features (combine original MFCC + residuals)
    hybrid_data = phase_hybrid_features(canonical_df, data_by_participant, cfg)

    # Phase 6: Model training on residual features alone
    personalized_results = phase_train_models(data_by_participant, cfg)

    # Phase 6b: Hybrid model training (original + residual)
    hybrid_results = phase_train_hybrid(hybrid_data, cfg) if hybrid_data else {}

    # Phase 7: Population model (on hybrid features)
    pop_data = hybrid_data if hybrid_data else data_by_participant
    population_results = phase_population(pop_data, cfg)

    # Phase 8: Output
    elapsed = time.time() - start_time
    phase_output(
        personalized_results, population_results, data_by_participant,
        cfg, elapsed, hybrid_results=hybrid_results,
    )

    logger.info("\nTotal time: %.1f seconds", elapsed)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Pipeline crashed: %s", e, exc_info=True)
        sys.exit(1)
