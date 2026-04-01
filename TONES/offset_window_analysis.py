#!/usr/bin/env python3
"""
TONES — Offset & Window Analysis
==================================
Two analyses that were missing from the hyperparameter sweep:

1. **Offset Sweep** (-30 to +30 min in 5-min steps per participant)
   Tests whether voice features detect glucose changes faster than CGM.
   CGM measures interstitial glucose with a known 5-15 min lag.
   If optimal offset is positive, voice "leads" CGM.

2. **Matching Window Sweep** (±5 to ±30 min)
   Tests how the CGM-to-voice matching window affects results.
   Smaller windows = stricter matching but fewer samples.

3. **Audio Segmentation Window** (different n_fft / hop_length)
   Tests how spectral resolution affects feature quality.

Usage:
    python offset_window_analysis.py
    python offset_window_analysis.py --participants Wolf Lara Sybille
    python offset_window_analysis.py --fast   # Quick run with fewer offsets
"""

import sys
import gc
import json
import time
import base64
import argparse
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_predict, LeaveOneOut, KFold

sys.path.insert(0, ".")
from tones.config import load_config, get_base_dir
from tones.data.loaders import (
    load_glucose_csv,
    collect_audio_files,
    parse_timestamp_from_filename,
    find_matching_glucose,
)
from tones.features.mfcc import MFCCExtractor
from tones.models.train import get_model, compute_metrics, mean_predictor_baseline
import librosa


# =============================================================================
# Configuration
# =============================================================================

OFFSET_RANGE = range(-30, 31, 5)       # -30 to +30 in 5-min steps
MATCHING_WINDOWS = [5, 10, 15, 20, 30]  # minutes
NFFT_SIZES = [512, 1024, 2048, 4096]    # librosa n_fft
HOP_FRACTIONS = [0.25, 0.5, 0.75]       # hop_length as fraction of n_fft
MODEL_NAMES = ["Ridge", "SVR", "BayesianRidge"]  # Fast, reliable models
MIN_SAMPLES = 15  # Lower threshold for offset analysis
N_MFCC = 20       # Use 20 MFCCs (best from sweep)


def setup_output():
    """Create output directories."""
    out_dir = Path("output/offset_window")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir


def print_header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}", flush=True)


# =============================================================================
# Part 1: Load Data — Audio Features + Raw Glucose for Re-matching
# =============================================================================

def load_participant_features_and_glucose(cfg, participants_filter=None):
    """
    For each participant:
    - Load glucose CSV (raw, for re-matching at different offsets)
    - Collect audio files, extract features
    - Store features + timestamps for re-use across offsets

    Returns dict: {participant: {features, timestamps, audio_paths, glucose_df}}
    """
    print_header("Loading Data: Audio Features + Glucose CSVs")
    base_dir = get_base_dir(cfg)
    participants = cfg.get("participants", {})

    if participants_filter:
        participants = {k: v for k, v in participants.items() if k in participants_filter}

    # Try to load from sweep cache first
    sweep_feat_dir = Path("output/sweep/features")
    use_cache = sweep_feat_dir.exists()

    extractor = MFCCExtractor(sr=16000, n_mfcc=N_MFCC, fmin=50, fmax=8000,
                              include_spectral=True, include_pitch=False, include_mel=False)

    all_data = {}

    for name, pcfg in participants.items():
        if not pcfg.get("glucose_csv"):
            continue

        # Load raw glucose data
        glucose_df = load_glucose_csv(
            pcfg["glucose_csv"],
            pcfg.get("glucose_unit", "auto"),
            base_dir,
        )
        if glucose_df.empty:
            print(f"  {name}: No glucose data, skipping", flush=True)
            continue

        # Try sweep cache first
        cache_file = sweep_feat_dir / f"20_base_{name}_X.npy"
        ts_cache = sweep_feat_dir / f"20_base_{name}_ts.json"

        if use_cache and cache_file.exists() and ts_cache.exists():
            X = np.load(cache_file)
            with open(ts_cache) as fp:
                ts_list = json.load(fp)
            timestamps = []
            for t in ts_list:
                try:
                    timestamps.append(datetime.fromisoformat(t))
                except (ValueError, TypeError):
                    timestamps.append(pd.Timestamp(t).to_pydatetime())

            if len(X) >= MIN_SAMPLES:
                all_data[name] = {
                    "features": X,
                    "timestamps": timestamps,
                    "glucose_df": glucose_df,
                    "default_offset": pcfg.get("optimal_offset", 0),
                }
                print(f"  {name}: {len(X)} samples (cached), glucose: {len(glucose_df)} readings", flush=True)
                continue

        # Extract fresh
        audio_files = collect_audio_files(
            pcfg.get("audio_dirs", []),
            pcfg.get("audio_ext", [".wav"]),
            base_dir,
        )

        if not audio_files:
            print(f"  {name}: No audio files found, skipping", flush=True)
            continue

        features = []
        timestamps = []

        for audio_path in audio_files:
            ts = parse_timestamp_from_filename(audio_path.name)
            if ts is None:
                continue
            try:
                y, _ = librosa.load(str(audio_path), sr=16000, mono=True)
                if len(y) < 8000:
                    continue
                feat = extractor.extract_from_array(y)
                if feat is not None:
                    features.append(feat)
                    timestamps.append(ts)
                del y
            except Exception:
                continue

        if len(features) >= MIN_SAMPLES:
            X = np.array(features)
            all_data[name] = {
                "features": X,
                "timestamps": timestamps,
                "glucose_df": glucose_df,
                "default_offset": pcfg.get("optimal_offset", 0),
            }
            print(f"  {name}: {len(X)} samples (extracted), glucose: {len(glucose_df)} readings", flush=True)
        else:
            print(f"  {name}: Only {len(features)} valid samples, skipping", flush=True)

        gc.collect()

    print(f"\nTotal: {len(all_data)} participants loaded", flush=True)
    return all_data


# =============================================================================
# Part 2: Offset Sweep Analysis
# =============================================================================

def match_glucose_at_offset(timestamps, glucose_df, offset_minutes, window_minutes=30):
    """
    Re-match audio timestamps to glucose at a given offset.

    Returns arrays of (glucose_values, valid_indices) where valid_indices
    indexes into the original timestamps/features arrays.
    """
    glucose_values = []
    valid_indices = []
    time_diffs = []

    for i, ts in enumerate(timestamps):
        glucose_val, time_diff = find_matching_glucose(
            ts, glucose_df,
            window_minutes=window_minutes,
            offset_minutes=offset_minutes,
            use_interpolation=True,
        )
        if glucose_val is not None:
            glucose_values.append(glucose_val)
            valid_indices.append(i)
            time_diffs.append(time_diff)

    return np.array(glucose_values), np.array(valid_indices), np.array(time_diffs)


def evaluate_at_offset(X, y, model_name="SVR"):
    """Run LOO/KFold CV and return metrics."""
    if len(X) < MIN_SAMPLES:
        return None

    X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    model = get_model(model_name)
    pipe = Pipeline([("s", RobustScaler()), ("m", model)])
    cv = LeaveOneOut() if len(X) <= 50 else KFold(10, shuffle=True, random_state=42)

    try:
        preds = cross_val_predict(pipe, X_clean, y, cv=cv)
        metrics = compute_metrics(y, preds)
        baseline = mean_predictor_baseline(y)
        metrics["baseline_mae"] = baseline["mae"]
        metrics["improvement"] = baseline["mae"] - metrics["mae"]
        metrics["predictions"] = preds
        return metrics
    except Exception as e:
        print(f"    Evaluation failed: {e}", flush=True)
        return None


def run_offset_sweep(all_data, offsets=None, model_name="SVR"):
    """
    Sweep offsets for all participants.

    Returns DataFrame with columns:
    participant, offset, n_samples, mae, r, rmse, baseline_mae, improvement
    """
    if offsets is None:
        offsets = list(OFFSET_RANGE)

    print_header(f"Offset Sweep Analysis ({model_name})")
    print(f"  Offsets: {offsets[0]} to {offsets[-1]} min, step {offsets[1]-offsets[0]} min")
    print(f"  Model: {model_name}")

    rows = []
    total = len(all_data) * len(offsets)
    idx = 0

    for pname, pdata in sorted(all_data.items()):
        X_all = pdata["features"]
        timestamps = pdata["timestamps"]
        glucose_df = pdata["glucose_df"]
        default_offset = pdata["default_offset"]

        print(f"\n  {pname} ({len(X_all)} samples, default offset: {default_offset} min):", flush=True)

        for offset in offsets:
            idx += 1
            # Re-match glucose at this offset
            y, valid_idx, time_diffs = match_glucose_at_offset(
                timestamps, glucose_df, offset_minutes=offset, window_minutes=30
            )

            if len(y) < MIN_SAMPLES:
                rows.append({
                    "participant": pname, "offset": offset,
                    "n_samples": len(y), "mae": np.nan, "r": np.nan,
                    "rmse": np.nan, "baseline_mae": np.nan,
                    "improvement": np.nan, "is_default": (offset == default_offset),
                    "avg_time_diff": np.nan,
                })
                continue

            X = X_all[valid_idx]
            metrics = evaluate_at_offset(X, y, model_name=model_name)

            if metrics is not None:
                rows.append({
                    "participant": pname, "offset": offset,
                    "n_samples": len(y),
                    "mae": round(metrics["mae"], 2),
                    "r": round(metrics["r"], 3),
                    "rmse": round(metrics["rmse"], 2),
                    "baseline_mae": round(metrics["baseline_mae"], 2),
                    "improvement": round(metrics["improvement"], 2),
                    "is_default": (offset == default_offset),
                    "avg_time_diff": round(np.mean(time_diffs), 1) if len(time_diffs) > 0 else np.nan,
                })
                marker = " <-- DEFAULT" if offset == default_offset else ""
                print(
                    f"    offset={offset:+3d}min: MAE={metrics['mae']:.2f}, r={metrics['r']:.3f}, "
                    f"n={len(y)}{marker}",
                    flush=True,
                )
            else:
                rows.append({
                    "participant": pname, "offset": offset,
                    "n_samples": len(y), "mae": np.nan, "r": np.nan,
                    "rmse": np.nan, "baseline_mae": np.nan,
                    "improvement": np.nan, "is_default": (offset == default_offset),
                    "avg_time_diff": np.nan,
                })

    return pd.DataFrame(rows)


def run_offset_sweep_multi_model(all_data, offsets=None):
    """Run offset sweep with multiple models and combine."""
    all_dfs = []
    for model_name in MODEL_NAMES:
        df = run_offset_sweep(all_data, offsets=offsets, model_name=model_name)
        df["model"] = model_name
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


# =============================================================================
# Part 3: Matching Window Analysis
# =============================================================================

def run_matching_window_sweep(all_data, model_name="SVR"):
    """Test different CGM matching windows (stricter = fewer samples but better quality)."""
    print_header(f"Matching Window Sweep ({model_name})")

    rows = []

    for pname, pdata in sorted(all_data.items()):
        X_all = pdata["features"]
        timestamps = pdata["timestamps"]
        glucose_df = pdata["glucose_df"]
        default_offset = pdata["default_offset"]

        print(f"\n  {pname}:", flush=True)

        for window in MATCHING_WINDOWS:
            y, valid_idx, time_diffs = match_glucose_at_offset(
                timestamps, glucose_df,
                offset_minutes=default_offset,
                window_minutes=window,
            )

            if len(y) < MIN_SAMPLES:
                rows.append({
                    "participant": pname, "window_minutes": window,
                    "n_samples": len(y), "mae": np.nan, "r": np.nan,
                    "avg_time_diff": np.nan,
                })
                print(f"    window=±{window}min: {len(y)} samples (too few)", flush=True)
                continue

            X = X_all[valid_idx]
            metrics = evaluate_at_offset(X, y, model_name=model_name)

            if metrics is not None:
                rows.append({
                    "participant": pname, "window_minutes": window,
                    "n_samples": len(y),
                    "mae": round(metrics["mae"], 2),
                    "r": round(metrics["r"], 3),
                    "avg_time_diff": round(np.mean(time_diffs), 1),
                })
                print(
                    f"    window=±{window}min: MAE={metrics['mae']:.2f}, r={metrics['r']:.3f}, "
                    f"n={len(y)}, avg_diff={np.mean(time_diffs):.1f}min",
                    flush=True,
                )

    return pd.DataFrame(rows)


# =============================================================================
# Part 4: Audio Segmentation Window Analysis
# =============================================================================

def extract_features_with_params(audio_paths_timestamps, n_fft=2048, hop_frac=0.5):
    """Extract MFCC features with specific n_fft and hop_length."""
    hop_length = max(1, int(n_fft * hop_frac))

    features = []
    valid_indices = []

    for i, (audio_path, ts) in enumerate(audio_paths_timestamps):
        try:
            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            if len(y) < 8000:
                continue

            # Extract MFCCs with custom n_fft and hop_length
            mfccs = librosa.feature.mfcc(
                y=y, sr=16000, n_mfcc=N_MFCC,
                n_fft=n_fft, hop_length=hop_length,
                fmin=50, fmax=8000,
            )
            feat = []
            feat.extend(np.mean(mfccs, axis=1))
            feat.extend(np.std(mfccs, axis=1))

            # Delta MFCCs
            delta = librosa.feature.delta(mfccs)
            feat.extend(np.mean(delta, axis=1))
            feat.extend(np.std(delta, axis=1))

            # Delta-delta
            delta2 = librosa.feature.delta(mfccs, order=2)
            feat.extend(np.mean(delta2, axis=1))
            feat.extend(np.std(delta2, axis=1))

            # Energy and ZCR
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
            feat.extend([np.mean(rms), np.std(rms)])
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
            feat.extend([np.mean(zcr), np.std(zcr)])

            features.append(np.array(feat, dtype=np.float32))
            valid_indices.append(i)
            del y
        except Exception:
            continue

    return np.array(features) if features else None, valid_indices


# =============================================================================
# Part 5: Visualization
# =============================================================================

COLORS_PARTICIPANT = {
    "Wolf": "#e74c3c", "Sybille": "#3498db", "Anja": "#2ecc71",
    "Margarita": "#9b59b6", "Vicky": "#f39c12", "Steffen": "#1abc9c",
    "Lara": "#e67e22", "Darav": "#34495e", "Joao": "#c0392b",
    "Alvar": "#16a085", "R_Rodolfo": "#8e44ad", "Christian_L": "#2c3e50",
}

COLORS_MODEL = {
    "Ridge": "#3498db", "SVR": "#e74c3c", "BayesianRidge": "#2ecc71",
}


def plot_offset_per_participant(offset_df, fig_dir, model_name="SVR"):
    """Plot offset sweep curves for each participant (MAE and r)."""
    df = offset_df[offset_df["model"] == model_name].copy()
    participants = sorted(df["participant"].unique())
    n_parts = len(participants)

    if n_parts == 0:
        return

    n_cols = min(3, n_parts)
    n_rows = (n_parts + n_cols - 1) // n_cols

    # --- MAE plots ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for idx, pname in enumerate(participants):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = df[df["participant"] == pname].sort_values("offset")
        color = COLORS_PARTICIPANT.get(pname, "#333")

        valid = sub.dropna(subset=["mae"])
        ax.plot(valid["offset"], valid["mae"], "o-", color=color, linewidth=2, markersize=5)

        # Mark optimal
        if not valid.empty:
            best_idx = valid["mae"].idxmin()
            best_offset = valid.loc[best_idx, "offset"]
            best_mae = valid.loc[best_idx, "mae"]
            ax.plot(best_offset, best_mae, "*", color="gold", markersize=15, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.annotate(f"Best: {best_offset:+d}min\nMAE={best_mae:.1f}",
                       xy=(best_offset, best_mae), xytext=(5, 10),
                       textcoords="offset points", fontsize=8, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        # Mark default offset
        default_row = sub[sub["is_default"]]
        if not default_row.empty and not default_row["mae"].isna().all():
            ax.axvline(default_row["offset"].values[0], color="gray", linestyle="--",
                      alpha=0.5, label=f"Default ({default_row['offset'].values[0]:+d})")
            ax.legend(fontsize=7)

        # Mark zero
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)

        # Shade "voice leads CGM" region
        ax.axvspan(0, 30, alpha=0.05, color="green")
        ax.axvspan(-30, 0, alpha=0.05, color="red")

        ax.set_title(f"{pname} (n={int(valid['n_samples'].median())})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Offset (min)", fontsize=9)
        ax.set_ylabel("MAE (mg/dL)", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide empty axes
    for idx in range(n_parts, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"Offset Sweep — MAE per Participant ({model_name})\n"
                 f"Green region: voice leads CGM (positive offset) | Red: CGM leads voice",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "offset_mae_per_participant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Correlation plots ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for idx, pname in enumerate(participants):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = df[df["participant"] == pname].sort_values("offset")
        color = COLORS_PARTICIPANT.get(pname, "#333")

        valid = sub.dropna(subset=["r"])
        ax.plot(valid["offset"], valid["r"], "s-", color=color, linewidth=2, markersize=5)

        # Mark peak r
        if not valid.empty:
            best_idx = valid["r"].idxmax()
            best_offset = valid.loc[best_idx, "offset"]
            best_r = valid.loc[best_idx, "r"]
            ax.plot(best_offset, best_r, "*", color="gold", markersize=15, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.annotate(f"Best: {best_offset:+d}min\nr={best_r:.3f}",
                       xy=(best_offset, best_r), xytext=(5, -15),
                       textcoords="offset points", fontsize=8, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        ax.axvline(0, color="black", linestyle=":", alpha=0.3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvspan(0, 30, alpha=0.05, color="green")
        ax.axvspan(-30, 0, alpha=0.05, color="red")

        ax.set_title(f"{pname}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Offset (min)", fontsize=9)
        ax.set_ylabel("Pearson r", fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(n_parts, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(f"Offset Sweep — Correlation per Participant ({model_name})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / "offset_r_per_participant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_offset_aggregate(offset_df, fig_dir):
    """Aggregate offset analysis across participants and models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Average MAE across participants per model
    ax = axes[0]
    for model_name in MODEL_NAMES:
        sub = offset_df[offset_df["model"] == model_name]
        avg = sub.groupby("offset").agg(
            avg_mae=("mae", "mean"),
            std_mae=("mae", "std"),
        ).reset_index()
        avg = avg.dropna()
        color = COLORS_MODEL.get(model_name, "gray")
        ax.plot(avg["offset"], avg["avg_mae"], "o-", color=color, label=model_name, linewidth=2)
        ax.fill_between(avg["offset"],
                        avg["avg_mae"] - avg["std_mae"],
                        avg["avg_mae"] + avg["std_mae"],
                        alpha=0.15, color=color)

    ax.axvline(0, color="black", linestyle=":", alpha=0.3)
    ax.axvspan(0, 30, alpha=0.05, color="green")
    ax.axvspan(-30, 0, alpha=0.05, color="red")
    ax.set_xlabel("Offset (min)", fontsize=11)
    ax.set_ylabel("Mean MAE (mg/dL)", fontsize=11)
    ax.set_title("Average MAE Across Participants", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Average r across participants per model
    ax = axes[1]
    for model_name in MODEL_NAMES:
        sub = offset_df[offset_df["model"] == model_name]
        avg = sub.groupby("offset").agg(
            avg_r=("r", "mean"),
            std_r=("r", "std"),
        ).reset_index()
        avg = avg.dropna()
        color = COLORS_MODEL.get(model_name, "gray")
        ax.plot(avg["offset"], avg["avg_r"], "s-", color=color, label=model_name, linewidth=2)
        ax.fill_between(avg["offset"],
                        avg["avg_r"] - avg["std_r"],
                        avg["avg_r"] + avg["std_r"],
                        alpha=0.15, color=color)

    ax.axvline(0, color="black", linestyle=":", alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvspan(0, 30, alpha=0.05, color="green")
    ax.axvspan(-30, 0, alpha=0.05, color="red")
    ax.set_xlabel("Offset (min)", fontsize=11)
    ax.set_ylabel("Mean Pearson r", fontsize=11)
    ax.set_title("Average Correlation Across Participants", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Optimal offset distribution (histogram)
    ax = axes[2]
    # For each participant + model, find optimal offset (min MAE)
    optimal_offsets = []
    for model_name in MODEL_NAMES:
        sub = offset_df[offset_df["model"] == model_name]
        for pname in sub["participant"].unique():
            psub = sub[sub["participant"] == pname].dropna(subset=["mae"])
            if not psub.empty:
                best_idx = psub["mae"].idxmin()
                optimal_offsets.append({
                    "participant": pname,
                    "model": model_name,
                    "optimal_offset": psub.loc[best_idx, "offset"],
                    "best_mae": psub.loc[best_idx, "mae"],
                })

    if optimal_offsets:
        opt_df = pd.DataFrame(optimal_offsets)

        # Use SVR for the histogram
        svr_opt = opt_df[opt_df["model"] == "SVR"]
        offsets_list = svr_opt["optimal_offset"].values
        bins = np.arange(-32.5, 35, 5)
        ax.hist(offsets_list, bins=bins, color="#3498db", edgecolor="white",
                alpha=0.7, label="SVR")

        mean_opt = np.mean(offsets_list)
        median_opt = np.median(offsets_list)
        ax.axvline(mean_opt, color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_opt:+.1f} min")
        ax.axvline(median_opt, color="orange", linestyle="-.", linewidth=2,
                   label=f"Median: {median_opt:+.1f} min")
        ax.axvline(0, color="black", linestyle=":", alpha=0.3)

        n_positive = np.sum(offsets_list > 0)
        n_zero = np.sum(offsets_list == 0)
        n_negative = np.sum(offsets_list < 0)
        ax.text(0.02, 0.98,
                f"Voice leads CGM: {n_positive}/{len(offsets_list)}\n"
                f"Simultaneous: {n_zero}/{len(offsets_list)}\n"
                f"CGM leads voice: {n_negative}/{len(offsets_list)}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Optimal Offset (min)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Optimal Offsets\n(per participant, SVR)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(fig_dir / "offset_aggregate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(optimal_offsets) if optimal_offsets else pd.DataFrame()


def plot_offset_heatmap(offset_df, fig_dir, model_name="SVR"):
    """Heatmap: participants × offsets showing MAE or r."""
    df = offset_df[offset_df["model"] == model_name].copy()

    for metric, cmap, title_suffix, vmin_fn in [
        ("mae", "RdYlGn_r", "MAE (mg/dL)", None),
        ("r", "RdYlGn", "Pearson r", None),
    ]:
        pivot = df.pivot_table(values=metric, index="participant", columns="offset", aggfunc="mean")
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6 + 2)))
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:+d}" for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_xlabel("Offset (minutes)", fontsize=11)
        ax.set_title(f"Offset Sweep — {title_suffix} ({model_name})", fontsize=13, fontweight="bold")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    fmt = f"{val:.1f}" if metric == "mae" else f"{val:.2f}"
                    color = "white" if (metric == "mae" and val > np.nanmean(pivot.values)) or \
                                       (metric == "r" and val < np.nanmean(pivot.values)) else "black"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=7, color=color)

                    # Mark best per participant
                    row_vals = pivot.values[i]
                    if metric == "mae":
                        is_best = val == np.nanmin(row_vals)
                    else:
                        is_best = val == np.nanmax(row_vals)
                    if is_best:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                   fill=False, edgecolor="gold", linewidth=2.5))

        fig.colorbar(im, ax=ax, label=title_suffix)
        plt.tight_layout()
        fig.savefig(fig_dir / f"offset_heatmap_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_matching_window(window_df, fig_dir):
    """Plot matching window analysis."""
    participants = sorted(window_df["participant"].unique())
    n_parts = len(participants)
    if n_parts == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. MAE vs window size
    ax = axes[0]
    for pname in participants:
        sub = window_df[window_df["participant"] == pname].sort_values("window_minutes")
        sub = sub.dropna(subset=["mae"])
        color = COLORS_PARTICIPANT.get(pname, "gray")
        ax.plot(sub["window_minutes"], sub["mae"], "o-", color=color, label=pname, linewidth=1.5)
    ax.set_xlabel("Matching Window (±min)", fontsize=11)
    ax.set_ylabel("MAE (mg/dL)", fontsize=11)
    ax.set_title("MAE vs Matching Window", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 2. Sample count vs window size
    ax = axes[1]
    for pname in participants:
        sub = window_df[window_df["participant"] == pname].sort_values("window_minutes")
        color = COLORS_PARTICIPANT.get(pname, "gray")
        ax.plot(sub["window_minutes"], sub["n_samples"], "s-", color=color, label=pname, linewidth=1.5)
    ax.set_xlabel("Matching Window (±min)", fontsize=11)
    ax.set_ylabel("Matched Samples", fontsize=11)
    ax.set_title("Sample Count vs Matching Window", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. Average match quality vs window size
    ax = axes[2]
    for pname in participants:
        sub = window_df[window_df["participant"] == pname].sort_values("window_minutes")
        sub = sub.dropna(subset=["avg_time_diff"])
        color = COLORS_PARTICIPANT.get(pname, "gray")
        ax.plot(sub["window_minutes"], sub["avg_time_diff"], "^-", color=color, label=pname, linewidth=1.5)
    ax.set_xlabel("Matching Window (±min)", fontsize=11)
    ax.set_ylabel("Avg Time Diff (min)", fontsize=11)
    ax.set_title("Match Quality vs Window", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / "matching_window_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Part 6: HTML Report
# =============================================================================

def embed_img(path):
    if not Path(path).exists():
        return "<p><i>[Image not generated]</i></p>"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;margin:10px 0;">'


def generate_html_report(offset_df, optimal_df, window_df, fig_dir, out_dir):
    """Generate comprehensive HTML report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_participants = offset_df["participant"].nunique()
    n_offsets = offset_df["offset"].nunique()

    # Compute summary statistics
    svr_df = offset_df[offset_df["model"] == "SVR"]
    svr_opt = optimal_df[optimal_df["model"] == "SVR"] if not optimal_df.empty else pd.DataFrame()

    # Best offset summary per participant
    best_offsets_html = ""
    if not svr_opt.empty:
        summary_rows = []
        for _, row in svr_opt.sort_values("participant").iterrows():
            pname = row["participant"]
            # Get default offset
            default = svr_df[svr_df["participant"] == pname]
            default_row = default[default["is_default"]]
            default_mae = default_row["mae"].values[0] if not default_row.empty else np.nan
            default_offset = default_row["offset"].values[0] if not default_row.empty else "N/A"

            leads = "Voice leads" if row["optimal_offset"] > 0 else ("Simultaneous" if row["optimal_offset"] == 0 else "CGM leads")
            summary_rows.append({
                "Participant": pname,
                "Default Offset": f"{default_offset:+d} min" if isinstance(default_offset, (int, float, np.integer)) else default_offset,
                "Optimal Offset": f"{int(row['optimal_offset']):+d} min",
                "Best MAE": f"{row['best_mae']:.2f}",
                "Default MAE": f"{default_mae:.2f}" if not np.isnan(default_mae) else "N/A",
                "Improvement": f"{default_mae - row['best_mae']:.2f}" if not np.isnan(default_mae) else "N/A",
                "Interpretation": leads,
            })
        best_offsets_html = pd.DataFrame(summary_rows).to_html(
            index=False, classes="", border=0, escape=False
        )

    # Aggregate stats
    if not svr_opt.empty:
        mean_optimal = svr_opt["optimal_offset"].mean()
        median_optimal = svr_opt["optimal_offset"].median()
        n_voice_leads = (svr_opt["optimal_offset"] > 0).sum()
        n_total = len(svr_opt)
        pct_voice_leads = 100 * n_voice_leads / n_total if n_total > 0 else 0
    else:
        mean_optimal = median_optimal = 0
        pct_voice_leads = 0
        n_voice_leads = n_total = 0

    # CGM lag interpretation
    if mean_optimal > 5:
        lag_interpretation = (
            f"The average optimal offset is <b>{mean_optimal:+.1f} minutes</b>, suggesting that voice features "
            f"correlate best with CGM readings taken {abs(mean_optimal):.0f} minutes <b>after</b> the voice recording. "
            f"This is consistent with the known CGM interstitial fluid lag (5-15 min). "
            f"<b>Voice appears to detect glucose changes faster than CGM for {n_voice_leads}/{n_total} "
            f"({pct_voice_leads:.0f}%) participants.</b>"
        )
    elif mean_optimal < -5:
        lag_interpretation = (
            f"The average optimal offset is <b>{mean_optimal:+.1f} minutes</b>, suggesting CGM readings "
            f"slightly precede the voice signal. This could indicate that voice features reflect slower "
            f"physiological responses to glucose changes."
        )
    else:
        lag_interpretation = (
            f"The average optimal offset is <b>{mean_optimal:+.1f} minutes</b>, indicating that voice features "
            f"and CGM readings are approximately synchronous. The known CGM lag may be offset by "
            f"voice feature response time."
        )

    # Window analysis summary
    window_summary_html = ""
    if not window_df.empty:
        window_avg = window_df.groupby("window_minutes").agg(
            avg_mae=("mae", "mean"),
            avg_samples=("n_samples", "mean"),
            avg_time_diff=("avg_time_diff", "mean"),
        ).reset_index().round(2)
        window_summary_html = window_avg.to_html(index=False, classes="", border=0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TONES — Offset &amp; Window Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1300px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
  h2 {{ color: #c0392b; margin-top: 40px; }}
  h3 {{ color: #34495e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
  th {{ background: #2c3e50; color: white; padding: 10px 8px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  tr:hover {{ background: #e8f4f8; }}
  .metric-box {{ display: inline-block; background: white; border: 1px solid #ddd; border-radius: 8px;
                 padding: 15px 25px; margin: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
  .metric-box .value {{ font-size: 28px; font-weight: bold; color: #c0392b; }}
  .metric-box .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
  .section {{ background: white; border-radius: 8px; padding: 25px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.08); }}
  .key-finding {{ background: #fef9e7; border-left: 4px solid #f39c12; padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
  .positive {{ color: #27ae60; font-weight: bold; }}
  .negative {{ color: #e74c3c; font-weight: bold; }}
  .footer {{ text-align: center; color: #95a5a6; font-size: 11px; margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; }}
  .explainer {{ background: #eaf2f8; border-radius: 8px; padding: 15px; margin: 15px 0; font-size: 13px; }}
</style>
</head>
<body>

<h1>TONES — CGM Offset &amp; Time Window Analysis</h1>
<p>Generated: {now} | Participants: {n_participants} | Offsets tested: {n_offsets} | Models: {', '.join(MODEL_NAMES)}</p>

<div class="section">
<h2>Executive Summary</h2>
<div style="text-align: center;">
  <div class="metric-box">
    <div class="value">{mean_optimal:+.1f} min</div>
    <div class="label">Mean Optimal Offset</div>
  </div>
  <div class="metric-box">
    <div class="value">{median_optimal:+.1f} min</div>
    <div class="label">Median Optimal Offset</div>
  </div>
  <div class="metric-box">
    <div class="value">{pct_voice_leads:.0f}%</div>
    <div class="label">Participants Where<br>Voice Leads CGM</div>
  </div>
  <div class="metric-box">
    <div class="value">{n_participants}</div>
    <div class="label">Participants<br>Analyzed</div>
  </div>
</div>

<div class="key-finding">
  <b>Key Finding:</b> {lag_interpretation}
</div>

<div class="explainer">
  <b>How to read the offset:</b> A positive offset (e.g., +15 min) means the model matches each voice recording
  to the CGM reading taken 15 minutes <i>later</i>. If positive offsets give the best prediction, it means
  the voice features are already capturing a glucose change that CGM won't report for another 15 minutes.
  This is plausible because CGM sensors measure interstitial fluid glucose, which lags behind actual blood glucose
  by 5&ndash;15 minutes. Voice features may reflect blood glucose more directly via autonomic nervous system effects
  on vocal fold tension, salivation, and muscle tone.
</div>
</div>

<div class="section">
<h2>1. Offset Sweep — Per Participant (MAE)</h2>
<p>For each participant, how does prediction error change as we shift the CGM&ndash;voice alignment?
   Gold stars mark the optimal offset. Green shading = voice leads CGM; red = CGM leads voice.</p>
{embed_img(fig_dir / "offset_mae_per_participant.png")}
</div>

<div class="section">
<h2>2. Offset Sweep — Per Participant (Correlation)</h2>
<p>Same analysis using Pearson r. Higher correlation at positive offsets supports the "voice leads CGM" hypothesis.</p>
{embed_img(fig_dir / "offset_r_per_participant.png")}
</div>

<div class="section">
<h2>3. Offset Heatmaps</h2>
<p>Gold borders indicate the best offset for each participant. Rows = participants, columns = offsets.</p>
{embed_img(fig_dir / "offset_heatmap_mae.png")}
{embed_img(fig_dir / "offset_heatmap_r.png")}
</div>

<div class="section">
<h2>4. Aggregate Offset Analysis</h2>
<p>Averaged across all participants. The shaded band shows ±1 standard deviation.
   If the curve dips (MAE) or peaks (r) at positive offsets, it suggests a systematic CGM lag.</p>
{embed_img(fig_dir / "offset_aggregate.png")}
</div>

<div class="section">
<h2>5. Optimal Offset Summary</h2>
<p>Per-participant optimal offset compared to the currently hardcoded default.</p>
{best_offsets_html}
</div>

<div class="section">
<h2>6. Matching Window Analysis</h2>
<p>How does the CGM matching window (strictness of time alignment) affect prediction quality?
   Smaller windows require closer temporal alignment but yield fewer matched samples.</p>
{embed_img(fig_dir / "matching_window_sweep.png")}

<h3>Window Size Summary (averaged across participants)</h3>
{window_summary_html}

<div class="explainer">
<b>Trade-off:</b> Tighter matching windows (e.g., &plusmn;5 min) ensure more accurate glucose labels
but may discard valid samples. Looser windows (&plusmn;30 min) retain more data but introduce
label noise from the time mismatch. The optimal window balances data quantity vs. label quality.
</div>
</div>

<div class="section">
<h2>Methodology</h2>
<h3>Offset Analysis</h3>
<ul>
  <li><b>Offsets tested:</b> {list(OFFSET_RANGE)[0]} to {list(OFFSET_RANGE)[-1]} min in 5-min steps</li>
  <li><b>Matching:</b> &plusmn;30 min window with linear interpolation at each offset</li>
  <li><b>Features:</b> 20 MFCCs + deltas + spectral (from pre-extracted sweep cache)</li>
  <li><b>Models:</b> {', '.join(MODEL_NAMES)}</li>
  <li><b>Evaluation:</b> LOO-CV (n&le;50) or 10-fold CV (n&gt;50)</li>
  <li><b>Minimum samples:</b> {MIN_SAMPLES} per participant-offset combination</li>
</ul>

<h3>Interpretation Guide</h3>
<ul>
  <li><b>Positive optimal offset (+N min):</b> Voice recording correlates best with CGM reading N minutes <i>later</i>.
      This means voice detects the glucose change before CGM reports it, consistent with known CGM interstitial lag.</li>
  <li><b>Zero optimal offset (0 min):</b> Voice and CGM are roughly synchronous for this participant.</li>
  <li><b>Negative optimal offset (-N min):</b> Voice correlates best with CGM reading N minutes <i>earlier</i>.
      This could mean voice features reflect a slower physiological response, or noise in the data.</li>
</ul>

<h3>Clinical Significance</h3>
<p>FreeStyle Libre CGM sensors measure <b>interstitial fluid glucose</b>, which lags behind blood glucose by
   approximately <b>5&ndash;15 minutes</b> (varying with glucose rate of change — larger lag during rapid changes).
   If voice features can predict glucose changes ahead of CGM, this has implications for:</p>
<ul>
  <li><b>Early warning systems:</b> Voice-based alerts could precede CGM alerts for hypoglycemia</li>
  <li><b>Complementary monitoring:</b> Voice could fill the CGM blind spot during rapid glucose transitions</li>
  <li><b>Non-invasive screening:</b> Voice analysis requires no sensor, making it accessible for broader populations</li>
</ul>
</div>

<div class="footer">
  ONVOX / TONES Project — Voice-Based Glucose Estimation<br>
  Offset &amp; Window Analysis Report | Generated by offset_window_analysis.py | {now}
</div>
</body>
</html>"""

    report_path = out_dir / "offset_window_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TONES Offset & Window Analysis")
    parser.add_argument("--participants", nargs="+", default=None, help="Filter participants")
    parser.add_argument("--fast", action="store_true", help="Quick run with fewer offsets")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    start_time = time.time()
    print_header("TONES — CGM Offset & Time Window Analysis")

    out_dir, fig_dir = setup_output()

    # Load config
    cfg = load_config(args.config)
    print(f"Config loaded from: {cfg['base_dir']}", flush=True)

    # Load data
    all_data = load_participant_features_and_glucose(cfg, args.participants)
    if not all_data:
        print("ERROR: No data loaded!", flush=True)
        sys.exit(1)

    # Offset range
    if args.fast:
        offsets = list(range(-20, 21, 10))  # Fewer steps for quick test
    else:
        offsets = list(OFFSET_RANGE)

    # =========================================================================
    # Analysis 1: Offset Sweep (multi-model)
    # =========================================================================
    offset_df = run_offset_sweep_multi_model(all_data, offsets=offsets)
    offset_df.to_csv(out_dir / "offset_sweep_results.csv", index=False)
    print(f"\nOffset results saved: {len(offset_df)} rows", flush=True)

    # =========================================================================
    # Analysis 2: Matching Window Sweep
    # =========================================================================
    window_df = run_matching_window_sweep(all_data, model_name="SVR")
    window_df.to_csv(out_dir / "matching_window_results.csv", index=False)
    print(f"\nWindow results saved: {len(window_df)} rows", flush=True)

    # =========================================================================
    # Visualizations
    # =========================================================================
    print_header("Generating Visualizations")

    for model_name in MODEL_NAMES:
        plot_offset_per_participant(offset_df, fig_dir, model_name=model_name)

    optimal_df = plot_offset_aggregate(offset_df, fig_dir)
    plot_offset_heatmap(offset_df, fig_dir, model_name="SVR")
    plot_matching_window(window_df, fig_dir)

    if not optimal_df.empty:
        optimal_df.to_csv(out_dir / "optimal_offsets.csv", index=False)

    # =========================================================================
    # HTML Report
    # =========================================================================
    print_header("Generating HTML Report")
    report_path = generate_html_report(offset_df, optimal_df, window_df, fig_dir, out_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"ANALYSIS COMPLETE in {elapsed:.1f}s", flush=True)
    print(f"  Offset results: {out_dir / 'offset_sweep_results.csv'}", flush=True)
    print(f"  Window results: {out_dir / 'matching_window_results.csv'}", flush=True)
    print(f"  Optimal offsets: {out_dir / 'optimal_offsets.csv'}", flush=True)
    print(f"  HTML Report:    {report_path}", flush=True)
    print(f"  Figures:        {fig_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
