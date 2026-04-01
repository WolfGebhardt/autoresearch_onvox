#!/usr/bin/env python3
"""
Metrics audit: compare model predictions against naive baselines
and compute clinically appropriate metrics.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, KFold

from tones.config import load_config, get_base_dir
from tones.data.loaders import load_participant_data
from tones.features.mfcc import MFCCExtractor
from tones.models.train import compute_metrics


def concordance_correlation(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient (CCC).
    Measures agreement, not just correlation. CCC = 1 means perfect agreement.
    Unlike Pearson r, CCC penalizes bias and scale shifts."""
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    ccc = 2 * cov / (var_t + var_p + (mean_t - mean_p)**2)
    return ccc


def mard(y_true, y_pred):
    """Mean Absolute Relative Difference (%) -- CGM industry standard."""
    return 100.0 * np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1.0))


def iso_15197_compliance(y_true, y_pred):
    """ISO 15197:2013 criteria for blood glucose monitors.
    For ref < 100 mg/dL: within +/-15 mg/dL
    For ref >= 100 mg/dL: within +/-15%"""
    compliant = 0
    for ref, pred in zip(y_true, y_pred):
        if ref < 100:
            if abs(pred - ref) <= 15:
                compliant += 1
        else:
            if abs(pred - ref) / ref <= 0.15:
                compliant += 1
    return 100.0 * compliant / len(y_true)


def main():
    cfg = load_config()
    base = get_base_dir()
    matching = cfg.get("matching", {})
    extractor = MFCCExtractor(sr=16000, n_mfcc=20, n_mels=64, fmin=100, fmax=8000)

    print("=" * 80)
    print("TONES -- METRICS AUDIT: Are we measuring the right thing?")
    print("=" * 80)

    all_refs, all_preds, all_means = [], [], []
    results = []

    for name, pcfg in cfg["participants"].items():
        pdata_df = load_participant_data(name, pcfg, base, matching)
        if pdata_df.empty:
            continue

        features, glucose_vals = [], []
        for _, row in pdata_df.iterrows():
            audio_path = Path(row["audio_path"])
            if not audio_path.exists():
                continue
            feat = extractor.extract_from_file(str(audio_path))
            if feat is not None and np.isfinite(feat).all():
                features.append(feat)
                glucose_vals.append(row["glucose_mg_dl"])

        if len(glucose_vals) < 20:
            continue

        X = np.array(features, dtype=np.float32)
        y = np.array(glucose_vals, dtype=np.float64)

        # --- SVR predictions (LOO/KFold) ---
        cv = LeaveOneOut() if len(y) <= 50 else KFold(n_splits=10, shuffle=True, random_state=42)
        preds = np.zeros_like(y)
        for tr, te in cv.split(X):
            pipe = Pipeline([("scaler", RobustScaler()), ("model", SVR(C=10, gamma="scale"))])
            pipe.fit(X[tr], y[tr])
            preds[te] = pipe.predict(X[te])

        # --- Naive baseline: predict participant's leave-one-out mean ---
        mean_preds = np.zeros_like(y)
        for i in range(len(y)):
            mean_preds[i] = np.mean(np.delete(y, i))

        # --- Metrics ---
        model_mae = np.mean(np.abs(preds - y))
        baseline_mae = np.mean(np.abs(mean_preds - y))
        improvement = baseline_mae - model_mae
        pct_improvement = 100.0 * improvement / baseline_mae

        r_model = stats.pearsonr(y, preds)[0] if np.std(preds) > 0 else 0
        r_baseline = 0.0  # Mean prediction always has r=0

        ccc_model = concordance_correlation(y, preds)
        mard_model = mard(y, preds)
        mard_baseline = mard(y, mean_preds)
        iso_model = iso_15197_compliance(y, preds)
        iso_baseline = iso_15197_compliance(y, mean_preds)

        glucose_std = np.std(y)
        glucose_range = np.max(y) - np.min(y)
        glucose_cv = 100.0 * glucose_std / np.mean(y)

        results.append({
            "name": name, "n": len(y),
            "glucose_mean": np.mean(y), "glucose_std": glucose_std,
            "glucose_range": glucose_range, "glucose_cv": glucose_cv,
            "model_mae": model_mae, "baseline_mae": baseline_mae,
            "improvement": improvement, "pct_improvement": pct_improvement,
            "pearson_r": r_model, "ccc": ccc_model,
            "mard_model": mard_model, "mard_baseline": mard_baseline,
            "iso_model": iso_model, "iso_baseline": iso_baseline,
        })

        all_refs.extend(y.tolist())
        all_preds.extend(preds.tolist())
        all_means.extend(mean_preds.tolist())

    # --- Print results ---
    print("\n" + "-" * 80)
    print("PART 1: Per-Participant Comparison -- Model vs. Predict-the-Mean Baseline")
    print("-" * 80)
    header = f"{'Name':<12} {'N':>4} {'Gluc SD':>8} {'Gluc CV%':>8} {'ModelMAE':>9} {'BaseMAE':>9} {'Improv%':>8} {'r':>7} {'CCC':>7} {'MARD%':>7} {'ISO%':>6}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["model_mae"]):
        print(f"{r['name']:<12} {r['n']:>4} {r['glucose_std']:>8.1f} {r['glucose_cv']:>8.1f} "
              f"{r['model_mae']:>9.2f} {r['baseline_mae']:>9.2f} {r['pct_improvement']:>7.1f}% "
              f"{r['pearson_r']:>7.3f} {r['ccc']:>7.3f} {r['mard_model']:>7.1f} {r['iso_model']:>5.1f}")

    # --- Aggregate ---
    refs = np.array(all_refs)
    preds = np.array(all_preds)
    means = np.array(all_means)

    print("\n" + "-" * 80)
    print("PART 2: Aggregate Metrics")
    print("-" * 80)
    print(f"  Total samples:           {len(refs)}")
    print(f"  Model MAE:               {np.mean(np.abs(preds - refs)):.2f} mg/dL")
    print(f"  Baseline MAE (mean):     {np.mean(np.abs(means - refs)):.2f} mg/dL")
    print(f"  Improvement over mean:   {np.mean(np.abs(means - refs)) - np.mean(np.abs(preds - refs)):.2f} mg/dL")
    print(f"  Pearson r (model):       {stats.pearsonr(refs, preds)[0]:.4f}")
    print(f"  CCC (model):             {concordance_correlation(refs, preds):.4f}")
    print(f"  MARD (model):            {mard(refs, preds):.2f}%")
    print(f"  MARD (baseline):         {mard(refs, means):.2f}%")
    print(f"  ISO 15197 (model):       {iso_15197_compliance(refs, preds):.1f}%")
    print(f"  ISO 15197 (baseline):    {iso_15197_compliance(refs, means):.1f}%")

    print("\n" + "-" * 80)
    print("PART 3: MAE by Glucose Range")
    print("-" * 80)
    for lo, hi, label in [(0, 80, "Hypo (<80)"), (80, 140, "Normal (80-140)"), (140, 500, "Hyper (>140)")]:
        mask = (refs >= lo) & (refs < hi)
        if mask.sum() > 0:
            model_m = np.mean(np.abs(preds[mask] - refs[mask]))
            base_m = np.mean(np.abs(means[mask] - refs[mask]))
            r_m = stats.pearsonr(refs[mask], preds[mask])[0] if mask.sum() > 5 and np.std(preds[mask]) > 0 else 0
            print(f"  {label:<20} n={mask.sum():>4}  Model MAE={model_m:>6.2f}  Baseline MAE={base_m:>6.2f}  "
                  f"Improvement={base_m - model_m:>6.2f}  r={r_m:>6.3f}")

    print("\n" + "-" * 80)
    print("PART 4: Honest Assessment")
    print("-" * 80)
    avg_pct_improv = np.mean([r["pct_improvement"] for r in results])
    n_positive_r = sum(1 for r in results if r["pearson_r"] > 0.1)
    n_positive_ccc = sum(1 for r in results if r["ccc"] > 0.1)
    print(f"  Avg improvement over mean: {avg_pct_improv:.1f}%")
    print(f"  Participants with r > 0.1: {n_positive_r}/{len(results)}")
    print(f"  Participants with CCC > 0.1: {n_positive_ccc}/{len(results)}")
    print()
    if avg_pct_improv < 15:
        print("  WARNING: Model improvement over predict-the-mean is modest.")
        print("  The low Pearson r values suggest the model captures central")
        print("  tendency but does NOT reliably track glucose DYNAMICS.")
    if n_positive_r < len(results) // 2:
        print("  WARNING: Fewer than half of participants show meaningful")
        print("  correlation (r > 0.1). The model may be predicting near")
        print("  each person's glucose mean rather than tracking variation.")


if __name__ == "__main__":
    main()
