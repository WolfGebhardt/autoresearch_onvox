#!/usr/bin/env python3
"""
TONES — Offset-by-Feature Sensitivity Analysis
================================================
Which features are most glucose-sensitive at which CGM-to-voice time offsets?

For each feature, we sweep CGM offsets from -30 to +30 minutes and compute
Pearson correlation with glucose at each offset. This reveals:
- Features that peak at +15–30 min: possibly reflect blood glucose more directly
  than CGM (which lags via interstitial fluid)
- Features that peak near 0: synchronous with interstitial glucose
- Features that peak at different offsets: different physiological pathways

Outputs:
- Per-feature optimal offset and peak |r| per participant
- Aggregate rankings: which features are most glucose-sensitive across participants
- Heatmaps: features × offsets
- Feature recommendations for app/API: use offset-optimized feature subsets

Usage:
    python offset_by_feature_analysis.py
    python offset_by_feature_analysis.py --participants Wolf Lara Anja
    python offset_by_feature_analysis.py --fast   # Fewer offsets for quick run
"""

import sys
import gc
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from tones.config import load_config, get_base_dir
from tones.data.loaders import (
    load_glucose_csv,
    collect_audio_files,
    parse_timestamp_from_filename,
    find_matching_glucose,
)
from tones.features.mfcc import MFCCExtractor

OFFSET_RANGE = list(range(-30, 31, 5))  # -30 to +30 in 5-min steps
MIN_SAMPLES = 15
N_MFCC = 20


def setup_output():
    """Create output directories."""
    out_dir = Path("output/offset_by_feature")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir


def load_participant_data(cfg, participants_filter=None):
    """
    Load features + timestamps + glucose for each participant.
    Reuses sweep cache when available (same as offset_window_analysis).
    """
    base_dir = get_base_dir(cfg)
    participants = cfg.get("participants", {})

    if participants_filter:
        participants = {k: v for k, v in participants.items() if k in participants_filter}

    sweep_feat_dir = Path("output/sweep/features")
    use_cache = sweep_feat_dir.exists()

    extractor = MFCCExtractor(sr=16000, n_mfcc=N_MFCC, fmin=50, fmax=8000,
                              include_spectral=True, include_pitch=False, include_mel=False)
    feature_names = extractor.feature_names

    all_data = {}
    for name, pcfg in participants.items():
        if not pcfg.get("glucose_csv"):
            continue

        glucose_df = load_glucose_csv(
            pcfg["glucose_csv"],
            pcfg.get("glucose_unit", "auto"),
            base_dir,
        )
        if glucose_df.empty:
            continue

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
                n_feat = X.shape[1]
                names = feature_names if len(feature_names) == n_feat else [f"feat_{i}" for i in range(n_feat)]
                all_data[name] = {
                    "features": X,
                    "timestamps": timestamps,
                    "glucose_df": glucose_df,
                    "feature_names": names,
                }
                continue

        # Extract fresh
        import librosa
        audio_files = collect_audio_files(
            pcfg.get("audio_dirs", []),
            pcfg.get("audio_ext", [".wav"]),
            base_dir,
        )
        if not audio_files:
            continue

        features, timestamps = [], []
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
            except Exception:
                continue

        if len(features) >= MIN_SAMPLES:
            all_data[name] = {
                "features": np.array(features),
                "timestamps": timestamps,
                "glucose_df": glucose_df,
                "feature_names": feature_names,
            }
        gc.collect()

    return all_data


def match_glucose_at_offset(timestamps, glucose_df, offset_minutes, window_minutes=30):
    """Re-match audio timestamps to glucose at a given offset."""
    glucose_values, valid_indices = [], []
    for i, ts in enumerate(timestamps):
        glucose_val, _ = find_matching_glucose(
            ts, glucose_df,
            window_minutes=window_minutes,
            offset_minutes=offset_minutes,
            use_interpolation=True,
        )
        if glucose_val is not None:
            glucose_values.append(glucose_val)
            valid_indices.append(i)
    return np.array(glucose_values), np.array(valid_indices)


def run_offset_by_feature(all_data, offsets=None):
    """
    For each participant, each feature, each offset: compute Pearson r with glucose.
    Returns nested dict: {participant: {feature_name: {offset: r, ...}}}
    plus per-feature optimal offset and peak |r|.
    """
    if offsets is None:
        offsets = OFFSET_RANGE

    results = {}
    for pname, pdata in sorted(all_data.items()):
        X_all = pdata["features"]
        timestamps = pdata["timestamps"]
        glucose_df = pdata["glucose_df"]
        feature_names = pdata["feature_names"]

        n_features = X_all.shape[1]
        if len(feature_names) != n_features:
            feature_names = [f"feat_{i}" for i in range(n_features)]

        results[pname] = {"by_offset": {}, "optimal": []}

        for offset in offsets:
            y, valid_idx = match_glucose_at_offset(
                timestamps, glucose_df, offset_minutes=offset, window_minutes=30
            )
            if len(y) < MIN_SAMPLES:
                continue

            X = X_all[valid_idx]
            X_clean = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

            r_values = []
            for j in range(n_features):
                feat = X_clean[:, j]
                if np.std(feat) < 1e-10 or np.std(y) < 1e-10:
                    r_values.append(0.0)
                else:
                    r, _ = stats.pearsonr(feat, y)
                    r_values.append(float(r))

            results[pname]["by_offset"][offset] = {
                "r": r_values,
                "n_samples": len(y),
            }

        # Compute optimal offset per feature
        for j in range(n_features):
            best_offset = None
            best_r = 0.0
            for offset in offsets:
                if offset not in results[pname]["by_offset"]:
                    continue
                r_val = results[pname]["by_offset"][offset]["r"][j]
                if abs(r_val) > abs(best_r):
                    best_r = r_val
                    best_offset = offset

            if best_offset is not None:
                results[pname]["optimal"].append({
                    "feature": feature_names[j],
                    "best_offset": best_offset,
                    "best_r": round(best_r, 4),
                    "abs_r": abs(best_r),
                })

        # Sort by |r| descending
        results[pname]["optimal"].sort(key=lambda x: -x["abs_r"])

    return results


def build_aggregate_rankings(results):
    """Aggregate across participants: which features appear most often as top glucose-sensitive?"""
    feature_scores = defaultdict(list)
    for pname, pdata in results.items():
        for item in pdata["optimal"][:20]:  # Top 20 per participant
            feature_scores[item["feature"]].append({
                "participant": pname,
                "best_offset": item["best_offset"],
                "best_r": item["best_r"],
                "abs_r": item["abs_r"],
            })

    # Compute aggregate score: mean peak |r| across participants where feature was in top 20
    aggregate = []
    for feat, scores in feature_scores.items():
        if not scores:
            continue
        mean_abs_r = np.mean([s["abs_r"] for s in scores])
        mean_offset = np.mean([s["best_offset"] for s in scores])
        n_participants = len(set(s["participant"] for s in scores))
        aggregate.append({
            "feature": feat,
            "mean_peak_abs_r": round(mean_abs_r, 4),
            "mean_optimal_offset": round(mean_offset, 1),
            "n_participants": n_participants,
        })

    aggregate.sort(key=lambda x: -x["mean_peak_abs_r"])
    return aggregate


def plot_heatmap(results, participant, feature_names, offsets, fig_dir):
    """Heatmap: features (rows) × offsets (cols), color = correlation."""
    if participant not in results:
        return

    pdata = results[participant]
    by_offset = pdata["by_offset"]
    if not by_offset:
        return

    # Use actual feature count from first offset's data
    first_offset = next(iter(by_offset))
    n_features = len(by_offset[first_offset]["r"])
    if len(feature_names) != n_features:
        feature_names = [f"feat_{i}" for i in range(n_features)]
    else:
        feature_names = list(feature_names)

    r_matrix = np.zeros((n_features, len(offsets)))
    r_matrix[:] = np.nan

    for col, offset in enumerate(offsets):
        if offset not in by_offset:
            continue
        r_matrix[:, col] = by_offset[offset]["r"]

    if np.all(np.isnan(r_matrix)):
        return

    fig, ax = plt.subplots(figsize=(14, max(10, n_features * 0.2)))
    im = ax.imshow(r_matrix, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels([f"{o:+d}" for o in offsets])
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("CGM offset (minutes, relative to voice)")
    ax.set_ylabel("Feature")
    ax.set_title(f"Offset-by-Feature Correlation: {participant}")
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    plt.savefig(fig_dir / f"offset_by_feature_heatmap_{participant}.png", dpi=120)
    plt.close()


def plot_top_features_per_offset(results, participant, top_n=15, fig_dir=None):
    """Bar chart: top N features by peak |r|, colored by optimal offset."""
    if participant not in results:
        return

    optimal = results[participant]["optimal"][:top_n]
    if not optimal:
        return

    features = [x["feature"] for x in optimal]
    r_vals = [x["best_r"] for x in optimal]
    offsets = [x["best_offset"] for x in optimal]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2E7D32" if o > 0 else "#C62828" if o < 0 else "#1565C0" for o in offsets]
    bars = ax.barh(range(len(features)), r_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Pearson r with glucose at optimal offset")
    ax.set_title(f"Top {top_n} Glucose-Sensitive Features: {participant}\n(green=voice leads, red=CGM leads, blue=sync)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    if fig_dir:
        plt.savefig(fig_dir / f"top_features_{participant}.png", dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Offset-by-feature sensitivity analysis")
    parser.add_argument("--participants", nargs="+", default=None)
    parser.add_argument("--fast", action="store_true", help="Use fewer offsets (-15, 0, +15)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    offsets = [-15, 0, 15] if args.fast else OFFSET_RANGE

    print("=" * 70)
    print("TONES — Offset-by-Feature Sensitivity Analysis")
    print("=" * 70)
    print(f"Offsets: {offsets[0]} to {offsets[-1]} min (step {offsets[1]-offsets[0] if len(offsets)>1 else 0})")
    print()

    cfg = load_config(args.config)
    all_data = load_participant_data(cfg, args.participants)
    if not all_data:
        print("No data loaded. Run hyperparameter_sweep.py or ensure sweep cache exists.")
        sys.exit(1)

    print(f"Loaded {len(all_data)} participants")
    for name, d in all_data.items():
        print(f"  {name}: {d['features'].shape[0]} samples, {d['features'].shape[1]} features")

    results = run_offset_by_feature(all_data, offsets)

    out_dir, fig_dir = setup_output()

    # Save per-participant optimal features
    optimal_csv_path = out_dir / "optimal_offset_by_feature.csv"
    rows = []
    for pname, pdata in results.items():
        for item in pdata["optimal"]:
            rows.append({
                "participant": pname,
                "feature": item["feature"],
                "best_offset_min": item["best_offset"],
                "best_r": item["best_r"],
            })
    pd.DataFrame(rows).to_csv(optimal_csv_path, index=False)
    print(f"\nSaved: {optimal_csv_path}")

    # Aggregate rankings
    aggregate = build_aggregate_rankings(results)
    agg_path = out_dir / "aggregate_feature_rankings.csv"
    pd.DataFrame(aggregate).to_csv(agg_path, index=False)
    print(f"Saved: {agg_path}")

    print("\nTop 15 features by mean peak |r| across participants:")
    for row in aggregate[:15]:
        print(f"  {row['feature']}: mean |r|={row['mean_peak_abs_r']:.3f}, "
              f"mean offset={row['mean_optimal_offset']:.0f}min, n={row['n_participants']}")

    # Plots
    feature_names = list(all_data.values())[0]["feature_names"]
    for pname in results:
        plot_heatmap(results, pname, feature_names, offsets, fig_dir)
        plot_top_features_per_offset(results, pname, top_n=15, fig_dir=fig_dir)

    # HTML report
    report_path = out_dir / "offset_by_feature_report.html"
    html = _build_html_report(results, aggregate, out_dir, fig_dir)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport: {report_path}")


def _build_html_report(results, aggregate, out_dir, fig_dir):
    rel_fig = "figures"
    participant_sections = ""
    for pname in sorted(results.keys()):
        opt = results[pname]["optimal"][:15]
        rows = "".join(
            f"<tr><td>{x['feature']}</td><td>{x['best_offset']:+.0f}</td><td>{x['best_r']:.3f}</td></tr>"
            for x in opt
        )
        participant_sections += f"""
        <h3>{pname}</h3>
        <img src="{rel_fig}/top_features_{pname}.png" alt="Top features" style="max-width:100%;">
        <table style="margin-top:10px;"><tr><th>Feature</th><th>Best offset (min)</th><th>r</th></tr>{rows}</table>
        """

    agg_rows = "".join(
        f"<tr><td>{r['feature']}</td><td>{r['mean_peak_abs_r']:.3f}</td>"
        f"<td>{r['mean_optimal_offset']:.0f}</td><td>{r['n_participants']}</td></tr>"
        for r in aggregate[:25]
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Offset-by-Feature Analysis</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:20px auto;}} table{{border-collapse:collapse;}}
th,td{{border:1px solid #ddd;padding:6px 10px;}} th{{background:#1976d2;color:white;}}</style></head>
<body>
<h1>Offset-by-Feature Sensitivity Analysis</h1>
<p>Which features are most glucose-sensitive at which CGM-to-voice time offsets?
Positive offset = voice recorded before CGM reading (voice may lead).
Negative offset = CGM reading before voice.</p>

<h2>Aggregate Rankings (mean peak |r| across participants)</h2>
<table><tr><th>Feature</th><th>Mean peak |r|</th><th>Mean optimal offset</th><th>N participants</th></tr>
{agg_rows}</table>

<h2>Per-Participant Top Features</h2>
{participant_sections}

<p><em>Generated {datetime.now().isoformat()}</em></p>
</body></html>"""


if __name__ == "__main__":
    main()
