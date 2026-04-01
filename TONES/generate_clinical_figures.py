#!/usr/bin/env python3
"""
Generate publication-quality clinical domain figures for TONES.

Outputs:
  - Clarke Error Grid (per-participant and aggregate)
  - Bland-Altman plot
  - Per-participant waterfall MAE comparison
  - Offset heatmap (voice leads CGM)
  - Model comparison bar chart
"""

import sys, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut, KFold

from tones.config import load_config, get_base_dir
from tones.data.loaders import load_participant_data
from tones.features.mfcc import MFCCExtractor
from tones.models.train import compute_metrics

OUT = Path("final_documentation/clinical_figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e74c3c", "E": "#8e44ad"}
P_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def clarke_zone_vectorized(refs, preds):
    zones = []
    for r, p in zip(refs, preds):
        if (r <= 70 and p <= 70) or (abs(p - r) / max(r, 1) <= 0.20):
            zones.append("A")
        elif abs(p - r) / max(r, 1) <= 0.40:
            zones.append("B")
        elif r > 180 and p < 70:
            zones.append("E")
        elif r < 70 and p > 180:
            zones.append("E")
        elif r >= 70 and p < 70:
            zones.append("D")
        else:
            zones.append("C")
    return zones


def extract_features_for_participant(pdata_df, cfg):
    sr = cfg.get("features", {}).get("sample_rate", 16000)
    n_mfcc = 20
    n_mels = 64
    fmin = 100
    fmax = 8000

    extractor = MFCCExtractor(
        sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
        fmin=fmin, fmax=fmax,
    )
    features, glucose_vals = [], []
    for _, row in pdata_df.iterrows():
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            continue
        try:
            feat = extractor.extract_from_file(str(audio_path))
            if feat is not None and np.isfinite(feat).all():
                features.append(feat)
                glucose_vals.append(row["glucose_mg_dl"])
        except Exception:
            continue
    if not features:
        return None, None
    return np.array(features, dtype=np.float32), np.array(glucose_vals, dtype=np.float64)


def plot_clarke_error_grid(refs, preds, labels, save_path):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    lo, hi = 40, max(max(refs), max(preds)) * 1.1
    hi = min(hi, 400)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.fill_between([lo, 70], lo, 70, alpha=0.06, color=COLORS["A"])
    xs = np.linspace(70, hi, 100)
    ax.fill_between(xs, xs * 0.8, xs * 1.2, alpha=0.06, color=COLORS["A"])

    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4, label="Perfect agreement")
    ax.plot([70, hi], [70 * 0.8, hi * 0.8], color=COLORS["A"], lw=1.0, alpha=0.4, ls="--")
    ax.plot([70, hi], [70 * 1.2, hi * 1.2], color=COLORS["A"], lw=1.0, alpha=0.4, ls="--")

    unique_labels = sorted(set(labels))
    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        r_sub = [r for r, m in zip(refs, mask) if m]
        p_sub = [p for p, m in zip(preds, mask) if m]
        ax.scatter(r_sub, p_sub, s=24, alpha=0.55,
                   c=P_COLORS[i % len(P_COLORS)],
                   edgecolors="white", linewidths=0.3, label=label, zorder=3)

    zones = clarke_zone_vectorized(refs, preds)
    total = len(zones)
    zc = {z: zones.count(z) for z in ["A", "B", "C", "D", "E"]}
    zp = {z: 100 * zc[z] / total for z in zc}

    info = "\n".join([f"Zone {z}: {zc[z]:>4d} ({zp[z]:5.1f}%)" for z in ["A", "B", "C", "D", "E"]])
    info += f"\n{'_' * 24}\nA+B:  {zc['A'] + zc['B']:>4d} ({zp['A'] + zp['B']:5.1f}%)"
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bbb'))

    ax.text(hi * 0.82, lo + 8, "A", fontsize=36, fontweight="bold", color=COLORS["A"], alpha=0.25)
    ax.text(hi * 0.50, lo + 8, "B", fontsize=28, fontweight="bold", color=COLORS["B"], alpha=0.2)

    ax.set_xlabel("Reference CGM Glucose (mg/dL)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Voice-Predicted Glucose (mg/dL)", fontsize=12, fontweight="bold")
    ax.set_title("Clarke Error Grid Analysis\nVoice-Based Non-Invasive Glucose Estimation (TONES)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.12)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(save_path).replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Clarke Error Grid -> {save_path}")
    return zc, zp


def plot_bland_altman(refs, preds, labels, save_path):
    refs, preds = np.array(refs), np.array(preds)
    means = (refs + preds) / 2
    diffs = preds - refs
    md, sd = np.mean(diffs), np.std(diffs)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    unique_labels = sorted(set(labels))
    for i, label in enumerate(unique_labels):
        mask = np.array([l == label for l in labels])
        ax.scatter(means[mask], diffs[mask], s=20, alpha=0.5,
                   c=P_COLORS[i % len(P_COLORS)], edgecolors="white",
                   linewidths=0.3, label=label, zorder=3)

    ax.axhline(md, color="#2c3e50", ls="--", lw=1.5, label=f"Mean bias: {md:.1f} mg/dL")
    ax.axhline(md + 1.96 * sd, color="#e74c3c", ls=":", lw=1.2,
               label=f"+1.96 SD: {md + 1.96 * sd:.1f}")
    ax.axhline(md - 1.96 * sd, color="#e74c3c", ls=":", lw=1.2,
               label=f"-1.96 SD: {md - 1.96 * sd:.1f}")
    ax.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.5)
    xlim = ax.get_xlim()
    ax.fill_between(xlim, md - 1.96 * sd, md + 1.96 * sd, alpha=0.05, color="#e74c3c")
    ax.set_xlim(xlim)

    ax.set_xlabel("Mean of Reference and Predicted (mg/dL)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Difference (Predicted - Reference) (mg/dL)", fontsize=11, fontweight="bold")
    ax.set_title("Bland-Altman Analysis -- Voice vs. CGM Glucose", fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(save_path).replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Bland-Altman -> {save_path}")
    return md, sd


def plot_waterfall(participant_results, save_path):
    names = sorted(participant_results.keys(), key=lambda k: participant_results[k]["mae"])
    maes = [participant_results[n]["mae"] for n in names]
    rs = [participant_results[n]["r"] for n in names]
    ns = [participant_results[n]["n"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    clrs = []
    for m in maes:
        if m < 10:
            clrs.append("#2ecc71")
        elif m < 15:
            clrs.append("#3498db")
        elif m < 20:
            clrs.append("#f39c12")
        else:
            clrs.append("#e74c3c")

    ax.barh(range(len(names)), maes, color=clrs, edgecolor="white", height=0.65)
    for i, (m, r, n) in enumerate(zip(maes, rs, ns)):
        ax.text(m + 0.3, i, f"  {m:.1f} mg/dL  (r = {r:.2f}, n = {n})",
                va="center", fontsize=9, fontweight="bold")

    avg_mae = np.mean(maes)
    ax.axvline(avg_mae, color="#2c3e50", ls="--", lw=1.2, alpha=0.6)
    ax.text(avg_mae + 0.2, len(names) - 0.3, f"Avg: {avg_mae:.1f}",
            fontsize=9, color="#2c3e50", fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean Absolute Error (mg/dL)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Participant Personalized Model Performance\n(SVR, LOO/10-fold CV, MFCC-20)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.invert_yaxis()
    legend_els = [
        mpatches.Patch(color="#2ecc71", label="Excellent (< 10)"),
        mpatches.Patch(color="#3498db", label="Good (10-15)"),
        mpatches.Patch(color="#f39c12", label="Fair (15-20)"),
        mpatches.Patch(color="#e74c3c", label="Poor (>= 20)"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=8, title="MAE (mg/dL)")
    ax.grid(True, axis="x", alpha=0.12)
    ax.set_xlim(0, max(maes) + 10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(save_path).replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Waterfall -> {save_path}")


def plot_offset_heatmap(offset_csv, save_path):
    df = pd.read_csv(offset_csv)
    df_svr = df[df["model"] == "SVR"].copy()
    if df_svr.empty:
        df_svr = df.copy()
    pivot = df_svr.pivot_table(index="participant", columns="offset", values="mae")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    cmap = LinearSegmentedColormap.from_list("rg", ["#2ecc71", "#f1c40f", "#e74c3c"])
    vmin = max(5, pivot.values.min() - 2)
    vmax = min(25, pivot.values.max() + 2)
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8, label="MAE (mg/dL)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c):+d} min" for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10, fontweight="bold")

    for i in range(pivot.shape[0]):
        best_j = int(np.argmin(pivot.values[i]))
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            fw = "bold" if j == best_j else "normal"
            clr = "white" if val > (vmin + vmax) / 2 else "black"
            marker = " *" if j == best_j else ""
            ax.text(j, i, f"{val:.1f}{marker}", ha="center", va="center",
                    fontsize=8, fontweight=fw, color=clr)

    zero_col = list(pivot.columns).index(0) if 0 in list(pivot.columns) else None
    if zero_col is not None:
        ax.axvline(zero_col, color="white", ls="--", lw=1.5, alpha=0.6)

    pos_cols = [c for c in pivot.columns if c > 0]
    n_pos_best = 0
    for i in range(pivot.shape[0]):
        best_offset = int(pivot.columns[np.argmin(pivot.values[i])])
        if best_offset > 0:
            n_pos_best += 1

    ax.set_xlabel("Time Offset (minutes) -- voice timestamp relative to CGM", fontsize=11, fontweight="bold")
    ax.set_title("Voice-CGM Temporal Offset Analysis (SVR)\n"
                 f"Positive offset = voice leads CGM  |  {n_pos_best}/{pivot.shape[0]} participants best at positive offset",
                 fontsize=12, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(save_path).replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Offset Heatmap -> {save_path}")

    best = {}
    for p in pivot.index:
        row = pivot.loc[p]
        best[p] = {"offset": int(row.idxmin()), "mae": float(row.min())}
    return best


def plot_model_comparison(pers_csv, save_path):
    df = pd.read_csv(pers_csv)
    grouped = df.groupby("model")["pers_mae"].agg(["mean", "min", "count"]).reset_index()
    grouped = grouped.sort_values("mean")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    bar_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    bars = ax.bar(range(len(grouped)), grouped["mean"],
                  color=bar_colors[:len(grouped)], edgecolor="white", width=0.6)
    for i, (_, row) in enumerate(grouped.iterrows()):
        ax.text(i, row["mean"] + 0.12,
                f"avg {row['mean']:.2f}\nbest {row['min']:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels(grouped["model"], fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean MAE (mg/dL)", fontsize=11, fontweight="bold")
    ax.set_title("Model Comparison -- Personalized MAE (Edge Optimization Sweep)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(str(save_path).replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Model Comparison -> {save_path}")


def main():
    print("=" * 70)
    print("TONES -- Clinical Domain Figure Generation")
    print("=" * 70)

    cfg = load_config()
    base = get_base_dir()
    matching = cfg.get("matching", {})

    print("\n1) Loading data and running per-participant SVR predictions...\n")

    all_refs, all_preds, all_labels = [], [], []
    participant_results = {}

    for name, pcfg in cfg["participants"].items():
        pdata_df = load_participant_data(name, pcfg, base, matching)
        if pdata_df.empty:
            print(f"  {name}: no matched data")
            continue

        X, y = extract_features_for_participant(pdata_df, cfg)
        if X is None or len(y) < 20:
            n = 0 if y is None else len(y)
            print(f"  {name}: skip (n={n}, need >= 20)")
            continue

        cv = LeaveOneOut() if len(y) <= 50 else KFold(n_splits=10, shuffle=True, random_state=42)
        preds = np.zeros_like(y, dtype=np.float64)
        for tr, te in cv.split(X):
            pipe = Pipeline([("scaler", RobustScaler()), ("model", SVR(C=10, gamma="scale"))])
            pipe.fit(X[tr], y[tr])
            preds[te] = pipe.predict(X[te])

        m = compute_metrics(y, preds)
        participant_results[name] = {"mae": float(m["mae"]), "r": float(m["r"]), "n": len(y)}
        all_refs.extend(y.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend([name] * len(y))
        print(f"  {name}: n={len(y)}, MAE={m['mae']:.2f}, r={m['r']:.3f}")

    if not all_refs:
        print("ERROR: No predictions generated.")
        sys.exit(1)

    print(f"\n  Total: {len(all_refs)} predictions across {len(participant_results)} participants\n")

    print("2) Generating clinical figures...\n")

    zc, zp = plot_clarke_error_grid(all_refs, all_preds, all_labels, OUT / "clarke_error_grid.png")
    md, sd = plot_bland_altman(all_refs, all_preds, all_labels, OUT / "bland_altman.png")
    plot_waterfall(participant_results, OUT / "participant_waterfall.png")

    offset_csv = Path("output/offset_window/offset_sweep_results.csv")
    best_offsets = {}
    if offset_csv.exists():
        best_offsets = plot_offset_heatmap(offset_csv, OUT / "offset_heatmap.png")
    else:
        print("  [SKIP] offset_sweep_results.csv not found")

    pers_csv = Path("output/edge_opt/leaderboard_personalized.csv")
    if pers_csv.exists():
        plot_model_comparison(pers_csv, OUT / "model_comparison.png")
    else:
        print("  [SKIP] leaderboard_personalized.csv not found")

    summary = {
        "n_total_samples": len(all_refs),
        "n_participants": len(participant_results),
        "clarke_zones": zc,
        "clarke_zone_pcts": {k: round(v, 1) for k, v in zp.items()},
        "clarke_ab_pct": round(zp["A"] + zp["B"], 1),
        "bland_altman_mean_bias": round(md, 2),
        "bland_altman_sd": round(sd, 2),
        "bland_altman_95_loa": [round(md - 1.96 * sd, 2), round(md + 1.96 * sd, 2)],
        "per_participant": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in participant_results.items()},
        "optimal_offsets_svr": best_offsets,
        "overall_mae": round(np.mean([v["mae"] for v in participant_results.values()]), 2),
        "overall_r": round(np.mean([v["r"] for v in participant_results.values()]), 3),
    }

    with open(OUT / "clinical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("CLINICAL FIGURES COMPLETE")
    print(f"  Zone A: {zp['A']:.1f}%  |  Zone A+B: {zp['A'] + zp['B']:.1f}%")
    print(f"  Overall MAE: {summary['overall_mae']} mg/dL  |  Overall r: {summary['overall_r']}")
    print(f"  Bland-Altman bias: {md:.2f} +/- {sd:.2f} mg/dL")
    print(f"  Figures: {OUT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
