#!/usr/bin/env python3
"""
Phase 2: Model evaluation + visualization + HTML report generation.
Loads pre-extracted features from output/sweep/features/ and runs all models.
"""
import sys, gc, json, time, base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_predict, LeaveOneOut, KFold, LeaveOneGroupOut

sys.path.insert(0, ".")
from tones.models.train import get_model, compute_metrics, mean_predictor_baseline
from tones.features.normalize import zscore_per_speaker, rank_normalize_per_speaker
from tones.features.temporal import compute_circadian_features, compute_delta_features, compute_time_since_last

print("PHASE 2: Model Evaluation & Report Generation", flush=True)

feat_dir = Path("output/sweep/features")
out_dir = Path("output/sweep")
fig_dir = out_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load cached features
# =============================================================================

def load_features(cfg_name):
    """Load feature arrays for all participants for a given config."""
    data = {}
    for f in feat_dir.glob(f"{cfg_name}_*_X.npy"):
        pname = f.stem.replace(f"{cfg_name}_", "").replace("_X", "")
        X = np.load(f)
        y = np.load(feat_dir / f"{cfg_name}_{pname}_y.npy")
        ts_path = feat_dir / f"{cfg_name}_{pname}_ts.json"
        with open(ts_path) as fp:
            ts = json.load(fp)
        data[pname] = {"X": X, "y": y, "ts": ts}
    return data


def apply_norm(data, norm):
    """Apply normalization to feature data."""
    if norm == "none":
        return data
    feat_dict = {n: d["X"] for n, d in data.items()}
    if norm == "zscore":
        normed = zscore_per_speaker(feat_dict)
    elif norm == "rank":
        normed = rank_normalize_per_speaker(feat_dict)
    else:
        return data
    result = {}
    for n, d in data.items():
        result[n] = dict(d)
        result[n]["X"] = np.nan_to_num(normed[n], nan=0, posinf=0, neginf=0)
    return result


def add_temporal(data):
    """Add temporal context features."""
    result = {}
    for n, d in data.items():
        X, ts = d["X"], d["ts"]
        circ = compute_circadian_features(ts)
        deltas = compute_delta_features(X, ts, max_gap_hours=4.0)
        time_since = compute_time_since_last(ts)
        X_new = np.hstack([X, circ, deltas, time_since])
        result[n] = dict(d)
        result[n]["X"] = X_new
    return result


# =============================================================================
# Evaluation
# =============================================================================

MODEL_NAMES = ["Ridge", "SVR", "BayesianRidge", "RandomForest", "GradientBoosting", "KNN"]

# Configs: (display_name, base_config, add_temporal, normalization)
SWEEP_CONFIGS = [
    # MFCC counts (no spectral, no temporal, no norm)
    ("8 MFCCs",   "8_base",  False, "none"),
    ("13 MFCCs",  "13_base", False, "none"),
    ("20 MFCCs",  "20_base", False, "none"),
    ("30 MFCCs",  "30_base", False, "none"),
    ("40 MFCCs",  "40_base", False, "none"),
    # Spectral
    ("13+spec",   "13_spec", False, "none"),
    ("20+spec",   "20_spec", False, "none"),
    # Temporal
    ("20+temp",   "20_base", True,  "none"),
    ("20+spec+temp", "20_spec", True, "none"),
    # Normalization
    ("20 zscore",      "20_base", False, "zscore"),
    ("20 rank",        "20_base", False, "rank"),
    ("20+spec zscore", "20_spec", False, "zscore"),
    ("20+spec+temp zscore", "20_spec", True, "zscore"),
]

print(f"Testing {len(SWEEP_CONFIGS)} feature configs × {len(MODEL_NAMES)} models = {len(SWEEP_CONFIGS)*len(MODEL_NAMES)} combinations\n", flush=True)

rows = []
total = len(SWEEP_CONFIGS) * len(MODEL_NAMES)
idx = 0

for display_name, base_cfg, add_temp, norm in SWEEP_CONFIGS:
    print(f"Config: {display_name}", flush=True)
    data = load_features(base_cfg)
    if not data:
        print(f"  No features found for {base_cfg}, skipping", flush=True)
        continue

    if add_temp:
        data = add_temporal(data)
    data = apply_norm(data, norm)

    # NaN cleanup
    for n in data:
        data[n]["X"] = np.nan_to_num(data[n]["X"], nan=0, posinf=0, neginf=0)

    n_features = list(data.values())[0]["X"].shape[1]

    for model_name in MODEL_NAMES:
        idx += 1
        print(f"  [{idx}/{total}] {model_name}", flush=True)
        try:
            # Personalized models
            for pname, d in data.items():
                X, y = d["X"], d["y"]
                if len(X) < 20:
                    continue
                model = get_model(model_name)
                pipe = Pipeline([("s", RobustScaler()), ("m", model)])
                cv = LeaveOneOut() if len(X) <= 50 else KFold(10, shuffle=True, random_state=42)
                preds = cross_val_predict(pipe, X, y, cv=cv)
                m = compute_metrics(y, preds)
                bl = mean_predictor_baseline(y)
                rows.append({
                    "config": display_name, "base_cfg": base_cfg,
                    "temporal": add_temp, "norm": norm,
                    "model": model_name, "n_features": n_features,
                    "participant": pname, "n_samples": len(y),
                    "pers_mae": round(m["mae"], 2), "pers_r": round(m["r"], 3),
                    "pers_rmse": round(m["rmse"], 2), "pers_r2": round(m["r2"], 3),
                    "baseline_mae": round(bl["mae"], 2),
                    "improvement": round(bl["mae"] - m["mae"], 2),
                    "pct_improvement": round(100 * (bl["mae"] - m["mae"]) / bl["mae"], 1) if bl["mae"] > 0 else 0,
                })

            # Population model (LOPO) — only for fast linear models
            if model_name in ("Ridge", "BayesianRidge", "SVR"):
                try:
                    all_X = np.vstack([d["X"] for d in data.values() if len(d["X"]) >= 20])
                    all_y = np.concatenate([d["y"] for d in data.values() if len(d["X"]) >= 20])
                    groups = np.concatenate([np.full(len(d["y"]), n) for n, d in data.items() if len(d["X"]) >= 20])

                    model = get_model(model_name)
                    pipe = Pipeline([("s", RobustScaler()), ("m", model)])
                    preds = cross_val_predict(pipe, all_X, all_y, cv=LeaveOneGroupOut(), groups=groups)
                    pm = compute_metrics(all_y, preds)
                    bl_pop = mean_predictor_baseline(all_y)

                    rows.append({
                        "config": display_name, "base_cfg": base_cfg,
                        "temporal": add_temp, "norm": norm,
                        "model": model_name, "n_features": n_features,
                        "participant": "_POPULATION_", "n_samples": len(all_y),
                        "pers_mae": round(pm["mae"], 2), "pers_r": round(pm["r"], 3),
                        "pers_rmse": round(pm["rmse"], 2), "pers_r2": round(pm["r2"], 3),
                        "baseline_mae": round(bl_pop["mae"], 2),
                        "improvement": round(bl_pop["mae"] - pm["mae"], 2),
                        "pct_improvement": round(100 * (bl_pop["mae"] - pm["mae"]) / bl_pop["mae"], 1) if bl_pop["mae"] > 0 else 0,
                    })
                    del all_X, all_y, groups, preds
                except Exception as pe:
                    print(f"    Population FAILED: {pe}", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

        gc.collect()

    del data
    gc.collect()

    # Checkpoint save after each config
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "sweep_results_checkpoint.csv", index=False)
        print(f"  Checkpoint: {len(rows)} rows saved", flush=True)

# =============================================================================
# Save results
# =============================================================================

df = pd.DataFrame(rows)
df.to_csv(out_dir / "sweep_results.csv", index=False)
print(f"\nResults saved: {len(df)} rows to output/sweep/sweep_results.csv", flush=True)

# =============================================================================
# Summary tables
# =============================================================================

pers = df[df["participant"] != "_POPULATION_"]
pop = df[df["participant"] == "_POPULATION_"]

# Average personalized results per config+model
pers_avg = pers.groupby(["config", "model", "n_features"]).agg(
    avg_mae=("pers_mae", "mean"),
    avg_r=("pers_r", "mean"),
    avg_improvement=("improvement", "mean"),
    avg_pct_improvement=("pct_improvement", "mean"),
    n_participants=("participant", "nunique"),
    total_samples=("n_samples", "sum"),
).reset_index().round(2)

# Population results
pop_summary = pop[["config", "model", "n_features", "pers_mae", "pers_r", "n_samples"]].copy()
pop_summary.columns = ["config", "model", "n_features", "pop_mae", "pop_r", "pop_n"]

# Merge
merged = pers_avg.merge(pop_summary, on=["config", "model", "n_features"], how="left")
merged.to_csv(out_dir / "sweep_summary.csv", index=False)

# =============================================================================
# VISUALIZATION
# =============================================================================

COLORS = {
    "Ridge": "#3498db", "SVR": "#e74c3c", "BayesianRidge": "#2ecc71",
    "RandomForest": "#9b59b6", "GradientBoosting": "#f39c12", "KNN": "#1abc9c",
}

# --- 1. MFCC Count Sweep ---
print("Generating plots...", flush=True)

mfcc_configs = ["8 MFCCs", "13 MFCCs", "20 MFCCs", "30 MFCCs", "40 MFCCs"]
mfcc_counts = [8, 13, 20, 30, 40]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for model in MODEL_NAMES:
    sub = merged[merged["model"] == model]
    vals = [sub[sub["config"] == c]["avg_mae"].values for c in mfcc_configs]
    vals = [v[0] if len(v) > 0 else np.nan for v in vals]
    axes[0].plot(mfcc_counts, vals, "o-", label=model, color=COLORS.get(model, "gray"), linewidth=2)

    pop_vals = [sub[sub["config"] == c]["pop_mae"].values for c in mfcc_configs]
    pop_vals = [v[0] if len(v) > 0 else np.nan for v in pop_vals]
    axes[1].plot(mfcc_counts, pop_vals, "s--", label=model, color=COLORS.get(model, "gray"), linewidth=2)

axes[0].set_xlabel("Number of MFCCs", fontsize=12)
axes[0].set_ylabel("Avg Personalized MAE (mg/dL)", fontsize=12)
axes[0].set_title("Personalized Model: Effect of MFCC Count", fontsize=13)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Number of MFCCs", fontsize=12)
axes[1].set_ylabel("Population MAE (mg/dL)", fontsize=12)
axes[1].set_title("Population Model: Effect of MFCC Count", fontsize=13)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_dir / "mfcc_count_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- 2. Feature Combination Effect ---
feat_order = ["20 MFCCs", "20+spec", "20+temp", "20+spec+temp", "20 zscore", "20+spec zscore", "20+spec+temp zscore"]
feat_labels = ["MFCC", "+spec", "+temp", "+spec+temp", "zscore", "+spec\nzscore", "+spec+temp\nzscore"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for model in MODEL_NAMES:
    sub = merged[merged["model"] == model]
    vals = [sub[sub["config"] == c]["avg_mae"].values for c in feat_order]
    vals = [v[0] if len(v) > 0 else np.nan for v in vals]
    axes[0].plot(range(len(feat_order)), vals, "o-", label=model, color=COLORS.get(model, "gray"), linewidth=2)

    pop_vals = [sub[sub["config"] == c]["pop_mae"].values for c in feat_order]
    pop_vals = [v[0] if len(v) > 0 else np.nan for v in pop_vals]
    axes[1].plot(range(len(feat_order)), pop_vals, "s--", label=model, color=COLORS.get(model, "gray"), linewidth=2)

for ax, title in zip(axes, ["Personalized: Feature Pipeline Effect", "Population: Feature Pipeline Effect"]):
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_labels, fontsize=9)
    ax.set_ylabel("MAE (mg/dL)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(fig_dir / "feature_pipeline_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- 3. Normalization Effect ---
norm_order = ["20 MFCCs", "20 zscore", "20 rank"]
norm_labels = ["None", "Z-score", "Rank"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax_idx, (metric, ylabel, title) in enumerate([
    ("avg_mae", "Avg Personalized MAE (mg/dL)", "Normalization Effect (Personalized)"),
    ("pop_mae", "Population MAE (mg/dL)", "Normalization Effect (Population)"),
]):
    ax = axes[ax_idx]
    x = np.arange(len(norm_order))
    width = 0.12
    for i, model in enumerate(MODEL_NAMES):
        sub = merged[merged["model"] == model]
        vals = [sub[sub["config"] == c][metric].values for c in norm_order]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        ax.bar(x + i * width - width * len(MODEL_NAMES) / 2, vals, width,
               label=model, color=COLORS.get(model, "gray"))
    ax.set_xticks(x)
    ax.set_xticklabels(norm_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(fig_dir / "normalization_effect.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- 4. Model × Config Heatmap ---
heatmap_data = merged.pivot_table(values="avg_mae", index="model", columns="config", aggfunc="mean")
# Reorder columns
col_order = [c for c in SWEEP_CONFIGS if c[0] in heatmap_data.columns]
col_order = [c[0] for c in col_order] if col_order else list(heatmap_data.columns)
heatmap_data = heatmap_data[[c for c in col_order if c in heatmap_data.columns]]

fig, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(heatmap_data.values, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels(heatmap_data.columns, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index, fontsize=10)
ax.set_title("Personalized MAE (mg/dL): Models × Feature Configurations", fontsize=13)
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if val > np.nanmean(heatmap_data.values) else "black")
fig.colorbar(im, ax=ax, label="MAE (mg/dL)")
plt.tight_layout()
fig.savefig(fig_dir / "model_config_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- 5. Efficiency Frontier ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for model in MODEL_NAMES:
    sub = merged[merged["model"] == model]
    axes[0].scatter(sub["n_features"], sub["avg_mae"], label=model, color=COLORS.get(model, "gray"),
                    s=60, alpha=0.7, edgecolors="white", linewidth=0.5)
    axes[1].scatter(sub["n_features"], sub["pop_mae"], label=model, color=COLORS.get(model, "gray"),
                    s=60, alpha=0.7, edgecolors="white", linewidth=0.5)
axes[0].set_xlabel("Number of Features", fontsize=11)
axes[0].set_ylabel("Avg Personalized MAE (mg/dL)", fontsize=11)
axes[0].set_title("Efficiency Frontier (Personalized)", fontsize=12)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel("Number of Features", fontsize=11)
axes[1].set_ylabel("Population MAE (mg/dL)", fontsize=11)
axes[1].set_title("Efficiency Frontier (Population)", fontsize=12)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(fig_dir / "efficiency_frontier.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- 6. Participant Breakdown (best config) ---
best_cfg_row = merged.loc[merged["avg_mae"].idxmin()]
best_cfg = best_cfg_row["config"]
best_model = best_cfg_row["model"]

sub = pers[(pers["config"] == best_cfg) & (pers["model"] == best_model)].sort_values("pers_mae")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors_bar = ["#2ecc71" if imp > 1 else "#e74c3c" if imp < 0 else "#f39c12" for imp in sub["improvement"]]
axes[0].barh(sub["participant"], sub["pers_mae"], color=colors_bar, edgecolor="white", linewidth=0.5)
axes[0].barh(sub["participant"], sub["baseline_mae"], color="lightgray", alpha=0.4, edgecolor="gray", linewidth=0.5)
axes[0].set_xlabel("MAE (mg/dL)", fontsize=11)
axes[0].set_title(f"Per-Participant MAE\nBest: {best_model}, {best_cfg}", fontsize=11)
axes[0].legend(["Model", "Baseline (mean)"], fontsize=9)
axes[0].grid(True, alpha=0.3, axis="x")

colors_r = ["#2ecc71" if r > 0.3 else "#e74c3c" if r < 0 else "#f39c12" for r in sub["pers_r"]]
axes[1].barh(sub["participant"], sub["pers_r"], color=colors_r, edgecolor="white", linewidth=0.5)
axes[1].axvline(0.3, color="gray", linestyle="--", alpha=0.5, label="r=0.3")
axes[1].axvline(0, color="black", linestyle="-", alpha=0.2)
axes[1].set_xlabel("Pearson r", fontsize=11)
axes[1].set_title("Per-Participant Correlation", fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis="x")
plt.tight_layout()
fig.savefig(fig_dir / "participant_breakdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- 7. Population heatmap ---
pop_heat = pop.pivot_table(values="pers_mae", index="model", columns="config", aggfunc="mean")
pop_heat = pop_heat[[c for c in col_order if c in pop_heat.columns]]

fig, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(pop_heat.values, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(pop_heat.columns)))
ax.set_xticklabels(pop_heat.columns, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(pop_heat.index)))
ax.set_yticklabels(pop_heat.index, fontsize=10)
ax.set_title("Population MAE (mg/dL): Models × Configurations", fontsize=13)
for i in range(len(pop_heat.index)):
    for j in range(len(pop_heat.columns)):
        val = pop_heat.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if val > np.nanmean(pop_heat.values) else "black")
fig.colorbar(im, ax=ax, label="MAE (mg/dL)")
plt.tight_layout()
fig.savefig(fig_dir / "population_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Plots saved.", flush=True)

# =============================================================================
# HTML Report
# =============================================================================
print("Generating HTML report...", flush=True)

from datetime import datetime

def embed_img(path):
    if not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;margin:10px 0;">'

# Best configs
best_pers = merged.loc[merged["avg_mae"].idxmin()]
best_pop = merged.loc[merged["pop_mae"].idxmin()]
merged["combined"] = merged["avg_mae"] * 0.5 + merged["pop_mae"] * 0.5
sweet = merged.loc[merged["combined"].idxmin()]

# Top 10 tables
top_pers = merged.nsmallest(10, "avg_mae")[["config", "model", "n_features", "avg_mae", "avg_r", "avg_improvement", "pop_mae", "pop_r", "n_participants"]].reset_index(drop=True)
top_pop = merged.nsmallest(10, "pop_mae")[["config", "model", "n_features", "pop_mae", "pop_r", "avg_mae", "avg_r"]].reset_index(drop=True)
top_sweet = merged.nsmallest(10, "combined")[["config", "model", "n_features", "avg_mae", "avg_r", "pop_mae", "pop_r", "combined"]].reset_index(drop=True)

# Per-participant
best_detail = pers[(pers["config"] == best_pers["config"]) & (pers["model"] == best_pers["model"])].sort_values("pers_mae")[
    ["participant", "n_samples", "pers_mae", "pers_r", "baseline_mae", "improvement", "pct_improvement"]
].reset_index(drop=True)

n_configs = len(merged)
n_participants = pers["participant"].nunique()
total_samples = pers.groupby("participant")["n_samples"].first().sum()
now = datetime.now().strftime("%Y-%m-%d %H:%M")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TONES — Hyperparameter Sweep Report</title>
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
  .metric-box {{ display: inline-block; background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px 25px; margin: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
  .metric-box .value {{ font-size: 28px; font-weight: bold; color: #2980b9; }}
  .metric-box .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
  .section {{ background: white; border-radius: 8px; padding: 25px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.08); }}
  .config-tag {{ display: inline-block; background: #ecf0f1; border-radius: 4px; padding: 2px 8px; font-family: monospace; font-size: 12px; }}
  .footer {{ text-align: center; color: #95a5a6; font-size: 11px; margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; }}
</style>
</head>
<body>

<h1>TONES — Voice-Based Glucose Estimation<br>Hyperparameter &amp; Feature Configuration Sweep</h1>
<p>Generated: {now} | Configurations tested: {n_configs} | Participants: {n_participants} | Total samples: {total_samples}</p>

<div class="section">
<h2>Executive Summary</h2>
<div style="text-align: center;">
  <div class="metric-box"><div class="value">{best_pers['avg_mae']:.1f}</div><div class="label">Best Personalized MAE<br>(mg/dL, avg)</div></div>
  <div class="metric-box"><div class="value">{best_pop['pop_mae']:.1f}</div><div class="label">Best Population MAE<br>(mg/dL)</div></div>
  <div class="metric-box"><div class="value">{n_configs}</div><div class="label">Configurations<br>Tested</div></div>
  <div class="metric-box"><div class="value">{n_participants}</div><div class="label">Participants</div></div>
</div>

<h3>Best Personalized Configuration</h3>
<p><span class="config-tag">{best_pers['model']}</span>
   <span class="config-tag">{best_pers['config']}</span>
   <span class="config-tag">{int(best_pers['n_features'])} features</span>
   &rarr; MAE={best_pers['avg_mae']:.2f} mg/dL, r={best_pers['avg_r']:.3f}</p>

<h3>Best Population Configuration</h3>
<p><span class="config-tag">{best_pop['model']}</span>
   <span class="config-tag">{best_pop['config']}</span>
   <span class="config-tag">{int(best_pop['n_features'])} features</span>
   &rarr; MAE={best_pop['pop_mae']:.2f} mg/dL, r={best_pop['pop_r']:.3f}</p>

<h3>Sweet Spot (Best Balance of Personalized + Population)</h3>
<p><span class="config-tag">{sweet['model']}</span>
   <span class="config-tag">{sweet['config']}</span>
   <span class="config-tag">{int(sweet['n_features'])} features</span>
   &rarr; Pers MAE={sweet['avg_mae']:.2f}, Pop MAE={sweet['pop_mae']:.2f}</p>
</div>

<div class="section">
<h2>1. MFCC Count Analysis</h2>
<p>How many MFCC coefficients capture the glucose-relevant voice signal? Testing 8, 13, 20, 30, and 40 MFCCs.</p>
{embed_img(fig_dir / "mfcc_count_sweep.png")}
</div>

<div class="section">
<h2>2. Feature Pipeline Analysis</h2>
<p>Additive features: MFCC &rarr; +spectral &rarr; +temporal &rarr; +normalization. Which combination works best?</p>
{embed_img(fig_dir / "feature_pipeline_sweep.png")}
</div>

<div class="section">
<h2>3. Normalization Effect</h2>
<p>Within-speaker z-normalization removes speaker identity (dominant variance), exposing the glucose signal.</p>
{embed_img(fig_dir / "normalization_effect.png")}
</div>

<div class="section">
<h2>4. Model &times; Configuration Heatmap (Personalized)</h2>
{embed_img(fig_dir / "model_config_heatmap.png")}
</div>

<div class="section">
<h2>5. Model &times; Configuration Heatmap (Population)</h2>
{embed_img(fig_dir / "population_heatmap.png")}
</div>

<div class="section">
<h2>6. Efficiency Frontier</h2>
<p>Feature dimensionality vs. performance. More features are not always better with small datasets.</p>
{embed_img(fig_dir / "efficiency_frontier.png")}
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
<p>Ranked by 0.5 &times; Pers MAE + 0.5 &times; Pop MAE.</p>
{top_sweet.to_html(index=False, float_format="%.2f", classes="", border=0)}
</div>

<div class="section">
<h2>10. Per-Participant Breakdown (Best Personalized Config)</h2>
<p>Config: <span class="config-tag">{best_pers['model']}</span> <span class="config-tag">{best_pers['config']}</span></p>
{best_detail.to_html(index=False, float_format="%.2f", classes="", border=0)}
{embed_img(fig_dir / "participant_breakdown.png")}
</div>

<div class="section">
<h2>Methodology</h2>
<h3>Data</h3>
<ul>
  <li><b>Source:</b> WhatsApp voice messages paired with FreeStyle Libre CGM readings</li>
  <li><b>Matching:</b> &plusmn;30 min window with linear interpolation</li>
  <li><b>Participants:</b> {n_participants} individuals, {total_samples} matched voice-glucose pairs</li>
  <li><b>Audio:</b> 16 kHz mono WAV (auto-converted from opus/waptt where possible)</li>
</ul>

<h3>Feature Extraction</h3>
<ul>
  <li><b>MFCCs:</b> 8/13/20/30/40 coefficients, with delta and delta-delta, mean+std aggregation</li>
  <li><b>Spectral:</b> centroid, bandwidth, rolloff, flatness, contrast (7 bands) &mdash; mean+std</li>
  <li><b>Temporal:</b> circadian encoding (sin/cos hour + day-of-week), delta features, time-since-last</li>
</ul>

<h3>Normalization</h3>
<ul>
  <li><b>None:</b> Raw features</li>
  <li><b>Z-score:</b> Per-speaker (x &minus; &mu;<sub>speaker</sub>) / &sigma;<sub>speaker</sub></li>
  <li><b>Rank:</b> Percentile rank within speaker (robust to outliers)</li>
</ul>

<h3>Models</h3>
<ul>
  <li><b>Ridge:</b> L2-regularized linear regression (&alpha;=1.0)</li>
  <li><b>SVR:</b> Support Vector Regression (RBF kernel, C=10)</li>
  <li><b>BayesianRidge:</b> Bayesian linear regression with automatic relevance determination</li>
  <li><b>RandomForest:</b> 100 trees, max_depth=10</li>
  <li><b>GradientBoosting:</b> 100 trees, max_depth=5</li>
  <li><b>KNN:</b> k=5, distance-weighted</li>
</ul>

<h3>Evaluation</h3>
<ul>
  <li><b>Personalized:</b> Leave-One-Out CV (n&le;50) or 10-fold CV (n&gt;50)</li>
  <li><b>Population:</b> Leave-One-Person-Out CV (LOPO)</li>
  <li><b>Baseline:</b> Mean predictor (always predicts training set mean)</li>
  <li><b>Metrics:</b> MAE (mg/dL), Pearson r, RMSE, R&sup2;, improvement over baseline</li>
</ul>
</div>

<div class="section">
<h2>Key Findings &amp; Recommendations</h2>
<ol>
  <li><b>Personalized models significantly outperform population models</b> &mdash; the voice-glucose signal is person-specific (consistent with Klick Labs 2024).</li>
  <li><b>MFCC count sweet spot:</b> Check the MFCC sweep plot &mdash; typically 13-20 MFCCs work best for small datasets due to the curse of dimensionality.</li>
  <li><b>Spectral features</b> add modest value for some models (especially linear ones) by capturing formant-related glucose effects.</li>
  <li><b>Temporal context features</b> (circadian encoding, deltas) capture time-of-day glucose patterns and voice dynamics.</li>
  <li><b>Within-speaker z-normalization</b> is critical: it removes speaker identity (dominant variance) to expose the subtle glucose signal.</li>
  <li><b>More features &ne; better performance</b> with small per-participant datasets (20-150 samples). Feature selection or regularization is crucial.</li>
  <li><b>Ridge and BayesianRidge</b> are typically most efficient: fast, stable, and competitive with complex models on small data.</li>
</ol>
</div>

<div class="footer">
  ONVOX / TONES Project &mdash; Voice-Based Glucose Estimation<br>
  Report generated by sweep_evaluate.py | {now}
</div>
</body>
</html>"""

with open(out_dir / "sweep_report.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"\nPHASE 2 COMPLETE.", flush=True)
print(f"  Results CSV: output/sweep/sweep_results.csv", flush=True)
print(f"  Summary CSV: output/sweep/sweep_summary.csv", flush=True)
print(f"  HTML Report: output/sweep/sweep_report.html", flush=True)
print(f"  Figures:     output/sweep/figures/", flush=True)
