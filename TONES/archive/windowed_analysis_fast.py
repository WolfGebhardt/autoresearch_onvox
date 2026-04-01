"""
Voice-Based Glucose Estimation - Fast Windowed Analysis
Optimized version with key window sizes only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import librosa
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneOut, cross_val_predict, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from scipy import stats

from voice_glucose_pipeline import PARTICIPANTS, BASE_DIR

# Output directories
OUTPUT_DIR = BASE_DIR / "documentation_v3"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Key window sizes to test
WINDOW_SIZES_MS = [500, 1000, 1500, 2000, 3000, 5000]


def extract_features_fast(audio_path, sr=16000, window_ms=2000):
    """Fast feature extraction with windowing."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=30)  # Limit to 30s

        window_samples = int(window_ms * sr / 1000)
        hop_samples = window_samples // 2

        if len(y) < window_samples:
            window_samples = len(y)
            hop_samples = len(y)

        window_features = []

        for start in range(0, max(1, len(y) - window_samples + 1), hop_samples):
            end = min(start + window_samples, len(y))
            y_window = y[start:end]

            if len(y_window) < 1000:  # Skip very short windows
                continue

            features = {}

            # MFCCs (faster with fewer)
            mfccs = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}'] = np.mean(mfccs[i])

            # MFCC deltas
            mfcc_delta = librosa.feature.delta(mfccs)
            for i in range(13):
                features[f'mfcc_delta_{i+1}'] = np.mean(mfcc_delta[i])

            # Key spectral features only
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y_window, sr=sr))
            features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y_window, sr=sr))
            features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y_window))
            features['rms'] = np.mean(librosa.feature.rms(y=y_window))

            window_features.append(features)

        if not window_features:
            return None

        # Aggregate
        aggregated = {}
        for fname in window_features[0].keys():
            values = [wf[fname] for wf in window_features]
            aggregated[f'{fname}_wmean'] = np.mean(values)
            aggregated[f'{fname}_wstd'] = np.std(values) if len(values) > 1 else 0

        aggregated['n_windows'] = len(window_features)
        return aggregated

    except Exception as e:
        return None


def load_participant_data(name, config, window_ms=2000):
    """Load and process data for one participant."""
    from voice_glucose_pipeline import load_multiple_glucose_csvs

    # Load glucose
    csv_paths = [BASE_DIR / p for p in config.get('glucose_csv', [])]
    existing_csvs = [p for p in csv_paths if p.exists()]
    if not existing_csvs:
        return None

    glucose_df = load_multiple_glucose_csvs(existing_csvs, config.get('glucose_unit', 'auto'))
    if glucose_df is None or len(glucose_df) == 0:
        return None

    # Find audio files
    audio_files = []
    for audio_dir in config.get('audio_dirs', []):
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(dir_path.glob('*.wav'))
            audio_files.extend(dir_path.glob('*.WAV'))

    audio_files = list(set(audio_files))
    if not audio_files:
        return None

    # Process samples
    samples = []
    glucose_from_filename = config.get('glucose_in_filename', False)

    for audio_path in audio_files:
        # Get glucose
        if glucose_from_filename:
            import re
            # Handle both "123_WhatsApp..." and "AnyConv.com__123_WhatsApp..."
            match = re.search(r'(?:^|__)(\d+)_', audio_path.name)
            if not match:
                continue
            glucose_mgdl = float(match.group(1))
        else:
            import re
            from datetime import timedelta

            # Parse timestamp
            match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})', str(audio_path))
            if not match:
                continue

            try:
                date_str = match.group(1)
                parts = date_str.split('-')
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                hour, minute = int(match.group(2)), int(match.group(3))
                timestamp = datetime(year, month, day, hour, minute)
            except:
                continue

            window = timedelta(minutes=15)
            mask = (glucose_df['timestamp'] >= timestamp - window) & \
                   (glucose_df['timestamp'] <= timestamp + window)
            candidates = glucose_df[mask]
            if len(candidates) == 0:
                continue

            time_diffs = abs(candidates['timestamp'] - timestamp)
            closest_idx = time_diffs.idxmin()
            glucose_mgdl = candidates.loc[closest_idx, 'glucose_mgdl']

        # Extract features
        features = extract_features_fast(audio_path, window_ms=window_ms)
        if features is None:
            continue

        features['glucose_mgdl'] = glucose_mgdl
        features['participant'] = name
        samples.append(features)

    if not samples:
        return None

    return pd.DataFrame(samples)


def clarke_zone(ref, pred):
    """Classify into Clarke zone."""
    if ref <= 70 and pred >= 180:
        return 'E'
    if ref >= 180 and pred <= 70:
        return 'E'
    if ref < 70 and pred < 70:
        return 'A'
    if ref >= 70 and abs(pred - ref) / ref <= 0.20:
        return 'A'
    if ref <= 70 and 70 < pred < 180:
        return 'D'
    if ref >= 240 and 70 <= pred <= 180:
        return 'D'
    if 70 <= ref <= 180 and pred > ref + 110:
        return 'C'
    return 'B'


def plot_clarke_grid(y_ref, y_pred, title, save_path):
    """Plot Clarke Error Grid."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)

    # Diagonal
    ax.plot([0, 400], [0, 400], 'k--', linewidth=1, alpha=0.5)

    # Zone boundaries
    ax.plot([70, 400/1.2], [84, 400], 'k-', linewidth=1.5)
    ax.plot([70, 400], [56, 320], 'k-', linewidth=1.5)
    ax.plot([70, 70], [0, 56], 'k-', linewidth=1.5)
    ax.plot([70, 70], [84, 180], 'k-', linewidth=1.5)
    ax.plot([180, 400], [70, 70], 'k-', linewidth=1.5)
    ax.plot([0, 70], [180, 180], 'k-', linewidth=1.5)
    ax.plot([240, 240], [70, 180], 'k-', linewidth=1.5)
    ax.plot([240, 400], [180, 180], 'k-', linewidth=1.5)
    ax.plot([70, 70], [180, 400], 'k-', linewidth=1.5)
    ax.plot([130, 180], [0, 70], 'k-', linewidth=1.5)

    # Count zones
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for ref, pred in zip(y_ref, y_pred):
        zones[clarke_zone(ref, pred)] += 1

    n = len(y_ref)
    pct = {k: v/n*100 for k, v in zones.items()}

    # Plot points
    ax.scatter(y_ref, y_pred, c='steelblue', alpha=0.6, s=40, edgecolors='white')

    # Zone labels
    ax.text(30, 15, 'A', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(300, 330, 'B', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(120, 330, 'C', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(30, 130, 'D', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(320, 130, 'D', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(30, 330, 'E', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(320, 30, 'E', fontsize=18, fontweight='bold', alpha=0.6)

    # Stats
    stats_text = f"Clarke Zone\n\nA = {pct['A']:.1f}%\nB = {pct['B']:.1f}%\nA+B = {pct['A']+pct['B']:.1f}%\nC = {pct['C']:.1f}%\nD = {pct['D']:.1f}%\nE = {pct['E']:.1f}%"
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Reference BG (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted BG (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return zones, pct


def main():
    print("="*70)
    print("VOICE-GLUCOSE ANALYSIS - FAST WINDOWED VERSION")
    print("="*70)

    models = {
        'ridge': Ridge(alpha=1.0),
        'bayesian_ridge': BayesianRidge(),
        'rf': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=42),
        'svr': SVR(kernel='rbf', C=1.0),
        'knn': KNeighborsRegressor(n_neighbors=5),
    }

    # Test window sizes
    window_results = []

    for window_ms in WINDOW_SIZES_MS:
        print(f"\n{'='*60}")
        print(f"WINDOW SIZE: {window_ms}ms")
        print(f"{'='*60}")

        datasets = {}
        for name, config in PARTICIPANTS.items():
            print(f"  Loading {name}...", end=" ")
            df = load_participant_data(name, config, window_ms=window_ms)
            if df is not None and len(df) >= 10:
                datasets[name] = df
                print(f"{len(df)} samples")
            else:
                print("skipped")

        if not datasets:
            continue

        # Evaluate models
        all_maes = []
        for name, df in datasets.items():
            feature_cols = [c for c in df.columns if c.endswith('_wmean') or c.endswith('_wstd')]
            X = df[feature_cols].values
            y = df['glucose_mgdl'].values

            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]

            if len(X) < 10:
                continue

            # Feature selection
            correlations = [(i, abs(np.corrcoef(X[:, i], y)[0, 1])) for i in range(X.shape[1]) if np.std(X[:, i]) > 0]
            correlations = [(i, c) for i, c in correlations if not np.isnan(c)]
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_idx = [c[0] for c in correlations[:20]]
            X_sel = X[:, top_idx] if top_idx else X

            best_mae = float('inf')
            for model_name, model in models.items():
                try:
                    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                    y_pred = cross_val_predict(pipe, X_sel, y, cv=LeaveOneOut())
                    mae = np.mean(np.abs(y - y_pred))
                    if mae < best_mae:
                        best_mae = mae
                except:
                    continue

            if best_mae < float('inf'):
                all_maes.append(best_mae)

        if all_maes:
            avg_mae = np.mean(all_maes)
            window_results.append({'window_ms': window_ms, 'avg_mae': avg_mae, 'datasets': datasets})
            print(f"\n  Average MAE: {avg_mae:.2f} mg/dL")

    # Find optimal
    if not window_results:
        print("No valid results!")
        return

    best = min(window_results, key=lambda x: x['avg_mae'])
    optimal_window = best['window_ms']
    datasets = best['datasets']

    print(f"\n{'='*70}")
    print(f"OPTIMAL WINDOW: {optimal_window}ms (MAE={best['avg_mae']:.2f})")
    print(f"{'='*70}")

    # Run final analysis
    print("\n" + "="*70)
    print("FINAL ANALYSIS WITH OPTIMAL WINDOW")
    print("="*70)

    personalized = {}
    all_preds = {}

    for name, df in datasets.items():
        print(f"\n--- {name} ({len(df)} samples) ---")

        feature_cols = [c for c in df.columns if c.endswith('_wmean') or c.endswith('_wstd')]
        X = df[feature_cols].values
        y = df['glucose_mgdl'].values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 10:
            continue

        # Feature selection
        correlations = [(i, abs(np.corrcoef(X[:, i], y)[0, 1])) for i in range(X.shape[1]) if np.std(X[:, i]) > 0]
        correlations = [(i, c) for i, c in correlations if not np.isnan(c)]
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_idx = [c[0] for c in correlations[:20]]
        X_sel = X[:, top_idx] if top_idx else X

        best_result = None
        for model_name, model in models.items():
            try:
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                y_pred = cross_val_predict(pipe, X_sel, y, cv=LeaveOneOut())
                mae = np.mean(np.abs(y - y_pred))
                rmse = np.sqrt(np.mean((y - y_pred)**2))
                r = np.corrcoef(y, y_pred)[0, 1]

                if best_result is None or mae < best_result['mae']:
                    best_result = {
                        'model': model_name, 'mae': mae, 'rmse': rmse,
                        'r': r if not np.isnan(r) else 0, 'y_pred': y_pred
                    }
            except:
                continue

        if best_result:
            personalized[name] = {
                'n_samples': len(y),
                'glucose_mean': np.mean(y),
                'glucose_std': np.std(y),
                'best_model': best_result['model'],
                'mae': best_result['mae'],
                'rmse': best_result['rmse'],
                'r': best_result['r']
            }
            all_preds[name] = {'y_true': y, 'y_pred': best_result['y_pred']}
            print(f"  Best: {best_result['model']}, MAE={best_result['mae']:.2f}, r={best_result['r']:.3f}")

    # Population model
    print("\n" + "="*70)
    print("POPULATION MODEL (Leave-One-Person-Out)")
    print("="*70)

    all_dfs = []
    for name, df in datasets.items():
        df_copy = df.copy()
        df_copy['participant'] = name
        all_dfs.append(df_copy)

    combined = pd.concat(all_dfs, ignore_index=True)
    feature_cols = [c for c in combined.columns if c.endswith('_wmean') or c.endswith('_wstd')]

    X_pop = combined[feature_cols].values
    y_pop = combined['glucose_mgdl'].values
    groups = combined['participant'].values

    mask = ~(np.isnan(X_pop).any(axis=1) | np.isnan(y_pop))
    X_pop, y_pop, groups = X_pop[mask], y_pop[mask], groups[mask]

    # Feature selection
    correlations = [(i, abs(np.corrcoef(X_pop[:, i], y_pop)[0, 1])) for i in range(X_pop.shape[1]) if np.std(X_pop[:, i]) > 0]
    correlations = [(i, c) for i, c in correlations if not np.isnan(c)]
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_idx = [c[0] for c in correlations[:20]]
    X_pop_sel = X_pop[:, top_idx]

    logo = LeaveOneGroupOut()
    pop_result = None

    for model_name in ['ridge', 'bayesian_ridge', 'rf']:
        try:
            pipe = Pipeline([('scaler', StandardScaler()), ('model', models[model_name])])
            y_pred_pop = cross_val_predict(pipe, X_pop_sel, y_pop, cv=logo, groups=groups)
            mae = np.mean(np.abs(y_pop - y_pred_pop))

            if pop_result is None or mae < pop_result['mae']:
                pop_result = {'model': model_name, 'mae': mae, 'y_pred': y_pred_pop, 'y_true': y_pop}
        except Exception as e:
            print(f"  {model_name}: {e}")

    if pop_result:
        print(f"\nPopulation Model: {pop_result['model']}, MAE={pop_result['mae']:.2f}")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)

    # Clarke Grid - Combined
    all_y_true = np.concatenate([p['y_true'] for p in all_preds.values()])
    all_y_pred = np.concatenate([p['y_pred'] for p in all_preds.values()])

    zones, pct = plot_clarke_grid(all_y_true, all_y_pred,
                                   f"Clarke Error Grid (Window={optimal_window}ms)",
                                   FIGURES_DIR / "clarke_error_grid.png")
    print(f"  Clarke A+B: {pct['A']+pct['B']:.1f}%")

    # Individual Clarke grids
    for name, preds in all_preds.items():
        plot_clarke_grid(preds['y_true'], preds['y_pred'],
                        f"Clarke Error Grid - {name}",
                        FIGURES_DIR / f"clarke_grid_{name}.png")

    # Window optimization plot
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = [r['window_ms'] for r in window_results]
    maes = [r['avg_mae'] for r in window_results]
    ax.plot(windows, maes, 'bo-', linewidth=2, markersize=10)
    ax.scatter([optimal_window], [best['avg_mae']], c='red', s=200, zorder=5, label=f'Optimal: {optimal_window}ms')
    ax.set_xlabel('Window Size (ms)')
    ax.set_ylabel('Average MAE (mg/dL)')
    ax.set_title('Window Size Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / "window_optimization.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    participants = list(personalized.keys())
    maes_list = [personalized[p]['mae'] for p in participants]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(participants)))

    axes[0, 0].bar(participants, maes_list, color=colors)
    axes[0, 0].axhline(np.mean(maes_list), color='red', linestyle='--', label=f'Mean: {np.mean(maes_list):.1f}')
    axes[0, 0].set_ylabel('MAE (mg/dL)')
    axes[0, 0].set_title('Personalized Model MAE')
    axes[0, 0].set_xticklabels(participants, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis='y', alpha=0.3)

    samples = [personalized[p]['n_samples'] for p in participants]
    axes[0, 1].bar(participants, samples, color=colors)
    axes[0, 1].set_ylabel('Samples')
    axes[0, 1].set_title('Dataset Size')
    axes[0, 1].set_xticklabels(participants, rotation=45, ha='right')
    axes[0, 1].grid(True, axis='y', alpha=0.3)

    rs = [personalized[p]['r'] for p in participants]
    axes[1, 0].bar(participants, rs, color=colors)
    axes[1, 0].set_ylabel('Pearson r')
    axes[1, 0].set_title('Prediction Correlation')
    axes[1, 0].set_xticklabels(participants, rotation=45, ha='right')
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    axes[1, 1].bar(['Personalized', 'Population'], [np.mean(maes_list), pop_result['mae']], color=['#3498db', '#e74c3c'])
    axes[1, 1].set_ylabel('MAE (mg/dL)')
    axes[1, 1].set_title('Personalized vs Population')
    axes[1, 1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Generate HTML report
    print("\nGenerating HTML report...")

    total_samples = sum(p['n_samples'] for p in personalized.values())
    avg_mae = np.mean([p['mae'] for p in personalized.values()])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Voice-Glucose Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; border-radius: 10px; }}
        h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        .metric {{ display: inline-block; background: #f0f4ff; padding: 15px 25px; border-radius: 8px; margin: 10px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure img {{ max-width: 100%; border-radius: 8px; }}
    </style>
</head>
<body>

<div class="header">
    <h1>Voice-Based Glucose Estimation</h1>
    <p>Technical Report with Windowed Analysis</p>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>

<div class="section">
    <h2>Summary</h2>
    <div class="metric"><div class="metric-value">{len(personalized)}</div><div>Participants</div></div>
    <div class="metric"><div class="metric-value">{total_samples}</div><div>Samples</div></div>
    <div class="metric"><div class="metric-value">{optimal_window}</div><div>Window (ms)</div></div>
    <div class="metric"><div class="metric-value">{avg_mae:.1f}</div><div>Avg MAE</div></div>
    <div class="metric"><div class="metric-value">{pct['A']+pct['B']:.1f}%</div><div>Clarke A+B</div></div>
</div>

<div class="section">
    <h2>Window Optimization</h2>
    <p>Tested windows: {', '.join(str(w) for w in WINDOW_SIZES_MS)} ms</p>
    <p><strong>Optimal: {optimal_window}ms</strong> with average MAE of {best['avg_mae']:.2f} mg/dL</p>
    <div class="figure"><img src="figures/window_optimization.png"></div>
</div>

<div class="section">
    <h2>Personalized Models (LOO-CV)</h2>
    <table>
        <tr><th>Participant</th><th>Samples</th><th>Best Model</th><th>MAE</th><th>RMSE</th><th>r</th></tr>
"""

    for name, r in personalized.items():
        html += f"<tr><td>{name}</td><td>{r['n_samples']}</td><td>{r['best_model']}</td><td>{r['mae']:.2f}</td><td>{r['rmse']:.2f}</td><td>{r['r']:.3f}</td></tr>"

    html += f"""
        <tr style="background:#e8f4ea;font-weight:bold;"><td>AVERAGE</td><td>{total_samples}</td><td>-</td><td>{avg_mae:.2f}</td><td>{np.mean([p['rmse'] for p in personalized.values()]):.2f}</td><td>{np.mean([p['r'] for p in personalized.values()]):.3f}</td></tr>
    </table>
    <div class="figure"><img src="figures/model_comparison.png"></div>
</div>

<div class="section">
    <h2>Population Model (LOPO)</h2>
    <p>Model: {pop_result['model']}, MAE: {pop_result['mae']:.2f} mg/dL</p>
</div>

<div class="section">
    <h2>Clarke Error Grid Analysis</h2>
    <table>
        <tr><th>Zone</th><th>Description</th><th>Count</th><th>%</th></tr>
        <tr style="background:#d4edda;"><td>A</td><td>Clinically accurate</td><td>{zones['A']}</td><td>{pct['A']:.1f}%</td></tr>
        <tr style="background:#fff3cd;"><td>B</td><td>Benign errors</td><td>{zones['B']}</td><td>{pct['B']:.1f}%</td></tr>
        <tr><td>C</td><td>Overcorrection</td><td>{zones['C']}</td><td>{pct['C']:.1f}%</td></tr>
        <tr style="background:#f8d7da;"><td>D</td><td>Failure to detect</td><td>{zones['D']}</td><td>{pct['D']:.1f}%</td></tr>
        <tr style="background:#f5c6cb;"><td>E</td><td>Dangerous</td><td>{zones['E']}</td><td>{pct['E']:.1f}%</td></tr>
        <tr style="font-weight:bold;"><td colspan="2">A+B (Acceptable)</td><td>{zones['A']+zones['B']}</td><td>{pct['A']+pct['B']:.1f}%</td></tr>
    </table>
    <div class="figure"><img src="figures/clarke_error_grid.png"></div>
</div>

</body>
</html>"""

    with open(OUTPUT_DIR / "report.html", 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Report: {OUTPUT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
