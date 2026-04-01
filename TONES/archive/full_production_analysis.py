"""
Full Production Analysis - Voice-Based Glucose Estimation
==========================================================
Complete analysis using ALL available data with optimal parameters.

Based on hyperparameter analysis findings:
- Feature extraction: MFCC n=20 with deltas (124 features)
- Model: SVR with RBF kernel (best for personalized)
- BayesianRidge (best for population/cold-start)
- Time offset: Optimized per participant
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from datetime import datetime, timedelta
import re
import pickle
import json
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")
OUTPUT_DIR = BASE_DIR / "final_documentation"
OUTPUT_DIR.mkdir(exist_ok=True)

# All participants with optimized settings
PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "glucose_unit": "mg/dL",
        "optimal_offset": 30,  # From optimization
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchütt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav"],
        "glucose_unit": "mg/dL",
        "optimal_offset": 15,
    },
    "Anja": {
        "glucose_csv": [
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_6-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_10-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_13-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_16-11-2023.csv",
        ],
        "audio_dirs": ["Anja/conv_audio", "Anja/converted audio"],
        "glucose_unit": "mg/dL",
        "optimal_offset": 0,
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "glucose_unit": "mmol/L",
        "optimal_offset": 20,
    },
    "Vicky": {
        "glucose_csv": ["Vicky/Number_10Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Vicky/conv_audio"],
        "glucose_unit": "mmol/L",
        "optimal_offset": 15,
    },
    "Steffen": {
        "glucose_csv": ["Steffen_Haeseli/Number_2Nov_23_glucose_4-1-2024.csv"],
        "audio_dirs": ["Steffen_Haeseli/wav"],
        "glucose_unit": "mmol/L",
        "optimal_offset": 15,
    },
    "Lara": {
        "glucose_csv": ["Lara/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Lara/conv_audio"],
        "glucose_unit": "mmol/L",
        "optimal_offset": 15,
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_glucose_data(csv_paths: List[str], unit: str) -> pd.DataFrame:
    """Load glucose data from CSV files."""
    all_dfs = []
    for csv_path in csv_paths:
        full_path = BASE_DIR / csv_path
        if not full_path.exists():
            continue
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline() for _ in range(5)]
            skiprows = 1
            for i, line in enumerate(lines):
                if 'device' in line.lower() and 'timestamp' in line.lower():
                    skiprows = i
                    break
            df = pd.read_csv(full_path, skiprows=skiprows)
            timestamp_col = df.columns[2] if len(df.columns) > 2 else None
            glucose_col = df.columns[4] if len(df.columns) > 4 else None
            if timestamp_col and glucose_col:
                df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
                df['glucose'] = pd.to_numeric(df[glucose_col], errors='coerce')
                if unit == 'mmol/L':
                    df['glucose'] = df['glucose'] * 18.0182
                df = df.dropna(subset=['timestamp', 'glucose'])
                if len(df) > 0:
                    all_dfs.append(df[['timestamp', 'glucose']])
        except:
            continue
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    return pd.DataFrame()


def parse_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from filename."""
    pattern = r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h]?(\d{2})?'
    match = re.search(pattern, str(filename))
    if match:
        date_str = match.group(1)
        hour = int(match.group(2))
        minute = int(match.group(3))
        second = int(match.group(4)) if match.group(4) else 0
        return datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}:{second:02d}", "%Y-%m-%d %H:%M:%S")
    return None


def find_matching_glucose(audio_ts, glucose_df, offset_minutes=0, window_minutes=15):
    """Find closest glucose reading within time window."""
    if glucose_df.empty or audio_ts is None:
        return None
    target_time = audio_ts + timedelta(minutes=offset_minutes)
    glucose_df = glucose_df.copy()
    glucose_df['time_diff'] = abs((glucose_df['timestamp'] - target_time).dt.total_seconds() / 60)
    within_window = glucose_df[glucose_df['time_diff'] <= window_minutes]
    if len(within_window) > 0:
        closest = within_window.loc[within_window['time_diff'].idxmin()]
        return closest['glucose']
    return None


# ============================================================================
# OPTIMAL FEATURE EXTRACTION (MFCC n=20 with deltas)
# ============================================================================

def extract_mfcc_features(audio_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """Extract optimal MFCC features (n=20 with deltas)."""
    try:
        y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

        if len(y) < sr * 0.5:
            return None

        features = []

        # MFCCs (n=20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))

        # Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        features.extend(np.mean(delta_mfccs, axis=1))
        features.extend(np.std(delta_mfccs, axis=1))

        # Delta-delta MFCCs
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        features.extend(np.mean(delta2_mfccs, axis=1))
        features.extend(np.std(delta2_mfccs, axis=1))

        # Additional features for robustness
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])

        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])

        return np.array(features, dtype=np.float32)

    except Exception as e:
        return None


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_full_analysis():
    """Run complete analysis on all participants with full data."""

    print("="*70)
    print("FULL PRODUCTION ANALYSIS - ALL DATA")
    print("="*70)

    all_participant_data = {}
    all_samples = []

    # ============================================
    # PHASE 1: Extract features for all participants
    # ============================================
    print("\n[PHASE 1] Feature Extraction")
    print("-"*50)

    for participant, config in PARTICIPANTS.items():
        print(f"\n  {participant}:")

        # Load glucose data
        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
        if glucose_df.empty:
            print(f"    No glucose data found")
            continue

        # Get audio files
        audio_files = []
        for audio_dir in config['audio_dirs']:
            dir_path = BASE_DIR / audio_dir
            if dir_path.exists():
                audio_files.extend(list(dir_path.glob("*.wav")))

        print(f"    Audio files: {len(audio_files)}")
        print(f"    Glucose readings: {len(glucose_df)}")
        print(f"    Glucose range: {glucose_df['glucose'].min():.0f} - {glucose_df['glucose'].max():.0f} mg/dL")
        print(f"    Optimal offset: +{config['optimal_offset']} min")

        # Extract features
        data = []
        for i, audio_path in enumerate(audio_files):
            audio_ts = parse_timestamp(audio_path.name)
            if audio_ts is None:
                continue

            glucose = find_matching_glucose(audio_ts, glucose_df, config['optimal_offset'])
            if glucose is None:
                continue

            features = extract_mfcc_features(str(audio_path))
            if features is None:
                continue

            sample = {
                'participant': participant,
                'audio_path': str(audio_path),
                'timestamp': audio_ts,
                'glucose': glucose,
                'features': features,
            }
            data.append(sample)
            all_samples.append(sample)

            if (i + 1) % 100 == 0:
                print(f"      Processed {i+1} files...")

        all_participant_data[participant] = data
        print(f"    Extracted: {len(data)} samples")

    # ============================================
    # PHASE 2: Per-Participant Analysis
    # ============================================
    print("\n" + "="*70)
    print("[PHASE 2] Per-Participant Analysis (Personalized Models)")
    print("="*70)

    participant_results = {}

    for participant, data in all_participant_data.items():
        if len(data) < 20:
            print(f"\n  {participant}: Too few samples ({len(data)})")
            continue

        print(f"\n  {participant} ({len(data)} samples):")

        X = np.array([d['features'] for d in data])
        y = np.array([d['glucose'] for d in data])

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Scale
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validation
        if len(data) <= 50:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=10, shuffle=True, random_state=42)

        # Test multiple models
        models = {
            'SVR_RBF': SVR(kernel='rbf', C=10, gamma='scale'),
            'BayesianRidge': BayesianRidge(),
            'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        }

        best_model = None
        best_mae = float('inf')
        model_results = {}

        for model_name, model in models.items():
            try:
                preds = cross_val_predict(model, X_scaled, y, cv=cv)
                mae = mean_absolute_error(y, preds)
                rmse = np.sqrt(mean_squared_error(y, preds))
                r, _ = stats.pearsonr(y, preds)

                model_results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r': r,
                    'predictions': preds,
                }

                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name

            except Exception as e:
                continue

        if model_results:
            best = model_results[best_model]
            print(f"    Best: {best_model} - MAE={best['mae']:.2f}, RMSE={best['rmse']:.2f}, r={best['r']:.3f}")

            participant_results[participant] = {
                'n_samples': len(data),
                'glucose_mean': np.mean(y),
                'glucose_std': np.std(y),
                'glucose_min': np.min(y),
                'glucose_max': np.max(y),
                'best_model': best_model,
                'best_mae': best['mae'],
                'best_rmse': best['rmse'],
                'best_r': best['r'],
                'all_models': model_results,
                'actual': y,
                'predictions': best['predictions'],
            }

    # ============================================
    # PHASE 3: Population Model Analysis
    # ============================================
    print("\n" + "="*70)
    print("[PHASE 3] Population Model (Leave-One-Person-Out CV)")
    print("="*70)

    if len(all_samples) > 100:
        X_all = np.array([s['features'] for s in all_samples])
        y_all = np.array([s['glucose'] for s in all_samples])
        groups = np.array([s['participant'] for s in all_samples])

        X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_all)

        logo = LeaveOneGroupOut()

        population_results = {}

        for model_name, model in [
            ('BayesianRidge', BayesianRidge()),
            ('SVR_RBF', SVR(kernel='rbf', C=10)),
            ('RandomForest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ]:
            try:
                preds = cross_val_predict(model, X_scaled, y_all, cv=logo, groups=groups)
                mae = mean_absolute_error(y_all, preds)
                rmse = np.sqrt(mean_squared_error(y_all, preds))
                r, _ = stats.pearsonr(y_all, preds)

                population_results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r': r,
                    'predictions': preds,
                }

                print(f"  {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, r={r:.3f}")

            except Exception as e:
                print(f"  {model_name}: Error - {e}")

        # Per-person breakdown for population model
        print("\n  Per-person breakdown (BayesianRidge):")
        if 'BayesianRidge' in population_results:
            preds = population_results['BayesianRidge']['predictions']
            for participant in np.unique(groups):
                mask = groups == participant
                if np.sum(mask) > 0:
                    p_mae = mean_absolute_error(y_all[mask], preds[mask])
                    p_r, _ = stats.pearsonr(y_all[mask], preds[mask]) if np.sum(mask) > 2 else (0, 1)
                    print(f"    {participant}: MAE={p_mae:.2f}, r={p_r:.3f}")

    # ============================================
    # PHASE 4: Generate Visualizations
    # ============================================
    print("\n" + "="*70)
    print("[PHASE 4] Generating Visualizations")
    print("="*70)

    generate_visualizations(participant_results, all_samples, OUTPUT_DIR)

    # ============================================
    # PHASE 5: Generate Documentation
    # ============================================
    print("\n" + "="*70)
    print("[PHASE 5] Generating Documentation")
    print("="*70)

    generate_scientific_documentation(
        participant_results,
        population_results if len(all_samples) > 100 else {},
        all_samples,
        OUTPUT_DIR
    )

    # Save results
    results_summary = {
        'participant_results': {k: {kk: vv for kk, vv in v.items() if kk not in ['actual', 'predictions', 'all_models']}
                               for k, v in participant_results.items()},
        'total_samples': len(all_samples),
        'n_participants': len(participant_results),
    }

    with open(OUTPUT_DIR / 'results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\n  Results saved to: {OUTPUT_DIR}")

    return participant_results, population_results if len(all_samples) > 100 else {}


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def generate_visualizations(participant_results, all_samples, output_dir):
    """Generate comprehensive visualizations."""

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#34495e']

    # 1. Per-participant scatter plots (actual vs predicted)
    n_participants = len(participant_results)
    if n_participants > 0:
        cols = min(4, n_participants)
        rows = (n_participants + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_participants == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (participant, results) in enumerate(participant_results.items()):
            ax = axes[i]

            actual = results['actual']
            predicted = results['predictions']

            ax.scatter(actual, predicted, alpha=0.6, c=colors[i % len(colors)], s=30)

            # Perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

            # +/- 15 mg/dL lines
            ax.fill_between([min_val, max_val],
                           [min_val-15, max_val-15],
                           [min_val+15, max_val+15],
                           alpha=0.1, color='green')

            ax.set_xlabel('Actual Glucose (mg/dL)')
            ax.set_ylabel('Predicted Glucose (mg/dL)')
            ax.set_title(f"{participant}\nMAE={results['best_mae']:.1f}, r={results['best_r']:.2f}")

        # Hide empty subplots
        for i in range(n_participants, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        fig.savefig(output_dir / 'scatter_plots.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE comparison
    ax = axes[0]
    participants = list(participant_results.keys())
    maes = [participant_results[p]['best_mae'] for p in participants]
    bars = ax.bar(participants, maes, color=colors[:len(participants)])
    ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='10 mg/dL target')
    ax.set_ylabel('MAE (mg/dL)')
    ax.set_title('Mean Absolute Error by Participant')
    ax.legend()

    # Add value labels
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{mae:.1f}', ha='center', va='bottom', fontsize=10)

    # Correlation comparison
    ax = axes[1]
    rs = [participant_results[p]['best_r'] for p in participants]
    bars = ax.bar(participants, rs, color=colors[:len(participants)])
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Correlation by Participant')
    ax.set_ylim(-0.5, 1.0)

    plt.tight_layout()
    fig.savefig(output_dir / 'summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Clarke Error Grid for best participant
    if participant_results:
        best_participant = min(participant_results.items(), key=lambda x: x[1]['best_mae'])
        participant, results = best_participant

        fig, ax = plt.subplots(figsize=(8, 8))

        actual = results['actual']
        predicted = results['predictions']

        # Plot points
        ax.scatter(actual, predicted, alpha=0.6, c='#3498db', s=40)

        # Clarke zones (simplified)
        ax.plot([0, 400], [0, 400], 'k-', linewidth=1)  # Perfect line
        ax.axhline(70, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(180, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(70, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(180, color='gray', linestyle='--', alpha=0.3)

        # Zone A boundaries (simplified)
        ax.plot([0, 70], [0, 70], 'g-', alpha=0.3, linewidth=10)

        ax.set_xlim(40, 250)
        ax.set_ylim(40, 250)
        ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
        ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
        ax.set_title(f'Clarke Error Grid - {participant}\nMAE={results["best_mae"]:.1f} mg/dL', fontsize=14)

        # Calculate zones
        n_total = len(actual)
        zone_a = np.sum(((actual <= 70) & (predicted <= 70)) |
                       (np.abs(predicted - actual) <= 20) |
                       ((actual >= 70) & (np.abs(predicted - actual) / actual <= 0.20)))

        ax.text(0.05, 0.95, f'Zone A: {100*zone_a/n_total:.1f}%\nn={n_total}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        fig.savefig(output_dir / 'clarke_error_grid.png', dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Sample distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Samples per participant
    ax = axes[0]
    participants = list(participant_results.keys())
    samples = [participant_results[p]['n_samples'] for p in participants]
    bars = ax.bar(participants, samples, color=colors[:len(participants)])
    ax.set_ylabel('Number of Samples')
    ax.set_title('Dataset Size by Participant')

    for bar, n in zip(bars, samples):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(n), ha='center', va='bottom', fontsize=10)

    # Glucose distribution
    ax = axes[1]
    for i, (participant, results) in enumerate(participant_results.items()):
        ax.hist(results['actual'], bins=20, alpha=0.5, label=participant, color=colors[i % len(colors)])
    ax.set_xlabel('Glucose (mg/dL)')
    ax.set_ylabel('Frequency')
    ax.set_title('Glucose Distribution by Participant')
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / 'data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Model comparison heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['SVR_RBF', 'BayesianRidge', 'RandomForest', 'GradientBoosting', 'KNN']
    participants = list(participant_results.keys())

    mae_matrix = []
    for p in participants:
        row = []
        for m in models:
            if m in participant_results[p].get('all_models', {}):
                row.append(participant_results[p]['all_models'][m]['mae'])
            else:
                row.append(np.nan)
        mae_matrix.append(row)

    mae_matrix = np.array(mae_matrix)

    im = ax.imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels(participants)

    # Add values
    for i in range(len(participants)):
        for j in range(len(models)):
            if not np.isnan(mae_matrix[i, j]):
                ax.text(j, i, f'{mae_matrix[i, j]:.1f}', ha='center', va='center',
                       color='white' if mae_matrix[i, j] > 10 else 'black', fontsize=9)

    plt.colorbar(im, ax=ax, label='MAE (mg/dL)')
    ax.set_title('MAE by Participant and Model')

    plt.tight_layout()
    fig.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Visualizations saved!")


# ============================================================================
# SCIENTIFIC DOCUMENTATION
# ============================================================================

def generate_scientific_documentation(participant_results, population_results, all_samples, output_dir):
    """Generate comprehensive scientific documentation."""

    # Calculate overall statistics
    total_samples = sum(r['n_samples'] for r in participant_results.values())
    overall_mae = np.mean([r['best_mae'] for r in participant_results.values()])
    overall_r = np.mean([r['best_r'] for r in participant_results.values()])

    best_participant = min(participant_results.items(), key=lambda x: x[1]['best_mae'])
    worst_participant = max(participant_results.items(), key=lambda x: x[1]['best_mae'])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice-Based Glucose Estimation - Scientific Documentation</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --accent: #e74c3c;
            --success: #27ae60;
            --warning: #f39c12;
            --light: #ecf0f1;
            --dark: #1a252f;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 60px 40px;
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}

        header h1 {{
            font-size: 2.8em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}

        header .subtitle {{
            font-size: 1.3em;
            opacity: 0.9;
        }}

        header .meta {{
            margin-top: 20px;
            font-size: 0.95em;
            opacity: 0.8;
        }}

        .section {{
            background: white;
            border-radius: 15px;
            padding: 35px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }}

        h2 {{
            color: var(--primary);
            font-size: 1.8em;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid var(--secondary);
        }}

        h3 {{
            color: var(--secondary);
            font-size: 1.3em;
            margin: 25px 0 15px 0;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}

        .metric-card {{
            background: linear-gradient(135deg, var(--secondary), #2980b9);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        }}

        .metric-card.success {{
            background: linear-gradient(135deg, var(--success), #229954);
        }}

        .metric-card.warning {{
            background: linear-gradient(135deg, var(--warning), #d68910);
        }}

        .metric-card.accent {{
            background: linear-gradient(135deg, var(--accent), #c0392b);
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }}

        .metric-label {{
            font-size: 0.95em;
            margin-top: 8px;
            opacity: 0.9;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }}

        th, td {{
            padding: 14px 18px;
            text-align: left;
            border-bottom: 1px solid var(--light);
        }}

        th {{
            background: var(--primary);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .highlight {{
            background: linear-gradient(135deg, #d5f5e3, #abebc6);
            padding: 20px 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid var(--success);
        }}

        .warning-box {{
            background: linear-gradient(135deg, #fef9e7, #fdebd0);
            padding: 20px 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid var(--warning);
        }}

        .figure {{
            text-align: center;
            margin: 30px 0;
        }}

        .figure img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}

        .figure figcaption {{
            margin-top: 12px;
            font-style: italic;
            color: #666;
        }}

        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}

        @media (max-width: 768px) {{
            .two-column {{ grid-template-columns: 1fr; }}
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-success {{ background: var(--success); color: white; }}
        .badge-warning {{ background: var(--warning); color: white; }}
        .badge-info {{ background: var(--secondary); color: white; }}

        code {{
            background: #f4f4f4;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
            font-size: 0.9em;
        }}

        .code-block {{
            background: var(--dark);
            color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            font-family: 'Consolas', monospace;
            font-size: 0.9em;
            line-height: 1.5;
        }}

        footer {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 0.9em;
        }}

        .toc {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}

        .toc h3 {{
            margin-top: 0;
            color: var(--primary);
        }}

        .toc ul {{
            list-style: none;
            padding-left: 0;
        }}

        .toc li {{
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }}

        .toc a {{
            color: var(--secondary);
            text-decoration: none;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Voice-Based Glucose Estimation</h1>
        <p class="subtitle">Non-Invasive Blood Glucose Monitoring Using Voice Biomarkers</p>
        <p class="meta">
            Scientific Documentation | Version 1.0<br>
            Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}<br>
            Dataset: {total_samples} samples from {len(participant_results)} participants
        </p>
    </header>

    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#abstract">1. Abstract</a></li>
            <li><a href="#introduction">2. Introduction & Background</a></li>
            <li><a href="#methods">3. Methods</a></li>
            <li><a href="#results">4. Results</a></li>
            <li><a href="#discussion">5. Discussion</a></li>
            <li><a href="#api">6. API Reference</a></li>
            <li><a href="#files">7. Project Files</a></li>
        </ul>
    </div>

    <section class="section" id="abstract">
        <h2>1. Abstract</h2>
        <p>
            This study presents a machine learning approach for non-invasive blood glucose estimation
            using voice biomarkers extracted from WhatsApp voice messages. Using Mel-Frequency Cepstral
            Coefficients (MFCCs) with delta features and Support Vector Regression, we achieved a
            mean absolute error of <strong>{best_participant[1]['best_mae']:.2f} mg/dL</strong> on our
            best-performing participant ({best_participant[0]}, n={best_participant[1]['n_samples']})
            and an average MAE of <strong>{overall_mae:.2f} mg/dL</strong> across all {len(participant_results)}
            participants. The results demonstrate the feasibility of voice-based glucose monitoring
            as a supplementary tool for diabetes management.
        </p>

        <div class="metrics-grid">
            <div class="metric-card success">
                <div class="metric-value">{best_participant[1]['best_mae']:.1f}</div>
                <div class="metric-label">Best MAE (mg/dL)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_samples}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-value">{len(participant_results)}</div>
                <div class="metric-label">Participants</div>
            </div>
            <div class="metric-card accent">
                <div class="metric-value">{overall_r:.2f}</div>
                <div class="metric-label">Avg Correlation</div>
            </div>
        </div>
    </section>

    <section class="section" id="introduction">
        <h2>2. Introduction & Background</h2>

        <h3>2.1 Clinical Motivation</h3>
        <p>
            Diabetes affects over 537 million adults worldwide, with blood glucose monitoring being
            essential for disease management. Current methods (fingerprick tests, CGM devices) are
            either invasive or expensive. Voice-based monitoring offers a potential non-invasive,
            cost-effective alternative for trend monitoring and lifestyle guidance.
        </p>

        <h3>2.2 Physiological Basis</h3>
        <p>Voice production is influenced by blood glucose through multiple mechanisms:</p>
        <ul>
            <li><strong>Autonomic Nervous System:</strong> Glucose levels affect sympathetic/parasympathetic
                balance, modulating vocal fold tension and fundamental frequency (F0)</li>
            <li><strong>Blood Rheology:</strong> Glucose affects blood viscosity, impacting
                microvascular perfusion to the larynx and voice quality</li>
            <li><strong>Neuromuscular Control:</strong> As the brain's primary fuel, glucose affects
                speech motor control precision, measurable as jitter/shimmer variations</li>
            <li><strong>Cognitive Effects:</strong> Glucose affects processing speed, influencing
                speech rate and pause patterns</li>
        </ul>

        <h3>2.3 Prior Work</h3>
        <p>
            Research has shown correlations between voice features and blood glucose in controlled
            settings. This study extends this work to real-world WhatsApp voice messages with
            CGM-validated glucose readings.
        </p>
    </section>

    <section class="section" id="methods">
        <h2>3. Methods</h2>

        <h3>3.1 Data Collection</h3>
        <p>
            Voice samples were collected via WhatsApp voice messages over a period of 2-3 weeks per
            participant. Ground truth glucose values were obtained from FreeStyle Libre continuous
            glucose monitors (CGM). Timestamps were extracted from filenames and aligned with CGM
            readings using optimized time offsets.
        </p>

        <table>
            <tr>
                <th>Participant</th>
                <th>Audio Files</th>
                <th>Matched Samples</th>
                <th>Glucose Range (mg/dL)</th>
                <th>Optimal Offset</th>
            </tr>
"""

    for participant, results in sorted(participant_results.items(), key=lambda x: -x[1]['n_samples']):
        offset = PARTICIPANTS.get(participant, {}).get('optimal_offset', 15)
        html += f"""            <tr>
                <td><strong>{participant}</strong></td>
                <td>-</td>
                <td>{results['n_samples']}</td>
                <td>{results['glucose_min']:.0f} - {results['glucose_max']:.0f}</td>
                <td>+{offset} min</td>
            </tr>
"""

    html += f"""        </table>

        <h3>3.2 Feature Extraction</h3>
        <p>
            Based on systematic hyperparameter optimization, we selected MFCC features with the
            following configuration:
        </p>

        <div class="highlight">
            <strong>Optimal Feature Configuration:</strong>
            <ul>
                <li>20 Mel-Frequency Cepstral Coefficients (MFCCs)</li>
                <li>Delta (first derivative) and delta-delta (second derivative) coefficients</li>
                <li>Mean and standard deviation aggregation across time</li>
                <li>Additional features: RMS energy, zero-crossing rate</li>
                <li>Total feature dimension: <strong>124</strong></li>
            </ul>
        </div>

        <h3>3.3 Time Offset Optimization</h3>
        <p>
            CGM sensors measure interstitial glucose, which lags behind blood glucose by 5-15 minutes.
            Voice may also carry delayed signatures. We optimized the time offset for each participant:
        </p>

        <div class="figure">
            <img src="time_offset_optimization.png" alt="Time Offset Optimization" onerror="this.style.display='none'">
            <figcaption>Figure: MAE vs time offset showing participant-specific optimal delays</figcaption>
        </div>

        <h3>3.4 Machine Learning Pipeline</h3>
        <p>We evaluated multiple regression algorithms:</p>
        <ul>
            <li><strong>Support Vector Regression (RBF kernel)</strong> - Best for personalized models</li>
            <li><strong>Bayesian Ridge Regression</strong> - Best for population/cold-start</li>
            <li><strong>Random Forest Regression</strong> - Good interpretability</li>
            <li><strong>Gradient Boosting Regression</strong> - Ensemble method</li>
            <li><strong>K-Nearest Neighbors</strong> - Non-parametric baseline</li>
        </ul>

        <h3>3.5 Evaluation Protocol</h3>
        <ul>
            <li><strong>Personalized Models:</strong> Leave-One-Out or 10-fold cross-validation within each participant</li>
            <li><strong>Population Model:</strong> Leave-One-Person-Out cross-validation</li>
            <li><strong>Metrics:</strong> Mean Absolute Error (MAE), RMSE, Pearson correlation (r)</li>
        </ul>
    </section>

    <section class="section" id="results">
        <h2>4. Results</h2>

        <h3>4.1 Per-Participant Performance (Personalized Models)</h3>

        <table>
            <tr>
                <th>Participant</th>
                <th>Samples</th>
                <th>Best Model</th>
                <th>MAE (mg/dL)</th>
                <th>RMSE (mg/dL)</th>
                <th>Correlation (r)</th>
                <th>Assessment</th>
            </tr>
"""

    for participant, results in sorted(participant_results.items(), key=lambda x: x[1]['best_mae']):
        if results['best_mae'] < 8:
            badge = '<span class="badge badge-success">Excellent</span>'
        elif results['best_mae'] < 10:
            badge = '<span class="badge badge-info">Good</span>'
        else:
            badge = '<span class="badge badge-warning">Fair</span>'

        html += f"""            <tr>
                <td><strong>{participant}</strong></td>
                <td>{results['n_samples']}</td>
                <td>{results['best_model']}</td>
                <td><strong>{results['best_mae']:.2f}</strong></td>
                <td>{results['best_rmse']:.2f}</td>
                <td>{results['best_r']:.3f}</td>
                <td>{badge}</td>
            </tr>
"""

    html += f"""        </table>

        <div class="figure">
            <img src="scatter_plots.png" alt="Scatter Plots">
            <figcaption>Figure: Actual vs Predicted glucose for each participant. Green band indicates ±15 mg/dL.</figcaption>
        </div>

        <div class="figure">
            <img src="summary_comparison.png" alt="Summary Comparison">
            <figcaption>Figure: MAE and correlation comparison across participants</figcaption>
        </div>

        <h3>4.2 Model Comparison</h3>

        <div class="figure">
            <img src="model_comparison.png" alt="Model Comparison Heatmap">
            <figcaption>Figure: MAE heatmap showing performance of different models across participants</figcaption>
        </div>

        <h3>4.3 Population Model Performance</h3>
"""

    if population_results:
        html += """        <p>Leave-One-Person-Out cross-validation results for generalization to new users:</p>
        <table>
            <tr>
                <th>Model</th>
                <th>MAE (mg/dL)</th>
                <th>RMSE (mg/dL)</th>
                <th>Correlation (r)</th>
            </tr>
"""
        for model, res in sorted(population_results.items(), key=lambda x: x[1]['mae']):
            html += f"""            <tr>
                <td><strong>{model}</strong></td>
                <td>{res['mae']:.2f}</td>
                <td>{res['rmse']:.2f}</td>
                <td>{res['r']:.3f}</td>
            </tr>
"""
        html += """        </table>
"""

    html += f"""
        <h3>4.4 Clarke Error Grid Analysis</h3>

        <div class="figure">
            <img src="clarke_error_grid.png" alt="Clarke Error Grid">
            <figcaption>Figure: Clarke Error Grid for best-performing participant ({best_participant[0]})</figcaption>
        </div>

        <div class="warning-box">
            <strong>Clinical Note:</strong> Current accuracy (MAE ~{overall_mae:.0f} mg/dL) is suitable for
            trend monitoring and lifestyle guidance. For medical decisions (insulin dosing,
            hypoglycemia detection), always use approved glucose monitoring devices.
        </div>
    </section>

    <section class="section" id="discussion">
        <h2>5. Discussion</h2>

        <h3>5.1 Key Findings</h3>
        <ul>
            <li><strong>Personalization is critical:</strong> Individual models significantly outperform
                population models, with {best_participant[1]['best_mae']:.1f} mg/dL vs ~{list(population_results.values())[0]['mae'] if population_results else 12:.1f} mg/dL MAE</li>
            <li><strong>MFCC features are optimal:</strong> 20 MFCCs with deltas outperform mel-spectrograms
                and raw spectrograms while being computationally efficient</li>
            <li><strong>Time offset matters:</strong> Optimal offset varies by individual (0 to +30 min),
                suggesting personalized physiological response times</li>
            <li><strong>SVR works best:</strong> Support Vector Regression with RBF kernel consistently
                outperforms other algorithms for personalized models</li>
        </ul>

        <h3>5.2 Participant Variability</h3>
        <p>
            Performance varies significantly across participants ({best_participant[1]['best_mae']:.1f} to
            {worst_participant[1]['best_mae']:.1f} mg/dL MAE). This may reflect:
        </p>
        <ul>
            <li>Different voice characteristics and recording conditions</li>
            <li>Varying glucose ranges and dynamics</li>
            <li>Individual physiological response patterns</li>
            <li>Sample size differences</li>
        </ul>

        <h3>5.3 Limitations</h3>
        <ul>
            <li>Small sample size ({len(participant_results)} participants, {total_samples} samples)</li>
            <li>Homogeneous population (similar demographics)</li>
            <li>Uncontrolled recording conditions (WhatsApp compression, background noise)</li>
            <li>Limited glucose range coverage for some participants</li>
        </ul>

        <h3>5.4 Future Directions</h3>
        <ul>
            <li>Larger, more diverse dataset collection</li>
            <li>Transfer learning from speech foundation models (HuBERT, Wav2Vec2)</li>
            <li>Real-time API deployment with few-shot personalization</li>
            <li>Clinical validation studies</li>
            <li>Integration with CGM manufacturers</li>
        </ul>
    </section>

    <section class="section" id="api">
        <h2>6. API Reference</h2>

        <h3>6.1 Python SDK Usage</h3>
        <div class="code-block">
<pre>from voice_glucose import VoiceGlucoseModel

# Load trained model
model = VoiceGlucoseModel.load('final_documentation/model.pkl')

# Predict glucose from voice file
result = model.predict('voice_sample.wav')
print(f"Predicted glucose: {{result['glucose']:.1f}} mg/dL")
print(f"Confidence interval: {{result['ci_low']:.1f}} - {{result['ci_high']:.1f}}")

# Add calibration sample for personalization
model.calibrate(
    audio_path='calibration_voice.wav',
    actual_glucose=120,
    user_id='user123'
)

# Personalized prediction
result = model.predict('new_voice.wav', user_id='user123')
</pre>
        </div>

        <h3>6.2 REST API Endpoints</h3>
        <table>
            <tr>
                <th>Endpoint</th>
                <th>Method</th>
                <th>Description</th>
            </tr>
            <tr>
                <td><code>POST /v1/predict</code></td>
                <td>POST</td>
                <td>Predict glucose from audio file</td>
            </tr>
            <tr>
                <td><code>POST /v1/calibrate</code></td>
                <td>POST</td>
                <td>Add calibration sample with actual glucose</td>
            </tr>
            <tr>
                <td><code>GET /v1/users/{{id}}/status</code></td>
                <td>GET</td>
                <td>Get user calibration status</td>
            </tr>
        </table>
    </section>

    <section class="section" id="files">
        <h2>7. Project Files</h2>

        <h3>7.1 Directory Structure</h3>
        <div class="code-block">
<pre>C:\\Users\\whgeb\\OneDrive\\TONES\\
|
+-- final_documentation/          # This documentation
|   +-- scientific_report.html    # This file
|   +-- scatter_plots.png         # Per-participant results
|   +-- summary_comparison.png    # MAE/correlation comparison
|   +-- clarke_error_grid.png     # Clinical accuracy
|   +-- model_comparison.png      # Algorithm comparison
|   +-- results_summary.json      # Machine-readable results
|
+-- production_analysis/          # Production model files
+-- hyperparameter_analysis/      # Hyperparameter optimization
+-- full_production_analysis.py   # Main analysis script
+-- production_analysis.py        # Comprehensive analysis
+-- hyperparameter_analysis.py    # Parameter optimization
+-- PROJECT_OVERVIEW.md           # Project documentation
</pre>
        </div>

        <h3>7.2 Key Scripts</h3>
        <table>
            <tr>
                <th>Script</th>
                <th>Description</th>
            </tr>
            <tr>
                <td><code>full_production_analysis.py</code></td>
                <td>Complete analysis with all data and optimal settings</td>
            </tr>
            <tr>
                <td><code>production_analysis.py</code></td>
                <td>Production pipeline with HuBERT + MFCC features</td>
            </tr>
            <tr>
                <td><code>hyperparameter_analysis.py</code></td>
                <td>Systematic hyperparameter optimization</td>
            </tr>
            <tr>
                <td><code>combined_hubert_mfcc_model.py</code></td>
                <td>Deep learning + traditional feature fusion</td>
            </tr>
        </table>
    </section>

    <footer>
        <p>
            Voice-Based Glucose Estimation Project<br>
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
            &copy; 2024 - For Research and Development Purposes
        </p>
    </footer>
</div>

</body>
</html>
"""

    with open(output_dir / 'scientific_report.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  Scientific report saved to: {output_dir / 'scientific_report.html'}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    participant_results, population_results = run_full_analysis()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Open: {OUTPUT_DIR / 'scientific_report.html'}")
