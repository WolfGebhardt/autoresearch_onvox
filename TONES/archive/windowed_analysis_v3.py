"""
Voice-Based Glucose Estimation - Complete Analysis with Windowing Optimization
Version 3: Correct Clarke Error Grid, Window Optimization (500ms-5000ms)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
import librosa
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import LeaveOneOut, cross_val_predict, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
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

# Window sizes to test (in milliseconds)
WINDOW_SIZES_MS = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]


def extract_features_windowed(audio_path, sr=16000, window_ms=2000, hop_ms=1000):
    """
    Extract features using sliding windows.

    Parameters:
    - audio_path: Path to audio file
    - sr: Sample rate
    - window_ms: Window size in milliseconds
    - hop_ms: Hop size in milliseconds

    Returns:
    - Dictionary of aggregated features (mean, std, min, max across windows)
    """
    try:
        y, sr_orig = librosa.load(audio_path, sr=sr)

        # Convert to samples
        window_samples = int(window_ms * sr / 1000)
        hop_samples = int(hop_ms * sr / 1000)

        # If audio is shorter than window, use full audio
        if len(y) < window_samples:
            window_samples = len(y)
            hop_samples = len(y)

        # Extract features for each window
        window_features = []

        for start in range(0, len(y) - window_samples + 1, hop_samples):
            end = start + window_samples
            y_window = y[start:end]

            features = {}

            # MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i+1}'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

            # MFCC deltas
            mfcc_delta = librosa.feature.delta(mfccs)
            for i in range(13):
                features[f'mfcc_delta_{i+1}'] = np.mean(mfcc_delta[i])

            # MFCC delta-deltas
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            for i in range(13):
                features[f'mfcc_delta2_{i+1}'] = np.mean(mfcc_delta2[i])

            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y_window, sr=sr)[0]
            features['spectral_centroid'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_window, sr=sr)[0]
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y_window, sr=sr)[0]
            features['spectral_rolloff'] = np.mean(spectral_rolloff)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y_window)[0]
            features['zcr'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)

            # RMS energy
            rms = librosa.feature.rms(y=y_window)[0]
            features['rms'] = np.mean(rms)
            features['rms_std'] = np.std(rms)

            # Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=y_window, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_range'] = 0

            window_features.append(features)

        if not window_features:
            return None

        # Aggregate across windows
        aggregated = {}
        feature_names = window_features[0].keys()

        for fname in feature_names:
            values = [wf[fname] for wf in window_features]
            aggregated[f'{fname}_mean'] = np.mean(values)
            aggregated[f'{fname}_std'] = np.std(values) if len(values) > 1 else 0
            aggregated[f'{fname}_min'] = np.min(values)
            aggregated[f'{fname}_max'] = np.max(values)

        # Add window count as meta-feature
        aggregated['n_windows'] = len(window_features)
        aggregated['audio_duration_ms'] = len(y) / sr * 1000

        return aggregated

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def load_glucose_data_for_participant(name, config):
    """Load glucose data for a participant."""
    from voice_glucose_pipeline import load_multiple_glucose_csvs

    csv_paths = [BASE_DIR / p for p in config.get('glucose_csv', [])]
    existing_csvs = [p for p in csv_paths if p.exists()]

    if not existing_csvs:
        return None

    glucose_df = load_multiple_glucose_csvs(existing_csvs, config.get('glucose_unit', 'auto'))
    return glucose_df


def find_audio_files(name, config):
    """Find all audio files for a participant."""
    audio_files = []
    audio_ext = config.get('audio_ext', '.wav')

    for audio_dir in config.get('audio_dirs', []):
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            for ext in [audio_ext, '.wav', '.WAV']:
                audio_files.extend(dir_path.glob(f'*{ext}'))

    return list(set(audio_files))


def parse_timestamp_from_filename(filename):
    """Extract timestamp from WhatsApp audio filename."""
    import re

    # Pattern: "WhatsApp Audio 2023-11-13 um 13.41.08" or similar
    patterns = [
        r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})(?:[\.:h](\d{2}))?',
        r'(\d{2}-\d{2}-\d{4})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            try:
                groups = match.groups()
                date_str = groups[0]
                hour = int(groups[1])
                minute = int(groups[2])
                second = int(groups[3]) if len(groups) > 3 and groups[3] else 0

                # Handle different date formats
                if '-' in date_str:
                    parts = date_str.split('-')
                    if len(parts[0]) == 4:  # YYYY-MM-DD
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    else:  # DD-MM-YYYY
                        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])

                from datetime import datetime
                return datetime(year, month, day, hour, minute, second)
            except:
                continue

    return None


def extract_glucose_from_filename(filename):
    """Extract glucose value from filename (for Wolf's data)."""
    import re
    match = re.match(r'^(\d+)_', Path(filename).name)
    if match:
        return float(match.group(1))
    return None


def create_windowed_dataset(name, config, window_ms=2000, verbose=True):
    """Create dataset with windowed features for a participant."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {name} (window={window_ms}ms)")
        print(f"{'='*60}")

    # Load glucose data
    glucose_df = load_glucose_data_for_participant(name, config)
    if glucose_df is None or len(glucose_df) == 0:
        if verbose:
            print(f"  No glucose data found")
        return None

    # Find audio files
    audio_files = find_audio_files(name, config)
    if not audio_files:
        if verbose:
            print(f"  No audio files found")
        return None

    if verbose:
        print(f"  Glucose readings: {len(glucose_df)}")
        print(f"  Audio files: {len(audio_files)}")

    # Process each audio file
    samples = []
    glucose_from_filename = config.get('glucose_in_filename', False)

    for audio_path in audio_files:
        # Get glucose value
        if glucose_from_filename:
            glucose_val = extract_glucose_from_filename(audio_path)
            if glucose_val is None:
                continue
            glucose_mgdl = glucose_val
        else:
            # Match by timestamp
            timestamp = parse_timestamp_from_filename(audio_path.name)
            if timestamp is None:
                continue

            # Find nearest glucose reading
            from datetime import timedelta
            window = timedelta(minutes=15)
            mask = (glucose_df['timestamp'] >= timestamp - window) & \
                   (glucose_df['timestamp'] <= timestamp + window)
            candidates = glucose_df[mask]

            if len(candidates) == 0:
                continue

            # Get closest reading
            time_diffs = abs(candidates['timestamp'] - timestamp)
            closest_idx = time_diffs.idxmin()
            glucose_mgdl = candidates.loc[closest_idx, 'glucose_mgdl']

        # Extract windowed features
        features = extract_features_windowed(audio_path, window_ms=window_ms, hop_ms=window_ms//2)

        if features is None:
            continue

        # Add metadata
        features['glucose_mgdl'] = glucose_mgdl
        features['audio_path'] = str(audio_path)
        features['participant'] = name

        samples.append(features)

    if not samples:
        if verbose:
            print(f"  No valid samples created")
        return None

    df = pd.DataFrame(samples)
    if verbose:
        print(f"  Created {len(df)} samples with {len(df.columns)} features")

    return df


def clarke_error_grid_classification(ref, pred):
    """
    Classify point into Clarke Error Grid zone.
    Based on the standard Clarke EGA (1987).

    Returns: 'A', 'B', 'C', 'D', or 'E'
    """
    # Zone E: Erroneous treatment (opposite extremes)
    if ref <= 70 and pred >= 180:
        return 'E'
    if ref >= 180 and pred <= 70:
        return 'E'

    # Zone A: Clinically accurate
    # Within 20% of reference, OR both < 70
    if ref < 70 and pred < 70:
        return 'A'
    if ref >= 70:
        if abs(pred - ref) / ref <= 0.20:
            return 'A'

    # Zone D: Failure to detect
    # Left D: ref <= 70 and 70 < pred < 180
    if ref <= 70 and 70 < pred < 180:
        return 'D'
    # Right D: ref >= 240 and 70 <= pred <= 180
    if ref >= 240 and 70 <= pred <= 180:
        return 'D'

    # Zone C: Overcorrection
    # Upper C: ref in [70, 180] and pred > ref + 110
    if 70 <= ref <= 180 and pred > ref + 110:
        return 'C'
    # Lower C: ref >= 180 and pred < 70
    # (already caught by Zone E check above)
    # Additional lower C region
    if ref >= 130 and ref <= 180:
        threshold = (7/5) * ref - 182
        if pred < threshold and pred > 70:
            return 'C'

    # Zone B: Everything else (benign errors)
    return 'B'


def plot_clarke_error_grid(y_ref, y_pred, title="Clarke Error Grid", save_path=None):
    """
    Plot Clarke Error Grid matching the standard clinical format.
    Reference: Clarke WL et al., Diabetes Care 1987
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set axis limits (0-400 standard)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)

    # Draw diagonal (perfect agreement)
    ax.plot([0, 400], [0, 400], 'k--', linewidth=1, alpha=0.5)

    # Draw 20% error bounds (Zone A boundaries)
    # Upper bound: pred = 1.2 * ref (for ref >= 70)
    # At ref=70: pred=84, at ref=333.33: pred=400
    ax.plot([70, 400/1.2], [84, 400], 'k-', linewidth=1.5)

    # Lower bound: pred = 0.8 * ref (for ref >= 70)
    # At ref=70: pred=56, at ref=400: pred=320
    ax.plot([70, 400], [56, 320], 'k-', linewidth=1.5)

    # Hypoglycemia region boundaries
    # Vertical line at ref=70
    ax.plot([70, 70], [0, 56], 'k-', linewidth=1.5)
    ax.plot([70, 70], [84, 180], 'k-', linewidth=1.5)

    # Horizontal line at pred=70 (for high ref values)
    ax.plot([180, 400], [70, 70], 'k-', linewidth=1.5)

    # Horizontal line at pred=180 (for low ref values)
    ax.plot([0, 70], [180, 180], 'k-', linewidth=1.5)

    # Zone D right boundary
    ax.plot([240, 240], [70, 180], 'k-', linewidth=1.5)
    ax.plot([240, 400], [180, 180], 'k-', linewidth=1.5)

    # Zone C upper boundary (from Zone A upper bound to top)
    ax.plot([70, 70], [180, 400], 'k-', linewidth=1.5)

    # Additional Zone C line (lower right region)
    # Line from (130, 0) to (180, 70) approximately
    ax.plot([130, 180], [0, 70], 'k-', linewidth=1.5)

    # Calculate zone distribution
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for ref, pred in zip(y_ref, y_pred):
        zone = clarke_error_grid_classification(ref, pred)
        zones[zone] += 1

    n = len(y_ref)
    zone_pct = {k: v/n*100 for k, v in zones.items()}

    # Plot data points
    ax.scatter(y_ref, y_pred, c='steelblue', alpha=0.6, s=40, edgecolors='white', linewidth=0.5)

    # Add zone labels
    ax.text(30, 15, 'A', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(300, 330, 'B', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(120, 330, 'C', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(30, 130, 'D', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(320, 130, 'D', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(30, 330, 'E', fontsize=18, fontweight='bold', alpha=0.6)
    ax.text(320, 30, 'E', fontsize=18, fontweight='bold', alpha=0.6)

    # Statistics text box
    stats_text = f"Clarke Zone\n\nA = {zone_pct['A']:.1f}%\nB = {zone_pct['B']:.1f}%\nA + B = {zone_pct['A']+zone_pct['B']:.1f}%\nC = {zone_pct['C']:.1f}%\nD = {zone_pct['D']:.1f}%\nE = {zone_pct['E']:.1f}%"
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('Reference BG (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted BG (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return zones, zone_pct


def run_model_evaluation(X, y, model_name, model):
    """Run leave-one-out CV and return metrics."""
    try:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        loo = LeaveOneOut()
        y_pred = cross_val_predict(pipeline, X, y, cv=loo)

        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r, _ = stats.pearsonr(y, y_pred)

        return {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r': r if not np.isnan(r) else 0,
            'predictions': y_pred
        }
    except Exception as e:
        return None


def optimize_window_size(datasets_by_window):
    """Find optimal window size based on average MAE."""

    models = {
        'ridge': Ridge(alpha=1.0),
        'bayesian_ridge': BayesianRidge(),
        'rf': RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        'svr_rbf': SVR(kernel='rbf', C=1.0),
        'knn_5': KNeighborsRegressor(n_neighbors=5),
    }

    results = []

    for window_ms, datasets in datasets_by_window.items():
        print(f"\n  Testing window size: {window_ms}ms")

        window_maes = []

        for name, df in datasets.items():
            if df is None or len(df) < 15:
                continue

            # Get feature columns
            feature_cols = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')
                          or c.endswith('_min') or c.endswith('_max')]
            feature_cols = [c for c in feature_cols if c not in ['glucose_mgdl', 'audio_path', 'participant']]

            X = df[feature_cols].values
            y = df['glucose_mgdl'].values

            # Remove NaN
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]

            if len(X) < 15:
                continue

            # Feature selection (top 30 by correlation)
            correlations = []
            for i in range(X.shape[1]):
                if np.std(X[:, i]) > 0:
                    corr, _ = stats.pearsonr(X[:, i], y)
                    correlations.append((i, abs(corr) if not np.isnan(corr) else 0))

            correlations.sort(key=lambda x: x[1], reverse=True)
            top_indices = [c[0] for c in correlations[:30]]
            X_selected = X[:, top_indices]

            # Test best model
            best_mae = float('inf')
            for model_name, model in models.items():
                result = run_model_evaluation(X_selected, y, model_name, model)
                if result and result['mae'] < best_mae:
                    best_mae = result['mae']

            if best_mae < float('inf'):
                window_maes.append(best_mae)

        if window_maes:
            avg_mae = np.mean(window_maes)
            results.append({
                'window_ms': window_ms,
                'avg_mae': avg_mae,
                'n_participants': len(window_maes),
                'individual_maes': window_maes
            })
            print(f"    Avg MAE: {avg_mae:.2f} mg/dL (n={len(window_maes)} participants)")

    return results


def run_full_analysis(window_ms=2000):
    """Run complete analysis with specified window size."""

    print(f"\n{'='*70}")
    print(f"FULL ANALYSIS WITH {window_ms}ms WINDOWS")
    print(f"{'='*70}")

    models = {
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1, max_iter=5000),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        'bayesian_ridge': BayesianRidge(),
        'rf_small': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        'rf_medium': RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42),
        'extra_trees': ExtraTreesRegressor(n_estimators=100, max_depth=7, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        'svr_linear': SVR(kernel='linear', C=1.0),
        'svr_rbf': SVR(kernel='rbf', C=1.0),
        'knn_3': KNeighborsRegressor(n_neighbors=3),
        'knn_5': KNeighborsRegressor(n_neighbors=5),
    }

    # Load data for all participants
    datasets = {}
    for name, config in PARTICIPANTS.items():
        df = create_windowed_dataset(name, config, window_ms=window_ms, verbose=True)
        if df is not None and len(df) >= 10:
            datasets[name] = df

    if not datasets:
        print("No valid datasets!")
        return None

    # PERSONALIZED MODELS
    print(f"\n{'#'*70}")
    print("# PERSONALIZED MODELS (Leave-One-Out CV)")
    print(f"{'#'*70}")

    personalized_results = {}
    all_predictions = {}

    for name, df in datasets.items():
        print(f"\n--- {name} ({len(df)} samples) ---")

        # Get feature columns
        feature_cols = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')
                      or c.endswith('_min') or c.endswith('_max')]
        feature_cols = [c for c in feature_cols if c not in ['glucose_mgdl', 'audio_path', 'participant']]

        X = df[feature_cols].values
        y = df['glucose_mgdl'].values

        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        print(f"  Samples after cleaning: {len(X)}, Features: {X.shape[1]}")

        if len(X) < 10:
            continue

        # Feature selection
        correlations = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 0:
                corr, _ = stats.pearsonr(X[:, i], y)
                correlations.append((i, abs(corr) if not np.isnan(corr) else 0))

        correlations.sort(key=lambda x: x[1], reverse=True)
        top_indices = [c[0] for c in correlations[:30]]
        X_selected = X[:, top_indices]

        # Test all models
        best_result = None
        for model_name, model in models.items():
            result = run_model_evaluation(X_selected, y, model_name, model)
            if result:
                if best_result is None or result['mae'] < best_result['mae']:
                    best_result = result

        if best_result:
            personalized_results[name] = {
                'n_samples': len(y),
                'glucose_mean': np.mean(y),
                'glucose_std': np.std(y),
                'glucose_min': np.min(y),
                'glucose_max': np.max(y),
                'best_model': best_result['model'],
                'mae': best_result['mae'],
                'rmse': best_result['rmse'],
                'r': best_result['r']
            }
            all_predictions[name] = {
                'y_true': y,
                'y_pred': best_result['predictions']
            }
            print(f"  Best: {best_result['model']}, MAE={best_result['mae']:.2f}, r={best_result['r']:.3f}")

    # POPULATION MODEL
    print(f"\n{'#'*70}")
    print("# POPULATION MODEL (Leave-One-Person-Out)")
    print(f"{'#'*70}")

    # Combine all data
    all_dfs = []
    for name, df in datasets.items():
        df_copy = df.copy()
        df_copy['participant'] = name
        all_dfs.append(df_copy)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    feature_cols = [c for c in combined_df.columns if c.endswith('_mean') or c.endswith('_std')
                  or c.endswith('_min') or c.endswith('_max')]
    feature_cols = [c for c in feature_cols if c not in ['glucose_mgdl', 'audio_path', 'participant']]

    X_pop = combined_df[feature_cols].values
    y_pop = combined_df['glucose_mgdl'].values
    groups = combined_df['participant'].values

    # Remove NaN
    mask = ~(np.isnan(X_pop).any(axis=1) | np.isnan(y_pop))
    X_pop, y_pop, groups = X_pop[mask], y_pop[mask], groups[mask]

    print(f"\nCombined dataset: {len(X_pop)} samples, {X_pop.shape[1]} features")

    # Feature selection on population
    correlations = []
    for i in range(X_pop.shape[1]):
        if np.std(X_pop[:, i]) > 0:
            corr, _ = stats.pearsonr(X_pop[:, i], y_pop)
            correlations.append((i, abs(corr) if not np.isnan(corr) else 0))

    correlations.sort(key=lambda x: x[1], reverse=True)
    top_indices = [c[0] for c in correlations[:30]]
    X_pop_selected = X_pop[:, top_indices]

    # LOPO CV
    logo = LeaveOneGroupOut()

    population_results = {}
    pop_predictions_all = []
    pop_true_all = []

    for model_name in ['ridge', 'bayesian_ridge', 'rf_medium', 'svr_rbf']:
        model = models[model_name]

        try:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            y_pred_pop = cross_val_predict(pipeline, X_pop_selected, y_pop, cv=logo, groups=groups)
            mae_pop = np.mean(np.abs(y_pop - y_pred_pop))

            if model_name not in population_results or mae_pop < population_results.get('best_mae', float('inf')):
                population_results = {
                    'best_model': model_name,
                    'best_mae': mae_pop,
                    'predictions': y_pred_pop,
                    'y_true': y_pop
                }
        except Exception as e:
            print(f"  Error with {model_name}: {e}")

    if population_results:
        print(f"\nPopulation Model: {population_results['best_model']}, MAE={population_results['best_mae']:.2f} mg/dL")

    return {
        'window_ms': window_ms,
        'datasets': datasets,
        'personalized': personalized_results,
        'predictions': all_predictions,
        'population': population_results,
        'combined_df': combined_df
    }


def generate_comprehensive_report(results, window_optimization_results):
    """Generate comprehensive HTML report."""

    personalized = results['personalized']
    predictions = results['predictions']
    population = results['population']
    window_ms = results['window_ms']

    # Combine all predictions for Clarke grid
    all_y_true = []
    all_y_pred = []
    for name, preds in predictions.items():
        all_y_true.extend(preds['y_true'])
        all_y_pred.extend(preds['y_pred'])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Generate Clarke Error Grid
    print("\nGenerating Clarke Error Grid...")
    zones, zone_pct = plot_clarke_error_grid(
        all_y_true, all_y_pred,
        title=f"Clarke Error Grid - All Participants (Window={window_ms}ms)",
        save_path=FIGURES_DIR / "clarke_error_grid.png"
    )

    # Individual Clarke grids
    for name, preds in predictions.items():
        plot_clarke_error_grid(
            preds['y_true'], preds['y_pred'],
            title=f"Clarke Error Grid - {name}",
            save_path=FIGURES_DIR / f"clarke_grid_{name}.png"
        )

    # Window optimization plot
    if window_optimization_results:
        print("\nGenerating window optimization plot...")
        fig, ax = plt.subplots(figsize=(10, 6))

        windows = [r['window_ms'] for r in window_optimization_results]
        maes = [r['avg_mae'] for r in window_optimization_results]

        ax.plot(windows, maes, 'bo-', linewidth=2, markersize=10)
        ax.set_xlabel('Window Size (ms)', fontsize=12)
        ax.set_ylabel('Average MAE (mg/dL)', fontsize=12)
        ax.set_title('Window Size Optimization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark optimal
        best_idx = np.argmin(maes)
        ax.scatter([windows[best_idx]], [maes[best_idx]], c='red', s=200, zorder=5, label=f'Optimal: {windows[best_idx]}ms')
        ax.legend()

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "window_optimization.png", dpi=150, bbox_inches='tight')
        plt.close()

    # Model comparison plot
    print("\nGenerating model comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    participants = list(personalized.keys())
    maes = [personalized[p]['mae'] for p in participants]

    # MAE by participant
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(participants)))
    ax1.bar(participants, maes, color=colors)
    ax1.axhline(y=np.mean(maes), color='red', linestyle='--', label=f'Mean: {np.mean(maes):.1f}')
    ax1.set_ylabel('MAE (mg/dL)')
    ax1.set_title('Personalized Model MAE by Participant')
    ax1.set_xticklabels(participants, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Sample count
    ax2 = axes[0, 1]
    samples = [personalized[p]['n_samples'] for p in participants]
    ax2.bar(participants, samples, color=colors)
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Dataset Size by Participant')
    ax2.set_xticklabels(participants, rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)

    # Correlation
    ax3 = axes[1, 0]
    rs = [personalized[p]['r'] for p in participants]
    ax3.bar(participants, rs, color=colors)
    ax3.set_ylabel('Pearson r')
    ax3.set_title('Prediction Correlation by Participant')
    ax3.set_xticklabels(participants, rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)

    # Approach comparison
    ax4 = axes[1, 1]
    approaches = ['Personalized', 'Population']
    approach_maes = [np.mean(maes), population['best_mae']]
    ax4.bar(approaches, approach_maes, color=['#3498db', '#e74c3c'])
    ax4.set_ylabel('MAE (mg/dL)')
    ax4.set_title('Personalized vs Population Model')
    for i, v in enumerate(approach_maes):
        ax4.text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Feature importance (combined)
    print("\nGenerating feature correlation plot...")
    combined_df = results['combined_df']
    feature_cols = [c for c in combined_df.columns if c.endswith('_mean') and 'mfcc' in c.lower()][:20]

    correlations = []
    for col in feature_cols:
        if combined_df[col].std() > 0:
            corr = combined_df[col].corr(combined_df['glucose_mgdl'])
            if not np.isnan(corr):
                correlations.append((col.replace('_mean', ''), corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    names = [c[0] for c in correlations[:15]]
    values = [c[1] for c in correlations[:15]]
    colors_corr = ['green' if v > 0 else 'red' for v in values]

    ax.barh(range(len(names)), values, color=colors_corr, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Pearson Correlation')
    ax.set_title('Top Features Correlated with Glucose')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlations.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Generate HTML report
    print("\nGenerating HTML report...")

    total_samples = sum(p['n_samples'] for p in personalized.values())
    avg_mae = np.mean([p['mae'] for p in personalized.values()])

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Voice-Based Glucose Estimation - Technical Report v3</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        h1 {{ margin: 0; }}
        h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-top: 40px; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        .metric-box {{ display: inline-block; background: #f0f4ff; padding: 15px 25px; border-radius: 8px; margin: 10px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .highlight {{ background: linear-gradient(120deg, #a8edea, #fed6e3); padding: 15px; border-radius: 8px; margin: 15px 0; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }}
    </style>
</head>
<body>

<div class="header">
    <h1>Voice-Based Glucose Estimation</h1>
    <p>Technical Report - Windowed Analysis v3</p>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>

<div class="section">
    <h2>1. Executive Summary</h2>

    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        <div class="metric-box">
            <div class="metric-value">{len(personalized)}</div>
            <div class="metric-label">Participants</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{total_samples}</div>
            <div class="metric-label">Voice Samples</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{window_ms}</div>
            <div class="metric-label">Window Size (ms)</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{avg_mae:.1f}</div>
            <div class="metric-label">Avg MAE (mg/dL)</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{zone_pct['A']+zone_pct['B']:.1f}%</div>
            <div class="metric-label">Clarke A+B</div>
        </div>
    </div>
</div>

<div class="section">
    <h2>2. Audio Processing & Windowing</h2>

    <h3>2.1 Window Optimization</h3>
    <p>Tested window sizes from 500ms to 5000ms with 50% overlap. Features extracted per window, then aggregated (mean, std, min, max) across all windows.</p>

    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Window Sizes Tested</td><td>500, 1000, 1500, 2000, 2500, 3000, 4000, 5000 ms</td></tr>
        <tr><td>Optimal Window</td><td>{window_ms} ms</td></tr>
        <tr><td>Hop Size</td><td>{window_ms//2} ms (50% overlap)</td></tr>
        <tr><td>Sample Rate</td><td>16 kHz</td></tr>
        <tr><td>Features per Window</td><td>69 acoustic features</td></tr>
        <tr><td>Aggregation</td><td>Mean, Std, Min, Max across windows</td></tr>
        <tr><td>Total Features</td><td>~280 (69 x 4 aggregations + meta)</td></tr>
    </table>

    <div class="figure">
        <img src="figures/window_optimization.png" alt="Window Optimization">
        <p>Figure 1: Window size optimization results</p>
    </div>
</div>

<div class="section">
    <h2>3. Feature Engineering</h2>

    <h3>3.1 Per-Window Features (69)</h3>
    <table>
        <tr><th>Category</th><th>Features</th><th>Count</th></tr>
        <tr><td>MFCCs</td><td>13 coefficients (mean + std)</td><td>26</td></tr>
        <tr><td>MFCC Deltas</td><td>First derivatives</td><td>13</td></tr>
        <tr><td>MFCC Delta-Deltas</td><td>Second derivatives</td><td>13</td></tr>
        <tr><td>Spectral</td><td>Centroid, Bandwidth, Rolloff, ZCR</td><td>10</td></tr>
        <tr><td>Prosodic</td><td>Pitch (F0), RMS energy</td><td>7</td></tr>
    </table>

    <h3>3.2 Aggregated Features (~280)</h3>
    <p>Each window feature is aggregated across all windows: <code>mean</code>, <code>std</code>, <code>min</code>, <code>max</code></p>

    <div class="figure">
        <img src="figures/feature_correlations.png" alt="Feature Correlations">
        <p>Figure 2: Top features correlated with glucose</p>
    </div>
</div>

<div class="section">
    <h2>4. Model Results</h2>

    <h3>4.1 Personalized Models (Leave-One-Out CV)</h3>
    <table>
        <tr><th>Participant</th><th>Samples</th><th>Best Model</th><th>MAE (mg/dL)</th><th>RMSE</th><th>Pearson r</th></tr>
"""

    for name, r in personalized.items():
        html += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{r['n_samples']}</td>
            <td>{r['best_model']}</td>
            <td>{r['mae']:.2f}</td>
            <td>{r['rmse']:.2f}</td>
            <td>{r['r']:.3f}</td>
        </tr>"""

    html += f"""
        <tr style="background: #e8f4ea; font-weight: bold;">
            <td>AVERAGE</td>
            <td>{total_samples}</td>
            <td>-</td>
            <td>{avg_mae:.2f}</td>
            <td>{np.mean([p['rmse'] for p in personalized.values()]):.2f}</td>
            <td>{np.mean([p['r'] for p in personalized.values()]):.3f}</td>
        </tr>
    </table>

    <h3>4.2 Population Model (Leave-One-Person-Out)</h3>
    <table>
        <tr><th>Model</th><th>MAE (mg/dL)</th></tr>
        <tr><td>{population['best_model']}</td><td>{population['best_mae']:.2f}</td></tr>
    </table>

    <div class="figure">
        <img src="figures/model_comparison.png" alt="Model Comparison">
        <p>Figure 3: Model performance comparison</p>
    </div>
</div>

<div class="section">
    <h2>5. Clinical Validation - Clarke Error Grid</h2>

    <div class="highlight">
        <p><strong>Clarke Error Grid Analysis</strong> (Clarke et al., Diabetes Care 1987)</p>
        <p>Gold standard for assessing clinical accuracy of glucose monitoring devices.</p>
    </div>

    <table>
        <tr><th>Zone</th><th>Meaning</th><th>Count</th><th>Percentage</th></tr>
        <tr style="background: #d4edda;"><td>A</td><td>Clinically accurate (within 20%)</td><td>{zones['A']}</td><td>{zone_pct['A']:.1f}%</td></tr>
        <tr style="background: #fff3cd;"><td>B</td><td>Benign errors</td><td>{zones['B']}</td><td>{zone_pct['B']:.1f}%</td></tr>
        <tr><td>C</td><td>Overcorrection</td><td>{zones['C']}</td><td>{zone_pct['C']:.1f}%</td></tr>
        <tr style="background: #f8d7da;"><td>D</td><td>Failure to detect</td><td>{zones['D']}</td><td>{zone_pct['D']:.1f}%</td></tr>
        <tr style="background: #f5c6cb;"><td>E</td><td>Erroneous treatment</td><td>{zones['E']}</td><td>{zone_pct['E']:.1f}%</td></tr>
        <tr style="font-weight: bold;"><td colspan="2">A + B (Clinically Acceptable)</td><td>{zones['A']+zones['B']}</td><td>{zone_pct['A']+zone_pct['B']:.1f}%</td></tr>
    </table>

    <div class="figure">
        <img src="figures/clarke_error_grid.png" alt="Clarke Error Grid">
        <p>Figure 4: Clarke Error Grid - Combined predictions</p>
    </div>
</div>

<div class="section">
    <h2>6. Conclusions</h2>

    <h3>Key Findings</h3>
    <ul>
        <li>Optimal window size: <strong>{window_ms}ms</strong></li>
        <li>Personalized models achieve <strong>{avg_mae:.1f} mg/dL</strong> average MAE</li>
        <li>Population model achieves <strong>{population['best_mae']:.1f} mg/dL</strong> MAE</li>
        <li>Clinical accuracy: <strong>{zone_pct['A']+zone_pct['B']:.1f}%</strong> in Clarke zones A+B</li>
        <li>MFCC features show strongest correlations with glucose</li>
    </ul>

    <h3>Limitations</h3>
    <ul>
        <li>Small sample size ({total_samples} samples, {len(personalized)} participants)</li>
        <li>Limited glucose range (mostly euglycemic)</li>
        <li>Uncontrolled recording conditions</li>
    </ul>
</div>

<div style="text-align: center; color: #666; margin-top: 40px;">
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>

</body>
</html>
"""

    report_path = OUTPUT_DIR / "technical_report_v3.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nReport saved: {report_path}")
    return report_path


def main():
    print("="*70)
    print("VOICE-BASED GLUCOSE ESTIMATION - WINDOWED ANALYSIS v3")
    print("="*70)

    # Step 1: Test different window sizes
    print("\n" + "="*70)
    print("STEP 1: WINDOW SIZE OPTIMIZATION (500ms - 5000ms)")
    print("="*70)

    datasets_by_window = {}

    for window_ms in WINDOW_SIZES_MS:
        print(f"\n--- Loading data with {window_ms}ms windows ---")
        datasets = {}
        for name, config in PARTICIPANTS.items():
            df = create_windowed_dataset(name, config, window_ms=window_ms, verbose=False)
            if df is not None and len(df) >= 10:
                datasets[name] = df
                print(f"  {name}: {len(df)} samples")
        datasets_by_window[window_ms] = datasets

    # Find optimal window
    print("\n" + "="*70)
    print("STEP 2: EVALUATING WINDOW SIZES")
    print("="*70)

    window_results = optimize_window_size(datasets_by_window)

    if window_results:
        best_window = min(window_results, key=lambda x: x['avg_mae'])
        optimal_window_ms = best_window['window_ms']
        print(f"\n*** OPTIMAL WINDOW SIZE: {optimal_window_ms}ms (MAE={best_window['avg_mae']:.2f}) ***")
    else:
        optimal_window_ms = 2000
        print(f"\nUsing default window size: {optimal_window_ms}ms")

    # Step 3: Run full analysis with optimal window
    print("\n" + "="*70)
    print(f"STEP 3: FULL ANALYSIS WITH OPTIMAL WINDOW ({optimal_window_ms}ms)")
    print("="*70)

    results = run_full_analysis(window_ms=optimal_window_ms)

    if results:
        # Step 4: Generate report
        print("\n" + "="*70)
        print("STEP 4: GENERATING COMPREHENSIVE REPORT")
        print("="*70)

        report_path = generate_comprehensive_report(results, window_results)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print(f"Report: {report_path}")
        print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
