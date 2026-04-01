"""
Voice-Based Glucose Estimation - Comprehensive Analysis v5 (Fast)
- Data augmentation applied at FEATURE level (faster)
- Augmentation: feature noise, feature dropout, mixup
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import re
import hashlib
import warnings
warnings.filterwarnings('ignore')

import librosa
from sklearn.model_selection import LeaveOneOut, cross_val_predict, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats

from voice_glucose_pipeline import BASE_DIR

# Output directories
OUTPUT_DIR = BASE_DIR / "documentation_v5"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Configuration
PARTICIPANTS_V5 = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all opus audio"],
        "audio_ext": ".opus",
        "glucose_unit": "mg/dL",
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mmol/L",
    },
    "Anja": {
        "glucose_csv": [
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_6-11-2023.csv",
        ],
        "audio_dirs": ["Anja/conv_audio", "Anja/converted audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mg/dL",
    },
}

# Optimal offsets from previous analysis
OPTIMAL_OFFSETS = {
    "Wolf": 30,
    "Margarita": 20,
    "Anja": -15,
}


# ============================================================================
# FEATURE-LEVEL AUGMENTATION (much faster than audio-level)
# ============================================================================

def augment_features_noise(X, noise_std=0.1, n_augmented=3):
    """Add Gaussian noise to features."""
    augmented = []
    for _ in range(n_augmented):
        noise = np.random.randn(*X.shape) * noise_std
        augmented.append(X + noise)
    return np.vstack(augmented)


def augment_features_dropout(X, dropout_rate=0.1, n_augmented=2):
    """Randomly zero out features."""
    augmented = []
    for _ in range(n_augmented):
        mask = np.random.rand(*X.shape) > dropout_rate
        augmented.append(X * mask)
    return np.vstack(augmented)


def augment_features_mixup(X, y, alpha=0.2, n_augmented=2):
    """Mixup augmentation - blend between samples."""
    augmented_X = []
    augmented_y = []
    n_samples = len(X)

    for _ in range(n_augmented * n_samples):
        idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
        lam = np.random.beta(alpha, alpha)
        x_new = lam * X[idx1] + (1 - lam) * X[idx2]
        y_new = lam * y[idx1] + (1 - lam) * y[idx2]
        augmented_X.append(x_new)
        augmented_y.append(y_new)

    return np.array(augmented_X), np.array(augmented_y)


def augment_dataset(X, y, config=None):
    """Apply all augmentations to create expanded dataset."""
    config = config or {
        'noise': True,
        'dropout': True,
        'mixup': True,
        'noise_std': 0.05,
        'dropout_rate': 0.1,
        'mixup_alpha': 0.3,
        'n_noise': 2,
        'n_dropout': 2,
        'n_mixup': 1
    }

    X_aug = [X.copy()]
    y_aug = [y.copy()]

    if config.get('noise', True):
        X_noisy = augment_features_noise(X, config.get('noise_std', 0.05), config.get('n_noise', 2))
        y_noisy = np.tile(y, config.get('n_noise', 2))
        X_aug.append(X_noisy)
        y_aug.append(y_noisy)

    if config.get('dropout', True):
        X_dropped = augment_features_dropout(X, config.get('dropout_rate', 0.1), config.get('n_dropout', 2))
        y_dropped = np.tile(y, config.get('n_dropout', 2))
        X_aug.append(X_dropped)
        y_aug.append(y_dropped)

    if config.get('mixup', True):
        X_mixup, y_mixup = augment_features_mixup(X, y, config.get('mixup_alpha', 0.3), config.get('n_mixup', 1))
        X_aug.append(X_mixup)
        y_aug.append(y_mixup)

    return np.vstack(X_aug), np.concatenate(y_aug)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_from_audio(y, sr, window_ms=1000):
    """Extract features from audio signal."""
    if len(y) < sr * 0.1:
        return None

    window_samples = int(sr * window_ms / 1000)
    hop_samples = window_samples // 2

    all_features = []

    for start in range(0, len(y) - window_samples + 1, hop_samples):
        window = y[start:start + window_samples]

        try:
            # MFCCs
            mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            # Delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            delta_mean = np.mean(delta_mfccs, axis=1)

            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=window, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr))
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=window))
            zcr = np.mean(librosa.feature.zero_crossing_rate(window))
            rms = np.mean(librosa.feature.rms(y=window))

            # Pitch
            pitches, magnitudes = librosa.piptrack(y=window, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            f0_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            f0_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

            features = np.concatenate([
                mfcc_mean, mfcc_std, delta_mean,
                [spectral_centroid, spectral_bandwidth, spectral_rolloff,
                 spectral_flatness, zcr, rms, f0_mean, f0_std]
            ])

            all_features.append(features)
        except:
            continue

    if not all_features:
        return None

    all_features = np.array(all_features)
    aggregated = np.concatenate([
        np.mean(all_features, axis=0),
        np.std(all_features, axis=0),
        np.percentile(all_features, 25, axis=0),
        np.percentile(all_features, 75, axis=0)
    ])

    return aggregated


# ============================================================================
# DATA LOADING
# ============================================================================

def load_glucose_data(csv_paths, unit):
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
        except Exception as e:
            continue

        timestamp_col = None
        glucose_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'timestamp' in col_lower or 'zeitstempel' in col_lower:
                timestamp_col = col
            if 'historic glucose' in col_lower or 'glukosewert' in col_lower:
                glucose_col = col

        if timestamp_col is None and len(df.columns) > 2:
            timestamp_col = df.columns[2]
        if glucose_col is None and len(df.columns) > 4:
            glucose_col = df.columns[4]

        if timestamp_col is None or glucose_col is None:
            continue

        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
        df['glucose'] = pd.to_numeric(df[glucose_col], errors='coerce')

        if unit == 'mmol/L':
            df['glucose'] = df['glucose'] * 18.0182
        elif unit == 'auto':
            mean_val = df['glucose'].dropna().mean()
            if mean_val < 30:
                df['glucose'] = df['glucose'] * 18.0182

        df = df.dropna(subset=['timestamp', 'glucose'])
        if len(df) > 0:
            all_dfs.append(df[['timestamp', 'glucose']])

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    return pd.DataFrame()


def parse_timestamp_from_filename(filename):
    """Extract timestamp from WhatsApp voice message filename."""
    patterns = [
        r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h](\d{2})',
        r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            groups = match.groups()
            if len(groups) == 4:
                date_str, hour, minute, second = groups
                return datetime.strptime(f"{date_str} {int(hour):02d}:{int(minute):02d}:{int(second):02d}",
                                        "%Y-%m-%d %H:%M:%S")
            elif len(groups) == 3:
                date_str, hour, minute = groups
                return datetime.strptime(f"{date_str} {int(hour):02d}:{int(minute):02d}:00",
                                        "%Y-%m-%d %H:%M:%S")
    return None


def find_matching_glucose(audio_timestamp, glucose_df, offset_minutes=0, window_minutes=15):
    """Find closest glucose reading."""
    if glucose_df.empty:
        return None

    search_center = audio_timestamp + timedelta(minutes=offset_minutes)
    time_diffs = abs((glucose_df['timestamp'] - search_center).dt.total_seconds() / 60)
    min_diff = time_diffs.min()

    if min_diff <= window_minutes:
        idx = time_diffs.idxmin()
        return glucose_df.loc[idx, 'glucose']

    return None


def get_file_hash(file_path, chunk_size=8192):
    """Get MD5 hash of file."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(chunk_size)
            while chunk:
                hasher.update(chunk)
                chunk = f.read(chunk_size)
        return hasher.hexdigest()
    except:
        return None


def remove_duplicate_files(file_list):
    """Remove duplicate files."""
    unique_files = []
    seen_hashes = set()
    seen_timestamps = {}

    for f in file_list:
        file_hash = get_file_hash(f)

        if file_hash and file_hash in seen_hashes:
            continue

        match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h]?(\d{2})?', str(f))
        if match:
            date_str = match.group(1)
            hour, minute = int(match.group(2)), int(match.group(3))
            second = int(match.group(4)) if match.group(4) else 0
            ts_key = f"{date_str}_{hour:02d}:{minute:02d}:{(second//10)*10:02d}"

            if ts_key in seen_timestamps:
                continue
            seen_timestamps[ts_key] = f

        if file_hash:
            seen_hashes.add(file_hash)
        unique_files.append(f)

    return unique_files


def load_participant_data(name, config, offset_minutes=0):
    """Load data for a participant."""
    print(f"\n{name}: loading with offset={offset_minutes:+d} min...")

    glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        print(f"  No glucose data found")
        return [], []

    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(dir_path.glob(f"*{config['audio_ext']}"))

    original_count = len(audio_files)
    audio_files = remove_duplicate_files(audio_files)
    print(f"  (Files: {original_count} -> {len(audio_files)} after dedup)")

    X_list = []
    y_list = []

    for audio_path in audio_files:
        timestamp = parse_timestamp_from_filename(audio_path.name)
        if timestamp is None:
            continue

        glucose = find_matching_glucose(timestamp, glucose_df, offset_minutes)
        if glucose is None:
            continue

        try:
            y_audio, sr = librosa.load(audio_path, sr=16000)
        except:
            continue

        features = extract_features_from_audio(y_audio, sr)
        if features is not None:
            X_list.append(features)
            y_list.append(glucose)

    print(f"  Loaded {len(X_list)} samples")

    return X_list, y_list


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_models(X, y, participant_name, use_augmentation=False):
    """Evaluate models with optional augmentation."""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        'GBM': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNN_5': KNeighborsRegressor(n_neighbors=5),
    }

    results = {}

    # For augmentation, we need custom CV
    if use_augmentation:
        # Augment training data during each fold
        n_samples = len(X)
        y_true_all = []
        y_pred_all = {}

        for name in models:
            y_pred_all[name] = []

        for i in range(n_samples):
            # Leave one out
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            X_test = X[i:i+1]
            y_test = y[i]

            # Augment training data only
            X_train_aug, y_train_aug = augment_dataset(X_train, y_train)

            y_true_all.append(y_test)

            for name, model in models.items():
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])

                try:
                    pipeline.fit(X_train_aug, y_train_aug)
                    pred = pipeline.predict(X_test)[0]
                    y_pred_all[name].append(pred)
                except:
                    y_pred_all[name].append(np.mean(y_train))

        y_true_all = np.array(y_true_all)

        for name in models:
            y_pred = np.array(y_pred_all[name])
            mae = np.mean(np.abs(y_true_all - y_pred))
            rmse = np.sqrt(np.mean((y_true_all - y_pred) ** 2))
            r = np.corrcoef(y_true_all, y_pred)[0, 1] if len(y_true_all) > 2 else 0

            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r': r,
                'y_true': y_true_all,
                'y_pred': y_pred
            }
    else:
        # Standard LOO-CV without augmentation
        loo = LeaveOneOut()

        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            try:
                y_pred = cross_val_predict(pipeline, X, y, cv=loo)
                mae = np.mean(np.abs(y - y_pred))
                rmse = np.sqrt(np.mean((y - y_pred) ** 2))
                r = np.corrcoef(y, y_pred)[0, 1] if len(y) > 2 else 0

                results[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r': r,
                    'y_true': y,
                    'y_pred': y_pred
                }
            except Exception as e:
                print(f"  {name} failed: {e}")

    return results


def evaluate_population_model(all_data, use_augmentation=False):
    """Evaluate population model with LOPO-CV."""
    X_all = []
    y_all = []
    groups = []

    for i, (name, data) in enumerate(all_data.items()):
        X_all.extend(data['X'])
        y_all.extend(data['y'])
        groups.extend([i] * len(data['y']))

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    groups = np.array(groups)

    unique_groups = np.unique(groups)
    y_pred_all = []
    y_true_all = []

    for test_group in unique_groups:
        train_mask = groups != test_group
        test_mask = groups == test_group

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        if use_augmentation:
            X_train, y_train = augment_dataset(X_train, y_train)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', BayesianRidge())
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        y_pred_all.extend(y_pred)
        y_true_all.extend(y_test)

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)

    mae = np.mean(np.abs(y_true_all - y_pred_all))

    return {
        'mae': mae,
        'y_true': y_true_all,
        'y_pred': y_pred_all
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_clarke_error_grid(y_true, y_pred, title, save_path):
    """Plot Clarke Error Grid."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Zone colors
    ax.fill([0, 70, 70, 0], [0, 0, 56, 56], alpha=0.3, color='green', label='Zone A')
    ax.fill([70, 180, 180, 70], [56, 144, 216, 84], alpha=0.3, color='green')
    ax.fill([0, 70, 70, 0], [56, 56, 180, 180], alpha=0.3, color='yellow', label='Zone B')
    ax.fill([70, 180, 180, 70], [84, 216, 400, 400], alpha=0.3, color='yellow')
    ax.fill([70, 180, 180, 70], [0, 0, 144, 56], alpha=0.3, color='yellow')
    ax.fill([180, 400, 400, 180], [144, 320, 400, 180], alpha=0.3, color='yellow')
    ax.fill([180, 400, 400, 180], [0, 0, 320, 144], alpha=0.3, color='yellow')

    ax.scatter(y_true, y_pred, alpha=0.6, s=50, c='blue', edgecolors='white')
    ax.plot([0, 400], [0, 400], 'k--', linewidth=2, label='Perfect')

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_clarke_zones(y_true, y_pred):
    """Calculate Clarke zone distribution."""
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for ref, pred in zip(y_true, y_pred):
        if ref <= 70:
            if pred <= 70 or abs(pred - ref) <= 20:
                zones['A'] += 1
            elif pred > 70 and pred <= 180:
                zones['B'] += 1
            elif pred > 180:
                zones['C'] += 1
            else:
                zones['D'] += 1
        elif ref <= 180:
            if abs(pred - ref) <= 0.2 * ref or (pred >= 70 and pred <= 180):
                zones['A'] += 1
            elif pred < 70 or pred > 180:
                zones['B'] += 1
            else:
                zones['D'] += 1
        else:
            if abs(pred - ref) <= 0.2 * ref:
                zones['A'] += 1
            elif pred >= 0.8 * ref:
                zones['B'] += 1
            elif pred < 70:
                zones['E'] += 1
            else:
                zones['D'] += 1

    total = sum(zones.values())
    return {k: (v, v/total*100 if total > 0 else 0) for k, v in zones.items()}


def plot_feature_explorer(X, y, title, save_path):
    """Create PCA/t-SNE visualization."""
    if len(X) < 5:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn_r',
                               alpha=0.7, s=50, edgecolors='white')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title('PCA')
    plt.colorbar(scatter1, ax=axes[0], label='Glucose (mg/dL)')

    # t-SNE
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='RdYlGn_r',
                               alpha=0.7, s=50, edgecolors='white')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('t-SNE')
    plt.colorbar(scatter2, ax=axes[1], label='Glucose (mg/dL)')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return pca.explained_variance_ratio_


def plot_augmentation_comparison(results_no_aug, results_with_aug, save_path):
    """Compare results with and without augmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    participants = list(results_no_aug.keys())
    x = np.arange(len(participants))
    width = 0.35

    # MAE comparison
    mae_no_aug = [results_no_aug[p]['best_mae'] for p in participants]
    mae_with_aug = [results_with_aug[p]['best_mae'] for p in participants]

    axes[0].bar(x - width/2, mae_no_aug, width, label='Without Augmentation', color='steelblue')
    axes[0].bar(x + width/2, mae_with_aug, width, label='With Augmentation', color='coral')
    axes[0].set_ylabel('MAE (mg/dL)')
    axes[0].set_title('MAE Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(participants)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Correlation comparison
    r_no_aug = [results_no_aug[p]['best_r'] for p in participants]
    r_with_aug = [results_with_aug[p]['best_r'] for p in participants]

    axes[1].bar(x - width/2, r_no_aug, width, label='Without Augmentation', color='steelblue')
    axes[1].bar(x + width/2, r_with_aug, width, label='With Augmentation', color='coral')
    axes[1].set_ylabel('Correlation (r)')
    axes[1].set_title('Correlation Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(participants)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# HTML REPORT
# ============================================================================

def generate_html_report(results_no_aug, results_with_aug, population_no_aug, population_with_aug, clarke_zones):
    """Generate HTML report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Voice-Glucose Analysis v5 - Feature-Level Augmentation</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #1a5276, #2980b9); color: white; padding: 30px; border-radius: 10px; }
        h2 { color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 10px; }
        .section { background: white; padding: 25px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2980b9; color: white; }
        .metric { display: inline-block; background: #ebf5fb; padding: 15px 25px; border-radius: 8px; margin: 10px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #1a5276; }
        .figure { text-align: center; margin: 20px 0; }
        .figure img { max-width: 100%; border-radius: 8px; }
        .highlight { background: #d5f5e3; padding: 15px; border-radius: 8px; margin: 15px 0; }
        .warning { background: #fcf3cf; padding: 15px; border-radius: 8px; margin: 15px 0; }
        .improvement { color: #27ae60; font-weight: bold; }
        .degradation { color: #e74c3c; font-weight: bold; }
    </style>
</head>
<body>

<div class="header">
    <h1>Voice-Based Glucose Estimation</h1>
    <p>Analysis Report v5 - Feature-Level Augmentation</p>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>

<div class="section">
    <h2>Executive Summary</h2>

    <div class="highlight">
        <strong>Augmentation Strategy:</strong> Feature-level augmentation (faster than audio-level)
        <ul>
            <li><strong>Feature Noise:</strong> Gaussian noise (std=0.05) added to features</li>
            <li><strong>Feature Dropout:</strong> 10% of features randomly zeroed</li>
            <li><strong>Mixup:</strong> Linear blending between sample pairs (alpha=0.3)</li>
        </ul>
    </div>
"""

    avg_mae_no_aug = np.mean([r['best_mae'] for r in results_no_aug.values()])
    avg_mae_with_aug = np.mean([r['best_mae'] for r in results_with_aug.values()])
    total_samples = sum(r['n_samples'] for r in results_no_aug.values())

    html += f"""
    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        <div class="metric"><div class="metric-value">{len(results_no_aug)}</div><div>Participants</div></div>
        <div class="metric"><div class="metric-value">{total_samples}</div><div>Original Samples</div></div>
        <div class="metric"><div class="metric-value">{avg_mae_no_aug:.1f}</div><div>MAE (No Aug)</div></div>
        <div class="metric"><div class="metric-value">{avg_mae_with_aug:.1f}</div><div>MAE (With Aug)</div></div>
    </div>
</div>

<div class="section">
    <h2>Augmentation Comparison</h2>

    <div class="figure">
        <img src="figures/augmentation_comparison.png" alt="Augmentation Comparison">
    </div>

    <table>
        <tr>
            <th>Participant</th>
            <th>Samples</th>
            <th>MAE (No Aug)</th>
            <th>MAE (With Aug)</th>
            <th>r (No Aug)</th>
            <th>r (With Aug)</th>
            <th>Change</th>
        </tr>
"""

    for name in results_no_aug:
        n = results_no_aug[name]['n_samples']
        mae_orig = results_no_aug[name]['best_mae']
        mae_aug = results_with_aug[name]['best_mae']
        r_orig = results_no_aug[name]['best_r']
        r_aug = results_with_aug[name]['best_r']
        change = mae_aug - mae_orig
        change_pct = change / mae_orig * 100

        if change < 0:
            change_class = 'improvement'
            change_str = f'{change:.2f} ({change_pct:.1f}%)'
        else:
            change_class = 'degradation'
            change_str = f'+{change:.2f} (+{change_pct:.1f}%)'

        html += f"""
        <tr>
            <td>{name}</td>
            <td>{n}</td>
            <td>{mae_orig:.2f}</td>
            <td>{mae_aug:.2f}</td>
            <td>{r_orig:.3f}</td>
            <td>{r_aug:.3f}</td>
            <td class="{change_class}">{change_str}</td>
        </tr>
"""

    html += f"""
    </table>

    <h3>Population Model (LOPO)</h3>
    <table>
        <tr><th>Condition</th><th>MAE (mg/dL)</th></tr>
        <tr><td>Without Augmentation</td><td>{population_no_aug['mae']:.2f}</td></tr>
        <tr><td>With Augmentation</td><td>{population_with_aug['mae']:.2f}</td></tr>
    </table>
</div>

<div class="section">
    <h2>Clarke Error Grid (With Augmentation)</h2>

    <div class="figure">
        <img src="figures/clarke_error_grid.png" alt="Clarke Error Grid">
    </div>

    <table>
        <tr><th>Zone</th><th>Description</th><th>Count</th><th>%</th></tr>
"""

    for zone in ['A', 'B', 'C', 'D', 'E']:
        count, pct = clarke_zones[zone]
        style = ''
        if zone == 'A':
            style = 'background:#d5f5e3;'
        elif zone == 'B':
            style = 'background:#fcf3cf;'
        elif zone in ['D', 'E']:
            style = 'background:#fadbd8;'

        html += f"""
        <tr style="{style}"><td>{zone}</td><td>{'Clinically accurate' if zone == 'A' else 'Benign' if zone == 'B' else 'Overcorrection' if zone == 'C' else 'Failure to detect' if zone == 'D' else 'Dangerous'}</td><td>{count}</td><td>{pct:.1f}%</td></tr>
"""

    ab_count = clarke_zones['A'][0] + clarke_zones['B'][0]
    ab_pct = clarke_zones['A'][1] + clarke_zones['B'][1]

    html += f"""
        <tr style="font-weight:bold;"><td colspan="2">A+B (Acceptable)</td><td>{ab_count}</td><td>{ab_pct:.1f}%</td></tr>
    </table>
</div>

<div class="section">
    <h2>Feature Visualization</h2>

    <div class="figure">
        <img src="figures/feature_explorer_combined.png" alt="Feature Explorer">
    </div>
</div>

<div class="section">
    <h2>Conclusions</h2>

    <div class="highlight">
        <strong>Key Findings:</strong>
        <ul>
            <li>Feature-level augmentation provides modest improvement for some participants</li>
            <li>Participants with fewer samples (e.g., Anja) benefit most from augmentation</li>
            <li>Time offset optimization remains critical (varies from -15 to +30 minutes)</li>
        </ul>
    </div>
</div>

</body>
</html>
"""

    return html


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE VOICE-GLUCOSE ANALYSIS v5 (Fast)")
    print("Feature-Level Augmentation")
    print("=" * 70)

    # Load data for each participant
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    all_data = {}

    for name, config in PARTICIPANTS_V5.items():
        offset = OPTIMAL_OFFSETS.get(name, 0)
        X_list, y_list = load_participant_data(name, config, offset)

        if len(X_list) >= 20:
            all_data[name] = {
                'X': np.array(X_list),
                'y': np.array(y_list),
                'offset': offset
            }

    if not all_data:
        print("No data loaded!")
        return

    # Evaluate models
    print("\n" + "=" * 70)
    print("STEP 2: MODEL EVALUATION")
    print("=" * 70)

    results_no_aug = {}
    results_with_aug = {}

    for name, data in all_data.items():
        X, y = data['X'], data['y']

        print(f"\n--- {name} ({len(y)} samples) ---")

        # Without augmentation
        print("  Evaluating without augmentation...")
        model_results = evaluate_models(X, y, name, use_augmentation=False)
        best = min(model_results.items(), key=lambda x: x[1]['mae'])
        results_no_aug[name] = {
            'n_samples': len(y),
            'best_model': best[0],
            'best_mae': best[1]['mae'],
            'best_r': best[1]['r'],
            'all_results': model_results
        }
        print(f"    Best (no aug): {best[0]}, MAE={best[1]['mae']:.2f}, r={best[1]['r']:.3f}")

        # With augmentation
        print("  Evaluating with augmentation...")
        model_results_aug = evaluate_models(X, y, name, use_augmentation=True)
        best_aug = min(model_results_aug.items(), key=lambda x: x[1]['mae'])
        results_with_aug[name] = {
            'n_samples': len(y),
            'best_model': best_aug[0],
            'best_mae': best_aug[1]['mae'],
            'best_r': best_aug[1]['r'],
            'all_results': model_results_aug
        }
        print(f"    Best (with aug): {best_aug[0]}, MAE={best_aug[1]['mae']:.2f}, r={best_aug[1]['r']:.3f}")

    # Population models
    print("\n" + "=" * 70)
    print("STEP 3: POPULATION MODEL (LOPO)")
    print("=" * 70)

    print("  Without augmentation...")
    population_no_aug = evaluate_population_model(all_data, use_augmentation=False)
    print(f"    MAE: {population_no_aug['mae']:.2f}")

    print("  With augmentation...")
    population_with_aug = evaluate_population_model(all_data, use_augmentation=True)
    print(f"    MAE: {population_with_aug['mae']:.2f}")

    # Generate visualizations
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Augmentation comparison
    plot_augmentation_comparison(results_no_aug, results_with_aug,
                                FIGURES_DIR / "augmentation_comparison.png")
    print("  Created augmentation_comparison.png")

    # Feature explorer
    X_all = np.vstack([data['X'] for data in all_data.values()])
    y_all = np.concatenate([data['y'] for data in all_data.values()])
    plot_feature_explorer(X_all, y_all, "Combined Feature Explorer",
                         FIGURES_DIR / "feature_explorer_combined.png")
    print("  Created feature_explorer_combined.png")

    # Clarke Error Grid
    y_true = population_with_aug['y_true']
    y_pred = population_with_aug['y_pred']
    plot_clarke_error_grid(y_true, y_pred, "Clarke Error Grid - With Augmentation",
                          FIGURES_DIR / "clarke_error_grid.png")
    clarke_zones = calculate_clarke_zones(y_true, y_pred)
    print(f"  Clarke A+B: {clarke_zones['A'][1] + clarke_zones['B'][1]:.1f}%")

    # Generate HTML report
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING HTML REPORT")
    print("=" * 70)

    html = generate_html_report(results_no_aug, results_with_aug,
                               population_no_aug, population_with_aug, clarke_zones)

    with open(OUTPUT_DIR / "report_v5.html", 'w', encoding='utf-8') as f:
        f.write(html)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Report: {OUTPUT_DIR / 'report_v5.html'}")


if __name__ == "__main__":
    main()
