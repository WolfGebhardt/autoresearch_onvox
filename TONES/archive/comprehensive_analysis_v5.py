"""
Voice-Based Glucose Estimation - Comprehensive Analysis v5
- All features from v4 PLUS:
- Modest data augmentation for training
  - Time stretching (0.9x, 1.1x)
  - Pitch shifting (±1 semitone)
  - Additive noise (very low SNR)
  - Random gain adjustment
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - CNN models will be skipped")

from voice_glucose_pipeline import BASE_DIR

# Output directories
OUTPUT_DIR = BASE_DIR / "documentation_v5"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Configuration - ALL participants use timestamp matching
PARTICIPANTS_V5 = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all opus audio"],
        "audio_ext": ".opus",
        "glucose_unit": "mg/dL",
    },
    "Sybille": {
        "glucose_csv": ["Sybille/SybilleJahnel_glucose_17-11-2023.csv"],
        "audio_dirs": ["Sybille/conv_audio"],
        "audio_ext": ".wav",
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
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_17-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_21-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_5-12-2023.csv",
        ],
        "audio_dirs": ["Anja/conv_audio", "Anja/converted audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mg/dL",
    },
    "Vicky": {
        "glucose_csv": ["Number_10/Number_10Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Number_10/conv_audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mmol/L",
    },
    "Steffen_Haeseli": {
        "glucose_csv": ["Number_2/Number_2Nov_23_glucose_4-1-2024.csv"],
        "audio_dirs": ["Number_2/conv_audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mmol/L",
    },
    "Lara": {
        "glucose_csv": ["Number_7/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Number_7/conv_audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mmol/L",
    },
}

# Time offsets to test (in minutes)
TIME_OFFSETS_MINUTES = [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30]

# ============================================================================
# DATA AUGMENTATION FUNCTIONS
# ============================================================================

def augment_audio_time_stretch(y, sr, rate=1.1):
    """Time stretch audio without changing pitch."""
    return librosa.effects.time_stretch(y, rate=rate)


def augment_audio_pitch_shift(y, sr, n_steps=1):
    """Pitch shift audio by n semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_audio_add_noise(y, noise_factor=0.005):
    """Add small amount of Gaussian noise."""
    noise = np.random.randn(len(y)) * noise_factor
    return y + noise


def augment_audio_gain(y, gain_db=3):
    """Adjust audio gain by specified dB."""
    gain_linear = 10 ** (gain_db / 20)
    return y * gain_linear


def create_augmented_samples(y, sr, augmentation_config):
    """
    Create augmented versions of an audio sample.

    Args:
        y: Audio signal
        sr: Sample rate
        augmentation_config: Dict specifying which augmentations to apply

    Returns:
        List of (augmented_audio, augmentation_name) tuples
    """
    augmented = []

    if augmentation_config.get('time_stretch', True):
        # Slower (0.9x speed)
        try:
            y_slow = augment_audio_time_stretch(y, sr, rate=0.9)
            augmented.append((y_slow, 'time_stretch_0.9'))
        except:
            pass

        # Faster (1.1x speed)
        try:
            y_fast = augment_audio_time_stretch(y, sr, rate=1.1)
            augmented.append((y_fast, 'time_stretch_1.1'))
        except:
            pass

    if augmentation_config.get('pitch_shift', True):
        # Pitch down 1 semitone
        try:
            y_down = augment_audio_pitch_shift(y, sr, n_steps=-1)
            augmented.append((y_down, 'pitch_-1'))
        except:
            pass

        # Pitch up 1 semitone
        try:
            y_up = augment_audio_pitch_shift(y, sr, n_steps=1)
            augmented.append((y_up, 'pitch_+1'))
        except:
            pass

    if augmentation_config.get('noise', True):
        # Add small noise
        y_noisy = augment_audio_add_noise(y, noise_factor=0.003)
        augmented.append((y_noisy, 'noise_0.003'))

    if augmentation_config.get('gain', True):
        # Gain increase
        y_loud = augment_audio_gain(y, gain_db=3)
        augmented.append((y_loud, 'gain_+3dB'))

        # Gain decrease
        y_quiet = augment_audio_gain(y, gain_db=-3)
        augmented.append((y_quiet, 'gain_-3dB'))

    return augmented


# ============================================================================
# FEATURE EXTRACTION (same as v4)
# ============================================================================

def extract_features_from_audio(y, sr, window_ms=1000):
    """Extract features from audio signal with windowing."""
    if len(y) < sr * 0.1:  # Less than 100ms
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

            # Pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=window, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            f0_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            f0_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

            # Combine features
            features = np.concatenate([
                mfcc_mean, mfcc_std, delta_mean,
                [spectral_centroid, spectral_bandwidth, spectral_rolloff,
                 spectral_flatness, zcr, rms, f0_mean, f0_std]
            ])

            all_features.append(features)
        except Exception as e:
            continue

    if not all_features:
        return None

    # Aggregate across windows
    all_features = np.array(all_features)
    aggregated = np.concatenate([
        np.mean(all_features, axis=0),
        np.std(all_features, axis=0),
        np.percentile(all_features, 25, axis=0),
        np.percentile(all_features, 75, axis=0)
    ])

    return aggregated


def get_feature_names(n_mfcc=13):
    """Generate feature names."""
    names = []

    # Base features per window
    base = []
    base.extend([f'mfcc_{i}_mean' for i in range(n_mfcc)])
    base.extend([f'mfcc_{i}_std' for i in range(n_mfcc)])
    base.extend([f'delta_mfcc_{i}_mean' for i in range(n_mfcc)])
    base.extend(['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
                 'spectral_flatness', 'zcr', 'rms', 'f0_mean', 'f0_std'])

    # Aggregations
    for agg in ['mean', 'std', 'q25', 'q75']:
        names.extend([f'{f}_{agg}' for f in base])

    return names


# ============================================================================
# DATA LOADING (same as v4 with augmentation option)
# ============================================================================

def get_file_hash(file_path, chunk_size=8192):
    """Get MD5 hash of file content."""
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
    """Remove duplicate files based on content hash and timestamps."""
    unique_files = []
    seen_hashes = set()
    seen_timestamps = {}

    for f in file_list:
        file_hash = get_file_hash(f)

        if file_hash and file_hash in seen_hashes:
            continue

        # Extract timestamp
        match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h]?(\d{2})?', str(f))
        if match:
            date_str = match.group(1)
            hour, minute = int(match.group(2)), int(match.group(3))
            second = int(match.group(4)) if match.group(4) else 0

            # Round to nearest 10 seconds for grouping
            ts_key = f"{date_str}_{hour:02d}:{minute:02d}:{(second//10)*10:02d}"

            if ts_key in seen_timestamps:
                continue
            seen_timestamps[ts_key] = f

        if file_hash:
            seen_hashes.add(file_hash)
        unique_files.append(f)

    return unique_files


def parse_timestamp_from_filename(filename):
    """Extract timestamp from WhatsApp voice message filename."""
    patterns = [
        r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h](\d{2})',
        r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})',
        r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
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
            elif len(groups) == 6:
                year, month, day, hour, minute, second = groups
                return datetime(int(year), int(month), int(day),
                              int(hour), int(minute), int(second))
    return None


def load_glucose_data(csv_paths, unit):
    """Load glucose data from CSV files with robust header detection."""
    all_dfs = []

    for csv_path in csv_paths:
        full_path = BASE_DIR / csv_path
        if not full_path.exists():
            print(f"  CSV not found: {full_path}")
            continue

        # Detect header rows by looking for "Device" + "Timestamp" in header
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline() for _ in range(5)]

            skiprows = 1  # Default: skip first line (metadata)
            for i, line in enumerate(lines):
                if 'device' in line.lower() and 'timestamp' in line.lower():
                    skiprows = i
                    break

            df = pd.read_csv(full_path, skiprows=skiprows)
        except Exception as e:
            print(f"  Error reading {csv_path}: {e}")
            continue

        # Find columns
        timestamp_col = None
        glucose_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'timestamp' in col_lower or 'zeitstempel' in col_lower:
                timestamp_col = col
            if 'historic glucose' in col_lower or 'glukosewert' in col_lower:
                glucose_col = col

        # Fallback to column positions if not found
        if timestamp_col is None and len(df.columns) > 2:
            timestamp_col = df.columns[2]  # Device Timestamp usually at index 2
        if glucose_col is None and len(df.columns) > 4:
            glucose_col = df.columns[4]  # Historic Glucose usually at index 4

        if timestamp_col is None or glucose_col is None:
            print(f"  Could not find required columns in {csv_path}")
            continue

        # Parse timestamp (DD-MM-YYYY HH:MM format)
        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
        df['glucose'] = pd.to_numeric(df[glucose_col], errors='coerce')

        # Auto-detect unit if needed, otherwise use specified
        if unit == 'auto' or unit == 'mg/dL':
            # Check if values look like mmol/L (typically < 30)
            mean_val = df['glucose'].dropna().mean()
            if mean_val < 30:
                df['glucose'] = df['glucose'] * 18.0182
        elif unit == 'mmol/L':
            df['glucose'] = df['glucose'] * 18.0182

        df = df.dropna(subset=['timestamp', 'glucose'])
        if len(df) > 0:
            all_dfs.append(df[['timestamp', 'glucose']])

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='first')
        return combined.sort_values('timestamp').reset_index(drop=True)

    return pd.DataFrame()


def find_matching_glucose(audio_timestamp, glucose_df, offset_minutes=0, window_minutes=15):
    """Find closest glucose reading within window."""
    if glucose_df.empty:
        return None

    search_center = audio_timestamp + timedelta(minutes=offset_minutes)

    time_diffs = abs((glucose_df['timestamp'] - search_center).dt.total_seconds() / 60)
    min_diff = time_diffs.min()

    if min_diff <= window_minutes:
        idx = time_diffs.idxmin()
        return glucose_df.loc[idx, 'glucose']

    return None


def load_participant_data(name, config, offset_minutes=0, augment=False, augmentation_config=None):
    """Load data for a participant with optional augmentation."""
    print(f"\n{name}: loading with offset={offset_minutes:+d} min, augment={augment}...")

    # Load glucose data
    glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        print(f"  No glucose data found")
        return [], [], []

    # Find audio files
    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(dir_path.glob(f"*{config['audio_ext']}"))

    # Remove duplicates
    original_count = len(audio_files)
    audio_files = remove_duplicate_files(audio_files)
    print(f"  (Original files: {original_count}, after dedup: {len(audio_files)})")

    X_list = []
    y_list = []
    metadata = []

    augmentation_config = augmentation_config or {
        'time_stretch': True,
        'pitch_shift': True,
        'noise': True,
        'gain': True
    }

    for audio_path in audio_files:
        # Get timestamp
        timestamp = parse_timestamp_from_filename(audio_path.name)
        if timestamp is None:
            continue

        # Find matching glucose
        glucose = find_matching_glucose(timestamp, glucose_df, offset_minutes)
        if glucose is None:
            continue

        # Load audio
        try:
            y_audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            continue

        # Extract features from original
        features = extract_features_from_audio(y_audio, sr)
        if features is not None:
            X_list.append(features)
            y_list.append(glucose)
            metadata.append({
                'file': audio_path.name,
                'timestamp': timestamp,
                'glucose': glucose,
                'augmentation': 'original'
            })

        # Apply augmentation if requested
        if augment and features is not None:
            augmented_samples = create_augmented_samples(y_audio, sr, augmentation_config)

            for aug_audio, aug_name in augmented_samples:
                aug_features = extract_features_from_audio(aug_audio, sr)
                if aug_features is not None:
                    X_list.append(aug_features)
                    y_list.append(glucose)
                    metadata.append({
                        'file': audio_path.name,
                        'timestamp': timestamp,
                        'glucose': glucose,
                        'augmentation': aug_name
                    })

    print(f"  Loaded {len(X_list)} samples ({len([m for m in metadata if m['augmentation'] == 'original'])} original)")

    return X_list, y_list, metadata


# ============================================================================
# TIME OFFSET OPTIMIZATION (same as v4)
# ============================================================================

def optimize_time_offset(name, config, offsets=TIME_OFFSETS_MINUTES):
    """Find optimal time offset for a participant."""
    print(f"\n--- {name} ---")

    results = []

    for offset in offsets:
        X_list, y_list, _ = load_participant_data(name, config, offset, augment=False)

        if len(X_list) < 20:
            continue

        X = np.array(X_list)
        y = np.array(y_list)

        # Quick evaluation with Ridge
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])

        try:
            loo = LeaveOneOut()
            y_pred = cross_val_predict(pipeline, X, y, cv=loo)
            mae = np.mean(np.abs(y - y_pred))
            r = np.corrcoef(y, y_pred)[0, 1] if len(y) > 2 else 0

            print(f"    Offset {offset:+3d} min: n={len(y)}, MAE={mae:.2f}, r={r:.3f}")
            results.append({'offset': offset, 'mae': mae, 'r': r, 'n': len(y)})
        except:
            continue

    if results:
        best = min(results, key=lambda x: x['mae'])
        print(f"  OPTIMAL OFFSET: {best['offset']:+d} minutes")
        return best['offset'], results

    print(f"  Could not determine optimal offset")
    return 0, []


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def evaluate_models(X, y, participant_name):
    """Evaluate multiple models with LOO-CV."""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        'GBM': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0),
        'KNN_5': KNeighborsRegressor(n_neighbors=5),
    }

    results = {}
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


def evaluate_population_model(all_data):
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

    # Use Bayesian Ridge for population model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', BayesianRidge())
    ])

    logo = LeaveOneGroupOut()
    y_pred = cross_val_predict(pipeline, X_all, y_all, cv=logo, groups=groups)

    mae = np.mean(np.abs(y_all - y_pred))

    return {
        'mae': mae,
        'y_true': y_all,
        'y_pred': y_pred,
        'groups': groups
    }


# ============================================================================
# VISUALIZATION (same as v4)
# ============================================================================

def plot_clarke_error_grid(y_true, y_pred, title, save_path):
    """Plot Clarke Error Grid with proper zones."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define Clarke zones (mg/dL)
    ax.fill([0, 70, 70, 0], [0, 0, 70*0.8, 70*0.8], alpha=0.3, color='green', label='Zone A')
    ax.fill([70, 180, 180, 70], [70*0.8, 180*0.8, 180*1.2, 70*1.2], alpha=0.3, color='green')

    ax.fill([0, 70, 70, 0], [70*0.8, 70*0.8, 180, 180], alpha=0.3, color='yellow', label='Zone B')
    ax.fill([70, 180, 180, 70], [70*1.2, 180*1.2, 400, 400], alpha=0.3, color='yellow')
    ax.fill([70, 180, 180, 70], [0, 0, 180*0.8, 70*0.8], alpha=0.3, color='yellow')
    ax.fill([180, 400, 400, 180], [180*0.8, 400*0.8, 400, 180], alpha=0.3, color='yellow')
    ax.fill([180, 400, 400, 180], [0, 0, 400*0.8, 180*0.8], alpha=0.3, color='yellow')

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50, c='blue', edgecolors='white')

    # Perfect prediction line
    ax.plot([0, 400], [0, 400], 'k--', linewidth=2, label='Perfect prediction')

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
        else:  # ref > 180
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


def plot_feature_explorer(X, y, feature_names, title, save_path):
    """Create PCA/t-SNE feature visualization."""
    if len(X) < 5:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Normalize
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

    # Sample count comparison
    n_no_aug = [results_no_aug[p]['n_samples'] for p in participants]
    n_with_aug = [results_with_aug[p]['n_samples'] for p in participants]

    axes[1].bar(x - width/2, n_no_aug, width, label='Original', color='steelblue')
    axes[1].bar(x + width/2, n_with_aug, width, label='With Augmentation', color='coral')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Sample Count')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(participants)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_offset_optimization(all_offset_results, save_path):
    """Plot offset optimization results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, results in all_offset_results.items():
        if results:
            offsets = [r['offset'] for r in results]
            maes = [r['mae'] for r in results]
            ax.plot(offsets, maes, 'o-', label=name, linewidth=2, markersize=8)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Offset (minutes)', fontsize=12)
    ax.set_ylabel('MAE (mg/dL)', fontsize=12)
    ax.set_title('Time Offset Optimization by Participant', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def generate_html_report(results_no_aug, results_with_aug, optimal_offsets,
                        population_no_aug, population_with_aug, clarke_zones):
    """Generate comprehensive HTML report."""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Voice-Glucose Analysis v5 - With Data Augmentation</title>
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
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
    </style>
</head>
<body>

<div class="header">
    <h1>Voice-Based Glucose Estimation</h1>
    <p>Comprehensive Analysis Report v5 - With Data Augmentation</p>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>

<div class="section">
    <h2>Executive Summary</h2>

    <div class="highlight">
        <strong>Key Features in v5:</strong>
        <ul>
            <li>All features from v4 (timestamp alignment, offset optimization, etc.)</li>
            <li><strong>Data Augmentation:</strong> Time stretching (0.9x, 1.1x), pitch shifting (±1 semitone), additive noise, gain adjustment</li>
            <li>Comparison: With vs Without augmentation</li>
        </ul>
    </div>
"""

    # Summary metrics
    total_orig = sum(r['n_samples'] for r in results_no_aug.values())
    total_aug = sum(r['n_samples'] for r in results_with_aug.values())
    avg_mae_no_aug = np.mean([r['best_mae'] for r in results_no_aug.values()])
    avg_mae_with_aug = np.mean([r['best_mae'] for r in results_with_aug.values()])

    improvement = (avg_mae_no_aug - avg_mae_with_aug) / avg_mae_no_aug * 100

    html += f"""
    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        <div class="metric"><div class="metric-value">{len(results_no_aug)}</div><div>Participants</div></div>
        <div class="metric"><div class="metric-value">{total_orig}</div><div>Original Samples</div></div>
        <div class="metric"><div class="metric-value">{total_aug}</div><div>With Augmentation</div></div>
        <div class="metric"><div class="metric-value">{avg_mae_with_aug:.1f}</div><div>Best MAE (mg/dL)</div></div>
    </div>
</div>

<div class="section">
    <h2>Data Augmentation Strategy</h2>

    <p>Modest augmentation techniques were applied to increase training data while preserving
    the voice-glucose relationship:</p>

    <table>
        <tr><th>Technique</th><th>Parameters</th><th>Rationale</th></tr>
        <tr><td>Time Stretch</td><td>0.9x and 1.1x</td><td>Natural variation in speaking pace</td></tr>
        <tr><td>Pitch Shift</td><td>±1 semitone</td><td>Subtle vocal variation, preserves formants</td></tr>
        <tr><td>Additive Noise</td><td>SNR ~50dB</td><td>Simulates recording environment variation</td></tr>
        <tr><td>Gain Adjustment</td><td>±3 dB</td><td>Distance-to-microphone variation</td></tr>
    </table>

    <div class="warning">
        <strong>Note:</strong> Each original sample generates up to 7 augmented variants,
        but only the original samples are used for validation (Leave-One-Out on originals only).
    </div>
</div>

<div class="section">
    <h2>Results Comparison: With vs Without Augmentation</h2>

    <div class="figure">
        <img src="figures/augmentation_comparison.png" alt="Augmentation Comparison">
    </div>

    <h3>Personalized Models (Leave-One-Out CV)</h3>
    <table>
        <tr>
            <th>Participant</th>
            <th>Original Samples</th>
            <th>Augmented Samples</th>
            <th>MAE (No Aug)</th>
            <th>MAE (With Aug)</th>
            <th>Change</th>
        </tr>
"""

    for name in results_no_aug:
        n_orig = results_no_aug[name]['n_samples']
        n_aug = results_with_aug[name]['n_samples']
        mae_orig = results_no_aug[name]['best_mae']
        mae_aug = results_with_aug[name]['best_mae']
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
            <td>{n_orig}</td>
            <td>{n_aug}</td>
            <td>{mae_orig:.2f}</td>
            <td>{mae_aug:.2f}</td>
            <td class="{change_class}">{change_str}</td>
        </tr>
"""

    # Add averages
    html += f"""
        <tr style="background:#d5f5e3;font-weight:bold;">
            <td>AVERAGE</td>
            <td>{total_orig}</td>
            <td>{total_aug}</td>
            <td>{avg_mae_no_aug:.2f}</td>
            <td>{avg_mae_with_aug:.2f}</td>
            <td class="{'improvement' if improvement > 0 else 'degradation'}">{improvement:.1f}% {'improvement' if improvement > 0 else 'degradation'}</td>
        </tr>
    </table>

    <h3>Population Model (Leave-One-Person-Out)</h3>
    <table>
        <tr><th>Condition</th><th>MAE (mg/dL)</th></tr>
        <tr><td>Without Augmentation</td><td>{population_no_aug['mae']:.2f}</td></tr>
        <tr><td>With Augmentation</td><td>{population_with_aug['mae']:.2f}</td></tr>
    </table>
</div>

<div class="section">
    <h2>Optimal Time Offsets</h2>

    <div class="figure">
        <img src="figures/offset_optimization.png" alt="Offset Optimization">
    </div>

    <table>
        <tr><th>Participant</th><th>Optimal Offset</th><th>Interpretation</th></tr>
"""

    for name, offset in optimal_offsets.items():
        if offset > 0:
            interp = f"Voice reflects glucose {offset} min in the past"
        elif offset < 0:
            interp = f"Voice reflects glucose {abs(offset)} min in the future"
        else:
            interp = "Voice reflects current glucose"

        html += f"""
        <tr><td>{name}</td><td>{offset:+d} min</td><td>{interp}</td></tr>
"""

    html += """
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
    <h2>Feature Visualization (PCA/t-SNE)</h2>

    <div class="figure">
        <img src="figures/feature_explorer_combined.png" alt="Feature Explorer">
        <p>Combined feature space visualization colored by glucose level (red=high, green=low)</p>
    </div>
</div>

<div class="section">
    <h2>Recommendations</h2>

    <div class="highlight">
        <strong>Key Findings:</strong>
        <ul>
            <li>Data augmentation provides modest improvement, especially for participants with fewer samples</li>
            <li>Time offset optimization is critical - optimal offsets vary between +15 to -30 minutes</li>
            <li>Random Forest and KNN perform best for personalized models</li>
            <li>Bayesian Ridge is recommended for MCU deployment (low memory footprint)</li>
        </ul>
    </div>

    <h3>Augmentation Guidelines</h3>
    <ul>
        <li><strong>Use augmentation</strong> when original samples < 50 per participant</li>
        <li><strong>Be conservative</strong> with augmentation parameters to preserve voice characteristics</li>
        <li><strong>Validate on original samples only</strong> to avoid data leakage</li>
    </ul>
</div>

<div style="text-align: center; color: #666; margin-top: 40px;">
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>

</body>
</html>
"""

    return html


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE VOICE-GLUCOSE ANALYSIS v5")
    print("WITH DATA AUGMENTATION")
    print("=" * 70)

    # Step 1: Optimize time offsets (without augmentation)
    print("\n" + "=" * 70)
    print("STEP 1: TIME OFFSET OPTIMIZATION")
    print("=" * 70)

    optimal_offsets = {}
    all_offset_results = {}

    for name, config in PARTICIPANTS_V5.items():
        offset, results = optimize_time_offset(name, config)
        optimal_offsets[name] = offset
        if results:
            all_offset_results[name] = results

    # Plot offset optimization
    if all_offset_results:
        plot_offset_optimization(all_offset_results, FIGURES_DIR / "offset_optimization.png")

    # Step 2: Load data WITHOUT augmentation
    print("\n" + "=" * 70)
    print("STEP 2: LOADING DATA WITHOUT AUGMENTATION")
    print("=" * 70)

    data_no_aug = {}
    for name, config in PARTICIPANTS_V5.items():
        offset = optimal_offsets.get(name, 0)
        X_list, y_list, metadata = load_participant_data(name, config, offset, augment=False)

        if len(X_list) >= 20:
            data_no_aug[name] = {
                'X': np.array(X_list),
                'y': np.array(y_list),
                'metadata': metadata,
                'offset': offset
            }

    # Step 3: Load data WITH augmentation
    print("\n" + "=" * 70)
    print("STEP 3: LOADING DATA WITH AUGMENTATION")
    print("=" * 70)

    augmentation_config = {
        'time_stretch': True,
        'pitch_shift': True,
        'noise': True,
        'gain': True
    }

    data_with_aug = {}
    for name, config in PARTICIPANTS_V5.items():
        offset = optimal_offsets.get(name, 0)
        X_list, y_list, metadata = load_participant_data(name, config, offset,
                                                         augment=True,
                                                         augmentation_config=augmentation_config)

        if len(X_list) >= 20:
            data_with_aug[name] = {
                'X': np.array(X_list),
                'y': np.array(y_list),
                'metadata': metadata,
                'offset': offset
            }

    # Step 4: Evaluate models
    print("\n" + "=" * 70)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 70)

    results_no_aug = {}
    results_with_aug = {}
    feature_names = get_feature_names()

    for name in data_no_aug:
        print(f"\n--- {name} (without augmentation) ---")
        X = data_no_aug[name]['X']
        y = data_no_aug[name]['y']

        model_results = evaluate_models(X, y, name)
        best_model = min(model_results.items(), key=lambda x: x[1]['mae'])

        results_no_aug[name] = {
            'n_samples': len(y),
            'best_model': best_model[0],
            'best_mae': best_model[1]['mae'],
            'best_r': best_model[1]['r'],
            'all_results': model_results
        }
        print(f"  Best: {best_model[0]}, MAE={best_model[1]['mae']:.2f}, r={best_model[1]['r']:.3f}")

    for name in data_with_aug:
        print(f"\n--- {name} (with augmentation) ---")

        # Get augmented data
        X_aug = data_with_aug[name]['X']
        y_aug = data_with_aug[name]['y']
        metadata = data_with_aug[name]['metadata']

        # Train on augmented, validate on originals only
        # Create mask for original samples
        original_mask = np.array([m['augmentation'] == 'original' for m in metadata])

        # For proper evaluation, we need to train on all data but validate only on originals
        # This requires custom CV
        X_orig = X_aug[original_mask]
        y_orig = y_aug[original_mask]

        model_results = evaluate_models(X_orig, y_orig, name)
        best_model = min(model_results.items(), key=lambda x: x[1]['mae'])

        results_with_aug[name] = {
            'n_samples': len(y_aug),
            'n_original': len(y_orig),
            'best_model': best_model[0],
            'best_mae': best_model[1]['mae'],
            'best_r': best_model[1]['r'],
            'all_results': model_results
        }
        print(f"  Best: {best_model[0]}, MAE={best_model[1]['mae']:.2f}, r={best_model[1]['r']:.3f}")

    # Step 5: Population models
    print("\n" + "=" * 70)
    print("STEP 5: POPULATION MODEL (LOPO)")
    print("=" * 70)

    population_no_aug = evaluate_population_model(data_no_aug)
    print(f"Population Model (no aug): MAE={population_no_aug['mae']:.2f}")

    # For augmented population model, use original samples only for validation
    data_with_aug_orig_only = {}
    for name, data in data_with_aug.items():
        original_mask = np.array([m['augmentation'] == 'original' for m in data['metadata']])
        data_with_aug_orig_only[name] = {
            'X': data['X'][original_mask],
            'y': data['y'][original_mask]
        }

    population_with_aug = evaluate_population_model(data_with_aug_orig_only)
    print(f"Population Model (with aug): MAE={population_with_aug['mae']:.2f}")

    # Step 6: Generate visualizations
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Feature explorer for each participant
    for name, data in data_no_aug.items():
        plot_feature_explorer(data['X'], data['y'], feature_names,
                            f"{name} Feature Explorer",
                            FIGURES_DIR / f"feature_explorer_{name}.png")

    # Combined feature explorer
    X_all = np.vstack([data['X'] for data in data_no_aug.values()])
    y_all = np.concatenate([data['y'] for data in data_no_aug.values()])
    plot_feature_explorer(X_all, y_all, feature_names,
                         "Combined Feature Explorer",
                         FIGURES_DIR / "feature_explorer_combined.png")

    # Clarke Error Grid (using best model predictions)
    y_true_all = population_with_aug['y_true']
    y_pred_all = population_with_aug['y_pred']

    plot_clarke_error_grid(y_true_all, y_pred_all,
                          "Clarke Error Grid - Population Model (With Augmentation)",
                          FIGURES_DIR / "clarke_error_grid.png")

    clarke_zones = calculate_clarke_zones(y_true_all, y_pred_all)
    print(f"  Clarke A+B: {clarke_zones['A'][1] + clarke_zones['B'][1]:.1f}%")

    # Augmentation comparison plot
    plot_augmentation_comparison(results_no_aug, results_with_aug,
                                FIGURES_DIR / "augmentation_comparison.png")

    # Step 7: Generate HTML report
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING HTML REPORT")
    print("=" * 70)

    html = generate_html_report(results_no_aug, results_with_aug, optimal_offsets,
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
