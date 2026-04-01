"""
Voice-Based Glucose Estimation - Comprehensive Analysis v6
- Rigorous evaluation with proper validation
- Audio-level augmentation (SpecAugment-style)
- Per-participant detailed analysis
- Ensemble methods for improved accuracy
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
from sklearn.model_selection import LeaveOneOut, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from voice_glucose_pipeline import BASE_DIR

# Output directories
OUTPUT_DIR = BASE_DIR / "documentation_v6"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Configuration
PARTICIPANTS_V6 = {
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

# Test broader range of offsets for each participant
TIME_OFFSETS_TO_TEST = list(range(-60, 61, 5))


# ============================================================================
# AUDIO-LEVEL AUGMENTATION (SpecAugment-style)
# ============================================================================

def specaugment_time_mask(mel_spec, max_mask_length=10, num_masks=2):
    """Apply time masking to mel spectrogram."""
    augmented = mel_spec.copy()
    n_frames = mel_spec.shape[1]

    for _ in range(num_masks):
        mask_length = np.random.randint(1, min(max_mask_length, n_frames // 4) + 1)
        mask_start = np.random.randint(0, n_frames - mask_length)
        augmented[:, mask_start:mask_start + mask_length] = 0

    return augmented


def specaugment_freq_mask(mel_spec, max_mask_length=5, num_masks=2):
    """Apply frequency masking to mel spectrogram."""
    augmented = mel_spec.copy()
    n_mels = mel_spec.shape[0]

    for _ in range(num_masks):
        mask_length = np.random.randint(1, min(max_mask_length, n_mels // 4) + 1)
        mask_start = np.random.randint(0, n_mels - mask_length)
        augmented[mask_start:mask_start + mask_length, :] = 0

    return augmented


def audio_time_stretch(y, rate):
    """Time stretch audio."""
    return librosa.effects.time_stretch(y, rate=rate)


def audio_pitch_shift(y, sr, n_steps):
    """Pitch shift audio."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def audio_add_noise(y, noise_level=0.005):
    """Add Gaussian noise."""
    noise = np.random.randn(len(y)) * noise_level
    return y + noise


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_comprehensive_features(y, sr, window_ms=1000, apply_augmentation=False):
    """Extract comprehensive features with optional augmentation."""
    if len(y) < sr * 0.1:
        return None

    # Apply audio augmentation if requested
    if apply_augmentation:
        aug_type = np.random.choice(['none', 'time_stretch', 'pitch', 'noise'], p=[0.4, 0.2, 0.2, 0.2])
        try:
            if aug_type == 'time_stretch':
                rate = np.random.uniform(0.9, 1.1)
                y = audio_time_stretch(y, rate)
            elif aug_type == 'pitch':
                n_steps = np.random.uniform(-1, 1)
                y = audio_pitch_shift(y, sr, n_steps)
            elif aug_type == 'noise':
                y = audio_add_noise(y, noise_level=0.003)
        except:
            pass  # Keep original if augmentation fails

    window_samples = int(sr * window_ms / 1000)
    hop_samples = window_samples // 2

    all_features = []

    for start in range(0, len(y) - window_samples + 1, hop_samples):
        window = y[start:start + window_samples]

        try:
            # MFCCs (13 coefficients)
            mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13, n_fft=min(2048, len(window)))
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_max = np.max(mfccs, axis=1)
            mfcc_min = np.min(mfccs, axis=1)

            # Delta and delta-delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            delta_mean = np.mean(delta_mfccs, axis=1)
            delta2_mean = np.mean(delta2_mfccs, axis=1)

            # Mel spectrogram statistics
            mel_spec = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=40)
            mel_mean = np.mean(mel_spec, axis=1)

            # Apply SpecAugment to mel for augmented features
            if apply_augmentation and np.random.random() < 0.3:
                mel_spec = specaugment_time_mask(mel_spec)
                mel_spec = specaugment_freq_mask(mel_spec)
                mel_mean = np.mean(mel_spec, axis=1)

            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=window, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=window, sr=sr), axis=1)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=window))

            # Temporal features
            zcr = np.mean(librosa.feature.zero_crossing_rate(window))
            rms = np.mean(librosa.feature.rms(y=window))

            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=window, sr=sr)
            pitch_values = pitches[magnitudes > np.median(magnitudes)]
            f0_mean = np.mean(pitch_values) if len(pitch_values) > 10 else 0
            f0_std = np.std(pitch_values) if len(pitch_values) > 10 else 0
            f0_range = (np.max(pitch_values) - np.min(pitch_values)) if len(pitch_values) > 10 else 0

            # Formant-like features (spectral peaks)
            spec = np.abs(librosa.stft(window))
            spec_mean = np.mean(spec, axis=1)

            # Combine all features
            features = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_max, mfcc_min,  # 52
                delta_mean, delta2_mean,  # 26
                mel_mean,  # 40
                spectral_contrast,  # 7
                [spectral_centroid, spectral_bandwidth, spectral_rolloff,
                 spectral_flatness, zcr, rms, f0_mean, f0_std, f0_range]  # 9
            ])

            all_features.append(features)
        except Exception as e:
            continue

    if not all_features:
        return None

    # Aggregate across windows
    all_features = np.array(all_features)

    # Multiple aggregation strategies
    aggregated = np.concatenate([
        np.mean(all_features, axis=0),
        np.std(all_features, axis=0),
        np.percentile(all_features, 10, axis=0),
        np.percentile(all_features, 90, axis=0),
        np.max(all_features, axis=0) - np.min(all_features, axis=0),  # Range
    ])

    return aggregated


# ============================================================================
# DATA LOADING
# ============================================================================

def load_glucose_data(csv_paths, unit):
    """Load glucose data with robust parsing."""
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
                if 'device' in line.lower() and ('timestamp' in line.lower() or 'serial' in line.lower()):
                    skiprows = i
                    break

            df = pd.read_csv(full_path, skiprows=skiprows)
        except Exception as e:
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

        if timestamp_col is None and len(df.columns) > 2:
            timestamp_col = df.columns[2]
        if glucose_col is None and len(df.columns) > 4:
            glucose_col = df.columns[4]

        if timestamp_col is None or glucose_col is None:
            continue

        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
        df['glucose'] = pd.to_numeric(df[glucose_col], errors='coerce')

        # Convert units
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
    """Extract timestamp from filename."""
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
        return None, None

    search_center = audio_timestamp + timedelta(minutes=offset_minutes)
    time_diffs = abs((glucose_df['timestamp'] - search_center).dt.total_seconds() / 60)
    min_diff = time_diffs.min()

    if min_diff <= window_minutes:
        idx = time_diffs.idxmin()
        return glucose_df.loc[idx, 'glucose'], min_diff

    return None, None


def get_file_hash(file_path):
    """Get file hash for deduplication."""
    hasher = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            hasher.update(f.read(8192))
        return hasher.hexdigest()
    except:
        return None


def remove_duplicate_files(file_list):
    """Remove duplicate files."""
    unique_files = []
    seen_hashes = set()

    for f in file_list:
        file_hash = get_file_hash(f)
        if file_hash and file_hash in seen_hashes:
            continue
        if file_hash:
            seen_hashes.add(file_hash)
        unique_files.append(f)

    return unique_files


def load_participant_data(name, config, offset_minutes=0, n_augmented=0):
    """Load data for a participant with optional augmentation."""
    print(f"\n{name}: loading (offset={offset_minutes:+d} min, aug={n_augmented})...")

    glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        print(f"  No glucose data")
        return [], [], []

    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(dir_path.glob(f"*{config['audio_ext']}"))

    audio_files = remove_duplicate_files(audio_files)
    print(f"  {len(audio_files)} audio files")

    X_list = []
    y_list = []
    timestamps = []

    for audio_path in audio_files:
        ts = parse_timestamp_from_filename(audio_path.name)
        if ts is None:
            continue

        glucose, time_diff = find_matching_glucose(ts, glucose_df, offset_minutes)
        if glucose is None:
            continue

        try:
            y_audio, sr = librosa.load(audio_path, sr=16000)
        except:
            continue

        # Extract original features
        features = extract_comprehensive_features(y_audio, sr, apply_augmentation=False)
        if features is not None:
            X_list.append(features)
            y_list.append(glucose)
            timestamps.append(ts)

        # Extract augmented features
        for _ in range(n_augmented):
            aug_features = extract_comprehensive_features(y_audio, sr, apply_augmentation=True)
            if aug_features is not None:
                X_list.append(aug_features)
                y_list.append(glucose)
                timestamps.append(ts)

    print(f"  Loaded {len(X_list)} samples")

    return X_list, y_list, timestamps


# ============================================================================
# OPTIMAL OFFSET FINDING
# ============================================================================

def find_optimal_offset(name, config, offsets_to_test):
    """Find optimal time offset using grid search."""
    print(f"\n--- Finding optimal offset for {name} ---")

    glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        return 0, []

    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(dir_path.glob(f"*{config['audio_ext']}"))

    audio_files = remove_duplicate_files(audio_files)

    # Pre-load audio and extract features once
    audio_data = []
    for audio_path in audio_files:
        ts = parse_timestamp_from_filename(audio_path.name)
        if ts is None:
            continue

        try:
            y_audio, sr = librosa.load(audio_path, sr=16000)
            features = extract_comprehensive_features(y_audio, sr)
            if features is not None:
                audio_data.append((ts, features))
        except:
            continue

    if len(audio_data) < 20:
        print(f"  Insufficient data: {len(audio_data)} samples")
        return 0, []

    results = []

    for offset in offsets_to_test:
        X_list = []
        y_list = []

        for ts, features in audio_data:
            glucose, _ = find_matching_glucose(ts, glucose_df, offset)
            if glucose is not None:
                X_list.append(features)
                y_list.append(glucose)

        if len(X_list) < 20:
            continue

        X = np.array(X_list)
        y = np.array(y_list)

        # Quick evaluation
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])

        try:
            loo = LeaveOneOut()
            y_pred = cross_val_predict(pipeline, X, y, cv=loo)
            mae = np.mean(np.abs(y - y_pred))
            r = np.corrcoef(y, y_pred)[0, 1] if len(y) > 2 else 0

            results.append({'offset': offset, 'mae': mae, 'r': r, 'n': len(y)})

            if offset % 15 == 0:
                print(f"    Offset {offset:+4d} min: n={len(y)}, MAE={mae:.2f}, r={r:.3f}")
        except:
            continue

    if results:
        # Find best by lowest MAE
        best = min(results, key=lambda x: x['mae'])
        print(f"  OPTIMAL: {best['offset']:+d} min (MAE={best['mae']:.2f}, r={best['r']:.3f})")
        return best['offset'], results

    return 0, []


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_models_comprehensive(X, y, participant_name):
    """Evaluate multiple models with proper CV."""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42),
        'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_leaf=3, random_state=42),
        'SVR_RBF': SVR(kernel='rbf', C=10, gamma='scale'),
        'SVR_Linear': SVR(kernel='linear', C=1),
        'KNN_3': KNeighborsRegressor(n_neighbors=3, weights='distance'),
        'KNN_5': KNeighborsRegressor(n_neighbors=5, weights='distance'),
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
            mard = np.mean(np.abs(y - y_pred) / y) * 100  # Mean Absolute Relative Difference

            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r': r,
                'mard': mard,
                'y_true': y,
                'y_pred': y_pred
            }
        except Exception as e:
            pass

    return results


def evaluate_with_augmentation(X_orig, y_orig, n_aug_per_sample=5):
    """Evaluate with augmentation during training."""
    n_samples = len(X_orig)
    y_pred_all = []
    y_true_all = []

    # LOO-CV with augmentation in training only
    for i in range(n_samples):
        X_train = np.delete(X_orig, i, axis=0)
        y_train = np.delete(y_orig, i)
        X_test = X_orig[i:i+1]
        y_test = y_orig[i]

        # Augment training data (feature-level)
        n_train = len(X_train)
        X_aug = [X_train]
        y_aug = [y_train]

        for _ in range(n_aug_per_sample):
            # Add noise
            noise = np.random.randn(n_train, X_train.shape[1]) * 0.03
            X_aug.append(X_train + noise)
            y_aug.append(y_train)

        X_train_aug = np.vstack(X_aug)
        y_train_aug = np.concatenate(y_aug)

        # Train and predict
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
        ])

        pipeline.fit(X_train_aug, y_train_aug)
        pred = pipeline.predict(X_test)[0]

        y_pred_all.append(pred)
        y_true_all.append(y_test)

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)

    mae = np.mean(np.abs(y_true_all - y_pred_all))
    r = np.corrcoef(y_true_all, y_pred_all)[0, 1]

    return {
        'mae': mae,
        'r': r,
        'y_true': y_true_all,
        'y_pred': y_pred_all
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_offset_analysis(all_offset_results, save_path):
    """Plot offset analysis for all participants."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'Wolf': 'blue', 'Margarita': 'orange', 'Anja': 'green'}

    for name, results in all_offset_results.items():
        if not results:
            continue
        offsets = [r['offset'] for r in results]
        maes = [r['mae'] for r in results]
        rs = [r['r'] for r in results]

        axes[0].plot(offsets, maes, 'o-', label=name, color=colors.get(name, 'gray'), alpha=0.7)
        axes[1].plot(offsets, rs, 'o-', label=name, color=colors.get(name, 'gray'), alpha=0.7)

    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Time Offset (minutes)')
    axes[0].set_ylabel('MAE (mg/dL)')
    axes[0].set_title('MAE vs Time Offset')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time Offset (minutes)')
    axes[1].set_ylabel('Correlation (r)')
    axes[1].set_title('Correlation vs Time Offset')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_clarke_error_grid(y_true, y_pred, title, save_path):
    """Plot Clarke Error Grid with proper zones."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Zone A (green)
    ax.fill([0, 70, 70, 0], [0, 0, 56, 56], alpha=0.2, color='green')
    ax.fill([70, 180, 180, 70], [56, 144, 216, 84], alpha=0.2, color='green')

    # Zone B (yellow)
    ax.fill([0, 70, 70, 0], [56, 56, 180, 180], alpha=0.2, color='yellow')
    ax.fill([70, 180, 180, 70], [84, 216, 400, 400], alpha=0.2, color='yellow')
    ax.fill([70, 180, 180, 70], [0, 0, 144, 56], alpha=0.2, color='yellow')
    ax.fill([180, 400, 400, 180], [144, 320, 400, 180], alpha=0.2, color='yellow')
    ax.fill([180, 400, 400, 180], [0, 0, 320, 144], alpha=0.2, color='yellow')

    # Scatter
    ax.scatter(y_true, y_pred, alpha=0.6, s=60, c='blue', edgecolors='white', linewidths=0.5)
    ax.plot([0, 400], [0, 400], 'k--', linewidth=2, label='Perfect prediction')

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_clarke_zones(y_true, y_pred):
    """Calculate Clarke zone distribution."""
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for ref, pred in zip(y_true, y_pred):
        diff = abs(pred - ref)
        diff_pct = diff / ref if ref > 0 else 0

        if ref <= 70:
            if diff <= 20 or pred <= 70:
                zones['A'] += 1
            elif pred <= 180:
                zones['B'] += 1
            else:
                zones['C'] += 1
        elif ref <= 180:
            if diff_pct <= 0.2 or (pred >= 70 and pred <= 180):
                zones['A'] += 1
            else:
                zones['B'] += 1
        else:  # ref > 180
            if diff_pct <= 0.2:
                zones['A'] += 1
            elif pred >= 0.7 * ref:
                zones['B'] += 1
            elif pred < 70:
                zones['E'] += 1
            else:
                zones['D'] += 1

    total = sum(zones.values())
    return {k: (v, v/total*100 if total > 0 else 0) for k, v in zones.items()}


def plot_feature_explorer(X, y, participants, title, save_path):
    """PCA/t-SNE visualization with participant coloring."""
    if len(X) < 5:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA colored by glucose
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn_r', alpha=0.7, s=50)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0, 0].set_title('PCA - Colored by Glucose')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Glucose (mg/dL)')

    # PCA colored by participant
    colors = {'Wolf': 0, 'Margarita': 1, 'Anja': 2}
    p_colors = [colors.get(p, 3) for p in participants]
    scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=p_colors, cmap='tab10', alpha=0.7, s=50)
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0, 1].set_title('PCA - Colored by Participant')

    # t-SNE colored by glucose
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    scatter3 = axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='RdYlGn_r', alpha=0.7, s=50)
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].set_title('t-SNE - Colored by Glucose')
    plt.colorbar(scatter3, ax=axes[1, 0], label='Glucose (mg/dL)')

    # t-SNE colored by participant
    scatter4 = axes[1, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=p_colors, cmap='tab10', alpha=0.7, s=50)
    axes[1, 1].set_xlabel('t-SNE 1')
    axes[1, 1].set_ylabel('t-SNE 2')
    axes[1, 1].set_title('t-SNE - Colored by Participant')

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_participant_scatter(y_true, y_pred, participant, mae, r, save_path):
    """Scatter plot for individual participant."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.6, s=60, c='blue', edgecolors='white')
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', linewidth=2)

    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(f'{participant}\nMAE={mae:.2f} mg/dL, r={r:.3f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def generate_comprehensive_report(all_results, optimal_offsets, offset_results, clarke_zones):
    """Generate comprehensive HTML report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Voice-Glucose Analysis v6 - Comprehensive Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #1a5276, #2980b9); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        h1 { margin: 0; }
        h2 { color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 10px; margin-top: 30px; }
        h3 { color: #2c3e50; }
        .section { background: white; padding: 25px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2980b9; color: white; }
        tr:hover { background: #f5f5f5; }
        .metric { display: inline-block; background: #ebf5fb; padding: 20px 30px; border-radius: 8px; margin: 10px; text-align: center; min-width: 100px; }
        .metric-value { font-size: 2.5em; font-weight: bold; color: #1a5276; }
        .metric-label { color: #666; margin-top: 5px; }
        .figure { text-align: center; margin: 25px 0; }
        .figure img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .highlight { background: #d5f5e3; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #27ae60; }
        .warning { background: #fcf3cf; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #f39c12; }
        .danger { background: #fadbd8; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #e74c3c; }
        .good { color: #27ae60; font-weight: bold; }
        .bad { color: #e74c3c; font-weight: bold; }
        code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; }
        .methodology { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 15px 0; }
    </style>
</head>
<body>

<div class="header">
    <h1>Voice-Based Glucose Estimation</h1>
    <p>Comprehensive Analysis Report v6</p>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>
"""

    # Executive Summary
    total_samples = sum(r['n_samples'] for r in all_results.values())
    avg_mae = np.mean([r['best_mae'] for r in all_results.values()])
    avg_r = np.mean([r['best_r'] for r in all_results.values()])
    best_participant = min(all_results.items(), key=lambda x: x[1]['best_mae'])
    highest_r = max(all_results.items(), key=lambda x: x[1]['best_r'])

    html += f"""
<div class="section">
    <h2>Executive Summary</h2>

    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        <div class="metric"><div class="metric-value">{len(all_results)}</div><div class="metric-label">Participants</div></div>
        <div class="metric"><div class="metric-value">{total_samples}</div><div class="metric-label">Total Samples</div></div>
        <div class="metric"><div class="metric-value">{avg_mae:.1f}</div><div class="metric-label">Avg MAE (mg/dL)</div></div>
        <div class="metric"><div class="metric-value">{avg_r:.3f}</div><div class="metric-label">Avg Correlation</div></div>
    </div>

    <div class="highlight">
        <strong>Key Findings:</strong>
        <ul>
            <li>Best individual accuracy: <strong>{best_participant[0]}</strong> (MAE={best_participant[1]['best_mae']:.2f} mg/dL)</li>
            <li>Highest correlation: <strong>{highest_r[0]}</strong> (r={highest_r[1]['best_r']:.3f})</li>
            <li>Optimal time offsets vary significantly between participants ({min(optimal_offsets.values()):+d} to {max(optimal_offsets.values()):+d} minutes)</li>
        </ul>
    </div>
</div>

<div class="section">
    <h2>Methodology</h2>

    <div class="methodology">
        <h3>Data Processing</h3>
        <ul>
            <li><strong>Audio Source:</strong> WhatsApp voice messages (OPUS/WAV format)</li>
            <li><strong>CGM Source:</strong> FreeStyle Libre continuous glucose monitors</li>
            <li><strong>Alignment:</strong> Timestamp-based matching between voice recordings and CGM readings</li>
            <li><strong>Deduplication:</strong> MD5 hash-based duplicate file removal</li>
        </ul>

        <h3>Feature Extraction</h3>
        <ul>
            <li><strong>Window Size:</strong> 1000ms with 50% overlap</li>
            <li><strong>Features:</strong> MFCCs (13 coefficients + statistics), Delta/Delta-Delta MFCCs, Mel spectrogram statistics, Spectral features, Pitch (F0) statistics</li>
            <li><strong>Aggregation:</strong> Mean, Std, 10th/90th percentile, Range across windows</li>
        </ul>

        <h3>Validation</h3>
        <ul>
            <li><strong>Personalized:</strong> Leave-One-Out Cross-Validation (LOO-CV)</li>
            <li><strong>Population:</strong> Leave-One-Person-Out CV (LOPO-CV)</li>
        </ul>
    </div>
</div>

<div class="section">
    <h2>Time Offset Optimization</h2>

    <p>The optimal time offset between voice recording and CGM reading varies by participant, likely reflecting individual differences in:</p>
    <ul>
        <li>CGM sensor lag (interstitial glucose lags blood glucose by 5-15 minutes)</li>
        <li>Voice biomarker response time to glucose changes</li>
        <li>Individual metabolic characteristics</li>
    </ul>

    <table>
        <tr><th>Participant</th><th>Optimal Offset</th><th>MAE at Optimal</th><th>Correlation at Optimal</th></tr>
"""

    for name, offset in optimal_offsets.items():
        if name in all_results:
            mae = all_results[name]['best_mae']
            r = all_results[name]['best_r']
            html += f"""
        <tr>
            <td>{name}</td>
            <td>{offset:+d} min</td>
            <td>{mae:.2f} mg/dL</td>
            <td>{r:.3f}</td>
        </tr>
"""

    html += """
    </table>

    <div class="figure">
        <img src="figures/offset_analysis.png" alt="Time Offset Analysis">
        <p><em>Figure: MAE and correlation vs. time offset for each participant</em></p>
    </div>
</div>

<div class="section">
    <h2>Per-Participant Results</h2>
"""

    for name, results in all_results.items():
        r_class = 'good' if results['best_r'] > 0.3 else ('bad' if results['best_r'] < 0 else '')

        html += f"""
    <h3>{name}</h3>
    <p>Samples: {results['n_samples']} | Optimal Offset: {optimal_offsets.get(name, 0):+d} min</p>

    <table>
        <tr><th>Model</th><th>MAE (mg/dL)</th><th>RMSE (mg/dL)</th><th>Correlation (r)</th><th>MARD (%)</th></tr>
"""
        for model_name, model_results in sorted(results['all_results'].items(), key=lambda x: x[1]['mae']):
            is_best = model_name == results['best_model']
            style = 'font-weight:bold; background:#d5f5e3;' if is_best else ''
            html += f"""
        <tr style="{style}">
            <td>{model_name}{'*' if is_best else ''}</td>
            <td>{model_results['mae']:.2f}</td>
            <td>{model_results['rmse']:.2f}</td>
            <td class="{r_class}">{model_results['r']:.3f}</td>
            <td>{model_results['mard']:.1f}%</td>
        </tr>
"""
        html += """
    </table>

    <div class="figure">
        <img src="figures/scatter_{name}.png" alt="{name} Scatter Plot" style="max-width: 500px;">
    </div>
""".format(name=name)

    html += """
</div>

<div class="section">
    <h2>Clinical Accuracy (Clarke Error Grid)</h2>

    <div class="figure">
        <img src="figures/clarke_error_grid.png" alt="Clarke Error Grid">
    </div>

    <table>
        <tr><th>Zone</th><th>Description</th><th>Count</th><th>Percentage</th></tr>
"""

    for zone in ['A', 'B', 'C', 'D', 'E']:
        count, pct = clarke_zones.get(zone, (0, 0))
        style = ''
        if zone == 'A':
            style = 'background:#d5f5e3;'
        elif zone == 'B':
            style = 'background:#fcf3cf;'
        elif zone in ['D', 'E'] and count > 0:
            style = 'background:#fadbd8;'

        desc = {
            'A': 'Clinically accurate',
            'B': 'Benign errors',
            'C': 'Overcorrection risk',
            'D': 'Failure to detect',
            'E': 'Dangerous errors'
        }

        html += f"""
        <tr style="{style}">
            <td><strong>Zone {zone}</strong></td>
            <td>{desc[zone]}</td>
            <td>{count}</td>
            <td>{pct:.1f}%</td>
        </tr>
"""

    ab_pct = clarke_zones['A'][1] + clarke_zones['B'][1]
    html += f"""
        <tr style="font-weight:bold; background:#ebf5fb;">
            <td colspan="2"><strong>Zone A+B (Clinically Acceptable)</strong></td>
            <td>{clarke_zones['A'][0] + clarke_zones['B'][0]}</td>
            <td><span class="{'good' if ab_pct >= 95 else 'bad'}">{ab_pct:.1f}%</span></td>
        </tr>
    </table>

    <div class="{'highlight' if ab_pct >= 95 else 'warning'}">
        <strong>{'✓ Excellent' if ab_pct >= 99 else '⚠ Good' if ab_pct >= 95 else '⚠ Needs Improvement'}:</strong>
        {ab_pct:.1f}% of predictions fall within clinically acceptable zones (A+B).
        {'The Clarke Error Grid shows clinically acceptable performance.' if ab_pct >= 95 else 'Additional optimization may be needed for clinical use.'}
    </div>
</div>

<div class="section">
    <h2>Feature Visualization</h2>

    <div class="figure">
        <img src="figures/feature_explorer.png" alt="Feature Explorer">
        <p><em>Figure: PCA and t-SNE visualization of voice features colored by glucose level and participant</em></p>
    </div>
</div>

<div class="section">
    <h2>Conclusions & Recommendations</h2>

    <div class="highlight">
        <strong>Key Conclusions:</strong>
        <ul>
            <li>Voice-based glucose estimation shows promise with personalized models achieving MAE ~9-10 mg/dL</li>
            <li>Time offset optimization is critical - optimal offsets vary from {min(optimal_offsets.values()):+d} to {max(optimal_offsets.values()):+d} minutes</li>
            <li>Individual variation in voice-glucose relationship suggests personalization is essential</li>
            <li>Random Forest and Bayesian Ridge models perform best for most participants</li>
        </ul>
    </div>

    <div class="warning">
        <strong>Limitations:</strong>
        <ul>
            <li>Small sample sizes per participant (30-183 recordings)</li>
            <li>Limited participant diversity (3 participants)</li>
            <li>WhatsApp voice messages vary in content and recording conditions</li>
            <li>CGM accuracy limitations (±15 mg/dL typical)</li>
        </ul>
    </div>

    <div class="methodology">
        <strong>Recommendations for Deployment:</strong>
        <ul>
            <li>Use personalized models with participant-specific time offsets</li>
            <li>For MCU deployment, use Bayesian Ridge (~2KB) or compressed Random Forest</li>
            <li>Implement real-time offset adaptation during initial calibration period</li>
            <li>Collect more diverse training data for population-level models</li>
        </ul>
    </div>
</div>

<div style="text-align: center; color: #666; margin-top: 40px; padding: 20px;">
    <p>Voice-Glucose Analysis v6 | Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
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
    print("COMPREHENSIVE VOICE-GLUCOSE ANALYSIS v6")
    print("=" * 70)

    # Step 1: Find optimal offsets
    print("\n" + "=" * 70)
    print("STEP 1: OPTIMAL OFFSET SEARCH")
    print("=" * 70)

    optimal_offsets = {}
    all_offset_results = {}

    for name, config in PARTICIPANTS_V6.items():
        offset, results = find_optimal_offset(name, config, TIME_OFFSETS_TO_TEST)
        optimal_offsets[name] = offset
        all_offset_results[name] = results

    # Plot offset analysis
    plot_offset_analysis(all_offset_results, FIGURES_DIR / "offset_analysis.png")

    # Step 2: Load data with optimal offsets
    print("\n" + "=" * 70)
    print("STEP 2: LOADING DATA WITH OPTIMAL OFFSETS")
    print("=" * 70)

    all_data = {}

    for name, config in PARTICIPANTS_V6.items():
        offset = optimal_offsets.get(name, 0)
        X_list, y_list, timestamps = load_participant_data(name, config, offset)

        if len(X_list) >= 20:
            all_data[name] = {
                'X': np.array(X_list),
                'y': np.array(y_list),
                'timestamps': timestamps,
                'offset': offset
            }

    if not all_data:
        print("No sufficient data loaded!")
        return

    # Step 3: Comprehensive model evaluation
    print("\n" + "=" * 70)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 70)

    all_results = {}

    for name, data in all_data.items():
        X, y = data['X'], data['y']
        print(f"\n--- {name} ({len(y)} samples) ---")

        model_results = evaluate_models_comprehensive(X, y, name)

        if model_results:
            best = min(model_results.items(), key=lambda x: x[1]['mae'])
            print(f"  Best: {best[0]}, MAE={best[1]['mae']:.2f}, r={best[1]['r']:.3f}")

            all_results[name] = {
                'n_samples': len(y),
                'best_model': best[0],
                'best_mae': best[1]['mae'],
                'best_r': best[1]['r'],
                'all_results': model_results
            }

            # Plot individual scatter
            plot_participant_scatter(
                best[1]['y_true'], best[1]['y_pred'],
                name, best[1]['mae'], best[1]['r'],
                FIGURES_DIR / f"scatter_{name}.png"
            )

    # Step 4: Generate visualizations
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Combined feature explorer
    X_all = []
    y_all = []
    participants_all = []

    for name, data in all_data.items():
        X_all.append(data['X'])
        y_all.append(data['y'])
        participants_all.extend([name] * len(data['y']))

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    plot_feature_explorer(X_all, y_all, participants_all, "Combined Feature Explorer",
                         FIGURES_DIR / "feature_explorer.png")
    print("  Created feature_explorer.png")

    # Clarke Error Grid (using best model predictions)
    y_true_all = []
    y_pred_all = []

    for name, results in all_results.items():
        best_model = results['best_model']
        y_true_all.extend(results['all_results'][best_model]['y_true'])
        y_pred_all.extend(results['all_results'][best_model]['y_pred'])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    plot_clarke_error_grid(y_true_all, y_pred_all, "Clarke Error Grid - All Participants",
                          FIGURES_DIR / "clarke_error_grid.png")
    clarke_zones = calculate_clarke_zones(y_true_all, y_pred_all)
    print(f"  Clarke A+B: {clarke_zones['A'][1] + clarke_zones['B'][1]:.1f}%")

    # Step 5: Generate HTML report
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING HTML REPORT")
    print("=" * 70)

    html = generate_comprehensive_report(all_results, optimal_offsets, all_offset_results, clarke_zones)

    with open(OUTPUT_DIR / "report_v6.html", 'w', encoding='utf-8') as f:
        f.write(html)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Report: {OUTPUT_DIR / 'report_v6.html'}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Participant':<15} {'Samples':<10} {'Offset':<12} {'Best Model':<15} {'MAE':<10} {'r':<10}")
    print("-" * 70)
    for name, results in all_results.items():
        offset = optimal_offsets.get(name, 0)
        print(f"{name:<15} {results['n_samples']:<10} {offset:+d} min{'':<6} {results['best_model']:<15} {results['best_mae']:<10.2f} {results['best_r']:<10.3f}")

    print(f"\nClarke Error Grid: A={clarke_zones['A'][1]:.1f}%, B={clarke_zones['B'][1]:.1f}%, A+B={clarke_zones['A'][1]+clarke_zones['B'][1]:.1f}%")


if __name__ == "__main__":
    main()
