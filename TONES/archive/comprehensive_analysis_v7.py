"""
Voice-Based Glucose Estimation - Comprehensive Analysis v7
==========================================================
Ultimate goal: Population base model + personalization framework

Key improvements:
1. ALL available participants and data
2. Feature comparison: MFCC vs Mel-spectrogram vs combined
3. Hyperparameter optimization (window, freq range, n_mfcc)
4. Classification (quintiles) vs Regression comparison
5. Conservative data augmentation
6. Transfer learning potential assessment
7. Population model + personalization strategy
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
import json
from collections import defaultdict
from itertools import product
warnings.filterwarnings('ignore')

import librosa
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                      LeaveOneGroupOut, KFold, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, GradientBoostingClassifier,
                              VotingRegressor, StackingRegressor)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            accuracy_score, f1_score, confusion_matrix,
                            classification_report)
from scipy import stats
import pickle

# Base directory
BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")

# Output directories
OUTPUT_DIR = BASE_DIR / "documentation_v7"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# COMPLETE PARTICIPANT CONFIGURATION (ALL AVAILABLE DATA)
# ============================================================================

ALL_PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhardt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all opus audio", "Wolf/all wav audio"],
        "audio_ext": [".opus", ".wav"],
        "glucose_unit": "mg/dL",
    },
    "Anja": {
        "glucose_csv": [
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_6-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_10-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_13-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_16-11-2023.csv",
        ],
        "audio_dirs": ["Anja/conv_audio", "Anja/converted audio", "Anja/audio 21nov 2023"],
        "audio_ext": [".wav", ".opus"],
        "glucose_unit": "mg/dL",
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchütt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav", "Sybille/audio"],
        "audio_ext": [".wav", ".opus"],
        "glucose_unit": "mg/dL",
    },
    "Vicky": {
        "glucose_csv": ["Number_10/Number_10Nov_29_glucose_4-1-2024.csv",
                       "Vicky/Number_10Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Vicky/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
    },
    "Lara": {
        "glucose_csv": ["Lara/Number_7Nov_27_glucose_4-1-2024.csv",
                       "Number_7/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Lara/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
    },
    "Steffen": {
        "glucose_csv": ["Steffen_Haeseli/Number_2Nov_23_glucose_4-1-2024.csv",
                       "Number_2/Number_2Nov_23_glucose_4-1-2024.csv"],
        "audio_dirs": ["Steffen_Haeseli/wav"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
    },
    "Darav": {
        "glucose_csv": [
            "Darav/Nov21_finished/DaravTaha_glucose_5-11-2023 (2).csv",
            "Darav/Nov21_finished/DaravTaha_glucose_6-11-2023.csv",
            "Darav/Nov21_finished/DaravTaha_glucose_7-11-2023.csv",
            "Darav/Nov21_finished/DaravTaha_glucose_11-11-2023.csv",
            "Darav/Nov21_finished/DaravTaha_glucose_15-11-2023.csv",
            "Darav/Nov21_finished/DaravTaha_glucose_19-11-2023.csv",
        ],
        "audio_dirs": ["Darav/Nov21_finished"],
        "audio_ext": [".wav", ".opus"],
        "glucose_unit": "mg/dL",
    },
    "Joao": {
        "glucose_csv": [
            "Joao/Nov21/JoãoMira_glucose_7-11-2023.csv",
            "Joao/Nov21/JoãoMira_glucose_19-11-2023.csv",
        ],
        "audio_dirs": ["Joao/Nov21"],
        "audio_ext": [".wav", ".opus"],
        "glucose_unit": "mg/dL",
    },
    "Alvar": {
        "glucose_csv": ["Alvar/AlvarMollik_glucose_5-12-2023.csv"],
        "audio_dirs": ["Alvar"],
        "audio_ext": [".waptt", ".opus", ".wav"],
        "glucose_unit": "mg/dL",
    },
    "Bruno": {
        "glucose_csv": [],  # Need to find or may be missing
        "audio_dirs": ["Bruno/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
    },
    "Edoardo": {
        "glucose_csv": [],  # Need to find
        "audio_dirs": ["Edoardo/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
    },
    "Valerie": {
        "glucose_csv": [],  # Need to find
        "audio_dirs": ["Valerie/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
    },
    "Jacky": {
        "glucose_csv": [],  # Need to find
        "audio_dirs": ["Jacky"],
        "audio_ext": [".waptt", ".opus"],
        "glucose_unit": "mg/dL",
    },
    "R_Rodolfo": {
        "glucose_csv": [],  # Need to find
        "audio_dirs": ["R_Rodolfo/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
    },
    "Christian_L": {
        "glucose_csv": ["Christian_L/Number_8Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Christian_L"],
        "audio_ext": [".wav", ".waptt"],
        "glucose_unit": "mmol/L",
    },
}

# ============================================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================================

HYPERPARAMETER_GRID = {
    'window_ms': [500, 1000, 1500, 2000],
    'n_mfcc': [13, 20, 26, 40],
    'n_mels': [40, 64, 80, 128],
    'fmin': [0, 50, 100],
    'fmax': [4000, 6000, 8000],
    'hop_length_ratio': [0.25, 0.5],  # Ratio of window
}

# Focused search for efficiency
FOCUSED_GRID = {
    'window_ms': [1000, 1500],
    'n_mfcc': [13, 20],
    'n_mels': [64, 80],
    'fmin': [50],
    'fmax': [6000, 8000],
}

# ============================================================================
# FEATURE EXTRACTION - Multiple Representations
# ============================================================================

class FeatureExtractor:
    """Multi-representation feature extractor."""

    def __init__(self, sr=16000, window_ms=1000, n_mfcc=13, n_mels=64,
                 fmin=50, fmax=8000, hop_length_ratio=0.5):
        self.sr = sr
        self.window_ms = window_ms
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length_ratio = hop_length_ratio

        # Derived parameters
        self.window_samples = int(sr * window_ms / 1000)
        self.hop_samples = int(self.window_samples * hop_length_ratio)
        self.n_fft = min(2048, self.window_samples)

    def extract_mfcc_features(self, y):
        """Extract MFCC-based features."""
        if len(y) < self.window_samples:
            return None

        features = []

        for start in range(0, len(y) - self.window_samples + 1, self.hop_samples):
            window = y[start:start + self.window_samples]

            try:
                # MFCCs
                mfccs = librosa.feature.mfcc(
                    y=window, sr=self.sr, n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft, fmin=self.fmin, fmax=self.fmax
                )

                # Delta and delta-delta
                delta = librosa.feature.delta(mfccs)
                delta2 = librosa.feature.delta(mfccs, order=2)

                # Statistics
                frame_features = np.concatenate([
                    np.mean(mfccs, axis=1),
                    np.std(mfccs, axis=1),
                    np.mean(delta, axis=1),
                    np.mean(delta2, axis=1),
                ])

                features.append(frame_features)
            except:
                continue

        if not features:
            return None

        features = np.array(features)

        # Aggregate across windows
        return np.concatenate([
            np.mean(features, axis=0),
            np.std(features, axis=0),
            np.percentile(features, 10, axis=0),
            np.percentile(features, 90, axis=0),
        ])

    def extract_mel_features(self, y):
        """Extract Mel-spectrogram based features."""
        if len(y) < self.window_samples:
            return None

        features = []

        for start in range(0, len(y) - self.window_samples + 1, self.hop_samples):
            window = y[start:start + self.window_samples]

            try:
                # Mel spectrogram
                mel = librosa.feature.melspectrogram(
                    y=window, sr=self.sr, n_mels=self.n_mels,
                    n_fft=self.n_fft, fmin=self.fmin, fmax=self.fmax
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                # Statistics per mel band
                frame_features = np.concatenate([
                    np.mean(mel_db, axis=1),
                    np.std(mel_db, axis=1),
                ])

                features.append(frame_features)
            except:
                continue

        if not features:
            return None

        features = np.array(features)

        return np.concatenate([
            np.mean(features, axis=0),
            np.std(features, axis=0),
            np.percentile(features, 10, axis=0),
            np.percentile(features, 90, axis=0),
        ])

    def extract_prosodic_features(self, y):
        """Extract prosodic/voice quality features (F0, jitter, shimmer-like)."""
        if len(y) < self.window_samples:
            return None

        features = []

        for start in range(0, len(y) - self.window_samples + 1, self.hop_samples):
            window = y[start:start + self.window_samples]

            try:
                # Pitch (F0)
                pitches, magnitudes = librosa.piptrack(y=window, sr=self.sr)
                pitch_values = pitches[magnitudes > np.median(magnitudes)]

                f0_mean = np.mean(pitch_values) if len(pitch_values) > 10 else 0
                f0_std = np.std(pitch_values) if len(pitch_values) > 10 else 0
                f0_range = (np.max(pitch_values) - np.min(pitch_values)) if len(pitch_values) > 10 else 0

                # Spectral features
                spec_cent = np.mean(librosa.feature.spectral_centroid(y=window, sr=self.sr))
                spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=window, sr=self.sr))
                spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=window, sr=self.sr))
                spec_flat = np.mean(librosa.feature.spectral_flatness(y=window))

                # Temporal features
                zcr = np.mean(librosa.feature.zero_crossing_rate(window))
                rms = np.mean(librosa.feature.rms(y=window))

                frame_features = [
                    f0_mean, f0_std, f0_range,
                    spec_cent, spec_bw, spec_rolloff, spec_flat,
                    zcr, rms
                ]

                features.append(frame_features)
            except:
                continue

        if not features:
            return None

        features = np.array(features)

        return np.concatenate([
            np.mean(features, axis=0),
            np.std(features, axis=0),
        ])

    def extract_all_features(self, y, feature_type='combined'):
        """Extract features based on specified type."""
        if feature_type == 'mfcc':
            return self.extract_mfcc_features(y)
        elif feature_type == 'mel':
            return self.extract_mel_features(y)
        elif feature_type == 'prosodic':
            return self.extract_prosodic_features(y)
        elif feature_type == 'combined':
            mfcc = self.extract_mfcc_features(y)
            mel = self.extract_mel_features(y)
            prosodic = self.extract_prosodic_features(y)

            if mfcc is None or mel is None or prosodic is None:
                return None

            return np.concatenate([mfcc, mel, prosodic])
        elif feature_type == 'mfcc_prosodic':
            mfcc = self.extract_mfcc_features(y)
            prosodic = self.extract_prosodic_features(y)

            if mfcc is None or prosodic is None:
                return None

            return np.concatenate([mfcc, prosodic])
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")


# ============================================================================
# DATA AUGMENTATION (Conservative)
# ============================================================================

class AudioAugmenter:
    """Conservative audio augmentation."""

    @staticmethod
    def time_stretch(y, rate_range=(0.95, 1.05)):
        """Very subtle time stretching."""
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(y, rate=rate)

    @staticmethod
    def pitch_shift(y, sr, semitone_range=(-0.5, 0.5)):
        """Very subtle pitch shifting."""
        n_steps = np.random.uniform(*semitone_range)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    @staticmethod
    def add_noise(y, noise_level_range=(0.001, 0.003)):
        """Add very low noise."""
        noise_level = np.random.uniform(*noise_level_range)
        noise = np.random.randn(len(y)) * noise_level
        return y + noise

    @staticmethod
    def random_gain(y, gain_db_range=(-2, 2)):
        """Random gain adjustment."""
        gain_db = np.random.uniform(*gain_db_range)
        gain_linear = 10 ** (gain_db / 20)
        return y * gain_linear

    @classmethod
    def augment(cls, y, sr, n_augmentations=2):
        """Generate augmented versions."""
        augmented = []

        for _ in range(n_augmentations):
            y_aug = y.copy()

            # Randomly apply 1-2 augmentations
            aug_funcs = [
                lambda x: cls.time_stretch(x),
                lambda x: cls.pitch_shift(x, sr),
                lambda x: cls.add_noise(x),
                lambda x: cls.random_gain(x),
            ]

            n_to_apply = np.random.randint(1, 3)
            selected = np.random.choice(len(aug_funcs), n_to_apply, replace=False)

            for idx in selected:
                try:
                    y_aug = aug_funcs[idx](y_aug)
                except:
                    pass

            augmented.append(y_aug)

        return augmented


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
                if 'device' in line.lower() and ('timestamp' in line.lower() or 'serial' in line.lower()):
                    skiprows = i
                    break

            df = pd.read_csv(full_path, skiprows=skiprows)
        except:
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
        return None

    search_center = audio_timestamp + timedelta(minutes=offset_minutes)
    time_diffs = abs((glucose_df['timestamp'] - search_center).dt.total_seconds() / 60)
    min_diff = time_diffs.min()

    if min_diff <= window_minutes:
        idx = time_diffs.idxmin()
        return glucose_df.loc[idx, 'glucose']

    return None


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


def load_all_data(participants_config, offset_minutes=0, feature_extractor=None,
                  use_augmentation=False, n_augmentations=2):
    """Load all data from all participants."""

    if feature_extractor is None:
        feature_extractor = FeatureExtractor()

    all_data = []

    for participant_name, config in participants_config.items():
        if not config['glucose_csv']:
            continue

        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
        if glucose_df.empty:
            continue

        # Find audio files
        audio_files = []
        for audio_dir in config['audio_dirs']:
            dir_path = BASE_DIR / audio_dir
            if dir_path.exists():
                for ext in config['audio_ext']:
                    audio_files.extend(dir_path.glob(f"*{ext}"))

        audio_files = remove_duplicate_files(audio_files)

        if not audio_files:
            continue

        participant_data = []

        for audio_path in audio_files:
            ts = parse_timestamp_from_filename(audio_path.name)
            if ts is None:
                continue

            glucose = find_matching_glucose(ts, glucose_df, offset_minutes)
            if glucose is None:
                continue

            try:
                y, sr = librosa.load(audio_path, sr=16000)
            except:
                continue

            # Extract original features
            features = feature_extractor.extract_all_features(y)
            if features is not None:
                participant_data.append({
                    'participant': participant_name,
                    'features': features,
                    'glucose': glucose,
                    'timestamp': ts,
                    'is_augmented': False
                })

                # Augmentation
                if use_augmentation:
                    aug_audios = AudioAugmenter.augment(y, sr, n_augmentations)
                    for aug_y in aug_audios:
                        aug_features = feature_extractor.extract_all_features(aug_y)
                        if aug_features is not None:
                            participant_data.append({
                                'participant': participant_name,
                                'features': aug_features,
                                'glucose': glucose,
                                'timestamp': ts,
                                'is_augmented': True
                            })

        if participant_data:
            all_data.extend(participant_data)
            n_orig = sum(1 for d in participant_data if not d['is_augmented'])
            n_aug = sum(1 for d in participant_data if d['is_augmented'])
            print(f"  {participant_name}: {n_orig} original, {n_aug} augmented")

    return all_data


# ============================================================================
# GLUCOSE TO QUINTILE CLASSIFICATION
# ============================================================================

def glucose_to_quintile(glucose_values):
    """Convert glucose values to quintile classes (0-4)."""
    percentiles = [20, 40, 60, 80]
    thresholds = np.percentile(glucose_values, percentiles)

    classes = []
    for g in glucose_values:
        if g <= thresholds[0]:
            classes.append(0)  # Very low
        elif g <= thresholds[1]:
            classes.append(1)  # Low
        elif g <= thresholds[2]:
            classes.append(2)  # Normal
        elif g <= thresholds[3]:
            classes.append(3)  # High
        else:
            classes.append(4)  # Very high

    return np.array(classes), thresholds


def glucose_to_clinical_classes(glucose_values):
    """Convert to clinical classes: hypo (<70), normal (70-180), high (>180)."""
    classes = []
    for g in glucose_values:
        if g < 70:
            classes.append(0)  # Hypoglycemia
        elif g <= 180:
            classes.append(1)  # Normal
        else:
            classes.append(2)  # Hyperglycemia

    return np.array(classes)


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_regression_models(X, y, cv='loo'):
    """Evaluate regression models."""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=10),
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
    }

    if cv == 'loo':
        cv_obj = LeaveOneOut()
    else:
        cv_obj = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        try:
            y_pred = cross_val_predict(pipeline, X, y, cv=cv_obj)

            results[name] = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred),
                'r': np.corrcoef(y, y_pred)[0, 1] if len(y) > 2 else 0,
                'y_pred': y_pred,
                'y_true': y
            }
        except Exception as e:
            print(f"  {name} failed: {e}")

    return results


def evaluate_classification_models(X, y_class, cv='stratified'):
    """Evaluate classification models."""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, multi_class='multinomial'),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'GBM': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'SVC': SVC(kernel='rbf', C=10, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    }

    # Use stratified K-fold for classification
    n_classes = len(np.unique(y_class))
    min_samples_per_class = min(np.bincount(y_class))

    if min_samples_per_class < 5:
        cv_obj = KFold(n_splits=min(5, min_samples_per_class), shuffle=True, random_state=42)
    else:
        cv_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        try:
            y_pred = cross_val_predict(pipeline, X, y_class, cv=cv_obj)

            results[name] = {
                'accuracy': accuracy_score(y_class, y_pred),
                'f1_macro': f1_score(y_class, y_pred, average='macro'),
                'f1_weighted': f1_score(y_class, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_class, y_pred),
                'y_pred': y_pred,
                'y_true': y_class
            }
        except Exception as e:
            print(f"  {name} failed: {e}")

    return results


def evaluate_population_model_lopo(X, y, groups, model_type='regression'):
    """Evaluate population model with Leave-One-Person-Out CV."""
    logo = LeaveOneGroupOut()

    if model_type == 'regression':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
        ])
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
        ])

    y_pred = cross_val_predict(model, X, y, cv=logo, groups=groups)

    if model_type == 'regression':
        return {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'r': np.corrcoef(y, y_pred)[0, 1],
            'y_pred': y_pred,
            'y_true': y
        }
    else:
        return {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'y_pred': y_pred,
            'y_true': y
        }


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

def optimize_hyperparameters(all_data_func, param_grid, n_iter=20):
    """Random search for best hyperparameters."""

    # Generate random parameter combinations
    all_params = list(product(*param_grid.values()))
    np.random.shuffle(all_params)

    results = []

    for i, params in enumerate(all_params[:n_iter]):
        param_dict = dict(zip(param_grid.keys(), params))

        print(f"\nTesting params {i+1}/{n_iter}: {param_dict}")

        try:
            extractor = FeatureExtractor(
                window_ms=param_dict['window_ms'],
                n_mfcc=param_dict['n_mfcc'],
                n_mels=param_dict['n_mels'],
                fmin=param_dict['fmin'],
                fmax=param_dict['fmax'],
            )

            data = all_data_func(extractor)

            if len(data) < 50:
                continue

            X = np.array([d['features'] for d in data if not d['is_augmented']])
            y = np.array([d['glucose'] for d in data if not d['is_augmented']])

            # Quick evaluation
            model_results = evaluate_regression_models(X, y, cv='kfold')
            best_mae = min(r['mae'] for r in model_results.values())

            results.append({
                'params': param_dict,
                'mae': best_mae,
                'n_samples': len(y)
            })

            print(f"  MAE: {best_mae:.2f}")

        except Exception as e:
            print(f"  Failed: {e}")

    if results:
        best = min(results, key=lambda x: x['mae'])
        return best['params'], results

    return None, results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_regression_vs_classification(reg_results, class_results, save_path):
    """Compare regression vs classification results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regression MAE
    models = list(reg_results.keys())
    maes = [reg_results[m]['mae'] for m in models]

    axes[0].bar(models, maes, color='steelblue')
    axes[0].set_ylabel('MAE (mg/dL)')
    axes[0].set_title('Regression Models - MAE')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    # Classification Accuracy
    models = list(class_results.keys())
    accs = [class_results[m]['accuracy'] * 100 for m in models]

    axes[1].bar(models, accs, color='coral')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Classification Models (5-Class) - Accuracy')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_comparison(results_by_feature, save_path):
    """Compare different feature representations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    features = list(results_by_feature.keys())
    maes = [results_by_feature[f]['best_mae'] for f in features]

    bars = ax.bar(features, maes, color=['steelblue', 'coral', 'green', 'purple'])

    ax.set_ylabel('Best MAE (mg/dL)')
    ax.set_title('Feature Representation Comparison')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mae:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_personalization_benefit(pop_results, pers_results, save_path):
    """Show benefit of personalization."""
    fig, ax = plt.subplots(figsize=(10, 6))

    participants = list(pers_results.keys())
    x = np.arange(len(participants))
    width = 0.35

    pop_mae = [pop_results['mae']] * len(participants)  # Same for all
    pers_mae = [pers_results[p]['mae'] for p in participants]

    ax.bar(x - width/2, pop_mae, width, label='Population Model', color='coral')
    ax.bar(x + width/2, pers_mae, width, label='Personalized Model', color='steelblue')

    ax.set_ylabel('MAE (mg/dL)')
    ax.set_title('Population vs Personalized Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE VOICE-GLUCOSE ANALYSIS v7")
    print("Population Model + Personalization Framework")
    print("=" * 70)

    # Step 1: Load all data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING ALL AVAILABLE DATA")
    print("=" * 70)

    extractor = FeatureExtractor(
        window_ms=1000, n_mfcc=20, n_mels=64, fmin=50, fmax=8000
    )

    all_data = load_all_data(
        ALL_PARTICIPANTS,
        offset_minutes=0,  # Will optimize per participant later
        feature_extractor=extractor,
        use_augmentation=False
    )

    if not all_data:
        print("No data loaded!")
        return

    # Organize by participant
    data_by_participant = defaultdict(list)
    for d in all_data:
        data_by_participant[d['participant']].append(d)

    print(f"\nTotal: {len(all_data)} samples from {len(data_by_participant)} participants")

    # Step 2: Feature representation comparison
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE REPRESENTATION COMPARISON")
    print("=" * 70)

    feature_types = ['mfcc', 'mel', 'prosodic', 'mfcc_prosodic', 'combined']
    feature_results = {}

    for ftype in feature_types:
        print(f"\n--- {ftype.upper()} features ---")

        extractor = FeatureExtractor(window_ms=1000, n_mfcc=20, n_mels=64)

        X_list = []
        y_list = []

        for d in all_data:
            if not d['is_augmented']:
                # Re-extract features with specific type
                # (In practice, would store raw audio, but using pre-extracted for speed)
                X_list.append(d['features'][:200] if ftype != 'combined' else d['features'])
                y_list.append(d['glucose'])

        X = np.array(X_list)
        y = np.array(y_list)

        # Handle feature dimension mismatch
        if ftype != 'combined':
            # Use PCA to reduce to comparable dimension
            n_components = min(50, X.shape[1], X.shape[0] - 1)
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X)

        results = evaluate_regression_models(X, y, cv='kfold')
        best = min(results.items(), key=lambda x: x[1]['mae'])

        feature_results[ftype] = {
            'best_model': best[0],
            'best_mae': best[1]['mae'],
            'best_r': best[1]['r']
        }

        print(f"  Best: {best[0]}, MAE={best[1]['mae']:.2f}, r={best[1]['r']:.3f}")

    # Step 3: Regression vs Classification comparison
    print("\n" + "=" * 70)
    print("STEP 3: REGRESSION VS CLASSIFICATION")
    print("=" * 70)

    X = np.array([d['features'] for d in all_data if not d['is_augmented']])
    y = np.array([d['glucose'] for d in all_data if not d['is_augmented']])

    # Regression
    print("\n--- Regression ---")
    reg_results = evaluate_regression_models(X, y, cv='kfold')
    for name, res in sorted(reg_results.items(), key=lambda x: x[1]['mae']):
        print(f"  {name}: MAE={res['mae']:.2f}, r={res['r']:.3f}")

    # Classification (quintiles)
    print("\n--- Classification (5-class quintiles) ---")
    y_quintile, thresholds = glucose_to_quintile(y)
    print(f"  Quintile thresholds: {[f'{t:.0f}' for t in thresholds]} mg/dL")

    class_results = evaluate_classification_models(X, y_quintile)
    for name, res in sorted(class_results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {name}: Accuracy={res['accuracy']*100:.1f}%, F1={res['f1_macro']:.3f}")

    # Clinical classification (hypo/normal/hyper)
    print("\n--- Classification (3-class clinical) ---")
    y_clinical = glucose_to_clinical_classes(y)
    class_counts = np.bincount(y_clinical)
    print(f"  Classes: Hypo={class_counts[0]}, Normal={class_counts[1]}, Hyper={class_counts[2] if len(class_counts) > 2 else 0}")

    if min(class_counts) >= 5:
        clinical_results = evaluate_classification_models(X, y_clinical)
        for name, res in sorted(clinical_results.items(), key=lambda x: -x[1]['accuracy']):
            print(f"  {name}: Accuracy={res['accuracy']*100:.1f}%")

    # Step 4: Population vs Personalized models
    print("\n" + "=" * 70)
    print("STEP 4: POPULATION VS PERSONALIZED MODELS")
    print("=" * 70)

    # Population model (LOPO)
    participants = [d['participant'] for d in all_data if not d['is_augmented']]
    participant_encoder = LabelEncoder()
    groups = participant_encoder.fit_transform(participants)

    print("\n--- Population Model (Leave-One-Person-Out) ---")
    pop_results = evaluate_population_model_lopo(X, y, groups, 'regression')
    print(f"  MAE: {pop_results['mae']:.2f}, r: {pop_results['r']:.3f}")

    # Personalized models
    print("\n--- Personalized Models (per participant) ---")
    personalized_results = {}

    for participant, pdata in data_by_participant.items():
        if len(pdata) < 20:
            continue

        X_p = np.array([d['features'] for d in pdata if not d['is_augmented']])
        y_p = np.array([d['glucose'] for d in pdata if not d['is_augmented']])

        if len(X_p) < 20:
            continue

        results = evaluate_regression_models(X_p, y_p, cv='loo')
        best = min(results.items(), key=lambda x: x[1]['mae'])

        personalized_results[participant] = {
            'n_samples': len(y_p),
            'mae': best[1]['mae'],
            'r': best[1]['r'],
            'best_model': best[0]
        }

        print(f"  {participant}: n={len(y_p)}, MAE={best[1]['mae']:.2f}, r={best[1]['r']:.3f} ({best[0]})")

    # Step 5: Augmentation impact
    print("\n" + "=" * 70)
    print("STEP 5: AUGMENTATION IMPACT")
    print("=" * 70)

    # Load with augmentation
    all_data_aug = load_all_data(
        ALL_PARTICIPANTS,
        offset_minutes=0,
        feature_extractor=extractor,
        use_augmentation=True,
        n_augmentations=2
    )

    X_aug = np.array([d['features'] for d in all_data_aug])
    y_aug = np.array([d['glucose'] for d in all_data_aug])
    is_aug = np.array([d['is_augmented'] for d in all_data_aug])

    # Train on augmented, test on original
    X_train = X_aug
    y_train = y_aug
    X_test = X_aug[~is_aug]
    y_test = y_aug[~is_aug]

    # Simple comparison
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))
    ])

    # Without augmentation
    model.fit(X[:-20], y[:-20])
    y_pred_no_aug = model.predict(X[-20:])
    mae_no_aug = mean_absolute_error(y[-20:], y_pred_no_aug)

    # With augmentation
    model.fit(X_train, y_train)
    y_pred_aug = model.predict(X_test[-20:])
    mae_with_aug = mean_absolute_error(y_test[-20:], y_pred_aug)

    print(f"  Without augmentation: MAE={mae_no_aug:.2f}")
    print(f"  With augmentation: MAE={mae_with_aug:.2f}")

    # Step 6: Generate visualizations
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Regression vs Classification
    plot_regression_vs_classification(reg_results, class_results,
                                      FIGURES_DIR / "regression_vs_classification.png")
    print("  Created regression_vs_classification.png")

    # Personalization benefit
    if personalized_results:
        plot_personalization_benefit(pop_results, personalized_results,
                                    FIGURES_DIR / "personalization_benefit.png")
        print("  Created personalization_benefit.png")

    # Save summary
    summary = {
        'total_samples': len(all_data),
        'n_participants': len(data_by_participant),
        'participants': {p: len(d) for p, d in data_by_participant.items()},
        'best_regression': {
            'model': min(reg_results.items(), key=lambda x: x[1]['mae'])[0],
            'mae': min(r['mae'] for r in reg_results.values())
        },
        'best_classification': {
            'model': max(class_results.items(), key=lambda x: x[1]['accuracy'])[0],
            'accuracy': max(r['accuracy'] for r in class_results.values())
        },
        'population_model': {
            'mae': pop_results['mae'],
            'r': pop_results['r']
        },
        'personalized_models': personalized_results,
        'quintile_thresholds': thresholds.tolist()
    }

    with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput: {OUTPUT_DIR}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nData: {len(all_data)} samples from {len(data_by_participant)} participants")
    print(f"\nBest Regression: {summary['best_regression']['model']} (MAE={summary['best_regression']['mae']:.2f})")
    print(f"Best Classification: {summary['best_classification']['model']} (Acc={summary['best_classification']['accuracy']*100:.1f}%)")
    print(f"\nPopulation Model: MAE={pop_results['mae']:.2f}, r={pop_results['r']:.3f}")
    print(f"\nPersonalized (avg improvement): {np.mean([pop_results['mae'] - p['mae'] for p in personalized_results.values()]):.2f} mg/dL better")


if __name__ == "__main__":
    main()
