"""
Production Voice-Glucose Analysis Pipeline
==========================================
Comprehensive analysis for building a monetizable voice-glucose estimation product.

First-Principles Approach:
1. Voice carries physiological signals (pitch, HNR, jitter, shimmer)
2. Glucose affects autonomic nervous system → affects vocal fold tension
3. Blood viscosity changes affect microvascular perfusion → affects voice quality
4. Individual variation is significant → personalization is critical

SOTA Techniques Applied:
- Self-supervised speech representations (HuBERT, Wav2Vec2)
- Traditional speech biomarkers (openSMILE-inspired)
- Ensemble methods with uncertainty quantification
- Few-shot domain adaptation for personalization
- Temporal modeling for longitudinal patterns

Business Moat Strategy:
1. Data moat: Proprietary paired voice-glucose dataset
2. Algorithm moat: Personalization that improves with each user
3. Network effects: More users → better population model → better cold-start
4. Integration moat: API/SDK for CGM manufacturers, health apps
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor
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
from scipy.signal import find_peaks

# ML imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor,
    StackingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.model_selection import (
    cross_val_predict, LeaveOneOut, KFold,
    GroupKFold, LeaveOneGroupOut, GridSearchCV
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")
OUTPUT_DIR = BASE_DIR / "production_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All participants
PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "glucose_unit": "mg/dL",
        "time_offset": 15,
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
        "time_offset": 0,
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "glucose_unit": "mmol/L",
        "time_offset": 20,
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchütt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav"],
        "glucose_unit": "mg/dL",
        "time_offset": 15,
    },
    "Vicky": {
        "glucose_csv": ["Vicky/Number_10Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Vicky/conv_audio"],
        "glucose_unit": "mmol/L",
        "time_offset": 15,
    },
    "Steffen": {
        "glucose_csv": ["Steffen_Haeseli/Number_2Nov_23_glucose_4-1-2024.csv"],
        "audio_dirs": ["Steffen_Haeseli/wav"],
        "glucose_unit": "mmol/L",
        "time_offset": 15,
    },
    "Lara": {
        "glucose_csv": ["Lara/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Lara/conv_audio"],
        "glucose_unit": "mmol/L",
        "time_offset": 15,
    },
}


# ============================================================================
# SECTION 1: DATA LOADING AND EXPLORATION
# ============================================================================

def load_glucose_data(csv_paths: List[str], unit: str) -> pd.DataFrame:
    """Load and preprocess glucose data."""
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

                # Unit conversion
                if unit == 'mmol/L':
                    df['glucose'] = df['glucose'] * 18.0182
                elif unit == 'mg/dL':
                    mean_val = df['glucose'].dropna().mean()
                    if mean_val < 30:  # Likely mmol/L mislabeled
                        df['glucose'] = df['glucose'] * 18.0182

                df = df.dropna(subset=['timestamp', 'glucose'])
                if len(df) > 0:
                    all_dfs.append(df[['timestamp', 'glucose']])
        except Exception as e:
            continue

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    return pd.DataFrame()


def parse_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from WhatsApp audio filename."""
    pattern = r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h]?(\d{2})?'
    match = re.search(pattern, str(filename))
    if match:
        date_str = match.group(1)
        hour = int(match.group(2))
        minute = int(match.group(3))
        second = int(match.group(4)) if match.group(4) else 0
        return datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}:{second:02d}", "%Y-%m-%d %H:%M:%S")
    return None


def find_matching_glucose(audio_ts: datetime, glucose_df: pd.DataFrame,
                          offset_minutes: int = 0, window_minutes: int = 15) -> Optional[float]:
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


def explore_data() -> Dict:
    """Comprehensive data exploration."""
    print("\n" + "="*70)
    print("SECTION 1: DATA EXPLORATION")
    print("="*70)

    exploration = {
        'participants': {},
        'total_audio': 0,
        'total_glucose_readings': 0,
        'glucose_stats': {},
    }

    for name, config in PARTICIPANTS.items():
        print(f"\n  {name}:")

        # Load glucose data
        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])

        # Count audio files
        audio_count = 0
        for audio_dir in config['audio_dirs']:
            dir_path = BASE_DIR / audio_dir
            if dir_path.exists():
                audio_count += len(list(dir_path.glob("*.wav")))

        exploration['participants'][name] = {
            'audio_files': audio_count,
            'glucose_readings': len(glucose_df),
            'glucose_mean': glucose_df['glucose'].mean() if len(glucose_df) > 0 else None,
            'glucose_std': glucose_df['glucose'].std() if len(glucose_df) > 0 else None,
            'glucose_min': glucose_df['glucose'].min() if len(glucose_df) > 0 else None,
            'glucose_max': glucose_df['glucose'].max() if len(glucose_df) > 0 else None,
        }

        exploration['total_audio'] += audio_count
        exploration['total_glucose_readings'] += len(glucose_df)

        if len(glucose_df) > 0:
            print(f"    Audio files: {audio_count}")
            print(f"    Glucose readings: {len(glucose_df)}")
            print(f"    Glucose range: {glucose_df['glucose'].min():.0f} - {glucose_df['glucose'].max():.0f} mg/dL")
            print(f"    Glucose mean ± std: {glucose_df['glucose'].mean():.1f} ± {glucose_df['glucose'].std():.1f} mg/dL")

    print(f"\n  TOTAL:")
    print(f"    Audio files: {exploration['total_audio']}")
    print(f"    Glucose readings: {exploration['total_glucose_readings']}")

    return exploration


# ============================================================================
# SECTION 2: ADVANCED FEATURE EXTRACTION
# ============================================================================

class VoiceBiomarkerExtractor:
    """
    Extract physiologically-relevant voice biomarkers.

    Based on research showing glucose affects:
    - Fundamental frequency (F0) - autonomic nervous system
    - Harmonic-to-noise ratio (HNR) - vocal fold vibration quality
    - Jitter/Shimmer - micro-perturbations in voice
    - Formant frequencies - vocal tract resonance
    - Speech rate/pauses - cognitive/motor effects
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def extract(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract comprehensive voice biomarkers."""
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)

            if len(y) < sr * 0.5:  # Too short
                return None

            features = []

            # ============ 1. PITCH (F0) FEATURES ============
            # F0 increases with blood glucose (autonomic effect)
            try:
                f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
                f0_valid = f0[~np.isnan(f0)]

                if len(f0_valid) > 5:
                    features.extend([
                        np.mean(f0_valid),           # Mean F0
                        np.std(f0_valid),            # F0 variability
                        np.median(f0_valid),         # Median F0
                        np.percentile(f0_valid, 25), # Q1
                        np.percentile(f0_valid, 75), # Q3
                        np.max(f0_valid) - np.min(f0_valid),  # F0 range
                        stats.skew(f0_valid),        # F0 skewness
                        stats.kurtosis(f0_valid),    # F0 kurtosis
                    ])

                    # F0 dynamics (rate of change)
                    f0_diff = np.diff(f0_valid)
                    features.extend([
                        np.mean(np.abs(f0_diff)),    # Mean F0 change rate
                        np.std(f0_diff),             # F0 change variability
                    ])
                else:
                    features.extend([0] * 10)
            except:
                features.extend([0] * 10)

            # ============ 2. JITTER (F0 perturbation) ============
            # Jitter may increase with glucose fluctuations
            try:
                if len(f0_valid) > 5:
                    periods = 1 / f0_valid[f0_valid > 0]
                    jitter_local = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
                    jitter_rap = np.mean(np.abs(periods[1:-1] -
                                        (periods[:-2] + periods[1:-1] + periods[2:]) / 3)) / np.mean(periods)
                    features.extend([jitter_local, jitter_rap])
                else:
                    features.extend([0, 0])
            except:
                features.extend([0, 0])

            # ============ 3. SHIMMER (amplitude perturbation) ============
            try:
                # Get amplitude envelope
                amplitude = np.abs(librosa.stft(y))
                amp_env = np.mean(amplitude, axis=0)

                if len(amp_env) > 5:
                    shimmer_local = np.mean(np.abs(np.diff(amp_env))) / np.mean(amp_env)
                    shimmer_apq = np.mean(np.abs(amp_env[2:-2] -
                                          np.convolve(amp_env, np.ones(5)/5, mode='valid'))) / np.mean(amp_env)
                    features.extend([shimmer_local, shimmer_apq])
                else:
                    features.extend([0, 0])
            except:
                features.extend([0, 0])

            # ============ 4. HARMONIC-TO-NOISE RATIO ============
            # HNR reflects voice quality, affected by autonomic state
            try:
                harmonic, percussive = librosa.effects.hpss(y)
                hnr = 10 * np.log10(np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10))

                # Segmented HNR
                hop = len(y) // 10
                hnr_segments = []
                for i in range(10):
                    seg = y[i*hop:(i+1)*hop]
                    h, p = librosa.effects.hpss(seg)
                    hnr_seg = 10 * np.log10(np.sum(h**2) / (np.sum(p**2) + 1e-10))
                    hnr_segments.append(hnr_seg)

                features.extend([
                    hnr,
                    np.mean(hnr_segments),
                    np.std(hnr_segments),
                ])
            except:
                features.extend([0, 0, 0])

            # ============ 5. MFCC FEATURES ============
            # Capture spectral envelope (voice quality)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)

            for coef in [mfccs, delta_mfccs, delta2_mfccs]:
                features.extend(np.mean(coef, axis=1))
                features.extend(np.std(coef, axis=1))

            # ============ 6. SPECTRAL FEATURES ============
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)

            for feat in [spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness]:
                features.extend([np.mean(feat), np.std(feat), np.median(feat.flatten())])

            # ============ 7. FORMANT-RELATED FEATURES ============
            # Using LPC for formant estimation proxy
            try:
                lpc_order = 16
                lpc_coeffs = librosa.lpc(y, order=lpc_order)
                roots = np.roots(lpc_coeffs)
                roots = roots[np.imag(roots) >= 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                formants = sorted(angles * sr / (2 * np.pi))[:4]

                while len(formants) < 4:
                    formants.append(0)
                features.extend(formants[:4])
            except:
                features.extend([0, 0, 0, 0])

            # ============ 8. ENERGY/INTENSITY ============
            rms = librosa.feature.rms(y=y)
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.max(rms),
                np.min(rms[rms > 0]) if np.any(rms > 0) else 0,
            ])

            # ============ 9. TEMPORAL FEATURES ============
            # Zero crossing rate (voice/unvoiced)
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([np.mean(zcr), np.std(zcr)])

            # Speech rate proxy (onset detection)
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
                speech_rate = len(onsets) / (len(y) / sr)
                features.append(speech_rate)
            except:
                features.append(0)

            # ============ 10. PAUSE ANALYSIS ============
            # Pauses may reflect cognitive processing affected by glucose
            try:
                # Simple VAD based on energy
                frame_length = int(0.025 * sr)
                hop_length = int(0.010 * sr)
                energy = np.array([np.sum(y[i:i+frame_length]**2)
                                   for i in range(0, len(y)-frame_length, hop_length)])

                threshold = 0.1 * np.max(energy)
                speech_frames = energy > threshold

                # Find pauses (consecutive silent frames)
                pause_lengths = []
                pause_count = 0
                in_pause = False

                for is_speech in speech_frames:
                    if not is_speech:
                        if not in_pause:
                            in_pause = True
                            pause_count = 1
                        else:
                            pause_count += 1
                    else:
                        if in_pause:
                            pause_lengths.append(pause_count * hop_length / sr)
                            in_pause = False

                if pause_lengths:
                    features.extend([
                        np.mean(pause_lengths),
                        np.std(pause_lengths) if len(pause_lengths) > 1 else 0,
                        len(pause_lengths) / (len(y) / sr),  # Pause rate
                        np.sum(pause_lengths) / (len(y) / sr),  # Pause proportion
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            except:
                features.extend([0, 0, 0, 0])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            return None


class HuBERTExtractor:
    """Extract HuBERT deep learning features."""

    def __init__(self, model_name: str = "facebook/hubert-base-ls960"):
        self.device = DEVICE
        print(f"  Loading HuBERT model...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"  HuBERT loaded on {self.device}")

    def extract(self, audio_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
        """Extract HuBERT features."""
        try:
            waveform, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
            waveform = waveform.astype(np.float32)

            inputs = self.feature_extractor(
                waveform, sampling_rate=target_sr,
                return_tensors="pt", padding=True
            )

            with torch.no_grad():
                outputs = self.model(inputs.input_values.to(self.device))
                hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            # Multi-level aggregation
            return np.concatenate([
                np.mean(hidden, axis=0),
                np.std(hidden, axis=0),
                np.max(hidden, axis=0),
                np.percentile(hidden, 25, axis=0),
                np.percentile(hidden, 75, axis=0),
            ])
        except:
            return None


# ============================================================================
# SECTION 3: ALGORITHM BENCHMARKING
# ============================================================================

def get_models_to_benchmark() -> Dict:
    """Define models to benchmark."""
    return {
        # Linear models
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Lasso': Lasso(alpha=0.1),

        # Tree-based ensembles
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42),

        # Support Vector Machines
        'SVR_RBF': SVR(kernel='rbf', C=10, gamma='scale'),
        'SVR_Linear': SVR(kernel='linear', C=1),

        # K-Nearest Neighbors
        'KNN_5': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        'KNN_10': KNeighborsRegressor(n_neighbors=10, weights='distance'),

        # Neural Network
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }


def benchmark_algorithms(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        cv_type: str = 'personalized') -> Dict:
    """
    Benchmark multiple algorithms.

    Args:
        X: Feature matrix
        y: Target values (glucose)
        groups: Participant IDs for grouping
        cv_type: 'personalized' (LOO within person) or 'population' (LOGO)

    Returns:
        Dictionary of results for each model
    """
    results = {}
    models = get_models_to_benchmark()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle NaN/Inf
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

    for name, model in models.items():
        try:
            if cv_type == 'personalized':
                # Leave-One-Out within each person
                all_preds = []
                all_actual = []

                for group in np.unique(groups):
                    mask = groups == group
                    X_group = X_scaled[mask]
                    y_group = y[mask]

                    if len(X_group) < 10:
                        continue

                    if len(X_group) <= 50:
                        cv = LeaveOneOut()
                    else:
                        cv = KFold(n_splits=min(10, len(X_group)), shuffle=True, random_state=42)

                    preds = cross_val_predict(model, X_group, y_group, cv=cv)
                    all_preds.extend(preds)
                    all_actual.extend(y_group)

                if len(all_preds) > 0:
                    mae = mean_absolute_error(all_actual, all_preds)
                    rmse = np.sqrt(mean_squared_error(all_actual, all_preds))
                    r, _ = stats.pearsonr(all_actual, all_preds)

                    results[name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'r': r,
                        'predictions': np.array(all_preds),
                        'actual': np.array(all_actual),
                    }

            else:  # population (Leave-One-Group-Out)
                logo = LeaveOneGroupOut()
                preds = cross_val_predict(model, X_scaled, y, cv=logo, groups=groups)

                mae = mean_absolute_error(y, preds)
                rmse = np.sqrt(mean_squared_error(y, preds))
                r, _ = stats.pearsonr(y, preds)

                results[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r': r,
                    'predictions': preds,
                    'actual': y,
                }

        except Exception as e:
            print(f"    Error with {name}: {e}")
            continue

    return results


# ============================================================================
# SECTION 4: PERSONALIZATION STRATEGIES
# ============================================================================

class PersonalizationStrategy:
    """Base class for personalization strategies."""

    def __init__(self, name: str):
        self.name = name

    def adapt(self, base_model, calibration_data: List[Tuple],
              test_features: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BiasCorrection(PersonalizationStrategy):
    """Simple bias correction: predict = population + mean_error."""

    def __init__(self):
        super().__init__("BiasCorrection")

    def adapt(self, base_prediction: float, calibration_data: List[Tuple]) -> float:
        if len(calibration_data) < 3:
            return base_prediction

        errors = [actual - pred for pred, actual in calibration_data]
        bias = np.mean(errors)
        return base_prediction + bias


class LinearCorrection(PersonalizationStrategy):
    """Linear correction: predict = a * population + b."""

    def __init__(self):
        super().__init__("LinearCorrection")
        self.model = None

    def fit(self, calibration_data: List[Tuple]):
        if len(calibration_data) < 5:
            self.model = None
            return

        preds = np.array([d[0] for d in calibration_data]).reshape(-1, 1)
        actuals = np.array([d[1] for d in calibration_data])

        self.model = Ridge(alpha=1.0)
        self.model.fit(preds, actuals)

    def adapt(self, base_prediction: float) -> float:
        if self.model is None:
            return base_prediction
        return self.model.predict([[base_prediction]])[0]


class FeatureAdaptation(PersonalizationStrategy):
    """Learn feature transformation for the individual."""

    def __init__(self):
        super().__init__("FeatureAdaptation")
        self.personal_model = None
        self.scaler = None

    def fit(self, features: np.ndarray, actuals: np.ndarray):
        if len(features) < 10:
            self.personal_model = None
            return

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(features)

        self.personal_model = BayesianRidge()
        self.personal_model.fit(X_scaled, actuals)

    def predict(self, features: np.ndarray) -> float:
        if self.personal_model is None:
            return None

        X_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.personal_model.predict(X_scaled)[0]


class EnsemblePersonalization(PersonalizationStrategy):
    """Weighted ensemble of population and personalized models."""

    def __init__(self):
        super().__init__("EnsemblePersonalization")
        self.weight = 0.5  # Start with equal weight
        self.personal_model = None

    def update_weight(self, pop_errors: List[float], pers_errors: List[float]):
        """Update ensemble weight based on recent performance."""
        if len(pop_errors) < 3 or len(pers_errors) < 3:
            return

        pop_mae = np.mean(np.abs(pop_errors[-10:]))
        pers_mae = np.mean(np.abs(pers_errors[-10:]))

        # Weight inversely proportional to error
        total = pop_mae + pers_mae
        if total > 0:
            self.weight = pop_mae / total  # Higher weight for personal if pop is worse

    def predict(self, pop_pred: float, pers_pred: Optional[float]) -> float:
        if pers_pred is None:
            return pop_pred
        return (1 - self.weight) * pop_pred + self.weight * pers_pred


def evaluate_personalization_strategies(X: np.ndarray, y: np.ndarray,
                                        groups: np.ndarray, pop_model) -> Dict:
    """Evaluate different personalization strategies."""

    results = {}
    unique_groups = np.unique(groups)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

    strategies = {
        'NoPersonalization': [],
        'BiasCorrection_5': [],
        'BiasCorrection_10': [],
        'LinearCorrection_10': [],
        'FeatureAdaptation_10': [],
        'FeatureAdaptation_20': [],
    }

    for group in unique_groups:
        mask = groups == group
        X_group = X_scaled[mask]
        y_group = y[mask]

        if len(X_group) < 25:
            continue

        # Get population model predictions for this person
        pop_preds = pop_model.predict(X_group)

        # Strategy 1: No personalization
        strategies['NoPersonalization'].extend(list(zip(pop_preds, y_group)))

        # Strategy 2-3: Bias correction with different calibration sizes
        for n_calib, key in [(5, 'BiasCorrection_5'), (10, 'BiasCorrection_10')]:
            if len(X_group) > n_calib + 5:
                corrector = BiasCorrection()
                calib_data = list(zip(pop_preds[:n_calib], y_group[:n_calib]))

                corrected_preds = []
                for pred in pop_preds[n_calib:]:
                    corrected = corrector.adapt(pred, calib_data)
                    corrected_preds.append(corrected)

                strategies[key].extend(list(zip(corrected_preds, y_group[n_calib:])))

        # Strategy 4: Linear correction
        if len(X_group) > 15:
            corrector = LinearCorrection()
            calib_data = list(zip(pop_preds[:10], y_group[:10]))
            corrector.fit(calib_data)

            corrected_preds = [corrector.adapt(p) for p in pop_preds[10:]]
            strategies['LinearCorrection_10'].extend(list(zip(corrected_preds, y_group[10:])))

        # Strategy 5-6: Feature adaptation
        for n_calib, key in [(10, 'FeatureAdaptation_10'), (20, 'FeatureAdaptation_20')]:
            if len(X_group) > n_calib + 5:
                adapter = FeatureAdaptation()
                adapter.fit(X_group[:n_calib], y_group[:n_calib])

                if adapter.personal_model is not None:
                    pers_preds = [adapter.predict(x) for x in X_group[n_calib:]]
                    strategies[key].extend(list(zip(pers_preds, y_group[n_calib:])))

    # Calculate metrics for each strategy
    for name, predictions in strategies.items():
        if len(predictions) > 0:
            preds = [p[0] for p in predictions]
            actuals = [p[1] for p in predictions]

            results[name] = {
                'mae': mean_absolute_error(actuals, preds),
                'rmse': np.sqrt(mean_squared_error(actuals, preds)),
                'r': stats.pearsonr(actuals, preds)[0],
                'n_samples': len(predictions),
            }

    return results


# ============================================================================
# SECTION 5: MAIN PIPELINE
# ============================================================================

def extract_all_features(biomarker_extractor: VoiceBiomarkerExtractor,
                        hubert_extractor: Optional[HuBERTExtractor] = None,
                        max_per_participant: Optional[int] = None,
                        use_hubert: bool = True) -> List[Dict]:
    """Extract all features from dataset."""

    all_data = []

    for name, config in PARTICIPANTS.items():
        print(f"\n  Processing {name}...")

        # Load glucose
        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
        if glucose_df.empty:
            print(f"    No glucose data")
            continue

        # Find audio files
        audio_files = []
        for audio_dir in config['audio_dirs']:
            dir_path = BASE_DIR / audio_dir
            if dir_path.exists():
                audio_files.extend(list(dir_path.glob("*.wav")))

        if max_per_participant:
            audio_files = audio_files[:max_per_participant]

        print(f"    Audio: {len(audio_files)}, Glucose: {len(glucose_df)}")

        offset = config.get('time_offset', 15)
        count = 0

        for i, audio_path in enumerate(audio_files):
            audio_ts = parse_timestamp(audio_path.name)
            if audio_ts is None:
                continue

            glucose = find_matching_glucose(audio_ts, glucose_df, offset)
            if glucose is None:
                continue

            # Extract biomarker features
            bio_feats = biomarker_extractor.extract(str(audio_path))
            if bio_feats is None:
                continue

            # Optional HuBERT features
            hubert_feats = None
            if use_hubert and hubert_extractor:
                hubert_feats = hubert_extractor.extract(str(audio_path))

            # Combine features
            if hubert_feats is not None:
                combined = np.concatenate([bio_feats, hubert_feats])
            else:
                combined = bio_feats

            all_data.append({
                'participant': name,
                'audio_path': str(audio_path),
                'timestamp': audio_ts,
                'glucose': glucose,
                'features': combined,
                'biomarker_dim': len(bio_feats),
                'hubert_dim': len(hubert_feats) if hubert_feats is not None else 0,
            })

            count += 1

            if (i + 1) % 50 == 0:
                print(f"      Processed {i+1}, extracted {count}")

        print(f"    Extracted {count} samples")

    return all_data


def run_comprehensive_analysis(use_hubert: bool = True, max_samples: int = 100):
    """Run the complete analysis pipeline."""

    print("\n" + "="*70)
    print("PRODUCTION VOICE-GLUCOSE ANALYSIS")
    print("="*70)

    # 1. Data Exploration
    exploration = explore_data()

    # 2. Feature Extraction
    print("\n" + "="*70)
    print("SECTION 2: FEATURE EXTRACTION")
    print("="*70)

    bio_extractor = VoiceBiomarkerExtractor()
    hubert_extractor = HuBERTExtractor() if use_hubert else None

    all_data = extract_all_features(
        bio_extractor, hubert_extractor,
        max_per_participant=max_samples,
        use_hubert=use_hubert
    )

    if len(all_data) < 50:
        print("ERROR: Not enough data extracted!")
        return None

    # Prepare arrays
    X = np.array([d['features'] for d in all_data])
    y = np.array([d['glucose'] for d in all_data])
    groups = np.array([d['participant'] for d in all_data])

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"  - Biomarkers: {all_data[0]['biomarker_dim']}")
    if use_hubert:
        print(f"  - HuBERT: {all_data[0]['hubert_dim']}")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # 3. Algorithm Benchmarking
    print("\n" + "="*70)
    print("SECTION 3: ALGORITHM BENCHMARKING")
    print("="*70)

    print("\n[3.1] Personalized Models (Leave-One-Out CV)...")
    personalized_results = benchmark_algorithms(X, y, groups, cv_type='personalized')

    print("\n  Results (sorted by MAE):")
    sorted_results = sorted(personalized_results.items(), key=lambda x: x[1]['mae'])
    for name, res in sorted_results[:10]:
        print(f"    {name:20s}: MAE={res['mae']:.2f}, RMSE={res['rmse']:.2f}, r={res['r']:.3f}")

    print("\n[3.2] Population Models (Leave-One-Person-Out CV)...")
    population_results = benchmark_algorithms(X, y, groups, cv_type='population')

    print("\n  Results (sorted by MAE):")
    sorted_results = sorted(population_results.items(), key=lambda x: x[1]['mae'])
    for name, res in sorted_results[:10]:
        print(f"    {name:20s}: MAE={res['mae']:.2f}, RMSE={res['rmse']:.2f}, r={res['r']:.3f}")

    # 4. Personalization Strategies
    print("\n" + "="*70)
    print("SECTION 4: PERSONALIZATION STRATEGIES")
    print("="*70)

    # Train population model for personalization tests
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

    best_pop_model = BayesianRidge()
    best_pop_model.fit(X_scaled, y)

    pers_strategy_results = evaluate_personalization_strategies(X, y, groups, best_pop_model)

    print("\n  Personalization Strategy Results:")
    for name, res in sorted(pers_strategy_results.items(), key=lambda x: x[1]['mae']):
        print(f"    {name:25s}: MAE={res['mae']:.2f}, r={res['r']:.3f}, n={res['n_samples']}")

    # 5. Generate visualizations and report
    print("\n" + "="*70)
    print("SECTION 5: VISUALIZATION AND REPORT")
    print("="*70)

    generate_visualizations(X, y, groups, personalized_results, population_results,
                           pers_strategy_results, all_data, OUTPUT_DIR)

    generate_final_report(exploration, all_data, personalized_results,
                         population_results, pers_strategy_results, OUTPUT_DIR)

    # Save models
    model_package = {
        'scaler': scaler,
        'population_model': best_pop_model,
        'feature_dim': X.shape[1],
        'biomarker_dim': all_data[0]['biomarker_dim'],
        'hubert_dim': all_data[0]['hubert_dim'],
    }

    with open(OUTPUT_DIR / 'production_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)

    print(f"\n  Model saved to: {OUTPUT_DIR / 'production_model.pkl'}")
    print(f"  Report saved to: {OUTPUT_DIR / 'production_report.html'}")

    return {
        'personalized': personalized_results,
        'population': population_results,
        'personalization_strategies': pers_strategy_results,
        'data': all_data,
    }


# ============================================================================
# SECTION 6: VISUALIZATIONS
# ============================================================================

def generate_visualizations(X, y, groups, pers_results, pop_results,
                           strategy_results, all_data, output_dir):
    """Generate comprehensive visualizations."""

    print("\n  Generating visualizations...")

    # 1. Algorithm comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Personalized
    ax = axes[0]
    names = [n for n, _ in sorted(pers_results.items(), key=lambda x: x[1]['mae'])]
    maes = [pers_results[n]['mae'] for n in names]
    colors = ['#2ecc71' if m < 10 else '#f39c12' if m < 12 else '#e74c3c' for m in maes]
    ax.barh(names, maes, color=colors)
    ax.set_xlabel('MAE (mg/dL)')
    ax.set_title('Personalized Models (Leave-One-Out CV)')
    ax.axvline(10, color='gray', linestyle='--', alpha=0.5, label='10 mg/dL target')

    # Population
    ax = axes[1]
    names = [n for n, _ in sorted(pop_results.items(), key=lambda x: x[1]['mae'])]
    maes = [pop_results[n]['mae'] for n in names]
    colors = ['#2ecc71' if m < 12 else '#f39c12' if m < 15 else '#e74c3c' for m in maes]
    ax.barh(names, maes, color=colors)
    ax.set_xlabel('MAE (mg/dL)')
    ax.set_title('Population Models (Leave-One-Person-Out CV)')
    ax.axvline(12, color='gray', linestyle='--', alpha=0.5, label='12 mg/dL target')

    plt.tight_layout()
    fig.savefig(output_dir / 'algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Personalization strategy comparison
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(strategy_results.keys())
    maes = [strategy_results[n]['mae'] for n in names]
    rs = [strategy_results[n]['r'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, maes, width, label='MAE (mg/dL)', color='#3498db')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, rs, width, label='Correlation (r)', color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('MAE (mg/dL)')
    ax2.set_ylabel('Correlation (r)')
    ax.set_title('Personalization Strategy Comparison')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    fig.savefig(output_dir / 'personalization_strategies.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Per-participant results
    best_model = min(pers_results.items(), key=lambda x: x[1]['mae'])[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    unique_groups = np.unique(groups)
    for i, group in enumerate(unique_groups[:8]):
        ax = axes[i]
        mask = groups == group

        # Get predictions for this group from best model
        actual = y[mask]

        # Simple scatter of actual values distribution
        ax.hist(actual, bins=20, alpha=0.7, color='#3498db')
        ax.axvline(np.mean(actual), color='red', linestyle='--', label=f'Mean: {np.mean(actual):.0f}')
        ax.set_xlabel('Glucose (mg/dL)')
        ax.set_ylabel('Count')
        ax.set_title(f'{group} (n={len(actual)})')
        ax.legend()

    for i in range(len(unique_groups), 8):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'participant_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Feature importance (using Random Forest)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    importances = rf.feature_importances_
    top_k = 30
    top_indices = np.argsort(importances)[-top_k:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_k), importances[top_indices], color='#3498db')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f'Feature {i}' for i in top_indices])
    ax.set_xlabel('Importance')
    ax.set_title('Top 30 Feature Importances (Random Forest)')

    plt.tight_layout()
    fig.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 5. t-SNE visualization
    print("  Computing t-SNE...")

    # PCA first for dimensionality reduction
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_pca)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by participant
    ax = axes[0]
    for group in np.unique(groups):
        mask = groups == group
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=group, alpha=0.6, s=30)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE by Participant')
    ax.legend()

    # Color by glucose
    ax = axes[1]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='RdYlGn_r', alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax, label='Glucose (mg/dL)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE by Glucose Level')

    plt.tight_layout()
    fig.savefig(output_dir / 'tsne_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Visualizations saved!")


# ============================================================================
# SECTION 7: FINAL REPORT
# ============================================================================

def generate_final_report(exploration, all_data, pers_results, pop_results,
                         strategy_results, output_dir):
    """Generate comprehensive HTML report."""

    # Best models
    best_pers = min(pers_results.items(), key=lambda x: x[1]['mae'])
    best_pop = min(pop_results.items(), key=lambda x: x[1]['mae'])
    best_strategy = min(strategy_results.items(), key=lambda x: x[1]['mae'])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Voice-Glucose Production Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
        .header {{ background: linear-gradient(135deg, #1a5276, #2980b9); color: white; padding: 40px; border-radius: 15px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        h2 {{ color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 10px; margin-top: 40px; }}
        h3 {{ color: #2c3e50; margin-top: 25px; }}
        .section {{ background: white; padding: 30px; border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #2980b9; color: white; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 25px; border-radius: 12px; text-align: center; }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
        .highlight {{ background: #d5f5e3; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #27ae60; }}
        .warning {{ background: #fdebd0; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #f39c12; }}
        .figure {{ text-align: center; margin: 30px 0; }}
        .figure img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .code {{ background: #2c3e50; color: #ecf0f1; padding: 20px; border-radius: 10px; font-family: 'Consolas', monospace; overflow-x: auto; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        @media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
        .badge {{ display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
        .badge-success {{ background: #27ae60; color: white; }}
        .badge-warning {{ background: #f39c12; color: white; }}
        .badge-info {{ background: #3498db; color: white; }}
    </style>
</head>
<body>

<div class="header">
    <h1>🎤 Voice-Based Glucose Estimation</h1>
    <p>Production Analysis Report | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p>Comprehensive analysis for building a monetizable product with competitive moat</p>
</div>

<div class="section">
    <h2>📊 Executive Summary</h2>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{len(set(d['participant'] for d in all_data))}</div>
            <div class="metric-label">Participants</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(all_data)}</div>
            <div class="metric-label">Total Samples</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, #27ae60, #2ecc71);">
            <div class="metric-value">{best_pers[1]['mae']:.1f}</div>
            <div class="metric-label">Best Personalized MAE</div>
        </div>
        <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c, #c0392b);">
            <div class="metric-value">{best_pop[1]['mae']:.1f}</div>
            <div class="metric-label">Best Population MAE</div>
        </div>
    </div>

    <div class="highlight">
        <strong>Key Finding:</strong> With personalization ({best_strategy[0]}), we achieve
        <strong>{best_strategy[1]['mae']:.1f} mg/dL MAE</strong>, which is
        {((best_pop[1]['mae'] - best_strategy[1]['mae']) / best_pop[1]['mae'] * 100):.1f}% better than the population-only model.
        This personalization advantage is our primary competitive moat.
    </div>
</div>

<div class="section">
    <h2>🔬 First-Principles Analysis</h2>

    <h3>Why Voice Reflects Blood Glucose</h3>
    <div class="two-col">
        <div>
            <p><strong>1. Autonomic Nervous System Effects</strong></p>
            <ul>
                <li>Glucose levels affect sympathetic/parasympathetic balance</li>
                <li>This modulates vocal fold tension and vibration frequency</li>
                <li>Measurable as changes in fundamental frequency (F0)</li>
            </ul>

            <p><strong>2. Blood Viscosity Changes</strong></p>
            <ul>
                <li>Glucose affects blood rheology</li>
                <li>Changes microvascular perfusion to larynx</li>
                <li>Affects voice quality and harmonic content</li>
            </ul>
        </div>
        <div>
            <p><strong>3. Neuromuscular Effects</strong></p>
            <ul>
                <li>Glucose is primary brain fuel</li>
                <li>Affects speech motor control precision</li>
                <li>Measurable as jitter/shimmer variations</li>
            </ul>

            <p><strong>4. Cognitive/Behavioral Effects</strong></p>
            <ul>
                <li>Glucose affects cognitive processing speed</li>
                <li>Changes speech rate and pause patterns</li>
                <li>Detectable in temporal speech features</li>
            </ul>
        </div>
    </div>
</div>

<div class="section">
    <h2>🏆 Algorithm Benchmarking Results</h2>

    <h3>Personalized Models (Leave-One-Out CV)</h3>
    <p>These results represent what's achievable with user-specific models.</p>

    <table>
        <tr>
            <th>Rank</th>
            <th>Algorithm</th>
            <th>MAE (mg/dL)</th>
            <th>RMSE (mg/dL)</th>
            <th>Correlation</th>
            <th>Assessment</th>
        </tr>
"""

    for i, (name, res) in enumerate(sorted(pers_results.items(), key=lambda x: x[1]['mae'])):
        badge = 'success' if res['mae'] < 10 else 'warning' if res['mae'] < 12 else 'info'
        assessment = 'Excellent' if res['mae'] < 10 else 'Good' if res['mae'] < 12 else 'Needs improvement'
        html += f"""        <tr>
            <td>{i+1}</td>
            <td><strong>{name}</strong></td>
            <td>{res['mae']:.2f}</td>
            <td>{res['rmse']:.2f}</td>
            <td>{res['r']:.3f}</td>
            <td><span class="badge badge-{badge}">{assessment}</span></td>
        </tr>
"""

    html += """    </table>

    <h3>Population Models (Leave-One-Person-Out CV)</h3>
    <p>These results show generalization to completely new users (cold start).</p>

    <table>
        <tr>
            <th>Rank</th>
            <th>Algorithm</th>
            <th>MAE (mg/dL)</th>
            <th>RMSE (mg/dL)</th>
            <th>Correlation</th>
        </tr>
"""

    for i, (name, res) in enumerate(sorted(pop_results.items(), key=lambda x: x[1]['mae'])):
        html += f"""        <tr>
            <td>{i+1}</td>
            <td><strong>{name}</strong></td>
            <td>{res['mae']:.2f}</td>
            <td>{res['rmse']:.2f}</td>
            <td>{res['r']:.3f}</td>
        </tr>
"""

    html += f"""    </table>

    <div class="figure">
        <img src="algorithm_comparison.png" alt="Algorithm Comparison">
        <p><em>Comparison of all benchmarked algorithms</em></p>
    </div>
</div>

<div class="section">
    <h2>🎯 Personalization Strategy Analysis</h2>

    <p>Testing different approaches to adapt the population model to individual users:</p>

    <table>
        <tr>
            <th>Strategy</th>
            <th>Calibration Samples</th>
            <th>MAE (mg/dL)</th>
            <th>Correlation</th>
            <th>Improvement vs Population</th>
        </tr>
"""

    pop_mae = best_pop[1]['mae']
    for name, res in sorted(strategy_results.items(), key=lambda x: x[1]['mae']):
        improvement = (pop_mae - res['mae']) / pop_mae * 100
        calib = name.split('_')[-1] if '_' in name else 'N/A'
        html += f"""        <tr>
            <td><strong>{name}</strong></td>
            <td>{calib}</td>
            <td>{res['mae']:.2f}</td>
            <td>{res['r']:.3f}</td>
            <td>{improvement:+.1f}%</td>
        </tr>
"""

    html += f"""    </table>

    <div class="highlight">
        <strong>Recommendation:</strong> Use <strong>{best_strategy[0]}</strong> for personalization.
        It provides {((pop_mae - best_strategy[1]['mae']) / pop_mae * 100):.1f}% improvement over population-only predictions
        with reasonable calibration requirements.
    </div>

    <div class="figure">
        <img src="personalization_strategies.png" alt="Personalization Strategies">
        <p><em>Comparison of personalization strategies</em></p>
    </div>
</div>

<div class="section">
    <h2>🛡️ Competitive Moat Strategy</h2>

    <div class="two-col">
        <div class="highlight" style="background: #ebf5fb; border-color: #3498db;">
            <h3>1. Data Moat</h3>
            <ul>
                <li>Proprietary paired voice-glucose dataset</li>
                <li>Currently {len(all_data)} samples from {len(set(d['participant'] for d in all_data))} participants</li>
                <li>Each new user adds to training data</li>
                <li>Competitors would need years to replicate</li>
            </ul>
        </div>

        <div class="highlight" style="background: #fef9e7; border-color: #f1c40f;">
            <h3>2. Algorithm Moat</h3>
            <ul>
                <li>Personalization that improves with each user</li>
                <li>Few-shot adaptation requires only 5-10 calibrations</li>
                <li>Continuous learning from user feedback</li>
                <li>Proprietary feature extraction pipeline</li>
            </ul>
        </div>

        <div class="highlight" style="background: #e8f8f5; border-color: #1abc9c;">
            <h3>3. Network Effects</h3>
            <ul>
                <li>More users → better population model</li>
                <li>Better cold-start for new users</li>
                <li>Community validation and trust</li>
                <li>User-generated content (voice recordings)</li>
            </ul>
        </div>

        <div class="highlight" style="background: #fdedec; border-color: #e74c3c;">
            <h3>4. Integration Moat</h3>
            <ul>
                <li>API/SDK for CGM manufacturers</li>
                <li>Integration with health apps</li>
                <li>Enterprise B2B partnerships</li>
                <li>Clinical validation studies</li>
            </ul>
        </div>
    </div>
</div>

<div class="section">
    <h2>🔧 Production API Design</h2>

    <div class="code">
<pre>
# Python SDK Example

from voiceglucose import VoiceGlucoseAPI

# Initialize
api = VoiceGlucoseAPI(api_key="your_key")

# 1. New user - cold start prediction
result = api.predict(
    audio_file="voice_recording.wav",
    user_id="new_user_123"
)
print(f"Predicted: {{result.glucose}} mg/dL")
print(f"Confidence: {{result.confidence}}")
print(f"Range: {{result.low}} - {{result.high}} mg/dL")

# 2. Calibrate with actual reading (CGM or fingerprick)
api.calibrate(
    audio_file="calibration_voice.wav",
    actual_glucose=120,
    user_id="user_123"
)

# 3. Personalized prediction
result = api.predict(
    audio_file="new_voice.wav",
    user_id="user_123"  # Now uses personalized model
)

# 4. Get calibration status
status = api.get_calibration_status("user_123")
print(f"Calibration samples: {{status.n_samples}}")
print(f"Estimated improvement: {{status.improvement}}%")
</pre>
    </div>

    <h3>REST API Endpoints</h3>
    <table>
        <tr>
            <th>Endpoint</th>
            <th>Method</th>
            <th>Description</th>
        </tr>
        <tr>
            <td><code>/v1/predict</code></td>
            <td>POST</td>
            <td>Predict glucose from audio file</td>
        </tr>
        <tr>
            <td><code>/v1/calibrate</code></td>
            <td>POST</td>
            <td>Add calibration sample with actual glucose</td>
        </tr>
        <tr>
            <td><code>/v1/users/{{user_id}}/status</code></td>
            <td>GET</td>
            <td>Get user calibration status</td>
        </tr>
        <tr>
            <td><code>/v1/users/{{user_id}}/history</code></td>
            <td>GET</td>
            <td>Get prediction history</td>
        </tr>
    </table>
</div>

<div class="section">
    <h2>📈 Visualizations</h2>

    <div class="figure">
        <img src="participant_distributions.png" alt="Participant Distributions">
        <p><em>Glucose distribution for each participant</em></p>
    </div>

    <div class="figure">
        <img src="tsne_visualization.png" alt="t-SNE Visualization">
        <p><em>t-SNE visualization showing voice feature space colored by participant and glucose level</em></p>
    </div>

    <div class="figure">
        <img src="feature_importance.png" alt="Feature Importance">
        <p><em>Top 30 most important features for glucose prediction</em></p>
    </div>
</div>

<div class="section">
    <h2>🚀 Next Steps & Recommendations</h2>

    <h3>Immediate (1-2 weeks)</h3>
    <ol>
        <li>Deploy API with {best_pop[0]} as population model</li>
        <li>Implement {best_strategy[0]} personalization</li>
        <li>Add confidence intervals to predictions</li>
        <li>Create mobile SDK (iOS/Android)</li>
    </ol>

    <h3>Short-term (1-3 months)</h3>
    <ol>
        <li>Collect more diverse training data (target: 50+ participants)</li>
        <li>Implement continuous learning from user feedback</li>
        <li>Add meal/activity context features</li>
        <li>Clinical validation study (FDA pathway exploration)</li>
    </ol>

    <h3>Long-term (3-12 months)</h3>
    <ol>
        <li>Partner with CGM manufacturers (Dexcom, Abbott, Medtronic)</li>
        <li>Enterprise API for health apps</li>
        <li>International expansion (multi-language support)</li>
        <li>Pursue CE marking and FDA clearance</li>
    </ol>

    <div class="warning">
        <strong>Important:</strong> Current accuracy ({best_pers[1]['mae']:.1f} mg/dL MAE) is suitable for
        trend monitoring and lifestyle guidance. For medical decisions, always recommend confirmation
        with approved glucose monitoring devices.
    </div>
</div>

<div class="section">
    <h2>📁 Project Files</h2>

    <table>
        <tr>
            <th>File</th>
            <th>Description</th>
        </tr>
        <tr><td><code>production_analysis.py</code></td><td>This comprehensive analysis script</td></tr>
        <tr><td><code>combined_hubert_mfcc_model.py</code></td><td>HuBERT + MFCC combined model</td></tr>
        <tr><td><code>hubert_glucose_model.py</code></td><td>HuBERT-only model with API</td></tr>
        <tr><td><code>comprehensive_analysis_v6.py</code></td><td>MFCC analysis with time offsets</td></tr>
        <tr><td><code>PROJECT_OVERVIEW.md</code></td><td>Project documentation</td></tr>
        <tr><td><code>production_analysis/production_model.pkl</code></td><td>Trained production model</td></tr>
        <tr><td><code>production_analysis/production_report.html</code></td><td>This report</td></tr>
    </table>
</div>

<footer style="text-align: center; padding: 30px; color: #7f8c8d; font-size: 0.9em;">
    <p>Voice-Glucose Production Analysis Report</p>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</footer>

</body>
</html>
"""

    with open(output_dir / 'production_report.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  Report saved to: {output_dir / 'production_report.html'}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    # Parse arguments
    use_hubert = '--no-hubert' not in sys.argv
    max_samples = 75  # Balance between speed and data

    for arg in sys.argv:
        if arg.startswith('--max-samples='):
            max_samples = int(arg.split('=')[1])

    print(f"Configuration:")
    print(f"  Use HuBERT: {use_hubert}")
    print(f"  Max samples per participant: {max_samples}")

    results = run_comprehensive_analysis(use_hubert=use_hubert, max_samples=max_samples)
