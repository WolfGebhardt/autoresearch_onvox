"""
Innovative Voice-Glucose Analysis
=================================
Novel approaches to improve glucose prediction from voice:

1. Multi-Offset Fusion: Use features from multiple time offsets simultaneously
2. Rate-of-Change Prediction: Predict glucose TREND (rising/falling/stable) instead of absolute value
3. Confidence-Weighted Predictions: Weight predictions by glucose stability
4. Voice-Glucose Lag Estimation: Estimate individual's physiological lag
5. Asymmetric Loss: Different penalties for over/under prediction (clinical safety)
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from datetime import datetime, timedelta
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy import stats, signal

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import BayesianRidge, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")
OUTPUT_DIR = BASE_DIR / "innovative_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

MMOL_TO_MGDL = 18.0182

# Optimal offsets discovered from enhanced_analysis.py
OPTIMAL_OFFSETS = {
    "Wolf": 15,
    "Sybille": -30,
    "Anja": 15,
    "Margarita": -30,
    "Vicky": 30,
    "Steffen": 20,
    "Lara": 20,
}

PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "glucose_unit": "mg/dL",
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchutt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav"],
        "glucose_unit": "mg/dL",
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
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "glucose_unit": "mmol/L",
    },
    "Lara": {
        "glucose_csv": ["Lara/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Lara/conv_audio"],
        "glucose_unit": "mmol/L",
    },
}


# ============================================================================
# DATA LOADING (reuse from enhanced_analysis)
# ============================================================================

def parse_audio_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from WhatsApp audio filename."""
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})[- ]um[- ](\d{2})\.(\d{2})\.(\d{2})',
        r'(\d{4})-(\d{2})-(\d{2})[- ]um[- ](\d{2})\.(\d{2})',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 6:
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]),
                                   int(groups[3]), int(groups[4]), int(groups[5]))
                else:
                    return datetime(int(groups[0]), int(groups[1]), int(groups[2]),
                                   int(groups[3]), int(groups[4]), 0)
            except ValueError:
                pass
    return None


def load_glucose_csv(csv_paths: List[str], unit: str) -> pd.DataFrame:
    """Load and merge glucose CSVs."""
    all_dfs = []

    for csv_path in csv_paths:
        full_path = BASE_DIR / csv_path

        if not full_path.exists():
            parent_dir = full_path.parent
            if parent_dir.exists():
                csv_files = list(parent_dir.glob("*.csv"))
                if csv_files:
                    full_path = csv_files[0]
                else:
                    continue
            else:
                continue

        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = [f.readline() for _ in range(5)]

            skiprows = 0
            for i, line in enumerate(lines):
                if 'Device' in line or 'Ger' in line:
                    skiprows = i
                    break

            df = pd.read_csv(full_path, skiprows=skiprows, encoding='utf-8', on_bad_lines='skip')

            glucose_col = None
            timestamp_col = None

            for col in df.columns:
                col_lower = col.lower()
                if 'historic glucose' in col_lower or 'glukosewert' in col_lower:
                    glucose_col = col
                if 'device timestamp' in col_lower or 'zeitstempel' in col_lower:
                    timestamp_col = col

            if glucose_col is None:
                continue

            if timestamp_col is None:
                timestamp_col = 'Device Timestamp'

            df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
            df['glucose_raw'] = pd.to_numeric(df[glucose_col], errors='coerce')

            if 'mmol' in glucose_col.lower():
                df['glucose_mgdl'] = df['glucose_raw'] * MMOL_TO_MGDL
            else:
                df['glucose_mgdl'] = df['glucose_raw']

            df = df[['timestamp', 'glucose_mgdl']].dropna()
            if len(df) > 0:
                all_dfs.append(df)

        except Exception as e:
            continue

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates('timestamp')
    return combined


def extract_mfcc_features(audio_path: Path, n_mfcc: int = 20) -> Optional[np.ndarray]:
    """Extract MFCC features with deltas."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) < sr * 0.5:
            return None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        features = []
        for feat in [mfcc, mfcc_delta, mfcc_delta2]:
            features.append(feat.mean(axis=1))
            features.append(feat.std(axis=1))

        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.array([rms.mean(), rms.std()]))
        features.append(np.array([zcr.mean(), zcr.std()]))

        return np.concatenate(features)
    except:
        return None


def get_glucose_context(glucose_df: pd.DataFrame, target_time: datetime,
                        window_minutes: int = 60) -> Dict:
    """Get rich glucose context around a time point."""
    if glucose_df.empty:
        return None

    target_ts = pd.Timestamp(target_time)

    # Get readings in window
    window_start = target_ts - timedelta(minutes=window_minutes)
    window_end = target_ts + timedelta(minutes=window_minutes)

    mask = (glucose_df['timestamp'] >= window_start) & (glucose_df['timestamp'] <= window_end)
    window_data = glucose_df[mask].copy()

    if len(window_data) < 3:
        return None

    # Find closest reading
    diffs = (window_data['timestamp'] - target_ts).abs()
    closest_idx = diffs.idxmin()
    closest_glucose = window_data.loc[closest_idx, 'glucose_mgdl']
    time_diff = diffs[closest_idx].total_seconds() / 60

    if time_diff > 15:
        return None

    # Calculate rate of change (mg/dL per minute)
    window_data = window_data.sort_values('timestamp')

    # Rate over last 15 minutes
    recent_mask = window_data['timestamp'] >= (target_ts - timedelta(minutes=15))
    recent = window_data[recent_mask]

    if len(recent) >= 2:
        time_span = (recent['timestamp'].iloc[-1] - recent['timestamp'].iloc[0]).total_seconds() / 60
        if time_span > 0:
            rate_15min = (recent['glucose_mgdl'].iloc[-1] - recent['glucose_mgdl'].iloc[0]) / time_span
        else:
            rate_15min = 0
    else:
        rate_15min = 0

    # Rate over last 30 minutes
    recent_mask_30 = window_data['timestamp'] >= (target_ts - timedelta(minutes=30))
    recent_30 = window_data[recent_mask_30]

    if len(recent_30) >= 2:
        time_span = (recent_30['timestamp'].iloc[-1] - recent_30['timestamp'].iloc[0]).total_seconds() / 60
        if time_span > 0:
            rate_30min = (recent_30['glucose_mgdl'].iloc[-1] - recent_30['glucose_mgdl'].iloc[0]) / time_span
        else:
            rate_30min = 0
    else:
        rate_30min = 0

    # Future glucose (for training rate-of-change models)
    future_mask = window_data['timestamp'] > target_ts
    future = window_data[future_mask]

    future_glucose_15 = None
    future_glucose_30 = None

    if len(future) > 0:
        # Find glucose ~15 min in future
        future_diffs_15 = (future['timestamp'] - (target_ts + timedelta(minutes=15))).abs()
        if future_diffs_15.min().total_seconds() / 60 < 10:
            future_glucose_15 = future.loc[future_diffs_15.idxmin(), 'glucose_mgdl']

        # Find glucose ~30 min in future
        future_diffs_30 = (future['timestamp'] - (target_ts + timedelta(minutes=30))).abs()
        if future_diffs_30.min().total_seconds() / 60 < 10:
            future_glucose_30 = future.loc[future_diffs_30.idxmin(), 'glucose_mgdl']

    # Stability metrics
    std_dev = window_data['glucose_mgdl'].std()
    glucose_range = window_data['glucose_mgdl'].max() - window_data['glucose_mgdl'].min()

    # Trend classification
    if rate_15min > 1:
        trend = 2  # Rising fast
    elif rate_15min > 0.3:
        trend = 1  # Rising
    elif rate_15min < -1:
        trend = -2  # Falling fast
    elif rate_15min < -0.3:
        trend = -1  # Falling
    else:
        trend = 0  # Stable

    return {
        'glucose': closest_glucose,
        'rate_15min': rate_15min,
        'rate_30min': rate_30min,
        'std_dev': std_dev,
        'range': glucose_range,
        'trend': trend,
        'future_glucose_15': future_glucose_15,
        'future_glucose_30': future_glucose_30,
        'time_diff': time_diff,
    }


# ============================================================================
# INNOVATION 1: Multi-Offset Fusion
# ============================================================================

def multi_offset_fusion_analysis(name: str, config: dict, offsets: List[int] = [-30, -15, 0, 15, 30]):
    """
    Use voice features matched at multiple time offsets simultaneously.

    Theory: Different voice features may correlate with glucose at different lags.
    By including features from multiple offsets, we let the model learn the
    optimal combination.
    """
    print(f"\n  [INNOVATION 1] Multi-Offset Fusion for {name}")

    glucose_df = load_glucose_csv(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        return None

    # Get audio files
    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(list(dir_path.glob("*.wav")))

    # Deduplicate
    seen = set()
    unique_audio = []
    for f in audio_files:
        hash_match = re.search(r'_([a-f0-9]{8})\.waptt\.wav$', f.name.lower())
        if hash_match:
            h = hash_match.group(1)
            if h not in seen:
                seen.add(h)
                unique_audio.append(f)
        else:
            unique_audio.append(f)

    # For each audio, get glucose at multiple offsets and combine features
    data = []

    for audio_path in unique_audio:
        audio_ts = parse_audio_timestamp(audio_path.name)
        if audio_ts is None:
            continue

        features = extract_mfcc_features(audio_path)
        if features is None:
            continue

        # Get glucose at each offset
        glucose_values = []
        valid_offsets = []

        for offset in offsets:
            target_time = pd.Timestamp(audio_ts + timedelta(minutes=offset))
            diffs = (glucose_df['timestamp'] - target_time).abs()
            min_idx = diffs.idxmin()

            if diffs[min_idx].total_seconds() / 60 <= 15:
                glucose_values.append(glucose_df.loc[min_idx, 'glucose_mgdl'])
                valid_offsets.append(offset)

        if len(valid_offsets) < 3:  # Need at least 3 offsets
            continue

        # Use optimal offset's glucose as target
        optimal_offset = OPTIMAL_OFFSETS.get(name, 15)
        if optimal_offset in valid_offsets:
            target_glucose = glucose_values[valid_offsets.index(optimal_offset)]
        else:
            target_glucose = glucose_values[len(glucose_values)//2]  # Middle offset

        # Only include samples where we have ALL offsets
        if len(valid_offsets) != len(offsets):
            continue

        # Create multi-offset feature vector
        # Include glucose values at different offsets as additional context
        multi_features = np.concatenate([
            features,
            np.array(glucose_values),  # Historical glucose context (fixed length)
        ])

        data.append({
            'features': multi_features,
            'glucose': target_glucose,
            'offsets_used': valid_offsets,
        })

    if len(data) < 15:
        print(f"    Too few samples: {len(data)}")
        return None

    X = np.vstack([d['features'] for d in data])
    y = np.array([d['glucose'] for d in data])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model
    model = SVR(kernel='rbf', C=10, gamma='scale')
    cv = LeaveOneOut() if len(data) < 50 else 10

    y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
    mae = mean_absolute_error(y, y_pred)
    r, _ = stats.pearsonr(y, y_pred)

    print(f"    Multi-Offset Fusion: MAE={mae:.2f}, r={r:.3f}, n={len(data)}")

    return {
        'mae': mae,
        'r': r,
        'n_samples': len(data),
        'predictions': y_pred,
        'actual': y,
    }


# ============================================================================
# INNOVATION 2: Rate-of-Change Prediction
# ============================================================================

def rate_of_change_analysis(name: str, config: dict):
    """
    Predict glucose TREND instead of absolute value.

    Theory: Voice may be more sensitive to glucose CHANGES (sympathetic nervous
    system response to rapid drops/rises) than to absolute glucose levels.

    Clinical relevance: Knowing if glucose is rising or falling is extremely
    valuable for diabetes management decisions.
    """
    print(f"\n  [INNOVATION 2] Rate-of-Change Prediction for {name}")

    glucose_df = load_glucose_csv(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        return None

    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(list(dir_path.glob("*.wav")))

    seen = set()
    unique_audio = []
    for f in audio_files:
        hash_match = re.search(r'_([a-f0-9]{8})\.waptt\.wav$', f.name.lower())
        if hash_match:
            h = hash_match.group(1)
            if h not in seen:
                seen.add(h)
                unique_audio.append(f)
        else:
            unique_audio.append(f)

    data = []

    for audio_path in unique_audio:
        audio_ts = parse_audio_timestamp(audio_path.name)
        if audio_ts is None:
            continue

        features = extract_mfcc_features(audio_path)
        if features is None:
            continue

        # Get glucose context
        context = get_glucose_context(glucose_df, audio_ts)
        if context is None:
            continue

        data.append({
            'features': features,
            'glucose': context['glucose'],
            'rate_15min': context['rate_15min'],
            'trend': context['trend'],
            'future_glucose_15': context['future_glucose_15'],
            'future_glucose_30': context['future_glucose_30'],
        })

    if len(data) < 15:
        print(f"    Too few samples: {len(data)}")
        return None

    X = np.vstack([d['features'] for d in data])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # 1. Predict rate of change (continuous)
    y_rate = np.array([d['rate_15min'] for d in data])

    model = SVR(kernel='rbf', C=10, gamma='scale')
    cv = LeaveOneOut() if len(data) < 50 else 10

    y_pred_rate = cross_val_predict(model, X_scaled, y_rate, cv=cv)
    mae_rate = mean_absolute_error(y_rate, y_pred_rate)
    r_rate, _ = stats.pearsonr(y_rate, y_pred_rate)

    print(f"    Rate prediction: MAE={mae_rate:.3f} mg/dL/min, r={r_rate:.3f}")
    results['rate'] = {'mae': mae_rate, 'r': r_rate}

    # 2. Predict trend (classification: rising/stable/falling)
    y_trend = np.array([d['trend'] for d in data])

    # Simplify to 3 classes
    y_trend_3class = np.sign(y_trend)  # -1, 0, 1

    from sklearn.svm import SVC
    trend_model = SVC(kernel='rbf', C=10, gamma='scale')

    y_pred_trend = cross_val_predict(trend_model, X_scaled, y_trend_3class, cv=cv)
    accuracy = (y_pred_trend == y_trend_3class).mean()

    print(f"    Trend classification accuracy: {accuracy:.1%}")
    results['trend_accuracy'] = accuracy

    # 3. Predict future glucose (if available)
    future_data = [(d['features'], d['future_glucose_15']) for d in data if d['future_glucose_15'] is not None]

    if len(future_data) >= 15:
        X_future = np.vstack([f[0] for f in future_data])
        y_future = np.array([f[1] for f in future_data])

        X_future_scaled = scaler.fit_transform(X_future)

        cv_future = LeaveOneOut() if len(future_data) < 50 else 10
        y_pred_future = cross_val_predict(model, X_future_scaled, y_future, cv=cv_future)
        mae_future = mean_absolute_error(y_future, y_pred_future)
        r_future, _ = stats.pearsonr(y_future, y_pred_future)

        print(f"    Future glucose (+15min) prediction: MAE={mae_future:.2f}, r={r_future:.3f}")
        results['future_prediction'] = {'mae': mae_future, 'r': r_future}

    return results


# ============================================================================
# INNOVATION 3: Physiological Lag Estimation
# ============================================================================

def estimate_physiological_lag(name: str, config: dict):
    """
    Estimate individual's voice-glucose physiological lag using cross-correlation.

    Theory: The optimal offset varies between individuals. We can use signal
    processing techniques to find the lag that maximizes correlation between
    voice features and glucose time series.
    """
    print(f"\n  [INNOVATION 3] Physiological Lag Estimation for {name}")

    glucose_df = load_glucose_csv(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        return None

    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(list(dir_path.glob("*.wav")))

    seen = set()
    unique_audio = []
    for f in audio_files:
        hash_match = re.search(r'_([a-f0-9]{8})\.waptt\.wav$', f.name.lower())
        if hash_match:
            h = hash_match.group(1)
            if h not in seen:
                seen.add(h)
                unique_audio.append(f)
        else:
            unique_audio.append(f)

    # Extract voice features with timestamps
    voice_data = []

    for audio_path in unique_audio:
        audio_ts = parse_audio_timestamp(audio_path.name)
        if audio_ts is None:
            continue

        features = extract_mfcc_features(audio_path)
        if features is None:
            continue

        voice_data.append({
            'timestamp': pd.Timestamp(audio_ts),
            'features': features,
            # Use first MFCC as representative voice feature
            'mfcc1_mean': features[0],
            # Use energy as another representative
            'energy': features[-4],  # RMS mean
        })

    if len(voice_data) < 20:
        print(f"    Too few samples: {len(voice_data)}")
        return None

    voice_df = pd.DataFrame(voice_data).sort_values('timestamp')

    # For each offset, calculate correlation between voice and glucose
    offsets = list(range(-60, 65, 5))  # -60 to +60 minutes in 5-min steps
    correlations_mfcc = []
    correlations_energy = []

    for offset in offsets:
        corrs_mfcc = []
        corrs_energy = []

        for _, row in voice_df.iterrows():
            target_time = row['timestamp'] + timedelta(minutes=offset)

            diffs = (glucose_df['timestamp'] - target_time).abs()
            min_idx = diffs.idxmin()

            if diffs[min_idx].total_seconds() / 60 <= 15:
                glucose = glucose_df.loc[min_idx, 'glucose_mgdl']
                corrs_mfcc.append((row['mfcc1_mean'], glucose))
                corrs_energy.append((row['energy'], glucose))

        if len(corrs_mfcc) >= 15:
            x_mfcc = [c[0] for c in corrs_mfcc]
            y_mfcc = [c[1] for c in corrs_mfcc]
            r_mfcc, _ = stats.pearsonr(x_mfcc, y_mfcc)

            x_energy = [c[0] for c in corrs_energy]
            y_energy = [c[1] for c in corrs_energy]
            r_energy, _ = stats.pearsonr(x_energy, y_energy)
        else:
            r_mfcc = 0
            r_energy = 0

        correlations_mfcc.append(r_mfcc)
        correlations_energy.append(r_energy)

    # Find peak correlation
    best_offset_mfcc = offsets[np.argmax(np.abs(correlations_mfcc))]
    best_corr_mfcc = correlations_mfcc[np.argmax(np.abs(correlations_mfcc))]

    best_offset_energy = offsets[np.argmax(np.abs(correlations_energy))]
    best_corr_energy = correlations_energy[np.argmax(np.abs(correlations_energy))]

    print(f"    MFCC1 best lag: {best_offset_mfcc:+d} min (r={best_corr_mfcc:.3f})")
    print(f"    Energy best lag: {best_offset_energy:+d} min (r={best_corr_energy:.3f})")

    return {
        'offsets': offsets,
        'correlations_mfcc': correlations_mfcc,
        'correlations_energy': correlations_energy,
        'best_offset_mfcc': best_offset_mfcc,
        'best_offset_energy': best_offset_energy,
        'best_corr_mfcc': best_corr_mfcc,
        'best_corr_energy': best_corr_energy,
    }


# ============================================================================
# INNOVATION 4: Asymmetric Clinical Loss Function
# ============================================================================

def asymmetric_loss_analysis(name: str, config: dict):
    """
    Train model with asymmetric loss that penalizes dangerous errors more.

    Theory: In diabetes management, predicting glucose as LOW when it's actually
    HIGH is more dangerous (missed hyperglycemia), and predicting HIGH when LOW
    is also dangerous (missed hypoglycemia, potential insulin overdose).

    We use quantile regression to get prediction intervals.
    """
    print(f"\n  [INNOVATION 4] Asymmetric Clinical Loss for {name}")

    glucose_df = load_glucose_csv(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        return None

    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(list(dir_path.glob("*.wav")))

    seen = set()
    unique_audio = []
    for f in audio_files:
        hash_match = re.search(r'_([a-f0-9]{8})\.waptt\.wav$', f.name.lower())
        if hash_match:
            h = hash_match.group(1)
            if h not in seen:
                seen.add(h)
                unique_audio.append(f)
        else:
            unique_audio.append(f)

    data = []
    optimal_offset = OPTIMAL_OFFSETS.get(name, 15)

    for audio_path in unique_audio:
        audio_ts = parse_audio_timestamp(audio_path.name)
        if audio_ts is None:
            continue

        features = extract_mfcc_features(audio_path)
        if features is None:
            continue

        target_time = pd.Timestamp(audio_ts + timedelta(minutes=optimal_offset))
        diffs = (glucose_df['timestamp'] - target_time).abs()
        min_idx = diffs.idxmin()

        if diffs[min_idx].total_seconds() / 60 <= 15:
            glucose = glucose_df.loc[min_idx, 'glucose_mgdl']
            data.append({'features': features, 'glucose': glucose})

    if len(data) < 15:
        print(f"    Too few samples: {len(data)}")
        return None

    X = np.vstack([d['features'] for d in data])
    y = np.array([d['glucose'] for d in data])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Quantile regression for prediction intervals
    from sklearn.ensemble import GradientBoostingRegressor

    # Low quantile (10th percentile) - lower bound
    model_low = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=100, random_state=42)
    # Median (50th percentile) - point estimate
    model_med = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100, random_state=42)
    # High quantile (90th percentile) - upper bound
    model_high = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=100, random_state=42)

    cv = KFold(n_splits=5, shuffle=True, random_state=42) if len(data) >= 25 else LeaveOneOut()

    y_pred_low = cross_val_predict(model_low, X_scaled, y, cv=cv)
    y_pred_med = cross_val_predict(model_med, X_scaled, y, cv=cv)
    y_pred_high = cross_val_predict(model_high, X_scaled, y, cv=cv)

    mae_med = mean_absolute_error(y, y_pred_med)
    r_med, _ = stats.pearsonr(y, y_pred_med)

    # Calculate coverage of prediction intervals
    coverage = ((y >= y_pred_low) & (y <= y_pred_high)).mean()
    interval_width = (y_pred_high - y_pred_low).mean()

    print(f"    Median prediction: MAE={mae_med:.2f}, r={r_med:.3f}")
    print(f"    80% interval coverage: {coverage:.1%} (width={interval_width:.1f} mg/dL)")

    # Clinical safety analysis
    # Count dangerous errors
    hypoglycemia_threshold = 70  # mg/dL
    hyperglycemia_threshold = 180  # mg/dL

    missed_hypos = ((y < hypoglycemia_threshold) & (y_pred_med >= hypoglycemia_threshold)).sum()
    missed_hypers = ((y > hyperglycemia_threshold) & (y_pred_med <= hyperglycemia_threshold)).sum()

    actual_hypos = (y < hypoglycemia_threshold).sum()
    actual_hypers = (y > hyperglycemia_threshold).sum()

    print(f"    Missed hypoglycemia: {missed_hypos}/{actual_hypos}")
    print(f"    Missed hyperglycemia: {missed_hypers}/{actual_hypers}")

    return {
        'mae': mae_med,
        'r': r_med,
        'coverage': coverage,
        'interval_width': interval_width,
        'predictions_low': y_pred_low,
        'predictions_med': y_pred_med,
        'predictions_high': y_pred_high,
        'actual': y,
    }


# ============================================================================
# INNOVATION 5: Ensemble with Offset Weighting
# ============================================================================

def offset_weighted_ensemble(name: str, config: dict):
    """
    Create an ensemble that weights models trained at different offsets
    based on their historical performance.

    Theory: Instead of picking one "optimal" offset, we can combine predictions
    from models at multiple offsets, weighted by their reliability.
    """
    print(f"\n  [INNOVATION 5] Offset-Weighted Ensemble for {name}")

    glucose_df = load_glucose_csv(config['glucose_csv'], config['glucose_unit'])
    if glucose_df.empty:
        return None

    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(list(dir_path.glob("*.wav")))

    seen = set()
    unique_audio = []
    for f in audio_files:
        hash_match = re.search(r'_([a-f0-9]{8})\.waptt\.wav$', f.name.lower())
        if hash_match:
            h = hash_match.group(1)
            if h not in seen:
                seen.add(h)
                unique_audio.append(f)
        else:
            unique_audio.append(f)

    # Offsets to use in ensemble
    offsets = [-30, -15, 0, 15, 30]

    # Collect data for each offset
    offset_data = {offset: [] for offset in offsets}

    for audio_path in unique_audio:
        audio_ts = parse_audio_timestamp(audio_path.name)
        if audio_ts is None:
            continue

        features = extract_mfcc_features(audio_path)
        if features is None:
            continue

        for offset in offsets:
            target_time = pd.Timestamp(audio_ts + timedelta(minutes=offset))
            diffs = (glucose_df['timestamp'] - target_time).abs()
            min_idx = diffs.idxmin()

            if diffs[min_idx].total_seconds() / 60 <= 15:
                glucose = glucose_df.loc[min_idx, 'glucose_mgdl']
                offset_data[offset].append({
                    'features': features,
                    'glucose': glucose,
                    'audio_path': str(audio_path),
                })

    # Find common samples across all offsets
    common_paths = set.intersection(*[
        set(d['audio_path'] for d in offset_data[o])
        for o in offsets if len(offset_data[o]) > 0
    ])

    if len(common_paths) < 15:
        print(f"    Too few common samples: {len(common_paths)}")
        return None

    # Build aligned datasets
    aligned_data = {offset: {} for offset in offsets}
    for offset in offsets:
        for d in offset_data[offset]:
            if d['audio_path'] in common_paths:
                aligned_data[offset][d['audio_path']] = d

    # Create feature matrices and train models
    scaler = StandardScaler()

    # Use first offset's features as base (features are same, just glucose differs)
    base_offset = offsets[0]
    sample_paths = sorted(common_paths)

    X = np.vstack([aligned_data[base_offset][p]['features'] for p in sample_paths])
    X_scaled = scaler.fit_transform(X)

    # Train model at each offset and get predictions
    offset_predictions = {}
    offset_weights = {}

    for offset in offsets:
        y_offset = np.array([aligned_data[offset][p]['glucose'] for p in sample_paths])

        model = SVR(kernel='rbf', C=10, gamma='scale')
        cv = LeaveOneOut() if len(sample_paths) < 50 else 10

        y_pred = cross_val_predict(model, X_scaled, y_offset, cv=cv)
        mae = mean_absolute_error(y_offset, y_pred)

        offset_predictions[offset] = y_pred
        # Weight inversely proportional to MAE
        offset_weights[offset] = 1.0 / (mae + 1)

    # Normalize weights
    total_weight = sum(offset_weights.values())
    for offset in offsets:
        offset_weights[offset] /= total_weight

    print(f"    Offset weights: " + ", ".join([f"{o:+d}min:{w:.2f}" for o, w in offset_weights.items()]))

    # Ensemble prediction
    y_ensemble = np.zeros(len(sample_paths))
    for offset in offsets:
        y_ensemble += offset_weights[offset] * offset_predictions[offset]

    # Use optimal offset's glucose as ground truth
    optimal_offset = OPTIMAL_OFFSETS.get(name, 15)
    if optimal_offset in offsets:
        y_true = np.array([aligned_data[optimal_offset][p]['glucose'] for p in sample_paths])
    else:
        y_true = np.array([aligned_data[0][p]['glucose'] for p in sample_paths])

    mae_ensemble = mean_absolute_error(y_true, y_ensemble)
    r_ensemble, _ = stats.pearsonr(y_true, y_ensemble)

    # Compare to single best offset
    best_single_mae = min(mean_absolute_error(
        np.array([aligned_data[offset][p]['glucose'] for p in sample_paths]),
        offset_predictions[offset]
    ) for offset in offsets)

    print(f"    Ensemble: MAE={mae_ensemble:.2f}, r={r_ensemble:.3f}")
    print(f"    Best single offset MAE: {best_single_mae:.2f}")
    print(f"    Improvement: {(best_single_mae - mae_ensemble):.2f} mg/dL ({(best_single_mae - mae_ensemble)/best_single_mae*100:.1f}%)")

    return {
        'mae_ensemble': mae_ensemble,
        'r_ensemble': r_ensemble,
        'best_single_mae': best_single_mae,
        'offset_weights': offset_weights,
        'n_samples': len(sample_paths),
    }


# ============================================================================
# MAIN
# ============================================================================

def run_innovative_analysis():
    """Run all innovative analysis methods."""

    print("="*70)
    print("INNOVATIVE VOICE-GLUCOSE ANALYSIS")
    print("="*70)

    all_results = {}

    # Focus on participants with most data
    focus_participants = ["Wolf", "Margarita", "Anja", "Lara"]

    for name in focus_participants:
        if name not in PARTICIPANTS:
            continue

        config = PARTICIPANTS[name]
        print(f"\n{'='*70}")
        print(f"PARTICIPANT: {name}")
        print(f"{'='*70}")

        results = {}

        # Innovation 1: Multi-Offset Fusion
        results['multi_offset'] = multi_offset_fusion_analysis(name, config)

        # Innovation 2: Rate-of-Change Prediction
        results['rate_of_change'] = rate_of_change_analysis(name, config)

        # Innovation 3: Physiological Lag Estimation
        results['lag_estimation'] = estimate_physiological_lag(name, config)

        # Innovation 4: Asymmetric Clinical Loss
        results['asymmetric'] = asymmetric_loss_analysis(name, config)

        # Innovation 5: Offset-Weighted Ensemble
        results['ensemble'] = offset_weighted_ensemble(name, config)

        all_results[name] = results

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    generate_innovative_visualizations(all_results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nBest Results by Method:")
    for name, results in all_results.items():
        print(f"\n  {name}:")
        if results.get('multi_offset'):
            print(f"    Multi-Offset Fusion: MAE={results['multi_offset']['mae']:.2f}")
        if results.get('rate_of_change') and 'rate' in results['rate_of_change']:
            print(f"    Rate Prediction r={results['rate_of_change']['rate']['r']:.3f}")
        if results.get('asymmetric'):
            print(f"    Quantile Regression: MAE={results['asymmetric']['mae']:.2f}, 80% coverage={results['asymmetric']['coverage']:.1%}")
        if results.get('ensemble'):
            print(f"    Ensemble: MAE={results['ensemble']['mae_ensemble']:.2f} (vs single best {results['ensemble']['best_single_mae']:.2f})")

    return all_results


def generate_innovative_visualizations(results: dict):
    """Generate visualizations for innovative methods."""

    # 1. Lag estimation curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        if result.get('lag_estimation'):
            lag = result['lag_estimation']
            ax.plot(lag['offsets'], lag['correlations_mfcc'], 'b-', label='MFCC1', linewidth=2)
            ax.plot(lag['offsets'], lag['correlations_energy'], 'r--', label='Energy', linewidth=2)
            ax.axvline(lag['best_offset_mfcc'], color='b', linestyle=':', alpha=0.5)
            ax.axvline(lag['best_offset_energy'], color='r', linestyle=':', alpha=0.5)
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Offset (minutes)')
            ax.set_ylabel('Correlation with Glucose')
            ax.set_title(f'{name}\nBest lag: MFCC={lag["best_offset_mfcc"]:+d}min, Energy={lag["best_offset_energy"]:+d}min')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lag_estimation.png', dpi=150)
    plt.close()
    print("  Saved: lag_estimation.png")

    # 2. Prediction intervals
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        if result.get('asymmetric'):
            asym = result['asymmetric']
            y = asym['actual']
            y_low = asym['predictions_low']
            y_med = asym['predictions_med']
            y_high = asym['predictions_high']

            # Sort by actual value for visualization
            sort_idx = np.argsort(y)
            x = np.arange(len(y))

            ax.fill_between(x, y_low[sort_idx], y_high[sort_idx], alpha=0.3, color='blue', label='80% PI')
            ax.plot(x, y_med[sort_idx], 'b-', label='Predicted', linewidth=1)
            ax.scatter(x, y[sort_idx], c='red', s=20, alpha=0.7, label='Actual', zorder=5)

            ax.set_xlabel('Sample (sorted by actual glucose)')
            ax.set_ylabel('Glucose (mg/dL)')
            ax.set_title(f'{name}: MAE={asym["mae"]:.1f}, Coverage={asym["coverage"]:.0%}')
            ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'prediction_intervals.png', dpi=150)
    plt.close()
    print("  Saved: prediction_intervals.png")

    # 3. Method comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Standard', 'Multi-Offset', 'Ensemble', 'Quantile']
    x = np.arange(len(results))
    width = 0.2

    for i, method in enumerate(methods):
        maes = []
        for name, result in results.items():
            if method == 'Standard':
                # Get from enhanced analysis results
                maes.append(None)  # We don't have this stored
            elif method == 'Multi-Offset' and result.get('multi_offset'):
                maes.append(result['multi_offset']['mae'])
            elif method == 'Ensemble' and result.get('ensemble'):
                maes.append(result['ensemble']['mae_ensemble'])
            elif method == 'Quantile' and result.get('asymmetric'):
                maes.append(result['asymmetric']['mae'])
            else:
                maes.append(None)

        valid_maes = [m if m is not None else 0 for m in maes]
        ax.bar(x + i * width, valid_maes, width, label=method)

    ax.set_ylabel('MAE (mg/dL)')
    ax.set_title('MAE Comparison Across Innovative Methods')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results.keys())
    ax.legend()
    ax.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='10 mg/dL target')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'method_comparison.png', dpi=150)
    plt.close()
    print("  Saved: method_comparison.png")


if __name__ == "__main__":
    results = run_innovative_analysis()
