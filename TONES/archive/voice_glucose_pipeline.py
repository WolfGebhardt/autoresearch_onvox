"""
Voice-Based Glucose Estimation Pipeline
========================================
Aligns WhatsApp voice messages with CGM glucose readings and builds
personalized models for glucose estimation from voice features.

Requirements:
    pip install pandas numpy librosa opensmile scikit-learn scipy matplotlib seaborn

Optional (for advanced features):
    pip install torch torchaudio transformers  # For Wav2Vec2 embeddings
    pip install praat-parselmouth              # For jitter/shimmer
"""

import os
import re
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Audio processing
import librosa

# Machine learning
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

# Try to import optional dependencies
try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    print("OpenSMILE not available. Install with: pip install opensmile")

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Parselmouth not available. Install with: pip install praat-parselmouth")


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(r"C:\Users\whgeb\OneDrive\TONES")

# Participant mapping: folder name -> configuration
# Now supports:
# - Multiple audio_dirs (list)
# - Multiple glucose_csv files (list) - will be merged
# - Auto glucose unit detection
# - Annotated audio directories
PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mg/dL",
        "glucose_in_filename": True,  # Glucose encoded as prefix: "131_WhatsApp..."
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchütt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav"],
        "annotated_audio": "Sybille/annotated_audio",  # Pre-labeled with glucose in subfolder names
        "audio_ext": ".wav",
        "glucose_unit": "mg/dL",
        "glucose_in_filename": False,
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],  # Converted WAV files
        "audio_ext": ".wav",
        "glucose_unit": "mmol/L",
        "glucose_in_filename": False,
    },
    "Anja": {
        "glucose_csv": [  # Multiple CSVs to merge
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_6-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_10-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_13-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_16-11-2023.csv",
        ],
        "audio_dirs": ["Anja/conv_audio", "Anja/converted audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mg/dL",
        "glucose_in_filename": False,
    },
    "Vicky": {
        "glucose_csv": ["Vicky/Number_10Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Vicky/conv_audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mmol/L",
        "glucose_in_filename": False,
    },
    "Steffen_Haeseli": {
        "glucose_csv": ["Steffen_Haeseli/Number_2Nov_23_glucose_4-1-2024.csv"],
        "audio_dirs": ["Steffen_Haeseli/wav"],
        "audio_ext": ".wav",
        "glucose_unit": "mmol/L",
        "glucose_in_filename": False,
    },
    "Lara": {
        "glucose_csv": ["Lara/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Lara/conv_audio"],
        "audio_ext": ".wav",
        "glucose_unit": "mmol/L",
        "glucose_in_filename": False,
    },
}

# Time window for matching audio to glucose (in minutes)
MATCH_WINDOW_MINUTES = 15


# =============================================================================
# FILENAME PARSING
# =============================================================================

def parse_glucose_from_filename(filename: str) -> Optional[float]:
    """
    Parse glucose value from filename prefix.
    Format: '131_WhatsApp Audio...' -> 131.0 (mg/dL)
    Also handles: 'AnyConv.com__131_WhatsApp...' and 'Wolf_131 WhatsApp...'
    """
    # Try various patterns
    patterns = [
        r'^(\d{2,3})_',                          # 131_WhatsApp...
        r'AnyConv\.com__(\d{2,3})_',             # AnyConv.com__131_WhatsApp...
        r'Wolf[_\s]+(\d{2,3})[_\s]+WhatsApp',    # Wolf_131 WhatsApp... or Wolf 131 WhatsApp...
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            glucose = float(match.group(1))
            # Sanity check: glucose should be in realistic range (40-400 mg/dL)
            if 40 <= glucose <= 400:
                return glucose

    return None


def parse_whatsapp_timestamp(filename: str) -> Optional[datetime]:
    """
    Parse timestamp from WhatsApp audio filename.
    Format: 'WhatsApp Audio 2023-11-05 um 17.57.26_102980fd.wav'
    """
    # Pattern for German WhatsApp format
    pattern = r'WhatsApp[- ]Audio[- ](\d{4}-\d{2}-\d{2})[- ]um[- ](\d{2})\.(\d{2})\.(\d{2})'
    match = re.search(pattern, filename)

    if match:
        date_str = match.group(1)
        hour = match.group(2)
        minute = match.group(3)
        second = match.group(4)

        try:
            dt = datetime.strptime(f"{date_str} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
            return dt
        except ValueError:
            pass

    return None


def detect_glucose_unit(df: pd.DataFrame) -> str:
    """
    Auto-detect glucose unit from column headers or values.
    """
    # Check column headers for unit indication
    for col in df.columns:
        col_lower = col.lower()
        if 'mmol/l' in col_lower:
            return 'mmol/L'
        if 'mg/dl' in col_lower:
            return 'mg/dL'

    # Fallback: check value ranges
    # mmol/L typically 2-20, mg/dL typically 40-400
    glucose_col = None
    for col in df.columns:
        if 'glucose' in col.lower() or 'glukose' in col.lower():
            glucose_col = col
            break

    if glucose_col:
        values = pd.to_numeric(df[glucose_col], errors='coerce').dropna()
        if len(values) > 0:
            mean_val = values.mean()
            if mean_val < 30:  # Likely mmol/L
                return 'mmol/L'
            else:  # Likely mg/dL
                return 'mg/dL'

    return 'mg/dL'  # Default


def load_glucose_data(csv_path: Path, unit: str = "auto") -> pd.DataFrame:
    """
    Load glucose CSV and parse timestamps.
    Handles both English and German column names.
    Auto-detects glucose unit if unit='auto'.
    Handles both Libre 3 (1 header row) and Libre Pro (2 header rows) formats.
    """
    # First, peek at the file to determine format
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_lines = [f.readline() for _ in range(5)]

    # Find the row that contains the actual column headers (Device, Serial Number, Timestamp)
    skiprows = 1  # Default: skip 1 header row (Libre 3 format)
    for i, line in enumerate(first_lines):
        line_lower = line.lower()
        if 'device' in line_lower and 'timestamp' in line_lower:
            skiprows = i
            break

    df = pd.read_csv(csv_path, skiprows=skiprows)

    # Auto-detect unit if needed
    if unit == "auto":
        unit = detect_glucose_unit(df)

    # Find timestamp column
    timestamp_col = None
    glucose_col = None

    for col in df.columns:
        if 'timestamp' in col.lower() or 'zeitstempel' in col.lower():
            timestamp_col = col
        if 'historic glucose' in col.lower() or 'glukosewert' in col.lower():
            glucose_col = col

    if timestamp_col is None and len(df.columns) > 2:
        # Try by position (usually column 3)
        timestamp_col = df.columns[2]
    if glucose_col is None and len(df.columns) > 4:
        # Try by position (usually column 5)
        glucose_col = df.columns[4]

    if timestamp_col is None or glucose_col is None:
        print(f"  Warning: Could not find timestamp/glucose columns in {csv_path.name}")
        print(f"  Columns: {list(df.columns)}")
        return pd.DataFrame(columns=['timestamp', 'glucose', 'glucose_mgdl'])

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
    df['glucose'] = pd.to_numeric(df[glucose_col], errors='coerce')

    # Convert to mg/dL if needed (for consistency)
    if unit == "mmol/L":
        df['glucose_mgdl'] = df['glucose'] * 18.0182
    else:
        df['glucose_mgdl'] = df['glucose']

    # Drop rows with missing data
    df = df.dropna(subset=['timestamp', 'glucose'])

    return df[['timestamp', 'glucose', 'glucose_mgdl']].sort_values('timestamp')


def load_multiple_glucose_csvs(csv_paths: List[Path], unit: str = "auto") -> pd.DataFrame:
    """
    Load and merge multiple glucose CSV files.
    Removes duplicates based on timestamp.
    """
    all_dfs = []

    for csv_path in csv_paths:
        if csv_path.exists():
            df = load_glucose_data(csv_path, unit)
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    # Concatenate all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)

    # Remove duplicates based on timestamp (keep first)
    combined = combined.drop_duplicates(subset=['timestamp'], keep='first')

    # Sort by timestamp
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    return combined


def find_matching_glucose(audio_timestamp: datetime,
                          glucose_df: pd.DataFrame,
                          window_minutes: int = 15) -> Tuple[Optional[float], Optional[float]]:
    """
    Find glucose reading closest to audio timestamp within window.
    Returns (glucose_value, time_difference_minutes).
    """
    window = timedelta(minutes=window_minutes)

    # Filter to readings within window
    mask = (glucose_df['timestamp'] >= audio_timestamp - window) & \
           (glucose_df['timestamp'] <= audio_timestamp + window)

    candidates = glucose_df[mask]

    if candidates.empty:
        return None, None

    # Find closest reading
    time_diffs = abs(candidates['timestamp'] - audio_timestamp)
    closest_idx = time_diffs.idxmin()
    closest_row = candidates.loc[closest_idx]

    time_diff_minutes = time_diffs.loc[closest_idx].total_seconds() / 60

    return closest_row['glucose_mgdl'], time_diff_minutes


# =============================================================================
# AUDIO FEATURE EXTRACTION
# =============================================================================

def extract_librosa_features(audio_path: str, sr: int = 16000) -> Dict[str, float]:
    """
    Extract acoustic features using librosa.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return {}

    # Skip very short audio
    if len(y) < sr * 0.5:  # Less than 0.5 seconds
        return {}

    features = {}

    # --- Pitch (F0) ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    if pitch_values:
        features['pitch_mean'] = np.mean(pitch_values)
        features['pitch_std'] = np.std(pitch_values)
        features['pitch_min'] = np.min(pitch_values)
        features['pitch_max'] = np.max(pitch_values)
        features['pitch_range'] = features['pitch_max'] - features['pitch_min']

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

    # MFCC deltas
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    for i in range(13):
        features[f'mfcc_delta_{i+1}_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_delta2_{i+1}_mean'] = np.mean(mfcc_delta2[i])

    # --- Energy/RMS ---
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_max'] = np.max(rms)

    # --- Spectral features ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)

    # --- Zero crossing rate ---
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # --- Tempo/rhythm ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])

    # --- Duration ---
    features['duration'] = len(y) / sr

    return features


def extract_parselmouth_features(audio_path: str) -> Dict[str, float]:
    """
    Extract jitter, shimmer, and HNR using Parselmouth (Praat).
    These are important voice quality measures affected by physiological state.
    """
    if not PARSELMOUTH_AVAILABLE:
        return {}

    try:
        sound = parselmouth.Sound(audio_path)

        # Get pitch
        pitch = call(sound, "To Pitch", 0.0, 75, 600)

        # Get point process for jitter/shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

        features = {}

        # Jitter (cycle-to-cycle variation in pitch period)
        features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features['jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)

        # Shimmer (cycle-to-cycle variation in amplitude)
        features['shimmer_local'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['shimmer_apq3'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        # Harmonics-to-Noise Ratio
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        features['hnr'] = call(harmonicity, "Get mean", 0, 0)

        return features

    except Exception as e:
        print(f"Parselmouth error for {audio_path}: {e}")
        return {}


def extract_opensmile_features(audio_path: str) -> Dict[str, float]:
    """
    Extract eGeMAPS features using OpenSMILE.
    eGeMAPS is a standardized set of 88 acoustic features for voice analysis.
    """
    if not OPENSMILE_AVAILABLE:
        return {}

    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        features_df = smile.process_file(audio_path)

        # Convert to dictionary
        return features_df.iloc[0].to_dict()

    except Exception as e:
        print(f"OpenSMILE error for {audio_path}: {e}")
        return {}


def extract_all_features(audio_path: str) -> Dict[str, float]:
    """
    Extract all available features from an audio file.
    """
    features = {}

    # Librosa features (always available)
    librosa_feats = extract_librosa_features(audio_path)
    features.update({f'librosa_{k}': v for k, v in librosa_feats.items()})

    # Parselmouth features (if available)
    praat_feats = extract_parselmouth_features(audio_path)
    features.update({f'praat_{k}': v for k, v in praat_feats.items()})

    # OpenSMILE features (if available)
    opensmile_feats = extract_opensmile_features(audio_path)
    features.update({f'opensmile_{k}': v for k, v in opensmile_feats.items()})

    return features


# =============================================================================
# DATA ALIGNMENT & DATASET CREATION
# =============================================================================

def extract_circadian_features(audio_timestamp: datetime) -> Dict[str, float]:
    """
    Extract circadian rhythm features from timestamp.
    Voice and glucose both have circadian patterns.
    """
    hour = audio_timestamp.hour
    minute = audio_timestamp.minute

    return {
        'hour_of_day': hour,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'is_morning': 1.0 if 6 <= hour < 12 else 0.0,
        'is_afternoon': 1.0 if 12 <= hour < 18 else 0.0,
        'is_evening': 1.0 if 18 <= hour < 22 else 0.0,
        'is_night': 1.0 if hour >= 22 or hour < 6 else 0.0,
        'minutes_since_midnight': hour * 60 + minute,
    }


def create_dataset_for_participant(participant_name: str,
                                   config: dict,
                                   window_minutes: int = 15,
                                   verbose: bool = True) -> pd.DataFrame:
    """
    Create aligned audio-glucose dataset for a single participant.

    Supports:
    1. glucose_in_filename=True: Glucose extracted from filename prefix (e.g., Wolf)
    2. glucose_in_filename=False: Match audio timestamp to CGM readings
    3. Multiple audio directories
    4. Multiple glucose CSV files (merged)
    """
    glucose_in_filename = config.get('glucose_in_filename', False)

    # Load glucose data - now supports multiple CSVs
    glucose_csv_paths = config.get('glucose_csv', [])
    if isinstance(glucose_csv_paths, str):
        glucose_csv_paths = [glucose_csv_paths]

    glucose_df = None
    csv_paths = [BASE_DIR / p for p in glucose_csv_paths]
    existing_csvs = [p for p in csv_paths if p.exists()]

    if existing_csvs:
        glucose_df = load_multiple_glucose_csvs(existing_csvs, config.get('glucose_unit', 'auto'))
        if verbose:
            print(f"\n{'='*60}")
            print(f"Participant: {participant_name}")
            print(f"Glucose CSVs loaded: {len(existing_csvs)}")
            print(f"Glucose readings: {len(glucose_df)}")
            print(f"Date range: {glucose_df['timestamp'].min()} to {glucose_df['timestamp'].max()}")
            print(f"Glucose range: {glucose_df['glucose_mgdl'].min():.1f} - {glucose_df['glucose_mgdl'].max():.1f} mg/dL")
    elif not glucose_in_filename:
        print(f"No glucose files found for {participant_name}")
        return pd.DataFrame()

    # Find audio files from all directories
    audio_dirs = config.get('audio_dirs', [])
    if isinstance(audio_dirs, str):
        audio_dirs = [audio_dirs]

    # Also check for legacy 'audio_dir' key
    if not audio_dirs and 'audio_dir' in config:
        audio_dirs = [config['audio_dir']]

    audio_files = []
    for audio_dir_path in audio_dirs:
        audio_dir = BASE_DIR / audio_dir_path
        if audio_dir.exists():
            audio_pattern = f"*{config['audio_ext']}"
            audio_files.extend(list(audio_dir.glob(audio_pattern)))

    if verbose:
        print(f"Audio files found: {len(audio_files)}")
        print(f"Glucose from filename: {glucose_in_filename}")

    # Process each audio file
    records = []
    matched = 0
    skipped_no_glucose = 0
    skipped_no_timestamp = 0

    # Track unique files to avoid duplicates (some files are duplicated with different naming)
    processed_hashes = set()

    for audio_path in audio_files:
        # Skip non-WAV files
        if config['audio_ext'] != '.wav':
            continue

        # Parse timestamp from filename
        audio_ts = parse_whatsapp_timestamp(audio_path.name)

        if audio_ts is None:
            skipped_no_timestamp += 1
            continue

        # Get glucose value
        if glucose_in_filename:
            # Extract glucose from filename prefix
            glucose_val = parse_glucose_from_filename(audio_path.name)
            time_diff = 0.0  # Assumed to be at exact time

            if glucose_val is None:
                skipped_no_glucose += 1
                continue
        else:
            # Match to CGM reading
            if glucose_df is None:
                continue

            glucose_val, time_diff = find_matching_glucose(audio_ts, glucose_df, window_minutes)

            if glucose_val is None:
                skipped_no_glucose += 1
                continue

        # Create a simple hash to detect duplicate content
        # Using timestamp + rounded glucose as proxy
        content_hash = f"{audio_ts.strftime('%Y%m%d%H%M')}_{int(glucose_val)}"
        if content_hash in processed_hashes:
            continue
        processed_hashes.add(content_hash)

        matched += 1

        # Extract features
        features = extract_all_features(str(audio_path))

        if not features:
            continue

        # Add circadian features
        circadian_feats = extract_circadian_features(audio_ts)
        features.update({f'circadian_{k}': v for k, v in circadian_feats.items()})

        record = {
            'participant': participant_name,
            'audio_file': audio_path.name,
            'audio_timestamp': audio_ts,
            'glucose_mgdl': glucose_val,
            'time_diff_minutes': time_diff,
            **features
        }
        records.append(record)

    if verbose:
        print(f"Matched audio-glucose pairs: {matched}")
        print(f"Skipped (no glucose match): {skipped_no_glucose}")
        print(f"Skipped (no timestamp): {skipped_no_timestamp}")
        print(f"Successfully processed: {len(records)}")

    return pd.DataFrame(records)


# =============================================================================
# MODEL TRAINING & EVALUATION
# =============================================================================

def train_personalized_model(df: pd.DataFrame,
                             feature_prefix: str = 'librosa_',
                             model_type: str = 'ridge',
                             verbose: bool = True) -> dict:
    """
    Train and evaluate a personalized glucose prediction model.
    Uses Leave-One-Out cross-validation for small datasets.
    """
    # Select features
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]

    if len(feature_cols) == 0:
        print(f"No features found with prefix '{feature_prefix}'")
        return {}

    X = df[feature_cols].values
    y = df['glucose_mgdl'].values

    # Handle missing/infinite values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select model
    if model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'elasticnet':
        model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    elif model_type == 'gbm':
        model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    else:
        model = Ridge(alpha=1.0)

    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X_scaled, y, cv=loo)

    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # Correlation
    pearson_r, pearson_p = pearsonr(y, y_pred)
    spearman_r, spearman_p = spearmanr(y, y_pred)

    results = {
        'n_samples': len(y),
        'n_features': len(feature_cols),
        'mae_mgdl': mae,
        'rmse_mgdl': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'y_true': y,
        'y_pred': y_pred,
        'feature_cols': feature_cols,
    }

    if verbose:
        print(f"\n--- Model Results ({model_type}) ---")
        print(f"Samples: {len(y)}, Features: {len(feature_cols)}")
        print(f"MAE: {mae:.2f} mg/dL")
        print(f"RMSE: {rmse:.2f} mg/dL")
        print(f"R²: {r2:.3f}")
        print(f"Pearson r: {pearson_r:.3f} (p={pearson_p:.4f})")
        print(f"Spearman r: {spearman_r:.3f} (p={spearman_p:.4f})")

    # Fit final model on all data
    model.fit(X_scaled, y)
    results['model'] = model
    results['scaler'] = scaler

    return results


def clarke_error_grid_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate Clarke Error Grid zones for glucose predictions.
    Zone A: Clinically accurate
    Zone B: Benign errors
    Zone C-E: Potentially dangerous errors
    """
    n = len(y_true)
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for ref, pred in zip(y_true, y_pred):
        if (ref <= 70 and pred <= 70) or \
           (ref >= 70 and pred >= 70 and abs(pred - ref) <= 0.2 * ref):
            zones['A'] += 1
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zones['E'] += 1
        elif (ref >= 70 and ref <= 290 and pred >= ref + 110) or \
             (ref >= 130 and ref <= 180 and pred <= 0.5 * (ref - 130)):
            zones['C'] += 1
        elif (ref >= 240 and pred <= 70 and pred >= 0.5 * (ref - 240)) or \
             (ref <= 175/3 and pred <= 180 and pred >= 70):
            zones['D'] += 1
        else:
            zones['B'] += 1

    # Convert to percentages
    zones_pct = {k: v / n * 100 for k, v in zones.items()}

    return {
        'zones_count': zones,
        'zones_pct': zones_pct,
        'clinically_acceptable': zones_pct['A'] + zones_pct['B']
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(participants: List[str] = None,
                 window_minutes: int = 15,
                 model_type: str = 'ridge',
                 optimize_offset: bool = False) -> dict:
    """
    Run the full pipeline for specified participants.

    Args:
        participants: List of participant names (None = all)
        window_minutes: Time window for glucose matching
        model_type: ML model to use ('ridge', 'rf', 'gbm', etc.)
        optimize_offset: If True, run offset optimization to find best alignment
    """
    if participants is None:
        participants = list(PARTICIPANTS.keys())

    all_results = {}

    for participant in participants:
        if participant not in PARTICIPANTS:
            print(f"Unknown participant: {participant}")
            continue

        config = PARTICIPANTS[participant]

        # Skip if audio needs conversion
        if config['audio_ext'] != '.wav':
            print(f"\n{participant}: Audio files need conversion to WAV first")
            continue

        # Create dataset
        df = create_dataset_for_participant(participant, config, window_minutes)

        if len(df) < 10:
            print(f"\n{participant}: Insufficient data ({len(df)} samples)")
            continue

        # Optional: Offset optimization
        offset_results = None
        if optimize_offset and not config.get('glucose_in_filename', False):
            try:
                from offset_optimization import run_offset_optimization

                glucose_path = BASE_DIR / config['glucose_csv']
                glucose_df = load_glucose_data(glucose_path, config['glucose_unit'])

                offset_results = run_offset_optimization(
                    participant, df, glucose_df, BASE_DIR
                )
            except Exception as e:
                print(f"Offset optimization failed: {e}")

        # Train model
        results = train_personalized_model(df, model_type=model_type)

        if results:
            # Clarke Error Grid analysis
            ceg = clarke_error_grid_analysis(results['y_true'], results['y_pred'])
            results['clarke_error_grid'] = ceg
            print(f"Clarke Error Grid - Zones A+B: {ceg['clinically_acceptable']:.1f}%")

        all_results[participant] = {
            'dataset': df,
            'model_results': results,
            'offset_optimization': offset_results
        }

    return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(results: dict, participant: str):
    """
    Plot model predictions vs actual glucose values.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")
        return

    if participant not in results:
        print(f"No results for {participant}")
        return

    model_results = results[participant]['model_results']
    y_true = model_results['y_true']
    y_pred = model_results['y_pred']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                  'r--', label='Perfect prediction')
    axes[0].set_xlabel('Actual Glucose (mg/dL)')
    axes[0].set_ylabel('Predicted Glucose (mg/dL)')
    axes[0].set_title(f'{participant}: Predicted vs Actual')
    axes[0].legend()

    # Residual plot
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=20, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Prediction Error (mg/dL)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'{participant}: Error Distribution')

    plt.tight_layout()
    plt.savefig(BASE_DIR / f'{participant}_results.png', dpi=150)
    plt.show()


# =============================================================================
# UTILITY: AUDIO CONVERSION
# =============================================================================

def convert_waptt_to_wav(input_dir: Path, output_dir: Path):
    """
    Convert WhatsApp .waptt files to .wav using ffmpeg.
    Requires ffmpeg to be installed and in PATH.

    Usage:
        convert_waptt_to_wav(
            Path("C:/Users/whgeb/OneDrive/TONES/Margarita"),
            Path("C:/Users/whgeb/OneDrive/TONES/Margarita/conv_audio")
        )
    """
    import subprocess

    output_dir.mkdir(exist_ok=True)

    waptt_files = list(input_dir.glob("*.waptt"))
    print(f"Found {len(waptt_files)} .waptt files to convert")

    for waptt_file in waptt_files:
        wav_file = output_dir / waptt_file.with_suffix('.wav').name

        if wav_file.exists():
            continue

        cmd = ['ffmpeg', '-i', str(waptt_file), '-ar', '16000', '-ac', '1', str(wav_file)]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Converted: {waptt_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {waptt_file.name}: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Voice-Based Glucose Estimation Pipeline")
    print("=" * 50)

    # Run for participants with WAV files ready
    participants_with_wav = ['Sybille', 'Steffen_Haeseli', 'Lara', 'Vicky']

    results = run_pipeline(
        participants=participants_with_wav,
        window_minutes=15,
        model_type='ridge'
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for participant, res in results.items():
        if 'model_results' in res and res['model_results']:
            mr = res['model_results']
            print(f"\n{participant}:")
            print(f"  Samples: {mr['n_samples']}")
            print(f"  MAE: {mr['mae_mgdl']:.2f} mg/dL")
            print(f"  Pearson r: {mr['pearson_r']:.3f}")
            if 'clarke_error_grid' in mr:
                print(f"  Clarke A+B: {mr['clarke_error_grid']['clinically_acceptable']:.1f}%")
