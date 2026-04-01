"""
Hyperparameter Analysis for Voice-Glucose Estimation
====================================================
Systematic comparison of:
1. Feature extraction methods (MFCC, Mel-spectrogram, raw spectrogram)
2. MFCC parameters (n_mfcc: 13, 20, 40)
3. Mel bands (n_mels: 40, 64, 128)
4. Frequency ranges (full, speech-focused, low-frequency)
5. Time windows (full audio, 3s, 5s, 10s segments)
6. Per-individual analysis with full data for Wolf & Sybille

Focus on Wolf (947 samples) and Sybille (546 samples) for robust analysis.
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
from scipy import stats

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")
OUTPUT_DIR = BASE_DIR / "hyperparameter_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Focus on participants with most data
PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "glucose_unit": "mg/dL",
        "time_offset": 15,
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchütt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav"],
        "glucose_unit": "mg/dL",
        "time_offset": 15,
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "glucose_unit": "mmol/L",
        "time_offset": 20,
    },
    "Anja": {
        "glucose_csv": [
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_6-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_10-11-2023.csv",
        ],
        "audio_dirs": ["Anja/conv_audio"],
        "glucose_unit": "mg/dL",
        "time_offset": 0,
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
    """Find closest glucose reading."""
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
# FEATURE EXTRACTION METHODS
# ============================================================================

class FeatureExtractor:
    """Configurable feature extractor for systematic comparison."""

    def __init__(self,
                 method: str = 'mfcc',
                 n_mfcc: int = 20,
                 n_mels: int = 40,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 fmin: float = 0,
                 fmax: float = None,
                 include_deltas: bool = True,
                 time_window: float = None,  # None = full audio, else seconds
                 sr: int = 16000):

        self.method = method
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.include_deltas = include_deltas
        self.time_window = time_window
        self.sr = sr

        self.name = self._generate_name()

    def _generate_name(self) -> str:
        """Generate descriptive name for this configuration."""
        parts = [self.method]

        if self.method == 'mfcc':
            parts.append(f"n{self.n_mfcc}")
        elif self.method in ['mel', 'melspec']:
            parts.append(f"m{self.n_mels}")

        if self.fmax:
            parts.append(f"f{int(self.fmax)}")

        if self.time_window:
            parts.append(f"t{int(self.time_window)}s")

        if self.include_deltas and self.method == 'mfcc':
            parts.append("d")

        return "_".join(parts)

    def extract(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract features from audio file."""
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)

            # Apply time window if specified
            if self.time_window:
                max_samples = int(self.time_window * sr)
                if len(y) > max_samples:
                    # Take middle segment
                    start = (len(y) - max_samples) // 2
                    y = y[start:start + max_samples]
                elif len(y) < max_samples * 0.5:
                    return None  # Too short

            if len(y) < sr * 0.5:
                return None

            features = []
            fmax = self.fmax if self.fmax else sr // 2

            if self.method == 'mfcc':
                # MFCC features
                mfccs = librosa.feature.mfcc(
                    y=y, sr=sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    fmin=self.fmin,
                    fmax=fmax
                )
                features.extend(np.mean(mfccs, axis=1))
                features.extend(np.std(mfccs, axis=1))

                if self.include_deltas:
                    delta_mfccs = librosa.feature.delta(mfccs)
                    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                    features.extend(np.mean(delta_mfccs, axis=1))
                    features.extend(np.std(delta_mfccs, axis=1))
                    features.extend(np.mean(delta2_mfccs, axis=1))
                    features.extend(np.std(delta2_mfccs, axis=1))

            elif self.method in ['mel', 'melspec']:
                # Mel spectrogram features
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    fmin=self.fmin,
                    fmax=fmax
                )
                mel_db = librosa.power_to_db(mel)
                features.extend(np.mean(mel_db, axis=1))
                features.extend(np.std(mel_db, axis=1))
                features.extend(np.percentile(mel_db, 25, axis=1))
                features.extend(np.percentile(mel_db, 75, axis=1))

            elif self.method == 'spectrogram':
                # Raw spectrogram (STFT) features
                D = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))

                # Reduce dimensionality by binning frequencies
                n_bins = 64
                freq_bins = np.array_split(D, n_bins, axis=0)
                binned = np.array([np.mean(b, axis=0) for b in freq_bins])

                features.extend(np.mean(binned, axis=1))
                features.extend(np.std(binned, axis=1))

            elif self.method == 'combined':
                # Combined MFCC + Mel + spectral
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
                features.extend(np.mean(mfccs, axis=1))
                features.extend(np.std(mfccs, axis=1))

                if self.include_deltas:
                    delta_mfccs = librosa.feature.delta(mfccs)
                    features.extend(np.mean(delta_mfccs, axis=1))
                    features.extend(np.std(delta_mfccs, axis=1))

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
                mel_db = librosa.power_to_db(mel)
                features.extend(np.mean(mel_db, axis=1))
                features.extend(np.std(mel_db, axis=1))

                # Spectral features
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

                features.extend([np.mean(cent), np.std(cent)])
                features.extend([np.mean(bw), np.std(bw)])
                features.extend([np.mean(rolloff), np.std(rolloff)])

            # Add common features for all methods
            # RMS energy
            rms = librosa.feature.rms(y=y)
            features.extend([np.mean(rms), np.std(rms)])

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([np.mean(zcr), np.std(zcr)])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            return None


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================

def evaluate_configuration(extractor: FeatureExtractor,
                          audio_files: List[Path],
                          glucose_df: pd.DataFrame,
                          time_offset: int,
                          max_samples: int = None) -> Dict:
    """Evaluate a single feature configuration."""

    # Extract features
    data = []
    for audio_path in audio_files[:max_samples] if max_samples else audio_files:
        audio_ts = parse_timestamp(audio_path.name)
        if audio_ts is None:
            continue

        glucose = find_matching_glucose(audio_ts, glucose_df, time_offset)
        if glucose is None:
            continue

        features = extractor.extract(str(audio_path))
        if features is None:
            continue

        data.append({
            'features': features,
            'glucose': glucose,
        })

    if len(data) < 20:
        return None

    # Prepare arrays
    X = np.array([d['features'] for d in data])
    y = np.array([d['glucose'] for d in data])

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate with multiple models
    models = {
        'BayesianRidge': BayesianRidge(),
        'SVR_RBF': SVR(kernel='rbf', C=10),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
    }

    results = {
        'config_name': extractor.name,
        'n_samples': len(data),
        'n_features': X.shape[1],
        'models': {},
    }

    cv = KFold(n_splits=min(5, len(data)//5), shuffle=True, random_state=42)

    for model_name, model in models.items():
        try:
            preds = cross_val_predict(model, X_scaled, y, cv=cv)
            mae = mean_absolute_error(y, preds)
            r, _ = stats.pearsonr(y, preds)

            results['models'][model_name] = {
                'mae': mae,
                'r': r,
            }
        except Exception as e:
            continue

    return results


def run_hyperparameter_analysis():
    """Run comprehensive hyperparameter analysis."""

    print("="*70)
    print("HYPERPARAMETER ANALYSIS")
    print("="*70)

    all_results = {}

    # Define configurations to test
    configurations = []

    # 1. MFCC with different n_mfcc
    for n_mfcc in [13, 20, 30, 40]:
        configurations.append(FeatureExtractor(
            method='mfcc', n_mfcc=n_mfcc, include_deltas=True
        ))

    # 2. MFCC without deltas
    configurations.append(FeatureExtractor(
        method='mfcc', n_mfcc=20, include_deltas=False
    ))

    # 3. Mel spectrogram with different n_mels
    for n_mels in [40, 64, 128]:
        configurations.append(FeatureExtractor(
            method='mel', n_mels=n_mels
        ))

    # 4. Raw spectrogram
    configurations.append(FeatureExtractor(method='spectrogram'))

    # 5. Different frequency ranges
    configurations.append(FeatureExtractor(
        method='mfcc', n_mfcc=20, fmin=80, fmax=3000, include_deltas=True
    ))  # Speech range

    configurations.append(FeatureExtractor(
        method='mfcc', n_mfcc=20, fmin=50, fmax=1000, include_deltas=True
    ))  # Low frequency (F0 + first formants)

    configurations.append(FeatureExtractor(
        method='mfcc', n_mfcc=20, fmin=300, fmax=4000, include_deltas=True
    ))  # Higher formants

    # 6. Different time windows
    for window in [3, 5, 10]:
        configurations.append(FeatureExtractor(
            method='mfcc', n_mfcc=20, include_deltas=True, time_window=window
        ))

    # 7. Combined features
    configurations.append(FeatureExtractor(method='combined', n_mfcc=20, n_mels=40))

    print(f"\nTesting {len(configurations)} configurations:")
    for cfg in configurations:
        print(f"  - {cfg.name}")

    # Process each participant
    for participant, config in PARTICIPANTS.items():
        print(f"\n{'='*60}")
        print(f"PARTICIPANT: {participant}")
        print(f"{'='*60}")

        # Load glucose data
        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
        if glucose_df.empty:
            print("  No glucose data")
            continue

        # Get audio files
        audio_files = []
        for audio_dir in config['audio_dirs']:
            dir_path = BASE_DIR / audio_dir
            if dir_path.exists():
                audio_files.extend(list(dir_path.glob("*.wav")))

        print(f"  Audio files: {len(audio_files)}")
        print(f"  Glucose readings: {len(glucose_df)}")

        # Use all data for Wolf and Sybille, limit others
        max_samples = None if participant in ['Wolf', 'Sybille'] else 150

        participant_results = []

        for cfg in configurations:
            print(f"\n  Testing: {cfg.name}...", end="", flush=True)

            result = evaluate_configuration(
                cfg, audio_files, glucose_df,
                config['time_offset'], max_samples
            )

            if result:
                # Get best model result
                best_model = min(result['models'].items(), key=lambda x: x[1]['mae'])
                print(f" n={result['n_samples']}, feat={result['n_features']}, "
                      f"MAE={best_model[1]['mae']:.2f} ({best_model[0]})")

                participant_results.append({
                    'config': cfg.name,
                    'n_samples': result['n_samples'],
                    'n_features': result['n_features'],
                    **{f"{m}_mae": v['mae'] for m, v in result['models'].items()},
                    **{f"{m}_r": v['r'] for m, v in result['models'].items()},
                })
            else:
                print(" FAILED")

        all_results[participant] = pd.DataFrame(participant_results)

    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY: BEST CONFIGURATIONS PER PARTICIPANT")
    print("="*70)

    summary_data = []

    for participant, df in all_results.items():
        if df.empty:
            continue

        # Find best configuration for each model
        for model in ['BayesianRidge', 'SVR_RBF', 'RandomForest']:
            mae_col = f"{model}_mae"
            if mae_col in df.columns:
                best_idx = df[mae_col].idxmin()
                best_row = df.loc[best_idx]

                summary_data.append({
                    'Participant': participant,
                    'Model': model,
                    'Best Config': best_row['config'],
                    'MAE': best_row[mae_col],
                    'r': best_row[f"{model}_r"],
                    'n_samples': best_row['n_samples'],
                    'n_features': best_row['n_features'],
                })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save results
    for participant, df in all_results.items():
        df.to_csv(OUTPUT_DIR / f"{participant}_hyperparameters.csv", index=False)

    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    # Generate visualization
    generate_comparison_plots(all_results, OUTPUT_DIR)

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return all_results, summary_df


def generate_comparison_plots(all_results: Dict, output_dir: Path):
    """Generate comparison visualizations."""

    for participant, df in all_results.items():
        if df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: MAE by configuration
        ax = axes[0]
        x = range(len(df))
        width = 0.25

        for i, model in enumerate(['BayesianRidge', 'SVR_RBF', 'RandomForest']):
            col = f"{model}_mae"
            if col in df.columns:
                ax.bar([xi + i*width for xi in x], df[col], width, label=model, alpha=0.8)

        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(df['config'], rotation=45, ha='right')
        ax.set_ylabel('MAE (mg/dL)')
        ax.set_title(f'{participant}: MAE by Configuration')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Correlation by configuration
        ax = axes[1]
        for i, model in enumerate(['BayesianRidge', 'SVR_RBF', 'RandomForest']):
            col = f"{model}_r"
            if col in df.columns:
                ax.bar([xi + i*width for xi in x], df[col], width, label=model, alpha=0.8)

        ax.set_xticks([xi + width for xi in x])
        ax.set_xticklabels(df['config'], rotation=45, ha='right')
        ax.set_ylabel('Correlation (r)')
        ax.set_title(f'{participant}: Correlation by Configuration')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / f"{participant}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    print("  Plots saved!")


# ============================================================================
# TIME OFFSET OPTIMIZATION
# ============================================================================

def optimize_time_offsets():
    """Find optimal time offset for each participant."""

    print("\n" + "="*70)
    print("TIME OFFSET OPTIMIZATION")
    print("="*70)

    offsets_to_test = list(range(-30, 35, 5))  # -30 to +30 in 5-min steps

    extractor = FeatureExtractor(method='mfcc', n_mfcc=20, include_deltas=True)

    results = {}

    for participant, config in PARTICIPANTS.items():
        print(f"\n{participant}:")

        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
        if glucose_df.empty:
            continue

        audio_files = []
        for audio_dir in config['audio_dirs']:
            dir_path = BASE_DIR / audio_dir
            if dir_path.exists():
                audio_files.extend(list(dir_path.glob("*.wav")))

        offset_results = []

        for offset in offsets_to_test:
            result = evaluate_configuration(
                extractor, audio_files, glucose_df,
                offset, max_samples=200
            )

            if result and 'BayesianRidge' in result['models']:
                mae = result['models']['BayesianRidge']['mae']
                offset_results.append({
                    'offset': offset,
                    'mae': mae,
                    'n_samples': result['n_samples'],
                })
                print(f"  Offset {offset:+3d} min: MAE={mae:.2f} (n={result['n_samples']})")

        if offset_results:
            best = min(offset_results, key=lambda x: x['mae'])
            results[participant] = {
                'best_offset': best['offset'],
                'best_mae': best['mae'],
                'all_offsets': offset_results,
            }
            print(f"  → Best offset: {best['offset']:+d} min (MAE={best['mae']:.2f})")

    # Plot offset curves
    fig, ax = plt.subplots(figsize=(12, 6))

    for participant, data in results.items():
        offsets = [r['offset'] for r in data['all_offsets']]
        maes = [r['mae'] for r in data['all_offsets']]
        ax.plot(offsets, maes, 'o-', label=participant, linewidth=2, markersize=6)

        # Mark best
        ax.axvline(data['best_offset'], linestyle='--', alpha=0.3)

    ax.set_xlabel('Time Offset (minutes)', fontsize=12)
    ax.set_ylabel('MAE (mg/dL)', fontsize=12)
    ax.set_title('Optimal Time Offset per Participant', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "time_offset_optimization.png", dpi=150, bbox_inches='tight')
    plt.close()

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run hyperparameter analysis
    results, summary = run_hyperparameter_analysis()

    # Run time offset optimization
    offset_results = optimize_time_offsets()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
