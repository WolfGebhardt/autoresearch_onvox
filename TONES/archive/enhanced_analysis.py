"""
Enhanced Voice-Glucose Analysis
===============================
Improved matching algorithm that:
1. Uses CGM CSV timestamps (not filename glucose prefix)
2. Tests multiple time windows to expand training data
3. Optimizes offset to find best voice-glucose lag
4. Validates filename glucose vs CGM reading
5. Handles unit conversion (mmol/L vs mg/dL)
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
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")
OUTPUT_DIR = BASE_DIR / "enhanced_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Conversion factor
MMOL_TO_MGDL = 18.0182

# Participants configuration
PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "glucose_unit": "mg/dL",
        "glucose_col": "Historic Glucose mg/dL",
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchutt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav"],
        "glucose_unit": "mg/dL",
        "glucose_col": "Historic Glucose mg/dL",
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
        "glucose_col": "Historic Glucose mg/dL",
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "glucose_unit": "mmol/L",
        "glucose_col": "Historic Glucose mmol/L",
    },
    "Vicky": {
        "glucose_csv": ["Vicky/Number_10Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Vicky/conv_audio"],
        "glucose_unit": "mmol/L",
        "glucose_col": "Historic Glucose mmol/L",
    },
    "Steffen": {
        "glucose_csv": ["Steffen_Haeseli/Number_2Nov_23_glucose_4-1-2024.csv"],
        "audio_dirs": ["Steffen_Haeseli/wav"],
        "glucose_unit": "mmol/L",
        "glucose_col": "Historic Glucose mmol/L",
    },
    "Lara": {
        "glucose_csv": ["Lara/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Lara/conv_audio"],
        "glucose_unit": "mmol/L",
        "glucose_col": "Historic Glucose mmol/L",
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def parse_audio_timestamp(filename: str) -> Optional[datetime]:
    """Extract timestamp from WhatsApp audio filename."""
    # Pattern: WhatsApp Audio 2023-11-08 um 17.30.35
    # or: WhatsApp-Audio-2023-11-08-um-17.30.35
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


def extract_filename_glucose(filename: str) -> Optional[float]:
    """Extract glucose value from filename prefix (e.g., '130_WhatsApp...')."""
    # Patterns: "130_WhatsApp", "Wolf 130 WhatsApp", "Wolf_130 WhatsApp"
    patterns = [
        r'^(\d+)_',  # 130_WhatsApp...
        r'Wolf[_ ](\d+)[_ ]WhatsApp',  # Wolf 130 WhatsApp or Wolf_130 WhatsApp
        r'AnyConv\.com__(\d+)_',  # AnyConv.com__130_WhatsApp
        r'AnyConv\.com__Wolf[_ ](\d+)',  # AnyConv.com__Wolf 130
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def load_glucose_csv(csv_paths: List[str], glucose_col: str, unit: str) -> pd.DataFrame:
    """Load and merge glucose CSVs with unit conversion."""
    all_dfs = []

    for csv_path in csv_paths:
        full_path = BASE_DIR / csv_path

        # Handle special characters in filename (Sybille)
        if not full_path.exists():
            # Try to find the file with glob
            parent_dir = full_path.parent
            pattern = full_path.name.replace('u', '?').replace('U', '?')
            matches = list(parent_dir.glob(pattern)) if parent_dir.exists() else []
            if matches:
                full_path = matches[0]
            else:
                # Try broader pattern
                if parent_dir.exists():
                    csv_files = list(parent_dir.glob("*.csv"))
                    if csv_files:
                        full_path = csv_files[0]
                    else:
                        print(f"    WARNING: CSV not found: {csv_path}")
                        continue
                else:
                    print(f"    WARNING: Directory not found: {parent_dir}")
                    continue

        try:
            # Detect header row by reading first few lines
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = [f.readline() for _ in range(5)]

            # Find the row with "Device" column (header row)
            skiprows = 0
            for i, line in enumerate(lines):
                if 'Device' in line or 'Ger' in line:  # German: Gerät
                    skiprows = i
                    break

            # Read CSV with detected skiprows
            df = pd.read_csv(full_path, skiprows=skiprows, encoding='utf-8', on_bad_lines='skip')

            # Find the glucose column (handle both English and German)
            glucose_col_actual = None
            timestamp_col = None

            for col in df.columns:
                col_lower = col.lower()
                # Glucose column
                if 'historic glucose' in col_lower or 'glukosewert-verlauf' in col_lower or 'glukosewert' in col_lower:
                    glucose_col_actual = col
                # Timestamp column
                if 'device timestamp' in col_lower or 'zeitstempel' in col_lower:
                    timestamp_col = col

            if glucose_col_actual is None:
                print(f"    WARNING: No glucose column in {csv_path}")
                print(f"    Available columns: {list(df.columns)[:5]}")
                continue

            if timestamp_col is None:
                timestamp_col = 'Device Timestamp'  # fallback

            # Parse timestamps
            df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')

            # Get glucose values
            df['glucose_raw'] = pd.to_numeric(df[glucose_col_actual], errors='coerce')

            # Convert to mg/dL if needed
            if 'mmol' in glucose_col_actual.lower():
                df['glucose_mgdl'] = df['glucose_raw'] * MMOL_TO_MGDL
            else:
                df['glucose_mgdl'] = df['glucose_raw']

            # Keep only valid rows
            df = df[['timestamp', 'glucose_mgdl', 'glucose_raw']].dropna()

            if len(df) > 0:
                all_dfs.append(df)
                print(f"    Loaded {len(df)} glucose readings from {full_path.name}")

        except Exception as e:
            print(f"    ERROR loading {csv_path}: {e}")
            import traceback
            traceback.print_exc()

    if not all_dfs:
        return pd.DataFrame()

    # Merge and sort
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').drop_duplicates('timestamp')
    return combined


def get_glucose_at_time(glucose_df: pd.DataFrame, target_time: datetime,
                        max_window_minutes: int = 15) -> Tuple[Optional[float], float, float]:
    """
    Get glucose value closest to target time within window.

    Returns: (glucose_value, time_diff_minutes, glucose_rate_of_change)
    """
    if glucose_df.empty:
        return None, float('inf'), 0.0

    # Calculate time differences
    diffs = (glucose_df['timestamp'] - target_time).abs()
    min_idx = diffs.idxmin()
    min_diff = diffs[min_idx].total_seconds() / 60

    if min_diff > max_window_minutes:
        return None, min_diff, 0.0

    glucose = glucose_df.loc[min_idx, 'glucose_mgdl']

    # Calculate rate of change (mg/dL per minute)
    try:
        nearby = glucose_df[(diffs < timedelta(minutes=30)).values]
        if len(nearby) >= 2:
            nearby = nearby.sort_values('timestamp')
            time_span = (nearby['timestamp'].iloc[-1] - nearby['timestamp'].iloc[0]).total_seconds() / 60
            if time_span > 0:
                glucose_change = nearby['glucose_mgdl'].iloc[-1] - nearby['glucose_mgdl'].iloc[0]
                rate = glucose_change / time_span
            else:
                rate = 0.0
        else:
            rate = 0.0
    except:
        rate = 0.0

    return glucose, min_diff, rate


def get_glucose_stability(glucose_df: pd.DataFrame, target_time: datetime,
                         window_minutes: int = 30) -> Tuple[float, float]:
    """
    Calculate glucose stability in a window around target time.

    Returns: (std_deviation, max_minus_min)
    """
    if glucose_df.empty:
        return float('inf'), float('inf')

    window_start = target_time - timedelta(minutes=window_minutes)
    window_end = target_time + timedelta(minutes=window_minutes)

    mask = (glucose_df['timestamp'] >= window_start) & (glucose_df['timestamp'] <= window_end)
    window_data = glucose_df.loc[mask, 'glucose_mgdl']

    if len(window_data) < 2:
        return float('inf'), float('inf')

    return window_data.std(), window_data.max() - window_data.min()


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_mfcc_features(audio_path: Path, n_mfcc: int = 20) -> Optional[np.ndarray]:
    """Extract MFCC features with deltas."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        if len(y) < sr * 0.5:  # Less than 0.5 seconds
            return None

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Deltas
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Aggregate statistics - collect as numpy arrays
        features = []
        for feat in [mfcc, mfcc_delta, mfcc_delta2]:
            features.append(feat.mean(axis=1))  # shape (n_mfcc,)
            features.append(feat.std(axis=1))   # shape (n_mfcc,)

        # Energy features
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.array([rms.mean(), rms.std()]))
        features.append(np.array([zcr.mean(), zcr.std()]))

        # Concatenate all features
        return np.concatenate(features)

    except Exception as e:
        return None


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_participant_with_offset(
    name: str,
    config: dict,
    offset_minutes: int = 0,
    max_window_minutes: int = 15,
    stability_threshold: float = 20.0,  # mg/dL max variation to accept
) -> dict:
    """
    Analyze a participant with a specific time offset.

    offset_minutes:
        Positive = voice recorded BEFORE CGM catches up (CGM lags behind blood glucose)
        Negative = voice recorded AFTER glucose changed

    Theory: CGM typically lags blood glucose by 5-15 minutes due to interstitial fluid.
    If voice responds to blood glucose faster than CGM, positive offset should help.
    """

    # Load glucose data
    glucose_df = load_glucose_csv(config['glucose_csv'], config.get('glucose_col', ''),
                                  config['glucose_unit'])

    if glucose_df.empty:
        return {'error': 'No glucose data'}

    # Get audio files
    audio_files = []
    for audio_dir in config['audio_dirs']:
        dir_path = BASE_DIR / audio_dir
        if dir_path.exists():
            audio_files.extend(list(dir_path.glob("*.wav")))

    # Remove duplicates (same content, different names)
    seen_hashes = set()
    unique_audio = []
    for f in audio_files:
        # Use the hash part of filename as dedup key
        hash_match = re.search(r'_([a-f0-9]{8})\.waptt\.wav$', f.name.lower())
        if hash_match:
            file_hash = hash_match.group(1)
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_audio.append(f)
        else:
            unique_audio.append(f)

    audio_files = unique_audio

    # Process each audio file
    data = []
    filename_validation = []

    for audio_path in audio_files:
        # Extract timestamp from filename
        audio_ts = parse_audio_timestamp(audio_path.name)
        if audio_ts is None:
            continue

        # Apply offset: look for glucose at (audio_time + offset)
        # Positive offset means: "The CGM will show the glucose X minutes LATER than voice"
        target_glucose_time = audio_ts + timedelta(minutes=offset_minutes)

        # Get glucose value
        glucose, time_diff, rate_of_change = get_glucose_at_time(
            glucose_df, target_glucose_time, max_window_minutes
        )

        if glucose is None:
            continue

        # Check stability (only use stable readings for training)
        std_dev, max_min_diff = get_glucose_stability(glucose_df, target_glucose_time, 30)

        # Validate against filename glucose (for Wolf)
        filename_glucose = extract_filename_glucose(audio_path.name)
        if filename_glucose is not None:
            filename_validation.append({
                'filename_glucose': filename_glucose,
                'cgm_glucose': glucose,
                'time_diff': time_diff,
                'difference': abs(filename_glucose - glucose),
            })

        # Extract features
        features = extract_mfcc_features(audio_path)
        if features is None:
            continue

        data.append({
            'audio_path': str(audio_path),
            'audio_timestamp': audio_ts,
            'glucose': glucose,
            'time_diff': time_diff,
            'rate_of_change': rate_of_change,
            'glucose_std': std_dev,
            'glucose_range': max_min_diff,
            'is_stable': max_min_diff <= stability_threshold,
            'filename_glucose': filename_glucose,
            'features': features,
        })

    if len(data) < 10:
        return {'error': f'Too few samples ({len(data)})'}

    # Build feature matrix
    X = np.vstack([d['features'] for d in data])
    y = np.array([d['glucose'] for d in data])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train and evaluate models
    models = {
        'SVR_RBF': SVR(kernel='rbf', C=10, gamma='scale'),
        'BayesianRidge': BayesianRidge(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'KNN': KNeighborsRegressor(n_neighbors=min(10, len(data)//2)),
    }

    results = {
        'n_samples': len(data),
        'n_stable': sum(1 for d in data if d['is_stable']),
        'glucose_mean': y.mean(),
        'glucose_std': y.std(),
        'glucose_min': y.min(),
        'glucose_max': y.max(),
        'offset_minutes': offset_minutes,
        'models': {},
    }

    # Cross-validation
    cv = LeaveOneOut() if len(data) < 50 else 10

    for model_name, model in models.items():
        try:
            y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            # Pearson correlation
            if y.std() > 0 and y_pred.std() > 0:
                r, p = stats.pearsonr(y, y_pred)
            else:
                r, p = 0, 1

            results['models'][model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'r': r,
                'p_value': p,
                'predictions': y_pred,
                'actual': y,
            }
        except Exception as e:
            results['models'][model_name] = {'error': str(e)}

    # Filename validation analysis
    if filename_validation:
        val_df = pd.DataFrame(filename_validation)
        results['filename_validation'] = {
            'n_validated': len(val_df),
            'mean_difference': val_df['difference'].mean(),
            'max_difference': val_df['difference'].max(),
            'correlation': val_df['filename_glucose'].corr(val_df['cgm_glucose']),
        }

    return results


def optimize_offset(name: str, config: dict,
                   offsets: List[int] = None,
                   max_window_minutes: int = 15) -> dict:
    """Find optimal time offset for a participant."""

    if offsets is None:
        offsets = list(range(-30, 35, 5))  # -30 to +30 in 5-min steps

    print(f"\n  Testing offsets for {name}...")

    offset_results = []

    for offset in offsets:
        result = analyze_participant_with_offset(
            name, config, offset_minutes=offset,
            max_window_minutes=max_window_minutes
        )

        if 'error' in result:
            continue

        best_mae = min(r['MAE'] for r in result['models'].values() if 'MAE' in r)
        best_model = min(result['models'].items(),
                        key=lambda x: x[1].get('MAE', float('inf')))[0]

        offset_results.append({
            'offset': offset,
            'best_mae': best_mae,
            'best_model': best_model,
            'n_samples': result['n_samples'],
            'best_r': result['models'][best_model].get('r', 0),
        })

        print(f"    Offset {offset:+3d} min: MAE={best_mae:.2f}, r={result['models'][best_model].get('r', 0):.3f}, n={result['n_samples']}")

    if not offset_results:
        return {'error': 'No valid offset results'}

    # Find optimal
    best_offset = min(offset_results, key=lambda x: x['best_mae'])

    return {
        'optimal_offset': best_offset['offset'],
        'optimal_mae': best_offset['best_mae'],
        'optimal_model': best_offset['best_model'],
        'all_results': offset_results,
    }


def test_window_sizes(name: str, config: dict, offset: int = 0,
                     windows: List[int] = None) -> dict:
    """Test different time window sizes for matching."""

    if windows is None:
        windows = [5, 10, 15, 20, 30, 45, 60]

    print(f"\n  Testing window sizes for {name} (offset={offset})...")

    window_results = []

    for window in windows:
        result = analyze_participant_with_offset(
            name, config, offset_minutes=offset,
            max_window_minutes=window
        )

        if 'error' in result:
            window_results.append({
                'window': window,
                'error': result['error'],
            })
            continue

        best_mae = min(r['MAE'] for r in result['models'].values() if 'MAE' in r)

        window_results.append({
            'window': window,
            'best_mae': best_mae,
            'n_samples': result['n_samples'],
            'n_stable': result['n_stable'],
        })

        print(f"    Window {window:2d} min: n={result['n_samples']}, stable={result['n_stable']}, MAE={best_mae:.2f}")

    return {'window_results': window_results}


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_full_analysis():
    """Run complete enhanced analysis."""

    print("="*70)
    print("ENHANCED VOICE-GLUCOSE ANALYSIS")
    print("="*70)

    all_results = {}

    for name, config in PARTICIPANTS.items():
        print(f"\n{'='*70}")
        print(f"ANALYZING: {name}")
        print(f"{'='*70}")

        # Phase 1: Optimize offset
        offset_result = optimize_offset(name, config)

        if 'error' in offset_result:
            print(f"  ERROR: {offset_result['error']}")
            continue

        optimal_offset = offset_result['optimal_offset']
        print(f"\n  OPTIMAL OFFSET: {optimal_offset:+d} min (MAE={offset_result['optimal_mae']:.2f})")

        # Phase 2: Test window sizes with optimal offset
        window_result = test_window_sizes(name, config, offset=optimal_offset)

        # Phase 3: Final analysis with optimal settings
        print(f"\n  Final analysis with offset={optimal_offset}, window=15min...")
        final_result = analyze_participant_with_offset(
            name, config,
            offset_minutes=optimal_offset,
            max_window_minutes=15
        )

        if 'error' not in final_result:
            print(f"\n  Results ({final_result['n_samples']} samples):")
            for model_name, metrics in final_result['models'].items():
                if 'MAE' in metrics:
                    print(f"    {model_name:15s}: MAE={metrics['MAE']:.2f}, r={metrics['r']:.3f}")

            # Filename validation
            if 'filename_validation' in final_result:
                fv = final_result['filename_validation']
                print(f"\n  Filename validation (n={fv['n_validated']}):")
                print(f"    Mean diff from CGM: {fv['mean_difference']:.1f} mg/dL")
                print(f"    Correlation: {fv['correlation']:.3f}")

        all_results[name] = {
            'offset_optimization': offset_result,
            'window_analysis': window_result,
            'final_result': final_result,
        }

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    generate_visualizations(all_results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Participant':<12} {'Offset':>8} {'Samples':>8} {'Best MAE':>10} {'Best r':>8} {'Model':<15}")
    print("-"*70)

    for name, result in all_results.items():
        if 'final_result' in result and 'models' in result['final_result']:
            final = result['final_result']
            offset = result['offset_optimization']['optimal_offset']
            best_model = min(final['models'].items(),
                           key=lambda x: x[1].get('MAE', float('inf')))
            print(f"{name:<12} {offset:>+7d}m {final['n_samples']:>8d} {best_model[1]['MAE']:>10.2f} {best_model[1]['r']:>8.3f} {best_model[0]:<15}")

    print(f"\nResults saved to: {OUTPUT_DIR}")

    return all_results


def generate_visualizations(results: dict):
    """Generate analysis visualizations."""

    # 1. Offset optimization curves
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        if 'offset_optimization' in result and 'all_results' in result['offset_optimization']:
            data = result['offset_optimization']['all_results']
            offsets = [d['offset'] for d in data]
            maes = [d['best_mae'] for d in data]

            ax.plot(offsets, maes, 'b-o', markersize=4)

            # Mark optimal
            opt_idx = maes.index(min(maes))
            ax.axvline(offsets[opt_idx], color='r', linestyle='--', alpha=0.5)
            ax.scatter([offsets[opt_idx]], [maes[opt_idx]], color='r', s=100, zorder=5)

            ax.set_xlabel('Offset (minutes)')
            ax.set_ylabel('MAE (mg/dL)')
            ax.set_title(f'{name}\nOptimal: {offsets[opt_idx]:+d} min')
            ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'offset_optimization.png', dpi=150)
    plt.close()
    print("  Saved: offset_optimization.png")

    # 2. Scatter plots for best models
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        if 'final_result' in result and 'models' in result['final_result']:
            final = result['final_result']
            best_model_name = min(final['models'].items(),
                                 key=lambda x: x[1].get('MAE', float('inf')))[0]
            best_model = final['models'][best_model_name]

            if 'actual' in best_model and 'predictions' in best_model:
                actual = best_model['actual']
                pred = best_model['predictions']

                ax.scatter(actual, pred, alpha=0.6, s=30)

                # Perfect prediction line
                min_val = min(actual.min(), pred.min())
                max_val = max(actual.max(), pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

                # +/- 15 mg/dL bands
                ax.fill_between([min_val, max_val],
                               [min_val-15, max_val-15],
                               [min_val+15, max_val+15],
                               alpha=0.1, color='green')

                ax.set_xlabel('Actual Glucose (mg/dL)')
                ax.set_ylabel('Predicted Glucose (mg/dL)')
                ax.set_title(f'{name}\nMAE={best_model["MAE"]:.1f}, r={best_model["r"]:.2f}')
                ax.set_aspect('equal')

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'predictions_scatter.png', dpi=150)
    plt.close()
    print("  Saved: predictions_scatter.png")

    # 3. Summary bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = []
    maes = []
    rs = []
    offsets = []

    for name, result in results.items():
        if 'final_result' in result and 'models' in result['final_result']:
            final = result['final_result']
            best = min(final['models'].items(), key=lambda x: x[1].get('MAE', float('inf')))
            names.append(name)
            maes.append(best[1]['MAE'])
            rs.append(best[1]['r'])
            offsets.append(result['offset_optimization']['optimal_offset'])

    if not names:
        print("  No valid results for summary chart")
        plt.close()
        return

    # MAE bars
    colors = plt.cm.RdYlGn_r(np.array(maes) / max(maes))
    bars = ax1.bar(names, maes, color=colors)
    ax1.set_ylabel('MAE (mg/dL)')
    ax1.set_title('Best Model MAE by Participant')
    ax1.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='10 mg/dL target')
    ax1.legend()

    # Add offset labels
    for bar, offset in zip(bars, offsets):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{offset:+d}m', ha='center', va='bottom', fontsize=9)

    # Correlation bars
    colors = plt.cm.RdYlGn(np.clip(np.array(rs), 0, 1))
    ax2.bar(names, rs, color=colors)
    ax2.set_ylabel('Pearson Correlation (r)')
    ax2.set_title('Prediction Correlation by Participant')
    ax2.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='r=0.5')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_comparison.png', dpi=150)
    plt.close()
    print("  Saved: summary_comparison.png")


if __name__ == "__main__":
    results = run_full_analysis()
