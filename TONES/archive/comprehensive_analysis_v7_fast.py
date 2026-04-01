"""
Voice-Based Glucose Estimation - Comprehensive Analysis v7 (Fast)
================================================================
Efficient version with VAD (Voice Activity Detection) support
"""

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
warnings.filterwarnings('ignore')

import librosa
from sklearn.model_selection import LeaveOneOut, cross_val_predict, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score

BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")
OUTPUT_DIR = BASE_DIR / "documentation_v7"
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================================================
# VOICE ACTIVITY DETECTION (VAD)
# ============================================================================

def apply_vad(y, sr, top_db=30, min_silence_ms=500, frame_length=2048, hop_length=512):
    """
    Apply Voice Activity Detection - remove long silences, keep short pauses.

    Args:
        y: Audio signal
        sr: Sample rate
        top_db: Threshold below peak to consider silence
        min_silence_ms: Minimum silence duration to remove (ms)
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis

    Returns:
        y_vad: Audio with long silences removed
        vad_ratio: Ratio of voiced to total duration
    """
    # Detect non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db,
                                       frame_length=frame_length,
                                       hop_length=hop_length)

    if len(intervals) == 0:
        return y, 1.0

    # Merge intervals that are separated by less than min_silence
    min_silence_samples = int(min_silence_ms * sr / 1000)
    merged = []

    for start, end in intervals:
        if merged and start - merged[-1][1] < min_silence_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append([start, end])

    # Extract and concatenate voiced segments
    if not merged:
        return y, 1.0

    chunks = [y[start:end] for start, end in merged]
    y_vad = np.concatenate(chunks)

    vad_ratio = len(y_vad) / len(y)

    return y_vad, vad_ratio


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(y, sr, n_mfcc=20, n_mels=64, use_vad=True):
    """Extract features with optional VAD preprocessing."""

    # Apply VAD if requested
    if use_vad:
        y, vad_ratio = apply_vad(y, sr, min_silence_ms=500)
        if len(y) < sr * 0.5:  # Less than 0.5s after VAD
            return None
    else:
        vad_ratio = 1.0

    window_ms = 1000
    window_samples = int(sr * window_ms / 1000)
    hop_samples = window_samples // 2

    if len(y) < window_samples:
        return None

    all_features = []

    for start in range(0, len(y) - window_samples + 1, hop_samples):
        window = y[start:start + window_samples]

        try:
            # MFCCs
            mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=n_mfcc, n_fft=min(2048, len(window)))
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)

            # Mel spectrogram
            mel = librosa.feature.melspectrogram(y=window, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Prosodic
            pitches, mags = librosa.piptrack(y=window, sr=sr)
            pitch_vals = pitches[mags > np.median(mags)]

            f0_mean = np.mean(pitch_vals) if len(pitch_vals) > 10 else 0
            f0_std = np.std(pitch_vals) if len(pitch_vals) > 10 else 0

            spec_cent = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=window, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(window))
            rms = np.mean(librosa.feature.rms(y=window))

            frame_features = np.concatenate([
                np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
                np.mean(delta, axis=1),
                np.mean(delta2, axis=1),
                np.mean(mel_db, axis=1), np.std(mel_db, axis=1),
                [f0_mean, f0_std, spec_cent, spec_bw, zcr, rms]
            ])

            all_features.append(frame_features)
        except:
            continue

    if not all_features:
        return None

    all_features = np.array(all_features)

    # Aggregate
    aggregated = np.concatenate([
        np.mean(all_features, axis=0),
        np.std(all_features, axis=0),
        np.percentile(all_features, 10, axis=0),
        np.percentile(all_features, 90, axis=0),
    ])

    # Add VAD ratio as feature
    aggregated = np.append(aggregated, vad_ratio)

    return aggregated


# ============================================================================
# DATA LOADING (Only participants with glucose data)
# ============================================================================

PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all opus audio"],
        "audio_ext": [".opus"],
        "glucose_unit": "mg/dL",
    },
    "Anja": {
        "glucose_csv": ["Anja/glucose 21nov 2023/AnjaZhao_glucose_6-11-2023.csv"],
        "audio_dirs": ["Anja/conv_audio"],
        "audio_ext": [".wav"],
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
        "audio_dirs": ["Sybille/audio_wav"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
    },
}


def load_glucose_data(csv_paths, unit):
    """Load glucose data."""
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
        except:
            continue

        timestamp_col = df.columns[2] if len(df.columns) > 2 else None
        glucose_col = df.columns[4] if len(df.columns) > 4 else None

        if timestamp_col is None or glucose_col is None:
            continue

        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
        df['glucose'] = pd.to_numeric(df[glucose_col], errors='coerce')

        if unit == 'mmol/L':
            df['glucose'] = df['glucose'] * 18.0182

        df = df.dropna(subset=['timestamp', 'glucose'])
        if len(df) > 0:
            all_dfs.append(df[['timestamp', 'glucose']])

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    return pd.DataFrame()


def parse_timestamp(filename):
    """Extract timestamp from filename."""
    patterns = [
        r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h]?(\d{2})?',
    ]

    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            groups = match.groups()
            date_str = groups[0]
            hour = int(groups[1])
            minute = int(groups[2])
            second = int(groups[3]) if groups[3] else 0
            return datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}:{second:02d}",
                                    "%Y-%m-%d %H:%M:%S")
    return None


def find_glucose(ts, glucose_df, offset_min=0, window_min=15):
    """Find matching glucose."""
    if glucose_df.empty:
        return None

    search = ts + timedelta(minutes=offset_min)
    diffs = abs((glucose_df['timestamp'] - search).dt.total_seconds() / 60)
    min_diff = diffs.min()

    if min_diff <= window_min:
        return glucose_df.loc[diffs.idxmin(), 'glucose']
    return None


def get_hash(path):
    """File hash for deduplication."""
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read(8192)).hexdigest()
    except:
        return None


def load_all_data(use_vad=True, offset_min=0):
    """Load all data from all participants."""
    all_data = []

    for name, config in PARTICIPANTS.items():
        print(f"\n  Loading {name}...")

        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
        if glucose_df.empty:
            print(f"    No glucose data")
            continue

        audio_files = []
        for audio_dir in config['audio_dirs']:
            dir_path = BASE_DIR / audio_dir
            if dir_path.exists():
                for ext in config['audio_ext']:
                    audio_files.extend(dir_path.glob(f"*{ext}"))

        # Deduplicate
        seen = set()
        unique = []
        for f in audio_files:
            h = get_hash(f)
            if h and h not in seen:
                seen.add(h)
                unique.append(f)

        print(f"    {len(unique)} audio files")

        count = 0
        for audio_path in unique:
            ts = parse_timestamp(audio_path.name)
            if ts is None:
                continue

            glucose = find_glucose(ts, glucose_df, offset_min)
            if glucose is None:
                continue

            try:
                y, sr = librosa.load(audio_path, sr=16000)
            except:
                continue

            features = extract_features(y, sr, use_vad=use_vad)
            if features is not None:
                all_data.append({
                    'participant': name,
                    'features': features,
                    'glucose': glucose,
                    'timestamp': ts
                })
                count += 1

        print(f"    {count} samples loaded")

    return all_data


# ============================================================================
# EVALUATION
# ============================================================================

def glucose_to_quintile(y):
    """Convert to quintiles."""
    thresholds = np.percentile(y, [20, 40, 60, 80])
    classes = np.digitize(y, thresholds)
    return classes, thresholds


def evaluate_regression(X, y):
    """Evaluate regression models."""
    models = {
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        try:
            y_pred = cross_val_predict(pipe, X, y, cv=cv)
            results[name] = {
                'mae': mean_absolute_error(y, y_pred),
                'r': np.corrcoef(y, y_pred)[0, 1],
                'y_pred': y_pred
            }
        except:
            pass

    return results


def evaluate_classification(X, y_class):
    """Evaluate classification models."""
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42),
    }

    # Use regular KFold if class distribution is uneven
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        try:
            y_pred = cross_val_predict(pipe, X, y_class, cv=cv)
            results[name] = {
                'accuracy': accuracy_score(y_class, y_pred),
                'f1': f1_score(y_class, y_pred, average='weighted'),
            }
        except:
            pass

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE VOICE-GLUCOSE ANALYSIS v7 (Fast)")
    print("With VAD (Voice Activity Detection)")
    print("=" * 70)

    # Step 1: Compare VAD vs No VAD
    print("\n" + "=" * 70)
    print("STEP 1: VAD COMPARISON")
    print("=" * 70)

    print("\n--- Loading WITHOUT VAD ---")
    data_no_vad = load_all_data(use_vad=False, offset_min=15)
    print(f"\nTotal without VAD: {len(data_no_vad)} samples")

    print("\n--- Loading WITH VAD ---")
    data_with_vad = load_all_data(use_vad=True, offset_min=15)
    print(f"\nTotal with VAD: {len(data_with_vad)} samples")

    if not data_with_vad:
        print("No data loaded!")
        return

    # Step 2: Evaluate both
    print("\n" + "=" * 70)
    print("STEP 2: REGRESSION COMPARISON")
    print("=" * 70)

    # Without VAD
    X_no_vad = np.array([d['features'] for d in data_no_vad])
    y_no_vad = np.array([d['glucose'] for d in data_no_vad])

    print("\n--- Without VAD ---")
    reg_no_vad = evaluate_regression(X_no_vad, y_no_vad)
    for name, res in sorted(reg_no_vad.items(), key=lambda x: x[1]['mae']):
        print(f"  {name}: MAE={res['mae']:.2f}, r={res['r']:.3f}")

    # With VAD
    X_vad = np.array([d['features'] for d in data_with_vad])
    y_vad = np.array([d['glucose'] for d in data_with_vad])

    print("\n--- With VAD ---")
    reg_vad = evaluate_regression(X_vad, y_vad)
    for name, res in sorted(reg_vad.items(), key=lambda x: x[1]['mae']):
        print(f"  {name}: MAE={res['mae']:.2f}, r={res['r']:.3f}")

    # Step 3: Classification comparison
    print("\n" + "=" * 70)
    print("STEP 3: CLASSIFICATION (5-class quintiles)")
    print("=" * 70)

    y_class, thresholds = glucose_to_quintile(y_vad)
    print(f"\nQuintile thresholds: {[f'{t:.0f}' for t in thresholds]} mg/dL")

    class_results = evaluate_classification(X_vad, y_class)
    for name, res in sorted(class_results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {name}: Accuracy={res['accuracy']*100:.1f}%, F1={res['f1']:.3f}")

    # Step 4: Per-participant analysis
    print("\n" + "=" * 70)
    print("STEP 4: PER-PARTICIPANT ANALYSIS")
    print("=" * 70)

    data_by_participant = defaultdict(list)
    for d in data_with_vad:
        data_by_participant[d['participant']].append(d)

    personalized_results = {}

    for participant, pdata in data_by_participant.items():
        if len(pdata) < 20:
            continue

        X_p = np.array([d['features'] for d in pdata])
        y_p = np.array([d['glucose'] for d in pdata])

        results = evaluate_regression(X_p, y_p)
        if results:
            best = min(results.items(), key=lambda x: x[1]['mae'])
            personalized_results[participant] = {
                'n': len(y_p),
                'mae': best[1]['mae'],
                'r': best[1]['r'],
                'model': best[0]
            }
            print(f"  {participant}: n={len(y_p)}, MAE={best[1]['mae']:.2f}, r={best[1]['r']:.3f} ({best[0]})")

    # Step 5: Population model
    print("\n" + "=" * 70)
    print("STEP 5: POPULATION MODEL")
    print("=" * 70)

    participants = [d['participant'] for d in data_with_vad]
    groups = LabelEncoder().fit_transform(participants)

    # Leave-One-Person-Out
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42))
    ])

    y_pred_pop = cross_val_predict(pipe, X_vad, y_vad, cv=logo, groups=groups)
    pop_mae = mean_absolute_error(y_vad, y_pred_pop)
    pop_r = np.corrcoef(y_vad, y_pred_pop)[0, 1]

    print(f"\n  Population Model (LOPO): MAE={pop_mae:.2f}, r={pop_r:.3f}")

    # Calculate personalization benefit
    if personalized_results:
        avg_pers_mae = np.mean([p['mae'] for p in personalized_results.values()])
        improvement = pop_mae - avg_pers_mae
        print(f"\n  Personalization benefit: {improvement:.2f} mg/dL better MAE")

    # Step 6: Generate visualizations
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # VAD comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE comparison
    conditions = ['Without VAD', 'With VAD']
    best_mae_no_vad = min(r['mae'] for r in reg_no_vad.values())
    best_mae_vad = min(r['mae'] for r in reg_vad.values())

    axes[0].bar(conditions, [best_mae_no_vad, best_mae_vad], color=['coral', 'steelblue'])
    axes[0].set_ylabel('MAE (mg/dL)')
    axes[0].set_title('VAD Impact on Regression MAE')
    axes[0].grid(axis='y', alpha=0.3)

    # Correlation comparison
    best_r_no_vad = max(r['r'] for r in reg_no_vad.values())
    best_r_vad = max(r['r'] for r in reg_vad.values())

    axes[1].bar(conditions, [best_r_no_vad, best_r_vad], color=['coral', 'steelblue'])
    axes[1].set_ylabel('Correlation (r)')
    axes[1].set_title('VAD Impact on Correlation')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'vad_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created vad_comparison.png")

    # Personalization benefit
    if personalized_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        participants = list(personalized_results.keys())
        x = np.arange(len(participants))
        width = 0.35

        pop_maes = [pop_mae] * len(participants)
        pers_maes = [personalized_results[p]['mae'] for p in participants]

        ax.bar(x - width/2, pop_maes, width, label='Population Model', color='coral')
        ax.bar(x + width/2, pers_maes, width, label='Personalized Model', color='steelblue')

        ax.set_ylabel('MAE (mg/dL)')
        ax.set_title('Population vs Personalized Model Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(participants)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'personalization_benefit.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Created personalization_benefit.png")

    # Save summary
    summary = {
        'vad_comparison': {
            'without_vad': {'mae': best_mae_no_vad, 'r': best_r_no_vad, 'n_samples': len(data_no_vad)},
            'with_vad': {'mae': best_mae_vad, 'r': best_r_vad, 'n_samples': len(data_with_vad)}
        },
        'population_model': {'mae': pop_mae, 'r': pop_r},
        'personalized_models': personalized_results,
        'classification_5class': {name: {'accuracy': res['accuracy'], 'f1': res['f1']}
                                   for name, res in class_results.items()},
        'quintile_thresholds_mgdl': thresholds.tolist()
    }

    with open(OUTPUT_DIR / 'analysis_summary_v7.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nOutput: {OUTPUT_DIR}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nVAD Impact:")
    print(f"  Without VAD: MAE={best_mae_no_vad:.2f}, r={best_r_no_vad:.3f}")
    print(f"  With VAD:    MAE={best_mae_vad:.2f}, r={best_r_vad:.3f}")
    vad_benefit = best_mae_no_vad - best_mae_vad
    print(f"  VAD benefit: {vad_benefit:.2f} mg/dL {'improvement' if vad_benefit > 0 else 'worse'}")

    print(f"\nClassification (5-class quintiles):")
    best_class = max(class_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"  Best: {best_class[0]} with {best_class[1]['accuracy']*100:.1f}% accuracy")

    print(f"\nPopulation vs Personalization:")
    print(f"  Population MAE: {pop_mae:.2f}")
    if personalized_results:
        avg_pers = np.mean([p['mae'] for p in personalized_results.values()])
        print(f"  Avg Personalized MAE: {avg_pers:.2f}")
        print(f"  Personalization benefit: {pop_mae - avg_pers:.2f} mg/dL")


if __name__ == "__main__":
    main()
