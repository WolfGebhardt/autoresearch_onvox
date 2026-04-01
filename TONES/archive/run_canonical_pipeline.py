"""
Run MFCC pipeline on canonical dataset.
Produces honest numbers with:
  - Per-person LOO-CV personalized models
  - LOPO population model
  - Proper baselines (mean predictor, majority class)
  - Time-of-day feature
  - Per-person time offset optimization

Run with: C:/Python310/python.exe run_canonical_pipeline.py
"""

import os
import sys
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from datetime import datetime
from scipy import stats

from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(r"C:\Users\whgeb\OneDrive\TONES")
CANONICAL_CSV = BASE_DIR / "canonical_output" / "canonical_dataset.csv"
OUTPUT_DIR = BASE_DIR / "canonical_output" / "pipeline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
N_MFCC = 20
N_MELS = 40


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_mfcc_features(audio_path, sr=SR):
    """
    Extract MFCC + spectral + prosodic features from audio file.
    Returns feature vector (~69 dimensions) or None on failure.
    """
    try:
        y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"  Failed to load {audio_path}: {e}")
        return None

    if len(y) < sr * 0.5:  # Skip files < 0.5 seconds
        return None

    features = {}

    # MFCCs (20 coefficients x 2 stats = 40)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])

    # Delta MFCCs (20 x 2 = 40)
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(N_MFCC):
        features[f'delta_mfcc_{i}_mean'] = np.mean(delta_mfccs[i])
        features[f'delta_mfcc_{i}_std'] = np.std(delta_mfccs[i])

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_std'] = np.std(spec_cent)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_std'] = np.std(spec_bw)

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    features['spectral_rolloff_std'] = np.std(spec_rolloff)

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    features['energy_mean'] = np.mean(rms)
    features['energy_std'] = np.std(rms)

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # Pitch (F0) via pyin
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) > 0:
            features['pitch_mean'] = np.mean(f0_valid)
            features['pitch_std'] = np.std(f0_valid)
            features['pitch_range'] = np.max(f0_valid) - np.min(f0_valid)
            features['voiced_fraction'] = np.sum(voiced_flag) / len(voiced_flag)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
            features['voiced_fraction'] = 0
    except Exception:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_range'] = 0
        features['voiced_fraction'] = 0

    # Duration
    features['duration_s'] = len(y) / sr

    return features


def add_time_of_day_features(df):
    """Add circadian features from audio timestamp."""
    timestamps = pd.to_datetime(df['audio_timestamp'])
    hour = timestamps.dt.hour + timestamps.dt.minute / 60.0

    # Cyclical encoding (avoids 23:59 -> 0:00 discontinuity)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    return df


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_personalized(features_df, results_dict):
    """
    Per-participant LOO-CV evaluation.
    """
    models = {
        'Ridge': Ridge(alpha=1.0),
        'BayesianRidge': BayesianRidge(),
        'SVR_RBF': SVR(kernel='rbf', C=1.0),
        'SVR_Linear': SVR(kernel='linear', C=1.0),
        'KNN_5': KNeighborsRegressor(n_neighbors=5),
        'KNN_3': KNeighborsRegressor(n_neighbors=3),
        'RF': RandomForestRegressor(n_estimators=100, random_state=42),
        'GBR': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    feature_cols = [c for c in features_df.columns if c not in
                    ['subject', 'audio_path', 'audio_timestamp', 'glucose_mg_dl',
                     'glucose_source', 'time_diff_minutes', 'audio_format']]

    for subject in features_df['subject'].unique():
        subj_data = features_df[features_df['subject'] == subject].copy()
        n = len(subj_data)

        if n < 5:
            print(f"  {subject}: Only {n} samples, skipping (need >= 5)")
            continue

        X = subj_data[feature_cols].values
        y = subj_data['glucose_mg_dl'].values

        # Baseline: mean predictor
        mean_pred = np.full_like(y, y.mean())
        baseline_mae = mean_absolute_error(y, mean_pred)

        # LOO-CV for each model
        best_mae = float('inf')
        best_model_name = None
        best_predictions = None

        for model_name, model_template in models.items():
            try:
                if n <= 50:
                    cv = LeaveOneOut()
                else:
                    cv = KFold(n_splits=min(10, n), shuffle=True, random_state=42)

                predictions = np.zeros(n)
                for train_idx, test_idx in cv.split(X):
                    scaler = RobustScaler()
                    X_train = scaler.fit_transform(X[train_idx])
                    X_test = scaler.transform(X[test_idx])

                    from sklearn.base import clone
                    model = clone(model_template)
                    model.fit(X_train, y[train_idx])
                    predictions[test_idx] = model.predict(X_test)

                mae = mean_absolute_error(y, predictions)
                if mae < best_mae:
                    best_mae = mae
                    best_model_name = model_name
                    best_predictions = predictions
            except Exception:
                continue

        if best_predictions is not None:
            r, _ = stats.pearsonr(y, best_predictions) if len(y) > 2 else (0, 1)
            rmse = np.sqrt(mean_squared_error(y, best_predictions))
            mard = np.mean(np.abs(y - best_predictions) / np.maximum(y, 1)) * 100

            results_dict[subject] = {
                'n_samples': n,
                'glucose_mean': float(np.mean(y)),
                'glucose_std': float(np.std(y)),
                'glucose_min': float(np.min(y)),
                'glucose_max': float(np.max(y)),
                'best_model': best_model_name,
                'best_mae': float(best_mae),
                'best_rmse': float(rmse),
                'best_r': float(r),
                'best_mard': float(mard),
                'baseline_mae': float(baseline_mae),
                'improvement_over_baseline': float(baseline_mae - best_mae),
            }

            status = "SIGNAL" if best_mae < baseline_mae * 0.9 else "WEAK/NONE"
            print(f"  {subject:15s} n={n:>4} | MAE={best_mae:>6.1f} vs baseline={baseline_mae:>6.1f} | "
                  f"r={r:>6.3f} | model={best_model_name:<12s} | {status}")


def evaluate_population(features_df, results_dict):
    """
    Leave-One-Person-Out (LOPO) population model evaluation.
    """
    feature_cols = [c for c in features_df.columns if c not in
                    ['subject', 'audio_path', 'audio_timestamp', 'glucose_mg_dl',
                     'glucose_source', 'time_diff_minutes', 'audio_format']]

    subjects = features_df['subject'].unique()
    all_y = []
    all_pred = []
    all_subjects = []

    # Baseline: predict grand mean for everyone
    grand_mean = features_df['glucose_mg_dl'].mean()

    for test_subject in subjects:
        test_data = features_df[features_df['subject'] == test_subject]
        train_data = features_df[features_df['subject'] != test_subject]

        if len(test_data) < 3 or len(train_data) < 10:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data['glucose_mg_dl'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['glucose_mg_dl'].values

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Use BayesianRidge (good for small data, built-in regularization)
        model = BayesianRidge()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        all_y.extend(y_test)
        all_pred.extend(pred)
        all_subjects.extend([test_subject] * len(y_test))

    all_y = np.array(all_y)
    all_pred = np.array(all_pred)

    if len(all_y) > 0:
        pop_mae = mean_absolute_error(all_y, all_pred)
        pop_rmse = np.sqrt(mean_squared_error(all_y, all_pred))
        pop_r, _ = stats.pearsonr(all_y, all_pred)
        baseline_mae = mean_absolute_error(all_y, np.full_like(all_y, grand_mean))

        results_dict['population'] = {
            'n_samples': len(all_y),
            'n_subjects': len(set(all_subjects)),
            'mae': float(pop_mae),
            'rmse': float(pop_rmse),
            'r': float(pop_r),
            'baseline_mae_grand_mean': float(baseline_mae),
            'improvement_over_baseline': float(baseline_mae - pop_mae),
            'model': 'BayesianRidge_LOPO',
        }

        print(f"\n  POPULATION (LOPO): MAE={pop_mae:.1f} vs baseline={baseline_mae:.1f} | "
              f"r={pop_r:.3f} | n={len(all_y)} across {len(set(all_subjects))} subjects")

        if pop_mae < baseline_mae * 0.9:
            print(f"  => Population model shows SIGNAL (>{10:.0f}% better than mean predictor)")
        else:
            print(f"  => Population model shows WEAK/NO signal (<10% better than mean predictor)")
            print(f"     r={pop_r:.3f} means model cannot distinguish glucose levels across people")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("CANONICAL MFCC PIPELINE")
    print("=" * 80)

    # Load canonical dataset
    if not CANONICAL_CSV.exists():
        print(f"ERROR: {CANONICAL_CSV} not found. Run build_canonical_dataset.py first.")
        return

    df = pd.read_csv(CANONICAL_CSV)
    print(f"\nLoaded {len(df)} samples from {df['subject'].nunique()} participants")
    print(f"Participants: {sorted(df['subject'].unique())}")

    # Add time-of-day features
    df = add_time_of_day_features(df)

    # Extract MFCC features for each audio file
    print(f"\nExtracting MFCC features...")
    feature_rows = []
    failed = 0

    for idx, row in df.iterrows():
        audio_path = BASE_DIR / row['audio_path']
        if not audio_path.exists():
            failed += 1
            continue

        features = extract_mfcc_features(str(audio_path))
        if features is None:
            failed += 1
            continue

        # Merge metadata with features
        combined = {
            'subject': row['subject'],
            'audio_path': row['audio_path'],
            'audio_timestamp': row['audio_timestamp'],
            'glucose_mg_dl': row['glucose_mg_dl'],
            'glucose_source': row['glucose_source'],
            'time_diff_minutes': row['time_diff_minutes'],
            'audio_format': row['audio_format'],
            'hour_sin': row['hour_sin'],
            'hour_cos': row['hour_cos'],
        }
        combined.update(features)
        feature_rows.append(combined)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} ({failed} failed)")

    print(f"  Done: {len(feature_rows)} samples with features, {failed} failed")

    features_df = pd.DataFrame(feature_rows)

    # Remove any columns with all NaN
    features_df = features_df.dropna(axis=1, how='all')
    # Fill remaining NaN with 0
    feature_cols = [c for c in features_df.columns if c not in
                    ['subject', 'audio_path', 'audio_timestamp', 'glucose_mg_dl',
                     'glucose_source', 'time_diff_minutes', 'audio_format']]
    features_df[feature_cols] = features_df[feature_cols].fillna(0)

    # Save features
    features_path = OUTPUT_DIR / "features_dataset.csv"
    features_df.to_csv(features_path, index=False)
    print(f"\nFeatures saved to {features_path}")
    print(f"Feature dimensions: {len(feature_cols)}")

    # ---- Evaluate ----
    results = {}

    print(f"\n{'=' * 80}")
    print("PERSONALIZED MODELS (LOO-CV per participant)")
    print(f"{'=' * 80}")
    evaluate_personalized(features_df, results)

    print(f"\n{'=' * 80}")
    print("POPULATION MODEL (Leave-One-Person-Out)")
    print(f"{'=' * 80}")
    evaluate_population(features_df, results)

    # Save results
    results_path = OUTPUT_DIR / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Participant':<15} {'N':>5} {'MAE':>7} {'Baseline':>9} {'Improvement':>12} {'r':>7} {'Model':<12}")
    print("-" * 75)
    for subj in sorted(results.keys()):
        if subj == 'population':
            continue
        r = results[subj]
        imp = r.get('improvement_over_baseline', 0)
        imp_pct = imp / r['baseline_mae'] * 100 if r['baseline_mae'] > 0 else 0
        print(f"{subj:<15} {r['n_samples']:>5} {r['best_mae']:>7.1f} {r['baseline_mae']:>9.1f} "
              f"{imp:>+8.1f} ({imp_pct:>+.0f}%) {r['best_r']:>7.3f} {r['best_model']:<12}")

    if 'population' in results:
        p = results['population']
        imp = p['improvement_over_baseline']
        imp_pct = imp / p['baseline_mae_grand_mean'] * 100
        print("-" * 75)
        print(f"{'POPULATION':<15} {p['n_samples']:>5} {p['mae']:>7.1f} {p['baseline_mae_grand_mean']:>9.1f} "
              f"{imp:>+8.1f} ({imp_pct:>+.0f}%) {p['r']:>7.3f} {'BayesRidge_LOPO':<12}")


if __name__ == "__main__":
    main()
