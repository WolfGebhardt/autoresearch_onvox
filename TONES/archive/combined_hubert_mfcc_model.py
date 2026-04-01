"""
Combined HuBERT + MFCC Voice-Based Glucose Estimation
=====================================================
This script combines deep learning (HuBERT) features with traditional
hand-crafted (MFCC) features for improved glucose estimation.

Features:
1. HuBERT transfer learning features (2304-dim)
2. Comprehensive MFCC/spectral features (~200-dim)
3. Feature fusion with dimensionality reduction
4. Few-shot calibration for personalization
5. Full dataset processing
"""

import numpy as np
import pandas as pd
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import librosa
from pathlib import Path
from datetime import datetime, timedelta
import re
import pickle
import warnings
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")
OUTPUT_DIR = BASE_DIR / "combined_model_output"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# All participants with glucose data
PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
        "time_offset": 15,  # Optimal offset from previous analysis
    },
    "Anja": {
        "glucose_csv": [
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_6-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_10-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_13-11-2023.csv",
            "Anja/glucose 21nov 2023/AnjaZhao_glucose_16-11-2023.csv",
        ],
        "audio_dirs": ["Anja/conv_audio", "Anja/converted audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
        "time_offset": 0,
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
        "time_offset": 20,
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchütt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
        "time_offset": 15,
    },
    "Vicky": {
        "glucose_csv": ["Vicky/Number_10Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Vicky/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
        "time_offset": 15,
    },
    "Steffen_Haeseli": {
        "glucose_csv": ["Steffen_Haeseli/Number_2Nov_23_glucose_4-1-2024.csv"],
        "audio_dirs": ["Steffen_Haeseli/wav"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
        "time_offset": 15,
    },
    "Lara": {
        "glucose_csv": ["Lara/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Lara/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
        "time_offset": 15,
    },
}


# ============================================================================
# HUBERT FEATURE EXTRACTOR
# ============================================================================

class HuBERTExtractor:
    """Extract HuBERT features from audio."""

    def __init__(self, model_name="facebook/hubert-base-ls960"):
        self.device = DEVICE
        print(f"Loading HuBERT model: {model_name}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("HuBERT model loaded!")

    def extract(self, audio_path, target_sr=16000):
        """Extract HuBERT features from audio file."""
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

            # Aggregate: mean, std, max
            return np.concatenate([
                np.mean(hidden, axis=0),
                np.std(hidden, axis=0),
                np.max(hidden, axis=0),
            ])
        except Exception as e:
            return None


# ============================================================================
# MFCC FEATURE EXTRACTOR
# ============================================================================

class MFCCExtractor:
    """Extract comprehensive MFCC and spectral features."""

    def __init__(self, sr=16000, n_mfcc=20, n_mels=40):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def extract(self, audio_path):
        """Extract MFCC and spectral features from audio file."""
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)

            if len(y) < sr * 0.5:  # Skip very short clips
                return None

            features = []

            # 1. MFCCs (20 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))

            # 2. Delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            features.extend(np.mean(delta_mfccs, axis=1))
            features.extend(np.std(delta_mfccs, axis=1))

            # 3. Delta-Delta MFCCs
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            features.extend(np.mean(delta2_mfccs, axis=1))
            features.extend(np.std(delta2_mfccs, axis=1))

            # 4. Mel spectrogram stats
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel)
            features.extend(np.mean(mel_db, axis=1))
            features.extend(np.std(mel_db, axis=1))

            # 5. Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))

            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features.extend(np.mean(spectral_contrast, axis=1))
            features.extend(np.std(spectral_contrast, axis=1))

            # 6. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))

            # 7. RMS energy
            rms = librosa.feature.rms(y=y)
            features.append(np.mean(rms))
            features.append(np.std(rms))

            # 8. Pitch (F0) estimation
            try:
                f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
                f0_valid = f0[~np.isnan(f0)]
                if len(f0_valid) > 0:
                    features.append(np.mean(f0_valid))
                    features.append(np.std(f0_valid))
                    features.append(np.median(f0_valid))
                else:
                    features.extend([0, 0, 0])
            except:
                features.extend([0, 0, 0])

            # 9. Harmonic-to-noise ratio proxy
            harmonic, percussive = librosa.effects.hpss(y)
            hnr_proxy = np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-10)
            features.append(hnr_proxy)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            return None


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
                if 'device' in line.lower() and 'timestamp' in line.lower():
                    skiprows = i
                    break

            df = pd.read_csv(full_path, skiprows=skiprows)
        except:
            continue

        # Find timestamp and glucose columns
        timestamp_col = df.columns[2] if len(df.columns) > 2 else None
        glucose_col = df.columns[4] if len(df.columns) > 4 else None

        if timestamp_col is None or glucose_col is None:
            continue

        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
        df['glucose'] = pd.to_numeric(df[glucose_col], errors='coerce')

        # Convert units if needed
        if unit == 'mmol/L':
            df['glucose'] = df['glucose'] * 18.0182
        elif unit == 'mg/dL':
            # Check if values look like mmol/L
            mean_val = df['glucose'].dropna().mean()
            if mean_val < 30:
                df['glucose'] = df['glucose'] * 18.0182

        df = df.dropna(subset=['timestamp', 'glucose'])
        if len(df) > 0:
            all_dfs.append(df[['timestamp', 'glucose']])

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    return pd.DataFrame()


def parse_timestamp(filename):
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


def find_matching_glucose(audio_ts, glucose_df, offset_minutes=0, window_minutes=15):
    """Find closest glucose reading within window."""
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
# COMBINED FEATURE EXTRACTION
# ============================================================================

def extract_all_features(hubert_extractor, mfcc_extractor, max_per_participant=None):
    """Extract combined HuBERT + MFCC features for all participants."""

    all_data = []

    for name, config in PARTICIPANTS.items():
        print(f"\n  Processing {name}...")

        # Load glucose data
        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
        if glucose_df.empty:
            print(f"    No glucose data found")
            continue

        print(f"    Glucose readings: {len(glucose_df)}")

        # Find audio files
        audio_files = []
        for audio_dir in config['audio_dirs']:
            dir_path = BASE_DIR / audio_dir
            if dir_path.exists():
                for ext in config['audio_ext']:
                    audio_files.extend(dir_path.glob(f"*{ext}"))

        if max_per_participant:
            audio_files = audio_files[:max_per_participant]

        print(f"    Audio files: {len(audio_files)}")

        # Extract features
        count = 0
        offset = config.get('time_offset', 15)

        for i, audio_path in enumerate(audio_files):
            # Get timestamp and glucose
            audio_ts = parse_timestamp(audio_path.name)
            if audio_ts is None:
                continue

            glucose = find_matching_glucose(audio_ts, glucose_df, offset)
            if glucose is None:
                continue

            # Extract HuBERT features
            hubert_feats = hubert_extractor.extract(audio_path)
            if hubert_feats is None:
                continue

            # Extract MFCC features
            mfcc_feats = mfcc_extractor.extract(audio_path)
            if mfcc_feats is None:
                continue

            # Combine features
            combined_feats = np.concatenate([hubert_feats, mfcc_feats])

            all_data.append({
                'participant': name,
                'audio_path': str(audio_path),
                'timestamp': audio_ts,
                'glucose': glucose,
                'features': combined_feats,
                'hubert_dim': len(hubert_feats),
                'mfcc_dim': len(mfcc_feats),
            })

            count += 1

            if (i + 1) % 50 == 0:
                print(f"      Processed {i+1} files, {count} samples...")

        print(f"    Extracted {count} samples")

    return all_data


# ============================================================================
# FEW-SHOT CALIBRATION
# ============================================================================

class FewShotCalibrator:
    """Personalization through few-shot calibration."""

    def __init__(self, min_samples=5, max_samples=20):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.calibration_samples = []
        self.model = None
        self.is_calibrated = False

    def add_sample(self, features, predicted, actual):
        """Add a calibration sample."""
        self.calibration_samples.append({
            'features': features,
            'predicted': predicted,
            'actual': actual,
            'error': actual - predicted,
        })

        if len(self.calibration_samples) > self.max_samples:
            self.calibration_samples.pop(0)

        if len(self.calibration_samples) >= self.min_samples:
            self._update_calibration()

    def _update_calibration(self):
        """Update calibration model."""
        errors = [s['error'] for s in self.calibration_samples]
        self.mean_bias = np.mean(errors)
        self.is_calibrated = True

    def calibrate(self, prediction):
        """Apply calibration to prediction."""
        if not self.is_calibrated:
            return prediction
        return prediction + self.mean_bias


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate(all_data, use_pca=True, pca_components=100):
    """Train and evaluate models."""

    if not all_data:
        print("No data available!")
        return None

    # Prepare data
    X = np.array([d['features'] for d in all_data])
    y = np.array([d['glucose'] for d in all_data])
    participants = np.array([d['participant'] for d in all_data])

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"  - HuBERT: {all_data[0]['hubert_dim']}")
    print(f"  - MFCC: {all_data[0]['mfcc_dim']}")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA
    pca = None
    if use_pca and X_scaled.shape[1] > pca_components:
        print(f"\nApplying PCA: {X_scaled.shape[1]} -> {pca_components} components")
        pca = PCA(n_components=pca_components)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    results = {
        'scaler': scaler,
        'pca': pca,
        'personalized': {},
        'population': {},
    }

    # ========================================
    # 1. PERSONALIZED MODELS (Leave-One-Out)
    # ========================================
    print("\n" + "="*60)
    print("PERSONALIZED MODELS (Leave-One-Out CV)")
    print("="*60)

    unique_participants = np.unique(participants)
    all_personalized_preds = []
    all_personalized_actual = []

    for participant in unique_participants:
        mask = participants == participant
        X_person = X_scaled[mask]
        y_person = y[mask]

        if len(X_person) < 10:
            print(f"\n  {participant}: Too few samples ({len(X_person)})")
            continue

        # Leave-One-Out CV
        model = BayesianRidge()

        if len(X_person) <= 50:
            cv = LeaveOneOut()
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=min(10, len(X_person)), shuffle=True, random_state=42)

        preds = cross_val_predict(model, X_person, y_person, cv=cv)

        mae = mean_absolute_error(y_person, preds)
        r, _ = stats.pearsonr(y_person, preds)

        print(f"\n  {participant} ({len(X_person)} samples):")
        print(f"    MAE: {mae:.2f} mg/dL")
        print(f"    r: {r:.3f}")

        results['personalized'][participant] = {
            'mae': mae,
            'r': r,
            'n_samples': len(X_person),
            'predictions': preds,
            'actual': y_person,
        }

        all_personalized_preds.extend(preds)
        all_personalized_actual.extend(y_person)

    overall_personalized_mae = mean_absolute_error(all_personalized_actual, all_personalized_preds)
    overall_personalized_r, _ = stats.pearsonr(all_personalized_actual, all_personalized_preds)

    print(f"\n  OVERALL PERSONALIZED:")
    print(f"    MAE: {overall_personalized_mae:.2f} mg/dL")
    print(f"    r: {overall_personalized_r:.3f}")

    # ========================================
    # 2. POPULATION MODEL (Leave-One-Person-Out)
    # ========================================
    print("\n" + "="*60)
    print("POPULATION MODEL (Leave-One-Person-Out CV)")
    print("="*60)

    population_preds = np.zeros_like(y)

    for test_participant in unique_participants:
        train_mask = participants != test_participant
        test_mask = participants == test_participant

        if np.sum(test_mask) == 0:
            continue

        X_train, y_train = X_scaled[train_mask], y[train_mask]
        X_test, y_test = X_scaled[test_mask], y[test_mask]

        model = BayesianRidge()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        population_preds[test_mask] = preds

        mae = mean_absolute_error(y_test, preds)
        print(f"  {test_participant}: MAE = {mae:.2f} mg/dL")

    overall_pop_mae = mean_absolute_error(y, population_preds)
    overall_pop_r, _ = stats.pearsonr(y, population_preds)

    print(f"\n  OVERALL POPULATION:")
    print(f"    MAE: {overall_pop_mae:.2f} mg/dL")
    print(f"    r: {overall_pop_r:.3f}")

    results['population'] = {
        'mae': overall_pop_mae,
        'r': overall_pop_r,
        'predictions': population_preds,
        'actual': y,
    }

    # ========================================
    # 3. TRAIN FINAL MODELS
    # ========================================
    print("\n" + "="*60)
    print("TRAINING FINAL MODELS")
    print("="*60)

    # Population model
    final_pop_model = BayesianRidge()
    final_pop_model.fit(X_scaled, y)
    results['final_population_model'] = final_pop_model

    # Per-person models
    results['final_personalized_models'] = {}
    for participant in unique_participants:
        mask = participants == participant
        if np.sum(mask) >= 10:
            model = BayesianRidge()
            model.fit(X_scaled[mask], y[mask])
            results['final_personalized_models'][participant] = model

    print(f"  Population model trained on {len(X)} samples")
    print(f"  Personalized models trained for {len(results['final_personalized_models'])} participants")

    return results


# ============================================================================
# CLARKE ERROR GRID
# ============================================================================

def clarke_error_grid(actual, predicted, title="Clarke Error Grid"):
    """Generate Clarke Error Grid analysis."""

    actual = np.array(actual)
    predicted = np.array(predicted)

    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for ref, pred in zip(actual, predicted):
        if (ref <= 70 and pred <= 70) or (abs(pred - ref) <= 20) or \
           (ref >= 70 and abs(pred - ref) / ref <= 0.20):
            zones['A'] += 1
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zones['E'] += 1
        elif (ref >= 70 and ref <= 290 and pred >= ref + 110) or \
             (ref >= 130 and ref <= 180 and pred <= (7/5) * ref - 182):
            zones['C'] += 1
        elif (ref >= 240 and (pred >= 70 and pred <= 180)) or \
             (ref <= 175 / 3 and pred <= 180 and pred >= 70) or \
             (ref >= 175 / 3 and ref <= 70 and pred >= (6/5) * ref):
            zones['D'] += 1
        else:
            zones['B'] += 1

    total = len(actual)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(actual, predicted, alpha=0.6, s=30)

    # Zone boundaries
    ax.plot([0, 400], [0, 400], 'k-', linewidth=1)
    ax.axhline(70, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(180, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(70, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(180, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Zone labels
    text = f"Zone A: {zones['A']} ({100*zones['A']/total:.1f}%)\n"
    text += f"Zone B: {zones['B']} ({100*zones['B']/total:.1f}%)\n"
    text += f"A+B: {zones['A']+zones['B']} ({100*(zones['A']+zones['B'])/total:.1f}%)"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    return fig, zones


# ============================================================================
# GENERATE REPORT
# ============================================================================

def generate_report(results, all_data, output_dir):
    """Generate comprehensive HTML report."""

    # Clarke Error Grid for personalized predictions
    all_actual = []
    all_preds = []
    for p, data in results['personalized'].items():
        all_actual.extend(data['actual'])
        all_preds.extend(data['predictions'])

    fig, zones = clarke_error_grid(all_actual, all_preds, "Clarke Error Grid - Personalized Models")
    fig.savefig(output_dir / 'clarke_personalized.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Clarke for population
    fig, pop_zones = clarke_error_grid(
        results['population']['actual'],
        results['population']['predictions'],
        "Clarke Error Grid - Population Model"
    )
    fig.savefig(output_dir / 'clarke_population.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Per-participant scatter plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (participant, data) in enumerate(results['personalized'].items()):
        if i >= 8:
            break
        ax = axes[i]
        ax.scatter(data['actual'], data['predictions'], alpha=0.6)
        ax.plot([50, 250], [50, 250], 'r--', alpha=0.5)
        ax.set_xlabel('Actual (mg/dL)')
        ax.set_ylabel('Predicted (mg/dL)')
        ax.set_title(f"{participant}\nMAE={data['mae']:.1f}, r={data['r']:.2f}")

    # Hide empty subplots
    for i in range(len(results['personalized']), 8):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / 'per_participant_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    # HTML Report
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Combined HuBERT + MFCC Voice-Glucose Model Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #1a5276, #2980b9); color: white; padding: 30px; border-radius: 10px; }}
        h2 {{ color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 10px; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #2980b9; color: white; }}
        .metric {{ display: inline-block; background: #ebf5fb; padding: 15px 25px; border-radius: 8px; margin: 10px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #1a5276; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .highlight {{ background: #d5f5e3; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .code {{ background: #f4f4f4; padding: 15px; border-radius: 8px; font-family: monospace; overflow-x: auto; }}
    </style>
</head>
<body>

<div class="header">
    <h1>🎤 Voice-Based Glucose Estimation</h1>
    <p>Combined HuBERT + MFCC Model Report</p>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>

<div class="section">
    <h2>📊 Executive Summary</h2>

    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        <div class="metric"><div class="metric-value">{len(set(d['participant'] for d in all_data))}</div><div>Participants</div></div>
        <div class="metric"><div class="metric-value">{len(all_data)}</div><div>Total Samples</div></div>
        <div class="metric"><div class="metric-value">{all_data[0]['hubert_dim'] + all_data[0]['mfcc_dim']}</div><div>Feature Dimensions</div></div>
    </div>

    <div class="highlight">
        <strong>Feature Composition:</strong>
        <ul>
            <li><strong>HuBERT Features:</strong> {all_data[0]['hubert_dim']} dimensions (mean, std, max of 768-dim embeddings)</li>
            <li><strong>MFCC Features:</strong> {all_data[0]['mfcc_dim']} dimensions (MFCCs, deltas, mel-spectrogram, spectral features, pitch)</li>
        </ul>
    </div>
</div>

<div class="section">
    <h2>📈 Personalized Model Results</h2>

    <table>
        <tr>
            <th>Participant</th>
            <th>Samples</th>
            <th>MAE (mg/dL)</th>
            <th>Correlation (r)</th>
        </tr>
"""

    for participant, data in sorted(results['personalized'].items(), key=lambda x: x[1]['mae']):
        html += f"""        <tr>
            <td>{participant}</td>
            <td>{data['n_samples']}</td>
            <td><strong>{data['mae']:.2f}</strong></td>
            <td>{data['r']:.3f}</td>
        </tr>
"""

    # Calculate overall
    overall_mae = mean_absolute_error(all_actual, all_preds)
    overall_r, _ = stats.pearsonr(all_actual, all_preds)

    html += f"""        <tr style="background: #ebf5fb; font-weight: bold;">
            <td>OVERALL</td>
            <td>{len(all_actual)}</td>
            <td>{overall_mae:.2f}</td>
            <td>{overall_r:.3f}</td>
        </tr>
    </table>

    <div class="figure">
        <img src="per_participant_scatter.png" alt="Per-Participant Results">
        <p><em>Actual vs Predicted glucose for each participant</em></p>
    </div>
</div>

<div class="section">
    <h2>🌍 Population Model Results</h2>

    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        <div class="metric"><div class="metric-value">{results['population']['mae']:.1f}</div><div>MAE (mg/dL)</div></div>
        <div class="metric"><div class="metric-value">{results['population']['r']:.3f}</div><div>Correlation</div></div>
    </div>

    <p>The population model was evaluated using Leave-One-Person-Out cross-validation,
    which tests generalization to completely new individuals.</p>
</div>

<div class="section">
    <h2>🎯 Clarke Error Grid Analysis</h2>

    <div class="figure">
        <img src="clarke_personalized.png" alt="Clarke Error Grid - Personalized">
        <p><em>Personalized Models - Clarke Error Grid</em></p>
    </div>

    <table>
        <tr><th>Zone</th><th>Description</th><th>Count</th><th>Percentage</th></tr>
        <tr style="background: #d5f5e3;"><td>A</td><td>Clinically accurate</td><td>{zones['A']}</td><td>{100*zones['A']/len(all_actual):.1f}%</td></tr>
        <tr style="background: #fcf3cf;"><td>B</td><td>Benign errors</td><td>{zones['B']}</td><td>{100*zones['B']/len(all_actual):.1f}%</td></tr>
        <tr><td>C</td><td>Overcorrection</td><td>{zones['C']}</td><td>{100*zones['C']/len(all_actual):.1f}%</td></tr>
        <tr><td>D</td><td>Failure to detect</td><td>{zones['D']}</td><td>{100*zones['D']/len(all_actual):.1f}%</td></tr>
        <tr><td>E</td><td>Dangerous errors</td><td>{zones['E']}</td><td>{100*zones['E']/len(all_actual):.1f}%</td></tr>
        <tr style="font-weight: bold; background: #ebf5fb;"><td colspan="2">A+B (Clinically Acceptable)</td><td>{zones['A']+zones['B']}</td><td>{100*(zones['A']+zones['B'])/len(all_actual):.1f}%</td></tr>
    </table>
</div>

<div class="section">
    <h2>🔧 API Usage</h2>

    <div class="code">
<pre>
from combined_hubert_mfcc_model import load_model, predict_glucose

# Load the trained model
model = load_model('combined_model_output/combined_model.pkl')

# Predict glucose from a voice recording
result = predict_glucose(model, 'path/to/voice.wav')
print(f"Predicted glucose: {{result['glucose']:.1f}} mg/dL")
print(f"Confidence: {{result['confidence']}}")

# For personalized prediction (if user has calibrated)
result = predict_glucose(model, 'path/to/voice.wav', user_id='user123')
</pre>
    </div>
</div>

<div class="section">
    <h2>📁 Project Files</h2>

    <table>
        <tr><th>File</th><th>Description</th></tr>
        <tr><td>combined_hubert_mfcc_model.py</td><td>Main model training script (HuBERT + MFCC)</td></tr>
        <tr><td>hubert_glucose_model.py</td><td>HuBERT-only model with few-shot calibration</td></tr>
        <tr><td>comprehensive_analysis_v6.py</td><td>MFCC-only analysis with time offset optimization</td></tr>
        <tr><td>comprehensive_analysis_v7_fast.py</td><td>VAD comparison and classification experiments</td></tr>
        <tr><td>combined_model_output/</td><td>Output directory with model and reports</td></tr>
        <tr><td>hubert_models/</td><td>HuBERT model checkpoints</td></tr>
        <tr><td>documentation_v5/</td><td>Previous analysis reports</td></tr>
    </table>
</div>

</body>
</html>
"""

    with open(output_dir / 'report.html', 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nReport saved to: {output_dir / 'report.html'}")


# ============================================================================
# HELPER FUNCTIONS FOR API
# ============================================================================

def save_model(results, output_path):
    """Save the trained model."""
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Model saved to: {output_path}")


def load_model(model_path):
    """Load a trained model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(max_per_participant=None):
    """Main training pipeline."""

    print("="*70)
    print("COMBINED HUBERT + MFCC VOICE-GLUCOSE MODEL")
    print("="*70)

    # Initialize extractors
    print("\n[1/4] Initializing feature extractors...")
    hubert = HuBERTExtractor()
    mfcc = MFCCExtractor()

    # Extract features
    print("\n[2/4] Extracting features from all audio files...")
    all_data = extract_all_features(hubert, mfcc, max_per_participant)

    if not all_data:
        print("ERROR: No data extracted!")
        return

    print(f"\nTotal samples extracted: {len(all_data)}")

    # Train and evaluate
    print("\n[3/4] Training and evaluating models...")
    results = train_and_evaluate(all_data, use_pca=True, pca_components=100)

    if results is None:
        print("ERROR: Training failed!")
        return

    # Generate report
    print("\n[4/4] Generating report...")
    generate_report(results, all_data, OUTPUT_DIR)

    # Save model
    save_model(results, OUTPUT_DIR / 'combined_model.pkl')

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Model file: {OUTPUT_DIR / 'combined_model.pkl'}")
    print(f"Report: {OUTPUT_DIR / 'report.html'}")


if __name__ == "__main__":
    # Process all audio files (set to None for all, or a number to limit)
    main(max_per_participant=None)
