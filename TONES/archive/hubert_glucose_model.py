"""
Voice-Based Glucose Estimation with HuBERT Transfer Learning
============================================================
Features:
1. HuBERT feature extraction (transfer learning)
2. Few-shot calibration for personalization
3. Population model + personalization pipeline
4. Production-ready API structure
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import librosa
from pathlib import Path
from datetime import datetime, timedelta
import re
import hashlib
import json
import pickle
import warnings
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneOut
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path("C:/Users/whgeb/OneDrive/TONES")
OUTPUT_DIR = BASE_DIR / "hubert_models"
OUTPUT_DIR.mkdir(exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================================
# HUBERT FEATURE EXTRACTOR
# ============================================================================

class HuBERTFeatureExtractor:
    """
    Extract features from audio using HuBERT (Hidden-Unit BERT).

    HuBERT learns robust speech representations through self-supervised
    learning on large amounts of unlabeled audio data.
    """

    def __init__(self, model_name="facebook/hubert-base-ls960", device=DEVICE):
        """
        Initialize HuBERT feature extractor.

        Args:
            model_name: HuBERT model from HuggingFace
            device: torch device (cuda/cpu)
        """
        self.device = device
        self.model_name = model_name

        print(f"Loading HuBERT model: {model_name}...")

        # Load feature extractor and model
        # Note: We use Wav2Vec2FeatureExtractor (not Processor) because we're
        # extracting features, not doing ASR which would need a tokenizer
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        print("HuBERT model loaded successfully!")

    def extract_features(self, audio_path, target_sr=16000):
        """
        Extract HuBERT features from an audio file.

        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (HuBERT expects 16kHz)

        Returns:
            features: numpy array of HuBERT features (768-dim for base model)
        """
        try:
            # Load audio using librosa (more compatible than torchaudio)
            waveform, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

            # Ensure float32
            waveform = waveform.astype(np.float32)

            # Process with HuBERT feature extractor
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.device)

            # Extract features
            with torch.no_grad():
                outputs = self.model(input_values)
                hidden_states = outputs.last_hidden_state  # [1, seq_len, 768]

            # Aggregate across time dimension
            # Use mean, std, max for rich representation
            features = hidden_states.squeeze(0).cpu().numpy()

            aggregated = np.concatenate([
                np.mean(features, axis=0),    # Mean across time
                np.std(features, axis=0),     # Std across time
                np.max(features, axis=0),     # Max across time
            ])

            return aggregated

        except Exception as e:
            print(f"Error extracting HuBERT features from {audio_path}: {e}")
            return None

    def extract_features_from_array(self, waveform, sr=16000):
        """
        Extract HuBERT features from a numpy array.

        Args:
            waveform: numpy array of audio samples
            sr: sample rate

        Returns:
            features: numpy array of HuBERT features
        """
        try:
            # Ensure float32
            waveform = waveform.astype(np.float32)

            # Process with feature extractor
            inputs = self.feature_extractor(
                waveform,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values)
                hidden_states = outputs.last_hidden_state

            features = hidden_states.squeeze(0).cpu().numpy()

            aggregated = np.concatenate([
                np.mean(features, axis=0),
                np.std(features, axis=0),
                np.max(features, axis=0),
            ])

            return aggregated

        except Exception as e:
            print(f"Error: {e}")
            return None


# ============================================================================
# FEW-SHOT CALIBRATION SYSTEM
# ============================================================================

class FewShotCalibrator:
    """
    Few-shot calibration system for personalizing glucose predictions.

    Uses a simple but effective approach:
    1. Start with population model predictions
    2. Learn a linear correction based on calibration samples
    3. Optionally learn user-specific time offset
    """

    def __init__(self, min_samples=5, max_samples=20):
        """
        Initialize calibrator.

        Args:
            min_samples: Minimum samples needed for calibration
            max_samples: Maximum samples to use (oldest dropped)
        """
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.calibration_samples = []
        self.correction_model = None
        self.optimal_offset = 0
        self.is_calibrated = False
        self.scaler = StandardScaler()

    def add_calibration_sample(self, features, predicted_glucose, actual_glucose, timestamp=None):
        """
        Add a calibration sample (when user provides fingerprick/CGM reading).

        Args:
            features: HuBERT features for the voice sample
            predicted_glucose: What the population model predicted
            actual_glucose: Actual glucose from CGM/fingerprick
            timestamp: When the sample was taken
        """
        sample = {
            'features': features,
            'predicted': predicted_glucose,
            'actual': actual_glucose,
            'timestamp': timestamp or datetime.now(),
            'error': actual_glucose - predicted_glucose
        }

        self.calibration_samples.append(sample)

        # Keep only most recent samples
        if len(self.calibration_samples) > self.max_samples:
            self.calibration_samples = self.calibration_samples[-self.max_samples:]

        # Re-calibrate if we have enough samples
        if len(self.calibration_samples) >= self.min_samples:
            self._update_calibration()

    def _update_calibration(self):
        """Update the calibration model based on collected samples."""

        # Prepare data
        X = np.array([s['features'] for s in self.calibration_samples])
        y_pred = np.array([s['predicted'] for s in self.calibration_samples])
        y_actual = np.array([s['actual'] for s in self.calibration_samples])

        # Method 1: Simple bias correction
        self.mean_error = np.mean(y_actual - y_pred)

        # Method 2: Linear correction model
        # Predict the error based on features
        errors = y_actual - y_pred

        # Combine features with predicted value for correction
        X_correction = np.column_stack([X, y_pred])

        # Fit correction model
        X_scaled = self.scaler.fit_transform(X_correction)
        self.correction_model = Ridge(alpha=1.0)
        self.correction_model.fit(X_scaled, errors)

        # Calculate calibration quality
        errors_corrected = self.correction_model.predict(X_scaled)
        self.calibration_mae = mean_absolute_error(errors, errors_corrected)

        self.is_calibrated = True

        print(f"Calibration updated with {len(self.calibration_samples)} samples")
        print(f"  Mean bias: {self.mean_error:.2f} mg/dL")
        print(f"  Correction MAE: {self.calibration_mae:.2f} mg/dL")

    def calibrate_prediction(self, features, population_prediction):
        """
        Apply calibration to a population model prediction.

        Args:
            features: HuBERT features for the voice sample
            population_prediction: Raw prediction from population model

        Returns:
            calibrated_prediction: Personalized prediction
        """
        if not self.is_calibrated:
            # Simple bias correction if calibrated, else return as-is
            if len(self.calibration_samples) > 0:
                return population_prediction + np.mean([s['error'] for s in self.calibration_samples])
            return population_prediction

        # Apply learned correction
        X_correction = np.column_stack([features.reshape(1, -1), [[population_prediction]]])
        X_scaled = self.scaler.transform(X_correction)
        correction = self.correction_model.predict(X_scaled)[0]

        return population_prediction + correction

    def get_calibration_stats(self):
        """Get calibration statistics."""
        if not self.calibration_samples:
            return {'status': 'not_calibrated', 'samples': 0}

        errors = [s['error'] for s in self.calibration_samples]

        return {
            'status': 'calibrated' if self.is_calibrated else 'collecting',
            'samples': len(self.calibration_samples),
            'min_required': self.min_samples,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'recent_errors': errors[-5:] if len(errors) >= 5 else errors
        }

    def save(self, path):
        """Save calibrator state."""
        state = {
            'calibration_samples': self.calibration_samples,
            'is_calibrated': self.is_calibrated,
            'mean_error': getattr(self, 'mean_error', 0),
            'optimal_offset': self.optimal_offset,
        }

        if self.correction_model is not None:
            state['correction_model'] = self.correction_model
            state['scaler'] = self.scaler

        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path):
        """Load calibrator state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.calibration_samples = state['calibration_samples']
        self.is_calibrated = state['is_calibrated']
        self.mean_error = state.get('mean_error', 0)
        self.optimal_offset = state.get('optimal_offset', 0)
        self.correction_model = state.get('correction_model')
        self.scaler = state.get('scaler', StandardScaler())


# ============================================================================
# GLUCOSE PREDICTION MODEL
# ============================================================================

class GlucosePredictionModel:
    """
    Complete glucose prediction model with:
    - HuBERT feature extraction
    - Population model
    - Few-shot personalization
    """

    def __init__(self, hubert_extractor=None):
        """
        Initialize the glucose prediction model.

        Args:
            hubert_extractor: Pre-initialized HuBERT extractor (optional)
        """
        self.hubert = hubert_extractor
        self.population_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.user_calibrators = {}  # Per-user calibration

    def initialize_hubert(self):
        """Initialize HuBERT feature extractor."""
        if self.hubert is None:
            self.hubert = HuBERTFeatureExtractor()

    def train_population_model(self, X, y, model_type='bayesian_ridge'):
        """
        Train the population model.

        Args:
            X: Features (HuBERT embeddings)
            y: Glucose values
            model_type: 'bayesian_ridge', 'random_forest', or 'ridge'
        """
        print(f"\nTraining population model ({model_type})...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Select model
        if model_type == 'bayesian_ridge':
            self.population_model = BayesianRidge()
        elif model_type == 'random_forest':
            self.population_model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42
            )
        else:
            self.population_model = Ridge(alpha=1.0)

        # Train
        self.population_model.fit(X_scaled, y)

        # Evaluate with CV
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(self.population_model, X_scaled, y, cv=cv)

        mae = mean_absolute_error(y, y_pred)
        r = np.corrcoef(y, y_pred)[0, 1]

        print(f"  Population model trained:")
        print(f"  - MAE: {mae:.2f} mg/dL")
        print(f"  - Correlation: {r:.3f}")
        print(f"  - Samples: {len(y)}")

        self.is_trained = True
        self.training_mae = mae
        self.training_r = r

        return {'mae': mae, 'r': r}

    def predict(self, audio_path_or_array, user_id=None, sr=16000):
        """
        Predict glucose from voice.

        Args:
            audio_path_or_array: Path to audio file or numpy array
            user_id: Optional user ID for personalized prediction
            sr: Sample rate (if array provided)

        Returns:
            prediction: Dict with glucose prediction and confidence
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_population_model first.")

        # Extract features
        if isinstance(audio_path_or_array, (str, Path)):
            features = self.hubert.extract_features(audio_path_or_array)
        else:
            features = self.hubert.extract_features_from_array(audio_path_or_array, sr)

        if features is None:
            return {'error': 'Feature extraction failed'}

        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Get population prediction
        pop_prediction = self.population_model.predict(features_scaled)[0]

        # Apply personalization if available
        if user_id and user_id in self.user_calibrators:
            calibrator = self.user_calibrators[user_id]
            final_prediction = calibrator.calibrate_prediction(features, pop_prediction)
            is_personalized = calibrator.is_calibrated
        else:
            final_prediction = pop_prediction
            is_personalized = False

        # Estimate confidence based on training performance
        confidence = max(0, 1 - (self.training_mae / 50))  # Rough confidence estimate

        return {
            'glucose_mgdl': float(final_prediction),
            'glucose_mmol': float(final_prediction / 18.0182),
            'population_prediction': float(pop_prediction),
            'is_personalized': is_personalized,
            'confidence': float(confidence),
            'features': features
        }

    def add_calibration(self, user_id, features, predicted, actual, timestamp=None):
        """
        Add a calibration sample for a user.

        Args:
            user_id: User identifier
            features: HuBERT features
            predicted: What was predicted
            actual: Actual glucose from CGM/fingerprick
            timestamp: When measured
        """
        if user_id not in self.user_calibrators:
            self.user_calibrators[user_id] = FewShotCalibrator()

        self.user_calibrators[user_id].add_calibration_sample(
            features, predicted, actual, timestamp
        )

    def get_user_calibration_status(self, user_id):
        """Get calibration status for a user."""
        if user_id not in self.user_calibrators:
            return {'status': 'no_calibration', 'samples': 0}

        return self.user_calibrators[user_id].get_calibration_stats()

    def save(self, path):
        """Save the complete model."""
        state = {
            'population_model': self.population_model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'training_mae': getattr(self, 'training_mae', None),
            'training_r': getattr(self, 'training_r', None),
            'user_calibrators': self.user_calibrators
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        print(f"Model saved to {path}")

    def load(self, path):
        """Load a saved model."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.population_model = state['population_model']
        self.scaler = state['scaler']
        self.is_trained = state['is_trained']
        self.training_mae = state.get('training_mae')
        self.training_r = state.get('training_r')
        self.user_calibrators = state.get('user_calibrators', {})

        print(f"Model loaded from {path}")


# ============================================================================
# DATA LOADING (Same as before)
# ============================================================================

PARTICIPANTS = {
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "audio_ext": [".wav"],
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
    match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h]?(\d{2})?', str(filename))
    if match:
        date_str = match.group(1)
        hour = int(match.group(2))
        minute = int(match.group(3))
        second = int(match.group(4)) if match.group(4) else 0
        return datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}:{second:02d}", "%Y-%m-%d %H:%M:%S")
    return None


def find_glucose(ts, glucose_df, offset_min=15, window_min=15):
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


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_hubert_glucose_model(max_samples_per_participant=100):
    """
    Train a HuBERT-based glucose prediction model.

    Args:
        max_samples_per_participant: Limit samples per participant for speed

    Returns:
        model: Trained GlucosePredictionModel
    """
    print("=" * 70)
    print("HUBERT GLUCOSE MODEL TRAINING")
    print("=" * 70)

    # Initialize HuBERT
    hubert = HuBERTFeatureExtractor()

    # Collect training data
    print("\n" + "=" * 70)
    print("EXTRACTING HUBERT FEATURES")
    print("=" * 70)

    all_features = []
    all_glucose = []
    all_participants = []
    all_timestamps = []

    for name, config in PARTICIPANTS.items():
        print(f"\n  Processing {name}...")

        glucose_df = load_glucose_data(config['glucose_csv'], config['glucose_unit'])
        if glucose_df.empty:
            print(f"    No glucose data")
            continue

        # Find audio files
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

        # Limit for speed
        if len(unique) > max_samples_per_participant:
            unique = unique[:max_samples_per_participant]

        print(f"    {len(unique)} audio files")

        count = 0
        for audio_path in unique:
            ts = parse_timestamp(audio_path.name)
            if ts is None:
                continue

            glucose = find_glucose(ts, glucose_df, offset_min=15)
            if glucose is None:
                continue

            # Extract HuBERT features
            features = hubert.extract_features(audio_path)
            if features is not None:
                all_features.append(features)
                all_glucose.append(glucose)
                all_participants.append(name)
                all_timestamps.append(ts)
                count += 1

                if count % 20 == 0:
                    print(f"      Processed {count} samples...")

        print(f"    {count} samples extracted")

    if not all_features:
        print("No data extracted!")
        return None

    print(f"\nTotal: {len(all_features)} samples from {len(set(all_participants))} participants")

    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_glucose)

    print(f"Feature dimension: {X.shape[1]}")

    # Create and train model
    print("\n" + "=" * 70)
    print("TRAINING POPULATION MODEL")
    print("=" * 70)

    model = GlucosePredictionModel(hubert_extractor=hubert)
    results = model.train_population_model(X, y, model_type='bayesian_ridge')

    # Test personalization simulation
    print("\n" + "=" * 70)
    print("SIMULATING FEW-SHOT PERSONALIZATION")
    print("=" * 70)

    # Group by participant
    participant_data = defaultdict(list)
    for i, p in enumerate(all_participants):
        participant_data[p].append({
            'features': all_features[i],
            'glucose': all_glucose[i],
            'timestamp': all_timestamps[i]
        })

    for participant, data in participant_data.items():
        if len(data) < 10:
            continue

        print(f"\n  {participant} ({len(data)} samples):")

        # Simulate: use first 5 samples for calibration, test on rest
        calibration_data = data[:5]
        test_data = data[5:]

        # Create calibrator
        calibrator = FewShotCalibrator(min_samples=5)

        # Get population predictions for calibration samples
        for sample in calibration_data:
            features = sample['features']
            features_scaled = model.scaler.transform(features.reshape(1, -1))
            pop_pred = model.population_model.predict(features_scaled)[0]

            calibrator.add_calibration_sample(
                features, pop_pred, sample['glucose'], sample['timestamp']
            )

        # Evaluate on test samples
        pop_errors = []
        cal_errors = []

        for sample in test_data:
            features = sample['features']
            features_scaled = model.scaler.transform(features.reshape(1, -1))
            pop_pred = model.population_model.predict(features_scaled)[0]

            cal_pred = calibrator.calibrate_prediction(features, pop_pred)

            pop_errors.append(abs(sample['glucose'] - pop_pred))
            cal_errors.append(abs(sample['glucose'] - cal_pred))

        pop_mae = np.mean(pop_errors)
        cal_mae = np.mean(cal_errors)
        improvement = pop_mae - cal_mae

        print(f"    Population MAE: {pop_mae:.2f} mg/dL")
        print(f"    Calibrated MAE: {cal_mae:.2f} mg/dL")
        print(f"    Improvement: {improvement:.2f} mg/dL ({improvement/pop_mae*100:.1f}%)")

        # Store calibrator
        model.user_calibrators[participant] = calibrator

    # Save model
    model_path = OUTPUT_DIR / "hubert_glucose_model.pkl"
    model.save(model_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel saved to: {model_path}")
    print(f"Population MAE: {results['mae']:.2f} mg/dL")
    print(f"Population r: {results['r']:.3f}")

    return model


# ============================================================================
# API SIMULATION
# ============================================================================

class GlucoseAPI:
    """
    Simulated API for glucose prediction service.

    This shows how the model would be used in production.
    """

    def __init__(self, model_path=None):
        """Initialize API with a trained model."""
        self.model = GlucosePredictionModel()

        if model_path and Path(model_path).exists():
            self.model.load(model_path)
            self.model.initialize_hubert()
        else:
            print("Warning: No model loaded. Train a model first.")

    def predict(self, audio_path, user_id=None):
        """
        Predict glucose from voice recording.

        Args:
            audio_path: Path to audio file
            user_id: Optional user ID for personalization

        Returns:
            JSON-like dict with prediction
        """
        result = self.model.predict(audio_path, user_id=user_id)

        # Remove features from response (too large)
        if 'features' in result:
            del result['features']

        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['user_id'] = user_id

        return result

    def calibrate(self, audio_path, actual_glucose, user_id):
        """
        Add a calibration point for a user.

        Args:
            audio_path: Path to the audio file
            actual_glucose: Actual glucose reading (mg/dL)
            user_id: User identifier

        Returns:
            Calibration status
        """
        # Get prediction and features
        result = self.model.predict(audio_path, user_id=user_id)

        if 'error' in result:
            return result

        # Add calibration
        self.model.add_calibration(
            user_id=user_id,
            features=result.get('features', np.zeros(768*3)),  # Placeholder if missing
            predicted=result['population_prediction'],
            actual=actual_glucose
        )

        return self.model.get_user_calibration_status(user_id)

    def get_calibration_status(self, user_id):
        """Get calibration status for a user."""
        return self.model.get_user_calibration_status(user_id)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Train the model
    model = train_hubert_glucose_model(max_samples_per_participant=50)

    if model:
        # Demo API usage
        print("\n" + "=" * 70)
        print("API USAGE DEMO")
        print("=" * 70)

        print("""
# Example API usage:

from hubert_glucose_model import GlucoseAPI

# Initialize API
api = GlucoseAPI(model_path='hubert_models/hubert_glucose_model.pkl')

# Predict glucose (population model)
result = api.predict('path/to/voice.wav')
print(f"Predicted glucose: {result['glucose_mgdl']:.1f} mg/dL")

# Add calibration sample (when user provides CGM/fingerprick reading)
api.calibrate('path/to/voice.wav', actual_glucose=120, user_id='user123')

# Predict with personalization
result = api.predict('path/to/voice.wav', user_id='user123')
print(f"Personalized prediction: {result['glucose_mgdl']:.1f} mg/dL")

# Check calibration status
status = api.get_calibration_status('user123')
print(f"Calibration: {status['samples']} samples, {status['status']}")
""")
