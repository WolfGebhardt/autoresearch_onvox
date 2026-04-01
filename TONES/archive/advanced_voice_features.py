"""
Advanced Voice Features for Glucose Estimation
===============================================
This module implements specialized acoustic features that research has
shown to correlate with blood glucose levels.

Key physiological mechanisms:
1. Hypoglycemia affects autonomic nervous system → vocal cord tension changes
2. Glucose levels affect cognitive processing → speech timing changes
3. Blood viscosity changes → subtle voice quality differences
4. Fatigue/alertness → prosodic variations

References:
- Pham et al. (2020) "Detecting Hypoglycemia from Voice"
- Sriram et al. (2020) "Voice-based Biomarkers for Health"
- Various diabetes-speech studies from 2018-2024
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from scipy import signal
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')


class AdvancedVoiceFeatureExtractor:
    """
    Extract voice features specifically relevant to glucose estimation.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def extract_all(self, audio_path: str) -> Dict[str, float]:
        """Extract all glucose-relevant voice features."""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return {}

        if len(y) < sr * 0.5:
            return {}

        features = {}

        # Core feature groups
        features.update(self._extract_pitch_features(y))
        features.update(self._extract_voice_quality_features(y))
        features.update(self._extract_energy_features(y))
        features.update(self._extract_timing_features(y))
        features.update(self._extract_spectral_features(y))
        features.update(self._extract_formant_features(y))
        features.update(self._extract_tremor_features(y))

        return features

    def _extract_pitch_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Pitch (F0) features - fundamental frequency of voice.
        Hypoglycemia can cause pitch instability and changes in mean pitch.
        """
        features = {}

        # Use pyin for more robust pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=50, fmax=500, sr=self.sr
        )

        # Remove unvoiced frames
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) > 0:
            features['f0_mean'] = np.mean(f0_voiced)
            features['f0_std'] = np.std(f0_voiced)
            features['f0_min'] = np.min(f0_voiced)
            features['f0_max'] = np.max(f0_voiced)
            features['f0_range'] = features['f0_max'] - features['f0_min']
            features['f0_median'] = np.median(f0_voiced)

            # F0 variability measures (important for glucose)
            features['f0_cv'] = features['f0_std'] / features['f0_mean'] if features['f0_mean'] > 0 else 0

            # F0 contour dynamics
            if len(f0_voiced) > 1:
                f0_diff = np.diff(f0_voiced)
                features['f0_delta_mean'] = np.mean(np.abs(f0_diff))
                features['f0_delta_std'] = np.std(f0_diff)

                # Pitch perturbation quotient (related to jitter)
                features['f0_ppq'] = np.mean(np.abs(f0_diff)) / features['f0_mean'] if features['f0_mean'] > 0 else 0

            # Distribution shape
            features['f0_skewness'] = skew(f0_voiced)
            features['f0_kurtosis'] = kurtosis(f0_voiced)

            # Voiced ratio (speech fluency indicator)
            features['voiced_ratio'] = np.sum(~np.isnan(f0)) / len(f0)

        return features

    def _extract_voice_quality_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Voice quality features - jitter, shimmer approximations.
        These reflect micro-instabilities in voice production.
        Hypoglycemia causes tremor which increases jitter/shimmer.
        """
        features = {}

        # Frame-level analysis
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.010 * self.sr)    # 10ms hop

        # Get amplitude envelope
        amplitude_envelope = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
        frame_energy = np.sum(amplitude_envelope ** 2, axis=0)

        if len(frame_energy) > 1:
            # Shimmer-like measure (amplitude perturbation)
            energy_diff = np.abs(np.diff(frame_energy))
            features['shimmer_approx'] = np.mean(energy_diff) / (np.mean(frame_energy) + 1e-10)

            # Amplitude perturbation quotient
            features['apq'] = np.mean(energy_diff) / np.mean(frame_energy[:-1] + 1e-10)

        # Harmonics-to-Noise Ratio approximation
        # High HNR = clear voice, Low HNR = breathy/hoarse
        S = np.abs(librosa.stft(y))
        harmonic, percussive = librosa.decompose.hpss(S)

        harmonic_energy = np.sum(harmonic ** 2)
        noise_energy = np.sum(percussive ** 2)

        if noise_energy > 0:
            features['hnr_approx'] = 10 * np.log10(harmonic_energy / noise_energy)
        else:
            features['hnr_approx'] = 30  # Max value if no noise

        # Cepstral Peak Prominence (voice quality measure)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        features['cpp'] = np.max(np.mean(mfcc, axis=1))

        return features

    def _extract_energy_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Energy/intensity features.
        Low glucose can cause fatigue → reduced vocal energy.
        """
        features = {}

        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_max'] = np.max(rms)
        features['energy_min'] = np.min(rms)
        features['energy_range'] = features['energy_max'] - features['energy_min']

        # Dynamic range
        if features['energy_min'] > 0:
            features['energy_dynamic_range'] = 20 * np.log10(features['energy_max'] / features['energy_min'])
        else:
            features['energy_dynamic_range'] = 0

        # Energy variation over time
        if len(rms) > 1:
            features['energy_delta_mean'] = np.mean(np.abs(np.diff(rms)))
            features['energy_cv'] = features['energy_std'] / features['energy_mean'] if features['energy_mean'] > 0 else 0

        # Loudness (perceptual)
        S = np.abs(librosa.stft(y)) ** 2
        features['loudness'] = np.mean(librosa.perceptual_weighting(S, librosa.fft_frequencies(sr=self.sr)))

        return features

    def _extract_timing_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Timing/rhythm features.
        Cognitive effects of glucose affect speech rate and pauses.
        """
        features = {}

        # Voice Activity Detection based features
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        threshold = np.mean(rms) * 0.5

        voiced_frames = rms > threshold
        total_frames = len(voiced_frames)

        features['speech_ratio'] = np.sum(voiced_frames) / total_frames if total_frames > 0 else 0

        # Pause analysis
        pause_lengths = []
        current_pause = 0

        for is_voiced in voiced_frames:
            if not is_voiced:
                current_pause += 1
            else:
                if current_pause > 0:
                    pause_lengths.append(current_pause)
                current_pause = 0

        if pause_lengths:
            # Convert to seconds (512 hop / 16000 sr = 0.032 sec per frame)
            frame_duration = 512 / self.sr
            pause_durations = np.array(pause_lengths) * frame_duration

            features['pause_count'] = len(pause_lengths)
            features['pause_mean_duration'] = np.mean(pause_durations)
            features['pause_max_duration'] = np.max(pause_durations)
            features['pause_total_duration'] = np.sum(pause_durations)
        else:
            features['pause_count'] = 0
            features['pause_mean_duration'] = 0
            features['pause_max_duration'] = 0
            features['pause_total_duration'] = 0

        # Tempo/speaking rate
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=self.sr)
        features['speaking_rate'] = float(tempo[0]) if len(tempo) > 0 else 0

        # Duration
        features['total_duration'] = len(y) / self.sr

        return features

    def _extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Spectral features - frequency distribution of voice.
        Changes in muscle tension affect spectral characteristics.
        """
        features = {}

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        features['spectral_centroid_mean'] = np.mean(centroid)
        features['spectral_centroid_std'] = np.std(centroid)

        # Spectral bandwidth (spread)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)

        # Spectral rolloff (high frequency content)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0]
        features['spectral_rolloff_mean'] = np.mean(rolloff)

        # Spectral flatness (noisiness)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        for i in range(contrast.shape[0]):
            features[f'spectral_contrast_band{i}_mean'] = np.mean(contrast[i])

        # Spectral flux (rate of change)
        S = np.abs(librosa.stft(y))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        features['spectral_flux_mean'] = np.mean(flux)
        features['spectral_flux_std'] = np.std(flux)

        return features

    def _extract_formant_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Formant-related features (approximations).
        Formants reflect vocal tract configuration.
        """
        features = {}

        # Use LPC to estimate formants
        try:
            # Pre-emphasis
            y_preemph = librosa.effects.preemphasis(y)

            # LPC analysis
            order = 12  # Typical for formant analysis
            a = librosa.lpc(y_preemph, order=order)

            # Find formant frequencies from LPC roots
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]  # Keep positive frequencies

            # Convert to frequencies
            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = sorted(angles * (self.sr / (2 * np.pi)))
            freqs = [f for f in freqs if 90 < f < 5000]  # Valid formant range

            if len(freqs) >= 3:
                features['formant_f1'] = freqs[0]
                features['formant_f2'] = freqs[1]
                features['formant_f3'] = freqs[2]
                features['formant_f2_f1_ratio'] = freqs[1] / freqs[0] if freqs[0] > 0 else 0

        except Exception:
            pass

        return features

    def _extract_tremor_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Tremor features - low frequency modulation.
        Hypoglycemia causes tremor (4-12 Hz modulation).
        """
        features = {}

        # Get amplitude envelope
        analytic_signal = signal.hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)

        # Downsample envelope for tremor analysis
        envelope_ds = signal.decimate(amplitude_envelope, 160)  # ~100 Hz
        env_sr = self.sr / 160

        # Analyze tremor frequency band (4-12 Hz)
        if len(envelope_ds) > int(env_sr):
            freqs, psd = signal.welch(envelope_ds, fs=env_sr, nperseg=min(256, len(envelope_ds)))

            # Tremor band (4-12 Hz)
            tremor_mask = (freqs >= 4) & (freqs <= 12)
            if np.any(tremor_mask):
                tremor_power = np.sum(psd[tremor_mask])
                total_power = np.sum(psd)

                features['tremor_power'] = tremor_power
                features['tremor_ratio'] = tremor_power / total_power if total_power > 0 else 0

                # Peak tremor frequency
                tremor_psd = psd[tremor_mask]
                tremor_freqs = freqs[tremor_mask]
                if len(tremor_psd) > 0:
                    features['tremor_peak_freq'] = tremor_freqs[np.argmax(tremor_psd)]

        # Pitch tremor (modulation of F0)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=self.sr)
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 10:
            # F0 modulation rate
            f0_diff = np.diff(f0_clean)
            features['pitch_tremor_intensity'] = np.std(f0_diff)

        return features


class GlucoseSpecificFeatures:
    """
    Features specifically designed for glucose-level classification/regression.
    Based on literature review of voice-glucose correlations.
    """

    @staticmethod
    def compute_hypoglycemia_indicators(features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute composite indicators that may signal hypoglycemia.
        """
        indicators = {}

        # Voice instability index (jitter + shimmer + tremor)
        jitter_like = features.get('f0_ppq', 0)
        shimmer_like = features.get('shimmer_approx', 0)
        tremor = features.get('tremor_ratio', 0)

        indicators['voice_instability_index'] = (
            0.4 * jitter_like +
            0.4 * shimmer_like +
            0.2 * tremor
        )

        # Cognitive load indicator (pause patterns, speaking rate)
        pause_ratio = features.get('pause_total_duration', 0) / features.get('total_duration', 1)
        speaking_rate = features.get('speaking_rate', 100)

        indicators['cognitive_load_index'] = (
            0.5 * pause_ratio +
            0.5 * (1 - min(speaking_rate / 150, 1))  # Normalized inverse rate
        )

        # Fatigue indicator (energy, pitch variability)
        energy = features.get('energy_mean', 0)
        energy_range = features.get('energy_dynamic_range', 0)
        pitch_var = features.get('f0_cv', 0)

        indicators['fatigue_index'] = (
            0.4 * (1 - min(energy / 0.1, 1)) +  # Low energy
            0.3 * (1 - min(energy_range / 30, 1)) +  # Low dynamic range
            0.3 * (1 - min(pitch_var / 0.3, 1))  # Low pitch variation
        )

        return indicators


def extract_glucose_relevant_features(audio_path: str) -> Dict[str, float]:
    """
    Main function to extract all glucose-relevant features.
    """
    extractor = AdvancedVoiceFeatureExtractor()
    features = extractor.extract_all(audio_path)

    if features:
        # Add composite indicators
        indicators = GlucoseSpecificFeatures.compute_hypoglycemia_indicators(features)
        features.update(indicators)

    return features


# Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        features = extract_glucose_relevant_features(audio_file)

        print(f"\nExtracted {len(features)} features from {audio_file}")
        print("\nKey features:")
        for key in ['f0_mean', 'f0_cv', 'energy_mean', 'tremor_ratio',
                    'voice_instability_index', 'cognitive_load_index', 'fatigue_index']:
            if key in features:
                print(f"  {key}: {features[key]:.4f}")
    else:
        print("Usage: python advanced_voice_features.py <audio_file.wav>")
