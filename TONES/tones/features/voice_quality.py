"""
Voice Quality Feature Extraction
===================================
Physiologically-motivated features that directly measure the neuromuscular
and autonomic effects of glucose on voice production.

Key physiological pathways:
  - Autonomic tremor (4-12 Hz) → vocal tremor, jitter, shimmer
  - Neuromuscular control → fine motor instability in articulators
  - Dehydration → vocal fold stiffness → formant shifts, reduced HNR
  - Cognitive load → pause patterns, speech rate changes

These features are NOT in the main MFCC pipeline but are likely the most
glucose-sensitive because they target specific physiological mechanisms.

Ported from archive/advanced_voice_features.py and extended with:
  - More robust jitter/shimmer computation
  - Tremor power spectral analysis
  - Cepstral Peak Prominence (CPP)
  - Voice instability composite index
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import librosa
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)


class VoiceQualityExtractor:
    """
    Extract voice quality features specifically relevant to glucose estimation.

    These features measure the micro-instabilities and spectral quality
    changes that glucose-induced physiological changes produce.

    Parameters
    ----------
    sr : int
        Target sample rate (default 16000).
    min_duration_sec : float
        Minimum audio duration in seconds.
    fmin : float
        Minimum F0 for pitch tracking.
    fmax : float
        Maximum F0 for pitch tracking.
    """

    def __init__(
        self,
        sr: int = 16000,
        min_duration_sec: float = 0.5,
        fmin: float = 50,
        fmax: float = 500,
    ):
        self.sr = sr
        self.min_duration_sec = min_duration_sec
        self.fmin = fmin
        self.fmax = fmax

    def extract_from_file(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract voice quality features from an audio file."""
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        except Exception as e:
            logger.warning("Failed to load %s: %s", audio_path, e)
            return None

        if len(y) < self.sr * self.min_duration_sec:
            return None

        return self.extract_from_array(y)

    def extract_from_array(self, y: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract voice quality features from a waveform.

        Returns a fixed-size feature vector containing:
          - Jitter features (3): local jitter, RAP, PPQ5
          - Shimmer features (3): local shimmer, APQ3, APQ5
          - Tremor features (4): power, ratio, peak freq, pitch tremor
          - Formant features (4): F1, F2, F3, F2/F1 ratio
          - HNR (1): harmonic-to-noise ratio
          - CPP (1): cepstral peak prominence
          - Pitch quality (5): F0 CV, skewness, kurtosis, voiced ratio, PPQ
          - Composites (3): instability index, energy CV, spectral flux mean
          Total: 24 features
        """
        if len(y) < self.sr * self.min_duration_sec:
            return None

        try:
            features = []

            # --- Pitch tracking (needed for jitter, tremor) ---
            # Limit to 5s for speed (pYIN is O(n^2) on long audio)
            max_samples = self.sr * 5
            y_pitch = y[:max_samples] if len(y) > max_samples else y
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y_pitch, fmin=self.fmin, fmax=self.fmax, sr=self.sr
            )
            f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])

            # --- Jitter features (pitch perturbation) ---
            features.extend(self._compute_jitter(f0_voiced))

            # --- Shimmer features (amplitude perturbation) ---
            features.extend(self._compute_shimmer(y))

            # --- Tremor features ---
            features.extend(self._compute_tremor(y, f0_voiced))

            # --- Formant features ---
            features.extend(self._compute_formants(y))

            # --- HNR ---
            features.append(self._compute_hnr(y))

            # --- CPP ---
            features.append(self._compute_cpp(y))

            # --- Pitch quality features ---
            features.extend(self._compute_pitch_quality(f0_voiced, f0))

            # --- Composite indicators ---
            features.extend(self._compute_composites(features, y))

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning("Voice quality extraction failed: %s", e)
            return None

    def _compute_jitter(self, f0_voiced: np.ndarray) -> List[float]:
        """
        Compute jitter (pitch perturbation) measures.

        Returns [local_jitter, rap, ppq5]
        """
        if len(f0_voiced) < 5:
            return [0.0, 0.0, 0.0]

        periods = 1.0 / f0_voiced  # Convert to period

        # Local jitter: mean absolute difference of consecutive periods
        diffs = np.abs(np.diff(periods))
        local_jitter = np.mean(diffs) / np.mean(periods) if np.mean(periods) > 0 else 0.0

        # RAP (Relative Average Perturbation): 3-point running average
        if len(periods) >= 3:
            smoothed = np.convolve(periods, np.ones(3) / 3, mode="valid")
            rap = np.mean(np.abs(periods[1:-1] - smoothed)) / np.mean(periods)
        else:
            rap = 0.0

        # PPQ5 (5-point Period Perturbation Quotient)
        if len(periods) >= 5:
            smoothed5 = np.convolve(periods, np.ones(5) / 5, mode="valid")
            ppq5 = np.mean(np.abs(periods[2:-2] - smoothed5)) / np.mean(periods)
        else:
            ppq5 = 0.0

        return [float(local_jitter), float(rap), float(ppq5)]

    def _compute_shimmer(self, y: np.ndarray) -> List[float]:
        """
        Compute shimmer (amplitude perturbation) measures.

        Returns [local_shimmer, apq3, apq5]
        """
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.010 * self.sr)     # 10ms hop

        # Frame-level peak amplitudes
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        peak_amplitudes = np.max(np.abs(frames), axis=0)

        if len(peak_amplitudes) < 5:
            return [0.0, 0.0, 0.0]

        # Filter out silence frames
        threshold = np.mean(peak_amplitudes) * 0.1
        voiced_amps = peak_amplitudes[peak_amplitudes > threshold]

        if len(voiced_amps) < 5:
            return [0.0, 0.0, 0.0]

        # Local shimmer
        diffs = np.abs(np.diff(voiced_amps))
        local_shimmer = np.mean(diffs) / np.mean(voiced_amps) if np.mean(voiced_amps) > 0 else 0.0

        # APQ3
        if len(voiced_amps) >= 3:
            smoothed = np.convolve(voiced_amps, np.ones(3) / 3, mode="valid")
            apq3 = np.mean(np.abs(voiced_amps[1:-1] - smoothed)) / np.mean(voiced_amps)
        else:
            apq3 = 0.0

        # APQ5
        if len(voiced_amps) >= 5:
            smoothed5 = np.convolve(voiced_amps, np.ones(5) / 5, mode="valid")
            apq5 = np.mean(np.abs(voiced_amps[2:-2] - smoothed5)) / np.mean(voiced_amps)
        else:
            apq5 = 0.0

        return [float(local_shimmer), float(apq3), float(apq5)]

    def _compute_tremor(self, y: np.ndarray, f0_voiced: np.ndarray) -> List[float]:
        """
        Compute tremor features (4-12 Hz modulation analysis).

        Returns [tremor_power, tremor_ratio, peak_tremor_freq, pitch_tremor_intensity]
        """
        tremor_power = 0.0
        tremor_ratio = 0.0
        peak_freq = 0.0
        pitch_tremor = 0.0

        try:
            # Amplitude envelope tremor
            analytic = sp_signal.hilbert(y)
            envelope = np.abs(analytic)

            # Downsample to ~100 Hz for tremor analysis
            decimation_factor = max(1, self.sr // 100)
            if len(envelope) > decimation_factor * 10:
                envelope_ds = sp_signal.decimate(envelope, decimation_factor)
                env_sr = self.sr / decimation_factor

                if len(envelope_ds) > int(env_sr):
                    nperseg = min(256, len(envelope_ds))
                    freqs, psd = sp_signal.welch(envelope_ds, fs=env_sr, nperseg=nperseg)

                    tremor_mask = (freqs >= 4) & (freqs <= 12)
                    if np.any(tremor_mask):
                        tremor_power = float(np.sum(psd[tremor_mask]))
                        total_power = float(np.sum(psd))
                        tremor_ratio = tremor_power / total_power if total_power > 0 else 0.0

                        tremor_psd = psd[tremor_mask]
                        tremor_freqs = freqs[tremor_mask]
                        peak_freq = float(tremor_freqs[np.argmax(tremor_psd)])
        except Exception:
            pass

        # Pitch tremor (F0 instability)
        if len(f0_voiced) > 10:
            f0_diff = np.diff(f0_voiced)
            pitch_tremor = float(np.std(f0_diff))

        return [tremor_power, tremor_ratio, peak_freq, pitch_tremor]

    def _compute_formants(self, y: np.ndarray) -> List[float]:
        """
        Estimate formant frequencies via LPC analysis.

        Returns [F1, F2, F3, F2/F1_ratio]
        """
        try:
            y_preemph = librosa.effects.preemphasis(y)
            order = min(12, len(y_preemph) // 100)
            if order < 4:
                return [0.0, 0.0, 0.0, 0.0]

            a = librosa.lpc(y_preemph, order=order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]

            angles = np.arctan2(np.imag(roots), np.real(roots))
            freqs = sorted(angles * (self.sr / (2 * np.pi)))
            freqs = [f for f in freqs if 90 < f < 5000]

            if len(freqs) >= 3:
                f1, f2, f3 = freqs[0], freqs[1], freqs[2]
                ratio = f2 / f1 if f1 > 0 else 0.0
                return [float(f1), float(f2), float(f3), float(ratio)]

        except Exception:
            pass

        return [0.0, 0.0, 0.0, 0.0]

    def _compute_hnr(self, y: np.ndarray) -> float:
        """
        Approximate Harmonic-to-Noise Ratio using HPSS decomposition.

        Returns HNR in dB.
        """
        try:
            S = np.abs(librosa.stft(y))
            harmonic, percussive = librosa.decompose.hpss(S)

            h_energy = np.sum(harmonic ** 2)
            n_energy = np.sum(percussive ** 2)

            if n_energy > 0:
                return float(10 * np.log10(h_energy / n_energy))
            return 30.0
        except Exception:
            return 0.0

    def _compute_cpp(self, y: np.ndarray) -> float:
        """
        Approximate Cepstral Peak Prominence.

        CPP is a measure of voice quality: higher = clearer voice.
        """
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
            return float(np.max(np.mean(mfcc, axis=1)))
        except Exception:
            return 0.0

    def _compute_pitch_quality(
        self, f0_voiced: np.ndarray, f0_full: np.ndarray
    ) -> List[float]:
        """
        Pitch quality features measuring voice stability.

        Returns [f0_cv, f0_skewness, f0_kurtosis, voiced_ratio, f0_ppq]
        """
        if len(f0_voiced) < 3:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        f0_mean = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)
        f0_cv = f0_std / f0_mean if f0_mean > 0 else 0.0

        f0_sk = float(skew(f0_voiced)) if len(f0_voiced) > 3 else 0.0
        f0_kurt = float(kurtosis(f0_voiced)) if len(f0_voiced) > 3 else 0.0

        voiced_ratio = float(np.sum(~np.isnan(f0_full)) / len(f0_full)) if len(f0_full) > 0 else 0.0

        # Pitch perturbation quotient
        if len(f0_voiced) > 1 and f0_mean > 0:
            f0_ppq = float(np.mean(np.abs(np.diff(f0_voiced))) / f0_mean)
        else:
            f0_ppq = 0.0

        return [float(f0_cv), f0_sk, f0_kurt, voiced_ratio, f0_ppq]

    def _compute_composites(self, features_so_far: List[float], y: np.ndarray) -> List[float]:
        """
        Composite indicators combining multiple voice quality dimensions.

        Returns [voice_instability_index, energy_cv, spectral_flux_mean]
        """
        # Voice instability = f(jitter, shimmer, tremor)
        jitter = features_so_far[0] if len(features_so_far) > 0 else 0.0
        shimmer = features_so_far[3] if len(features_so_far) > 3 else 0.0
        tremor_ratio = features_so_far[7] if len(features_so_far) > 7 else 0.0
        instability = 0.4 * jitter + 0.4 * shimmer + 0.2 * tremor_ratio

        # Energy coefficient of variation
        rms = librosa.feature.rms(y=y)[0]
        energy_cv = float(np.std(rms) / (np.mean(rms) + 1e-10))

        # Spectral flux (rate of spectral change)
        try:
            S = np.abs(librosa.stft(y))
            flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
            spectral_flux = float(np.mean(flux))
        except Exception:
            spectral_flux = 0.0

        return [float(instability), energy_cv, spectral_flux]

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names for interpretability."""
        return [
            # Jitter (3)
            "jitter_local", "jitter_rap", "jitter_ppq5",
            # Shimmer (3)
            "shimmer_local", "shimmer_apq3", "shimmer_apq5",
            # Tremor (4)
            "tremor_power", "tremor_ratio", "tremor_peak_freq", "pitch_tremor_intensity",
            # Formants (4)
            "formant_f1", "formant_f2", "formant_f3", "formant_f2_f1_ratio",
            # HNR (1)
            "hnr",
            # CPP (1)
            "cpp",
            # Pitch quality (5)
            "f0_cv", "f0_skewness", "f0_kurtosis", "voiced_ratio", "f0_ppq",
            # Composites (3)
            "voice_instability_index", "energy_cv", "spectral_flux_mean",
        ]

    @property
    def n_features(self) -> int:
        return 24


def create_voice_quality_extractor(cfg: Dict = None) -> VoiceQualityExtractor:
    """Create a VoiceQualityExtractor from config."""
    if cfg is None:
        cfg = {}
    feat_cfg = cfg.get("features", {})
    return VoiceQualityExtractor(
        sr=feat_cfg.get("sample_rate", 16000),
    )
