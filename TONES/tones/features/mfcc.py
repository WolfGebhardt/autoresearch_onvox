"""
MFCC Feature Extraction
========================
Canonical MFCC + delta + spectral feature extraction for the TONES pipeline.

Based on hyperparameter optimization: n_mfcc=20 with deltas yields 124 features
and achieves the best MAE on personalized models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import librosa

logger = logging.getLogger(__name__)


class MFCCExtractor:
    """
    Extract MFCC and spectral features from audio files.

    Combines:
    - MFCC coefficients (mean + std)
    - Delta and delta-delta MFCCs (mean + std)
    - RMS energy and zero-crossing rate
    - Optional: spectral centroid, bandwidth, rolloff, contrast
    - Optional: pitch (F0) via pYIN
    - Optional: harmonic-to-noise ratio proxy

    Parameters
    ----------
    sr : int
        Target sample rate (default 16000).
    n_mfcc : int
        Number of MFCC coefficients (default 20).
    n_mels : int
        Number of mel bands for mel-spectrogram features (default 64).
    fmin : float
        Minimum frequency for MFCC/mel (default 50 Hz).
    fmax : float
        Maximum frequency for MFCC/mel (default 8000 Hz).
    include_spectral : bool
        Include spectral centroid, bandwidth, rolloff, contrast, flatness.
    include_pitch : bool
        Include F0 estimation via pYIN.
    include_mel : bool
        Include mel-spectrogram statistics.
    min_duration_sec : float
        Skip audio shorter than this.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 20,
        n_mels: int = 64,
        fmin: float = 50,
        fmax: float = 8000,
        include_spectral: bool = True,
        include_pitch: bool = True,
        include_mel: bool = False,
        min_duration_sec: float = 0.5,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.include_spectral = include_spectral
        self.include_pitch = include_pitch
        self.include_mel = include_mel
        self.min_duration_sec = min_duration_sec

    def extract_from_file(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract features from an audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to the audio file.

        Returns
        -------
        np.ndarray or None
            Feature vector, or None if extraction failed.
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        except Exception as e:
            logger.warning("Failed to load %s: %s", audio_path, e)
            return None

        if len(y) < sr * self.min_duration_sec:
            logger.debug("Audio too short (%d samples): %s", len(y), audio_path)
            return None

        return self.extract_from_array(y)

    def extract_from_array(self, y: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features from a waveform array.

        Parameters
        ----------
        y : np.ndarray
            Audio waveform (mono, at self.sr sample rate).

        Returns
        -------
        np.ndarray or None
            Feature vector.
        """
        if len(y) < self.sr * self.min_duration_sec:
            return None

        try:
            features = []

            # --- Core: MFCCs ---
            mfccs = librosa.feature.mfcc(
                y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                fmin=self.fmin, fmax=self.fmax,
                htk=False,
            )
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))

            # --- Delta MFCCs ---
            delta = librosa.feature.delta(mfccs)
            features.extend(np.mean(delta, axis=1))
            features.extend(np.std(delta, axis=1))

            # --- Delta-delta MFCCs ---
            delta2 = librosa.feature.delta(mfccs, order=2)
            features.extend(np.mean(delta2, axis=1))
            features.extend(np.std(delta2, axis=1))

            # --- Energy ---
            rms = librosa.feature.rms(y=y)
            features.extend([np.mean(rms), np.std(rms)])

            # --- Zero-crossing rate ---
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([np.mean(zcr), np.std(zcr)])

            # --- Optional: Spectral features ---
            if self.include_spectral:
                features.extend(self._extract_spectral(y))

            # --- Optional: Pitch (F0) ---
            if self.include_pitch:
                features.extend(self._extract_pitch(y))

            # --- Optional: Mel-spectrogram ---
            if self.include_mel:
                features.extend(self._extract_mel(y))

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning("Feature extraction failed: %s", e)
            return None

    def _extract_spectral(self, y: np.ndarray) -> list:
        """Extract spectral features."""
        feats = []

        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        feats.extend([np.mean(centroid), np.std(centroid)])

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)
        feats.extend([np.mean(bandwidth), np.std(bandwidth)])

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        feats.extend([np.mean(rolloff), np.std(rolloff)])

        flatness = librosa.feature.spectral_flatness(y=y)
        feats.extend([np.mean(flatness), np.std(flatness)])

        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
        feats.extend(np.mean(contrast, axis=1))
        feats.extend(np.std(contrast, axis=1))

        return feats

    def _extract_pitch(self, y: np.ndarray) -> list:
        """Extract pitch (F0) features using pYIN. Uses max 5s of audio for speed."""
        try:
            # Limit to 5 seconds to avoid pYIN being extremely slow on long recordings
            max_samples = self.sr * 5
            y_clip = y[:max_samples] if len(y) > max_samples else y
            f0, voiced_flag, _ = librosa.pyin(y_clip, fmin=50, fmax=400, sr=self.sr)
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                return [np.mean(f0_valid), np.std(f0_valid), np.median(f0_valid)]
        except Exception:
            pass
        return [0.0, 0.0, 0.0]

    def _extract_mel(self, y: np.ndarray) -> list:
        """Extract mel-spectrogram statistics."""
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
            htk=False,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feats = []
        feats.extend(np.mean(mel_db, axis=1))
        feats.extend(np.std(mel_db, axis=1))
        return feats

    @property
    def feature_names(self) -> list:
        """Return a list of feature names (for interpretability)."""
        names = []
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"mfcc_{i}_{stat}")
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"delta_mfcc_{i}_{stat}")
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"delta2_mfcc_{i}_{stat}")
        names.extend(["rms_mean", "rms_std", "zcr_mean", "zcr_std"])

        if self.include_spectral:
            names.extend([
                "spectral_centroid_mean", "spectral_centroid_std",
                "spectral_bandwidth_mean", "spectral_bandwidth_std",
                "spectral_rolloff_mean", "spectral_rolloff_std",
                "spectral_flatness_mean", "spectral_flatness_std",
            ])
            for stat in ["mean", "std"]:
                for i in range(7):
                    names.append(f"spectral_contrast_{i}_{stat}")

        if self.include_pitch:
            names.extend(["f0_mean", "f0_std", "f0_median"])

        if self.include_mel:
            for stat in ["mean", "std"]:
                for i in range(self.n_mels):
                    names.append(f"mel_{i}_{stat}")

        return names


def create_extractor_from_config(cfg: Dict) -> MFCCExtractor:
    """Create an MFCCExtractor from the features section of config.yaml."""
    feat_cfg = cfg.get("features", {})
    return MFCCExtractor(
        sr=feat_cfg.get("sample_rate", 16000),
        n_mfcc=feat_cfg.get("n_mfcc", 20),
        n_mels=feat_cfg.get("n_mels", 64),
        fmin=feat_cfg.get("fmin", 50),
        fmax=feat_cfg.get("fmax", 8000),
        include_spectral=True,
        include_pitch=True,
        include_mel=False,
    )
