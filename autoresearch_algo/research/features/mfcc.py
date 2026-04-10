"""
MFCC Feature Extractor — Wrapper for autoresearch hyperparameter sweep.
========================================================================

Wraps either the ONVOX CanonicalFeatureExtractor (if available) or a
standalone librosa-based extraction. Provides the interface expected by
hyperparameter_sweep.py's extract_features_config().
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MFCCExtractor:
    """MFCC + optional spectral/pitch feature extractor.

    Matches the autoresearch hyperparameter_sweep.py interface:
        - extract_from_array(y) -> np.ndarray or None
        - feature_names property for dimensionality

    Args:
        sr: Sample rate (default 16000)
        n_mfcc: Number of MFCC coefficients
        fmin: Minimum frequency for mel filterbank
        fmax: Maximum frequency for mel filterbank
        include_spectral: Include spectral features (centroid, bandwidth, rolloff, flatness, contrast)
        include_pitch: Include pitch (F0) features
        include_mel: Include raw mel-spectrogram stats (usually False)
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 20,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        include_spectral: bool = True,
        include_pitch: bool = True,
        include_mel: bool = False,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax
        self.include_spectral = include_spectral
        self.include_pitch = include_pitch
        self.include_mel = include_mel
        self._n_fft = 2048
        self._hop_length = 512
        self._n_mels = 64
        self._feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        names = []
        # MFCCs: mean + std
        for i in range(self.n_mfcc):
            names.append(f"mfcc_{i}_mean")
            names.append(f"mfcc_{i}_std")
        # Delta MFCCs: mean + std
        for i in range(self.n_mfcc):
            names.append(f"delta_mfcc_{i}_mean")
            names.append(f"delta_mfcc_{i}_std")
        # Delta-delta MFCCs: mean + std
        for i in range(self.n_mfcc):
            names.append(f"delta2_mfcc_{i}_mean")
            names.append(f"delta2_mfcc_{i}_std")

        if self.include_spectral:
            names.extend([
                "spectral_centroid_mean", "spectral_centroid_std",
                "spectral_bandwidth_mean", "spectral_bandwidth_std",
                "spectral_rolloff_mean", "spectral_rolloff_std",
                "spectral_flatness_mean", "spectral_flatness_std",
                "spectral_contrast_mean", "spectral_contrast_std",
            ])

        if self.include_pitch:
            names.extend(["f0_mean", "f0_std", "f0_median"])

        return names

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    def extract_from_array(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from a raw audio waveform array.

        Args:
            y: Audio waveform as numpy array (mono, at self.sr sample rate)

        Returns:
            1D numpy array of features, or None on failure.
        """
        import librosa

        if len(y) < self.sr * 0.5:
            return None

        try:
            features = []

            # MFCCs
            mfccs = librosa.feature.mfcc(
                y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                n_fft=self._n_fft, hop_length=self._hop_length, n_mels=self._n_mels,
                fmin=self.fmin, fmax=self.fmax,
            )
            for i in range(self.n_mfcc):
                features.append(float(np.mean(mfccs[i])))
                features.append(float(np.std(mfccs[i])))

            # Delta MFCCs
            delta_mfccs = librosa.feature.delta(mfccs)
            for i in range(self.n_mfcc):
                features.append(float(np.mean(delta_mfccs[i])))
                features.append(float(np.std(delta_mfccs[i])))

            # Delta-delta MFCCs
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            for i in range(self.n_mfcc):
                features.append(float(np.mean(delta2_mfccs[i])))
                features.append(float(np.std(delta2_mfccs[i])))

            # Spectral features
            if self.include_spectral:
                spec_cent = librosa.feature.spectral_centroid(
                    y=y, sr=self.sr, n_fft=self._n_fft, hop_length=self._hop_length
                )[0]
                features.extend([float(np.mean(spec_cent)), float(np.std(spec_cent))])

                spec_bw = librosa.feature.spectral_bandwidth(
                    y=y, sr=self.sr, n_fft=self._n_fft, hop_length=self._hop_length
                )[0]
                features.extend([float(np.mean(spec_bw)), float(np.std(spec_bw))])

                spec_roll = librosa.feature.spectral_rolloff(
                    y=y, sr=self.sr, n_fft=self._n_fft, hop_length=self._hop_length
                )[0]
                features.extend([float(np.mean(spec_roll)), float(np.std(spec_roll))])

                spec_flat = librosa.feature.spectral_flatness(
                    y=y, n_fft=self._n_fft, hop_length=self._hop_length
                )[0]
                features.extend([float(np.mean(spec_flat)), float(np.std(spec_flat))])

                spec_contrast = librosa.feature.spectral_contrast(
                    y=y, sr=self.sr, n_fft=self._n_fft, hop_length=self._hop_length
                )
                features.extend([float(np.mean(spec_contrast)), float(np.std(spec_contrast))])

            # Pitch features
            if self.include_pitch:
                try:
                    f0, voiced_flag, voiced_probs = librosa.pyin(
                        y, fmin=self.fmin, fmax=min(self.fmax, 500.0),
                        sr=self.sr, frame_length=self._n_fft,
                    )
                    f0_voiced = f0[~np.isnan(f0)]
                    if len(f0_voiced) > 0:
                        features.extend([
                            float(np.mean(f0_voiced)),
                            float(np.std(f0_voiced)),
                            float(np.median(f0_voiced)),
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0])
                except Exception:
                    features.extend([0.0, 0.0, 0.0])

            return np.array(features, dtype=np.float32)

        except Exception as e:
            logger.warning("Feature extraction failed: %s", e)
            return None
