"""
Voice Quality Feature Extractor — Jitter, Shimmer, HNR, Voiced Ratio, F0 CV.
==============================================================================

Extracts voice quality features for the autoresearch pipeline.
Reuses computation patterns from CanonicalFeatureExtractor's extended features.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VoiceQualityExtractor:
    """Extract voice quality features from audio waveform.

    Features (in order):
        - jitter_local: cycle-to-cycle F0 perturbation
        - jitter_rap: relative average perturbation
        - jitter_ppq5: five-point period perturbation quotient
        - shimmer_local: cycle-to-cycle amplitude perturbation
        - shimmer_apq3: three-point amplitude perturbation quotient
        - shimmer_apq5: five-point amplitude perturbation quotient
        - tremor_power: 4-12 Hz power in amplitude contour
        - formant_f1: first formant frequency
        - formant_f2: second formant frequency
        - formant_f3: third formant frequency
        - hnr: harmonics-to-noise ratio
        - cpp: cepstral peak prominence
        - f0_cv: F0 coefficient of variation
        - f0_skew: F0 distribution skewness
        - f0_kurtosis: F0 distribution kurtosis
        - voiced_ratio: fraction of voiced frames
    """

    FEATURE_NAMES = [
        "jitter_local", "jitter_rap", "jitter_ppq5",
        "shimmer_local", "shimmer_apq3", "shimmer_apq5",
        "tremor_power",
        "formant_f1", "formant_f2", "formant_f3",
        "hnr", "cpp",
        "f0_cv", "f0_skew", "f0_kurtosis",
        "voiced_ratio",
    ]

    def __init__(self, sr: int = 16000, fmin: float = 65.0, fmax: float = 500.0):
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.n_features = len(self.FEATURE_NAMES)

    def extract_from_array(self, y: np.ndarray) -> Optional[np.ndarray]:
        """Extract voice quality features from audio array.

        Returns 1D numpy array of features, or None on failure.
        Falls back to zero-vector if pYIN fails (matching ONVOX convention).
        """
        import librosa
        from scipy import signal as scipy_signal

        if len(y) < self.sr * 0.5:
            return None

        features = np.zeros(self.n_features, dtype=np.float32)

        try:
            # F0 extraction via pYIN
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=self.fmin, fmax=self.fmax,
                sr=self.sr, frame_length=2048,
            )
            voiced_mask = ~np.isnan(f0)
            f0_voiced = f0[voiced_mask]
            n_frames = len(f0)
            n_voiced = len(f0_voiced)

            # Voiced ratio
            voiced_ratio = n_voiced / max(n_frames, 1)
            features[self.FEATURE_NAMES.index("voiced_ratio")] = voiced_ratio

            if n_voiced < 3:
                # Not enough voiced frames for quality metrics
                return features

            # F0 statistics
            f0_mean = np.mean(f0_voiced)
            f0_std = np.std(f0_voiced)
            f0_cv = f0_std / max(f0_mean, 1e-8)
            features[self.FEATURE_NAMES.index("f0_cv")] = f0_cv

            from scipy.stats import skew, kurtosis
            features[self.FEATURE_NAMES.index("f0_skew")] = float(skew(f0_voiced))
            features[self.FEATURE_NAMES.index("f0_kurtosis")] = float(kurtosis(f0_voiced))

            # Jitter (cycle-to-cycle F0 perturbation)
            periods = 1.0 / np.clip(f0_voiced, 50, 500)
            if len(periods) > 1:
                period_diffs = np.abs(np.diff(periods))
                jitter_local = np.mean(period_diffs) / max(np.mean(periods), 1e-8)
                features[self.FEATURE_NAMES.index("jitter_local")] = jitter_local

                # RAP: relative average perturbation (3-point)
                if len(periods) >= 3:
                    rap_diffs = []
                    for i in range(1, len(periods) - 1):
                        avg3 = (periods[i-1] + periods[i] + periods[i+1]) / 3
                        rap_diffs.append(abs(periods[i] - avg3))
                    features[self.FEATURE_NAMES.index("jitter_rap")] = np.mean(rap_diffs) / max(np.mean(periods), 1e-8)

                # PPQ5: five-point period perturbation quotient
                if len(periods) >= 5:
                    ppq5_diffs = []
                    for i in range(2, len(periods) - 2):
                        avg5 = np.mean(periods[i-2:i+3])
                        ppq5_diffs.append(abs(periods[i] - avg5))
                    features[self.FEATURE_NAMES.index("jitter_ppq5")] = np.mean(ppq5_diffs) / max(np.mean(periods), 1e-8)

            # Shimmer (amplitude perturbation)
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
            rms_voiced = rms[voiced_mask[:len(rms)]] if len(rms) >= len(voiced_mask) else rms

            if len(rms_voiced) > 1:
                amp_diffs = np.abs(np.diff(rms_voiced))
                shimmer_local = np.mean(amp_diffs) / max(np.mean(rms_voiced), 1e-8)
                features[self.FEATURE_NAMES.index("shimmer_local")] = shimmer_local

                if len(rms_voiced) >= 3:
                    apq3_diffs = []
                    for i in range(1, len(rms_voiced) - 1):
                        avg3 = (rms_voiced[i-1] + rms_voiced[i] + rms_voiced[i+1]) / 3
                        apq3_diffs.append(abs(rms_voiced[i] - avg3))
                    features[self.FEATURE_NAMES.index("shimmer_apq3")] = np.mean(apq3_diffs) / max(np.mean(rms_voiced), 1e-8)

                if len(rms_voiced) >= 5:
                    apq5_diffs = []
                    for i in range(2, len(rms_voiced) - 2):
                        avg5 = np.mean(rms_voiced[i-2:i+3])
                        apq5_diffs.append(abs(rms_voiced[i] - avg5))
                    features[self.FEATURE_NAMES.index("shimmer_apq5")] = np.mean(apq5_diffs) / max(np.mean(rms_voiced), 1e-8)

            # Tremor power (4-12 Hz in amplitude contour)
            if len(rms) > 16:
                frame_rate = self.sr / hop_length
                freqs, psd = scipy_signal.welch(rms, fs=frame_rate, nperseg=min(len(rms), 64))
                tremor_mask = (freqs >= 4) & (freqs <= 12)
                if tremor_mask.any():
                    features[self.FEATURE_NAMES.index("tremor_power")] = float(np.sum(psd[tremor_mask]))

            # HNR (harmonics-to-noise ratio)
            try:
                autocorr = np.correlate(y[:self.sr], y[:self.sr], mode="full")
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / max(autocorr[0], 1e-8)
                # Find first peak after minimum period
                min_lag = int(self.sr / self.fmax)
                max_lag = int(self.sr / self.fmin)
                search = autocorr[min_lag:max_lag]
                if len(search) > 0:
                    peak_idx = np.argmax(search) + min_lag
                    r_peak = autocorr[peak_idx]
                    if 0 < r_peak < 1:
                        hnr = 10 * np.log10(r_peak / max(1 - r_peak, 1e-8))
                        features[self.FEATURE_NAMES.index("hnr")] = float(np.clip(hnr, -20, 40))
            except Exception:
                pass

            # CPP (Cepstral Peak Prominence)
            try:
                ceps = np.abs(np.fft.rfft(np.log(np.abs(np.fft.rfft(y[:self.sr])) + 1e-8)))
                min_q = int(self.sr / self.fmax)
                max_q = int(self.sr / self.fmin)
                search_ceps = ceps[min_q:max_q]
                if len(search_ceps) > 0:
                    peak_val = np.max(search_ceps)
                    baseline = np.mean(search_ceps)
                    cpp = 20 * np.log10(max(peak_val, 1e-8) / max(baseline, 1e-8))
                    features[self.FEATURE_NAMES.index("cpp")] = float(np.clip(cpp, -10, 30))
            except Exception:
                pass

            # Formants (LPC-based estimation)
            try:
                from scipy.signal import lfilter
                # LPC via autocorrelation method
                order = 12
                windowed = y[:min(len(y), self.sr)] * np.hamming(min(len(y), self.sr))
                r = np.correlate(windowed, windowed, mode="full")
                r = r[len(r)//2:len(r)//2 + order + 1]

                # Levinson-Durbin recursion
                a = np.zeros(order + 1)
                a[0] = 1.0
                e = r[0]
                for i in range(1, order + 1):
                    lam = -np.sum(a[:i] * r[i:0:-1]) / max(e, 1e-8)
                    a_new = a.copy()
                    for j in range(1, i + 1):
                        a_new[j] = a[j] + lam * a[i - j]
                    a = a_new
                    e = e * (1 - lam * lam)

                # Find formants from LPC roots
                roots = np.roots(a)
                roots = roots[np.imag(roots) > 0]
                angles = np.arctan2(np.imag(roots), np.real(roots))
                formant_freqs = sorted(angles * self.sr / (2 * np.pi))
                formant_freqs = [f for f in formant_freqs if 90 < f < 5000]

                if len(formant_freqs) >= 1:
                    features[self.FEATURE_NAMES.index("formant_f1")] = formant_freqs[0]
                if len(formant_freqs) >= 2:
                    features[self.FEATURE_NAMES.index("formant_f2")] = formant_freqs[1]
                if len(formant_freqs) >= 3:
                    features[self.FEATURE_NAMES.index("formant_f3")] = formant_freqs[2]
            except Exception:
                pass

        except Exception as e:
            logger.warning("Voice quality extraction failed: %s", e)
            # Return zero-vector fallback (ONVOX convention)

        return features
