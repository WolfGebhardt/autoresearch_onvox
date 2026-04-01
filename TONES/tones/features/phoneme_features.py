"""
Phoneme-Level Physiological Residual Features
================================================
The core innovation: instead of computing features over entire recordings
(where phonemic content dominates ~60-70% of acoustic variance), we:

1. Extract acoustic features per phoneme segment (vowel clusters, consonant clusters)
2. Build a personal baseline per phoneme category for each speaker
3. Compute the RESIDUAL: how each instance deviates from the speaker's baseline
4. Aggregate residuals per recording → features that capture physiological state

This removes the dominant content confound and amplifies the glucose signal
from ~2% of total variance to ~10-30% of residual variance.

Key insight: within a specific vowel category (e.g., all /a/-like segments),
acoustic variation is much smaller, and glucose effects on vocal fold tension,
mucosal viscosity, and neural motor control become a much larger fraction
of the remaining variance.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa

from tones.features.phoneme_align import AlignmentResult, PhonemeSegment

logger = logging.getLogger(__name__)


# ─── Feature names ──────────────────────────────────────────────────────────────

# Per-segment acoustic features (13 dimensions)
SEGMENT_FEATURE_NAMES = [
    "f0_mean",          # Fundamental frequency (vocal fold tension)
    "f0_std",           # Pitch stability
    "energy_mean",      # RMS energy
    "energy_std",       # Energy stability
    "hnr",              # Harmonic-to-noise ratio (voice clarity)
    "jitter",           # Pitch perturbation (neuromuscular control)
    "shimmer",          # Amplitude perturbation
    "spectral_centroid", # Spectral brightness
    "spectral_tilt",    # Spectral slope (breathiness indicator)
    "mfcc1_mean",       # 1st MFCC (overall spectral shape)
    "mfcc2_mean",       # 2nd MFCC (spectral contrast)
    "mfcc3_mean",       # 3rd MFCC (finer spectral shape)
    "duration",         # Segment duration (speaking rate component)
]
N_SEGMENT_FEATURES = len(SEGMENT_FEATURE_NAMES)

# Recording-level aggregated residual feature names
# For each category (vowel, consonant), we compute:
#   - mean residual (13 dims)
#   - std of residual (13 dims)
#   - magnitude of residual vector (1 dim) = "how different is voice overall"
# Plus cross-category features (3 dims)
# Total = 2 categories * (13 + 13 + 1) + 3 = 57 features

def get_residual_feature_names() -> List[str]:
    """Return names of the final recording-level residual features."""
    names = []
    for cat in ["vowel", "consonant"]:
        for feat in SEGMENT_FEATURE_NAMES:
            names.append(f"residual_{cat}_{feat}_mean")
        for feat in SEGMENT_FEATURE_NAMES:
            names.append(f"residual_{cat}_{feat}_std")
        names.append(f"residual_{cat}_magnitude")
    # Cross-category
    names.append("residual_vowel_consonant_ratio")  # ratio of vowel to consonant deviation
    names.append("n_vowel_segments")
    names.append("n_consonant_segments")
    return names


N_RESIDUAL_FEATURES = len(get_residual_feature_names())  # 57


# ─── Per-segment feature extraction ────────────────────────────────────────────

def extract_segment_features(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    min_samples: int = 256,
) -> Optional[np.ndarray]:
    """
    Extract acoustic features from a single phoneme segment.
    
    Parameters
    ----------
    y : np.ndarray
        Full audio waveform (mono).
    sr : int
        Sample rate.
    start_sec, end_sec : float
        Segment boundaries in seconds.
    min_samples : int
        Minimum number of audio samples for a valid segment.
    
    Returns
    -------
    np.ndarray or None
        Feature vector of length N_SEGMENT_FEATURES, or None if too short.
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    
    # Clamp to audio bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(y), end_sample)
    
    segment = y[start_sample:end_sample]
    
    if len(segment) < min_samples:
        return None
    
    try:
        features = np.zeros(N_SEGMENT_FEATURES, dtype=np.float32)
        
        # ── F0 (pitch) ──
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f0, voiced, _ = librosa.pyin(
                    segment, fmin=65, fmax=500, sr=sr,
                    frame_length=max(641, min(1024, len(segment))),
                )
            f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            if len(f0_valid) > 0:
                features[0] = np.mean(f0_valid)   # f0_mean
                features[1] = np.std(f0_valid)     # f0_std
                
                # ── Jitter (pitch perturbation) ──
                if len(f0_valid) > 2:
                    periods = 1.0 / (f0_valid + 1e-10)
                    diffs = np.abs(np.diff(periods))
                    features[5] = np.mean(diffs) / (np.mean(periods) + 1e-10)
        except Exception:
            pass
        
        # ── Energy ──
        rms = librosa.feature.rms(y=segment, frame_length=min(512, len(segment)),
                                   hop_length=min(256, len(segment) // 2 + 1))[0]
        if len(rms) > 0:
            features[2] = np.mean(rms)   # energy_mean
            features[3] = np.std(rms)    # energy_std
        
        # ── HNR (harmonic-to-noise ratio) ──
        try:
            S = np.abs(librosa.stft(segment, n_fft=min(512, len(segment))))
            if S.shape[1] > 0:
                harmonic, percussive = librosa.decompose.hpss(S)
                h_energy = np.sum(harmonic ** 2)
                n_energy = np.sum(percussive ** 2)
                features[4] = 10 * np.log10(h_energy / (n_energy + 1e-10)) if n_energy > 0 else 20.0
        except Exception:
            pass
        
        # ── Shimmer (amplitude perturbation) ──
        frame_len = min(int(0.025 * sr), len(segment))
        hop_len = max(1, frame_len // 2)
        if frame_len > 0 and len(segment) >= frame_len:
            try:
                frames = librosa.util.frame(segment, frame_length=frame_len, hop_length=hop_len)
                peaks = np.max(np.abs(frames), axis=0)
                valid_peaks = peaks[peaks > np.mean(peaks) * 0.1]
                if len(valid_peaks) > 2:
                    diffs = np.abs(np.diff(valid_peaks))
                    features[6] = np.mean(diffs) / (np.mean(valid_peaks) + 1e-10)
            except Exception:
                pass
        
        # ── Spectral centroid ──
        try:
            cent = librosa.feature.spectral_centroid(
                y=segment, sr=sr, n_fft=min(512, len(segment)),
                hop_length=min(256, len(segment) // 2 + 1),
            )
            features[7] = np.mean(cent) if cent.size > 0 else 0.0
        except Exception:
            pass
        
        # ── Spectral tilt (slope of log-spectrum) ──
        try:
            S_mag = np.abs(librosa.stft(segment, n_fft=min(512, len(segment))))
            if S_mag.shape[0] > 1 and S_mag.shape[1] > 0:
                mean_spectrum = np.mean(S_mag, axis=1)
                log_spec = np.log(mean_spectrum + 1e-10)
                x = np.arange(len(log_spec))
                if len(x) > 1:
                    slope = np.polyfit(x, log_spec, 1)[0]
                    features[8] = slope
        except Exception:
            pass
        
        # ── MFCCs 1-3 ──
        try:
            mfccs = librosa.feature.mfcc(
                y=segment, sr=sr, n_mfcc=4,
                n_fft=min(512, len(segment)),
                hop_length=min(256, len(segment) // 2 + 1),
            )
            if mfccs.shape[1] > 0:
                features[9] = np.mean(mfccs[1])   # mfcc1 (skip mfcc0=energy)
                features[10] = np.mean(mfccs[2])  # mfcc2
                features[11] = np.mean(mfccs[3])  # mfcc3
        except Exception:
            pass
        
        # ── Duration ──
        features[12] = end_sec - start_sec
        
        return features
    
    except Exception as e:
        logger.debug("Segment feature extraction failed: %s", e)
        return None


# ─── Per-recording phoneme feature extraction ──────────────────────────────────

@dataclass
class RecordingPhonemeData:
    """Phoneme-level features for a single recording."""
    audio_path: str
    vowel_features: np.ndarray     # (n_vowels, N_SEGMENT_FEATURES)
    consonant_features: np.ndarray  # (n_consonants, N_SEGMENT_FEATURES)
    vowel_labels: List[str]         # grapheme text for each vowel segment
    consonant_labels: List[str]     # grapheme text for each consonant segment
    n_vowels: int = 0
    n_consonants: int = 0


def extract_phoneme_features_for_recording(
    audio_path: str,
    alignment: AlignmentResult,
    sr: int = 16000,
    min_segment_ms: int = 30,
) -> Optional[RecordingPhonemeData]:
    """
    Extract per-phoneme features for a single recording.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file.
    alignment : AlignmentResult
        Word/phoneme alignment from Whisper.
    sr : int
        Sample rate.
    min_segment_ms : int
        Minimum segment duration in milliseconds. Shorter segments are skipped.
    
    Returns
    -------
    RecordingPhonemeData or None
        Per-phoneme features organized by category.
    """
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception as e:
        logger.warning("Failed to load audio %s: %s", audio_path, e)
        return None
    
    min_duration_sec = min_segment_ms / 1000.0
    min_samples = int(min_duration_sec * sr)
    
    vowel_feats = []
    consonant_feats = []
    vowel_labels = []
    consonant_labels = []
    
    for phoneme in alignment.phonemes:
        duration = phoneme.end - phoneme.start
        if duration < min_duration_sec:
            continue
        
        feats = extract_segment_features(y, sr, phoneme.start, phoneme.end, min_samples)
        if feats is None:
            continue
        
        if phoneme.category == "vowel":
            vowel_feats.append(feats)
            vowel_labels.append(phoneme.text)
        elif phoneme.category == "consonant":
            consonant_feats.append(feats)
            consonant_labels.append(phoneme.text)
    
    vowel_arr = np.array(vowel_feats) if vowel_feats else np.empty((0, N_SEGMENT_FEATURES))
    cons_arr = np.array(consonant_feats) if consonant_feats else np.empty((0, N_SEGMENT_FEATURES))
    
    return RecordingPhonemeData(
        audio_path=audio_path,
        vowel_features=vowel_arr,
        consonant_features=cons_arr,
        vowel_labels=vowel_labels,
        consonant_labels=consonant_labels,
        n_vowels=len(vowel_feats),
        n_consonants=len(consonant_feats),
    )


# ─── Per-speaker phoneme baseline ──────────────────────────────────────────────

@dataclass
class SpeakerPhonemeBaseline:
    """
    Per-speaker baseline statistics for each phoneme category.
    
    Built from ALL recordings of this speaker. Represents "what this speaker's
    vowels/consonants normally sound like." The residual from this baseline
    is the physiological signal.
    """
    speaker: str
    vowel_mean: np.ndarray          # (N_SEGMENT_FEATURES,)
    vowel_std: np.ndarray           # (N_SEGMENT_FEATURES,)
    consonant_mean: np.ndarray      # (N_SEGMENT_FEATURES,)
    consonant_std: np.ndarray       # (N_SEGMENT_FEATURES,)
    n_vowel_instances: int = 0
    n_consonant_instances: int = 0


def build_speaker_baseline(
    recordings: List[RecordingPhonemeData],
    speaker: str,
) -> SpeakerPhonemeBaseline:
    """
    Build a phoneme baseline for a single speaker from all their recordings.
    
    This pools all vowel instances and all consonant instances across all
    recordings to compute the speaker's typical acoustic profile per category.
    
    Parameters
    ----------
    recordings : list of RecordingPhonemeData
        All phoneme-level data for this speaker.
    speaker : str
        Speaker name.
    
    Returns
    -------
    SpeakerPhonemeBaseline
        Mean and std for each phoneme category.
    """
    all_vowels = []
    all_consonants = []
    
    for rec in recordings:
        if rec.vowel_features.shape[0] > 0:
            all_vowels.append(rec.vowel_features)
        if rec.consonant_features.shape[0] > 0:
            all_consonants.append(rec.consonant_features)
    
    if all_vowels:
        vowels = np.vstack(all_vowels)
        vowel_mean = np.mean(vowels, axis=0)
        vowel_std = np.std(vowels, axis=0) + 1e-8
    else:
        vowel_mean = np.zeros(N_SEGMENT_FEATURES)
        vowel_std = np.ones(N_SEGMENT_FEATURES)
    
    if all_consonants:
        consonants = np.vstack(all_consonants)
        consonant_mean = np.mean(consonants, axis=0)
        consonant_std = np.std(consonants, axis=0) + 1e-8
    else:
        consonant_mean = np.zeros(N_SEGMENT_FEATURES)
        consonant_std = np.ones(N_SEGMENT_FEATURES)
    
    baseline = SpeakerPhonemeBaseline(
        speaker=speaker,
        vowel_mean=vowel_mean,
        vowel_std=vowel_std,
        consonant_mean=consonant_mean,
        consonant_std=consonant_std,
        n_vowel_instances=sum(r.n_vowels for r in recordings),
        n_consonant_instances=sum(r.n_consonants for r in recordings),
    )
    
    logger.info(
        "  %s baseline: %d vowel instances, %d consonant instances",
        speaker, baseline.n_vowel_instances, baseline.n_consonant_instances,
    )
    
    return baseline


# ─── Residual computation ──────────────────────────────────────────────────────

def compute_recording_residuals(
    recording: RecordingPhonemeData,
    baseline: SpeakerPhonemeBaseline,
) -> np.ndarray:
    """
    Compute the aggregated residual feature vector for a single recording.
    
    For each phoneme segment in the recording:
      residual = (segment_features - baseline_mean) / baseline_std
    
    Then aggregate across all segments per category:
      - mean residual (captures central tendency of deviation)
      - std of residual (captures variability of deviation)
      - magnitude = L2 norm of mean residual (scalar "how different" measure)
    
    Parameters
    ----------
    recording : RecordingPhonemeData
        Per-phoneme features for this recording.
    baseline : SpeakerPhonemeBaseline
        Speaker's baseline statistics.
    
    Returns
    -------
    np.ndarray
        Feature vector of length N_RESIDUAL_FEATURES (57 dims).
    """
    features = np.zeros(N_RESIDUAL_FEATURES, dtype=np.float32)
    idx = 0
    
    for cat, cat_features, cat_mean, cat_std in [
        ("vowel", recording.vowel_features, baseline.vowel_mean, baseline.vowel_std),
        ("consonant", recording.consonant_features, baseline.consonant_mean, baseline.consonant_std),
    ]:
        if cat_features.shape[0] > 0:
            # Z-score each segment relative to baseline
            residuals = (cat_features - cat_mean) / cat_std
            
            # Mean residual across all segments of this category in this recording
            mean_residual = np.mean(residuals, axis=0)
            std_residual = np.std(residuals, axis=0) if residuals.shape[0] > 1 else np.zeros(N_SEGMENT_FEATURES)
            magnitude = np.linalg.norm(mean_residual)
            
            features[idx:idx + N_SEGMENT_FEATURES] = mean_residual
            idx += N_SEGMENT_FEATURES
            features[idx:idx + N_SEGMENT_FEATURES] = std_residual
            idx += N_SEGMENT_FEATURES
            features[idx] = magnitude
            idx += 1
        else:
            # No segments of this category — fill with zeros
            idx += 2 * N_SEGMENT_FEATURES + 1
    
    # Cross-category features
    vowel_mag = features[2 * N_SEGMENT_FEATURES]  # vowel magnitude
    cons_mag = features[2 * (2 * N_SEGMENT_FEATURES + 1) - 1]  # consonant magnitude
    features[idx] = vowel_mag / (cons_mag + 1e-10)  # vowel/consonant deviation ratio
    idx += 1
    features[idx] = recording.n_vowels
    idx += 1
    features[idx] = recording.n_consonants
    idx += 1
    
    return features


# ─── High-level: full pipeline for one speaker ─────────────────────────────────

def extract_residual_features_for_speaker(
    recordings: List[RecordingPhonemeData],
    speaker: str,
) -> Tuple[np.ndarray, SpeakerPhonemeBaseline]:
    """
    Full pipeline: build baseline, compute residuals for all recordings.
    
    Parameters
    ----------
    recordings : list of RecordingPhonemeData
        All phoneme-level data for this speaker.
    speaker : str
        Speaker name.
    
    Returns
    -------
    tuple of (features, baseline)
        features: np.ndarray of shape (n_recordings, N_RESIDUAL_FEATURES)
        baseline: SpeakerPhonemeBaseline
    """
    baseline = build_speaker_baseline(recordings, speaker)
    
    features = []
    for rec in recordings:
        residual = compute_recording_residuals(rec, baseline)
        features.append(residual)
    
    features_arr = np.array(features, dtype=np.float32)
    
    logger.info(
        "  %s: %d recordings → %d-dim residual features",
        speaker, len(recordings), features_arr.shape[1],
    )
    
    return features_arr, baseline


# ─── Analysis helpers ───────────────────────────────────────────────────────────

def analyze_phoneme_sensitivity(
    recordings: List[RecordingPhonemeData],
    glucose_values: np.ndarray,
    baseline: SpeakerPhonemeBaseline,
) -> Dict[str, float]:
    """
    Analyze which phoneme-level features correlate most with glucose.
    
    This helps identify which specific acoustic dimensions carry the
    glucose signal (e.g., "vowel jitter correlates r=0.4 with glucose").
    
    Parameters
    ----------
    recordings : list of RecordingPhonemeData
        Per-phoneme data for all recordings.
    glucose_values : np.ndarray
        Matched glucose values (mg/dL).
    baseline : SpeakerPhonemeBaseline
        Speaker baseline.
    
    Returns
    -------
    dict
        {feature_name: pearson_r} for features with |r| > 0.1
    """
    from scipy import stats
    
    n = len(recordings)
    if n < 10:
        return {}
    
    # Compute per-recording mean residuals for vowels
    correlations = {}
    
    for cat, cat_mean, cat_std in [
        ("vowel", baseline.vowel_mean, baseline.vowel_std),
        ("consonant", baseline.consonant_mean, baseline.consonant_std),
    ]:
        for feat_idx, feat_name in enumerate(SEGMENT_FEATURE_NAMES):
            residuals = []
            for rec in recordings:
                cat_feats = rec.vowel_features if cat == "vowel" else rec.consonant_features
                if cat_feats.shape[0] > 0:
                    seg_residuals = (cat_feats[:, feat_idx] - cat_mean[feat_idx]) / cat_std[feat_idx]
                    residuals.append(np.mean(seg_residuals))
                else:
                    residuals.append(0.0)
            
            residuals = np.array(residuals)
            if np.std(residuals) > 1e-8 and np.std(glucose_values) > 1e-8:
                r, p = stats.pearsonr(residuals, glucose_values)
                if abs(r) > 0.1:
                    key = f"{cat}_{feat_name}"
                    correlations[key] = round(float(r), 3)
    
    # Sort by absolute correlation
    correlations = dict(sorted(correlations.items(), key=lambda x: -abs(x[1])))
    return correlations
