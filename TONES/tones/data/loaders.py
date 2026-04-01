"""
Unified data loading for TONES pipeline.
==========================================
Consolidates all glucose CSV loading, timestamp parsing, glucose matching,
and audio file collection into one canonical implementation.

This replaces the 8+ duplicated copies across the legacy scripts.
"""

import hashlib
import logging
import re
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Audio Format Conversion (opus/waptt -> wav)
# =============================================================================

def _find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg executable."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    # Common Windows locations
    for candidate in [
        r"C:\Users\whgeb\OneDrive\TONES\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]:
        if Path(candidate).exists():
            return candidate
    return None


def convert_audio_to_wav(
    input_path: Path,
    output_dir: Optional[Path] = None,
    sr: int = 16000,
) -> Optional[Path]:
    """
    Convert an opus or waptt audio file to WAV using ffmpeg.

    Parameters
    ----------
    input_path : Path
        Path to the source audio file.
    output_dir : Path, optional
        Directory for converted files. Defaults to a '.converted' subdir.
    sr : int
        Target sample rate.

    Returns
    -------
    Path or None
        Path to the converted WAV file, or None if conversion failed.
    """
    if input_path.suffix.lower() == ".wav":
        return input_path

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        logger.warning(
            "ffmpeg not found — cannot convert %s. Install ffmpeg to use opus/waptt files.",
            input_path.name,
        )
        return None

    if output_dir is None:
        output_dir = input_path.parent / ".converted"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (input_path.stem + ".wav")

    # Skip if already converted
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    try:
        cmd = [
            ffmpeg, "-y", "-i", str(input_path),
            "-ar", str(sr), "-ac", "1",
            "-sample_fmt", "s16",
            str(output_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and output_path.exists():
            return output_path
        else:
            logger.warning(
                "ffmpeg conversion failed for %s: %s",
                input_path.name, result.stderr.decode(errors="replace")[:200],
            )
            return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out for %s", input_path.name)
        return None
    except Exception as e:
        logger.warning("Conversion error for %s: %s", input_path.name, e)
        return None


# =============================================================================
# Timestamp Parsing
# =============================================================================

# Compiled patterns for speed (most specific first)
_TIMESTAMP_PATTERNS = [
    # "2023-11-11 um 20.19.20" / "2023-11-11 at 20.19.20" / hyphenated
    re.compile(r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)\s*(\d{1,2})\.(\d{2})\.(\d{2})'),
    # "2023-11-11 um 20.19" (no seconds)
    re.compile(r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)\s*(\d{1,2})\.(\d{2})'),
    # Fallback: "2023-11-11-um-20.19.20" fully hyphenated
    re.compile(r'(\d{4}-\d{2}-\d{2})-um-(\d{1,2})\.(\d{2})\.(\d{2})'),
    # Very flexible: date + H:MM:SS or H.MM.SS with optional separator
    re.compile(r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)?\s*(\d{1,2})[\.:h](\d{2})[\.:h]?(\d{2})?'),
]


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract a datetime from a WhatsApp audio filename.

    Handles these formats (and variants):
      - "WhatsApp Audio 2023-11-11 um 20.19.20_hash.ext"
      - "WhatsApp-Audio-2023-11-11-um-20.19.20_hash.ext"
      - "131_WhatsApp Audio 2023-11-06 um 20.59.43_hash.ext"  (Wolf prefix)
      - "AnyConv.com__101_WhatsApp Audio 2023-11-08 um 12.32.08_hash.ext"
      - "Wolf 99 WhatsApp Audio 2023-11-02 um 18.08.01_hash.ext"

    Parameters
    ----------
    filename : str
        The filename (not full path) to parse.

    Returns
    -------
    datetime or None
        Parsed timestamp, or None if no pattern matched.
    """
    for pattern in _TIMESTAMP_PATTERNS:
        match = pattern.search(str(filename))
        if match:
            groups = match.groups()
            try:
                date_str = groups[0]
                hour = int(groups[1])
                minute = int(groups[2])
                second = int(groups[3]) if len(groups) >= 4 and groups[3] is not None else 0
                return datetime.strptime(
                    f"{date_str} {hour:02d}:{minute:02d}:{second:02d}",
                    "%Y-%m-%d %H:%M:%S",
                )
            except (ValueError, IndexError):
                continue
    return None


# =============================================================================
# Glucose CSV Loading
# =============================================================================

# Keywords for identifying columns across languages
_TIMESTAMP_KEYWORDS = ["timestamp", "zeitstempel", "carimbo", "data/hora"]
_GLUCOSE_KEYWORDS = [
    "historic glucose", "glukosewert-verlauf", "glukosewert",
    "histórico de glicose", "historic glucose mg/dl",
]
_HEADER_KEYWORDS = ["device", "gerät", "serial", "seriennummer", "dispositivo"]


def load_glucose_csv(
    csv_paths: List[str],
    glucose_unit: str,
    base_dir: Path,
) -> pd.DataFrame:
    """
    Load and merge glucose data from one or more FreeStyle Libre CSV exports.

    Handles English, German, and Portuguese CSV formats. Automatically detects
    header rows and column names.

    Parameters
    ----------
    csv_paths : list of str
        Relative paths to CSV files (relative to base_dir).
    glucose_unit : str
        One of "mg/dL", "mmol/L", or "auto".
    base_dir : Path
        Project root directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['timestamp', 'glucose_mg_dl'], sorted by time.
        Empty DataFrame if no valid data found.
    """
    all_dfs: List[pd.DataFrame] = []

    for csv_rel in csv_paths:
        full_path = base_dir / csv_rel
        if not full_path.exists():
            logger.warning("CSV not found: %s", full_path)
            continue

        try:
            df = _load_single_csv(full_path, glucose_unit)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                logger.info(
                    "Loaded %d readings from %s (%s to %s)",
                    len(df), csv_rel,
                    df["timestamp"].min().strftime("%Y-%m-%d"),
                    df["timestamp"].max().strftime("%Y-%m-%d"),
                )
        except Exception as e:
            logger.warning("Failed to load %s: %s", csv_rel, e)

    if not all_dfs:
        return pd.DataFrame(columns=["timestamp", "glucose_mg_dl"])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = (
        combined
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return combined


def _load_single_csv(full_path: Path, glucose_unit: str) -> Optional[pd.DataFrame]:
    """Load a single FreeStyle Libre CSV file."""
    # Detect header row (FreeStyle Libre has 1-3 metadata lines)
    with open(full_path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    skiprows = 0
    for i, line in enumerate(lines[:5]):
        lower = line.lower()
        if any(kw in lower for kw in _HEADER_KEYWORDS):
            skiprows = i
            break

    df = pd.read_csv(full_path, skiprows=skiprows, encoding="utf-8-sig")

    # Identify columns
    timestamp_col = _find_column(df.columns, _TIMESTAMP_KEYWORDS)
    glucose_col = _find_column(df.columns, _GLUCOSE_KEYWORDS)

    # Fallback to positional columns (standard FreeStyle Libre layout)
    if timestamp_col is None and len(df.columns) > 2:
        timestamp_col = df.columns[2]
    if glucose_col is None and len(df.columns) > 3:
        glucose_col = df.columns[3]

    if timestamp_col is None or glucose_col is None:
        logger.warning("Could not identify columns in %s: %s", full_path.name, list(df.columns[:8]))
        return None

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df[timestamp_col], format="%d-%m-%Y %H:%M", errors="coerce")
    if df["timestamp"].isna().all():
        df["timestamp"] = pd.to_datetime(df[timestamp_col], dayfirst=True, errors="coerce")

    df["glucose"] = pd.to_numeric(df[glucose_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "glucose"])

    if len(df) == 0:
        return None

    # Convert to mg/dL
    df["glucose_mg_dl"] = _convert_to_mg_dl(df["glucose"], glucose_unit)

    return df[["timestamp", "glucose_mg_dl"]]


def _find_column(columns: pd.Index, keywords: List[str]) -> Optional[str]:
    """Find a column matching any of the given keywords."""
    for col in columns:
        col_lower = col.lower().strip()
        if any(kw in col_lower for kw in keywords):
            return col
    return None


def _convert_to_mg_dl(values: pd.Series, unit: str) -> pd.Series:
    """Convert glucose values to mg/dL."""
    if unit == "mmol/L":
        return values * 18.0182
    elif unit == "mg/dL":
        return values
    else:  # auto-detect
        mean_val = values.mean()
        if mean_val < 30:  # mmol/L values are typically 3-20
            return values * 18.0182
        return values


# =============================================================================
# Glucose Matching
# =============================================================================

def find_matching_glucose(
    audio_timestamp: datetime,
    glucose_df: pd.DataFrame,
    window_minutes: int = 30,
    offset_minutes: int = 0,
    use_interpolation: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find glucose value at a given audio timestamp.

    Uses linear interpolation between the two bracketing CGM readings when
    possible, falling back to nearest-neighbor.

    Parameters
    ----------
    audio_timestamp : datetime
        Timestamp of the audio recording.
    glucose_df : pd.DataFrame
        Must have columns ['timestamp', 'glucose_mg_dl'].
    window_minutes : int
        Maximum time difference allowed (in minutes).
    offset_minutes : int
        Time offset to apply before matching (accounts for CGM lag).
    use_interpolation : bool
        If True, use linear interpolation; otherwise use nearest-neighbor.

    Returns
    -------
    tuple of (glucose_mg_dl, time_diff_minutes)
        Returns (None, None) if no match found within window.
    """
    if glucose_df.empty:
        return None, None

    search_time = audio_timestamp + timedelta(minutes=offset_minutes)
    ts = pd.Timestamp(search_time)

    if use_interpolation:
        return _match_interpolated(ts, glucose_df, window_minutes)
    else:
        return _match_nearest(ts, glucose_df, window_minutes)


def _match_interpolated(
    ts: pd.Timestamp, glucose_df: pd.DataFrame, window_minutes: int
) -> Tuple[Optional[float], Optional[float]]:
    """Linear interpolation between bracketing CGM readings."""
    time_diffs_sec = (glucose_df["timestamp"] - ts).dt.total_seconds()

    before_mask = time_diffs_sec <= 0
    after_mask = time_diffs_sec >= 0

    has_before = before_mask.any()
    has_after = after_mask.any()

    if not has_before and not has_after:
        return None, None

    before_diff_min = float("inf")
    after_diff_min = float("inf")

    if has_before:
        before_idx = time_diffs_sec[before_mask].idxmax()
        before_diff_min = abs(time_diffs_sec[before_idx]) / 60

    if has_after:
        after_idx = time_diffs_sec[after_mask].idxmin()
        after_diff_min = abs(time_diffs_sec[after_idx]) / 60

    nearest_diff = min(before_diff_min, after_diff_min)
    if nearest_diff > window_minutes:
        return None, None

    # Interpolate if we have readings on both sides within window
    if (has_before and before_diff_min <= window_minutes and
            has_after and after_diff_min <= window_minutes):
        before_glucose = glucose_df.loc[before_idx, "glucose_mg_dl"]
        after_glucose = glucose_df.loc[after_idx, "glucose_mg_dl"]
        total_span = before_diff_min + after_diff_min

        if total_span == 0:
            return float(before_glucose), 0.0

        weight_before = after_diff_min / total_span
        weight_after = before_diff_min / total_span
        interpolated = weight_before * before_glucose + weight_after * after_glucose
        return float(interpolated), float(nearest_diff)

    # Only one side available
    if has_before and before_diff_min <= window_minutes:
        return float(glucose_df.loc[before_idx, "glucose_mg_dl"]), float(before_diff_min)
    if has_after and after_diff_min <= window_minutes:
        return float(glucose_df.loc[after_idx, "glucose_mg_dl"]), float(after_diff_min)

    return None, None


def _match_nearest(
    ts: pd.Timestamp, glucose_df: pd.DataFrame, window_minutes: int
) -> Tuple[Optional[float], Optional[float]]:
    """Nearest-neighbor glucose matching."""
    time_diffs_min = abs((glucose_df["timestamp"] - ts).dt.total_seconds() / 60)
    min_diff = time_diffs_min.min()

    if min_diff <= window_minutes:
        idx = time_diffs_min.idxmin()
        return float(glucose_df.loc[idx, "glucose_mg_dl"]), float(min_diff)

    return None, None


# =============================================================================
# Audio File Collection
# =============================================================================

def get_file_hash(file_path: Path) -> Optional[str]:
    """MD5 hash of first 8KB for fast deduplication."""
    try:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            hasher.update(f.read(8192))
        return hasher.hexdigest()
    except Exception:
        return None


def collect_audio_files(
    audio_dirs: List[str],
    audio_ext: List[str],
    base_dir: Path,
    deduplicate: bool = True,
    auto_convert: bool = True,
    target_sr: int = 16000,
    recursive: bool = True,
) -> List[Path]:
    """
    Collect unique audio files from directories, deduplicating by content hash.
    Automatically converts opus/waptt files to WAV if ffmpeg is available.

    Parameters
    ----------
    audio_dirs : list of str
        Relative directory paths containing audio files.
    audio_ext : list of str
        File extensions to include (e.g., [".wav", ".opus"]).
    base_dir : Path
        Project root directory.
    deduplicate : bool
        If True, skip files with duplicate content hashes.
    auto_convert : bool
        If True, auto-convert non-WAV files to WAV using ffmpeg.
    target_sr : int
        Target sample rate for conversion.
    recursive : bool
        If True, search subdirectories (**/*.ext). If False, top-level only (*.ext).

    Returns
    -------
    list of Path
        Sorted list of unique audio file paths (all WAV after conversion).
    """
    all_files: List[Path] = []
    seen_hashes: set = set()
    # Track filenames already seen (to deduplicate across formats)
    seen_stems: set = set()
    converted_count = 0

    for dir_rel in audio_dirs:
        dir_path = base_dir / dir_rel
        if not dir_path.exists():
            logger.warning("Audio dir not found: %s", dir_path)
            continue

        for ext in audio_ext:
            glob_pattern = f"**/*{ext}" if recursive else f"*{ext}"
            for f in sorted(dir_path.glob(glob_pattern)):
                if deduplicate:
                    file_hash = get_file_hash(f)
                    if file_hash and file_hash in seen_hashes:
                        continue
                    if file_hash:
                        seen_hashes.add(file_hash)

                # Convert non-WAV files
                if auto_convert and f.suffix.lower() in (".opus", ".waptt"):
                    converted = convert_audio_to_wav(f, sr=target_sr)
                    if converted is not None:
                        # Check if we already have the WAV version of this file
                        stem = converted.stem
                        if stem in seen_stems:
                            continue
                        seen_stems.add(stem)
                        all_files.append(converted)
                        converted_count += 1
                    continue

                seen_stems.add(f.stem)
                all_files.append(f)

    if converted_count > 0:
        logger.info("Auto-converted %d non-WAV files to WAV", converted_count)

    return all_files


# =============================================================================
# High-Level: Load All Participant Data
# =============================================================================

def compute_glucose_rate_of_change(
    audio_timestamp: datetime,
    glucose_df: pd.DataFrame,
    window_minutes: int = 15,
) -> Optional[float]:
    """
    Compute glucose rate of change at a given timestamp using CGM data.

    Uses glucose readings within +/- window_minutes to estimate the
    instantaneous rate of change (mg/dL per minute).

    Parameters
    ----------
    audio_timestamp : datetime
        Time at which to compute rate of change.
    glucose_df : pd.DataFrame
        Must have columns ['timestamp', 'glucose_mg_dl'].
    window_minutes : int
        Look-back and look-ahead window.

    Returns
    -------
    float or None
        Rate of change in mg/dL/min, or None if not computable.
    """
    if glucose_df.empty or len(glucose_df) < 2:
        return None

    ts = pd.Timestamp(audio_timestamp)
    time_diffs_sec = (glucose_df["timestamp"] - ts).dt.total_seconds()

    # Find readings before and after
    before_mask = (time_diffs_sec >= -window_minutes * 60) & (time_diffs_sec < 0)
    after_mask = (time_diffs_sec > 0) & (time_diffs_sec <= window_minutes * 60)

    if not before_mask.any() or not after_mask.any():
        return None

    # Get nearest before and nearest after
    before_idx = time_diffs_sec[before_mask].idxmax()
    after_idx = time_diffs_sec[after_mask].idxmin()

    g_before = glucose_df.loc[before_idx, "glucose_mg_dl"]
    g_after = glucose_df.loc[after_idx, "glucose_mg_dl"]
    t_diff_min = (time_diffs_sec[after_idx] - time_diffs_sec[before_idx]) / 60.0

    if t_diff_min <= 0:
        return None

    return (g_after - g_before) / t_diff_min


def classify_glucose_rate(rate: Optional[float], threshold: float = 1.0) -> str:
    """
    Classify glucose rate of change into rising/stable/falling.

    Parameters
    ----------
    rate : float or None
        Rate of change in mg/dL/min.
    threshold : float
        Threshold for rising/falling classification (default 1.0 mg/dL/min).

    Returns
    -------
    str
        One of "rising", "stable", "falling", or "unknown".
    """
    if rate is None:
        return "unknown"
    if rate > threshold:
        return "rising"
    elif rate < -threshold:
        return "falling"
    else:
        return "stable"


def classify_glucose_regime(glucose_mg_dl: float) -> str:
    """
    Classify glucose into clinically-relevant regimes.

    Parameters
    ----------
    glucose_mg_dl : float
        Glucose value in mg/dL.

    Returns
    -------
    str
        One of "hypo_risk", "normal", "hyper_risk".
    """
    if glucose_mg_dl < 80:
        return "hypo_risk"
    elif glucose_mg_dl > 140:
        return "hyper_risk"
    else:
        return "normal"


def load_participant_data(
    participant_name: str,
    participant_cfg: Dict,
    base_dir: Path,
    matching_cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Load all matched audio-glucose pairs for a single participant.

    Parameters
    ----------
    participant_name : str
        Name of the participant.
    participant_cfg : dict
        Participant configuration from config.yaml.
    base_dir : Path
        Project root directory.
    matching_cfg : dict, optional
        Matching configuration (window_minutes, use_interpolation).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, audio_path, audio_timestamp,
        glucose_mg_dl, time_diff_minutes, audio_format.
    """
    if matching_cfg is None:
        matching_cfg = {"window_minutes": 30, "use_interpolation": True}

    window = matching_cfg.get("window_minutes", 30)
    interpolate = matching_cfg.get("use_interpolation", True)
    offset = participant_cfg.get("optimal_offset", 0)

    # Load glucose
    glucose_df = load_glucose_csv(
        participant_cfg["glucose_csv"],
        participant_cfg.get("glucose_unit", "auto"),
        base_dir,
    )
    if glucose_df.empty:
        logger.warning("%s: No glucose data loaded", participant_name)
        return pd.DataFrame()

    # Collect audio files
    audio_files = collect_audio_files(
        participant_cfg["audio_dirs"],
        participant_cfg.get("audio_ext", [".wav"]),
        base_dir,
    )

    if not audio_files:
        logger.warning("%s: No audio files found", participant_name)
        return pd.DataFrame()

    # Match
    rows = []
    for audio_path in audio_files:
        audio_ts = parse_timestamp_from_filename(audio_path.name)
        if audio_ts is None:
            continue

        glucose_val, time_diff = find_matching_glucose(
            audio_ts, glucose_df, window, offset, interpolate
        )

        if glucose_val is not None:
            # Compute rate of change and regime classification
            rate = compute_glucose_rate_of_change(audio_ts, glucose_df, window_minutes=15)
            rate_label = classify_glucose_rate(rate)
            regime = classify_glucose_regime(glucose_val)

            rows.append({
                "subject": participant_name,
                "audio_path": str(audio_path),
                "audio_timestamp": audio_ts.isoformat(),
                "glucose_mg_dl": round(glucose_val, 2),
                "time_diff_minutes": round(time_diff, 2) if time_diff else None,
                "audio_format": audio_path.suffix,
                "glucose_rate": round(rate, 4) if rate is not None else None,
                "glucose_rate_label": rate_label,
                "glucose_regime": regime,
            })

    result = pd.DataFrame(rows)

    # Deduplicate: multiple audio files may match same timestamp (same recording, different names).
    # Keep the row with smallest time_diff_minutes (best CGM match).
    if len(result) > 1:
        result["_time_diff"] = result["time_diff_minutes"].fillna(float("inf"))
        before = len(result)
        result = (
            result.sort_values("_time_diff")
            .drop_duplicates(subset=["subject", "audio_timestamp"], keep="first")
            .drop(columns=["_time_diff"])
        )
        if len(result) < before:
            logger.info("%s: Deduplicated %d -> %d rows (same timestamp)", participant_name, before, len(result))
    logger.info(
        "%s: %d/%d audio files matched to glucose",
        participant_name, len(result), len(audio_files),
    )
    return result


def load_all_participants(cfg: Dict) -> pd.DataFrame:
    """
    Load matched audio-glucose data for ALL configured participants.

    Parameters
    ----------
    cfg : dict
        Full configuration from load_config().

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame for all participants.
    """
    base_dir = Path(cfg["base_dir"])
    matching_cfg = cfg.get("matching", {})
    participants = cfg.get("participants", {})

    all_dfs = []
    for name, pcfg in participants.items():
        if not pcfg.get("glucose_csv"):
            logger.info("%s: No glucose CSVs configured, skipping", name)
            continue

        df = load_participant_data(name, pcfg, base_dir, matching_cfg)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No data loaded from any participant!")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total: %d matched samples from %d participants", len(result), len(all_dfs))
    return result
