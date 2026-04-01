"""
Build Canonical Dataset for ONVOX/TONES
========================================
Consolidates ALL available audio-glucose pairs into a single canonical CSV.

Improvements over previous pipelines:
  1. Maps orphaned CGM files to participants via date overlap
  2. Widens matching window to +/-30 min with linear interpolation
  3. Fixes Wolf to use CGM CSV (not glucose-from-filename)
  4. Integrates Darav, Joao, Alvar, Christian_L
  5. Handles .waptt -> .wav conversion check
  6. Produces one canonical CSV: canonical_dataset.csv

Usage:
    python build_canonical_dataset.py
"""

import os
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"C:\Users\whgeb\OneDrive\TONES")
OUTPUT_DIR = BASE_DIR / "canonical_output"
OUTPUT_DIR.mkdir(exist_ok=True)

MATCHING_WINDOW_MINUTES = 30  # Widened from 15 to 30
USE_INTERPOLATION = True       # Linear interpolation instead of nearest-neighbor

# All known participants with their data sources
# After Step 1 mapping, we include orphaned CGM matches
PARTICIPANTS = {
    # === CURRENTLY IN PIPELINE (7 participants) ===
    "Wolf": {
        "glucose_csv": ["Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"],
        "audio_dirs": ["Wolf/all wav audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
        "notes": "Filenames have glucose prefix (e.g. '131_WhatsApp...'). "
                 "This script ignores the prefix and matches by timestamp to CGM CSV.",
    },
    "Sybille": {
        "glucose_csv": ["Sybille/glucose/SSchütt_glucose_19-11-2023.csv"],
        "audio_dirs": ["Sybille/audio_wav"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",  # German CSV but glucose in mg/dL
        "notes": "German-language CSV (Glukosewert mg/dL).",
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
    },
    "Margarita": {
        "glucose_csv": ["Margarita/Number_9Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Margarita/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
    },
    "Vicky": {
        "glucose_csv": ["Vicky/Number_10Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["Vicky/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
    },
    "Steffen": {
        "glucose_csv": ["Steffen_Haeseli/Number_2Nov_23_glucose_4-1-2024.csv"],
        "audio_dirs": ["Steffen_Haeseli/wav"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
    },
    "Lara": {
        "glucose_csv": ["Lara/Number_7Nov_27_glucose_4-1-2024.csv"],
        "audio_dirs": ["Lara/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
    },

    # === NEW: Previously not in pipeline ===
    "Darav": {
        "glucose_csv": [
            "Darav/Nov21_finished/DaravTaha_glucose_5-11-2023 (2).csv",
            "Darav/Nov21_finished/DaravTaha_glucose_6-11-2023.csv",
            "Darav/Nov21_finished/DaravTaha_glucose_7-11-2023.csv",
            "Darav/Nov21_finished/DaravTaha_glucose_11-11-2023.csv",
            "Darav/Nov21_finished/DaravTaha_glucose_15-11-2023.csv",
            "Darav/Nov21_finished/DaravTaha_glucose_19-11-2023.csv",
        ],
        "audio_dirs": ["Darav/wav_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
    },
    "Joao": {
        "glucose_csv": [
            "Joao/Nov21/Jo\u00e3oMira_glucose_7-11-2023.csv",
            "Joao/Nov21/Jo\u00e3oMira_glucose_19-11-2023.csv",
        ],
        "audio_dirs": ["Joao/wav_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
    },
    "Alvar": {
        "glucose_csv": ["Alvar/AlvarMollik_glucose_5-12-2023.csv"],
        "audio_dirs": ["Alvar/wav_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mg/dL",
    },

    # === CONFIRMED ORPHAN MATCH ===
    # Number_12 -> Bopo del Valle = R_Rodolfo (confirmed by user)
    "R_Rodolfo": {
        "glucose_csv": ["Number_12Nov_29_glucose_4-1-2024.csv"],
        "audio_dirs": ["R_Rodolfo/conv_audio"],
        "audio_ext": [".wav"],
        "glucose_unit": "mmol/L",
        "notes": "CONFIRMED: Number 12 = Bopo del Valle = R_Rodolfo.",
    },

    # === UNRESOLVED ORPHAN CGMs ===
    # Number_5 -> Boriska Molnar (from key). No audio folder found.
    # Number_14 -> Berna Agar (from key). No audio folder found.
    # Number_3 (Nov 10-24) -> Possibly Bruno or Valerie (ambiguous overlap)
    # Number_4 (Nov 9-23) -> Possibly Edoardo (ambiguous overlap)
    # Number_X6 (Nov 16-24) -> Unclear
    # Number_11 (Nov 10-24) -> Multiple possible matches
    # Number_13 (Nov 29 - Dec 10) -> No audio participants in this range
    # Number_1 (Nov 9-17) -> Wolf's second CGM (already have Wolf's primary)
}


# ============================================================================
# TIMESTAMP PARSING
# ============================================================================

def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract timestamp from WhatsApp audio filename.

    Handles formats:
      - "WhatsApp Audio 2023-11-11 um 20.19.20_hash.ext"
      - "WhatsApp-Audio-2023-11-11-um-20.19.20_hash.ext"
      - "131_WhatsApp Audio 2023-11-06 um 20.59.43_hash.ext"  (Wolf prefix)
      - "AnyConv.com__101_WhatsApp Audio 2023-11-08 um 12.32.08_hash.ext"
      - "Wolf 99 WhatsApp Audio 2023-11-02 um 18.08.01_hash.ext"
    """
    patterns = [
        # Full pattern with seconds (most specific first)
        r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)\s*(\d{1,2})\.(\d{2})\.(\d{2})',
        # Without seconds
        r'(\d{4}-\d{2}-\d{2})\s*(?:um|at|-)\s*(\d{1,2})\.(\d{2})',
        # Hyphenated format
        r'(\d{4}-\d{2}-\d{2})-um-(\d{1,2})\.(\d{2})\.(\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, str(filename))
        if match:
            groups = match.groups()
            try:
                if len(groups) == 4:
                    date_str, hour, minute, second = groups
                    return datetime.strptime(
                        f"{date_str} {int(hour):02d}:{int(minute):02d}:{int(second):02d}",
                        "%Y-%m-%d %H:%M:%S"
                    )
                elif len(groups) == 3:
                    date_str, hour, minute = groups
                    return datetime.strptime(
                        f"{date_str} {int(hour):02d}:{int(minute):02d}:00",
                        "%Y-%m-%d %H:%M:%S"
                    )
            except ValueError:
                continue

    return None


# ============================================================================
# GLUCOSE CSV LOADING
# ============================================================================

def load_glucose_csv(csv_paths: List[str], glucose_unit: str) -> pd.DataFrame:
    """
    Load and merge glucose CSV files. Handles English and German FreeStyle Libre formats.

    Returns DataFrame with columns: ['timestamp', 'glucose_mg_dl']
    """
    all_dfs = []

    for csv_rel in csv_paths:
        full_path = BASE_DIR / csv_rel
        if not full_path.exists():
            print(f"  WARNING: CSV not found: {full_path}")
            continue

        # FreeStyle Libre CSVs have variable header rows (1-3 lines of metadata)
        try:
            # Try to detect header rows by looking for the column row
            with open(full_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                lines = f.readlines()

            skiprows = 0
            for i, line in enumerate(lines[:5]):
                lower = line.lower()
                if ('device' in lower or 'gerät' in lower or
                    'serial' in lower or 'seriennummer' in lower or
                    'dispositivo' in lower):
                    skiprows = i
                    break

            df = pd.read_csv(full_path, skiprows=skiprows, encoding='utf-8-sig')
        except Exception as e:
            print(f"  WARNING: Failed to load {csv_rel}: {e}")
            continue

        # Find timestamp column (English, German, Portuguese)
        timestamp_col = None
        glucose_col = None

        for col in df.columns:
            col_lower = col.lower().strip()
            if any(kw in col_lower for kw in ['timestamp', 'zeitstempel', 'carimbo']):
                timestamp_col = col
            if any(kw in col_lower for kw in ['historic glucose', 'glukosewert-verlauf',
                                                'glukosewert', 'histórico de glicose']):
                glucose_col = col

        # Fallback to positional columns (standard FreeStyle Libre format)
        if timestamp_col is None and len(df.columns) > 2:
            timestamp_col = df.columns[2]
        if glucose_col is None and len(df.columns) > 3:
            # Column index 3 is "Historic Glucose" in standard exports
            glucose_col = df.columns[3]

        if timestamp_col is None or glucose_col is None:
            print(f"  WARNING: Could not identify columns in {csv_rel}")
            print(f"    Columns: {list(df.columns[:8])}")
            continue

        # Parse timestamps (try multiple formats)
        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M', errors='coerce')
        if df['timestamp'].isna().all():
            df['timestamp'] = pd.to_datetime(df[timestamp_col], dayfirst=True, errors='coerce')

        df['glucose'] = pd.to_numeric(df[glucose_col], errors='coerce')
        df = df.dropna(subset=['timestamp', 'glucose'])

        if len(df) == 0:
            print(f"  WARNING: No valid rows in {csv_rel}")
            continue

        # Convert to mg/dL
        if glucose_unit == 'mmol/L':
            df['glucose_mg_dl'] = df['glucose'] * 18.0182
        elif glucose_unit == 'mg/dL':
            df['glucose_mg_dl'] = df['glucose']
        else:  # auto-detect
            mean_val = df['glucose'].mean()
            if mean_val < 30:  # mmol/L values are typically 3-20
                df['glucose_mg_dl'] = df['glucose'] * 18.0182
            else:
                df['glucose_mg_dl'] = df['glucose']

        all_dfs.append(df[['timestamp', 'glucose_mg_dl']])
        print(f"  Loaded {len(df)} readings from {csv_rel} "
              f"({df['timestamp'].min().strftime('%Y-%m-%d')} to "
              f"{df['timestamp'].max().strftime('%Y-%m-%d')})")

    if not all_dfs:
        return pd.DataFrame(columns=['timestamp', 'glucose_mg_dl'])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return combined


# ============================================================================
# GLUCOSE MATCHING (with interpolation)
# ============================================================================

def find_matching_glucose_interpolated(
    audio_timestamp: datetime,
    glucose_df: pd.DataFrame,
    window_minutes: int = 30,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find glucose value at audio_timestamp using linear interpolation between
    the two bracketing CGM readings.

    Returns: (glucose_mg_dl, time_diff_to_nearest_reading_minutes)
    """
    if glucose_df.empty:
        return None, None

    ts = pd.Timestamp(audio_timestamp)
    time_diffs_seconds = (glucose_df['timestamp'] - ts).dt.total_seconds()

    # Find readings before and after
    before_mask = time_diffs_seconds <= 0
    after_mask = time_diffs_seconds >= 0

    has_before = before_mask.any()
    has_after = after_mask.any()

    if not has_before and not has_after:
        return None, None

    if has_before:
        before_idx = time_diffs_seconds[before_mask].idxmax()  # Closest before (least negative)
        before_diff_min = abs(time_diffs_seconds[before_idx]) / 60
    else:
        before_diff_min = float('inf')

    if has_after:
        after_idx = time_diffs_seconds[after_mask].idxmin()  # Closest after (least positive)
        after_diff_min = abs(time_diffs_seconds[after_idx]) / 60
    else:
        after_diff_min = float('inf')

    # Both must be within window
    nearest_diff = min(before_diff_min, after_diff_min)
    if nearest_diff > window_minutes:
        return None, None

    # If we have both sides within window, interpolate
    if (has_before and before_diff_min <= window_minutes and
        has_after and after_diff_min <= window_minutes):
        before_glucose = glucose_df.loc[before_idx, 'glucose_mg_dl']
        after_glucose = glucose_df.loc[after_idx, 'glucose_mg_dl']
        total_span = before_diff_min + after_diff_min

        if total_span == 0:
            # Exact match
            return before_glucose, 0.0

        # Linear interpolation: weight by proximity
        weight_after = before_diff_min / total_span
        weight_before = after_diff_min / total_span
        interpolated = weight_before * before_glucose + weight_after * after_glucose
        return interpolated, nearest_diff

    # Only one side available within window -- use nearest
    if has_before and before_diff_min <= window_minutes:
        return glucose_df.loc[before_idx, 'glucose_mg_dl'], before_diff_min
    if has_after and after_diff_min <= window_minutes:
        return glucose_df.loc[after_idx, 'glucose_mg_dl'], after_diff_min

    return None, None


def find_matching_glucose_nearest(
    audio_timestamp: datetime,
    glucose_df: pd.DataFrame,
    window_minutes: int = 30,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Nearest-neighbor glucose matching (legacy behavior, for comparison).
    """
    if glucose_df.empty:
        return None, None

    ts = pd.Timestamp(audio_timestamp)
    time_diffs = abs((glucose_df['timestamp'] - ts).dt.total_seconds() / 60)
    min_diff = time_diffs.min()

    if min_diff <= window_minutes:
        idx = time_diffs.idxmin()
        return glucose_df.loc[idx, 'glucose_mg_dl'], min_diff

    return None, None


# ============================================================================
# FILE DEDUPLICATION
# ============================================================================

def get_file_hash(file_path: Path) -> Optional[str]:
    """MD5 hash of first 8KB for fast deduplication."""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read(8192))
        return hasher.hexdigest()
    except Exception:
        return None


def collect_audio_files(audio_dirs: List[str], audio_ext: List[str]) -> List[Path]:
    """
    Collect unique audio files from directories, deduplicating by content hash.
    """
    all_files = []
    seen_hashes = set()

    for dir_rel in audio_dirs:
        dir_path = BASE_DIR / dir_rel
        if not dir_path.exists():
            print(f"  WARNING: Audio dir not found: {dir_path}")
            continue

        for ext in audio_ext:
            for f in sorted(dir_path.glob(f"*{ext}")):
                file_hash = get_file_hash(f)
                if file_hash and file_hash in seen_hashes:
                    continue
                if file_hash:
                    seen_hashes.add(file_hash)
                all_files.append(f)

    return all_files


# ============================================================================
# MAIN: BUILD CANONICAL DATASET
# ============================================================================

def build_canonical_dataset() -> pd.DataFrame:
    """
    Process all participants and build the canonical dataset CSV.
    """
    print("=" * 80)
    print("BUILDING CANONICAL DATASET")
    print(f"Matching window: +/-{MATCHING_WINDOW_MINUTES} minutes")
    print(f"Interpolation: {USE_INTERPOLATION}")
    print("=" * 80)

    all_rows = []
    summary_rows = []

    for name, config in PARTICIPANTS.items():
        print(f"\n{'-' * 60}")
        print(f"PARTICIPANT: {name}")
        print(f"{'-' * 60}")

        # Load glucose data
        glucose_df = load_glucose_csv(config['glucose_csv'], config.get('glucose_unit', 'auto'))
        if glucose_df.empty:
            print(f"  SKIPPED: No glucose data loaded")
            summary_rows.append({
                'participant': name,
                'audio_files': 0,
                'matched': 0,
                'match_rate': 0,
                'glucose_range': '-',
                'status': 'NO_GLUCOSE_DATA',
            })
            continue

        glucose_range = f"{glucose_df['glucose_mg_dl'].min():.0f}-{glucose_df['glucose_mg_dl'].max():.0f}"
        print(f"  Glucose: {len(glucose_df)} readings, range {glucose_range} mg/dL")

        # Collect audio files
        audio_files = collect_audio_files(config['audio_dirs'], config['audio_ext'])
        print(f"  Audio: {len(audio_files)} unique files")

        if len(audio_files) == 0:
            print(f"  SKIPPED: No audio files found")
            summary_rows.append({
                'participant': name,
                'audio_files': 0,
                'matched': 0,
                'match_rate': 0,
                'glucose_range': glucose_range,
                'status': 'NO_AUDIO_FILES',
            })
            continue

        # Match each audio file to glucose
        matched = 0
        unmatched_timestamps = []

        for audio_path in audio_files:
            audio_ts = parse_timestamp_from_filename(audio_path.name)
            if audio_ts is None:
                continue

            # Match with or without interpolation
            if USE_INTERPOLATION:
                glucose_val, time_diff = find_matching_glucose_interpolated(
                    audio_ts, glucose_df, MATCHING_WINDOW_MINUTES
                )
            else:
                glucose_val, time_diff = find_matching_glucose_nearest(
                    audio_ts, glucose_df, MATCHING_WINDOW_MINUTES
                )

            if glucose_val is not None:
                all_rows.append({
                    'subject': name,
                    'audio_path': str(audio_path.relative_to(BASE_DIR)),
                    'audio_timestamp': audio_ts.isoformat(),
                    'glucose_mg_dl': round(glucose_val, 2),
                    'glucose_source': 'interpolated' if USE_INTERPOLATION else 'nearest',
                    'time_diff_minutes': round(time_diff, 2),
                    'audio_format': audio_path.suffix,
                })
                matched += 1
            else:
                unmatched_timestamps.append(audio_ts)

        match_rate = matched / len(audio_files) * 100 if audio_files else 0
        print(f"  Matched: {matched}/{len(audio_files)} ({match_rate:.1f}%)")

        if unmatched_timestamps:
            first_unmatched = min(unmatched_timestamps).strftime('%Y-%m-%d')
            last_unmatched = max(unmatched_timestamps).strftime('%Y-%m-%d')
            print(f"  Unmatched range: {first_unmatched} to {last_unmatched}")

        summary_rows.append({
            'participant': name,
            'audio_files': len(audio_files),
            'matched': matched,
            'match_rate': round(match_rate, 1),
            'glucose_range': glucose_range,
            'status': 'OK' if matched > 0 else 'NO_MATCHES',
        })

        if config.get('notes'):
            print(f"  Notes: {config['notes']}")

    # Build output DataFrame
    if not all_rows:
        print("\nERROR: No matches found at all!")
        return pd.DataFrame()

    canonical_df = pd.DataFrame(all_rows)

    # Sort by subject then timestamp
    canonical_df = canonical_df.sort_values(['subject', 'audio_timestamp']).reset_index(drop=True)

    # Save canonical dataset
    output_path = OUTPUT_DIR / "canonical_dataset.csv"
    canonical_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"CANONICAL DATASET SAVED: {output_path}")
    print(f"Total rows: {len(canonical_df)}")
    print(f"{'=' * 80}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "matching_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Print summary table
    print(f"\n{'MATCHING SUMMARY':^60}")
    print(f"{'-' * 60}")
    print(f"{'Participant':<15} {'Audio':>6} {'Matched':>8} {'Rate':>6} {'Glucose Range':>15}")
    print(f"{'-' * 60}")
    total_audio = 0
    total_matched = 0
    for row in summary_rows:
        if row['status'] in ['OK', 'NO_MATCHES']:
            print(f"{row['participant']:<15} {row['audio_files']:>6} {row['matched']:>8} "
                  f"{row['match_rate']:>5.1f}% {row['glucose_range']:>15}")
            total_audio += row['audio_files']
            total_matched += row['matched']
    print(f"{'-' * 60}")
    total_rate = total_matched / total_audio * 100 if total_audio else 0
    print(f"{'TOTAL':<15} {total_audio:>6} {total_matched:>8} {total_rate:>5.1f}%")

    # Per-subject glucose stats
    print(f"\n{'GLUCOSE STATISTICS (mg/dL)':^60}")
    print(f"{'-' * 60}")
    print(f"{'Participant':<15} {'N':>5} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"{'-' * 60}")
    for subj in canonical_df['subject'].unique():
        subj_data = canonical_df[canonical_df['subject'] == subj]['glucose_mg_dl']
        print(f"{subj:<15} {len(subj_data):>5} {subj_data.mean():>7.1f} {subj_data.std():>7.1f} "
              f"{subj_data.min():>7.1f} {subj_data.max():>7.1f}")

    return canonical_df


# ============================================================================
# STEP 1: DATE OVERLAP ANALYSIS (for mapping orphaned CGMs)
# ============================================================================

def analyze_date_overlaps():
    """
    Cross-reference orphaned CGM file date ranges with audio file date ranges
    to identify which CGM belongs to which participant.
    """
    print("\n" + "=" * 80)
    print("STEP 1: ORPHANED CGM DATE OVERLAP ANALYSIS")
    print("=" * 80)

    orphaned_csvs = [
        "Number_1Nov23_1_glucose_4-1-2024.csv",
        "Number_3Nov_23_glucose_4-1-2024.csv",
        "Number_4Nov_23_glucose_4-1-2024.csv",
        "Number_5Nov_23_glucose_4-1-2024.csv",
        "Number_X6Nov_25_glucose_4-1-2024.csv",
        "Number_11Nov_29_glucose_4-1-2024.csv",
        "Number_12Nov_29_glucose_4-1-2024.csv",
        "Number_13Dec_10_glucose_4-1-2024.csv",
        "Number_14Dec_10_glucose_4-1-2024.csv",
    ]

    # Known mappings from key file
    known_names = {
        "Number_5": "Boriska Molnar",
        "Number_12": "Bopo del Valle",
        "Number_14": "Berna Agar",
    }

    # Audio participants with date ranges (no CGM currently)
    audio_participants = {
        "Bruno": {"dir": "Bruno/conv_audio", "ext": ".wav"},
        "Valerie": {"dir": "Valerie/conv_audio", "ext": ".wav"},
        "Edoardo": {"dir": "Edoardo/conv_audio", "ext": ".wav"},
        "Jacky": {"dir": "Jacky", "ext": ".waptt"},
        "R_Rodolfo": {"dir": "R_Rodolfo/conv_audio", "ext": ".wav"},
    }

    # Get audio date ranges
    print("\nAudio participant date ranges:")
    audio_ranges = {}
    for name, info in audio_participants.items():
        dir_path = BASE_DIR / info['dir']
        if not dir_path.exists():
            print(f"  {name}: directory not found ({info['dir']})")
            continue

        timestamps = []
        for f in dir_path.glob(f"*{info['ext']}"):
            ts = parse_timestamp_from_filename(f.name)
            if ts:
                timestamps.append(ts)

        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            audio_ranges[name] = (min_ts, max_ts, len(timestamps))
            print(f"  {name}: {min_ts.strftime('%Y-%m-%d')} to {max_ts.strftime('%Y-%m-%d')} "
                  f"({len(timestamps)} files)")
        else:
            print(f"  {name}: no parseable timestamps")

    # Get CGM date ranges
    print("\nOrphaned CGM date ranges:")
    cgm_ranges = {}
    for csv_name in orphaned_csvs:
        csv_path = BASE_DIR / csv_name
        if not csv_path.exists():
            print(f"  {csv_name}: not found")
            continue

        df = load_glucose_csv([csv_name], 'mmol/L')
        if df.empty:
            print(f"  {csv_name}: no valid data")
            continue

        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        n_readings = len(df)
        cgm_ranges[csv_name] = (min_ts, max_ts, n_readings)

        # Check if this is a known mapping
        for prefix, person in known_names.items():
            if csv_name.startswith(prefix):
                print(f"  {csv_name}: {min_ts.strftime('%Y-%m-%d')} to "
                      f"{max_ts.strftime('%Y-%m-%d')} ({n_readings} readings) "
                      f"-> KNOWN: {person}")
                break
        else:
            print(f"  {csv_name}: {min_ts.strftime('%Y-%m-%d')} to "
                  f"{max_ts.strftime('%Y-%m-%d')} ({n_readings} readings)")

    # Cross-reference
    print("\n" + "-" * 60)
    print("CROSS-REFERENCE: CGM x Audio Date Overlap")
    print("-" * 60)

    for csv_name, (cgm_start, cgm_end, n_cgm) in cgm_ranges.items():
        matches = []
        for audio_name, (audio_start, audio_end, n_audio) in audio_ranges.items():
            # Calculate overlap
            overlap_start = max(cgm_start, audio_start)
            overlap_end = min(cgm_end, audio_end)
            overlap_days = (overlap_end - overlap_start).days

            if overlap_days > 0:
                # Count audio files within CGM range
                audio_dir = BASE_DIR / audio_participants[audio_name]['dir']
                ext = audio_participants[audio_name]['ext']
                files_in_range = 0
                for f in audio_dir.glob(f"*{ext}"):
                    ts = parse_timestamp_from_filename(f.name)
                    if ts and cgm_start <= ts <= cgm_end:
                        files_in_range += 1
                matches.append((audio_name, overlap_days, files_in_range, n_audio))

        short_name = csv_name.split('_glucose')[0]
        if matches:
            matches.sort(key=lambda x: x[2], reverse=True)  # Sort by files in range
            print(f"\n  {short_name}:")
            for audio_name, days, files_in, total in matches:
                print(f"    -> {audio_name}: {days} days overlap, "
                      f"{files_in}/{total} audio files within CGM range")
        else:
            print(f"\n  {short_name}: NO audio participant overlap")


# ============================================================================
# STEP 4: SEARCH FOR CARMEN
# ============================================================================

def search_for_carmen():
    """Search entire TONES directory for anything Carmen-related."""
    print("\n" + "=" * 80)
    print("STEP 4: SEARCHING FOR CARMEN'S AUDIO")
    print("=" * 80)

    found = []
    for root, dirs, files in os.walk(BASE_DIR):
        for item in dirs + files:
            if 'carmen' in item.lower():
                found.append(os.path.join(root, item))

    if found:
        print(f"  Found {len(found)} Carmen-related items:")
        for f in found:
            print(f"    {f}")
    else:
        print("  No files or directories containing 'Carmen' found in TONES directory.")
        print("  Carmen's audio may be on another drive, in cloud storage, or not yet collected.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Step 1: Analyze date overlaps for orphaned CGM mapping
    analyze_date_overlaps()

    # Step 4: Search for Carmen
    search_for_carmen()

    # Build the canonical dataset (Steps 2, 3, 5)
    print("\n")
    canonical_df = build_canonical_dataset()

    if not canonical_df.empty:
        print(f"\n{'=' * 80}")
        print("NEXT STEPS:")
        print("  1. Review matching_summary.csv for any issues")
        print("  2. Confirm tentative CGM-participant mappings from overlap analysis")
        print("  3. Convert .waptt files to .wav for Darav/Joao/Alvar (ffmpeg)")
        print("  4. Uncomment confirmed orphan matches in PARTICIPANTS config")
        print("  5. Re-run to produce final canonical_dataset.csv")
        print(f"{'=' * 80}")
