"""
ONVOX Data Loaders — Load matched voice-glucose pairs for research.
====================================================================

Primary path: loads from pre-matched CSVs in data/processed/matched_data_v2_strict/
Fallback: raw audio matching from project directory per config.yaml participants.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Project root for finding pre-matched CSVs
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
MATCHED_DIR = PROJECT_ROOT / "data" / "processed" / "matched_data_v2_strict"

# Participant name -> matched CSV name mapping (handles naming differences)
NAME_MAP = {
    "Steffen": "Steffen_Haeseli",
    "R_Rodolfo": "Bodo_del_Valle",
}


def load_participant_data(
    name: str,
    pcfg: dict,
    base_dir: Path,
    matching_cfg: dict,
) -> pd.DataFrame:
    """Load matched voice-glucose data for one participant.

    First tries pre-matched CSV from data/processed/matched_data_v2_strict/.
    Falls back to raw audio matching from config if CSV not found.

    Returns DataFrame with columns: audio_path, audio_timestamp, glucose_mg_dl
    """
    # Try pre-matched CSV first
    csv_name = NAME_MAP.get(name, name)
    csv_path = MATCHED_DIR / f"matched_{csv_name}.csv"

    if csv_path.exists():
        logger.info("  %s: Loading pre-matched CSV: %s", name, csv_path.name)
        df = pd.read_csv(csv_path)

        # Normalize column names
        if "glucose_mg_dl" not in df.columns and "glucose_mg_dL" in df.columns:
            df = df.rename(columns={"glucose_mg_dL": "glucose_mg_dl"})

        # Ensure required columns exist
        required = ["audio_path", "glucose_mg_dl"]
        if not all(c in df.columns for c in required):
            logger.warning("  %s: CSV missing required columns: %s", name, required)
            return pd.DataFrame()

        # Add audio_timestamp if missing
        if "audio_timestamp" not in df.columns:
            if "voice_timestamp" in df.columns:
                df["audio_timestamp"] = df["voice_timestamp"]
            else:
                df["audio_timestamp"] = ""

        # Verify audio paths exist
        valid_mask = df["audio_path"].apply(lambda p: Path(str(p)).exists())
        n_missing = (~valid_mask).sum()
        if n_missing > 0:
            logger.info("  %s: %d/%d audio paths missing, keeping those that exist", name, n_missing, len(df))
        df = df[valid_mask].reset_index(drop=True)

        return df

    # Fallback: raw audio matching
    logger.info("  %s: No pre-matched CSV, attempting raw audio matching", name)
    return _match_raw_audio(name, pcfg, base_dir, matching_cfg)


def _match_raw_audio(
    name: str,
    pcfg: dict,
    base_dir: Path,
    matching_cfg: dict,
) -> pd.DataFrame:
    """Match raw audio files to glucose readings by timestamp.

    Uses filenames containing glucose values (e.g., "Voice 260303_114441 113 new sensor.m4a")
    or timestamp-based matching against CGM CSV files.
    """
    audio_paths = collect_audio_files(
        pcfg.get("audio_dirs", []),
        base_dir,
        pcfg.get("audio_ext", [".wav", ".opus"]),
    )

    if not audio_paths:
        logger.warning("  %s: No audio files found", name)
        return pd.DataFrame()

    # Load glucose CSVs
    glucose_csvs = pcfg.get("glucose_csv", [])
    glucose_unit = pcfg.get("glucose_unit", "mg/dL")
    all_glucose = []

    for csv_rel in glucose_csvs:
        csv_path = base_dir / csv_rel
        if not csv_path.exists():
            logger.warning("  %s: Glucose CSV not found: %s", name, csv_path)
            continue
        try:
            gdf = pd.read_csv(csv_path)
            all_glucose.append(gdf)
        except Exception as e:
            logger.warning("  %s: Failed to read glucose CSV: %s", name, e)

    if not all_glucose:
        logger.warning("  %s: No glucose data loaded", name)
        return pd.DataFrame()

    # For now, return empty if no pre-matched CSV exists
    # (full raw matching logic is in tools/rematch_strict_v2.py)
    logger.info("  %s: Raw audio matching not fully implemented, "
                "use rematch_strict_v2.py to create matched CSVs first", name)
    return pd.DataFrame()


def collect_audio_files(
    audio_dirs: List[str],
    base_dir: Path,
    extensions: List[str],
) -> List[Path]:
    """Glob for audio files in specified directories.

    Args:
        audio_dirs: Relative directory paths within base_dir
        base_dir: Base directory (project root)
        extensions: File extensions to match (e.g., [".wav", ".opus", ".m4a"])

    Returns:
        List of absolute paths to audio files, sorted by name.
    """
    found = []
    for rel_dir in audio_dirs:
        dir_path = base_dir / rel_dir
        if not dir_path.exists():
            continue
        for ext in extensions:
            # Handle compound extensions like .waptt
            pattern = f"*{ext}"
            found.extend(dir_path.glob(pattern))
            # Also check for .waptt.opus compound extensions
            if ext == ".opus":
                found.extend(dir_path.glob("*.waptt.opus"))

    # Deduplicate and sort
    unique = sorted(set(found), key=lambda p: p.name)
    return unique
