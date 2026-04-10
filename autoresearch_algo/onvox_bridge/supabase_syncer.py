#!/usr/bin/env python3
"""
Supabase Data Syncer — Download production calibrations to local NPZ files.
============================================================================

Downloads calibrations from Supabase, parses feature vectors (dict-before-list),
filters metadata keys, zero-pads to consistent dimensions, and writes per-user
NPZ files to data/synced/features/.

Usage:
    python -m tools.autoresearch.onvox_bridge.supabase_syncer
    python -m tools.autoresearch.onvox_bridge.supabase_syncer --min-samples 10
"""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SYNC_DIR = PROJECT_ROOT / "data" / "synced"
FEATURES_DIR = SYNC_DIR / "features"
CALIBRATIONS_DIR = SYNC_DIR / "calibrations"
MANIFEST_PATH = SYNC_DIR / "sync_manifest.json"


def _get_client():
    """Get Supabase client (reuses DataExporter pattern)."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY required in .env")

    from supabase import create_client
    return create_client(url, key)


def _fetch_table(client, table_name: str) -> list:
    """Fetch all rows from a table with pagination (1000-row pages)."""
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        resp = client.table(table_name).select('*').range(
            offset, offset + page_size - 1
        ).execute()
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    return all_rows


def parse_feature_vector(fv) -> Optional[List[float]]:
    """Parse a feature vector from Supabase JSON column.

    CRITICAL: dict check BEFORE list check — Supabase JSON columns return
    dict with string keys {"0": 1.23, "1": 0.45, ...}, not lists.
    """
    if fv is None:
        return None

    if isinstance(fv, dict):
        # Filter metadata keys (e.g., _mel_scale, _migrated_to_db)
        n = sum(1 for k in fv if k.isdigit())
        if n == 0:
            return None
        return [float(fv.get(str(i), 0)) for i in range(n)]

    if isinstance(fv, list):
        return [float(x) for x in fv]

    if isinstance(fv, str):
        try:
            parsed = json.loads(fv)
            return parse_feature_vector(parsed)
        except Exception:
            return None

    return None


def sync_calibrations(
    min_samples: int = 5,
    client=None,
) -> Dict[str, dict]:
    """Download calibrations from Supabase, parse, and write per-user NPZ files.

    Returns dict of {user_id: {n_samples, dim, npz_path, checksum}}.
    """
    if client is None:
        client = _get_client()

    logger.info("Fetching calibrations from Supabase...")
    calibrations = _fetch_table(client, "calibrations")
    logger.info("Fetched %d total calibration rows", len(calibrations))

    # Group by user_id
    by_user: Dict[str, List[dict]] = {}
    for cal in calibrations:
        uid = cal.get("user_id")
        if not uid:
            continue
        by_user.setdefault(uid, []).append(cal)

    logger.info("Found %d unique users", len(by_user))

    # Ensure output dirs exist
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATIONS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for user_id, cals in by_user.items():
        features = []
        glucose_values = []
        timestamps = []

        for cal in cals:
            fv = parse_feature_vector(cal.get("feature_vector"))
            gv = cal.get("reference_glucose")

            if fv is None or gv is None:
                continue

            try:
                gv_float = float(gv)
            except (TypeError, ValueError):
                continue

            features.append(fv)
            glucose_values.append(gv_float)
            timestamps.append(cal.get("created_at", ""))

        if len(features) < min_samples:
            logger.info("  %s: %d samples (< %d min), skipping", user_id[:8], len(features), min_samples)
            continue

        # Zero-pad to consistent dimension within user
        dims = [len(f) for f in features]
        max_dim = max(dims)
        if len(set(dims)) > 1:
            features = [
                f + [0.0] * (max_dim - len(f)) if len(f) < max_dim else f
                for f in features
            ]

        X = np.array(features, dtype=np.float32)
        y = np.array(glucose_values, dtype=np.float32)
        ts = np.array(timestamps, dtype=object)

        # Write NPZ
        npz_path = FEATURES_DIR / f"{user_id}_features.npz"
        np.savez_compressed(
            npz_path,
            features=X,
            glucose=y,
            timestamps=ts,
            dim=np.array([max_dim]),
        )

        # Also save raw calibrations as JSON for debugging
        cal_path = CALIBRATIONS_DIR / f"{user_id}_calibrations.json"
        cal_path.write_text(json.dumps(cals, indent=2, default=str), encoding="utf-8")

        # Compute checksum
        checksum = hashlib.md5(X.tobytes() + y.tobytes()).hexdigest()[:12]

        results[user_id] = {
            "n_samples": len(glucose_values),
            "dim": max_dim,
            "npz_path": str(npz_path),
            "checksum": checksum,
            "glucose_range": [float(y.min()), float(y.max())],
            "glucose_mean": float(y.mean()),
        }
        logger.info(
            "  %s: %d samples, %d-dim, glucose %.0f-%.0f (mean %.1f)",
            user_id[:8], len(glucose_values), max_dim,
            y.min(), y.max(), y.mean(),
        )

    # Write manifest
    manifest = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "total_users": len(results),
        "total_samples": sum(r["n_samples"] for r in results.values()),
        "users": results,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "Sync complete: %d users, %d total samples. Manifest: %s",
        len(results),
        manifest["total_samples"],
        MANIFEST_PATH,
    )

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sync Supabase calibrations to local NPZ")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum samples per user")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sync_calibrations(min_samples=args.min_samples)


if __name__ == "__main__":
    main()
