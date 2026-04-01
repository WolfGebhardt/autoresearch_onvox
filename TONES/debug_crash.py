#!/usr/bin/env python3
"""Diagnose which audio files crash during feature extraction."""
import sys
import gc
import pandas as pd
import numpy as np

# Load canonical dataset
df = pd.read_csv('output/canonical_dataset.csv')
print(f"Total: {len(df)} samples", flush=True)

# Try loading each audio file around the crash point (250+)
import librosa

from tones.config import load_config
from tones.features.mfcc import create_extractor_from_config
from tones.features.cache import FeatureCache
from pathlib import Path

cfg = load_config()
extractor = create_extractor_from_config(cfg)
feat_cfg = cfg.get("features", {})
cache_dir = feat_cfg.get("cache_dir", ".cache/features")
base_cache_dir = str(Path(cfg["base_dir"]) / cache_dir)
cache = FeatureCache(cache_dir=base_cache_dir, enabled=True)
mfcc_id = f"mfcc_n{extractor.n_mfcc}_sr{extractor.sr}"

# Count cached vs uncached
cached = 0
uncached_indices = []
for i in range(len(df)):
    path = df.iloc[i]["audio_path"]
    if cache.get(path, mfcc_id) is not None:
        cached += 1
    else:
        uncached_indices.append(i)

print(f"Cached: {cached}, Uncached: {len(uncached_indices)}", flush=True)
print(f"First 10 uncached indices: {uncached_indices[:10]}", flush=True)

# Try extracting each uncached file individually
for idx in uncached_indices:
    path = df.iloc[idx]["audio_path"]
    subject = df.iloc[idx]["subject"]
    print(f"  [{idx}] {subject}: {path} ...", end="", flush=True)
    try:
        features = extractor.extract_from_file(path)
        if features is not None:
            cache.put(path, mfcc_id, features)
            print(f" OK (shape={features.shape})", flush=True)
        else:
            print(" FAILED (returned None)", flush=True)
    except Exception as e:
        print(f" ERROR: {e}", flush=True)
    gc.collect()

print("Done!", flush=True)
