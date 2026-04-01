#!/usr/bin/env python3
"""Phase 1: Extract features for all MFCC configs and save to disk."""
import sys, gc, json, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from pathlib import Path
from tones.config import load_config, get_base_dir
from tones.data.loaders import load_participant_data
from tones.features.mfcc import MFCCExtractor
import librosa

print("PHASE 1: Feature Extraction", flush=True)
cfg = load_config(None)
base_dir = get_base_dir(cfg)
out_dir = Path("output/sweep/features")
out_dir.mkdir(parents=True, exist_ok=True)

# Load matched data (paths + glucose)
participants = cfg.get("participants", {})
matching_cfg = cfg.get("matching", {})
pdata = {}
for name, pcfg in participants.items():
    if not pcfg.get("glucose_csv"):
        continue
    df = load_participant_data(name, pcfg, base_dir, matching_cfg)
    if len(df) >= 20:
        pdata[name] = df
        print(f"  {name}: {len(df)} matched samples", flush=True)
print(f"Total: {len(pdata)} participants\n", flush=True)

# MFCC configs to extract
CONFIGS = [
    ("8_base",  8,  False, False),
    ("13_base", 13, False, False),
    ("20_base", 20, False, False),
    ("30_base", 30, False, False),
    ("40_base", 40, False, False),
    ("13_spec", 13, True,  False),
    ("20_spec", 20, True,  False),
]

# Extract for each config
for cfg_name, n_mfcc, inc_spec, inc_pitch in CONFIGS:
    print(f"Extracting {cfg_name} (n_mfcc={n_mfcc}, spectral={inc_spec})...", flush=True)
    ext = MFCCExtractor(sr=16000, n_mfcc=n_mfcc, fmin=50, fmax=8000,
                        include_spectral=inc_spec, include_pitch=inc_pitch, include_mel=False)
    
    for pname, df in pdata.items():
        feats, gluc, ts_list = [], [], []
        for _, row in df.iterrows():
            try:
                y, _ = librosa.load(str(row["audio_path"]), sr=16000, mono=True)
                if len(y) < 8000:
                    continue
                f = ext.extract_from_array(y)
                if f is not None:
                    feats.append(f)
                    gluc.append(row["glucose_mg_dl"])
                    ts_list.append(str(row["audio_timestamp"]))
                del y
            except Exception:
                continue
        
        if len(feats) >= 20:
            X = np.array(feats)
            np.save(out_dir / f"{cfg_name}_{pname}_X.npy", X)
            np.save(out_dir / f"{cfg_name}_{pname}_y.npy", np.array(gluc))
            with open(out_dir / f"{cfg_name}_{pname}_ts.json", "w") as fp:
                json.dump(ts_list, fp)
            print(f"  {pname}: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
        else:
            print(f"  {pname}: only {len(feats)} valid samples, skipping", flush=True)
        
        del feats
        gc.collect()
    
    print(f"  Done: {cfg_name}", flush=True)
    gc.collect()

print("\nPHASE 1 COMPLETE. Features saved to output/sweep/features/", flush=True)
