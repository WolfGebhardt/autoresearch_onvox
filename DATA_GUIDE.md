# ONVOX AutoResearch — Data Handling Guide

How to set up data, run the autonomous research loop, and connect to production.

---

## 1. Data Sources Overview

The system uses **two data tracks** that feed into the same evaluation pipeline:

| Track | Source | Format | Location |
|-------|--------|--------|----------|
| **A — Research audio** | Local CGM CSVs + raw audio files | WAV/Opus per participant | `base_dir` in `config.yaml` |
| **B — Production features** | Supabase `calibrations` table | Pre-extracted feature vectors (10–119 dim) | `data/synced/features/*.npz` |

Track A is always available (offline research data). Track B is optional and requires Supabase credentials.

---

## 2. Track A: Research Audio Data

### 2.1 Directory Structure

The research data lives in a local directory (the `base_dir` from `config.yaml`). Each participant has their own folder:

```
ONVOX_DATA/                          ← base_dir in config.yaml
├── Wolf/
│   ├── all glucose/
│   │   └── HenningGebhard_glucose_19-11-2023.csv    ← CGM export (CSV)
│   ├── all wav audio/
│   │   ├── Voice 260303_114441 113 new sensor.wav    ← voice recording
│   │   └── ...
│   └── all opus audio/
│       └── ...
├── Anja/
│   ├── glucose 21nov 2023/
│   │   ├── AnjaZhao_glucose_6-11-2023.csv
│   │   └── ...
│   └── conv_audio/
│       └── ...
├── Margarita/
│   └── ...
└── ... (11 participants total)
```

### 2.2 CGM CSV Format

CGM exports come from FreeStyle Libre sensors. Two formats exist:

**English format** (mg/dL):
```csv
Device,Serial Number,Device Timestamp,Record Type,Historic Glucose mg/dL,...
FreeStyle LibreLink,ABC123,11-06-2023 08:15,0,95,...
```

**German format** (mg/dL, different column names):
```csv
Gerät,Seriennummer,Gerätezeitstempel,Aufzeichnungstyp,Glukosewert-Verlauf mg/dL,...
```

**mmol/L format** (some participants):
The system auto-converts mmol/L to mg/dL using the standard factor (×18.0182).

Each participant's `glucose_unit` in `config.yaml` specifies which unit their CSV uses.

### 2.3 Audio File Formats

Accepted extensions: `.wav`, `.opus`, `.waptt` (WhatsApp voice note format)

Audio is resampled to **16 kHz mono** during feature extraction. Files shorter than 0.5 seconds are skipped.

Filenames often contain timestamps and sometimes glucose values embedded in the name (legacy format from early data collection). The matching pipeline ignores filename glucose values and matches by timestamp.

### 2.4 Voice-Glucose Matching

Matching pairs each audio file with the nearest CGM reading. Configured in `config.yaml`:

```yaml
matching:
  window_minutes: 30       # Max time gap between audio and CGM reading
  use_interpolation: true  # Linearly interpolate between bracketing CGM points
  min_audio_duration_sec: 0.5
```

**Pre-matched CSVs** (preferred): The matching has already been run and results saved to:
```
data/processed/matched_data_v2_strict/matched_{Participant}.csv
```

Each CSV contains:
```csv
audio_path,audio_timestamp,glucose_mg_dl,cgm_timestamp,time_diff_min
/full/path/to/audio.wav,2023-11-06T08:15:00,95.0,2023-11-06T08:12:00,3.0
```

The loader (`research/data/loaders.py`) tries these CSVs first, falling back to raw matching only if they don't exist.

### 2.5 Participant Configuration

`config.yaml` defines each participant:

```yaml
participants:
  Wolf:
    glucose_csv:
      - "Wolf/all glucose/HenningGebhard_glucose_19-11-2023.csv"
    audio_dirs:
      - "Wolf/all wav audio"
      - "Wolf/all opus audio"
    audio_ext: [".wav", ".opus", ".waptt"]
    glucose_unit: "mg/dL"
    optimal_offset: 30        # CGM lag offset in minutes (per participant)
```

Key fields:
- **`glucose_csv`**: Paths relative to `base_dir`. Can be multiple CSVs (different date ranges).
- **`audio_dirs`**: Directories to scan for audio. Multiple dirs handled (some participants have split folders).
- **`glucose_unit`**: `"mg/dL"` or `"mmol/L"` — auto-converted during loading.
- **`optimal_offset`**: Per-participant CGM lag in minutes. The autonomous loop also searches lag as a hyperparameter (`cgm_lag_min`).

### 2.6 Setting Up on a New Machine

1. Place participant data folders at a path of your choice.
2. Edit `config.yaml` → set `base_dir` to that path.
   - Or set the environment variable `AUTORESEARCH_BASE_DIR` (overrides config).
3. If you have pre-matched CSVs, place them in `data/processed/matched_data_v2_strict/`.
4. Run `python hyperparameter_sweep.py --quick` to verify loading works.

**Privacy**: Audio and CGM data are **never committed** to git. The `.gitignore` excludes data directories. Keep raw data local only.

---

## 3. Track B: Production Data (Supabase)

### 3.1 What It Is

Production data comes from the ONVOX app: real users recording voice samples and entering finger-prick glucose references through the app. This data is stored in Supabase.

### 3.2 Supabase Schema

The `calibrations` table:

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid | Primary key |
| `user_id` | uuid | Full 36-char UUID — **never use short prefixes** |
| `feature_vector` | jsonb | **Returns as dict** `{"0": 1.23, "1": 0.45, ...}` — handle `isinstance(x, dict)` BEFORE `isinstance(x, list)` |
| `reference_glucose` | float | Glucose in mg/dL — **NOT** `glucose_value` |
| `created_at` | timestamptz | When the calibration was recorded |
| `device_info` | jsonb | Device metadata (optional) |
| `audio_duration_s` | float | Recording duration (optional) |
| `recording_quality` | float | Quality score 0-1 (optional) |

**Critical trap**: Supabase JSON columns return Python dicts with string keys, not lists. Code must always check `isinstance(x, dict)` before `isinstance(x, list)`. This has caused 3 critical silent-failure bugs.

### 3.3 Syncing Production Data

```bash
# Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in .env
python -m onvox_bridge.supabase_syncer
python -m onvox_bridge.supabase_syncer --min-samples 10  # Skip users with few samples
```

This:
1. Fetches all calibrations from Supabase (paginated, 1000 rows/page)
2. Groups by `user_id`
3. Parses feature vectors (dict→list, filters metadata keys like `_mel_scale`)
4. Zero-pads to consistent dimension within each user
5. Writes per-user NPZ files to `data/synced/features/{user_id}_features.npz`
6. Writes a sync manifest to `data/synced/sync_manifest.json`

Each NPZ contains:
- `features`: numpy array `(n_samples, n_features)`
- `glucose`: numpy array `(n_samples,)`
- `timestamps`: array of ISO timestamp strings
- `dim`: the feature dimension

### 3.4 Feature Vector Dimensions

Production feature vectors come from the on-device SDK and may have different dimensions depending on SDK version:

| SDK Version | Dimensions | Notes |
|------------|-----------|-------|
| Pre-v2.6.0 | 10 | Edge features only |
| v2.6.0+ | 103 | Full canonical set |
| v2.8.0+ | 119 | Canonical + windowed + derived + DOW |

The syncer zero-pads shorter vectors. The autoresearch evaluation adapts to available dimensions.

### 3.5 Feature Metadata Keys

Some feature vectors contain metadata entries prefixed with `_`:
- `_mel_scale`: `"htk"` or `"slaney"` (mel filterbank type)
- `_migrated_to_db`: migration flag

These are **not features** — filter them with `k.startswith('_')` or iterate `range(n_features)`.

---

## 4. How the Autonomous Loop Uses Data

### 4.1 Data Loading

At startup, `main()` loads both tracks:

```
Track A: load_config() → load_all_audio() → participant_data dict
Track B: load_production_data() → production_data dict (optional)
```

### 4.2 Evaluation Pipeline

For each candidate configuration, `evaluate_one_dual()` runs:

1. **Track A evaluation**:
   - Apply CGM lag offset (`cgm_lag_min`) via temporal interpolation
   - Extract features with given config (n_mfcc, spectral, pitch, temporal, normalization)
   - Run personalized evaluation (per-participant LOO or k-fold CV)
   - Run population evaluation (LOSO — Leave-One-Subject-Out)
   - Run temporal evaluation (chronological train/test split)

2. **Track B evaluation** (if production data available):
   - Map feature_key to closest production feature subset
   - Run personalized evaluation on production users
   - Blend scores: 60% Track A + 40% Track B

### 4.3 Scoring Formula

```
selection_score = balance - pers_r_bonus + temporal_penalty + signal_gate_penalty + temp_r_penalty

where:
  balance             = 0.85 * pers_mae + 0.15 * pop_mae
  pers_r_bonus        = max(0, pers_r - 0.1) * 3.0
  temporal_penalty    = max(0, temp_mae - pers_mae)
  signal_gate_penalty = max(0, 0.30 - signal_gate_pass_rate) * 3.0
  temp_r_penalty      = max(0, 0.05 - temp_r) * 2.0
```

Lower is better. Population MAE is down-weighted to 15% because 22 research stages confirmed no population-level voice-glucose signal (r = -0.098).

### 4.4 Signal Gate

Per-participant signal detection gate (must pass ALL three):
- Pearson r > 0.3
- Improvement over baseline > 10%
- p-value < 0.05

The `signal_gate_pass_rate` is the fraction of participants passing all three criteria.

---

## 5. WhatsApp Audio Data

### 5.1 How It Arrives

Users send voice notes to the ONVOX WhatsApp number. The flow:

```
User voice note → Twilio → /api/whatsapp/webhook →
  Download audio → Extract features on server →
  Store in Supabase calibrations table
```

WhatsApp voice notes are Opus-encoded. The server:
1. Downloads the audio via Twilio media URL
2. Decodes Opus → PCM (16 kHz mono)
3. Extracts the full 119-dim feature vector
4. Stores in `calibrations.feature_vector` as JSONB

### 5.2 Using WhatsApp Data in Research

WhatsApp calibrations land in the same `calibrations` table as app calibrations. After running `supabase_syncer`, they appear in the per-user NPZ files alongside app-recorded samples.

No special handling is needed — the syncer treats all calibrations identically. The `device_info` column can distinguish source if needed.

### 5.3 Audio Format Notes

- WhatsApp Opus: variable bitrate, typically 16 kHz
- The `.waptt` extension is WhatsApp's internal format (renamed `.opus`)
- Research audio (Track A) includes `.waptt` files from early WhatsApp-based data collection

---

## 6. Production Bridge: Promotion Gate

When the autonomous loop finds a configuration that passes the signal gate on production data, it queues it for the ONVOX BackgroundTrainer:

```
autoresearch config → PromotionGate.translate_config() → BackgroundTrainer format
```

Translation maps:
- `feature_key` → `feature_subset` (e.g., `"mfcc+spectral+pitch"` → `"personal_10"`)
- `model_name` → `model_type` (e.g., `"Ridge"` → `"Ridge"`)
- Adds default alpha schedule (`"100/n"` for Ridge)

Queued configs are written to `output/promotion_queue.json` for manual review before deploying to the production trainer.

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `config.yaml` | Participant definitions, data paths, feature/model settings |
| `research/config.py` | Config loader (finds config.yaml, resolves paths) |
| `research/data/loaders.py` | Load matched CSVs or raw audio per participant |
| `research/features/mfcc.py` | MFCC extraction (librosa) |
| `research/features/voice_quality.py` | Jitter, shimmer, HNR, formants |
| `research/features/temporal.py` | Circadian, delta, time-since features |
| `research/features/normalize.py` | Z-score, rank normalization |
| `research/models/train.py` | Model registry, training, metrics |
| `research/evaluation/metrics.py` | Clarke Error Grid, MAE, MARD |
| `research/evaluation/temporal_cv.py` | Temporal cross-validation |
| `hyperparameter_sweep.py` | Systematic sweep across all configs |
| `autoresearch/autonomous_llm_loop.py` | LLM-driven autonomous experiment loop |
| `onvox_bridge/supabase_syncer.py` | Download Supabase calibrations → NPZ |
| `onvox_bridge/production_data_loader.py` | Load synced NPZ for evaluation |
| `onvox_bridge/promotion_gate.py` | Signal gate check + config translation |

---

## 8. Quick Start Checklist

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Set up data**: Place participant folders, set `base_dir` in `config.yaml`
3. **Verify data loads**: `python hyperparameter_sweep.py --quick`
4. **Start Ollama**: `ollama serve` (needs a model like `gemma4:26b` or `qwen2.5:7b`)
5. **Run the loop**: `python autoresearch/autonomous_llm_loop.py --optimizer-mode v2 --model gemma4:26b`
6. **Monitor**: Open the Streamlit dashboard → AutoResearch tab, or run the GUI monitor

### Optional: Production data
7. **Set Supabase credentials** in `.env`: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`
8. **Sync**: `python -m onvox_bridge.supabase_syncer`
9. **Re-run loop** — it auto-detects synced production data

---

## 9. Known Data Issues

- **Two user ID systems**: Supabase auth UUID vs localStorage anonymous ID. The mapping between them is maintained manually.
- **Mixed mel scale**: Calibrations before SDK v2.6.0 used HTK mel scale; v2.6.0+ uses Slaney. Flagged with `_mel_scale` metadata key. The adaptive model (tau=8) handles the transition naturally.
- **Dimension mismatches**: Server GP is 10-dim, SDK GP may be 26-dim. The syncer zero-pads. Evaluation code must handle variable dimensions defensively.
- **`reference_glucose` not `glucose_value`**: The Supabase column is `reference_glucose`. Using `glucose_value` returns NULL silently.
