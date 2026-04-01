# TONES — Voice-Based Glucose Estimation

Non-invasive blood glucose monitoring using voice biomarkers extracted from WhatsApp voice messages, paired with FreeStyle Libre CGM readings.

## Project Structure

```
TONES/
├── config.yaml              # Single source of truth for all settings
├── run_pipeline.py          # Main entry point
├── requirements.txt         # Python dependencies
│
├── tones/                   # Core package
│   ├── config.py            # Configuration loader
│   ├── data/
│   │   └── loaders.py       # Glucose CSV loading, timestamp parsing, matching
│   ├── features/
│   │   ├── mfcc.py          # MFCC feature extraction
│   │   ├── hubert.py        # HuBERT transfer learning features
│   │   └── cache.py         # Feature caching (joblib)
│   ├── models/
│   │   └── train.py         # Personalized + population model training
│   └── evaluation/
│       └── metrics.py       # Clarke Error Grid, visualization, reporting
│
├── archive/                 # Legacy scripts (kept for reference)
│   ├── comprehensive_analysis_v4.py
│   ├── comprehensive_analysis_v5.py
│   ├── comprehensive_analysis_v5_fast.py
│   ├── comprehensive_analysis_v6.py
│   ├── comprehensive_analysis_v7.py
│   ├── full_production_analysis.py
│   ├── combined_hubert_mfcc_model.py
│   ├── enhanced_analysis.py
│   └── ...
│
├── Wolf/                    # Participant data folders
├── Anja/
├── Sybille/
├── ...
│
└── output/                  # Generated results (gitignored)
    ├── canonical_dataset.csv
    ├── results_summary.json
    └── figures/
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

### 3. Common options

```bash
# Build dataset only (no model training)
python run_pipeline.py --data-only

# Run for specific participants
python run_pipeline.py --participants Wolf Lara

# Disable feature caching (re-extract everything)
python run_pipeline.py --no-cache

# Verbose logging
python run_pipeline.py -v

# Use a custom config file
python run_pipeline.py --config /path/to/config.yaml
```

## Configuration

All settings are in `config.yaml`:

- **Participant configs**: CSV paths, audio directories, glucose units, time offsets
- **Matching parameters**: Window size (default 30 min), interpolation on/off
- **Feature extraction**: n_mfcc, sample rate, frequency range
- **Model settings**: Which models to train, hyperparameters, CV strategy

To add a new participant, add an entry to the `participants` section of `config.yaml`.

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| 1. **Build Dataset** | Match audio files to CGM readings by timestamp |
| 2. **Extract Features** | MFCC + delta features (cached to disk) |
| 3. **Personalized Models** | Per-participant LOO/K-fold CV (SVR, BayesianRidge, etc.) |
| 4. **Population Model** | Leave-One-Person-Out CV across all participants |
| 5. **Output** | Scatter plots, Clarke Error Grid, JSON summary |

## Key Design Decisions

- **MFCC features** (not HuBERT) for the canonical pipeline. HuBERT's 2304 dimensions cause overfitting with <1000 samples (see `ALGORITHM_IMPROVEMENT_ROADMAP.md`).
- **Linear interpolation** for glucose matching instead of nearest-neighbor.
- **Feature caching** via `.cache/features/` — HuBERT extraction takes minutes per file; MFCC is fast but caching still helps during iteration.
- **RobustScaler** instead of StandardScaler for better outlier handling.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Main pipeline: MFCC + VQ + temporal, personalized + population |
| `phoneme_residual_pipeline.py` | Phoneme-level residuals, hybrid models |
| `offset_window_analysis.py` | Optimal CGM-to-voice offset per participant |
| `offset_by_feature_analysis.py` | Optimal offset per feature (for feature selection) |

## Next Steps

See **[NEXT_STEPS_PLAN.md](NEXT_STEPS_PLAN.md)** for the full roadmap:
- Offset-by-feature analysis (feature selection for app/API)
- Standardized phrase + clustering (pre-personalization)
- Tiered onboarding (classification → regression)
- Privacy-preserving API (features only, no raw voice)
- API (Twilio/WhatsApp) + app (Lovable) deployment

## Environment Variable

Set `TONES_BASE_DIR` to override the project root directory:

```bash
export TONES_BASE_DIR=/path/to/TONES
python run_pipeline.py
```

If not set, the pipeline auto-detects the root from the location of `config.yaml`.
