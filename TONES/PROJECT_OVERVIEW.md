# Voice-Based Glucose Estimation Project

## Overview

This project develops a machine learning pipeline to estimate blood glucose levels from WhatsApp voice messages, using CGM (Continuous Glucose Monitoring) data as ground truth labels.

## Project Structure

```
C:\Users\whgeb\OneDrive\TONES\
│
├── 📊 DATA DIRECTORIES
│   ├── Wolf/                     # 947 WAV files, mg/dL glucose
│   ├── Sybille/                  # 546 WAV files, mg/dL glucose
│   ├── Margarita/                # 108 WAV files, mmol/L glucose
│   ├── Anja/                     # 91 WAV files, mg/dL glucose (4 CSV files)
│   ├── Vicky/                    # 79 WAV files, mmol/L glucose
│   ├── Steffen_Haeseli/          # 39 WAV files, mmol/L glucose
│   └── Lara/                     # 32 WAV files, mmol/L glucose
│
├── 📁 MODEL OUTPUT DIRECTORIES
│   ├── combined_model_output/    # Combined HuBERT+MFCC model results
│   ├── hubert_models/            # HuBERT-only model checkpoints
│   ├── documentation_v5/         # Previous analysis reports
│   └── documentation_v6/         # v6 analysis reports
│
├── 🐍 MAIN SCRIPTS
│   ├── combined_hubert_mfcc_model.py   # ⭐ MAIN: Combined HuBERT + MFCC model
│   ├── hubert_glucose_model.py         # HuBERT transfer learning model
│   ├── comprehensive_analysis_v6.py    # MFCC analysis with time offsets
│   ├── comprehensive_analysis_v7_fast.py # VAD comparison experiments
│   └── comprehensive_analysis_v5.py    # Data augmentation experiments
│
├── 📄 DOCUMENTATION
│   ├── PROJECT_OVERVIEW.md       # This file
│   └── combined_model_output/report.html  # Generated HTML report
│
└── 📁 AUXILIARY
    ├── voice_glucose_pipeline.py # Original pipeline (deprecated)
    └── *.py                      # Other experimental scripts
```

## Key Files

### 1. `combined_hubert_mfcc_model.py` ⭐ MAIN MODEL

The primary production model combining deep learning and traditional features:

**Features:**
- **HuBERT Features** (2304 dimensions): Pre-trained transformer embeddings
- **MFCC Features** (~200 dimensions): MFCCs, deltas, mel-spectrogram, spectral features, pitch

**Usage:**
```python
from combined_hubert_mfcc_model import load_model, main

# Train the model
main(max_per_participant=None)  # Process all audio

# Load trained model
model = load_model('combined_model_output/combined_model.pkl')
```

### 2. `hubert_glucose_model.py` - HuBERT Transfer Learning

Standalone HuBERT model with few-shot calibration for personalization:

**Key Classes:**
- `HuBERTFeatureExtractor`: Extract 768×3=2304 dim features
- `FewShotCalibrator`: Personalization with 5-20 calibration samples
- `GlucoseAPI`: Production-ready API interface

**Usage:**
```python
from hubert_glucose_model import GlucoseAPI

api = GlucoseAPI(model_path='hubert_models/hubert_glucose_model.pkl')

# Predict
result = api.predict('voice.wav')
print(f"Glucose: {result['glucose_mgdl']:.1f} mg/dL")

# Calibrate (when user provides actual glucose reading)
api.calibrate('voice.wav', actual_glucose=120, user_id='user123')

# Personalized prediction
result = api.predict('voice.wav', user_id='user123')
```

### 3. `comprehensive_analysis_v6.py` - MFCC Analysis

Traditional MFCC-based analysis with:
- Time offset optimization (-30 to +30 minutes)
- Per-participant personalized models
- Clarke Error Grid analysis
- Feature importance visualization

**Best Results (from v6):**
| Participant | MAE (mg/dL) | Correlation | Optimal Offset |
|-------------|-------------|-------------|----------------|
| Wolf        | 8.37        | 0.385       | +15 min        |
| Anja        | 8.75        | 0.776       | 0 min          |
| Margarita   | 9.28        | -0.236      | +20 min        |

### 4. `comprehensive_analysis_v7_fast.py` - Experiments

Experimental comparisons:
- **VAD Analysis**: Found that removing pauses slightly hurts performance
- **Classification vs Regression**: 5-class quintile classification only achieved 27% accuracy
- **Personalization Benefit**: ~1.86 mg/dL improvement over population model

## Model Approaches

### Approach A: Personalized Models
- One model per person
- Leave-One-Out cross-validation
- Best accuracy for individuals with enough data

### Approach B: Population Model
- Single model for all participants
- Leave-One-Person-Out cross-validation
- Tests generalization to new users

### Approach C: Transfer Learning + Personalization
- Start with population model
- Fine-tune with few-shot calibration (5-20 samples)
- Best of both worlds

## Technical Details

### Feature Extraction

**HuBERT Features:**
- Model: `facebook/hubert-base-ls960`
- Input: 16kHz mono audio
- Output: 768-dimensional embeddings per frame
- Aggregation: mean, std, max → 2304 dimensions

**MFCC Features:**
- MFCCs (20 coefficients) + deltas + delta-deltas
- Mel-spectrogram statistics (40 bands)
- Spectral centroid, bandwidth, rolloff, contrast
- Zero crossing rate, RMS energy
- Pitch (F0) via PYIN

### Time Alignment

CGM sensors have 5-15 minute lag vs actual blood glucose. Optimal offsets vary per person:
- Wolf: +15 minutes
- Anja: 0 minutes
- Margarita: +20 minutes

### Glucose Unit Conversion

```python
mg_dL = mmol_L * 18.0182
```

## Evaluation Metrics

1. **MAE** (Mean Absolute Error): Primary metric, in mg/dL
2. **Correlation (r)**: Pearson correlation coefficient
3. **Clarke Error Grid**: Clinical accuracy zones
   - Zone A: Clinically accurate
   - Zone B: Benign errors
   - Zone A+B ≥ 99%: Clinical acceptability threshold

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install librosa torch torchaudio transformers
```

## Running the Pipeline

```bash
cd "C:/Users/whgeb/OneDrive/TONES"

# Run combined model (full training)
python combined_hubert_mfcc_model.py

# Run HuBERT-only model
python hubert_glucose_model.py

# Run MFCC analysis
python comprehensive_analysis_v6.py
```

## Output Files

After running `combined_hubert_mfcc_model.py`:

```
combined_model_output/
├── combined_model.pkl          # Trained model (pickle)
├── report.html                 # HTML report with visualizations
├── clarke_personalized.png     # Clarke Error Grid (personalized)
├── clarke_population.png       # Clarke Error Grid (population)
└── per_participant_scatter.png # Per-participant results
```

## Research Insights

1. **Voice-glucose signal is subtle**: Aggressive data augmentation hurts performance
2. **Personalization matters**: ~2 mg/dL improvement over population model
3. **Time alignment is critical**: Optimal offset varies per person (0 to +20 min)
4. **Deep learning helps**: HuBERT captures patterns not in hand-crafted features
5. **Feature fusion works**: Combining HuBERT + MFCC provides complementary information
6. **Pauses contain information**: VAD that removes pauses slightly degrades results

## Key Scripts (Current)

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Main pipeline: MFCC + VQ + temporal, personalized + population models |
| `phoneme_residual_pipeline.py` | Phoneme-level residuals, hybrid models (validated best approach) |
| `offset_window_analysis.py` | Optimal CGM-to-voice offset per participant |
| `offset_by_feature_analysis.py` | Optimal offset per feature (for feature selection) |
| `hyperparameter_sweep.py` | Feature and model configuration sweeps |

## Next Steps

See **NEXT_STEPS_PLAN.md** (Corrected v2, February 2026) for the full roadmap:
- **Phase 1:** Phoneme residuals + standardized phrase (MFA, per-phoneme baselines)
- **Phase 2:** Calibration budget analysis + voice-profile clustering
- **Phase 3:** Tiered onboarding + hydration indicator (Day 1 value)
- **Phase 4:** Privacy-preserving pipeline (features only, no raw voice)
- **Phases 5–8:** API + app, production phoneme residuals, BP track, model improvement loop

## Future Improvements

1. **More data**: Collect from more participants
2. **End-to-end fine-tuning**: Train HuBERT on glucose prediction directly
3. **Temporal modeling**: Use sequence models (LSTM, Transformer) for multiple recordings
4. **Meal/activity context**: Include time of day, recent meals, activity level
5. **Real-time API**: Deploy as web service for mobile app integration

## Contact

Project Directory: `C:\Users\whgeb\OneDrive\TONES`
