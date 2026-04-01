# Voice-Based Glucose Estimation: A Complete Methodology from Scratch

**ONVOX / TONES Project**
**Version 1.0 -- February 2026**

> If we had to build an effective and efficient way to estimate glucose levels from voice again using the data in this project, knowing everything we know now, this is exactly how we would do it.

---

## Table of Contents

0. [Lessons Learned: What Went Wrong and Why](#part-0-lessons-learned)
1. [Data Strategy](#part-1-data-strategy)
2. [Audio Processing Pipeline](#part-2-audio-processing-pipeline)
3. [Feature Engineering](#part-3-feature-engineering)
4. [Modeling Strategy](#part-4-modeling-strategy)
5. [Evaluation Framework](#part-5-evaluation-framework)
6. [Code Architecture](#part-6-code-architecture)
7. [Product Strategy](#part-7-product-strategy)
8. [Scaling Roadmap](#part-8-scaling-roadmap)
9. [Research Context and References](#appendix-research-context)

---

## Part 0: Lessons Learned

Before describing what to do, we must be honest about what went wrong. These are the mistakes that cost weeks of work and produced misleading results.

### 0.1 The Pipeline Proliferation Problem

The TONES codebase accumulated at least **7 independent analysis pipelines** that evolved organically: `voice_glucose_pipeline.py`, `comprehensive_analysis_v4.py` through `v7_fast.py`, `combined_hubert_mfcc_model.py`, `production_analysis.py`, and `full_production_analysis.py`. Each implemented slightly different timestamp matching, different participant subsets, different feature sets, and different evaluation protocols.

The critical symptom: `combined_hubert_mfcc_model.py` reported population MAE of **53.8 mg/dL** while `comprehensive_analysis_v6.py` reported **10.9 mg/dL** on overlapping data. A 5x discrepancy on the same underlying recordings is not a modeling difference -- it is a data pipeline bug.

**Root causes:**
- Different timestamp matching logic across scripts (some used +/-15 min, others +/-30)
- Different deduplication strategies
- Different participant inclusion lists (6, 7, or 11 participants)
- Wolf's glucose-from-filename bug (see 0.2)

**Lesson: There must be exactly ONE canonical dataset and ONE evaluation pipeline. Everything else is an experiment built on top of that single source of truth.**

### 0.2 The Glucose-from-Filename Bug

Wolf's audio files are named with self-reported glucose values as prefixes: `131_WhatsApp Audio 2023-11-06...`. The original pipeline (`voice_glucose_pipeline.py`) had `glucose_in_filename: True` for Wolf, parsing these prefixes as ground truth glucose values instead of matching to the CGM CSV.

Self-reported values taken at recording time are naturally better correlated with the recording than CGM readings taken up to 30 minutes away. This inflated Wolf's numbers from the honest **MAE 10.26, r=0.455** (CGM-matched) to the misleading **MAE 5.69, r=0.674** (self-reported).

**Lesson: Ground truth must always come from the CGM sensor, never from self-reported values embedded in filenames. The canonical dataset script (`build_canonical_dataset.py`) correctly ignores filename prefixes and matches exclusively by CGM timestamp.**

### 0.3 The HuBERT Dimensionality Trap

HuBERT-base (`facebook/hubert-base-ls960`) extracts 768-dimensional hidden states. With three aggregation statistics (mean, std, max across time frames), this produces **2,304 features per audio file**. When combined with ~200 MFCC features, the total feature space reaches ~2,500 dimensions.

With only ~500-750 usable samples, this creates a **p/n ratio of 3.3-5.0** -- deep into curse-of-dimensionality territory. The result was predictable: the MFCC-only pipeline with 97 features consistently outperformed the 2,500-feature combined pipeline on population generalization.

**Lesson: With fewer than 1,000 samples, feature dimensionality must be kept below ~150. If using SSL embeddings, aggressive PCA reduction to 20-30 components is mandatory before concatenation with other features.**

### 0.4 Classification Baseline Inflation

The project reported 79.9% accuracy on 5-class glucose classification (quintiles). This was compared to a 20% uniform random baseline, suggesting a 60-percentage-point improvement.

The problem: glucose values in this dataset cluster heavily around the euglycemic range (80-130 mg/dL). The correct majority-class baseline is **56.8%**, not 20%. The actual improvement is only ~23 percentage points -- modest, and likely driven by the model learning to predict the majority class more often.

**Lesson: Classification accuracy must always be compared against the majority-class baseline, never against uniform random chance. Better yet, report precision and recall per class, and use AUC-ROC which is threshold-independent.**

### 0.5 Suspicious Transfer Learning Results

The `hubert_glucose_model.py` reported transfer learning results with average r=0.957 and MAE=3.29 mg/dL across 7 participants. These numbers are suspiciously good -- better than most CGM sensors achieve against laboratory references. They have not been independently validated and likely suffer from data leakage (training and evaluation on overlapping or non-temporally-split data).

**Lesson: Any result that looks too good to be true probably is. All claims require validation against held-out temporal splits where training data precedes test data chronologically.**

### 0.6 What Actually Worked

From the canonical pipeline (`pipeline_results.json`), here are the honest numbers:

| Participant | N | Best MAE | Baseline MAE | Improvement | r | Signal? |
|---|---|---|---|---|---|---|
| Lara | 32 | 6.87 | 8.99 | +2.12 | 0.506 | YES |
| Wolf | 156 | 10.26 | 12.31 | +2.05 | 0.455 | YES |
| Sybille | 55 | 13.26 | 14.79 | +1.54 | 0.456 | YES |
| Vicky | 79 | 10.71 | 11.38 | +0.67 | 0.323 | MARGINAL |
| Joao | 89 | 10.36 | 10.92 | +0.56 | 0.176 | NO |
| Margarita | 107 | 9.71 | 9.72 | +0.01 | 0.040 | NO |
| Darav | 91 | 17.03 | 16.75 | -0.28 | 0.140 | NO |
| Anja | 84 | 10.22 | 10.07 | -0.15 | 0.018 | NO |
| Steffen | 38 | 8.59 | 8.36 | -0.23 | -1.000 | NO |
| R_Rodolfo | 7 | 7.20 | 7.02 | -0.18 | 0.313 | TOO FEW |
| Alvar | 14 | 14.02 | 13.33 | -0.69 | -0.312 | TOO FEW |
| **Population** | **752** | **14.29** | **12.76** | **-1.52** | **-0.124** | **NO** |

**Only 3 of 11 participants show genuine signal (r > 0.4 AND improvement > 1 mg/dL). The population model is worse than simply predicting the grand mean. This is the honest foundation to build from.**

---

## Part 1: Data Strategy

### 1.1 Maximize Existing Data Before Collecting More

The canonical dataset achieves 752 matched audio-glucose pairs from 11 participants. But there are significant untapped data assets:

**Wolf's unexploited audio:** The "all opus audio" directory contains 297 unique opus recordings (Nov 2-12, 2023), but the canonical pipeline only reads "all wav audio" (186 files, of which only 154 have unique WhatsApp hashes -- 32 are conversion duplicates). That means **143 unique Wolf recordings are currently unused** because they exist only in opus format and haven't been converted to WAV. The CGM covers Oct 31 - Nov 12, so all 297 should have glucose matches.

**Unintegrated participants:**
- **Christian_L** (Number_8): Has both audio (.waptt) and a CGM CSV -- should be integrated
- **Bruno** (~100 audio files), **Valerie** (~114), **Edoardo** (~46), **Jacky** (~31): Have audio but no confirmed CGM match. Cross-referencing their audio date ranges against the 9 orphaned CGM CSVs (Number_1, 3, 4, 5, X6, 11, 13, 14) via the `analyze_date_overlaps()` function may rescue some

**Realistic expectation:** Expanding Wolf from 156 to ~280+ samples and adding Christian_L could bring the total from 752 to ~900-1,000 samples. Resolving orphaned CGMs might add 50-200 more.

### 1.2 Data Collection Protocol for New Participants

The current dataset uses unstructured WhatsApp voice messages -- spontaneous speech of variable content, duration, and recording quality. The University of Bern's 2025 hypoglycemia study (Lehmann et al., published in *Diabetes Care*) achieved 90% detection accuracy using a **standardized multi-task protocol** with just 22 participants:

- **Sustained vowel** /a:/ for 5 seconds -- captures F0 stability, jitter, shimmer
- **Standard reading passage** (~30 seconds) -- captures prosodic features, fluency, rhythm
- **Spontaneous speech** (describe an image, answer a prompt) -- captures natural variation
- **Rapid syllable repetition** (/pa-ta-ka/) -- captures motor control precision

**Recommended protocol for new data collection:**

1. **Minimum target:** 50 participants x 50 recordings each = 2,500 matched samples
2. **Per recording session:** All four speech tasks (total ~90 seconds of audio)
3. **Timing:** Record within **5 minutes** of a CGM scan (not +/-30 minutes)
4. **CGM hardware:** OTC sensors -- Dexcom Stelo ($89/mo) or Abbott Lingo (~$49/sensor)
5. **Metadata per session:** meal timing (fasted/postprandial), physical activity level, subjective stress, ambient noise description, hydration status
6. **Duration:** 14 days per participant (one full sensor cycle)
7. **Diversity requirements:** Equal gender split, age range 18-65, include pre-diabetic/T2D participants (not just healthy normals)

### 1.3 Data Quality Gates

Before any audio-glucose pair enters the dataset:

| Gate | Criterion | Action if Fails |
|---|---|---|
| Audio duration | >= 2 seconds | Discard |
| Audio sample rate | 16 kHz (or resample if higher) | Resample |
| Estimated SNR | >= 10 dB | Flag as low quality |
| Glucose value | 40-400 mg/dL | Discard (likely sensor error) |
| Timestamp match | < 15 min preferred | Flag 15-30 min matches |
| File uniqueness | MD5 hash on first 8KB | Deduplicate |

For training set construction, actively ensure **representation across the full glucose range**. Most participants are euglycemic (80-130 mg/dL), which means the model gets very few examples of hypo- or hyperglycemia. If a participant has fewer than 3 samples below 80 or above 150 mg/dL, flag this as a training limitation for that individual.

### 1.4 The Minimum Viable Dataset

Current literature suggests these approximate thresholds:

- **Personalized model** (per-person calibration): 20-30 high-quality samples per participant
- **Population model** (cross-subject generalization): 50+ diverse participants with 2,500+ total samples
- **Clinical validation** (regulatory-grade): 200+ participants, IRB-approved, prospective design

We currently have enough for personalized models on ~6 participants (those with 30+ samples). We are far from population or clinical thresholds.

---

## Part 2: Audio Processing Pipeline

### 2.1 Format Normalization

The TONES project contains three audio formats:

- **`.wav`** -- PCM audio, ready to use (most participant directories)
- **`.waptt`** -- WhatsApp voice message containers (opus-encoded, librosa on Python 3.10 can load directly)
- **`.opus`** -- Opus codec files (Wolf's "all opus audio", 297 files)

The normalization pipeline:

```
Raw audio (any format)
  -> librosa.load(path, sr=16000, mono=True)
  -> Peak normalize to [-1.0, 1.0]
  -> librosa.effects.trim(y, top_db=30)  # Remove leading/trailing silence
  -> Save as 16kHz mono float32 WAV
```

The existing `convert_waptt_to_wav.py` implements this correctly. Extend it to handle all formats and all participants uniformly.

### 2.2 Preprocessing Decisions (Evidence-Based)

These decisions are backed by experimental results from v5-v7 analyses:

| Processing Step | Decision | Evidence |
|---|---|---|
| **Voice Activity Detection (VAD)** | DO NOT APPLY | Without VAD: MAE 10.24, r=0.368. With VAD: MAE 10.47, r=0.343. Pauses contain glucose-related information (speech hesitancy, timing). |
| **Data augmentation** (time-stretch, pitch-shift) | DO NOT APPLY | v5 analysis showed augmentation degrades performance. The glucose signal resides in natural vocal characteristics; distorting them destroys the signal. |
| **Noise reduction** | LIGHT ONLY | Apply spectral gating only if SNR < 10 dB. Do not apply aggressive denoising -- it removes vocal texture information. |
| **Silence trimming** | YES | `librosa.effects.trim(top_db=30)` -- generous threshold to keep speech-adjacent pauses while removing dead air. |
| **Resampling** | YES, to 16 kHz | All SSL models expect 16 kHz. MFCCs can use any rate but standardizing avoids bugs. |

### 2.3 Single Unified Audio Processor

Replace the fragmented conversion scripts with one module:

```python
# audio_processor.py
def load_audio(path: str) -> np.ndarray:
    """Load any format to 16kHz mono float32."""
    y, _ = librosa.load(path, sr=16000, mono=True)
    return y

def normalize(y: np.ndarray) -> np.ndarray:
    """Peak normalize to [-1, 1]."""
    peak = np.abs(y).max()
    return y / peak if peak > 0 else y

def trim_silence(y: np.ndarray) -> np.ndarray:
    """Remove leading/trailing silence."""
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    return trimmed

def quality_check(y: np.ndarray) -> dict:
    """Return duration, estimated SNR, validity flag."""
    duration = len(y) / 16000
    # Estimate SNR from top-10% energy vs bottom-10%
    frame_energies = librosa.feature.rms(y=y)[0]
    sorted_e = np.sort(frame_energies)
    noise_floor = sorted_e[:max(1, len(sorted_e)//10)].mean()
    signal_level = sorted_e[-max(1, len(sorted_e)//10):].mean()
    snr_db = 10 * np.log10(signal_level / noise_floor) if noise_floor > 0 else 40.0
    return {"duration": duration, "snr_db": snr_db, "is_valid": duration >= 2.0}
```

---

## Part 3: Feature Engineering

### 3.1 Core MFCC Feature Set (97 dimensions -- the Proven Foundation)

The canonical pipeline's feature set has been validated as effective for the 3 participants that show signal:

- **20 MFCCs** x (mean, std) = 40 features -- captures vocal tract shape
- **20 delta-MFCCs** x (mean, std) = 40 features -- captures temporal dynamics
- **Spectral centroid, bandwidth, rolloff** (mean, std each) = 6 features -- timbral brightness/warmth
- **Energy** (RMS mean, std) = 2 features -- vocal effort/loudness
- **Zero crossing rate** (mean, std) = 2 features -- noisiness/breathiness
- **Pitch** (F0 mean, std, range, voiced_fraction) = 4 features -- fundamental frequency (directly linked to glucose per Klick 2024: 0.02 Hz per 1 mg/dL)
- **Duration** = 1 feature -- speech rate proxy
- **Time-of-day** (hour_sin, hour_cos) = 2 features -- circadian glucose pattern

**Total: 97 dimensions.** With 752 samples, this gives a feature-to-sample ratio of ~7.7:1, well within the safe zone for regularized linear models and tree ensembles.

### 3.2 Advanced Prosodic Features (Add 4-6 Selectively)

From `advanced_voice_features.py`, these features are physiologically grounded for glucose estimation:

| Feature | Description | Glucose Relevance |
|---|---|---|
| F0 coefficient of variation | Pitch instability measure | Hypoglycemia increases vocal tremor |
| Jitter (F0 perturbation quotient) | Cycle-to-cycle F0 variation | Neuromuscular control affected by glucose |
| Shimmer approximation | Amplitude perturbation | Vocal fold tension changes with glucose |
| Voiced ratio | Fraction of voiced vs. unvoiced frames | Speech fluency degrades at extreme glucose |
| Harmonic-to-noise ratio (HNR) | Voice clarity measure | Glucose affects vocal fold closure |
| Speech rate (syllables/second) | Speaking tempo | Cognitive effects of hypo/hyperglycemia |

Add these 4-6 features to the core 97, bringing total to ~103. **Do NOT add all ~40 features** from the advanced module -- too many weak features make regularization harder with small N.

### 3.3 SSL Embeddings: WavLM with PCA (the Right Way to Use Deep Representations)

No published study has combined self-supervised speech models with voice-based glucose estimation. This represents a genuine research opportunity, but it must be done correctly given the small dataset:

1. **Use `microsoft/wavlm-base-plus`** instead of `facebook/hubert-base-ls960`. WavLM consistently outperforms HuBERT on paralinguistic tasks (speaker identity, emotion, age estimation) because it was trained with an utterance mixing strategy that preserves speaker characteristics.

2. **Extract last hidden state only**, aggregate with **mean pooling** across time frames = **768 dimensions** (not 2,304 from mean+std+max).

3. **Apply PCA to reduce to 20-25 components** using the training set. This is critical -- 768 raw dimensions would recreate the HuBERT dimensionality trap. PCA on the training fold, transform applied to test fold. Never fit PCA on the full dataset.

4. **Concatenate with MFCC + prosodic features:**

```
Final feature vector: 97 (MFCC) + 6 (prosodic) + 25 (WavLM-PCA) = 128 dimensions
```

The ratio of 128 features to 752 samples (~5.9:1) is manageable for regularized regression. If we expand the dataset to ~1,000 samples, the ratio drops to ~7.8:1, even safer.

### 3.4 Why This Specific Feature Budget

The feature engineering strategy follows one principle: **maximize information per dimension, minimize dimensions per sample.**

| Feature Set | Dimensions | Information Type | Cost |
|---|---|---|---|
| MFCCs + deltas | 80 | Vocal tract shape + dynamics | Very cheap |
| Spectral/energy/ZCR | 10 | Timbral quality | Very cheap |
| Pitch (F0) | 4 | Fundamental frequency (directly linked to glucose) | Cheap |
| Time-of-day | 2 | Circadian context | Free |
| Duration | 1 | Speech behavior | Free |
| Prosodic (advanced) | 6 | Voice quality/stability | Cheap |
| WavLM-PCA | 25 | Learned paralinguistic representations | Expensive (GPU) |
| **Total** | **128** | | |

If GPU inference is not available (e.g., mobile deployment), drop the WavLM-PCA features. The 103-dimension MFCC+prosodic set is the minimum viable feature set and is already proven to capture the glucose signal in 3/11 participants.

---

## Part 4: Modeling Strategy

### 4.1 Accept the Personalization Dependency

The data is unambiguous:

- **Population model:** r = -0.124, MAE 14.29 vs baseline 12.76 -- **worse than predicting the grand mean**
- **Personalized models:** 3/11 participants show r > 0.4

This aligns with the literature. The 2020 review by Cecchi et al. stated: "generalization beyond one speaker in instantaneous glucose estimation can be impossible, because in endocrinology research the trend is that many aspects regarding diabetes are highly individual and average responses are useless."

The Klick Labs 2024 study (505 participants with CGM) found that F0 increases 0.02 Hz per 1 mg/dL glucose **within an individual** -- the relationship is real but person-specific. Different people have different baseline F0, different vocal fold properties, and different glucose-voice coupling strengths.

**The primary modeling target is personalized models, not population models.** A population model serves only as a warm-start prior for few-shot calibration. It does not need to be accurate on its own.

### 4.2 Model Selection: Ensemble of Three

From the canonical results, the winning models vary by participant (GBR for Lara/Wolf/Sybille, SVR_RBF for Anja/Margarita, RF for Darav/Joao/Vicky). Rather than selecting per-participant (which overfits the model selection itself), use a **3-model ensemble with equal-weight averaging:**

1. **Ridge regression** (alpha=1.0) -- linear baseline, fast, regularized
2. **SVR with RBF kernel** (C=1.0) -- captures nonlinear patterns, robust to outliers
3. **Gradient Boosting Regressor** (100 trees) -- captures feature interactions, often the best individual model

The ensemble prediction is simply `(ridge_pred + svr_pred + gbr_pred) / 3`. This is more robust than any single model across participants and avoids the meta-optimization of model selection.

### 4.3 Temporal Validation (Critical Fix)

The current pipeline uses Leave-One-Out Cross-Validation (LOO-CV) for personalized models. This is **problematic for time-series data** because it trains on future observations to predict past ones. In production, the model will only ever see past data at prediction time.

**Implement chronological split validation:**

- Sort each participant's data by timestamp
- **Primary split:** First 70% chronologically for training, last 30% for testing
- **For participants with 50+ samples:** Use expanding-window validation -- train on samples 1..K, predict K+1, then train on 1..K+1, predict K+2, etc. Report the average MAE across all forward-looking predictions.

This is a more honest evaluation and will likely produce worse numbers than LOO-CV. That's the point -- it reflects real-world performance.

### 4.4 Per-Person Time Offset Optimization

Glucose affects voice with a delay. Different people may show vocal changes anywhere from 0 to 30 minutes after a glucose shift. The `offset_optimization.py` script already implements this:

- For each participant, test offsets from -30 to +30 minutes in 5-minute steps (13 candidates)
- At each offset, shift all audio timestamps by that amount before matching to CGM readings
- Use cross-validated MAE to select the optimal offset
- **This optimization must be done inside the cross-validation loop** -- do not select the offset on the test set

Current hard-coded offsets from previous analysis: Wolf +15 min, Anja 0 min, Margarita +20 min. These should be validated on the temporal split, not taken at face value.

### 4.5 Few-Shot Calibration Architecture

This is the architecture for a shipping product:

```
1. POPULATION BASE MODEL (weak but universal)
   - Trained on all available participants' data (LOPO validated)
   - Expected performance: near-zero correlation (current r = -0.124)
   - Purpose: provide a starting prediction and feature normalization

2. CALIBRATION PHASE (user provides paired data)
   - User wears OTC CGM for 7-14 days
   - Records 20-50 voice samples paired with CGM readings
   - System learns personalized affine transform:
     y_personal = a * features_personal + b
   - Alternatively: trains fresh personalized model if N >= 30

3. PRODUCTION INFERENCE
   - After calibration, user records voice only (no CGM)
   - System extracts features, applies personalized model
   - Returns estimated glucose + confidence interval
   - Model retrains periodically if user provides occasional CGM spot-checks
```

The population model is a warm-start, not the product. The product is the calibrated personalized model.

### 4.6 What NOT to Do

| Approach | Why Not | When It Becomes Viable |
|---|---|---|
| Fine-tune WavLM end-to-end | 90M+ parameters, <1,000 samples = catastrophic overfitting | N >= 2,500 samples (Phase 3) |
| Neural network regressors (MLP) | Overfits with small N, no regularization advantage over Ridge | N >= 5,000+ samples |
| Stacking ensembles | Requires out-of-fold predictions on already small datasets; meta-learner overfits | N >= 2,000+ |
| Deep temporal models (LSTM/Transformer) | Requires sequential data points per participant; most have <100 | N >= 100 per participant with controlled timing |
| Feature selection (LASSO, RFE) | With 128 features and ~30-150 samples per person, selection is unstable | N >= 500 per participant |

---

## Part 5: Evaluation Framework

### 5.1 Mandatory Baselines

Every result must be reported alongside these baselines:

1. **Mean predictor:** Predict the training set mean for all test samples. This is the "no-information" baseline. Any model that doesn't beat this has no signal.

2. **Participant-mean predictor (for population model):** Predict each participant's known training mean. This reveals whether a population model is just learning per-person averages or actually tracking glucose variation.

3. **Previous-value predictor (for temporal models):** Predict the last known glucose value. This is the "persistence" baseline -- surprisingly hard to beat for slowly-varying signals.

### 5.2 Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** (mg/dL) | mean(|pred - actual|) | Primary metric, clinically interpretable |
| **Pearson r** | correlation(pred, actual) | Key signal indicator -- does the model track glucose variation? |
| **MARD** (%) | mean(|pred - actual| / actual * 100) | Percentage error, comparable across glucose ranges |
| **Clarke Error Grid** | Zone A+B % | Clinical acceptability (Zone A = clinically accurate, Zone B = benign error) |
| **Improvement** | (baseline_MAE - model_MAE) / baseline_MAE * 100 | Percentage improvement over mean predictor |

### 5.3 Signal Detection Threshold

A personalized model "has signal" if and only if:

- **Improvement over mean predictor > 10%** (not just marginally better)
- **Pearson r > 0.3** (weak-to-moderate correlation with actual glucose)
- **p-value < 0.05** for r (not just by chance in small N)

By these criteria, our current 3 signal-bearing participants:
- **Lara:** +24% improvement, r=0.506 -- YES
- **Wolf:** +17% improvement, r=0.455 -- YES
- **Sybille:** +10% improvement, r=0.456 -- BORDERLINE (just meets the 10% threshold)

### 5.4 Statistical Significance

With small per-participant sample sizes (7-156), results are noisy. Report:

- **95% confidence intervals on MAE** via bootstrap (1,000 iterations, resample test set with replacement)
- **p-value** from paired t-test: model absolute errors vs. mean-predictor absolute errors
- **Effect size** (Cohen's d): quantifies practical significance beyond statistical significance

For the population model, use **permutation testing**: randomly shuffle glucose labels 1,000 times, re-evaluate the model each time, and report the fraction of permuted runs that achieve equal or better r. If the observed r is within the permutation distribution, the model has no real signal.

### 5.5 Clinical Relevance Benchmarks

For context only -- we are nowhere near these thresholds:

| Standard | Requirement | Current State |
|---|---|---|
| **ISO 15197** (SMBG accuracy) | 95% within +/-15 mg/dL (if glucose < 100) or +/-15% (if >= 100) | Best personalized: ~60-70% in Zone A |
| **CGM-grade MARD** | < 10% | Best personalized: 6.9% (Lara), 9.7% (Wolf) |
| **Clarke Error Grid** | Zone A+B >= 99% | Not evaluated systematically |

**Honest framing: This is a wellness directional signal, not a diagnostic tool. The claim is "track your glucose trend," not "measure your glucose level."**

---

## Part 6: Code Architecture

### 6.1 Scripts to Keep and Refactor

| Script | Action | Reason |
|---|---|---|
| `build_canonical_dataset.py` | **KEEP, extend** | Single source of truth for data. Add Christian_L, expand Wolf opus files, resolve orphans. |
| `run_canonical_pipeline.py` | **KEEP, refactor** | Clean MFCC pipeline. Add temporal validation, ensemble, WavLM-PCA features. |
| `convert_waptt_to_wav.py` | **KEEP** | Correctly converts .waptt to .wav at 16kHz mono. Extend to .opus and all participants. |
| `advanced_voice_features.py` | **KEEP** | Well-structured prosodic feature extractor. Cherry-pick 4-6 features. |
| `offset_optimization.py` | **KEEP** | Clean `OffsetOptimizer` class. Integrate into canonical pipeline. |

### 6.2 Scripts to Discard

| Script | Reason |
|---|---|
| `voice_glucose_pipeline.py` | Contains the `glucose_in_filename: True` bug. Fully superseded. |
| `comprehensive_analysis_v4.py` through `v7_fast.py` | Five versions of the same analysis with incremental changes. All superseded by `run_canonical_pipeline.py`. |
| `combined_hubert_mfcc_model.py` | Produced the 53.8 MAE result. The 2,500-dimension approach is proven harmful at current sample size. |
| `hubert_glucose_model.py` | Frozen HuBERT features alone are worse than MFCC. Transfer learning results (r=0.957) unvalidated. |
| `production_analysis.py`, `full_production_analysis.py` | Kitchen-sink scripts importing GPs, stacking, neural networks -- premature complexity. |
| `innovative_analysis.py`, `enhanced_analysis.py` | Experimental ideas (multi-offset fusion, rate-of-change) that should be re-implemented cleanly on canonical pipeline if needed. |
| `wav2vec_features.py` | wav2vec2-base features. If SSL is used, WavLM is strictly better. |
| `windowed_analysis_v3.py`, `windowed_analysis_fast.py` | Experimental windowed approaches, not integrated. |
| `hyperparameter_analysis.py`, `model_comparison.py` | One-off experiments. |
| `generate_documentation.py`, `generate_documentation_v2.py` | Report generators for deprecated pipelines. |

### 6.3 Target Clean Architecture

```
TONES/
  data/
    raw/                          # Original .waptt/.opus/.wav per participant
    processed/                    # Standardized 16kHz mono WAV
    canonical_dataset.csv         # Single source of truth

  src/
    data/
      build_dataset.py            # From build_canonical_dataset.py
      convert_audio.py            # From convert_waptt_to_wav.py
      match_glucose.py            # Timestamp matching + interpolation
    features/
      mfcc_features.py            # Core 97-dim MFCC/spectral/pitch extraction
      prosodic_features.py        # Selected advanced features (from advanced_voice_features.py)
      ssl_features.py             # WavLM embedding extraction + PCA
      time_features.py            # Time-of-day cyclical encoding
      combine.py                  # Feature fusion into final vector
    models/
      personalized.py             # Per-participant LOO-CV / temporal-split models
      population.py               # LOPO population model
      calibration.py              # Few-shot calibration (affine transform)
      ensemble.py                 # Ridge + SVR_RBF + GBR ensemble
    evaluation/
      metrics.py                  # MAE, r, MARD, Clarke grid
      baselines.py                # Mean predictor, majority class, previous-value
      temporal_cv.py              # Chronological split validation
      visualization.py            # Scatter plots, Clarke grid, per-participant summaries
    utils/
      audio_io.py                 # Load, normalize, quality check
      glucose_io.py               # FreeStyle Libre CSV parsing (EN/DE/PT)

  scripts/
    run_pipeline.py               # Main entry point
    run_experiments.py             # Experimental variations (feature ablation, etc.)

  results/
    pipeline_results.json         # Canonical results
    features_dataset.csv          # Feature matrix
    figures/                      # Generated plots
```

This structure has clear separation of concerns, no duplicate logic, and a single path from raw data to evaluated model.

---

## Part 7: Product Strategy

### 7.1 Tier 1: Calibrated CGM-Companion (Shippable Now)

**What it is:** A mobile app that requires a 7-14 day calibration period with an OTC CGM (Dexcom Stelo or Abbott Lingo). After calibration, the app estimates glucose trends from voice alone.

**Honest claim:** "Track your glucose trend between CGM readings after a personalized calibration period."

**What it is NOT:** A glucose meter. It does not replace CGM or fingerprick testing.

**Target users:**
- People currently wearing OTC CGMs who want cheaper ongoing monitoring after their sensor expires
- Health-curious early adopters who want passive glucose awareness
- Biohackers / quantified-self community

**Pricing:** $9.99-14.99/month after free calibration period

**Technical requirements:**
- ONVOX app (lovable.dev: https://github.com/WolfGebhardt/onvox) captures voice
- Backend runs feature extraction + personalized model inference
- Calibration wizard guides user through CGM pairing
- Results displayed as directional trend (rising/stable/falling), not absolute number

**Revenue potential:** $500K-$2M ARR with 4,000-13,000 subscribers

### 7.2 Tier 2: General Wellness Tool (Requires Population r > 0.3)

**Gate:** Population model Pearson r > 0.3 AND MAE < 15 mg/dL on held-out participants from a 50+ participant study.

**What changes:** Users get directional glucose estimates without any calibration period. Less accurate than Tier 1 but zero friction.

**Pricing:** $14.99-24.99/month (broader market, lower accuracy)

**Revenue potential:** $5M-$20M ARR

### 7.3 Tier 3: Medical-Adjacent Screening (Requires Clinical Validation)

**Gate:** IRB-approved study, n > 200, published peer-reviewed results, FDA General Wellness exemption confirmed.

**What it is:** B2B API for health systems, insurers, corporate wellness programs. Screens for pre-diabetes risk from a 30-second voice sample.

**Pricing:** B2B APIs at $5-15/patient/month

**Revenue potential:** $20M-$100M+ ARR

### 7.4 The Existential Risk

The population correlation of r = -0.124 raises a fundamental question: **Is voice-glucose coupling too individual to ever generalize across people?**

If 50+ diverse participants still show population r < 0.2, the mass-market no-calibration thesis (Tier 2) is falsified. ONVOX would remain viable as a calibration-required CGM companion (Tier 1), which is a real but smaller market.

The Klick Labs 2024 study (505 participants) found that F0 increases with glucose **within individuals** but the relationship is person-specific. This is consistent with our findings. Population generalization may require normalizing out individual baseline vocal characteristics before looking for glucose-related variation -- an unsolved problem in the literature.

**The Phase 3 recruitment (50+ participants) is the definitive test of whether Tier 2 is achievable. Do not invest beyond Tier 1 until this question is answered.**

---

## Part 8: Scaling Roadmap

### Phase 1: Data Rescue (Weeks 1-2, Cost: $0)

- [ ] Convert Wolf's 297 opus files to WAV (add "all opus audio" to audio_dirs, extend `convert_waptt_to_wav.py`)
- [ ] Integrate Christian_L (Number_8 CGM + .waptt audio)
- [ ] Cross-reference orphaned CGMs with Bruno, Valerie, Edoardo, Jacky
- [ ] Convert all .waptt files for unmatched participants
- [ ] Implement chronological validation split in `run_canonical_pipeline.py`
- [ ] Re-run canonical pipeline with expanded data
- [ ] **Target:** 900-1,000+ matched samples, 12-15 participants

### Phase 2: Feature + Model Upgrade (Months 1-2, Cost: $0)

- [ ] Add 4-6 prosodic features from `advanced_voice_features.py`
- [ ] Implement WavLM-base-plus embeddings with PCA reduction to 25 dims
- [ ] Build 3-model ensemble (Ridge + SVR_RBF + GBR)
- [ ] Implement per-person offset optimization inside CV loop
- [ ] Run full ablation study: MFCC-only vs MFCC+prosodic vs MFCC+prosodic+WavLM
- [ ] Independently validate transfer learning claims (r=0.957)
- [ ] **Gate:** At least 4/11 participants show r > 0.3 with temporal validation

### Phase 3: Scale Data Collection (Months 2-4, Cost: $15-25K)

- [ ] Design standardized recording protocol (4 speech tasks per session)
- [ ] Recruit 50+ participants with OTC CGMs ($89/mo x 50 = $4,450 for sensors)
- [ ] Build data collection app or integrate into ONVOX
- [ ] 14-day collection period per participant
- [ ] Retrain population model on combined dataset (original 11 + new 50+)
- [ ] **Decision gate:** If population r > 0.3, proceed to Tier 2. If not, focus exclusively on Tier 1.

### Phase 4: Ship Tier 1 Product (Months 4-8, Cost: $10-20K)

- [ ] Build calibration wizard in ONVOX app
- [ ] Backend API for feature extraction + personalized inference
- [ ] 7-day free trial with CGM requirement
- [ ] Launch to CGM-wearing early adopters
- [ ] Collect real-world data from consenting users (data flywheel)
- [ ] Iterate on model with production data

### Phase 5: Clinical Validation (Months 8-18, Cost: $50-150K)

- [ ] Only if Tier 1 shows product-market fit AND population r > 0.3 at scale
- [ ] IRB application
- [ ] Prospective study: 200+ participants, diverse demographics
- [ ] Pre-register hypotheses and analysis plan
- [ ] Submit for peer review
- [ ] If successful: pursue FDA General Wellness exemption, launch Tier 2/3

---

## Appendix: Research Context

### Key Studies Informing This Methodology

**1. Lehmann et al. (2025) -- "Listening to Hypoglycemia," *Diabetes Care***
- 22 T1D participants, controlled clinical setting
- Smartphone voice recordings during induced hypoglycemia
- ML algorithm correctly detected hypoglycemia in ~90% of cases (reading task) and ~87% (syllable repetition)
- Features: pitch, volume, resonance, clarity, sound dynamics
- Published in a top-tier diabetes journal -- the strongest clinical evidence for voice-glucose coupling
- Limitation: controlled setting, not real-world; binary classification (hypo vs normal), not regression

**2. Klick Labs / Colive Voice (2024) -- T2D Screening, *PLOS Digital Health***
- 607 US participants
- Used BYOL-S/CvT embeddings (~1,024 deep-learned features) + 6,000 acoustic features
- Gender-specific algorithms: AUC 75% (men), 71% (women) for T2D detection
- Important: this detects T2D *status* (yes/no), not instantaneous glucose level

**3. Klick Labs (2024) -- F0 and Glucose, *Scientific Reports***
- 505 participants with CGM, up to 6 recordings/day for 14 days
- Found F0 increases 0.02 Hz per 1 mg/dL glucose increase (p < 0.001)
- The effect is **within-individual** -- confirms personalization dependency
- This is the largest voice-glucose paired dataset in the published literature

**4. Klick Labs (2020) -- Voice Biomarker Potential, *bioRxiv***
- 44 healthy participants, 1,454 voice recordings
- 196 voice biomarkers characterized
- Random forest classifier: 78.66% accuracy, AUC 0.83 for 3-class glucose (high/normal/low)
- Limitation: relatively small N, pre-print not peer-reviewed

**5. Ahmadli et al. (2024) -- "Hearing Your Blood Sugar," *arXiv***
- Logistic regression + Ridge regularization on vocal features
- Investigates blood vessel dynamics during voice production as mechanism
- Promising preliminary results, not yet peer-reviewed

**6. Cecchi et al. (2020) -- Review, *Journal of Voice***
- First systematic review of blood glucose estimation from voice
- Key finding: "generalization beyond one speaker in instantaneous glucose estimation can be impossible"
- Cataloged five different computational approaches from different research fields
- Reported results ranging from "nonrandom patterns in a few subjects" to "98% accuracy in 7,000 subjects" -- wide variance suggesting methodological inconsistencies

### Research Gap: SSL Speech Models for Glucose Estimation

As of February 2026, **no published study has combined self-supervised speech models (WavLM, HuBERT, wav2vec 2.0) with voice-based glucose estimation.** The existing literature uses either handcrafted acoustic features (MFCCs, eGeMAPS) or task-specific learned embeddings (BYOL-S/CvT).

WavLM and HuBERT have been shown to capture paralinguistic information (speaker identity, emotion, age, pathology) in their intermediate layers. Given that glucose affects voice through neuromuscular and physiological mechanisms similar to those captured by paralinguistic models, applying SSL speech representations to glucose estimation is a natural extension -- and a publishable novelty.

**However, our own experience confirms the dimensionality warning: raw SSL embeddings (768-2,304 dims) will overfit catastrophically with fewer than 1,000 samples. PCA reduction to 20-25 dimensions is non-negotiable.**

---

*This document represents the honest, evidence-based methodology for voice-based glucose estimation given the data, results, and research landscape as of February 2026. It prioritizes what works over what sounds impressive, and what can ship over what might work in theory.*
