# TONES / ONVOX — Corrected Next Steps Plan v2

*February 2026*

This document supersedes the original NEXT_STEPS_PLAN. Key corrections: phoneme residual extraction promoted from Phase 6 to Phase 1, Multi-Offset Fusion removed from all results (data leakage), calibration sample budget analysis added, hydration indicator added as Day 1 feature.

---

## Corrections Applied

### 1. Multi-Offset Fusion: Removed (Data Leakage)

Multi-Offset Fusion feeds glucose values from adjacent CGM timepoints (-30, -15, 0, +15, +30 min) as input features alongside voice features. CGM autocorrelation at 15-minute intervals is ~0.95. The model learns to interpolate glucose from glucose; voice features become irrelevant. The 5.15 mg/dL MAE result is glucose interpolation, not voice-based estimation.

**Corrected best results (single-offset, personalized):**

| Participant | Samples | Best Method | MAE (mg/dL) | Correlation |
|-------------|---------|-------------|-------------|-------------|
| Lara | 32 | Standard (+20min) | 6.01 | 0.60 |
| Steffen | 38 | Standard (+20min) | 7.88 | — |
| Wolf | 175 | Standard (+15min) | 8.28 | 0.39 |
| Margarita | 107 | Standard (-30min) | 8.78 | 0.32 |
| Anja | 83 | Standard (+15min) | 10.34 | 0.56 |
| Vicky | 79 | Standard (+30min) | 10.46 | 0.09 |
| Sybille | 55 | Standard (-30min) | 11.25 | 0.30 |

These are honest, defensible numbers. Trend accuracy (67–83%) remains valid and clinically valuable. All investor materials, scientific reports, and internal documentation must use these corrected figures.

### 2. Phoneme Residuals: Promoted from Phase 6 → Phase 1

Phoneme-level residual extraction is the core technical differentiator. Every downstream decision — feature selection, offset optimization, calibration requirements, model architecture — becomes more informative when computed on residual features (10–30% glucose signal) versus whole-utterance MFCCs (<2% glucose signal). Deferring it to week 10–16 means Phases 1–5 are built on the weaker signal and must be re-validated.

The fixed phrase (originally Phase 2) and phoneme alignment are the same dependency. A known phrase makes Montreal Forced Aligner deterministic, fast, and independent of Whisper. They ship together.

### 3. Offset-by-Feature: Demoted from Phase 1 → Sanity Check

Per-feature offset analysis on whole-utterance MFCCs produces unstable correlation maps dominated by phonemic variance. Run it for orientation; do not build production feature selection on it. Meaningful offset-by-feature analysis requires residual features (post-Phase 1).

### 4. Calibration Budget Analysis: Added (Phase 2)

The plan specified "20–50+ pairs" without empirical basis. Learning curve analysis on existing data determines the minimum viable calibration set, which directly controls onboarding duration and Tier transition timing.

### 5. Hydration Indicator: Added (Phase 3, Day 1 Feature)

Vocal fold hydration has direct, well-documented acoustic effects (increased jitter, shimmer, phonation threshold pressure). A hydration indicator requires zero calibration and provides immediate value during the glucose calibration period, transforming onboarding from "record and wait" to "here's your hydration status now."

---

## Revised Phase Plan

### Phase 1: Phoneme Residuals + Standardized Phrase (Weeks 1–3)

**Goal**: Define fixed phrase, build phoneme alignment, extract residual features, validate improvement over whole-utterance approach.

**Rationale**: This is the single highest-leverage technical task. It addresses the fundamental signal-to-noise problem — phonemic content accounts for 60–70% of acoustic variance, burying the <2% glucose signal. Phoneme-level normalization against personal baselines removes the dominant confound at its source, amplifying the physiological residual to 10–30% of remaining variance.

**Week 1: Phrase definition + alignment pipeline**

1. **Define standardized phrase.**
   Target: "What is my glucose level right now?" (25–30 phonemes, covers vowels /ɪ/, /aɪ/, /u/, /ɛ/, /aʊ/; nasals /m/, /n/; fricatives /s/, /z/, /ʒ/; stops /t/, /k/; approximants /w/, /l/, /ɹ/).
   Alternative: "The rainbow is a division of white light into many beautiful colors." (phonetically balanced, covers broader phoneme inventory, established in speech pathology).
   Selection criterion: maximum phoneme diversity in minimum duration (5–8 seconds).

2. **Install and configure Montreal Forced Aligner.**
   MFA on a known phrase is deterministic — no ASR uncertainty. Alignment runs in <1 second per recording on CPU. No Whisper dependency.
   ```bash
   mfa model download acoustic english_mfa
   mfa model download dictionary english_mfa
   mfa align corpus/ english_mfa english_mfa output/ --clean
   ```

3. **Build phoneme segmentation pipeline.**
   Input: WAV + known transcript → MFA alignment → per-phoneme time boundaries.
   For existing WhatsApp data (no fixed phrase): use Whisper for transcription, then MFA for alignment. This is the fallback for retrospective analysis.

**Week 2: Per-phoneme baseline + residual extraction**

4. **Extract per-phoneme features.**
   For each phoneme instance in each recording, extract:
   - F0 (mean, std, slope) via pYIN
   - Jitter (local, RAP, ppq5)
   - Shimmer (local, apq3, apq5)
   - HNR
   - Formant frequencies F1–F4 and bandwidths (vowels only)
   - Energy (RMS, spectral tilt)
   - Duration
   - MFCC 1–13 (mean over phoneme segment)

5. **Compute per-speaker, per-phoneme baselines.**
   For each participant, for each phoneme, compute feature means and standard deviations across all recordings. Requires ≥10 instances per phoneme (with 84+ recordings per participant, each phoneme appears 50–200+ times in the fixed phrase scenario; in WhatsApp data, frequency depends on content).

6. **Extract residual features.**
   For each phoneme instance: `residual = (feature_value - baseline_mean) / baseline_std`.
   Aggregate residuals per recording: mean residual across all phoneme instances → recording-level feature vector (~50–80 dimensions depending on phoneme inventory and feature set).

**Week 3: Validation + offset-by-feature on residuals**

7. **Compare residual vs. whole-utterance performance.**
   Run existing personalized model pipeline (Ridge, RF, BayesianRidge) on:
   - (a) Original MFCC features (173 dims) — existing baseline
   - (b) Phoneme residual features (~60 dims)
   - (c) Hybrid: original + residual (~230 dims)
   Evaluation: temporal train/test split (first 70% train, last 30% test), per-participant MAE and Pearson r.

8. **Offset-by-feature on residual features.**
   Now meaningful: sweep offsets -30 to +30 min, compute per-residual-feature correlation with glucose. Identify which phonemes and which features are most glucose-sensitive at which lags.

9. **Phoneme sensitivity ranking.**
   Rank phonemes by glucose correlation of their residual features. Hypothesis: voiced vowels (/a/, /i/) carry the strongest signal via mucosal viscosity; nasals (/m/, /n/) via velopharyngeal control; stops (/t/, /k/) via timing precision. This ranking is a novel scientific finding and potential publication.

**Deliverables:**
- Standardized phrase specification with phoneme inventory
- MFA alignment pipeline (script + documentation)
- Per-phoneme residual extraction pipeline
- Validation report: residual vs. whole-utterance vs. hybrid
- Offset-by-feature heatmap on residual features
- Phoneme sensitivity ranking
- Feature selection recommendations for production

**Success criterion:** Residual features achieve higher Pearson r than whole-utterance MFCCs for ≥4 of 7 participants with sufficient data.

**Risk mitigation:** If MFA alignment quality is poor on WhatsApp audio (variable quality, background noise), fall back to Whisper transcription + MFA alignment. If alignment fails entirely, use sustained vowel segments extracted by VAD — a degraded but still useful version of phoneme-level analysis.

---

### Phase 2: Calibration Budget Analysis + Pre-Personalization (Weeks 3–5)

**Goal**: Determine minimum viable calibration samples. Build voice-profile clustering for improved population models.

**Rationale**: Calibration duration directly determines user onboarding experience. At 3 recordings/day: 10 pairs = 3.3 days, 20 pairs = 6.7 days, 50 pairs = 16.7 days. Knowing where the learning curve flattens is a product-defining decision, not a nice-to-have.

**Tasks:**

1. **Learning curve analysis.**
   For each participant with ≥50 samples (Wolf, Margarita, Anja, Vicky):
   - Temporal ordering of all recordings
   - Train on first N samples (N = 5, 10, 15, 20, 30, 50), test on remaining
   - Record MAE, Pearson r, and 5-class classification accuracy at each N
   - Plot learning curves with confidence intervals (repeat with 5 random temporal splits)
   - Determine the "knee" — the N where accuracy gain per additional sample drops below 5%

2. **Minimum viable calibration report.**
   Output: recommended calibration count for each tier transition:
   - Tier 1→2: classification improvement threshold
   - Tier 2→3: regression viability threshold
   - Expected onboarding duration at 3, 6, 10 recordings/day

3. **Voice-profile clustering.**
   K-means or hierarchical clustering on standardized phrase features (or whole-utterance features for existing data).
   Variables: mean F0, F0 range, speaking rate, spectral centroid, formant dispersion.
   Target: 3–5 clusters capturing gender, pitch range, and speaking style.
   Evaluate: LOPO within-cluster vs. global population model.

4. **Cluster-specific population models.**
   Train one 5-class classifier per cluster.
   Evaluation: does cluster assignment improve population-level classification accuracy?

**Deliverables:**
- Learning curve plots per participant
- Minimum viable calibration count recommendation
- Clustering pipeline and cluster definitions
- Cluster-specific vs. global model comparison report

---

### Phase 3: Tiered Onboarding + Hydration Indicator (Weeks 5–7)

**Goal**: Implement user journey with immediate value (hydration) and progressive glucose capability.

**Rationale**: Users need value from recording 1. A hydration indicator provides this without calibration. Glucose classification (5-class) follows at Tier 1, improving with calibration data. Regression unlocks only when sufficient paired data exists.

**Tiered flow:**

| Tier | Trigger | Input | Output | Model |
|------|---------|-------|--------|-------|
| **0** | First recording | Standardized phrase | Hydration status (normal / mild dehydration / dehydrated) + voice profile + cluster assignment | Hydration classifier (no calibration) |
| **1** | Cluster assigned | Phrase | Hydration + 5-class glucose estimate + confidence | Cluster population classifier |
| **2** | 5–N calibrations (N from Phase 2) | Phrase + glucose labels | Hydration + improved 5-class glucose + confidence | Light personalization |
| **3** | ≥N calibrations (N from Phase 2) | Phrase + glucose labels | Hydration + continuous glucose (mg/dL) + trend direction + confidence interval | Full personalized regression |

**Hydration indicator implementation:**

Hydration affects voice through well-characterized pathways: vocal fold surface hydration reduces phonation threshold pressure, decreases jitter and shimmer, and increases HNR. Systemic dehydration does the reverse. Features:
- Jitter (local) — increases with dehydration
- Shimmer (local) — increases with dehydration
- HNR — decreases with dehydration
- CPP (cepstral peak prominence) — decreases with dehydration
- Phonation threshold pressure (estimated from spectral tilt)

Threshold-based classifier using population norms from speech pathology literature (Sivasankar & Leydon, 2010). No per-user calibration required. Accuracy target: directional correctness (hydrated vs. dehydrated), not quantitative measurement.

**Glucose classification:**

5-class binning (absolute thresholds for Tier 1, personal quintiles for Tier 2+):

| Class | Tier 1 (Absolute) | Tier 2+ (Personal Quintiles) |
|-------|-------------------|------------------------------|
| Very Low | <70 mg/dL | <Q10 |
| Low | 70–90 mg/dL | Q10–Q30 |
| Normal | 90–120 mg/dL | Q30–Q70 |
| High | 120–160 mg/dL | Q70–Q90 |
| Very High | >160 mg/dL | >Q90 |

**Tasks:**

1. Hydration feature extraction and classifier.
2. 5-class glucose classifier with absolute and quintile modes.
3. Tier transition logic: track calibration count, trigger model switches.
4. Status tracking: `calibration_count`, `current_tier`, `personalization_ready`, `days_to_next_tier`.

**Deliverables:**
- Hydration classifier with population baselines
- 5-class glucose classifier (absolute and quintile modes)
- Tier state machine with transition logic
- Tier transition UX specifications

---

### Phase 4: Privacy-Preserving Feature Pipeline (Weeks 6–8)

**Goal**: Audio never leaves device. Only numeric feature vectors transmitted.

**Rationale**: Raw voice is biometric, identifiable, and carries legal exposure (GDPR Art. 9, HIPAA). Feature vectors (~200–300 floats) are smaller, faster to transmit, non-identifiable, and sufficient for all model operations. This is the single most important architectural decision for regulatory and trust positioning.

**Architecture:**

```
┌─────────────────────────────────────────────┐
│                  ON-DEVICE                   │
│                                              │
│  Mic → WAV (16kHz, 16-bit) → Feature        │
│  Extraction (MFCCs, VQ, F0, jitter,         │
│  shimmer, HNR, formants) → DELETE audio      │
│              │                               │
│              ▼                               │
│  Feature vector (JSON, ~300 floats)          │
│  + timestamp + device_id                     │
└──────────────┬───────────────────────────────┘
               │ HTTPS (TLS 1.3)
               ▼
┌─────────────────────────────────────────────┐
│              ONVOX SERVER                    │
│                                              │
│  Auth → Validate → Load user model →         │
│  Predict → Return estimate + confidence      │
│                                              │
│  Store: features + labels (with consent)     │
│  Never: raw audio                            │
└─────────────────────────────────────────────┘
```

**Feature schema (draft):**

```json
{
  "version": "1.0",
  "timestamp": "ISO8601",
  "device_id": "uuid",
  "user_id": "uuid",
  "phrase_id": "glucose_standard_v1",
  "features": {
    "mfcc": [40 floats],
    "delta_mfcc": [40 floats],
    "delta2_mfcc": [40 floats],
    "f0": {"mean": float, "std": float, "min": float, "max": float, "slope": float},
    "jitter": {"local": float, "rap": float, "ppq5": float},
    "shimmer": {"local": float, "apq3": float, "apq5": float},
    "hnr": float,
    "formants": {"f1": float, "f2": float, "f3": float, "f4": float,
                 "b1": float, "b2": float, "b3": float, "b4": float},
    "energy": {"rms_mean": float, "rms_std": float, "zcr_mean": float},
    "phoneme_residuals": [57 floats]
  },
  "calibration": {"glucose_mgdl": float, "source": "cgm|fingerstick|manual", "timestamp": "ISO8601"}
}
```

**Tasks:**

1. Feature schema specification (JSON Schema, versioned).
2. Reference extraction implementation (Python, matching training pipeline exactly).
3. Extraction parity tests: verify feature vectors from app extractor match training pipeline within ε < 0.01.
4. Secure transmission: HTTPS, JWT auth, rate limiting (max 20 requests/min/user).
5. Data retention policy: features stored 90 days by default, indefinite with research consent, deletion on user request within 72 hours.

**Deliverables:**
- Feature schema v1.0 (JSON Schema)
- Reference extraction SDK (Python package)
- Parity test suite
- Data retention and privacy policy document
- Data flow diagram for compliance review

---

### Phase 5: API + App Integration (Weeks 8–11)

**Goal**: Production API serving predictions. Dual-channel: direct app + Twilio/WhatsApp.

**API design (FastAPI):**

| Endpoint | Method | Auth | Input | Output |
|----------|--------|------|-------|--------|
| `/v1/predict` | POST | JWT | Feature vector (JSON) | `{tier, hydration_status, glucose_class, glucose_mgdl?, confidence, trend?, next_tier_in}` |
| `/v1/calibrate` | POST | JWT | `{user_id, glucose_mgdl, timestamp, source}` | `{calibration_count, current_tier, samples_to_next_tier}` |
| `/v1/status` | GET | JWT | `{user_id}` | `{calibration_count, current_tier, personalization_ready, model_version, last_prediction}` |
| `/v1/history` | GET | JWT | `{user_id, from, to}` | `[{timestamp, glucose_class, glucose_mgdl?, confidence, hydration}]` |
| `/v1/onboard` | POST | JWT | `{user_id, phrase_audio_features}` | `{cluster_id, voice_profile, hydration_status}` |

**Response schema (`/v1/predict`):**

```json
{
  "tier": 2,
  "hydration": {"status": "normal", "confidence": 0.82},
  "glucose": {
    "class": "normal",
    "class_confidence": 0.71,
    "mgdl": null,
    "trend": "rising",
    "trend_confidence": 0.76
  },
  "meta": {
    "model_version": "v1.2.0",
    "calibration_count": 14,
    "samples_to_regression": 6
  }
}
```

At Tier 3, `glucose.mgdl` populates with continuous estimate and `glucose.confidence_interval` is added.

**Model serving:**

- ONNX Runtime for inference (~5ms per prediction)
- Per-user model storage: PostgreSQL JSONB for model coefficients (Ridge/BayesianRidge models are <10KB per user)
- Model loading: LRU cache of hot user models, cold load from DB (<50ms)
- Population/cluster models: loaded at startup, versioned, A/B testable

**Twilio/WhatsApp integration:**

- Twilio webhook receives WhatsApp voice message
- Server-side: download audio → extract features → predict → reply with text
- Privacy trade-off: audio is processed server-side for WhatsApp channel (user consented)
- Response format: "Your voice suggests [normal/high/low] glucose levels. Hydration: [good/drink water]. [Trend: rising/falling/stable]."

**App integration (Lovable / React Native):**

- Recording screen: fixed phrase prompt, recording indicator, quality feedback
- On-device feature extraction (Phase 4 SDK)
- Calibration screen: enter glucose value + source (CGM sync, manual input)
- Dashboard: current estimate, trend arrow, hydration status, calibration progress bar, history chart
- Settings: data sharing consent, recording frequency reminders, account management

**Tasks:**

1. FastAPI application with all endpoints
2. Model registry and per-user model management
3. Twilio webhook handler
4. App recording + feature extraction integration
5. App calibration flow
6. App dashboard
7. End-to-end integration testing

**Deliverables:**
- Deployed API (staging + production)
- API documentation (OpenAPI/Swagger)
- Twilio/WhatsApp integration
- App build with recording, calibration, and dashboard
- Integration test suite

---

### Phase 6: Phoneme Residuals in Production (Weeks 10–13)

**Goal**: Integrate phoneme-level residual features into the production prediction pipeline.

**Rationale**: Phase 1 validates phoneme residuals in research. This phase operationalizes them. The fixed phrase from Phase 1 makes this tractable — MFA on a known phrase runs in <1 second on any modern phone CPU, no Whisper needed.

**On-device alignment options:**

| Option | Size | Latency | Quality | Privacy |
|--------|------|---------|---------|---------|
| MFA on fixed phrase (on-device) | ~50MB | <1s | High (known text) | Full (no audio sent) |
| Whisper tiny + MFA (on-device) | ~190MB | ~5-10s | Good | Full |
| Whisper base via API (cloud) | 0 on device | ~2-3s + network | Best | Audio sent to server |
| Skip alignment (whole-utterance only) | 0 | 0 | Baseline | Full |

**Recommended**: MFA on fixed phrase, on-device. The phrase is known, so no ASR step is needed. MFA acoustic model (~40MB) runs on CPU. Phoneme boundaries are deterministic given known text and audio.

**Tasks:**

1. MFA integration for mobile (iOS: Core ML wrapper; Android: TFLite or native binary)
2. Phoneme residual computation on-device using stored personal baselines
3. Baseline initialization: first 14 days of recordings build per-phoneme baselines on-device
4. Baseline update: exponential moving average as new recordings arrive
5. Feature vector extension: original features + phoneme residuals in single transmission
6. Model retraining with hybrid features (population + personalized)

**Deliverables:**
- On-device MFA integration (iOS + Android)
- Phoneme residual extraction SDK (mobile)
- Baseline management (initialization, update, storage)
- Updated models trained on hybrid features
- A/B test framework: hybrid vs. original features in production

---

### Phase 7: Blood Pressure Track (Weeks 12–16)

**Goal**: Add voice-based blood pressure estimation as a second health metric.

**Rationale**: Two published feasibility studies support voice-BP detection (Klick Labs IEEE Access 2024: 84% accuracy women, 77% men; Karger Digital Biomarkers 2025: hypertension screening from random speech). Patent landscape for voice-based BP is significantly less crowded than glucose. BP adds immediate product value, diversifies the clinical story, and provides a second data point for multi-biomarker wellness profiling.

**Implementation:**

1. Literature review and feature identification for BP-sensitive voice features
2. If BP reference data available (some participants may have cuff readings): build personalized BP models using same phoneme residual pipeline
3. If no BP reference data: implement population-level BP risk classification (normal/elevated/high) using published feature sets
4. Add BP output to API response and app dashboard
5. Patent filing: voice-based BP detection method using phoneme-level residuals (novel combination, not in any published work)

**Deliverables:**
- BP feature extraction pipeline
- BP classification model (population-level minimum)
- API and app integration
- Patent application draft for voice-based BP via phoneme residuals

---

### Phase 8: Model Improvement Loop (Ongoing from Week 8)

**Goal**: Use incoming production data to continuously improve models.

**Tasks:**

1. **Anonymized feature store.**
   Features + optional glucose labels, stripped of user identifiers, stored for research use (with explicit opt-in consent).

2. **Retraining pipeline.**
   Monthly retrain of cluster/population models. Automated evaluation against held-out test sets. A/B deployment of new models with rollback capability.

3. **Personalization monitoring.**
   Track per-user: calibration count, prediction accuracy (when ground truth available), model drift, recalibration triggers.

4. **Recalibration detection.**
   Voice baselines drift (illness, aging, weight change). Monitor prediction residuals; when systematic bias exceeds threshold, prompt user for recalibration recordings.

5. **Cross-user learning.**
   As user base grows: identify which phoneme residual features generalize vs. which are user-specific. Improve population priors for faster personalization of new users.

**Deliverables:**
- Data pipeline (feature ingestion → anonymization → storage → retraining)
- Retraining schedule and evaluation protocol
- A/B testing framework
- Recalibration detection and notification logic
- Quarterly model performance reports

---

## Dependency Graph

```
Phase 1: Phoneme Residuals + Phrase ──┐
                                      ├── Phase 2: Calibration Budget + Clustering
                                      │        │
                                      │        ▼
                                      │   Phase 3: Tiered Onboarding + Hydration
                                      │        │
                                      ▼        ▼
                              Phase 4: Privacy Pipeline
                                      │
                                      ▼
                              Phase 5: API + App ──────── Phase 8: Model Loop (ongoing)
                                      │
                                      ▼
                              Phase 6: Phoneme Residuals in Production
                                      │
                                      ▼
                              Phase 7: Blood Pressure Track
```

Phases 1–3 are sequential (each informs the next).
Phase 4 can begin in parallel with Phase 3 (feature schema defined in Phase 1).
Phase 5 requires Phases 3 + 4.
Phase 6 requires Phase 5 (production infrastructure).
Phase 7 can begin research in parallel with Phase 6.
Phase 8 begins when Phase 5 deploys.

---

## Timeline Summary

| Week | Phase | Key Output |
|------|-------|------------|
| 1–3 | 1: Phoneme Residuals + Phrase | MFA pipeline, residual features, validation report, phoneme ranking |
| 3–5 | 2: Calibration Budget + Clustering | Learning curves, minimum viable N, cluster models |
| 5–7 | 3: Tiered Onboarding + Hydration | Tier state machine, hydration classifier, 5-class glucose classifier |
| 6–8 | 4: Privacy Pipeline | Feature schema, extraction SDK, parity tests, privacy docs |
| 8–11 | 5: API + App | Deployed API, Twilio integration, app with recording + calibration + dashboard |
| 10–13 | 6: Production Phoneme Residuals | On-device MFA, mobile residual extraction, hybrid models |
| 12–16 | 7: Blood Pressure Track | BP classification, API integration, patent draft |
| 8+ | 8: Model Improvement | Retraining pipeline, A/B testing, recalibration detection |

Total to MVP (Tier 0–2 functional, API live, app deployed): **~11 weeks**.
Total to full personalized regression (Tier 3): **~13 weeks** (includes production phoneme residuals).

---

## Key Scripts (Updated)

| Script | Purpose | Phase |
|--------|---------|-------|
| `phoneme_residual_pipeline.py` | Phoneme alignment + residual extraction + hybrid models | 1 |
| `calibration_learning_curves.py` | Min viable calibration analysis | 2 |
| `run_pipeline.py` | Main pipeline: MFCC + VQ + temporal, personalized + population | 2, 3 |
| `offset_by_feature_analysis.py` | Offset per feature (run on residual features) | 1 |
| `hydration_classifier.py` | Hydration status from voice features | 3 |
| `hyperparameter_sweep.py` | Feature and model sweeps | 2 |
| `innovative_analysis.py` | Trend prediction, quantile regression (NOT Multi-Offset Fusion) | 3 |

---

## Metrics That Matter

**Do not report:**
- Multi-Offset Fusion MAE (data leakage)
- Clarke Error Grid percentages without stating glucose range and population (misleading for normoglycemic subjects where most predictions fall in Zone A by default)
- Population model results as evidence of system capability (population models fail; only personalized results count)

**Do report:**
- Per-participant MAE with single-offset personalized models
- Pearson r per participant
- 5-class classification accuracy (more honest than regression for current data volumes)
- Trend accuracy (67–83%, clinically actionable)
- Calibration learning curve: MAE vs. N samples
- Phoneme sensitivity ranking (novel scientific contribution)
- Comparison: residual features vs. whole-utterance features (core innovation validation)

---

## Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Phoneme residuals don't improve over whole-utterance | High — core thesis invalidated | 0.25 | Validate in Week 3; if fails, pivot to sustained vowel protocol (controlled phonation) |
| MFA alignment quality poor on smartphone audio | Medium — delays Phase 6 | 0.3 | Fixed phrase reduces alignment difficulty; fallback to Whisper for edge cases |
| Calibration requires >30 samples (>10 day onboarding) | High — user attrition | 0.4 | Hydration indicator provides Day 1 value; 5-class classification provides Tier 1 value |
| Insufficient participants for cluster validation | Medium — cluster models untested | 0.5 | Launch with global population model; add clustering when user base reaches 50+ |
| Regulatory challenge to wellness positioning | High — blocks launch | 0.15 | January 2026 FDA guidance explicitly includes glucose as wellness parameter; consult regulatory counsel before launch |
| Privacy breach (feature vectors re-identified) | Critical — trust destruction | 0.05 | Feature vectors are non-identifiable; no raw audio stored; penetration testing pre-launch |
