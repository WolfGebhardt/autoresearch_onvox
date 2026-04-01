# ONVOX / TONES — Algorithm Improvement Roadmap & Market Opportunity Analysis

**Date:** February 8, 2026
**Version:** 2.0 (Revised with honest assessment and corrected priorities)
**Document Type:** Strategic Technical & Commercial Roadmap
**Status:** Internal Planning Document

---

## Table of Contents

1. [Honest Current State Assessment](#1-honest-current-state-assessment)
   - 1A. Two Pipelines, Contradictory Results
   - 1B. Population Model Reality Check
   - 1C. Classification Claim Correction
   - 1D. Data Utilization Gap
   - 1E. HuBERT Dimensionality Problem
2. [Verified Performance Data](#2-verified-performance-data)
3. [Algorithm & Model Improvements (Corrected Priority Order)](#3-algorithm--model-improvements-corrected-priority-order)
   - Priority 0: Fix Data Utilization
   - Priority 1: Consolidate to One Pipeline
   - Priority 2: Accept Personalization Dependency
   - Priority 3: Targeted Feature Engineering
   - Priority 4: Fix Classification Claims
   - Priority 5: Model Architecture Upgrades (After Data Issues Resolved)
4. [Market Segments & Willingness to Pay](#4-market-segments--willingness-to-pay)
5. [Product-Market Fit by Accuracy Tier](#5-product-market-fit-by-accuracy-tier)
6. [Highest-ROI Improvement Priorities (Revised)](#6-highest-roi-improvement-priorities-revised)
7. [Competitive Positioning](#7-competitive-positioning)
8. [Pricing Recommendations](#8-pricing-recommendations)
9. [Summary: What to Do Now](#9-summary-what-to-do-now)
10. [Appendix A: Participant & Data Inventory](#appendix-a-participant--data-inventory)
11. [Appendix B: Market Data Sources](#appendix-b-market-data-sources)

---

## 1. Honest Current State Assessment

### 1A. Two Pipelines Produce Contradictory Results

ONVOX/TONES currently has two independent analysis paths running on overlapping but not identical data. Their results diverge by 5x on the most important metric:

| Dimension | Original HuBERT+MFCC Pipeline | Separate Session Pipeline |
|-----------|-------------------------------|--------------------------|
| **Script** | `combined_hubert_mfcc_model.py` | `comprehensive_analysis_v6.py` / `v7` |
| **Features** | HuBERT 2,304-dim + 200 MFCC (~2,500 total) | 60-69 MFCC/spectral/prosodic |
| **Population MAE** | **53.8 mg/dL** | **10.9 mg/dL** |
| **Personalized MAE** | 10-16 mg/dL | 8.8-9.6 mg/dL |
| **Population correlation** | Negative or near-zero | 0.105 (still near-zero) |
| **Participants** | 6-11 subjects, 2,110 samples | 7 subjects, 459-480 samples |
| **Matching rate** | 49% of available audio | Not stated (uses deduplication + timestamp alignment) |

**Diagnosis:** The 5x difference in population MAE (53.8 vs 10.9) from overlapping raw data means the original pipeline has severe data processing bugs, likely in timestamp matching, feature extraction, or both. The separate session pipeline produces more credible numbers but still shows near-zero population correlation.

---

### 1B. Population Model Reality Check

**Both pipelines show the same underlying reality: the population model correlation is near-zero or negative.**

The separate session's BayesianRidge achieves MAE 10.9 mg/dL with r=0.105. This looks acceptable on the surface but is misleading:

- Most participants are euglycemic (glucose range 80-130 mg/dL)
- Simply predicting ~100 mg/dL for everyone yields MAE ~11 automatically
- A model with r=0.105 cannot distinguish whether someone's glucose is 85 or 125
- It is predicting the grand mean with a small bias correction, not detecting glucose from voice

**Implication:** A no-calibration mass-market product does not exist yet with current data and methods. The population model is a mean predictor, not a glucose estimator.

---

### 1C. Classification Claim Correction

The reported 79.9% 5-class classification accuracy is inflated by test set imbalance:

- Test set class distribution: [17, 6, 7, 30, 79]
- Class 4 alone represents 56.8% of the test set
- A trivial majority-class classifier achieves 56.8%
- The model's 79.9% is 23 points above that baseline, which is meaningful but NOT "4x better than random"
- The correct baseline is majority-class (56.8%), not uniform random (20%)

**Corrected interpretation:** The model adds ~23 percentage points above a naive classifier. This is real signal but must be reported honestly.

---

### 1D. Data Utilization Gap

Only 49% of available data is being used:

| Participant | Audio Files | Matched | Utilization |
|-------------|------------|---------|-------------|
| Wolf | 947 | ~550 | 58% |
| Sybille | 546 | ~262 | 48% |
| Anja | 91 | ~28 | 31% |
| Margarita | 108 | 108 | 100% |
| Vicky | 79 | 79 | 100% |
| Steffen | 39 | 38 | 97% |
| Lara | 32 | 32 | 100% |
| **Total** | **4,295** | **2,110** | **49%** |

**Additional unused data:**

| Data Source | Status |
|------------|--------|
| Carmen | 4,268 CGM records, entirely unused, no audio located in project |
| 9 orphaned root-level CGM CSVs | Numbers 1, 3, 4, 5 (Boriska), X6, 11, 12 (Bopo), 13, 14 (Berna) -- not matched to audio |
| Bruno | 100+ audio files, no CGM match |
| Valerie | 114 audio files, no CGM match |
| Edoardo | 46 audio files, no CGM match |
| Jacky | 31 audio files, no CGM match |
| Darav | 6 CGM files, audio exists, not in pipeline |
| Joao | 2 CGM files, audio exists, not in pipeline |
| Alvar | 1 CGM file, audio exists, not in pipeline |

**Participant mapping key** (from `key glucose person freestyle libre 2023 Antler.txt`):

| Number | Name | Status |
|--------|------|--------|
| 2 | Steffen Haeseli | In pipeline |
| 5 | Boriska Molnar | CSV exists at root, no audio folder |
| 7 | Lara Planet | In pipeline |
| 8 | Christian Legler | Has folder, not in pipeline |
| 9 | Margo Grobovskaya (Margarita) | In pipeline |
| 10 | Viktoria Pilinko (Vicky) | In pipeline |
| 12 | Bopo del Valle | CSV exists at root, no audio folder |
| 14 | Berna Agar | CSV exists at root, no audio folder |
| 1, 3, 4, X6, 11, 13 | Unknown | CSVs exist, no names, no audio |

---

### 1E. HuBERT May Be Hurting the Population Model

The separate session (MFCC-only, 60-69 features) achieves better population model performance than the original pipeline (HuBERT + MFCC, 2,500 features).

**Why:** HuBERT embeddings are 2,304 dimensions of generic speech representation. With ~500 usable samples, this creates a severe p>>n problem (far more features than observations). The curse of dimensionality dominates: the model learns noise specific to each speaker's voice identity rather than glucose-relevant signal.

**Evidence:** The MFCC-only pipeline with 60-69 features on fewer samples outperforms the combined 2,500-feature pipeline on population generalization.

---

## 2. Verified Performance Data

### Per-Participant Personalized Models (from `results_summary.json`, Feb 2026)

| Participant | Samples | Glucose Range (mg/dL) | Best Model | MAE (mg/dL) | Correlation (r) |
|-------------|---------|----------------------|------------|-------------|-----------------|
| Wolf | 175 | 80-134 (mean 100) | KNN | 5.69 | 0.674 |
| Lara | 32 | 85-119 (mean 98) | GradientBoosting | 6.12 | 0.542 |
| Steffen | 38 | 76-117 (mean 96) | BayesianRidge | 8.01 | -1.000 |
| Margarita | 108 | 68-124 (mean 94) | SVR_RBF | 8.86 | 0.224 |
| Anja | 84 | 83-139 (mean 108) | SVR_RBF | 9.96 | 0.240 |
| Vicky | 79 | 58-133 (mean 92) | SVR_RBF | 10.95 | 0.071 |
| Sybille | 55 | 78-160 (mean 106) | BayesianRidge | 14.98 | -0.216 |

**Total samples in pipeline:** 571 (across 7 participants)

### Critical Observations

- Only Wolf (r=0.674) and Lara (r=0.542) show meaningful correlation
- Steffen (r=-1.000) indicates a degenerate model (likely predicting a constant)
- Sybille (r=-0.216), Vicky (r=0.071), Margarita (r=0.224), Anja (r=0.240) are near-zero or negative
- Most participants have narrow glucose ranges (~80-130 mg/dL), making the prediction task harder
- **Personalization "works" primarily for Wolf and Lara; for others it's producing low MAE by predicting near the mean**

### Transfer Learning Results (Reported, Requires Validation)

| Participant | MAE (mg/dL) | Correlation |
|-------------|-------------|-------------|
| Lara | 2.35 | 0.972 |
| Wolf | 2.99 | 0.985 |
| Sybille | 3.11 | 0.983 |
| Margarita | 3.18 | 0.957 |
| Vicky | 3.41 | 0.959 |
| Steffen | 3.52 | 0.931 |
| Anja | 4.49 | 0.910 |
| **Average** | **3.29** | **0.957** |

**Caution:** These numbers are suspiciously good given the poor personalized model correlations above. The transfer learning evaluation methodology needs independent validation for potential data leakage (e.g., LOO-CV without temporal ordering on time-series data).

---

## 3. Algorithm & Model Improvements (Corrected Priority Order)

The previous version of this document recommended scaling the dataset first and fine-tuning HuBERT. The corrected priority order addresses foundational issues before pursuing advanced techniques.

### Priority 0: Fix Data Utilization Before Touching Any Algorithm

**The single highest-ROI action is not data collection, not fine-tuning HuBERT, and not building features. It is matching the other 51% of existing audio to CGM readings.**

| Action | How | Expected Impact |
|--------|-----|----------------|
| Widen matching window from +/-15 to +/-30 or +/-45 minutes | Modify timestamp matching in pipeline | Could recover hundreds of Wolf/Sybille/Anja samples |
| Fix timestamp parsing across different WhatsApp export formats | Debug the matching code; handle edge cases | Addresses the 31% Anja utilization rate |
| Map the 9 orphaned root-level CGM files to audio participants | Cross-reference dates, sensor IDs, and the participant key | Could add 3-5 new participants (Boriska, Bopo, Berna, Numbers 1/3/4/11/13) |
| Integrate Darav (6 CSVs), Joao (2 CSVs), Alvar (1 CSV) | These have both audio and CGM data but are not in the pipeline | 3 additional participants, free |
| Locate Carmen's audio (4,268 CGM records exist) | Check WhatsApp exports, other drives, cloud storage | Potentially the largest single-participant dataset |

**Cost:** $0
**Timeline:** 1-2 weeks
**Impact:** Could take usable samples from ~571 to ~1,500+ without recruiting anyone

---

### Priority 1: Settle on One Pipeline and Validate It Properly

Two codebases producing contradictory numbers is untenable for a product or for investor communication.

| Action | Detail |
|--------|--------|
| Adopt the separate session pipeline as canonical | MFCC-only with proper deduplication, timestamp alignment, and per-person time-offset optimization produces more credible results |
| Run it on the full maximized dataset (after Priority 0) | Generate one set of honest numbers |
| Implement proper temporal train/test splits | LOO-CV without temporal ordering is problematic for time-series; use chronological splits |
| Validate transfer learning claims | The 3.29 MAE / r=0.957 numbers need independent verification for data leakage |
| Report against correct baselines | Population: compare to mean predictor; Classification: compare to majority-class baseline (56.8%), not uniform random (20%) |

**Cost:** Engineering time only
**Timeline:** 1-2 weeks
**Impact:** Establishes ground truth for all subsequent decisions

---

### Priority 2: Accept the Personalization Dependency and Ship the Calibrated Product

The data is unambiguous:
- **Personalization works** (MAE 5-11 mg/dL, meaningful correlation for Wolf/Lara)
- **Population models don't** (near-zero correlation regardless of MAE)

| Action | Detail |
|--------|--------|
| Ship the Lovable app (ONVOX) with CGM/fingerprick calibration requirement | The correct product for now; honest about what it can do |
| Target CGM-wearing audience first | Dexcom Stelo ($89/mo) and Abbott Lingo (~$49/sensor) users are the accessible wedge |
| Collect real-world data from these users | Creates the data flywheel: more users improve the population model |
| Graduate to no-calibration only when cross-subject correlation exceeds ~0.3 with MAE <15 | Set a concrete, measurable gate for the mass-market claim |

**Cost:** $10-20K development (Lovable app is already started)
**Timeline:** 2-3 months
**Impact:** First revenue + data collection flywheel

**Critical warning:** Trying to ship a no-calibration product with r=0.1 will destroy credibility. The tech press and diabetes community will test it immediately. Ship what works.

---

### Priority 3: Targeted Feature Engineering (Not Expansive)

More features is not the answer. With 42-69 features and ~500 samples, top correlations max out at |r|=0.159 (energy_mean). Adding 60 more weak features makes regularization harder.

**Do these:**

| Feature | Why | Cost |
|---------|-----|------|
| **Time-of-day** | Glucose has strong circadian patterns; this is free signal not currently used | Zero |
| **Per-person time offset optimization** | Already shown to matter (0 to +30 min variation); bake into pipeline as automatic calibration | Engineering time |
| **PCA on HuBERT embeddings (20-30 components) before feeding to regression** | If HuBERT is used at all, reduce from 2,304 to 20-30 dimensions to avoid p>>n curse | Engineering time |

**Do NOT do these (yet):**

| Feature | Why Not |
|---------|---------|
| Full F1-F4 formant tracking | Adding more features to a small dataset worsens overfitting |
| Glottal flow inverse filtering | Complex to implement, marginal expected benefit with current sample size |
| Whisper.cpp forced alignment | Engineering-heavy; benefit unproven for glucose |
| CPPS voice quality measures | Adding to already-saturated feature space |

---

### Priority 4: Fix the 79.9% Classification Claim

| Action | Detail |
|--------|--------|
| Re-run classification with stratified temporal split | Ensure class balance in train/val/test |
| Report against majority-class baseline (56.8%) | Not uniform random (20%) |
| Expected honest result | Likely 55-65% on balanced 5-class, which is still 3x random and still real signal |
| Use this corrected number in all materials | Honest reporting builds credibility with investors, clinicians, and regulators |

---

### Priority 5: Model Architecture Upgrades (After Priorities 0-4)

These are the right things to do eventually, but premature until the data and pipeline foundations are solid.

| Improvement | Current State | Proposed Change | Expected Impact | Prerequisite |
|-------------|--------------|-----------------|----------------|--------------|
| **Fine-tune transformer backbone** | Frozen HuBERT (feature extraction only) | Fine-tune last 4-6 layers on voice-glucose task | 20-40% MAE reduction on personalized models | Need 1,000+ samples to avoid overfitting a 90M+ param model |
| **Upgrade to WavLM-Large** | HuBERT-base (90M params, 768-dim) | WavLM-Large (300M params, 1024-dim) | Better speaker-specific physiological cues | Same data requirement |
| **Add temporal modeling** | Each sample predicted independently | LSTM/Transformer over user's sample sequence | Enables trend predictions ("glucose rising/falling") | Need consistent per-user time series (20+ sequential samples per person) |
| **Multi-task learning** | Predict glucose only | Jointly predict glucose + rate-of-change + risk class | Auxiliary tasks regularize training | Need rate-of-change labels (CGM data at sufficient frequency) |
| **Uncertainty quantification** | Simple confidence based on training MAE | Monte Carlo dropout or deep ensembles | Per-prediction confidence intervals | Required for clinical credibility and B2B |
| **On-device inference** | Server-side processing | Knowledge distillation to ~20M params; Core ML / TF Lite export | Offline use, privacy, no server costs | Need a stable model architecture first |

---

## 4. Market Segments & Willingness to Pay

### 4.1 Market Context

| Metric | Value | Source |
|--------|-------|-------|
| Adults with diabetes globally (2024) | 589 million | IDF Atlas 11th Ed. |
| Projected by 2050 | 853M-1.3B | IDF / Lancet |
| Undiagnosed diabetics | 252 million | IDF |
| Pre-diabetics (impaired glucose tolerance) globally | 635 million | IDF |
| Pre-diabetics in the US | 97.6 million (>80% unaware) | CDC |
| Type 1 diabetes globally | 9.5 million | IDF |
| Global CGM market (2025) | $13.28 billion | Mordor Intelligence |
| CGM users globally | >9 million (~1.5% penetration) | Sequenex |
| Global health expenditure on diabetes (2024) | >$1 trillion | IDF |
| US direct medical cost of diabetes (2022) | $307 billion | ADA |
| Voice biomarker market (2025) | $1.1-1.6 billion | SNS Insider / BIS Research |
| Voice biomarker market projected (2033) | $4.2-5.4 billion (14-16% CAGR) | SNS Insider |
| Non-invasive glucose monitoring market (2025) | $2B-9.3B (varies by definition) | Grand View / Mordor |
| Corporate wellness market (2026) | $69.8 billion | Grand View Research |
| Healthcare API market (2025) | $0.34 billion | MarketGenics |
| Healthcare data monetization market (2025) | $0.58 billion | MarketsandMarkets |
| OTC CGM market (US, 2024) | $48.6 million | Grand View Research |
| CGM market concentration | Abbott 56.3%, Dexcom 35.1%, Medtronic 6.9% | Sequenex |

### 4.2 Segment Analysis

| Segment | Addressable Size | WTP per Month | Pain Point ONVOX Solves | Required Accuracy (MAE) | Priority |
|---------|-----------------|---------------|------------------------|------------------------|----------|
| **Pre-diabetics (wellness screening)** | 97.6M (US), 635M (global) | $10-25 | "Am I at risk?" -- 80%+ are unaware; no monitoring habit | <25 mg/dL acceptable for screening (wellness, not diagnostic) | **#1 -- Largest TAM, lowest accuracy bar** |
| **Type 2 diabetics (non-insulin)** | ~33M (US) | $25-75 | Hate finger-pricks; 52% willing to pay $75+/mo for CGM; want trend awareness between meals | <15 mg/dL for clinically meaningful trends | **#2 -- High WTP, proven demand** |
| **Corporate wellness / employers** | 92% of large firms offer programs | $5-15/employee | Diabetes costs employers $327B/yr; need scalable screening at $30-75/employee/yr | Classification accuracy >75% for risk stratification | **#3 -- Scalable B2B revenue** |
| **CGM app partners (B2B2C)** | Levels, Nutrisense, Signos, January AI | $0.50-2/user | Extend monitoring between sensor changes; reduce sensor costs | <20 mg/dL as supplementary signal | **#4 -- Revenue with minimal sales effort** |
| **Telehealth / RPM platforms** | $0.34B healthcare API market | $5-15/patient (PUPM) | Add glucose screening to voice-based telehealth calls | Trend detection + risk classification | **#5 -- High-margin API revenue** |
| **Insurance / payers** | $18.4B corporate wellness market | $2-5/member | Pre-diabetes identification saves $10K+/patient in avoided T2D costs | Sensitivity >80% for screening | **#6 -- Outcome-based contracts** |
| **Type 1 diabetics** | 9.5M globally | $0 incremental | Already use CGMs continuously; voice adds nothing | <5 mg/dL (nearly impossible) | **Skip -- wrong market** |

### 4.3 Willingness to Pay Detail

**Type 2 Diabetics:**
- 52% willing to spend $75+/month on CGM sensors/supplies (PREPARE 4 CGM study, 2025)
- Average out-of-pocket paid by CGM users: $91/month
- Strong WTP for weight loss, HbA1c reduction, hypoglycemia avoidance
- Primary driver of OTC CGM adoption

**Pre-Diabetics:**
- Currently pay $0 for monitoring (most do not monitor at all)
- Estimated WTP: $25-100/month based on OTC CGM pricing and DTC health app benchmarks
- Diabetes prevention programs cost-effective at $50,000/QALY threshold

**Wellness Consumers:**
- Dominate OTC CGM market with 62% share (2024)
- Currently paying $49-99/month for OTC CGMs (Abbott Lingo, Dexcom Stelo)
- Pay $143-449/month for integrated CGM + coaching apps (Signos, Nutrisense)
- Price sensitivity higher than diabetics; engagement is discretionary

**Corporate Wellness / Employers:**
- Health risk assessments: $30-75/employee/year
- Fitness programs: $10-40/employee/month
- Mental health digital services: $3-15/employee/month
- 92% of large firms (500+ employees) offer wellness programs
- Diabetes prevention is a major employer priority

### 4.4 Comparable Product Pricing

| Product | Model | Monthly Cost | What's Included |
|---------|-------|-------------|-----------------|
| Dexcom Stelo (OTC) | Subscription | $89/mo or $99 one-time for 2 sensors | 15-day CGM sensor, no Rx needed |
| Abbott Lingo (OTC) | Per sensor | ~$49/sensor (~$98/mo) | 14-day sensor, wellness-focused |
| Abbott FreeStyle Libre (Rx) | Subscription | ~$80/mo cash; <$40 with insurance | 14-day sensor |
| Levels Health | Subscription | $24/mo (app only) to $125/mo (complete) | App + optional CGM + coaching |
| Nutrisense | Subscription | $225-399/mo | 2 FreeStyle Libre sensors + dietitian |
| Signos | Subscription | ~$143-449/mo | Dexcom CGM + AI coaching |
| January AI | Subscription | $288/mo (30-day) | CGM for 14 days then AI-only |

---

## 5. Product-Market Fit by Accuracy Tier

**Critical framework: what ONVOX can sell depends directly on what accuracy is achieved AND what it can honestly claim.**

### Tier 1: Current State (Ship Now)

**Personalized MAE 5-11 mg/dL (requires CGM calibration) | Population model: NOT USABLE (r~0.1)**

Sellable as: **Personalized wellness tracker with calibration requirement**

| Feature | Price Point | Segment | Honest Claim |
|---------|-------------|---------|-------------|
| "Metabolic voice check" after calibration with OTC CGM | $9.99/mo | Early adopter wellness consumers | "Track your glucose trend between CGM readings" |
| SDK for CGM apps to add voice-between-sensors | $0.50/user/mo | B2B2C (Levels, January AI) | "Supplementary signal, not replacement" |
| Research API for academic studies | $5K-25K/yr | Universities, pharma | Research use only |

**Estimated revenue potential:** $500K-$2M ARR
**What NOT to claim:** "Check your glucose with just your voice" (requires calibration; population model cannot do this)

---

### Tier 2: After Data Scale-Up + Pipeline Fix

**Population MAE <10 mg/dL WITH r>0.3 | Personalized MAE <5 mg/dL | 50+ participants, 5K+ samples**

Sellable as: **General wellness glucose awareness tool (minimal or no calibration)**

| Feature | Price Point | Segment |
|---------|-------------|---------|
| Standalone app "Voice glucose awareness" | $14.99-24.99/mo | Pre-diabetics, wellness consumers |
| B2B screening API for telehealth platforms | $5-15 PUPM | Telehealth, RPM companies |
| Corporate wellness voice screening | $30-50/employee/yr | Employers (large firms) |
| CGM cost reduction -- voice fills gaps between sensor changes | $0.50-2/user/mo | B2B2C |

**Estimated revenue potential:** $5M-$20M ARR
**Gate to enter Tier 2:** Cross-subject correlation r>0.3 AND MAE<15 mg/dL on held-out participants

---

### Tier 3: With Clinical Validation

**Population MAE <8 mg/dL WITH r>0.5 | Validated in IRB-approved clinical study (n=200+)**

Sellable as: **Medical-adjacent screening tool (FDA general wellness exempt)**

| Feature | Price Point | Segment |
|---------|-------------|---------|
| Insurance-integrated screening ("talk to check") | $2-5/member/mo | Payers, self-insured employers |
| Clinical decision support (trend alerts for RPM) | $15-25 PUPM | Health systems |
| Pharma companion diagnostic | $50K-500K/study | Pharmaceutical companies |
| White-label SDK for phone/watch OEMs | $0.10-0.50/device/mo | Samsung Health, Apple Health, Google Fit |

**Estimated revenue potential:** $20M-$100M+ ARR
**Gate to enter Tier 3:** IRB-approved study, n>200, diverse demographics, published peer review

---

## 6. Highest-ROI Improvement Priorities (Revised)

The previous version of this document had the wrong priority order. Here is the corrected sequence:

| Priority | Action | Cost | Timeline | Revenue Impact | Why This Order |
|----------|--------|------|----------|----------------|---------------|
| **0** | **Fix data utilization** -- match the other 51% of audio to CGM readings; widen matching window; map orphaned CSVs; integrate Darav, Joao, Alvar; locate Carmen's audio | $0 | 1-2 weeks | Could 3x usable samples from 571 to 1,500+ | Free data is the cheapest data |
| **1** | **Consolidate to one pipeline** -- adopt the MFCC pipeline as canonical; run on full dataset; implement temporal train/test splits; validate transfer learning claims | $0 | 1-2 weeks | Establishes ground truth for all decisions | Cannot make good decisions with contradictory numbers |
| **2** | **Ship the calibrated product** -- Lovable/ONVOX app with CGM/fingerprick calibration; target Dexcom Stelo and Abbott Lingo users | $10-20K | 2-3 months | First revenue ($500K-2M ARR); starts data flywheel | Revenue + data collection, honest about capabilities |
| **3** | **Targeted feature engineering** -- time-of-day, automatic per-person offset optimization, PCA on HuBERT to 20-30 dims | $0 | 1-2 weeks | Better personalized accuracy; better population model | Low-cost improvements with proven benefit |
| **4** | **Fix classification claims** -- re-run with stratified splits, report against majority-class baseline | $0 | 2-3 days | Credibility with investors, clinicians, regulators | Honest numbers are foundational |
| **5** | **Scale dataset externally** -- recruit 50+ participants with OTC CGMs | $15-25K | 3-6 months | Unlocks Tier 2 ($5-20M ARR) | Only after internal data is maximized |
| **6** | **Fine-tune HuBERT/WavLM backbone** -- unfreeze last 4-6 layers | $2-5K compute | 1-2 months | 20-40% personalized MAE reduction | Only viable with 1,000+ samples |
| **7** | **Add uncertainty quantification** -- Monte Carlo dropout or deep ensembles | $0 | 2-4 weeks | Required for B2B clinical credibility | Needed before enterprise sales |
| **8** | **Add temporal modeling** -- LSTM over user history for trend predictions | $0 | 3-4 weeks | "Glucose rising/falling" -- the feature users most want | Requires stable sequential per-user data |
| **9** | **On-device inference** -- knowledge distillation, Core ML / TF Lite export | $5-10K | 1-2 months | Privacy, offline use, no server costs | Needs stable model architecture first |
| **10** | **Clinical validation study** -- n=200, IRB-approved, diverse demographics | $50-150K | 6-12 months | Required for Tier 3 ($20-100M+ ARR) | Expensive; do after product-market fit confirmed |

---

## 7. Competitive Positioning

### 7.1 ONVOX vs. Alternatives

| Competitor Category | ONVOX Advantage | ONVOX Disadvantage |
|--------------------|----------------|-------------------|
| **CGM sensors** ($49-99/mo: Abbott Lingo, Dexcom Stelo) | No hardware, no needle, no cost per test, no waste | Lower accuracy, not continuous, requires calibration |
| **Finger-prick glucometers** ($0.50-1/strip) | Painless, instant, zero consumables | Lower accuracy |
| **Optical/spectroscopic devices** (GlucoTrack, Cnoga) | No device purchase; works on any smartphone | Less validated physical basis |
| **Other voice biomarker companies** (Kintsugi, Sonde, Canary Speech) | No competitor targets glucose -- first mover | They have more funding, clinical validation, established B2B channels |

### 7.2 Voice Biomarker Competitive Landscape

| Company | Focus | Funding | Status |
|---------|-------|---------|--------|
| Kintsugi | Depression/anxiety (20-sec speech) | ~$28M | FDA De Novo pending |
| Sonde Health | Mental, respiratory, cognitive (30-sec voice) | $16M+ | KT Corp, Qualcomm, AFWERX partnerships |
| Canary Speech | Depression, Alzheimer's, Parkinson's (ambient) | Not disclosed | Microsoft Cloud for Healthcare partner |
| Ellipsis Health | AI care management, vocal biomarker screening | ~$49M | Backed by Salesforce, Khosla, CVS Health Ventures |
| **ONVOX/TONES** | **Glucose estimation from voice** | -- | **Research stage; no direct glucose competitor** |

**No existing voice biomarker company targets glucose. ONVOX is the first mover.**

### 7.3 Key Moats to Build

1. **Data moat** -- Every calibrating user adds to the population model. Competitors need years to replicate paired voice-glucose data.
2. **Personalization moat** -- Few-shot calibration improves with each interaction, creating switching costs.
3. **First-mover in voice-glucose** -- No competitor exists in this niche. Patent the approach immediately.
4. **Network effects** -- More users produce a better population model, which improves cold-start for new users (virtuous cycle).

### 7.4 Existential Risk

The near-zero population correlation could reflect genuine biological reality: the voice-glucose signal may be too individual to ever generalize across people. If true, no-calibration will never work, and ONVOX is permanently a calibration-required supplementary tool. This is a viable business (CGM companion) but a smaller one than the mass-market vision. The data scaling effort (Priority 5) is the test: if 50+ diverse participants still show r<0.2 on population models, the mass-market thesis is falsified.

---

## 8. Pricing Recommendations

| Product | Model | Price | Justification |
|---------|-------|-------|---------------|
| **Consumer App (B2C)** | Freemium + subscription | Free: 3 checks/day. Pro: $14.99/mo unlimited + trends + history | Undercuts CGM apps ($99-449/mo) by 10x |
| **API (B2B)** | Per-prediction + monthly minimum | $0.05/prediction, $500/mo minimum | Competitive with health API pricing ($0.002-0.08/call) |
| **Enterprise SDK (B2B)** | Per-user/month | $2-5/active user/month | Cheaper than Glooko ($15 PUPM) since supplementary |
| **Corporate Wellness (B2B)** | Per-employee/year | $36-60/employee/year ($3-5/mo) | Within corporate wellness budget ($30-75/employee/yr for HRAs) |
| **Research License** | Annual | $10K-50K/year | Standard for academic/pharma data access |

### Pricing Rationale

- **B2C:** Must be dramatically cheaper than CGM-based solutions to justify lower accuracy. At $14.99/mo vs. $99-449/mo, the value proposition is convenience and cost savings for directional glucose awareness.
- **B2B API:** Per-prediction pricing aligns cost with usage. $500/mo minimum ensures meaningful revenue. At $0.05/prediction and ~100 predictions/user/month = $5/user/month, competitive with PUPM models.
- **Corporate wellness:** Must fit within existing budgets. At $3-5/employee/month, cheaper than fitness programs ($10-40/employee/month) and comparable to mental health digital services ($3-15/employee/month).

---

## 9. Summary: What to Do Now

### Immediate (Weeks 1-2): Fix Foundations -- $0

1. **Match the other 51% of audio** to CGM readings. Widen matching window. Map orphaned CSVs. Integrate Darav, Joao, Alvar. Locate Carmen's audio. This alone could triple the usable dataset from 571 to 1,500+.

2. **Kill one pipeline.** Adopt the MFCC-based pipeline as canonical. Run it on the full maximized dataset. Generate one honest set of numbers with temporal train/test splits.

3. **Add time-of-day as a feature.** It's free, glucose has strong circadian patterns, and you're not using it.

### Short-term (Months 1-3): Ship What Works -- $10-20K

4. **Ship the ONVOX app with calibration requirement.** Target Dexcom Stelo / Abbott Lingo users. Be honest: "Track your glucose trend between CGM readings." Do not claim no-calibration glucose estimation until population r>0.3.

5. **Collect real-world data from app users.** This creates the flywheel: more users, better population model, eventually enabling reduced or no calibration.

### Medium-term (Months 3-6): Scale -- $15-25K

6. **Recruit 50+ external participants with OTC CGMs.** This is the experiment that tests whether the mass-market thesis is true (population r>0.3) or false (forever calibration-dependent).

7. **Fine-tune the transformer backbone** once you have 1,000+ samples. Before that, it will overfit.

### Long-term (Months 6-12): Validate -- $50-150K

8. **Run clinical validation study** (n=200, IRB-approved) only after product-market fit is confirmed and population model shows real signal.

---

## Appendix A: Participant & Data Inventory

### Currently in Pipeline (7 participants, 571 matched samples)

| Participant | Folder | Audio Files | Matched Samples | Glucose Unit | CGM Source |
|-------------|--------|-------------|-----------------|-------------|------------|
| Wolf | Wolf/ | 947 | 175 | mg/dL | FreeStyle Libre |
| Sybille | Sybille/ | 546 | 55 | mg/dL | FreeStyle Libre |
| Anja | Anja/ | 91 | 84 | mg/dL | FreeStyle Libre |
| Margarita | Margarita/ | 108 | 108 | mmol/L | FreeStyle Libre |
| Vicky | Vicky/ | 79 | 79 | mmol/L | FreeStyle Libre |
| Steffen | Steffen_Haeseli/ | 39 | 38 | mmol/L | FreeStyle Libre |
| Lara | Lara/ | 32 | 32 | mmol/L | FreeStyle Libre |

### Not in Pipeline but Have Data (3+ participants)

| Participant | Audio | CGM | Why Not Used |
|-------------|-------|-----|-------------|
| Darav | Yes (Nov21_finished/) | 6 CSVs, mg/dL | Not integrated |
| Joao | Yes (Nov21/) | 2 CSVs, mg/dL | Not integrated |
| Alvar | Yes (.waptt files) | 1 CSV, mg/dL | Not integrated |
| Christian_L | Yes (folder exists) | Number 8 CSV | Not integrated |

### Audio Only, No CGM Match Yet

| Participant | Audio Files | Potential CGM Match |
|-------------|-------------|-------------------|
| Bruno | ~100 (conv_audio) | Possibly one of the orphaned numbered CSVs |
| Valerie | ~114 (conv_audio) | Possibly one of the orphaned numbered CSVs |
| Edoardo | ~46 (conv_audio) | Possibly one of the orphaned numbered CSVs |
| Jacky | ~31 (.waptt, .opus) | Possibly one of the orphaned numbered CSVs |
| R_Rodolfo | Unknown (conv_audio) | Possibly one of the orphaned numbered CSVs |

### Orphaned CGM CSVs at Root Level

| File | Size | Identified Person |
|------|------|------------------|
| Number_1Nov23_1_glucose_4-1-2024.csv | 56 KB | Unknown |
| Number_3Nov_23_glucose_4-1-2024.csv | 93 KB | Unknown |
| Number_4Nov_23_glucose_4-1-2024.csv | 95 KB | Unknown |
| Number_5Nov_23_glucose_4-1-2024.csv | 92 KB | Boriska Molnar |
| Number_X6Nov_25_glucose_4-1-2024.csv | 57 KB | Unknown (X6 indicates issue) |
| Number_11Nov_29_glucose_4-1-2024.csv | 95 KB | Unknown |
| Number_12Nov_29_glucose_4-1-2024.csv | 83 KB | Bopo del Valle |
| Number_13Dec_10_glucose_4-1-2024.csv | 75 KB | Unknown |
| Number_14Dec_10_glucose_4-1-2024.csv | 95 KB | Berna Agar |

### Unlocated Data

| Participant | Known Data | Status |
|-------------|-----------|--------|
| Carmen | 4,268 CGM records (Portuguese FreeStyle Libre 3) | Audio not found in project directory |

---

## Appendix B: Market Data Sources

### Voice Biomarker Market
- SNS Insider: Vocal Biomarkers Market Size, January 2026
- BIS Research: Voice Biomarkers in Healthcare 2025
- Precedence Research: Vocal Biomarkers Market Report

### Diabetes and Glucose Monitoring
- IDF Diabetes Atlas, 11th Edition (2024)
- Mordor Intelligence: Continuous Glucose Monitoring Market Report
- Grand View Research: US OTC Continuous Glucose Monitoring Devices Market
- Sequenex: State of the Global CGM Market 2025
- CDC National Diabetes Statistics Report

### Consumer Pricing and WTP
- Dexcom Stelo pricing (MedTech Dive)
- Abbott Lingo and FreeStyle Libre pricing (GoodRx)
- CGM app cost comparison (SNAQ.ai)
- PREPARE 4 CGM willingness-to-pay study (Annals of Family Medicine, 2025)

### Digital Health APIs and B2B Pricing
- Redox public pricing documentation
- Glooko B2B model analysis
- MarketsandMarkets: Healthcare Data Monetization Market, 2025
- MarketGenics: Healthcare API Market Growth 2025

### Clinical Accuracy Standards
- ISO 15197:2013 standard for blood glucose monitoring systems
- FDA guidance for blood glucose monitoring test systems (2016)
- FDA General Wellness Policy guidance (January 2026)
- PMC: MARD and ISO 15197 relationship analysis
- PMC: Measures of Accuracy for Continuous Glucose Monitoring

### Corporate Wellness
- Grand View Research: Corporate Wellness Market Report
- ADA: Economic Costs of Diabetes in the US (2022)

### Voice-Glucose Research
- Klick Inc. / Mayo Clinic: Voice-based T2D prediction studies
- Colive Voice / ADA: Voice AI algorithm for Type 2 diabetes prediction
- Frontiers in Clinical Diabetes and Healthcare: Voice-based prediabetes prediction (2025)

---

*Document generated February 8, 2026. Version 2.0 incorporates honest assessment of pipeline contradictions, corrected performance claims, and revised priority ordering based on actual data analysis. All market data reflects most recent available sources.*
