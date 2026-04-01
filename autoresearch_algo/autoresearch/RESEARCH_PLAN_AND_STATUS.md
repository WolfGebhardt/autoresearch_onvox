# TONES autonomous research — plan, status, and next steps

This document consolidates **goals**, **rationale**, **data quality assumptions**, **what has been implemented**, and **recommended next actions**. Operational run instructions remain in `program_tones.md`.

---

## 1. Overall goals

### 1.1 Scientific / product

- **Primary:** Discover voice-based configurations that **reduce glucose estimation error** on **matched voice–CGM pairs**, with honest **per-user (personalized)** and **time-ordered (temporal)** evaluation.
- **Secondary:** Monitor **population LOSO** (`pop_mae`, `pop_r`) as a **screen**, not the main success criterion. External ONVOX-style evidence suggests **weak or absent population correlation** on voice-only is common; **personal** models can still be useful when **per-user sample density** is high.
- **Safety-aware framing:** Metrics include **Clarke A+B**, **MARD**, **bias**; the loop is a **research screen**, not a clinical device claim.

### 1.2 Engineering

- **Reproducible** pipeline: fixed seeds where applicable, versioned **feature extract** and **eval caches**, logged **TSV** runs.
- **Autonomous search:** LLM-assisted proposals plus **deterministic guardrails** (exploration balance, neighbors, fallbacks).
- **Efficiency:** Parallel evaluation (v2), staged ranking, early-stop pruning of expensive temporal eval when a run cannot beat the best score.

---

## 2. Rationale (why this design)

| Theme | Rationale |
|--------|-----------|
| **Personal vs population** | Voice–glucose signal is often **user-specific**; LOSO aggregates dilute or invert correlation. **`selection_score` weights personal MAE higher than population** (0.65 / 0.35) so the optimizer does not chase `pop_r` alone. |
| **Signal gate** | Per-participant **OOF** checks: **r**, **% improvement vs mean baseline**, and **permutation p-value** on Pearson r (`p_value_perm`). Reduces trusting spurious correlation on small series. |
| **Compact features** | High-dimensional stacks (**`all_features`**) on **small cohorts** risk overfitting; a **10-D `personal_10`** bundle tests parsimony vs full **`mfcc+spectral+pitch+vq`**. |
| **Mel scale** | **Slaney / non-HTK** mel for MFCC (`htk=False` in `MFCCExtractor`) avoids mixing HTK vs Slaney mel definitions across tooling. |
| **Physics-style GP** | **`PhysicsGP`** (sklearn GPR) probes smooth nonlinear structure; **train rows capped** in personalized CV; **LOSO/temporal** use a **BayesianRidge surrogate** because full GP at LOPO scale is prohibitive. |
| **Search diversity** | **Tight neighbors** exploit known **`keep`** sweet spots; **wild** slots force **novel model/feature/lag** combinations with **optional warm-up policy bypass** so exploration does not stall in Ridge-only phases. |

---

## 3. Data quality and assumptions

### 3.1 What the data are

- **Paired samples:** Audio clips aligned to CGM labels within configured **time windows** (`config.yaml` → `matching`).
- **Per-participant series:** Timestamps drive **time-respecting CV** (not i.i.d. shuffled CV for the main personalized path).

### 3.2 Limitations (interpret results accordingly)

| Issue | Implication |
|--------|-------------|
| **Small effective n per user** | High variance in per-user metrics; **mean** personalized MAE can be driven by one participant. |
| **Cohort size ~ hundreds of clips** | **`all_features`** and heavy nonlinear models can **overfit**; watch **`selection_score` tail** and **temporal** penalty. |
| **Device / recording variability** | MFCC + spectral paths are more portable than raw energy; still **within-speaker normalization** matters (`none` / `zscore` / `rank`). |
| **CGM vs voice timing** | **`cgm_lag_min`** sweeps **voice-leading / coincident / lagging** CGM; physiological plausibility favors **small positive lags** in some settings — treat **negative lags** as **controls**, not defaults. |
| **Label noise** | CGM interpolation and matching window add noise; focus on **relative** improvements and **ranking** configs, not absolute mg/dL claims. |

### 3.3 Data hygiene rules (project)

- **Do not** modify raw participant files.
- **Do** use existing loaders and matched pairs only.

---

## 4. What has been implemented (technical)

### 4.1 Metrics and models (`TONES/tones/models/train.py`, `hyperparameter_sweep.py`)

- **`permutation_p_value_pearson(y_true, y_pred, n_perm)`** — default permutations via env **`TONES_PERM_N`** (default 400).
- **`PhysicsGP`** — `GaussianProcessRegressor` with **RBF + WhiteKernel**; personalized CV **subsamples** training rows to **`PHYSICS_GP_MAX_TRAIN` (200)** per fold.
- **`evaluate_personalized`** adds **`p_value_perm`** to per-participant metrics.
- **`evaluate_population` / `evaluate_temporal`:** if **`model_name == "PhysicsGP"`**, fit **`BayesianRidge`** instead (surrogate), recorded in metrics as **`population_surrogate_model` / `temporal_surrogate_model`**.

### 4.2 Features

- **`FEATURE_COMBOS["personal_10"]`:** Full **`mfcc+spectral+pitch+vq`** extract, then **column slice** to 10 names (MFCC0–1 means, centroid, F0 stats, jitter/shimmer/HNR/f0_cv). Incompatible with **`use_temporal`** in the same bundle (slice skipped with warning if temporal is on).
- **`extract_features_config(..., feature_key=...)`** — `run_sweep` passes **`combo_name`**; autonomous **`get_cached_features`** passes **`feature_key`**.
- **`tones/features/mfcc.py`:** **`htk=False`** on **`librosa.feature.mfcc`** and **`melspectrogram`** (explicit Slaney-style path).
- **`config.yaml`:** Comment that mel/MFCC uses **non-HTK** mel.

### 4.3 Autonomous loop (`autoresearch/tones_autonomous_llm_loop.py`)

- **`selection_score`:** **`0.65 * pers_mae + 0.35 * pop_mae`** (+ temporal and correlation penalties as implemented).
- **Signal gate:** **`p_value_perm < 0.05`** with **r** and **relative improvement** thresholds (see code — not parametric multiplicity across all participants).
- **Cache/schema:** **`FEATURE_EXTRACT_CACHE_VERSION = "3"`**, **`EVAL_CACHE_SCHEMA = 4`** (v4 adds structured TSV columns; invalidates older `eval_cache.json` rows).
- **Multi-fidelity stage 1 (v2):** Uncached batch candidates run **`evaluate_one(..., max_participants=stage1_participants, include_temporal=False, eval_phase=stage1_proxy)`** in parallel; **`selection_score`** on that subset ranks them; top **`stage1_top_k`** proceed to **full** eval (temporal on). On exception, falls back to **`stage1_heuristic_score`**.
- **CGM lag grid:** **`CGM_LAG_OPTIONS_MIN`** includes **25, 35, 40, 45, 60** (minute) for finer alignment experiments.
- **TSV columns:** **`eval_phase`** (`full` / `stage1_proxy`), **`failure_tags`** (comma-separated), **`ab_tag`** (`ab:<axis>` when exactly one axis differs from best **`keep`** incumbent).
- **LLM prompt:** **`_research_synthesis_block`** injects failure-tag counts and lag coverage from recent history.
- **Priors:** **`ONVOX_PRIOR_*`** extended with **`personal_10`** where applicable.
- **Throughput defaults (launcher + argparse):** **`ParallelWorkers=5`**, **`BatchSize=10`**, **`Stage1TopK=4`** (see `start_tones_autonomous.ps1`).
- **Candidate sources:** **`tight_neighbor`** (small moves around top **`keep`** rows), **`neighbor`** (wide one-hop), **`wild`** (novel configs, **policy bypass** in picker), plus **`underexplored`**, **`diversity`**, **`llm`**, **`fallback`**.
- **Batch floors:** Minimum **`tight_neighbor`** and **`wild`** slots per v2 batch (see **`MIN_TIGHT_NEIGHBOR_PER_BATCH`**, **`MIN_WILD_PER_BATCH`** in code).

### 4.4 Snapshots and mirror

- **Evolution snapshots** under `TONES/output/autoresearch/evolution_snapshots/<timestamp>/` (TSV, `eval_cache.json`, `status.json`, `manifest.json`).
- **`autoresearch_algo/`** git repo mirrors key scripts (`hyperparameter_sweep.py`, `tones/models/train.py`, `autoresearch/tones_autonomous_llm_loop.py`, etc.) — **canonical task code** for day-to-day work is under **`TONES/`**.

---

## 5. Empirical status (from logged runs — illustrative)

Values drift as new rows append; recompute from **`autonomous_runs_v2.tsv`** for current numbers.

- **Best unique `keep` seen (lowest `selection_score`):** roughly **`~10.96`** — **`BayesianRidge`**, **`n_mfcc=20`**, **`mfcc+spectral+pitch`**, **`rank`**, **`cgm_lag_min=30`** (example **`exp_key`:** `BayesianRidge|20|mfcc+spectral+pitch|rank|lag30`).
- **`personal_10`:** Many trials; **no `keep`** in the analyzed window — typically **worse `selection_score`** than the best full bundle on aggregate (parsimony vs information tradeoff on this cohort).
- **`all_features`:** **High variance** in **`selection_score`** (bad tail ~20+); **not** a stable default for small-n screening without stronger regularization or a different objective weight.

**Interpretation:** Treat these as **historical search outcomes**, not final science — new **`tight_neighbor` / `wild`** logic will change the proposal mix going forward.

---

## 6. What we should do next

### 6.1 Search and evaluation (near term)

1. **Restart the autonomous loop** after code changes so **new proposal logic** and **cache versions** load.
2. **Fair A/B for `personal_10`:** Fix **`BayesianRidge`**, **`n_mfcc=20`**, **same `normalization` and `cgm_lag_min`** as the current best **`mfcc+spectral+pitch`** row, then compare — many logged **`personal_10`** rows varied other axes simultaneously.
3. **Neighbor grid on the incumbent:** **`cgm_lag`** options now include **25, 30, 35, 40, 45, 60**; compare against best **`keep`** using **`ab_tag`** for single-axis attribution.
4. **Monitor failure tags** in notes (`low_pop_r`, `weak_signal_gate`, etc.) — if **`wild`** runs dominate **`discard`**, consider lowering **`MIN_WILD_PER_BATCH`** or tightening wild scoring.

### 6.2 Code / ops (optional)

- **`--stage1-participants`:** Wired to **real** cheap eval (subset of participants, no temporal) for v2 ranking; heuristic remains **fallback** on eval exceptions.
- **Root git:** Consider **`git init`** at `AutoResearch_WHG` or symlink strategy so **`TONES/`** changes are versioned alongside **`autoresearch_algo`**.
- **Operational:** “More calibrations per user” remains **data collection**, not a code switch.

### 6.3 Documentation hygiene

- Keep **`program_tones.md`** for **how to run** and guardrails.
- Update **this file** when the **best `keep`**, **objective weights**, or **major pipeline** pieces change.

### 6.4 Longer-term (not yet implemented)

- **SMAC3 / BOHB** wrapping `run_sweep` or the autonomous objective; warm-start from best **`keep`** rows.
- **Per-user stacking ensemble** (meta-learner on OOF preds); **nested hold-out** for unbiased winner reporting.
- **Benjamini–Hochberg** across batch multiplicity for **`p_value_perm`** (family-wise error).
- **Residual / `delta_from_baseline`** features; **structured GP kernels** (periodic + linear on time).
- **Per–model-family worker pools** to avoid slow models blocking fast ones.

---

## 7. Key file map

| Area | Path |
|------|------|
| Sweep & eval | `TONES/hyperparameter_sweep.py` |
| Models / permutation | `TONES/tones/models/train.py` |
| MFCC / mel | `TONES/tones/features/mfcc.py` |
| Autonomous loop | `TONES/autoresearch/tones_autonomous_llm_loop.py` |
| Launcher | `TONES/autoresearch/start_tones_autonomous.ps1` |
| Config | `TONES/config.yaml` |
| Run logs | `TONES/output/autoresearch/autonomous_runs_v2.tsv`, `eval_cache.json`, `status.json` |

---

## 8. Revision note

- **Created:** 2026-03-31 (consolidates goals, implemented work through tight/wild search and ONVOX-aligned metrics).
- **2026-04-01:** Strategy pass — multi-fidelity stage1, expanded lags, structured TSV tags + LLM synthesis block, **`EVAL_CACHE_SCHEMA = 4`**.
- **Update** when objectives or pipeline semantics change materially.
