# autoresearch for TONES (voice + CGM)

Use this program to run autonomous model discovery on the local TONES dataset at:

`C:\Users\whgeb\AutoResearch_WHG\TONES`

Goal: find population and personalized glucose estimation algorithms that improve accuracy using voice features matched to CGM labels.

## Scope

- Treat `TONES` as the task repo.
- Use existing paired voice/CGM data and loaders in `TONES`.
- Prioritize algorithms and feature configurations that improve both:
  - personalized MAE (per participant)
  - population MAE (leave-one-person-out)

## Primary objective

Lower **personalized** error and **time-respecting** error first. Treat **population LOSO**
(`pop_mae` / `pop_r`) as a **secondary** screen: external ONVOX research shows **0/22** stages
passing a strict population signal gate on voice-only, while **personal** models can pass
when data density per user is high. This repo’s `selection_score` still blends both —
use it for balanced search, but interpret **weak population r** as **expected**, not failure.

Secondary priorities:
- better temporal validation (`temp_mae`, walk-forward / chrono)
- stable per-participant performance (avoid one outlier driving the mean)
- simpler configurations if quality is similar
- **compact** feature bundles over maximal `all_features` on small cohorts (overfitting risk)

## SOTA + metric validation mandate

At regular checkpoints, perform a short "SOTA + metric sanity" review before proposing
new search directions:

- Confirm the current optimization target still matches intended use:
  - screening / trend detection
  - personalization after calibration
  - safety-aware behavior in hypo/hyper ranges
- Validate that metric emphasis is appropriate for this dataset:
  - MAE and correlation are necessary but not sufficient
  - include temporal robustness and clinically-oriented checks
- If metric mismatch is detected, update candidate ranking rules before continuing.

## In-scope files (TONES)

- `hyperparameter_sweep.py` (main search loop, quick/full search spaces)
- `sweep_evaluate.py` (evaluation tooling)
- `tones/models/train.py` (model definitions / defaults)
- `tones/features/*` (feature extraction variants)
- `config.yaml` (participants, matching, feature/model options)

## Constraints

- Do not modify raw participant data files.
- Keep all evaluation on matched voice/CGM pairs from existing pipeline.
- Favor reproducible settings and explicit seeds where possible.
- Keep output under `TONES/output/sweep`.

## Baseline and loop

1. Baseline run:
   - `python hyperparameter_sweep.py --quick`
2. Read:
   - `output/sweep/sweep_summary.csv`
   - `output/sweep/sweep_results.csv`
3. Identify best:
   - personalized config (min `pers_mae`)
   - population config (min `pop_mae`)
   - balanced config (e.g. mean of `pers_mae` and `pop_mae`)
4. Propose one targeted change at a time, then rerun quick sweep.
5. Keep changes only if they improve target metrics or clearly simplify with no regression.

## Guardrails (updated)

- Never fail a cycle due to duplicate proposals; always evaluate an unseen fallback candidate.
- Keep exploration balanced across:
  - normalization (`none`, `zscore`, `rank`)
  - feature families (`mfcc_only`, `mfcc+spectral`, `mfcc+spectral+pitch`, and temporal/vq variants)
- After discovering strong configs, perform one-hop neighbor search around best-kept settings
  (change one axis at a time: model, MFCC count, normalization, or feature family).
- Log and monitor `error` rate; if it exceeds 20% over recent cycles, switch to deterministic
  unseen-candidate mode until stability recovers.
- Prefer text/code models for planning loops; avoid vision-first models as primary planner.

## ONVOX / production ML learnings (aligned with autonomous search)

Synthesize these when steering the loop (full detail lives in product `ML_LEARNINGS` docs):

| Finding | Implication for this repo |
|--------|---------------------------|
| Population signal gate **0/22** passes on broad voice-only experiments | Do **not** expect `pop_r` to turn strongly positive; prioritize **pers_*** and **temp_*** trends. |
| **Personal** GP / Ridge with physics-aware features can pass gate (e.g. MAE ~7, r~0.78 on dense user) | Dense per-user data matters more than feature count; favor **regularized** models and **moderate** MFCC + spectral + pitch paths. |
| **Shuffled** CV **inflates** scores vs LOSO / temporal | Autoresearch uses **time-respecting** personalized CV and temporal splits — keep that honest protocol. |
| HuBERT / contrastive / 68-dim nonlinear stacks | **Deprioritize** unless you add strict leakage checks; production stages showed **no** population signal and contrastive had **leakage** when eval was shuffled. |
| VAD, augmentation, `all_features` on **~800** samples | **Harmful or neutral** — VAD removes informative pauses; augmentation breaks F0/spectral cues; huge stacks **overfit**. |
| CGM **rate of change** as feature | **Never** — label leakage at inference. |
| Mel HTK vs Slaney mismatch | If MFCCs mix scales across users, metrics can be **anti-correlated**; prefer **one** extraction pipeline per run (`config`/librosa settings). |
| Improvement gate uses **relative** MAE drop vs **mean** baseline | Matches `pct_improvement` in `hyperparameter_sweep.py` (already consistent). |

**Improvement priorities (external, ordered):** more calibrations per user \> mel-filter policy for scarce users \> GP transfer \> expand compact feature sets \> windowed models only with long clips.

## ONVOX memory priors (search policy)

Apply these priors in autonomous search policy:

- Early-cycle preference:
  - cycles 1-4: temporal-aware feature sets + robust linear models first
  - cycles 5-14: prioritize `Ridge` / `BayesianRidge` with `n_mfcc` in `13` or `20`
- Feature priors:
  - prioritize `mfcc+spectral`, `mfcc+spectral+pitch`, `mfcc+spectral+pitch+temporal`
- Continue exploring other models/configs after warm-up, but keep priors in stage-1 ranking.
- Keep objective safety-aware:
  - preserve temporal robustness penalties
  - include clinical quality checks (e.g., Clarke A+B behavior, calibration bias)
- Add explicit signal gate tracking:
  - participant-level gate: `r > 0.3` AND `improvement > 10%` AND **permutation** `p_value_perm < 0.05` on OOF preds
  - penalize candidates with very low gate pass rate.
- `selection_score` weights **personal** MAE higher than population (0.65 / 0.35) given weak population voice-only signal.
- Optional models/features: `PhysicsGP` (GPR with train caps in personalized CV; LOSO/temporal use `BayesianRidge` surrogate), `personal_10` compact 10-D bundle (MFCC0–1 means, centroid, F0, jitter/shimmer/HNR/f0_cv).
- Prefer honest temporal protocol:
  - chronological 80/20 holdout for small participant histories
  - walk-forward validation for larger histories.
- Explicitly search CGM/voice temporal offset:
  - include positive and control offsets (e.g., -15..+30 min)
  - prioritize plausible positive lags (voice may lead interstitial CGM due to diffusion delay)
  - retain some negative/zero controls to avoid confirmation bias.

## Suggested search directions

- Feature sets:
  - MFCC count (`13`, `20`, then wider if promising)
  - add/remove spectral, temporal, and voice-quality blocks
  - compare within-speaker normalization strategies
- Algorithms:
  - linear regularized models (Ridge, BayesianRidge)
  - nonlinear models (SVR, RandomForest, GradientBoosting, KNN)
- Validation:
  - prioritize improvements that hold in temporal split, not only CV

## Output format for each cycle

Report:
- best personalized algorithm + settings + MAE/r
- best population algorithm + settings + MAE/r
- balanced recommendation for deployment
- participants that regress, if any
- next single experiment to run

At checkpoint cycles, also report:
- why the current metric bundle is or is not sufficient
- what metric weight/penalty adjustments were applied
