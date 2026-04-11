# ONVOX AutoResearch — Autonomous Experiment Loop

Use this program to run autonomous model discovery on the ONVOX research dataset.

**Goal**: Find personalized glucose estimation algorithms that improve accuracy using voice features matched to CGM labels.

**Key finding**: There is NO population-level voice-glucose signal (r = -0.098, 0/22 stages pass gate). The product is personal adaptation. The scoring formula reflects this: 85% weight on personal MAE, 15% on population.

## Scope

- Use existing paired voice/CGM data via `config.yaml` participants.
- Optimize for **personal model accuracy** (pers_mae, pers_r).
- Population metrics are tracked but down-weighted.

## Primary Objective

Lower `pers_mae` (mg/dL) and raise `pers_r`, while keeping models practical for deployment.

Secondary priorities:
- Better temporal validation (`temp_mae` should not exceed `pers_mae`)
- Higher signal gate pass rate (r > 0.3 AND improvement > 10% AND p < 0.05)
- Simpler configurations if quality is similar

## Scoring Formula

```
selection_score = balance - pers_r_bonus + temporal_penalty + signal_gate_penalty + temp_r_penalty

balance             = 0.85 * pers_mae + 0.15 * pop_mae
pers_r_bonus        = max(0, pers_r - 0.1) * 3.0
temporal_penalty    = max(0, temp_mae - pers_mae)
signal_gate_penalty = max(0, 0.30 - signal_gate_pass_rate) * 3.0
temp_r_penalty      = max(0, 0.05 - temp_r) * 2.0
```

Lower is better.

## In-scope Files

- `hyperparameter_sweep.py` (main search loop, quick/full search spaces)
- `research/models/train.py` (model definitions / defaults)
- `research/features/*` (feature extraction variants)
- `config.yaml` (participants, matching, feature/model options)
- `onvox_bridge/*` (production data sync and promotion gates)

## Constraints

- Do not modify raw participant data files.
- Keep all evaluation on matched voice/CGM pairs from existing pipeline.
- Favor reproducible settings and explicit seeds where possible.
- Keep output under `output/`.

## Search Policy

### ONVOX Memory Priors

- Early-cycle preference:
  - Cycles 1-4: temporal-aware feature sets + robust linear models first
  - Cycles 5-14: prioritize Ridge / BayesianRidge with n_mfcc in 13 or 20
- Feature priors:
  - Prioritize `mfcc+spectral`, `mfcc+spectral+pitch`, `mfcc+spectral+pitch+temporal`
- After warm-up, explore freely but keep priors in stage-1 ranking.

### CGM Lag Search

- Include positive and control offsets (-15 to +30 min)
- Prioritize plausible positive lags (voice may lead interstitial CGM due to diffusion delay)
- Retain negative/zero controls to avoid confirmation bias

### Guardrails

- Never fail a cycle due to duplicate proposals; always evaluate an unseen fallback candidate.
- Keep exploration balanced across normalization, feature families, and model types.
- After discovering strong configs, perform one-hop neighbor search around best-kept settings.
- At least 1 LLM-proposed and 1 diversity-forced candidate per batch.
- Cap any single source at 60% of batch to prevent exploration collapse.

## Baseline and Loop

1. Quick baseline: `python hyperparameter_sweep.py --quick`
2. Start autonomous loop: `python autoresearch/autonomous_llm_loop.py --optimizer-mode v2`
3. Monitor via Streamlit dashboard (AutoResearch tab) or GUI monitor
4. Results logged to `output/autoresearch/autonomous_runs_v2.tsv`
5. Passing configs queued in `output/promotion_queue.json` for production deployment
