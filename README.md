# ONVOX AutoResearch

Autonomous LLM-driven experiment loop for voice-glucose research: hyperparameter sweep, personal-focused optimization, and monitoring tools.

## Layout

- **`autoresearch_algo/`** — Core code: autonomous loop, research framework, production bridge, monitoring.
  - `autoresearch/` — Autonomous LLM loop + monitors
  - `research/` — Research framework (config, data loaders, features, evaluation, models)
  - `onvox_bridge/` — Production integration (Supabase sync, promotion gates)
  - `hyperparameter_sweep.py` — Systematic sweep across model/feature/normalization space
- **`memory from onvox/`** — Project memory / rationale notes.

## Quick start (autonomous loop)

From `autoresearch_algo/autoresearch/`:

```bash
python autonomous_llm_loop.py --optimizer-mode v2
```

Or on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_autonomous.ps1 -Background
```

GUI monitor:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_autonomous.ps1 -GuiMonitor
```

## Optimization Objective

The loop uses a personal-focused scoring formula:

```
selection_score = balance - pers_r_bonus + temporal_penalty + signal_gate_penalty + temp_r_penalty

where:
  balance        = 0.85 * pers_mae + 0.15 * pop_mae
  pers_r_bonus   = max(0, pers_r - 0.1) * 3.0
  temporal_penalty = max(0, temp_mae - pers_mae)
  signal_gate_penalty = max(0, 0.30 - pass_rate) * 3.0
  temp_r_penalty = max(0, 0.05 - temp_r) * 2.0
```

Population MAE is down-weighted to 15% because 22 research stages confirmed no population-level voice-glucose signal exists (r=-0.098). The product is personal adaptation.

## Requirements

Python 3.11+, local [Ollama](https://ollama.com) for LLM proposals, and `config.yaml` / data paths as documented in the project.

## Data note

Participant CGM/voice folders are gitignored (sensitive). Keep data only on your machine; update `config.yaml` paths to your local data layout.
