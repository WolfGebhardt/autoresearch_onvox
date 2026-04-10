# autoresearch_algo

Autonomous search framework for voice-based glucose estimation experiments using matched CGM-voice data.

This repository contains the current working loop, monitor tools, and core evaluation logic used to run a balanced exploration/exploitation search over model, feature, normalization, and CGM lag hypotheses.

## What This Includes

- Autonomous loop with:
  - local LLM proposal generation (Ollama),
  - history-aware candidate routing,
  - exploration/exploitation balancing,
  - source bandit weighting,
  - cross-run bootstrap memory,
  - early-stop pruning and evaluation cache.
- Monitoring:
  - terminal dashboard,
  - GUI dashboard with plots, targets, and snapshots,
  - watchdog auto-health restart mode.
- Evaluation and metrics:
  - personalized, population, and temporal validation,
  - fold-safe normalization,
  - clinical metrics (MARD, Clarke A+B, bias),
  - failure taxonomy tagging for discarded runs.

## Repository Structure

- `autoresearch/tones_autonomous_llm_loop.py` - main autonomous optimizer loop.
- `autoresearch/start_tones_autonomous.ps1` - launcher/manager (run, stop, status, watchdog, monitors).
- `autoresearch/monitor_autonomous_progress.py` - terminal monitor and health checks.
- `autoresearch/monitor_autonomous_gui.py` - GUI monitor with time-series plots.
- `autoresearch/program_tones.md` - system prompt and domain priors.
- `hyperparameter_sweep.py` - core feature extraction/evaluation entry points.
- `research/models/train.py` - model registry and metric computation.
- `research/evaluation/metrics.py` - Clarke Error Grid utilities.
- `research/evaluation/temporal_cv.py` - temporal validation utilities.
- `research/features/normalize.py` - normalization functions.
- `research/features/temporal.py` - temporal feature engineering.
- `research/config.py` and `config.yaml` - config loading and dataset paths.

## Quick Start (Windows / PowerShell)

1. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

2. Install and run Ollama, and pull a planning model (example):

```powershell
ollama pull qwen2.5-coder:7b
```

3. Update dataset paths in `config.yaml` to your local data.

4. Start autonomous loop in background:

```powershell
powershell -ExecutionPolicy Bypass -File ".\autoresearch\start_tones_autonomous.ps1" -Background
```

5. Check status:

```powershell
powershell -ExecutionPolicy Bypass -File ".\autoresearch\start_tones_autonomous.ps1" -Status
```

## Monitoring Commands

- Terminal live watch:

```powershell
powershell -ExecutionPolicy Bypass -File ".\autoresearch\start_tones_autonomous.ps1" -Watch
```

- Popup monitor window:

```powershell
powershell -ExecutionPolicy Bypass -File ".\autoresearch\start_tones_autonomous.ps1" -PopupWatch
```

- GUI monitor:

```powershell
powershell -ExecutionPolicy Bypass -File ".\autoresearch\start_tones_autonomous.ps1" -GuiMonitor
```

- Watchdog (auto-restart on repeated health failures):

```powershell
powershell -ExecutionPolicy Bypass -File ".\autoresearch\start_tones_autonomous.ps1" -Watchdog -HealthMaxStaleMinutes 8 -WatchdogIntervalSec 30 -WatchdogConsecutiveFails 3
```

## Optimization Objective (Current Behavior)

The loop minimizes a multi-objective `selection_score` combining:

- MAE balance (personalized + population),
- temporal robustness penalties,
- correlation penalties,
- clinical quality penalties (e.g., Clarke A+B),
- signal gate penalty.

This is a constrained, practical objective for real progress tracking. It is not a single-loss model fit.

## Outputs

By default, outputs are written to:

- `output/autoresearch/status.json`
- `output/autoresearch/autonomous_runs_v2.tsv`
- `output/autoresearch/eval_cache.json`
- `output/autoresearch/loop.log`
- `output/autoresearch/loop.err.log`

## Notes

- This repo is currently tuned for local experimentation with CPU-heavy evaluation and local LLM planning.
- If you run multiple launchers/watchdogs at once, progress can degrade due to process contention. Prefer one loop + one watchdog instance.
