# AutoResearch_WHG

TONES voice→glucose research workspace: autonomous LLM-driven experiment loop, hyperparameter sweep, and monitoring tools.

## Layout

- **`TONES/`** — Primary application code (`hyperparameter_sweep.py`, `tones/`, `autoresearch/`).
- **`autoresearch_algo/`** — Mirrored task scripts and docs for portability.
- **`memory from onvox/`** — Project memory / rationale notes.

## Quick start (autonomous loop)

From `TONES/autoresearch/`:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_tones_autonomous.ps1 -Background
```

GUI monitor:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_tones_autonomous.ps1 -GuiMonitor
```

See `TONES/autoresearch/RESEARCH_PLAN_AND_STATUS.md` for methodology and status.

## Git note

If this tree was created from repos that had their own `.git` folders, those were renamed to `.git_repo_backup/` at convert time so a single top-level repository could track everything. Remove the backup folders locally when you no longer need the old history.

**Participant CGM/voice folders under `TONES/` are gitignored** (sensitive). Keep data only on your machine; clone the repo elsewhere and restore `config.yaml` paths to your local data layout.

### Push to GitHub

1. Create a **new** repository on GitHub (HTTPS). A **private** repo is recommended.
2. In this directory:

```powershell
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

Use a [personal access token](https://github.com/settings/tokens) as the password if prompted for HTTPS, or configure SSH and use `git@github.com:YOUR_USER/YOUR_REPO.git`.

## Requirements

Python 3.11+, local [Ollama](https://ollama.com) for LLM proposals, and TONES `config.yaml` / data paths as documented in the TONES project.
