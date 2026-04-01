#!/usr/bin/env python3
"""
Autonomous TONES experiment loop driven by a local Ollama LLM.

This script proposes one experiment at a time, evaluates it against
matched CGM-voice pairs, logs results, and keeps iterating.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from itertools import product
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

# TONES root = parent of this file's directory
TONES_ROOT = Path(__file__).resolve().parent.parent
if str(TONES_ROOT) not in sys.path:
    sys.path.insert(0, str(TONES_ROOT))

from tones.config import load_config  # type: ignore
from hyperparameter_sweep import (  # type: ignore
    FEATURE_COMBOS,
    MODEL_NAMES,
    extract_features_config,
    evaluate_personalized,
    evaluate_population,
    evaluate_temporal,
    load_all_audio,
)


NORM_METHODS = ["none", "zscore", "rank"]
N_MFCC_OPTIONS = [8, 13, 20, 30, 40]
# Finer positive lags for CGM–voice alignment experiments (strategy: neighbor grid near sweet spots).
CGM_LAG_OPTIONS_MIN = [-15, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60]
ONVOX_PRIOR_MODELS = {"Ridge", "BayesianRidge", "SVR"}
ONVOX_PRIOR_FEATURES = {
    "mfcc+spectral",
    "mfcc+spectral+pitch",
    "mfcc+spectral+pitch+temporal",
    "personal_10",
}
ONVOX_PRIOR_MFCC = {13, 20}
DEFAULT_MODEL_PREF = [
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "llama3.1:latest",
    "mistral:7b",
    "deepseek-r1:7b",
    "phi4:latest",
]
SOURCE_NAMES = {
    "llm",
    "neighbor",
    "tight_neighbor",
    "wild",
    "underexplored",
    "fallback",
    "diversity",
}
MIN_LLM_PER_BATCH = 1
MIN_DIVERSITY_PER_BATCH = 1
# Exploit top keep rows: small perturbations of lag / mfcc / norm / linear model / feature family.
MIN_TIGHT_NEIGHBOR_PER_BATCH = 2
# Novel high-variance configs (nonlinear models, rare bundles); may bypass cycle policy when needed.
MIN_WILD_PER_BATCH = 1
MAX_SOURCE_FRAC = 0.60

# Perturb lags only when the sum is still an allowed CGM lag option.
TIGHT_LAG_DELTAS = (-10, -5, 5, 10)
# Prefer these for "wild" exploratory slots (innovation / distance from typical Ridge keeps).
WILD_MODEL_BIAS = (
    "PhysicsGP",
    "ExtraTrees",
    "GradientBoosting",
    "SVR",
    "KNN",
    "Huber",
    "RandomForest",
    "Lasso",
    "ElasticNet",
)
WILD_FEATURE_BIAS = (
    "all_features",
    "personal_10",
    "mfcc_only",
    "mfcc+spectral+pitch+temporal",
    "mfcc+spectral+pitch+vq",
    "mfcc+spectral+pitch",
)

# Bump when feature extraction / lag logic changes so caches stay valid.
FEATURE_EXTRACT_CACHE_VERSION = "3"
EVAL_CACHE_SCHEMA = 4
FEATURE_CACHE_MAX_ENTRIES = 32
_feature_cache_lock = threading.Lock()
_feature_extract_cache: "OrderedDict[tuple, Dict[str, Dict]]" = OrderedDict()

LOG_FIELDS = [
    "timestamp",
    "cycle",
    "llm_model",
    "status",
    "source",
    "model_name",
    "n_mfcc",
    "cgm_lag_min",
    "feature_key",
    "normalization",
    "exp_key",
    "pers_mae",
    "pers_r",
    "pop_mae",
    "pop_r",
    "temp_mae",
    "temp_r",
    "pers_mard",
    "pop_mard",
    "temp_mard",
    "pop_clarke_ab_pct",
    "pop_bias",
    "temp_bias",
    "signal_gate_pass_rate",
    "signal_gate_penalty",
    "balance",
    "selection_score",
    "temporal_penalty",
    "correlation_penalty",
    "eval_phase",
    "failure_tags",
    "ab_tag",
    "participants",
    "notes",
]

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string"},
        "n_mfcc": {"type": "integer"},
        "cgm_lag_min": {"type": "integer"},
        "normalization": {"type": "string"},
        "feature_key": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "model_name",
        "n_mfcc",
        "cgm_lag_min",
        "normalization",
        "feature_key",
        "rationale",
    ],
}


@dataclass
class EvalResult:
    model_name: str
    n_mfcc: int
    cgm_lag_min: int
    feature_key: str
    normalization: str
    pers_mae: float
    pers_r: float
    pop_mae: float
    pop_r: float
    temp_mae: float
    temp_r: float
    pers_mard: float
    pop_mard: float
    temp_mard: float
    pop_clarke_ab_pct: float
    pop_bias: float
    temp_bias: float
    signal_gate_pass_rate: float
    signal_gate_penalty: float
    balance: float
    selection_score: float
    temporal_penalty: float
    correlation_penalty: float
    eval_phase: str
    n_participants: int
    notes: str


@dataclass
class Candidate:
    source: str
    model_name: str
    n_mfcc: int
    cgm_lag_min: int
    feature_key: str
    normalization: str
    exp_key: str
    rationale: str


def _flags_from_feature_key(feature_key: str) -> Dict[str, bool]:
    cfg = FEATURE_COMBOS[feature_key]
    return {
        "include_spectral": bool(cfg["include_spectral"]),
        "include_pitch": bool(cfg["include_pitch"]),
        "use_vq": bool(cfg["use_vq"]),
        "use_temporal": bool(cfg["use_temporal"]),
    }


def _list_ollama_models() -> List[str]:
    proc = subprocess.run(
        ["ollama", "list"],
        cwd=str(TONES_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []
    names = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def pick_local_llm(preferred: Optional[str]) -> str:
    installed = _list_ollama_models()
    if not installed:
        raise RuntimeError("No local Ollama models are installed.")

    def _is_healthy(model_name: str) -> bool:
        proc = subprocess.run(
            ["ollama", "show", model_name],
            cwd=str(TONES_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0

    if preferred:
        # Vision-first models are poor planners for this text-only optimization loop.
        if preferred.lower().startswith("llava"):
            fallback = "qwen2.5-coder:7b"
            if fallback in installed and _is_healthy(fallback):
                return fallback
        if preferred in installed:
            if not _is_healthy(preferred):
                raise RuntimeError(
                    f"Requested model '{preferred}' exists but is unhealthy. "
                    f"Try: ollama pull {preferred}"
                )
            return preferred
        raise RuntimeError(
            f"Requested model '{preferred}' is not installed. Installed: {installed}"
        )
    for model in DEFAULT_MODEL_PREF:
        if model in installed and _is_healthy(model):
            return model
    # Last resort: pick first healthy model from installed list.
    for model in installed:
        if _is_healthy(model):
            return model
    raise RuntimeError("No healthy local Ollama model found. Try re-pulling one model.")


_OLLAMA_HTTP: Dict[str, float] = {
    "timeout_sec": 300.0,
    "retries": 3.0,
    "retry_sleep_sec": 2.0,
}


def configure_ollama_http(timeout_sec: int, retries: int, retry_sleep_sec: float) -> None:
    """Set HTTP behavior for `call_ollama_chat` (called from main after argparse)."""
    _OLLAMA_HTTP["timeout_sec"] = float(max(30, timeout_sec))
    _OLLAMA_HTTP["retries"] = float(max(1, retries))
    _OLLAMA_HTTP["retry_sleep_sec"] = float(max(0.5, retry_sleep_sec))


def call_ollama_chat(
    model: str, prompt: str, timeout_sec: Optional[int] = None, schema: Optional[Dict] = None
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You design ML experiments. Return only strict JSON. "
                    "No markdown, no explanations outside JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    if schema is not None:
        payload["format"] = schema
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t = float(timeout_sec) if timeout_sec is not None else _OLLAMA_HTTP["timeout_sec"]
    retries = int(_OLLAMA_HTTP["retries"])
    sleep_s = float(_OLLAMA_HTTP["retry_sleep_sec"])
    last_err: Optional[BaseException] = None
    body: Optional[str] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=t) as resp:
                body = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}. Body: {err_body[:600]}"
            ) from exc
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            last_err = exc
            if attempt + 1 >= retries:
                break
            time.sleep(sleep_s)
    if body is None:
        if isinstance(last_err, urllib.error.URLError):
            raise RuntimeError(
                "Could not reach Ollama at http://127.0.0.1:11434. Is 'ollama serve' running?"
            ) from last_err
        raise RuntimeError(
            f"Ollama request failed after {retries} attempt(s) (timeout or connection error). Last: {last_err!r}"
        ) from last_err
    data = json.loads(body)
    return data["message"]["content"]


def _extract_json(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _all_candidate_configs() -> List[Tuple[str, str, int, str, str, int]]:
    """Return all valid experiment candidates as tuples.

    Tuple format:
      (exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min)
    """
    candidates: List[Tuple[str, str, int, str, str, int]] = []
    for model_name, n_mfcc, normalization, feature_key, cgm_lag_min in product(
        MODEL_NAMES,
        N_MFCC_OPTIONS,
        NORM_METHODS,
        FEATURE_COMBOS.keys(),
        CGM_LAG_OPTIONS_MIN,
    ):
        exp_key = f"{model_name}|{n_mfcc}|{feature_key}|{normalization}|lag{cgm_lag_min}"
        candidates.append(
            (exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min)
        )
    return candidates


def _pick_untried_candidate(
    tried_keys: set, cycle: int
) -> Optional[Tuple[str, str, int, str, str, int]]:
    """Pick an unseen candidate deterministically based on cycle index."""
    remaining = [c for c in _all_candidate_configs() if c[0] not in tried_keys]
    if not remaining:
        return None
    idx = (cycle - 1) % len(remaining)
    return remaining[idx]


def _underexplored_normalization(ok_rows: List[Dict[str, str]]) -> Optional[str]:
    counts = {n: 0 for n in NORM_METHODS}
    for row in ok_rows:
        norm = str(row.get("normalization", ""))
        if norm in counts:
            counts[norm] += 1
    if not counts:
        return None
    return min(counts, key=lambda k: counts[k])


def _underexplored_feature(ok_rows: List[Dict[str, str]]) -> Optional[str]:
    counts = {k: 0 for k in FEATURE_COMBOS.keys()}
    for row in ok_rows:
        fk = str(row.get("feature_key", ""))
        if fk in counts:
            counts[fk] += 1
    if not counts:
        return None
    return min(counts, key=lambda k: counts[k])


def _model_family(model_name: str) -> str:
    m = str(model_name)
    if m in {"Ridge", "BayesianRidge", "ElasticNet", "Lasso"}:
        return "linear"
    if m in {"RandomForest", "ExtraTrees"}:
        return "tree"
    if m in {"SVR"}:
        return "kernel"
    if m in {"Huber"}:
        return "robust"
    return "other"


def _best_rows(history: List[Dict[str, str]], top_n: int = 5) -> List[Dict[str, str]]:
    keep_rows: List[Tuple[float, Dict[str, str]]] = []
    for row in history:
        if row.get("status") != "keep":
            continue
        try:
            sc = float(row.get("selection_score", row.get("balance", "nan")))
            if math.isfinite(sc):
                keep_rows.append((sc, row))
        except ValueError:
            continue
    keep_rows.sort(key=lambda x: x[0])
    return [r for _, r in keep_rows[:top_n]]


def _tight_neighbor_candidates_from_best(
    history: List[Dict[str, str]], tried_keys: set
) -> List[Tuple[str, str, int, str, str, int]]:
    """Small, practical perturbations around top ``keep`` rows (sweet-spot exploitation).

    Varies lag (small deltas), adjacent ``n_mfcc`` on the grid, all norms, Ridge↔BayesianRidge,
    and a short list of nearby feature bundles — not the full Cartesian one-hop space.
    """
    lag_allowed = set(CGM_LAG_OPTIONS_MIN)
    mfcc_order = list(N_MFCC_OPTIONS)
    out: List[Tuple[str, str, int, str, str, int]] = []

    def _feat_tight_variants(base_feat: str) -> List[str]:
        v = [base_feat]
        if base_feat == "mfcc+spectral+pitch":
            for x in (
                "mfcc+spectral+pitch+vq",
                "mfcc+spectral+pitch+temporal",
                "mfcc+spectral",
            ):
                if x in FEATURE_COMBOS and x not in v:
                    v.append(x)
        elif base_feat == "mfcc+spectral+pitch+vq":
            for x in ("mfcc+spectral+pitch", "mfcc+spectral+pitch+temporal"):
                if x in FEATURE_COMBOS and x not in v:
                    v.append(x)
        elif base_feat == "mfcc+spectral":
            for x in ("mfcc+spectral+pitch", "mfcc+spectral+pitch+vq"):
                if x in FEATURE_COMBOS and x not in v:
                    v.append(x)
        return v

    for row in _best_rows(history, top_n=5):
        try:
            base_model = str(row["model_name"])
            base_mfcc = int(float(row["n_mfcc"]))
            base_feat = str(row["feature_key"])
            base_norm = str(row["normalization"])
            base_lag = int(float(row.get("cgm_lag_min", "0")))
        except Exception:
            continue

        mfcc_opts = {base_mfcc}
        if base_mfcc in mfcc_order:
            i = mfcc_order.index(base_mfcc)
            if i > 0:
                mfcc_opts.add(mfcc_order[i - 1])
            if i + 1 < len(mfcc_order):
                mfcc_opts.add(mfcc_order[i + 1])

        lag_opts = {base_lag}
        for d in TIGHT_LAG_DELTAS:
            cand = base_lag + d
            if cand in lag_allowed:
                lag_opts.add(cand)

        model_opts = [base_model]
        if base_model in ("Ridge", "BayesianRidge"):
            for m in ("Ridge", "BayesianRidge"):
                if m != base_model and m in MODEL_NAMES:
                    model_opts.append(m)

        feat_opts = _feat_tight_variants(base_feat)

        for model_name in model_opts:
            for n_mfcc in sorted(mfcc_opts):
                for normalization in NORM_METHODS:
                    for feature_key in feat_opts:
                        for cgm_lag_min in sorted(lag_opts):
                            exp_key = (
                                f"{model_name}|{n_mfcc}|{feature_key}|{normalization}|lag{cgm_lag_min}"
                            )
                            if exp_key in tried_keys:
                                continue
                            out.append(
                                (
                                    exp_key,
                                    model_name,
                                    n_mfcc,
                                    normalization,
                                    feature_key,
                                    cgm_lag_min,
                                )
                            )

    seen = set()
    deduped: List[Tuple[str, str, int, str, str, int]] = []
    for t in out:
        if t[0] in seen:
            continue
        seen.add(t[0])
        deduped.append(t)
    return deduped[:160]


def _neighbor_candidates_from_best(
    history: List[Dict[str, str]], tried_keys: set
) -> List[Tuple[str, str, int, str, str, int]]:
    """Generate unseen one-hop neighbors around best kept configs."""
    neighbors: List[Tuple[str, str, int, str, str, int]] = []
    best = _best_rows(history, top_n=4)
    for row in best:
        try:
            base_model = str(row["model_name"])
            base_mfcc = int(float(row["n_mfcc"]))
            base_feat = str(row["feature_key"])
            base_norm = str(row["normalization"])
            base_lag = int(float(row.get("cgm_lag_min", "0")))
        except Exception:
            continue

        variants = []
        # Change one axis at a time to stay close to a good region.
        for m in MODEL_NAMES:
            if m != base_model:
                variants.append((m, base_mfcc, base_norm, base_feat, base_lag))
        for n in N_MFCC_OPTIONS:
            if n != base_mfcc:
                variants.append((base_model, n, base_norm, base_feat, base_lag))
        for norm in NORM_METHODS:
            if norm != base_norm:
                variants.append((base_model, base_mfcc, norm, base_feat, base_lag))
        for feat in FEATURE_COMBOS.keys():
            if feat != base_feat:
                variants.append((base_model, base_mfcc, base_norm, feat, base_lag))
        for lag in CGM_LAG_OPTIONS_MIN:
            if lag != base_lag:
                variants.append((base_model, base_mfcc, base_norm, base_feat, lag))

        for model_name, n_mfcc, normalization, feature_key, cgm_lag_min in variants:
            exp_key = (
                f"{model_name}|{n_mfcc}|{feature_key}|{normalization}|lag{cgm_lag_min}"
            )
            if exp_key not in tried_keys:
                neighbors.append(
                    (
                        exp_key,
                        model_name,
                        n_mfcc,
                        normalization,
                        feature_key,
                        cgm_lag_min,
                    )
                )

    return neighbors


def _passes_onvox_cycle_policy(
    cycle: int,
    model_name: str,
    n_mfcc: int,
    feature_key: str,
    cgm_lag_min: int,
    *,
    policy_bypass: bool = False,
) -> bool:
    """Apply ONVOX-informed cycle policy to improve early search efficiency.

    Wild exploratory slots set ``policy_bypass=True`` so nonlinear / rare bundles
    are still reachable during warm-up cycles.
    """
    if policy_bypass:
        return True
    if cycle <= 4:
        # Early: emphasize temporal priors and simpler robust models.
        return (
            ("temporal" in feature_key)
            and (model_name in {"Ridge", "BayesianRidge"})
            and (cgm_lag_min in {5, 10, 15, 20})
        )
    if cycle <= 14:
        # Warm-up: keep close to known strong personal model family.
        return (
            model_name in {"Ridge", "BayesianRidge"}
            and n_mfcc in ONVOX_PRIOR_MFCC
            and feature_key in ONVOX_PRIOR_FEATURES
        )
    return True


def _onvox_prior_bonus(
    model_name: str, n_mfcc: int, feature_key: str, cgm_lag_min: int
) -> float:
    bonus = 0.0
    if model_name in ONVOX_PRIOR_MODELS:
        bonus -= 0.4
    if feature_key in ONVOX_PRIOR_FEATURES:
        bonus -= 0.4
    if n_mfcc in ONVOX_PRIOR_MFCC:
        bonus -= 0.2
    # Diffusion-delay prior: positive lag is more plausible than negative lag.
    if cgm_lag_min in {5, 10, 15, 20}:
        bonus -= 0.3
    elif cgm_lag_min < 0:
        bonus += 0.1
    # ONVOX research: full 103-style stacks overfit on small n; prefer compact sets.
    if feature_key == "all_features":
        bonus += 0.45
    return bonus


def _pick_diversity_candidate(
    tried: set, cycle: int, history: List[Dict[str, str]]
) -> Optional[Candidate]:
    eval_rows = [r for r in history if r.get("status") in ("keep", "discard")]
    model_counts: Dict[str, int] = {}
    feat_counts: Dict[str, int] = {}
    norm_counts: Dict[str, int] = {}
    lag_counts: Dict[str, int] = {}
    fam_counts: Dict[str, int] = {}
    for r in eval_rows:
        m = str(r.get("model_name", ""))
        f = str(r.get("feature_key", ""))
        n = str(r.get("normalization", ""))
        lag = str(r.get("cgm_lag_min", "0"))
        if m:
            model_counts[m] = model_counts.get(m, 0) + 1
            fam = _model_family(m)
            fam_counts[fam] = fam_counts.get(fam, 0) + 1
        if f:
            feat_counts[f] = feat_counts.get(f, 0) + 1
        if n:
            norm_counts[n] = norm_counts.get(n, 0) + 1
        lag_counts[lag] = lag_counts.get(lag, 0) + 1

    pool = [
        c
        for c in _all_candidate_configs()
        if c[0] not in tried and _passes_onvox_cycle_policy(cycle, c[1], c[2], c[4], c[5])
    ]
    if not pool:
        return None

    scored: List[Tuple[float, Tuple[str, str, int, str, str, int]]] = []
    for c in pool:
        exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min = c
        family = _model_family(model_name)
        novelty = 0.0
        novelty += 1.00 / (1.0 + model_counts.get(model_name, 0))
        novelty += 0.90 / (1.0 + feat_counts.get(feature_key, 0))
        novelty += 0.70 / (1.0 + norm_counts.get(normalization, 0))
        novelty += 0.80 / (1.0 + lag_counts.get(str(cgm_lag_min), 0))
        novelty += 0.90 / (1.0 + fam_counts.get(family, 0))
        prior = -_onvox_prior_bonus(model_name, n_mfcc, feature_key, cgm_lag_min)
        score = novelty + 0.15 * prior
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: min(12, len(scored))]
    _, pick = top[(cycle - 1) % len(top)]
    exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min = pick
    return Candidate(
        source="diversity",
        model_name=model_name,
        n_mfcc=n_mfcc,
        cgm_lag_min=cgm_lag_min,
        feature_key=feature_key,
        normalization=normalization,
        exp_key=exp_key,
        rationale="diversity_forced_slot",
    )


def _pick_wild_candidate(
    tried: set, cycle: int, history: List[Dict[str, str]]
) -> Optional[Candidate]:
    """High-variance exploratory configs (novel models/bundles/lags); policy may bypass warm-up."""
    keep_models: set = set()
    keep_feats: set = set()
    for r in history:
        if r.get("status") != "keep":
            continue
        keep_models.add(str(r.get("model_name", "")))
        keep_feats.add(str(r.get("feature_key", "")))

    pool = [
        c
        for c in _all_candidate_configs()
        if c[0] not in tried
        and _passes_onvox_cycle_policy(
            cycle, c[1], c[2], c[4], c[5], policy_bypass=True
        )
    ]
    if not pool:
        return None

    scored: List[Tuple[float, Tuple[str, str, int, str, str, int]]] = []
    for c in pool:
        _, model_name, _, _, feature_key, cgm_lag_min = c
        wild = 0.0
        if model_name not in keep_models:
            wild += 2.0
        if model_name in WILD_MODEL_BIAS:
            wild += 2.2
        if feature_key not in keep_feats:
            wild += 1.1
        if feature_key in WILD_FEATURE_BIAS:
            wild += 1.4
        if cgm_lag_min < 0 or abs(cgm_lag_min) >= 20:
            wild += 0.6
        if _model_family(model_name) in {"tree", "kernel", "other"}:
            wild += 0.8
        scored.append((wild, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    span = min(48, max(1, len(scored)))
    top = scored[:span]
    _, pick = top[(cycle * 17 + 5) % len(top)]
    exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min = pick
    return Candidate(
        source="wild",
        model_name=model_name,
        n_mfcc=n_mfcc,
        cgm_lag_min=cgm_lag_min,
        feature_key=feature_key,
        normalization=normalization,
        exp_key=exp_key,
        rationale="guardrail_wild_explore",
    )


def _pick_tight_injection_candidate(
    tried: set, cycle: int, history: List[Dict[str, str]]
) -> Optional[Candidate]:
    """Single tight neighbor for explicit batch floors (may overlap with choose_next path)."""
    cands = [
        n
        for n in _tight_neighbor_candidates_from_best(history, tried)
        if _passes_onvox_cycle_policy(cycle, n[1], n[2], n[4], n[5])
    ]
    if not cands:
        return None
    pick = cands[(cycle * 19 + 7) % len(cands)]
    exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min = pick
    return Candidate(
        source="tight_neighbor",
        model_name=model_name,
        n_mfcc=n_mfcc,
        cgm_lag_min=cgm_lag_min,
        feature_key=feature_key,
        normalization=normalization,
        exp_key=exp_key,
        rationale="guardrail_tight_sweet_spot",
    )


def _source_of_row(row: Dict[str, str]) -> str:
    src = str(row.get("source", "")).strip().lower()
    if src in SOURCE_NAMES:
        return src
    notes = str(row.get("notes", ""))
    if "diversity_forced_slot" in notes:
        return "diversity"
    if "guardrail_neighbor_of_best_keep" in notes:
        return "neighbor"
    if "guardrail_tight_sweet_spot" in notes:
        return "tight_neighbor"
    if "guardrail_wild_explore" in notes:
        return "wild"
    if "guardrail_fallback_unseen_candidate" in notes:
        return "fallback"
    if "guardrail_underexplored_axes" in notes:
        return "underexplored"
    return "llm"


def _propose_llm_candidate(
    llm_model: str, tried: set, best_selection_score: Optional[float], cycle: int, history: List[Dict[str, str]]
) -> Optional[Candidate]:
    for _ in range(5):
        proposal, _ = propose_config(llm_model, tried, best_selection_score, cycle, history)
        proposal["source"] = "llm"
        try:
            source, model_name, n_mfcc, cgm_lag_min, normalization, packed = normalize_proposal(proposal)
        except Exception:
            continue
        feature_key, exp_key = packed.split("|", 1)
        if exp_key in tried:
            continue
        if not _passes_onvox_cycle_policy(cycle, model_name, n_mfcc, feature_key, cgm_lag_min):
            continue
        return Candidate(
            source=source,
            model_name=model_name,
            n_mfcc=n_mfcc,
            cgm_lag_min=cgm_lag_min,
            feature_key=feature_key,
            normalization=normalization,
            exp_key=exp_key,
            rationale=str(proposal.get("rationale", "")).strip()[:300] or "llm_balanced_slot",
        )
    return None


def propose_config(
    llm_model: str,
    tried_keys: set,
    best_selection_score: Optional[float],
    cycle: int,
    history: List[Dict[str, str]],
) -> Tuple[Dict, str]:
    recent_json = json.dumps(history[-8:], indent=2)
    tried_preview = sorted(list(tried_keys))[-20:]
    synthesis = _research_synthesis_block(history)
    prompt = f"""
Design the next glucose-estimation experiment for TONES.

Allowed values:
- n_mfcc: {N_MFCC_OPTIONS}
- cgm_lag_min: {CGM_LAG_OPTIONS_MIN}
- model_name: {MODEL_NAMES}
- normalization: {NORM_METHODS}
- feature_key: one of {list(FEATURE_COMBOS.keys())}

Current cycle: {cycle}
Best selection_score so far (lower is better): {best_selection_score}
Recently evaluated rows:
{recent_json}

Research synthesis (failure tags + lag coverage; use to avoid repeating dead ends and to propose novel axes):
{synthesis}

Already tried experiment keys (avoid duplicates):
{tried_preview}

Priors from ONVOX memory + ML research (22+ stages):
- Population LOSO often shows weak/negative r on voice-only; trustworthy gains are usually
  PERSONAL (per-user) and TIME-ORDERED validation — do not chase population r alone.
- Search strategy: combine TIGHT moves around the best-known ``keep`` configs (same feature family,
  small lag/mfcc/norm tweaks, Ridge/BayesianRidge) with WILD experiments (different model families,
  PhysicsGP, all_features, personal_10, extreme lags) — both are intentionally run each batch.
- Signal gate (routing heuristic): per participant r>0.3 AND >10% relative MAE improvement
  vs mean baseline AND permutation p<0.05 on OOF preds — rare at population level; common failure tags: low_pop_r.
- Favor compact features: mfcc+spectral (+pitch / +temporal / +vq) over "all_features"
  unless evidence says otherwise (high-dim hurt on small cohorts in production research).
- Dead ends (do not prioritize): VAD stripping, pitch/time augmentation, CGM rate-of-change
  as input (label leakage at inference), shuffled CV as "honest" eval, raw energy as proxy
  for level (device-dependent), contrastive/embeddings without strict LOSO.
- What worked elsewhere: strong personal linear/GP tracks, physics-informed length scales,
  temporal priors (time-of-day), quality weighting — here we only have sklearn models;
  prefer Ridge/BayesianRidge and honest temporal metrics.
- Early calibration tends to work best with Ridge/BayesianRidge.
- Strong defaults: n_mfcc in [13, 20] and feature_key in
  ["mfcc+spectral", "mfcc+spectral+pitch", "mfcc+spectral+pitch+temporal"].
- Explore CGM lag; prefer positive lags (+5..+20 min) with controls.

Return JSON only with keys:
{{
        "model_name": "...",
        "n_mfcc": 13,
        "cgm_lag_min": 10,
        "normalization": "zscore",
        "feature_key": "mfcc+spectral+pitch",
  "rationale": "short reason"
}}
"""
    raw = call_ollama_chat(llm_model, prompt, schema=PROPOSAL_SCHEMA)
    data = _extract_json(raw)
    return data, raw


def normalize_proposal(data: Dict) -> Tuple[str, str, int, int, str, str]:
    source = str(data.get("source", "llm")).strip().lower() or "llm"
    model_name = str(data.get("model_name", "")).strip()
    n_mfcc = int(data.get("n_mfcc"))
    cgm_lag_min = int(data.get("cgm_lag_min", 0))
    normalization = str(data.get("normalization", "")).strip()
    feature_key = str(data.get("feature_key", "")).strip()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Invalid model_name: {model_name}")
    if n_mfcc not in N_MFCC_OPTIONS:
        raise ValueError(f"Invalid n_mfcc: {n_mfcc}")
    if cgm_lag_min not in CGM_LAG_OPTIONS_MIN:
        raise ValueError(f"Invalid cgm_lag_min: {cgm_lag_min}")
    if normalization not in NORM_METHODS:
        raise ValueError(f"Invalid normalization: {normalization}")
    if feature_key not in FEATURE_COMBOS:
        raise ValueError(f"Invalid feature_key: {feature_key}")
    if source not in SOURCE_NAMES:
        source = "llm"
    exp_key = f"{model_name}|{n_mfcc}|{feature_key}|{normalization}|lag{cgm_lag_min}"
    return source, model_name, n_mfcc, cgm_lag_min, normalization, feature_key + "|" + exp_key


def choose_next_candidate(
    llm_model: str,
    tried: set,
    best_selection_score: Optional[float],
    cycle: int,
    history: List[Dict[str, str]],
) -> Tuple[Dict, str]:
    """Guardrailed candidate chooser:
    1) Enforce exploration balance (normalization, feature families)
    2) Tight neighbors around best ``keep`` rows, then wider one-hop neighbors
    3) LLM proposal
    4) Deterministic unseen fallback

    Batched mode also enforces minimum ``tight_neighbor`` and ``wild`` slots per batch.
    """
    ok_rows = [r for r in history if r.get("status") in ("keep", "discard")]

    # 1) Balance underexplored axes to avoid rank-only collapse.
    target_norm = _underexplored_normalization(ok_rows)
    target_feat = _underexplored_feature(ok_rows)
    if target_norm and target_feat:
        constrained = [
            c
            for c in _all_candidate_configs()
            if c[0] not in tried
            and c[3] == target_norm
            and c[4] == target_feat
            and _passes_onvox_cycle_policy(cycle, c[1], c[2], c[4], c[5])
        ]
        if constrained:
            exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min = constrained[
                (cycle - 1) % len(constrained)
            ]
            return (
                {
                    "source": "underexplored",
                    "model_name": model_name,
                    "n_mfcc": n_mfcc,
                    "cgm_lag_min": cgm_lag_min,
                    "normalization": normalization,
                    "feature_key": feature_key,
                    "rationale": "guardrail_underexplored_axes",
                },
                exp_key,
            )

    # 2) Exploit around best known configs: tight sweet-spot moves first, then wide one-hop.
    tight = [
        n
        for n in _tight_neighbor_candidates_from_best(history, tried)
        if _passes_onvox_cycle_policy(cycle, n[1], n[2], n[4], n[5])
    ]
    wide = [
        n
        for n in _neighbor_candidates_from_best(history, tried)
        if _passes_onvox_cycle_policy(cycle, n[1], n[2], n[4], n[5])
    ]
    tight_keys = {n[0] for n in tight}
    merged: List[Tuple[Tuple[str, str, int, str, str, int], bool]] = [
        (n, True) for n in tight
    ] + [(n, False) for n in wide if n[0] not in tight_keys]
    if merged:
        pick, is_tight = merged[(cycle - 1) % len(merged)]
        exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min = pick
        return (
            {
                "source": "tight_neighbor" if is_tight else "neighbor",
                "model_name": model_name,
                "n_mfcc": n_mfcc,
                "cgm_lag_min": cgm_lag_min,
                "normalization": normalization,
                "feature_key": feature_key,
                "rationale": (
                    "guardrail_tight_sweet_spot"
                    if is_tight
                    else "guardrail_neighbor_of_best_keep"
                ),
            },
            exp_key,
        )

    # 3) LLM proposal (with duplicate retries).
    for _ in range(4):
        proposal, _ = propose_config(llm_model, tried, best_selection_score, cycle, history)
        source, model_name, n_mfcc, cgm_lag_min, normalization, packed = normalize_proposal(proposal)
        feature_key, exp_key = packed.split("|", 1)
        if exp_key not in tried and _passes_onvox_cycle_policy(
            cycle, model_name, n_mfcc, feature_key, cgm_lag_min
        ):
            proposal["source"] = "llm"
            return proposal, exp_key

    # 4) Last-resort deterministic fallback.
    remaining = [
        c
        for c in _all_candidate_configs()
        if c[0] not in tried and _passes_onvox_cycle_policy(cycle, c[1], c[2], c[4], c[5])
    ]
    fallback = remaining[(cycle - 1) % len(remaining)] if remaining else _pick_untried_candidate(tried, cycle)
    if fallback is None:
        raise RuntimeError("Search space exhausted: no non-duplicate candidate remains.")
    exp_key, model_name, n_mfcc, normalization, feature_key, cgm_lag_min = fallback
    return (
        {
            "source": "fallback",
            "model_name": model_name,
            "n_mfcc": n_mfcc,
            "cgm_lag_min": cgm_lag_min,
            "normalization": normalization,
            "feature_key": feature_key,
            "rationale": "guardrail_fallback_unseen_candidate",
        },
        exp_key,
    )


def propose_candidate_batch(
    llm_model: str,
    tried: set,
    best_selection_score: Optional[float],
    cycle_start: int,
    history: List[Dict[str, str]],
    batch_size: int,
) -> List[Candidate]:
    def _replace_one(preferred_old_sources: List[str], new_cand: Candidate) -> bool:
        for src in preferred_old_sources:
            for idx in range(len(out) - 1, -1, -1):
                if out[idx].source != src:
                    continue
                local_tried.discard(out[idx].exp_key)
                out[idx] = new_cand
                local_tried.add(new_cand.exp_key)
                return True
        return False

    local_tried = set(tried)
    out: List[Candidate] = []
    target_batch = max(batch_size, 1)
    for i in range(target_batch):
        proposal, exp_key = choose_next_candidate(
            llm_model=llm_model,
            tried=local_tried,
            best_selection_score=best_selection_score,
            cycle=cycle_start + i,
            history=history,
        )
        source, model_name, n_mfcc, cgm_lag_min, normalization, packed = normalize_proposal(proposal)
        feature_key, normalized_exp_key = packed.split("|", 1)
        if normalized_exp_key != exp_key:
            exp_key = normalized_exp_key
        if exp_key in local_tried:
            continue
        out.append(
            Candidate(
                source=source,
                model_name=model_name,
                n_mfcc=n_mfcc,
                cgm_lag_min=cgm_lag_min,
                feature_key=feature_key,
                normalization=normalization,
                exp_key=exp_key,
                rationale=str(proposal.get("rationale", "")).strip()[:300] or "v2_batch",
            )
        )
        local_tried.add(exp_key)

    # Force at least one novelty-oriented slot each batch.
    diversity = _pick_diversity_candidate(local_tried, cycle_start + len(out), history)
    src_counts: Dict[str, int] = {}
    for c in out:
        src_counts[c.source] = src_counts.get(c.source, 0) + 1
    if diversity is not None and src_counts.get("diversity", 0) < MIN_DIVERSITY_PER_BATCH:
        if not _replace_one(["neighbor", "underexplored", "fallback", "llm"], diversity):
            if len(out) < target_batch:
                out.append(diversity)
                local_tried.add(diversity.exp_key)
        src_counts["diversity"] = src_counts.get("diversity", 0) + 1

    # Ensure at least one LLM-driven proposal per batch.
    src_counts = {}
    for c in out:
        src_counts[c.source] = src_counts.get(c.source, 0) + 1
    if src_counts.get("llm", 0) < MIN_LLM_PER_BATCH:
        llm_slot = _propose_llm_candidate(
            llm_model=llm_model,
            tried=local_tried,
            best_selection_score=best_selection_score,
            cycle=cycle_start + len(out),
            history=history,
        )
        if llm_slot is not None:
            if not _replace_one(["fallback", "neighbor", "underexplored", "diversity"], llm_slot):
                if len(out) < target_batch:
                    out.append(llm_slot)
                    local_tried.add(llm_slot.exp_key)

    # Floor: tight sweet-spot neighbors (small moves around top ``keep`` rows).
    min_tight = max(1, min(MIN_TIGHT_NEIGHBOR_PER_BATCH, max(1, (target_batch + 2) // 4)))
    for attempt in range(36):
        if sum(1 for c in out if c.source == "tight_neighbor") >= min_tight:
            break
        cand = _pick_tight_injection_candidate(
            local_tried, cycle_start + attempt + 1, history
        )
        if cand is None:
            break
        if cand.exp_key in local_tried:
            continue
        if not _replace_one(
            ["fallback", "underexplored", "neighbor", "diversity", "llm"], cand
        ):
            if len(out) < target_batch:
                out.append(cand)
                local_tried.add(cand.exp_key)
            else:
                break

    # Floor: wild / novel configs (nonlinear models, rare bundles; policy bypass in picker).
    min_wild = max(1, min(MIN_WILD_PER_BATCH, max(1, target_batch // 8)))
    for attempt in range(48):
        if sum(1 for c in out if c.source == "wild") >= min_wild:
            break
        cand = _pick_wild_candidate(
            local_tried, cycle_start + attempt + 200, history
        )
        if cand is None:
            break
        if cand.exp_key in local_tried:
            continue
        if not _replace_one(
            ["fallback", "underexplored", "neighbor", "diversity", "llm"], cand
        ):
            if len(out) < target_batch:
                out.append(cand)
                local_tried.add(cand.exp_key)
            else:
                break

    # Cap source dominance to keep exploration/exploitation balanced.
    src_counts = {}
    for c in out:
        src_counts[c.source] = src_counts.get(c.source, 0) + 1
    cap = max(1, int(math.ceil(target_batch * MAX_SOURCE_FRAC)))
    for dominant in ["underexplored", "neighbor", "fallback", "llm", "diversity"]:
        while src_counts.get(dominant, 0) > cap:
            replacement = None
            if src_counts.get("diversity", 0) < MIN_DIVERSITY_PER_BATCH:
                replacement = _pick_diversity_candidate(local_tried, cycle_start + len(out), history)
            if replacement is None and src_counts.get("llm", 0) < MIN_LLM_PER_BATCH:
                replacement = _propose_llm_candidate(
                    llm_model=llm_model,
                    tried=local_tried,
                    best_selection_score=best_selection_score,
                    cycle=cycle_start + len(out),
                    history=history,
                )
            if replacement is None:
                break
            if not _replace_one([dominant], replacement):
                break
            src_counts[dominant] = max(0, src_counts.get(dominant, 0) - 1)
            src_counts[replacement.source] = src_counts.get(replacement.source, 0) + 1

    return out


def evaluate_one(
    participant_data: Dict[str, Dict],
    model_name: str,
    n_mfcc: int,
    cgm_lag_min: int,
    normalization: str,
    feature_key: str,
    include_temporal: bool = True,
    early_stop_cutoff: Optional[float] = None,
    early_stop_margin: float = 0.0,
    max_participants: Optional[int] = None,
    eval_phase: str = "full",
) -> EvalResult:
    fdata = get_cached_features(
        participant_data,
        cgm_lag_min=cgm_lag_min,
        n_mfcc=n_mfcc,
        feature_key=feature_key,
        normalization=normalization,
    )
    if not fdata:
        raise RuntimeError("Feature extraction returned no valid participant data.")

    # Multi-fidelity: cheap stage uses first N participants (sorted keys) without changing features.
    if max_participants is not None and len(fdata) >= 2:
        cap = min(max(int(max_participants), 2), len(fdata))
        if cap < len(fdata):
            keys = sorted(fdata.keys())[:cap]
            fdata = {k: fdata[k] for k in keys}

    p_res = evaluate_personalized(fdata, model_name, normalization_method=normalization)
    pop_res = evaluate_population(fdata, model_name, normalization_method=normalization)
    t_res = (
        evaluate_temporal(fdata, model_name, normalization_method=normalization)
        if include_temporal
        else {}
    )

    if not p_res or not pop_res:
        raise RuntimeError("Evaluation returned empty personalized or population result.")

    pers_mae = mean([v["mae"] for v in p_res.values()])
    pers_r = mean([v["r"] for v in p_res.values()])
    pers_mard = mean([float(v.get("mard", 0.0)) for v in p_res.values()])
    pop_mae = float(pop_res["mae"])
    pop_r = float(pop_res["r"])
    pop_mard = float(pop_res.get("mard", float("nan")))
    pop_clarke_ab_pct = float(pop_res.get("clarke_ab_pct", float("nan")))
    pop_bias = float(pop_res.get("bias", float("nan")))
    # Weight personal MAE higher than population (ONVOX: population signal often weak).
    balance = 0.65 * pers_mae + 0.35 * pop_mae

    # ONVOX signal gate (routing heuristic; not multiplicity-corrected across folds):
    # per participant: r>0.3 AND improvement>10% AND permutation p<0.05 on OOF preds.
    gate_total = 0
    gate_pass = 0
    for v in p_res.values():
        gate_total += 1
        p_perm = float(v.get("p_value_perm", 1.0))
        if (
            float(v.get("r", 0.0)) > 0.3
            and float(v.get("pct_improvement", 0.0)) > 10.0
            and p_perm < 0.05
        ):
            gate_pass += 1
    signal_gate_pass_rate = (gate_pass / gate_total) if gate_total else 0.0
    # Penalize low signal evidence; no penalty once >=30% participants pass gate.
    signal_gate_penalty = max(0.0, 0.30 - signal_gate_pass_rate) * 3.0
    pop_r_penalty = max(0.0, 0.10 - pop_r) * 4.0
    clarke_penalty = (
        max(0.0, 95.0 - pop_clarke_ab_pct) * 0.05
        if math.isfinite(pop_clarke_ab_pct)
        else 0.5
    )

    # Safe early-stop pruning:
    # Minimum achievable selection_score if temporal_penalty and temp_r_penalty were zero:
    # balance + pop_r_penalty + clarke_penalty + signal_gate_penalty.
    # Compare that to the best kept score (early_stop_cutoff). Do not double-count clarke.
    optimistic_lower_bound = (
        balance + pop_r_penalty + clarke_penalty + signal_gate_penalty
    )
    should_early_stop = (
        include_temporal
        and early_stop_cutoff is not None
        and math.isfinite(early_stop_cutoff)
        and optimistic_lower_bound > (early_stop_cutoff + max(early_stop_margin, 0.0))
    )
    if should_early_stop:
        temp_mae = float("nan")
        temp_r = float("nan")
        temp_mard = float("nan")
        temp_bias = float("nan")
        temporal_penalty = 0.0
        temp_r_penalty = 0.0
        correlation_penalty = pop_r_penalty + clarke_penalty + signal_gate_penalty
        selection_score = optimistic_lower_bound
        notes = "early_stop_no_temporal"
        return EvalResult(
            model_name=model_name,
            n_mfcc=n_mfcc,
            cgm_lag_min=cgm_lag_min,
            feature_key=feature_key,
            normalization=normalization,
            pers_mae=pers_mae,
            pers_r=pers_r,
            pop_mae=pop_mae,
            pop_r=pop_r,
            temp_mae=temp_mae,
            temp_r=temp_r,
            pers_mard=pers_mard,
            pop_mard=pop_mard,
            temp_mard=temp_mard,
            pop_clarke_ab_pct=pop_clarke_ab_pct,
            pop_bias=pop_bias,
            temp_bias=temp_bias,
            signal_gate_pass_rate=signal_gate_pass_rate,
            signal_gate_penalty=signal_gate_penalty,
            balance=balance,
            selection_score=selection_score,
            temporal_penalty=temporal_penalty,
            correlation_penalty=correlation_penalty,
            eval_phase=eval_phase,
            n_participants=len(p_res),
            notes=notes,
        )

    temp_mae = mean([v["mae"] for v in t_res.values()]) if t_res else float("nan")
    temp_r = mean([v["r"] for v in t_res.values()]) if t_res else float("nan")
    temp_mard = mean([float(v.get("mard", 0.0)) for v in t_res.values()]) if t_res else float("nan")
    temp_bias = mean([float(v.get("bias", 0.0)) for v in t_res.values()]) if t_res else float("nan")

    # Multi-objective selection score:
    # - keep MAE balance as core target
    # - penalize temporal leakage (temp_mae worse than CV balance)
    # - penalize weak/negative correlation for population and temporal robustness
    temporal_penalty = (
        max(0.0, temp_mae - balance) if (include_temporal and math.isfinite(temp_mae)) else 0.0
    )
    temp_r_penalty = (
        max(0.0, 0.05 - temp_r) * 2.0
        if (include_temporal and math.isfinite(temp_r))
        else 0.0
    )
    correlation_penalty = pop_r_penalty + temp_r_penalty + clarke_penalty + signal_gate_penalty
    selection_score = balance + temporal_penalty + correlation_penalty

    return EvalResult(
        model_name=model_name,
        n_mfcc=n_mfcc,
        cgm_lag_min=cgm_lag_min,
        feature_key=feature_key,
        normalization=normalization,
        pers_mae=pers_mae,
        pers_r=pers_r,
        pop_mae=pop_mae,
        pop_r=pop_r,
        temp_mae=temp_mae,
        temp_r=temp_r,
        pers_mard=pers_mard,
        pop_mard=pop_mard,
        temp_mard=temp_mard,
        pop_clarke_ab_pct=pop_clarke_ab_pct,
        pop_bias=pop_bias,
        temp_bias=temp_bias,
        signal_gate_pass_rate=signal_gate_pass_rate,
        signal_gate_penalty=signal_gate_penalty,
        balance=balance,
        selection_score=selection_score,
        temporal_penalty=temporal_penalty,
        correlation_penalty=correlation_penalty,
        eval_phase=eval_phase,
        n_participants=len(p_res),
        notes="ok" if include_temporal else "stage1_fast",
    )


def _safe_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _ts_to_epoch_seconds(value) -> Optional[float]:
    try:
        if isinstance(value, datetime):
            return float(value.timestamp())
        return float(datetime.fromisoformat(str(value)).timestamp())
    except Exception:
        return None


def _apply_cgm_lag(
    participant_data: Dict[str, Dict], cgm_lag_min: int
) -> Dict[str, Dict]:
    if cgm_lag_min == 0:
        return participant_data
    shifted: Dict[str, Dict] = {}
    lag_s = float(cgm_lag_min) * 60.0
    for name, pdata in participant_data.items():
        ts_list = pdata.get("timestamps", [])
        glucose = np.asarray(pdata.get("glucose", []), dtype=float)
        t = np.array([_ts_to_epoch_seconds(x) for x in ts_list], dtype=float)
        # fallback to original when timestamps are not parseable.
        if len(t) != len(glucose) or len(t) == 0 or np.any(np.isnan(t)):
            shifted[name] = dict(pdata)
            continue
        order = np.argsort(t)
        x = t[order]
        y = glucose[order]
        target = t + lag_s
        y_shifted = np.interp(target, x, y, left=float(y[0]), right=float(y[-1]))
        copied = dict(pdata)
        copied["glucose"] = y_shifted
        shifted[name] = copied
    return shifted


def _fingerprint_participant_data(data: Dict[str, Dict]) -> str:
    """Stable hash of raw cohort (paths + glucose) for feature-cache keys."""
    h = hashlib.sha256()
    for name in sorted(data.keys()):
        p = data[name]
        paths = p.get("audio_paths", [])
        g = np.asarray(p.get("glucose", []), dtype=float)
        h.update(name.encode("utf-8"))
        h.update(str(len(paths)).encode("ascii", errors="replace"))
        if g.size:
            h.update(np.ascontiguousarray(g).tobytes())
    return h.hexdigest()[:40]


def get_cached_features(
    participant_data: Dict[str, Dict],
    cgm_lag_min: int,
    n_mfcc: int,
    feature_key: str,
    normalization: str,
) -> Dict[str, Dict]:
    """Re-use feature matrices for identical (data, lag, mfcc, feature bundle, norm)."""
    fp = _fingerprint_participant_data(participant_data)
    key = (
        FEATURE_EXTRACT_CACHE_VERSION,
        fp,
        cgm_lag_min,
        n_mfcc,
        feature_key,
        normalization,
    )
    with _feature_cache_lock:
        if key in _feature_extract_cache:
            _feature_extract_cache.move_to_end(key)
            return _feature_extract_cache[key]
    flags = _flags_from_feature_key(feature_key)
    lagged_data = _apply_cgm_lag(participant_data, cgm_lag_min)
    fdata = extract_features_config(
        lagged_data,
        n_mfcc=n_mfcc,
        include_spectral=flags["include_spectral"],
        include_pitch=flags["include_pitch"],
        use_vq=flags["use_vq"],
        use_temporal=flags["use_temporal"],
        normalization=normalization,
        feature_key=feature_key,
    )
    with _feature_cache_lock:
        _feature_extract_cache[key] = fdata
        while len(_feature_extract_cache) > FEATURE_CACHE_MAX_ENTRIES:
            _feature_extract_cache.popitem(last=False)
    return fdata


def default_row(cycle: int, llm_model: str) -> Dict[str, str]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cycle": str(cycle),
        "llm_model": llm_model,
        "status": "error",
        "source": "",
        "model_name": "",
        "n_mfcc": "",
        "cgm_lag_min": "",
        "feature_key": "",
        "normalization": "",
        "exp_key": "",
        "pers_mae": "",
        "pers_r": "",
        "pop_mae": "",
        "pop_r": "",
        "temp_mae": "",
        "temp_r": "",
        "pers_mard": "",
        "pop_mard": "",
        "temp_mard": "",
        "pop_clarke_ab_pct": "",
        "pop_bias": "",
        "temp_bias": "",
        "signal_gate_pass_rate": "",
        "signal_gate_penalty": "",
        "balance": "",
        "selection_score": "",
        "temporal_penalty": "",
        "correlation_penalty": "",
        "eval_phase": "",
        "failure_tags": "",
        "ab_tag": "",
        "participants": "",
        "notes": "",
    }


def row_from_eval(
    cycle: int,
    llm_model: str,
    status: str,
    source: str,
    exp_key: str,
    result: EvalResult,
    notes: str,
    ab_tag: str = "",
) -> Dict[str, str]:
    row = default_row(cycle=cycle, llm_model=llm_model)
    ft = ",".join(failure_tags(result))
    row.update(
        {
            "status": status,
            "source": source,
            "model_name": result.model_name,
            "n_mfcc": str(result.n_mfcc),
            "cgm_lag_min": str(result.cgm_lag_min),
            "feature_key": result.feature_key,
            "normalization": result.normalization,
            "exp_key": exp_key,
            "pers_mae": f"{result.pers_mae:.4f}",
            "pers_r": f"{result.pers_r:.4f}",
            "pop_mae": f"{result.pop_mae:.4f}",
            "pop_r": f"{result.pop_r:.4f}",
            "temp_mae": f"{result.temp_mae:.4f}" if math.isfinite(result.temp_mae) else "",
            "temp_r": f"{result.temp_r:.4f}" if math.isfinite(result.temp_r) else "",
            "pers_mard": f"{result.pers_mard:.4f}",
            "pop_mard": f"{result.pop_mard:.4f}" if math.isfinite(result.pop_mard) else "",
            "temp_mard": f"{result.temp_mard:.4f}" if math.isfinite(result.temp_mard) else "",
            "pop_clarke_ab_pct": f"{result.pop_clarke_ab_pct:.2f}" if math.isfinite(result.pop_clarke_ab_pct) else "",
            "pop_bias": f"{result.pop_bias:.4f}" if math.isfinite(result.pop_bias) else "",
            "temp_bias": f"{result.temp_bias:.4f}" if math.isfinite(result.temp_bias) else "",
            "signal_gate_pass_rate": f"{result.signal_gate_pass_rate:.4f}",
            "signal_gate_penalty": f"{result.signal_gate_penalty:.4f}",
            "balance": f"{result.balance:.4f}",
            "selection_score": f"{result.selection_score:.4f}",
            "temporal_penalty": f"{result.temporal_penalty:.4f}",
            "correlation_penalty": f"{result.correlation_penalty:.4f}",
            "eval_phase": result.eval_phase,
            "failure_tags": ft,
            "ab_tag": ab_tag,
            "participants": str(result.n_participants),
            "notes": notes[:300],
        }
    )
    return row


def load_cache(cache_path: Path) -> Dict[str, Dict[str, str]]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        ver = data.get("_eval_cache_schema", 1)
        if ver != EVAL_CACHE_SCHEMA:
            print(
                f"[warn] eval_cache schema {ver} != {EVAL_CACHE_SCHEMA}; "
                "ignoring disk cache (metrics or CV may have changed). Delete "
                f"{cache_path.name} to silence this after an intentional upgrade."
            )
            return {}
        out: Dict[str, Dict[str, str]] = {}
        for k, v in data.items():
            if str(k).startswith("_"):
                continue
            if isinstance(v, dict):
                out[str(k)] = dict(v)
        return out
    except Exception:
        pass
    return {}


def save_cache(cache_path: Path, cache: Dict[str, Dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_eval_cache_schema": EVAL_CACHE_SCHEMA, **cache}
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evalresult_from_row(row: Dict[str, str]) -> EvalResult:
    lag_val = int(_safe_float(row.get("cgm_lag_min", "0"), 0.0))
    return EvalResult(
        model_name=str(row.get("model_name", "")),
        n_mfcc=int(float(row.get("n_mfcc", "0"))),
        cgm_lag_min=lag_val,
        feature_key=str(row.get("feature_key", "")),
        normalization=str(row.get("normalization", "")),
        pers_mae=_safe_float(row.get("pers_mae", "")),
        pers_r=_safe_float(row.get("pers_r", "")),
        pop_mae=_safe_float(row.get("pop_mae", "")),
        pop_r=_safe_float(row.get("pop_r", "")),
        temp_mae=_safe_float(row.get("temp_mae", "")),
        temp_r=_safe_float(row.get("temp_r", "")),
        pers_mard=_safe_float(row.get("pers_mard", ""), float("nan")),
        pop_mard=_safe_float(row.get("pop_mard", ""), float("nan")),
        temp_mard=_safe_float(row.get("temp_mard", ""), float("nan")),
        pop_clarke_ab_pct=_safe_float(row.get("pop_clarke_ab_pct", ""), float("nan")),
        pop_bias=_safe_float(row.get("pop_bias", ""), float("nan")),
        temp_bias=_safe_float(row.get("temp_bias", ""), float("nan")),
        signal_gate_pass_rate=_safe_float(row.get("signal_gate_pass_rate", ""), 0.0),
        signal_gate_penalty=_safe_float(row.get("signal_gate_penalty", ""), 0.0),
        balance=_safe_float(row.get("balance", "")),
        selection_score=_safe_float(row.get("selection_score", "")),
        temporal_penalty=_safe_float(row.get("temporal_penalty", "")),
        correlation_penalty=_safe_float(row.get("correlation_penalty", "")),
        eval_phase=str(row.get("eval_phase", "full") or "full"),
        n_participants=int(float(row.get("participants", "0"))),
        notes=str(row.get("notes", "")),
    )


def failure_tags(result: EvalResult) -> List[str]:
    tags: List[str] = []
    if result.pop_r < 0.05:
        tags.append("low_pop_r")
    if math.isfinite(result.temp_r) and result.temp_r < 0.05:
        tags.append("low_temp_r")
    if math.isfinite(result.pop_mard) and result.pop_mard > 15.0:
        tags.append("high_pop_mard")
    if math.isfinite(result.pop_clarke_ab_pct) and result.pop_clarke_ab_pct < 99.0:
        tags.append("low_clarke_ab")
    if result.signal_gate_pass_rate < 0.10:
        tags.append("weak_signal_gate")
    if result.temporal_penalty > 1.0:
        tags.append("high_temporal_penalty")
    if result.correlation_penalty > 1.2:
        tags.append("high_correlation_penalty")
    if result.feature_key == "all_features" and result.selection_score > 18.0:
        tags.append("overfit_tail")
    if result.cgm_lag_min in {25, 35, 40, 45, 60}:
        tags.append("fine_lag_grid")
    return tags


def _best_incumbent_row(history: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Best selection_score among ``keep`` rows (for single-axis A/B tagging)."""
    best: Optional[Dict[str, str]] = None
    best_sc: Optional[float] = None
    for r in history:
        if r.get("status") != "keep":
            continue
        sc = _safe_float(str(r.get("selection_score", "")))
        if not math.isfinite(sc):
            continue
        if best_sc is None or sc < best_sc:
            best_sc = sc
            best = r
    return best


def _ab_tag_for_candidate(
    cand: Candidate, incumbent: Optional[Dict[str, str]]
) -> str:
    if not incumbent:
        return ""
    diffs: List[str] = []
    if cand.model_name != str(incumbent.get("model_name", "")):
        diffs.append("model_name")
    if str(cand.n_mfcc) != str(incumbent.get("n_mfcc", "")).strip():
        diffs.append("n_mfcc")
    if str(cand.cgm_lag_min) != str(incumbent.get("cgm_lag_min", "")).strip():
        diffs.append("cgm_lag_min")
    if cand.feature_key != str(incumbent.get("feature_key", "")):
        diffs.append("feature_key")
    if cand.normalization != str(incumbent.get("normalization", "")):
        diffs.append("normalization")
    if len(diffs) == 1:
        return f"ab:{diffs[0]}"
    return ""


def _ab_tag_from_parts(
    model_name: str,
    n_mfcc: int,
    cgm_lag_min: int,
    feature_key: str,
    normalization: str,
    incumbent: Optional[Dict[str, str]],
) -> str:
    if not incumbent:
        return ""
    c = Candidate(
        source="llm",
        model_name=model_name,
        n_mfcc=n_mfcc,
        cgm_lag_min=cgm_lag_min,
        feature_key=feature_key,
        normalization=normalization,
        exp_key="_",
        rationale="",
    )
    return _ab_tag_for_candidate(c, incumbent)


def _research_synthesis_block(history: List[Dict[str, str]], window: int = 120) -> str:
    """Failure-tag counts and lag coverage for LLM hypothesis generation."""
    rows = [r for r in history[-window:] if r.get("status") in ("keep", "discard", "error")]
    if not rows:
        return "(no prior rows in window)"
    tag_counts: Dict[str, int] = {}
    for r in rows:
        ft = str(r.get("failure_tags", "")).strip()
        if not ft:
            # Fallback: parse fail_tags= from legacy notes
            notes = str(r.get("notes", ""))
            if "fail_tags=" in notes:
                part = notes.split("fail_tags=", 1)[-1].split(";", 1)[0].split(",")
                for t in part:
                    t = t.strip()
                    if t:
                        tag_counts[t] = tag_counts.get(t, 0) + 1
            continue
        for t in ft.split(","):
            t = t.strip()
            if t:
                tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:14]
    lags = sorted(
        {str(r.get("cgm_lag_min", "")) for r in rows if str(r.get("cgm_lag_min", ""))},
        key=lambda x: int(x) if x.lstrip("-").isdigit() else 0,
    )
    return (
        f"failure_tag_counts (approx, last {len(rows)} rows): {top_tags}\n"
        f"cgm_lag_min values seen: {lags}"
    )


def stage1_heuristic_score(candidate: Candidate, history: List[Dict[str, str]]) -> float:
    """Cheap pre-screen score (lower is better), no feature extraction."""
    ok_rows = [r for r in history if r.get("status") in {"keep", "discard"}]
    if not ok_rows:
        return 100.0

    def _axis_stats(axis: str, value: str) -> Tuple[float, int]:
        vals: List[float] = []
        cnt = 0
        for r in ok_rows:
            if str(r.get(axis, "")) != value:
                continue
            sc = _safe_float(str(r.get("selection_score", "")))
            if math.isfinite(sc):
                vals.append(sc)
                cnt += 1
        if not vals:
            return 40.0, 0
        return min(vals), cnt

    m_best, m_cnt = _axis_stats("model_name", candidate.model_name)
    f_best, f_cnt = _axis_stats("feature_key", candidate.feature_key)
    n_best, n_cnt = _axis_stats("normalization", candidate.normalization)
    l_best, l_cnt = _axis_stats("cgm_lag_min", str(candidate.cgm_lag_min))

    # Source bandit weighting:
    # Prefer sources that historically yield keeps, with beta-prior smoothing.
    keep = 0
    total = 0
    for r in ok_rows:
        if _source_of_row(r) != candidate.source:
            continue
        st = str(r.get("status", ""))
        if st in {"keep", "discard"}:
            total += 1
            if st == "keep":
                keep += 1
    alpha = 1.0 + keep
    beta = 1.0 + max(total - keep, 0)
    keep_mean = alpha / (alpha + beta)
    # Lower is better score; negative means preferred.
    source_bandit_bonus = -(keep_mean - 0.35) * 0.8

    novelty_bonus = -0.2 * sum(1 for c in [m_cnt, f_cnt, n_cnt, l_cnt] if c == 0)
    prior_bonus = _onvox_prior_bonus(
        candidate.model_name,
        candidate.n_mfcc,
        candidate.feature_key,
        candidate.cgm_lag_min,
    )
    return (
        0.40 * m_best
        + 0.25 * f_best
        + 0.15 * n_best
        + 0.20 * l_best
        + novelty_bonus
        + prior_bonus
        + source_bandit_bonus
    )


def append_row(tsv_path: Path, row: Dict[str, str]) -> None:
    needs_header = (not tsv_path.exists()) or tsv_path.stat().st_size == 0
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = LOG_FIELDS
    with tsv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        if needs_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def ensure_log_schema(tsv_path: Path) -> None:
    if (not tsv_path.exists()) or tsv_path.stat().st_size == 0:
        return
    try:
        with tsv_path.open("r", encoding="utf-8", newline="") as f:
            first_line = f.readline().strip("\r\n")
    except Exception:
        return
    header = first_line.split("\t") if first_line else []
    if header == LOG_FIELDS:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = tsv_path.with_name(f"{tsv_path.stem}.schema_mismatch_backup_{ts}.tsv")
    tsv_path.replace(backup)
    print(f"[warn] Log schema mismatch; moved old log to: {backup}")


def bootstrap_policy_history_from_backups(
    log_path: Path, max_rows: int = 400
) -> List[Dict[str, str]]:
    parent = log_path.parent
    stem = log_path.stem
    backups = sorted(
        parent.glob(f"{stem}.schema_mismatch_backup_*.tsv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not backups:
        return []
    out: List[Dict[str, str]] = []
    for fp in backups[:4]:
        try:
            with fp.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for raw in reader:
                    row = {k: raw.get(k, "") for k in LOG_FIELDS}
                    if "cgm_lag_min" not in raw:
                        row["cgm_lag_min"] = "0"
                    row["source"] = _source_of_row(row)
                    if row.get("status") in {"keep", "discard", "error"}:
                        out.append(row)
        except Exception:
            continue
        if len(out) >= max_rows:
            break
    return out[-max_rows:]


def read_rows(tsv_path: Path) -> List[Dict[str, str]]:
    if not tsv_path.exists():
        return []
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: List[Dict[str, str]] = []
        for raw in reader:
            out = {k: raw.get(k, "") for k in LOG_FIELDS}
            # Backward compatibility: old logs without cgm_lag_min.
            if "cgm_lag_min" not in raw:
                out["cgm_lag_min"] = "0"
            if "source" not in raw or not str(out.get("source", "")).strip():
                notes = str(out.get("notes", ""))
                if "diversity_forced_slot" in notes:
                    out["source"] = "diversity"
                elif "guardrail_neighbor_of_best_keep" in notes:
                    out["source"] = "neighbor"
                elif "guardrail_fallback_unseen_candidate" in notes:
                    out["source"] = "fallback"
                elif "guardrail_underexplored_axes" in notes:
                    out["source"] = "underexplored"
                else:
                    out["source"] = "llm"
            rows.append(out)
        return rows


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous local-LLM loop for TONES.")
    parser.add_argument("--config", default=None, help="Path to TONES config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (default: best available local fit).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="0 means run forever. Positive integer limits cycles.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.03,
        help="Minimum improvement in selection_score (lower is better) required to mark a run as keep.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=3,
        help="Pause between cycles.",
    )
    parser.add_argument(
        "--status-file",
        default=None,
        help="Path for runtime status JSON (default: output/autoresearch/status.json).",
    )
    parser.add_argument(
        "--pid-file",
        default=None,
        help="Path for loop PID file (default: output/autoresearch/loop.pid).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path for runs TSV log (default: output/autoresearch/autonomous_runs_v2.tsv).",
    )
    parser.add_argument(
        "--optimizer-mode",
        choices=["classic", "v2"],
        default="v2",
        help="classic: one-by-one loop; v2: batched staged parallel search.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=5,
        help="Parallel workers for staged/full evaluation in v2 mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of proposed candidates per v2 iteration.",
    )
    parser.add_argument(
        "--stage1-participants",
        type=int,
        default=4,
        help="Participants used for cheap stage1 scoring in v2 mode.",
    )
    parser.add_argument(
        "--stage1-top-k",
        type=int,
        default=4,
        help="Top stage1 candidates promoted to full evaluation in v2 mode.",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="JSON cache file for completed experiment results in v2 mode.",
    )
    parser.add_argument(
        "--early-stop-margin",
        type=float,
        default=0.4,
        help="Prune candidate if optimistic score is above (best + margin).",
    )
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="Disable safe early-stop pruning of temporal evaluation.",
    )
    parser.add_argument(
        "--ollama-timeout-sec",
        type=int,
        default=300,
        help="HTTP timeout for each Ollama /api/chat request (default 300).",
    )
    parser.add_argument(
        "--ollama-retries",
        type=int,
        default=3,
        help="Retries on timeout or connection errors to Ollama (default 3).",
    )
    parser.add_argument(
        "--ollama-retry-sleep-sec",
        type=float,
        default=2.0,
        help="Seconds to wait between Ollama retries (default 2).",
    )
    args = parser.parse_args()

    configure_ollama_http(
        args.ollama_timeout_sec,
        args.ollama_retries,
        args.ollama_retry_sleep_sec,
    )

    llm_model = pick_local_llm(args.model)
    log_path = (
        Path(args.log_file)
        if args.log_file
        else TONES_ROOT / "output" / "autoresearch" / "autonomous_runs_v2.tsv"
    )
    status_path = (
        Path(args.status_file)
        if args.status_file
        else TONES_ROOT / "output" / "autoresearch" / "status.json"
    )
    pid_path = (
        Path(args.pid_file)
        if args.pid_file
        else TONES_ROOT / "output" / "autoresearch" / "loop.pid"
    )
    cache_path = (
        Path(args.cache_file)
        if args.cache_file
        else TONES_ROOT / "output" / "autoresearch" / "eval_cache.json"
    )

    print(f"[init] Using TONES root: {TONES_ROOT}")
    print(f"[init] Using Ollama model: {llm_model}")
    print(f"[init] Log file: {log_path}")
    print(f"[init] Status file: {status_path}")

    started_at = datetime.now().isoformat(timespec="seconds")
    # Publish startup heartbeat immediately so monitors don't show stale old PID/state.
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    write_json(
        status_path,
        {
            "running": True,
            "pid": os.getpid(),
            "started_at": started_at,
            "llm_model": llm_model,
            "log_path": str(log_path),
            "last_update": started_at,
            "cycle": 0,
            "phase": "starting",
            "best_selection_score": None,
            "status_file": str(status_path),
            "pid_file": str(pid_path),
            "optimizer_mode": args.optimizer_mode,
            "last_result": None,
        },
    )

    write_json(
        status_path,
        {
            "running": True,
            "pid": os.getpid(),
            "started_at": started_at,
            "llm_model": llm_model,
            "log_path": str(log_path),
            "last_update": datetime.now().isoformat(timespec="seconds"),
            "cycle": 0,
            "phase": "loading_data",
            "best_selection_score": None,
            "status_file": str(status_path),
            "pid_file": str(pid_path),
            "optimizer_mode": args.optimizer_mode,
            "last_result": None,
        },
    )

    cfg = load_config(args.config)
    participant_data = load_all_audio(cfg)
    if not participant_data:
        raise RuntimeError("No participant data loaded from config.")
    print(f"[init] Loaded participant groups: {len(participant_data)}")

    ensure_log_schema(log_path)
    history = read_rows(log_path)
    if not history:
        seeded = bootstrap_policy_history_from_backups(log_path, max_rows=500)
        if seeded:
            print(f"[init] Bootstrapped policy history rows from backups: {len(seeded)}")
            history = seeded
    tried = set()
    best_selection_score = None
    for row in history:
        exp_key = row.get("exp_key", "")
        if exp_key:
            tried.add(exp_key)
        try:
            # Prefer multi-objective score if present, else legacy balance.
            bal = float(row.get("selection_score", row.get("balance", "nan")))
            status = row.get("status", "")
            if status == "keep" and (best_selection_score is None or bal < best_selection_score):
                best_selection_score = bal
        except ValueError:
            pass

    cycle = 0
    write_json(
        status_path,
        {
            "running": True,
            "pid": os.getpid(),
            "started_at": started_at,
            "llm_model": llm_model,
            "log_path": str(log_path),
            "last_update": started_at,
            "cycle": 0,
            "best_selection_score": best_selection_score,
            "last_result": None,
        },
    )

    def write_runtime_status(phase: str, cycle_value: int, extra: Optional[Dict] = None) -> None:
        payload = {
            "running": True,
            "pid": os.getpid(),
            "started_at": started_at,
            "last_update": datetime.now().isoformat(timespec="seconds"),
            "llm_model": llm_model,
            "cycle": cycle_value,
            "phase": phase,
            "best_selection_score": best_selection_score,
            "log_path": str(log_path),
            "status_file": str(status_path),
            "pid_file": str(pid_path),
            "optimizer_mode": args.optimizer_mode,
        }
        if extra:
            payload.update(extra)
        try:
            write_json(status_path, payload)
        except Exception:
            pass

    try:
        if args.optimizer_mode == "classic":
            while True:
                cycle += 1
                if args.max_cycles > 0 and cycle > args.max_cycles:
                    print("[done] Reached max cycles.")
                    break

                print(f"\n[cycle {cycle}] Proposing next experiment...")
                write_runtime_status("proposing", cycle)
                row = default_row(cycle=cycle, llm_model=llm_model)

                try:
                    proposal, chosen_exp_key = choose_next_candidate(
                        llm_model=llm_model,
                        tried=tried,
                        best_selection_score=best_selection_score,
                        cycle=cycle,
                        history=history,
                    )

                    source, model_name, n_mfcc, cgm_lag_min, normalization, packed = normalize_proposal(proposal)
                    feature_key, exp_key = packed.split("|", 1)
                    if exp_key != chosen_exp_key:
                        chosen_exp_key = exp_key
                    rationale = str(proposal.get("rationale", "")).strip()[:300]
                    print(
                        "[cycle %d] Evaluating (%s) %s n_mfcc=%s lag=%smin features=%s norm=%s"
                        % (cycle, source, model_name, n_mfcc, cgm_lag_min, feature_key, normalization)
                    )
                    write_runtime_status(
                        "evaluating",
                        cycle,
                        {
                            "candidate": {
                                "source": source,
                                "model_name": model_name,
                                "n_mfcc": n_mfcc,
                                "cgm_lag_min": cgm_lag_min,
                                "feature_key": feature_key,
                                "normalization": normalization,
                                "exp_key": chosen_exp_key,
                            }
                        },
                    )
                    row["source"] = source

                    eval_result = evaluate_one(
                        participant_data=participant_data,
                        model_name=model_name,
                        n_mfcc=n_mfcc,
                        cgm_lag_min=cgm_lag_min,
                        normalization=normalization,
                        feature_key=feature_key,
                        include_temporal=True,
                        early_stop_cutoff=None if args.disable_early_stop else best_selection_score,
                        early_stop_margin=max(args.early_stop_margin, 0.0),
                    )

                    current_score = eval_result.selection_score
                    improved = best_selection_score is None or (
                        current_score <= (best_selection_score - args.min_improvement)
                    )
                    status = "keep" if improved else "discard"
                    if status == "keep":
                        best_selection_score = current_score
                    base_notes = (rationale + ";" + eval_result.notes) if eval_result.notes else (rationale or "ok")
                    if status == "discard":
                        tags = failure_tags(eval_result)
                        if tags:
                            base_notes = f"{base_notes};fail_tags={','.join(tags)}"
                    inc_row = _best_incumbent_row(history)
                    ab = _ab_tag_from_parts(
                        model_name,
                        n_mfcc,
                        cgm_lag_min,
                        feature_key,
                        normalization,
                        inc_row,
                    )
                    row = row_from_eval(
                        cycle=cycle,
                        llm_model=llm_model,
                        status=status,
                        source=source,
                        exp_key=chosen_exp_key,
                        result=eval_result,
                        notes=base_notes,
                        ab_tag=ab,
                    )
                    tried.add(chosen_exp_key)
                    print(
                        "[cycle %d] %s | pers_mae=%.3f pop_mae=%.3f balance=%.3f score=%.3f"
                        % (
                            cycle,
                            status.upper(),
                            eval_result.pers_mae,
                            eval_result.pop_mae,
                            eval_result.balance,
                            eval_result.selection_score,
                        )
                    )
                except Exception as exc:
                    row["status"] = "error"
                    row["notes"] = str(exc)[:500]
                    print(f"[cycle {cycle}] ERROR: {exc}")

                append_row(log_path, row)
                history.append(row)
                write_runtime_status("idle", cycle, {"last_result": row})
                time.sleep(max(args.sleep_seconds, 0))
        else:
            workers = max(args.parallel_workers, 1)
            batch_size = max(args.batch_size, workers)
            top_k = max(1, min(args.stage1_top_k, batch_size))
            cache = load_cache(cache_path)
            if cache:
                for k in cache.keys():
                    tried.add(str(k))
            print(
                f"[init:v2] workers={workers} batch_size={batch_size} top_k={top_k} "
                f"stage1=cheap_eval(n<={max(2, args.stage1_participants)}) cache_entries={len(cache)}"
            )
            while True:
                if args.max_cycles > 0 and cycle >= args.max_cycles:
                    print("[done] Reached max cycles.")
                    break

                write_runtime_status("proposing_batch", cycle)
                candidates = propose_candidate_batch(
                    llm_model=llm_model,
                    tried=tried,
                    best_selection_score=best_selection_score,
                    cycle_start=cycle + 1,
                    history=history,
                    batch_size=batch_size,
                )
                if not candidates:
                    print("[done] No more candidates available.")
                    break

                stage1_scores: Dict[str, float] = {}
                uncached = [c for c in candidates if c.exp_key not in cache]

                if uncached:
                    write_runtime_status(
                        "stage1_cheap_eval",
                        cycle,
                        {"batch_size": len(candidates), "uncached": len(uncached)},
                    )
                    n_sub = max(
                        2,
                        min(int(args.stage1_participants), len(participant_data)),
                    )

                    def _stage1_eval(c: Candidate) -> Tuple[str, float]:
                        try:
                            er = evaluate_one(
                                participant_data,
                                c.model_name,
                                c.n_mfcc,
                                c.cgm_lag_min,
                                c.normalization,
                                c.feature_key,
                                include_temporal=False,
                                early_stop_cutoff=None,
                                early_stop_margin=0.0,
                                max_participants=n_sub,
                                eval_phase="stage1_proxy",
                            )
                            return c.exp_key, er.selection_score
                        except Exception:
                            return c.exp_key, stage1_heuristic_score(c, history)

                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futs = [ex.submit(_stage1_eval, c) for c in uncached]
                        pending_s1 = set(futs)
                        completed_s1 = 0
                        while pending_s1:
                            done_s1, pending_s1 = wait(
                                pending_s1,
                                timeout=15.0,
                                return_when=FIRST_COMPLETED,
                            )
                            # Heartbeat while cheap evals run (stage2 already does this; stage1 could take many minutes).
                            write_runtime_status(
                                "stage1_cheap_eval",
                                cycle,
                                {
                                    "batch_size": len(candidates),
                                    "uncached": len(uncached),
                                    "stage1_completed": completed_s1,
                                    "stage1_total": len(uncached),
                                    "stage1_pending": len(pending_s1),
                                },
                            )
                            for fut in done_s1:
                                ek, sc = fut.result()
                                stage1_scores[ek] = sc
                                completed_s1 += 1

                selected: List[Candidate] = []
                selected.extend([c for c in candidates if c.exp_key in cache])
                scored_candidates = [c for c in uncached if c.exp_key in stage1_scores]
                scored_candidates.sort(key=lambda c: stage1_scores[c.exp_key])
                for c in scored_candidates[:top_k]:
                    selected.append(c)
                selected = selected[:batch_size]
                if not selected:
                    # All failed in stage1; log one error row to keep visibility.
                    cycle += 1
                    err = default_row(cycle=cycle, llm_model=llm_model)
                    err["notes"] = "v2_stage1_all_failed"
                    append_row(log_path, err)
                    history.append(err)
                    write_runtime_status("idle", cycle, {"last_result": err})
                    continue

                write_runtime_status(
                    "stage2_full_eval",
                    cycle,
                    {"selected_full_eval": len(selected), "proposed": len(candidates)},
                )
                full_results: Dict[str, EvalResult] = {}
                full_errors: Dict[str, str] = {}
                to_eval = [c for c in selected if c.exp_key not in cache]
                if to_eval:
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futures = {
                            ex.submit(
                                evaluate_one,
                                participant_data,
                                c.model_name,
                                c.n_mfcc,
                                c.cgm_lag_min,
                                c.normalization,
                                c.feature_key,
                                True,
                                None if args.disable_early_stop else best_selection_score,
                                max(args.early_stop_margin, 0.0),
                            ): c
                            for c in to_eval
                        }
                        pending = set(futures.keys())
                        completed = 0
                        while pending:
                            done, pending = wait(
                                pending,
                                timeout=15.0,
                                return_when=FIRST_COMPLETED,
                            )
                            # Heartbeat while long evaluations are running.
                            write_runtime_status(
                                "stage2_full_eval",
                                cycle,
                                {
                                    "selected_full_eval": len(selected),
                                    "proposed": len(candidates),
                                    "stage2_completed": completed,
                                    "stage2_total": len(to_eval),
                                    "stage2_pending": len(pending),
                                },
                            )
                            for fut in done:
                                cand = futures[fut]
                                try:
                                    full_results[cand.exp_key] = fut.result()
                                except Exception as exc:
                                    full_errors[cand.exp_key] = str(exc)[:300]
                                completed += 1

                new_rows: List[Dict[str, str]] = []
                incumbent_row = _best_incumbent_row(history)
                for cand in selected:
                    cycle += 1
                    if args.max_cycles > 0 and cycle > args.max_cycles:
                        break
                    if cand.exp_key in full_errors:
                        row = default_row(cycle=cycle, llm_model=llm_model)
                        row.update(
                            {
                                "status": "error",
                                "source": cand.source,
                                "model_name": cand.model_name,
                                "n_mfcc": str(cand.n_mfcc),
                                "cgm_lag_min": str(cand.cgm_lag_min),
                                "feature_key": cand.feature_key,
                                "normalization": cand.normalization,
                                "exp_key": cand.exp_key,
                                "notes": f"v2_full_error: {full_errors[cand.exp_key]}",
                            }
                        )
                    else:
                        if cand.exp_key in cache:
                            eval_result = evalresult_from_row(cache[cand.exp_key])
                            note = f"{cand.rationale};cache_hit"
                        else:
                            eval_result = full_results[cand.exp_key]
                            cache[cand.exp_key] = row_from_eval(
                                cycle=0,
                                llm_model=llm_model,
                                status="cached",
                                source=cand.source,
                                exp_key=cand.exp_key,
                                result=eval_result,
                                notes=cand.rationale,
                                ab_tag="",
                            )
                            note = cand.rationale
                        current_score = eval_result.selection_score
                        improved = best_selection_score is None or (
                            current_score <= (best_selection_score - args.min_improvement)
                        )
                        status = "keep" if improved else "discard"
                        if status == "keep":
                            best_selection_score = current_score
                        base_notes = (note + ";" + eval_result.notes) if eval_result.notes else note
                        if status == "discard":
                            tags = failure_tags(eval_result)
                            if tags:
                                base_notes = f"{base_notes};fail_tags={','.join(tags)}"
                        ab = _ab_tag_for_candidate(cand, incumbent_row)
                        row = row_from_eval(
                            cycle=cycle,
                            llm_model=llm_model,
                            status=status,
                            source=cand.source,
                            exp_key=cand.exp_key,
                            result=eval_result,
                            notes=base_notes,
                            ab_tag=ab,
                        )
                    append_row(log_path, row)
                    history.append(row)
                    new_rows.append(row)
                    tried.add(cand.exp_key)

                save_cache(cache_path, cache)
                if new_rows:
                    write_runtime_status("idle", cycle, {"last_result": new_rows[-1]})
                time.sleep(max(args.sleep_seconds, 0))
    finally:
        # Mark stopped state and clean pid file.
        try:
            write_json(
                status_path,
                {
                    "running": False,
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "ended_at": datetime.now().isoformat(timespec="seconds"),
                    "llm_model": llm_model,
                    "cycle": cycle,
                    "best_selection_score": best_selection_score,
                    "log_path": str(log_path),
                    "status_file": str(status_path),
                    "pid_file": str(pid_path),
                },
            )
        except Exception:
            pass
        try:
            if pid_path.exists():
                pid_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
