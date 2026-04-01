#!/usr/bin/env python3
"""
Edge-inspired audio optimization for TONES glucose modeling.

Searches over:
- Window sizes (100-3000 ms)
- Feature families (MFCC, MFE, log-spectrogram)
- Frequency ranges and MFCC counts
- Light augmentation choices (including optional reverb)
- Model families for population and personalized tasks

Outputs:
- output/edge_opt/leaderboard_personalized.csv
- output/edge_opt/leaderboard_population.csv
- output/edge_opt/leaderboard_calibration.csv
- output/edge_opt/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneOut, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from tones.config import get_base_dir, load_config
from tones.data.loaders import load_participant_data
from tones.models.train import compute_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str  # mfcc, mfe, spec
    window_ms: int
    hop_ratio: float
    n_mfcc: int
    n_mels: int
    n_fft: int
    fmin: int
    fmax: int


@dataclass(frozen=True)
class AugSpec:
    name: str  # none, light, light_reverb
    add_noise_std: float
    gain_db_max: float
    shift_ms_max: int
    reverb: bool


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("numba", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_model(name: str):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "BayesianRidge":
        return BayesianRidge()
    if name == "SVR":
        return SVR(C=10, gamma="scale", kernel="rbf")
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    if name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=400,
            early_stopping=True,
            random_state=42,
        )
    raise ValueError(f"Unknown model: {name}")


def load_wave_cache(cfg: Dict, participants_filter: Optional[List[str]] = None) -> Dict[str, Dict]:
    base_dir = get_base_dir(cfg)
    matching_cfg = cfg.get("matching", {})
    participants = cfg.get("participants", {})
    if participants_filter:
        participants = {k: v for k, v in participants.items() if k in participants_filter}

    cache: Dict[str, Dict] = {}
    for name, pcfg in participants.items():
        if not pcfg.get("glucose_csv"):
            continue
        df = load_participant_data(name, pcfg, base_dir, matching_cfg)
        if df.empty or len(df) < 20:
            continue

        waves, yvals, ts = [], [], []
        for row in df.itertuples(index=False):
            try:
                y, _ = librosa.load(str(row.audio_path), sr=16000, mono=True)
                if len(y) < 1600:
                    continue
                waves.append(y.astype(np.float32))
                yvals.append(float(row.glucose_mg_dl))
                ts.append(str(row.audio_timestamp))
            except Exception:
                continue

        if len(waves) < 20:
            continue

        cache[name] = {
            "waves": waves,
            "glucose": np.asarray(yvals, dtype=np.float32),
            "timestamps": np.asarray(ts),
        }
        logger.info("%s: %d valid waveform/glucose pairs", name, len(waves))

    logger.info("Loaded %d participants for optimization", len(cache))
    return cache


def center_crop_or_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]
    pad_left = (target_len - len(y)) // 2
    pad_right = target_len - len(y) - pad_left
    return np.pad(y, (pad_left, pad_right), mode="reflect")


def lightweight_reverb(y: np.ndarray, sr: int) -> np.ndarray:
    ir_len = max(32, int(0.03 * sr))
    t = np.linspace(0.0, 1.0, ir_len, dtype=np.float32)
    ir = np.exp(-6.0 * t)
    ir[0] = 1.0
    wet = np.convolve(y, ir, mode="full")[: len(y)]
    wet = wet / (np.max(np.abs(wet)) + 1e-8)
    return wet.astype(np.float32)


def augment_audio(y: np.ndarray, sr: int, aug: AugSpec, rng: np.random.Generator) -> np.ndarray:
    out = y.copy()
    if aug.shift_ms_max > 0:
        max_shift = int(sr * (aug.shift_ms_max / 1000.0))
        if max_shift > 0:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            out = np.roll(out, shift)
    if aug.gain_db_max > 0:
        gain_db = float(rng.uniform(-aug.gain_db_max, aug.gain_db_max))
        out = out * (10 ** (gain_db / 20.0))
    if aug.add_noise_std > 0:
        noise = rng.normal(0.0, aug.add_noise_std, size=len(out)).astype(np.float32)
        out = out + noise
    if aug.reverb:
        out = 0.8 * out + 0.2 * lightweight_reverb(out, sr)

    peak = np.max(np.abs(out)) + 1e-8
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def extract_feature(y: np.ndarray, sr: int, spec: FeatureSpec) -> np.ndarray:
    win_len = max(256, int(sr * (spec.window_ms / 1000.0)))
    hop_len = max(64, int(win_len * spec.hop_ratio))

    if spec.family == "mfcc":
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=spec.n_mfcc,
            n_fft=spec.n_fft,
            hop_length=hop_len,
            fmin=spec.fmin,
            fmax=spec.fmax,
        )
        n_frames = mfcc.shape[1]
        delta_width = min(9, n_frames if (n_frames % 2 == 1) else (n_frames - 1))
        if delta_width >= 3:
            d1 = librosa.feature.delta(mfcc, width=delta_width)
            d2 = librosa.feature.delta(mfcc, order=2, width=delta_width)
        else:
            # Very short windows can have too few frames for robust delta estimates.
            d1 = np.zeros_like(mfcc)
            d2 = np.zeros_like(mfcc)
        feats = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.mean(d1, axis=1),
                np.std(d1, axis=1),
                np.mean(d2, axis=1),
                np.std(d2, axis=1),
            ]
        )
        return feats.astype(np.float32)

    if spec.family == "mfe":
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=spec.n_fft,
            hop_length=hop_len,
            n_mels=spec.n_mels,
            fmin=spec.fmin,
            fmax=spec.fmax,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feats = np.concatenate([np.mean(mel_db, axis=1), np.std(mel_db, axis=1)])
        return feats.astype(np.float32)

    if spec.family == "spec":
        stft = np.abs(librosa.stft(y, n_fft=spec.n_fft, hop_length=hop_len)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=spec.n_fft)
        mask = (freqs >= spec.fmin) & (freqs <= spec.fmax)
        stft = stft[mask]
        logp = librosa.power_to_db(stft + 1e-10, ref=np.max)
        feats = np.concatenate([np.mean(logp, axis=1), np.std(logp, axis=1)])
        return feats.astype(np.float32)

    raise ValueError(f"Unsupported family: {spec.family}")


def build_features(
    wave_cache: Dict[str, Dict],
    spec: FeatureSpec,
    aug: AugSpec,
    random_seed: int = 42,
) -> Dict[str, Dict]:
    sr = 16000
    target_len = int(sr * (spec.window_ms / 1000.0))
    rng = np.random.default_rng(random_seed)
    out: Dict[str, Dict] = {}

    for participant, pdata in wave_cache.items():
        X = []
        y = pdata["glucose"]
        ts = pdata["timestamps"]
        for wave in pdata["waves"]:
            cropped = center_crop_or_pad(wave, target_len)
            feat = extract_feature(cropped, sr=sr, spec=spec)
            X.append(feat)
        if not X:
            continue
        out[participant] = {
            "X": np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            "y": y,
            "ts": ts,
            "target_len": target_len,
            "rng_state": rng.bit_generator.state,
            "waves": pdata["waves"],
        }
    return out


def augment_train_split(
    X_train: np.ndarray,
    idx_train: np.ndarray,
    waves: List[np.ndarray],
    spec: FeatureSpec,
    aug: AugSpec,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if aug.name == "none":
        return X_train, idx_train

    sr = 16000
    target_len = int(sr * (spec.window_ms / 1000.0))
    X_aug = []
    for i in idx_train:
        y_aug = augment_audio(center_crop_or_pad(waves[int(i)], target_len), sr, aug, rng)
        X_aug.append(extract_feature(y_aug, sr, spec))
    if not X_aug:
        return X_train, idx_train
    X_all = np.vstack([X_train, np.asarray(X_aug, dtype=np.float32)])
    idx_all = np.concatenate([idx_train, idx_train])
    return X_all, idx_all


def eval_personalized(
    features: Dict[str, Dict],
    spec: FeatureSpec,
    aug: AugSpec,
    model_name: str,
) -> Dict[str, float]:
    maes, rs = [], []
    rng = np.random.default_rng(42)
    for pdata in features.values():
        X = pdata["X"]
        y = pdata["y"]
        waves = pdata["waves"]
        if len(y) < 20:
            continue
        cv = LeaveOneOut() if len(y) <= 50 else KFold(n_splits=10, shuffle=True, random_state=42)
        preds = np.zeros_like(y, dtype=np.float64)
        for train_idx, test_idx in cv.split(X):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_train_aug, idx_all = augment_train_split(X_train, train_idx, waves, spec, aug, rng)
            y_train_aug = y[idx_all]

            pipe = Pipeline([("scaler", RobustScaler()), ("model", get_model(model_name))])
            pipe.fit(X_train_aug, y_train_aug)
            preds[test_idx] = pipe.predict(X[test_idx])

        m = compute_metrics(y, preds)
        maes.append(m["mae"])
        rs.append(m["r"])
    if not maes:
        return {"mae": np.nan, "r": np.nan, "n_participants": 0}
    return {
        "mae": float(np.mean(maes)),
        "r": float(np.mean(rs)),
        "n_participants": len(maes),
    }


def eval_population(
    features: Dict[str, Dict],
    spec: FeatureSpec,
    aug: AugSpec,
    model_name: str,
) -> Dict[str, float]:
    rows = []
    for participant, pdata in features.items():
        for i in range(len(pdata["y"])):
            rows.append((participant, i, pdata["X"][i], float(pdata["y"][i])))
    if not rows:
        return {"mae": np.nan, "r": np.nan, "n_samples": 0}

    groups = np.asarray([r[0] for r in rows])
    idx = np.asarray([r[1] for r in rows], dtype=np.int32)
    X = np.asarray([r[2] for r in rows], dtype=np.float32)
    y = np.asarray([r[3] for r in rows], dtype=np.float32)

    logo = LeaveOneGroupOut()
    preds = np.zeros_like(y, dtype=np.float64)
    rng = np.random.default_rng(43)
    for train_idx, test_idx in logo.split(X, y, groups):
        train_groups = groups[train_idx]
        train_orig_idx = idx[train_idx]
        X_train = X[train_idx]
        y_train = y[train_idx]

        if aug.name != "none":
            X_aug = []
            y_aug = []
            for g, local_i in zip(train_groups, train_orig_idx):
                wave = features[g]["waves"][int(local_i)]
                y_w = center_crop_or_pad(wave, int(16000 * (spec.window_ms / 1000.0)))
                y_w = augment_audio(y_w, 16000, aug, rng)
                X_aug.append(extract_feature(y_w, 16000, spec))
                y_aug.append(features[g]["y"][int(local_i)])
            if X_aug:
                X_train = np.vstack([X_train, np.asarray(X_aug, dtype=np.float32)])
                y_train = np.concatenate([y_train, np.asarray(y_aug, dtype=np.float32)])

        pipe = Pipeline([("scaler", RobustScaler()), ("model", get_model(model_name))])
        pipe.fit(X_train, y_train)
        preds[test_idx] = pipe.predict(X[test_idx])

    m = compute_metrics(y, preds)
    return {"mae": float(m["mae"]), "r": float(m["r"]), "n_samples": int(len(y))}


def eval_calibration(
    features: Dict[str, Dict],
    model_name: str,
    shots: int,
) -> Dict[str, float]:
    # Leave-one-person-out population training, then few-shot linear calibration.
    participant_names = list(features.keys())
    maes = []
    for holdout in participant_names:
        train_X, train_y = [], []
        for p in participant_names:
            if p == holdout:
                continue
            train_X.append(features[p]["X"])
            train_y.append(features[p]["y"])
        if not train_X:
            continue
        X_train = np.vstack(train_X)
        y_train = np.concatenate(train_y)
        base_pipe = Pipeline([("scaler", RobustScaler()), ("model", get_model(model_name))])
        base_pipe.fit(X_train, y_train)

        X_h = features[holdout]["X"]
        y_h = features[holdout]["y"]
        if len(y_h) <= shots + 3:
            continue

        order = np.arange(len(y_h))
        calib_idx = order[:shots]
        test_idx = order[shots:]

        y_pred_calib = base_pipe.predict(X_h[calib_idx])
        y_true_calib = y_h[calib_idx]
        # linear calibration y' = a*y + b
        A = np.vstack([y_pred_calib, np.ones_like(y_pred_calib)]).T
        try:
            a, b = np.linalg.lstsq(A, y_true_calib, rcond=None)[0]
        except Exception:
            a, b = 1.0, 0.0
        y_pred_test = base_pipe.predict(X_h[test_idx])
        y_pred_adj = a * y_pred_test + b
        maes.append(mean_absolute_error(y_h[test_idx], y_pred_adj))

    if not maes:
        return {"mae": np.nan, "n_participants": 0}
    return {"mae": float(np.mean(maes)), "n_participants": len(maes)}


def search_space(quick: bool) -> Tuple[List[FeatureSpec], List[AugSpec], List[str], List[int]]:
    if quick:
        windows = [300, 1000, 2000]
        families = ["mfcc", "mfe", "spec"]
        n_mfcc_list = [13, 20]
        fmins = [100]
        fmaxs = [8000]
        n_fft = 512
        hops = [0.5]
        n_mels = 40
        models = ["Ridge", "SVR", "BayesianRidge"]
        shots = [5, 10]
    else:
        windows = [100, 300, 700, 1000, 1500, 2000, 3000]
        families = ["mfcc", "mfe", "spec"]
        n_mfcc_list = [10, 13, 20, 30]
        fmins = [50, 100, 300]
        fmaxs = [6000, 8000, 10000]
        n_fft = 1024
        hops = [0.5, 0.25]
        n_mels = 64
        models = ["Ridge", "BayesianRidge", "SVR", "RandomForest", "GradientBoosting"]
        shots = [5, 10, 15]

    features: List[FeatureSpec] = []
    for win in windows:
        for family in families:
            for hop in hops:
                for fmin in fmins:
                    for fmax in fmaxs:
                        if fmax <= fmin:
                            continue
                        if family == "mfcc":
                            for n_mfcc in n_mfcc_list:
                                features.append(
                                    FeatureSpec(
                                        name=f"mfcc_w{win}_h{hop}_m{n_mfcc}_{fmin}-{fmax}",
                                        family=family,
                                        window_ms=win,
                                        hop_ratio=hop,
                                        n_mfcc=n_mfcc,
                                        n_mels=n_mels,
                                        n_fft=n_fft,
                                        fmin=fmin,
                                        fmax=fmax,
                                    )
                                )
                        else:
                            features.append(
                                FeatureSpec(
                                    name=f"{family}_w{win}_h{hop}_{fmin}-{fmax}",
                                    family=family,
                                    window_ms=win,
                                    hop_ratio=hop,
                                    n_mfcc=20,
                                    n_mels=n_mels,
                                    n_fft=n_fft,
                                    fmin=fmin,
                                    fmax=fmax,
                                )
                            )

    augs = [
        AugSpec("none", 0.0, 0.0, 0, False),
        AugSpec("light", 0.002, 3.0, 40, False),
        AugSpec("light_reverb", 0.002, 3.0, 40, True),
    ]
    return features, augs, models, shots


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge-inspired optimization for TONES")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--participants", nargs="+", default=None)
    parser.add_argument("--quick", action="store_true", help="Smaller search space")
    parser.add_argument("--max-configs", type=int, default=0, help="Cap evaluated feature specs")
    parser.add_argument("--models", nargs="+", default=None, help="Override model list, e.g. Ridge BayesianRidge")
    parser.add_argument("--no-calibration", action="store_true", help="Skip few-shot calibration evaluation")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    random.seed(42)
    np.random.seed(42)

    t0 = time.time()
    cfg = load_config(args.config)
    base_dir = get_base_dir(cfg)
    out_dir = base_dir / "output" / "edge_opt"
    out_dir.mkdir(parents=True, exist_ok=True)

    wave_cache = load_wave_cache(cfg, args.participants)
    if not wave_cache:
        raise SystemExit("No participant data loaded.")

    feature_specs, aug_specs, model_names, shots = search_space(args.quick)
    if args.models:
        model_names = args.models
    if args.no_calibration:
        shots = []
    if args.max_configs > 0:
        feature_specs = feature_specs[: args.max_configs]

    logger.info(
        "Evaluating %d feature specs x %d augmentations x %d models",
        len(feature_specs),
        len(aug_specs),
        len(model_names),
    )

    pers_rows = []
    pop_rows = []
    calib_rows = []

    for i, fspec in enumerate(feature_specs, start=1):
        logger.info("[%d/%d] Feature spec: %s", i, len(feature_specs), fspec.name)
        for aspec in aug_specs:
            features = build_features(wave_cache, fspec, aspec)
            if not features:
                continue
            for model in model_names:
                pers = eval_personalized(features, fspec, aspec, model)
                pop = eval_population(features, fspec, aspec, model)
                pers_rows.append(
                    {
                        "feature_spec": fspec.name,
                        "family": fspec.family,
                        "window_ms": fspec.window_ms,
                        "hop_ratio": fspec.hop_ratio,
                        "n_mfcc": fspec.n_mfcc,
                        "fmin": fspec.fmin,
                        "fmax": fspec.fmax,
                        "augmentation": aspec.name,
                        "model": model,
                        "pers_mae": round(pers["mae"], 4) if np.isfinite(pers["mae"]) else np.nan,
                        "pers_r": round(pers["r"], 4) if np.isfinite(pers["r"]) else np.nan,
                        "n_participants": pers["n_participants"],
                    }
                )
                pop_rows.append(
                    {
                        "feature_spec": fspec.name,
                        "family": fspec.family,
                        "window_ms": fspec.window_ms,
                        "hop_ratio": fspec.hop_ratio,
                        "n_mfcc": fspec.n_mfcc,
                        "fmin": fspec.fmin,
                        "fmax": fspec.fmax,
                        "augmentation": aspec.name,
                        "model": model,
                        "pop_mae": round(pop["mae"], 4) if np.isfinite(pop["mae"]) else np.nan,
                        "pop_r": round(pop["r"], 4) if np.isfinite(pop["r"]) else np.nan,
                        "n_samples": pop["n_samples"],
                    }
                )
                for k in shots:
                    calib = eval_calibration(features, model_name=model, shots=k)
                    calib_rows.append(
                        {
                            "feature_spec": fspec.name,
                            "family": fspec.family,
                            "window_ms": fspec.window_ms,
                            "hop_ratio": fspec.hop_ratio,
                            "n_mfcc": fspec.n_mfcc,
                            "fmin": fspec.fmin,
                            "fmax": fspec.fmax,
                            "augmentation": aspec.name,
                            "model": model,
                            "calibration_shots": k,
                            "calib_mae": round(calib["mae"], 4) if np.isfinite(calib["mae"]) else np.nan,
                            "n_participants": calib["n_participants"],
                        }
                    )

    df_p = pd.DataFrame(pers_rows).sort_values("pers_mae")
    df_pop = pd.DataFrame(pop_rows).sort_values("pop_mae")
    if calib_rows:
        df_cal = pd.DataFrame(calib_rows).sort_values("calib_mae")
    else:
        df_cal = pd.DataFrame(
            columns=[
                "feature_spec", "family", "window_ms", "hop_ratio", "n_mfcc", "fmin",
                "fmax", "augmentation", "model", "calibration_shots", "calib_mae", "n_participants",
            ]
        )

    df_p.to_csv(out_dir / "leaderboard_personalized.csv", index=False)
    df_pop.to_csv(out_dir / "leaderboard_population.csv", index=False)
    df_cal.to_csv(out_dir / "leaderboard_calibration.csv", index=False)

    summary = {
        "runtime_sec": round(time.time() - t0, 2),
        "quick": args.quick,
        "feature_specs_evaluated": len(feature_specs),
        "augmentations": [a.name for a in aug_specs],
        "models": model_names,
        "top_personalized": df_p.head(10).to_dict(orient="records"),
        "top_population": df_pop.head(10).to_dict(orient="records"),
        "top_calibration": df_cal.head(10).to_dict(orient="records"),
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved results to %s", out_dir)
    logger.info("Top personalized MAE: %.3f", float(df_p.iloc[0]["pers_mae"]) if not df_p.empty else float("nan"))
    logger.info("Top population MAE: %.3f", float(df_pop.iloc[0]["pop_mae"]) if not df_pop.empty else float("nan"))
    logger.info("Top calibration MAE: %.3f", float(df_cal.iloc[0]["calib_mae"]) if not df_cal.empty else float("nan"))


if __name__ == "__main__":
    main()
