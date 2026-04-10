"""
Promotion Gate — Checks autoresearch results against ONVOX signal gate
and translates configs to BackgroundTrainer format.
======================================================================

Signal gate: r > 0.3 AND improvement > 10% AND p < 0.05
(matches BackgroundTrainer.R_THRESHOLD, IMPROVEMENT_THRESHOLD, P_THRESHOLD)
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
QUEUE_PATH = OUTPUT_DIR / "promotion_queue.json"

# Signal detection thresholds (must match BackgroundTrainer)
R_THRESHOLD = 0.3
IMPROVEMENT_THRESHOLD = 0.10  # 10%
P_THRESHOLD = 0.05

# BackgroundTrainer's FEATURE_SUBSETS for config mapping
BT_FEATURE_SUBSETS = {
    'edge_10': list(range(10)),
    'personal_10': [0, 1, 80, 99, 90, 91, 97, 98, 100, 101],
    'dynamics_10': [20, 48, 52, 59, 80, 85, 90, 91, 97, 98],
    'dynamics_windowed_9': list(range(103, 112)),
    'personal_14': [0, 1, 80, 99, 90, 91, 97, 98, 100, 101, 112, 113, 114, 115],
    'temporal_4': [95, 96, 117, 118],
    'personal_10_time': [0, 1, 80, 99, 90, 91, 97, 98, 100, 101, 95, 96, 117, 118],
    'mfcc_13': list(range(13)),
    'spectral_8': [0, 1, 80, 81, 82, 83, 84, 85],
    'full': None,
}

# Model name mapping: autoresearch name -> BackgroundTrainer model_type
MODEL_MAP = {
    "Ridge": "Ridge",
    "BayesianRidge": "BayesianRidge",
    "SVR": "SVR",
    "ElasticNet": "ElasticNet",
    "Lasso": "Lasso",
    "Huber": "Huber",
    "RandomForest": "RandomForest",
    "GradientBoosting": "GradientBoosting",
    "ExtraTrees": "ExtraTrees",
    "KNN": "KNN",
}

# Feature combo -> closest BT feature subset
FEATURE_COMBO_MAP = {
    "mfcc_only": "mfcc_13",
    "mfcc+spectral": "spectral_8",
    "mfcc+spectral+pitch": "personal_10",
    "mfcc+spectral+pitch+vq": "personal_14",
    "mfcc+spectral+pitch+temporal": "personal_10_time",
    "all_features": "full",
}


class PromotionGate:
    """Evaluates autoresearch results and queues passing configs for BackgroundTrainer."""

    def __init__(self):
        self.queue_path = QUEUE_PATH

    def evaluate(self, result: dict) -> bool:
        """Check if a result passes the ONVOX signal gate.

        Args:
            result: Dict with keys pers_r, pers_mae, baseline_mae (or pct_improvement),
                    and optionally p_value.

        Returns:
            True if the result passes all three gate criteria.
        """
        r = float(result.get("pers_r", 0))
        pct_improvement = float(result.get("pct_improvement", result.get("pers_pct_improvement", 0)))
        p_value = float(result.get("p_value", 1.0))

        # If pct_improvement not directly available, compute from MAE
        if pct_improvement == 0 and "pers_mae" in result and "baseline_mae" in result:
            baseline = float(result["baseline_mae"])
            if baseline > 0:
                pct_improvement = 100 * (baseline - float(result["pers_mae"])) / baseline

        passes_r = r > R_THRESHOLD
        passes_improvement = (pct_improvement / 100.0) > IMPROVEMENT_THRESHOLD
        passes_p = p_value < P_THRESHOLD

        return passes_r and passes_improvement and passes_p

    def translate_config(self, autoresearch_config: dict) -> Tuple[str, Optional[str], str]:
        """Translate autoresearch config to BackgroundTrainer (model_type, alpha_param, feature_subset) tuple.

        Args:
            autoresearch_config: Dict with model_name, n_mfcc, feature_key, normalization.

        Returns:
            Tuple of (model_type, alpha_or_params, feature_subset_name) matching FULL_CONFIGS format.
        """
        model_name = str(autoresearch_config.get("model_name", "Ridge"))
        feature_key = str(autoresearch_config.get("feature_key", "mfcc+spectral+pitch"))
        n_mfcc = int(autoresearch_config.get("n_mfcc", 20))

        # Map model name
        model_type = MODEL_MAP.get(model_name, model_name)

        # Map alpha parameter
        if model_type == "Ridge":
            alpha_param = "100/n"  # Default ONVOX alpha schedule
        elif model_type in ("BayesianRidge", "PhysicsGP"):
            alpha_param = None
        elif model_type == "SVR":
            alpha_param = None
        else:
            alpha_param = None

        # Map feature subset
        feature_subset = FEATURE_COMBO_MAP.get(feature_key, "personal_10")

        # If n_mfcc is unusual, might map to different subset
        if n_mfcc <= 10 and feature_key == "mfcc_only":
            feature_subset = "edge_10"
        elif n_mfcc >= 30 and feature_key in ("mfcc_only", "mfcc+spectral"):
            feature_subset = "full"

        return model_type, alpha_param, feature_subset

    def queue(self, config: dict, result: dict) -> None:
        """Append a passing config to the promotion queue.

        Args:
            config: The autoresearch experiment config.
            result: The evaluation result metrics.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Load existing queue
        queue = []
        if self.queue_path.exists():
            try:
                queue = json.loads(self.queue_path.read_text(encoding="utf-8"))
            except Exception:
                queue = []

        model_type, alpha_param, feature_subset = self.translate_config(config)

        entry = {
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "autoresearch_config": config,
            "bt_config": {
                "model_type": model_type,
                "alpha_param": alpha_param,
                "feature_subset": feature_subset,
            },
            "bt_tuple": f"('{model_type}', {repr(alpha_param)}, '{feature_subset}')",
            "metrics": {
                "pers_mae": result.get("pers_mae"),
                "pers_r": result.get("pers_r"),
                "pop_mae": result.get("pop_mae"),
                "pop_r": result.get("pop_r"),
                "selection_score": result.get("selection_score"),
                "signal_gate_pass_rate": result.get("signal_gate_pass_rate"),
            },
            "status": "pending",
        }

        # Deduplicate by bt_tuple
        existing_tuples = {e.get("bt_tuple") for e in queue}
        if entry["bt_tuple"] in existing_tuples:
            logger.info("Config already in queue: %s", entry["bt_tuple"])
            return

        queue.append(entry)
        self.queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
        logger.info("Queued for promotion: %s", entry["bt_tuple"])


def check_and_queue_result(config: dict, result: dict) -> bool:
    """Convenience function: evaluate result against gate and queue if passing.

    Returns True if the result was queued.
    """
    gate = PromotionGate()
    if gate.evaluate(result):
        gate.queue(config, result)
        return True
    return False
