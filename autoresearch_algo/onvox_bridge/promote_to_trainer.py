#!/usr/bin/env python3
"""
Promote to BackgroundTrainer — Review and apply queued autoresearch promotions.
===============================================================================

CLI tool that reads promotion_queue.json and either:
  --dry-run: Lists pending promotions with metrics (default)
  --apply:   Prints the FULL_CONFIGS entries to add to background_trainer.py
             and marks them as applied in the log.

Does NOT auto-edit background_trainer.py (too risky). Outputs the config
tuple and human copies it.

Usage:
    python -m tools.autoresearch.onvox_bridge.promote_to_trainer
    python -m tools.autoresearch.onvox_bridge.promote_to_trainer --apply
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
QUEUE_PATH = OUTPUT_DIR / "promotion_queue.json"
LOG_PATH = OUTPUT_DIR / "promotion_log.json"


def load_queue() -> list:
    if not QUEUE_PATH.exists():
        return []
    try:
        return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_queue(queue: list) -> None:
    QUEUE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def load_log() -> list:
    if not LOG_PATH.exists():
        return []
    try:
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_log(log: list) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")


def print_pending(queue: list) -> None:
    pending = [e for e in queue if e.get("status") == "pending"]
    if not pending:
        print("No pending promotions in queue.")
        return

    print(f"\n{'='*70}")
    print(f"Pending Promotions ({len(pending)} configs)")
    print(f"{'='*70}")

    for i, entry in enumerate(pending, 1):
        bt = entry.get("bt_config", {})
        metrics = entry.get("metrics", {})
        ar_cfg = entry.get("autoresearch_config", {})

        print(f"\n#{i} — {entry.get('bt_tuple', '?')}")
        print(f"  BackgroundTrainer: model={bt.get('model_type')}, "
              f"alpha={bt.get('alpha_param')}, features={bt.get('feature_subset')}")
        print(f"  Autoresearch:      model={ar_cfg.get('model_name')}, "
              f"n_mfcc={ar_cfg.get('n_mfcc')}, "
              f"features={ar_cfg.get('feature_key')}, "
              f"norm={ar_cfg.get('normalization')}")
        print(f"  Metrics:           pers_mae={metrics.get('pers_mae', '?')}, "
              f"pers_r={metrics.get('pers_r', '?')}, "
              f"pop_mae={metrics.get('pop_mae', '?')}, "
              f"pop_r={metrics.get('pop_r', '?')}")
        print(f"  Selection score:   {metrics.get('selection_score', '?')}")
        print(f"  Signal gate rate:  {metrics.get('signal_gate_pass_rate', '?')}")
        print(f"  Queued at:         {entry.get('queued_at', '?')}")

    print(f"\n{'='*70}")
    print("Run with --apply to generate FULL_CONFIGS entries")


def apply_promotions(queue: list) -> None:
    pending = [e for e in queue if e.get("status") == "pending"]
    if not pending:
        print("No pending promotions to apply.")
        return

    log = load_log()

    print(f"\n{'='*70}")
    print("FULL_CONFIGS entries to add to background_trainer.py:")
    print(f"{'='*70}")
    print()
    print("# --- Autoresearch promotions (%s) ---" % datetime.now().strftime("%Y-%m-%d"))

    for entry in pending:
        bt_tuple = entry.get("bt_tuple", "")
        metrics = entry.get("metrics", {})
        print(f"    {bt_tuple},  "
              f"# autoresearch: pers_mae={metrics.get('pers_r', '?')}, "
              f"score={metrics.get('selection_score', '?')}")

        # Mark as applied
        entry["status"] = "applied"
        entry["applied_at"] = datetime.now(timezone.utc).isoformat()

        # Log
        log.append({
            "action": "applied",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bt_tuple": bt_tuple,
            "metrics": metrics,
        })

    print()
    print(f"{'='*70}")
    print(f"Copy the above into FULL_CONFIGS in background_trainer.py")
    print(f"Then sync to onvox-ai2 and push for production deployment.")
    print(f"{'='*70}")

    save_queue(queue)
    save_log(log)
    print(f"\nMarked {len(pending)} entries as applied.")


def main():
    parser = argparse.ArgumentParser(description="Review/apply autoresearch promotions")
    parser.add_argument("--apply", action="store_true", help="Generate FULL_CONFIGS entries and mark as applied")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    queue = load_queue()

    if args.apply:
        apply_promotions(queue)
    else:
        print_pending(queue)


if __name__ == "__main__":
    main()
