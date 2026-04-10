#!/usr/bin/env python3
"""GUI monitor with color plots and snapshot exports for ONVOX AutoResearch loop."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def read_status(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def as_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def as_dt(value: str) -> Optional[datetime]:
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s.replace(" ", "T", 1))
    except Exception:
        return None


def last_eval_timestamp(rows: List[Dict[str, str]]) -> Optional[datetime]:
    best: Optional[datetime] = None
    for r in rows:
        ts = as_dt(str(r.get("timestamp", "")))
        if ts is not None and (best is None or ts > best):
            best = ts
    return best


def apply_datetime_xaxis(ax, xs: List[datetime]) -> None:
    if not xs:
        return
    t0, t1 = min(xs), max(xs)
    span_sec = max(1.0, (t1 - t0).total_seconds())
    pad = timedelta(seconds=max(120.0, min(6 * 3600.0, span_sec * 0.04)))
    ax.set_xlim(t0 - pad, t1 + pad)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    if span_sec <= 36 * 3600:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    else:
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def evaluated_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("status") in {"keep", "discard", "error"}]


def metric_series(rows: List[Dict[str, str]], key: str) -> Tuple[List[datetime], List[float]]:
    xs: List[datetime] = []
    ys: List[float] = []
    for row in rows:
        if row.get("status") not in {"keep", "discard"}:
            continue
        val = as_float(row.get(key, ""))
        ts = as_dt(row.get("timestamp", ""))
        if val is None or ts is None:
            continue
        xs.append(ts)
        ys.append(val)
    return xs, ys


def metric_series_index(rows: List[Dict[str, str]], key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    idx = 0
    for row in rows:
        if row.get("status") not in {"keep", "discard"}:
            continue
        val = as_float(row.get(key, ""))
        if val is None:
            continue
        idx += 1
        xs.append(idx)
        ys.append(val)
    return xs, ys


def running_best(values: List[float], lower_is_better: bool = True) -> List[float]:
    out: List[float] = []
    best = float("inf") if lower_is_better else float("-inf")
    for v in values:
        if lower_is_better:
            best = min(best, v)
        else:
            best = max(best, v)
        out.append(best)
    return out


def summarize(rows: List[Dict[str, str]]) -> Dict[str, float]:
    total = len(rows)
    keep = sum(1 for r in rows if r.get("status") == "keep")
    discard = sum(1 for r in rows if r.get("status") == "discard")
    error = sum(1 for r in rows if r.get("status") == "error")
    return {
        "total": float(total),
        "keep": float(keep),
        "discard": float(discard),
        "error": float(error),
        "keep_rate": (keep / total) if total else 0.0,
        "error_rate": (error / total) if total else 0.0,
    }


def early_stop_stats(rows: List[Dict[str, str]]) -> Dict[str, float]:
    eval_rows = [r for r in rows if r.get("status") in {"keep", "discard"}]
    n_eval = len(eval_rows)
    n_early = sum(
        1 for r in eval_rows if "early_stop_no_temporal" in str(r.get("notes", ""))
    )
    return {
        "early_stop": float(n_early),
        "evaluated": float(n_eval),
        "rate": (n_early / n_eval) if n_eval else 0.0,
    }


def recent_metric_avg(rows: List[Dict[str, str]], key: str, window: int = 20) -> Optional[float]:
    vals: List[float] = []
    for r in rows:
        if r.get("status") not in {"keep", "discard"}:
            continue
        v = as_float(r.get(key, ""))
        if v is not None:
            vals.append(v)
    vals = vals[-window:]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def recent_rationale_usage(
    rows: List[Dict[str, str]], needle: str, window: int = 20
) -> Dict[str, float]:
    eval_rows = [r for r in rows if r.get("status") in {"keep", "discard"}]
    recent = eval_rows[-window:] if len(eval_rows) > window else eval_rows
    total = len(recent)
    hits = sum(1 for r in recent if needle in str(r.get("notes", "")))
    return {
        "hits": float(hits),
        "total": float(total),
        "rate": (hits / total) if total else 0.0,
    }


def source_keep_rates(rows: List[Dict[str, str]], window: int = 50) -> str:
    def _source_of(r: Dict[str, str]) -> str:
        src = str(r.get("source", "")).strip()
        if src:
            return src
        notes = str(r.get("notes", ""))
        if "diversity_forced_slot" in notes:
            return "diversity"
        if "guardrail_neighbor_of_best_keep" in notes:
            return "neighbor"
        if "guardrail_fallback_unseen_candidate" in notes:
            return "fallback"
        if "guardrail_underexplored_axes" in notes:
            return "underexplored"
        return "llm"

    eval_rows = [r for r in rows if r.get("status") in {"keep", "discard"}]
    recent = eval_rows[-window:] if len(eval_rows) > window else eval_rows
    counts: Dict[str, int] = {}
    keeps: Dict[str, int] = {}
    for r in recent:
        src = _source_of(r)
        counts[src] = counts.get(src, 0) + 1
        if r.get("status") == "keep":
            keeps[src] = keeps.get(src, 0) + 1
    if not counts:
        return "-"
    items = []
    for src, cnt in counts.items():
        k = keeps.get(src, 0)
        rate = (k / cnt) if cnt else 0.0
        items.append((src, cnt, k, rate))
    items.sort(key=lambda x: (-x[1], x[0]))
    top = items[:4]
    return ", ".join([f"{src}:{k}/{cnt}({rate:.0%})" for src, cnt, k, rate in top])


def restart_hint_command() -> str:
    root = Path(__file__).resolve().parent
    launcher = root / "start_tones_autonomous.ps1"
    return (
        f'powershell -ExecutionPolicy Bypass -File "{launcher}" -Stop; '
        f'powershell -ExecutionPolicy Bypass -File "{launcher}" -Background'
    )


def resolve_snapshot_dir(tsv_path: Path, arg_value: str) -> Path:
    p = Path(arg_value)
    if p.is_absolute():
        return p
    return tsv_path.parent / p


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI monitor for ONVOX AutoResearch autonomous loop.")
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--tsv-file", required=True)
    parser.add_argument("--interval-ms", type=int, default=4000, help="Refresh interval in milliseconds.")
    parser.add_argument("--target-selection-score", type=float, default=12.0, help="Reference line for selection score.")
    parser.add_argument("--target-mae", type=float, default=12.0, help="Reference line for MAE charts.")
    parser.add_argument("--target-pop-r", type=float, default=0.10, help="Reference line for population correlation.")
    parser.add_argument("--target-temp-r", type=float, default=0.05, help="Reference line for temporal correlation.")
    parser.add_argument(
        "--stale-alert-minutes",
        type=int,
        default=8,
        help="Show stale warning when data age exceeds this many minutes.",
    )
    parser.add_argument("--snapshot-minutes", type=int, default=10, help="Save PNG dashboard every N minutes (0 disables).")
    parser.add_argument(
        "--snapshot-dir",
        default="monitor_snapshots",
        help="Directory to write dashboard PNG snapshots (absolute or relative to TSV folder).",
    )
    args = parser.parse_args()

    status_path = Path(args.status_file)
    tsv_path = Path(args.tsv_file)
    snapshot_dir = resolve_snapshot_dir(tsv_path, args.snapshot_dir)
    if args.snapshot_minutes > 0:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.canvas.manager.set_window_title("ONVOX Autonomous Monitor")
    last_snapshot_bucket: Optional[int] = None

    def refresh(_frame: int) -> None:
        nonlocal last_snapshot_bucket
        status = read_status(status_path)
        rows = evaluated_rows(read_rows(tsv_path))
        summary = summarize(rows)
        early = early_stop_stats(rows)
        pop_clarke_ab = recent_metric_avg(rows, "pop_clarke_ab_pct", 20)
        pop_mard = recent_metric_avg(rows, "pop_mard", 20)
        pop_bias = recent_metric_avg(rows, "pop_bias", 20)
        gate_rate = recent_metric_avg(rows, "signal_gate_pass_rate", 20)
        gate_pen = recent_metric_avg(rows, "signal_gate_penalty", 20)
        diversity_recent = recent_rationale_usage(rows, "diversity_forced_slot", 20)
        src_rates = source_keep_rates(rows, 50)
        now = datetime.now()

        for ax in axes.flatten():
            ax.clear()

        # 1) Selection score with running best.
        x_s, y_s = metric_series(rows, "selection_score")
        ax = axes[0, 0]
        if y_s:
            ax.plot(x_s, y_s, color="#1f77b4", linewidth=1.6, label="selection_score")
            ax.plot(x_s, running_best(y_s, lower_is_better=True), color="#2ca02c", linewidth=1.6, label="best_so_far")
            ax.scatter([x_s[-1]], [y_s[-1]], color="#d62728", s=28, label="latest")
            ax.axhline(args.target_selection_score, color="#444444", linestyle="--", linewidth=1.0, label=f"target={args.target_selection_score:g}")
            ax.set_title("Selection Score (lower is better)")
            ax.set_xlabel("Time (local)")
            ax.set_ylabel("Score")
            ax.legend(loc="upper right")
            apply_datetime_xaxis(ax, x_s)
        else:
            ax.set_title("Selection Score")
            ax.text(0.5, 0.5, "No score data yet", ha="center", va="center", transform=ax.transAxes)

        # 2) MAE trends.
        ax = axes[0, 1]
        mae_specs = [
            ("pers_mae", "#d62728", "Personalized MAE"),
            ("pop_mae", "#ff7f0e", "Population MAE"),
            ("temp_mae", "#9467bd", "Temporal MAE"),
        ]
        drew = False
        mae_xs_all: List[datetime] = []
        for key, color, label in mae_specs:
            x, y = metric_series(rows, key)
            if y:
                mae_xs_all.extend(x)
                ax.plot(x, y, color=color, linewidth=1.5, label=label)
                drew = True
        ax.axhline(args.target_mae, color="#444444", linestyle="--", linewidth=1.0, label=f"target_mae={args.target_mae:g}")
        ax.set_title("MAE Metrics")
        ax.set_xlabel("Time (local)")
        ax.set_ylabel("MAE")
        if drew:
            apply_datetime_xaxis(ax, mae_xs_all)
            ax.legend(loc="upper right")
        else:
            ax.text(0.5, 0.5, "No MAE data yet", ha="center", va="center", transform=ax.transAxes)

        # 3) Correlation trends.
        ax = axes[1, 0]
        corr_specs = [
            ("pop_r", "#17becf", "Population r"),
            ("temp_r", "#8c564b", "Temporal r"),
        ]
        drew = False
        corr_xs_all: List[datetime] = []
        for key, color, label in corr_specs:
            x, y = metric_series(rows, key)
            if y:
                corr_xs_all.extend(x)
                ax.plot(x, y, color=color, linewidth=1.5, label=label)
                drew = True
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.axhline(args.target_pop_r, color="#17becf", linestyle="--", linewidth=1.0, alpha=0.85, label=f"target_pop_r={args.target_pop_r:g}")
        ax.axhline(args.target_temp_r, color="#8c564b", linestyle="--", linewidth=1.0, alpha=0.85, label=f"target_temp_r={args.target_temp_r:g}")
        ax.set_title("Correlation Metrics (higher is better)")
        ax.set_xlabel("Time (local)")
        ax.set_ylabel("Pearson r")
        if drew:
            apply_datetime_xaxis(ax, corr_xs_all)
            ax.legend(loc="lower right")
        else:
            ax.text(0.5, 0.5, "No correlation data yet", ha="center", va="center", transform=ax.transAxes)

        # 4) Status panel + stacked windows.
        ax = axes[1, 1]
        win_sizes = [20, 50, 200]
        labels = ["20", "50", "200"]
        keep_vals: List[float] = []
        discard_vals: List[float] = []
        err_vals: List[float] = []
        for w in win_sizes:
            window_rows = rows[-w:] if len(rows) > w else rows
            s = summarize(window_rows)
            keep_vals.append(s["keep"])
            discard_vals.append(s["discard"])
            err_vals.append(s["error"])
        x = list(range(len(win_sizes)))
        ax.bar(x, keep_vals, color="#2ca02c", label="keep")
        ax.bar(x, discard_vals, bottom=keep_vals, color="#ffbf00", label="discard")
        stacked_bottom = [k + d for k, d in zip(keep_vals, discard_vals)]
        ax.bar(x, err_vals, bottom=stacked_bottom, color="#d62728", label="error")
        ax.set_xticks(x, labels)
        ax.set_title("Outcome Mix by Recent Window")
        ax.set_xlabel("Window size (rows)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")

        info = (
            f"Running: {status.get('running', False)}\n"
            f"PID: {status.get('pid', '-')}\n"
            f"Model: {status.get('llm_model', '-')}\n"
            f"Cycle: {status.get('cycle', '-')}\n"
            f"Phase: {status.get('phase', '-')}\n"
            f"Best score: {status.get('best_selection_score', '-')}\n"
            f"Rows loaded: {int(summary['total'])}\n"
            f"Early-stop pruned: {int(early['early_stop'])}/{int(early['evaluated'])} ({early['rate']:.1%})\n"
            f"Clinical(20): AB={('-' if pop_clarke_ab is None else f'{pop_clarke_ab:.2f}%')}, "
            f"MARD={('-' if pop_mard is None else f'{pop_mard:.2f}%')}, "
            f"Bias={('-' if pop_bias is None else f'{pop_bias:.2f}')}\n"
            f"SignalGate(20): pass_rate={('-' if gate_rate is None else f'{gate_rate:.3f}')}, "
            f"penalty={('-' if gate_pen is None else f'{gate_pen:.3f}')}\n"
            f"Innovation(20): diversity_slot={int(diversity_recent['hits'])}/{int(diversity_recent['total'])} "
            f"({diversity_recent['rate']:.1%})\n"
            f"Source keep-rate(50): {src_rates}\n"
            f"Total: {int(summary['total'])} | Keep rate: {summary['keep_rate']:.1%} | Error rate: {summary['error_rate']:.1%}\n"
            f"Targets: score<={args.target_selection_score:g}, mae<={args.target_mae:g}, pop_r>={args.target_pop_r:g}, temp_r>={args.target_temp_r:g}"
        )
        ax.text(
            1.02,
            0.5,
            info,
            transform=ax.transAxes,
            fontsize=9,
            va="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#999999"},
        )

        last_update = as_dt(str(status.get("last_update", "")))
        stale_sec = (now - last_update).total_seconds() if last_update else None
        stale_alert = (
            stale_sec is not None
            and stale_sec > max(args.stale_alert_minutes, 1) * 60
            and bool(status.get("running", False))
        )
        live = bool(
            status.get("running", False)
            and isinstance(status.get("pid", None), int)
            and stale_sec is not None
            and stale_sec <= max(args.interval_ms / 1000.0 * 3, 20)
        )
        live_text = "LIVE" if live else "STALE"
        age_text = "-" if stale_sec is None else f"{stale_sec:.0f}s old"
        last_ev = last_eval_timestamp(rows)
        tsv_bit = "TSV last=—"
        if last_ev is not None:
            lag = (now - last_ev).total_seconds()
            tsv_bit = (
                f"TSV last={last_ev.strftime('%m-%d %H:%M')} ({lag / 3600.0:.1f}h ago)"
                if lag >= 3600
                else f"TSV last={last_ev.strftime('%m-%d %H:%M')} ({lag / 60.0:.0f}m ago)"
            )
        try:
            fig.autofmt_xdate(rotation=24)
        except Exception:
            pass
        fig.suptitle(
            f"ONVOX Autonomous Monitor | {live_text} | monitor={now.strftime('%H:%M:%S')} | status_age={age_text} | {tsv_bit}",
            fontsize=11,
            fontweight="bold",
            color=("#2ca02c" if live else "#d62728"),
        )
        if stale_alert:
            fig.text(
                0.01,
                0.01,
                f"STALE ALERT: data age > {args.stale_alert_minutes}m. Restart hint: {restart_hint_command()}",
                fontsize=8,
                color="#d62728",
            )
        fig.tight_layout()

        if args.snapshot_minutes > 0:
            bucket_seconds = max(args.snapshot_minutes, 1) * 60
            bucket = int(now.timestamp() // bucket_seconds)
            if bucket != last_snapshot_bucket:
                last_snapshot_bucket = bucket
                out_file = snapshot_dir / f"dashboard_{now.strftime('%Y%m%d_%H%M%S')}.png"
                fig.savefig(out_file, dpi=140, bbox_inches="tight")

    refresh(0)
    _anim = FuncAnimation(fig, refresh, interval=max(args.interval_ms, 1000), cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
