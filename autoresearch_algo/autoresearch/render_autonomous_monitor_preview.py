#!/usr/bin/env python3
"""One-off render of ONVOX autonomous monitor layout (sample data) for docs/preview. Non-interactive (Agg)."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import plotting helpers from the real monitor
from monitor_autonomous_gui import (
    early_stop_stats,
    evaluated_rows,
    metric_series,
    read_rows,
    read_status,
    recent_metric_avg,
    recent_rationale_usage,
    running_best,
    source_keep_rates,
    summarize,
)


def _fake_rows(n: int = 40) -> list[dict[str, str]]:
    base = datetime(2026, 4, 4, 10, 0, 0)
    rows: list[dict[str, str]] = []
    import random

    random.seed(42)
    score = 18.0
    for i in range(n):
        ts = base + timedelta(minutes=3 * i)
        score = max(8.0, score + random.uniform(-1.2, 0.4))
        st = random.choices(["keep", "discard", "error"], weights=[0.45, 0.5, 0.05])[0]
        rows.append(
            {
                "timestamp": ts.isoformat(timespec="seconds"),
                "status": st,
                "selection_score": f"{score:.3f}",
                "pers_mae": f"{10 + random.uniform(-2, 2):.2f}",
                "pop_mae": f"{11 + random.uniform(-2, 2):.2f}",
                "temp_mae": f"{12 + random.uniform(-2, 2):.2f}",
                "pop_r": f"{0.02 + random.uniform(-0.15, 0.12):.4f}",
                "temp_r": f"{0.01 + random.uniform(-0.1, 0.08):.4f}",
                "pop_clarke_ab_pct": f"{85 + random.uniform(-5, 5):.2f}",
                "pop_mard": f"{12 + random.uniform(-3, 3):.2f}",
                "pop_bias": f"{random.uniform(-2, 2):.2f}",
                "signal_gate_pass_rate": f"{random.uniform(0.2, 0.6):.3f}",
                "signal_gate_penalty": f"{random.uniform(0.0, 0.3):.3f}",
                "notes": "diversity_forced_slot" if random.random() < 0.15 else "",
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Render PNG matching monitor_autonomous_gui layout (sample or real TSV)."
    )
    p.add_argument("-o", "--output", type=Path, default=Path(__file__).resolve().parent / "autonomous_monitor_preview.png")
    p.add_argument(
        "--tsv-file",
        type=Path,
        default=None,
        help="Real autonomous_runs*.tsv (evaluated rows only). If omitted, uses synthetic data.",
    )
    p.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="Optional status.json from the loop; if missing, status panel uses placeholders.",
    )
    args_ns = p.parse_args()

    if args_ns.tsv_file and args_ns.tsv_file.exists():
        rows = evaluated_rows(read_rows(args_ns.tsv_file))
        status_path = args_ns.status_file
        if status_path and status_path.exists():
            status = read_status(status_path)
        else:
            status = {
                "running": False,
                "pid": "-",
                "llm_model": rows[-1].get("llm_model", "-") if rows else "-",
                "cycle": rows[-1].get("cycle", "-") if rows else "-",
                "phase": "snapshot (no status.json)",
                "best_selection_score": "-",
                "last_update": datetime.now().isoformat(timespec="seconds"),
            }
        footer = f"DATA: {args_ns.tsv_file.name} ({len(rows)} evaluated rows). For live window: python monitor_autonomous_gui.py --tsv-file ... --status-file ..."
    else:
        rows = _fake_rows(45)
        status = {
            "running": True,
            "pid": 4242,
            "llm_model": "qwen2.5-coder:7b",
            "cycle": 12,
            "phase": "evaluate",
            "best_selection_score": "9.842",
            "last_update": datetime.now().isoformat(timespec="seconds"),
        }
        footer = "PREVIEW: synthetic data — run monitor_autonomous_gui.py with real status.json + autonomous_runs*.tsv for live feed."

    args = args_ns

    target_selection_score = 12.0
    target_mae = 12.0
    target_pop_r = 0.10
    target_temp_r = 0.05
    stale_alert_minutes = 8
    interval_ms = 4000

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    now = datetime.now()

    summary = summarize(rows)
    early = early_stop_stats(rows)
    pop_clarke_ab = recent_metric_avg(rows, "pop_clarke_ab_pct", 20)
    pop_mard = recent_metric_avg(rows, "pop_mard", 20)
    pop_bias = recent_metric_avg(rows, "pop_bias", 20)
    gate_rate = recent_metric_avg(rows, "signal_gate_pass_rate", 20)
    gate_pen = recent_metric_avg(rows, "signal_gate_penalty", 20)
    diversity_recent = recent_rationale_usage(rows, "diversity_forced_slot", 20)
    src_rates = source_keep_rates(rows, 50)

    x_s, y_s = metric_series(rows, "selection_score")
    ax = axes[0, 0]
    if y_s:
        ax.plot(x_s, y_s, color="#1f77b4", linewidth=1.6, label="selection_score")
        ax.plot(x_s, running_best(y_s, lower_is_better=True), color="#2ca02c", linewidth=1.6, label="best_so_far")
        ax.scatter([x_s[-1]], [y_s[-1]], color="#d62728", s=28, label="latest")
        ax.axhline(target_selection_score, color="#444444", linestyle="--", linewidth=1.0, label=f"target={target_selection_score:g}")
        ax.set_title("Selection Score (lower is better)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Score")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax = axes[0, 1]
    for key, color, label in [
        ("pers_mae", "#d62728", "Personalized MAE"),
        ("pop_mae", "#ff7f0e", "Population MAE"),
        ("temp_mae", "#9467bd", "Temporal MAE"),
    ]:
        x, y = metric_series(rows, key)
        if y:
            ax.plot(x, y, color=color, linewidth=1.5, label=label)
    ax.axhline(target_mae, color="#444444", linestyle="--", linewidth=1.0, label=f"target_mae={target_mae:g}")
    ax.set_title("MAE Metrics")
    ax.set_xlabel("Time")
    ax.set_ylabel("MAE")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax = axes[1, 0]
    for key, color, label in [
        ("pop_r", "#17becf", "Population r"),
        ("temp_r", "#8c564b", "Temporal r"),
    ]:
        x, y = metric_series(rows, key)
        if y:
            ax.plot(x, y, color=color, linewidth=1.5, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(target_pop_r, color="#17becf", linestyle="--", linewidth=1.0, alpha=0.85, label=f"target_pop_r={target_pop_r:g}")
    ax.axhline(target_temp_r, color="#8c564b", linestyle="--", linewidth=1.0, alpha=0.85, label=f"target_temp_r={target_temp_r:g}")
    ax.set_title("Correlation Metrics (higher is better)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pearson r")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax = axes[1, 1]
    win_sizes = [20, 50, 200]
    labels = ["20", "50", "200"]
    keep_vals: list[float] = []
    discard_vals: list[float] = []
    err_vals: list[float] = []
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
        f"Targets: score<={target_selection_score:g}, mae<={target_mae:g}, pop_r>={target_pop_r:g}, temp_r>={target_temp_r:g}"
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

    last_update = datetime.fromisoformat(status["last_update"])
    stale_sec = (now - last_update).total_seconds()
    live = stale_sec <= max(interval_ms / 1000.0 * 3, 20)
    live_text = "LIVE" if live else "STALE"
    age_text = f"{stale_sec:.0f}s old"
    fig.suptitle(
        f"ONVOX Autonomous Research Progress Dashboard | feed={live_text} | monitor={now.strftime('%H:%M:%S')} | data_age={age_text}",
        fontsize=12,
        fontweight="bold",
        color=("#2ca02c" if live else "#d62728"),
    )
    fig.text(0.5, 0.01, footer, fontsize=8, ha="center", color="#555555")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
