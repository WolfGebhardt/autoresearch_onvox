#!/usr/bin/env python3
"""GUI monitor with color plots and snapshot exports for TONES loop."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Visual theme — dark, high-contrast, readable at a glance
# ---------------------------------------------------------------------------
def _theme() -> Dict[str, str]:
    # Okabe–Ito–inspired palette (colorblind-friendly) on dark background.
    return {
        "bg": "#0b1220",
        "panel": "#111827",
        "panel_elev": "#1a2332",
        "text": "#f1f5f9",
        "muted": "#94a3b8",
        "grid": "#334155",
        "c1": "#56B4E9",  # sky blue
        "c2": "#E69F00",  # orange
        "c3": "#009E73",  # bluish green
        "c4": "#F0E442",  # yellow
        "c5": "#D55E00",  # vermillion
        "c6": "#CC79A7",  # reddish purple
        "accent": "#56B4E9",
        "good": "#009E73",
        "warn": "#E69F00",
        "bad": "#D55E00",
        "rose": "#CC79A7",
        "violet": "#CC79A7",
        "amber": "#E69F00",
        "cyan": "#56B4E9",
    }


def _apply_figure_style(fig: plt.Figure, t: Dict[str, str]) -> None:
    fig.patch.set_facecolor(t["bg"])
    fig.set_edgecolor(t["bg"])


def _style_data_axis(ax: plt.Axes, t: Dict[str, str]) -> None:
    ax.set_facecolor(t["panel"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])
    ax.title.set_color(t["text"])
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("600")
    ax.grid(True, alpha=0.28, color=t["grid"], linestyle="-", linewidth=0.6)
    for s in ax.spines.values():
        s.set_color(t["grid"])
        s.set_linewidth(0.8)


def _legend_top_center(ax: plt.Axes, t: Dict[str, str], ncol: int) -> None:
    """Center legend above the plot so lines, scatter, and recent points stay visible."""
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=ncol,
        framealpha=0.92,
        facecolor=t["panel_elev"],
        edgecolor=t["grid"],
        labelcolor=t["text"],
        borderaxespad=0.0,
        fontsize=8,
        columnspacing=0.95,
        handlelength=1.5,
        handletextpad=0.45,
    )


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
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def status_pid(value: object) -> Optional[int]:
    """Normalize pid from JSON (int, float, or numeric string)."""
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value == int(value):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        v = int(value.strip())
        return v if v > 0 else None
    return None


def pid_is_alive(pid: int) -> bool:
    """True if an OS process with this PID exists (loop may not touch status.json for a long time)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def evaluated_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [r for r in rows if r.get("status") in {"keep", "discard", "error"}]


def read_text_tail_lines(path: Path, max_lines: int, max_line_chars: int = 180) -> str:
    """Last N lines without loading huge files whole."""
    if not path.exists():
        return f"({path.name} not found)"
    try:
        size = path.stat().st_size
        if size == 0:
            return "(empty)"
        with path.open("rb") as f:
            chunk_bytes = min(size, 196_608)
            f.seek(size - chunk_bytes)
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        if chunk_bytes < size and lines:
            lines = lines[1:]
        tail = lines[-max_lines:] if len(lines) > max_lines else lines
        out: List[str] = []
        for ln in tail:
            if len(ln) > max_line_chars:
                ln = ln[: max_line_chars - 1] + "…"
            out.append(ln)
        return "\n".join(out) if out else "(empty)"
    except OSError as exc:
        return f"({path.name}: {exc})"


def format_processing_hint(status: Dict) -> str:
    """Human-readable 'what is it doing now' from status.json (no extra log parsing)."""
    if not status:
        return "status.json missing or empty."
    lines: List[str] = [
        f"phase={status.get('phase', '—')}  cycle={status.get('cycle', '—')}  "
        f"running={status.get('running', False)}"
    ]
    ph = str(status.get("phase", ""))
    if ph == "stage1_cheap_eval":
        lines.append(
            f"Stage-1 (cheap): {status.get('stage1_completed', '?')}/{status.get('stage1_total', '?')} done, "
            f"{status.get('stage1_pending', '?')} still running — sklearn CV + features per candidate."
        )
    elif ph == "stage2_full_eval":
        lines.append(
            f"Stage-2 (full): {status.get('stage2_completed', '?')}/{status.get('stage2_total', '?')} done, "
            f"{status.get('stage2_pending', '?')} pending — temporal + population eval."
        )
    elif ph == "proposing_batch":
        lines.append("LLM is proposing the next candidate batch (Ollama).")
    elif ph in {"loading_data", "starting"}:
        lines.append("Loading config / participant audio…")
    elif ph == "idle":
        lines.append("Idle / writing TSV between batches.")
    bs = status.get("batch_size")
    if bs is not None:
        lines.append(f"batch_size={bs}  uncached={status.get('uncached', '—')}")
    lines.append("Eval stack: CPU (librosa + scikit-learn); not the PyTorch train.py path.")
    return "\n".join(lines)


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
        return "—"
    items = []
    for src, cnt in counts.items():
        k = keeps.get(src, 0)
        rate = (k / cnt) if cnt else 0.0
        items.append((src, cnt, k, rate))
    items.sort(key=lambda x: (-x[1], x[0]))
    top = items[:5]
    return "  ·  ".join([f"{src} {k}/{cnt} ({rate:.0%})" for src, cnt, k, rate in top])


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


def _empty_message(ax: plt.Axes, title: str, subtitle: str, t: Dict[str, str]) -> None:
    ax.set_title(title, color=t["text"], fontsize=11, fontweight="600")
    ax.text(
        0.5,
        0.45,
        subtitle,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color=t["muted"],
        wrap=True,
    )
    ax.text(
        0.5,
        0.28,
        "Starts updating after the first evaluations complete.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
        color=t["grid"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI monitor for TONES autonomous loop.")
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
    parser.add_argument(
        "--live-max-stale-sec",
        type=int,
        default=600,
        help=(
            "Max seconds since status.json last_update to show LIVE (default 10 min). "
            "Eval batches (stage1/stage2) often run minutes without status writes; the old 20s rule was too strict."
        ),
    )
    parser.add_argument("--snapshot-minutes", type=int, default=10, help="Save PNG dashboard every N minutes (0 disables).")
    parser.add_argument(
        "--snapshot-dpi",
        type=int,
        default=160,
        help="PNG snapshot resolution (higher = sharper files, larger size).",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="monitor_snapshots",
        help="Directory to write dashboard PNG snapshots (absolute or relative to TSV folder).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Background loop stdout (default: <status-dir>/loop.log).",
    )
    parser.add_argument(
        "--err-log-file",
        default=None,
        help="Background loop stderr (default: <status-dir>/loop.err.log).",
    )
    parser.add_argument("--log-stdout-lines", type=int, default=14, help="Lines of loop.log to show.")
    parser.add_argument("--log-stderr-lines", type=int, default=5, help="Lines of loop.err.log to show.")
    args = parser.parse_args()

    status_path = Path(args.status_file)
    tsv_path = Path(args.tsv_file)
    log_out_path = Path(args.log_file) if args.log_file else status_path.parent / "loop.log"
    log_err_path = Path(args.err_log_file) if args.err_log_file else status_path.parent / "loop.err.log"
    snapshot_dir = resolve_snapshot_dir(tsv_path, args.snapshot_dir)
    if args.snapshot_minutes > 0:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    t = _theme()
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.unicode_minus"] = False
    # Space between title and plot (Text.title has no set_pad in recent Matplotlib).
    plt.rcParams["axes.titlepad"] = 10

    fig = plt.figure(figsize=(15, 10.2), facecolor=t["bg"])
    _apply_figure_style(fig, t)
    gs = GridSpec(
        4,
        3,
        figure=fig,
        # Taller banner row so the main title and subtitle are not cramped.
        height_ratios=[0.13, 1.0, 1.0, 0.34],
        width_ratios=[1.0, 1.0, 0.52],
        hspace=0.48,
        wspace=0.32,
        left=0.06,
        right=0.98,
        top=0.91,
        bottom=0.05,
    )

    ax_banner = fig.add_subplot(gs[0, :])
    ax00 = fig.add_subplot(gs[1, 0])
    ax01 = fig.add_subplot(gs[1, 1])
    ax10 = fig.add_subplot(gs[2, 0])
    ax11 = fig.add_subplot(gs[2, 1])
    ax_side = fig.add_subplot(gs[1:3, 2])
    ax_log = fig.add_subplot(gs[3, :])

    try:
        fig.canvas.manager.set_window_title("TONES · Research dashboard")
    except Exception:
        pass

    plot_axes = (ax00, ax01, ax10, ax11)
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

        for ax in plot_axes:
            ax.clear()
            _style_data_axis(ax, t)

        # --- Banner ---
        ax_banner.clear()
        ax_banner.axis("off")
        ax_banner.set_facecolor(t["bg"])
        last_update = as_dt(str(status.get("last_update", "")))
        stale_sec = (now - last_update).total_seconds() if last_update else None
        file_age_sec: Optional[float] = None
        if status_path.exists():
            try:
                mtime = datetime.fromtimestamp(status_path.stat().st_mtime)
                file_age_sec = (now - mtime).total_seconds()
            except OSError:
                pass

        running = bool(status.get("running", False))
        stale_alert = (
            stale_sec is not None
            and stale_sec > max(args.stale_alert_minutes, 1) * 60
            and running
        )
        # LIVE = running + PID alive or recent heartbeat. STOPPED = loop exited (not "STALE").
        max_stale = float(max(30, args.live_max_stale_sec))
        spid = status_pid(status.get("pid"))
        os_alive = spid is not None and pid_is_alive(spid)
        heartbeat_ok = (stale_sec is not None and stale_sec <= max_stale) or (
            stale_sec is None
            and file_age_sec is not None
            and file_age_sec <= max_stale
            and running
        )
        live = bool(running and spid is not None and (os_alive or heartbeat_ok))

        def _fmt_sec(sec: float) -> str:
            return f"{sec:.0f}s" if sec < 120 else f"{sec / 60.0:.1f}m"

        if not running:
            feed_label = "STOPPED"
            feed_color = t["warn"]
            if stale_sec is not None:
                age_text = _fmt_sec(stale_sec)
            elif file_age_sec is not None:
                age_text = f"file {_fmt_sec(file_age_sec)}"
            else:
                ended = as_dt(str(status.get("ended_at", "")))
                age_text = f"ended {ended.strftime('%m/%d %H:%M')}" if ended else "—"
        elif live:
            feed_label = "LIVE"
            feed_color = t["good"]
            if stale_sec is not None:
                age_text = _fmt_sec(stale_sec)
            elif file_age_sec is not None:
                age_text = f"file {_fmt_sec(file_age_sec)}"
            else:
                age_text = "no last_update"
        else:
            feed_label = "STALE"
            feed_color = t["bad"]
            if stale_sec is not None:
                age_text = _fmt_sec(stale_sec)
            elif file_age_sec is not None:
                age_text = f"file {_fmt_sec(file_age_sec)}"
            else:
                age_text = "—"

        ax_banner.add_patch(
            mpatches.FancyBboxPatch(
                (0.02, 0.18),
                0.14,
                0.64,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor=t["panel_elev"],
                edgecolor=t["grid"],
                linewidth=0.8,
                transform=ax_banner.transAxes,
            )
        )
        ax_banner.text(
            0.09,
            0.5,
            feed_label,
            ha="center",
            va="center",
            transform=ax_banner.transAxes,
            fontsize=11,
            fontweight="700",
            color=feed_color,
        )
        ax_banner.text(
            0.20,
            0.78,
            "Voice → glucose · autonomous research",
            ha="left",
            va="center",
            transform=ax_banner.transAxes,
            fontsize=14,
            fontweight="700",
            color=t["text"],
        )
        ax_banner.text(
            0.20,
            0.38,
            f"Monitor {now.strftime('%H:%M:%S')}  ·  Data age {age_text}  ·  "
            f"{len(rows)} eval rows",
            ha="left",
            va="center",
            transform=ax_banner.transAxes,
            fontsize=9,
            color=t["muted"],
        )

        # 1) Selection score
        x_s, y_s = metric_series(rows, "selection_score")
        ax = ax00
        if y_s:
            ax.plot(x_s, y_s, color=t["accent"], linewidth=2.0, label="Each eval", alpha=0.95)
            ax.plot(
                x_s,
                running_best(y_s, lower_is_better=True),
                color=t["good"],
                linewidth=2.0,
                label="Best so far",
            )
            ax.scatter([x_s[-1]], [y_s[-1]], color=t["rose"], s=42, zorder=5, label="Latest", edgecolors=t["text"], linewidths=0.6)
            ax.axhline(
                args.target_selection_score,
                color=t["muted"],
                linestyle="--",
                linewidth=1.1,
                alpha=0.85,
                label=f"Target ≤ {args.target_selection_score:g}",
            )
            ax.set_title("Selection score — lower is better")
            ax.set_xlabel("Time")
            ax.set_ylabel("Score")
            _legend_top_center(ax, t, ncol=4)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        else:
            _empty_message(ax, "Selection score", "No scores logged yet.", t)

        # 2) MAE
        ax = ax01
        mae_specs = [
            ("pers_mae", t["c5"], "Personalized"),
            ("pop_mae", t["c2"], "Population"),
            ("temp_mae", t["c6"], "Temporal"),
        ]
        drew = False
        for key, color, label in mae_specs:
            x, y = metric_series(rows, key)
            if y:
                ax.plot(x, y, color=color, linewidth=1.9, label=label, alpha=0.92)
                drew = True
        ax.axhline(
            args.target_mae,
            color=t["muted"],
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label=f"Target ≤ {args.target_mae:g}",
        )
        ax.set_title("Mean absolute error (mg/dL)")
        ax.set_xlabel("Time")
        ax.set_ylabel("MAE")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        if drew:
            _legend_top_center(ax, t, ncol=4)
        else:
            _empty_message(ax, "MAE", "No MAE series yet.", t)

        # 3) Correlation
        ax = ax10
        corr_specs = [
            ("pop_r", t["c1"], "Population r"),
            ("temp_r", t["c3"], "Temporal r"),
        ]
        drew = False
        for key, color, label in corr_specs:
            x, y = metric_series(rows, key)
            if y:
                ax.plot(x, y, color=color, linewidth=1.9, label=label, alpha=0.92)
                drew = True
        ax.axhline(0.0, color=t["muted"], linewidth=1.0, alpha=0.45)
        ax.axhline(
            args.target_pop_r,
            color=t["c1"],
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label=f"Pop target ≥ {args.target_pop_r:g}",
        )
        ax.axhline(
            args.target_temp_r,
            color=t["c3"],
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label=f"Temp target ≥ {args.target_temp_r:g}",
        )
        ax.set_title("Correlation (Pearson r) — higher is better")
        ax.set_xlabel("Time")
        ax.set_ylabel("r")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        if drew:
            _legend_top_center(ax, t, ncol=4)
        else:
            _empty_message(ax, "Correlation", "No correlation series yet.", t)

        # 4) Outcome mix
        ax = ax11
        win_sizes = [20, 50, 200]
        labels = ["Last 20", "Last 50", "Last 200"]
        keep_vals: List[float] = []
        discard_vals: List[float] = []
        err_vals: List[float] = []
        for w in win_sizes:
            window_rows = rows[-w:] if len(rows) > w else rows
            s = summarize(window_rows)
            keep_vals.append(s["keep"])
            discard_vals.append(s["discard"])
            err_vals.append(s["error"])
        xpos = list(range(len(win_sizes)))
        w = 0.62
        ax.bar(xpos, keep_vals, width=w, color=t["good"], label="Keep", edgecolor=t["bg"], linewidth=0.6)
        ax.bar(xpos, discard_vals, width=w, bottom=keep_vals, color=t["warn"], label="Discard", edgecolor=t["bg"], linewidth=0.6)
        stacked_bottom = [k + d for k, d in zip(keep_vals, discard_vals)]
        ax.bar(xpos, err_vals, width=w, bottom=stacked_bottom, color=t["bad"], label="Error", edgecolor=t["bg"], linewidth=0.6)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=8, color=t["muted"])
        ax.set_title("Outcomes by window")
        ax.set_xlabel("Evaluation window")
        ax.set_ylabel("Count")
        _legend_top_center(ax, t, ncol=3)

        # --- Side panel: structured KPI ---
        ax_side.clear()
        ax_side.axis("off")
        ax_side.set_facecolor(t["bg"])

        best_s = status.get("best_selection_score", "—")
        phase = status.get("phase", "—")
        cycle = status.get("cycle", "—")

        def line(k: str, v: str) -> str:
            return f"{k:16} {v}"

        blocks = [
            ("RUN", [
                line("Status", "running" if status.get("running") else "idle"),
                line("Phase", str(phase)),
                line("Cycle", str(cycle)),
                line("PID", str(status.get("pid", "—"))),
                line("LLM", str(status.get("llm_model", "—"))),
                line("Best score", str(best_s)),
            ]),
            ("VOLUME", [
                line("Rows", f"{int(summary['total'])}"),
                line("Keep rate", f"{summary['keep_rate']:.1%}"),
                line("Error rate", f"{summary['error_rate']:.1%}"),
                line("Early-stop", f"{int(early['early_stop'])}/{int(early['evaluated'])} ({early['rate']:.0%})"),
            ]),
            ("CLINICAL (avg 20)", [
                line("Clarke A+B", "—" if pop_clarke_ab is None else f"{pop_clarke_ab:.2f}%"),
                line("Pop MARD", "—" if pop_mard is None else f"{pop_mard:.2f}%"),
                line("Bias", "—" if pop_bias is None else f"{pop_bias:.2f}"),
            ]),
            ("SIGNAL GATE (20)", [
                line("Pass rate", "—" if gate_rate is None else f"{gate_rate:.3f}"),
                line("Penalty", "—" if gate_pen is None else f"{gate_pen:.3f}"),
            ]),
            ("EXPLORATION", [
                line("Diversity slot", f"{int(diversity_recent['hits'])}/{int(diversity_recent['total'])} ({diversity_recent['rate']:.0%})"),
                line("Sources (50)", src_rates[:120] + ("…" if len(src_rates) > 120 else "")),
            ]),
        ]

        ax_side.add_patch(
            mpatches.FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.02,rounding_size=0.015",
                facecolor=t["panel"],
                edgecolor=t["grid"],
                linewidth=0.9,
                transform=ax_side.transAxes,
                zorder=0,
            )
        )

        y = 0.96
        ax_side.text(
            0.05,
            y,
            "Summary",
            transform=ax_side.transAxes,
            fontsize=13,
            fontweight="700",
            color=t["text"],
            va="top",
            zorder=2,
        )
        y -= 0.065
        for title, lines in blocks:
            ax_side.text(
                0.05,
                y,
                title,
                transform=ax_side.transAxes,
                fontsize=10,
                fontweight="700",
                color=t["accent"],
                va="top",
                zorder=2,
            )
            y -= 0.042
            for ln in lines:
                ax_side.text(
                    0.07,
                    y,
                    ln,
                    transform=ax_side.transAxes,
                    fontsize=8.5,
                    family="monospace",
                    color=t["muted"],
                    va="top",
                    zorder=2,
                )
                y -= 0.032
            y -= 0.016

        ax_side.text(
            0.05,
            0.14,
            "Targets",
            transform=ax_side.transAxes,
            fontsize=10,
            fontweight="700",
            color=t["accent"],
            va="bottom",
            zorder=2,
        )
        ax_side.text(
            0.07,
            0.1,
            f"score ≤ {args.target_selection_score:g}  ·  MAE ≤ {args.target_mae:g}\n"
            f"pop r ≥ {args.target_pop_r:g}  ·  temp r ≥ {args.target_temp_r:g}",
            transform=ax_side.transAxes,
            fontsize=8,
            color=t["muted"],
            va="top",
            zorder=2,
        )

        # --- Loop log tail (same files as background `python -u ... > loop.log 2> loop.err`) ---
        ax_log.clear()
        ax_log.axis("off")
        ax_log.set_facecolor(t["bg"])
        ax_log.add_patch(
            mpatches.FancyBboxPatch(
                (0.005, 0.02),
                0.99,
                0.96,
                boxstyle="round,pad=0.01,rounding_size=0.012",
                facecolor=t["panel"],
                edgecolor=t["grid"],
                linewidth=0.8,
                transform=ax_log.transAxes,
                zorder=0,
            )
        )
        hint = format_processing_hint(status)
        out_tail = read_text_tail_lines(log_out_path, args.log_stdout_lines)
        err_tail = read_text_tail_lines(log_err_path, args.log_stderr_lines)
        log_block = (
            "── Status (what the loop is doing) ──\n"
            + hint
            + "\n\n── loop.log (tail) ──\n"
            + out_tail
            + "\n\n── loop.err.log (tail) ──\n"
            + err_tail
        )
        ax_log.text(
            0.012,
            0.98,
            log_block,
            transform=ax_log.transAxes,
            fontsize=7,
            family="monospace",
            color=t["muted"],
            va="top",
            ha="left",
            linespacing=1.12,
            zorder=2,
        )

        if stale_alert:
            fig.text(
                0.5,
                0.008,
                f"Stale alert: no update for >{args.stale_alert_minutes} min while running.  {restart_hint_command()}",
                ha="center",
                fontsize=8,
                color=t["bad"],
                wrap=True,
            )

        if args.snapshot_minutes > 0:
            bucket_seconds = max(args.snapshot_minutes, 1) * 60
            bucket = int(now.timestamp() // bucket_seconds)
            if bucket != last_snapshot_bucket:
                last_snapshot_bucket = bucket
                out_file = snapshot_dir / f"dashboard_{now.strftime('%Y%m%d_%H%M%S')}.png"
                fig.savefig(
                    out_file,
                    dpi=max(72, args.snapshot_dpi),
                    bbox_inches="tight",
                    facecolor=t["bg"],
                )

    refresh(0)
    _anim = FuncAnimation(fig, refresh, interval=max(args.interval_ms, 1000), cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
