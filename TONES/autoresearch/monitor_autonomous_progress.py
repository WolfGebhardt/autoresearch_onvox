#!/usr/bin/env python3
"""Monitor TONES autonomous loop progress/status."""

from __future__ import annotations

import argparse
import csv
import json
import math
import msvcrt
import os
import shutil
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

TONES_ROOT = Path(__file__).resolve().parent.parent
STATUS_DEFAULT = TONES_ROOT / "output" / "autoresearch" / "status.json"
TSV_DEFAULT = TONES_ROOT / "output" / "autoresearch" / "autonomous_runs.tsv"
LAUNCHER_PATH = TONES_ROOT / "autoresearch" / "start_tones_autonomous.ps1"
SPARK_CHARS = " .:-=+*#%@"


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


USE_COLOR = _supports_color()
RESET = "\033[0m"
COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bold": "\033[1m",
}


def colorize(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{COLORS.get(color, '')}{text}{RESET}"


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


def summarize(rows: List[Dict[str, str]]) -> Dict:
    out = {"total": len(rows), "keep": 0, "discard": 0, "error": 0}
    for r in rows:
        s = r.get("status", "")
        if s in out:
            out[s] += 1
    out["error_rate"] = (out["error"] / out["total"]) if out["total"] else 0.0
    return out


def _parse_float(value: str) -> Optional[float]:
    try:
        num = float(value)
        if math.isfinite(num):
            return num
    except Exception:
        return None
    return None


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _extract_scores(rows: List[Dict[str, str]]) -> List[float]:
    scores: List[float] = []
    for row in rows:
        if row.get("status") not in {"keep", "discard"}:
            continue
        score = _parse_float(row.get("selection_score", ""))
        if score is not None:
            scores.append(score)
    return scores


def _best_so_far(series: List[float]) -> List[float]:
    out: List[float] = []
    best = float("inf")
    for x in series:
        best = min(best, x)
        out.append(best)
    return out


def _resample(values: List[float], width: int) -> List[float]:
    if not values:
        return []
    if len(values) <= width:
        return values
    out: List[float] = []
    step = len(values) / width
    for i in range(width):
        idx = int(i * step)
        out.append(values[min(idx, len(values) - 1)])
    return out


def sparkline(values: List[float], width: int = 60, invert: bool = False) -> str:
    vals = _resample(values, max(width, 1))
    if not vals:
        return "-"
    vmin = min(vals)
    vmax = max(vals)
    if vmax <= vmin:
        return "=" * len(vals)
    out = []
    for v in vals:
        n = (v - vmin) / (vmax - vmin)
        if invert:
            n = 1.0 - n
        idx = int(round(n * (len(SPARK_CHARS) - 1)))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def _recent_eval_rate_per_hour(rows: List[Dict[str, str]], window: int = 30) -> float:
    recent = [r for r in rows if r.get("status") in {"keep", "discard"}][-window:]
    if len(recent) < 2:
        return 0.0
    t0 = _parse_timestamp(recent[0].get("timestamp", ""))
    t1 = _parse_timestamp(recent[-1].get("timestamp", ""))
    if not t0 or not t1:
        return 0.0
    hours = (t1 - t0).total_seconds() / 3600.0
    if hours <= 0:
        return 0.0
    return (len(recent) - 1) / hours


def _split_bar(keep: int, discard: int, error: int, width: int = 40) -> str:
    total = keep + discard + error
    if total <= 0:
        return "." * width
    k = int(round(width * keep / total))
    d = int(round(width * discard / total))
    e = width - k - d
    if e < 0:
        e = 0
    seg_keep = colorize("K" * k, "green")
    seg_discard = colorize("D" * d, "yellow")
    seg_error = colorize("E" * e, "red")
    return f"{seg_keep}{seg_discard}{seg_error}"


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _evaluated_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    eval_rows = [r for r in rows if r.get("status") in {"keep", "discard", "error"}]
    def _sort_key(r: Dict[str, str]):
        c = _parse_int(r.get("cycle", ""))
        t = _parse_timestamp(r.get("timestamp", ""))
        return (c if c is not None else 10**9, t or datetime.min)
    return sorted(eval_rows, key=_sort_key)


def _score_series(rows: List[Dict[str, str]]) -> List[float]:
    vals: List[float] = []
    for r in rows:
        if r.get("status") in {"keep", "discard"}:
            s = _parse_float(r.get("selection_score", ""))
            if s is not None:
                vals.append(s)
    return vals


def _metric_series(rows: List[Dict[str, str]], field: str) -> List[float]:
    vals: List[float] = []
    for r in rows:
        if r.get("status") not in {"keep", "discard"}:
            continue
        v = _parse_float(r.get(field, ""))
        if v is not None:
            vals.append(v)
    return vals


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _status_chip(status: str) -> str:
    if status == "keep":
        return colorize("KEEP", "green")
    if status == "discard":
        return colorize("DISCARD", "yellow")
    if status == "error":
        return colorize("ERROR", "red")
    return status or "-"


def _rows_by_key_best(rows: List[Dict[str, str]], key: str, max_items: int = 5) -> List[tuple]:
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

    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for r in rows:
        if key == "source":
            k = _source_of(r)
        else:
            k = r.get(key, "-")
        stats.setdefault(k, {"count": 0, "keep": 0, "best": None})
        stats[k]["count"] = int(stats[k]["count"] or 0) + 1
        if r.get("status") == "keep":
            stats[k]["keep"] = int(stats[k]["keep"] or 0) + 1
        sc = _parse_float(r.get("selection_score", ""))
        if sc is not None:
            best = stats[k]["best"]
            stats[k]["best"] = sc if best is None else min(float(best), sc)
    items = []
    for k, v in stats.items():
        items.append((k, int(v["count"] or 0), int(v["keep"] or 0), v["best"]))
    items.sort(key=lambda x: (x[3] if x[3] is not None else 10**9, -x[1], x[0]))
    return items[:max_items]


def _bar(pct: float, width: int = 24, fill: str = "#") -> str:
    p = max(0.0, min(1.0, pct))
    n = int(round(p * width))
    return fill * n + "." * (width - n)


def _minutes_since(ts: Optional[datetime]) -> Optional[float]:
    if ts is None:
        return None
    return (datetime.now() - ts).total_seconds() / 60.0


def _seconds_since(ts: Optional[datetime]) -> Optional[float]:
    if ts is None:
        return None
    return (datetime.now() - ts).total_seconds()


def _early_stop_stats(rows: List[Dict[str, str]]) -> Dict[str, float]:
    eval_rows = [r for r in rows if r.get("status") in {"keep", "discard"}]
    n_eval = len(eval_rows)
    n_early = sum(
        1
        for r in eval_rows
        if "early_stop_no_temporal" in str(r.get("notes", ""))
    )
    rate = (n_early / n_eval) if n_eval else 0.0
    return {"early_stop": n_early, "evaluated": n_eval, "rate": rate}


def _recent_metric_avg(rows: List[Dict[str, str]], field: str, window: int = 20) -> Optional[float]:
    vals = _metric_series(rows, field)[-window:]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _recent_rationale_usage(
    rows: List[Dict[str, str]], needle: str, window: int = 20
) -> Dict[str, float]:
    eval_rows = [r for r in rows if r.get("status") in {"keep", "discard"}]
    recent = eval_rows[-window:] if len(eval_rows) > window else eval_rows
    total = len(recent)
    hits = sum(1 for r in recent if needle in str(r.get("notes", "")))
    rate = (hits / total) if total else 0.0
    return {"hits": float(hits), "total": float(total), "rate": rate}


def _restart_hint() -> str:
    launcher = str(LAUNCHER_PATH)
    return (
        f'powershell -ExecutionPolicy Bypass -File "{launcher}" -Stop; '
        f'powershell -ExecutionPolicy Bypass -File "{launcher}" -Background'
    )


def _metric_delta(values: List[float], lower_is_better: bool) -> Dict[str, Optional[float]]:
    if not values:
        return {"start": None, "last": None, "best": None, "delta": None, "delta_pct": None}
    start = values[0]
    last = values[-1]
    best = min(values) if lower_is_better else max(values)
    delta = (start - last) if lower_is_better else (last - start)
    delta_pct = (delta / abs(start)) if start not in (0.0, -0.0) else None
    return {"start": start, "last": last, "best": best, "delta": delta, "delta_pct": delta_pct}


def _delta_text(stats: Dict[str, Optional[float]]) -> str:
    d = stats.get("delta")
    p = stats.get("delta_pct")
    if d is None:
        return "-"
    sign = "+" if d >= 0 else "-"
    if p is None:
        return f"{sign}{abs(d):.4f}"
    return f"{sign}{abs(d):.4f} ({sign}{abs(p):.1%})"


def _read_key_nonblocking() -> str:
    if os.name != "nt":
        return ""
    try:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            return ch.lower()
    except Exception:
        return ""
    return ""


def pid_alive(pid_value) -> bool:
    alive = False
    if isinstance(pid_value, int):
        try:
            if os.name == "nt":
                proc = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        f"Get-Process -Id {pid_value} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                alive = proc.returncode == 0 and proc.stdout.strip() != ""
            else:
                os.kill(pid_value, 0)
                alive = True
        except Exception:
            alive = False
    return alive


def evaluate_health(
    status: Dict,
    rows: List[Dict[str, str]],
    window: int,
    max_error_rate: float,
    min_evaluated: int,
    require_running: bool,
    max_stale_minutes: int,
) -> Dict:
    recent_rows = rows[-window:] if len(rows) > window else rows
    s_recent = summarize(recent_rows)
    evaluated_recent = s_recent["keep"] + s_recent["discard"]
    running = bool(status.get("running", False))
    alive = pid_alive(status.get("pid", "-"))
    last_up = _parse_timestamp(str(status.get("last_update", "")))
    stale_min = _minutes_since(last_up)
    stale_ok = True
    if max_stale_minutes > 0:
        stale_ok = (
            stale_min is not None
            and stale_min <= float(max_stale_minutes)
        )

    checks = {
        "running": (running and alive) if require_running else True,
        "stale": stale_ok,
        "error_rate": s_recent["error_rate"] <= max_error_rate,
        "evaluated_count": evaluated_recent >= min_evaluated,
    }
    ok = all(checks.values())

    return {
        "ok": ok,
        "checks": checks,
        "recent_summary": s_recent,
        "evaluated_recent": evaluated_recent,
        "window": window,
        "max_error_rate": max_error_rate,
        "min_evaluated": min_evaluated,
        "running": running,
        "pid_alive": alive,
        "stale_minutes": stale_min,
        "max_stale_minutes": max_stale_minutes,
    }


def print_snapshot(
    status: Dict,
    rows: List[Dict[str, str]],
    watch_interval_s: int = 5,
    stale_alert_minutes: int = 8,
) -> None:
    term_width = shutil.get_terminal_size((120, 30)).columns
    width = max(100, min(term_width, 180))
    eval_rows = _evaluated_rows(rows)
    s = summarize(eval_rows)
    recent_rows = eval_rows[-200:] if len(eval_rows) > 200 else eval_rows
    recent_50 = eval_rows[-50:] if len(eval_rows) > 50 else eval_rows
    recent_20 = eval_rows[-20:] if len(eval_rows) > 20 else eval_rows
    s_recent = summarize(recent_rows)
    s_50 = summarize(recent_50)
    s_20 = summarize(recent_20)
    last = status.get("last_result") or (eval_rows[-1] if eval_rows else {})
    print(colorize("=" * width, "cyan"))
    print(colorize("TONES Autonomous Monitor (Advanced)", "bold"))
    running = bool(status.get("running", False))
    pid = status.get("pid", "-")
    alive = pid_alive(pid)
    last_up = _parse_timestamp(str(status.get("last_update", "")))
    stale_min = _minutes_since(last_up)
    stale_sec = _seconds_since(last_up)
    stale_txt = "-" if stale_min is None else f"{stale_min:.1f}m ago"
    freshness_budget = max(3 * watch_interval_s, 20)
    is_live = bool(running and alive and stale_sec is not None and stale_sec <= freshness_budget)
    live_badge = colorize("LIVE", "green") if is_live else colorize("STALE", "yellow")
    now_txt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"state={colorize('RUNNING', 'green') if (running and alive) else colorize('STOPPED', 'red')} "
        f"| feed={live_badge} | pid={pid} | model={status.get('llm_model', '-')} "
        f"| cycle={status.get('cycle', '-')} | phase={status.get('phase', '-')}"
    )
    print(f"monitor_time={now_txt} | last_update={status.get('last_update', '-')} ({stale_txt})")
    stale_alert = (
        stale_sec is not None
        and stale_sec > max(stale_alert_minutes, 1) * 60
        and running
        and alive
    )
    if stale_alert:
        print(
            colorize(
                f"[stale-alert] data older than {stale_alert_minutes}m; loop may be busy or stuck.",
                "red",
            )
        )
        print(colorize(f"[restart-hint] {_restart_hint()}", "yellow"))
    print(colorize("-" * width, "cyan"))
    keep_rate = (s["keep"] / s["total"]) if s["total"] else 0.0
    print(
        f"evaluated={s['total']}  keep={s['keep']}  discard={s['discard']}  error={s['error']}  "
        f"keep_rate={keep_rate:.1%}  error_rate={s['error_rate']:.1%}"
    )
    print(f"mix       {_split_bar(s['keep'], s['discard'], s['error'], width=48)}")
    print(
        f"recent20  K={s_20['keep']:>3} D={s_20['discard']:>3} E={s_20['error']:>2} | "
        f"keep_rate={(s_20['keep'] / s_20['total'] if s_20['total'] else 0.0):.1%}"
    )
    print(
        f"recent50  K={s_50['keep']:>3} D={s_50['discard']:>3} E={s_50['error']:>2} | "
        f"keep_rate={(s_50['keep'] / s_50['total'] if s_50['total'] else 0.0):.1%}"
    )
    print(
        f"recent200 K={s_recent['keep']:>3} D={s_recent['discard']:>3} E={s_recent['error']:>2} | "
        f"keep_rate={(s_recent['keep'] / s_recent['total'] if s_recent['total'] else 0.0):.1%}"
    )
    early = _early_stop_stats(eval_rows)
    print(
        f"early_stop_pruned={int(early['early_stop'])}/{int(early['evaluated'])} "
        f"({early['rate']:.1%})"
    )
    clarke_ab_recent = _recent_metric_avg(eval_rows, "pop_clarke_ab_pct", window=20)
    pop_mard_recent = _recent_metric_avg(eval_rows, "pop_mard", window=20)
    pop_bias_recent = _recent_metric_avg(eval_rows, "pop_bias", window=20)
    gate_rate_recent = _recent_metric_avg(eval_rows, "signal_gate_pass_rate", window=20)
    gate_penalty_recent = _recent_metric_avg(eval_rows, "signal_gate_penalty", window=20)
    diversity_recent = _recent_rationale_usage(
        eval_rows, "diversity_forced_slot", window=20
    )
    print(
        "clinical_recent20 "
        f"pop_clarke_ab={_format_float(clarke_ab_recent, 2)}% "
        f"pop_mard={_format_float(pop_mard_recent, 2)}% "
        f"pop_bias={_format_float(pop_bias_recent, 2)}"
    )
    print(
        "signal_gate_recent20 "
        f"pass_rate={_format_float(gate_rate_recent, 3)} "
        f"penalty={_format_float(gate_penalty_recent, 3)}"
    )
    print(
        "innovation_recent20 "
        f"diversity_slot={int(diversity_recent['hits'])}/{int(diversity_recent['total'])} "
        f"({diversity_recent['rate']:.1%})"
    )
    eval_rate = _recent_eval_rate_per_hour(rows, window=30)
    print(f"throughput ~{eval_rate:.1f} evals/hour (recent)")
    print(colorize("-" * width, "cyan"))

    scores = _score_series(eval_rows)
    if scores:
        score_stats = _metric_delta(scores, lower_is_better=True)
        print(
            f"selection_score: best={_format_float(score_stats['best'])}  last={_format_float(score_stats['last'])}  "
            f"start={_format_float(score_stats['start'])}  improvement={_delta_text(score_stats)}"
        )
        print(
            f"rolling avg (10/25/50): "
            f"{_format_float(_avg(scores[-10:]))} / {_format_float(_avg(scores[-25:]))} / {_format_float(_avg(scores[-50:]))}"
        )
        recent_scores = scores[-60:]
        best_series = _best_so_far(scores)[-60:]
        print(colorize("trend(recent): lower is better", "magenta"))
        print(f"  {colorize(sparkline(recent_scores, width=min(90, width - 10), invert=True), 'yellow')}")
        print(colorize("trend(best-so-far):", "magenta"))
        print(f"  {colorize(sparkline(best_series, width=min(90, width - 10), invert=True), 'green')}")
    else:
        print("selection_score: -")

    print(colorize("-" * width, "cyan"))
    print(colorize("metric trends (last 80 evals)", "magenta"))
    metric_specs = [
        ("pers_mae", True, "pers_mae"),
        ("pop_mae", True, "pop_mae"),
        ("temp_mae", True, "temp_mae"),
        ("pop_r", False, "pop_r"),
        ("temp_r", False, "temp_r"),
        ("sig_gate", False, "signal_gate_pass_rate"),
    ]
    for label, lower_better, field in metric_specs:
        vals = _metric_series(eval_rows, field)[-80:]
        if not vals:
            print(f"  {label:<8} no data")
            continue
        stats = _metric_delta(vals, lower_better)
        chart = sparkline(vals, width=min(70, width - 46), invert=lower_better)
        direction = "low->good" if lower_better else "high->good"
        chart_col = "yellow" if lower_better else "green"
        print(
            f"  {label:<8} {colorize(chart, chart_col)} "
            f"last={_format_float(stats['last'])} best={_format_float(stats['best'])} "
            f"delta={_delta_text(stats)} {direction}"
        )

    print(colorize("-" * width, "cyan"))
    model_top = _rows_by_key_best(eval_rows, "model_name", max_items=4)
    feat_top = _rows_by_key_best(eval_rows, "feature_key", max_items=4)
    norm_top = _rows_by_key_best(eval_rows, "normalization", max_items=4)
    source_top = _rows_by_key_best(eval_rows, "source", max_items=6)
    if model_top:
        print(colorize("by model (count / keep / best score)", "blue"))
        for name, cnt, keep, best in model_top:
            kr = (keep / cnt) if cnt else 0.0
            print(f"  {name:<12} {cnt:>3} / {keep:>3}  keep_rate={kr:>6.1%}  best={_format_float(best)}")
    if feat_top:
        print(colorize("by feature set (count / keep / best score)", "blue"))
        for name, cnt, keep, best in feat_top:
            kr = (keep / cnt) if cnt else 0.0
            short = (name[:36] + "...") if len(name) > 39 else name
            print(f"  {short:<40} {cnt:>3} / {keep:>3}  keep_rate={kr:>6.1%}  best={_format_float(best)}")
    if norm_top:
        print(colorize("by normalization (count / keep / best score)", "blue"))
        for name, cnt, keep, best in norm_top:
            kr = (keep / cnt) if cnt else 0.0
            print(f"  {name:<10} {cnt:>3} / {keep:>3}  keep_rate={kr:>6.1%}  best={_format_float(best)}")
    if source_top:
        print(colorize("by source (count / keep / best score)", "blue"))
        for name, cnt, keep, best in source_top:
            kr = (keep / cnt) if cnt else 0.0
            src = name or "-"
            print(f"  {src:<12} {cnt:>3} / {keep:>3}  keep_rate={kr:>6.1%}  best={_format_float(best)}")

    print(colorize("-" * width, "cyan"))
    seen = set()
    top_pool = []
    for r in sorted(
        [x for x in eval_rows if _parse_float(x.get("selection_score", "")) is not None],
        key=lambda x: float(x["selection_score"]),
    ):
        sig = (
            r.get("exp_key", ""),
            r.get("selection_score", ""),
            r.get("cycle", ""),
        )
        if sig in seen:
            continue
        seen.add(sig)
        top_pool.append(r)
    top_rows = top_pool[:5]
    if top_rows:
        print(colorize("top configs (best selection_score)", "green"))
        for i, r in enumerate(top_rows, 1):
            sc = _format_float(_parse_float(r.get("selection_score", "")))
            print(
                f"  #{i} score={sc}  model={r.get('model_name','-')}  n_mfcc={r.get('n_mfcc','-')}  "
                f"feat={r.get('feature_key','-')}  norm={r.get('normalization','-')}  "
                f"source={r.get('source','-')}  cycle={r.get('cycle','-')}"
            )

    if last:
        seq = "".join(
            "K" if r.get("status") == "keep" else ("D" if r.get("status") == "discard" else "E")
            for r in eval_rows[-40:]
        )
        print(colorize("-" * width, "cyan"))
        print(f"recent outcomes (40): {seq or '-'}")
        print(
            f"last_result: status={_status_chip(last.get('status','-'))} "
            f"cycle={last.get('cycle','-')} "
            f"model={last.get('model_name','-')} feat={last.get('feature_key','-')} "
            f"norm={last.get('normalization','-')} score={last.get('selection_score','-')}"
        )
    print(colorize("=" * width, "cyan"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor TONES autonomous loop progress.")
    parser.add_argument("--status-file", default=str(STATUS_DEFAULT))
    parser.add_argument("--tsv-file", default=str(TSV_DEFAULT))
    parser.add_argument("--watch", action="store_true", help="Continuously refresh output.")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval seconds.")
    parser.add_argument("--tui", action="store_true", help="Interactive watch mode with keyboard shortcuts.")
    parser.add_argument(
        "--stale-alert-minutes",
        type=int,
        default=8,
        help="Show stale warning/restart hint if feed age exceeds this many minutes.",
    )
    parser.add_argument("--health", action="store_true", help="Run health checks and exit with code.")
    parser.add_argument("--health-window", type=int, default=100, help="Recent row window for health check.")
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.30,
        help="Maximum tolerated error rate in health window (0-1).",
    )
    parser.add_argument(
        "--min-evaluated",
        type=int,
        default=10,
        help="Minimum keep+discard rows required in health window.",
    )
    parser.add_argument(
        "--require-running",
        action="store_true",
        help="Health fails if loop process is not running.",
    )
    parser.add_argument(
        "--max-stale-minutes",
        type=int,
        default=8,
        help="Health fails if status last_update is older than this many minutes (0 disables stale check).",
    )
    args = parser.parse_args()

    status_path = Path(args.status_file)
    tsv_path = Path(args.tsv_file)

    status = read_status(status_path)
    rows = read_rows(tsv_path)

    if args.health:
        health = evaluate_health(
            status=status,
            rows=rows,
            window=max(args.health_window, 1),
            max_error_rate=max(min(args.max_error_rate, 1.0), 0.0),
            min_evaluated=max(args.min_evaluated, 0),
            require_running=args.require_running,
            max_stale_minutes=max(args.max_stale_minutes, 0),
        )
        state = "HEALTHY" if health["ok"] else "UNHEALTHY"
        state_colored = colorize(state, "green" if health["ok"] else "red")
        print(
            f"{state_colored} | recent_window={health['window']} "
            f"error_rate={health['recent_summary']['error_rate']:.1%} "
            f"evaluated={health['evaluated_recent']} "
            f"running={health['running']} pid_alive={health['pid_alive']} "
            f"stale_min={_format_float(health.get('stale_minutes'), 1)}"
        )
        print(
            "checks:",
            f"running={colorize(str(health['checks']['running']), 'green' if health['checks']['running'] else 'red')}",
            f"stale={colorize(str(health['checks']['stale']), 'green' if health['checks']['stale'] else 'red')}",
            f"error_rate={colorize(str(health['checks']['error_rate']), 'green' if health['checks']['error_rate'] else 'red')}",
            f"evaluated_count={colorize(str(health['checks']['evaluated_count']), 'green' if health['checks']['evaluated_count'] else 'red')}",
        )
        raise SystemExit(0 if health["ok"] else 1)

    if not args.watch:
        print_snapshot(
            status,
            rows,
            watch_interval_s=max(args.interval, 1),
            stale_alert_minutes=max(args.stale_alert_minutes, 1),
        )
        return

    try:
        interval = max(args.interval, 1)
        paused = False
        refresh_now = False
        loop_mode = args.tui or True
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            live_status = read_status(status_path)
            live_rows = read_rows(tsv_path)
            print_snapshot(
                live_status,
                live_rows,
                watch_interval_s=interval,
                stale_alert_minutes=max(args.stale_alert_minutes, 1),
            )
            if loop_mode:
                print(
                    colorize(
                        f"[shortcuts] q=quit  p=pause/resume  r=refresh  1/2/5/0=set interval(s)  current={interval}s",
                        "cyan",
                    )
                )
                if paused:
                    print(colorize("[paused] auto-refresh is paused; press p to resume.", "yellow"))
            wait_s = 0.1
            waited = 0.0
            target = 0.1 if refresh_now else interval
            refresh_now = False
            while waited < target:
                key = _read_key_nonblocking() if loop_mode else ""
                if key == "q":
                    return
                if key == "p":
                    paused = not paused
                    break
                if key == "r":
                    refresh_now = True
                    break
                if key in {"1", "2", "5"}:
                    interval = int(key)
                    break
                if key == "0":
                    interval = 10
                    break
                if not paused:
                    time.sleep(wait_s)
                    waited += wait_s
                else:
                    time.sleep(wait_s)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
