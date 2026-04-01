"""
Evaluation Metrics & Clarke Error Grid
========================================
Clinical accuracy evaluation for glucose predictions.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# =============================================================================
# Clarke Error Grid
# =============================================================================

def clarke_error_grid(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, int]:
    """
    Classify predictions into Clarke Error Grid zones (A-E).

    Zone A: Clinically accurate (within 20% or both ≤70)
    Zone B: Benign errors (would not lead to inappropriate treatment)
    Zone C: Overcorrection errors
    Zone D: Failure to detect (dangerous under/over)
    Zone E: Erroneous treatment (most dangerous)

    Parameters
    ----------
    actual : np.ndarray
        Reference glucose values (mg/dL).
    predicted : np.ndarray
        Predicted glucose values (mg/dL).

    Returns
    -------
    dict
        Zone counts: {"A": n, "B": n, "C": n, "D": n, "E": n}
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    zones = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    for ref, pred in zip(actual, predicted):
        if (ref <= 70 and pred <= 70) or \
           abs(pred - ref) <= 20 or \
           (ref >= 70 and abs(pred - ref) / ref <= 0.20):
            zones["A"] += 1
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zones["E"] += 1
        elif (70 <= ref <= 290 and pred >= ref + 110) or \
             (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
            zones["C"] += 1
        elif (ref >= 240 and 70 <= pred <= 180) or \
             (ref <= 175 / 3 and 70 <= pred <= 180) or \
             (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
            zones["D"] += 1
        else:
            zones["B"] += 1

    return zones


def clarke_zone_percentages(zones: Dict[str, int]) -> Dict[str, float]:
    """Convert zone counts to percentages."""
    total = sum(zones.values())
    if total == 0:
        return {k: 0.0 for k in zones}
    return {k: 100 * v / total for k, v in zones.items()}


# =============================================================================
# Visualization
# =============================================================================

def plot_clarke_error_grid(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Clarke Error Grid",
    save_path: str = None,
) -> plt.Figure:
    """
    Generate a Clarke Error Grid plot.

    Parameters
    ----------
    actual : np.ndarray
        Reference glucose values (mg/dL).
    predicted : np.ndarray
        Predicted glucose values (mg/dL).
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    plt.Figure
    """
    zones = clarke_error_grid(actual, predicted)
    pct = clarke_zone_percentages(zones)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actual, predicted, alpha=0.6, s=30, c="#3498db", edgecolors="white", linewidth=0.5)

    # Perfect prediction line
    ax.plot([0, 400], [0, 400], "k-", linewidth=1, alpha=0.7)

    # Clinical thresholds
    for val in [70, 180]:
        ax.axhline(val, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(val, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlim(40, max(250, actual.max() * 1.1))
    ax.set_ylim(40, max(250, predicted.max() * 1.1))
    ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=12)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Zone annotation
    text = (
        f"Zone A: {zones['A']} ({pct['A']:.1f}%)\n"
        f"Zone B: {zones['B']} ({pct['B']:.1f}%)\n"
        f"A+B: {zones['A'] + zones['B']} ({pct['A'] + pct['B']:.1f}%)"
    )
    ax.text(
        0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Clarke Error Grid saved: %s", save_path)

    return fig


def plot_scatter_per_participant(
    participant_results: Dict,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot actual vs predicted scatter for each participant.

    Parameters
    ----------
    participant_results : dict
        {name: {"actual": array, "predictions": array, "mae": float, "r": float}}
    save_path : str, optional
        Path to save figure.
    """
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#34495e"]
    n = len(participant_results)
    if n == 0:
        return plt.figure()

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, (name, res) in enumerate(participant_results.items()):
        ax = axes[i]
        actual = res["actual"]
        predicted = res["predictions"]

        ax.scatter(actual, predicted, alpha=0.6, c=colors[i % len(colors)], s=30)

        # Perfect prediction line
        all_vals = np.concatenate([actual, predicted])
        vmin, vmax = all_vals.min(), all_vals.max()
        ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.5, linewidth=1)

        # +/- 15 mg/dL band
        ax.fill_between(
            [vmin, vmax],
            [vmin - 15, vmax - 15],
            [vmin + 15, vmax + 15],
            alpha=0.1, color="green",
        )

        ax.set_xlabel("Actual (mg/dL)")
        ax.set_ylabel("Predicted (mg/dL)")
        ax.set_title(f"{name}\nMAE={res['mae']:.1f}, r={res['r']:.2f}")

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Scatter plots saved: %s", save_path)

    return fig


def plot_model_comparison(
    participant_results: Dict,
    save_path: str = None,
) -> plt.Figure:
    """
    Bar charts comparing MAE and correlation across participants.
    """
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#34495e"]

    names = list(participant_results.keys())
    maes = [participant_results[p]["mae"] for p in names]
    rs = [participant_results[p]["r"] for p in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE
    ax = axes[0]
    bars = ax.bar(names, maes, color=colors[:len(names)])
    ax.axhline(10, color="red", linestyle="--", alpha=0.5, label="10 mg/dL target")
    ax.set_ylabel("MAE (mg/dL)")
    ax.set_title("Mean Absolute Error by Participant")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{mae:.1f}", ha="center", va="bottom", fontsize=9)

    # Correlation
    ax = axes[1]
    bars = ax.bar(names, rs, color=colors[:len(names)])
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.set_ylabel("Correlation (r)")
    ax.set_title("Pearson Correlation by Participant")
    ax.set_ylim(-0.5, 1.0)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Model comparison saved: %s", save_path)

    return fig
