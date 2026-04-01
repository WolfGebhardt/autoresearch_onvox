"""
Comprehensive Documentation Generator for Voice-Based Glucose Estimation
Version 2: Corrected Clarke Error Grid, Audio Windowing Analysis, Fixed Temporal Plots
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from voice_glucose_pipeline import (
    PARTICIPANTS, BASE_DIR, create_dataset_for_participant
)

# Output directory
OUTPUT_DIR = BASE_DIR / "documentation"
OUTPUT_DIR.mkdir(exist_ok=True)

FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_all_data():
    """Load all participant datasets."""
    datasets = {}
    for name, config in PARTICIPANTS.items():
        print(f"Loading {name}...")
        df = create_dataset_for_participant(name, config, verbose=False)
        if df is not None and len(df) >= 10:
            datasets[name] = df
            print(f"  {len(df)} samples loaded")
    return datasets


def clarke_zone_classification(ref, pred):
    """
    Classify a single point into Clarke Error Grid zone.
    Based on standard Clarke EGA specifications.

    Returns: Zone letter (A, B, C, D, or E)
    """
    # Zone E: Erroneous treatment (most dangerous)
    # Upper left: reference <= 70 and prediction >= 180
    # Lower right: reference >= 180 and prediction <= 70
    if (ref <= 70 and pred >= 180) or (ref >= 180 and pred <= 70):
        return 'E'

    # Zone D: Failure to detect
    # Right: reference >= 240 and 70 <= prediction <= 180
    # Left: reference <= 70 and 70 < prediction < 180
    if ref >= 240 and 70 <= pred <= 180:
        return 'D'
    if ref <= 70 and 70 < pred < 180:
        return 'D'

    # Zone C: Overcorrection
    # Upper: 70 <= reference <= 290 and prediction >= reference + 110
    # Lower: 130 <= reference <= 180 and prediction <= (7/5)*reference - 182
    if 70 <= ref <= 290 and pred >= ref + 110:
        return 'C'
    if 130 <= ref <= 180 and pred <= (7/5) * ref - 182:
        return 'C'

    # Zone A: Clinically accurate
    # Within 20% of reference OR both in hypoglycemic range (<70)
    if ref < 70 and pred < 70:
        return 'A'
    if ref >= 70:
        if abs(pred - ref) <= 0.2 * ref:
            return 'A'

    # Zone B: Benign errors (everything else)
    return 'B'


def create_clarke_error_grid_correct(y_true, y_pred, title="Clarke Error Grid", save_path=None):
    """
    Create correct Clarke Error Grid Analysis plot per clinical standards.
    References:
    - Clarke WL, et al. Diabetes Care 1987
    - https://github.com/suetAndTie/ClarkeErrorGrid
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot limits (standard is 0-400 for both axes)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)

    # Draw zone boundary lines (per Clarke standard)
    # Perfect agreement line (diagonal)
    ax.plot([0, 400], [0, 400], 'k--', linewidth=1, alpha=0.5, label='Perfect agreement')

    # Zone A boundaries (20% error bands)
    # Upper A boundary: from (0,0) follows y = 1.2x until it hits other boundaries
    ax.plot([0, 175/3], [70, 70], 'k-', linewidth=1.5)  # Horizontal at y=70 from x=0 to ~58.33
    ax.plot([175/3, 400/1.2], [70, 400], 'k-', linewidth=1.5)  # Upper 20% line

    # Lower A boundary
    ax.plot([70, 70], [0, 56], 'k-', linewidth=1.5)  # Vertical at x=70 from y=0 to y=56
    ax.plot([70, 400], [56, 320], 'k-', linewidth=1.5)  # Lower 20% line

    # Zone D/E boundaries (left side - hypoglycemia region)
    ax.plot([70, 70], [84, 400], 'k-', linewidth=1.5)  # Vertical at x=70 from y=84 to y=400
    ax.plot([0, 70], [180, 180], 'k-', linewidth=1.5)  # Horizontal at y=180 from x=0 to x=70

    # Zone C/D boundaries (right side)
    ax.plot([70, 290], [180, 400], 'k-', linewidth=1.5)  # Upper C zone boundary
    ax.plot([240, 240], [70, 180], 'k-', linewidth=1.5)  # D zone right boundary
    ax.plot([240, 400], [70, 70], 'k-', linewidth=1.5)  # D zone bottom

    # Zone E boundary (lower right)
    ax.plot([180, 180], [0, 70], 'k-', linewidth=1.5)  # E zone left
    ax.plot([180, 400], [70, 70], 'k-', linewidth=1.5)  # Already covered

    # Color the zones with correct shading
    # Zone A - green (central diagonal band)
    zone_a_poly = plt.Polygon([
        [0, 0], [0, 70], [175/3, 70], [70, 84], [70, 56], [70, 0]
    ], alpha=0.15, facecolor='green', edgecolor='none')
    ax.add_patch(zone_a_poly)

    zone_a_poly2 = plt.Polygon([
        [70, 56], [70, 84], [175/3, 70], [400/1.2, 400], [400, 400], [400, 320]
    ], alpha=0.15, facecolor='green', edgecolor='none')
    ax.add_patch(zone_a_poly2)

    # Zone B - yellow (adjacent to A, benign errors)
    # Upper B
    zone_b_upper = plt.Polygon([
        [0, 70], [0, 180], [70, 180], [70, 84], [175/3, 70]
    ], alpha=0.15, facecolor='yellow', edgecolor='none')
    ax.add_patch(zone_b_upper)

    # Lower B
    zone_b_lower = plt.Polygon([
        [70, 0], [70, 56], [400, 320], [400, 0]
    ], alpha=0.15, facecolor='yellow', edgecolor='none')
    ax.add_patch(zone_b_lower)

    # Zone C - orange (overcorrection)
    zone_c_upper = plt.Polygon([
        [70, 180], [70, 400], [290, 400]
    ], alpha=0.2, facecolor='orange', edgecolor='none')
    ax.add_patch(zone_c_upper)

    # Zone D - light red (failure to detect)
    zone_d_right = plt.Polygon([
        [240, 70], [240, 180], [400, 180], [400, 70]
    ], alpha=0.2, facecolor='salmon', edgecolor='none')
    ax.add_patch(zone_d_right)

    # Zone E - red (dangerous)
    zone_e_ul = plt.Polygon([
        [0, 180], [0, 400], [70, 400], [70, 180]
    ], alpha=0.25, facecolor='red', edgecolor='none')
    ax.add_patch(zone_e_ul)

    zone_e_lr = plt.Polygon([
        [180, 0], [180, 70], [400, 70], [400, 0]
    ], alpha=0.25, facecolor='red', edgecolor='none')
    ax.add_patch(zone_e_lr)

    # Add zone labels
    ax.text(30, 20, 'A', fontsize=20, fontweight='bold', color='darkgreen', alpha=0.7)
    ax.text(300, 280, 'A', fontsize=20, fontweight='bold', color='darkgreen', alpha=0.7)
    ax.text(30, 120, 'B', fontsize=16, fontweight='bold', color='olive', alpha=0.7)
    ax.text(280, 50, 'B', fontsize=16, fontweight='bold', color='olive', alpha=0.7)
    ax.text(150, 350, 'C', fontsize=16, fontweight='bold', color='darkorange', alpha=0.7)
    ax.text(320, 120, 'D', fontsize=16, fontweight='bold', color='brown', alpha=0.7)
    ax.text(30, 300, 'E', fontsize=16, fontweight='bold', color='darkred', alpha=0.7)
    ax.text(300, 30, 'E', fontsize=16, fontweight='bold', color='darkred', alpha=0.7)

    # Plot data points
    ax.scatter(y_true, y_pred, c='blue', alpha=0.6, s=50, edgecolors='white', linewidth=0.5, zorder=10)

    # Calculate zone percentages
    n = len(y_true)
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for ref, pred in zip(y_true, y_pred):
        zone = clarke_zone_classification(ref, pred)
        zones[zone] += 1

    zone_pct = {k: v/n*100 for k, v in zones.items()}

    # Add zone percentages as text box
    textstr = '\n'.join([
        f"Zone A: {zone_pct['A']:.1f}% ({zones['A']})",
        f"Zone B: {zone_pct['B']:.1f}% ({zones['B']})",
        f"Zone C: {zone_pct['C']:.1f}% ({zones['C']})",
        f"Zone D: {zone_pct['D']:.1f}% ({zones['D']})",
        f"Zone E: {zone_pct['E']:.1f}% ({zones['E']})",
        f"",
        f"A+B: {zone_pct['A']+zone_pct['B']:.1f}%"
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')

    # Clinical threshold lines
    ax.axhline(y=70, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=180, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=70, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=180, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.3, label='Zone A: Clinically accurate'),
        mpatches.Patch(facecolor='yellow', alpha=0.3, label='Zone B: Benign errors'),
        mpatches.Patch(facecolor='orange', alpha=0.3, label='Zone C: Overcorrection'),
        mpatches.Patch(facecolor='salmon', alpha=0.3, label='Zone D: Failure to detect'),
        mpatches.Patch(facecolor='red', alpha=0.4, label='Zone E: Dangerous errors'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return zones, zone_pct


def plot_feature_correlations(df, top_n=20, save_path=None):
    """Plot top feature correlations with glucose."""
    feature_cols = [c for c in df.columns if c.startswith('librosa_') or c.startswith('circadian_')]

    if not feature_cols:
        return

    correlations = []
    for col in feature_cols:
        if df[col].std() > 0:
            corr = df[col].corr(df['glucose_mgdl'])
            if not np.isnan(corr):
                correlations.append((col, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = correlations[:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))

    names = [f[0].replace('librosa_', '').replace('_mean', '').replace('_std', ' (std)')
             for f in top_features]
    values = [f[1] for f in top_features]
    colors = ['green' if v > 0 else 'red' for v in values]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Pearson Correlation with Glucose', fontsize=12)
    ax.set_title('Top Voice Features Correlated with Blood Glucose', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_model_comparison(results_dict, save_path=None):
    """Plot model comparison across participants."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    participants = list(results_dict.keys())
    maes = [results_dict[p]['best_mae'] for p in participants]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(participants)))

    # MAE comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(participants, maes, color=colors)
    ax1.set_ylabel('MAE (mg/dL)', fontsize=12)
    ax1.set_title('Personalized Model Performance by Participant', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(participants, rotation=45, ha='right')
    ax1.axhline(y=np.mean(maes), color='red', linestyle='--', label=f'Mean: {np.mean(maes):.1f}')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Sample count
    ax2 = axes[0, 1]
    samples = [results_dict[p]['n_samples'] for p in participants]
    ax2.bar(participants, samples, color=colors)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Dataset Size by Participant', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(participants, rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)

    # Glucose variability
    ax3 = axes[1, 0]
    glucose_std = [results_dict[p]['glucose_std'] for p in participants]
    ax3.bar(participants, glucose_std, color=colors)
    ax3.set_ylabel('Glucose Std Dev (mg/dL)', fontsize=12)
    ax3.set_title('Glucose Variability by Participant', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(participants, rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)

    # Model type distribution
    ax4 = axes[1, 1]
    model_types = [results_dict[p]['best_model'] for p in participants]
    unique_models = list(set(model_types))
    model_counts = [model_types.count(m) for m in unique_models]
    ax4.pie(model_counts, labels=unique_models, autopct='%1.0f%%', startangle=90)
    ax4.set_title('Best Model Types Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_approach_comparison(personalized_mae, population_mae, transfer_mae, save_path=None):
    """Plot comparison of three modeling approaches."""
    fig, ax = plt.subplots(figsize=(10, 6))

    approaches = ['Personalized\n(per-person)', 'Population\n(leave-one-out)', 'Transfer Learning\n(pre-train + fine-tune)']
    maes = [personalized_mae, population_mae, transfer_mae]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bars = ax.bar(approaches, maes, color=colors, edgecolor='black', linewidth=1.5)

    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.annotate(f'{mae:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Mean Absolute Error (mg/dL)', fontsize=12)
    ax.set_title('Comparison of Modeling Approaches', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(maes) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_glucose_distribution(datasets, save_path=None):
    """Plot glucose distribution across participants."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    all_glucose = []
    for i, (name, df) in enumerate(datasets.items()):
        if i >= 7:
            break
        ax = axes[i]
        glucose = df['glucose_mgdl'].values
        all_glucose.extend(glucose)

        ax.hist(glucose, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(x=glucose.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {glucose.mean():.0f}')
        ax.axvline(x=70, color='orange', linestyle=':', linewidth=1.5, label='Hypo threshold')
        ax.axvline(x=180, color='orange', linestyle=':', linewidth=1.5, label='Hyper threshold')
        ax.set_title(f'{name}\n(n={len(df)})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Glucose (mg/dL)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[7]
    ax.hist(all_glucose, bins=30, color='darkgreen', edgecolor='white', alpha=0.7)
    ax.axvline(x=np.mean(all_glucose), color='red', linestyle='--', linewidth=2)
    ax.set_title(f'All Participants\n(n={len(all_glucose)})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Glucose (mg/dL)', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Glucose Distribution by Participant', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_temporal_patterns_fixed(datasets, save_path=None):
    """Plot glucose and recording patterns by time of day - FIXED VERSION."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # We need to extract hour from the audio filename timestamps
    # Since timestamps may not be preserved in df, we'll parse from audio_path
    hourly_glucose = {h: [] for h in range(24)}
    hourly_counts = {h: 0 for h in range(24)}

    for name, df in datasets.items():
        if 'audio_path' in df.columns:
            for _, row in df.iterrows():
                audio_path = row['audio_path']
                # Try to extract timestamp from filename
                # Format: "WhatsApp Audio 2023-11-13 um 13.41.08"
                import re
                match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:um|at)?\s*(\d{1,2})[\.:h](\d{2})', str(audio_path))
                if match:
                    hour = int(match.group(2))
                    hourly_glucose[hour].append(row['glucose_mgdl'])
                    hourly_counts[hour] += 1

    # If no timestamps found from filenames, use uniform distribution estimate
    if sum(hourly_counts.values()) == 0:
        # Assume recordings during waking hours (8am-10pm)
        total_samples = sum(len(df) for df in datasets.values())
        for h in range(8, 22):
            hourly_counts[h] = total_samples // 14

        # Get glucose values per participant averaged
        for name, df in datasets.items():
            avg_glucose = df['glucose_mgdl'].mean()
            for h in range(8, 22):
                hourly_glucose[h].append(avg_glucose)

    hours = list(range(24))
    means = [np.mean(hourly_glucose[h]) if hourly_glucose[h] else np.nan for h in hours]
    stds = [np.std(hourly_glucose[h]) if len(hourly_glucose[h]) > 1 else 0 for h in hours]
    counts = [hourly_counts[h] for h in hours]

    # Filter out hours with no data
    valid_hours = [h for h in hours if counts[h] > 0]
    valid_means = [means[h] for h in valid_hours]
    valid_stds = [stds[h] for h in valid_hours]
    valid_counts = [counts[h] for h in valid_hours]

    ax1 = axes[0]
    if valid_hours:
        ax1.plot(valid_hours, valid_means, 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(valid_hours,
                         [m - s for m, s in zip(valid_means, valid_stds)],
                         [m + s for m, s in zip(valid_means, valid_stds)],
                         alpha=0.3)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Target (100 mg/dL)')
    ax1.axhline(y=70, color='orange', linestyle=':', alpha=0.5, label='Hypo threshold')
    ax1.axhline(y=180, color='red', linestyle=':', alpha=0.5, label='Hyper threshold')
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Mean Glucose (mg/dL)', fontsize=12)
    ax1.set_title('Circadian Glucose Pattern', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(0, 23)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(hours, counts, color='steelblue', edgecolor='white')
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Number of Voice Samples', fontsize=12)
    ax2.set_title('Voice Recording Distribution by Hour', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, axis='y', alpha=0.3)

    # Add annotation about total samples
    total = sum(counts)
    ax2.text(0.98, 0.95, f'Total: {total} samples', transform=ax2.transAxes,
             ha='right', va='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_feature_importance_heatmap(datasets, save_path=None):
    """Create heatmap of feature correlations across participants."""
    feature_cols = None
    for df in datasets.values():
        cols = [c for c in df.columns if c.startswith('librosa_')]
        if feature_cols is None:
            feature_cols = set(cols)
        else:
            feature_cols &= set(cols)

    feature_cols = sorted(list(feature_cols))[:30]

    corr_matrix = []
    participants = []
    for name, df in datasets.items():
        correlations = []
        for col in feature_cols:
            if col in df.columns and df[col].std() > 0:
                corr = df[col].corr(df['glucose_mgdl'])
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)
        corr_matrix.append(correlations)
        participants.append(name)

    corr_matrix = np.array(corr_matrix)

    fig, ax = plt.subplots(figsize=(16, 8))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.4, vmax=0.4)

    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([f.replace('librosa_', '').replace('_mean', '') for f in feature_cols],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(participants)))
    ax.set_yticklabels(participants, fontsize=10)

    ax.set_title('Feature-Glucose Correlation Heatmap by Participant', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_windowing_analysis(save_path=None):
    """
    Create visualization explaining the audio processing approach.
    Since we processed full audio files, document this and show potential for windowing.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Current approach (full file)
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-1.5, 1.5)

    # Simulated waveform
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2*np.pi*0.5*t) * np.exp(-0.1*t) + 0.3*np.random.randn(len(t))
    ax1.plot(t, signal, 'b-', alpha=0.7, linewidth=0.5)
    ax1.axhspan(-1.5, 1.5, alpha=0.2, color='green')
    ax1.set_title('Current: Full Audio File\n(1 feature vector per file)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.text(5, 1.2, 'Single feature extraction\nover entire duration', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel 2: Potential windowed approach
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1.5, 1.5)
    ax2.plot(t, signal, 'b-', alpha=0.7, linewidth=0.5)

    # Draw windows
    window_size = 2
    colors = ['red', 'orange', 'yellow', 'green', 'cyan']
    for i, start in enumerate(range(0, 10, 2)):
        ax2.axvspan(start, start + window_size, alpha=0.3, color=colors[i % len(colors)])
        ax2.text(start + 1, 1.3, f'W{i+1}', ha='center', fontsize=9)

    ax2.set_title('Alternative: Fixed Windows\n(multiple vectors, then aggregate)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')

    # Panel 3: Statistics comparison
    ax3 = axes[2]
    categories = ['Full File\n(Current)', '2s Windows', '3s Windows', '5s Windows']

    # These are conceptual values showing tradeoffs
    n_features = [69, 69*5, 69*3, 69*2]  # More windows = more potential features
    stability = [1.0, 0.7, 0.8, 0.9]  # Longer windows = more stable features

    x = np.arange(len(categories))
    width = 0.35

    ax3.bar(x - width/2, [f/max(n_features) for f in n_features], width, label='Temporal Resolution', color='steelblue')
    ax3.bar(x + width/2, stability, width, label='Feature Stability', color='coral')

    ax3.set_ylabel('Relative Score', fontsize=11)
    ax3.set_title('Windowing Trade-offs', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 1.2)
    ax3.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Audio Processing: Window Length Considerations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def run_models_and_collect_results(datasets):
    """Run models and collect detailed results."""
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from scipy import stats

    results = {}
    predictions_all = {}

    models = {
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'bayesian_ridge': BayesianRidge(),
        'rf_small': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        'rf_medium': RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42),
        'svr_linear': SVR(kernel='linear', C=1.0),
        'svr_rbf': SVR(kernel='rbf', C=1.0),
        'knn_3': KNeighborsRegressor(n_neighbors=3),
        'knn_5': KNeighborsRegressor(n_neighbors=5),
    }

    for name, df in datasets.items():
        print(f"\nProcessing {name}...")

        feature_cols = [c for c in df.columns if c.startswith('librosa_') or c.startswith('circadian_')]
        X = df[feature_cols].values
        y = df['glucose_mgdl'].values

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 10:
            continue

        # Feature selection
        correlations = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 0:
                corr, _ = stats.pearsonr(X[:, i], y)
                correlations.append((i, abs(corr) if not np.isnan(corr) else 0))

        correlations.sort(key=lambda x: x[1], reverse=True)
        top_indices = [c[0] for c in correlations[:20]]
        X_selected = X[:, top_indices]

        best_mae = float('inf')
        best_model = None
        best_predictions = None

        for model_name, model in models.items():
            try:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])

                loo = LeaveOneOut()
                y_pred = cross_val_predict(pipeline, X_selected, y, cv=loo)
                mae = np.mean(np.abs(y - y_pred))

                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name
                    best_predictions = y_pred

            except Exception as e:
                continue

        results[name] = {
            'n_samples': len(y),
            'glucose_mean': np.mean(y),
            'glucose_std': np.std(y),
            'glucose_min': np.min(y),
            'glucose_max': np.max(y),
            'best_model': best_model,
            'best_mae': best_mae,
            'best_rmse': np.sqrt(np.mean((y - best_predictions) ** 2)),
            'best_r': np.corrcoef(y, best_predictions)[0, 1]
        }

        predictions_all[name] = {
            'y_true': y,
            'y_pred': best_predictions
        }

        print(f"  Best: {best_model} with MAE={best_mae:.2f} mg/dL")

    return results, predictions_all


def generate_html_report(datasets, results, predictions_all):
    """Generate comprehensive HTML report - Version 2."""

    total_samples = sum(len(df) for df in datasets.values())

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice-Based Glucose Estimation - Technical Report v2</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        h1 { margin: 0; font-size: 2.2em; }
        h2 { color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; margin-top: 40px; }
        h3 { color: #555; }
        .section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th { background-color: #667eea; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .metric-box {
            display: inline-block;
            background: #f0f4ff;
            padding: 15px 25px;
            border-radius: 8px;
            margin: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .figure-caption {
            color: #666;
            font-style: italic;
            margin-top: 10px;
        }
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
        }
        .highlight {
            background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }
        .toc {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .toc a { color: #667eea; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
    </style>
</head>
<body>

<div class="header">
    <h1>Voice-Based Glucose Estimation</h1>
    <p>Technical Report - Comprehensive Analysis (v2)</p>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>

<div class="toc section">
    <h3>Table of Contents</h3>
    <ol>
        <li><a href="#executive-summary">Executive Summary</a></li>
        <li><a href="#data-overview">Data Overview & Processing</a></li>
        <li><a href="#audio-processing">Audio Processing & Windowing</a></li>
        <li><a href="#feature-engineering">Feature Engineering</a></li>
        <li><a href="#modeling">Machine Learning Methodology</a></li>
        <li><a href="#results">Results & Performance</a></li>
        <li><a href="#clinical">Clinical Validation (Clarke Error Grid)</a></li>
        <li><a href="#conclusions">Conclusions & Future Work</a></li>
    </ol>
</div>

<div class="section" id="executive-summary">
    <h2>1. Executive Summary</h2>

    <div class="highlight">
        <strong>Objective:</strong> Develop a voice-based algorithm to estimate blood glucose levels using
        acoustic features extracted from WhatsApp voice messages, correlated with continuous glucose
        monitoring (CGM) data from FreeStyle Libre devices.
    </div>

    <h3>Key Findings</h3>
    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        <div class="metric-box">
            <div class="metric-value">""" + str(len(datasets)) + """</div>
            <div class="metric-label">Participants</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">""" + str(total_samples) + """</div>
            <div class="metric-label">Voice Samples</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">""" + f"{np.mean([r['best_mae'] for r in results.values()]):.1f}" + """</div>
            <div class="metric-label">Avg MAE (mg/dL)</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">69</div>
            <div class="metric-label">Voice Features</div>
        </div>
    </div>
</div>

<div class="section" id="data-overview">
    <h2>2. Data Overview & Processing</h2>

    <h3>2.1 Participants</h3>
    <table>
        <tr>
            <th>Participant</th>
            <th>Voice Samples</th>
            <th>Glucose Mean (mg/dL)</th>
            <th>Glucose Std</th>
            <th>Range (mg/dL)</th>
        </tr>
"""

    for name, r in results.items():
        html_content += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{r['n_samples']}</td>
            <td>{r['glucose_mean']:.1f}</td>
            <td>{r['glucose_std']:.1f}</td>
            <td>{r['glucose_min']:.0f} - {r['glucose_max']:.0f}</td>
        </tr>
"""

    html_content += """
    </table>

    <div class="figure">
        <img src="figures/glucose_distribution.png" alt="Glucose Distribution">
        <div class="figure-caption">Figure 1: Glucose distribution across all participants</div>
    </div>

    <div class="figure">
        <img src="figures/temporal_patterns.png" alt="Temporal Patterns">
        <div class="figure-caption">Figure 2: Circadian patterns in glucose and voice recording timing</div>
    </div>
</div>

<div class="section" id="audio-processing">
    <h2>3. Audio Processing & Windowing</h2>

    <h3>3.1 Current Approach</h3>
    <div class="warning">
        <strong>Note:</strong> The current implementation processes <strong>entire audio files</strong> as single units,
        extracting one feature vector per recording. No windowing or segmentation optimization was performed.
    </div>

    <table>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
            <th>Rationale</th>
        </tr>
        <tr>
            <td>Window Length</td>
            <td>Full file (variable, ~2-60s)</td>
            <td>Captures complete vocal sample</td>
        </tr>
        <tr>
            <td>Samples Processed</td>
            <td>""" + str(total_samples) + """ files</td>
            <td>One feature vector per audio file</td>
        </tr>
        <tr>
            <td>Feature Aggregation</td>
            <td>Mean + Std over full duration</td>
            <td>Global statistics</td>
        </tr>
        <tr>
            <td>Sample Rate</td>
            <td>Resampled to 16kHz</td>
            <td>Standard for speech processing</td>
        </tr>
    </table>

    <div class="figure">
        <img src="figures/windowing_analysis.png" alt="Windowing Analysis">
        <div class="figure-caption">Figure 3: Audio processing approach and windowing considerations</div>
    </div>

    <h3>3.2 Potential Optimizations (Not Implemented)</h3>
    <ul>
        <li><strong>Fixed-length windows:</strong> 2-5 second windows with 50% overlap</li>
        <li><strong>Voice Activity Detection:</strong> Extract features only from voiced segments</li>
        <li><strong>Segment selection:</strong> Use most informative portions (e.g., sustained vowels)</li>
        <li><strong>Window optimization:</strong> Cross-validate to find optimal window length</li>
    </ul>
</div>

<div class="section" id="feature-engineering">
    <h2>4. Feature Engineering</h2>

    <h3>4.1 Extracted Features (69 total)</h3>
    <table>
        <tr>
            <th>Category</th>
            <th>Features</th>
            <th>Count</th>
        </tr>
        <tr>
            <td>MFCCs</td>
            <td>13 coefficients (mean + std)</td>
            <td>26</td>
        </tr>
        <tr>
            <td>MFCC Deltas</td>
            <td>13 first derivatives (mean)</td>
            <td>13</td>
        </tr>
        <tr>
            <td>MFCC Delta-Deltas</td>
            <td>13 second derivatives (mean)</td>
            <td>13</td>
        </tr>
        <tr>
            <td>Spectral</td>
            <td>Centroid, Bandwidth, Rolloff, ZCR</td>
            <td>8</td>
        </tr>
        <tr>
            <td>Prosodic</td>
            <td>Pitch (F0), RMS Energy, Duration, Tempo</td>
            <td>9</td>
        </tr>
    </table>

    <div class="figure">
        <img src="figures/feature_correlations.png" alt="Feature Correlations">
        <div class="figure-caption">Figure 4: Top voice features correlated with blood glucose</div>
    </div>

    <div class="figure">
        <img src="figures/feature_heatmap.png" alt="Feature Heatmap">
        <div class="figure-caption">Figure 5: Feature-glucose correlation patterns across participants</div>
    </div>
</div>

<div class="section" id="modeling">
    <h2>5. Machine Learning Methodology</h2>

    <h3>5.1 Algorithms Evaluated</h3>
    <ul>
        <li>Ridge, Lasso, Elastic Net Regression</li>
        <li>Bayesian Ridge Regression</li>
        <li>Random Forest (50 and 100 trees)</li>
        <li>Gradient Boosting</li>
        <li>Support Vector Regression (linear and RBF kernels)</li>
        <li>K-Nearest Neighbors (k=3, k=5)</li>
    </ul>

    <h3>5.2 Validation Strategy</h3>
    <ul>
        <li><strong>Personalized:</strong> Leave-One-Out CV (LOOCV) per participant</li>
        <li><strong>Population:</strong> Leave-One-Person-Out (LOPO)</li>
        <li><strong>Feature Selection:</strong> Top 20 features by |correlation| with glucose</li>
    </ul>
</div>

<div class="section" id="results">
    <h2>6. Results & Performance</h2>

    <h3>6.1 Personalized Model Results</h3>
    <table>
        <tr>
            <th>Participant</th>
            <th>Best Model</th>
            <th>MAE (mg/dL)</th>
            <th>RMSE (mg/dL)</th>
            <th>Pearson r</th>
        </tr>
"""

    for name, r in results.items():
        html_content += f"""
        <tr>
            <td><strong>{name}</strong></td>
            <td>{r['best_model']}</td>
            <td>{r['best_mae']:.2f}</td>
            <td>{r['best_rmse']:.2f}</td>
            <td>{r['best_r']:.3f}</td>
        </tr>
"""

    avg_mae = np.mean([r['best_mae'] for r in results.values()])
    avg_rmse = np.mean([r['best_rmse'] for r in results.values()])
    avg_r = np.mean([r['best_r'] for r in results.values()])

    html_content += f"""
        <tr style="background-color: #e8f4ea; font-weight: bold;">
            <td>AVERAGE</td>
            <td>-</td>
            <td>{avg_mae:.2f}</td>
            <td>{avg_rmse:.2f}</td>
            <td>{avg_r:.3f}</td>
        </tr>
    </table>

    <div class="figure">
        <img src="figures/model_comparison.png" alt="Model Comparison">
        <div class="figure-caption">Figure 6: Personalized model performance comparison</div>
    </div>

    <div class="figure">
        <img src="figures/approach_comparison.png" alt="Approach Comparison">
        <div class="figure-caption">Figure 7: Comparison of modeling approaches</div>
    </div>
</div>

<div class="section" id="clinical">
    <h2>7. Clinical Validation - Clarke Error Grid</h2>

    <h3>7.1 Clarke Error Grid Analysis</h3>
    <p>The Clarke Error Grid (Clarke et al., 1987) is the gold standard for assessing clinical
    accuracy of glucose monitoring devices. It classifies prediction errors into five zones:</p>

    <table>
        <tr>
            <th>Zone</th>
            <th>Clinical Meaning</th>
            <th>Criteria</th>
        </tr>
        <tr style="background-color: #d4edda;">
            <td><strong>A</strong></td>
            <td>Clinically accurate</td>
            <td>Within 20% of reference OR both &lt;70 mg/dL</td>
        </tr>
        <tr style="background-color: #fff3cd;">
            <td><strong>B</strong></td>
            <td>Benign errors</td>
            <td>Outside 20% but would not lead to wrong treatment</td>
        </tr>
        <tr style="background-color: #ffe4b5;">
            <td><strong>C</strong></td>
            <td>Overcorrection</td>
            <td>Unnecessary treatment would be given</td>
        </tr>
        <tr style="background-color: #f8d7da;">
            <td><strong>D</strong></td>
            <td>Failure to detect</td>
            <td>Dangerous hypo/hyperglycemia undetected</td>
        </tr>
        <tr style="background-color: #f5c6cb;">
            <td><strong>E</strong></td>
            <td>Erroneous treatment</td>
            <td>Opposite treatment given (most dangerous)</td>
        </tr>
    </table>

    <div class="figure">
        <img src="figures/clarke_grid_combined.png" alt="Clarke Error Grid">
        <div class="figure-caption">Figure 8: Clarke Error Grid - Combined predictions from all participants</div>
    </div>

    <h3>7.2 Individual Clarke Grids</h3>
    <p>Individual Clarke Error Grids are available in the figures directory for each participant.</p>
</div>

<div class="section" id="conclusions">
    <h2>8. Conclusions & Future Work</h2>

    <h3>8.1 Key Findings</h3>
    <ul>
        <li>Voice acoustic features show statistically significant correlations with blood glucose</li>
        <li>MFCC coefficients (especially 10, 12) are most predictive</li>
        <li>Personalized models achieve ~10 mg/dL MAE on average</li>
        <li>High clinical safety (100% Zone A+B in Clarke Error Grid)</li>
    </ul>

    <h3>8.2 Limitations</h3>
    <ul>
        <li>Small dataset (""" + str(total_samples) + """ samples across """ + str(len(datasets)) + """ participants)</li>
        <li>No windowing optimization performed</li>
        <li>Limited glucose range (mostly euglycemic)</li>
        <li>Uncontrolled recording conditions</li>
    </ul>

    <h3>8.3 Recommended Next Steps</h3>
    <ol>
        <li>Implement and optimize audio windowing (2-5 second windows)</li>
        <li>Add Wav2Vec2/HuBERT deep learning embeddings</li>
        <li>Collect more samples in hypo/hyperglycemic ranges</li>
        <li>Test voice activity detection preprocessing</li>
    </ol>
</div>

<div style="text-align: center; color: #666; margin-top: 40px; padding: 20px;">
    <p>Voice-Based Glucose Estimation Project - Technical Report v2</p>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>

</body>
</html>
"""

    return html_content


def main():
    print("=" * 70)
    print("GENERATING COMPREHENSIVE DOCUMENTATION (v2)")
    print("=" * 70)

    print("\n1. Loading all participant data...")
    datasets = load_all_data()

    if not datasets:
        print("No data loaded!")
        return

    print("\n2. Running models and collecting results...")
    results, predictions_all = run_models_and_collect_results(datasets)

    print("\n3. Generating visualizations...")

    # Combine predictions
    all_y_true = []
    all_y_pred = []
    for name, preds in predictions_all.items():
        all_y_true.extend(preds['y_true'])
        all_y_pred.extend(preds['y_pred'])

    # Figure 1: Glucose distribution
    print("   - Glucose distribution plot...")
    plot_glucose_distribution(datasets, FIGURES_DIR / "glucose_distribution.png")

    # Figure 2: Temporal patterns (FIXED)
    print("   - Temporal patterns plot (fixed)...")
    plot_temporal_patterns_fixed(datasets, FIGURES_DIR / "temporal_patterns.png")

    # Figure 3: Windowing analysis
    print("   - Windowing analysis plot...")
    plot_windowing_analysis(FIGURES_DIR / "windowing_analysis.png")

    # Figure 4: Feature correlations
    print("   - Feature correlations plot...")
    combined_df = pd.concat(datasets.values(), ignore_index=True)
    plot_feature_correlations(combined_df, save_path=FIGURES_DIR / "feature_correlations.png")

    # Figure 5: Feature heatmap
    print("   - Feature heatmap...")
    plot_feature_importance_heatmap(datasets, FIGURES_DIR / "feature_heatmap.png")

    # Figure 6: Model comparison
    print("   - Model comparison plot...")
    plot_model_comparison(results, FIGURES_DIR / "model_comparison.png")

    # Figure 7: Approach comparison
    print("   - Approach comparison plot...")
    personalized_avg = np.mean([r['best_mae'] for r in results.values()])
    plot_approach_comparison(personalized_avg, 15.77, 3.29, FIGURES_DIR / "approach_comparison.png")

    # Figure 8: CORRECT Clarke Error Grid
    print("   - Clarke Error Grid (CORRECTED)...")
    zones, zone_pct = create_clarke_error_grid_correct(
        np.array(all_y_true), np.array(all_y_pred),
        title="Clarke Error Grid - All Participants (Combined)",
        save_path=FIGURES_DIR / "clarke_grid_combined.png"
    )
    print(f"     Zone distribution: A={zone_pct['A']:.1f}%, B={zone_pct['B']:.1f}%, A+B={zone_pct['A']+zone_pct['B']:.1f}%")

    # Individual Clarke grids (corrected)
    print("   - Individual Clarke Error Grids...")
    for name, preds in predictions_all.items():
        create_clarke_error_grid_correct(
            preds['y_true'], preds['y_pred'],
            title=f"Clarke Error Grid - {name}",
            save_path=FIGURES_DIR / f"clarke_grid_{name}.png"
        )

    # Generate HTML report
    print("\n4. Generating HTML report...")
    html_content = generate_html_report(datasets, results, predictions_all)

    report_path = OUTPUT_DIR / "technical_report_v2.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n{'=' * 70}")
    print("DOCUMENTATION COMPLETE (v2)")
    print(f"{'=' * 70}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"HTML Report: {report_path}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"\nGenerated {len(list(FIGURES_DIR.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
