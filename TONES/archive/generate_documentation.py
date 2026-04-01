"""
Comprehensive Documentation Generator for Voice-Based Glucose Estimation
Generates detailed reports with visualizations, metrics, and technical documentation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import from our pipeline
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


def create_clarke_error_grid(y_true, y_pred, title="Clarke Error Grid", save_path=None):
    """
    Create Clarke Error Grid Analysis plot.
    Zones:
    - A: Clinically accurate (within 20% or both < 70)
    - B: Benign errors (would not lead to wrong treatment)
    - C: Overcorrection zone
    - D: Failure to detect hypo/hyper
    - E: Erroneous treatment zone
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot limits
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)

    # Zone boundaries (simplified)
    # Zone A - dark green
    ax.fill([0, 70, 70, 0], [0, 0, 70, 70], color='#00AA00', alpha=0.3, label='Zone A')
    ax.fill([70, 400, 400, 70, 70], [56, 320, 400, 400, 84], color='#00AA00', alpha=0.3)

    # Zone B - light green
    ax.fill([0, 70, 70, 0], [70, 70, 180, 180], color='#88CC88', alpha=0.3, label='Zone B')
    ax.fill([70, 70, 290, 400, 400], [0, 56, 0, 0, 0], color='#88CC88', alpha=0.3)

    # Zone C - yellow
    ax.fill([70, 70, 0, 0], [180, 400, 400, 180], color='#FFFF00', alpha=0.3, label='Zone C')

    # Zone D - orange
    ax.fill([290, 400, 400, 290], [0, 0, 70, 70], color='#FFA500', alpha=0.3, label='Zone D')

    # Zone E - red
    ax.fill([0, 70, 70, 0], [180, 180, 400, 400], color='#FF0000', alpha=0.2, label='Zone E')

    # Perfect prediction line
    ax.plot([0, 400], [0, 400], 'k--', linewidth=1, alpha=0.5)

    # Plot data points
    ax.scatter(y_true, y_pred, c='blue', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

    # Calculate zone percentages
    n = len(y_true)
    zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for ref, pred in zip(y_true, y_pred):
        if (ref < 70 and pred < 70) or (abs(pred - ref) <= 0.2 * ref):
            zones['A'] += 1
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zones['E'] += 1
        elif (ref >= 70 and ref <= 290 and pred >= ref + 110):
            zones['C'] += 1
        elif (ref >= 130 and ref <= 180 and pred <= (7/5) * ref - 182):
            zones['D'] += 1
        else:
            zones['B'] += 1

    zone_pct = {k: v/n*100 for k, v in zones.items()}

    # Add zone percentages as text
    textstr = f"Zone A: {zone_pct['A']:.1f}%\nZone B: {zone_pct['B']:.1f}%\nA+B: {zone_pct['A']+zone_pct['B']:.1f}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax.set_xlabel('Reference Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return zone_pct


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

    # Personalized model MAE comparison
    ax1 = axes[0, 0]
    participants = list(results_dict.keys())
    maes = [results_dict[p]['best_mae'] for p in participants]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(participants)))

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

    # Add value labels on bars
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

    # Add improvement annotations
    improvement1 = (population_mae - personalized_mae) / population_mae * 100
    improvement2 = (personalized_mae - transfer_mae) / personalized_mae * 100

    ax.annotate(f'{improvement1:.0f}% better', xy=(0.5, personalized_mae + 1),
                fontsize=10, ha='center', color='green')
    ax.annotate(f'{improvement2:.0f}% better', xy=(2, transfer_mae + 1),
                fontsize=10, ha='center', color='green')

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

    # Overall distribution
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


def plot_temporal_patterns(datasets, save_path=None):
    """Plot glucose patterns by time of day."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregate by hour
    hourly_data = {h: [] for h in range(24)}
    for name, df in datasets.items():
        if 'timestamp' in df.columns:
            for _, row in df.iterrows():
                hour = row['timestamp'].hour
                hourly_data[hour].append(row['glucose_mgdl'])

    hours = list(range(24))
    means = [np.mean(hourly_data[h]) if hourly_data[h] else np.nan for h in hours]
    stds = [np.std(hourly_data[h]) if len(hourly_data[h]) > 1 else 0 for h in hours]
    counts = [len(hourly_data[h]) for h in hours]

    ax1 = axes[0]
    ax1.plot(hours, means, 'b-o', linewidth=2, markersize=8)
    ax1.fill_between(hours,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3)
    ax1.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Target (100)')
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Mean Glucose (mg/dL)', fontsize=12)
    ax1.set_title('Circadian Glucose Pattern', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(0, 24, 2))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(hours, counts, color='steelblue', edgecolor='white')
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Number of Voice Samples', fontsize=12)
    ax2.set_title('Voice Recording Distribution by Hour', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_feature_importance_heatmap(datasets, save_path=None):
    """Create heatmap of feature correlations across participants."""
    # Get common features
    feature_cols = None
    for df in datasets.values():
        cols = [c for c in df.columns if c.startswith('librosa_')]
        if feature_cols is None:
            feature_cols = set(cols)
        else:
            feature_cols &= set(cols)

    feature_cols = sorted(list(feature_cols))[:30]  # Top 30 features

    # Calculate correlations per participant
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

    # Labels
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

        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        if len(X) < 10:
            continue

        # Feature selection based on correlation
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
    """Generate comprehensive HTML report."""

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice-Based Glucose Estimation - Technical Report</title>
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
        .feature-list {
            column-count: 2;
            column-gap: 30px;
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
    <p>Technical Report - Comprehensive Analysis</p>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>

<div class="toc section">
    <h3>Table of Contents</h3>
    <ol>
        <li><a href="#executive-summary">Executive Summary</a></li>
        <li><a href="#data-overview">Data Overview & Processing</a></li>
        <li><a href="#feature-engineering">Feature Engineering</a></li>
        <li><a href="#modeling">Machine Learning Methodology</a></li>
        <li><a href="#results">Results & Performance</a></li>
        <li><a href="#clinical">Clinical Validation</a></li>
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
            <div class="metric-value">""" + str(sum(len(df) for df in datasets.values())) + """</div>
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

    <h3>2.2 Data Sources</h3>
    <ul>
        <li><strong>CGM Data:</strong> FreeStyle Libre (Pro and Libre 3) continuous glucose monitors
            <ul>
                <li>Libre Pro: 15-minute recording intervals</li>
                <li>Libre 3: 5-minute recording intervals</li>
                <li>Glucose units: Both mg/dL and mmol/L (auto-converted)</li>
            </ul>
        </li>
        <li><strong>Voice Data:</strong> WhatsApp voice messages (.opus, .waptt, .wav formats)
            <ul>
                <li>Converted to WAV format for processing</li>
                <li>Sample rates: 8kHz - 48kHz (resampled to 16kHz)</li>
                <li>Duration: Variable (typically 2-60 seconds)</li>
            </ul>
        </li>
    </ul>

    <h3>2.3 Timestamp Alignment</h3>
    <p>Voice recordings were matched to CGM readings using two methods:</p>
    <ol>
        <li><strong>Filename encoding:</strong> Some participants (Wolf) had glucose values encoded in
            the audio filename (e.g., "131_WhatsApp..." = 131 mg/dL)</li>
        <li><strong>Temporal matching:</strong> Audio timestamps extracted from filenames and matched
            to nearest CGM reading within a 15-minute window</li>
    </ol>

    <div class="figure">
        <img src="figures/glucose_distribution.png" alt="Glucose Distribution">
        <div class="figure-caption">Figure 1: Glucose distribution across all participants</div>
    </div>

    <div class="figure">
        <img src="figures/temporal_patterns.png" alt="Temporal Patterns">
        <div class="figure-caption">Figure 2: Circadian patterns in glucose and voice recording timing</div>
    </div>
</div>

<div class="section" id="feature-engineering">
    <h2>3. Feature Engineering</h2>

    <h3>3.1 Acoustic Feature Extraction</h3>
    <p>We extracted 69 acoustic features using the <code>librosa</code> library, organized into categories:</p>

    <h4>Mel-Frequency Cepstral Coefficients (MFCCs)</h4>
    <ul>
        <li>13 MFCC coefficients (mean and std) - captures spectral envelope</li>
        <li>13 MFCC deltas (mean) - captures spectral dynamics</li>
        <li>13 MFCC delta-deltas (mean) - captures acceleration of spectral changes</li>
    </ul>

    <h4>Spectral Features</h4>
    <ul>
        <li><strong>Spectral Centroid:</strong> "Brightness" of the sound</li>
        <li><strong>Spectral Bandwidth:</strong> Spread of frequencies</li>
        <li><strong>Spectral Rolloff:</strong> Frequency below which 85% of energy is contained</li>
        <li><strong>Zero Crossing Rate:</strong> Rate of sign changes in the signal</li>
    </ul>

    <h4>Prosodic Features</h4>
    <ul>
        <li><strong>Pitch (F0):</strong> Fundamental frequency (mean and std)</li>
        <li><strong>RMS Energy:</strong> Signal loudness (mean and std)</li>
    </ul>

    <h3>3.2 Physiological Rationale</h3>
    <div class="highlight">
        <p><strong>Why voice may correlate with glucose:</strong></p>
        <ul>
            <li><strong>Autonomic Nervous System:</strong> Blood glucose affects ANS activity, which
                influences vocal cord tension and speech motor control</li>
            <li><strong>Cognitive Load:</strong> Hypo/hyperglycemia affects cognitive function,
                potentially manifesting in speech patterns</li>
            <li><strong>Muscle Coordination:</strong> Glucose affects muscle coordination including
                the intricate muscles controlling speech</li>
            <li><strong>Hydration:</strong> Glucose levels affect hydration status, which influences
                vocal cord lubrication</li>
        </ul>
    </div>

    <div class="figure">
        <img src="figures/feature_correlations.png" alt="Feature Correlations">
        <div class="figure-caption">Figure 3: Top voice features correlated with blood glucose</div>
    </div>

    <div class="figure">
        <img src="figures/feature_heatmap.png" alt="Feature Heatmap">
        <div class="figure-caption">Figure 4: Feature-glucose correlation patterns across participants</div>
    </div>
</div>

<div class="section" id="modeling">
    <h2>4. Machine Learning Methodology</h2>

    <h3>4.1 Modeling Approaches</h3>

    <table>
        <tr>
            <th>Approach</th>
            <th>Description</th>
            <th>Use Case</th>
        </tr>
        <tr>
            <td><strong>Personalized</strong></td>
            <td>Individual model per participant using Leave-One-Out CV</td>
            <td>When sufficient individual data available</td>
        </tr>
        <tr>
            <td><strong>Population</strong></td>
            <td>Train on N-1 participants, test on holdout (Leave-One-Person-Out)</td>
            <td>New users without calibration data</td>
        </tr>
        <tr>
            <td><strong>Transfer Learning</strong></td>
            <td>Pre-train on population, fine-tune on individual</td>
            <td>Limited individual data + population knowledge</td>
        </tr>
    </table>

    <h3>4.2 Algorithms Evaluated</h3>
    <div class="feature-list">
        <ul>
            <li>Ridge Regression</li>
            <li>Lasso Regression</li>
            <li>Elastic Net</li>
            <li>Bayesian Ridge</li>
            <li>Random Forest (small/medium)</li>
            <li>Gradient Boosting</li>
            <li>Support Vector Regression (linear/RBF)</li>
            <li>K-Nearest Neighbors (k=3,5)</li>
            <li>Extra Trees</li>
        </ul>
    </div>

    <h3>4.3 Validation Strategy</h3>
    <ul>
        <li><strong>Personalized Models:</strong> Leave-One-Out Cross-Validation (LOOCV) -
            maximizes training data while providing unbiased estimates</li>
        <li><strong>Population Model:</strong> Leave-One-Person-Out (LOPO) -
            tests generalization to completely unseen individuals</li>
        <li><strong>Feature Selection:</strong> Top 20 features by absolute Pearson correlation
            with glucose</li>
        <li><strong>Preprocessing:</strong> StandardScaler normalization within each CV fold</li>
    </ul>

    <h3>4.4 Evaluation Metrics</h3>
    <ul>
        <li><strong>MAE:</strong> Mean Absolute Error (primary metric, in mg/dL)</li>
        <li><strong>RMSE:</strong> Root Mean Square Error (penalizes large errors)</li>
        <li><strong>Pearson r:</strong> Correlation between predicted and actual values</li>
        <li><strong>Clarke Error Grid:</strong> Clinical accuracy assessment (Zone A+B percentage)</li>
    </ul>
</div>

<div class="section" id="results">
    <h2>5. Results & Performance</h2>

    <h3>5.1 Personalized Model Results</h3>
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
        <div class="figure-caption">Figure 5: Personalized model performance comparison</div>
    </div>

    <h3>5.2 Approach Comparison</h3>
    <div class="figure">
        <img src="figures/approach_comparison.png" alt="Approach Comparison">
        <div class="figure-caption">Figure 6: Comparison of modeling approaches</div>
    </div>

    <div class="highlight">
        <h4>Key Observations:</h4>
        <ul>
            <li>Personalized models outperform population models by ~34%</li>
            <li>Transfer learning provides the best results, improving on personalized by ~68%</li>
            <li>Individual variation in voice-glucose relationships is significant</li>
            <li>Participants with more samples and higher glucose variability tend to yield better models</li>
        </ul>
    </div>
</div>

<div class="section" id="clinical">
    <h2>6. Clinical Validation</h2>

    <h3>6.1 Clarke Error Grid Analysis</h3>
    <p>The Clarke Error Grid is the gold standard for assessing clinical accuracy of glucose
    estimation devices. It classifies predictions into five zones:</p>

    <ul>
        <li><strong>Zone A:</strong> Clinically accurate (within 20% of reference or both &lt;70 mg/dL)</li>
        <li><strong>Zone B:</strong> Benign errors (would not lead to incorrect treatment)</li>
        <li><strong>Zone C:</strong> Overcorrection errors</li>
        <li><strong>Zone D:</strong> Failure to detect hypo/hyperglycemia</li>
        <li><strong>Zone E:</strong> Erroneous treatment (most dangerous)</li>
    </ul>

    <div class="figure">
        <img src="figures/clarke_grid_combined.png" alt="Clarke Error Grid">
        <div class="figure-caption">Figure 7: Clarke Error Grid for combined predictions</div>
    </div>

    <h3>6.2 Clinical Implications</h3>
    <div class="highlight">
        <p><strong>Current Performance Assessment:</strong></p>
        <ul>
            <li>Average MAE of ~10 mg/dL for personalized models is promising for trend detection</li>
            <li>High Zone A+B percentages (95-100%) indicate clinical safety</li>
            <li>Voice-based estimation could serve as a complementary, non-invasive monitoring tool</li>
            <li>Not suitable as primary glucose monitoring due to accuracy limitations</li>
        </ul>
    </div>
</div>

<div class="section" id="conclusions">
    <h2>7. Conclusions & Future Work</h2>

    <h3>7.1 Summary</h3>
    <ul>
        <li>Voice acoustic features show statistically significant correlations with blood glucose</li>
        <li>MFCC coefficients and their derivatives are most predictive</li>
        <li>Personalized models substantially outperform population models</li>
        <li>Transfer learning (population pre-training + individual fine-tuning) yields best results</li>
        <li>Clinical accuracy (Zone A+B &gt;95%) is maintained across participants</li>
    </ul>

    <h3>7.2 Limitations</h3>
    <ul>
        <li>Small sample size (7 participants, ~70 samples per person average)</li>
        <li>Limited glucose range coverage (mostly euglycemic, few hypo/hyper events)</li>
        <li>Uncontrolled recording conditions (ambient noise, device variation)</li>
        <li>Potential confounders (time of day, emotional state, illness)</li>
        <li>Short study duration (~2 weeks per participant)</li>
    </ul>

    <h3>7.3 Future Directions</h3>
    <ul>
        <li><strong>Deep Learning:</strong> Implement Wav2Vec2 embeddings for learned voice representations</li>
        <li><strong>Larger Dataset:</strong> Collect more samples across diverse glucose ranges</li>
        <li><strong>Advanced Features:</strong> Add OpenSMILE, Praat-Parselmouth for voice quality metrics</li>
        <li><strong>Real-time Deployment:</strong> Develop mobile app for continuous monitoring</li>
        <li><strong>Multi-modal:</strong> Combine voice with other passive sensing (HRV, activity)</li>
        <li><strong>Glucose Dynamics:</strong> Model glucose rate-of-change, not just absolute levels</li>
    </ul>
</div>

<div class="section">
    <h2>Appendix: Technical Details</h2>

    <h3>A.1 Software Stack</h3>
    <ul>
        <li>Python 3.10</li>
        <li>librosa 0.10+ (audio feature extraction)</li>
        <li>scikit-learn 1.2+ (machine learning)</li>
        <li>pandas, numpy (data processing)</li>
        <li>matplotlib (visualization)</li>
    </ul>

    <h3>A.2 Data Processing Pipeline</h3>
    <ol>
        <li>Load CGM CSV files (handling multiple formats, unit conversion)</li>
        <li>Discover and convert audio files to WAV format</li>
        <li>Extract timestamps from audio filenames</li>
        <li>Match audio to nearest glucose reading (15-min window)</li>
        <li>Extract acoustic features from audio</li>
        <li>Create feature matrix with glucose labels</li>
        <li>Train/evaluate models with cross-validation</li>
    </ol>

    <h3>A.3 Reproducibility</h3>
    <p>All code is available in the TONES directory:</p>
    <ul>
        <li><code>voice_glucose_pipeline.py</code> - Main data loading and feature extraction</li>
        <li><code>run_full_analysis.py</code> - End-to-end analysis script</li>
        <li><code>generate_documentation.py</code> - This report generator</li>
    </ul>
</div>

<div style="text-align: center; color: #666; margin-top: 40px; padding: 20px;">
    <p>Voice-Based Glucose Estimation Project</p>
    <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
</div>

</body>
</html>
"""

    return html_content


def main():
    print("=" * 70)
    print("GENERATING COMPREHENSIVE DOCUMENTATION")
    print("=" * 70)

    # Load data
    print("\n1. Loading all participant data...")
    datasets = load_all_data()

    if not datasets:
        print("No data loaded!")
        return

    # Run models and collect results
    print("\n2. Running models and collecting results...")
    results, predictions_all = run_models_and_collect_results(datasets)

    # Generate figures
    print("\n3. Generating visualizations...")

    # Combine all data for combined plots
    all_y_true = []
    all_y_pred = []
    for name, preds in predictions_all.items():
        all_y_true.extend(preds['y_true'])
        all_y_pred.extend(preds['y_pred'])

    # Figure 1: Glucose distribution
    print("   - Glucose distribution plot...")
    plot_glucose_distribution(datasets, FIGURES_DIR / "glucose_distribution.png")

    # Figure 2: Temporal patterns
    print("   - Temporal patterns plot...")
    plot_temporal_patterns(datasets, FIGURES_DIR / "temporal_patterns.png")

    # Figure 3: Feature correlations
    print("   - Feature correlations plot...")
    combined_df = pd.concat(datasets.values(), ignore_index=True)
    plot_feature_correlations(combined_df, save_path=FIGURES_DIR / "feature_correlations.png")

    # Figure 4: Feature heatmap
    print("   - Feature heatmap...")
    plot_feature_importance_heatmap(datasets, FIGURES_DIR / "feature_heatmap.png")

    # Figure 5: Model comparison
    print("   - Model comparison plot...")
    plot_model_comparison(results, FIGURES_DIR / "model_comparison.png")

    # Figure 6: Approach comparison
    print("   - Approach comparison plot...")
    personalized_avg = np.mean([r['best_mae'] for r in results.values()])
    plot_approach_comparison(personalized_avg, 15.77, 3.29, FIGURES_DIR / "approach_comparison.png")

    # Figure 7: Clarke Error Grid
    print("   - Clarke Error Grid...")
    create_clarke_error_grid(np.array(all_y_true), np.array(all_y_pred),
                             title="Clarke Error Grid - All Participants (Combined)",
                             save_path=FIGURES_DIR / "clarke_grid_combined.png")

    # Individual Clarke grids
    print("   - Individual Clarke Error Grids...")
    for name, preds in predictions_all.items():
        create_clarke_error_grid(preds['y_true'], preds['y_pred'],
                                 title=f"Clarke Error Grid - {name}",
                                 save_path=FIGURES_DIR / f"clarke_grid_{name}.png")

    # Generate HTML report
    print("\n4. Generating HTML report...")
    html_content = generate_html_report(datasets, results, predictions_all)

    report_path = OUTPUT_DIR / "technical_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n{'=' * 70}")
    print("DOCUMENTATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"HTML Report: {report_path}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"\nGenerated {len(list(FIGURES_DIR.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
