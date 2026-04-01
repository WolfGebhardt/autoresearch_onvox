"""
Voice-Based Glucose Estimation - Full Analysis Pipeline
=======================================================
End-to-end script to run the complete analysis:
1. Load data for all participants
2. Extract features (traditional + optional Wav2Vec2)
3. Train personalized models
4. Train population model
5. Compare approaches
6. Generate results report

Usage:
    python run_full_analysis.py                    # Run all participants
    python run_full_analysis.py --participant Wolf # Run single participant
    python run_full_analysis.py --wav2vec          # Include Wav2Vec2 features
    python run_full_analysis.py --quick            # Quick test mode
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from voice_glucose_pipeline import (
    create_dataset_for_participant, PARTICIPANTS, BASE_DIR,
    train_personalized_model, clarke_error_grid_analysis
)
from model_comparison import (
    ModelComparison, PersonalizedModelTrainer, TransferLearningApproach,
    FeatureSelector, run_model_comparison_pipeline
)

# Optional imports
try:
    from wav2vec_features import get_wav2vec_features, TORCH_AVAILABLE
except ImportError:
    TORCH_AVAILABLE = False
    get_wav2vec_features = None

try:
    from offset_optimization import run_offset_optimization
    OFFSET_OPT_AVAILABLE = True
except ImportError:
    OFFSET_OPT_AVAILABLE = False


class FullAnalysisPipeline:
    """
    Complete analysis pipeline for voice-based glucose estimation.
    """

    def __init__(self, use_wav2vec: bool = False, verbose: bool = True):
        """
        Initialize pipeline.

        Args:
            use_wav2vec: Whether to include Wav2Vec2 features
            verbose: Print progress information
        """
        self.use_wav2vec = use_wav2vec and TORCH_AVAILABLE
        self.verbose = verbose
        self.results = {}

        if self.use_wav2vec:
            print("Wav2Vec2 features ENABLED (will be slower)")
        else:
            print("Using traditional acoustic features only")

    def load_all_participants(self, participants: list = None) -> Dict[str, pd.DataFrame]:
        """
        Load datasets for all participants.

        Args:
            participants: List of participant names (None = all)

        Returns:
            Dictionary of DataFrames
        """
        if participants is None:
            participants = list(PARTICIPANTS.keys())

        datasets = {}

        for name in participants:
            if name not in PARTICIPANTS:
                print(f"Unknown participant: {name}")
                continue

            config = PARTICIPANTS[name]

            # Skip if audio format is not WAV
            if config.get('audio_ext', '.wav') != '.wav':
                print(f"\n{name}: Skipping (audio needs conversion)")
                continue

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Loading: {name}")

            df = create_dataset_for_participant(name, config, verbose=self.verbose)

            if df is not None and len(df) >= 5:
                datasets[name] = df
                print(f"  Loaded {len(df)} samples")
            else:
                print(f"  Insufficient data")

        return datasets

    def run_personalized_models(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Train and evaluate personalized models for each participant.
        """
        print("\n" + "#" * 70)
        print("# PERSONALIZED MODELS")
        print("#" * 70)

        results = {}

        for name, df in datasets.items():
            print(f"\n{'='*60}")
            print(f"Participant: {name} ({len(df)} samples)")
            print("=" * 60)

            # Run model comparison
            comparison_results = run_model_comparison_pipeline(df, feature_prefix='librosa_')

            if comparison_results:
                results[name] = {
                    'n_samples': len(df),
                    'best_model': comparison_results['best_model'],
                    'best_mae': comparison_results['best_mae'],
                    'comparison': comparison_results['comparison_results']
                }

                # Also try with opensmile features if available
                opensmile_cols = [c for c in df.columns if c.startswith('opensmile_')]
                if opensmile_cols:
                    print("\nWith OpenSMILE features:")
                    opensmile_results = run_model_comparison_pipeline(df, feature_prefix='opensmile_')
                    if opensmile_results and opensmile_results['best_mae'] < results[name]['best_mae']:
                        results[name]['best_mae_opensmile'] = opensmile_results['best_mae']
                        results[name]['best_model_opensmile'] = opensmile_results['best_model']

        return results

    def run_population_model(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Train population model using all participants.
        """
        print("\n" + "#" * 70)
        print("# POPULATION MODEL (Leave-One-Person-Out)")
        print("#" * 70)

        # Combine all datasets
        all_dfs = []
        for name, df in datasets.items():
            df = df.copy()
            df['participant'] = name
            all_dfs.append(df)

        if not all_dfs:
            return {}

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal samples: {len(combined)}")
        print(f"Participants: {combined['participant'].nunique()}")

        # Get features
        feature_cols = [c for c in combined.columns if c.startswith('librosa_')]

        if len(feature_cols) == 0:
            print("No features found")
            return {}

        X = combined[feature_cols].values
        y = combined['glucose_mgdl'].values
        groups = combined['participant'].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Feature selection
        selector = FeatureSelector()
        X_selected, selected_features = selector.correlation_selection(
            X, y, feature_cols, top_k=min(30, len(feature_cols))
        )

        # Leave-one-person-out cross-validation
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.preprocessing import RobustScaler
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error
        from scipy.stats import pearsonr

        logo = LeaveOneGroupOut()
        scaler = RobustScaler()

        y_true_all = []
        y_pred_all = []
        person_results = {}

        for train_idx, test_idx in logo.split(X_selected, y, groups):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_person = groups[test_idx[0]]

            # Scale
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

            # Per-person metrics
            person_mae = mean_absolute_error(y_test, y_pred)
            person_r = pearsonr(y_test, y_pred)[0] if len(y_test) > 2 else 0
            person_results[test_person] = {
                'mae': person_mae,
                'pearson_r': person_r,
                'n_samples': len(y_test)
            }

            print(f"  {test_person}: MAE={person_mae:.2f}, r={person_r:.3f} (n={len(y_test)})")

        # Overall metrics
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        overall_mae = mean_absolute_error(y_true_all, y_pred_all)
        overall_r = pearsonr(y_true_all, y_pred_all)[0]

        print(f"\nOVERALL: MAE={overall_mae:.2f} mg/dL, r={overall_r:.3f}")

        return {
            'overall_mae': overall_mae,
            'overall_pearson_r': overall_r,
            'per_person': person_results,
            'n_total': len(y_true_all)
        }

    def run_transfer_learning(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run transfer learning approach.
        """
        print("\n" + "#" * 70)
        print("# TRANSFER LEARNING (Population -> Personalized)")
        print("#" * 70)

        # Combine all datasets
        all_dfs = []
        for name, df in datasets.items():
            df = df.copy()
            df['participant'] = name
            all_dfs.append(df)

        if not all_dfs:
            return {}

        combined = pd.concat(all_dfs, ignore_index=True)

        # Get features
        feature_cols = [c for c in combined.columns if c.startswith('librosa_')]
        X_all = combined[feature_cols].values
        y_all = combined['glucose_mgdl'].values
        persons = combined['participant'].values

        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        # Initialize transfer learning
        transfer = TransferLearningApproach()

        # Pre-train on all data
        print("\nPre-training population model...")
        pop_results = transfer.fit_population(X_all, y_all, persons)
        print(f"Population MAE: {pop_results['mae']:.2f} mg/dL")

        # Fine-tune per person
        print("\nFine-tuning per person...")
        person_results = {}

        for name in datasets.keys():
            mask = persons == name
            X_person = X_all[mask]
            y_person = y_all[mask]

            if len(y_person) < 5:
                continue

            ft_results = transfer.fine_tune_person(name, X_person, y_person)
            person_results[name] = ft_results
            print(f"  {name}: MAE={ft_results['mae']:.2f}, r={ft_results['pearson_r']:.3f}")

        return {
            'population': pop_results,
            'fine_tuned': person_results
        }

    def generate_report(self, personalized: Dict, population: Dict, transfer: Dict) -> str:
        """
        Generate summary report.
        """
        report = []
        report.append("\n" + "=" * 70)
        report.append("VOICE-BASED GLUCOSE ESTIMATION - RESULTS SUMMARY")
        report.append("=" * 70)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Personalized results
        report.append("\n\n### PERSONALIZED MODELS (per participant)")
        report.append("-" * 50)
        report.append(f"{'Participant':<20} {'Samples':<10} {'Best Model':<15} {'MAE (mg/dL)':<12}")
        report.append("-" * 50)

        for name, res in sorted(personalized.items()):
            report.append(f"{name:<20} {res['n_samples']:<10} {res['best_model']:<15} {res['best_mae']:.2f}")

        # Population results
        if population:
            report.append("\n\n### POPULATION MODEL (Leave-One-Person-Out)")
            report.append("-" * 50)
            report.append(f"Overall MAE: {population['overall_mae']:.2f} mg/dL")
            report.append(f"Overall Pearson r: {population['overall_pearson_r']:.3f}")

        # Transfer learning results
        if transfer and 'fine_tuned' in transfer:
            report.append("\n\n### TRANSFER LEARNING (Fine-tuned)")
            report.append("-" * 50)
            for name, res in sorted(transfer['fine_tuned'].items()):
                report.append(f"{name:<20} MAE: {res['mae']:.2f}, r: {res['pearson_r']:.3f}")

        # Comparison
        report.append("\n\n### APPROACH COMPARISON")
        report.append("-" * 50)

        if personalized:
            avg_personalized = np.mean([r['best_mae'] for r in personalized.values()])
            report.append(f"Avg Personalized MAE: {avg_personalized:.2f} mg/dL")

        if population:
            report.append(f"Population MAE: {population['overall_mae']:.2f} mg/dL")

        if transfer and 'fine_tuned' in transfer:
            avg_transfer = np.mean([r['mae'] for r in transfer['fine_tuned'].values()])
            report.append(f"Avg Transfer Learning MAE: {avg_transfer:.2f} mg/dL")

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def run(self, participants: list = None) -> Dict:
        """
        Run complete analysis pipeline.
        """
        # Load data
        datasets = self.load_all_participants(participants)

        if not datasets:
            print("No data loaded!")
            return {}

        # Personalized models
        personalized_results = self.run_personalized_models(datasets)

        # Population model (requires multiple participants)
        if len(datasets) >= 2:
            population_results = self.run_population_model(datasets)
            transfer_results = self.run_transfer_learning(datasets)
        else:
            print("\nSkipping population and transfer models (need 2+ participants)")
            population_results = {}
            transfer_results = {}

        # Generate report
        report = self.generate_report(
            personalized_results,
            population_results,
            transfer_results
        )
        print(report)

        # Save results
        self.results = {
            'datasets': {k: len(v) for k, v in datasets.items()},
            'personalized': personalized_results,
            'population': population_results,
            'transfer': transfer_results
        }

        # Save report to file
        report_path = BASE_DIR / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

        return self.results


def main():
    parser = argparse.ArgumentParser(description='Voice-Based Glucose Estimation Analysis')
    parser.add_argument('--participant', '-p', type=str, help='Single participant to analyze')
    parser.add_argument('--wav2vec', action='store_true', help='Include Wav2Vec2 features')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (Wolf only)')
    parser.add_argument('--list', action='store_true', help='List available participants')

    args = parser.parse_args()

    if args.list:
        print("Available participants:")
        for name, config in PARTICIPANTS.items():
            audio_ext = config.get('audio_ext', '.wav')
            ready = "Ready" if audio_ext == '.wav' else "Needs conversion"
            print(f"  {name}: {ready}")
        return

    # Determine participants
    if args.quick:
        participants = ['Wolf']  # Largest dataset
    elif args.participant:
        participants = [args.participant]
    else:
        participants = None  # All

    # Run pipeline
    pipeline = FullAnalysisPipeline(use_wav2vec=args.wav2vec)
    results = pipeline.run(participants)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
