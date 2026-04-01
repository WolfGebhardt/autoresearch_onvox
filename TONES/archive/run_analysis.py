"""
Voice-Based Glucose Estimation - Quick Start
=============================================
Run this script to analyze your data and build models.

Usage:
    python run_analysis.py [participant_name]

Example:
    python run_analysis.py Sybille
    python run_analysis.py all
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from voice_glucose_pipeline import (
    create_dataset_for_participant, PARTICIPANTS, BASE_DIR,
    train_personalized_model, plot_results
)
from model_comparison import run_model_comparison_pipeline
from advanced_voice_features import extract_glucose_relevant_features

import pandas as pd
import numpy as np


def analyze_participant(name: str, verbose: bool = True):
    """
    Run full analysis for a single participant.
    """
    if name not in PARTICIPANTS:
        print(f"Unknown participant: {name}")
        print(f"Available: {list(PARTICIPANTS.keys())}")
        return None

    config = PARTICIPANTS[name]

    if config['audio_ext'] != '.wav':
        print(f"{name}: Audio needs conversion to WAV first")
        print("Use ffmpeg to convert .waptt files:")
        print(f"  ffmpeg -i input.waptt -ar 16000 -ac 1 output.wav")
        return None

    print(f"\n{'#' * 70}")
    print(f"# ANALYZING: {name}")
    print(f"{'#' * 70}")

    # Create dataset
    df = create_dataset_for_participant(name, config, window_minutes=15, verbose=verbose)

    if df is None or len(df) < 5:
        print(f"Insufficient data for {name}")
        return None

    # Save dataset
    output_path = BASE_DIR / f"{name}_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    # Model comparison
    results = run_model_comparison_pipeline(df, feature_prefix='librosa_')

    if results:
        print(f"\n{'='*60}")
        print(f"BEST MODEL FOR {name}: {results['best_model']}")
        print(f"MAE: {results['best_mae']:.2f} mg/dL")
        print(f"{'='*60}")

    return {'dataset': df, 'results': results}


def analyze_all():
    """
    Analyze all participants with WAV files.
    """
    all_results = {}

    for name, config in PARTICIPANTS.items():
        if config['audio_ext'] == '.wav':
            result = analyze_participant(name)
            if result:
                all_results[name] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL PARTICIPANTS")
    print("=" * 70)

    summary_data = []
    for name, result in all_results.items():
        if 'results' in result and result['results']:
            summary_data.append({
                'Participant': name,
                'Samples': len(result['dataset']),
                'Best Model': result['results']['best_model'],
                'MAE (mg/dL)': f"{result['results']['best_mae']:.1f}"
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

    return all_results


def quick_test():
    """
    Quick test with first available participant.
    """
    for name, config in PARTICIPANTS.items():
        if config['audio_ext'] == '.wav':
            audio_dir = BASE_DIR / config['audio_dir']
            wav_files = list(audio_dir.glob("*.wav"))
            if wav_files:
                print(f"Testing with: {name}")
                print(f"Found {len(wav_files)} WAV files")

                # Test feature extraction on first file
                test_file = wav_files[0]
                print(f"\nExtracting features from: {test_file.name}")

                features = extract_glucose_relevant_features(str(test_file))
                print(f"Extracted {len(features)} features")

                if features:
                    print("\nSample features:")
                    for key in list(features.keys())[:10]:
                        print(f"  {key}: {features[key]:.4f}")
                return
    print("No WAV files found to test")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg == 'all':
            analyze_all()
        elif arg == 'test':
            quick_test()
        else:
            # Try to match participant name
            matched = None
            for name in PARTICIPANTS.keys():
                if arg in name.lower():
                    matched = name
                    break

            if matched:
                analyze_participant(matched)
            else:
                print(f"Unknown argument: {arg}")
                print("\nUsage:")
                print("  python run_analysis.py <participant>  - Analyze one participant")
                print("  python run_analysis.py all           - Analyze all participants")
                print("  python run_analysis.py test          - Quick feature extraction test")
                print(f"\nAvailable participants: {list(PARTICIPANTS.keys())}")
    else:
        print("Voice-Based Glucose Estimation Pipeline")
        print("=" * 50)
        print("\nUsage:")
        print("  python run_analysis.py <participant>  - Analyze one participant")
        print("  python run_analysis.py all           - Analyze all participants")
        print("  python run_analysis.py test          - Quick feature extraction test")
        print(f"\nParticipants: {list(PARTICIPANTS.keys())}")
        print("\nRunning quick test...")
        quick_test()
