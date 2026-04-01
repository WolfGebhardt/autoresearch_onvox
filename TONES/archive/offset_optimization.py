"""
Time Offset Optimization for Voice-Glucose Alignment
=====================================================
CGM sensors have a known lag (5-15 minutes) relative to blood glucose.
Voice biomarkers may reflect glucose changes at different delays.

This module:
1. Tests multiple time offsets to find optimal alignment
2. Uses glucose dynamics filtering to select stable periods for more reliable matching
3. Widens the time window for samples with low glucose variability

Key insight: If voice correlates better with CGM readings from X minutes ago/ahead,
this tells us about the relative lag between voice changes and CGM detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


@dataclass
class AlignmentResult:
    """Results from a single alignment configuration."""
    offset_minutes: int
    window_minutes: int
    n_samples: int
    mae: float
    pearson_r: float
    glucose_std: float  # Average glucose std in matching windows


class OffsetOptimizer:
    """
    Find optimal time offset between voice recordings and CGM readings.
    """

    def __init__(self, glucose_df: pd.DataFrame):
        """
        Args:
            glucose_df: DataFrame with 'timestamp' and 'glucose_mgdl' columns
        """
        self.glucose_df = glucose_df.sort_values('timestamp').reset_index(drop=True)

    def get_glucose_at_offset(self,
                               audio_timestamp: datetime,
                               offset_minutes: int,
                               window_minutes: int = 15) -> Tuple[Optional[float], Optional[float]]:
        """
        Get glucose value at (audio_time + offset).

        Args:
            audio_timestamp: Time of voice recording
            offset_minutes: Offset to apply (positive = look forward, negative = look back)
            window_minutes: Search window around target time

        Returns:
            (glucose_value, glucose_std_in_window) or (None, None)
        """
        target_time = audio_timestamp + timedelta(minutes=offset_minutes)
        window = timedelta(minutes=window_minutes)

        # Find readings within window
        mask = (self.glucose_df['timestamp'] >= target_time - window) & \
               (self.glucose_df['timestamp'] <= target_time + window)

        candidates = self.glucose_df[mask]

        if candidates.empty:
            return None, None

        # Get closest reading
        time_diffs = abs(candidates['timestamp'] - target_time)
        closest_idx = time_diffs.idxmin()
        glucose_val = candidates.loc[closest_idx, 'glucose_mgdl']

        # Calculate glucose variability in window (useful for filtering)
        glucose_std = candidates['glucose_mgdl'].std() if len(candidates) > 1 else 0.0

        return glucose_val, glucose_std

    def get_glucose_with_dynamics_filter(self,
                                          audio_timestamp: datetime,
                                          offset_minutes: int = 0,
                                          window_minutes: int = 30,
                                          max_rate_of_change: float = 2.0) -> Tuple[Optional[float], float]:
        """
        Get glucose value, but only if glucose is relatively stable in the window.

        This allows wider windows while ensuring the voice recording corresponds
        to a consistent glucose state.

        Args:
            audio_timestamp: Time of voice recording
            offset_minutes: Time offset to apply
            window_minutes: Window to search (can be larger due to stability check)
            max_rate_of_change: Maximum mg/dL per minute to consider "stable"

        Returns:
            (glucose_value, rate_of_change) or (None, float)
        """
        target_time = audio_timestamp + timedelta(minutes=offset_minutes)
        window = timedelta(minutes=window_minutes)

        # Get all readings in window
        mask = (self.glucose_df['timestamp'] >= target_time - window) & \
               (self.glucose_df['timestamp'] <= target_time + window)

        candidates = self.glucose_df[mask].sort_values('timestamp')

        if len(candidates) < 2:
            return None, float('inf')

        # Calculate rate of change
        glucose_values = candidates['glucose_mgdl'].values
        time_diffs = candidates['timestamp'].diff().dt.total_seconds() / 60  # minutes

        # mg/dL per minute
        rates = np.abs(np.diff(glucose_values) / time_diffs.iloc[1:].values)
        max_rate = np.max(rates) if len(rates) > 0 else 0.0

        if max_rate > max_rate_of_change:
            return None, max_rate

        # Use mean glucose in stable period
        glucose_val = candidates['glucose_mgdl'].mean()

        return glucose_val, max_rate


class OffsetExperiment:
    """
    Run experiments to find optimal voice-glucose alignment.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def test_offsets(self,
                     audio_features_df: pd.DataFrame,
                     glucose_df: pd.DataFrame,
                     offsets: List[int] = None,
                     windows: List[int] = None) -> List[AlignmentResult]:
        """
        Test multiple offset/window combinations.

        Args:
            audio_features_df: DataFrame with audio features and 'audio_timestamp'
            glucose_df: DataFrame with glucose readings
            offsets: List of offsets to test (minutes, can be negative)
            windows: List of window sizes to test (minutes)

        Returns:
            List of AlignmentResult sorted by MAE
        """
        if offsets is None:
            # Test offsets from -30 to +30 minutes (5 min steps)
            offsets = list(range(-30, 35, 5))

        if windows is None:
            windows = [10, 15, 20, 30]

        optimizer = OffsetOptimizer(glucose_df)
        results = []

        # Get feature columns
        feature_cols = [c for c in audio_features_df.columns
                        if c.startswith('librosa_') or c.startswith('praat_') or c.startswith('opensmile_')]

        if not feature_cols:
            print("No feature columns found")
            return []

        for offset in offsets:
            for window in windows:
                # Align each audio recording with glucose at offset
                aligned_data = []

                for idx, row in audio_features_df.iterrows():
                    audio_ts = row['audio_timestamp']
                    if isinstance(audio_ts, str):
                        audio_ts = pd.to_datetime(audio_ts)

                    glucose_val, glucose_std = optimizer.get_glucose_at_offset(
                        audio_ts, offset, window
                    )

                    if glucose_val is not None:
                        aligned_data.append({
                            'idx': idx,
                            'glucose': glucose_val,
                            'glucose_std': glucose_std
                        })

                if len(aligned_data) < 10:
                    continue

                # Build X, y
                aligned_indices = [d['idx'] for d in aligned_data]
                X = audio_features_df.loc[aligned_indices, feature_cols].values
                y = np.array([d['glucose'] for d in aligned_data])
                glucose_stds = np.array([d['glucose_std'] for d in aligned_data])

                # Handle missing values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                # Train model with LOO-CV
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = Ridge(alpha=1.0)
                loo = LeaveOneOut()

                try:
                    y_pred = cross_val_predict(model, X_scaled, y, cv=loo)
                    mae = mean_absolute_error(y, y_pred)
                    r, _ = pearsonr(y, y_pred)
                except Exception:
                    continue

                results.append(AlignmentResult(
                    offset_minutes=offset,
                    window_minutes=window,
                    n_samples=len(y),
                    mae=mae,
                    pearson_r=r,
                    glucose_std=np.mean(glucose_stds)
                ))

        # Sort by MAE
        results.sort(key=lambda x: x.mae)

        return results

    def test_dynamics_filtered(self,
                                audio_features_df: pd.DataFrame,
                                glucose_df: pd.DataFrame,
                                max_rates: List[float] = None) -> List[AlignmentResult]:
        """
        Test glucose dynamics filtering to expand usable data.

        Uses wider windows but only accepts samples with stable glucose.

        Args:
            audio_features_df: DataFrame with audio features
            glucose_df: DataFrame with glucose readings
            max_rates: Maximum rate of change values to test (mg/dL per minute)

        Returns:
            List of results
        """
        if max_rates is None:
            # 1 mg/dL/min = relatively stable
            # 2 mg/dL/min = moderate dynamics
            # 3 mg/dL/min = higher dynamics allowed
            max_rates = [1.0, 1.5, 2.0, 2.5, 3.0]

        optimizer = OffsetOptimizer(glucose_df)
        results = []

        feature_cols = [c for c in audio_features_df.columns
                        if c.startswith('librosa_') or c.startswith('praat_') or c.startswith('opensmile_')]

        if not feature_cols:
            print("No feature columns found")
            return []

        for max_rate in max_rates:
            # Use wider window (30 min) with dynamics filtering
            aligned_data = []

            for idx, row in audio_features_df.iterrows():
                audio_ts = row['audio_timestamp']
                if isinstance(audio_ts, str):
                    audio_ts = pd.to_datetime(audio_ts)

                glucose_val, rate = optimizer.get_glucose_with_dynamics_filter(
                    audio_ts,
                    offset_minutes=0,
                    window_minutes=30,
                    max_rate_of_change=max_rate
                )

                if glucose_val is not None:
                    aligned_data.append({
                        'idx': idx,
                        'glucose': glucose_val,
                        'rate': rate
                    })

            if len(aligned_data) < 10:
                continue

            # Build and evaluate model
            aligned_indices = [d['idx'] for d in aligned_data]
            X = audio_features_df.loc[aligned_indices, feature_cols].values
            y = np.array([d['glucose'] for d in aligned_data])

            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(alpha=1.0)
            loo = LeaveOneOut()

            try:
                y_pred = cross_val_predict(model, X_scaled, y, cv=loo)
                mae = mean_absolute_error(y, y_pred)
                r, _ = pearsonr(y, y_pred)
            except Exception:
                continue

            results.append(AlignmentResult(
                offset_minutes=0,
                window_minutes=30,
                n_samples=len(y),
                mae=mae,
                pearson_r=r,
                glucose_std=max_rate  # Using this field to store max_rate for this mode
            ))

            print(f"Max rate {max_rate:.1f} mg/dL/min: {len(y)} samples, MAE={mae:.2f}, r={r:.3f}")

        return results


def print_offset_results(results: List[AlignmentResult], top_n: int = 10):
    """Pretty print offset optimization results."""
    print("\n" + "=" * 70)
    print("OFFSET OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"{'Offset':>8} {'Window':>8} {'Samples':>8} {'MAE':>10} {'Pearson r':>10}")
    print("-" * 70)

    for r in results[:top_n]:
        print(f"{r.offset_minutes:>8} {r.window_minutes:>8} {r.n_samples:>8} "
              f"{r.mae:>10.2f} {r.pearson_r:>10.3f}")

    if results:
        best = results[0]
        print("-" * 70)
        print(f"BEST: Offset={best.offset_minutes}min, Window={best.window_minutes}min")
        print(f"      MAE={best.mae:.2f} mg/dL, r={best.pearson_r:.3f}")

        # Interpretation
        if best.offset_minutes < 0:
            print(f"\nInterpretation: Voice changes PRECEDE CGM by ~{abs(best.offset_minutes)} minutes")
            print("This suggests voice reflects blood glucose before CGM can detect it.")
        elif best.offset_minutes > 0:
            print(f"\nInterpretation: Voice changes LAG CGM by ~{best.offset_minutes} minutes")
            print("This suggests voice responds after glucose changes are detected by CGM.")
        else:
            print("\nInterpretation: Voice and CGM are approximately synchronized")


def run_offset_optimization(participant_name: str,
                            audio_features_df: pd.DataFrame,
                            glucose_df: pd.DataFrame,
                            base_dir: Path = None) -> Dict:
    """
    Run full offset optimization for a participant.

    Returns:
        Dictionary with optimization results
    """
    if base_dir is None:
        base_dir = Path(r"C:\Users\whgeb\OneDrive\TONES")

    experiment = OffsetExperiment(base_dir)

    print(f"\n{'#' * 70}")
    print(f"# OFFSET OPTIMIZATION: {participant_name}")
    print(f"{'#' * 70}")

    # Test different offsets
    print("\n1. Testing time offsets (-30 to +30 minutes)...")
    offset_results = experiment.test_offsets(
        audio_features_df, glucose_df,
        offsets=list(range(-30, 35, 5)),
        windows=[10, 15, 20]
    )

    print_offset_results(offset_results)

    # Test dynamics filtering
    print("\n2. Testing glucose dynamics filtering (wider windows, stable periods)...")
    dynamics_results = experiment.test_dynamics_filtered(
        audio_features_df, glucose_df,
        max_rates=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    )

    return {
        'offset_results': offset_results,
        'dynamics_results': dynamics_results,
        'best_offset': offset_results[0] if offset_results else None
    }


# Example usage
if __name__ == "__main__":
    print("Offset Optimization Module")
    print("Use run_offset_optimization(participant, audio_df, glucose_df)")
