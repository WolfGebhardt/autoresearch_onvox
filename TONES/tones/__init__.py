"""
TONES — Voice-Based Glucose Estimation Package
===============================================
Non-invasive blood glucose monitoring using voice biomarkers.

Usage:
    from tones.config import load_config
    from tones.data.loaders import load_participant_data
    from tones.features.mfcc import MFCCExtractor
    from tones.models.train import train_personalized, train_population
    from tones.evaluation.metrics import evaluate_regression
"""

__version__ = "1.0.0"
