"""
Quick runner for combined model with limited samples for faster results.
Run the full model later with: python combined_hubert_mfcc_model.py
"""

from combined_hubert_mfcc_model import main

if __name__ == "__main__":
    # Process 100 samples per participant for faster results
    # Set to None for full processing
    main(max_per_participant=100)
