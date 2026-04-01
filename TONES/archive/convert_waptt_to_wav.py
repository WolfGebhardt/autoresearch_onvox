"""
Convert .waptt files to .wav for participants that only have .waptt audio.
Uses librosa to load and soundfile to save as 16kHz mono WAV.

Run with: C:/Python310/python.exe convert_waptt_to_wav.py
"""

import os
import librosa
import soundfile as sf
from pathlib import Path

BASE_DIR = Path(r"C:\Users\whgeb\OneDrive\TONES")

# Participants that need .waptt -> .wav conversion
CONVERT_DIRS = {
    "Darav": {
        "source": "Darav/Nov21_finished",
        "output": "Darav/wav_audio",
    },
    "Joao": {
        "source": "Joao/Nov21",
        "output": "Joao/wav_audio",
    },
    "Alvar": {
        "source": "Alvar",
        "output": "Alvar/wav_audio",
    },
}

SR = 16000  # Target sample rate

def convert_all():
    total_converted = 0
    total_skipped = 0
    total_failed = 0

    for name, config in CONVERT_DIRS.items():
        source_dir = BASE_DIR / config["source"]
        output_dir = BASE_DIR / config["output"]
        output_dir.mkdir(parents=True, exist_ok=True)

        waptt_files = sorted([f for f in source_dir.iterdir() if f.suffix == '.waptt'])
        print(f"\n{name}: {len(waptt_files)} .waptt files in {config['source']}")

        converted = 0
        for wf in waptt_files:
            out_name = wf.stem + ".wav"
            out_path = output_dir / out_name

            if out_path.exists():
                total_skipped += 1
                continue

            try:
                y, sr = librosa.load(str(wf), sr=SR, mono=True)
                sf.write(str(out_path), y, SR)
                converted += 1
            except Exception as e:
                print(f"  FAILED: {wf.name}: {e}")
                total_failed += 1

        print(f"  Converted: {converted}, Skipped (exist): {total_skipped}, Failed: {total_failed}")
        total_converted += converted

    print(f"\nDONE: {total_converted} converted, {total_skipped} skipped, {total_failed} failed")


if __name__ == "__main__":
    convert_all()
