"""
Wav2Vec2 Deep Learning Feature Extraction
==========================================
Extracts learned representations from audio using pre-trained Wav2Vec2 model.
These embeddings capture complex patterns not accessible via hand-crafted features.

Requirements:
    pip install torch torchaudio transformers

Usage:
    extractor = Wav2VecExtractor()
    features = extractor.extract("audio.wav")
"""

import numpy as np
from typing import Dict, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check for torch availability
try:
    import torch
    import torchaudio
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/transformers not available. Install with:")
    print("  pip install torch torchaudio transformers")


class Wav2VecExtractor:
    """
    Extract Wav2Vec2 embeddings from audio files.

    Features extracted:
    - Mean, std, max of hidden states across time
    - Layer-wise statistics from middle layers (most informative for paralinguistics)
    - Pooled 768-dimensional embedding
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = None):
        """
        Initialize Wav2Vec2 feature extractor.

        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None for auto-detect
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch and transformers required. Install with: pip install torch torchaudio transformers")

        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading Wav2Vec2 model on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Wav2Vec2 model loaded successfully")

        self.prefix = 'wav2vec'

    def extract(self, audio_path: str, target_sr: int = 16000) -> Dict[str, float]:
        """
        Extract Wav2Vec2 features from an audio file.

        Args:
            audio_path: Path to WAV file
            target_sr: Target sample rate (Wav2Vec2 expects 16kHz)

        Returns:
            Dictionary of features
        """
        if not TORCH_AVAILABLE:
            return {}

        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Resample if necessary
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Flatten to 1D
            waveform = waveform.squeeze().numpy()

            # Skip very short audio
            if len(waveform) < target_sr * 0.5:  # Less than 0.5 seconds
                return {}

            # Process through Wav2Vec2
            inputs = self.processor(
                waveform,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(input_values, output_hidden_states=True)

            # Get last hidden state
            last_hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # [T, 768]

            # Get hidden states from different layers
            hidden_states = outputs.hidden_states  # Tuple of [1, T, 768] for each layer

            features = {}

            # === Global pooling of last hidden state ===
            features.update(self._pool_embeddings(last_hidden, 'last'))

            # === Layer-specific features ===
            # Middle layers (6-8) often best for paralinguistics
            for layer_idx in [4, 6, 8, 10]:
                if layer_idx < len(hidden_states):
                    layer_hidden = hidden_states[layer_idx].squeeze(0).cpu().numpy()
                    features.update(self._pool_embeddings(layer_hidden, f'layer{layer_idx}'))

            # === Temporal dynamics ===
            features.update(self._temporal_features(last_hidden))

            return features

        except Exception as e:
            print(f"Wav2Vec2 extraction error for {audio_path}: {e}")
            return {}

    def _pool_embeddings(self, embeddings: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Pool embeddings across time dimension.

        Args:
            embeddings: [T, D] array
            prefix: Prefix for feature names

        Returns:
            Dictionary of pooled features
        """
        features = {}

        # Global statistics
        mean_emb = np.mean(embeddings, axis=0)
        std_emb = np.std(embeddings, axis=0)
        max_emb = np.max(embeddings, axis=0)

        # Store first N dimensions (768 is too many)
        n_dims = 32  # Reduce dimensionality

        for i in range(n_dims):
            features[f'{prefix}_mean_d{i}'] = float(mean_emb[i])
            features[f'{prefix}_std_d{i}'] = float(std_emb[i])

        # Summary statistics across all dimensions
        features[f'{prefix}_mean_norm'] = float(np.linalg.norm(mean_emb))
        features[f'{prefix}_std_mean'] = float(np.mean(std_emb))
        features[f'{prefix}_max_mean'] = float(np.mean(max_emb))

        return features

    def _temporal_features(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal dynamics from embeddings.

        Args:
            embeddings: [T, D] array

        Returns:
            Dictionary of temporal features
        """
        features = {}

        # Temporal change
        if embeddings.shape[0] > 1:
            diffs = np.diff(embeddings, axis=0)
            diff_norms = np.linalg.norm(diffs, axis=1)

            features['temporal_diff_mean'] = float(np.mean(diff_norms))
            features['temporal_diff_std'] = float(np.std(diff_norms))
            features['temporal_diff_max'] = float(np.max(diff_norms))

        # First vs last embedding (captures overall change)
        if embeddings.shape[0] > 10:
            first_emb = np.mean(embeddings[:5], axis=0)
            last_emb = np.mean(embeddings[-5:], axis=0)
            features['first_last_distance'] = float(np.linalg.norm(last_emb - first_emb))

        return features


class Wav2VecBatchExtractor:
    """
    Batch extraction for multiple files with caching.
    """

    def __init__(self, cache_dir: Path = None):
        self.extractor = None
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_extractor(self):
        """Lazy load the extractor."""
        if self.extractor is None:
            self.extractor = Wav2VecExtractor()

    def extract_batch(self, audio_paths: list, show_progress: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Extract features from multiple audio files.

        Args:
            audio_paths: List of paths to WAV files
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping filename to features
        """
        self._ensure_extractor()

        results = {}
        total = len(audio_paths)

        for i, audio_path in enumerate(audio_paths):
            if show_progress and i % 10 == 0:
                print(f"Processing {i+1}/{total}: {Path(audio_path).name}")

            features = self.extractor.extract(str(audio_path))
            if features:
                results[str(audio_path)] = features

        return results


def extract_wav2vec_features(audio_path: str) -> Dict[str, float]:
    """
    Convenience function to extract Wav2Vec2 features.

    Args:
        audio_path: Path to WAV file

    Returns:
        Dictionary of features with 'wav2vec_' prefix
    """
    if not TORCH_AVAILABLE:
        return {}

    try:
        extractor = Wav2VecExtractor()
        features = extractor.extract(audio_path)
        return {f'wav2vec_{k}': v for k, v in features.items()}
    except Exception as e:
        print(f"Wav2Vec2 error: {e}")
        return {}


# Singleton extractor for efficiency
_wav2vec_extractor = None


def get_wav2vec_features(audio_path: str) -> Dict[str, float]:
    """
    Get Wav2Vec2 features using singleton extractor (more efficient for batch processing).
    """
    global _wav2vec_extractor

    if not TORCH_AVAILABLE:
        return {}

    try:
        if _wav2vec_extractor is None:
            _wav2vec_extractor = Wav2VecExtractor()

        features = _wav2vec_extractor.extract(audio_path)
        return {f'wav2vec_{k}': v for k, v in features.items()}
    except Exception as e:
        print(f"Wav2Vec2 error: {e}")
        return {}


# Test
if __name__ == "__main__":
    import sys

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping test.")
        sys.exit(0)

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(f"Testing Wav2Vec2 extraction on: {audio_file}")

        extractor = Wav2VecExtractor()
        features = extractor.extract(audio_file)

        print(f"\nExtracted {len(features)} features")
        print("\nSample features:")
        for key in list(features.keys())[:10]:
            print(f"  {key}: {features[key]:.6f}")
    else:
        print("Usage: python wav2vec_features.py <audio_file.wav>")
        print("\nTesting model loading...")
        if TORCH_AVAILABLE:
            extractor = Wav2VecExtractor()
            print("Model loaded successfully!")
