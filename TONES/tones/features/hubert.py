"""
HuBERT Feature Extraction
===========================
Transfer learning features from the HuBERT speech foundation model.

Extracts 2304-dimensional embeddings (mean + std + max of 768-dim hidden states).
Optional PCA reduction to mitigate the curse of dimensionality.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import librosa

logger = logging.getLogger(__name__)

# Lazy-loaded to avoid importing torch at module level
_hubert_model = None
_hubert_extractor = None
_device = None


def _ensure_model_loaded(model_name: str = "facebook/hubert-base-ls960"):
    """Lazy-load HuBERT model on first use."""
    global _hubert_model, _hubert_extractor, _device

    if _hubert_model is not None:
        return

    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading HuBERT model: %s (device: %s)", model_name, _device)

    _hubert_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    _hubert_model = HubertModel.from_pretrained(model_name)
    _hubert_model.to(_device)
    _hubert_model.eval()

    logger.info("HuBERT model loaded successfully")


class HuBERTExtractor:
    """
    Extract HuBERT embeddings from audio.

    Produces a fixed-length vector by aggregating the last hidden state
    across time using mean, std, and max pooling.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    target_sr : int
        Target sample rate (HuBERT expects 16000).
    pca_components : int or None
        If set, apply PCA to reduce dimensionality. Recommended for
        population models to avoid p >> n overfitting (e.g., 30-50).
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        target_sr: int = 16000,
        pca_components: Optional[int] = None,
    ):
        self.model_name = model_name
        self.target_sr = target_sr
        self.pca_components = pca_components
        self._pca = None

    def extract_from_file(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract HuBERT features from an audio file.

        Returns
        -------
        np.ndarray or None
            2304-dimensional feature vector (or PCA-reduced), or None on failure.
        """
        try:
            waveform, _ = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
        except Exception as e:
            logger.warning("Failed to load %s: %s", audio_path, e)
            return None

        return self.extract_from_array(waveform)

    def extract_from_array(self, waveform: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract HuBERT features from a waveform array.

        Parameters
        ----------
        waveform : np.ndarray
            Mono audio at self.target_sr sample rate.
        """
        import torch

        _ensure_model_loaded(self.model_name)

        try:
            waveform = waveform.astype(np.float32)

            inputs = _hubert_extractor(
                waveform,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = _hubert_model(inputs.input_values.to(_device))
                hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

            # Aggregate: mean, std, max pooling -> 3 * 768 = 2304 dims
            features = np.concatenate([
                np.mean(hidden, axis=0),
                np.std(hidden, axis=0),
                np.max(hidden, axis=0),
            ])

            return features.astype(np.float32)

        except Exception as e:
            logger.warning("HuBERT extraction failed: %s", e)
            return None

    def fit_pca(self, feature_matrix: np.ndarray):
        """
        Fit PCA on a matrix of HuBERT features for dimensionality reduction.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Shape (n_samples, 2304).
        """
        if self.pca_components is None:
            return

        from sklearn.decomposition import PCA

        n_components = min(self.pca_components, feature_matrix.shape[0] - 1, feature_matrix.shape[1])
        self._pca = PCA(n_components=n_components)
        self._pca.fit(feature_matrix)
        logger.info(
            "PCA fitted: %d -> %d components (%.1f%% variance explained)",
            feature_matrix.shape[1], n_components,
            self._pca.explained_variance_ratio_.sum() * 100,
        )

    def transform_pca(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA reduction to features."""
        if self._pca is None:
            return features
        if features.ndim == 1:
            return self._pca.transform(features.reshape(1, -1)).flatten()
        return self._pca.transform(features)
