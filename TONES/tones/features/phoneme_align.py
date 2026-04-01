"""
Phoneme-Level Alignment via Whisper
======================================
Uses OpenAI Whisper to transcribe audio and obtain word-level timestamps.
Words are then decomposed into approximate vowel/consonant segments for
phoneme-controlled physiological residual extraction.

The key insight: glucose affects the vocal apparatus (fold viscosity, tension,
neural control) — but those effects are tiny compared to phonemic content.
By aligning to phonemes, we can control for content and extract the residual
physiological signal where glucose actually lives.

Requires: openai-whisper (pip install openai-whisper)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── Phoneme category definitions ──────────────────────────────────────────────
# We use a simplified phoneme model based on orthographic approximation.
# Vowel segments are the most informative for glucose because they expose
# vocal fold vibration directly (voiced, sustained, formant-rich).

VOWEL_CHARS = set("aeiouäöüàáâãèéêìíîòóôõùúûAEIOUÄÖÜ")
VOWEL_DIGRAPHS = {"ai", "au", "ei", "eu", "ou", "ie", "oo", "ee", "ea", "oa", "ue"}


@dataclass
class WordSegment:
    """A single word with start/end timestamps from Whisper."""
    word: str
    start: float  # seconds
    end: float    # seconds
    confidence: float = 0.0


@dataclass
class PhonemeSegment:
    """An approximate phoneme/grapheme cluster with timestamps."""
    text: str
    start: float  # seconds
    end: float    # seconds
    category: str  # "vowel", "consonant", "mixed"
    word: str = ""  # parent word


@dataclass
class AlignmentResult:
    """Full alignment result for one audio recording."""
    audio_path: str
    transcript: str
    language: str
    words: List[WordSegment] = field(default_factory=list)
    phonemes: List[PhonemeSegment] = field(default_factory=list)
    n_vowel_segments: int = 0
    n_consonant_segments: int = 0
    duration_sec: float = 0.0


# ─── Whisper model management ──────────────────────────────────────────────────

_whisper_model = None
_whisper_model_name = None


def load_whisper_model(model_name: str = "base") -> "whisper.Whisper":
    """
    Load (or reuse cached) Whisper model.
    
    Parameters
    ----------
    model_name : str
        Whisper model size: "tiny", "base", "small", "medium", "large".
        "base" is a good trade-off: fast enough for 700+ files, accurate enough
        for word-level alignment. Word timestamps don't improve much with larger models.
    """
    global _whisper_model, _whisper_model_name
    
    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model
    
    import whisper
    logger.info("Loading Whisper model '%s'...", model_name)
    _whisper_model = whisper.load_model(model_name)
    _whisper_model_name = model_name
    logger.info("Whisper model '%s' loaded.", model_name)
    return _whisper_model


# ─── Core alignment ────────────────────────────────────────────────────────────

def transcribe_and_align(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None,
) -> Optional[AlignmentResult]:
    """
    Transcribe an audio file and extract word-level timestamps using Whisper.
    
    Audio is loaded via librosa (which doesn't require ffmpeg) and passed
    as a numpy array to Whisper's transcribe(), bypassing Whisper's own
    ffmpeg-based audio loading.
    
    Parameters
    ----------
    audio_path : str or Path
        Path to audio file (WAV preferred).
    model_name : str
        Whisper model size.
    language : str, optional
        Force language detection (e.g., "en", "de"). None = auto-detect.
    
    Returns
    -------
    AlignmentResult or None
        Alignment with word timestamps and approximate phoneme segments.
    """
    import librosa
    
    audio_path = str(audio_path)
    
    if not Path(audio_path).exists():
        logger.warning("Audio file not found: %s", audio_path)
        return None
    
    try:
        model = load_whisper_model(model_name)
        
        # Load audio with librosa (works without ffmpeg on Windows)
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_array = y.astype(np.float32)
        
        # Transcribe with word-level timestamps
        # Pass numpy array instead of file path to bypass ffmpeg dependency
        result = model.transcribe(
            audio_array,
            word_timestamps=True,
            language=language,
            fp16=False,  # CPU compatibility; GPU will still use fp16 internally
        )
        
        transcript = result.get("text", "").strip()
        detected_lang = result.get("language", "unknown")
        
        if not transcript:
            logger.debug("Empty transcript for %s", audio_path)
            return AlignmentResult(
                audio_path=audio_path,
                transcript="",
                language=detected_lang,
            )
        
        # Extract word segments from Whisper output
        words = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                word_text = w.get("word", "").strip()
                if word_text:
                    words.append(WordSegment(
                        word=word_text,
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        confidence=w.get("probability", 0.0),
                    ))
        
        # Decompose words into approximate phoneme segments
        phonemes = _decompose_words_to_phonemes(words)
        
        n_vowels = sum(1 for p in phonemes if p.category == "vowel")
        n_consonants = sum(1 for p in phonemes if p.category == "consonant")
        
        # Estimate duration from last word end
        duration = words[-1].end if words else 0.0
        
        return AlignmentResult(
            audio_path=audio_path,
            transcript=transcript,
            language=detected_lang,
            words=words,
            phonemes=phonemes,
            n_vowel_segments=n_vowels,
            n_consonant_segments=n_consonants,
            duration_sec=duration,
        )
    
    except Exception as e:
        logger.warning("Transcription failed for %s: %s", audio_path, e)
        return None


def _decompose_words_to_phonemes(words: List[WordSegment]) -> List[PhonemeSegment]:
    """
    Decompose word segments into approximate vowel/consonant clusters.
    
    Since we don't have a true phoneme-level forced aligner, we use a
    principled approximation: split each word into vowel clusters and
    consonant clusters, distribute the word duration proportionally
    (vowels get ~60% of duration weight, consonants ~40%, reflecting
    typical acoustic energy distribution).
    
    This is sufficient because:
    - We don't need exact phoneme boundaries
    - We need consistent *categorization* so the same vowel type gets 
      compared across recordings
    - The residual computation averages over many instances, so small
      timing errors wash out
    """
    phonemes = []
    
    for ws in words:
        word = ws.word.lower().strip()
        if not word:
            continue
        
        # Split word into alternating vowel/consonant clusters
        clusters = _split_into_clusters(word)
        
        if not clusters:
            continue
        
        # Distribute word duration across clusters
        # Vowels tend to be longer than consonants in speech
        total_weight = 0.0
        weights = []
        for text, cat in clusters:
            w = len(text) * (1.5 if cat == "vowel" else 0.8)
            weights.append(w)
            total_weight += w
        
        if total_weight == 0:
            continue
        
        word_duration = ws.end - ws.start
        current_time = ws.start
        
        for (text, cat), weight in zip(clusters, weights):
            frac = weight / total_weight
            segment_duration = word_duration * frac
            
            phonemes.append(PhonemeSegment(
                text=text,
                start=current_time,
                end=current_time + segment_duration,
                category=cat,
                word=ws.word,
            ))
            current_time += segment_duration
    
    return phonemes


def _split_into_clusters(word: str) -> List[Tuple[str, str]]:
    """
    Split a word into alternating vowel/consonant grapheme clusters.
    
    Examples:
        "hello" -> [("h", "consonant"), ("e", "vowel"), ("ll", "consonant"), ("o", "vowel")]
        "strength" -> [("str", "consonant"), ("e", "vowel"), ("ngth", "consonant")]
        "audio" -> [("au", "vowel"), ("d", "consonant"), ("io", "vowel")]
    """
    if not word:
        return []
    
    # Remove non-alphabetic characters
    word = re.sub(r'[^a-zäöüàáâãèéêìíîòóôõùúû]', '', word.lower())
    if not word:
        return []
    
    clusters = []
    current = word[0]
    current_is_vowel = word[0] in VOWEL_CHARS
    
    for ch in word[1:]:
        ch_is_vowel = ch in VOWEL_CHARS
        if ch_is_vowel == current_is_vowel:
            current += ch
        else:
            cat = "vowel" if current_is_vowel else "consonant"
            clusters.append((current, cat))
            current = ch
            current_is_vowel = ch_is_vowel
    
    # Don't forget the last cluster
    cat = "vowel" if current_is_vowel else "consonant"
    clusters.append((current, cat))
    
    return clusters


# ─── Batch alignment with caching ──────────────────────────────────────────────

def align_batch(
    audio_paths: List[str],
    model_name: str = "base",
    cache_dir: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict[str, AlignmentResult]:
    """
    Transcribe and align a batch of audio files, with optional disk caching.
    
    Parameters
    ----------
    audio_paths : list of str
        Paths to audio files.
    model_name : str
        Whisper model size.
    cache_dir : str, optional
        Directory for alignment cache. None = no caching.
    language : str, optional
        Force language.
    
    Returns
    -------
    dict
        {audio_path: AlignmentResult}
    """
    import json
    
    results = {}
    cached = 0
    computed = 0
    failed = 0
    
    cache_path = Path(cache_dir) if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
    
    for i, audio_path in enumerate(audio_paths):
        audio_path = str(audio_path)
        
        # Check cache
        if cache_path:
            cache_file = cache_path / (_path_hash(audio_path) + ".json")
            if cache_file.exists():
                try:
                    alignment = _load_cached_alignment(cache_file)
                    if alignment is not None:
                        results[audio_path] = alignment
                        cached += 1
                        continue
                except Exception:
                    pass
        
        # Compute alignment
        alignment = transcribe_and_align(audio_path, model_name, language)
        
        if alignment is not None:
            results[audio_path] = alignment
            computed += 1
            
            # Save to cache
            if cache_path:
                try:
                    _save_cached_alignment(
                        cache_path / (_path_hash(audio_path) + ".json"),
                        alignment,
                    )
                except Exception as e:
                    logger.debug("Cache save failed: %s", e)
        else:
            failed += 1
        
        if (i + 1) % 25 == 0:
            logger.info(
                "  Aligned %d/%d (cached=%d, computed=%d, failed=%d)",
                i + 1, len(audio_paths), cached, computed, failed,
            )
    
    logger.info(
        "Alignment complete: %d cached, %d computed, %d failed out of %d total",
        cached, computed, failed, len(audio_paths),
    )
    return results


def _path_hash(path: str) -> str:
    """Short hash of a file path for cache keys."""
    import hashlib
    return hashlib.sha256(path.encode()).hexdigest()[:16]


def _save_cached_alignment(cache_file: Path, alignment: AlignmentResult):
    """Serialize an AlignmentResult to JSON."""
    import json
    
    data = {
        "audio_path": alignment.audio_path,
        "transcript": alignment.transcript,
        "language": alignment.language,
        "duration_sec": alignment.duration_sec,
        "n_vowel_segments": alignment.n_vowel_segments,
        "n_consonant_segments": alignment.n_consonant_segments,
        "words": [
            {"word": w.word, "start": w.start, "end": w.end, "confidence": w.confidence}
            for w in alignment.words
        ],
        "phonemes": [
            {"text": p.text, "start": p.start, "end": p.end,
             "category": p.category, "word": p.word}
            for p in alignment.phonemes
        ],
    }
    
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_cached_alignment(cache_file: Path) -> Optional[AlignmentResult]:
    """Deserialize an AlignmentResult from JSON."""
    import json
    
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    words = [
        WordSegment(word=w["word"], start=w["start"], end=w["end"],
                    confidence=w.get("confidence", 0.0))
        for w in data.get("words", [])
    ]
    
    phonemes = [
        PhonemeSegment(text=p["text"], start=p["start"], end=p["end"],
                       category=p["category"], word=p.get("word", ""))
        for p in data.get("phonemes", [])
    ]
    
    return AlignmentResult(
        audio_path=data["audio_path"],
        transcript=data["transcript"],
        language=data.get("language", "unknown"),
        words=words,
        phonemes=phonemes,
        n_vowel_segments=data.get("n_vowel_segments", 0),
        n_consonant_segments=data.get("n_consonant_segments", 0),
        duration_sec=data.get("duration_sec", 0.0),
    )
